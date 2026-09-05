from TokenSim.llm.llm_request import Request
from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.block_pool import BlockPool
from TokenSim.block.block_table import BlockTable
from TokenSim.block.kv_cache_manager import KVCacheManager, PrefixReusePlan
from TokenSim.errors import OutOfBlocksError, SimulationStateError
from TokenSim.kv_working_set.placement import gpu_resident_blocks, retained_tokens
from typing import Tuple


def _streaming_keep_blocks(
    blocks: list[PhysicalTokenBlock],
    context_len: int,
    block_size: int,
    sink_tokens: int,
    window_tokens: int,
) -> list[PhysicalTokenBlock]:
    """Keep sink prefix blocks and newest window blocks; drop the middle."""
    if not blocks or block_size <= 0:
        return []
    kept = retained_tokens(context_len, sink_tokens, window_tokens)
    if kept <= 0:
        return []
    if kept >= context_len:
        return list(blocks)
    sink = min(max(0, int(sink_tokens)), context_len)
    window = kept - sink
    sink_blocks = (sink + block_size - 1) // block_size if sink else 0
    window_blocks = (window + block_size - 1) // block_size if window else 0
    n = len(blocks)
    if sink_blocks + window_blocks >= n:
        return list(blocks)
    keep_idx = set(range(min(sink_blocks, n)))
    keep_idx.update(range(max(0, n - window_blocks), n))
    return [blocks[i] for i in sorted(keep_idx)]


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
        kv_cache_manager: KVCacheManager | None = None,
    ) -> None:
        self.device = device
        self.block_size: int = block_size
        self.num_blocks: int = int(num_blocks)
        self.kv_cache_manager = kv_cache_manager
        self.pool = BlockPool(device, block_size, num_blocks)
        self.free_blocks = self.pool.free_blocks

    def allocate(self) -> PhysicalTokenBlock:
        if not self.get_num_free_blocks():
            raise OutOfBlocksError(
                f"out of {self.device.name} memory; no free blocks are available"
            )
        block = self.pool.pop_free_block()
        if block.cached:
            if self.kv_cache_manager is None:
                block.clear_cache_metadata()
            else:
                self.kv_cache_manager.evict(block)
        if block.ref_count != 0:
            raise SimulationStateError(
                f"{self.device.name} block {block.block_number} is free but "
                + f"has ref_count {block.ref_count}"
            )
        block.ref_count += 1
        return block

    def acquire(self, block: PhysicalTokenBlock) -> None:
        self.pool.validate_owned(block)
        if block.ref_count == 0:
            self.pool.remove_from_free_queue(block)
        block.ref_count += 1
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.touch(block)

    def free(self, block: PhysicalTokenBlock | None = None) -> None:
        if block is None:
            raise SimulationStateError(
                "BlockAllocator.free() requires a physical block"
            )
        self.pool.validate_owned(block)
        if block.ref_count <= 0:
            raise SimulationStateError(
                f"block {block.block_number} ref_count is already {block.ref_count}"
            )
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.release_block(block, self.free_blocks)
        else:
            block.ref_count -= 1
            if block.ref_count == 0:
                block.clear_cache_metadata()
                self.pool.append_free_block(block)

    def get_num_free_blocks(self) -> int:
        return self.pool.get_num_free_blocks()

    def get_num_allocated_blocks(self) -> int:
        return self.num_blocks - self.get_num_free_blocks()

    def set_num_free_blocks(self, block_num):
        self.pool.set_num_free_blocks(block_num)
        self.free_blocks = self.pool.free_blocks

    def get_status(self) -> Tuple[int, int, int]:
        return (
            self.get_num_free_blocks(),
            self.get_num_allocated_blocks(),
            self.num_blocks,
        )


class BlockManager:
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        model: str = "unknown",
        watermark: float = 0.01,
        gpu_frac: float = 1.0,
        sink_tokens: int = 0,
        window_tokens: int = 0,
        streaming_attention: bool = False,
        cache_tokens: int = 0,
    ):
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.gpu_frac = gpu_frac
        self.sink_tokens = max(0, int(sink_tokens))
        self.window_tokens = max(0, int(window_tokens))
        self.streaming_attention = bool(streaming_attention)
        self.cache_tokens = max(0, int(cache_tokens))

        self.block_table = BlockTable()
        self.kv_cache_manager = KVCacheManager(block_size=block_size, model=model)

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(
            Device.GPU,
            block_size,
            num_gpu_blocks,
            kv_cache_manager=self.kv_cache_manager,
        )
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size, num_cpu_blocks)
        self._reserved_gpu_blocks: list[PhysicalTokenBlock] = []

    def _constrain_gpu_occupancy(self) -> bool:
        return self.gpu_frac < 1.0 or self.streaming_attention

    def _occupancy_frac(self, req: Request) -> float:
        """Prefill/recompute keep the full context on HBM; decode uses gpu_frac."""
        if req.is_prefill or getattr(req, "needs_recompute", False):
            return 1.0
        return self.gpu_frac

    def _gpu_target_blocks(self, req: Request) -> int:
        if self.streaming_attention and not (
            req.is_prefill or getattr(req, "needs_recompute", False)
        ):
            tokens = retained_tokens(
                req.context_len, self.sink_tokens, self.window_tokens
            )
            if tokens <= 0 or self.block_size <= 0:
                return 0
            return (tokens + self.block_size - 1) // self.block_size
        decode = self._occupancy_frac(req) < 1.0
        return gpu_resident_blocks(
            req.context_len,
            self.block_size,
            self._occupancy_frac(req),
            self.sink_tokens if decode else 0,
            cache_tokens=self.cache_tokens if decode else 0,
        )

    def can_allocate(self, req: Request) -> bool:
        if self._constrain_gpu_occupancy():
            num_required_blocks = self._gpu_target_blocks(req)
            num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
            return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks
        plan = self.kv_cache_manager.plan_reuse(req)
        num_required_blocks = plan.miss_block_count
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        # Zero-ref hits sit in the free queue until acquired, so they must not
        # be counted as capacity available for miss/private allocations.
        num_free_gpu_blocks -= plan.zero_ref_hit_blocks
        return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, req: Request):
        # Allocate new physical token blocks that will store the prompt&generated tokens.
        if self._constrain_gpu_occupancy():
            needed = self._gpu_target_blocks(req)
            blocks = [self.gpu_allocator.allocate() for _ in range(needed)]
            self.block_table.add_blocks(req.id, blocks)
            self._sync_request_blocks(req, blocks)
            return
        plan = self.kv_cache_manager.plan_reuse(req)
        self.kv_cache_manager.apply_plan(req, plan)

        blocks = list(plan.hit_blocks)
        for block in plan.hit_blocks:
            self.gpu_allocator.acquire(block)

        for _ in range(plan.miss_block_count):
            block = self.gpu_allocator.allocate()
            blocks.append(block)
        self.block_table.add_blocks(req.id, blocks)
        self._sync_request_blocks(req, blocks)
        self._mark_input_blocks_pending(req, blocks, plan)

    def get_gpu_status(self) -> Tuple[int, int, int]:
        return self.gpu_allocator.get_status()

    def get_num_blocks(self, req: Request) -> int:
        return self._num_blocks(req)

    def _num_blocks(self, req: Request) -> int:
        return self.block_table.get_num_blocks(req.id) or req.num_physical_token_blocks

    def free(self, req: Request):
        self.commit_finished_cache(req)
        self.release_request_blocks(req)

    def commit_finished_cache(self, req: Request) -> None:
        self._commit_input_cache(req)
        self.kv_cache_manager.register_output_blocks(
            req,
            self.block_table.get_blocks(req.id),
        )

    def release_request_blocks(self, req: Request) -> None:
        blocks = self.block_table.pop_blocks(req.id) or list(req._physical_token_blocks)
        for block in blocks:
            self._free_block(block)
        req._physical_token_blocks.clear()
        self.kv_cache_manager.forget_request(req.id)

    def _append_target_blocks(self, req: Request) -> int:
        if self._constrain_gpu_occupancy():
            return self._gpu_target_blocks(req)
        return req.num_logical_token_blocks

    def trim_to_gpu_target(self, req: Request) -> int:
        """Free GPU blocks down to decode residency. Spill I/O is not charged."""
        if not self._constrain_gpu_occupancy():
            return 0
        if req.is_prefill or getattr(req, "needs_recompute", False):
            return 0
        target = self._gpu_target_blocks(req)
        blocks = self.block_table.get_blocks(req.id)
        if len(blocks) <= target:
            return 0
        if self.streaming_attention:
            keep = _streaming_keep_blocks(
                blocks,
                req.context_len,
                self.block_size,
                self.sink_tokens,
                self.window_tokens,
            )
            spill = [block for block in blocks if block not in set(keep)]
        else:
            spill = blocks[: len(blocks) - target]
            keep = blocks[len(blocks) - target :]
        for block in spill:
            self._free_block(block)
        self.block_table.set_blocks(req.id, keep)
        self._sync_request_blocks(req, keep)
        return len(spill)

    def can_append_slot(self, req: Request, reserved_blocks: int = 0) -> bool:
        required_blocks = int(
            self.block_table.get_num_blocks(req.id) < self._append_target_blocks(req)
        )
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        return num_free_gpu_blocks >= required_blocks + reserved_blocks

    def append_slot(self, req: Request):
        """Allocate a physical slot for a new token."""
        if self.block_table.get_num_blocks(req.id) < self._append_target_blocks(req):
            # The request has a new logical block, which
            # happens in Scheduler.update_output_tokens().
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            self.block_table.add_block(req.id, block)
            req._append_physical_block(block)

    def get_request_block_counts(self, requests: list[Request]) -> dict[int, int]:
        return {req.id: self._num_blocks(req) for req in requests}

    def reserve_gpu_blocks(self, num_blocks: int):
        allocated: list[PhysicalTokenBlock] = []
        for _ in range(num_blocks):
            try:
                block = self.gpu_allocator.allocate()
            except Exception:
                for block in allocated:
                    self.gpu_allocator.free(block)
                raise
            allocated.append(block)
        self._reserved_gpu_blocks.extend(allocated)

    def release_reserved_blocks(self, num_blocks: int | None = None):
        if num_blocks is None:
            num_blocks = len(self._reserved_gpu_blocks)
        if num_blocks < 0:
            raise SimulationStateError(
                f"cannot release a negative block count: {num_blocks}"
            )
        if num_blocks > len(self._reserved_gpu_blocks):
            raise SimulationStateError(
                f"cannot release {num_blocks} reserved GPU blocks; "
                + f"only {len(self._reserved_gpu_blocks)} are reserved"
            )
        for _ in range(num_blocks):
            self.gpu_allocator.free(self._reserved_gpu_blocks.pop(0))

    def _free_block(self, block: PhysicalTokenBlock) -> None:
        if block.device == Device.GPU:
            self.gpu_allocator.free(block)
        elif block.device == Device.CPU:
            self.cpu_allocator.free(block)
        else:
            raise SimulationStateError(f"unknown block device {block.device}")

    @staticmethod
    def _validate_blocks_device(
        req: Request,
        blocks: list[PhysicalTokenBlock],
        expected_device: Device,
        operation: str,
    ) -> None:
        for block in blocks:
            if block.device != expected_device:
                raise SimulationStateError(
                    f"request {req.id} has {block.device.name} block {block} "
                    + f"during {operation}; expected {expected_device.name}"
                )

    @staticmethod
    def _sync_request_blocks(req: Request, blocks: list[PhysicalTokenBlock]) -> None:
        req._physical_token_blocks.clear()
        for block in blocks:
            req._append_physical_block(block)

    def _mark_input_blocks_pending(
        self,
        req: Request,
        blocks: list[PhysicalTokenBlock],
        plan: PrefixReusePlan,
    ) -> None:
        if not plan.input_keys:
            req.input_cache_keys = []
            return
        req.input_cache_keys = list(plan.input_keys)
        for index, key in enumerate(plan.input_keys):
            if index < plan.hit_block_count or index >= len(blocks):
                continue
            block = blocks[index]
            block.block_hash = key
            block.is_full = True
            block.cached = False

    def commit_input_cache(self, req: Request) -> None:
        self._commit_input_cache(req)

    def _commit_input_cache(self, req: Request) -> None:
        if req.input_cache_committed:
            return
        blocks = self.block_table.get_blocks(req.id)
        input_keys = getattr(req, "input_cache_keys", [])
        if input_keys:
            self.kv_cache_manager.register_blocks(blocks[: len(input_keys)], input_keys)
        req.input_cache_committed = True
