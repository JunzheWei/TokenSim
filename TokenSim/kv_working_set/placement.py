from __future__ import annotations

from dataclasses import dataclass
from math import floor

from TokenSim.kv_working_set.config import WorkingSetConfig


@dataclass(frozen=True)
class ContextSplit:
    gpu: int
    dram: int
    ssd: int
    gpu_start: int
    dram_start: int
    ssd_start: int

    @property
    def context_len(self) -> int:
        return self.gpu + self.dram + self.ssd


def gpu_resident_tokens(context_len: int, gpu_frac: float) -> int:
    """Newest tokens kept on GPU; matches ``split_context`` GPU count."""
    return floor(max(0, int(context_len)) * gpu_frac)


def gpu_resident_blocks(context_len: int, block_size: int, gpu_frac: float) -> int:
    """GPU KV blocks charged for ``context_len`` tokens at ``gpu_frac``."""
    if block_size <= 0:
        return 0
    tokens = gpu_resident_tokens(context_len, gpu_frac)
    if tokens <= 0:
        return 0
    return (tokens + block_size - 1) // block_size


def split_context(context_len: int, config: WorkingSetConfig) -> ContextSplit:
    """Split live context tokens across GPU / DRAM / SSD.

    ``sliding_window`` assigns the newest tokens to GPU, then DRAM, then SSD.
    ``static_fraction`` uses the same counts; range fields still follow that
    newest-to-oldest layout so tests can assert GPU residency of the tail.
    """
    context_len = max(0, int(context_len))
    gpu = floor(context_len * config.gpu_frac)
    dram = floor(context_len * config.dram_frac)
    ssd = context_len - gpu - dram
    gpu_start = context_len - gpu
    dram_start = gpu_start - dram
    return ContextSplit(
        gpu=gpu,
        dram=dram,
        ssd=ssd,
        gpu_start=gpu_start,
        dram_start=dram_start,
        ssd_start=0,
    )
