from __future__ import annotations

from dataclasses import dataclass

from TokenSim.config.constants import _GB
from TokenSim.kv_working_set.config import MediaReadConfig, WorkingSetConfig
from TokenSim.kv_working_set.placement import ContextSplit, split_context

_EPS = 1e-9


@dataclass(frozen=True)
class FetchCost:
    latency: float
    dram_tokens: int
    ssd_tokens: int
    dram_bytes: int
    ssd_bytes: int
    split: ContextSplit
    next_fetched_end: int = 0


def media_read_latency(bytes_: int, media: MediaReadConfig | None) -> float:
    if bytes_ <= 0 or media is None:
        return 0.0
    return media.read_latency_us / 1e6 + bytes_ / _GB / max(_EPS, media.read_bw_gbps)


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def fetch_cost(
    context_len: int,
    config: WorkingSetConfig,
    size_per_token: int,
    fetched_end: int = 0,
) -> FetchCost:
    """Charge DRAM/SSD reads only for tokens not yet faulted in.

    Storage covers ``[0, gpu_start)``. Tokens in ``[0, fetched_end)`` already
    paid I/O. Later decode steps only pay for the window sliding into storage.
    """
    split = split_context(context_len, config)
    size_per_token = max(0, int(size_per_token))
    fetched_end = max(0, int(fetched_end))
    miss_start = min(fetched_end, split.gpu_start)
    miss_end = split.gpu_start
    dram_tokens = _overlap(miss_start, miss_end, split.dram_start, split.gpu_start)
    ssd_tokens = _overlap(miss_start, miss_end, split.ssd_start, split.dram_start)
    dram_bytes = dram_tokens * size_per_token
    ssd_bytes = ssd_tokens * size_per_token
    latency = media_read_latency(dram_bytes, config.dram) + media_read_latency(
        ssd_bytes,
        config.ssd,
    )
    return FetchCost(
        latency=latency,
        dram_tokens=dram_tokens,
        ssd_tokens=ssd_tokens,
        dram_bytes=dram_bytes,
        ssd_bytes=ssd_bytes,
        split=split,
        next_fetched_end=max(fetched_end, split.gpu_start),
    )


def decode_fetch_for_requests(
    requests: list[object],
    config: WorkingSetConfig,
    size_per_token: int,
) -> FetchCost:
    """Sum per-request decode page-fault costs and advance each watermark."""
    total = FetchCost(
        latency=0.0,
        dram_tokens=0,
        ssd_tokens=0,
        dram_bytes=0,
        ssd_bytes=0,
        split=split_context(0, config),
    )
    if not config.enabled or not requests:
        return total
    latency = 0.0
    dram_tokens = 0
    ssd_tokens = 0
    dram_bytes = 0
    ssd_bytes = 0
    last_split = total.split
    last_fetched = 0
    for req in requests:
        if getattr(req, "is_prefill", False) or getattr(req, "needs_recompute", False):
            continue
        context_len = getattr(req, "prefill_len", 0) + getattr(req, "generation_idx", 0)
        fetched_end = int(getattr(req, "kv_ws_fetched_end", 0) or 0)
        cost = fetch_cost(context_len, config, size_per_token, fetched_end)
        setattr(req, "kv_ws_fetched_end", cost.next_fetched_end)
        latency += cost.latency
        dram_tokens += cost.dram_tokens
        ssd_tokens += cost.ssd_tokens
        dram_bytes += cost.dram_bytes
        ssd_bytes += cost.ssd_bytes
        last_split = cost.split
        last_fetched = cost.next_fetched_end
    return FetchCost(
        latency=latency,
        dram_tokens=dram_tokens,
        ssd_tokens=ssd_tokens,
        dram_bytes=dram_bytes,
        ssd_bytes=ssd_bytes,
        split=last_split,
        next_fetched_end=last_fetched,
    )
