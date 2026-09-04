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


def media_read_latency(bytes_: int, media: MediaReadConfig | None) -> float:
    if bytes_ <= 0 or media is None:
        return 0.0
    return media.read_latency_us / 1e6 + bytes_ / _GB / max(_EPS, media.read_bw_gbps)


def fetch_cost(
    context_len: int,
    config: WorkingSetConfig,
    size_per_token: int,
) -> FetchCost:
    split = split_context(context_len, config)
    size_per_token = max(0, int(size_per_token))
    dram_bytes = split.dram * size_per_token
    ssd_bytes = split.ssd * size_per_token
    latency = media_read_latency(dram_bytes, config.dram) + media_read_latency(
        ssd_bytes,
        config.ssd,
    )
    return FetchCost(
        latency=latency,
        dram_tokens=split.dram,
        ssd_tokens=split.ssd,
        dram_bytes=dram_bytes,
        ssd_bytes=ssd_bytes,
        split=split,
    )


def decode_fetch_for_requests(
    requests: list[object],
    config: WorkingSetConfig,
    size_per_token: int,
) -> FetchCost:
    """Sum per-request decode fetch costs for the scheduled batch."""
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
    for req in requests:
        if getattr(req, "is_prefill", False) or getattr(req, "needs_recompute", False):
            continue
        context_len = getattr(req, "prefill_len", 0) + getattr(req, "generation_idx", 0)
        cost = fetch_cost(context_len, config, size_per_token)
        latency += cost.latency
        dram_tokens += cost.dram_tokens
        ssd_tokens += cost.ssd_tokens
        dram_bytes += cost.dram_bytes
        ssd_bytes += cost.ssd_bytes
        last_split = cost.split
    return FetchCost(
        latency=latency,
        dram_tokens=dram_tokens,
        ssd_tokens=ssd_tokens,
        dram_bytes=dram_bytes,
        ssd_bytes=ssd_bytes,
        split=last_split,
    )
