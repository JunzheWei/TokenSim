from __future__ import annotations

from dataclasses import dataclass

from TokenSim.config.constants import _GB
from TokenSim.kv_working_set.config import (
    IO_SIZE_COALESCED,
    QD_LATENCY_KNEE,
    MediaReadConfig,
    WorkingSetConfig,
)
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
    dram_ios: int = 0
    ssd_ios: int = 0


def media_n_ios(bytes_: int, media: MediaReadConfig | None) -> int:
    if bytes_ <= 0 or media is None:
        return 0
    if media.io_size_bytes <= IO_SIZE_COALESCED:
        return 1
    return (int(bytes_) + media.io_size_bytes - 1) // media.io_size_bytes


def media_read_latency(bytes_: int, media: MediaReadConfig | None) -> float:
    if bytes_ <= 0 or media is None:
        return 0.0
    return media.read_latency_us / 1e6 + bytes_ / _GB / max(_EPS, media.read_bw_gbps)


def _interpolate_qd_latency_us(
    points: tuple[tuple[float, float], ...],
    qd: float,
) -> float:
    if not points:
        return 0.0
    if qd <= points[0][0]:
        return points[0][1]
    if qd >= points[-1][0]:
        return points[-1][1]
    for (q0, lat0), (q1, lat1) in zip(points, points[1:]):
        if q0 <= qd <= q1:
            if q1 == q0:
                return lat0
            return lat0 + (lat1 - lat0) * (qd - q0) / (q1 - q0)
    return points[-1][1]


def qd_latency_us(media: MediaReadConfig, qd: int) -> float:
    qd = max(1, int(qd))
    if media.qd_latency_us:
        return _interpolate_qd_latency_us(media.qd_latency_us, float(qd))
    if qd <= QD_LATENCY_KNEE:
        return media.read_latency_us
    return media.read_latency_us * (qd / QD_LATENCY_KNEE)


def media_queue_latency(
    bytes_: int,
    n_ios: int,
    media: MediaReadConfig | None,
) -> float:
    """Roofline: max(command drain, transfer). Prefill is not charged here."""
    if bytes_ <= 0 or media is None:
        return 0.0
    t_bw = bytes_ / _GB / max(_EPS, media.read_bw_gbps)
    n_ios = max(1, int(n_ios))
    qd = min(n_ios, media.qd_cap)
    lat_s = qd_latency_us(media, qd) / 1e6
    t_iops = n_ios * lat_s / qd
    return max(t_iops, t_bw)


def media_access_latency(
    bytes_: int,
    media: MediaReadConfig | None,
) -> float:
    if bytes_ <= 0 or media is None:
        return 0.0
    if not media.queueing_enabled():
        return media_read_latency(bytes_, media)
    return media_queue_latency(bytes_, media_n_ios(bytes_, media), media)


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
    dram_ios = media_n_ios(dram_bytes, config.dram)
    ssd_ios = media_n_ios(ssd_bytes, config.ssd)
    latency = media_access_latency(dram_bytes, config.dram) + media_access_latency(
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
        dram_ios=dram_ios,
        ssd_ios=ssd_ios,
    )


def decode_fetch_for_requests(
    requests: list[object],
    config: WorkingSetConfig,
    size_per_token: int,
) -> FetchCost:
    """Page-fault decode reads; share the SSD/DRAM queue when io_size > 0."""
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
    dram_tokens = 0
    ssd_tokens = 0
    dram_bytes = 0
    ssd_bytes = 0
    dram_ios = 0
    ssd_ios = 0
    dram_independent = 0.0
    ssd_independent = 0.0
    last_split = total.split
    last_fetched = 0
    for req in requests:
        if getattr(req, "is_prefill", False) or getattr(req, "needs_recompute", False):
            continue
        context_len = getattr(req, "prefill_len", 0) + getattr(req, "generation_idx", 0)
        fetched_end = int(getattr(req, "kv_ws_fetched_end", 0) or 0)
        cost = fetch_cost(context_len, config, size_per_token, fetched_end)
        setattr(req, "kv_ws_fetched_end", cost.next_fetched_end)
        dram_independent += media_access_latency(cost.dram_bytes, config.dram)
        ssd_independent += media_access_latency(cost.ssd_bytes, config.ssd)
        dram_tokens += cost.dram_tokens
        ssd_tokens += cost.ssd_tokens
        dram_bytes += cost.dram_bytes
        ssd_bytes += cost.ssd_bytes
        dram_ios += cost.dram_ios
        ssd_ios += cost.ssd_ios
        last_split = cost.split
        last_fetched = cost.next_fetched_end
    dram_q = config.dram is not None and config.dram.queueing_enabled()
    ssd_q = config.ssd is not None and config.ssd.queueing_enabled()
    t_dram = (
        media_queue_latency(dram_bytes, dram_ios, config.dram)
        if dram_q
        else dram_independent
    )
    t_ssd = (
        media_queue_latency(ssd_bytes, ssd_ios, config.ssd) if ssd_q else ssd_independent
    )
    return FetchCost(
        latency=t_dram + t_ssd,
        dram_tokens=dram_tokens,
        ssd_tokens=ssd_tokens,
        dram_bytes=dram_bytes,
        ssd_bytes=ssd_bytes,
        split=last_split,
        next_fetched_end=last_fetched,
        dram_ios=dram_ios,
        ssd_ios=ssd_ios,
    )
