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
    sink: int = 0

    @property
    def context_len(self) -> int:
        return self.gpu + self.dram + self.ssd


def retained_tokens(
    context_len: int,
    sink_tokens: int,
    window_tokens: int,
) -> int:
    """StreamingLLM live tokens: sink ∪ window, counted once if they overlap."""
    context_len = max(0, int(context_len))
    sink = min(max(0, int(sink_tokens)), context_len)
    window = max(0, int(window_tokens))
    return sink + min(window, context_len - sink)


def attention_tokens(context_len: int, config: WorkingSetConfig) -> int:
    """Decode attention tokens.

    StreamingLLM: sink ∪ window. Sparse+offload: sink ∪ window, or
    sink ∪ top-k when ``select_tokens`` is set.
    """
    if config.streaming_attention:
        return retained_tokens(
            context_len, config.sink_tokens, config.window_tokens
        )
    if config.sparse:
        return retained_tokens(
            context_len, config.sink_tokens, config.attended_span()
        )
    return max(0, int(context_len))


def gpu_resident_tokens(
    context_len: int,
    gpu_frac: float,
    sink_tokens: int = 0,
    window_tokens: int = 0,
    streaming_attention: bool = False,
    cache_tokens: int = 0,
) -> int:
    """GPU-resident tokens: ``split_context`` GPU count plus the selection
    page cache (``cache_tokens``, bounded by the cold hole)."""
    if streaming_attention:
        return retained_tokens(context_len, sink_tokens, window_tokens)
    context_len = max(0, int(context_len))
    tail = floor(context_len * gpu_frac)
    sink = min(max(0, int(sink_tokens)), context_len)
    if sink == 0:
        resident = tail
    elif sink + tail >= context_len:
        resident = context_len
    else:
        resident = sink + tail
    return min(context_len, resident + max(0, int(cache_tokens)))


def gpu_resident_blocks(
    context_len: int,
    block_size: int,
    gpu_frac: float,
    sink_tokens: int = 0,
    window_tokens: int = 0,
    streaming_attention: bool = False,
    cache_tokens: int = 0,
) -> int:
    """GPU KV blocks charged for ``context_len`` tokens at ``gpu_frac``."""
    if block_size <= 0:
        return 0
    tokens = gpu_resident_tokens(
        context_len,
        gpu_frac,
        sink_tokens,
        window_tokens,
        streaming_attention,
        cache_tokens,
    )
    if tokens <= 0:
        return 0
    return (tokens + block_size - 1) // block_size


def split_context(context_len: int, config: WorkingSetConfig) -> ContextSplit:
    """Split live context tokens across GPU / DRAM / SSD.

    ``sliding_window`` assigns the newest tokens to GPU, then DRAM, then SSD.
    ``static_fraction`` uses the same counts; range fields still follow that
    newest-to-oldest layout so tests can assert GPU residency of the tail.

    When ``config.streaming_attention``, only sink ∪ window stay on GPU;
    the middle is evicted (DRAM = SSD = 0).

    When ``config.sparse``, GPU is ``[0, sink) ∪ [S - tail, S)`` (sink is
    extra, not taken from the tail). DRAM is still ``floor(S × dram_frac)``
    of the remaining hole; SSD is the rest.
    """
    context_len = max(0, int(context_len))
    if config.streaming_attention:
        kept = retained_tokens(
            context_len, config.sink_tokens, config.window_tokens
        )
        sink = min(max(0, int(config.sink_tokens)), context_len)
        window = kept - sink
        return ContextSplit(
            gpu=kept,
            dram=0,
            ssd=0,
            gpu_start=context_len - window,
            dram_start=context_len - window,
            ssd_start=sink,
            sink=sink,
        )
    tail = floor(context_len * config.gpu_frac)
    dram = floor(context_len * config.dram_frac)
    if not config.sparse:
        gpu = tail
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

    sink = min(max(0, int(config.sink_tokens)), context_len)
    hole_start = sink
    hole_end = context_len - tail
    if hole_start >= hole_end:
        return ContextSplit(
            gpu=context_len,
            dram=0,
            ssd=0,
            gpu_start=0,
            dram_start=0,
            ssd_start=0,
            sink=sink,
        )
    gpu = sink + tail
    hole = hole_end - hole_start
    dram = min(dram, hole)
    ssd = hole - dram
    gpu_start = hole_end
    dram_start = gpu_start - dram
    return ContextSplit(
        gpu=gpu,
        dram=dram,
        ssd=ssd,
        gpu_start=gpu_start,
        dram_start=dram_start,
        ssd_start=sink,
        sink=sink,
    )
