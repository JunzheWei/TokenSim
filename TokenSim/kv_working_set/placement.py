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
