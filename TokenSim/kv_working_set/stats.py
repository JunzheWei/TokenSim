from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from TokenSim.kv_working_set.config import WorkingSetConfig
from TokenSim.kv_working_set.fetch import FetchCost


@dataclass
class WorkingSetStats:
    kv_ws_enabled: bool = False
    kv_ws_placement: str = "sliding_window"
    kv_ws_gpu_frac: float = 1.0
    kv_ws_dram_frac: float = 0.0
    kv_ws_ssd_frac: float = 0.0
    kv_ws_fetch_latency: float = 0.0
    kv_ws_dram_read_bytes: int = 0
    kv_ws_ssd_read_bytes: int = 0
    kv_ws_dram_read_tokens: int = 0
    kv_ws_ssd_read_tokens: int = 0
    kv_ws_dram_ios: int = 0
    kv_ws_ssd_ios: int = 0

    @classmethod
    def from_config(cls, config: WorkingSetConfig | None) -> "WorkingSetStats":
        if config is None or not config.enabled:
            return cls()
        return cls(
            kv_ws_enabled=True,
            kv_ws_placement=config.placement,
            kv_ws_gpu_frac=config.gpu_frac,
            kv_ws_dram_frac=config.dram_frac,
            kv_ws_ssd_frac=config.ssd_frac,
        )

    def record_fetch(self, cost: FetchCost) -> None:
        self.kv_ws_fetch_latency += cost.latency
        self.kv_ws_dram_read_bytes += cost.dram_bytes
        self.kv_ws_ssd_read_bytes += cost.ssd_bytes
        self.kv_ws_dram_read_tokens += cost.dram_tokens
        self.kv_ws_ssd_read_tokens += cost.ssd_tokens
        self.kv_ws_dram_ios += cost.dram_ios
        self.kv_ws_ssd_ios += cost.ssd_ios

    def aggregate(self, other: "WorkingSetStats") -> "WorkingSetStats":
        return WorkingSetStats(
            kv_ws_enabled=self.kv_ws_enabled or other.kv_ws_enabled,
            kv_ws_placement=(
                self.kv_ws_placement
                if self.kv_ws_enabled
                else other.kv_ws_placement
            ),
            kv_ws_gpu_frac=(
                self.kv_ws_gpu_frac if self.kv_ws_enabled else other.kv_ws_gpu_frac
            ),
            kv_ws_dram_frac=(
                self.kv_ws_dram_frac if self.kv_ws_enabled else other.kv_ws_dram_frac
            ),
            kv_ws_ssd_frac=(
                self.kv_ws_ssd_frac if self.kv_ws_enabled else other.kv_ws_ssd_frac
            ),
            kv_ws_fetch_latency=self.kv_ws_fetch_latency + other.kv_ws_fetch_latency,
            kv_ws_dram_read_bytes=self.kv_ws_dram_read_bytes
            + other.kv_ws_dram_read_bytes,
            kv_ws_ssd_read_bytes=self.kv_ws_ssd_read_bytes + other.kv_ws_ssd_read_bytes,
            kv_ws_dram_read_tokens=self.kv_ws_dram_read_tokens
            + other.kv_ws_dram_read_tokens,
            kv_ws_ssd_read_tokens=self.kv_ws_ssd_read_tokens
            + other.kv_ws_ssd_read_tokens,
            kv_ws_dram_ios=self.kv_ws_dram_ios + other.kv_ws_dram_ios,
            kv_ws_ssd_ios=self.kv_ws_ssd_ios + other.kv_ws_ssd_ios,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kv_ws_enabled": self.kv_ws_enabled,
            "kv_ws_placement": self.kv_ws_placement,
            "kv_ws_gpu_frac": self.kv_ws_gpu_frac,
            "kv_ws_dram_frac": self.kv_ws_dram_frac,
            "kv_ws_ssd_frac": self.kv_ws_ssd_frac,
            "kv_ws_fetch_latency": self.kv_ws_fetch_latency,
            "kv_ws_dram_read_bytes": self.kv_ws_dram_read_bytes,
            "kv_ws_ssd_read_bytes": self.kv_ws_ssd_read_bytes,
            "kv_ws_dram_read_tokens": self.kv_ws_dram_read_tokens,
            "kv_ws_ssd_read_tokens": self.kv_ws_ssd_read_tokens,
            "kv_ws_dram_ios": self.kv_ws_dram_ios,
            "kv_ws_ssd_ios": self.kv_ws_ssd_ios,
        }
