from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from TokenSim.errors import ConfigurationError

SUPPORTED_PLACEMENTS = {"sliding_window", "static_fraction"}
SUPPORTED_OVERLAPS = {"blocking", "layer_prefetch"}
FRACTION_SUM_TOLERANCE = 1e-6
IO_SIZE_COALESCED = 0
IO_SIZE_128K = 131072
QD_LATENCY_KNEE = 32
DEFAULT_QD_CAP = 32
DEFAULT_PCIE_BW_GBPS = 50.0


@dataclass(frozen=True)
class MediaReadConfig:
    read_latency_us: float
    read_bw_gbps: float
    io_size_bytes: int = IO_SIZE_COALESCED
    qd_cap: int = DEFAULT_QD_CAP
    qd_latency_us: tuple[tuple[float, float], ...] = ()
    write_latency_us: float | None = None
    write_bw_gbps: float | None = None

    def queueing_enabled(self) -> bool:
        return self.io_size_bytes > IO_SIZE_COALESCED

    def effective_write_latency_us(self) -> float:
        if self.write_latency_us is None:
            return self.read_latency_us
        return self.write_latency_us

    def effective_write_bw_gbps(self) -> float:
        if self.write_bw_gbps is None:
            return self.read_bw_gbps
        return self.write_bw_gbps


@dataclass(frozen=True)
class WorkingSetConfig:
    enabled: bool = False
    placement: str = "sliding_window"
    gpu_frac: float = 1.0
    dram_frac: float = 0.0
    ssd_frac: float = 0.0
    overlap: str = "blocking"
    pcie_bw_gbps: float = DEFAULT_PCIE_BW_GBPS
    dram: MediaReadConfig | None = None
    ssd: MediaReadConfig | None = None
    # Optional metadata only. Decode/prefill HBM traffic uses TransformerRoofline
    # (H200 BW_TBs=4.8). fetch_cost / spill_cost never read this field.
    hbm: MediaReadConfig | None = None
    # Keep full KV on the hierarchy; decode attends to sink ∪ window only.
    sparse: bool = False
    # Evict the middle; GPU holds only sink ∪ window (no DRAM/SSD).
    streaming_attention: bool = False
    sink_tokens: int = 4
    window_tokens: int = 256
    # Importance-based page selection (Quest-style top-k), statistical model:
    # decode attends sink ∪ k selected tokens; selection is uniform over the
    # non-sink context, so the cold share of k is fetched each step.
    # 0 keeps the sliding-window selection. Requires ``sparse``.
    select_tokens: int = 0
    # Plan C on top of select_tokens: a per-request GPU page cache holding
    # recently selected cold pages (extra GPU residency beyond gpu_frac), and
    # the fraction of each step's selection that repeats the previous step.
    # Cold hit ratio = reuse + (1 - reuse) × min(1, cache / hole).
    select_cache_tokens: int = 0
    select_reuse: float = 0.0

    def attended_span(self) -> int:
        """Tokens attended beyond sink: top-k selection if set, else window."""
        return self.select_tokens if self.select_tokens > 0 else self.window_tokens

    def __post_init__(self) -> None:
        if self.placement not in SUPPORTED_PLACEMENTS:
            raise ConfigurationError(
                "unsupported working-set placement "
                + f"{self.placement!r}; expected one of "
                + f"{sorted(SUPPORTED_PLACEMENTS)}"
            )
        if self.overlap not in SUPPORTED_OVERLAPS:
            raise ConfigurationError(
                "unsupported working-set overlap "
                + f"{self.overlap!r}; expected one of "
                + f"{sorted(SUPPORTED_OVERLAPS)}"
            )
        if self.pcie_bw_gbps <= 0.0:
            raise ConfigurationError(
                f"pcie_bw_gbps must be > 0, got {self.pcie_bw_gbps}"
            )
        for name in ("gpu_frac", "dram_frac", "ssd_frac"):
            value = getattr(self, name)
            if value < 0.0 or value > 1.0:
                raise ConfigurationError(f"{name} must be in [0, 1], got {value}")
        frac_sum = self.gpu_frac + self.dram_frac + self.ssd_frac
        if abs(frac_sum - 1.0) > FRACTION_SUM_TOLERANCE:
            raise ConfigurationError(
                "gpu_frac + dram_frac + ssd_frac must sum to 1, "
                + f"got {frac_sum}"
            )
        self._require_media("dram", self.dram_frac, self.dram)
        self._require_media("ssd", self.ssd_frac, self.ssd)
        if self.hbm is not None:
            _validate_media("hbm", self.hbm)
        if self.sink_tokens < 0:
            raise ConfigurationError(
                f"sink_tokens must be >= 0, got {self.sink_tokens}"
            )
        if self.window_tokens < 0:
            raise ConfigurationError(
                f"window_tokens must be >= 0, got {self.window_tokens}"
            )
        if self.select_tokens < 0:
            raise ConfigurationError(
                f"select_tokens must be >= 0, got {self.select_tokens}"
            )
        if self.select_tokens > 0 and (not self.sparse or self.streaming_attention):
            raise ConfigurationError(
                "select_tokens requires sparse=true and streaming_attention=false"
            )
        if self.select_cache_tokens < 0:
            raise ConfigurationError(
                f"select_cache_tokens must be >= 0, got {self.select_cache_tokens}"
            )
        if not 0.0 <= self.select_reuse < 1.0:
            raise ConfigurationError(
                f"select_reuse must be in [0, 1), got {self.select_reuse}"
            )
        if (self.select_cache_tokens > 0 or self.select_reuse > 0.0) and (
            self.select_tokens <= 0
        ):
            raise ConfigurationError(
                "select_cache_tokens / select_reuse require select_tokens > 0"
            )
        if 0 < self.select_cache_tokens < self.select_tokens:
            raise ConfigurationError(
                "select_cache_tokens must hold one step's selection "
                f"(>= select_tokens={self.select_tokens}), got {self.select_cache_tokens}"
            )

    def any_queueing(self) -> bool:
        return any(
            media is not None and media.queueing_enabled()
            for media in (self.dram, self.ssd)
        )

    @staticmethod
    def _require_media(
        name: str,
        frac: float,
        media: MediaReadConfig | None,
    ) -> None:
        if frac <= 0.0:
            return
        if media is None:
            raise ConfigurationError(
                f"{name} media config is required when {name}_frac > 0"
            )
        _validate_media(name, media)

    @classmethod
    def disabled(cls) -> "WorkingSetConfig":
        return cls(enabled=False, gpu_frac=1.0, dram_frac=0.0, ssd_frac=0.0)

    @classmethod
    def from_file(cls, filename: str | Path) -> "WorkingSetConfig":
        payload = json.loads(Path(filename).read_text())
        if not isinstance(payload, dict):
            raise ConfigurationError("working-set config must be a JSON object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkingSetConfig":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            placement=str(payload.get("placement", "sliding_window")),
            gpu_frac=float(payload.get("gpu_frac", 1.0)),
            dram_frac=float(payload.get("dram_frac", 0.0)),
            ssd_frac=float(payload.get("ssd_frac", 0.0)),
            overlap=str(payload.get("overlap", "blocking")),
            pcie_bw_gbps=float(
                payload.get("pcie_bw_gbps", DEFAULT_PCIE_BW_GBPS)
            ),
            dram=_parse_media(payload.get("dram")),
            ssd=_parse_media(payload.get("ssd")),
            hbm=_parse_media(payload.get("hbm")),
            sparse=bool(payload.get("sparse", False)),
            streaming_attention=bool(payload.get("streaming_attention", False)),
            sink_tokens=int(payload.get("sink_tokens", 4)),
            window_tokens=int(payload.get("window_tokens", 256)),
            select_tokens=int(payload.get("select_tokens", 0)),
            select_cache_tokens=int(payload.get("select_cache_tokens", 0)),
            select_reuse=float(payload.get("select_reuse", 0.0)),
        )


def _parse_media(value: Any) -> MediaReadConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigurationError("media config must be an object")
    if "read_latency_us" not in value or "read_bw_gbps" not in value:
        raise ConfigurationError(
            "media config requires read_latency_us and read_bw_gbps"
        )
    write_latency = value.get("write_latency_us")
    write_bw = value.get("write_bw_gbps")
    return MediaReadConfig(
        read_latency_us=float(value["read_latency_us"]),
        read_bw_gbps=float(value["read_bw_gbps"]),
        io_size_bytes=int(value.get("io_size_bytes", IO_SIZE_COALESCED)),
        qd_cap=int(value.get("qd_cap", DEFAULT_QD_CAP)),
        qd_latency_us=_parse_qd_latency(value.get("qd_latency_us")),
        write_latency_us=(
            None if write_latency is None else float(write_latency)
        ),
        write_bw_gbps=None if write_bw is None else float(write_bw),
    )


def _parse_qd_latency(value: Any) -> tuple[tuple[float, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigurationError("qd_latency_us must be a list of [qd, latency_us]")
    points: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ConfigurationError(
                "qd_latency_us entries must be [qd, latency_us] pairs"
            )
        qd = float(item[0])
        lat = float(item[1])
        if qd <= 0.0:
            raise ConfigurationError("qd_latency_us qd must be > 0")
        if lat < 0.0:
            raise ConfigurationError("qd_latency_us latency must be >= 0")
        points.append((qd, lat))
    points.sort(key=lambda pair: pair[0])
    return tuple(points)


def _validate_media(name: str, media: MediaReadConfig) -> None:
    if media.read_latency_us < 0.0:
        raise ConfigurationError(f"{name} read_latency_us must be >= 0")
    if media.read_bw_gbps <= 0.0:
        raise ConfigurationError(f"{name} read_bw_gbps must be > 0")
    if media.io_size_bytes < IO_SIZE_COALESCED:
        raise ConfigurationError(f"{name} io_size_bytes must be >= 0")
    if media.qd_cap < 1:
        raise ConfigurationError(f"{name} qd_cap must be >= 1")
    if media.write_latency_us is not None and media.write_latency_us < 0.0:
        raise ConfigurationError(f"{name} write_latency_us must be >= 0")
    if media.write_bw_gbps is not None and media.write_bw_gbps <= 0.0:
        raise ConfigurationError(f"{name} write_bw_gbps must be > 0")
