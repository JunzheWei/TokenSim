#!/usr/bin/env python3
"""GQA Poisson QPS-knee sweep: gpu_frac 100% → 10% in 5% steps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kv_working_set_test_report"))

from bench_lib import (  # noqa: E402
    IO_SIZE_128K,
    SSD_SLC,
    base_hier,
    cache_gpu_blocks,
    find_config_knee,
    write_cfg,
)

OUT = Path(__file__).resolve().parent
MODEL = ROOT / "data/psla/llama-70b-gqa.json"
GPU_BLOCKS = cache_gpu_blocks("LLaMa2-70B-GQA")
DRAM_SHARE = 5.0 / 7.0


def split_fracs(gpu_frac: float) -> tuple[float, float, float]:
    off = 1.0 - gpu_frac
    if off <= 0.0:
        return gpu_frac, 0.0, 0.0
    dram = off * DRAM_SHARE
    ssd = 1.0 - gpu_frac - dram
    return gpu_frac, dram, ssd


def main() -> int:
    summary = []
    percents = list(range(100, 5, -5))
    print(f"GQA gpu_blocks={GPU_BLOCKS}", flush=True)
    for pct in percents:
        gpu = pct / 100.0
        g, d, s = split_fracs(gpu)
        tag = f"gpu_{pct:03d}"
        cfg = base_hier(gpu_frac=g, dram_frac=d, ssd_frac=s, ssd=dict(SSD_SLC))
        if s <= 0.0:
            cfg.pop("ssd", None)
            cfg["ssd_frac"] = 0.0
        cfg_path = write_cfg(OUT / f"{tag}.json", cfg)
        print(f"=== {tag} gpu={g:.2f} dram={d:.4f} ssd={s:.4f} ===", flush=True)
        row = find_config_knee(
            tag,
            OUT / tag,
            cfg_path,
            {
                "gpu_frac": g,
                "dram_frac": d,
                "ssd_frac": s,
                "io_size_bytes": 0 if s <= 0.0 else IO_SIZE_128K,
            },
            model=MODEL,
            gpu_blocks=GPU_BLOCKS,
        )
        summary.append(row)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
