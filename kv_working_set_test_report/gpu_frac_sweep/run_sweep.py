#!/usr/bin/env python3
"""Burst sweep: gpu_frac 100% → 10% in 5% steps, occupancy-constrained."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
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
    for pct in percents:
        gpu = pct / 100.0
        g, d, s = split_fracs(gpu)
        tag = f"gpu_{pct:03d}"
        cfg_path = OUT / f"{tag}.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "placement": "sliding_window",
                    "gpu_frac": g,
                    "dram_frac": d,
                    "ssd_frac": s,
                    "overlap": "blocking",
                    "dram": {"read_latency_us": 2.0, "read_bw_gbps": 50.0},
                    "ssd": {"read_latency_us": 100.0, "read_bw_gbps": 7.0},
                    "hbm": {"read_latency_us": 0.0, "read_bw_gbps": 2000.0},
                },
                indent=2,
            )
            + "\n"
        )
        results_dir = OUT / tag
        cmd = [
            sys.executable,
            str(ROOT / "benchmark.py"),
            "--batching",
            "paged-attn",
            "--qps",
            "10",
            "--distribution",
            "burst",
            "--cluster",
            str(ROOT / "data/clusters/1_h200/h1.json"),
            "--model",
            str(ROOT / "data/psla/llama-70b.json"),
            "--verbose",
            "none",
            "--kv_working_set_config",
            str(cfg_path),
            "--results_path",
            str(results_dir),
        ]
        print(f"=== {tag} gpu={g:.2f} dram={d:.4f} ssd={s:.4f} ===", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, check=False)
        if proc.returncode != 0:
            print(f"FAILED {tag} rc={proc.returncode}", flush=True)
            return proc.returncode
        result = json.loads((results_dir / "result_inf.json").read_text())
        row = {
            "gpu_frac": g,
            "dram_frac": d,
            "ssd_frac": s,
            "duration": result["duration"],
            "output_token_ps": result["output_token_ps"],
            "output_qps": result["output_qps"],
            "ttft_p50": result["prefill_time"]["p50"],
            "ttft_p99": result["prefill_time"]["p99"],
            "ttft_min": result["prefill_time"].get("min", None),
            "tpot_p50": result["decode_time"]["p50"],
            "tpot_p99": result["decode_time"]["p99"],
            "preemption_count": result["preemption_count"],
            "recomputation_count": result["recomputation_count"],
            "recomputed_tokens": result["recomputed_tokens"],
            "recompute_service_time": result["recompute_service_time"],
            "kv_ws_fetch_latency": result["kv_ws_fetch_latency"],
            "kv_ws_dram_read_tokens": result["kv_ws_dram_read_tokens"],
            "kv_ws_ssd_read_tokens": result["kv_ws_ssd_read_tokens"],
        }
        summary.append(row)
        print(
            f"  tok/s={row['output_token_ps']:.1f} tpot_p50={row['tpot_p50']*1e3:.1f}ms "
            f"ttft_p50={row['ttft_p50']:.1f}s preempt={row['preemption_count']} "
            f"fetch={row['kv_ws_fetch_latency']:.2f}s",
            flush=True,
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
