from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from TokenSim.kv_working_set.knee import little_concurrency, search_knee_qps
from TokenSim.kv_working_set.placement import gpu_resident_blocks

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "data/clusters/1_h200/h1.json"
MODEL = ROOT / "data/psla/llama-70b.json"
WATERMARK_BLOCKS = 5
GPU_BLOCKS = 518
AVAIL_GPU_BLOCKS = GPU_BLOCKS - WATERMARK_BLOCKS
IO_SIZE_128K = 131072
DRAM = {"read_latency_us": 2.0, "read_bw_gbps": 50.0}
HBM = {"read_latency_us": 0.0, "read_bw_gbps": 2000.0}
SSD_SLC = {
    "read_latency_us": 13.0,
    "read_bw_gbps": 14.0,
    "io_size_bytes": IO_SIZE_128K,
    "qd_cap": 32,
    "qd_latency_us": [
        [1, 13.0],
        [32, 13.0],
        [64, 26.0],
        [128, 52.0],
        [256, 104.0],
        [512, 208.0],
    ],
}


def peak_b(gpu_frac: float, context: int = 1024, block_size: int = 16) -> int:
    blocks = gpu_resident_blocks(context, block_size, gpu_frac)
    if blocks <= 0:
        return 0
    return AVAIL_GPU_BLOCKS // blocks


def result_filename(qps: float) -> str:
    return f"result_{qps:g}.json"


def run_benchmark(
    results_dir: Path,
    qps: float,
    config_path: Path | None,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "benchmark.py"),
        "--batching",
        "paged-attn",
        "--qps",
        f"{qps:g}",
        "--distribution",
        "poisson",
        "--cluster",
        str(CLUSTER),
        "--model",
        str(MODEL),
        "--verbose",
        "none",
        "--results_path",
        str(results_dir),
    ]
    if config_path is not None:
        cmd.extend(["--kv_working_set_config", str(config_path)])
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = results_dir / f"stdout_{qps:g}.log"
    log_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"FAILED qps={qps:g} rc={proc.returncode}")
    return json.loads((results_dir / result_filename(qps)).read_text())


_SPILL_RESULT_KEYS = (
    "kv_ws_spill_latency",
    "kv_ws_dram_write_bytes",
    "kv_ws_ssd_write_bytes",
)


def metrics_from_result(result: dict, offered_qps: float) -> dict:
    missing = [key for key in _SPILL_RESULT_KEYS if key not in result]
    if missing:
        raise KeyError(
            "result JSON missing spill fields "
            f"{missing}; re-run benchmark.py so LLMResult persists them"
        )
    tpot = result["decode_time"]["p50"]
    request_p50 = result["request_time"]["p50"]
    return {
        "offered_qps": offered_qps,
        "output_qps": result["output_qps"],
        "output_token_ps": result["output_token_ps"],
        "ttft_p50": result["prefill_time"]["p50"],
        "ttft_p99": result["prefill_time"]["p99"],
        "ttft_max": result["prefill_time"]["max"],
        "tpot_p50": tpot,
        "tpot_p99": result["decode_time"]["p99"],
        "tpot_max": result["decode_time"]["max"],
        "request_time_p50": request_p50,
        "little_n": little_concurrency(offered_qps, request_p50),
        "duration": result["duration"],
        "preemption_count": result["preemption_count"],
        "recomputation_count": result["recomputation_count"],
        "recomputed_tokens": result.get("recomputed_tokens", 0),
        "kv_ws_fetch_latency": result.get("kv_ws_fetch_latency", 0.0),
        "kv_ws_spill_latency": result["kv_ws_spill_latency"],
        "kv_ws_dram_read_tokens": result.get("kv_ws_dram_read_tokens", 0),
        "kv_ws_ssd_read_tokens": result.get("kv_ws_ssd_read_tokens", 0),
        "kv_ws_dram_ios": result.get("kv_ws_dram_ios", 0),
        "kv_ws_ssd_ios": result.get("kv_ws_ssd_ios", 0),
        "kv_ws_dram_write_bytes": result["kv_ws_dram_write_bytes"],
        "kv_ws_ssd_write_bytes": result["kv_ws_ssd_write_bytes"],
    }


def find_config_knee(
    tag: str,
    results_dir: Path,
    config_path: Path | None,
    extra: dict,
) -> dict:
    print(f"=== knee {tag} ===", flush=True)

    def evaluate(qps: float) -> dict:
        print(f"  qps={qps:g}", flush=True)
        result = run_benchmark(results_dir, qps, config_path)
        return metrics_from_result(result, qps)

    qps_star, points = search_knee_qps(evaluate)
    knee = dict(points[qps_star])
    gpu_frac = float(extra.get("gpu_frac", 1.0 if config_path is None else 0.3))
    row = {
        "tag": tag,
        **extra,
        **knee,
        "qps_star": qps_star,
        "n_star": knee["little_n"],
        "peak_b": peak_b(gpu_frac),
        "curve": [
            {**points[q], "offered_qps": q, "stable": q <= qps_star + 1e-12}
            for q in sorted(points)
        ],
    }
    print(
        f"  λ*={qps_star:g} N*={row['n_star']:.2f} tok/s={row['output_token_ps']:.1f} "
        f"tpot_p50={row['tpot_p50']*1e3:.1f}ms ttft_p99={row['ttft_p99']:.2f}s "
        f"PeakB={row['peak_b']}",
        flush=True,
    )
    return row


def write_cfg(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def base_hier(
    *,
    gpu_frac: float = 0.3,
    dram_frac: float = 0.5,
    ssd_frac: float = 0.2,
    ssd: dict | None = None,
    overlap: str = "layer_prefetch",
) -> dict:
    return {
        "enabled": True,
        "placement": "sliding_window",
        "gpu_frac": gpu_frac,
        "dram_frac": dram_frac,
        "ssd_frac": ssd_frac,
        "overlap": overlap,
        "pcie_bw_gbps": 50.0,
        "dram": dict(DRAM),
        "ssd": dict(ssd or SSD_SLC),
        "hbm": dict(HBM),
    }
