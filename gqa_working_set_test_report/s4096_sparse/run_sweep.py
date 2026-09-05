#!/usr/bin/env python3
"""GQA S=4096 knees: all_gpu vs hier_full vs sparse_offload, then hier_full latency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kv_working_set_test_report"))

from bench_lib import (  # noqa: E402
    cache_gpu_blocks,
    find_config_knee,
    metrics_from_result,
    peak_b,
    run_benchmark,
    write_cfg,
)
from TokenSim.kv_working_set.placement import retained_tokens  # noqa: E402

OUT = Path(__file__).resolve().parent
MODEL = ROOT / "data/psla/llama-70b-gqa-4k.json"
HIER_FULL = ROOT / "data/kv_working_set/hier_30_0_70_slc_4k.json"
HIER_SPARSE = ROOT / "data/kv_working_set/hier_30_0_70_slc_4k_sparse256.json"
HIER_SELECT = ROOT / "data/kv_working_set/hier_30_0_70_slc_4k_select256.json"
HIER_CACHE = ROOT / "data/kv_working_set/hier_30_0_70_slc_4k_cache512.json"
# Plan C sensitivity: GPU page-cache tokens × step-to-step reuse (assumed, no trace).
CACHE_TOKENS = (512, 1024)
REUSE = (0.0, 0.5, 0.8)
# all_gpu's knee search never probes 0.14 (sparse arms' λ*); add it for the curves.
EXTRA_ALL_GPU_QPS = (0.14,)
GPU_BLOCKS = cache_gpu_blocks("LLaMa2-70B-GQA")
CONTEXT = 4096
PREFILL_CONTEXT = 2048
SINK = 4
WINDOW = 256
LATENCIES_US = (13, 25, 50, 100)
OVERLAPS = ("layer_prefetch", "blocking")


def _peak_fields(gpu_frac: float, *, streaming: bool = False) -> dict:
    sink = SINK if streaming else 0
    window = WINDOW if streaming else 0
    kept = retained_tokens(CONTEXT, SINK, WINDOW) if streaming else CONTEXT
    return {
        "retained_kv_tokens": kept,
        "peak_b": peak_b(
            gpu_frac,
            context=CONTEXT,
            gpu_blocks=GPU_BLOCKS,
            sink_tokens=sink,
            window_tokens=window,
            streaming_attention=streaming,
        ),
        "peak_b_prefill": peak_b(1.0, context=PREFILL_CONTEXT, gpu_blocks=GPU_BLOCKS),
    }


def _knee_row(tag: str, config_path: Path | None, extra: dict) -> dict:
    return find_config_knee(
        tag,
        OUT / tag,
        config_path,
        extra,
        model=MODEL,
        gpu_blocks=GPU_BLOCKS,
        context=CONTEXT,
    )


def _reuse_knee(tag: str, extra: dict) -> dict | None:
    summary_path = OUT / "summary.json"
    if not summary_path.exists():
        return None
    for row in json.loads(summary_path.read_text()):
        if row.get("tag") != tag:
            continue
        merged = {**row, **extra}
        merged.update(_peak_fields(float(extra.get("gpu_frac", 1.0)), streaming=False))
        return merged
    return None


def _add_curve_points(row: dict, config_path: Path | None, qps_list: tuple[float, ...]) -> None:
    have = {round(float(p["offered_qps"]), 4) for p in row["curve"]}
    for qps in qps_list:
        if round(qps, 4) in have:
            continue
        print(f"  extra {row['tag']} qps={qps:g}", flush=True)
        result = run_benchmark(OUT / row["tag"], qps, config_path, model=MODEL)
        point = metrics_from_result(result, qps)
        point["stable"] = qps <= float(row["qps_star"]) + 1e-12
        row["curve"].append(point)
    row["curve"].sort(key=lambda p: p["offered_qps"])


def _select_extra(cache_tokens: int, reuse: float) -> dict:
    return {
        "gpu_frac": 0.3,
        "dram_frac": 0.0,
        "ssd_frac": 0.7,
        "sparse": True,
        "sink_tokens": SINK,
        "window_tokens": WINDOW,
        "select_tokens": WINDOW,
        "select_cache_tokens": cache_tokens,
        "select_reuse": reuse,
    }


def run_knees() -> list[dict]:
    print(f"GQA gpu_blocks={GPU_BLOCKS} context={CONTEXT}", flush=True)
    all_gpu = _reuse_knee("all_gpu", {"gpu_frac": 1.0})
    if all_gpu is None:
        all_gpu = _knee_row("all_gpu", None, {"gpu_frac": 1.0})
    else:
        print("  reuse all_gpu", flush=True)
    _add_curve_points(all_gpu, None, EXTRA_ALL_GPU_QPS)
    hier_full = _knee_row(
        "hier_full",
        HIER_FULL,
        {"gpu_frac": 0.3, "dram_frac": 0.0, "ssd_frac": 0.7},
    )
    sparse_offload = _knee_row(
        "sparse_offload",
        HIER_SPARSE,
        {
            "gpu_frac": 0.3,
            "dram_frac": 0.0,
            "ssd_frac": 0.7,
            "sparse": True,
            "sink_tokens": SINK,
            "window_tokens": WINDOW,
        },
    )
    select_offload = _knee_row(
        "select_offload",
        HIER_SELECT,
        {
            "gpu_frac": 0.3,
            "dram_frac": 0.0,
            "ssd_frac": 0.7,
            "sparse": True,
            "sink_tokens": SINK,
            "window_tokens": WINDOW,
            "select_tokens": WINDOW,
        },
    )
    cache_offload = _knee_row("cache_offload", HIER_CACHE, _select_extra(512, 0.0))
    rows = [all_gpu, hier_full, sparse_offload, select_offload, cache_offload]
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {OUT / 'summary.json'}", flush=True)
    return rows


def run_cache_grid() -> list[dict]:
    print("=== plan C grid: cache_tokens × reuse ===", flush=True)
    base = json.loads(HIER_SELECT.read_text())
    rows = []
    for cache_tokens in CACHE_TOKENS:
        for reuse in REUSE:
            tag = f"cache_{cache_tokens}_r{int(reuse * 100):02d}"
            payload = dict(base)
            payload["select_cache_tokens"] = cache_tokens
            payload["select_reuse"] = reuse
            cfg_path = write_cfg(OUT / f"{tag}.json", payload)
            rows.append(_knee_row(tag, cfg_path, _select_extra(cache_tokens, reuse)))
    (OUT / "cache_grid.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {OUT / 'cache_grid.json'}", flush=True)
    return rows


def run_latency_sweep(hier_full_qps: float) -> list[dict]:
    existing = OUT / "latency_summary.json"
    print(f"=== latency sweep offered_qps={hier_full_qps:g} ===", flush=True)
    base = json.loads(HIER_FULL.read_text())
    rows = []
    for latency_us in LATENCIES_US:
        for overlap in OVERLAPS:
            tag = f"lat_{latency_us:03d}_{overlap}"
            payload = json.loads(json.dumps(base))
            payload["overlap"] = overlap
            payload["ssd"]["read_latency_us"] = float(latency_us)
            cfg_path = write_cfg(OUT / f"{tag}.json", payload)
            print(f"  {tag}", flush=True)
            result = run_benchmark(OUT / tag, hier_full_qps, cfg_path, model=MODEL)
            row = {
                "tag": tag,
                "read_latency_us": latency_us,
                "overlap": overlap,
                "gpu_frac": 0.3,
                **metrics_from_result(result, hier_full_qps),
            }
            rows.append(row)
            print(
                f"    tok/s={row['output_token_ps']:.1f} "
                f"tpot_p50={row['tpot_p50']*1e3:.1f}ms "
                f"ttft_p99={row['ttft_p99']:.2f}s "
                f"fetch={row['kv_ws_fetch_latency']:.3f}s "
                f"ssd_ios={row['kv_ws_ssd_ios']}",
                flush=True,
            )
    existing.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {existing}", flush=True)
    return rows


def main() -> int:
    knees = run_knees()
    run_cache_grid()
    hier_full = next(row for row in knees if row["tag"] == "hier_full")
    run_latency_sweep(float(hier_full["qps_star"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
