#!/usr/bin/env python3
"""gpu_frac sweep at dram_frac=0: fetch_cost curves and measured token/s."""

from __future__ import annotations

import json
import sys
from itertools import product
from math import floor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kv_working_set_test_report"))

from bench_lib import (  # noqa: E402
    cache_gpu_blocks,
    metrics_from_result,
    peak_b,
    run_benchmark,
    write_cfg,
)
from TokenSim.kv_working_set.config import MediaReadConfig, WorkingSetConfig
from TokenSim.kv_working_set.fetch import fetch_cost
from TokenSim.kv_working_set.placement import split_context

HERE = Path(__file__).resolve().parent
MODEL = ROOT / "data/psla/llama-70b-gqa-4k.json"
SIZE_PER_TOKEN = 327_680
SINK = 4
WINDOW = 256
PREFILL = 2048
CONTEXT = 4096
BENCH_FRACS = (0.30, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04)
BENCH_QPS_LIST = (0.08, 0.14)
# Roofline sparse decode step at B=1, sink+window, after DECODE_SCALE (report §2).
ROOFLINE_STEP_MS = 59.22
SSD = MediaReadConfig(
    read_latency_us=13.0,
    read_bw_gbps=14.0,
    io_size_bytes=4096,
    qd_cap=64,
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _sparse_config(gpu_frac: float, select_tokens: int = 0) -> WorkingSetConfig:
    return WorkingSetConfig(
        enabled=True,
        placement="sliding_window",
        gpu_frac=gpu_frac,
        dram_frac=0.0,
        ssd_frac=1.0 - gpu_frac,
        overlap="layer_prefetch",
        sparse=True,
        sink_tokens=SINK,
        window_tokens=WINDOW,
        select_tokens=select_tokens,
        ssd=SSD,
    )


def sweep_fetch_cost() -> list[dict]:
    rows = []
    gpu_blocks = cache_gpu_blocks("LLaMa2-70B-GQA")
    for step in range(2, 31):
        gpu_frac = step / 100.0
        config = _sparse_config(gpu_frac)
        select = _sparse_config(gpu_frac, select_tokens=WINDOW)
        row: dict = {"gpu_frac": gpu_frac}
        for context, key in ((PREFILL, "prefill"), (CONTEXT, "end")):
            split = split_context(context, config)
            cost = fetch_cost(context, config, SIZE_PER_TOKEN)
            tail = floor(context * gpu_frac)
            row[f"{key}_tail"] = tail
            row[f"{key}_gpu"] = split.gpu
            row[f"{key}_ssd"] = split.ssd
            row[f"{key}_fetch_ssd"] = cost.ssd_tokens
            row[f"{key}_fetch_ms"] = cost.latency * 1e3
            sel_cost = fetch_cost(context, select, SIZE_PER_TOKEN)
            row[f"{key}_select_fetch_ssd"] = sel_cost.ssd_tokens
            row[f"{key}_select_fetch_ms"] = sel_cost.latency * 1e3
        row["peak_b"] = peak_b(
            gpu_frac,
            context=CONTEXT,
            gpu_blocks=gpu_blocks,
            sink_tokens=SINK,
        )
        rows.append(row)
    (HERE / "gpu_frac_fetch.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def plot_fetch(rows: list[dict]) -> None:
    xs = [r["gpu_frac"] * 100 for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    axes[0].plot(
        xs,
        [r["prefill_fetch_ssd"] for r in rows],
        marker="o",
        markersize=4,
        color="#4c78a8",
        label="S=2048 (decode start)",
    )
    axes[0].plot(
        xs,
        [r["end_fetch_ssd"] for r in rows],
        marker="s",
        markersize=4,
        color="#e45756",
        label="S=4096 (decode end)",
    )
    axes[0].plot(
        xs,
        [r["prefill_select_fetch_ssd"] for r in rows],
        linestyle=":",
        color="#4c78a8",
        label="select256 S=2048",
    )
    axes[0].plot(
        xs,
        [r["end_select_fetch_ssd"] for r in rows],
        linestyle=":",
        color="#e45756",
        label="select256 S=4096",
    )
    axes[0].axvline(12.5, color="#4c78a8", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].axvline(6.25, color="#e45756", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_xlabel("gpu_frac (%)")
    axes[0].set_ylabel("SSD fetch tokens / step")
    axes[0].set_title("fetch_cost  SSD tokens  (dram_frac=0)")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(31, 1)

    axes[1].plot(
        xs,
        [r["prefill_fetch_ms"] for r in rows],
        marker="o",
        markersize=4,
        color="#4c78a8",
        label="S=2048",
    )
    axes[1].plot(
        xs,
        [r["end_fetch_ms"] for r in rows],
        marker="s",
        markersize=4,
        color="#e45756",
        label="S=4096",
    )
    axes[1].plot(
        xs,
        [r["prefill_select_fetch_ms"] for r in rows],
        linestyle=":",
        color="#4c78a8",
        label="select256 S=2048",
    )
    axes[1].plot(
        xs,
        [r["end_select_fetch_ms"] for r in rows],
        linestyle=":",
        color="#e45756",
        label="select256 S=4096",
    )
    axes[1].axvline(12.5, color="#4c78a8", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].axvline(6.25, color="#e45756", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_xlabel("gpu_frac (%)")
    axes[1].set_ylabel("fetch_cost latency (ms)")
    axes[1].set_title("fetch_cost  latency")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(31, 1)

    fig.suptitle(
        "Sparse decode, dram_frac=0 — solid: window=256, dotted: uniform select256  "
        "(dashed: 12.5% start / 6.25% end)"
    )
    fig.tight_layout()
    path = HERE / "fig_gpu_frac_fetch.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _payload(gpu_frac: float, select_tokens: int = 0) -> dict:
    return {
        "enabled": True,
        "placement": "sliding_window",
        "gpu_frac": gpu_frac,
        "dram_frac": 0.0,
        "ssd_frac": 1.0 - gpu_frac,
        "overlap": "layer_prefetch",
        "pcie_bw_gbps": 50.0,
        "sparse": True,
        "sink_tokens": SINK,
        "window_tokens": WINDOW,
        "select_tokens": select_tokens,
        "ssd": {
            "read_latency_us": 13.0,
            "read_bw_gbps": 14.0,
            "io_size_bytes": 4096,
            "qd_cap": 64,
        },
        "hbm": {"read_latency_us": 0.0, "read_bw_gbps": 2000.0},
    }


def run_token_sweep() -> list[dict]:
    path = HERE / "gpu_frac_tokens.json"
    existing: list[dict] = []
    if path.exists():
        existing = json.loads(path.read_text())
    for row in existing:
        row.setdefault("mode", "window")
    have = {
        (row["mode"], round(float(row["gpu_frac"]), 2), round(float(row["offered_qps"]), 2))
        for row in existing
    }
    gpu_blocks = cache_gpu_blocks("LLaMa2-70B-GQA")
    rows = list(existing)
    modes = (("window", "gfrac", 0), ("select", "gsel", WINDOW))
    for (mode, prefix, select), gpu_frac in product(modes, BENCH_FRACS):
        tag = f"{prefix}_{gpu_frac:.2f}".replace(".", "p")
        cfg = write_cfg(HERE / f"{tag}.json", _payload(gpu_frac, select))
        for qps in BENCH_QPS_LIST:
            key = (mode, round(gpu_frac, 2), round(qps, 2))
            if key in have:
                print(f"  reuse {tag} qps={qps:g}", flush=True)
                continue
            print(f"  {tag} qps={qps:g}", flush=True)
            result = run_benchmark(HERE / tag, qps, cfg, model=MODEL)
            row = {
                "tag": tag,
                "mode": mode,
                "gpu_frac": gpu_frac,
                "ssd_frac": 1.0 - gpu_frac,
                "peak_b": peak_b(
                    gpu_frac,
                    context=CONTEXT,
                    gpu_blocks=gpu_blocks,
                    sink_tokens=SINK,
                ),
                **metrics_from_result(result, qps),
            }
            rows.append(row)
            have.add(key)
            print(
                f"    tok/s={row['output_token_ps']:.1f} "
                f"tpot={row['tpot_p50']*1e3:.2f}ms "
                f"ssd_read={row['kv_ws_ssd_read_tokens']}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["mode"], row["offered_qps"], -row["gpu_frac"]))
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {path}", flush=True)
    return rows


def _rows_for_qps(rows: list[dict], qps: float, mode: str = "window") -> list[dict]:
    matched = [
        row
        for row in rows
        if abs(float(row["offered_qps"]) - qps) < 1e-9
        and row.get("mode", "window") == mode
    ]
    matched.sort(key=lambda row: -row["gpu_frac"])
    return matched


def plot_tokens(rows: list[dict], fetch_rows: list[dict]) -> None:
    # (qps, mode) -> (color, marker, linestyle, label). window solid, select dotted.
    styles = {
        (0.08, "window"): ("#54a24b", "o", "-", "window256 @0.08"),
        (0.14, "window"): ("#4c78a8", "s", "-", "window256 @0.14"),
        (0.08, "select"): ("#54a24b", "o", ":", "select256 @0.08"),
        (0.14, "select"): ("#4c78a8", "s", ":", "select256 @0.14"),
    }
    by_key = {key: _rows_for_qps(rows, key[0], key[1]) for key in styles}

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for key, (color, marker, ls, label) in styles.items():
        series = by_key[key]
        if not series:
            continue
        xs = [r["gpu_frac"] * 100 for r in series]
        axes[0].plot(
            xs,
            [r["output_token_ps"] for r in series],
            marker=marker,
            color=color,
            linestyle=ls,
            linewidth=2,
            label=label,
        )
    axes[0].set_xlabel("gpu_frac (%)")
    axes[0].set_ylabel("system token/s")
    axes[0].set_title("System token/s (arrival-limited)")
    axes[0].set_xlim(31, 3)
    axes[0].set_ylim(0, 650)
    axes[0].legend(fontsize=8)

    ax2 = axes[1].twinx()
    for mode, ls in (("window", "-"), ("select", ":")):
        series_08 = by_key[(0.08, mode)]
        if not series_08:
            continue
        xs = [r["gpu_frac"] * 100 for r in series_08]
        axes[1].plot(
            xs,
            [r["tpot_p50"] * 1e3 for r in series_08],
            marker="o",
            color="#c45c26",
            linestyle=ls,
            linewidth=2,
            label=f"TPOT p50 {mode}256 @0.08",
        )
        ax2.plot(
            xs,
            [r["kv_ws_ssd_read_tokens"] / 1e6 for r in series_08],
            marker="s",
            color="#4c78a8",
            linestyle=ls,
            linewidth=1.6,
            label=f"SSD read tokens (M) {mode}256",
        )
    axes[1].set_xlabel("gpu_frac (%)")
    axes[1].set_ylabel("TPOT p50 (ms)", color="#c45c26")
    ax2.set_ylabel("SSD read tokens (M)", color="#4c78a8")
    axes[1].set_title("TPOT and measured SSD reads (offered 0.08)")
    axes[1].set_xlim(31, 3)
    axes[1].spines["right"].set_visible(True)
    lines = axes[1].get_lines() + ax2.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines], fontsize=7)
    fig.suptitle("Sparse offload, dram_frac=0 — solid: window256, dotted: uniform select256")
    fig.tight_layout()
    path = HERE / "fig_gpu_frac_tokens.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for key, (color, marker, ls, label) in styles.items():
        series = by_key[key]
        if not series:
            continue
        xs = [r["gpu_frac"] * 100 for r in series]
        speed = [1.0 / r["tpot_p50"] for r in series]
        axes[0].plot(
            xs, speed, marker=marker, color=color, linestyle=ls, linewidth=2, label=label
        )
    axes[0].set_xlabel("gpu_frac (%)")
    axes[0].set_ylabel("decode token/s per request  (1 / TPOT)")
    axes[0].set_title("Measured per-request decode speed")
    axes[0].set_xlim(31, 3)
    axes[0].legend(fontsize=8)

    fx = [r["gpu_frac"] * 100 for r in fetch_rows]
    axes[1].plot(
        fx,
        [1000.0 / (ROOFLINE_STEP_MS + r["prefill_fetch_ms"]) for r in fetch_rows],
        color="#4c78a8",
        linewidth=2,
        label="S=2048  1000/(59.22ms+fetch)",
    )
    axes[1].plot(
        fx,
        [1000.0 / (ROOFLINE_STEP_MS + r["end_fetch_ms"]) for r in fetch_rows],
        color="#e45756",
        linewidth=2,
        label="S=4096  1000/(59.22ms+fetch)",
    )
    axes[1].plot(
        fx,
        [1000.0 / (ROOFLINE_STEP_MS + r["prefill_select_fetch_ms"]) for r in fetch_rows],
        color="#4c78a8",
        linestyle=":",
        linewidth=2,
        label="select256 S=2048",
    )
    axes[1].plot(
        fx,
        [1000.0 / (ROOFLINE_STEP_MS + r["end_select_fetch_ms"]) for r in fetch_rows],
        color="#e45756",
        linestyle=":",
        linewidth=2,
        label="select256 S=4096",
    )
    axes[1].axvline(12.5, color="#4c78a8", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].axvline(6.25, color="#e45756", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_xlabel("gpu_frac (%)")
    axes[1].set_ylabel("decode token/s per request")
    axes[1].set_title("Roofline + fetch_cost step speed")
    axes[1].set_xlim(31, 1)
    axes[1].legend(fontsize=8)
    fig.suptitle("Inference speed in tokens — per-request decode rate vs gpu_frac")
    fig.tight_layout()
    path = HERE / "fig_gpu_frac_speed.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main() -> int:
    _style()
    fetch_rows = sweep_fetch_cost()
    plot_fetch(fetch_rows)
    token_rows = run_token_sweep()
    plot_tokens(token_rows, fetch_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
