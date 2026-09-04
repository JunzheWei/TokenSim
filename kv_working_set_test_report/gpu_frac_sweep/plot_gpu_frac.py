#!/usr/bin/env python3
"""Figures for gpu_frac_inference_speed.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from math import floor

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "summary.json"
WATERMARK_BLOCKS = 5
GPU_BLOCKS = 518
AVAIL = GPU_BLOCKS - WATERMARK_BLOCKS


def gpu_resident_blocks(context_len: int, block_size: int, gpu_frac: float) -> int:
    tokens = floor(max(0, int(context_len)) * gpu_frac)
    if tokens <= 0 or block_size <= 0:
        return 0
    return (tokens + block_size - 1) // block_size


def peak_b(gpu_frac: float, context: int = 1024) -> int:
    blocks = gpu_resident_blocks(context, 16, gpu_frac)
    return AVAIL // blocks if blocks else 0


def plateau_spans(pcts: list[int], bvals: list[int]) -> list[tuple[float, float, int]]:
    spans: list[tuple[float, float, int]] = []
    start = 0
    half = 2.5
    for i in range(1, len(bvals) + 1):
        if i == len(bvals) or bvals[i] != bvals[start]:
            lo = min(pcts[start], pcts[i - 1]) - half
            hi = max(pcts[start], pcts[i - 1]) + half
            spans.append((lo, hi, bvals[start]))
            start = i
    return spans


def main() -> None:
    rows = json.loads(SUMMARY.read_text())
    pcts = [int(round(r["gpu_frac"] * 100)) for r in rows]
    tok = [r["output_token_ps"] for r in rows]
    tpot_ms = [r["tpot_p50"] * 1e3 for r in rows]
    ttft = [r["ttft_p50"] for r in rows]
    b_dec = [peak_b(r["gpu_frac"], 1024) for r in rows]
    b_pre = [peak_b(r["gpu_frac"], 512) for r in rows]
    fetch = [r["kv_ws_fetch_latency"] for r in rows]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
        }
    )

    # --- Figure 1: token/s with peak-B plateaus ---
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.set_ylim(240, 1160)
    cmap = plt.cm.Blues
    unique_b = sorted(set(b_dec))
    b_to_color = {
        b: cmap(0.28 + 0.62 * i / max(1, len(unique_b) - 1)) for i, b in enumerate(unique_b)
    }
    for lo, hi, b in plateau_spans(pcts, b_dec):
        ax.axvspan(lo, hi, color=b_to_color[b], alpha=0.38, zorder=0)

    ax.plot(pcts, tok, color="#1f4e79", linewidth=2.2, zorder=3)
    for x, y, b in zip(pcts, tok, b_dec):
        ax.plot(x, y, marker="o", markersize=7.5, color=b_to_color[b],
                markeredgecolor="#1f4e79", zorder=4)

    callouts = {100: (0, 18), 90: (6, 22), 50: (4, 16), 30: (4, 18), 10: (-18, 12)}
    for x, y, b, p in zip(pcts, tok, b_dec, pcts):
        if p not in callouts:
            continue
        ox, oy = callouts[p]
        ax.annotate(
            f"B={b}",
            xy=(x, y),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=9,
            color="#1f4e79",
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": "#1f4e79", "lw": 0.8},
            zorder=5,
        )

    ax2 = ax.twinx()
    ax2.step(pcts, b_dec, where="mid", color="#c45c26", linewidth=2.0, zorder=2)
    ax2.plot(pcts, b_dec, linestyle="None", marker="s", markersize=5, color="#c45c26")
    ax2.set_ylabel("Peak B at S=1024 (concurrent requests)", color="#c45c26")
    ax2.tick_params(axis="y", colors="#c45c26")
    ax2.set_ylim(0, 85)
    ax2.spines["right"].set_color("#c45c26")
    ax2.grid(False)

    ax.set_xlim(103, 7)
    ax.set_xlabel("GPU KV fraction (gpu_frac)")
    ax.set_ylabel("System token/s")
    ax.set_title(
        "System token/s vs GPU KV fraction — plateaus follow peak B\n"
        "Peak B = 513 GPU blocks / ceil(floor(1024 × gpu_frac) / 16)"
    )
    ax.set_xticks(pcts)
    ax.set_xticklabels([f"{p}%" for p in pcts], rotation=45, ha="right")
    legend = [
        Line2D([0], [0], color="#1f4e79", marker="o", label="System token/s (left)"),
        Line2D([0], [0], color="#c45c26", marker="s", label="Peak B at S=1024 (right)"),
        Line2D(
            [0],
            [0],
            color="#8eb6d4",
            linewidth=10,
            alpha=0.5,
            label="Shaded band = constant peak B",
        ),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(HERE / "fig_tokens_peak_b.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: token/s vs peak B ---
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(b_dec, tok, color="#9aa5b1", linewidth=1.4, zorder=1)
    sc = ax.scatter(b_dec, tok, c=pcts, cmap="viridis_r", s=64, zorder=2, edgecolors="white")
    for x, y, p in zip(b_dec, tok, pcts):
        ax.annotate(
            f"{p}%",
            (x, y),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color="#334155",
        )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("gpu_frac (%)")
    ax.set_xlabel("Peak B at S=1024")
    ax.set_ylabel("System token/s")
    ax.set_title("System token/s tracks peak B (sublinear: TPOT also rises)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_tokens_vs_peak_b.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: TTFT / TPOT ---
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    axes[0].plot(pcts, ttft, color="#2a9d8f", marker="o", linewidth=2)
    axes[0].set_xlim(103, 7)
    axes[0].set_xticks(pcts)
    axes[0].set_xticklabels([f"{p}%" for p in pcts], rotation=45, ha="right")
    axes[0].set_xlabel("GPU KV fraction")
    axes[0].set_ylabel("TTFT p50 (s)")
    axes[0].set_title("TTFT p50 falls as more requests admit together")
    axes[1].plot(pcts, tpot_ms, color="#c45c26", marker="o", linewidth=2)
    axes[1].set_xlim(103, 7)
    axes[1].set_xticks(pcts)
    axes[1].set_xticklabels([f"{p}%" for p in pcts], rotation=45, ha="right")
    axes[1].set_xlabel("GPU KV fraction")
    axes[1].set_ylabel("TPOT p50 (ms)")
    axes[1].set_title("TPOT p50 rises with larger decode batches")
    fig.tight_layout()
    fig.savefig(HERE / "fig_ttft_tpot_frac.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: Bpre vs Bdec (explains 90% bump at same Bdec=8) ---
    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    ax.step(pcts, b_dec, where="mid", color="#c45c26", linewidth=2.2, label="Peak B at S=1024 (decode)")
    ax.step(pcts, b_pre, where="mid", color="#1f4e79", linewidth=2.2, label="Peak B at S=512 (prefill)")
    ax.set_xlim(103, 7)
    ax.set_xticks(pcts)
    ax.set_xticklabels([f"{p}%" for p in pcts], rotation=45, ha="right")
    ax.set_xlabel("GPU KV fraction")
    ax.set_ylabel("Peak concurrent requests")
    ax.set_title("Prefill vs decode peak B — 90% raises B_pre 16→17 while B_dec stays 8")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_peak_b_prefill_decode.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("wrote", HERE / "fig_tokens_peak_b.png")
    print("B_dec", b_dec)
    print("B_pre", b_pre)
    print("fetch", [round(x, 2) for x in fetch])


if __name__ == "__main__":
    main()
