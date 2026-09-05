#!/usr/bin/env python3
"""TPOT / TTFT / token/s figures for gqa_s4096_slc4k_sparse.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ARMS = ("all_gpu", "hier_full", "sparse_offload", "select_offload", "cache_offload")
LABELS = {
    "all_gpu": "all_gpu_full",
    "hier_full": "hier_full",
    "sparse_offload": "sparse_offload",
    "select_offload": "select_offload",
    "cache_offload": "cache_offload",
}
COLORS = {
    "all_gpu": "#4c78a8",
    "hier_full": "#e45756",
    "sparse_offload": "#54a24b",
    "select_offload": "#f58518",
    "cache_offload": "#b279a2",
}


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


def _load_rows(path: Path) -> dict[str, dict]:
    return {row["tag"]: row for row in json.loads(path.read_text())}


def plot_knee_bars(knees: dict[str, dict]) -> None:
    names = [LABELS[tag] for tag in ARMS]
    colors = [COLORS[tag] for tag in ARMS]
    tok = [knees[tag]["output_token_ps"] for tag in ARMS]
    tpot = [knees[tag]["tpot_p50"] * 1e3 for tag in ARMS]
    ttft50 = [knees[tag]["ttft_p50"] for tag in ARMS]
    ttft99 = [knees[tag]["ttft_p99"] for tag in ARMS]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
    axes[0].bar(names, tok, color=colors)
    axes[0].set_ylabel("token/s")
    axes[0].set_title("System token/s at λ*")

    axes[1].bar(names, tpot, color=colors)
    axes[1].set_ylabel("ms")
    axes[1].set_title("TPOT p50 at λ*")

    x = range(len(names))
    width = 0.38
    axes[2].bar([i - width / 2 for i in x], ttft50, width, color=colors, label="p50")
    axes[2].bar(
        [i + width / 2 for i in x],
        ttft99,
        width,
        color=colors,
        alpha=0.45,
        label="p99",
    )
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(names)
    axes[2].set_ylabel("s")
    axes[2].set_title("TTFT at λ*")
    axes[2].legend()
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15, labelsize=9)

    fig.suptitle("GQA S=4096 knees — full attention vs sparse+offload")
    fig.tight_layout()
    fig.savefig(HERE / "fig_knee_ttft_tpot_tokens.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_qps_curves(knees: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    series = (
        (axes[0], "output_token_ps", 1.0, "System token/s", False),
        (axes[1], "tpot_p50", 1e3, "TPOT p50 (ms)", False),
        (axes[2], "ttft_p99", 1.0, "TTFT p99 (s)", True),
    )
    for ax, key, scale, ylabel, logy in series:
        for tag in ARMS:
            xs = [p["offered_qps"] for p in knees[tag]["curve"]]
            ys = [p[key] * scale for p in knees[tag]["curve"]]
            ax.plot(xs, ys, marker="o", color=COLORS[tag], label=LABELS[tag])
            ax.axvline(
                knees[tag]["qps_star"],
                color=COLORS[tag],
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
        ax.set_xlabel("offered QPS")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("GQA S=4096 vs offered QPS (dashed = that arm's λ*)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_qps_ttft_tpot_tokens.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_latency(rows: list[dict]) -> None:
    by_overlap: dict[str, list[dict]] = {"layer_prefetch": [], "blocking": []}
    for row in rows:
        by_overlap[row["overlap"]].append(row)
    for group in by_overlap.values():
        group.sort(key=lambda r: r["read_latency_us"])

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    styles = {
        "layer_prefetch": ("#4c78a8", "layer_prefetch"),
        "blocking": ("#f58518", "blocking"),
    }
    panels = (
        (axes[0], "output_token_ps", 1.0, "System token/s", False),
        (axes[1], "tpot_p50", 1e3, "TPOT p50 (ms)", True),
        (axes[2], "ttft_p99", 1.0, "TTFT p99 (s)", True),
    )
    for ax, key, scale, ylabel, logy in panels:
        for overlap, (color, label) in styles.items():
            xs = [r["read_latency_us"] for r in by_overlap[overlap]]
            ys = [r[key] * scale for r in by_overlap[overlap]]
            ax.plot(xs, ys, marker="o", color=color, label=label)
        ax.set_xlabel("SSD read_latency_us")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks([13, 25, 50, 100])
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("hier_full 30/0/70 latency sweep — 4K, qd_cap=64")
    fig.tight_layout()
    fig.savefig(HERE / "fig_latency_ttft_tpot_tokens.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    knees = _load_rows(HERE / "summary.json")
    latency = json.loads((HERE / "latency_summary.json").read_text())
    plot_knee_bars(knees)
    plot_qps_curves(knees)
    plot_latency(latency)
    print("wrote", HERE / "fig_knee_ttft_tpot_tokens.png")
    print("wrote", HERE / "fig_qps_ttft_tpot_tokens.png")
    print("wrote", HERE / "fig_latency_ttft_tpot_tokens.png")


if __name__ == "__main__":
    main()
