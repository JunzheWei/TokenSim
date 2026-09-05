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

    fig.suptitle("GQA S=4096 knees — N3X SLC 13 µs, 4K")
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
    fig.suptitle("GQA S=4096 vs offered QPS — N3X SLC 13 µs (dashed = that arm's λ*)")
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


def plot_drive(rows: list[dict]) -> None:
    labels = [row["arm"] for row in rows]
    slc_tok = [row["n3x_slc"]["output_token_ps"] for row in rows]
    n3_tok = [row["n3"]["output_token_ps"] for row in rows]
    slc_tpot = [row["n3x_slc"]["tpot_p50"] * 1e3 for row in rows]
    n3_tpot = [row["n3"]["tpot_p50"] * 1e3 for row in rows]
    slc_qps = [row["n3x_slc"]["qps_star"] for row in rows]
    n3_qps = [row["n3"]["qps_star"] for row in rows]
    x = list(range(len(labels)))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
    axes[0].bar([i - width / 2 for i in x], slc_qps, width, color="#54a24b", label="N3X SLC 13 µs")
    axes[0].bar([i + width / 2 for i in x], n3_qps, width, color="#e45756", label="N3 50 µs")
    axes[0].set_ylabel("λ*")
    axes[0].set_title("Knee QPS")
    axes[1].bar([i - width / 2 for i in x], slc_tok, width, color="#54a24b", label="N3X SLC 13 µs")
    axes[1].bar([i + width / 2 for i in x], n3_tok, width, color="#e45756", label="N3 50 µs")
    axes[1].set_ylabel("token/s")
    axes[1].set_title("System token/s at λ*")
    axes[2].bar([i - width / 2 for i in x], slc_tpot, width, color="#54a24b", label="N3X SLC 13 µs")
    axes[2].bar([i + width / 2 for i in x], n3_tpot, width, color="#e45756", label="N3 50 µs")
    axes[2].set_ylabel("ms")
    axes[2].set_title("TPOT p50 at λ*")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("N3 vs N3X SLC — 4K, qd_cap=64, 14 GB/s (only L changes)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_drive_n3_vs_slc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_drive_tokens(rows: list[dict], all_gpu: dict | None = None) -> None:
    """System token/s: SLC vs N3 curves, plus bars at λ* and at QPS=0.08."""
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))
    qps_ref = 0.08
    if all_gpu is not None:
        xs = [p["offered_qps"] for p in all_gpu["curve"]]
        ys = [p["output_token_ps"] for p in all_gpu["curve"]]
        axes[0].plot(xs, ys, color="#9e9e9e", linestyle=":", marker=".", label="all_gpu_full")
    for row in rows:
        arm = row["arm"]
        color = COLORS[arm]
        for key, style, mark, suffix in (
            ("n3x_slc", "-", "o", "SLC"),
            ("n3", "--", "s", "N3"),
        ):
            curve = row[key]["curve"]
            xs = [p["offered_qps"] for p in curve]
            ys = [p["output_token_ps"] for p in curve]
            axes[0].plot(xs, ys, color=color, linestyle=style, marker=mark, label=f"{arm} {suffix}")
    axes[0].set_xlabel("offered QPS")
    axes[0].set_ylabel("token/s")
    axes[0].set_title("System token/s vs QPS")
    axes[0].legend(fontsize=6.5, ncol=2)

    labels = [row["arm"] for row in rows]
    x = list(range(len(labels)))
    width = 0.36
    slc_star = [row["n3x_slc"]["output_token_ps"] for row in rows]
    n3_star = [row["n3"]["output_token_ps"] for row in rows]
    axes[1].bar([i - width / 2 for i in x], slc_star, width, color="#54a24b", label="N3X SLC 13 µs")
    axes[1].bar([i + width / 2 for i in x], n3_star, width, color="#e45756", label="N3 50 µs")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, fontsize=9)
    axes[1].set_ylabel("token/s")
    axes[1].set_title("System token/s at each arm's λ*")
    axes[1].legend(fontsize=8)

    def _tok_at(curve: list[dict], qps: float) -> float | None:
        for point in curve:
            if abs(point["offered_qps"] - qps) < 1e-9:
                return point["output_token_ps"]
        return None

    plot08 = [
        row
        for row in rows
        if _tok_at(row["n3x_slc"]["curve"], qps_ref) is not None
        and _tok_at(row["n3"]["curve"], qps_ref) is not None
    ]
    labels08 = [row["arm"] for row in plot08]
    x08 = list(range(len(labels08)))
    slc_08 = [_tok_at(row["n3x_slc"]["curve"], qps_ref) for row in plot08]
    n3_08 = [_tok_at(row["n3"]["curve"], qps_ref) for row in plot08]
    axes[2].bar([i - width / 2 for i in x08], slc_08, width, color="#54a24b", label="N3X SLC 13 µs")
    axes[2].bar([i + width / 2 for i in x08], n3_08, width, color="#e45756", label="N3 50 µs")
    axes[2].set_xticks(x08)
    axes[2].set_xticklabels(labels08, rotation=15, fontsize=9)
    axes[2].set_ylabel("token/s")
    axes[2].set_title(f"System token/s at QPS={qps_ref:g} (same load)")
    axes[2].legend(fontsize=8)
    fig.suptitle("N3 vs N3X SLC — system token/s (4K, qd_cap=64, 14 GB/s)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_drive_tokens.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    knees = _load_rows(HERE / "summary.json")
    latency = json.loads((HERE / "latency_summary.json").read_text())
    plot_knee_bars(knees)
    plot_qps_curves(knees)
    plot_latency(latency)
    drive_path = HERE / "drive_compare.json"
    if drive_path.exists():
        drive_rows = json.loads(drive_path.read_text())
        plot_drive(drive_rows)
        plot_drive_tokens(drive_rows, knees.get("all_gpu"))
        print("wrote", HERE / "fig_drive_n3_vs_slc.png")
        print("wrote", HERE / "fig_drive_tokens.png")
    print("wrote", HERE / "fig_knee_ttft_tpot_tokens.png")
    print("wrote", HERE / "fig_qps_ttft_tpot_tokens.png")
    print("wrote", HERE / "fig_latency_ttft_tpot_tokens.png")


if __name__ == "__main__":
    main()
