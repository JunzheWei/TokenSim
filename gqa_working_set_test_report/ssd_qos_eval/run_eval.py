#!/usr/bin/env python3
"""GQA Poisson QPS-knee eval: all-GPU vs hierarchical, plus SLC/MLC/N3 and qd_cap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kv_working_set_test_report"))

from bench_lib import (  # noqa: E402
    IO_SIZE_128K,
    base_hier,
    cache_gpu_blocks,
    find_config_knee,
    write_cfg,
)

REPORT = ROOT / "gqa_working_set_test_report"
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "kv_working_set"
MODEL = ROOT / "data/psla/llama-70b-gqa.json"
GPU_BLOCKS = cache_gpu_blocks("LLaMa2-70B-GQA")
DRIVES = {
    "slc": {"read_latency_us": 13.0, "file": "hier_n3x_slc.json"},
    "n3x": {"read_latency_us": 18.0, "file": "hier_n3x.json"},
    "n3": {"read_latency_us": 50.0, "file": "hier_n3.json"},
}


def knee(tag: str, results_dir: Path, config_path: Path | None, extra: dict) -> dict:
    return find_config_knee(
        tag,
        results_dir,
        config_path,
        extra,
        model=MODEL,
        gpu_blocks=GPU_BLOCKS,
    )


def queued_ssd(latency_us: float, qd_cap: int, io_size: int = IO_SIZE_128K) -> dict:
    cfg = base_hier()
    cfg["ssd"] = {
        "read_latency_us": latency_us,
        "read_bw_gbps": 14.0,
        "io_size_bytes": io_size,
        "qd_cap": qd_cap,
    }
    return cfg


def coalesced_ssd(latency_us: float) -> dict:
    cfg = base_hier()
    cfg["ssd"] = {
        "read_latency_us": latency_us,
        "read_bw_gbps": 14.0,
        "io_size_bytes": 0,
    }
    return cfg


def plot_official(all_gpu: dict, hier: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for ax, key, ylabel, scale in (
        (axes[0], "output_qps", "output QPS", 1.0),
        (axes[1], "ttft_p99", "TTFT p99 (s)", 1.0),
    ):
        for row, color, label in (
            (all_gpu, "#4c78a8", "all GPU"),
            (hier, "#f58518", "hierarchical"),
        ):
            xs = [p["offered_qps"] for p in row["curve"]]
            ys = [p[key] * scale for p in row["curve"]]
            ax.plot(xs, ys, marker="o", color=color, label=label)
        ax.axvline(all_gpu["qps_star"], color="#4c78a8", linestyle="--", linewidth=1)
        ax.axvline(hier["qps_star"], color="#f58518", linestyle="--", linewidth=1)
        ax.set_xlabel("offered QPS")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle("GQA-8 Poisson QPS knee (128KiB DMA + layer prefetch)")
    fig.savefig(REPORT / "fig_qps_knee.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    labels = ["λ*", "N*", "tok/s", "TPOT p50 ms"]
    gpu_vals = [
        all_gpu["qps_star"],
        all_gpu["n_star"],
        all_gpu["output_token_ps"],
        all_gpu["tpot_p50"] * 1e3,
    ]
    hier_vals = [
        hier["qps_star"],
        hier["n_star"],
        hier["output_token_ps"],
        hier["tpot_p50"] * 1e3,
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.8))
    for ax, label, g, h in zip(axes, labels, gpu_vals, hier_vals):
        ax.bar(["all GPU", "hier"], [g, h], color=["#4c78a8", "#f58518"])
        ax.set_title(label)
    fig.suptitle("GQA-8 official knee metrics")
    fig.savefig(REPORT / "fig_ttft_tpot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_qos(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_tag = {row["tag"]: row for row in rows}
    drive_tags = ["queued_slc", "queued_n3x", "queued_n3"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    axes[0].bar(drive_tags, [by_tag[t]["qps_star"] for t in drive_tags], color="#4c78a8")
    axes[0].set_ylabel("λ* (r/s)")
    axes[0].set_title("qd_cap=32 knee QPS")
    axes[1].bar(
        drive_tags,
        [by_tag[t]["output_token_ps"] for t in drive_tags],
        color="#f58518",
    )
    axes[1].set_ylabel("token/s at λ*")
    axes[1].set_title("qd_cap=32 tok/s")
    fig.savefig(OUT / "fig_drive_rank.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for drive, color in (("slc", "#4c78a8"), ("n3", "#e45756")):
        caps = []
        vals = []
        for cap in (8, 32, 128, 512):
            tag = f"{drive}_qd{cap}"
            if tag == "slc_qd32":
                tag = "queued_slc"
            elif tag == "n3_qd32":
                tag = "queued_n3"
            if tag not in by_tag:
                continue
            caps.append(cap)
            vals.append(by_tag[tag]["qps_star"])
        ax.plot(caps, vals, marker="o", label=drive, color=color)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 32, 128, 512], ["8", "32", "128", "512"])
    ax.set_xlabel("qd_cap")
    ax.set_ylabel("λ* (r/s)")
    ax.legend()
    ax.set_title("GQA-8 qd_cap sweep knee QPS (SLC vs N3, 128KiB)")
    fig.savefig(OUT / "fig_qd_cap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    print(f"GQA gpu_blocks={GPU_BLOCKS}", flush=True)

    all_gpu = knee(
        "all_gpu",
        REPORT / "all_gpu_poisson",
        None,
        {"kind": "official", "drive": None, "io_size_bytes": 0, "gpu_frac": 1.0},
    )
    summary.append(all_gpu)
    (REPORT / "all_gpu_poisson.json").write_text(json.dumps(all_gpu, indent=2) + "\n")

    hier = knee(
        "hier_slc",
        REPORT / "hier_poisson",
        DATA / "hier_30_50_20.json",
        {
            "kind": "official",
            "drive": "slc",
            "io_size_bytes": IO_SIZE_128K,
            "qd_cap": 32,
            "gpu_frac": 0.3,
        },
    )
    summary.append(hier)
    (REPORT / "hier_poisson.json").write_text(json.dumps(hier, indent=2) + "\n")

    for name, spec in DRIVES.items():
        path = write_cfg(
            OUT / f"coalesced_{name}.json",
            coalesced_ssd(spec["read_latency_us"]),
        )
        summary.append(
            knee(
                f"coalesced_{name}",
                OUT / f"coalesced_{name}",
                path,
                {
                    "kind": "coalesced",
                    "drive": name,
                    "io_size_bytes": 0,
                    "gpu_frac": 0.3,
                    "read_latency_us": spec["read_latency_us"],
                },
            )
        )

    for name, spec in DRIVES.items():
        if name == "slc":
            slc_row = dict(hier)
            slc_row["tag"] = "queued_slc"
            slc_row["kind"] = "queued"
            summary.append(slc_row)
            continue
        summary.append(
            knee(
                f"queued_{name}",
                OUT / f"queued_{name}",
                DATA / spec["file"],
                {
                    "kind": "queued",
                    "drive": name,
                    "io_size_bytes": IO_SIZE_128K,
                    "qd_cap": 32,
                    "gpu_frac": 0.3,
                    "read_latency_us": spec["read_latency_us"],
                },
            )
        )

    path_4k = write_cfg(OUT / "queued_slc_4k.json", queued_ssd(13.0, 32, io_size=4096))
    summary.append(
        knee(
            "queued_slc_4k",
            OUT / "queued_slc_4k",
            path_4k,
            {
                "kind": "queued_4k",
                "drive": "slc",
                "io_size_bytes": 4096,
                "qd_cap": 32,
                "gpu_frac": 0.3,
            },
        )
    )

    for name in ("slc", "n3"):
        latency = DRIVES[name]["read_latency_us"]
        for cap in (8, 128, 512):
            path = write_cfg(
                OUT / f"{name}_qd{cap}.json",
                queued_ssd(latency, cap),
            )
            summary.append(
                knee(
                    f"{name}_qd{cap}",
                    OUT / f"{name}_qd{cap}",
                    path,
                    {
                        "kind": "qd_cap",
                        "drive": name,
                        "io_size_bytes": IO_SIZE_128K,
                        "qd_cap": cap,
                        "gpu_frac": 0.3,
                        "read_latency_us": latency,
                    },
                )
            )

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_official(summary[0], summary[1])
    plot_qos(summary)
    print(f"wrote {OUT / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
