#!/usr/bin/env python3
"""Burst eval: all-GPU vs hierarchical, plus SLC/MLC/N3 and qd_cap sweep."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "kv_working_set_test_report"
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "kv_working_set"
CLUSTER = ROOT / "data/clusters/1_h200/h1.json"
MODEL = ROOT / "data/psla/llama-70b.json"

DRAM = {"read_latency_us": 2.0, "read_bw_gbps": 50.0}
HBM = {"read_latency_us": 0.0, "read_bw_gbps": 2000.0}
DRIVES = {
    "slc": {"read_latency_us": 13.0, "file": "hier_n3x_slc.json"},
    "n3x": {"read_latency_us": 18.0, "file": "hier_n3x.json"},
    "n3": {"read_latency_us": 50.0, "file": "hier_n3.json"},
}


def _base_hier() -> dict:
    return {
        "enabled": True,
        "placement": "sliding_window",
        "gpu_frac": 0.3,
        "dram_frac": 0.5,
        "ssd_frac": 0.2,
        "overlap": "blocking",
        "dram": dict(DRAM),
        "hbm": dict(HBM),
    }


def coalesced_ssd(latency_us: float) -> dict:
    cfg = _base_hier()
    cfg["ssd"] = {
        "read_latency_us": latency_us,
        "read_bw_gbps": 14.0,
        "io_size_bytes": 0,
    }
    return cfg


def queued_ssd(latency_us: float, qd_cap: int) -> dict:
    cfg = _base_hier()
    cfg["ssd"] = {
        "read_latency_us": latency_us,
        "read_bw_gbps": 14.0,
        "io_size_bytes": 4096,
        "qd_cap": qd_cap,
    }
    return cfg


def _parse_stdout(text: str) -> dict:
    def _f(pattern: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    return {
        "ttft_avg": _f(r"Average prefill latency: ([0-9.eE+-]+)"),
        "ttft_min": _f(r"Min prefill latency: ([0-9.eE+-]+)"),
        "tpot_avg": _f(r"Average decode latency: ([0-9.eE+-]+)"),
        "stdout_token_ps": _f(r"Thoughput: .* r/s, ([0-9.eE+-]+) token/s"),
    }


def _row(tag: str, result: dict, extra: dict, stdout: dict) -> dict:
    tpot = result["decode_time"]["p50"]
    return {
        "tag": tag,
        **extra,
        "duration": result["duration"],
        "output_token_ps": result["output_token_ps"],
        "output_qps": result["output_qps"],
        "ttft_p50": result["prefill_time"]["p50"],
        "ttft_p99": result["prefill_time"]["p99"],
        "ttft_max": result["prefill_time"]["max"],
        "ttft_min": stdout.get("ttft_min"),
        "ttft_avg": stdout.get("ttft_avg"),
        "tpot_p50": tpot,
        "tpot_p99": result["decode_time"]["p99"],
        "tpot_max": result["decode_time"]["max"],
        "tpot_avg": stdout.get("tpot_avg"),
        "stdout_token_ps": stdout.get("stdout_token_ps"),
        "per_request_token_ps": 1.0 / tpot if tpot else None,
        "preemption_count": result["preemption_count"],
        "recomputation_count": result["recomputation_count"],
        "recomputed_tokens": result["recomputed_tokens"],
        "kv_ws_fetch_latency": result.get("kv_ws_fetch_latency", 0.0),
        "kv_ws_dram_read_tokens": result.get("kv_ws_dram_read_tokens", 0),
        "kv_ws_ssd_read_tokens": result.get("kv_ws_ssd_read_tokens", 0),
        "kv_ws_dram_ios": result.get("kv_ws_dram_ios", 0),
        "kv_ws_ssd_ios": result.get("kv_ws_ssd_ios", 0),
    }


def run_arm(
    tag: str,
    results_dir: Path,
    config_path: Path | None,
    extra: dict,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"=== {tag} ===", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = results_dir / "stdout.log"
    log_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"FAILED {tag} rc={proc.returncode}")
    result = json.loads((results_dir / "result_inf.json").read_text())
    row = _row(tag, result, extra, _parse_stdout(proc.stdout))
    print(
        f"  tok/s={row['output_token_ps']:.1f} tpot_p50={row['tpot_p50']*1e3:.1f}ms "
        f"ttft_p50={row['ttft_p50']:.1f}s preempt={row['preemption_count']} "
        f"fetch={row['kv_ws_fetch_latency']:.2f}s",
        flush=True,
    )
    return row


def write_cfg(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def plot_official(all_gpu: dict, hier: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["TTFT p50", "TTFT p99", "TPOT p50", "TPOT p99"]
    gpu_vals = [
        all_gpu["ttft_p50"],
        all_gpu["ttft_p99"],
        all_gpu["tpot_p50"] * 1e3,
        all_gpu["tpot_p99"] * 1e3,
    ]
    hier_vals = [
        hier["ttft_p50"],
        hier["ttft_p99"],
        hier["tpot_p50"] * 1e3,
        hier["tpot_p99"] * 1e3,
    ]
    ratios = [h / g if g else 0.0 for h, g in zip(hier_vals, gpu_vals)]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(labels, ratios, color="#4c78a8")
    ax.axhline(1.0, color="#333", linewidth=1)
    ax.set_ylabel("hierarchical / all-GPU")
    ax.set_title("Official burst slowdown (per-step cold-set fetch + SLC 4K)")
    fig.savefig(REPORT / "fig_slowdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].bar(["all GPU", "hierarchical"], [all_gpu["ttft_p50"], hier["ttft_p50"]], color="#4c78a8")
    axes[0].set_ylabel("TTFT p50 (s)")
    axes[1].bar(
        ["all GPU", "hierarchical"],
        [all_gpu["tpot_p50"] * 1e3, hier["tpot_p50"] * 1e3],
        color="#f58518",
    )
    axes[1].set_ylabel("TPOT p50 (ms)")
    fig.suptitle("Official burst TTFT / TPOT")
    fig.savefig(REPORT / "fig_ttft_tpot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    names = ["system tok/s", "stdout tok/s", "per-request tok/s"]
    gpu_tok = [
        all_gpu["output_token_ps"],
        all_gpu.get("stdout_token_ps") or 0.0,
        all_gpu["per_request_token_ps"],
    ]
    hier_tok = [
        hier["output_token_ps"],
        hier.get("stdout_token_ps") or 0.0,
        hier["per_request_token_ps"],
    ]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar([i - 0.18 for i in x], gpu_tok, width=0.36, label="all GPU")
    ax.bar([i + 0.18 for i in x], hier_tok, width=0.36, label="hierarchical")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("token/s")
    ax.legend()
    ax.set_title("Official burst throughput")
    fig.savefig(REPORT / "fig_tokens.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_qos(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_tag = {row["tag"]: row for row in rows}

    drive_tags = ["queued_slc", "queued_n3x", "queued_n3"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    axes[0].bar(drive_tags, [by_tag[t]["output_token_ps"] for t in drive_tags], color="#4c78a8")
    axes[0].set_ylabel("token/s")
    axes[0].set_title("qd_cap=32 tok/s")
    axes[1].bar(
        drive_tags,
        [by_tag[t]["kv_ws_fetch_latency"] for t in drive_tags],
        color="#e45756",
    )
    axes[1].set_ylabel("Σ fetch (s)")
    axes[1].set_title("qd_cap=32 fetch")
    fig.savefig(OUT / "fig_drive_rank.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for drive, color in (("slc", "#4c78a8"), ("n3", "#e45756")):
        caps = []
        toks = []
        for cap in (8, 32, 128, 512):
            tag = f"{drive}_qd{cap}"
            if tag == "slc_qd32":
                tag = "queued_slc"
            elif tag == "n3_qd32":
                tag = "queued_n3"
            if tag not in by_tag:
                continue
            caps.append(cap)
            toks.append(by_tag[tag]["output_token_ps"])
        ax.plot(caps, toks, marker="o", label=drive, color=color)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 32, 128, 512], ["8", "32", "128", "512"])
    ax.set_xlabel("qd_cap")
    ax.set_ylabel("token/s")
    ax.legend()
    ax.set_title("qd_cap sweep (SLC vs N3)")
    fig.savefig(OUT / "fig_qd_cap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    summary.append(
        run_arm(
            "all_gpu",
            REPORT / "all_gpu_burst",
            None,
            {"kind": "official", "drive": None, "io_size_bytes": 0, "qd_cap": None},
        )
    )
    shutil.copyfile(
        REPORT / "all_gpu_burst" / "result_inf.json",
        REPORT / "all_gpu_burst.json",
    )
    summary.append(
        run_arm(
            "hier_slc",
            REPORT / "hier_burst",
            DATA / "hier_30_50_20.json",
            {
                "kind": "official",
                "drive": "slc",
                "io_size_bytes": 4096,
                "qd_cap": 32,
            },
        )
    )
    shutil.copyfile(
        REPORT / "hier_burst" / "result_inf.json",
        REPORT / "hier_burst.json",
    )

    for name, spec in DRIVES.items():
        path = write_cfg(OUT / f"coalesced_{name}.json", coalesced_ssd(spec["read_latency_us"]))
        summary.append(
            run_arm(
                f"coalesced_{name}",
                OUT / f"coalesced_{name}",
                path,
                {
                    "kind": "coalesced",
                    "drive": name,
                    "io_size_bytes": 0,
                    "qd_cap": None,
                    "read_latency_us": spec["read_latency_us"],
                },
            )
        )

    for name, spec in DRIVES.items():
        summary.append(
            run_arm(
                f"queued_{name}",
                OUT / f"queued_{name}",
                DATA / spec["file"],
                {
                    "kind": "queued",
                    "drive": name,
                    "io_size_bytes": 4096,
                    "qd_cap": 32,
                    "read_latency_us": spec["read_latency_us"],
                },
            )
        )

    for name in ("slc", "n3"):
        latency = DRIVES[name]["read_latency_us"]
        for cap in (8, 128, 512):
            path = write_cfg(OUT / f"{name}_qd{cap}.json", queued_ssd(latency, cap))
            summary.append(
                run_arm(
                    f"{name}_qd{cap}",
                    OUT / f"{name}_qd{cap}",
                    path,
                    {
                        "kind": "qd_cap",
                        "drive": name,
                        "io_size_bytes": 4096,
                        "qd_cap": cap,
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
