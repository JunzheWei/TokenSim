from __future__ import annotations

from typing import Callable

LIGHT_QPS = 0.02
MAX_QPS = 1.0
GOODPUT_RATIO = 0.90
TTFT_P99_MULT = 3.0
QPS_RESOLUTION = 0.02


def snap_qps(qps: float) -> float:
    steps = round(float(qps) / QPS_RESOLUTION)
    return round(max(QPS_RESOLUTION, steps * QPS_RESOLUTION), 4)


def is_stable(row: dict, offered_qps: float, light_ttft_p99: float) -> bool:
    if offered_qps <= 0.0:
        return False
    output_qps = float(row["output_qps"])
    if output_qps / offered_qps < GOODPUT_RATIO:
        return False
    cap = TTFT_P99_MULT * max(float(light_ttft_p99), 1e-9)
    return float(row["ttft_p99"]) <= cap


def little_concurrency(offered_qps: float, request_time_p50: float) -> float:
    return float(offered_qps) * float(request_time_p50)


def search_knee_qps(
    evaluate: Callable[[float], dict],
    *,
    light_qps: float = LIGHT_QPS,
    max_qps: float = MAX_QPS,
    resolution: float = QPS_RESOLUTION,
) -> tuple[float, dict[float, dict]]:
    """Geometric probe then binary search for the largest stable offered QPS."""
    points: dict[float, dict] = {}

    def run(qps: float) -> dict:
        qps = snap_qps(qps)
        cached = points.get(qps)
        if cached is None:
            cached = evaluate(qps)
            points[qps] = cached
        return cached

    light = snap_qps(light_qps)
    light_row = run(light)
    light_p99 = float(light_row["ttft_p99"])
    last_stable: float | None = (
        light if is_stable(light_row, light, light_p99) else None
    )
    first_fail: float | None = None
    qps = snap_qps(light * 2.0)
    cap = snap_qps(max_qps)
    while qps <= cap + 1e-12:
        row = run(qps)
        if is_stable(row, qps, light_p99):
            last_stable = qps
            nxt = snap_qps(qps * 2.0)
            if nxt <= qps:
                break
            qps = nxt
            continue
        first_fail = qps
        break
    if last_stable is None:
        return light, points
    if first_fail is None:
        return last_stable, points
    lo = last_stable
    hi = first_fail
    while hi - lo > resolution + 1e-12:
        mid = snap_qps((lo + hi) / 2.0)
        if mid <= lo or mid >= hi:
            break
        row = run(mid)
        if is_stable(row, mid, light_p99):
            lo = mid
        else:
            hi = mid
    return lo, points
