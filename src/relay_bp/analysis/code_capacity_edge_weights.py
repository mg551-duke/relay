from __future__ import annotations

from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .code_capacity_common import (
    CodeCapacityProblem,
    build_edge_damping_messages,
    build_edge_message_weights,
    decode_with_memory_edge_damping_and_edge_weights,
    evaluate_decode_result,
    make_min_sum_tracer,
)


def interval_from_center_width(center: float, width: float) -> tuple[float, float]:
    if float(width) < 0.0:
        raise ValueError("width must be non-negative")
    half_width = float(width) / 2.0
    return (float(center) - half_width, float(center) + half_width)


def build_memory_strengths(
    n_bits: int,
    *,
    interval: tuple[float, float],
    coeff_seed: int,
    support_bits: tuple[int, ...] | list[int] | None = None,
    outside_value: float = 0.0,
) -> np.ndarray:
    low, high = float(interval[0]), float(interval[1])
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"Memory interval endpoints must be finite; got {interval}.")
    if low > high:
        raise ValueError(f"Memory interval low must be <= high; got {interval}.")
    if not np.isfinite(float(outside_value)):
        raise ValueError(f"outside_value must be finite; got {outside_value}.")
    weights = np.full(int(n_bits), float(outside_value), dtype=np.float64)
    if support_bits is None:
        target_indices = np.arange(int(n_bits), dtype=int)
    else:
        target_indices = np.unique(np.asarray(support_bits, dtype=int))
    if target_indices.size == 0:
        return weights
    if np.any(target_indices < 0) or np.any(target_indices >= int(n_bits)):
        raise ValueError(f"support_bits must stay within [0, {int(n_bits)}).")
    if np.isclose(low, high):
        weights[target_indices] = low
    else:
        rng = np.random.default_rng(int(coeff_seed))
        weights[target_indices] = rng.uniform(low, high, size=target_indices.size)
    return weights


def evaluate_edge_weight_heatmap(
    *,
    problem: CodeCapacityProblem,
    cases: list[dict[str, Any]],
    max_iter: int,
    alpha: float | None,
    centers: list[float],
    widths: list[float],
    base_seed: int,
    random_draws_per_point: int,
    support_only: bool,
    memory_interval: tuple[float, float] | None = None,
    damping_interval: tuple[float, float] | None = None,
    show_progress: bool = False,
    show_inner_case_progress: bool = False,
) -> dict[str, Any]:
    tracer = make_min_sum_tracer(
        problem,
        max_iter=max_iter,
        alpha=alpha,
        gamma0=0.0 if memory_interval is not None else None,
    )
    shape = (len(widths), len(centers))
    logical_success_rate = np.full(shape, np.nan, dtype=np.float64)
    convergence_rate = np.full(shape, np.nan, dtype=np.float64)
    exact_recovery_rate = np.full(shape, np.nan, dtype=np.float64)
    mean_iterations = np.full(shape, np.nan, dtype=np.float64)
    trial_count = np.zeros(shape, dtype=np.int64)
    interval_lows = np.full(shape, np.nan, dtype=np.float64)
    interval_highs = np.full(shape, np.nan, dtype=np.float64)

    parameter_points = [
        (width_idx, float(width), center_idx, float(center))
        for width_idx, width in enumerate(widths)
        for center_idx, center in enumerate(centers)
    ]
    parameter_points_iter = tqdm(
        parameter_points,
        total=len(parameter_points),
        desc="edge-weight grid",
        disable=not show_progress,
    )

    for width_idx, width, center_idx, center in parameter_points_iter:
        interval = interval_from_center_width(center, width)
        interval_lows[width_idx, center_idx] = interval[0]
        interval_highs[width_idx, center_idx] = interval[1]
        logical_success_sum = 0.0
        convergence_sum = 0.0
        exact_sum = 0.0
        iterations_sum = 0.0
        trials = 0
        cases_iter = tqdm(
            cases,
            total=len(cases),
            desc=f"w={width:.3f}, c={center:.3f}",
            leave=False,
            disable=(not show_progress) or (not show_inner_case_progress),
        )
        for case in cases_iter:
            case_id = int(case.get("case_id", case.get("sample_idx", 0)))
            support_bits = case["support_bits"] if support_only and "support_bits" in case else None
            for draw_idx in range(int(random_draws_per_point)):
                coeff_seed = (
                    int(base_seed)
                    + 1_000_003 * case_id
                    + 10_007 * int(width_idx)
                    + 1_009 * int(center_idx)
                    + int(draw_idx)
                )
                edge_message_weights = build_edge_message_weights(
                    problem.hz,
                    interval=interval,
                    coeff_seed=coeff_seed,
                    support_bits=support_bits,
                )
                edge_damping_messages = None
                if damping_interval is not None:
                    edge_damping_messages = build_edge_damping_messages(
                        problem.hz,
                        interval=damping_interval,
                        coeff_seed=coeff_seed + 104_729,
                        support_bits=support_bits,
                    )
                memory_strengths = None
                if memory_interval is not None:
                    memory_strengths = build_memory_strengths(
                        problem.n_bits,
                        interval=memory_interval,
                        coeff_seed=coeff_seed + 130_363,
                        support_bits=support_bits,
                    )
                result = decode_with_memory_edge_damping_and_edge_weights(
                    tracer,
                    case["syndrome"],
                    n_bits=problem.n_bits,
                    memory_strengths=memory_strengths,
                    edge_damping_messages=edge_damping_messages,
                    edge_message_weights=edge_message_weights,
                )
                evaluation = evaluate_decode_result(
                    result=result,
                    error=case["error"],
                    lz=problem.lz,
                )
                logical_success_sum += float(evaluation.logical_success)
                convergence_sum += float(evaluation.converged)
                exact_sum += float(evaluation.exact_recovery)
                iterations_sum += float(evaluation.iterations)
                trials += 1
        if trials:
            logical_success_rate[width_idx, center_idx] = logical_success_sum / trials
            convergence_rate[width_idx, center_idx] = convergence_sum / trials
            exact_recovery_rate[width_idx, center_idx] = exact_sum / trials
            mean_iterations[width_idx, center_idx] = iterations_sum / trials
            trial_count[width_idx, center_idx] = trials
            parameter_points_iter.set_postfix(
                width=f"{width:.3f}",
                center=f"{center:.3f}",
                logical=f"{logical_success_rate[width_idx, center_idx]:.3f}",
                conv=f"{convergence_rate[width_idx, center_idx]:.3f}",
            )

    return {
        "centers": np.asarray(centers, dtype=np.float64),
        "widths": np.asarray(widths, dtype=np.float64),
        "interval_lows": interval_lows,
        "interval_highs": interval_highs,
        "logical_success_rate": logical_success_rate,
        "convergence_rate": convergence_rate,
        "exact_recovery_rate": exact_recovery_rate,
        "mean_iterations": mean_iterations,
        "trial_count": trial_count,
        "memory_interval": None if memory_interval is None else tuple(float(v) for v in memory_interval),
        "damping_interval": None if damping_interval is None else tuple(float(v) for v in damping_interval),
    }


def edge_weight_heatmap_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width_idx, width in enumerate(artifact["widths"]):
        for center_idx, center in enumerate(artifact["centers"]):
            rows.append(
                {
                    "center": float(center),
                    "width": float(width),
                    "low": float(artifact["interval_lows"][width_idx, center_idx]),
                    "high": float(artifact["interval_highs"][width_idx, center_idx]),
                    "logical_success_rate": float(artifact["logical_success_rate"][width_idx, center_idx]),
                    "convergence_rate": float(artifact["convergence_rate"][width_idx, center_idx]),
                    "exact_recovery_rate": float(artifact["exact_recovery_rate"][width_idx, center_idx]),
                    "mean_iterations": float(artifact["mean_iterations"][width_idx, center_idx]),
                    "trial_count": int(artifact["trial_count"][width_idx, center_idx]),
                }
            )
    return rows


def best_edge_weight_rows(rows: list[dict[str, Any]], *, topk: int = 10) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["logical_success_rate"]),
            -float(row["convergence_rate"]),
            float(row["mean_iterations"]),
            -float(row["exact_recovery_rate"]),
            float(row["width"]),
            float(row["center"]),
        ),
    )[: max(1, int(topk))]
