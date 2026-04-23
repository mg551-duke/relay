from __future__ import annotations

from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .code_capacity_common import (
    CodeCapacityProblem,
    build_edge_damping_messages,
    build_half_stabilizer_case_from_seed,
    decode_with_edge_damping,
    decode_with_plain_bp,
    evaluate_decode_result,
    make_min_sum_tracer,
)


def interval_from_center_width(center: float, width: float) -> tuple[float, float]:
    half_width = max(0.0, float(width) / 2.0)
    low = max(0.0, float(center) - half_width)
    high = min(1.0, float(center) + half_width)
    if low > high:
        low, high = high, low
    return (low, high)


def evaluate_interval_heatmap(
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
    show_progress: bool = False,
    show_inner_case_progress: bool = False,
) -> dict[str, Any]:
    tracer = make_min_sum_tracer(problem, max_iter=max_iter, alpha=alpha, gamma0=None)
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
        desc="heatmap grid",
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
            for draw_idx in range(int(random_draws_per_point)):
                coeff_seed = (
                    int(base_seed)
                    + 1_000_003 * int(case.get("case_id", case.get("sample_idx", 0)))
                    + 10_007 * int(width_idx)
                    + 1_009 * int(center_idx)
                    + int(draw_idx)
                )
                support_bits = case["support_bits"] if support_only and "support_bits" in case else None
                messages = build_edge_damping_messages(
                    problem.hz,
                    interval=interval,
                    coeff_seed=coeff_seed,
                    support_bits=support_bits,
                )
                result = decode_with_edge_damping(
                    tracer,
                    case["syndrome"],
                    messages,
                    n_bits=problem.n_bits,
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
    }


def heatmap_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
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


def evaluate_decoder_family(
    *,
    problem: CodeCapacityProblem,
    cases: list[dict[str, Any]],
    max_iter: int,
    alpha: float | None,
    same_damping_value: float,
    random_interval: tuple[float, float],
    base_seed: int,
    random_seed_offsets: list[int],
    support_only: bool,
) -> dict[str, list[dict[str, Any]]]:
    plain_tracer = make_min_sum_tracer(problem, max_iter=max_iter, alpha=alpha, gamma0=None)
    damping_tracer = make_min_sum_tracer(problem, max_iter=max_iter, alpha=alpha, gamma0=None)
    rows: dict[str, list[dict[str, Any]]] = {
        "bp": [],
        "damp_same": [],
        "damp_random_interval": [],
    }

    for case in cases:
        support_bits = case["support_bits"] if support_only and "support_bits" in case else None
        plain_result = decode_with_plain_bp(plain_tracer, case["syndrome"], n_bits=problem.n_bits)
        plain_eval = evaluate_decode_result(result=plain_result, error=case["error"], lz=problem.lz)
        rows["bp"].append(
            _case_row("bp", case, plain_result, plain_eval, coeff_seed=None, coeff_seed_offset=None)
        )

        same_messages = build_edge_damping_messages(
            problem.hz,
            interval=(same_damping_value, same_damping_value),
            coeff_seed=0,
            support_bits=support_bits,
        )
        same_result = decode_with_edge_damping(
            damping_tracer,
            case["syndrome"],
            same_messages,
            n_bits=problem.n_bits,
        )
        same_eval = evaluate_decode_result(result=same_result, error=case["error"], lz=problem.lz)
        rows["damp_same"].append(
            _case_row(
                "damp_same",
                case,
                same_result,
                same_eval,
                coeff_seed=0,
                coeff_seed_offset=0,
            )
        )

        for coeff_seed_offset in random_seed_offsets:
            coeff_seed = int(base_seed) + (10_000 * int(coeff_seed_offset)) + int(
                case.get("case_id", case.get("sample_idx", 0))
            )
            random_messages = build_edge_damping_messages(
                problem.hz,
                interval=random_interval,
                coeff_seed=coeff_seed,
                support_bits=support_bits,
            )
            random_result = decode_with_edge_damping(
                damping_tracer,
                case["syndrome"],
                random_messages,
                n_bits=problem.n_bits,
            )
            random_eval = evaluate_decode_result(
                result=random_result,
                error=case["error"],
                lz=problem.lz,
            )
            rows["damp_random_interval"].append(
                _case_row(
                    "damp_random_interval",
                    case,
                    random_result,
                    random_eval,
                    coeff_seed=coeff_seed,
                    coeff_seed_offset=coeff_seed_offset,
                )
            )

    return rows


def run_trace_suite(
    *,
    problem: CodeCapacityProblem,
    core_seed: int,
    max_iter: int,
    alpha: float | None,
    same_damping_value: float,
    random_interval: tuple[float, float],
    random_seed_offsets: list[int],
    support_only: bool,
) -> dict[str, Any]:
    case = build_half_stabilizer_case_from_seed(problem, core_seed)
    traced_bits = tuple(case["flipped_bits"])
    plain_tracer = make_min_sum_tracer(problem, max_iter=max_iter, alpha=alpha, gamma0=None)
    damping_tracer = make_min_sum_tracer(problem, max_iter=max_iter, alpha=alpha, gamma0=None)
    runs = [
        _trace_run(
            decoder_kind="bp",
            tracer=plain_tracer,
            syndrome=case["syndrome"],
            traced_bits=traced_bits,
            decode_kind="plain",
            n_bits=problem.n_bits,
            error=case["error"],
            lz=problem.lz,
        ),
        _trace_run(
            decoder_kind="damp_same",
            tracer=damping_tracer,
            syndrome=case["syndrome"],
            traced_bits=traced_bits,
            decode_kind="damping",
            n_bits=problem.n_bits,
            error=case["error"],
            lz=problem.lz,
            edge_messages=build_edge_damping_messages(
                problem.hz,
                interval=(same_damping_value, same_damping_value),
                coeff_seed=0,
                support_bits=case["support_bits"] if support_only else None,
            ),
            coeff_seed=0,
            coeff_seed_offset=0,
        ),
    ]
    for coeff_seed_offset in random_seed_offsets:
        coeff_seed = int(core_seed) + (10_000 * int(coeff_seed_offset))
        runs.append(
            _trace_run(
                decoder_kind="damp_random_interval",
                tracer=damping_tracer,
                syndrome=case["syndrome"],
                traced_bits=traced_bits,
                decode_kind="damping",
                n_bits=problem.n_bits,
                error=case["error"],
                lz=problem.lz,
                edge_messages=build_edge_damping_messages(
                    problem.hz,
                    interval=random_interval,
                    coeff_seed=coeff_seed,
                    support_bits=case["support_bits"] if support_only else None,
                ),
                coeff_seed=coeff_seed,
                coeff_seed_offset=coeff_seed_offset,
            )
        )
    return {"case": case, "traced_bits": traced_bits, "runs": runs}


def _case_row(
    decoder_kind: str,
    case: dict[str, Any],
    result: Any,
    evaluation: Any,
    *,
    coeff_seed: int | None,
    coeff_seed_offset: int | None,
) -> dict[str, Any]:
    return {
        "decoder_kind": decoder_kind,
        "case_id": int(case.get("case_id", case.get("sample_idx", -1))),
        "row_idx": int(case.get("row_idx", -1)),
        "row_weight": int(case.get("row_weight", -1)),
        "half_subset_idx": int(case.get("half_subset_idx", -1)),
        "support_bits": list(case.get("support_bits", ())),
        "flipped_bits": list(case.get("flipped_bits", ())),
        "syndrome_weight": int(case.get("syndrome_weight", 0)),
        "coeff_seed": coeff_seed,
        "coeff_seed_offset": coeff_seed_offset,
        "converged": bool(result.success),
        "logical_success": bool(evaluation.logical_success),
        "exact_recovery": bool(evaluation.exact_recovery),
        "logical_weight": int(evaluation.logical_weight),
        "iterations": int(result.iterations),
        "damping_mode": str(result.damping_mode),
        "damping_min": result.damping_min,
        "damping_max": result.damping_max,
    }


def _trace_run(
    *,
    decoder_kind: str,
    tracer: Any,
    syndrome: np.ndarray,
    traced_bits: tuple[int, ...],
    decode_kind: str,
    n_bits: int,
    error: np.ndarray,
    lz: np.ndarray,
    edge_messages: np.ndarray | None = None,
    coeff_seed: int | None = None,
    coeff_seed_offset: int | None = None,
) -> dict[str, Any]:
    tracer.reset()
    tracer.set_memory_strengths(np.zeros(int(n_bits), dtype=np.float64))
    tracer.clear_explicit_c_damp_messages()
    if decode_kind == "damping":
        assert edge_messages is not None
        tracer.set_explicit_c_damp_messages(edge_messages)
    snapshots = [tracer.snapshot(np.asarray(syndrome, dtype=np.uint8))]
    while tracer.current_iteration < snapshots[-1].max_iter and not snapshots[-1].success:
        snapshots.append(tracer.run_iteration(np.asarray(syndrome, dtype=np.uint8)))
    final_result = snapshots[-1]
    evaluation = evaluate_decode_result(result=final_result, error=error, lz=lz)
    posterior_trace = np.vstack(
        [np.asarray(snapshot.posterior_ratios, dtype=np.float64) for snapshot in snapshots]
    )
    return {
        "decoder_kind": decoder_kind,
        "coeff_seed": coeff_seed,
        "coeff_seed_offset": coeff_seed_offset,
        "iterations": np.asarray([int(snapshot.iterations) for snapshot in snapshots], dtype=int),
        "posterior_trace": posterior_trace,
        "traced_bits": traced_bits,
        "final_iterations": int(final_result.iterations),
        "success": bool(final_result.success),
        "final_decoding": np.asarray(final_result.decoding, dtype=np.uint8),
        "logical_success": bool(evaluation.logical_success),
        "exact_recovery": bool(evaluation.exact_recovery),
        "logical_weight": int(evaluation.logical_weight),
        "damping_mode": str(final_result.damping_mode),
        "damping_min": final_result.damping_min,
        "damping_max": final_result.damping_max,
    }
