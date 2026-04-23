from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import relay_bp

from .code_capacity_common import (
    CodeCapacityProblem,
    decode_with_plain_bp,
    enumerate_half_stabilizer_cases,
    evaluate_decode_result,
    make_min_sum_tracer,
)


EXPECTED_HALF_NULL_SEMANTICS = "heuristic_low_weight_independent_nulls_in_kernel_of_[Hz;Lz]"
DEFAULT_GAP_THRESHOLD_EXACT = 1e-12
DEFAULT_GAP_THRESHOLD_NEAR = 1e-3
PAIR_CLASS_ORDER = (
    "exact_equal",
    "near_equal",
    "mild_bias",
    "biased",
    "strong_bias",
)
DECODED_MATCH_VALUES = ("half_a", "half_b", "neither", "both")


def validate_half_null_problem(problem: CodeCapacityProblem) -> None:
    semantics = str(problem.metadata.get("Hx_semantics", ""))
    if semantics and semantics != EXPECTED_HALF_NULL_SEMANTICS:
        raise ValueError(
            "This analysis expects projected circuit half-null exports with "
            f"Hx semantics {EXPECTED_HALF_NULL_SEMANTICS!r}; got {semantics!r}."
        )


def enumerate_half_null_cases(
    problem: CodeCapacityProblem,
    *,
    selected_row_weights: set[int] | None = None,
    max_cases: int | None = None,
    shuffle_seed: int | None = None,
) -> list[dict[str, Any]]:
    validate_half_null_problem(problem)
    return enumerate_half_stabilizer_cases(
        problem,
        selected_row_weights=selected_row_weights,
        max_cases=max_cases,
        shuffle_seed=shuffle_seed,
    )


def pair_half_null_cases(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for case in cases:
        support_bits = tuple(int(bit) for bit in case["support_bits"])
        flipped_bits = tuple(int(bit) for bit in case["flipped_bits"])
        flipped_set = set(flipped_bits)
        complement_bits = tuple(bit for bit in support_bits if bit not in flipped_set)
        half_a_bits, half_b_bits = sorted((flipped_bits, complement_bits))
        key = (
            int(case["row_idx"]),
            support_bits,
            half_a_bits,
            half_b_bits,
        )
        grouped.setdefault(key, []).append(case)

    pairs: list[dict[str, Any]] = []
    for (row_idx, support_bits, half_a_bits, half_b_bits), entries in grouped.items():
        by_flipped = {
            tuple(int(bit) for bit in entry["flipped_bits"]): entry
            for entry in entries
        }
        if half_a_bits not in by_flipped or half_b_bits not in by_flipped:
            continue
        pairs.append(
            {
                "pair_id": int(len(pairs)),
                "row_idx": int(row_idx),
                "row_weight": int(by_flipped[half_a_bits]["row_weight"]),
                "support_bits": support_bits,
                "case_a": by_flipped[half_a_bits],
                "case_b": by_flipped[half_b_bits],
            }
        )
    return pairs


def classify_posterior_gap(
    gap: float,
    *,
    gap_threshold_exact: float = DEFAULT_GAP_THRESHOLD_EXACT,
    gap_threshold_near: float = DEFAULT_GAP_THRESHOLD_NEAR,
) -> str:
    gap = float(gap)
    if gap < 0.0:
        raise ValueError(f"gap must be non-negative; got {gap}.")
    if gap_threshold_exact < 0.0:
        raise ValueError(f"gap_threshold_exact must be non-negative; got {gap_threshold_exact}.")
    if gap_threshold_near < gap_threshold_exact:
        raise ValueError(
            "gap_threshold_near must be greater than or equal to gap_threshold_exact."
        )
    if gap <= float(gap_threshold_exact):
        return "exact_equal"
    if gap <= float(gap_threshold_near):
        return "near_equal"
    if gap <= 1e-1:
        return "mild_bias"
    if gap <= 1.0:
        return "biased"
    return "strong_bias"


def compute_half_null_pair_posteriors(
    problem: CodeCapacityProblem,
    pairs: Sequence[dict[str, Any]],
    *,
    gap_threshold_exact: float = DEFAULT_GAP_THRESHOLD_EXACT,
    gap_threshold_near: float = DEFAULT_GAP_THRESHOLD_NEAR,
) -> list[dict[str, Any]]:
    validate_half_null_problem(problem)
    priors = _clipped_priors(problem.error_priors)
    log_odds = np.log(priors / (1.0 - priors))
    pair_rows: list[dict[str, Any]] = []
    for pair in pairs:
        case_a = pair["case_a"]
        case_b = pair["case_b"]
        error_a = np.asarray(case_a["error"], dtype=np.uint8)
        error_b = np.asarray(case_b["error"], dtype=np.uint8)
        half_a_bits = tuple(int(bit) for bit in case_a["flipped_bits"])
        half_b_bits = tuple(int(bit) for bit in case_b["flipped_bits"])
        half_a_log_odds = float(log_odds[list(half_a_bits)].sum())
        half_b_log_odds = float(log_odds[list(half_b_bits)].sum())
        posterior_gap_log_odds = abs(half_a_log_odds - half_b_log_odds)
        pair_rows.append(
            {
                "pair_id": int(pair["pair_id"]),
                "row_idx": int(pair["row_idx"]),
                "row_weight": int(pair["row_weight"]),
                "support_bits": list(pair["support_bits"]),
                "half_a_bits": list(half_a_bits),
                "half_b_bits": list(half_b_bits),
                "syndrome_weight": int(np.asarray(case_a["syndrome"], dtype=np.uint8).sum()),
                "half_a_log_odds_sum": half_a_log_odds,
                "half_b_log_odds_sum": half_b_log_odds,
                "posterior_gap_log_odds": float(posterior_gap_log_odds),
                "pair_class": classify_posterior_gap(
                    posterior_gap_log_odds,
                    gap_threshold_exact=gap_threshold_exact,
                    gap_threshold_near=gap_threshold_near,
                ),
                "half_a_log_prob": _event_log_prob(error_a, priors),
                "half_b_log_prob": _event_log_prob(error_b, priors),
            }
        )
    return pair_rows


def summarize_half_null_pair_posteriors(
    pair_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for pair_class in PAIR_CLASS_ORDER:
        class_rows = [row for row in pair_rows if row["pair_class"] == pair_class]
        if class_rows:
            gaps = np.asarray([row["posterior_gap_log_odds"] for row in class_rows], dtype=np.float64)
            summary[pair_class] = {
                "num_pairs": int(len(class_rows)),
                "min_gap": float(gaps.min()),
                "median_gap": float(np.median(gaps)),
                "max_gap": float(gaps.max()),
            }
        else:
            summary[pair_class] = {
                "num_pairs": 0,
                "min_gap": 0.0,
                "median_gap": 0.0,
                "max_gap": 0.0,
            }
    return summary


def audit_half_null_complement_symmetry(
    *,
    problem: CodeCapacityProblem,
    pairs: Sequence[dict[str, Any]],
    max_iter: int,
    alpha: float | None,
    max_pairs: int | None = None,
    gap_threshold_exact: float = DEFAULT_GAP_THRESHOLD_EXACT,
    gap_threshold_near: float = DEFAULT_GAP_THRESHOLD_NEAR,
) -> dict[str, Any]:
    candidate_rows = compute_half_null_pair_posteriors(
        problem,
        pairs,
        gap_threshold_exact=gap_threshold_exact,
        gap_threshold_near=gap_threshold_near,
    )
    selected_rows = candidate_rows[: max_pairs if max_pairs is not None else len(candidate_rows)]
    pair_rows = _decode_half_null_pair_rows(
        problem=problem,
        pairs=pairs,
        pair_rows=selected_rows,
        max_iter=max_iter,
        alpha=alpha,
    )
    summary = _summarize_half_null_symmetry_rows(pair_rows)
    return {
        "summary": summary,
        "pair_rows": pair_rows,
        "exact_split_examples": [row for row in pair_rows if row["exact_recovery_split"]][:10],
    }


def audit_half_null_posteriors(
    *,
    problem: CodeCapacityProblem,
    pairs: Sequence[dict[str, Any]],
    max_iter: int,
    alpha: float | None,
    pair_classes: Sequence[str] | None = None,
    max_pairs_per_class: int | None = None,
    gap_threshold_exact: float = DEFAULT_GAP_THRESHOLD_EXACT,
    gap_threshold_near: float = DEFAULT_GAP_THRESHOLD_NEAR,
) -> dict[str, Any]:
    validate_half_null_problem(problem)
    candidate_rows = compute_half_null_pair_posteriors(
        problem,
        pairs,
        gap_threshold_exact=gap_threshold_exact,
        gap_threshold_near=gap_threshold_near,
    )
    selected_rows = _select_pair_rows(
        pair_rows=candidate_rows,
        pair_classes=pair_classes,
        max_pairs_per_class=max_pairs_per_class,
    )
    decoded_rows = _decode_half_null_pair_rows(
        problem=problem,
        pairs=pairs,
        pair_rows=selected_rows,
        max_iter=max_iter,
        alpha=alpha,
    )
    return {
        "summary": _summarize_half_null_symmetry_rows(decoded_rows),
        "pair_rows": decoded_rows,
        "pair_class_counts": summarize_half_null_pair_posteriors(candidate_rows),
        "pair_class_summary": _summarize_pair_classes(decoded_rows),
        "decoded_matches_counts": _count_decoded_matches(decoded_rows),
        "representative_examples": {
            "exact_equal": _representative_examples(decoded_rows, "exact_equal", limit=5),
            "near_equal": _representative_examples(decoded_rows, "near_equal", limit=5),
            "strong_bias": _representative_examples(decoded_rows, "strong_bias", limit=5),
        },
        "gap_threshold_exact": float(gap_threshold_exact),
        "gap_threshold_near": float(gap_threshold_near),
    }


def trace_half_null_pair(
    *,
    problem: CodeCapacityProblem,
    pair: dict[str, Any],
    max_iter: int,
    alpha: float | None,
) -> dict[str, Any]:
    validate_half_null_problem(problem)
    trace_a = _trace_case(problem, pair["case_a"], max_iter=max_iter, alpha=alpha)
    trace_b = _trace_case(problem, pair["case_b"], max_iter=max_iter, alpha=alpha)
    return {
        "pair_id": int(pair["pair_id"]),
        "row_idx": int(pair["row_idx"]),
        "support_bits": list(pair["support_bits"]),
        "half_a_bits": list(pair["case_a"]["flipped_bits"]),
        "half_b_bits": list(pair["case_b"]["flipped_bits"]),
        "trace_a": trace_a,
        "trace_b": trace_b,
        "same_iteration_grid": bool(np.array_equal(trace_a["iterations"], trace_b["iterations"])),
        "same_posterior_trace": bool(
            trace_a["posterior_trace"].shape == trace_b["posterior_trace"].shape
            and np.allclose(trace_a["posterior_trace"], trace_b["posterior_trace"])
        ),
        "same_decoding": bool(np.array_equal(trace_a["final_decoding"], trace_b["final_decoding"])),
        "same_logical_success": bool(trace_a["logical_success"] == trace_b["logical_success"]),
    }


def _decode_half_null_pair_rows(
    *,
    problem: CodeCapacityProblem,
    pairs: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
    max_iter: int,
    alpha: float | None,
) -> list[dict[str, Any]]:
    validate_half_null_problem(problem)
    decoder = relay_bp.MinSumBPDecoderF64(
        problem.hz,
        error_priors=problem.error_priors,
        max_iter=int(max_iter),
        alpha=alpha,
    )
    priors = _clipped_priors(problem.error_priors)
    pairs_by_id = {int(pair["pair_id"]): pair for pair in pairs}
    decoded_rows: list[dict[str, Any]] = []
    for pair_row in pair_rows:
        pair = pairs_by_id[int(pair_row["pair_id"])]
        case_a = pair["case_a"]
        case_b = pair["case_b"]
        syndrome_a = np.asarray(case_a["syndrome"], dtype=np.uint8)
        syndrome_b = np.asarray(case_b["syndrome"], dtype=np.uint8)
        error_a = np.asarray(case_a["error"], dtype=np.uint8)
        error_b = np.asarray(case_b["error"], dtype=np.uint8)
        logical_a = np.asarray((problem.lz @ error_a) % 2, dtype=np.uint8).reshape(-1)
        logical_b = np.asarray((problem.lz @ error_b) % 2, dtype=np.uint8).reshape(-1)

        result_a = decoder.decode_detailed(syndrome_a)
        result_b = decoder.decode_detailed(syndrome_b)
        decoding_a = np.asarray(result_a.decoding, dtype=np.uint8)
        decoding_b = np.asarray(result_b.decoding, dtype=np.uint8)
        evaluation_a = evaluate_decode_result(result=result_a, error=error_a, lz=problem.lz)
        evaluation_b = evaluate_decode_result(result=result_b, error=error_b, lz=problem.lz)

        support_vector = np.zeros(problem.n_bits, dtype=np.uint8)
        support_vector[list(pair["support_bits"])] = 1
        decoded_matches = _decoded_matches(decoding_a, error_a, error_b)
        decoded_rows.append(
            {
                **pair_row,
                "syndrome_equal": bool(np.array_equal(syndrome_a, syndrome_b)),
                "logical_signature_equal": bool(np.array_equal(logical_a, logical_b)),
                "decoder_output_equal": bool(np.array_equal(decoding_a, decoding_b)),
                "decoder_iterations_a": int(result_a.iterations),
                "decoder_iterations_b": int(result_b.iterations),
                "decoder_iterations_equal": int(result_a.iterations) == int(result_b.iterations),
                "decoder_success_a": bool(result_a.success),
                "decoder_success_b": bool(result_b.success),
                "decoder_success_equal": bool(result_a.success) == bool(result_b.success),
                "logical_success_equal": bool(evaluation_a.logical_success)
                == bool(evaluation_b.logical_success),
                "exact_recovery_a": bool(evaluation_a.exact_recovery),
                "exact_recovery_b": bool(evaluation_b.exact_recovery),
                "exact_recovery_split": bool(evaluation_a.exact_recovery)
                != bool(evaluation_b.exact_recovery),
                "logical_success_a": bool(evaluation_a.logical_success),
                "logical_success_b": bool(evaluation_b.logical_success),
                "logical_weight_a": int(evaluation_a.logical_weight),
                "logical_weight_b": int(evaluation_b.logical_weight),
                "decoded_bits": np.flatnonzero(decoding_a).astype(int).tolist(),
                "decoded_log_prob": _event_log_prob(decoding_a, priors),
                "decoded_matches": decoded_matches,
                "residual_relation_holds": bool(
                    np.array_equal(
                        np.bitwise_xor(evaluation_a.residual, evaluation_b.residual),
                        support_vector,
                    )
                ),
            }
        )
    return decoded_rows


def _summarize_half_null_symmetry_rows(
    pair_rows: Sequence[dict[str, Any]],
) -> dict[str, float | int]:
    num_pairs = len(pair_rows)
    return {
        "num_pairs": int(num_pairs),
        "syndrome_mismatches": int(sum(not row["syndrome_equal"] for row in pair_rows)),
        "logical_signature_mismatches": int(
            sum(not row["logical_signature_equal"] for row in pair_rows)
        ),
        "decoder_output_mismatches": int(
            sum(not row["decoder_output_equal"] for row in pair_rows)
        ),
        "decoder_iteration_mismatches": int(
            sum(not row["decoder_iterations_equal"] for row in pair_rows)
        ),
        "decoder_success_mismatches": int(
            sum(not row["decoder_success_equal"] for row in pair_rows)
        ),
        "logical_success_mismatches": int(
            sum(not row["logical_success_equal"] for row in pair_rows)
        ),
        "residual_relation_failures": int(
            sum(not row["residual_relation_holds"] for row in pair_rows)
        ),
        "exact_recovery_splits": int(sum(row["exact_recovery_split"] for row in pair_rows)),
        "exact_recovery_split_rate": (
            float(sum(row["exact_recovery_split"] for row in pair_rows) / num_pairs)
            if num_pairs
            else 0.0
        ),
    }


def _summarize_pair_classes(
    pair_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for pair_class in PAIR_CLASS_ORDER:
        class_rows = [row for row in pair_rows if row["pair_class"] == pair_class]
        if not class_rows:
            summary[pair_class] = {
                "num_pairs": 0,
                "convergence_rate": 0.0,
                "logical_success_rate": 0.0,
                "mean_iterations": 0.0,
                "exact_recovery_split_rate": 0.0,
            }
            continue
        convergence_rate = np.mean(
            [
                0.5 * (float(row["decoder_success_a"]) + float(row["decoder_success_b"]))
                for row in class_rows
            ]
        )
        logical_success_rate = np.mean(
            [
                0.5 * (float(row["logical_success_a"]) + float(row["logical_success_b"]))
                for row in class_rows
            ]
        )
        mean_iterations = np.mean(
            [
                0.5 * (float(row["decoder_iterations_a"]) + float(row["decoder_iterations_b"]))
                for row in class_rows
            ]
        )
        exact_recovery_split_rate = np.mean(
            [float(row["exact_recovery_split"]) for row in class_rows]
        )
        summary[pair_class] = {
            "num_pairs": int(len(class_rows)),
            "convergence_rate": float(convergence_rate),
            "logical_success_rate": float(logical_success_rate),
            "mean_iterations": float(mean_iterations),
            "exact_recovery_split_rate": float(exact_recovery_split_rate),
        }
    return summary


def _count_decoded_matches(pair_rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in DECODED_MATCH_VALUES}
    for row in pair_rows:
        counts[str(row["decoded_matches"])] += 1
    return counts


def _representative_examples(
    pair_rows: Sequence[dict[str, Any]],
    pair_class: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    class_rows = [row for row in pair_rows if row["pair_class"] == pair_class]
    if pair_class == "strong_bias":
        class_rows = sorted(
            class_rows,
            key=lambda row: (-float(row["posterior_gap_log_odds"]), int(row["pair_id"])),
        )
    else:
        class_rows = sorted(
            class_rows,
            key=lambda row: (float(row["posterior_gap_log_odds"]), int(row["pair_id"])),
        )
    return class_rows[:limit]


def _select_pair_rows(
    *,
    pair_rows: Sequence[dict[str, Any]],
    pair_classes: Sequence[str] | None,
    max_pairs_per_class: int | None,
) -> list[dict[str, Any]]:
    allowed_classes = None if pair_classes is None else {str(value) for value in pair_classes}
    selected_rows: list[dict[str, Any]] = []
    per_class_counts = {pair_class: 0 for pair_class in PAIR_CLASS_ORDER}
    for row in pair_rows:
        pair_class = str(row["pair_class"])
        if allowed_classes is not None and pair_class not in allowed_classes:
            continue
        if max_pairs_per_class is not None and per_class_counts[pair_class] >= int(max_pairs_per_class):
            continue
        selected_rows.append(row)
        per_class_counts[pair_class] += 1
    return selected_rows


def _decoded_matches(
    decoded: np.ndarray,
    error_a: np.ndarray,
    error_b: np.ndarray,
) -> str:
    matches_a = bool(np.array_equal(decoded, error_a))
    matches_b = bool(np.array_equal(decoded, error_b))
    if matches_a and matches_b:
        return "both"
    if matches_a:
        return "half_a"
    if matches_b:
        return "half_b"
    return "neither"


def _trace_case(
    problem: CodeCapacityProblem,
    case: dict[str, Any],
    *,
    max_iter: int,
    alpha: float | None,
) -> dict[str, Any]:
    tracer = make_min_sum_tracer(problem, max_iter=max_iter, alpha=alpha, gamma0=None)
    syndrome = np.asarray(case["syndrome"], dtype=np.uint8)
    result = decode_with_plain_bp(tracer, syndrome, n_bits=problem.n_bits)
    evaluation = evaluate_decode_result(
        result=result,
        error=np.asarray(case["error"], dtype=np.uint8),
        lz=problem.lz,
    )

    tracer.reset()
    tracer.clear_explicit_c_damp_messages()
    tracer.set_memory_strengths(np.zeros(problem.n_bits, dtype=np.float64))
    snapshots = [tracer.snapshot(syndrome)]
    while tracer.current_iteration < snapshots[-1].max_iter and not snapshots[-1].success:
        snapshots.append(tracer.run_iteration(syndrome))

    return {
        "iterations": np.asarray([int(snapshot.iterations) for snapshot in snapshots], dtype=int),
        "posterior_trace": np.vstack(
            [np.asarray(snapshot.posterior_ratios, dtype=np.float64) for snapshot in snapshots]
        ),
        "final_decoding": np.asarray(result.decoding, dtype=np.uint8),
        "logical_success": bool(evaluation.logical_success),
        "exact_recovery": bool(evaluation.exact_recovery),
    }


def _clipped_priors(error_priors: np.ndarray) -> np.ndarray:
    priors = np.asarray(error_priors, dtype=np.float64).reshape(-1)
    return np.clip(priors, 1e-15, 1.0 - 1e-15)


def _event_log_prob(error: np.ndarray, priors: np.ndarray) -> float:
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    priors = np.asarray(priors, dtype=np.float64).reshape(-1)
    return float((error * np.log(priors) + (1.0 - error) * np.log1p(-priors)).sum())
