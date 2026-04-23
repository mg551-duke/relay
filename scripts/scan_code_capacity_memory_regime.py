#!/usr/bin/env python3
"""Scan p-regimes and coarse static-memory intervals for code-capacity experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from relay_bp.analysis import (
    evaluate_memory_weight_vector,
    load_code_capacity_problem,
    sample_random_error_cases,
    with_uniform_error_rate,
)


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in str(raw).split(",")]
    parsed = [float(item) for item in values if item]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return parsed


def _sample_static_interval_weights(
    *,
    n_bits: int,
    low: float,
    high: float,
    seed: int,
) -> np.ndarray:
    if high < low:
        raise ValueError(f"Interval high must be >= low; got ({low}, {high}).")
    if np.isclose(low, high):
        return np.full(int(n_bits), float(low), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    return rng.uniform(float(low), float(high), size=int(n_bits)).astype(np.float64)


def _aggregate_interval_metrics(
    *,
    problem,
    cases,
    low: float,
    high: float,
    draws_per_point: int,
    base_seed: int,
    max_iter: int,
    alpha: float | None,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
) -> dict[str, float]:
    total_loss = 0.0
    total_logical = 0.0
    total_exact = 0.0
    total_convergence = 0.0
    total_iterations = 0.0
    draw_count = max(1, int(draws_per_point))
    for draw_idx in range(draw_count):
        weights = _sample_static_interval_weights(
            n_bits=problem.n_bits,
            low=low,
            high=high,
            seed=int(base_seed) + draw_idx,
        )
        metrics = evaluate_memory_weight_vector(
            problem=problem,
            cases=cases,
            memory_strengths=weights,
            max_iter=max_iter,
            alpha=alpha,
            logical_weight_penalty=logical_weight_penalty,
            convergence_penalty=convergence_penalty,
            iteration_penalty=iteration_penalty,
        )
        total_loss += float(metrics["loss_mean"])
        total_logical += float(metrics["logical_success_rate"])
        total_exact += float(metrics["exact_recovery_rate"])
        total_convergence += float(metrics["convergence_rate"])
        total_iterations += float(metrics["mean_iterations"])
    return {
        "loss_mean": total_loss / draw_count,
        "logical_success_rate": total_logical / draw_count,
        "exact_recovery_rate": total_exact / draw_count,
        "convergence_rate": total_convergence / draw_count,
        "mean_iterations": total_iterations / draw_count,
    }


def _recommend_p(summary_rows: list[dict[str, object]], *, perfect_threshold: float) -> dict[str, object] | None:
    eligible: list[tuple[float, dict[str, object]]] = []
    fallback: list[tuple[float, dict[str, object]]] = []
    for row in summary_rows:
        bp = row["bp"]
        best_interval = row["best_interval"]
        logical_gain = float(best_interval["logical_success_rate"]) - float(bp["logical_success_rate"])
        convergence_gain = float(best_interval["convergence_rate"]) - float(bp["convergence_rate"])
        score = logical_gain + 0.35 * convergence_gain
        fallback.append((score, row))
        bp_nontrivial = 0.0 < float(bp["logical_success_rate"]) < float(perfect_threshold)
        interval_nontrivial = (
            float(best_interval["logical_success_rate"]) < float(perfect_threshold)
            or float(best_interval["convergence_rate"]) < float(perfect_threshold)
        )
        if bp_nontrivial and interval_nontrivial:
            eligible.append((score, row))
    if eligible:
        return max(eligible, key=lambda item: item[0])[1]
    if fallback:
        return max(fallback, key=lambda item: item[0])[1]
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--p-values", type=_parse_float_list, default=_parse_float_list("0.01,0.02,0.03,0.05,0.07,0.09"))
    parser.add_argument(
        "--center-values",
        type=_parse_float_list,
        default=_parse_float_list("0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"),
    )
    parser.add_argument(
        "--width-values",
        type=_parse_float_list,
        default=_parse_float_list("0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0"),
    )
    parser.add_argument("--n-error-samples", type=int, default=256)
    parser.add_argument("--draws-per-point", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--logical-weight-penalty", type=float, default=4.0)
    parser.add_argument("--convergence-penalty", type=float, default=1.0)
    parser.add_argument("--iteration-penalty", type=float, default=0.05)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--perfect-threshold", type=float, default=0.999999)
    parser.add_argument("--top-intervals", type=int, default=10)
    parser.add_argument("--use-file-priors", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_problem = load_code_capacity_problem(args.code_path)
    summary_rows: list[dict[str, object]] = []
    for p_idx, p in enumerate(args.p_values):
        problem = base_problem if args.use_file_priors else with_uniform_error_rate(base_problem, p)
        cases = sample_random_error_cases(
            problem,
            num_samples=args.n_error_samples,
            error_rate=p,
            base_seed=int(args.base_seed) + 10_000 * p_idx,
        )
        bp_metrics = evaluate_memory_weight_vector(
            problem=problem,
            cases=cases,
            memory_strengths=None,
            max_iter=args.max_iter,
            alpha=args.alpha,
            logical_weight_penalty=args.logical_weight_penalty,
            convergence_penalty=args.convergence_penalty,
            iteration_penalty=args.iteration_penalty,
        )
        interval_rows: list[dict[str, object]] = []
        for width_idx, width in enumerate(args.width_values):
            for center_idx, center in enumerate(args.center_values):
                low = float(center) - float(width) / 2.0
                high = float(center) + float(width) / 2.0
                metrics = _aggregate_interval_metrics(
                    problem=problem,
                    cases=cases,
                    low=low,
                    high=high,
                    draws_per_point=args.draws_per_point,
                    base_seed=int(args.base_seed) + 1_000_000 * p_idx + 10_000 * width_idx + 100 * center_idx,
                    max_iter=args.max_iter,
                    alpha=args.alpha,
                    logical_weight_penalty=args.logical_weight_penalty,
                    convergence_penalty=args.convergence_penalty,
                    iteration_penalty=args.iteration_penalty,
                )
                interval_rows.append(
                    {
                        "center": float(center),
                        "width": float(width),
                        "interval_low": low,
                        "interval_high": high,
                        **metrics,
                    }
                )
        ranked_intervals = sorted(
            interval_rows,
            key=lambda row: (
                float(row["loss_mean"]),
                -float(row["logical_success_rate"]),
                -float(row["convergence_rate"]),
                float(row["mean_iterations"]),
            ),
        )
        summary_rows.append(
            {
                "p": float(p),
                "bp": bp_metrics,
                "best_interval": ranked_intervals[0],
                "top_intervals": ranked_intervals[: max(1, int(args.top_intervals))],
            }
        )

    payload = {
        "code_path": str(Path(args.code_path).resolve()),
        "n_bits": int(base_problem.n_bits),
        "n_logicals": int(base_problem.n_logicals),
        "n_error_samples": int(args.n_error_samples),
        "draws_per_point": int(args.draws_per_point),
        "max_iter": int(args.max_iter),
        "alpha": None if args.alpha is None else float(args.alpha),
        "use_file_priors": bool(args.use_file_priors),
        "p_summaries": summary_rows,
        "recommended": _recommend_p(summary_rows, perfect_threshold=float(args.perfect_threshold)),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
