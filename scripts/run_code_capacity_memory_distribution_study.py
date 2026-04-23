#!/usr/bin/env python3
"""Run a scriptable Gaussian-vs-uniform memory-distribution study for code-capacity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from relay_bp.analysis import (
    EquivalenceThresholds,
    MemorySamplingSpec,
    best_heatmap_rows,
    bootstrap_confidence_interval,
    default_export_code_paths,
    enumerate_half_stabilizer_cases,
    evaluate_equivalence,
    evaluate_gaussian_refinement_heatmap,
    evaluate_matched_support_heatmaps,
    evaluate_memory_weight_vector,
    evaluate_sampling_strategy,
    load_code_capacity_problem,
    matched_support_heatmap_rows,
    sample_random_error_cases,
    train_gaussian_memory_distribution,
    train_static_memory_weights,
    with_uniform_error_rate,
)
from relay_bp.analysis.code_capacity_gaussian_training import GaussianCEMTrainingConfig
from relay_bp.analysis.code_capacity_training import StaticMemoryTrainingConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORTS = default_export_code_paths(REPO_ROOT)


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in str(raw).split(",")]
    parsed = [float(item) for item in values if item]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return parsed


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(",")]
    parsed = [int(item) for item in values if item]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer value.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-name", choices=sorted(DEFAULT_EXPORTS), default="surface13")
    parser.add_argument("--code-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-file-priors", action="store_true")
    parser.add_argument("--p-values", type=_parse_float_list, default=_parse_float_list("0.02,0.05,0.07,0.09,0.11"))
    parser.add_argument("--center-values", type=_parse_float_list, default=_parse_float_list("0.0,0.1,0.2,0.3,0.4,0.5"))
    parser.add_argument("--width-values", type=_parse_float_list, default=_parse_float_list("0.0,0.2,0.4,0.6,0.8"))
    parser.add_argument("--random-error-samples", type=int, default=512)
    parser.add_argument("--heatmap-draws-per-point", type=int, default=8)
    parser.add_argument("--comparison-seeds", type=int, default=16)
    parser.add_argument("--comparison-k-values", type=_parse_int_list, default=_parse_int_list("4,8"))
    parser.add_argument("--random-application-scope", choices=["all_bits"], default="all_bits")
    parser.add_argument(
        "--half-application-scope",
        choices=["all_bits", "stabilizer_support"],
        default="stabilizer_support",
    )
    parser.add_argument("--half-case-limit", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--baseline-memory-value", type=float, default=0.15)
    parser.add_argument("--logical-weight-penalty", type=float, default=4.0)
    parser.add_argument("--convergence-penalty", type=float, default=1.0)
    parser.add_argument("--iteration-penalty", type=float, default=0.05)
    parser.add_argument("--perfect-threshold", type=float, default=0.999999)
    parser.add_argument("--top-heatmap-points", type=int, default=8)
    parser.add_argument("--gaussian-refinement-points", type=int, default=9)
    parser.add_argument("--gaussian-variance-values", type=_parse_float_list, default=None)
    parser.add_argument("--train-static", action="store_true")
    parser.add_argument("--train-gaussian", action="store_true")
    parser.add_argument("--train-half-limit", type=int, default=128)
    parser.add_argument("--train-random-validation-samples", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--train-warmup-steps", type=int, default=10)
    parser.add_argument("--train-mixed-steps", type=int, default=10)
    parser.add_argument("--train-finetune-steps", type=int, default=10)
    parser.add_argument("--train-validation-interval", type=int, default=2)
    parser.add_argument("--static-initialization-num-candidates", type=int, default=32)
    parser.add_argument("--static-initialization-topk", type=int, default=6)
    parser.add_argument("--gaussian-initialization-num-candidates", type=int, default=32)
    parser.add_argument("--gaussian-initialization-topk", type=int, default=8)
    parser.add_argument("--gaussian-candidate-count", type=int, default=64)
    parser.add_argument("--gaussian-elite-count", type=int, default=8)
    parser.add_argument("--gaussian-smoothing", type=float, default=0.2)
    parser.add_argument("--equivalence-bootstrap-samples", type=int, default=2000)
    parser.add_argument("--equivalence-logical-margin", type=float, default=-0.01)
    parser.add_argument("--equivalence-convergence-margin", type=float, default=-0.02)
    parser.add_argument("--equivalence-iteration-ratio-ceiling", type=float, default=1.10)
    parser.add_argument("--base-seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    code_path = Path(args.code_path) if args.code_path is not None else DEFAULT_EXPORTS[args.code_name]
    base_problem = load_code_capacity_problem(code_path)

    p_scan_payload = _run_p_scan(base_problem, args)
    recommended = p_scan_payload["recommended"]
    selected_p = float(recommended["p"])
    study_problem = base_problem if args.use_file_priors else with_uniform_error_rate(base_problem, selected_p)
    task_payloads = {
        "random_error": _run_task_heatmaps(
            task_name="random_error",
            problem=study_problem,
            args=args,
            output_dir=output_dir,
            p_value=selected_p,
        ),
        "half_stabilizer": _run_task_heatmaps(
            task_name="half_stabilizer",
            problem=study_problem,
            args=args,
            output_dir=output_dir,
            p_value=selected_p,
        ),
    }

    training_interval = {
        "low": float(task_payloads["random_error"]["best_support_overall"]["interval_low"]),
        "high": float(task_payloads["random_error"]["best_support_overall"]["interval_high"]),
    }
    training_payloads: dict[str, Any] = {}
    if args.train_static:
        training_payloads["static"] = _run_static_training(
            problem=study_problem,
            args=args,
            output_dir=output_dir,
            selected_p=selected_p,
            interval=training_interval,
        )
    if args.train_gaussian:
        training_payloads["gaussian"] = _run_gaussian_training(
            problem=study_problem,
            args=args,
            output_dir=output_dir,
            selected_p=selected_p,
            interval=training_interval,
        )

    comparisons = {
        "random_error": _run_method_comparison(
            task_name="random_error",
            problem=study_problem,
            args=args,
            task_payload=task_payloads["random_error"],
            training_payloads=training_payloads,
            p_value=selected_p,
            output_dir=output_dir,
        ),
        "half_stabilizer": _run_method_comparison(
            task_name="half_stabilizer",
            problem=study_problem,
            args=args,
            task_payload=task_payloads["half_stabilizer"],
            training_payloads=training_payloads,
            p_value=selected_p,
            output_dir=output_dir,
        ),
    }

    summary = {
        "code_path": str(code_path.resolve()),
        "code_name": None if args.code_path is not None else args.code_name,
        "n_bits": int(base_problem.n_bits),
        "n_logicals": int(base_problem.n_logicals),
        "selected_random_error_rate": selected_p,
        "training_interval": training_interval,
        "p_scan": p_scan_payload,
        "tasks": task_payloads,
        "training": training_payloads,
        "comparisons": comparisons,
    }
    _write_json(output_dir / "study_summary.json", summary)
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


def _run_p_scan(base_problem, args) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for p_idx, p_value in enumerate(args.p_values):
        problem = base_problem if args.use_file_priors else with_uniform_error_rate(base_problem, p_value)
        cases = sample_random_error_cases(
            problem,
            num_samples=args.random_error_samples,
            error_rate=p_value,
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
        artifact = evaluate_matched_support_heatmaps(
            problem=problem,
            cases=cases,
            max_iter=args.max_iter,
            alpha=args.alpha,
            centers=args.center_values,
            widths=args.width_values,
            base_seed=int(args.base_seed) + 300_000 + 10_000 * p_idx,
            random_draws_per_point=args.heatmap_draws_per_point,
            application_scope=args.random_application_scope,
            selection_mode="single_draw",
            logical_weight_penalty=args.logical_weight_penalty,
            convergence_penalty=args.convergence_penalty,
            iteration_penalty=args.iteration_penalty,
        )
        best_row = best_heatmap_rows(matched_support_heatmap_rows(artifact), topk=1)[0]
        row = {
            "p": float(p_value),
            "bp": bp_metrics,
            "best_sampler": best_row,
            "score": float(
                (float(best_row["logical_success_rate"]) - float(bp_metrics["logical_success_rate"]))
                + 0.35 * (float(best_row["convergence_rate"]) - float(bp_metrics["convergence_rate"]))
            ),
        }
        rows.append(row)

    recommended = _recommend_p(rows, perfect_threshold=float(args.perfect_threshold))
    payload = {
        "rows": rows,
        "recommended": recommended,
    }
    _write_json(Path(args.output_dir) / "p_scan.json", payload)
    _write_rows_csv(Path(args.output_dir) / "p_scan.csv", _flatten_p_scan_rows(rows))
    return payload


def _recommend_p(rows: list[dict[str, Any]], *, perfect_threshold: float) -> dict[str, Any]:
    eligible: list[tuple[float, dict[str, Any]]] = []
    fallback: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        bp = row["bp"]
        sampler = row["best_sampler"]
        score = float(row["score"])
        fallback.append((score, row))
        bp_nontrivial = (
            (0.0 < float(bp["logical_success_rate"]) < perfect_threshold)
            or (0.0 < float(bp["convergence_rate"]) < perfect_threshold)
        )
        sampler_nontrivial = (
            float(sampler["logical_success_rate"]) < perfect_threshold
            or float(sampler["convergence_rate"]) < perfect_threshold
        )
        if bp_nontrivial and sampler_nontrivial:
            eligible.append((score, row))
    pool = eligible if eligible else fallback
    return max(pool, key=lambda item: item[0])[1]


def _flatten_p_scan_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat_rows.append(
            {
                "p": float(row["p"]),
                "bp_logical_success_rate": float(row["bp"]["logical_success_rate"]),
                "bp_convergence_rate": float(row["bp"]["convergence_rate"]),
                "best_family": row["best_sampler"]["family"],
                "best_center": float(row["best_sampler"]["center"]),
                "best_width": float(row["best_sampler"]["width"]),
                "best_variance": float(row["best_sampler"].get("variance", np.nan)),
                "best_logical_success_rate": float(row["best_sampler"]["logical_success_rate"]),
                "best_convergence_rate": float(row["best_sampler"]["convergence_rate"]),
                "best_exact_recovery_rate": float(row["best_sampler"]["exact_recovery_rate"]),
                "best_mean_iterations": float(row["best_sampler"]["mean_iterations"]),
                "score": float(row["score"]),
            }
        )
    return flat_rows


def _run_task_heatmaps(
    *,
    task_name: str,
    problem,
    args,
    output_dir: Path,
    p_value: float,
) -> dict[str, Any]:
    cases = _build_cases(problem=problem, task_name=task_name, p_value=p_value, args=args, seed_offset=0)
    application_scope = args.random_application_scope if task_name == "random_error" else args.half_application_scope
    artifact = evaluate_matched_support_heatmaps(
        problem=problem,
        cases=cases,
        max_iter=args.max_iter,
        alpha=args.alpha,
        centers=args.center_values,
        widths=args.width_values,
        base_seed=int(args.base_seed) + (500_000 if task_name == "random_error" else 600_000),
        random_draws_per_point=args.heatmap_draws_per_point,
        application_scope=application_scope,
        selection_mode="single_draw",
        logical_weight_penalty=args.logical_weight_penalty,
        convergence_penalty=args.convergence_penalty,
        iteration_penalty=args.iteration_penalty,
    )
    rows = matched_support_heatmap_rows(artifact)
    for row in rows:
        row["variance"] = float(row["sigma"]) ** 2
    uniform_best = best_heatmap_rows([row for row in rows if row["family"] == "uniform_interval"], topk=1)[0]
    gaussian_support_best = best_heatmap_rows([row for row in rows if row["family"] == "truncated_gaussian"], topk=1)[0]
    best_support_overall = min([uniform_best, gaussian_support_best], key=_method_sort_key)
    _write_json(output_dir / f"{task_name}_matched_support.json", artifact)
    _write_rows_csv(output_dir / f"{task_name}_matched_support.csv", rows)
    _write_rows_csv(
        output_dir / f"{task_name}_matched_support_best_points.csv",
        best_heatmap_rows(rows, topk=args.top_heatmap_points),
    )
    _plot_matched_support_heatmap(
        artifact,
        title=f"{task_name.replace('_', ' ').title()} Matched-Support Heatmaps",
        output_path=output_dir / f"{task_name}_matched_support.png",
    )

    refinement_low = float(gaussian_support_best["interval_low"])
    refinement_high = float(gaussian_support_best["interval_high"])
    width = max(0.0, refinement_high - refinement_low)
    if args.gaussian_variance_values is not None:
        refinement_variances = np.asarray(args.gaussian_variance_values, dtype=np.float64)
    else:
        sigma_low = max(1e-3, width / 16.0)
        sigma_high = max(sigma_low, width / 2.0)
        refinement_variances = np.linspace(sigma_low**2, sigma_high**2, int(args.gaussian_refinement_points))
    refinement_means = np.linspace(refinement_low, refinement_high, int(args.gaussian_refinement_points))
    refinement_sigmas = np.sqrt(np.maximum(refinement_variances, 0.0))
    refinement_artifact = evaluate_gaussian_refinement_heatmap(
        problem=problem,
        cases=cases,
        max_iter=args.max_iter,
        alpha=args.alpha,
        means=refinement_means.tolist(),
        sigmas=refinement_sigmas.tolist(),
        low=refinement_low,
        high=refinement_high,
        base_seed=int(args.base_seed) + (700_000 if task_name == "random_error" else 800_000),
        random_draws_per_point=args.heatmap_draws_per_point,
        application_scope=application_scope,
        selection_mode="single_draw",
        logical_weight_penalty=args.logical_weight_penalty,
        convergence_penalty=args.convergence_penalty,
        iteration_penalty=args.iteration_penalty,
    )
    refinement_rows = []
    from relay_bp.analysis import gaussian_refinement_heatmap_rows

    refinement_rows = gaussian_refinement_heatmap_rows(refinement_artifact)
    for row in refinement_rows:
        row["variance"] = float(row["sigma"]) ** 2
    gaussian_best = best_heatmap_rows(refinement_rows, topk=1)[0]
    _write_json(output_dir / f"{task_name}_gaussian_refinement.json", refinement_artifact)
    _write_rows_csv(output_dir / f"{task_name}_gaussian_refinement.csv", refinement_rows)
    _write_rows_csv(
        output_dir / f"{task_name}_gaussian_refinement_best_points.csv",
        best_heatmap_rows(refinement_rows, topk=args.top_heatmap_points),
    )
    _plot_gaussian_refinement_heatmap(
        refinement_artifact,
        title=f"{task_name.replace('_', ' ').title()} Gaussian Refinement",
        output_path=output_dir / f"{task_name}_gaussian_refinement.png",
    )

    return {
        "application_scope": application_scope,
        "num_cases": int(len(cases)),
        "best_support_overall": best_support_overall,
        "best_uniform": uniform_best,
        "best_gaussian_support": gaussian_support_best,
        "best_gaussian": gaussian_best,
        "best_by_metric": {
            "uniform": _best_rows_by_metric([row for row in rows if row["family"] == "uniform_interval"]),
            "gaussian_support": _best_rows_by_metric([row for row in rows if row["family"] == "truncated_gaussian"]),
            "gaussian_refinement": _best_rows_by_metric(refinement_rows),
        },
    }


def _run_static_training(*, problem, args, output_dir: Path, selected_p: float, interval: dict[str, float]) -> dict[str, Any]:
    checkpoint_dir = output_dir / "static_training"
    config = StaticMemoryTrainingConfig(
        code_path=str(problem.path),
        checkpoint_dir=str(checkpoint_dir),
        max_iter=args.max_iter,
        alpha=args.alpha,
        train_random_error_rate=selected_p,
        validation_random_error_rate=selected_p,
        half_train_limit=args.train_half_limit,
        half_validation_limit=min(args.train_half_limit, max(1, args.train_half_limit // 2)),
        random_validation_samples=args.train_random_validation_samples,
        batch_size=args.train_batch_size,
        validation_interval=args.train_validation_interval,
        warmup_steps=args.train_warmup_steps,
        mixed_steps=args.train_mixed_steps,
        finetune_steps=args.train_finetune_steps,
        selection_half_weight=1.0,
        logical_weight_penalty=args.logical_weight_penalty,
        convergence_penalty=args.convergence_penalty,
        iteration_penalty=args.iteration_penalty,
        baseline_memory_value=args.baseline_memory_value,
        memory_min=float(interval["low"]),
        memory_max=float(interval["high"]),
        initialization_mode="interval_topk_average",
        initialization_num_candidates=args.static_initialization_num_candidates,
        initialization_topk=args.static_initialization_topk,
        base_seed=args.base_seed,
        resume=False,
    )
    return train_static_memory_weights(problem, config=config)


def _run_gaussian_training(*, problem, args, output_dir: Path, selected_p: float, interval: dict[str, float]) -> dict[str, Any]:
    checkpoint_dir = output_dir / "gaussian_training"
    config = GaussianCEMTrainingConfig(
        code_path=str(problem.path),
        checkpoint_dir=str(checkpoint_dir),
        max_iter=args.max_iter,
        alpha=args.alpha,
        train_random_error_rate=selected_p,
        validation_random_error_rate=selected_p,
        half_train_limit=args.train_half_limit,
        half_validation_limit=min(args.train_half_limit, max(1, args.train_half_limit // 2)),
        random_validation_samples=args.train_random_validation_samples,
        batch_size=args.train_batch_size,
        warmup_steps=args.train_warmup_steps,
        mixed_steps=args.train_mixed_steps,
        finetune_steps=args.train_finetune_steps,
        selection_half_weight=1.0,
        logical_weight_penalty=args.logical_weight_penalty,
        convergence_penalty=args.convergence_penalty,
        iteration_penalty=args.iteration_penalty,
        baseline_memory_value=args.baseline_memory_value,
        memory_min=float(interval["low"]),
        memory_max=float(interval["high"]),
        initialization_low=float(interval["low"]),
        initialization_high=float(interval["high"]),
        initialization_num_candidates=args.gaussian_initialization_num_candidates,
        initialization_topk=args.gaussian_initialization_topk,
        candidate_count=args.gaussian_candidate_count,
        elite_count=args.gaussian_elite_count,
        smoothing=args.gaussian_smoothing,
        validation_interval=args.train_validation_interval,
        base_seed=args.base_seed,
        resume=False,
    )
    return train_gaussian_memory_distribution(problem, config=config)


def _run_method_comparison(
    *,
    task_name: str,
    problem,
    args,
    task_payload: dict[str, Any],
    training_payloads: dict[str, Any],
    p_value: float,
    output_dir: Path,
) -> dict[str, Any]:
    methods = _build_methods(problem=problem, args=args, task_payload=task_payload, training_payloads=training_payloads)
    per_method_rows: dict[str, list[dict[str, Any]]] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for method in methods:
        rows = _evaluate_method_over_seeds(
            problem=problem,
            args=args,
            task_name=task_name,
            p_value=p_value,
            method=method,
        )
        per_method_rows[method["name"]] = rows
        aggregate_rows.append(
            {
                "method": method["name"],
                "kind": method["kind"],
                "logical_success_rate": float(np.mean([row["logical_success_rate"] for row in rows])),
                "exact_recovery_rate": float(np.mean([row["exact_recovery_rate"] for row in rows])),
                "convergence_rate": float(np.mean([row["convergence_rate"] for row in rows])),
                "mean_iterations": float(np.mean([row["mean_iterations"] for row in rows])),
                "loss_mean": float(np.mean([row["loss_mean"] for row in rows])),
                "seed_count": int(len(rows)),
            }
        )

    _write_rows_csv(output_dir / f"{task_name}_method_comparison.csv", aggregate_rows)
    _write_rows_csv(
        output_dir / f"{task_name}_method_comparison_per_seed.csv",
        [
            {"method": method_name, **row}
            for method_name, rows in per_method_rows.items()
            for row in rows
        ],
    )

    best_single_draw = min(
        (row for row in aggregate_rows if row["kind"] == "stochastic_single_draw"),
        key=_method_sort_key,
    )
    best_practical_kdraw = min(
        (row for row in aggregate_rows if row["kind"] == "stochastic_kdraw_practical"),
        key=_method_sort_key,
    )
    thresholds = EquivalenceThresholds(
        logical_margin=args.equivalence_logical_margin,
        convergence_margin=args.equivalence_convergence_margin,
        iteration_ratio_ceiling=args.equivalence_iteration_ratio_ceiling,
        bootstrap_samples=args.equivalence_bootstrap_samples,
        seed=args.base_seed,
    )
    equivalence: dict[str, Any] = {}
    baseline_rows = per_method_rows[best_single_draw["method"]]
    for row in aggregate_rows:
        if row["kind"] not in {"deterministic", "deterministic_trained"}:
            continue
        equivalence[row["method"]] = evaluate_equivalence(
            candidate_rows=per_method_rows[row["method"]],
            baseline_rows=baseline_rows,
            thresholds=thresholds,
        )

    pairwise = {}
    if "uniform_single_draw" in per_method_rows and "gaussian_single_draw" in per_method_rows:
        pairwise["gaussian_vs_uniform_single_draw"] = _bootstrap_method_difference(
            candidate_rows=per_method_rows["gaussian_single_draw"],
            baseline_rows=per_method_rows["uniform_single_draw"],
            bootstrap_samples=args.equivalence_bootstrap_samples,
            seed=args.base_seed + 19,
        )
    for k_value in args.comparison_k_values:
        uniform_name = f"uniform_k_draw_practical_{k_value}"
        gaussian_name = f"gaussian_k_draw_practical_{k_value}"
        if uniform_name in per_method_rows and gaussian_name in per_method_rows:
            pairwise[f"gaussian_vs_uniform_practical_k{k_value}"] = _bootstrap_method_difference(
                candidate_rows=per_method_rows[gaussian_name],
                baseline_rows=per_method_rows[uniform_name],
                bootstrap_samples=args.equivalence_bootstrap_samples,
                seed=args.base_seed + 101 + int(k_value),
            )

    payload = {
        "best_single_draw_baseline": best_single_draw,
        "best_practical_kdraw_baseline": best_practical_kdraw,
        "aggregate_rows": aggregate_rows,
        "equivalence_to_best_single_draw": equivalence,
        "pairwise_comparisons": pairwise,
    }
    _write_json(output_dir / f"{task_name}_method_comparison.json", payload)
    return payload


def _build_methods(*, problem, args, task_payload: dict[str, Any], training_payloads: dict[str, Any]) -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = [
        {"name": "bp", "kind": "deterministic", "type": "bp"},
        {
            "name": "same_memory",
            "kind": "deterministic",
            "type": "static",
            "weights": np.full(problem.n_bits, float(args.baseline_memory_value), dtype=np.float64),
        },
    ]
    uniform_best = task_payload["best_uniform"]
    gaussian_best = task_payload["best_gaussian"]
    methods.append(
        {
            "name": "uniform_single_draw",
            "kind": "stochastic_single_draw",
            "type": "sampler",
            "sampling_spec": MemorySamplingSpec(
                sampling_family="uniform_interval",
                application_scope=task_payload["application_scope"],
                selection_mode="single_draw",
                low=float(uniform_best["interval_low"]),
                high=float(uniform_best["interval_high"]),
            ),
        }
    )
    methods.append(
        {
            "name": "gaussian_single_draw",
            "kind": "stochastic_single_draw",
            "type": "sampler",
            "sampling_spec": MemorySamplingSpec(
                sampling_family="truncated_gaussian",
                application_scope=task_payload["application_scope"],
                selection_mode="single_draw",
                low=float(gaussian_best["low"]),
                high=float(gaussian_best["high"]),
                mean=float(gaussian_best["mean"]),
                sigma=float(gaussian_best["sigma"]),
            ),
        }
    )
    for k_value in args.comparison_k_values:
        for family_name, spec in (
            ("uniform", methods[-2]["sampling_spec"]),
            ("gaussian", methods[-1]["sampling_spec"]),
        ):
            for selection_mode, kind in (
                ("k_draw_oracle", "stochastic_kdraw_oracle"),
                ("k_draw_practical", "stochastic_kdraw_practical"),
            ):
                methods.append(
                    {
                        "name": f"{family_name}_{selection_mode}_{k_value}",
                        "kind": kind,
                        "type": "sampler",
                        "sampling_spec": MemorySamplingSpec(
                            sampling_family=spec.sampling_family,
                            application_scope=spec.application_scope,
                            selection_mode=selection_mode,
                            draw_count=int(k_value),
                            low=spec.low,
                            high=spec.high,
                            mean=spec.mean,
                            mean_vector=spec.mean_vector,
                            sigma=spec.sigma,
                        ),
                    }
                )
    if "static" in training_payloads:
        methods.append(
            {
                "name": "trained_static_vector",
                "kind": "deterministic_trained",
                "type": "static",
                "weights": np.load(Path(training_payloads["static"]["best_weights_path"])),
            }
        )
    if "gaussian" in training_payloads:
        gaussian_mean = np.load(Path(training_payloads["gaussian"]["best_distribution_mean_path"]))
        gaussian_dist = _load_json(Path(training_payloads["gaussian"]["best_distribution_path"]))
        methods.append(
            {
                "name": "learned_gaussian_representative",
                "kind": "deterministic_trained",
                "type": "static",
                "weights": gaussian_mean,
            }
        )
        methods.append(
            {
                "name": "learned_gaussian_single_draw",
                "kind": "stochastic_single_draw",
                "type": "sampler",
                "sampling_spec": MemorySamplingSpec(
                    sampling_family="truncated_gaussian",
                    application_scope=task_payload["application_scope"],
                    selection_mode="single_draw",
                    low=float(gaussian_dist["low"]),
                    high=float(gaussian_dist["high"]),
                    mean_vector=gaussian_mean,
                    sigma=float(gaussian_dist["sigma"]),
                ),
            }
        )
    return methods


def _evaluate_method_over_seeds(*, problem, args, task_name: str, p_value: float, method: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shared_half_cases = None
    if task_name == "half_stabilizer":
        shared_half_cases = _build_cases(problem=problem, task_name=task_name, p_value=p_value, args=args, seed_offset=0)
    for seed_idx in range(int(args.comparison_seeds)):
        cases = shared_half_cases
        if cases is None:
            cases = _build_cases(
                problem=problem,
                task_name=task_name,
                p_value=p_value,
                args=args,
                seed_offset=10_000 + seed_idx,
            )
        if method["type"] == "bp":
            metrics = evaluate_memory_weight_vector(
                problem=problem,
                cases=cases,
                memory_strengths=None,
                max_iter=args.max_iter,
                alpha=args.alpha,
                logical_weight_penalty=args.logical_weight_penalty,
                convergence_penalty=args.convergence_penalty,
                iteration_penalty=args.iteration_penalty,
            )
        elif method["type"] == "static":
            metrics = evaluate_memory_weight_vector(
                problem=problem,
                cases=cases,
                memory_strengths=np.asarray(method["weights"], dtype=np.float64),
                max_iter=args.max_iter,
                alpha=args.alpha,
                logical_weight_penalty=args.logical_weight_penalty,
                convergence_penalty=args.convergence_penalty,
                iteration_penalty=args.iteration_penalty,
            )
        else:
            metrics = evaluate_sampling_strategy(
                problem=problem,
                cases=cases,
                sampling_spec=method["sampling_spec"],
                max_iter=args.max_iter,
                alpha=args.alpha,
                logical_weight_penalty=args.logical_weight_penalty,
                convergence_penalty=args.convergence_penalty,
                iteration_penalty=args.iteration_penalty,
                base_seed=int(args.base_seed) + 20_000 * seed_idx,
            )
        rows.append({"seed": int(seed_idx), **metrics})
    return rows


def _build_cases(*, problem, task_name: str, p_value: float, args, seed_offset: int) -> list[dict[str, Any]]:
    if task_name == "random_error":
        return sample_random_error_cases(
            problem,
            num_samples=args.random_error_samples,
            error_rate=p_value,
            base_seed=int(args.base_seed) + int(seed_offset),
        )
    return enumerate_half_stabilizer_cases(
        problem,
        max_cases=args.half_case_limit,
        shuffle_seed=int(args.base_seed) + int(seed_offset),
    )


def _plot_matched_support_heatmap(artifact: dict[str, Any], *, title: str, output_path: Path) -> None:
    centers = np.asarray(artifact["centers"], dtype=np.float64)
    widths = np.asarray(artifact["widths"], dtype=np.float64)
    families = [("uniform_interval", "Uniform"), ("truncated_gaussian", "Gaussian")]
    metrics = [
        ("logical_success_rate", "Logical Success"),
        ("convergence_rate", "Convergence"),
        ("mean_iterations", "Mean Iterations"),
    ]
    fig, axes = plt.subplots(len(families), len(metrics), figsize=(12, 6), constrained_layout=True)
    for row_idx, (family_name, family_label) in enumerate(families):
        for col_idx, (metric_name, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                artifact["families"][family_name][metric_name],
                origin="lower",
                aspect="auto",
                extent=_heatmap_extent(centers, widths),
                cmap="viridis",
            )
            ax.set_title(f"{family_label}: {metric_label}")
            ax.set_xlabel("Center")
            ax.set_ylabel("Width")
            fig.colorbar(image, ax=ax, shrink=0.85)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_gaussian_refinement_heatmap(artifact: dict[str, Any], *, title: str, output_path: Path) -> None:
    means = np.asarray(artifact["means"], dtype=np.float64)
    variances = np.asarray(artifact["sigmas"], dtype=np.float64) ** 2
    metrics = [
        ("logical_success_rate", "Logical Success"),
        ("convergence_rate", "Convergence"),
        ("mean_iterations", "Mean Iterations"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.6), constrained_layout=True)
    for ax, (metric_name, metric_label) in zip(axes, metrics):
        image = ax.imshow(
            artifact[metric_name],
            origin="lower",
            aspect="auto",
            extent=_heatmap_extent(means, variances),
            cmap="viridis",
        )
        ax.set_title(metric_label)
        ax.set_xlabel("Mean")
        ax.set_ylabel("Variance")
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _heatmap_extent(x_values: np.ndarray, y_values: np.ndarray) -> list[float]:
    return [
        *_axis_bounds(x_values),
        *_axis_bounds(y_values),
    ]


def _axis_bounds(values: np.ndarray) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return (value - 0.5, value + 0.5)
    step = float(np.min(np.diff(np.sort(values))))
    return (float(np.min(values)) - step / 2.0, float(np.max(values)) + step / 2.0)


def _method_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["loss_mean"]),
        -float(row["logical_success_rate"]),
        -float(row["convergence_rate"]),
        float(row["mean_iterations"]),
    )


def _best_rows_by_metric(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    return {
        "loss": min(rows, key=lambda row: _method_sort_key(row)),
        "logical_success": max(
            rows,
            key=lambda row: (
                float(row["logical_success_rate"]),
                float(row["convergence_rate"]),
                float(row["exact_recovery_rate"]),
                -float(row["mean_iterations"]),
            ),
        ),
        "convergence": max(
            rows,
            key=lambda row: (
                float(row["convergence_rate"]),
                float(row["logical_success_rate"]),
                float(row["exact_recovery_rate"]),
                -float(row["mean_iterations"]),
            ),
        ),
        "exact_recovery": max(
            rows,
            key=lambda row: (
                float(row["exact_recovery_rate"]),
                float(row["logical_success_rate"]),
                float(row["convergence_rate"]),
                -float(row["mean_iterations"]),
            ),
        ),
        "speed": min(
            rows,
            key=lambda row: (
                float(row["mean_iterations"]),
                -float(row["logical_success_rate"]),
                -float(row["convergence_rate"]),
                -float(row["exact_recovery_rate"]),
            ),
        ),
    }


def _bootstrap_method_difference(
    *,
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(candidate_rows) != len(baseline_rows):
        raise ValueError("candidate_rows and baseline_rows must have the same length.")
    logical_diff = [
        float(candidate["logical_success_rate"]) - float(baseline["logical_success_rate"])
        for candidate, baseline in zip(candidate_rows, baseline_rows)
    ]
    convergence_diff = [
        float(candidate["convergence_rate"]) - float(baseline["convergence_rate"])
        for candidate, baseline in zip(candidate_rows, baseline_rows)
    ]
    exact_diff = [
        float(candidate["exact_recovery_rate"]) - float(baseline["exact_recovery_rate"])
        for candidate, baseline in zip(candidate_rows, baseline_rows)
    ]
    iteration_ratio = [
        float(candidate["mean_iterations"]) / max(float(baseline["mean_iterations"]), 1e-12)
        for candidate, baseline in zip(candidate_rows, baseline_rows)
    ]
    return {
        "logical_success_difference": bootstrap_confidence_interval(
            logical_diff,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "convergence_difference": bootstrap_confidence_interval(
            convergence_diff,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "exact_recovery_difference": bootstrap_confidence_interval(
            exact_diff,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 2,
        ),
        "iteration_ratio": bootstrap_confidence_interval(
            iteration_ratio,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 3,
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
