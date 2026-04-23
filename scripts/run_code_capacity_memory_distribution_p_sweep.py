#!/usr/bin/env python3
"""Sweep Gaussian-vs-uniform memory distributions across multiple p values for one code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from relay_bp.analysis import (
    MemorySamplingSpec,
    bootstrap_confidence_interval,
    enumerate_half_stabilizer_cases,
    evaluate_gaussian_refinement_heatmap,
    evaluate_matched_support_heatmaps,
    evaluate_memory_weight_vector,
    evaluate_sampling_strategy,
    gaussian_refinement_heatmap_rows,
    load_code_capacity_problem,
    matched_support_heatmap_rows,
    sample_random_error_cases,
    with_uniform_error_rate,
)


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in str(raw).split(",")]
    parsed = [float(item) for item in values if item]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-label", required=True)
    parser.add_argument("--code-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--p-values", type=_parse_float_list, required=True)
    parser.add_argument("--center-values", type=_parse_float_list, required=True)
    parser.add_argument("--width-values", type=_parse_float_list, required=True)
    parser.add_argument("--scan-random-samples", type=int, default=512)
    parser.add_argument("--holdout-random-samples", type=int, default=1024)
    parser.add_argument("--heatmap-draws", type=int, default=4)
    parser.add_argument("--comparison-seeds", type=int, default=8)
    parser.add_argument("--k-draw", type=int, default=4)
    parser.add_argument("--gaussian-refinement-points", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--logical-weight-penalty", type=float, default=4.0)
    parser.add_argument("--convergence-penalty", type=float, default=1.0)
    parser.add_argument("--iteration-penalty", type=float, default=0.05)
    parser.add_argument("--baseline-memory-value", type=float, default=0.15)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_problem = load_code_capacity_problem(args.code_path)
    half_cases = enumerate_half_stabilizer_cases(base_problem)
    summary = {
        "code_label": args.code_label,
        "code_path": str(Path(args.code_path).resolve()),
        "n_bits": int(base_problem.n_bits),
        "n_logicals": int(base_problem.n_logicals),
        "half_case_count": int(len(half_cases)),
        "scan_random_samples": int(args.scan_random_samples),
        "holdout_random_samples": int(args.holdout_random_samples),
        "heatmap_draws": int(args.heatmap_draws),
        "comparison_seeds": int(args.comparison_seeds),
        "k_draw": int(args.k_draw),
        "p_summaries": [],
    }

    for p_idx, p_value in enumerate(args.p_values):
        print(f"[{args.code_label}] p={p_value:.4f}", flush=True)
        problem = with_uniform_error_rate(base_problem, p_value)
        random_scan_cases = sample_random_error_cases(
            problem,
            num_samples=args.scan_random_samples,
            error_rate=p_value,
            base_seed=int(args.base_seed) + 100_000 * p_idx,
        )
        task_results = {
            "random_error": _scan_task(
                problem=problem,
                cases=random_scan_cases,
                task_name="random_error",
                application_scope="all_bits",
                args=args,
                seed_offset=10_000 * (p_idx + 1),
            ),
            "half_stabilizer": _scan_task(
                problem=problem,
                cases=half_cases,
                task_name="half_stabilizer",
                application_scope="stabilizer_support",
                args=args,
                seed_offset=20_000 * (p_idx + 1),
            ),
        }

        random_holdout_cases_by_seed = [
            sample_random_error_cases(
                problem,
                num_samples=args.holdout_random_samples,
                error_rate=p_value,
                base_seed=int(args.base_seed) + 1_000_000 + 10_000 * p_idx + seed_idx,
            )
            for seed_idx in range(int(args.comparison_seeds))
        ]
        half_holdout_cases_by_seed = [half_cases for _ in range(int(args.comparison_seeds))]
        holdout = {
            "random_error": _evaluate_holdout(
                problem=problem,
                cases_by_seed=random_holdout_cases_by_seed,
                task_scan=task_results["random_error"],
                args=args,
                seed_offset=30_000 * (p_idx + 1),
            ),
            "half_stabilizer": _evaluate_holdout(
                problem=problem,
                cases_by_seed=half_holdout_cases_by_seed,
                task_scan=task_results["half_stabilizer"],
                args=args,
                seed_offset=40_000 * (p_idx + 1),
            ),
        }
        summary["p_summaries"].append(
            {
                "p": float(p_value),
                "task_scans": task_results,
                "holdout": holdout,
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    _plot_metric_curves(summary, output_dir=output_dir)
    _plot_difference_curves(summary, output_dir=output_dir)
    _plot_parameter_curves(summary, output_dir=output_dir)
    _write_plot_readme(summary, output_dir=output_dir)
    print(summary_path)


def _scan_task(
    *,
    problem,
    cases,
    task_name: str,
    application_scope: str,
    args,
    seed_offset: int,
) -> dict[str, Any]:
    support_artifact = evaluate_matched_support_heatmaps(
        problem=problem,
        cases=cases,
        max_iter=args.max_iter,
        alpha=args.alpha,
        centers=args.center_values,
        widths=args.width_values,
        base_seed=int(args.base_seed) + int(seed_offset),
        random_draws_per_point=args.heatmap_draws,
        application_scope=application_scope,
        selection_mode="single_draw",
        logical_weight_penalty=args.logical_weight_penalty,
        convergence_penalty=args.convergence_penalty,
        iteration_penalty=args.iteration_penalty,
    )
    support_rows = matched_support_heatmap_rows(support_artifact)
    for row in support_rows:
        row["variance"] = float(row["sigma"]) ** 2
    support_by_family = {
        "uniform": _best_rows_by_metric([row for row in support_rows if row["family"] == "uniform_interval"]),
        "gaussian_support": _best_rows_by_metric(
            [row for row in support_rows if row["family"] == "truncated_gaussian"]
        ),
    }

    gaussian_support_loss = support_by_family["gaussian_support"]["loss"]
    support_width = max(
        0.0,
        float(gaussian_support_loss["interval_high"]) - float(gaussian_support_loss["interval_low"]),
    )
    variance_low = max((support_width / 16.0) ** 2, 1e-5)
    variance_high = max((support_width / 2.0) ** 2, variance_low)
    refinement_variances = np.linspace(variance_low, variance_high, int(args.gaussian_refinement_points))
    refinement_means = np.linspace(
        float(gaussian_support_loss["interval_low"]),
        float(gaussian_support_loss["interval_high"]),
        int(args.gaussian_refinement_points),
    )
    refinement_artifact = evaluate_gaussian_refinement_heatmap(
        problem=problem,
        cases=cases,
        max_iter=args.max_iter,
        alpha=args.alpha,
        means=refinement_means.tolist(),
        sigmas=np.sqrt(refinement_variances).tolist(),
        low=float(gaussian_support_loss["interval_low"]),
        high=float(gaussian_support_loss["interval_high"]),
        base_seed=int(args.base_seed) + int(seed_offset) + 5_000,
        random_draws_per_point=args.heatmap_draws,
        application_scope=application_scope,
        selection_mode="single_draw",
        logical_weight_penalty=args.logical_weight_penalty,
        convergence_penalty=args.convergence_penalty,
        iteration_penalty=args.iteration_penalty,
    )
    refinement_rows = gaussian_refinement_heatmap_rows(refinement_artifact)
    for row in refinement_rows:
        row["variance"] = float(row["sigma"]) ** 2

    return {
        "task_name": task_name,
        "application_scope": application_scope,
        "num_cases": int(len(cases)),
        "support_best_by_metric": support_by_family,
        "gaussian_refined_best_by_metric": _best_rows_by_metric(refinement_rows),
    }


def _evaluate_holdout(
    *,
    problem,
    cases_by_seed,
    task_scan: dict[str, Any],
    args,
    seed_offset: int,
) -> dict[str, Any]:
    bp_rows = _evaluate_rows(
        problem=problem,
        cases_by_seed=cases_by_seed,
        method={"type": "bp"},
        args=args,
        sampler_seed_base=int(args.base_seed) + int(seed_offset),
    )
    same_rows = _evaluate_rows(
        problem=problem,
        cases_by_seed=cases_by_seed,
        method={
            "type": "static",
            "weights": np.full(problem.n_bits, float(args.baseline_memory_value), dtype=np.float64),
        },
        args=args,
        sampler_seed_base=int(args.base_seed) + int(seed_offset),
    )

    uniform_loss_row = task_scan["support_best_by_metric"]["uniform"]["loss"]
    gaussian_loss_row = task_scan["gaussian_refined_best_by_metric"]["loss"]
    application_scope = task_scan["application_scope"]

    uniform_single_rows = _evaluate_rows(
        problem=problem,
        cases_by_seed=cases_by_seed,
        method={
            "type": "sampler",
            "sampling_spec": _row_to_uniform_spec(uniform_loss_row, application_scope),
        },
        args=args,
        sampler_seed_base=int(args.base_seed) + int(seed_offset) + 10_000,
    )
    gaussian_single_rows = _evaluate_rows(
        problem=problem,
        cases_by_seed=cases_by_seed,
        method={
            "type": "sampler",
            "sampling_spec": _row_to_gaussian_spec(gaussian_loss_row, application_scope),
        },
        args=args,
        sampler_seed_base=int(args.base_seed) + int(seed_offset) + 20_000,
    )
    uniform_k_rows = _evaluate_rows(
        problem=problem,
        cases_by_seed=cases_by_seed,
        method={
            "type": "sampler",
            "sampling_spec": _row_to_uniform_spec(
                uniform_loss_row,
                application_scope,
                selection_mode="k_draw_practical",
                draw_count=int(args.k_draw),
            ),
        },
        args=args,
        sampler_seed_base=int(args.base_seed) + int(seed_offset) + 30_000,
    )
    gaussian_k_rows = _evaluate_rows(
        problem=problem,
        cases_by_seed=cases_by_seed,
        method={
            "type": "sampler",
            "sampling_spec": _row_to_gaussian_spec(
                gaussian_loss_row,
                application_scope,
                selection_mode="k_draw_practical",
                draw_count=int(args.k_draw),
            ),
        },
        args=args,
        sampler_seed_base=int(args.base_seed) + int(seed_offset) + 40_000,
    )

    return {
        "bp": _aggregate_rows(bp_rows),
        "same_memory": _aggregate_rows(same_rows),
        "uniform_single_draw": _aggregate_rows(uniform_single_rows),
        "gaussian_single_draw": _aggregate_rows(gaussian_single_rows),
        "uniform_practical_k": _aggregate_rows(uniform_k_rows),
        "gaussian_practical_k": _aggregate_rows(gaussian_k_rows),
        "gaussian_minus_uniform_single_draw": _bootstrap_method_difference(
            gaussian_single_rows,
            uniform_single_rows,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.base_seed) + int(seed_offset) + 50_000,
        ),
        "gaussian_minus_uniform_practical_k": _bootstrap_method_difference(
            gaussian_k_rows,
            uniform_k_rows,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.base_seed) + int(seed_offset) + 60_000,
        ),
        "selected_uniform_loss_row": uniform_loss_row,
        "selected_gaussian_loss_row": gaussian_loss_row,
    }


def _evaluate_rows(*, problem, cases_by_seed, method: dict[str, Any], args, sampler_seed_base: int) -> list[dict[str, Any]]:
    rows = []
    for seed_idx, cases in enumerate(cases_by_seed):
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
                base_seed=int(sampler_seed_base) + 10_007 * seed_idx,
            )
        rows.append({"seed": int(seed_idx), **metrics})
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seed_count": int(len(rows)),
        "logical_success_rate": float(np.mean([row["logical_success_rate"] for row in rows])),
        "convergence_rate": float(np.mean([row["convergence_rate"] for row in rows])),
        "exact_recovery_rate": float(np.mean([row["exact_recovery_rate"] for row in rows])),
        "mean_iterations": float(np.mean([row["mean_iterations"] for row in rows])),
        "loss_mean": float(np.mean([row["loss_mean"] for row in rows])),
    }


def _best_rows_by_metric(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "loss": min(rows, key=lambda row: _scan_sort_key(row)),
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


def _scan_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["loss_mean"]),
        -float(row["logical_success_rate"]),
        -float(row["convergence_rate"]),
        float(row["mean_iterations"]),
    )


def _row_to_uniform_spec(
    row: dict[str, Any],
    application_scope: str,
    *,
    selection_mode: str = "single_draw",
    draw_count: int = 1,
) -> MemorySamplingSpec:
    return MemorySamplingSpec(
        sampling_family="uniform_interval",
        application_scope=application_scope,
        selection_mode=selection_mode,
        draw_count=int(draw_count),
        low=float(row["interval_low"]),
        high=float(row["interval_high"]),
    )


def _row_to_gaussian_spec(
    row: dict[str, Any],
    application_scope: str,
    *,
    selection_mode: str = "single_draw",
    draw_count: int = 1,
) -> MemorySamplingSpec:
    if "interval_low" in row:
        low = float(row["interval_low"])
        high = float(row["interval_high"])
        mean = float(row["center"])
        sigma = float(row["sigma"])
    else:
        low = float(row["low"])
        high = float(row["high"])
        mean = float(row["mean"])
        sigma = float(row["sigma"])
    return MemorySamplingSpec(
        sampling_family="truncated_gaussian",
        application_scope=application_scope,
        selection_mode=selection_mode,
        draw_count=int(draw_count),
        low=low,
        high=high,
        mean=mean,
        sigma=sigma,
    )


def _bootstrap_method_difference(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
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


def _plot_metric_curves(summary: dict[str, Any], *, output_dir: Path) -> None:
    p_values = np.asarray([row["p"] for row in summary["p_summaries"]], dtype=np.float64)
    for task_name in ("random_error", "half_stabilizer"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        metrics = [
            ("logical_success_rate", "Logical Success"),
            ("convergence_rate", "Convergence"),
            ("exact_recovery_rate", "Exact Recovery"),
            ("mean_iterations", "Mean Iterations"),
        ]
        series = {
            "BP": [row["holdout"][task_name]["bp"] for row in summary["p_summaries"]],
            "Uniform Single": [row["holdout"][task_name]["uniform_single_draw"] for row in summary["p_summaries"]],
            "Gaussian Single": [row["holdout"][task_name]["gaussian_single_draw"] for row in summary["p_summaries"]],
            f"Uniform k={summary['k_draw']}": [
                row["holdout"][task_name]["uniform_practical_k"] for row in summary["p_summaries"]
            ],
            f"Gaussian k={summary['k_draw']}": [
                row["holdout"][task_name]["gaussian_practical_k"] for row in summary["p_summaries"]
            ],
        }
        for ax, (metric_name, title) in zip(axes.ravel(), metrics):
            for label, rows in series.items():
                ax.plot(p_values, [float(row[metric_name]) for row in rows], marker="o", label=label)
            ax.set_title(title)
            ax.set_xlabel("p")
            ax.set_ylabel("rate" if metric_name != "mean_iterations" else "iterations")
            ax.grid(True, alpha=0.3)
        axes[1, 1].legend(loc="best", fontsize=8)
        fig.suptitle(
            f"{summary['code_label']} {task_name.replace('_', ' ').title()} Metrics "
            f"(higher is better except mean iterations)"
        )
        fig.savefig(output_dir / f"{task_name}_metric_curves.png", dpi=150)
        plt.close(fig)


def _plot_difference_curves(summary: dict[str, Any], *, output_dir: Path) -> None:
    p_values = np.asarray([row["p"] for row in summary["p_summaries"]], dtype=np.float64)
    for task_name in ("random_error", "half_stabilizer"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        metrics = [
            ("logical_success_difference", "Gaussian - Uniform Logical (95% CI)"),
            ("convergence_difference", "Gaussian - Uniform Convergence (95% CI)"),
            ("exact_recovery_difference", "Gaussian - Uniform Exact (95% CI)"),
            ("iteration_ratio", "Gaussian / Uniform Iterations (95% CI)"),
        ]
        for ax, (metric_name, title) in zip(axes.ravel(), metrics):
            single_rows = [
                row["holdout"][task_name]["gaussian_minus_uniform_single_draw"][metric_name]
                for row in summary["p_summaries"]
            ]
            k_rows = [
                row["holdout"][task_name]["gaussian_minus_uniform_practical_k"][metric_name]
                for row in summary["p_summaries"]
            ]
            ax.plot(p_values, [float(row["mean"]) for row in single_rows], marker="o", label="single draw mean")
            ax.fill_between(
                p_values,
                [float(row["lower"]) for row in single_rows],
                [float(row["upper"]) for row in single_rows],
                alpha=0.2,
            )
            ax.plot(
                p_values,
                [float(row["mean"]) for row in k_rows],
                marker="s",
                label=f"k={summary['k_draw']} mean",
            )
            ax.fill_between(
                p_values,
                [float(row["lower"]) for row in k_rows],
                [float(row["upper"]) for row in k_rows],
                alpha=0.2,
            )
            if metric_name != "iteration_ratio":
                ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
            else:
                ax.axhline(1.0, color="black", linewidth=1, alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel("p")
            ax.set_ylabel("difference" if metric_name != "iteration_ratio" else "ratio")
            ax.grid(True, alpha=0.3)
        axes[1, 1].legend(loc="best", fontsize=8)
        fig.suptitle(f"{summary['code_label']} {task_name.replace('_', ' ').title()} Gaussian vs Uniform")
        fig.savefig(output_dir / f"{task_name}_difference_curves.png", dpi=150)
        plt.close(fig)


def _plot_parameter_curves(summary: dict[str, Any], *, output_dir: Path) -> None:
    p_values = np.asarray([row["p"] for row in summary["p_summaries"]], dtype=np.float64)
    for task_name in ("random_error", "half_stabilizer"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        uniform_rows = [row["holdout"][task_name]["selected_uniform_loss_row"] for row in summary["p_summaries"]]
        gaussian_rows = [row["holdout"][task_name]["selected_gaussian_loss_row"] for row in summary["p_summaries"]]

        axes[0, 0].plot(p_values, [float(row["interval_low"]) for row in uniform_rows], marker="o", label="uniform low")
        axes[0, 0].plot(p_values, [float(row["interval_high"]) for row in uniform_rows], marker="o", label="uniform high")
        axes[0, 0].set_title("Uniform Interval")
        axes[0, 0].set_xlabel("p")
        axes[0, 0].set_ylabel("memory value")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(loc="best", fontsize=8)

        axes[0, 1].plot(p_values, [float(row["low"]) for row in gaussian_rows], marker="s", label="gaussian low")
        axes[0, 1].plot(p_values, [float(row["high"]) for row in gaussian_rows], marker="s", label="gaussian high")
        axes[0, 1].plot(p_values, [float(row["mean"]) for row in gaussian_rows], marker="^", label="gaussian mean")
        axes[0, 1].set_title("Gaussian Support And Mean")
        axes[0, 1].set_xlabel("p")
        axes[0, 1].set_ylabel("memory value")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(loc="best", fontsize=8)

        axes[1, 0].plot(p_values, [float(row["variance"]) for row in uniform_rows], marker="o")
        axes[1, 0].set_title("Uniform Support-Matched Variance Reference")
        axes[1, 0].set_xlabel("p")
        axes[1, 0].set_ylabel("variance")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(p_values, [float(row["variance"]) for row in gaussian_rows], marker="s")
        axes[1, 1].set_title("Gaussian Variance")
        axes[1, 1].set_xlabel("p")
        axes[1, 1].set_ylabel("variance")
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(f"{summary['code_label']} {task_name.replace('_', ' ').title()} Best-Loss Parameters")
        fig.savefig(output_dir / f"{task_name}_parameter_curves.png", dpi=150)
        plt.close(fig)


def _write_plot_readme(summary: dict[str, Any], *, output_dir: Path) -> None:
    task_labels = {
        "random_error": "Random-error task",
        "half_stabilizer": "Full half-stabilizer catalogue",
    }
    lines = [
        f"# {summary['code_label']} Plot Guide",
        "",
        "## What Is In This Folder",
        "",
        "- `summary.json`: machine-readable sweep results for all scanned p values.",
        "- `*_metric_curves.png`: absolute performance curves for BP, uniform sampling, and Gaussian sampling.",
        "- `*_difference_curves.png`: Gaussian-minus-uniform comparison with bootstrap confidence bands.",
        "- `*_parameter_curves.png`: the best-loss support and variance chosen by the scan at each p.",
        "",
        "## Glossary",
        "",
        "- `random_error`: sampled i.i.d. code-capacity errors at the plotted physical error rate `p`.",
        "- `half_stabilizer`: the full enumerated half-stabilizer fault catalogue scored with decoder priors set by `p`.",
        "- `BP`: plain belief propagation with no memory weights.",
        f"- `Uniform Single`: one draw from the best-loss uniform interval found by the scan.",
        f"- `Gaussian Single`: one draw from the best-loss truncated Gaussian found by the scan.",
        f"- `Uniform k={summary['k_draw']}` / `Gaussian k={summary['k_draw']}`: practical repeated draws using the deterministic selector.",
        "- `logical_success_rate`: fraction of cases with no logical error after decoding. Higher is better.",
        "- `convergence_rate`: fraction of cases where the decoder declared convergence. Higher is better.",
        "- `exact_recovery_rate`: fraction of cases where the decoded error exactly matches the true error. Higher is better.",
        "- `mean_iterations`: mean decoder iterations. Lower is better.",
        "- `loss_mean`: the scan/training selection loss combining logical, convergence, and iteration penalties.",
        "- `support`: the allowed interval `[low, high]` for sampled memory values.",
        "- `variance`: Gaussian variance `sigma^2` inside the selected support.",
        "",
        "## How To Read The Plots",
        "",
        "### Metric Curves",
        "",
        "- Three things matter here: who has the highest logical, convergence, and exact curves, and who has the lowest mean-iterations curve.",
        "- The metric panels are absolute performance. If Gaussian is above uniform on a rate metric, Gaussian is better there.",
        "- In the mean-iterations panel, lower is better.",
        "",
        "### Difference Curves",
        "",
        "- The first three panels are `Gaussian - Uniform`. Positive values favor Gaussian; negative values favor uniform.",
        "- The shaded regions are bootstrap 95% confidence intervals across holdout seeds.",
        "- If the shaded band crosses `0`, treat that p value as a tie on that metric.",
        "- The iteration panel is `Gaussian / Uniform` mean iterations. Values below `1` favor Gaussian, above `1` favor uniform.",
        "- If the iteration-ratio band crosses `1`, treat that p value as a speed tie.",
        "",
        "### Parameter Curves",
        "",
        "- These are the best-loss parameters selected by the scan, not hand-picked values.",
        "- `Uniform Interval` shows the winning low/high bounds for the uniform sampler.",
        "- `Gaussian Support And Mean` shows the winning Gaussian support `[low, high]` and internal mean.",
        "- `Gaussian Variance` is the winning Gaussian variance at each p.",
        "- `Uniform Support-Matched Variance Reference` is only a reference derived from the matched-support sweep; it is not the true variance of the uniform distribution.",
        "",
        "## Practical Interpretation Rules",
        "",
        "- If a Gaussian difference band stays entirely above `0` on logical or convergence, Gaussian has a decisive advantage at that p.",
        "- If a Gaussian difference band stays entirely below `0`, uniform is decisively better at that p.",
        "- If the band straddles `0`, the run does not separate the two models confidently on that metric.",
        "- Prefer the random-error panels for decoder usefulness under the chosen physical error rate.",
        "- Use the half-stabilizer panels to see whether the learned/sampled memory family is helping on the structured difficult catalogue.",
        "",
        "## Caveats",
        "",
        "- The half-stabilizer catalogue is deterministic; repeated seeds there reflect sampler variation, not new fault instances.",
        "- Very large iteration ratios can happen when the baseline method already has near-zero mean iterations. In that regime, the absolute mean-iterations panel is easier to trust than the ratio panel alone.",
        "- The selected parameters are tied to the current scan grid. A finer grid can move the chosen interval or variance even when performance barely changes.",
        "",
        "## Sweep Settings",
        "",
        f"- p grid: `{', '.join(f'{row['p']:.4f}' for row in summary['p_summaries'])}`",
        f"- random scan cases per p: `{summary['scan_random_samples']}`",
        f"- random holdout cases per seed: `{summary['holdout_random_samples']}`",
        f"- holdout seeds: `{summary['comparison_seeds']}`",
        f"- practical repeated draws: `k={summary['k_draw']}`",
        f"- half-stabilizer catalogue size: `{summary['half_case_count']}`",
        "",
        "## Task Files",
        "",
    ]
    for task_name, task_label in task_labels.items():
        lines.extend(
            [
                f"### {task_label}",
                "",
                f"- metrics: `{task_name}_metric_curves.png`",
                f"- differences: `{task_name}_difference_curves.png`",
                f"- parameters: `{task_name}_parameter_curves.png`",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


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
