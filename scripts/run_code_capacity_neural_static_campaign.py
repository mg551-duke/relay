#!/usr/bin/env python3
"""Run and aggregate neural static-distillation campaigns for raw code-capacity codes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_code_capacity_static_assignment_campaign import (
    FROZEN_P_GRIDS,
    infer_bank_size,
    resolve_code_path,
    summarize_bank_members,
    verdict_from_comparisons,
    write_rows_csv,
)
from scripts.train_code_capacity_neural_static_bank import add_run_arguments, json_ready, parse_int_list, run_training_unit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one neural-distillation training unit.")
    add_run_arguments(run)

    manifest = subparsers.add_parser("write-manifest", help="Write a CSV manifest for the frozen raw-code grid.")
    manifest.add_argument("--output-path", type=Path, required=True)
    manifest.add_argument("--families", default="memory,damping,joint")
    manifest.add_argument("--bank-sizes", type=parse_int_list, default=parse_int_list("1,2,4,8"))
    manifest.add_argument("--restarts", type=int, default=4)
    manifest.add_argument("--code-labels", default="surface5_raw,surface13_raw,gross_raw,two_gross_raw")
    manifest.add_argument("--output-root", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate per-run summaries.")
    aggregate.add_argument("--runs-root", type=Path, required=True)
    aggregate.add_argument("--output-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        summary = run_campaign_unit(args)
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        return
    if args.command == "write-manifest":
        write_manifest(args)
        return
    if args.command == "aggregate":
        aggregate_runs(args)
        return
    raise ValueError(f"Unsupported command: {args.command!r}")


def run_campaign_unit(args) -> dict[str, Any]:
    training_summary = run_training_unit(args)
    run_dir = Path(training_summary["run_dir"])
    exact_summary = training_summary["training"]
    random_test_comparison = exact_summary.get("random_test_comparison")
    half_comparison = exact_summary.get("half_comparison")
    if random_test_comparison is None or half_comparison is None:
        raise ValueError("Neural training summary did not include exact comparison payloads.")

    code_path = Path(training_summary["code_path"])
    from relay_bp.analysis import load_code_capacity_problem, with_uniform_error_rate, enumerate_half_stabilizer_cases, sample_random_error_cases

    base_problem = load_code_capacity_problem(code_path)
    problem = with_uniform_error_rate(base_problem, float(training_summary["p_value"]))
    half_cases = enumerate_half_stabilizer_cases(problem)
    random_test_cases = sample_random_error_cases(
        problem,
        num_samples=int(args.random_test_samples),
        error_rate=float(training_summary["p_value"]),
        base_seed=int(args.base_seed) + 200_000 + 10_000 * int(args.restart_index),
    )
    best_memory_bank = training_summary.get("best_memory_bank")
    best_damping_bank = training_summary.get("best_damping_bank")
    bank_member_diagnostics = summarize_bank_members(
        problem=problem,
        family=str(training_summary["family"]),
        memory_bank=best_memory_bank,
        damping_bank=best_damping_bank,
        random_test_cases=random_test_cases,
        half_cases=half_cases,
        max_iter=int(args.max_iter),
        alpha=args.alpha,
        logical_failure_penalty=float(args.logical_failure_penalty),
        logical_weight_penalty=float(args.logical_weight_penalty),
        convergence_penalty=float(args.convergence_penalty),
        iteration_penalty=float(args.iteration_penalty),
    )
    collapse_analysis = analyze_bank_collapse(
        memory_bank=best_memory_bank,
        damping_bank=best_damping_bank,
        family=str(training_summary["family"]),
        memory_span=0.6,
        damping_span=1.0,
    )
    verdict = verdict_from_comparisons(random_test_comparison, half_comparison)
    negative_reason = negative_reason_from_summary(
        training=exact_summary,
        collapse_analysis=collapse_analysis,
        verdict=verdict,
    )
    summary = {
        **{
            key: value
            for key, value in training_summary.items()
            if key not in {"best_memory_bank", "best_damping_bank", "best_distilled_memory_bank", "best_distilled_damping_bank"}
        },
        "random_test_comparison": random_test_comparison,
        "half_comparison": half_comparison,
        "bank_member_diagnostics": bank_member_diagnostics,
        "collapse_analysis": collapse_analysis,
        "verdict": verdict,
        "negative_reason": negative_reason,
    }
    (run_dir / "random_test_comparison.json").write_text(
        json.dumps(json_ready(random_test_comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "half_comparison.json").write_text(
        json.dumps(json_ready(half_comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "bank_member_diagnostics.json").write_text(
        json.dumps(json_ready(bank_member_diagnostics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "collapse_analysis.json").write_text(
        json.dumps(json_ready(collapse_analysis), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def write_manifest(args) -> None:
    code_labels = [label.strip() for label in str(args.code_labels).split(",") if label.strip()]
    families = [item.strip() for item in str(args.families).split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for code_label in code_labels:
        if code_label not in FROZEN_P_GRIDS:
            raise ValueError(f"Unsupported code_label: {code_label!r}")
        code_path = resolve_code_path(code_label, None)
        for p_value in FROZEN_P_GRIDS[code_label]:
            for family in families:
                for bank_size in args.bank_sizes:
                    for restart_index in range(max(1, int(args.restarts))):
                        rows.append(
                            {
                                "code_label": code_label,
                                "code_path": str(code_path.resolve()),
                                "p_value": float(p_value),
                                "family": family,
                                "bank_size": int(bank_size),
                                "restart_index": int(restart_index),
                                "output_dir": str(Path(args.output_root).resolve()),
                            }
                        )
    write_rows_csv(Path(args.output_path), rows)


def aggregate_runs(args) -> None:
    run_summaries = []
    for summary_path in sorted(Path(args.runs_root).glob("**/run_summary.json")):
        run_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    grouped: dict[tuple[str, float, str, int], list[dict[str, Any]]] = {}
    for summary in run_summaries:
        key = (
            str(summary["code_label"]),
            float(summary["p_value"]),
            str(summary["family"]),
            int(summary["bank_size"]),
        )
        grouped.setdefault(key, []).append(summary)
    best_by_group = {key: max(rows, key=aggregate_group_key) for key, rows in grouped.items()}
    best_by_code_p_family = best_by_code_p_family_map(best_by_group)
    payload = {
        "run_count": int(len(run_summaries)),
        "group_count": int(len(best_by_group)),
        "best_by_group": {str(key): value for key, value in best_by_group.items()},
        "best_by_code_p_family": {str(key): value for key, value in best_by_code_p_family.items()},
    }
    Path(args.output_path).write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def analyze_bank_collapse(
    *,
    memory_bank: np.ndarray | None,
    damping_bank: np.ndarray | None,
    family: str,
    memory_span: float,
    damping_span: float,
) -> dict[str, Any]:
    bank_size = infer_bank_size(memory_bank, damping_bank)
    if bank_size <= 1:
        return {
            "bank_size": int(bank_size),
            "collapsed": False,
            "max_normalized_pairwise_distance": None,
            "mean_normalized_pairwise_distance": None,
        }
    pairwise: list[float] = []
    for left_idx in range(bank_size):
        for right_idx in range(left_idx + 1, bank_size):
            distances = []
            if family in {"memory", "joint"} and memory_bank is not None:
                distances.append(
                    float(np.mean(np.abs(memory_bank[left_idx] - memory_bank[right_idx])) / max(float(memory_span), 1e-12))
                )
            if family in {"damping", "joint"} and damping_bank is not None:
                distances.append(
                    float(np.mean(np.abs(damping_bank[left_idx] - damping_bank[right_idx])) / max(float(damping_span), 1e-12))
                )
            pairwise.append(float(np.mean(distances)) if distances else 0.0)
    max_distance = float(max(pairwise)) if pairwise else 0.0
    mean_distance = float(np.mean(pairwise)) if pairwise else 0.0
    return {
        "bank_size": int(bank_size),
        "collapsed": bool(max_distance < 0.02),
        "max_normalized_pairwise_distance": max_distance,
        "mean_normalized_pairwise_distance": mean_distance,
        "pairwise_normalized_distances": pairwise,
    }


def negative_reason_from_summary(
    *,
    training: dict[str, Any],
    collapse_analysis: dict[str, Any],
    verdict: str,
) -> str | None:
    if str(verdict) != "negative_conclusion":
        return None
    if bool(collapse_analysis.get("collapsed")):
        return "bank_collapse"
    if training.get("distilled_beats_exact_baseline_validation") is False:
        return "surrogate_mismatch"
    return "true_repeated_random_advantage"


def aggregate_group_key(summary: dict[str, Any]) -> tuple[int, float, float, float]:
    verdict_rank = {"negative_conclusion": 0, "near_equal": 1, "success": 2}
    comparison = summary["random_test_comparison"]
    bank = comparison["bank_practical"]
    baseline = comparison["random_practical"]
    logical_gap = float(bank["logical_success_rate"]) - float(baseline["logical_success_rate"])
    iteration_ratio = float(bank["mean_iterations"]) / max(float(baseline["mean_iterations"]), 1e-12)
    return (
        int(verdict_rank.get(str(summary["verdict"]), 0)),
        float(logical_gap),
        -float(iteration_ratio),
        float(bank["convergence_rate"]),
    )


def best_by_code_p_family_map(
    best_by_group: dict[tuple[str, float, str, int], dict[str, Any]]
) -> dict[tuple[str, float, str], dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for (code_label, p_value, family, _bank_size), summary in best_by_group.items():
        grouped.setdefault((code_label, p_value, family), []).append(summary)
    return {key: max(rows, key=aggregate_group_key) for key, rows in grouped.items()}


if __name__ == "__main__":
    main()
