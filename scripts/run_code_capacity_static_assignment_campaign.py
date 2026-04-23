#!/usr/bin/env python3
"""Train and evaluate deterministic static assignment banks for raw code-capacity codes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from relay_bp.analysis import (
    ScalarDistributionSpec,
    StaticAssignmentBankConfig,
    compare_bank_to_random_baselines,
    enumerate_half_stabilizer_cases,
    evaluate_assignment_bank,
    load_code_capacity_problem,
    sample_random_error_cases,
    scan_damping_distribution_support,
    scan_memory_distribution_support,
    train_assignment_bank_cem,
    with_uniform_error_rate,
)


FROZEN_P_GRIDS = {
    "surface5_raw": [0.01, 0.05, 0.11, 0.21, 0.50],
    "surface13_raw": [0.005, 0.015, 0.03, 0.045, 0.10],
    "gross_raw": [0.015, 0.03, 0.04, 0.06, 0.10],
    "two_gross_raw": [0.015, 0.03, 0.04, 0.05, 0.08],
}

RAW_CODE_PATH_CANDIDATES = {
    "surface5_raw": [Path(r"C:\Users\User\Documents\Code from BPGD\surface5_HxHzLxLz.npz")],
    "surface13_raw": [Path(r"C:\Users\User\Documents\Code from BPGD\surface13_HxHzLxLz.npz")],
    "gross_raw": [Path(r"C:\Users\User\Documents\projects-git\bpgd_low_llr_and_hybrid\gross_HxHzLxLz.npz")],
    "two_gross_raw": [Path(r"C:\Users\User\Documents\projects-git\bpgd_low_llr_and_hybrid\two_gross_HxHzLxLz.npz")],
}


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


def default_raw_code_paths() -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for label, candidates in RAW_CODE_PATH_CANDIDATES.items():
        for candidate in candidates:
            if candidate.exists():
                resolved[label] = candidate
                break
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one training unit.")
    run.add_argument("--code-label", required=True)
    run.add_argument("--code-path", type=Path, default=None)
    run.add_argument("--p-value", type=float, required=True)
    run.add_argument("--family", choices=["memory", "damping", "joint"], required=True)
    run.add_argument("--bank-size", type=int, required=True)
    run.add_argument("--restart-index", type=int, default=0)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--center-values", type=_parse_float_list, default=_parse_float_list("0.0,0.1,0.2,0.3,0.4,0.5"))
    run.add_argument("--width-values", type=_parse_float_list, default=_parse_float_list("0.0,0.2,0.4,0.6,0.8,1.0"))
    run.add_argument("--gaussian-refinement-points", type=int, default=9)
    run.add_argument("--scan-random-cases", type=int, default=512)
    run.add_argument("--random-train-samples", type=int, default=4096)
    run.add_argument("--random-validation-samples", type=int, default=2048)
    run.add_argument("--random-test-samples", type=int, default=4096)
    run.add_argument("--candidate-count", type=int, default=64)
    run.add_argument("--elite-count", type=int, default=8)
    run.add_argument("--rounds", type=int, default=80)
    run.add_argument("--validation-interval", type=int, default=5)
    run.add_argument("--practical-k", type=int, default=8)
    run.add_argument("--comparison-repeats", type=int, default=16)
    run.add_argument("--bootstrap-samples", type=int, default=2000)
    run.add_argument("--max-iter", type=int, default=40)
    run.add_argument("--alpha", type=float, default=1.0)
    run.add_argument("--selection-half-weight", type=float, default=1.0)
    run.add_argument("--logical-failure-penalty", type=float, default=20.0)
    run.add_argument("--logical-weight-penalty", type=float, default=4.0)
    run.add_argument("--convergence-penalty", type=float, default=1.0)
    run.add_argument("--iteration-penalty", type=float, default=0.1)
    run.add_argument("--diversity-penalty", type=float, default=0.01)
    run.add_argument("--base-seed", type=int, default=0)

    manifest = subparsers.add_parser("write-manifest", help="Write a CSV manifest for Slurm arrays.")
    manifest.add_argument("--output-path", type=Path, required=True)
    manifest.add_argument("--families", default="memory,damping")
    manifest.add_argument("--bank-sizes", type=_parse_int_list, default=_parse_int_list("1,2,4"))
    manifest.add_argument("--restarts", type=int, default=8)
    manifest.add_argument("--code-labels", default="surface5_raw,surface13_raw,gross_raw,two_gross_raw")
    manifest.add_argument("--output-root", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate run summaries and optionally emit a joint shortlist manifest.")
    aggregate.add_argument("--runs-root", type=Path, required=True)
    aggregate.add_argument("--output-path", type=Path, required=True)
    aggregate.add_argument("--shortlist-manifest-path", type=Path, default=None)
    aggregate.add_argument("--bank8-shortlist-manifest-path", type=Path, default=None)
    aggregate.add_argument("--joint-bank-sizes", type=_parse_int_list, default=_parse_int_list("1,2,4"))
    aggregate.add_argument("--joint-restarts", type=int, default=12)
    aggregate.add_argument("--bank8-restarts", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        summary = run_training_unit(args)
        print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
        return
    if args.command == "write-manifest":
        write_manifest(args)
        return
    if args.command == "aggregate":
        aggregate_runs(args)
        return
    raise ValueError(f"Unsupported command: {args.command!r}")


def run_training_unit(args) -> dict[str, Any]:
    code_path = resolve_code_path(args.code_label, args.code_path)
    run_dir = (
        Path(args.output_dir)
        / args.code_label
        / f"p_{_format_float(args.p_value)}"
        / args.family
        / f"bank_{int(args.bank_size)}"
        / f"restart_{int(args.restart_index)}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    base_problem = load_code_capacity_problem(code_path)
    problem = with_uniform_error_rate(base_problem, float(args.p_value))
    half_cases = enumerate_half_stabilizer_cases(problem)
    if not half_cases:
        raise ValueError("No half-stabilizer cases were generated for this code.")
    random_train_cases = sample_random_error_cases(
        problem,
        num_samples=int(args.random_train_samples),
        error_rate=float(args.p_value),
        base_seed=int(args.base_seed) + 10_000 * int(args.restart_index),
    )
    random_validation_cases = sample_random_error_cases(
        problem,
        num_samples=int(args.random_validation_samples),
        error_rate=float(args.p_value),
        base_seed=int(args.base_seed) + 100_000 + 10_000 * int(args.restart_index),
    )
    random_test_cases = sample_random_error_cases(
        problem,
        num_samples=int(args.random_test_samples),
        error_rate=float(args.p_value),
        base_seed=int(args.base_seed) + 200_000 + 10_000 * int(args.restart_index),
    )
    scan_random_cases = random_train_cases[: min(int(args.scan_random_cases), len(random_train_cases))]

    support_payload: dict[str, Any] = {}
    memory_support = None
    damping_support = None
    if args.family in {"memory", "joint"}:
        support_payload["memory"] = scan_memory_distribution_support(
            problem=problem,
            random_cases=scan_random_cases,
            half_cases=half_cases,
            centers=args.center_values,
            widths=args.width_values,
            gaussian_refinement_points=int(args.gaussian_refinement_points),
            draw_count=int(args.practical_k),
            max_iter=int(args.max_iter),
            alpha=args.alpha,
            logical_failure_penalty=float(args.logical_failure_penalty),
            logical_weight_penalty=float(args.logical_weight_penalty),
            convergence_penalty=float(args.convergence_penalty),
            iteration_penalty=float(args.iteration_penalty),
            selection_half_weight=float(args.selection_half_weight),
            base_seed=int(args.base_seed) + 300_000 + 10_000 * int(args.restart_index),
        )
        memory_support = support_payload["memory"]["best_overall"]
    if args.family in {"damping", "joint"}:
        support_payload["damping"] = scan_damping_distribution_support(
            problem=problem,
            random_cases=scan_random_cases,
            half_cases=half_cases,
            centers=args.center_values,
            widths=args.width_values,
            gaussian_refinement_points=int(args.gaussian_refinement_points),
            draw_count=int(args.practical_k),
            max_iter=int(args.max_iter),
            alpha=args.alpha,
            logical_failure_penalty=float(args.logical_failure_penalty),
            logical_weight_penalty=float(args.logical_weight_penalty),
            convergence_penalty=float(args.convergence_penalty),
            iteration_penalty=float(args.iteration_penalty),
            selection_half_weight=float(args.selection_half_weight),
            base_seed=int(args.base_seed) + 600_000 + 10_000 * int(args.restart_index),
        )
        damping_support = support_payload["damping"]["best_overall"]

    config = StaticAssignmentBankConfig(
        code_path=str(code_path),
        checkpoint_dir=str(run_dir / "training"),
        family=str(args.family),
        bank_size=int(args.bank_size),
        max_iter=int(args.max_iter),
        alpha=args.alpha,
        train_random_error_rate=float(args.p_value),
        validation_random_error_rate=float(args.p_value),
        test_random_error_rate=float(args.p_value),
        random_train_samples=int(args.random_train_samples),
        random_validation_samples=int(args.random_validation_samples),
        random_test_samples=int(args.random_test_samples),
        candidate_count=int(args.candidate_count),
        elite_count=int(args.elite_count),
        rounds=int(args.rounds),
        validation_interval=int(args.validation_interval),
        practical_k=int(args.practical_k),
        logical_failure_penalty=float(args.logical_failure_penalty),
        logical_weight_penalty=float(args.logical_weight_penalty),
        convergence_penalty=float(args.convergence_penalty),
        iteration_penalty=float(args.iteration_penalty),
        selection_half_weight=float(args.selection_half_weight),
        diversity_penalty=float(args.diversity_penalty),
        base_seed=int(args.base_seed) + 1_000_000 * int(args.restart_index),
    )
    training_summary = train_assignment_bank_cem(
        problem,
        config=config,
        half_cases=half_cases,
        random_train_cases=random_train_cases,
        random_validation_cases=random_validation_cases,
        memory_support=memory_support,
        damping_support=damping_support,
    )

    memory_distribution = None if memory_support is None else distribution_from_support(memory_support)
    damping_distribution = None if damping_support is None else distribution_from_support(damping_support)
    test_comparison = compare_bank_to_random_baselines(
        problem=problem,
        cases=random_test_cases,
        family=str(args.family),
        memory_bank=training_summary["best_memory_bank"],
        damping_bank=training_summary["best_damping_bank"],
        memory_distribution=memory_distribution,
        damping_distribution=damping_distribution,
        max_iter=int(args.max_iter),
        alpha=args.alpha,
        k_draw=int(args.practical_k),
        logical_failure_penalty=float(args.logical_failure_penalty),
        logical_weight_penalty=float(args.logical_weight_penalty),
        convergence_penalty=float(args.convergence_penalty),
        iteration_penalty=float(args.iteration_penalty),
        bootstrap_samples=int(args.bootstrap_samples),
        repeat_count=int(args.comparison_repeats),
        base_seed=int(args.base_seed) + 2_000_000 + 10_000 * int(args.restart_index),
    )
    half_comparison = compare_bank_to_random_baselines(
        problem=problem,
        cases=half_cases,
        family=str(args.family),
        memory_bank=training_summary["best_memory_bank"],
        damping_bank=training_summary["best_damping_bank"],
        memory_distribution=memory_distribution,
        damping_distribution=damping_distribution,
        max_iter=int(args.max_iter),
        alpha=args.alpha,
        k_draw=int(args.practical_k),
        logical_failure_penalty=float(args.logical_failure_penalty),
        logical_weight_penalty=float(args.logical_weight_penalty),
        convergence_penalty=float(args.convergence_penalty),
        iteration_penalty=float(args.iteration_penalty),
        bootstrap_samples=int(args.bootstrap_samples),
        repeat_count=int(args.comparison_repeats),
        base_seed=int(args.base_seed) + 3_000_000 + 10_000 * int(args.restart_index),
    )
    bank_member_diagnostics = summarize_bank_members(
        problem=problem,
        family=str(args.family),
        memory_bank=training_summary["best_memory_bank"],
        damping_bank=training_summary["best_damping_bank"],
        random_test_cases=random_test_cases,
        half_cases=half_cases,
        max_iter=int(args.max_iter),
        alpha=args.alpha,
        logical_failure_penalty=float(args.logical_failure_penalty),
        logical_weight_penalty=float(args.logical_weight_penalty),
        convergence_penalty=float(args.convergence_penalty),
        iteration_penalty=float(args.iteration_penalty),
    )
    summary = {
        "code_label": str(args.code_label),
        "code_path": str(code_path.resolve()),
        "p_value": float(args.p_value),
        "family": str(args.family),
        "bank_size": int(args.bank_size),
        "restart_index": int(args.restart_index),
        "run_dir": str(run_dir),
        "support_scan": support_payload,
        "training": {
            key: value
            for key, value in training_summary.items()
            if key not in {"best_memory_bank", "best_damping_bank"}
        },
        "random_test_comparison": test_comparison,
        "half_comparison": half_comparison,
        "verdict": verdict_from_comparisons(test_comparison, half_comparison),
    }
    (run_dir / "support_scan.json").write_text(
        json.dumps(_json_ready(support_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "random_test_comparison.json").write_text(
        json.dumps(_json_ready(test_comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "half_comparison.json").write_text(
        json.dumps(_json_ready(half_comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "bank_member_diagnostics.json").write_text(
        json.dumps(_json_ready(bank_member_diagnostics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_manifest(args) -> None:
    raw_paths = default_raw_code_paths()
    code_labels = [label.strip() for label in str(args.code_labels).split(",") if label.strip()]
    families = [item.strip() for item in str(args.families).split(",") if item.strip()]
    rows = []
    for code_label in code_labels:
        if code_label not in FROZEN_P_GRIDS:
            raise ValueError(f"Unsupported code_label: {code_label!r}")
        code_path = resolve_code_path(code_label, raw_paths.get(code_label))
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
    best_by_group = {
        key: max(rows, key=_aggregate_group_key)
        for key, rows in grouped.items()
    }
    best_by_code_p_family = _best_by_code_p_family(best_by_group)
    payload = {
        "run_count": int(len(run_summaries)),
        "group_count": int(len(best_by_group)),
        "best_by_group": {str(key): value for key, value in best_by_group.items()},
        "best_by_code_p_family": {str(key): value for key, value in best_by_code_p_family.items()},
    }
    Path(args.output_path).write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")

    if args.shortlist_manifest_path is not None:
        shortlist_rows = []
        for key, summary in best_by_code_p_family.items():
            code_label, p_value, family = key
            if family not in {"memory", "damping"}:
                continue
            if qualifies_for_joint_shortlist(summary):
                code_path = resolve_code_path(code_label, None)
                for bank_size in args.joint_bank_sizes:
                    for restart_index in range(max(1, int(args.joint_restarts))):
                        shortlist_rows.append(
                            {
                                "code_label": code_label,
                                "code_path": str(code_path.resolve()),
                                "p_value": float(p_value),
                                "family": "joint",
                                "bank_size": int(bank_size),
                                "restart_index": int(restart_index),
                                "output_dir": str(Path(args.runs_root).resolve()),
                            }
                        )
        write_rows_csv(Path(args.shortlist_manifest_path), shortlist_rows)
    if args.bank8_shortlist_manifest_path is not None:
        shortlist_rows = []
        for key, summary in best_by_group.items():
            code_label, p_value, family, bank_size = key
            if family not in {"memory", "damping"} or int(bank_size) != 4:
                continue
            if needs_bank8_escalation(summary):
                code_path = resolve_code_path(code_label, None)
                for restart_index in range(max(1, int(args.bank8_restarts))):
                    shortlist_rows.append(
                        {
                            "code_label": code_label,
                            "code_path": str(code_path.resolve()),
                            "p_value": float(p_value),
                            "family": family,
                            "bank_size": 8,
                            "restart_index": int(restart_index),
                            "output_dir": str(Path(args.runs_root).resolve()),
                        }
                    )
        write_rows_csv(Path(args.bank8_shortlist_manifest_path), shortlist_rows)


def resolve_code_path(code_label: str, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).resolve()
    raw_paths = default_raw_code_paths()
    if code_label in raw_paths:
        return raw_paths[code_label].resolve()
    raise ValueError(f"No default path found for {code_label!r}; pass --code-path.")


def distribution_from_support(row: dict[str, Any]) -> ScalarDistributionSpec:
    if "mean" in row and "sigma" in row:
        return ScalarDistributionSpec(
            sampling_family="truncated_gaussian",
            low=float(row["low"]),
            high=float(row["high"]),
            mean=float(row["mean"]),
            sigma=float(row["sigma"]),
        )
    return ScalarDistributionSpec(
        sampling_family="uniform_interval",
        low=float(row["low"]),
        high=float(row["high"]),
    )


def summarize_bank_members(
    *,
    problem,
    family: str,
    memory_bank,
    damping_bank,
    random_test_cases,
    half_cases,
    max_iter: int,
    alpha: float | None,
    logical_failure_penalty: float,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
) -> list[dict[str, Any]]:
    diagnostics = []
    bank_size = infer_bank_size(memory_bank, damping_bank)
    for bank_idx in range(bank_size):
        member_memory = None if memory_bank is None else memory_bank[bank_idx : bank_idx + 1]
        member_damping = None if damping_bank is None else damping_bank[bank_idx : bank_idx + 1]
        random_metrics = evaluate_assignment_bank(
            problem=problem,
            cases=random_test_cases,
            family=family,
            memory_bank=member_memory,
            damping_bank=member_damping,
            max_iter=max_iter,
            alpha=alpha,
            selection_mode="practical",
            logical_failure_penalty=logical_failure_penalty,
            logical_weight_penalty=logical_weight_penalty,
            convergence_penalty=convergence_penalty,
            iteration_penalty=iteration_penalty,
        )
        half_metrics = evaluate_assignment_bank(
            problem=problem,
            cases=half_cases,
            family=family,
            memory_bank=member_memory,
            damping_bank=member_damping,
            max_iter=max_iter,
            alpha=alpha,
            selection_mode="practical",
            logical_failure_penalty=logical_failure_penalty,
            logical_weight_penalty=logical_weight_penalty,
            convergence_penalty=convergence_penalty,
            iteration_penalty=iteration_penalty,
        )
        diagnostics.append(
            {
                "bank_idx": int(bank_idx),
                "random_test": random_metrics,
                "half": half_metrics,
            }
        )
    return diagnostics


def infer_bank_size(memory_bank, damping_bank) -> int:
    if memory_bank is not None:
        return int(memory_bank.shape[0])
    if damping_bank is not None:
        return int(damping_bank.shape[0])
    return 0


def verdict_from_comparisons(random_test_comparison: dict[str, Any], half_comparison: dict[str, Any]) -> str:
    random_practical = random_test_comparison["practical_difference"]
    half_practical = half_comparison["practical_difference"]
    if _meets_success(random_practical) and _meets_success(half_practical):
        return "success"
    if _meets_near_equal(random_practical) and _meets_near_equal(half_practical):
        return "near_equal"
    return "negative_conclusion"


def qualifies_for_joint_shortlist(summary: dict[str, Any]) -> bool:
    comparison = summary["random_test_comparison"]
    bank = comparison["bank_practical"]
    baseline = comparison["random_practical"]
    logical_gap = float(bank["logical_success_rate"]) - float(baseline["logical_success_rate"])
    iteration_ratio = float(bank["mean_iterations"]) / max(float(baseline["mean_iterations"]), 1e-12)
    return (logical_gap >= -0.01) or (iteration_ratio <= 0.85 and logical_gap >= -0.02)


def _meets_success(report: dict[str, Any]) -> bool:
    return (
        float(report["logical_success_difference"]["lower"]) >= 0.0
        and float(report["convergence_difference"]["lower"]) >= -0.01
        and float(report["iteration_ratio"]["upper"]) <= 1.05
    )


def _meets_near_equal(report: dict[str, Any]) -> bool:
    return (
        float(report["logical_success_difference"]["lower"]) >= -0.002
        and float(report["convergence_difference"]["lower"]) >= -0.02
        and float(report["iteration_ratio"]["upper"]) <= 1.10
    )


def _aggregate_group_key(summary: dict[str, Any]) -> tuple[int, float, float, float]:
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
        float(comparison["bank_practical"]["convergence_rate"]),
    )


def needs_bank8_escalation(summary: dict[str, Any]) -> bool:
    comparison = summary["random_test_comparison"]
    bank = comparison["bank_practical"]
    baseline = comparison["random_practical"]
    logical_gap = float(bank["logical_success_rate"]) - float(baseline["logical_success_rate"])
    iteration_ratio = float(bank["mean_iterations"]) / max(float(baseline["mean_iterations"]), 1e-12)
    return logical_gap < -0.002 or iteration_ratio > 1.05


def _best_by_code_p_family(
    best_by_group: dict[tuple[str, float, str, int], dict[str, Any]]
) -> dict[tuple[str, float, str], dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for (code_label, p_value, family, _bank_size), summary in best_by_group.items():
        grouped.setdefault((code_label, p_value, family), []).append(summary)
    return {
        key: max(rows, key=_aggregate_group_key)
        for key, rows in grouped.items()
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: float) -> str:
    return str(value).replace(".", "p")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
    except ModuleNotFoundError:
        pass
    return value


if __name__ == "__main__":
    main()
