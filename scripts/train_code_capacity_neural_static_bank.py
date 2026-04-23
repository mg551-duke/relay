#!/usr/bin/env python3
"""Train one neural-distilled static assignment bank for a raw code-capacity code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from relay_bp.analysis import (
    NeuralStaticDistillationConfig,
    enumerate_half_stabilizer_cases,
    load_code_capacity_problem,
    sample_random_error_cases,
    train_neural_static_bank,
    with_uniform_error_rate,
)
from scripts.run_code_capacity_static_assignment_campaign import resolve_code_path


def parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in str(raw).split(",")]
    parsed = [float(item) for item in values if item]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return parsed


def parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(",")]
    parsed = [int(item) for item in values if item]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer value.")
    return parsed


def add_run_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--code-label", required=True)
    parser.add_argument("--code-path", type=Path, default=None)
    parser.add_argument("--p-value", type=float, required=True)
    parser.add_argument("--family", choices=["memory", "damping", "joint"], required=True)
    parser.add_argument("--bank-size", type=int, required=True)
    parser.add_argument("--restart-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random-train-samples", type=int, default=4096)
    parser.add_argument("--random-validation-samples", type=int, default=2048)
    parser.add_argument("--random-test-samples", type=int, default=4096)
    parser.add_argument("--random-train-batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--mixed-steps", type=int, default=20)
    parser.add_argument("--finetune-steps", type=int, default=20)
    parser.add_argument("--consolidation-steps", type=int, default=10)
    parser.add_argument("--mixed-half-fraction", type=float, default=0.5)
    parser.add_argument("--finetune-half-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--selector-temperature-start", type=float, default=1.0)
    parser.add_argument("--selector-temperature-end", type=float, default=0.05)
    parser.add_argument("--selector-temperature-consolidation", type=float, default=0.02)
    parser.add_argument("--logical-loss-weight", type=float, default=6.0)
    parser.add_argument("--syndrome-loss-weight", type=float, default=3.0)
    parser.add_argument("--iteration-trace-weight", type=float, default=1.0)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.05)
    parser.add_argument("--teacher-regularization-weight", type=float, default=0.01)
    parser.add_argument("--teacher-default-gamma", type=float, default=1.0)
    parser.add_argument("--teacher-default-delta", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--initialization-candidates", type=int, default=8)
    parser.add_argument("--initialization-subset-random-cases", type=int, default=128)
    parser.add_argument("--initialization-subset-half-cases", type=int, default=32)
    parser.add_argument("--selection-half-weight", type=float, default=1.0)
    parser.add_argument("--diversity-penalty", type=float, default=0.01)
    parser.add_argument("--diversity-threshold-fraction", type=float, default=0.05)
    parser.add_argument("--logical-failure-penalty", type=float, default=20.0)
    parser.add_argument("--logical-weight-penalty", type=float, default=4.0)
    parser.add_argument("--convergence-penalty", type=float, default=1.0)
    parser.add_argument("--iteration-penalty", type=float, default=0.1)
    parser.add_argument("--practical-k", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--comparison-repeats", type=int, default=16)
    parser.add_argument("--support-scan-random-cases", type=int, default=512)
    parser.add_argument("--center-values", type=parse_float_list, default=parse_float_list("0.0,0.1,0.2,0.3,0.4,0.5"))
    parser.add_argument("--width-values", type=parse_float_list, default=parse_float_list("0.0,0.2,0.4,0.6,0.8,1.0"))
    parser.add_argument("--gaussian-refinement-points", type=int, default=9)
    parser.add_argument("--exact-polish-candidate-count", type=int, default=16)
    parser.add_argument("--exact-polish-elite-count", type=int, default=4)
    parser.add_argument("--exact-polish-rounds", type=int, default=12)
    parser.add_argument("--exact-polish-validation-interval", type=int, default=2)
    parser.add_argument("--exact-polish-random-train-batch-size", type=int, default=128)
    parser.add_argument("--exact-polish-initialization-candidates", type=int, default=1)
    parser.add_argument("--exact-polish-memory-sigma", type=float, default=None)
    parser.add_argument("--exact-polish-damping-sigma", type=float, default=None)
    parser.add_argument("--exact-polish-sigma-floor", type=float, default=1e-3)
    parser.add_argument("--exact-polish-memory-sigma-ceiling", type=float, default=None)
    parser.add_argument("--exact-polish-damping-sigma-ceiling", type=float, default=None)
    parser.add_argument("--no-exact-baseline", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-seed", type=int, default=0)
    return parser


def build_parser() -> argparse.ArgumentParser:
    return add_run_arguments(argparse.ArgumentParser(description=__doc__))


def run_training_unit(args) -> dict[str, Any]:
    code_path = resolve_code_path(str(args.code_label), args.code_path)
    run_dir = (
        Path(args.output_dir)
        / str(args.code_label)
        / f"p_{format_float(float(args.p_value))}"
        / str(args.family)
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

    config = NeuralStaticDistillationConfig(
        code_path=str(code_path),
        checkpoint_dir=str(run_dir / "training"),
        family=str(args.family),
        bank_size=int(args.bank_size),
        max_iter=int(args.max_iter),
        alpha=args.alpha,
        train_random_error_rate=float(args.p_value),
        validation_random_error_rate=float(args.p_value),
        test_random_error_rate=float(args.p_value),
        random_train_batch_size=int(args.random_train_batch_size),
        warmup_steps=int(args.warmup_steps),
        mixed_steps=int(args.mixed_steps),
        finetune_steps=int(args.finetune_steps),
        consolidation_steps=int(args.consolidation_steps),
        mixed_half_fraction=float(args.mixed_half_fraction),
        finetune_half_fraction=float(args.finetune_half_fraction),
        learning_rate=float(args.learning_rate),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        weight_decay=float(args.weight_decay),
        selector_temperature_start=float(args.selector_temperature_start),
        selector_temperature_end=float(args.selector_temperature_end),
        selector_temperature_consolidation=float(args.selector_temperature_consolidation),
        logical_loss_weight=float(args.logical_loss_weight),
        syndrome_loss_weight=float(args.syndrome_loss_weight),
        iteration_trace_weight=float(args.iteration_trace_weight),
        confidence_loss_weight=float(args.confidence_loss_weight),
        teacher_regularization_weight=float(args.teacher_regularization_weight),
        teacher_default_gamma=float(args.teacher_default_gamma),
        teacher_default_delta=float(args.teacher_default_delta),
        validation_interval=int(args.validation_interval),
        initialization_candidates=int(args.initialization_candidates),
        initialization_subset_random_cases=int(args.initialization_subset_random_cases),
        initialization_subset_half_cases=int(args.initialization_subset_half_cases),
        selection_half_weight=float(args.selection_half_weight),
        diversity_penalty=float(args.diversity_penalty),
        diversity_threshold_fraction=float(args.diversity_threshold_fraction),
        logical_failure_penalty=float(args.logical_failure_penalty),
        logical_weight_penalty=float(args.logical_weight_penalty),
        convergence_penalty=float(args.convergence_penalty),
        iteration_penalty=float(args.iteration_penalty),
        practical_k=int(args.practical_k),
        bootstrap_samples=int(args.bootstrap_samples),
        comparison_repeats=int(args.comparison_repeats),
        support_scan_random_cases=int(args.support_scan_random_cases),
        support_centers=tuple(float(value) for value in args.center_values),
        support_widths=tuple(float(value) for value in args.width_values),
        gaussian_refinement_points=int(args.gaussian_refinement_points),
        exact_polish_candidate_count=int(args.exact_polish_candidate_count),
        exact_polish_elite_count=int(args.exact_polish_elite_count),
        exact_polish_rounds=int(args.exact_polish_rounds),
        exact_polish_validation_interval=int(args.exact_polish_validation_interval),
        exact_polish_random_train_batch_size=int(args.exact_polish_random_train_batch_size),
        exact_polish_initialization_candidates=int(args.exact_polish_initialization_candidates),
        exact_polish_memory_sigma=args.exact_polish_memory_sigma,
        exact_polish_damping_sigma=args.exact_polish_damping_sigma,
        exact_polish_sigma_floor=float(args.exact_polish_sigma_floor),
        exact_polish_memory_sigma_ceiling=args.exact_polish_memory_sigma_ceiling,
        exact_polish_damping_sigma_ceiling=args.exact_polish_damping_sigma_ceiling,
        run_exact_baseline=not bool(args.no_exact_baseline),
        device=str(args.device),
        base_seed=int(args.base_seed) + 1_000_000 * int(args.restart_index),
    )
    summary = train_neural_static_bank(
        problem,
        config=config,
        half_cases=half_cases,
        random_train_cases=random_train_cases,
        random_validation_cases=random_validation_cases,
        random_test_cases=random_test_cases,
    )
    run_summary = {
        "code_label": str(args.code_label),
        "code_path": str(code_path.resolve()),
        "p_value": float(args.p_value),
        "family": str(args.family),
        "bank_size": int(args.bank_size),
        "restart_index": int(args.restart_index),
        "run_dir": str(run_dir),
        "training": {
            key: value
            for key, value in summary.items()
            if key not in {"best_memory_bank", "best_damping_bank", "best_distilled_memory_bank", "best_distilled_damping_bank"}
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(json_ready(run_summary), indent=2, sort_keys=True), encoding="utf-8")
    return {
        **run_summary,
        "best_memory_bank": summary.get("best_memory_bank"),
        "best_damping_bank": summary.get("best_damping_bank"),
        "best_distilled_memory_bank": summary.get("best_distilled_memory_bank"),
        "best_distilled_damping_bank": summary.get("best_distilled_damping_bank"),
    }


def format_float(value: float) -> str:
    return str(value).replace(".", "p")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    args = build_parser().parse_args()
    summary = run_training_unit(args)
    printable = {
        key: value
        for key, value in summary.items()
        if key not in {"best_memory_bank", "best_damping_bank", "best_distilled_memory_bank", "best_distilled_damping_bank"}
    }
    print(json.dumps(json_ready(printable), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
