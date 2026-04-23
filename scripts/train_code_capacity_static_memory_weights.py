#!/usr/bin/env python3
"""Train static disordered memory weights on code-capacity NPZ exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relay_bp.analysis import default_export_code_paths, load_code_capacity_problem, with_uniform_error_rate
from relay_bp.analysis.code_capacity_training import (
    StaticMemoryTrainingConfig,
    train_static_memory_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_PATH = default_export_code_paths(REPO_ROOT)["surface13"]
DEFAULT_CHECKPOINT_DIR = (
    REPO_ROOT / "examples" / "notebook_data" / "static_memory_training" / "surface13_static_memory_es"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-path", type=Path, default=DEFAULT_CODE_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-random-error-rate", type=float, default=0.1)
    parser.add_argument("--validation-random-error-rate", type=float, default=0.1)
    parser.add_argument("--half-train-limit", type=int, default=128)
    parser.add_argument("--half-validation-limit", type=int, default=64)
    parser.add_argument("--random-validation-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--perturbations-per-step", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.075)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--initial-raw-std", type=float, default=0.5)
    parser.add_argument("--memory-min", type=float, default=-0.3)
    parser.add_argument("--memory-max", type=float, default=0.3)
    parser.add_argument("--baseline-memory-value", type=float, default=0.15)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--mixed-steps", type=int, default=40)
    parser.add_argument("--finetune-steps", type=int, default=40)
    parser.add_argument("--mixed-half-fraction", type=float, default=0.5)
    parser.add_argument("--finetune-half-fraction", type=float, default=0.2)
    parser.add_argument("--finetune-replay-fraction", type=float, default=0.3)
    parser.add_argument("--selection-half-weight", type=float, default=0.75)
    parser.add_argument("--logical-weight-penalty", type=float, default=4.0)
    parser.add_argument("--convergence-penalty", type=float, default=1.0)
    parser.add_argument("--iteration-penalty", type=float, default=0.05)
    parser.add_argument("--max-hard-replay-cases", type=int, default=32)
    parser.add_argument(
        "--initialization-mode",
        choices=["gaussian", "interval_best", "interval_topk_average"],
        default="gaussian",
    )
    parser.add_argument("--initialization-num-candidates", type=int, default=32)
    parser.add_argument("--initialization-topk", type=int, default=4)
    parser.add_argument("--prior-error-rate", type=float, default=None)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    problem = load_code_capacity_problem(args.code_path)
    if args.prior_error_rate is not None:
        problem = with_uniform_error_rate(problem, args.prior_error_rate)
    config = StaticMemoryTrainingConfig(
        code_path=str(Path(args.code_path).resolve()),
        checkpoint_dir=str(Path(args.checkpoint_dir).resolve()),
        max_iter=args.max_iter,
        alpha=args.alpha,
        train_random_error_rate=args.train_random_error_rate,
        validation_random_error_rate=args.validation_random_error_rate,
        half_train_limit=args.half_train_limit,
        half_validation_limit=args.half_validation_limit,
        random_validation_samples=args.random_validation_samples,
        batch_size=args.batch_size,
        perturbations_per_step=args.perturbations_per_step,
        noise_std=args.noise_std,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        initial_raw_std=args.initial_raw_std,
        memory_min=args.memory_min,
        memory_max=args.memory_max,
        baseline_memory_value=args.baseline_memory_value,
        validation_interval=args.validation_interval,
        warmup_steps=args.warmup_steps,
        mixed_steps=args.mixed_steps,
        finetune_steps=args.finetune_steps,
        mixed_half_fraction=args.mixed_half_fraction,
        finetune_half_fraction=args.finetune_half_fraction,
        finetune_replay_fraction=args.finetune_replay_fraction,
        selection_half_weight=args.selection_half_weight,
        logical_weight_penalty=args.logical_weight_penalty,
        convergence_penalty=args.convergence_penalty,
        iteration_penalty=args.iteration_penalty,
        max_hard_replay_cases=args.max_hard_replay_cases,
        initialization_mode=args.initialization_mode,
        initialization_num_candidates=args.initialization_num_candidates,
        initialization_topk=args.initialization_topk,
        base_seed=args.base_seed,
        resume=not args.no_resume,
    )
    try:
        summary = train_static_memory_weights(problem, config=config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
