from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .code_capacity_common import (
    CodeCapacityProblem,
    decode_with_memory_strengths,
    decode_with_plain_bp,
    enumerate_half_stabilizer_cases,
    evaluate_decode_result,
    make_min_sum_tracer,
    sample_random_error_cases,
)


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required for static memory training. Install it with `pip install -e .[ml]`."
        ) from exc
    return torch


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    steps: int
    half_fraction: float
    replay_fraction: float


@dataclass(frozen=True)
class StaticMemoryTrainingConfig:
    code_path: str
    checkpoint_dir: str
    max_iter: int = 40
    alpha: float | None = 1.0
    train_random_error_rate: float = 0.1
    validation_random_error_rate: float = 0.1
    half_train_limit: int = 128
    half_validation_limit: int = 64
    random_validation_samples: int = 64
    batch_size: int = 16
    perturbations_per_step: int = 4
    noise_std: float = 0.075
    learning_rate: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    initial_raw_std: float = 0.5
    memory_min: float = -0.3
    memory_max: float = 0.3
    baseline_memory_value: float = 0.15
    validation_interval: int = 10
    warmup_steps: int = 30
    mixed_steps: int = 40
    finetune_steps: int = 40
    mixed_half_fraction: float = 0.5
    finetune_half_fraction: float = 0.2
    finetune_replay_fraction: float = 0.3
    selection_half_weight: float = 0.75
    logical_weight_penalty: float = 4.0
    convergence_penalty: float = 1.0
    iteration_penalty: float = 0.05
    max_hard_replay_cases: int = 32
    initialization_mode: str = "gaussian"
    initialization_num_candidates: int = 32
    initialization_topk: int = 4
    base_seed: int = 0
    resume: bool = True

    @property
    def total_steps(self) -> int:
        return int(self.warmup_steps + self.mixed_steps + self.finetune_steps)


def bounded_memory_strengths(raw_params: Any, *, memory_min: float, memory_max: float) -> Any:
    torch = require_torch()
    center = (float(memory_min) + float(memory_max)) / 2.0
    half_span = (float(memory_max) - float(memory_min)) / 2.0
    if half_span <= 0.0:
        raise ValueError("memory_max must be greater than memory_min.")
    return center + half_span * torch.tanh(raw_params)


def raw_memory_parameters(
    memory_strengths: np.ndarray,
    *,
    memory_min: float,
    memory_max: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    center = (float(memory_min) + float(memory_max)) / 2.0
    half_span = (float(memory_max) - float(memory_min)) / 2.0
    if half_span <= 0.0:
        raise ValueError("memory_max must be greater than memory_min.")
    scaled = (np.asarray(memory_strengths, dtype=np.float64) - center) / half_span
    clipped = np.clip(scaled, -1.0 + float(epsilon), 1.0 - float(epsilon))
    return np.arctanh(clipped).astype(np.float64)


def average_top_candidate_memory_strengths(
    candidate_memory_strengths: Sequence[np.ndarray],
    candidate_losses: Sequence[float],
    *,
    topk: int,
) -> np.ndarray:
    if not candidate_memory_strengths:
        raise ValueError("candidate_memory_strengths must not be empty.")
    losses = np.asarray(candidate_losses, dtype=np.float64)
    if losses.shape != (len(candidate_memory_strengths),):
        raise ValueError("candidate_losses must match candidate_memory_strengths length.")
    order = np.argsort(losses)
    topk = max(1, min(int(topk), len(candidate_memory_strengths)))
    selected_indices = order[:topk]
    selected_losses = losses[selected_indices]
    selected_vectors = np.asarray(
        [np.asarray(candidate_memory_strengths[int(index)], dtype=np.float64) for index in selected_indices],
        dtype=np.float64,
    )
    loss_scale = max(float(np.std(selected_losses)), 1e-6)
    logits = -((selected_losses - float(selected_losses.min())) / loss_scale)
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weights = weights / float(np.sum(weights))
    return np.sum(selected_vectors * weights[:, None], axis=0, dtype=np.float64)


def evaluate_memory_weight_vector(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    memory_strengths: np.ndarray | None,
    max_iter: int,
    alpha: float | None,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    return_case_rows: bool = False,
) -> dict[str, Any]:
    tracer = make_min_sum_tracer(
        problem,
        max_iter=max_iter,
        alpha=alpha,
        gamma0=0.0 if memory_strengths is not None else None,
    )
    logical_success_sum = 0.0
    exact_recovery_sum = 0.0
    convergence_sum = 0.0
    iterations_sum = 0.0
    logical_weight_sum = 0.0
    loss_sum = 0.0
    case_rows: list[dict[str, Any]] = []

    if memory_strengths is not None:
        memory_strengths = np.asarray(memory_strengths, dtype=np.float64)
        if memory_strengths.shape != (problem.n_bits,):
            raise ValueError(
                f"memory_strengths has shape {memory_strengths.shape}, expected {(problem.n_bits,)}."
            )

    for case in cases:
        syndrome = np.asarray(case["syndrome"], dtype=np.uint8)
        if memory_strengths is None:
            result = decode_with_plain_bp(tracer, syndrome, n_bits=problem.n_bits)
        else:
            result = decode_with_memory_strengths(tracer, syndrome, memory_strengths)
        evaluation = evaluate_decode_result(result=result, error=case["error"], lz=problem.lz)
        loss = _case_loss(
            logical_weight=evaluation.logical_weight,
            converged=evaluation.converged,
            iterations=evaluation.iterations,
            max_iter=max_iter,
            logical_weight_penalty=logical_weight_penalty,
            convergence_penalty=convergence_penalty,
            iteration_penalty=iteration_penalty,
        )
        logical_success_sum += float(evaluation.logical_success)
        exact_recovery_sum += float(evaluation.exact_recovery)
        convergence_sum += float(evaluation.converged)
        iterations_sum += float(evaluation.iterations)
        logical_weight_sum += float(evaluation.logical_weight)
        loss_sum += loss
        if return_case_rows:
            case_rows.append(
                {
                    "case_id": int(case.get("case_id", case.get("sample_idx", -1))),
                    "row_idx": int(case.get("row_idx", -1)),
                    "row_weight": int(case.get("row_weight", -1)),
                    "logical_success": bool(evaluation.logical_success),
                    "exact_recovery": bool(evaluation.exact_recovery),
                    "converged": bool(evaluation.converged),
                    "logical_weight": int(evaluation.logical_weight),
                    "iterations": int(evaluation.iterations),
                    "loss": float(loss),
                    "support_bits": list(case.get("support_bits", ())),
                    "flipped_bits": list(case.get("flipped_bits", ())),
                }
            )

    num_cases = max(1, len(cases))
    metrics = {
        "num_cases": int(len(cases)),
        "loss_mean": float(loss_sum / num_cases),
        "logical_success_rate": float(logical_success_sum / num_cases),
        "exact_recovery_rate": float(exact_recovery_sum / num_cases),
        "convergence_rate": float(convergence_sum / num_cases),
        "mean_iterations": float(iterations_sum / num_cases),
        "mean_logical_weight": float(logical_weight_sum / num_cases),
    }
    if return_case_rows:
        metrics["case_rows"] = case_rows
    return metrics


def train_static_memory_weights(
    problem: CodeCapacityProblem,
    *,
    config: StaticMemoryTrainingConfig,
) -> dict[str, Any]:
    torch = require_torch()
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    half_train_cases, half_validation_cases = _build_half_case_splits(problem, config)
    random_validation_cases = sample_random_error_cases(
        problem,
        num_samples=config.random_validation_samples,
        error_rate=config.validation_random_error_rate,
        base_seed=config.base_seed + 50_000,
    )
    baseline_metrics = _compute_baseline_metrics(
        problem=problem,
        config=config,
        half_validation_cases=half_validation_cases,
        random_validation_cases=random_validation_cases,
    )
    _write_json(checkpoint_dir / "config.json", _config_payload(problem, config))
    _write_json(checkpoint_dir / "baseline_metrics.json", baseline_metrics)

    (
        raw_params,
        optimizer,
        completed_steps,
        best_selection_loss,
        hard_replay_case_ids,
        best_validation_row,
    ) = _load_or_init_state(
        problem=problem,
        config=config,
        checkpoint_dir=checkpoint_dir,
        half_validation_cases=half_validation_cases,
        random_validation_cases=random_validation_cases,
    )
    phases = [
        CurriculumPhase(name="warmup_half_stabilizer", steps=config.warmup_steps, half_fraction=1.0, replay_fraction=0.0),
        CurriculumPhase(
            name="mixed_half_and_random",
            steps=config.mixed_steps,
            half_fraction=float(config.mixed_half_fraction),
            replay_fraction=0.0,
        ),
        CurriculumPhase(
            name="random_dominant_finetune",
            steps=config.finetune_steps,
            half_fraction=float(config.finetune_half_fraction),
            replay_fraction=float(config.finetune_replay_fraction),
        ),
    ]
    total_steps = config.total_steps
    if completed_steps >= total_steps:
        summary = _finalize_training_artifacts(
            raw_params=raw_params,
            checkpoint_dir=checkpoint_dir,
            config=config,
            problem=problem,
            completed_steps=completed_steps,
            best_selection_loss=best_selection_loss,
            resumed=True,
        )
        summary["baseline_metrics"] = baseline_metrics
        return summary

    rng = np.random.default_rng(config.base_seed + completed_steps)
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(int(config.base_seed) + int(completed_steps))
    train_metrics_path = checkpoint_dir / "train_metrics.csv"
    validation_metrics_path = checkpoint_dir / "validation_metrics.csv"

    for step in range(completed_steps, total_steps):
        phase = _phase_for_step(phases, step)
        batch_cases, batch_counts = _sample_training_batch(
            problem=problem,
            config=config,
            phase=phase,
            step=step,
            rng=rng,
            half_train_cases=half_train_cases,
            hard_replay_cases=_lookup_cases_by_id(half_validation_cases, hard_replay_case_ids),
        )
        gradient = _estimate_gradient(
            raw_params=raw_params,
            batch_cases=batch_cases,
            problem=problem,
            config=config,
            torch_generator=torch_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        raw_params.grad = gradient
        optimizer.step()

        current_weights = (
            bounded_memory_strengths(
                raw_params.detach(),
                memory_min=config.memory_min,
                memory_max=config.memory_max,
            )
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        train_metrics = evaluate_memory_weight_vector(
            problem=problem,
            cases=batch_cases,
            memory_strengths=current_weights,
            max_iter=config.max_iter,
            alpha=config.alpha,
            logical_weight_penalty=config.logical_weight_penalty,
            convergence_penalty=config.convergence_penalty,
            iteration_penalty=config.iteration_penalty,
        )
        _append_csv_row(
            train_metrics_path,
            {
                "step": int(step + 1),
                "phase": phase.name,
                "batch_half_cases": int(batch_counts["half"]),
                "batch_replay_cases": int(batch_counts["replay"]),
                "batch_random_cases": int(batch_counts["random"]),
                **train_metrics,
            },
        )

        should_validate = ((step + 1) % config.validation_interval == 0) or (step + 1 == total_steps)
        if should_validate:
            half_validation = evaluate_memory_weight_vector(
                problem=problem,
                cases=half_validation_cases,
                memory_strengths=current_weights,
                max_iter=config.max_iter,
                alpha=config.alpha,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
                return_case_rows=True,
            )
            random_validation = evaluate_memory_weight_vector(
                problem=problem,
                cases=random_validation_cases,
                memory_strengths=current_weights,
                max_iter=config.max_iter,
                alpha=config.alpha,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            )
            selection_loss = float(
                random_validation["loss_mean"]
                + float(config.selection_half_weight) * half_validation["loss_mean"]
            )
            hard_replay_case_ids = [
                int(row["case_id"])
                for row in sorted(
                    half_validation["case_rows"],
                    key=lambda row: (row["logical_success"], -row["loss"], -row["logical_weight"]),
                )
                if not row["logical_success"]
            ][: config.max_hard_replay_cases]
            validation_row = {
                "step": int(step + 1),
                "phase": phase.name,
                "selection_loss": selection_loss,
                "hard_replay_pool_size": int(len(hard_replay_case_ids)),
                "half_validation_loss_mean": float(half_validation["loss_mean"]),
                "half_validation_logical_success_rate": float(half_validation["logical_success_rate"]),
                "half_validation_exact_recovery_rate": float(half_validation["exact_recovery_rate"]),
                "half_validation_convergence_rate": float(half_validation["convergence_rate"]),
                "half_validation_mean_iterations": float(half_validation["mean_iterations"]),
                "half_validation_mean_logical_weight": float(half_validation["mean_logical_weight"]),
                "random_validation_loss_mean": float(random_validation["loss_mean"]),
                "random_validation_logical_success_rate": float(random_validation["logical_success_rate"]),
                "random_validation_exact_recovery_rate": float(random_validation["exact_recovery_rate"]),
                "random_validation_convergence_rate": float(random_validation["convergence_rate"]),
                "random_validation_mean_iterations": float(random_validation["mean_iterations"]),
                "random_validation_mean_logical_weight": float(random_validation["mean_logical_weight"]),
                "half_validation_bp_logical_success_rate": float(
                    baseline_metrics["half_validation"]["bp"]["logical_success_rate"]
                ),
                "half_validation_same_mem_logical_success_rate": float(
                    baseline_metrics["half_validation"]["same_memory"]["logical_success_rate"]
                ),
                "random_validation_bp_logical_success_rate": float(
                    baseline_metrics["random_validation"]["bp"]["logical_success_rate"]
                ),
                "random_validation_same_mem_logical_success_rate": float(
                    baseline_metrics["random_validation"]["same_memory"]["logical_success_rate"]
                ),
            }
            _append_csv_row(validation_metrics_path, validation_row)
            np.save(checkpoint_dir / "latest_weights.npy", current_weights)
            if _is_better_validation_row(validation_row, best_validation_row):
                best_selection_loss = selection_loss
                best_validation_row = dict(validation_row)
                np.save(checkpoint_dir / "best_weights.npy", current_weights)
                _write_json(
                    checkpoint_dir / "best_validation.json",
                    validation_row,
                )
            _save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                raw_params=raw_params.detach(),
                optimizer=optimizer,
                completed_steps=step + 1,
                best_selection_loss=best_selection_loss,
                hard_replay_case_ids=hard_replay_case_ids,
            )
        else:
            _save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                raw_params=raw_params.detach(),
                optimizer=optimizer,
                completed_steps=step + 1,
                best_selection_loss=best_selection_loss,
                hard_replay_case_ids=hard_replay_case_ids,
            )

    summary = _finalize_training_artifacts(
        raw_params=raw_params,
        checkpoint_dir=checkpoint_dir,
        config=config,
        problem=problem,
        completed_steps=total_steps,
        best_selection_loss=best_selection_loss,
        resumed=bool(completed_steps),
    )
    summary["baseline_metrics"] = baseline_metrics
    return summary


def _case_loss(
    *,
    logical_weight: int,
    converged: bool,
    iterations: int,
    max_iter: int,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
) -> float:
    iteration_term = float(iterations) / max(1, int(max_iter))
    return (
        float(logical_weight_penalty) * float(logical_weight)
        + float(convergence_penalty) * float(not converged)
        + float(iteration_penalty) * iteration_term
    )


def _build_half_case_splits(
    problem: CodeCapacityProblem,
    config: StaticMemoryTrainingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = enumerate_half_stabilizer_cases(
        problem,
        shuffle_seed=config.base_seed,
    )
    if not cases:
        raise ValueError("No half-stabilizer cases were generated from the provided code.")
    if len(cases) == 1:
        return cases, cases
    train_count = min(int(config.half_train_limit), max(1, len(cases) - 1))
    validation_count = min(int(config.half_validation_limit), max(1, len(cases) - train_count))
    priority_cases = _priority_half_cases(
        problem=problem,
        cases=cases,
        config=config,
    )
    half_train_cases, half_validation_cases = _sample_half_case_split(
        cases=cases,
        priority_cases=priority_cases,
        train_count=train_count,
        validation_count=validation_count,
        base_seed=config.base_seed,
    )
    if not half_validation_cases:
        split_point = max(1, len(half_train_cases) // 4)
        half_validation_cases = list(half_train_cases[-split_point:])
        half_train_cases = list(half_train_cases[:-split_point] or half_train_cases)
    return half_train_cases, half_validation_cases


def _compute_baseline_metrics(
    *,
    problem: CodeCapacityProblem,
    config: StaticMemoryTrainingConfig,
    half_validation_cases: Sequence[dict[str, Any]],
    random_validation_cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    same_memory = np.full(problem.n_bits, float(config.baseline_memory_value), dtype=np.float64)
    return {
        "half_validation": {
            "bp": evaluate_memory_weight_vector(
                problem=problem,
                cases=half_validation_cases,
                memory_strengths=None,
                max_iter=config.max_iter,
                alpha=config.alpha,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            ),
            "same_memory": evaluate_memory_weight_vector(
                problem=problem,
                cases=half_validation_cases,
                memory_strengths=same_memory,
                max_iter=config.max_iter,
                alpha=config.alpha,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            ),
        },
        "random_validation": {
            "bp": evaluate_memory_weight_vector(
                problem=problem,
                cases=random_validation_cases,
                memory_strengths=None,
                max_iter=config.max_iter,
                alpha=config.alpha,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            ),
            "same_memory": evaluate_memory_weight_vector(
                problem=problem,
                cases=random_validation_cases,
                memory_strengths=same_memory,
                max_iter=config.max_iter,
                alpha=config.alpha,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            ),
        },
    }


def _initialize_raw_params(
    *,
    problem: CodeCapacityProblem,
    config: StaticMemoryTrainingConfig,
    checkpoint_dir: Path,
    half_validation_cases: Sequence[dict[str, Any]],
    random_validation_cases: Sequence[dict[str, Any]],
) -> Any:
    torch = require_torch()
    mode = str(config.initialization_mode).strip().lower()
    if mode in {"", "gaussian"}:
        return _gaussian_raw_params(problem=problem, config=config)
    if mode not in {"interval_best", "interval_topk_average"}:
        raise ValueError(f"Unsupported initialization_mode: {config.initialization_mode!r}")

    candidates: list[tuple[str, np.ndarray]] = [
        (
            "interval_center",
            np.full(
                problem.n_bits,
                (float(config.memory_min) + float(config.memory_max)) / 2.0,
                dtype=np.float64,
            ),
        )
    ]
    for candidate_idx in range(max(0, int(config.initialization_num_candidates))):
        candidates.append(
            (
                f"random_{candidate_idx}",
                _sample_interval_memory_strengths(
                    n_bits=problem.n_bits,
                    low=float(config.memory_min),
                    high=float(config.memory_max),
                    seed=int(config.base_seed) + 900_000 + candidate_idx,
                ),
            )
        )

    candidate_rows: list[dict[str, Any]] = []
    candidate_vectors: list[np.ndarray] = []
    candidate_losses: list[float] = []
    for label, candidate in candidates:
        half_metrics = evaluate_memory_weight_vector(
            problem=problem,
            cases=half_validation_cases,
            memory_strengths=candidate,
            max_iter=config.max_iter,
            alpha=config.alpha,
            logical_weight_penalty=config.logical_weight_penalty,
            convergence_penalty=config.convergence_penalty,
            iteration_penalty=config.iteration_penalty,
        )
        random_metrics = evaluate_memory_weight_vector(
            problem=problem,
            cases=random_validation_cases,
            memory_strengths=candidate,
            max_iter=config.max_iter,
            alpha=config.alpha,
            logical_weight_penalty=config.logical_weight_penalty,
            convergence_penalty=config.convergence_penalty,
            iteration_penalty=config.iteration_penalty,
        )
        selection_loss = float(
            random_metrics["loss_mean"] + float(config.selection_half_weight) * half_metrics["loss_mean"]
        )
        candidate_rows.append(
            {
                "label": label,
                "selection_loss": selection_loss,
                "half_validation_loss_mean": float(half_metrics["loss_mean"]),
                "half_validation_logical_success_rate": float(half_metrics["logical_success_rate"]),
                "half_validation_convergence_rate": float(half_metrics["convergence_rate"]),
                "half_validation_mean_iterations": float(half_metrics["mean_iterations"]),
                "random_validation_loss_mean": float(random_metrics["loss_mean"]),
                "random_validation_logical_success_rate": float(random_metrics["logical_success_rate"]),
                "random_validation_convergence_rate": float(random_metrics["convergence_rate"]),
                "random_validation_mean_iterations": float(random_metrics["mean_iterations"]),
                "weight_min": float(np.min(candidate)),
                "weight_max": float(np.max(candidate)),
                "weight_mean": float(np.mean(candidate)),
                "weight_std": float(np.std(candidate)),
            }
        )
        candidate_vectors.append(candidate)
        candidate_losses.append(selection_loss)

    best_index = int(np.argmin(np.asarray(candidate_losses, dtype=np.float64)))
    if mode == "interval_best":
        selected_memory_strengths = candidate_vectors[best_index]
        selection_mode = "best_candidate"
    else:
        selected_memory_strengths = average_top_candidate_memory_strengths(
            candidate_vectors,
            candidate_losses,
            topk=config.initialization_topk,
        )
        selection_mode = f"topk_average_{max(1, min(int(config.initialization_topk), len(candidate_vectors)))}"

    selected_half_metrics = evaluate_memory_weight_vector(
        problem=problem,
        cases=half_validation_cases,
        memory_strengths=selected_memory_strengths,
        max_iter=config.max_iter,
        alpha=config.alpha,
        logical_weight_penalty=config.logical_weight_penalty,
        convergence_penalty=config.convergence_penalty,
        iteration_penalty=config.iteration_penalty,
    )
    selected_random_metrics = evaluate_memory_weight_vector(
        problem=problem,
        cases=random_validation_cases,
        memory_strengths=selected_memory_strengths,
        max_iter=config.max_iter,
        alpha=config.alpha,
        logical_weight_penalty=config.logical_weight_penalty,
        convergence_penalty=config.convergence_penalty,
        iteration_penalty=config.iteration_penalty,
    )
    _write_json(
        checkpoint_dir / "initialization_summary.json",
        {
            "initialization_mode": mode,
            "selection_mode": selection_mode,
            "candidate_count": int(len(candidate_rows)),
            "selection_half_weight": float(config.selection_half_weight),
            "memory_interval": {
                "low": float(config.memory_min),
                "high": float(config.memory_max),
            },
            "best_candidate_label": str(candidate_rows[best_index]["label"]),
            "best_candidate_selection_loss": float(candidate_rows[best_index]["selection_loss"]),
            "selected_weight_min": float(np.min(selected_memory_strengths)),
            "selected_weight_max": float(np.max(selected_memory_strengths)),
            "selected_weight_mean": float(np.mean(selected_memory_strengths)),
            "selected_weight_std": float(np.std(selected_memory_strengths)),
            "selected_half_validation": selected_half_metrics,
            "selected_random_validation": selected_random_metrics,
            "candidates": candidate_rows,
        },
    )
    raw_params = torch.from_numpy(
        raw_memory_parameters(
            selected_memory_strengths,
            memory_min=config.memory_min,
            memory_max=config.memory_max,
        )
    ).to(dtype=torch.float64)
    raw_params.requires_grad_(True)
    return raw_params


def _gaussian_raw_params(
    *,
    problem: CodeCapacityProblem,
    config: StaticMemoryTrainingConfig,
) -> Any:
    torch = require_torch()
    raw_params = torch.zeros(problem.n_bits, dtype=torch.float64, requires_grad=True)
    if float(config.initial_raw_std) > 0.0:
        torch.manual_seed(int(config.base_seed))
        raw_params = (
            float(config.initial_raw_std) * torch.randn(problem.n_bits, dtype=torch.float64)
        ).requires_grad_(True)
    return raw_params


def _sample_interval_memory_strengths(
    *,
    n_bits: int,
    low: float,
    high: float,
    seed: int,
) -> np.ndarray:
    low = float(low)
    high = float(high)
    if high < low:
        raise ValueError(f"Interval high must be >= low; got ({low}, {high}).")
    if np.isclose(low, high):
        return np.full(int(n_bits), low, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    return rng.uniform(low, high, size=int(n_bits)).astype(np.float64)


def _load_or_init_state(
    *,
    problem: CodeCapacityProblem,
    config: StaticMemoryTrainingConfig,
    checkpoint_dir: Path,
    half_validation_cases: Sequence[dict[str, Any]],
    random_validation_cases: Sequence[dict[str, Any]],
) -> tuple[Any, Any, int, float, list[int], dict[str, Any] | None]:
    torch = require_torch()
    checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
    best_validation_path = checkpoint_dir / "best_validation.json"
    best_validation_row = (
        json.loads(best_validation_path.read_text(encoding="utf-8"))
        if best_validation_path.exists()
        else None
    )
    if not config.resume or not checkpoint_path.exists():
        raw_params = _initialize_raw_params(
            problem=problem,
            config=config,
            checkpoint_dir=checkpoint_dir,
            half_validation_cases=half_validation_cases,
            random_validation_cases=random_validation_cases,
        )
        optimizer = torch.optim.Adam(
            [raw_params],
            lr=float(config.learning_rate),
            betas=(float(config.adam_beta1), float(config.adam_beta2)),
            weight_decay=float(config.weight_decay),
        )
        return raw_params, optimizer, 0, float("inf"), [], None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_params = checkpoint["raw_params"].detach().clone().to(dtype=torch.float64)
    raw_params.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [raw_params],
        lr=float(config.learning_rate),
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        weight_decay=float(config.weight_decay),
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        raw_params,
        optimizer,
        int(checkpoint.get("completed_steps", 0)),
        float(checkpoint.get("best_selection_loss", float("inf"))),
        [int(case_id) for case_id in checkpoint.get("hard_replay_case_ids", [])],
        best_validation_row,
    )


def _case_identifier(case: dict[str, Any]) -> int:
    return int(case.get("case_id", case.get("sample_idx", -1)))


def _case_logical_weight(problem: CodeCapacityProblem, case: dict[str, Any]) -> int:
    error = np.asarray(case["error"], dtype=np.uint8)
    logical_signature = np.asarray((problem.lz @ error) % 2, dtype=np.uint8).reshape(-1)
    return int(logical_signature.sum())


def _sample_without_replacement(
    cases: Sequence[dict[str, Any]],
    count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if count <= 0 or not cases:
        return []
    if count >= len(cases):
        return list(cases)
    indices = rng.choice(len(cases), size=int(count), replace=False)
    return [cases[int(index)] for index in indices]


def _sample_half_case_split(
    *,
    cases: Sequence[dict[str, Any]],
    priority_cases: Sequence[dict[str, Any]],
    train_count: int,
    validation_count: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(base_seed))
    all_cases = list(cases)
    train_cases: list[dict[str, Any]] = []
    validation_cases: list[dict[str, Any]] = []
    used_case_ids: set[int] = set()

    if validation_count > 0 and priority_cases:
        chosen = priority_cases[0]
        validation_cases.append(chosen)
        used_case_ids.add(_case_identifier(chosen))
    remaining_priority_cases = [
        case for case in priority_cases if _case_identifier(case) not in used_case_ids
    ]
    if train_count > 0 and remaining_priority_cases:
        chosen = remaining_priority_cases[0]
        train_cases.append(chosen)
        used_case_ids.add(_case_identifier(chosen))

    remaining_cases = [
        case for case in all_cases if _case_identifier(case) not in used_case_ids
    ]
    validation_cases.extend(
        _sample_without_replacement(
            remaining_cases,
            validation_count - len(validation_cases),
            rng,
        )
    )
    used_case_ids.update(_case_identifier(case) for case in validation_cases)
    remaining_cases = [
        case for case in all_cases if _case_identifier(case) not in used_case_ids
    ]
    train_cases.extend(
        _sample_without_replacement(
            remaining_cases,
            train_count - len(train_cases),
            rng,
        )
    )
    return train_cases, validation_cases


def _priority_half_cases(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    config: StaticMemoryTrainingConfig,
) -> list[dict[str, Any]]:
    case_by_id = {_case_identifier(case): case for case in cases}
    raw_logical_weight_by_id = {
        case_id: _case_logical_weight(problem, case)
        for case_id, case in case_by_id.items()
    }
    baseline_rows = evaluate_memory_weight_vector(
        problem=problem,
        cases=cases,
        memory_strengths=None,
        max_iter=config.max_iter,
        alpha=config.alpha,
        logical_weight_penalty=config.logical_weight_penalty,
        convergence_penalty=config.convergence_penalty,
        iteration_penalty=config.iteration_penalty,
        return_case_rows=True,
    )["case_rows"]

    def is_priority_row(row: dict[str, Any]) -> bool:
        case_id = int(row["case_id"])
        return (
            raw_logical_weight_by_id.get(case_id, 0) > 0
            or not bool(row["logical_success"])
            or not bool(row["converged"])
            or not bool(row["exact_recovery"])
        )

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        case_id = int(row["case_id"])
        raw_logical_weight = raw_logical_weight_by_id.get(case_id, 0)
        return (
            bool(row["logical_success"]),
            bool(row["converged"]),
            -int(row["iterations"]),
            raw_logical_weight == 0,
            bool(row["exact_recovery"]),
            -raw_logical_weight,
            -int(row["logical_weight"]),
            case_id,
        )

    return [
        case_by_id[int(row["case_id"])]
        for row in sorted(
            (row for row in baseline_rows if is_priority_row(row)),
            key=sort_key,
        )
        if int(row["case_id"]) in case_by_id
    ]


def _validation_tie_break_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["random_validation_convergence_rate"]),
        -float(row["random_validation_mean_iterations"]),
        float(row["random_validation_logical_success_rate"]),
        float(row["random_validation_exact_recovery_rate"]),
        -float(row["random_validation_mean_logical_weight"]),
        float(row["half_validation_convergence_rate"]),
        -float(row["half_validation_mean_iterations"]),
        float(row["half_validation_logical_success_rate"]),
        float(row["half_validation_exact_recovery_rate"]),
        -float(row["half_validation_mean_logical_weight"]),
    )


def _is_better_validation_row(
    candidate: dict[str, Any],
    current_best: dict[str, Any] | None,
    *,
    tolerance: float = 1e-12,
) -> bool:
    if current_best is None:
        return True
    candidate_loss = float(candidate["selection_loss"])
    current_loss = float(current_best["selection_loss"])
    if candidate_loss < current_loss - tolerance:
        return True
    if candidate_loss > current_loss + tolerance:
        return False
    return _validation_tie_break_key(candidate) > _validation_tie_break_key(current_best)


def _sample_training_batch(
    *,
    problem: CodeCapacityProblem,
    config: StaticMemoryTrainingConfig,
    phase: CurriculumPhase,
    step: int,
    rng: np.random.Generator,
    half_train_cases: Sequence[dict[str, Any]],
    hard_replay_cases: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    batch_size = int(config.batch_size)
    replay_count = min(
        int(round(batch_size * max(0.0, min(1.0, phase.replay_fraction)))),
        len(hard_replay_cases),
    )
    half_count = int(round(batch_size * max(0.0, min(1.0, phase.half_fraction))))
    half_count = max(half_count, replay_count)
    fresh_half_count = max(0, half_count - replay_count)
    random_count = max(0, batch_size - fresh_half_count - replay_count)

    batch_cases: list[dict[str, Any]] = []
    batch_cases.extend(_sample_with_replacement(half_train_cases, fresh_half_count, rng))
    batch_cases.extend(_sample_with_replacement(hard_replay_cases, replay_count, rng))
    if random_count:
        batch_cases.extend(
            sample_random_error_cases(
                problem,
                num_samples=random_count,
                error_rate=config.train_random_error_rate,
                base_seed=config.base_seed + 100_000 + (1_000 * int(step)),
            )
        )
    rng.shuffle(batch_cases)
    return batch_cases, {
        "half": int(fresh_half_count),
        "replay": int(replay_count),
        "random": int(random_count),
    }


def _sample_with_replacement(
    cases: Sequence[dict[str, Any]],
    count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if count <= 0 or not cases:
        return []
    indices = rng.integers(0, len(cases), size=int(count))
    return [cases[int(index)] for index in indices]


def _estimate_gradient(
    *,
    raw_params: Any,
    batch_cases: Sequence[dict[str, Any]],
    problem: CodeCapacityProblem,
    config: StaticMemoryTrainingConfig,
    torch_generator: Any,
) -> Any:
    torch = require_torch()
    gradient = torch.zeros_like(raw_params)
    noise_std = float(config.noise_std)
    for _ in range(int(config.perturbations_per_step)):
        epsilon = torch.randn(raw_params.shape, generator=torch_generator, dtype=raw_params.dtype)
        plus_weights = (
            bounded_memory_strengths(
                raw_params.detach() + noise_std * epsilon,
                memory_min=config.memory_min,
                memory_max=config.memory_max,
            )
            .cpu()
            .numpy()
        )
        minus_weights = (
            bounded_memory_strengths(
                raw_params.detach() - noise_std * epsilon,
                memory_min=config.memory_min,
                memory_max=config.memory_max,
            )
            .cpu()
            .numpy()
        )
        plus_metrics = evaluate_memory_weight_vector(
            problem=problem,
            cases=batch_cases,
            memory_strengths=plus_weights,
            max_iter=config.max_iter,
            alpha=config.alpha,
            logical_weight_penalty=config.logical_weight_penalty,
            convergence_penalty=config.convergence_penalty,
            iteration_penalty=config.iteration_penalty,
        )
        minus_metrics = evaluate_memory_weight_vector(
            problem=problem,
            cases=batch_cases,
            memory_strengths=minus_weights,
            max_iter=config.max_iter,
            alpha=config.alpha,
            logical_weight_penalty=config.logical_weight_penalty,
            convergence_penalty=config.convergence_penalty,
            iteration_penalty=config.iteration_penalty,
        )
        gradient = gradient + ((plus_metrics["loss_mean"] - minus_metrics["loss_mean"]) / (2.0 * noise_std)) * epsilon
    gradient = gradient / max(1, int(config.perturbations_per_step))
    return gradient


def _phase_for_step(phases: Sequence[CurriculumPhase], step: int) -> CurriculumPhase:
    running = 0
    for phase in phases:
        running += int(phase.steps)
        if step < running:
            return phase
    return phases[-1]


def _lookup_cases_by_id(cases: Sequence[dict[str, Any]], case_ids: Sequence[int]) -> list[dict[str, Any]]:
    by_id = {
        int(case.get("case_id", case.get("sample_idx", -1))): case
        for case in cases
    }
    return [by_id[case_id] for case_id in case_ids if case_id in by_id]


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    raw_params: Any,
    optimizer: Any,
    completed_steps: int,
    best_selection_loss: float,
    hard_replay_case_ids: Sequence[int],
) -> None:
    torch = require_torch()
    torch.save(
        {
            "raw_params": raw_params.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "completed_steps": int(completed_steps),
            "best_selection_loss": float(best_selection_loss),
            "hard_replay_case_ids": [int(case_id) for case_id in hard_replay_case_ids],
        },
        checkpoint_dir / "checkpoint_latest.pt",
    )


def _finalize_training_artifacts(
    *,
    raw_params: Any,
    checkpoint_dir: Path,
    config: StaticMemoryTrainingConfig,
    problem: CodeCapacityProblem,
    completed_steps: int,
    best_selection_loss: float,
    resumed: bool,
) -> dict[str, Any]:
    final_weights = (
        bounded_memory_strengths(
            raw_params.detach(),
            memory_min=config.memory_min,
            memory_max=config.memory_max,
        )
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    np.save(checkpoint_dir / "learned_weights.npy", final_weights)
    summary = {
        "code_path": str(problem.path),
        "checkpoint_dir": str(checkpoint_dir),
        "completed_steps": int(completed_steps),
        "best_selection_loss": float(best_selection_loss),
        "resumed": bool(resumed),
        "train_metrics_csv": str(checkpoint_dir / "train_metrics.csv"),
        "validation_metrics_csv": str(checkpoint_dir / "validation_metrics.csv"),
        "latest_checkpoint": str(checkpoint_dir / "checkpoint_latest.pt"),
        "learned_weights_path": str(checkpoint_dir / "learned_weights.npy"),
        "best_weights_path": str(checkpoint_dir / "best_weights.npy"),
    }
    _write_json(checkpoint_dir / "training_summary.json", summary)
    return summary


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _config_payload(problem: CodeCapacityProblem, config: StaticMemoryTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["n_bits"] = int(problem.n_bits)
    payload["n_logicals"] = int(problem.n_logicals)
    payload["metadata"] = problem.metadata
    return payload
