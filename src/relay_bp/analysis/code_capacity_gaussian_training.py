from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .code_capacity_common import CodeCapacityProblem, sample_random_error_cases
from .code_capacity_memory import MemorySamplingSpec, evaluate_sampling_strategy, sample_memory_strengths
from .code_capacity_training import (
    CurriculumPhase,
    _append_csv_row,
    _build_half_case_splits,
    _compute_baseline_metrics,
    _is_better_validation_row,
    _lookup_cases_by_id,
    _phase_for_step,
    _sample_training_batch,
    _write_json,
    average_top_candidate_memory_strengths,
    evaluate_memory_weight_vector,
)


@dataclass(frozen=True)
class GaussianCEMTrainingConfig:
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
    warmup_steps: int = 20
    mixed_steps: int = 20
    finetune_steps: int = 20
    mixed_half_fraction: float = 0.5
    finetune_half_fraction: float = 0.2
    finetune_replay_fraction: float = 0.3
    selection_half_weight: float = 0.75
    logical_weight_penalty: float = 4.0
    convergence_penalty: float = 1.0
    iteration_penalty: float = 0.05
    max_hard_replay_cases: int = 32
    memory_min: float = -0.3
    memory_max: float = 0.3
    baseline_memory_value: float = 0.15
    initialization_low: float | None = None
    initialization_high: float | None = None
    initialization_num_candidates: int = 32
    initialization_topk: int = 8
    initial_sigma: float | None = None
    sigma_floor: float = 0.01
    sigma_ceiling: float | None = None
    candidate_count: int = 64
    elite_count: int = 8
    smoothing: float = 0.2
    validation_interval: int = 5
    validation_distribution_draws: int = 8
    base_seed: int = 0
    resume: bool = True

    @property
    def total_steps(self) -> int:
        return int(self.warmup_steps + self.mixed_steps + self.finetune_steps)


def clipped_mean_vector(mean_vector: np.ndarray, *, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(mean_vector, dtype=np.float64), float(low), float(high)).astype(np.float64)


def train_gaussian_memory_distribution(
    problem: CodeCapacityProblem,
    *,
    config: GaussianCEMTrainingConfig,
) -> dict[str, Any]:
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
        mean_vector,
        sigma,
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
            checkpoint_dir=checkpoint_dir,
            config=config,
            problem=problem,
            completed_steps=completed_steps,
            best_selection_loss=best_selection_loss,
            resumed=True,
            mean_vector=mean_vector,
            sigma=sigma,
        )
        summary["baseline_metrics"] = baseline_metrics
        return summary

    rng = np.random.default_rng(config.base_seed + completed_steps)
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
        candidate_vectors, candidate_losses, candidate_metrics = _evaluate_candidate_batch(
            problem=problem,
            config=config,
            mean_vector=mean_vector,
            sigma=sigma,
            cases=batch_cases,
            step=step,
        )
        elite_mean, elite_sigma, elite_indices = _fit_elite_distribution(
            candidate_vectors,
            candidate_losses,
            elite_count=config.elite_count,
        )
        rho = max(0.0, min(1.0, float(config.smoothing)))
        low = float(config.memory_min)
        high = float(config.memory_max)
        mean_vector = clipped_mean_vector(
            (1.0 - rho) * mean_vector + rho * elite_mean,
            low=low,
            high=high,
        )
        sigma = _clip_sigma(
            (1.0 - rho) * float(sigma) + rho * float(elite_sigma),
            config=config,
        )

        representative_metrics = evaluate_memory_weight_vector(
            problem=problem,
            cases=batch_cases,
            memory_strengths=mean_vector,
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
                "candidate_loss_min": float(np.min(candidate_losses)),
                "candidate_loss_mean": float(np.mean(candidate_losses)),
                "candidate_loss_max": float(np.max(candidate_losses)),
                "candidate_logical_success_rate_max": float(
                    max(float(metrics["logical_success_rate"]) for metrics in candidate_metrics)
                ),
                "elite_count": int(len(elite_indices)),
                "sigma": float(sigma),
                "mean_weight_min": float(np.min(mean_vector)),
                "mean_weight_max": float(np.max(mean_vector)),
                "mean_weight_std": float(np.std(mean_vector)),
                "representative_loss_mean": float(representative_metrics["loss_mean"]),
                "representative_logical_success_rate": float(representative_metrics["logical_success_rate"]),
                "representative_convergence_rate": float(representative_metrics["convergence_rate"]),
                "representative_mean_iterations": float(representative_metrics["mean_iterations"]),
            },
        )

        should_validate = ((step + 1) % config.validation_interval == 0) or (step + 1 == total_steps)
        if should_validate:
            distribution_half = _evaluate_distribution(
                problem=problem,
                cases=half_validation_cases,
                config=config,
                mean_vector=mean_vector,
                sigma=sigma,
                draw_count=config.validation_distribution_draws,
                base_seed=config.base_seed + 700_000 + 10_000 * int(step),
                return_case_rows=True,
            )
            distribution_random = _evaluate_distribution(
                problem=problem,
                cases=random_validation_cases,
                config=config,
                mean_vector=mean_vector,
                sigma=sigma,
                draw_count=config.validation_distribution_draws,
                base_seed=config.base_seed + 900_000 + 10_000 * int(step),
            )
            representative_half = evaluate_memory_weight_vector(
                problem=problem,
                cases=half_validation_cases,
                memory_strengths=mean_vector,
                max_iter=config.max_iter,
                alpha=config.alpha,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            )
            representative_random = evaluate_memory_weight_vector(
                problem=problem,
                cases=random_validation_cases,
                memory_strengths=mean_vector,
                max_iter=config.max_iter,
                alpha=config.alpha,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            )
            selection_loss = float(
                distribution_random["loss_mean"]
                + float(config.selection_half_weight) * distribution_half["loss_mean"]
            )
            hard_replay_case_ids = [
                int(row["case_id"])
                for row in sorted(
                    distribution_half["case_rows"],
                    key=lambda row: (
                        bool(row["logical_success"]),
                        -float(row["loss"]),
                        -int(row["logical_weight"]),
                    ),
                )
                if not bool(row["logical_success"])
            ][: int(config.max_hard_replay_cases)]
            validation_row = {
                "step": int(step + 1),
                "phase": phase.name,
                "selection_loss": selection_loss,
                "hard_replay_pool_size": int(len(hard_replay_case_ids)),
                "sigma": float(sigma),
                "half_validation_loss_mean": float(distribution_half["loss_mean"]),
                "half_validation_logical_success_rate": float(distribution_half["logical_success_rate"]),
                "half_validation_exact_recovery_rate": float(distribution_half["exact_recovery_rate"]),
                "half_validation_convergence_rate": float(distribution_half["convergence_rate"]),
                "half_validation_mean_iterations": float(distribution_half["mean_iterations"]),
                "half_validation_mean_logical_weight": float(distribution_half["mean_logical_weight"]),
                "random_validation_loss_mean": float(distribution_random["loss_mean"]),
                "random_validation_logical_success_rate": float(distribution_random["logical_success_rate"]),
                "random_validation_exact_recovery_rate": float(distribution_random["exact_recovery_rate"]),
                "random_validation_convergence_rate": float(distribution_random["convergence_rate"]),
                "random_validation_mean_iterations": float(distribution_random["mean_iterations"]),
                "random_validation_mean_logical_weight": float(distribution_random["mean_logical_weight"]),
                "representative_half_validation_loss_mean": float(representative_half["loss_mean"]),
                "representative_half_validation_logical_success_rate": float(representative_half["logical_success_rate"]),
                "representative_half_validation_exact_recovery_rate": float(representative_half["exact_recovery_rate"]),
                "representative_half_validation_convergence_rate": float(representative_half["convergence_rate"]),
                "representative_half_validation_mean_iterations": float(representative_half["mean_iterations"]),
                "representative_half_validation_mean_logical_weight": float(representative_half["mean_logical_weight"]),
                "representative_random_validation_loss_mean": float(representative_random["loss_mean"]),
                "representative_random_validation_logical_success_rate": float(representative_random["logical_success_rate"]),
                "representative_random_validation_exact_recovery_rate": float(representative_random["exact_recovery_rate"]),
                "representative_random_validation_convergence_rate": float(representative_random["convergence_rate"]),
                "representative_random_validation_mean_iterations": float(representative_random["mean_iterations"]),
                "representative_random_validation_mean_logical_weight": float(representative_random["mean_logical_weight"]),
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
            np.save(checkpoint_dir / "latest_distribution_mean.npy", mean_vector)
            _write_json(
                checkpoint_dir / "latest_distribution.json",
                _distribution_payload(checkpoint_dir / "latest_distribution_mean.npy", mean_vector, sigma, config),
            )
            if _is_better_validation_row(validation_row, best_validation_row):
                best_selection_loss = selection_loss
                best_validation_row = dict(validation_row)
                np.save(checkpoint_dir / "best_distribution_mean.npy", mean_vector)
                _write_json(
                    checkpoint_dir / "best_distribution.json",
                    _distribution_payload(checkpoint_dir / "best_distribution_mean.npy", mean_vector, sigma, config),
                )
                _write_json(checkpoint_dir / "best_validation.json", validation_row)
            _save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                mean_vector=mean_vector,
                sigma=sigma,
                completed_steps=step + 1,
                best_selection_loss=best_selection_loss,
                hard_replay_case_ids=hard_replay_case_ids,
            )
        else:
            _save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                mean_vector=mean_vector,
                sigma=sigma,
                completed_steps=step + 1,
                best_selection_loss=best_selection_loss,
                hard_replay_case_ids=hard_replay_case_ids,
            )

    summary = _finalize_training_artifacts(
        checkpoint_dir=checkpoint_dir,
        config=config,
        problem=problem,
        completed_steps=total_steps,
        best_selection_loss=best_selection_loss,
        resumed=bool(completed_steps),
        mean_vector=mean_vector,
        sigma=sigma,
    )
    summary["baseline_metrics"] = baseline_metrics
    return summary


def _fit_elite_distribution(
    candidate_vectors: Sequence[np.ndarray],
    candidate_losses: Sequence[float],
    *,
    elite_count: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    if not candidate_vectors:
        raise ValueError("candidate_vectors must not be empty.")
    losses = np.asarray(candidate_losses, dtype=np.float64)
    if losses.shape != (len(candidate_vectors),):
        raise ValueError("candidate_losses must match candidate_vectors length.")
    elite_count = max(1, min(int(elite_count), len(candidate_vectors)))
    elite_indices = np.argsort(losses)[:elite_count]
    elite_vectors = [np.asarray(candidate_vectors[int(index)], dtype=np.float64) for index in elite_indices]
    elite_losses = [float(losses[int(index)]) for index in elite_indices]
    elite_mean = average_top_candidate_memory_strengths(
        elite_vectors,
        elite_losses,
        topk=elite_count,
    )
    elite_matrix = np.asarray(elite_vectors, dtype=np.float64)
    elite_sigma = float(np.sqrt(np.mean((elite_matrix - elite_mean[None, :]) ** 2)))
    return elite_mean, elite_sigma, elite_indices.astype(np.int64)


def _evaluate_candidate_batch(
    *,
    problem: CodeCapacityProblem,
    config: GaussianCEMTrainingConfig,
    mean_vector: np.ndarray,
    sigma: float,
    cases: Sequence[dict[str, Any]],
    step: int,
) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, Any]]]:
    candidate_vectors: list[np.ndarray] = []
    candidate_losses: list[float] = []
    candidate_metrics: list[dict[str, Any]] = []
    base_seed = int(config.base_seed) + 1_000_000 + 100_000 * int(step)
    for candidate_idx in range(int(config.candidate_count)):
        vector = _sample_gaussian_candidate(
            problem=problem,
            config=config,
            mean_vector=mean_vector,
            sigma=sigma,
            seed=base_seed + candidate_idx,
        )
        metrics = evaluate_memory_weight_vector(
            problem=problem,
            cases=cases,
            memory_strengths=vector,
            max_iter=config.max_iter,
            alpha=config.alpha,
            logical_weight_penalty=config.logical_weight_penalty,
            convergence_penalty=config.convergence_penalty,
            iteration_penalty=config.iteration_penalty,
        )
        candidate_vectors.append(vector)
        candidate_losses.append(float(metrics["loss_mean"]))
        candidate_metrics.append(metrics)
    return candidate_vectors, np.asarray(candidate_losses, dtype=np.float64), candidate_metrics


def _sample_gaussian_candidate(
    *,
    problem: CodeCapacityProblem,
    config: GaussianCEMTrainingConfig,
    mean_vector: np.ndarray,
    sigma: float,
    seed: int,
) -> np.ndarray:
    sampling_spec = MemorySamplingSpec(
        sampling_family="truncated_gaussian",
        application_scope="all_bits",
        selection_mode="single_draw",
        draw_count=1,
        low=float(config.memory_min),
        high=float(config.memory_max),
        mean_vector=np.asarray(mean_vector, dtype=np.float64),
        sigma=float(sigma),
    )
    return sample_memory_strengths(
        problem=problem,
        case={"support_bits": tuple(range(problem.n_bits))},
        sampling_spec=sampling_spec,
        seed=int(seed),
    )


def _evaluate_distribution(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    config: GaussianCEMTrainingConfig,
    mean_vector: np.ndarray,
    sigma: float,
    draw_count: int,
    base_seed: int,
    return_case_rows: bool = False,
) -> dict[str, Any]:
    repeats = max(1, int(draw_count))
    aggregate: dict[str, float] = {
        "loss_mean": 0.0,
        "logical_success_rate": 0.0,
        "exact_recovery_rate": 0.0,
        "convergence_rate": 0.0,
        "mean_iterations": 0.0,
        "mean_logical_weight": 0.0,
    }
    first_metrics: dict[str, Any] | None = None
    for repeat_idx in range(repeats):
        sampling_spec = MemorySamplingSpec(
            sampling_family="truncated_gaussian",
            application_scope="all_bits",
            selection_mode="single_draw",
            draw_count=1,
            low=float(config.memory_min),
            high=float(config.memory_max),
            mean_vector=np.asarray(mean_vector, dtype=np.float64),
            sigma=float(sigma),
        )
        metrics = evaluate_sampling_strategy(
            problem=problem,
            cases=cases,
            sampling_spec=sampling_spec,
            max_iter=config.max_iter,
            alpha=config.alpha,
            logical_weight_penalty=config.logical_weight_penalty,
            convergence_penalty=config.convergence_penalty,
            iteration_penalty=config.iteration_penalty,
            base_seed=int(base_seed) + 10_007 * repeat_idx,
            return_case_rows=return_case_rows and repeat_idx == 0,
        )
        if first_metrics is None:
            first_metrics = metrics
        for key in aggregate:
            aggregate[key] += float(metrics[key])
    assert first_metrics is not None
    result = {
        "num_cases": int(first_metrics["num_cases"]),
        **{key: float(value / repeats) for key, value in aggregate.items()},
    }
    if return_case_rows:
        result["case_rows"] = first_metrics["case_rows"]
    return result


def _load_or_init_state(
    *,
    problem: CodeCapacityProblem,
    config: GaussianCEMTrainingConfig,
    checkpoint_dir: Path,
    half_validation_cases: Sequence[dict[str, Any]],
    random_validation_cases: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, float, int, float, list[int], dict[str, Any] | None]:
    checkpoint_path = checkpoint_dir / "checkpoint_latest.npz"
    best_validation_path = checkpoint_dir / "best_validation.json"
    best_validation_row = (
        None if not best_validation_path.exists() else _load_json(best_validation_path)
    )
    if not config.resume or not checkpoint_path.exists():
        mean_vector, sigma = _initialize_distribution(
            problem=problem,
            config=config,
            checkpoint_dir=checkpoint_dir,
            half_validation_cases=half_validation_cases,
            random_validation_cases=random_validation_cases,
        )
        return mean_vector, sigma, 0, float("inf"), [], best_validation_row

    checkpoint = np.load(checkpoint_path, allow_pickle=False)
    mean_vector = checkpoint["mean_vector"].astype(np.float64)
    sigma = float(checkpoint["sigma"][0])
    completed_steps = int(checkpoint["completed_steps"][0])
    best_selection_loss = float(checkpoint["best_selection_loss"][0])
    hard_replay_case_ids = checkpoint["hard_replay_case_ids"].astype(np.int64).tolist()
    return mean_vector, sigma, completed_steps, best_selection_loss, hard_replay_case_ids, best_validation_row


def _initialize_distribution(
    *,
    problem: CodeCapacityProblem,
    config: GaussianCEMTrainingConfig,
    checkpoint_dir: Path,
    half_validation_cases: Sequence[dict[str, Any]],
    random_validation_cases: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, float]:
    init_low = float(config.memory_min if config.initialization_low is None else config.initialization_low)
    init_high = float(config.memory_max if config.initialization_high is None else config.initialization_high)
    if init_high < init_low:
        raise ValueError("initialization_high must be >= initialization_low.")
    uniform_spec = MemorySamplingSpec(
        sampling_family="uniform_interval",
        application_scope="all_bits",
        selection_mode="single_draw",
        draw_count=1,
        low=init_low,
        high=init_high,
    )
    candidates: list[tuple[str, np.ndarray]] = [
        (
            "interval_center",
            np.full(problem.n_bits, (init_low + init_high) / 2.0, dtype=np.float64),
        )
    ]
    for candidate_idx in range(max(0, int(config.initialization_num_candidates))):
        candidates.append(
            (
                f"random_{candidate_idx}",
                sample_memory_strengths(
                    problem=problem,
                    case={"support_bits": tuple(range(problem.n_bits))},
                    sampling_spec=uniform_spec,
                    seed=int(config.base_seed) + 200_000 + candidate_idx,
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
                "half_validation_logical_success_rate": float(half_metrics["logical_success_rate"]),
                "half_validation_convergence_rate": float(half_metrics["convergence_rate"]),
                "half_validation_mean_iterations": float(half_metrics["mean_iterations"]),
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

    elite_mean, elite_sigma, elite_indices = _fit_elite_distribution(
        candidate_vectors,
        candidate_losses,
        elite_count=config.initialization_topk,
    )
    interval_width = max(0.0, init_high - init_low)
    default_sigma = interval_width / 4.0
    sigma_seed = default_sigma if config.initial_sigma is None else float(config.initial_sigma)
    sigma = _clip_sigma(max(float(elite_sigma), float(sigma_seed)), config=config)
    mean_vector = clipped_mean_vector(elite_mean, low=float(config.memory_min), high=float(config.memory_max))

    _write_json(
        checkpoint_dir / "initialization_summary.json",
        {
            "memory_interval": {"low": init_low, "high": init_high},
            "candidate_count": int(len(candidate_rows)),
            "elite_count": int(len(elite_indices)),
            "selected_sigma": float(sigma),
            "selected_weight_min": float(np.min(mean_vector)),
            "selected_weight_max": float(np.max(mean_vector)),
            "selected_weight_mean": float(np.mean(mean_vector)),
            "selected_weight_std": float(np.std(mean_vector)),
            "candidates": candidate_rows,
        },
    )
    return mean_vector, sigma


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    mean_vector: np.ndarray,
    sigma: float,
    completed_steps: int,
    best_selection_loss: float,
    hard_replay_case_ids: Sequence[int],
) -> None:
    np.savez_compressed(
        checkpoint_dir / "checkpoint_latest.npz",
        mean_vector=np.asarray(mean_vector, dtype=np.float64),
        sigma=np.asarray([float(sigma)], dtype=np.float64),
        completed_steps=np.asarray([int(completed_steps)], dtype=np.int64),
        best_selection_loss=np.asarray([float(best_selection_loss)], dtype=np.float64),
        hard_replay_case_ids=np.asarray([int(case_id) for case_id in hard_replay_case_ids], dtype=np.int64),
    )


def _finalize_training_artifacts(
    *,
    checkpoint_dir: Path,
    config: GaussianCEMTrainingConfig,
    problem: CodeCapacityProblem,
    completed_steps: int,
    best_selection_loss: float,
    resumed: bool,
    mean_vector: np.ndarray,
    sigma: float,
) -> dict[str, Any]:
    mean_vector = clipped_mean_vector(mean_vector, low=float(config.memory_min), high=float(config.memory_max))
    final_mean_path = checkpoint_dir / "distribution_mean.npy"
    np.save(final_mean_path, mean_vector)
    distribution_path = checkpoint_dir / "distribution.json"
    _write_json(distribution_path, _distribution_payload(final_mean_path, mean_vector, sigma, config))
    summary = {
        "code_path": str(problem.path),
        "checkpoint_dir": str(checkpoint_dir),
        "completed_steps": int(completed_steps),
        "best_selection_loss": float(best_selection_loss),
        "resumed": bool(resumed),
        "train_metrics_csv": str(checkpoint_dir / "train_metrics.csv"),
        "validation_metrics_csv": str(checkpoint_dir / "validation_metrics.csv"),
        "latest_checkpoint": str(checkpoint_dir / "checkpoint_latest.npz"),
        "distribution_path": str(distribution_path),
        "distribution_mean_path": str(final_mean_path),
        "best_distribution_path": str(checkpoint_dir / "best_distribution.json"),
        "best_distribution_mean_path": str(checkpoint_dir / "best_distribution_mean.npy"),
    }
    _write_json(checkpoint_dir / "training_summary.json", summary)
    return summary


def _distribution_payload(
    mean_path: Path,
    mean_vector: np.ndarray,
    sigma: float,
    config: GaussianCEMTrainingConfig,
) -> dict[str, Any]:
    return {
        "sampling_family": "truncated_gaussian",
        "application_scope": "all_bits",
        "selection_mode": "single_draw",
        "low": float(config.memory_min),
        "high": float(config.memory_max),
        "sigma": float(sigma),
        "mean_weights_path": str(mean_path),
        "mean_weight_min": float(np.min(mean_vector)),
        "mean_weight_max": float(np.max(mean_vector)),
        "mean_weight_mean": float(np.mean(mean_vector)),
        "mean_weight_std": float(np.std(mean_vector)),
    }


def _clip_sigma(value: float, *, config: GaussianCEMTrainingConfig) -> float:
    ceiling = float(config.sigma_ceiling) if config.sigma_ceiling is not None else float(config.memory_max - config.memory_min)
    return float(np.clip(float(value), float(config.sigma_floor), ceiling))


def _config_payload(problem: CodeCapacityProblem, config: GaussianCEMTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["n_bits"] = int(problem.n_bits)
    payload["n_logicals"] = int(problem.n_logicals)
    payload["metadata"] = problem.metadata
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
