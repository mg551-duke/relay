# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from pathlib import Path
import tempfile

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from relay_bp.analysis.code_capacity_common import (
    CodeCapacityProblem,
    default_export_code_paths,
    load_code_capacity_problem,
    with_uniform_error_rate,
)

torch = pytest.importorskip("torch", reason="torch is required for static memory training tests")

from relay_bp.analysis.code_capacity_training import (  # noqa: E402
    StaticMemoryTrainingConfig,
    _build_half_case_splits,
    _case_logical_weight,
    _is_better_validation_row,
    average_top_candidate_memory_strengths,
    bounded_memory_strengths,
    evaluate_memory_weight_vector,
    raw_memory_parameters,
    train_static_memory_weights,
)


def _toy_problem() -> CodeCapacityProblem:
    hx = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    hz = csc_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )
    )
    lz = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    return CodeCapacityProblem(
        path=Path("toy_training_code.npz"),
        hx=hx,
        hz=hz,
        lz=lz,
        error_priors=np.full(4, 0.1, dtype=np.float64),
        metadata={"name": "toy-training"},
    )


def _memory_sensitive_problem() -> tuple[CodeCapacityProblem, dict[str, object], np.ndarray]:
    hx = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    hz = csc_matrix(
        np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.uint8,
        )
    )
    lz = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    error = np.array([1, 0, 0, 0], dtype=np.uint8)
    syndrome = np.asarray((hz.toarray() @ error) % 2, dtype=np.uint8)
    return (
        CodeCapacityProblem(
            path=Path("memory_sensitive_code.npz"),
            hx=hx,
            hz=hz,
            lz=lz,
            error_priors=np.full(4, 0.1, dtype=np.float64),
            metadata={"name": "memory-sensitive"},
        ),
        {"sample_idx": 0, "error": error, "syndrome": syndrome},
        np.array([-0.2495907938505691, 0.19958648859203865, 0.17225898449321003, -0.1563783342042287]),
    )


def _split_problem_with_rare_logicals() -> CodeCapacityProblem:
    hx = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    hz = csc_matrix(np.eye(4, dtype=np.uint8))
    lz = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    return CodeCapacityProblem(
        path=Path("split_problem.npz"),
        hx=hx,
        hz=hz,
        lz=lz,
        error_priors=np.full(4, 0.1, dtype=np.float64),
        metadata={"name": "split-problem"},
    )


def test_bounded_memory_strengths_stays_in_interval():
    raw = torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float64)
    bounded = bounded_memory_strengths(raw, memory_min=-0.3, memory_max=0.3).numpy()
    assert bounded[0] >= -0.3 - 1e-9
    assert bounded[1] == pytest.approx(0.0)
    assert bounded[2] <= 0.3 + 1e-9


def test_raw_memory_parameters_round_trip():
    original = np.array([-0.2, -0.05, 0.0, 0.17, 0.24], dtype=np.float64)
    raw = raw_memory_parameters(original, memory_min=-0.3, memory_max=0.3)
    rebuilt = bounded_memory_strengths(
        torch.tensor(raw, dtype=torch.float64),
        memory_min=-0.3,
        memory_max=0.3,
    ).numpy()
    assert rebuilt == pytest.approx(original)


def test_average_top_candidate_memory_strengths_biases_toward_lower_loss():
    candidates = [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([10.0, 10.0], dtype=np.float64),
    ]
    averaged = average_top_candidate_memory_strengths(candidates, [0.1, 0.2, 5.0], topk=2)
    assert averaged[0] > 0.0
    assert averaged[0] < 1.0
    assert averaged[0] < 0.6


def test_with_uniform_error_rate_replaces_priors_without_mutating_problem():
    problem = _toy_problem()
    updated = with_uniform_error_rate(problem, 0.23)
    assert updated is not problem
    assert np.allclose(updated.error_priors, 0.23)
    assert np.allclose(problem.error_priors, 0.1)
    assert updated.metadata["uniform_error_rate"] == pytest.approx(0.23)


def test_evaluate_memory_weight_vector_enables_memory_mode():
    problem, case, memory_strengths = _memory_sensitive_problem()
    common = dict(
        problem=problem,
        cases=[case],
        max_iter=8,
        alpha=1.0,
        logical_weight_penalty=4.0,
        convergence_penalty=1.0,
        iteration_penalty=0.05,
    )

    bp_metrics = evaluate_memory_weight_vector(memory_strengths=None, **common)
    mem_metrics = evaluate_memory_weight_vector(memory_strengths=memory_strengths, **common)

    assert bp_metrics["convergence_rate"] == pytest.approx(0.0)
    assert mem_metrics["convergence_rate"] == pytest.approx(1.0)
    assert mem_metrics["mean_iterations"] < bp_metrics["mean_iterations"]


def test_half_case_split_reserves_logical_nontrivial_cases_for_validation():
    problem = _split_problem_with_rare_logicals()
    config = StaticMemoryTrainingConfig(
        code_path=str(problem.path),
        checkpoint_dir=str(Path.cwd() / ".tmp" / "unused-split-checkpoint"),
        half_train_limit=4,
        half_validation_limit=2,
        random_validation_samples=1,
        warmup_steps=1,
        mixed_steps=1,
        finetune_steps=1,
        base_seed=7,
    )

    half_train_cases, half_validation_cases = _build_half_case_splits(problem, config)

    train_nontrivial = sum(_case_logical_weight(problem, case) > 0 for case in half_train_cases)
    validation_nontrivial = sum(_case_logical_weight(problem, case) > 0 for case in half_validation_cases)

    assert train_nontrivial >= 1
    assert validation_nontrivial >= 1


def test_surface13_half_validation_catches_bp_logical_failures():
    repo_root = Path(__file__).resolve().parents[2]
    problem = load_code_capacity_problem(default_export_code_paths(repo_root)["surface13"])
    config = StaticMemoryTrainingConfig(
        code_path=str(problem.path),
        checkpoint_dir=str(repo_root / ".tmp" / "unused-surface13-checkpoint"),
        max_iter=20,
        alpha=1.0,
        half_train_limit=12,
        half_validation_limit=6,
        logical_weight_penalty=4.0,
        convergence_penalty=1.0,
        iteration_penalty=0.05,
        base_seed=7,
    )

    _, half_validation_cases = _build_half_case_splits(problem, config)
    metrics = evaluate_memory_weight_vector(
        problem=problem,
        cases=half_validation_cases,
        memory_strengths=None,
        max_iter=config.max_iter,
        alpha=config.alpha,
        logical_weight_penalty=config.logical_weight_penalty,
        convergence_penalty=config.convergence_penalty,
        iteration_penalty=config.iteration_penalty,
    )

    assert metrics["logical_success_rate"] < 1.0
    assert metrics["convergence_rate"] < 1.0


def test_best_validation_tie_break_prefers_faster_result():
    current = {
        "selection_loss": 1.0,
        "random_validation_logical_success_rate": 0.8,
        "random_validation_exact_recovery_rate": 0.1,
        "random_validation_mean_logical_weight": 0.2,
        "random_validation_convergence_rate": 0.5,
        "random_validation_mean_iterations": 12.0,
        "half_validation_logical_success_rate": 1.0,
        "half_validation_exact_recovery_rate": 0.0,
        "half_validation_mean_logical_weight": 0.0,
        "half_validation_convergence_rate": 0.8,
        "half_validation_mean_iterations": 9.0,
    }
    candidate = dict(current)
    candidate["random_validation_mean_iterations"] = 8.0

    assert _is_better_validation_row(candidate, current)
    assert not _is_better_validation_row(current, candidate)


def test_static_memory_training_writes_artifacts_and_resumes():
    problem = _toy_problem()
    base_tmp = Path.cwd() / ".tmp" / "pytest-static-memory-training"
    base_tmp.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=base_tmp) as tmp_dir:
        checkpoint_dir = Path(tmp_dir) / "static_memory_training"
        config = StaticMemoryTrainingConfig(
            code_path=str(problem.path),
            checkpoint_dir=str(checkpoint_dir),
            max_iter=6,
            alpha=1.0,
            half_train_limit=4,
            half_validation_limit=2,
            random_validation_samples=3,
            batch_size=3,
            perturbations_per_step=1,
            noise_std=0.05,
            learning_rate=0.1,
            validation_interval=1,
            warmup_steps=1,
            mixed_steps=1,
            finetune_steps=1,
            base_seed=2,
            resume=True,
        )

        summary = train_static_memory_weights(problem, config=config)

        assert summary["completed_steps"] == 3
        assert (checkpoint_dir / "config.json").exists()
        assert (checkpoint_dir / "baseline_metrics.json").exists()
        assert (checkpoint_dir / "checkpoint_latest.pt").exists()
        assert (checkpoint_dir / "learned_weights.npy").exists()
        assert (checkpoint_dir / "train_metrics.csv").exists()
        assert (checkpoint_dir / "validation_metrics.csv").exists()

        resumed_config = StaticMemoryTrainingConfig(
            **{
                **config.__dict__,
                "warmup_steps": 1,
                "mixed_steps": 1,
                "finetune_steps": 2,
            }
        )
        resumed_summary = train_static_memory_weights(problem, config=resumed_config)
        assert resumed_summary["completed_steps"] == 4
