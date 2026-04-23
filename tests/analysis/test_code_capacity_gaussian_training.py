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

from relay_bp.analysis.code_capacity_common import CodeCapacityProblem
from relay_bp.analysis.code_capacity_gaussian_training import (
    GaussianCEMTrainingConfig,
    _fit_elite_distribution,
    clipped_mean_vector,
    train_gaussian_memory_distribution,
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
        path=Path("toy_gaussian_training_code.npz"),
        hx=hx,
        hz=hz,
        lz=lz,
        error_priors=np.full(4, 0.1, dtype=np.float64),
        metadata={"name": "toy-gaussian-training"},
    )


def test_clipped_mean_vector_respects_bounds():
    clipped = clipped_mean_vector(np.array([-2.0, -0.1, 0.4]), low=-0.3, high=0.2)
    assert clipped == pytest.approx(np.array([-0.3, -0.1, 0.2]))


def test_fit_elite_distribution_moves_toward_low_loss_region():
    elite_mean, elite_sigma, elite_indices = _fit_elite_distribution(
        [
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([10.0, 10.0], dtype=np.float64),
        ],
        [0.1, 0.2, 5.0],
        elite_count=2,
    )

    assert elite_indices.tolist() == [0, 1]
    assert 0.0 < elite_mean[0] < 1.0
    assert elite_sigma > 0.0
    assert elite_sigma < 1.0


def test_gaussian_training_writes_bounded_reproducible_artifacts():
    problem = _toy_problem()
    base_tmp = Path.cwd() / ".tmp" / "pytest-gaussian-memory-training"
    base_tmp.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=base_tmp) as first_tmp, tempfile.TemporaryDirectory(dir=base_tmp) as second_tmp:
        config_kwargs = {
            "code_path": str(problem.path),
            "max_iter": 6,
            "alpha": 1.0,
            "half_train_limit": 4,
            "half_validation_limit": 2,
            "random_validation_samples": 3,
            "batch_size": 3,
            "warmup_steps": 1,
            "mixed_steps": 1,
            "finetune_steps": 1,
            "initialization_num_candidates": 4,
            "initialization_topk": 2,
            "candidate_count": 4,
            "elite_count": 2,
            "validation_interval": 1,
            "validation_distribution_draws": 1,
            "base_seed": 7,
            "resume": False,
        }
        first_config = GaussianCEMTrainingConfig(
            checkpoint_dir=str(Path(first_tmp) / "gaussian_memory_training"),
            **config_kwargs,
        )
        second_config = GaussianCEMTrainingConfig(
            checkpoint_dir=str(Path(second_tmp) / "gaussian_memory_training"),
            **config_kwargs,
        )

        first_summary = train_gaussian_memory_distribution(problem, config=first_config)
        second_summary = train_gaussian_memory_distribution(problem, config=second_config)

        first_dir = Path(first_config.checkpoint_dir)
        assert first_summary["completed_steps"] == 3
        assert (first_dir / "config.json").exists()
        assert (first_dir / "baseline_metrics.json").exists()
        assert (first_dir / "checkpoint_latest.npz").exists()
        assert (first_dir / "distribution.json").exists()
        assert (first_dir / "distribution_mean.npy").exists()
        assert (first_dir / "train_metrics.csv").exists()
        assert (first_dir / "validation_metrics.csv").exists()

        first_mean = np.load(first_dir / "distribution_mean.npy")
        second_mean = np.load(Path(second_config.checkpoint_dir) / "distribution_mean.npy")
        assert np.all(first_mean >= first_config.memory_min - 1e-12)
        assert np.all(first_mean <= first_config.memory_max + 1e-12)
        assert first_mean == pytest.approx(second_mean)
