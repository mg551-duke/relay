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
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import csv

import numpy as np
import pytest
from scipy.sparse import csc_matrix

import scripts.run_code_capacity_static_assignment_campaign as assignment_campaign
from relay_bp.analysis import (
    StaticAssignmentBankConfig,
    decode_with_edge_damping,
    decode_with_memory_and_edge_damping,
    decode_with_memory_strengths,
    decode_with_plain_bp,
    enumerate_half_stabilizer_cases,
    make_min_sum_tracer,
    sample_random_error_cases,
    train_assignment_bank_cem,
)
from relay_bp.analysis.code_capacity_assignment_bank import (
    _bank_diversity_penalty,
    _select_assignment_row,
)
from relay_bp.analysis.code_capacity_common import CodeCapacityProblem


def _toy_problem() -> CodeCapacityProblem:
    hx = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    hz_dense = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    lz = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    return CodeCapacityProblem(
        path=Path("toy_assignment_bank_code.npz"),
        hx=hx,
        hz=csc_matrix(hz_dense),
        lz=lz,
        error_priors=np.full(4, 0.1, dtype=np.float64),
        metadata={"name": "toy-assignment-bank"},
    )


def _write_toy_npz(path: Path) -> None:
    problem = _toy_problem()
    np.savez(
        path,
        Hx=problem.hx,
        Hz=problem.hz.toarray(),
        Lz=problem.lz,
        error_priors=problem.error_priors,
    )


def _result_signature(result) -> tuple[bool, int, tuple[int, ...]]:
    return (
        bool(result.success),
        int(result.iterations),
        tuple(np.asarray(result.decoding, dtype=np.uint8).tolist()),
    )


def _memory_support() -> dict[str, float]:
    return {
        "low": -0.15,
        "high": 0.20,
        "mean": 0.05,
        "sigma": 0.04,
    }


def _damping_support() -> dict[str, float]:
    return {
        "low": 0.35,
        "high": 0.95,
        "mean": 0.70,
        "sigma": 0.08,
    }


def _tiny_training_summary(
    *,
    problem: CodeCapacityProblem,
    family: str,
    checkpoint_dir: Path,
    base_seed: int,
) -> dict[str, object]:
    half_cases = enumerate_half_stabilizer_cases(problem, max_cases=2, shuffle_seed=3)
    random_train_cases = sample_random_error_cases(problem, num_samples=4, error_rate=0.1, base_seed=17)
    random_validation_cases = sample_random_error_cases(problem, num_samples=3, error_rate=0.1, base_seed=29)
    config = StaticAssignmentBankConfig(
        code_path=str(problem.path),
        checkpoint_dir=str(checkpoint_dir),
        family=family,
        bank_size=2,
        max_iter=6,
        alpha=1.0,
        random_train_samples=4,
        random_validation_samples=3,
        random_test_samples=4,
        random_train_batch_size=3,
        candidate_count=4,
        elite_count=2,
        rounds=2,
        validation_interval=1,
        practical_k=2,
        initialization_candidates=4,
        base_seed=base_seed,
    )
    return train_assignment_bank_cem(
        problem,
        config=config,
        half_cases=half_cases,
        random_train_cases=random_train_cases,
        random_validation_cases=random_validation_cases,
        memory_support=None if family == "damping" else _memory_support(),
        damping_support=None if family == "memory" else _damping_support(),
    )


def test_joint_decode_helper_matches_plain_memory_and_damping_wrappers():
    problem = _toy_problem()
    case = enumerate_half_stabilizer_cases(problem, max_cases=1, shuffle_seed=1)[0]
    memory = np.array([-0.10, 0.20, 0.05, -0.02], dtype=np.float64)
    damping = np.linspace(0.4, 0.9, problem.hz.nnz, dtype=np.float64)

    plain_result = decode_with_plain_bp(
        make_min_sum_tracer(problem, max_iter=6, alpha=1.0, gamma0=0.0),
        case["syndrome"],
        n_bits=problem.n_bits,
    )
    joint_plain_result = decode_with_memory_and_edge_damping(
        make_min_sum_tracer(problem, max_iter=6, alpha=1.0, gamma0=0.0),
        case["syndrome"],
        n_bits=problem.n_bits,
        memory_strengths=None,
        edge_damping_messages=None,
    )
    assert _result_signature(plain_result) == _result_signature(joint_plain_result)

    memory_result = decode_with_memory_strengths(
        make_min_sum_tracer(problem, max_iter=6, alpha=1.0, gamma0=0.0),
        case["syndrome"],
        memory,
    )
    joint_memory_result = decode_with_memory_and_edge_damping(
        make_min_sum_tracer(problem, max_iter=6, alpha=1.0, gamma0=0.0),
        case["syndrome"],
        n_bits=problem.n_bits,
        memory_strengths=memory,
        edge_damping_messages=None,
    )
    assert _result_signature(memory_result) == _result_signature(joint_memory_result)

    damping_result = decode_with_edge_damping(
        make_min_sum_tracer(problem, max_iter=6, alpha=1.0, gamma0=None),
        case["syndrome"],
        damping,
        n_bits=problem.n_bits,
    )
    joint_damping_result = decode_with_memory_and_edge_damping(
        make_min_sum_tracer(problem, max_iter=6, alpha=1.0, gamma0=None),
        case["syndrome"],
        n_bits=problem.n_bits,
        memory_strengths=None,
        edge_damping_messages=damping,
    )
    assert _result_signature(damping_result) == _result_signature(joint_damping_result)


def test_assignment_row_selection_matches_practical_and_oracle_priority_order():
    practical_choice = _select_assignment_row(
        [
            {
                "bank_idx": 0,
                "residual_syndrome_weight": 1,
                "converged": True,
                "iterations": 1,
                "decoding_weight": 1,
                "logical_success": True,
                "logical_weight": 0,
            },
            {
                "bank_idx": 1,
                "residual_syndrome_weight": 0,
                "converged": False,
                "iterations": 1,
                "decoding_weight": 1,
                "logical_success": False,
                "logical_weight": 1,
            },
            {
                "bank_idx": 2,
                "residual_syndrome_weight": 0,
                "converged": True,
                "iterations": 5,
                "decoding_weight": 1,
                "logical_success": True,
                "logical_weight": 0,
            },
            {
                "bank_idx": 3,
                "residual_syndrome_weight": 0,
                "converged": True,
                "iterations": 3,
                "decoding_weight": 5,
                "logical_success": True,
                "logical_weight": 0,
            },
            {
                "bank_idx": 4,
                "residual_syndrome_weight": 0,
                "converged": True,
                "iterations": 3,
                "decoding_weight": 1,
                "logical_success": True,
                "logical_weight": 0,
            },
        ],
        selection_mode="practical",
    )
    assert int(practical_choice["bank_idx"]) == 4

    oracle_choice = _select_assignment_row(
        [
            {"bank_idx": 0, "logical_success": False, "logical_weight": 0, "iterations": 1},
            {"bank_idx": 1, "logical_success": True, "logical_weight": 2, "iterations": 1},
            {"bank_idx": 2, "logical_success": True, "logical_weight": 1, "iterations": 5},
            {"bank_idx": 3, "logical_success": True, "logical_weight": 1, "iterations": 3},
        ],
        selection_mode="oracle",
    )
    assert int(oracle_choice["bank_idx"]) == 3


def test_duplicate_bank_penalty_flags_collapsed_members():
    problem = _toy_problem()
    config = StaticAssignmentBankConfig(
        code_path=str(problem.path),
        checkpoint_dir=str(Path.cwd() / ".tmp" / "unused-assignment-bank-diversity"),
        family="memory",
        bank_size=2,
        candidate_count=1,
        elite_count=1,
        rounds=1,
        validation_interval=1,
        initialization_candidates=1,
    )
    duplicate_penalty = _bank_diversity_penalty(
        family="memory",
        memory_bank=np.zeros((2, problem.n_bits), dtype=np.float64),
        damping_bank=None,
        config=config,
    )
    separated_penalty = _bank_diversity_penalty(
        family="memory",
        memory_bank=np.array(
            [
                np.full(problem.n_bits, -0.3, dtype=np.float64),
                np.full(problem.n_bits, 0.3, dtype=np.float64),
            ]
        ),
        damping_bank=None,
        config=config,
    )
    assert duplicate_penalty > 0.0
    assert separated_penalty == pytest.approx(0.0)


@pytest.mark.parametrize("family", ["memory", "damping", "joint"])
def test_assignment_bank_training_writes_bounded_reproducible_artifacts(family: str):
    problem = _toy_problem()
    base_tmp = Path.cwd() / ".tmp" / "pytest-assignment-bank-training"
    base_tmp.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=base_tmp) as first_tmp, TemporaryDirectory(dir=base_tmp) as second_tmp:
        first_summary = _tiny_training_summary(
            problem=problem,
            family=family,
            checkpoint_dir=Path(first_tmp) / family,
            base_seed=7,
        )
        second_summary = _tiny_training_summary(
            problem=problem,
            family=family,
            checkpoint_dir=Path(second_tmp) / family,
            base_seed=7,
        )

        checkpoint_dir = Path(first_summary["checkpoint_dir"])
        assert (checkpoint_dir / "config.json").exists()
        assert (checkpoint_dir / "initialization_summary.json").exists()
        assert (checkpoint_dir / "checkpoint_latest.npz").exists()
        assert (checkpoint_dir / "train_metrics.csv").exists()
        assert (checkpoint_dir / "validation_metrics.csv").exists()
        assert (checkpoint_dir / "training_summary.json").exists()

        if family in {"memory", "joint"}:
            first_memory = np.asarray(first_summary["best_memory_bank"], dtype=np.float64)
            second_memory = np.asarray(second_summary["best_memory_bank"], dtype=np.float64)
            assert np.all(first_memory >= -0.3 - 1e-12)
            assert np.all(first_memory <= 0.3 + 1e-12)
            assert first_memory == pytest.approx(second_memory)

        if family in {"damping", "joint"}:
            first_damping = np.asarray(first_summary["best_damping_bank"], dtype=np.float64)
            second_damping = np.asarray(second_summary["best_damping_bank"], dtype=np.float64)
            assert np.all(first_damping >= -1e-12)
            assert np.all(first_damping <= 1.0 + 1e-12)
            assert first_damping == pytest.approx(second_damping)


@pytest.mark.parametrize("family", ["memory", "damping", "joint"])
@pytest.mark.parametrize("bank_size", [1, 2])
def test_static_assignment_campaign_runner_smoke(family: str, bank_size: int):
    base_tmp = Path.cwd() / ".tmp" / "pytest-assignment-bank-runner"
    base_tmp.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=base_tmp) as tmp_dir:
        tmp_path = Path(tmp_dir)
        code_path = tmp_path / "toy_assignment_bank_code.npz"
        _write_toy_npz(code_path)
        args = SimpleNamespace(
            code_label="toy_raw",
            code_path=code_path,
            p_value=0.1,
            family=family,
            bank_size=bank_size,
            restart_index=0,
            output_dir=tmp_path / "runs",
            center_values=[0.0, 0.2],
            width_values=[0.2],
            gaussian_refinement_points=3,
            scan_random_cases=2,
            random_train_samples=4,
            random_validation_samples=2,
            random_test_samples=4,
            candidate_count=2,
            elite_count=1,
            rounds=1,
            validation_interval=1,
            practical_k=2,
            comparison_repeats=2,
            bootstrap_samples=64,
            max_iter=5,
            alpha=1.0,
            selection_half_weight=1.0,
            logical_failure_penalty=20.0,
            logical_weight_penalty=4.0,
            convergence_penalty=1.0,
            iteration_penalty=0.1,
            diversity_penalty=0.01,
            base_seed=11,
        )

        summary = assignment_campaign.run_training_unit(args)
        run_dir = Path(summary["run_dir"])

        assert summary["family"] == family
        assert int(summary["bank_size"]) == bank_size
        assert (run_dir / "run_summary.json").exists()
        assert (run_dir / "support_scan.json").exists()
        assert (run_dir / "random_test_comparison.json").exists()
        assert (run_dir / "half_comparison.json").exists()
        assert (run_dir / "bank_member_diagnostics.json").exists()
        assert (run_dir / "training" / "checkpoint_latest.npz").exists()


def test_write_manifest_smoke_uses_frozen_p_grid(monkeypatch: pytest.MonkeyPatch):
    base_tmp = Path.cwd() / ".tmp" / "pytest-assignment-bank-manifest"
    base_tmp.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=base_tmp) as tmp_dir:
        tmp_path = Path(tmp_dir)
        code_path = tmp_path / "toy_assignment_bank_code.npz"
        _write_toy_npz(code_path)
        monkeypatch.setattr(
            assignment_campaign,
            "default_raw_code_paths",
            lambda: {"surface5_raw": code_path.resolve()},
        )
        manifest_path = tmp_path / "manifest.csv"
        assignment_campaign.write_manifest(
            SimpleNamespace(
                output_path=manifest_path,
                families="memory,damping",
                bank_sizes=[1],
                restarts=1,
                code_labels="surface5_raw",
                output_root=tmp_path / "campaign_runs",
            )
        )

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == len(assignment_campaign.FROZEN_P_GRIDS["surface5_raw"]) * 2
        assert {row["family"] for row in rows} == {"memory", "damping"}
        assert {row["code_label"] for row in rows} == {"surface5_raw"}
