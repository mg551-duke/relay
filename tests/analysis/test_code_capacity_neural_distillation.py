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

import numpy as np
import pytest
from scipy.sparse import csc_matrix

torch = pytest.importorskip("torch")

import scripts.run_code_capacity_neural_static_campaign as neural_campaign
import scripts.train_code_capacity_neural_static_bank as neural_train_script
from relay_bp.analysis import (
    CompiledTannerGraph,
    NeuralStaticDistillationConfig,
    enumerate_half_stabilizer_cases,
    load_code_capacity_problem,
    sample_random_error_cases,
    train_neural_static_bank,
    with_uniform_error_rate,
)
from relay_bp.analysis.code_capacity_common import CodeCapacityProblem
from relay_bp.analysis.code_capacity_neural_distillation import (
    _compiled_to_torch,
    _damping_bank_from_logits,
    _memory_bank_from_logits,
    _soft_selector_weights,
    _soft_zero_parity_bce,
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
        path=Path("toy_neural_distillation_code.npz"),
        hx=hx,
        hz=hz,
        lz=lz,
        error_priors=np.full(4, 0.1, dtype=np.float64),
        metadata={"name": "toy-neural-distillation"},
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


def _tiny_neural_summary(
    *,
    problem: CodeCapacityProblem,
    family: str,
    checkpoint_dir: Path,
    base_seed: int,
    bank_size: int = 2,
) -> dict[str, object]:
    half_cases = enumerate_half_stabilizer_cases(problem, max_cases=2, shuffle_seed=3)
    random_train_cases = sample_random_error_cases(problem, num_samples=4, error_rate=0.1, base_seed=17)
    random_validation_cases = sample_random_error_cases(problem, num_samples=3, error_rate=0.1, base_seed=29)
    random_test_cases = sample_random_error_cases(problem, num_samples=3, error_rate=0.1, base_seed=41)
    config = NeuralStaticDistillationConfig(
        code_path=str(problem.path),
        checkpoint_dir=str(checkpoint_dir),
        family=family,
        bank_size=bank_size,
        max_iter=6,
        alpha=1.0,
        random_train_batch_size=2,
        warmup_steps=1,
        mixed_steps=1,
        finetune_steps=1,
        consolidation_steps=1,
        validation_interval=1,
        initialization_candidates=2,
        initialization_subset_random_cases=2,
        initialization_subset_half_cases=2,
        support_scan_random_cases=2,
        support_centers=(0.1,),
        support_widths=(0.2,),
        gaussian_refinement_points=3,
        exact_polish_candidate_count=4,
        exact_polish_elite_count=2,
        exact_polish_rounds=2,
        exact_polish_validation_interval=1,
        exact_polish_random_train_batch_size=2,
        exact_polish_initialization_candidates=1,
        bootstrap_samples=100,
        comparison_repeats=2,
        base_seed=base_seed,
        run_exact_baseline=True,
        device="cpu",
    )
    return train_neural_static_bank(
        problem,
        config=config,
        half_cases=half_cases,
        random_train_cases=random_train_cases,
        random_validation_cases=random_validation_cases,
        random_test_cases=random_test_cases,
        memory_support=None if family == "damping" else _memory_support(),
        damping_support=None if family == "memory" else _damping_support(),
    )


def test_compiled_tanner_graph_matches_problem_structure():
    problem = _toy_problem()
    compiled = CompiledTannerGraph.from_problem(problem)

    assert compiled.num_checks == 3
    assert compiled.num_bits == 4
    assert compiled.num_edges == int(problem.hz.nnz)
    assert compiled.num_logicals == 1
    assert compiled.edge_to_bit.tolist() == [0, 1, 1, 2, 2, 3]
    assert compiled.edge_to_check.tolist() == [0, 0, 1, 1, 2, 2]
    assert compiled.check_to_bit.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert compiled.logical_to_bit.tolist() == [[0, 2]]


def test_soft_zero_parity_bce_vanishes_on_hard_even_parity():
    problem = _toy_problem()
    compiled = CompiledTannerGraph.from_problem(problem)
    torch_graph = _compiled_to_torch(compiled, torch=torch, device=torch.device("cpu"))
    residual_prob = torch.ones((1, 1, problem.n_bits), dtype=torch.float64)
    even_loss = _soft_zero_parity_bce(
        residual_prob,
        torch_graph.check_to_bit,
        torch_graph.check_to_bit_mask,
        torch=torch,
    )
    odd_residual_prob = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]], dtype=torch.float64)
    odd_loss = _soft_zero_parity_bce(
        odd_residual_prob,
        torch_graph.logical_to_bit,
        torch_graph.logical_to_bit_mask,
        torch=torch,
    )

    assert float(torch.max(torch.abs(even_loss)).item()) < 1e-5
    assert float(torch.min(odd_loss).item()) > 1.0


def test_soft_selector_weights_harden_with_temperature():
    scores = torch.tensor([[0.1], [1.0]], dtype=torch.float64)
    smooth = _soft_selector_weights(scores, temperature=1.0, torch=torch)
    hard = _soft_selector_weights(scores, temperature=0.01, torch=torch)

    assert float(smooth[0, 0].item()) > 0.5
    assert float(hard[0, 0].item()) > 0.999
    assert float(hard[1, 0].item()) < 1e-6


def test_memory_and_damping_parameterizations_respect_bounds():
    config = NeuralStaticDistillationConfig(
        code_path="toy.npz",
        checkpoint_dir=str(Path.cwd() / ".tmp" / "unused-neural-bounds"),
        family="joint",
        bank_size=2,
        memory_min=-0.3,
        memory_max=0.3,
        damping_min=0.1,
        damping_max=0.9,
    )
    memory_logits = torch.nn.Parameter(torch.tensor([[-10.0, 0.0, 10.0]], dtype=torch.float64))
    damping_logits = torch.nn.Parameter(torch.tensor([[-10.0, 0.0, 10.0]], dtype=torch.float64))

    memory_bank = _memory_bank_from_logits(memory_logits, config, torch)
    damping_bank = _damping_bank_from_logits(damping_logits, config, torch)

    assert torch.all(memory_bank >= config.memory_min - 1e-12)
    assert torch.all(memory_bank <= config.memory_max + 1e-12)
    assert torch.all(damping_bank >= config.damping_min - 1e-12)
    assert torch.all(damping_bank <= config.damping_max + 1e-12)


@pytest.mark.parametrize("family", ["memory", "damping", "joint"])
def test_neural_static_bank_training_writes_bounded_reproducible_artifacts(family: str):
    problem = _toy_problem()
    base_tmp = Path.cwd() / ".tmp" / "pytest-neural-static-distillation"
    base_tmp.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=base_tmp) as first_tmp, TemporaryDirectory(dir=base_tmp) as second_tmp:
        first_summary = _tiny_neural_summary(
            problem=problem,
            family=family,
            checkpoint_dir=Path(first_tmp) / family,
            base_seed=7,
        )
        second_summary = _tiny_neural_summary(
            problem=problem,
            family=family,
            checkpoint_dir=Path(second_tmp) / family,
            base_seed=7,
        )

        checkpoint_dir = Path(first_summary["checkpoint_dir"])
        assert (checkpoint_dir / "config.json").exists()
        assert (checkpoint_dir / "support_scan.json").exists()
        assert (checkpoint_dir / "support_initialization_summary.json").exists()
        assert (checkpoint_dir / "surrogate_checkpoint_latest.npz").exists()
        assert (checkpoint_dir / "surrogate_train_metrics.csv").exists()
        assert (checkpoint_dir / "surrogate_validation_metrics.csv").exists()
        assert (checkpoint_dir / "teacher_diagnostics.json").exists()
        assert (checkpoint_dir / "training_summary.json").exists()

        if family in {"memory", "joint"}:
            first_bank = np.asarray(first_summary["best_memory_bank"], dtype=np.float64)
            second_bank = np.asarray(second_summary["best_memory_bank"], dtype=np.float64)
            assert first_bank.shape == (2, problem.n_bits)
            assert np.all(first_bank >= -0.3 - 1e-12)
            assert np.all(first_bank <= 0.3 + 1e-12)
            assert first_bank == pytest.approx(second_bank)
            assert (checkpoint_dir / "best_memory_bank.npy").exists()
            assert (checkpoint_dir / "best_distilled_memory_bank.npy").exists()
        if family in {"damping", "joint"}:
            first_bank = np.asarray(first_summary["best_damping_bank"], dtype=np.float64)
            second_bank = np.asarray(second_summary["best_damping_bank"], dtype=np.float64)
            assert first_bank.shape == (2, problem.hz.nnz)
            assert np.all(first_bank >= -1e-12)
            assert np.all(first_bank <= 1.0 + 1e-12)
            assert first_bank == pytest.approx(second_bank)
            assert (checkpoint_dir / "best_damping_bank.npy").exists()
            assert (checkpoint_dir / "best_distilled_damping_bank.npy").exists()
        assert "distilled_beats_exact_baseline_validation" in first_summary
        assert "practical_difference" in first_summary["random_test_comparison"]
        assert "oracle_difference" in first_summary["random_test_comparison"]
        assert "practical_difference" in first_summary["half_comparison"]
        assert "oracle_difference" in first_summary["half_comparison"]


def test_neural_campaign_run_writes_verdict_and_collapse_reports():
    problem = _toy_problem()
    base_tmp = Path.cwd() / ".tmp" / "pytest-neural-campaign"
    base_tmp.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=base_tmp) as tmp_dir:
        code_path = Path(tmp_dir) / "toy_code.npz"
        output_dir = Path(tmp_dir) / "runs"
        _write_toy_npz(code_path)
        args = SimpleNamespace(
            code_label="toy_raw",
            code_path=code_path,
            p_value=0.1,
            family="memory",
            bank_size=2,
            restart_index=0,
            output_dir=output_dir,
            max_iter=6,
            alpha=1.0,
            random_train_samples=4,
            random_validation_samples=3,
            random_test_samples=3,
            random_train_batch_size=2,
            warmup_steps=1,
            mixed_steps=1,
            finetune_steps=1,
            consolidation_steps=1,
            mixed_half_fraction=0.5,
            finetune_half_fraction=0.25,
            learning_rate=0.02,
            adam_beta1=0.9,
            adam_beta2=0.999,
            weight_decay=0.0,
            selector_temperature_start=1.0,
            selector_temperature_end=0.05,
            selector_temperature_consolidation=0.02,
            logical_loss_weight=6.0,
            syndrome_loss_weight=3.0,
            iteration_trace_weight=1.0,
            confidence_loss_weight=0.05,
            teacher_regularization_weight=0.01,
            teacher_default_gamma=1.0,
            teacher_default_delta=1.0,
            validation_interval=1,
            initialization_candidates=2,
            initialization_subset_random_cases=2,
            initialization_subset_half_cases=2,
            selection_half_weight=1.0,
            diversity_penalty=0.01,
            diversity_threshold_fraction=0.05,
            logical_failure_penalty=20.0,
            logical_weight_penalty=4.0,
            convergence_penalty=1.0,
            iteration_penalty=0.1,
            practical_k=2,
            bootstrap_samples=100,
            comparison_repeats=2,
            support_scan_random_cases=2,
            center_values=[0.1],
            width_values=[0.2],
            gaussian_refinement_points=3,
            exact_polish_candidate_count=4,
            exact_polish_elite_count=2,
            exact_polish_rounds=2,
            exact_polish_validation_interval=1,
            exact_polish_random_train_batch_size=2,
            exact_polish_initialization_candidates=1,
            exact_polish_memory_sigma=None,
            exact_polish_damping_sigma=None,
            exact_polish_sigma_floor=1e-3,
            exact_polish_memory_sigma_ceiling=None,
            exact_polish_damping_sigma_ceiling=None,
            no_exact_baseline=False,
            device="cpu",
            base_seed=7,
        )
        summary = neural_campaign.run_campaign_unit(args)
        run_dir = Path(summary["run_dir"])

        assert summary["verdict"] in {"success", "near_equal", "negative_conclusion"}
        assert "collapse_analysis" in summary
        assert (run_dir / "run_summary.json").exists()
        assert (run_dir / "bank_member_diagnostics.json").exists()
        assert (run_dir / "collapse_analysis.json").exists()


@pytest.mark.parametrize(
    ("code_label", "p_value"),
    [("surface5_raw", 0.11), ("surface13_raw", 0.03)],
)
def test_raw_code_smoke_runs_if_local_code_is_available(code_label: str, p_value: float):
    try:
        code_path = neural_campaign.resolve_code_path(code_label, None)
    except ValueError:
        pytest.skip(f"No local code path configured for {code_label}.")
    if not code_path.exists():
        pytest.skip(f"Missing local code file for {code_label}: {code_path}")

    base_problem = load_code_capacity_problem(code_path)
    problem = with_uniform_error_rate(base_problem, float(p_value))
    half_cases = enumerate_half_stabilizer_cases(problem, max_cases=2, shuffle_seed=1)
    random_train_cases = sample_random_error_cases(problem, num_samples=2, error_rate=float(p_value), base_seed=11)
    random_validation_cases = sample_random_error_cases(problem, num_samples=2, error_rate=float(p_value), base_seed=13)
    random_test_cases = sample_random_error_cases(problem, num_samples=2, error_rate=float(p_value), base_seed=17)
    base_tmp = Path.cwd() / ".tmp" / "pytest-neural-raw-smoke"
    base_tmp.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=base_tmp) as tmp_dir:
        config = NeuralStaticDistillationConfig(
            code_path=str(code_path),
            checkpoint_dir=str(Path(tmp_dir) / code_label),
            family="memory",
            bank_size=1,
            max_iter=6,
            alpha=1.0,
            random_train_batch_size=2,
            warmup_steps=1,
            mixed_steps=0,
            finetune_steps=0,
            consolidation_steps=0,
            validation_interval=1,
            initialization_candidates=1,
            initialization_subset_random_cases=2,
            initialization_subset_half_cases=2,
            support_scan_random_cases=2,
            support_centers=(0.1,),
            support_widths=(0.2,),
            gaussian_refinement_points=1,
            exact_polish_candidate_count=2,
            exact_polish_elite_count=1,
            exact_polish_rounds=1,
            exact_polish_validation_interval=1,
            exact_polish_random_train_batch_size=2,
            exact_polish_initialization_candidates=1,
            bootstrap_samples=20,
            comparison_repeats=1,
            run_exact_baseline=False,
            device="cpu",
            base_seed=19,
        )
        summary = train_neural_static_bank(
            problem,
            config=config,
            half_cases=half_cases,
            random_train_cases=random_train_cases,
            random_validation_cases=random_validation_cases,
            random_test_cases=random_test_cases,
            memory_support=_memory_support(),
        )

        assert np.asarray(summary["best_memory_bank"]).shape == (1, problem.n_bits)
        assert "random_test_comparison" in summary
