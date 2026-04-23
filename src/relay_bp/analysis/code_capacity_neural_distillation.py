from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .code_capacity_assignment_bank import (
    ScalarDistributionSpec,
    StaticAssignmentBankConfig,
    _evaluate_bank_validation,
    _is_better_bank_validation,
    compare_bank_to_random_baselines,
    evaluate_assignment_bank,
    scan_damping_distribution_support,
    scan_memory_distribution_support,
    train_assignment_bank_cem,
)
from .code_capacity_common import CodeCapacityProblem
from .code_capacity_training import _append_csv_row, _write_json, raw_memory_parameters, require_torch

FAMILY_CHOICES = {"memory", "damping", "joint"}


@dataclass(frozen=True)
class CompiledTannerGraph:
    num_checks: int
    num_bits: int
    num_edges: int
    num_logicals: int
    edge_to_check: np.ndarray
    edge_to_bit: np.ndarray
    check_to_edge: np.ndarray
    check_to_edge_mask: np.ndarray
    check_to_bit: np.ndarray
    check_to_bit_mask: np.ndarray
    bit_to_edge: np.ndarray
    bit_to_edge_mask: np.ndarray
    logical_to_bit: np.ndarray
    logical_to_bit_mask: np.ndarray
    log_prior_ratios: np.ndarray

    @classmethod
    def from_problem(cls, problem: CodeCapacityProblem) -> "CompiledTannerGraph":
        hz_csc = problem.hz.tocsc()
        hz_csr = problem.hz.tocsr()
        edge_to_check = np.asarray(hz_csc.indices, dtype=np.int64)
        edge_to_bit = np.empty(problem.hz.nnz, dtype=np.int64)
        for bit_idx in range(problem.n_bits):
            start = int(hz_csc.indptr[bit_idx])
            end = int(hz_csc.indptr[bit_idx + 1])
            edge_to_bit[start:end] = int(bit_idx)
        check_edge_lists = []
        check_bit_lists = []
        for check_idx in range(problem.hz.shape[0]):
            start = int(hz_csr.indptr[check_idx])
            end = int(hz_csr.indptr[check_idx + 1])
            bits = np.asarray(hz_csr.indices[start:end], dtype=np.int64)
            check_bit_lists.append(bits.tolist())
            edges = []
            for bit_idx in bits.tolist():
                bit_start = int(hz_csc.indptr[bit_idx])
                bit_end = int(hz_csc.indptr[bit_idx + 1])
                matches = np.flatnonzero(edge_to_check[bit_start:bit_end] == check_idx)
                if matches.size != 1:
                    raise ValueError(
                        f"Expected one Tanner edge for check={check_idx}, bit={bit_idx}; found {matches.size}."
                    )
                edges.append(int(bit_start + int(matches[0])))
            check_edge_lists.append(edges)
        bit_edge_lists = []
        for bit_idx in range(problem.n_bits):
            start = int(hz_csc.indptr[bit_idx])
            end = int(hz_csc.indptr[bit_idx + 1])
            bit_edge_lists.append(list(range(start, end)))
        logical_bit_lists = [
            np.flatnonzero(np.asarray(problem.lz[row_idx], dtype=np.uint8) % 2).astype(np.int64).tolist()
            for row_idx in range(problem.n_logicals)
        ]
        return cls(
            num_checks=int(problem.hz.shape[0]),
            num_bits=int(problem.n_bits),
            num_edges=int(problem.hz.nnz),
            num_logicals=int(problem.n_logicals),
            edge_to_check=edge_to_check,
            edge_to_bit=edge_to_bit,
            check_to_edge=_pad_index_rows(check_edge_lists),
            check_to_edge_mask=_pad_mask_rows(check_edge_lists),
            check_to_bit=_pad_index_rows(check_bit_lists),
            check_to_bit_mask=_pad_mask_rows(check_bit_lists),
            bit_to_edge=_pad_index_rows(bit_edge_lists),
            bit_to_edge_mask=_pad_mask_rows(bit_edge_lists),
            logical_to_bit=_pad_index_rows(logical_bit_lists),
            logical_to_bit_mask=_pad_mask_rows(logical_bit_lists),
            log_prior_ratios=np.log((1.0 - problem.error_priors) / np.maximum(problem.error_priors, 1e-12)),
        )


@dataclass(frozen=True)
class NeuralStaticDistillationConfig:
    code_path: str
    checkpoint_dir: str
    family: str
    bank_size: int = 1
    max_iter: int = 40
    alpha: float | None = 1.0
    train_random_error_rate: float = 0.1
    validation_random_error_rate: float = 0.1
    test_random_error_rate: float = 0.1
    random_train_batch_size: int = 32
    warmup_steps: int = 20
    mixed_steps: int = 20
    finetune_steps: int = 20
    consolidation_steps: int = 10
    mixed_half_fraction: float = 0.5
    finetune_half_fraction: float = 0.25
    learning_rate: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    selector_temperature_start: float = 1.0
    selector_temperature_end: float = 0.05
    selector_temperature_consolidation: float = 0.02
    logical_loss_weight: float = 6.0
    syndrome_loss_weight: float = 3.0
    iteration_trace_weight: float = 1.0
    confidence_loss_weight: float = 0.05
    teacher_regularization_weight: float = 0.01
    teacher_default_delta: float = 1.0
    teacher_default_gamma: float = 1.0
    validation_interval: int = 5
    initialization_candidates: int = 8
    initialization_subset_random_cases: int = 128
    initialization_subset_half_cases: int = 32
    selection_half_weight: float = 1.0
    diversity_penalty: float = 0.01
    diversity_threshold_fraction: float = 0.05
    logical_failure_penalty: float = 20.0
    logical_weight_penalty: float = 4.0
    convergence_penalty: float = 1.0
    iteration_penalty: float = 0.1
    memory_min: float = -0.3
    memory_max: float = 0.3
    damping_min: float = 0.0
    damping_max: float = 1.0
    practical_k: int = 8
    bootstrap_samples: int = 2000
    comparison_repeats: int = 16
    support_scan_random_cases: int = 512
    support_centers: tuple[float, ...] = field(default_factory=lambda: (0.0, 0.1, 0.2, 0.3, 0.4, 0.5))
    support_widths: tuple[float, ...] = field(default_factory=lambda: (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    gaussian_refinement_points: int = 9
    exact_polish_candidate_count: int = 16
    exact_polish_elite_count: int = 4
    exact_polish_rounds: int = 12
    exact_polish_validation_interval: int = 2
    exact_polish_random_train_batch_size: int = 128
    exact_polish_initialization_candidates: int = 1
    exact_polish_memory_sigma: float | None = None
    exact_polish_damping_sigma: float | None = None
    exact_polish_sigma_floor: float = 1e-3
    exact_polish_memory_sigma_ceiling: float | None = None
    exact_polish_damping_sigma_ceiling: float | None = None
    run_exact_baseline: bool = True
    device: str = "cpu"
    base_seed: int = 0

    @property
    def total_surrogate_steps(self) -> int:
        return int(self.warmup_steps + self.mixed_steps + self.finetune_steps)

    def __post_init__(self) -> None:
        if str(self.family) not in FAMILY_CHOICES:
            raise ValueError(f"Unsupported family: {self.family!r}")
        if int(self.bank_size) <= 0:
            raise ValueError("bank_size must be positive.")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive.")
        if int(self.random_train_batch_size) <= 0:
            raise ValueError("random_train_batch_size must be positive.")
        if int(self.validation_interval) <= 0:
            raise ValueError("validation_interval must be positive.")
        if float(self.memory_max) < float(self.memory_min):
            raise ValueError("memory_max must be >= memory_min.")
        if float(self.damping_max) < float(self.damping_min):
            raise ValueError("damping_max must be >= damping_min.")


def evaluate_surrogate_static_bank(
    *,
    compiled_graph: CompiledTannerGraph,
    cases: Sequence[dict[str, Any]],
    family: str,
    memory_bank: np.ndarray | None,
    damping_bank: np.ndarray | None,
    max_iter: int,
    alpha: float | None,
    selector_temperature: float = 0.05,
    teacher_parameters: dict[str, np.ndarray] | None = None,
    memory_min: float = -0.3,
    memory_max: float = 0.3,
    damping_min: float = 0.0,
    damping_max: float = 1.0,
    device: str = "cpu",
    return_case_rows: bool = False,
) -> dict[str, Any]:
    torch = require_torch()
    family = _validate_family(family)
    device_obj = _resolve_device(torch, device)
    compiled = _compiled_to_torch(compiled_graph, torch=torch, device=device_obj)
    errors, syndromes = _cases_to_torch(cases, compiled_graph, torch=torch, device=device_obj)
    memory_tensor = None if memory_bank is None else torch.as_tensor(np.asarray(memory_bank, dtype=np.float64), dtype=torch.float64, device=device_obj)
    damping_tensor = None if damping_bank is None else torch.as_tensor(np.asarray(damping_bank, dtype=np.float64), dtype=torch.float64, device=device_obj)
    teacher = _teacher_tensor_state(
        torch=torch,
        family=family,
        max_iter=int(max_iter),
        alpha=alpha,
        teacher_parameters=teacher_parameters,
        device=device_obj,
    )
    surrogate = _surrogate_forward(
        compiled=compiled,
        family=family,
        syndromes=syndromes,
        errors=errors,
        memory_bank=memory_tensor,
        damping_bank=damping_tensor,
        teacher=teacher,
        selector_temperature=float(selector_temperature),
        memory_min=float(memory_min),
        memory_max=float(memory_max),
        damping_min=float(damping_min),
        damping_max=float(damping_max),
        return_trace=True,
    )
    return _surrogate_metrics_from_forward(
        surrogate=surrogate,
        cases=cases,
        return_case_rows=return_case_rows,
    )


def train_neural_static_bank(
    problem: CodeCapacityProblem,
    *,
    config: NeuralStaticDistillationConfig,
    half_cases: Sequence[dict[str, Any]],
    random_train_cases: Sequence[dict[str, Any]],
    random_validation_cases: Sequence[dict[str, Any]],
    random_test_cases: Sequence[dict[str, Any]] | None = None,
    memory_support: dict[str, Any] | None = None,
    damping_support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    compiled_graph = CompiledTannerGraph.from_problem(problem)
    device_obj = _resolve_device(torch, config.device)
    compiled = _compiled_to_torch(compiled_graph, torch=torch, device=device_obj)

    support_payload: dict[str, Any] = {}
    if config.family in {"memory", "joint"}:
        if memory_support is None:
            support_payload["memory"] = scan_memory_distribution_support(
                problem=problem,
                random_cases=list(random_train_cases[: min(len(random_train_cases), int(config.support_scan_random_cases))]),
                half_cases=half_cases,
                centers=list(config.support_centers),
                widths=list(config.support_widths),
                gaussian_refinement_points=int(config.gaussian_refinement_points),
                draw_count=int(config.practical_k),
                max_iter=int(config.max_iter),
                alpha=config.alpha,
                logical_failure_penalty=float(config.logical_failure_penalty),
                logical_weight_penalty=float(config.logical_weight_penalty),
                convergence_penalty=float(config.convergence_penalty),
                iteration_penalty=float(config.iteration_penalty),
                selection_half_weight=float(config.selection_half_weight),
                base_seed=int(config.base_seed) + 110_003,
            )
            memory_support = support_payload["memory"]["best_overall"]
        else:
            support_payload["memory"] = {"best_overall": memory_support}
    if config.family in {"damping", "joint"}:
        if damping_support is None:
            support_payload["damping"] = scan_damping_distribution_support(
                problem=problem,
                random_cases=list(random_train_cases[: min(len(random_train_cases), int(config.support_scan_random_cases))]),
                half_cases=half_cases,
                centers=list(config.support_centers),
                widths=list(config.support_widths),
                gaussian_refinement_points=int(config.gaussian_refinement_points),
                draw_count=int(config.practical_k),
                max_iter=int(config.max_iter),
                alpha=config.alpha,
                logical_failure_penalty=float(config.logical_failure_penalty),
                logical_weight_penalty=float(config.logical_weight_penalty),
                convergence_penalty=float(config.convergence_penalty),
                iteration_penalty=float(config.iteration_penalty),
                selection_half_weight=float(config.selection_half_weight),
                base_seed=int(config.base_seed) + 310_003,
            )
            damping_support = support_payload["damping"]["best_overall"]
        else:
            support_payload["damping"] = {"best_overall": damping_support}

    _write_json(checkpoint_dir / "config.json", _config_payload(problem, config))
    _write_json(checkpoint_dir / "support_scan.json", support_payload)
    initialization = _initialize_distilled_banks(
        problem=problem,
        config=config,
        half_cases=half_cases,
        random_train_cases=random_train_cases,
        memory_support=memory_support,
        damping_support=damping_support,
        torch=torch,
        device=device_obj,
    )
    _write_json(checkpoint_dir / "support_initialization_summary.json", initialization["summary"])

    memory_logits = initialization["memory_logits"]
    damping_logits = initialization["damping_logits"]
    teacher_params = _initialize_teacher_parameters(
        torch=torch,
        config=config,
        device=device_obj,
    )
    trainable_parameters = [param for param in [memory_logits, damping_logits, *teacher_params.values()] if param is not None]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=float(config.learning_rate),
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        weight_decay=float(config.weight_decay),
    )
    surrogate_train_csv = checkpoint_dir / "surrogate_train_metrics.csv"
    surrogate_validation_csv = checkpoint_dir / "surrogate_validation_metrics.csv"

    rng = np.random.default_rng(int(config.base_seed))
    best_validation_row: dict[str, Any] | None = None
    best_distilled_memory_bank: np.ndarray | None = None
    best_distilled_damping_bank: np.ndarray | None = None
    teacher_trace_rows: list[dict[str, Any]] = []

    phase_rows = (
        [("warmup", 1.0)] * int(config.warmup_steps)
        + [("mixed", float(config.mixed_half_fraction))] * int(config.mixed_steps)
        + [("finetune", float(config.finetune_half_fraction))] * int(config.finetune_steps)
    )
    total_neural_steps = len(phase_rows)
    for step_idx, (phase_name, half_fraction) in enumerate(phase_rows):
        selector_temperature = _selector_temperature(config, step_idx, total_neural_steps)
        batch_cases = _sample_mixed_cases(
            half_cases=half_cases,
            random_cases=random_train_cases,
            half_fraction=float(half_fraction),
            batch_size=int(config.random_train_batch_size),
            rng=rng,
        )
        errors, syndromes = _cases_to_torch(batch_cases, compiled_graph, torch=torch, device=device_obj)
        objective = _training_objective(
            compiled=compiled,
            family=config.family,
            errors=errors,
            syndromes=syndromes,
            memory_bank=_memory_bank_from_logits(memory_logits, config, torch) if memory_logits is not None else None,
            damping_bank=_damping_bank_from_logits(damping_logits, config, torch) if damping_logits is not None else None,
            teacher=teacher_params,
            config=config,
            selector_temperature=float(selector_temperature),
            return_forward=True,
        )
        optimizer.zero_grad(set_to_none=True)
        objective["loss"].backward()
        optimizer.step()

        _append_csv_row(
            surrogate_train_csv,
            {
                "step": int(step_idx + 1),
                "phase": phase_name,
                "selector_temperature": float(selector_temperature),
                "loss": float(objective["loss"].detach().cpu().item()),
                "soft_logical_loss": float(objective["soft_logical_loss"].detach().cpu().item()),
                "soft_syndrome_loss": float(objective["soft_syndrome_loss"].detach().cpu().item()),
                "iteration_trace_loss": float(objective["iteration_trace_loss"].detach().cpu().item()),
                "confidence_loss": float(objective["confidence_loss"].detach().cpu().item()),
                "teacher_regularization": float(objective["teacher_regularization"].detach().cpu().item()),
                "diversity_penalty": float(objective["diversity_penalty"].detach().cpu().item()),
                "proxy_logical_success_rate": float(objective["proxy_metrics"]["logical_success_rate"]),
                "proxy_convergence_rate": float(objective["proxy_metrics"]["convergence_rate"]),
                "proxy_mean_iterations": float(objective["proxy_metrics"]["mean_iterations"]),
            },
        )
        teacher_trace_rows.append(
            {
                "step": int(step_idx + 1),
                "selector_temperature": float(selector_temperature),
                **_teacher_summary_row(teacher_params),
            }
        )
        if ((step_idx + 1) % int(config.validation_interval) == 0) or (step_idx + 1 == total_neural_steps):
            distilled_memory_bank = (
                None
                if memory_logits is None
                else _memory_bank_from_logits(memory_logits, config, torch).detach().cpu().numpy().astype(np.float64)
            )
            distilled_damping_bank = (
                None
                if damping_logits is None
                else _damping_bank_from_logits(damping_logits, config, torch).detach().cpu().numpy().astype(np.float64)
            )
            surrogate_validation = evaluate_surrogate_static_bank(
                compiled_graph=compiled_graph,
                cases=random_validation_cases,
                family=config.family,
                memory_bank=distilled_memory_bank,
                damping_bank=distilled_damping_bank,
                max_iter=config.max_iter,
                alpha=config.alpha,
                selector_temperature=float(config.selector_temperature_end),
                teacher_parameters=_teacher_state_numpy(teacher_params),
                memory_min=float(config.memory_min),
                memory_max=float(config.memory_max),
                damping_min=float(config.damping_min),
                damping_max=float(config.damping_max),
                device=str(device_obj),
            )
            exact_validation = _evaluate_bank_validation(
                problem=problem,
                config=_exact_config_from_neural(problem, config, checkpoint_dir / "exact_polish"),
                half_cases=half_cases,
                random_validation_cases=random_validation_cases,
                memory_bank=distilled_memory_bank,
                damping_bank=distilled_damping_bank,
                round_idx=step_idx,
            )
            validation_row = {
                **exact_validation,
                "surrogate_proxy_logical_success_rate": float(surrogate_validation["logical_success_rate"]),
                "surrogate_proxy_convergence_rate": float(surrogate_validation["convergence_rate"]),
                "surrogate_proxy_mean_iterations": float(surrogate_validation["mean_iterations"]),
                "selector_temperature": float(selector_temperature),
            }
            _append_csv_row(surrogate_validation_csv, validation_row)
            _save_surrogate_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step_idx=step_idx,
                memory_logits=memory_logits,
                damping_logits=damping_logits,
                teacher_params=teacher_params,
            )
            if _is_better_bank_validation(validation_row, best_validation_row):
                best_validation_row = dict(validation_row)
                best_distilled_memory_bank = distilled_memory_bank
                best_distilled_damping_bank = distilled_damping_bank
                if best_distilled_memory_bank is not None:
                    np.save(checkpoint_dir / "best_distilled_memory_bank.npy", best_distilled_memory_bank)
                if best_distilled_damping_bank is not None:
                    np.save(checkpoint_dir / "best_distilled_damping_bank.npy", best_distilled_damping_bank)
                _write_json(checkpoint_dir / "best_validation.json", best_validation_row)

    for parameter in teacher_params.values():
        parameter.requires_grad_(False)
    consolidation_parameters = [param for param in [memory_logits, damping_logits] if param is not None]
    if consolidation_parameters and int(config.consolidation_steps) > 0:
        consolidation_optimizer = torch.optim.Adam(
            consolidation_parameters,
            lr=float(config.learning_rate),
            betas=(float(config.adam_beta1), float(config.adam_beta2)),
            weight_decay=float(config.weight_decay),
        )
        for consolidation_idx in range(int(config.consolidation_steps)):
            batch_cases = _sample_mixed_cases(
                half_cases=half_cases,
                random_cases=random_train_cases,
                half_fraction=float(config.finetune_half_fraction),
                batch_size=int(config.random_train_batch_size),
                rng=rng,
            )
            errors, syndromes = _cases_to_torch(batch_cases, compiled_graph, torch=torch, device=device_obj)
            objective = _training_objective(
                compiled=compiled,
                family=config.family,
                errors=errors,
                syndromes=syndromes,
                memory_bank=_memory_bank_from_logits(memory_logits, config, torch) if memory_logits is not None else None,
                damping_bank=_damping_bank_from_logits(damping_logits, config, torch) if damping_logits is not None else None,
                teacher=teacher_params,
                config=config,
                selector_temperature=float(config.selector_temperature_consolidation),
                return_forward=False,
            )
            consolidation_optimizer.zero_grad(set_to_none=True)
            objective["loss"].backward()
            consolidation_optimizer.step()
            _append_csv_row(
                surrogate_train_csv,
                {
                    "step": int(total_neural_steps + consolidation_idx + 1),
                    "phase": "consolidation",
                    "selector_temperature": float(config.selector_temperature_consolidation),
                    "loss": float(objective["loss"].detach().cpu().item()),
                    "soft_logical_loss": float(objective["soft_logical_loss"].detach().cpu().item()),
                    "soft_syndrome_loss": float(objective["soft_syndrome_loss"].detach().cpu().item()),
                    "iteration_trace_loss": float(objective["iteration_trace_loss"].detach().cpu().item()),
                    "confidence_loss": float(objective["confidence_loss"].detach().cpu().item()),
                    "teacher_regularization": float(objective["teacher_regularization"].detach().cpu().item()),
                    "diversity_penalty": float(objective["diversity_penalty"].detach().cpu().item()),
                    "proxy_logical_success_rate": float("nan"),
                    "proxy_convergence_rate": float("nan"),
                    "proxy_mean_iterations": float("nan"),
                },
            )

    if best_distilled_memory_bank is None and memory_logits is not None:
        best_distilled_memory_bank = _memory_bank_from_logits(memory_logits, config, torch).detach().cpu().numpy().astype(np.float64)
        np.save(checkpoint_dir / "best_distilled_memory_bank.npy", best_distilled_memory_bank)
    if best_distilled_damping_bank is None and damping_logits is not None:
        best_distilled_damping_bank = _damping_bank_from_logits(damping_logits, config, torch).detach().cpu().numpy().astype(np.float64)
        np.save(checkpoint_dir / "best_distilled_damping_bank.npy", best_distilled_damping_bank)

    _write_json(
        checkpoint_dir / "teacher_diagnostics.json",
        _json_ready(
            {
                "rows": teacher_trace_rows,
                "final_teacher_state": _teacher_state_numpy(teacher_params),
                "selector_temperature_start": float(config.selector_temperature_start),
                "selector_temperature_end": float(config.selector_temperature_end),
                "selector_temperature_consolidation": float(config.selector_temperature_consolidation),
            }
        ),
    )

    exact_polish_summary = train_assignment_bank_cem(
        problem,
        config=_exact_config_from_neural(problem, config, checkpoint_dir / "exact_polish"),
        half_cases=half_cases,
        random_train_cases=random_train_cases,
        random_validation_cases=random_validation_cases,
        memory_support=memory_support,
        damping_support=damping_support,
        initial_memory_bank=best_distilled_memory_bank,
        initial_damping_bank=best_distilled_damping_bank,
    )
    exact_baseline_summary = None
    beats_exact_baseline_validation = None
    if config.run_exact_baseline:
        exact_baseline_summary = train_assignment_bank_cem(
            problem,
            config=_exact_config_from_neural(problem, config, checkpoint_dir / "exact_baseline"),
            half_cases=half_cases,
            random_train_cases=random_train_cases,
            random_validation_cases=random_validation_cases,
            memory_support=memory_support,
            damping_support=damping_support,
        )
        beats_exact_baseline_validation = _is_better_bank_validation(
            exact_polish_summary["best_validation"],
            exact_baseline_summary["best_validation"],
        )

    if exact_polish_summary["best_memory_bank"] is not None:
        np.save(checkpoint_dir / "best_memory_bank.npy", exact_polish_summary["best_memory_bank"])
    if exact_polish_summary["best_damping_bank"] is not None:
        np.save(checkpoint_dir / "best_damping_bank.npy", exact_polish_summary["best_damping_bank"])

    summary = {
        "code_path": str(problem.path),
        "checkpoint_dir": str(checkpoint_dir),
        "family": str(config.family),
        "bank_size": int(config.bank_size),
        "support_scan": support_payload,
        "support_initialization_summary": initialization["summary"],
        "surrogate_train_metrics_csv": str(surrogate_train_csv),
        "surrogate_validation_metrics_csv": str(surrogate_validation_csv),
        "best_validation": best_validation_row,
        "best_distilled_memory_bank_path": None
        if best_distilled_memory_bank is None
        else str(checkpoint_dir / "best_distilled_memory_bank.npy"),
        "best_distilled_damping_bank_path": None
        if best_distilled_damping_bank is None
        else str(checkpoint_dir / "best_distilled_damping_bank.npy"),
        "best_memory_bank_path": None
        if exact_polish_summary["best_memory_bank"] is None
        else str(checkpoint_dir / "best_memory_bank.npy"),
        "best_damping_bank_path": None
        if exact_polish_summary["best_damping_bank"] is None
        else str(checkpoint_dir / "best_damping_bank.npy"),
        "teacher_diagnostics_path": str(checkpoint_dir / "teacher_diagnostics.json"),
        "exact_polish": {
            key: value
            for key, value in exact_polish_summary.items()
            if key not in {"best_memory_bank", "best_damping_bank"}
        },
        "exact_baseline": None
        if exact_baseline_summary is None
        else {
            key: value
            for key, value in exact_baseline_summary.items()
            if key not in {"best_memory_bank", "best_damping_bank"}
        },
        "distilled_beats_exact_baseline_validation": beats_exact_baseline_validation,
    }
    if random_test_cases is not None:
        memory_distribution = None if memory_support is None else _distribution_from_support(memory_support)
        damping_distribution = None if damping_support is None else _distribution_from_support(damping_support)
        summary["random_test_comparison"] = compare_bank_to_random_baselines(
            problem=problem,
            cases=random_test_cases,
            family=str(config.family),
            memory_bank=exact_polish_summary["best_memory_bank"],
            damping_bank=exact_polish_summary["best_damping_bank"],
            memory_distribution=memory_distribution,
            damping_distribution=damping_distribution,
            max_iter=int(config.max_iter),
            alpha=config.alpha,
            k_draw=int(config.practical_k),
            logical_failure_penalty=float(config.logical_failure_penalty),
            logical_weight_penalty=float(config.logical_weight_penalty),
            convergence_penalty=float(config.convergence_penalty),
            iteration_penalty=float(config.iteration_penalty),
            bootstrap_samples=int(config.bootstrap_samples),
            repeat_count=int(config.comparison_repeats),
            base_seed=int(config.base_seed) + 2_400_003,
        )
        summary["half_comparison"] = compare_bank_to_random_baselines(
            problem=problem,
            cases=half_cases,
            family=str(config.family),
            memory_bank=exact_polish_summary["best_memory_bank"],
            damping_bank=exact_polish_summary["best_damping_bank"],
            memory_distribution=memory_distribution,
            damping_distribution=damping_distribution,
            max_iter=int(config.max_iter),
            alpha=config.alpha,
            k_draw=int(config.practical_k),
            logical_failure_penalty=float(config.logical_failure_penalty),
            logical_weight_penalty=float(config.logical_weight_penalty),
            convergence_penalty=float(config.convergence_penalty),
            iteration_penalty=float(config.iteration_penalty),
            bootstrap_samples=int(config.bootstrap_samples),
            repeat_count=int(config.comparison_repeats),
            base_seed=int(config.base_seed) + 2_900_003,
        )
    _write_json(checkpoint_dir / "training_summary.json", _json_ready(summary))
    return {
        **summary,
        "best_memory_bank": exact_polish_summary["best_memory_bank"],
        "best_damping_bank": exact_polish_summary["best_damping_bank"],
        "best_distilled_memory_bank": best_distilled_memory_bank,
        "best_distilled_damping_bank": best_distilled_damping_bank,
    }


def _training_objective(
    *,
    compiled: "_TorchCompiledTannerGraph",
    family: str,
    errors,
    syndromes,
    memory_bank,
    damping_bank,
    teacher: dict[str, Any],
    config: NeuralStaticDistillationConfig,
    selector_temperature: float,
    return_forward: bool,
) -> dict[str, Any]:
    torch = require_torch()
    surrogate = _surrogate_forward(
        compiled=compiled,
        family=family,
        syndromes=syndromes,
        errors=errors,
        memory_bank=memory_bank,
        damping_bank=damping_bank,
        teacher=teacher,
        selector_temperature=float(selector_temperature),
        memory_min=float(config.memory_min),
        memory_max=float(config.memory_max),
        damping_min=float(config.damping_min),
        damping_max=float(config.damping_max),
        return_trace=True,
    )
    loss = surrogate["selected_objective"]
    diversity_penalty = _torch_bank_diversity_penalty(
        memory_bank=memory_bank,
        damping_bank=damping_bank,
        family=family,
        memory_span=float(config.memory_max) - float(config.memory_min),
        damping_span=float(config.damping_max) - float(config.damping_min),
        threshold_fraction=float(config.diversity_threshold_fraction),
        torch=torch,
    )
    teacher_regularization = _teacher_regularization(
        teacher=teacher,
        family=family,
        config=config,
        torch=torch,
    )
    total_loss = (
        loss
        + float(config.diversity_penalty) * diversity_penalty
        + float(config.teacher_regularization_weight) * teacher_regularization
    )
    payload = {
        "loss": total_loss,
        "selected_loss": loss,
        "soft_logical_loss": surrogate["selected_soft_logical_loss"],
        "soft_syndrome_loss": surrogate["selected_soft_syndrome_loss"],
        "iteration_trace_loss": surrogate["selected_iteration_trace_loss"],
        "confidence_loss": surrogate["selected_confidence_loss"],
        "teacher_regularization": teacher_regularization,
        "diversity_penalty": diversity_penalty,
        "proxy_metrics": _surrogate_metrics_from_forward(surrogate=surrogate, cases=None, return_case_rows=False),
    }
    if return_forward:
        payload["surrogate"] = surrogate
    return payload


@dataclass(frozen=True)
class _TorchCompiledTannerGraph:
    edge_to_check: Any
    edge_to_bit: Any
    check_to_edge: Any
    check_to_edge_mask: Any
    check_to_bit: Any
    check_to_bit_mask: Any
    logical_to_bit: Any
    logical_to_bit_mask: Any
    log_prior_ratios: Any
    num_checks: int
    num_bits: int
    num_edges: int
    num_logicals: int


def _compiled_to_torch(compiled: CompiledTannerGraph, *, torch, device) -> _TorchCompiledTannerGraph:
    return _TorchCompiledTannerGraph(
        edge_to_check=torch.as_tensor(compiled.edge_to_check, dtype=torch.long, device=device),
        edge_to_bit=torch.as_tensor(compiled.edge_to_bit, dtype=torch.long, device=device),
        check_to_edge=torch.as_tensor(compiled.check_to_edge, dtype=torch.long, device=device),
        check_to_edge_mask=torch.as_tensor(compiled.check_to_edge_mask, dtype=torch.bool, device=device),
        check_to_bit=torch.as_tensor(compiled.check_to_bit, dtype=torch.long, device=device),
        check_to_bit_mask=torch.as_tensor(compiled.check_to_bit_mask, dtype=torch.bool, device=device),
        logical_to_bit=torch.as_tensor(compiled.logical_to_bit, dtype=torch.long, device=device),
        logical_to_bit_mask=torch.as_tensor(compiled.logical_to_bit_mask, dtype=torch.bool, device=device),
        log_prior_ratios=torch.as_tensor(compiled.log_prior_ratios, dtype=torch.float64, device=device),
        num_checks=int(compiled.num_checks),
        num_bits=int(compiled.num_bits),
        num_edges=int(compiled.num_edges),
        num_logicals=int(compiled.num_logicals),
    )


def _cases_to_torch(cases: Sequence[dict[str, Any]], compiled: CompiledTannerGraph, *, torch, device):
    errors = np.asarray([np.asarray(case["error"], dtype=np.uint8) for case in cases], dtype=np.uint8)
    syndromes = np.asarray([np.asarray(case["syndrome"], dtype=np.uint8) for case in cases], dtype=np.uint8)
    if errors.ndim != 2:
        errors = errors.reshape(len(cases), compiled.num_bits)
    if syndromes.ndim != 2:
        syndromes = syndromes.reshape(len(cases), compiled.num_checks)
    return (
        torch.as_tensor(errors, dtype=torch.float64, device=device),
        torch.as_tensor(syndromes, dtype=torch.float64, device=device),
    )


def _surrogate_forward(
    *,
    compiled: _TorchCompiledTannerGraph,
    family: str,
    syndromes,
    errors,
    memory_bank,
    damping_bank,
    teacher: dict[str, Any],
    selector_temperature: float,
    memory_min: float,
    memory_max: float,
    damping_min: float,
    damping_max: float,
    return_trace: bool,
) -> dict[str, Any]:
    torch = require_torch()
    family = _validate_family(family)
    batch_size = int(errors.shape[0])
    bank_size = int(
        memory_bank.shape[0]
        if memory_bank is not None
        else damping_bank.shape[0]
    )
    prior = compiled.log_prior_ratios.view(1, 1, compiled.num_bits).expand(bank_size, batch_size, compiled.num_bits)
    posterior = prior.clone()
    c2v = torch.zeros((bank_size, batch_size, compiled.num_edges), dtype=torch.float64, device=prior.device)
    v2c = prior[:, :, compiled.edge_to_bit]
    trace_posteriors = [posterior.clone()]
    trace_syndrome_losses = []
    trace_logical_losses = []
    use_memory = family in {"memory", "joint"} and memory_bank is not None
    use_damping = family in {"damping", "joint"} and damping_bank is not None

    teacher_values = {
        key: _teacher_parameter_value(value, torch=torch)
        for key, value in teacher.items()
    }
    for iter_idx in range(int(teacher_values["alpha"].shape[0])):
        alpha_t = teacher_values["alpha"][iter_idx]
        beta_t = teacher_values["beta"][iter_idx]
        gamma_t = teacher_values["gamma"][iter_idx] if "gamma" in teacher_values else None
        delta_t = teacher_values["delta"][iter_idx] if "delta" in teacher_values else None
        new_c2v = c2v.clone()
        for check_idx in range(compiled.num_checks):
            active_mask = compiled.check_to_edge_mask[check_idx]
            edge_indices = compiled.check_to_edge[check_idx][active_mask]
            if edge_indices.numel() == 0:
                continue
            syndrome_sign = 1.0 - (2.0 * syndromes[:, check_idx]).view(1, batch_size)
            for local_edge_idx, edge_idx_tensor in enumerate(edge_indices):
                edge_idx = int(edge_idx_tensor.item())
                if edge_indices.numel() == 1:
                    incoming = v2c[:, :, edge_idx : edge_idx + 1]
                else:
                    others = torch.cat((edge_indices[:local_edge_idx], edge_indices[local_edge_idx + 1 :]))
                    incoming = v2c.index_select(2, others)
                signs = torch.where(incoming < 0.0, -torch.ones_like(incoming), torch.ones_like(incoming))
                sign_product = torch.prod(signs, dim=2) * syndrome_sign
                min_abs = torch.min(torch.abs(incoming), dim=2).values
                check_message = alpha_t * sign_product * min_abs
                if use_damping:
                    effective_damping = torch.clamp(delta_t * damping_bank[:, edge_idx].view(bank_size, 1), damping_min, damping_max)
                    new_c2v[:, :, edge_idx] = (
                        effective_damping * check_message
                        + (1.0 - effective_damping) * c2v[:, :, edge_idx]
                    )
                else:
                    new_c2v[:, :, edge_idx] = check_message
        c2v = new_c2v
        variable_prior = prior
        if use_memory:
            effective_memory = torch.clamp(
                gamma_t * memory_bank.view(bank_size, 1, compiled.num_bits),
                memory_min,
                memory_max,
            )
            variable_prior = (prior * (1.0 - effective_memory)) + (posterior * effective_memory)
        sum_c2v = torch.zeros((bank_size, batch_size, compiled.num_bits), dtype=torch.float64, device=prior.device)
        edge_index = compiled.edge_to_bit.view(1, 1, compiled.num_edges).expand(bank_size, batch_size, compiled.num_edges)
        sum_c2v.scatter_add_(2, edge_index, c2v)
        posterior = beta_t * (variable_prior + sum_c2v)
        v2c = posterior[:, :, compiled.edge_to_bit] - c2v
        trace_posteriors.append(posterior.clone())
        residual_prob = _residual_probabilities(posterior, errors)
        trace_syndrome_losses.append(_soft_zero_parity_bce(residual_prob, compiled.check_to_bit, compiled.check_to_bit_mask, torch=torch))
        trace_logical_losses.append(_soft_zero_parity_bce(residual_prob, compiled.logical_to_bit, compiled.logical_to_bit_mask, torch=torch))

    final_residual_prob = _residual_probabilities(posterior, errors)
    soft_logical_loss = _soft_zero_parity_bce(final_residual_prob, compiled.logical_to_bit, compiled.logical_to_bit_mask, torch=torch)
    soft_syndrome_loss = _soft_zero_parity_bce(final_residual_prob, compiled.check_to_bit, compiled.check_to_bit_mask, torch=torch)
    iteration_trace_loss = torch.mean(torch.stack(trace_syndrome_losses, dim=0), dim=0)
    confidence_loss = _confidence_loss_from_logits(posterior, torch=torch)
    member_objective = (
        soft_logical_loss
        + soft_syndrome_loss
        + iteration_trace_loss
        + confidence_loss
    )
    soft_selector = _soft_selector_weights(member_objective, temperature=float(selector_temperature), torch=torch)
    selected_objective = torch.mean(torch.sum(soft_selector * member_objective, dim=0))
    selected_soft_logical_loss = torch.mean(torch.sum(soft_selector * soft_logical_loss, dim=0))
    selected_soft_syndrome_loss = torch.mean(torch.sum(soft_selector * soft_syndrome_loss, dim=0))
    selected_iteration_trace_loss = torch.mean(torch.sum(soft_selector * iteration_trace_loss, dim=0))
    selected_confidence_loss = torch.mean(torch.sum(soft_selector * confidence_loss, dim=0))
    return {
        "final_logits": posterior,
        "trace_logits": torch.stack(trace_posteriors, dim=0) if return_trace else None,
        "member_objective": member_objective,
        "soft_selector": soft_selector,
        "selected_objective": selected_objective,
        "selected_soft_logical_loss": selected_soft_logical_loss,
        "selected_soft_syndrome_loss": selected_soft_syndrome_loss,
        "selected_iteration_trace_loss": selected_iteration_trace_loss,
        "selected_confidence_loss": selected_confidence_loss,
        "soft_logical_loss": soft_logical_loss,
        "soft_syndrome_loss": soft_syndrome_loss,
        "iteration_trace_loss": iteration_trace_loss,
        "confidence_loss": confidence_loss,
        "errors": errors,
        "syndromes": syndromes,
        "compiled": compiled,
    }


def _surrogate_metrics_from_forward(
    *,
    surrogate: dict[str, Any],
    cases: Sequence[dict[str, Any]] | None,
    return_case_rows: bool,
) -> dict[str, Any]:
    torch = require_torch()
    compiled = surrogate["compiled"]
    final_logits = surrogate["final_logits"]
    trace_logits = surrogate["trace_logits"]
    errors = surrogate["errors"]
    bank_size = int(final_logits.shape[0])
    batch_size = int(final_logits.shape[1])
    final_bits = (final_logits < 0.0).to(dtype=torch.int64)
    error_bits = errors.to(dtype=torch.int64).unsqueeze(0).expand(bank_size, batch_size, compiled.num_bits)
    residual_bits = torch.remainder(final_bits + error_bits, 2)
    check_parity = _hard_zero_parity(residual_bits, compiled.check_to_bit, compiled.check_to_bit_mask, torch=torch)
    logical_parity = _hard_zero_parity(residual_bits, compiled.logical_to_bit, compiled.logical_to_bit_mask, torch=torch)
    residual_syndrome_weight = torch.sum(check_parity, dim=2)
    logical_weight = torch.sum(logical_parity, dim=2)
    logical_success = logical_weight == 0
    converged = residual_syndrome_weight == 0
    exact_recovery = torch.sum(residual_bits, dim=2) == 0
    decoding_weight = torch.sum(final_bits, dim=2)
    iteration_proxy = _hard_iteration_proxy(trace_logits, errors, compiled, torch=torch)

    case_rows = []
    selected_rows = []
    for case_idx in range(batch_size):
        rows = []
        for bank_idx in range(bank_size):
            rows.append(
                {
                    "bank_idx": int(bank_idx),
                    "logical_success": bool(logical_success[bank_idx, case_idx].item()),
                    "logical_weight": int(logical_weight[bank_idx, case_idx].item()),
                    "converged": bool(converged[bank_idx, case_idx].item()),
                    "iterations": int(iteration_proxy[bank_idx, case_idx].item()),
                    "decoding_weight": int(decoding_weight[bank_idx, case_idx].item()),
                    "residual_syndrome_weight": int(residual_syndrome_weight[bank_idx, case_idx].item()),
                    "exact_recovery": bool(exact_recovery[bank_idx, case_idx].item()),
                    "soft_objective": float(surrogate["member_objective"][bank_idx, case_idx].detach().cpu().item()),
                }
            )
        selected = min(
            rows,
            key=lambda row: (
                int(row["residual_syndrome_weight"]),
                not bool(row["converged"]),
                int(row["iterations"]),
                int(row["decoding_weight"]),
                int(row["bank_idx"]),
            ),
        )
        selected_rows.append(selected)
        if return_case_rows:
            case = None if cases is None else cases[case_idx]
            case_rows.append(
                {
                    "case_id": -1 if case is None else int(case.get("case_id", case.get("sample_idx", -1))),
                    "row_idx": -1 if case is None else int(case.get("row_idx", -1)),
                    "row_weight": -1 if case is None else int(case.get("row_weight", -1)),
                    "support_bits": [] if case is None else list(case.get("support_bits", ())),
                    "flipped_bits": [] if case is None else list(case.get("flipped_bits", ())),
                    **selected,
                }
            )

    num_cases = max(1, len(selected_rows))
    metrics = {
        "num_cases": int(len(selected_rows)),
        "bank_size": int(bank_size),
        "logical_success_rate": float(np.mean([float(row["logical_success"]) for row in selected_rows])),
        "exact_recovery_rate": float(np.mean([float(row["exact_recovery"]) for row in selected_rows])),
        "convergence_rate": float(np.mean([float(row["converged"]) for row in selected_rows])),
        "mean_iterations": float(np.mean([float(row["iterations"]) for row in selected_rows])),
        "mean_logical_weight": float(np.mean([float(row["logical_weight"]) for row in selected_rows])),
        "soft_objective_mean": float(np.mean([float(row["soft_objective"]) for row in selected_rows])),
        "selected_soft_logical_loss": float(surrogate["selected_soft_logical_loss"].detach().cpu().item()),
        "selected_soft_syndrome_loss": float(surrogate["selected_soft_syndrome_loss"].detach().cpu().item()),
        "selected_iteration_trace_loss": float(surrogate["selected_iteration_trace_loss"].detach().cpu().item()),
        "selected_confidence_loss": float(surrogate["selected_confidence_loss"].detach().cpu().item()),
    }
    if return_case_rows:
        metrics["case_rows"] = case_rows
    return metrics


def _hard_iteration_proxy(trace_logits, errors, compiled: _TorchCompiledTannerGraph, *, torch):
    if trace_logits is None:
        bank_size, batch_size = int(errors.shape[0]), int(errors.shape[1])
        return torch.full((bank_size, batch_size), 0, dtype=torch.int64, device=errors.device)
    trace_bits = (trace_logits < 0.0).to(dtype=torch.int64)
    error_bits = errors.to(dtype=torch.int64).unsqueeze(0).unsqueeze(0)
    error_bits = error_bits.expand(trace_bits.shape[0], trace_bits.shape[1], trace_bits.shape[2], trace_bits.shape[3])
    residual = torch.remainder(trace_bits + error_bits, 2)
    trace_check = _hard_zero_parity_trace(residual, compiled.check_to_bit, compiled.check_to_bit_mask, torch=torch)
    zero_weight = torch.sum(trace_check, dim=3) == 0
    iteration_proxy = torch.full((trace_bits.shape[1], trace_bits.shape[2]), int(trace_bits.shape[0] - 1), dtype=torch.int64, device=trace_bits.device)
    for trace_idx in range(trace_bits.shape[0]):
        mask = zero_weight[trace_idx]
        iteration_proxy = torch.where(mask & (iteration_proxy == int(trace_bits.shape[0] - 1)), torch.full_like(iteration_proxy, int(trace_idx)), iteration_proxy)
    return iteration_proxy


def _hard_zero_parity(residual_bits, padded_indices, padded_mask, *, torch):
    gathered = residual_bits[:, :, padded_indices]
    mask = padded_mask.view(1, 1, padded_mask.shape[0], padded_mask.shape[1])
    masked = torch.where(mask, gathered, torch.zeros_like(gathered))
    return torch.remainder(torch.sum(masked, dim=3), 2)


def _hard_zero_parity_trace(residual_bits, padded_indices, padded_mask, *, torch):
    gathered = residual_bits[:, :, :, padded_indices]
    mask = padded_mask.view(1, 1, 1, padded_mask.shape[0], padded_mask.shape[1])
    masked = torch.where(mask, gathered, torch.zeros_like(gathered))
    return torch.remainder(torch.sum(masked, dim=4), 2)


def _residual_probabilities(posterior_logits, errors):
    torch = require_torch()
    decoding_prob = torch.sigmoid(-posterior_logits)
    return torch.where(errors.unsqueeze(0) > 0.5, 1.0 - decoding_prob, decoding_prob)


def _soft_zero_parity_bce(residual_prob, padded_indices, padded_mask, *, torch):
    if padded_indices.numel() == 0:
        return torch.zeros((residual_prob.shape[0], residual_prob.shape[1]), dtype=residual_prob.dtype, device=residual_prob.device)
    gathered = residual_prob[:, :, padded_indices]
    mask = padded_mask.view(1, 1, padded_mask.shape[0], padded_mask.shape[1])
    neutral = torch.ones_like(gathered)
    transformed = torch.where(mask, 1.0 - (2.0 * gathered), neutral)
    parity_one = 0.5 * (1.0 - torch.prod(transformed, dim=3))
    parity_one = torch.clamp(parity_one, 1e-6, 1.0 - 1e-6)
    return -torch.mean(torch.log1p(-parity_one), dim=2)


def _confidence_loss_from_logits(posterior_logits, *, torch):
    probs = torch.clamp(torch.sigmoid(-posterior_logits), 1e-6, 1.0 - 1e-6)
    entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
    return torch.mean(entropy, dim=2)


def _soft_selector_weights(member_scores, *, temperature: float, torch):
    if int(member_scores.shape[0]) == 1:
        return torch.ones_like(member_scores)
    temp = max(float(temperature), 1e-6)
    logits = -member_scores / temp
    logits = logits - torch.max(logits, dim=0, keepdim=True).values
    weights = torch.softmax(logits, dim=0)
    return weights


def _initialize_distilled_banks(
    *,
    problem: CodeCapacityProblem,
    config: NeuralStaticDistillationConfig,
    half_cases: Sequence[dict[str, Any]],
    random_train_cases: Sequence[dict[str, Any]],
    memory_support: dict[str, Any] | None,
    damping_support: dict[str, Any] | None,
    torch,
    device,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(config.base_seed))
    random_subset = list(random_train_cases[: min(len(random_train_cases), int(config.initialization_subset_random_cases))])
    half_subset = list(half_cases[: min(len(half_cases), int(config.initialization_subset_half_cases))])
    memory_candidates = []
    damping_candidates = []
    scored = []
    for candidate_idx in range(max(1, int(config.initialization_candidates))):
        seed_rng = np.random.default_rng(int(config.base_seed) + 10_003 * candidate_idx)
        memory_bank = None
        damping_bank = None
        if config.family in {"memory", "joint"}:
            memory_bank = _sample_initial_bank_from_support(
                shape=(int(config.bank_size), int(problem.n_bits)),
                support=memory_support,
                low=float(config.memory_min),
                high=float(config.memory_max),
                rng=seed_rng,
            )
        if config.family in {"damping", "joint"}:
            damping_bank = _sample_initial_bank_from_support(
                shape=(int(config.bank_size), int(problem.hz.nnz)),
                support=damping_support,
                low=float(config.damping_min),
                high=float(config.damping_max),
                rng=seed_rng,
            )
        loss_terms = []
        if random_subset:
            random_metrics = evaluate_assignment_bank(
                problem=problem,
                cases=random_subset,
                family=config.family,
                memory_bank=memory_bank,
                damping_bank=damping_bank,
                max_iter=config.max_iter,
                alpha=config.alpha,
                selection_mode="practical",
                logical_failure_penalty=config.logical_failure_penalty,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            )
            loss_terms.append(float(random_metrics["loss_mean"]))
        if half_subset:
            half_metrics = evaluate_assignment_bank(
                problem=problem,
                cases=half_subset,
                family=config.family,
                memory_bank=memory_bank,
                damping_bank=damping_bank,
                max_iter=config.max_iter,
                alpha=config.alpha,
                selection_mode="practical",
                logical_failure_penalty=config.logical_failure_penalty,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            )
            loss_terms.append(float(config.selection_half_weight) * float(half_metrics["loss_mean"]))
        scored.append((float(np.sum(loss_terms)), memory_bank, damping_bank))
        memory_candidates.append(memory_bank)
        damping_candidates.append(damping_bank)
    scored.sort(key=lambda item: item[0])
    best_loss, best_memory_bank, best_damping_bank = scored[0]
    summary = {
        "candidate_count": int(len(scored)),
        "best_initial_selection_loss": float(best_loss),
        "memory_support": memory_support,
        "damping_support": damping_support,
    }
    memory_logits = None
    damping_logits = None
    if best_memory_bank is not None:
        memory_logits = torch.nn.Parameter(
            torch.as_tensor(
                _memory_logits_from_values(best_memory_bank, config=config),
                dtype=torch.float64,
                device=device,
            )
        )
        summary["memory_initial_mean"] = float(np.mean(best_memory_bank))
        summary["memory_initial_std"] = float(np.std(best_memory_bank))
    if best_damping_bank is not None:
        damping_logits = torch.nn.Parameter(
            torch.as_tensor(
                _damping_logits_from_values(best_damping_bank, config=config),
                dtype=torch.float64,
                device=device,
            )
        )
        summary["damping_initial_mean"] = float(np.mean(best_damping_bank))
        summary["damping_initial_std"] = float(np.std(best_damping_bank))
    return {
        "memory_logits": memory_logits,
        "damping_logits": damping_logits,
        "summary": summary,
    }


def _sample_initial_bank_from_support(
    *,
    shape: tuple[int, int],
    support: dict[str, Any] | None,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if support is not None and "mean" in support:
        sigma = float(support.get("sigma", max((high - low) / 4.0, 1e-6)))
        sampled = rng.normal(loc=float(support["mean"]), scale=sigma, size=shape)
        return np.clip(sampled, low, high).astype(np.float64)
    return rng.uniform(low, high, size=shape).astype(np.float64)


def _initialize_teacher_parameters(*, torch, config: NeuralStaticDistillationConfig, device):
    alpha_default = 1.0 if config.alpha is None else float(config.alpha)
    alpha = torch.nn.Parameter(torch.full((int(config.max_iter),), _inverse_softplus(alpha_default), dtype=torch.float64, device=device))
    beta = torch.nn.Parameter(torch.full((int(config.max_iter),), _inverse_softplus(1.0), dtype=torch.float64, device=device))
    teacher = {"alpha": alpha, "beta": beta}
    if config.family in {"memory", "joint"}:
        teacher["gamma"] = torch.nn.Parameter(
            torch.full((int(config.max_iter),), _inverse_softplus(float(config.teacher_default_gamma)), dtype=torch.float64, device=device)
        )
    if config.family in {"damping", "joint"}:
        teacher["delta"] = torch.nn.Parameter(
            torch.full((int(config.max_iter),), _inverse_softplus(float(config.teacher_default_delta)), dtype=torch.float64, device=device)
        )
    return teacher


def _teacher_tensor_state(*, torch, family: str, max_iter: int, alpha: float | None, teacher_parameters: dict[str, np.ndarray] | None, device):
    alpha_default = 1.0 if alpha is None else float(alpha)
    state = {
        "alpha": torch.full((int(max_iter),), alpha_default, dtype=torch.float64, device=device),
        "beta": torch.full((int(max_iter),), 1.0, dtype=torch.float64, device=device),
    }
    if family in {"memory", "joint"}:
        state["gamma"] = torch.full((int(max_iter),), 1.0, dtype=torch.float64, device=device)
    if family in {"damping", "joint"}:
        state["delta"] = torch.full((int(max_iter),), 1.0, dtype=torch.float64, device=device)
    if teacher_parameters is not None:
        for key in list(state):
            if key in teacher_parameters:
                state[key] = torch.as_tensor(np.asarray(teacher_parameters[key], dtype=np.float64), dtype=torch.float64, device=device)
    return state


def _teacher_regularization(*, teacher: dict[str, Any], family: str, config: NeuralStaticDistillationConfig, torch):
    losses = []
    alpha_target = 1.0 if config.alpha is None else float(config.alpha)
    losses.append(torch.mean((_positive_parameter(teacher["alpha"], torch=torch) - alpha_target) ** 2))
    losses.append(torch.mean((_positive_parameter(teacher["beta"], torch=torch) - 1.0) ** 2))
    if family in {"memory", "joint"} and "gamma" in teacher:
        losses.append(torch.mean((_positive_parameter(teacher["gamma"], torch=torch) - float(config.teacher_default_gamma)) ** 2))
    if family in {"damping", "joint"} and "delta" in teacher:
        losses.append(torch.mean((_positive_parameter(teacher["delta"], torch=torch) - float(config.teacher_default_delta)) ** 2))
    return torch.mean(torch.stack(losses, dim=0)) if losses else torch.zeros((), dtype=torch.float64)


def _teacher_summary_row(teacher: dict[str, Any]) -> dict[str, float]:
    return {
        "alpha_mean": float(np.mean(_positive_parameter_numpy(teacher["alpha"]))),
        "beta_mean": float(np.mean(_positive_parameter_numpy(teacher["beta"]))),
        "gamma_mean": float(np.mean(_positive_parameter_numpy(teacher["gamma"]))) if "gamma" in teacher else float("nan"),
        "delta_mean": float(np.mean(_positive_parameter_numpy(teacher["delta"]))) if "delta" in teacher else float("nan"),
    }


def _teacher_state_numpy(teacher: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: _positive_parameter_numpy(value)
        for key, value in teacher.items()
    }


def _positive_parameter(parameter, *, torch):
    return torch.nn.functional.softplus(parameter) + 1e-6


def _positive_parameter_numpy(parameter) -> np.ndarray:
    torch = require_torch()
    return _positive_parameter(parameter, torch=torch).detach().cpu().numpy().astype(np.float64)


def _teacher_parameter_value(parameter, *, torch):
    if getattr(parameter, "requires_grad", False):
        return _positive_parameter(parameter, torch=torch)
    return parameter


def _memory_bank_from_logits(logits, config: NeuralStaticDistillationConfig, torch):
    center = (float(config.memory_min) + float(config.memory_max)) / 2.0
    half_span = (float(config.memory_max) - float(config.memory_min)) / 2.0
    return center + (half_span * torch.tanh(logits))


def _damping_bank_from_logits(logits, config: NeuralStaticDistillationConfig, torch):
    scaled = torch.sigmoid(logits)
    return float(config.damping_min) + (float(config.damping_max) - float(config.damping_min)) * scaled


def _memory_logits_from_values(values: np.ndarray, *, config: NeuralStaticDistillationConfig) -> np.ndarray:
    return raw_memory_parameters(
        np.asarray(values, dtype=np.float64),
        memory_min=float(config.memory_min),
        memory_max=float(config.memory_max),
    )


def _damping_logits_from_values(
    values: np.ndarray,
    *,
    config: NeuralStaticDistillationConfig,
    epsilon: float = 1e-6,
) -> np.ndarray:
    span = float(config.damping_max) - float(config.damping_min)
    if span <= 0.0:
        raise ValueError("damping_max must be greater than damping_min.")
    scaled = (np.asarray(values, dtype=np.float64) - float(config.damping_min)) / span
    clipped = np.clip(scaled, float(epsilon), 1.0 - float(epsilon))
    return np.log(clipped / (1.0 - clipped)).astype(np.float64)


def _torch_bank_diversity_penalty(
    *,
    memory_bank,
    damping_bank,
    family: str,
    memory_span: float,
    damping_span: float,
    threshold_fraction: float,
    torch,
):
    penalties = []
    if family in {"memory", "joint"} and memory_bank is not None and int(memory_bank.shape[0]) > 1:
        penalties.append(_torch_duplicate_penalty(memory_bank, span=max(memory_span, 1e-6), threshold_fraction=threshold_fraction, torch=torch))
    if family in {"damping", "joint"} and damping_bank is not None and int(damping_bank.shape[0]) > 1:
        penalties.append(_torch_duplicate_penalty(damping_bank, span=max(damping_span, 1e-6), threshold_fraction=threshold_fraction, torch=torch))
    if not penalties:
        return torch.zeros((), dtype=torch.float64, device=(memory_bank if memory_bank is not None else damping_bank).device if (memory_bank is not None or damping_bank is not None) else None)
    return torch.mean(torch.stack(penalties, dim=0))


def _torch_duplicate_penalty(bank, *, span: float, threshold_fraction: float, torch):
    threshold = max(float(span) * float(threshold_fraction), 1e-9)
    penalties = []
    for first_idx in range(int(bank.shape[0])):
        for second_idx in range(first_idx + 1, int(bank.shape[0])):
            diff = torch.mean(torch.abs(bank[first_idx] - bank[second_idx]))
            penalties.append(torch.relu(torch.tensor(threshold, dtype=bank.dtype, device=bank.device) - diff) / threshold)
    if not penalties:
        return torch.zeros((), dtype=bank.dtype, device=bank.device)
    return torch.mean(torch.stack(penalties, dim=0))


def _sample_mixed_cases(
    *,
    half_cases: Sequence[dict[str, Any]],
    random_cases: Sequence[dict[str, Any]],
    half_fraction: float,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    half_count = min(int(round(float(batch_size) * max(0.0, min(1.0, half_fraction)))), int(batch_size))
    random_count = max(0, int(batch_size) - half_count)
    cases = []
    if half_cases and half_count > 0:
        half_indices = rng.integers(0, len(half_cases), size=int(half_count))
        cases.extend([half_cases[int(index)] for index in half_indices])
    if random_cases and random_count > 0:
        random_indices = rng.integers(0, len(random_cases), size=int(random_count))
        cases.extend([random_cases[int(index)] for index in random_indices])
    rng.shuffle(cases)
    return cases


def _selector_temperature(config: NeuralStaticDistillationConfig, step_idx: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(config.selector_temperature_end)
    frac = float(step_idx) / float(max(total_steps - 1, 1))
    start = max(float(config.selector_temperature_start), 1e-6)
    end = max(float(config.selector_temperature_end), 1e-6)
    log_temp = ((1.0 - frac) * np.log(start)) + (frac * np.log(end))
    return float(np.exp(log_temp))


def _distribution_from_support(row: dict[str, Any]) -> ScalarDistributionSpec:
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


def _exact_config_from_neural(
    problem: CodeCapacityProblem,
    config: NeuralStaticDistillationConfig,
    checkpoint_dir: Path,
) -> StaticAssignmentBankConfig:
    return StaticAssignmentBankConfig(
        code_path=str(problem.path),
        checkpoint_dir=str(checkpoint_dir),
        family=str(config.family),
        bank_size=int(config.bank_size),
        max_iter=int(config.max_iter),
        alpha=config.alpha,
        train_random_error_rate=float(config.train_random_error_rate),
        validation_random_error_rate=float(config.validation_random_error_rate),
        test_random_error_rate=float(config.test_random_error_rate),
        random_train_samples=0,
        random_validation_samples=0,
        random_test_samples=0,
        random_train_batch_size=int(config.exact_polish_random_train_batch_size),
        candidate_count=int(config.exact_polish_candidate_count),
        elite_count=int(config.exact_polish_elite_count),
        rounds=int(config.exact_polish_rounds),
        validation_interval=int(config.exact_polish_validation_interval),
        practical_k=int(config.practical_k),
        logical_failure_penalty=float(config.logical_failure_penalty),
        logical_weight_penalty=float(config.logical_weight_penalty),
        convergence_penalty=float(config.convergence_penalty),
        iteration_penalty=float(config.iteration_penalty),
        selection_half_weight=float(config.selection_half_weight),
        diversity_penalty=float(config.diversity_penalty),
        diversity_threshold_fraction=float(config.diversity_threshold_fraction),
        memory_min=float(config.memory_min),
        memory_max=float(config.memory_max),
        damping_min=float(config.damping_min),
        damping_max=float(config.damping_max),
        initial_memory_sigma=config.exact_polish_memory_sigma,
        initial_damping_sigma=config.exact_polish_damping_sigma,
        sigma_floor=float(config.exact_polish_sigma_floor),
        memory_sigma_ceiling=config.exact_polish_memory_sigma_ceiling,
        damping_sigma_ceiling=config.exact_polish_damping_sigma_ceiling,
        initialization_candidates=int(config.exact_polish_initialization_candidates),
        base_seed=int(config.base_seed) + 4_200_003,
    )


def _save_surrogate_checkpoint(
    *,
    checkpoint_dir: Path,
    step_idx: int,
    memory_logits,
    damping_logits,
    teacher_params: dict[str, Any],
) -> None:
    payload = {
        "step_idx": np.asarray([int(step_idx + 1)], dtype=np.int64),
        "memory_logits": np.asarray([] if memory_logits is None else memory_logits.detach().cpu().numpy(), dtype=np.float64),
        "damping_logits": np.asarray([] if damping_logits is None else damping_logits.detach().cpu().numpy(), dtype=np.float64),
    }
    for key, value in teacher_params.items():
        payload[f"{key}_raw"] = np.asarray(value.detach().cpu().numpy(), dtype=np.float64)
    np.savez_compressed(checkpoint_dir / "surrogate_checkpoint_latest.npz", **payload)


def _config_payload(problem: CodeCapacityProblem, config: NeuralStaticDistillationConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "code_path": str(problem.path),
        "n_bits": int(problem.n_bits),
        "n_checks": int(problem.hz.shape[0]),
        "nnz": int(problem.hz.nnz),
    }


def _resolve_device(torch, device_name: str):
    name = str(device_name).strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return float(np.log(np.expm1(value)))


def _validate_family(family: str) -> str:
    family = str(family)
    if family not in FAMILY_CHOICES:
        raise ValueError(f"Unsupported family: {family!r}")
    return family


def _pad_index_rows(rows: Sequence[Sequence[int]]) -> np.ndarray:
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return np.zeros((len(rows), 0), dtype=np.int64)
    padded = np.full((len(rows), width), -1, dtype=np.int64)
    for row_idx, row in enumerate(rows):
        if row:
            padded[row_idx, : len(row)] = np.asarray(row, dtype=np.int64)
    return padded


def _pad_mask_rows(rows: Sequence[Sequence[int]]) -> np.ndarray:
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return np.zeros((len(rows), 0), dtype=bool)
    mask = np.zeros((len(rows), width), dtype=bool)
    for row_idx, row in enumerate(rows):
        if row:
            mask[row_idx, : len(row)] = True
    return mask


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
