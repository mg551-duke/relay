#!/usr/bin/env python3
"""Sinter-based Relay-BP-5 performance sweep with mean iteration tracking.

This is a circuit-level performance runner for the
`bicycle_bivariate_144_12_12_memory_Z` family using Z-basis detector filtering.
It is modeled on `run_bivariate_bicycle144_12_12_native_bpgd_cluster_Z_basis.py`
but focuses on six Relay-BP-5 variants:

1. Relay-BP-5 without damping
2. Relay-BP-5 with random damping in [0.85, 0.95]
3. Relay-BP-5 with trained two-point discrete memory
4. Relay-BP-5 with trained two-point discrete memory and damping 0.9
5. Relay-BP-5 with trained two-point discrete memory and damping in [0.85, 0.95]
6. Relay-BP-5 with continuous interval memory and damping 0.9
7. Relay-BP-5 with the top-10 trained static discrete assignments and damping 0.9

Two optional static-assignment decoders can be appended with
`--static-discrete-gammas-path`, `--static-continuous-gammas-path`, and
`--static-discrete-assignment-bank-base`.

Shared decoder settings:
- p sweep: 0.001, 0.002, 0.003, 0.004, 0.005
- max logical errors per p: 50
- max shots per p: 1e8
- gamma0 = 0.125
- relay weights sampled from [-0.24, 0.66], except for trained discrete memory
- R = 601 total legs (600 relay continuation legs)
- stop after 5 converged relay legs

The runner still uses `sinter.collect`, but uses a custom sinter sampler so it
can aggregate iteration totals into `custom_counts` without persisting per-shot
detail records.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sinter
import stim

from relay_bp.stim import CheckMatrices, SinterDecoder_RelayBP
from relay_bp.stim.sinter.runner import _filter_detectors_by_basis
from relay_bp.stim.sinter.utils import write_stats
from sinter._decoding._sampler import CompiledSampler, Sampler

SELECTED_P_VALUES = [0.001, 0.002, 0.003, 0.004, 0.005]
ERRORS_BY_P = {
    0.001: 50,
    0.002: 50,
    0.003: 50,
    0.004: 50,
    0.005: 50,
}
MAX_SHOTS_SAFETY_CAP = 100_000_000

BASIS_FILTER = "Z"

RELAY_ALPHA = 1.0
RELAY_GAMMA0 = 0.125
RELAY_PRE_ITER = 80
RELAY_NUM_SETS = 600
RELAY_SET_MAX_ITER = 60
RELAY_GAMMA_DIST_INTERVAL = (-0.24, 0.66)
RELAY_STOP_NCONV = 5
RELAY_POSTERIORS = True
RELAY_FIXED_DAMPING = 0.9
RELAY_DAMPING_INTERVAL = (0.85, 0.95)
# Native wrapper order: (negative, positive, p_positive).
RELAY_BERNOULLI_GAMMA = (
    -0.0687506131581227,
    0.3557972204473916,
    0.5451025293701163,
)
RELAY_BASE_SEED = 0

DEFAULT_DECODER_SEQUENCE = [
    "relay-bp5-r601-no-damping",
    "relay-bp5-r601-damp085-095",
    "relay-bp5-r601-discrete-memory",
    "relay-bp5-r601-discrete-memory-damp0p9",
    "relay-bp5-r601-discrete-memory-damp085-095",
    "relay-bp5-r601-damp0p9",
]
STATIC_DISCRETE_DECODER_NAME = "relay-bp5-r601-static-discrete-memory"
STATIC_CONTINUOUS_DECODER_NAME = "relay-bp5-r601-static-continuous-memory"
STATIC_DISCRETE_TOPK_DAMP_DECODER_NAME = "relay-bp5-r601-static-discrete-top10-damp0p9"
MESSAGE_MIX_NO_GAMMA_DECODER_NAME = "relay-bp5-r601-message-mix-no-gamma"
MESSAGE_MIX_GAMMA0125_DECODER_NAME = "relay-bp5-r601-message-mix-gamma0p125"
DEFAULT_DECODER_INDICES: list[int] | None = None
DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_STATIC_DISCRETE_ASSIGNMENT_BANK_TOP_K = 10


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "tests" / "testdata" / "bicycle_bivariate").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root containing tests/testdata/bicycle_bivariate "
        f"starting from {start}."
    )


def resolve_num_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return max(1, multiprocessing.cpu_count())


def parse_name_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"stim_path": str(path)}
    for part in path.stem.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        metadata[key] = value
    if "error_rate" in metadata:
        metadata["p"] = float(metadata["error_rate"])
    return metadata


def parse_float_csv(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_static_gammas(
    path: Path | None,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    resolved = path.resolve()
    gammas = np.asarray(np.load(resolved), dtype=np.float64)
    if gammas.ndim == 1:
        gammas = gammas.reshape(1, -1)
    if gammas.ndim != 2 or gammas.shape[0] != 1:
        raise ValueError(
            f"Static gamma artifact must be 1D or shape (1, n_variables); got {gammas.shape} from {resolved}."
        )
    if not np.all(np.isfinite(gammas)):
        raise ValueError(
            f"Static gamma artifact contains non-finite values: {resolved}"
        )
    metadata = {
        "path": str(resolved),
        "shape": [int(dim) for dim in gammas.shape],
        "sha256": file_sha256(resolved),
    }
    return gammas, metadata


def load_message_mix_best_params(
    path: Path | None,
) -> tuple[
    tuple[float, float, float, float, float, float] | None,
    tuple[float, float, float] | None,
    dict[str, Any] | None,
]:
    if path is None:
        return None, None, None
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Message-mix params path does not exist: {resolved}")
    best_path = resolve_message_mix_best_params_path(resolved)
    payload = load_json_if_dict(best_path)
    if payload is None:
        raise ValueError(
            f"Expected JSON object in message-mix params file: {best_path}"
        )
    params_payload = payload.get("params") if "params" in payload else payload
    if not isinstance(params_payload, dict):
        raise ValueError(f"Message-mix params JSON missing params object: {best_path}")
    names = [
        "fresh_negative",
        "fresh_positive",
        "fresh_p_positive",
        "previous_negative",
        "previous_positive",
        "previous_p_positive",
    ]
    if "message_mix_bernoulli" in params_payload:
        values = tuple(float(value) for value in params_payload["message_mix_bernoulli"])
    else:
        values = tuple(float(params_payload[name]) for name in names)
    second_previous_names = [
        "second_previous_negative",
        "second_previous_positive",
        "second_previous_p_positive",
    ]
    if "message_mix_second_previous_bernoulli" in params_payload:
        second_previous_values = tuple(
            float(value)
            for value in params_payload["message_mix_second_previous_bernoulli"]
        )
    elif all(name in params_payload for name in second_previous_names):
        second_previous_values = tuple(
            float(params_payload[name]) for name in second_previous_names
        )
    else:
        second_previous_values = None
    if not all(np.isfinite(values)):
        raise ValueError(f"Message-mix params contain non-finite values: {best_path}")
    if second_previous_values is not None and not all(np.isfinite(second_previous_values)):
        raise ValueError(
            f"Second-previous message-mix params contain non-finite values: {best_path}"
        )
    metadata = {
        "path": str(best_path),
        "source": str(resolved),
        "sha256": file_sha256(best_path),
        "message_mix_bernoulli": list(values),
    }
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        metadata["metrics"] = metrics
    if second_previous_values is not None:
        metadata["message_mix_second_previous_bernoulli"] = list(
            second_previous_values
        )
    return values, second_previous_values, metadata


def resolve_message_mix_best_params_path(path: Path) -> Path:
    if path.is_file():
        return path
    direct = path / "best_params.json"
    if direct.exists():
        return direct
    seed_candidates = []
    for seed_dir in path.iterdir():
        candidate = seed_dir / "best_params.json"
        if (
            seed_dir.is_dir()
            and seed_dir.name.startswith("seed_")
            and candidate.exists()
        ):
            payload = load_json_if_dict(candidate)
            metrics = payload.get("metrics") if isinstance(payload, dict) else None
            rank_key = (
                metrics_rank_key(metrics)
                if isinstance(metrics, dict)
                else (float("inf"), float("inf"), 0.0)
            )
            seed_candidates.append((rank_key, seed_dir_sort_key(seed_dir), candidate))
    if not seed_candidates:
        raise FileNotFoundError(
            f"No best_params.json found under message-mix training output base: {path}"
        )
    seed_candidates.sort(key=lambda item: (item[0], item[1]))
    return seed_candidates[0][2]


def metrics_rank_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("logical_failure_rate", float("inf"))),
        float(metrics.get("mean_iterations", float("inf"))),
        -float(metrics.get("convergence_rate", 0.0)),
    )


def load_static_discrete_assignment_bank(
    base_path: Path | None,
    *,
    top_k: int = DEFAULT_STATIC_DISCRETE_ASSIGNMENT_BANK_TOP_K,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if base_path is None:
        return None, None
    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}.")

    resolved = base_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Static discrete assignment bank base does not exist: {resolved}"
        )

    seed_dirs = [
        directory
        for directory in resolved.iterdir()
        if directory.is_dir() and directory.name.startswith("seed_")
    ]
    candidate_dirs = sorted(seed_dirs, key=seed_dir_sort_key)
    if not candidate_dirs and (resolved / "discrete_best_static_gammas.npy").exists():
        candidate_dirs = [resolved]

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    expected_width: int | None = None
    for seed_dir in candidate_dirs:
        gammas_path = seed_dir / "discrete_best_static_gammas.npy"
        if not gammas_path.exists():
            skipped.append(
                {"seed_dir": str(seed_dir.resolve()), "reason": "missing gammas"}
            )
            continue

        metrics, metrics_source, metrics_path = load_static_discrete_metrics(seed_dir)
        if metrics is None or metrics_source is None or metrics_path is None:
            skipped.append(
                {"seed_dir": str(seed_dir.resolve()), "reason": "missing metrics"}
            )
            continue

        gammas = load_static_gamma_vector(gammas_path)
        if expected_width is None:
            expected_width = int(gammas.shape[0])
        elif gammas.shape[0] != expected_width:
            raise ValueError(
                "Static assignment gamma width mismatch: "
                f"{gammas_path.resolve()} has {gammas.shape[0]}, expected {expected_width}."
            )

        rank_key = metrics_rank_key(metrics)
        candidates.append(
            {
                "seed_dir": str(seed_dir.resolve()),
                "gammas_path": str(gammas_path.resolve()),
                "gammas_sha256": file_sha256(gammas_path.resolve()),
                "metrics_path": str(metrics_path.resolve()),
                "metrics_source": metrics_source,
                "metrics": metrics,
                "rank_key": list(rank_key),
                "gammas": gammas,
            }
        )

    if len(candidates) < top_k:
        raise ValueError(
            f"Need at least {top_k} valid static discrete assignments under {resolved}; "
            f"found {len(candidates)}. Skipped entries: {skipped}"
        )

    selected = sorted(candidates, key=lambda candidate: tuple(candidate["rank_key"]))[
        :top_k
    ]
    ranked_bank = np.vstack([candidate["gammas"] for candidate in selected]).astype(
        np.float64, copy=False
    )
    bank = relay_explicit_gammas_from_ranked_bank(ranked_bank)
    metadata = {
        "base_path": str(resolved),
        "top_k": int(top_k),
        "shape": [int(dim) for dim in bank.shape],
        "ranked_shape": [int(dim) for dim in ranked_bank.shape],
        "relay_rotation_indexing": (
            "Relay continuation legs use explicit_gammas[set_idx % row_count] "
            "with set_idx starting at 1; rows are shifted so leg 1 uses rank 1."
        ),
        "candidate_count": int(len(candidates)),
        "skipped": skipped,
        "selected_assignments": [
            {key: value for key, value in candidate.items() if key != "gammas"}
            for candidate in selected
        ],
    }
    return bank, metadata


def relay_explicit_gammas_from_ranked_bank(ranked_bank: np.ndarray) -> np.ndarray:
    if ranked_bank.ndim != 2:
        raise ValueError(
            f"Expected a 2D ranked gamma bank; got shape {ranked_bank.shape}."
        )
    if ranked_bank.shape[0] <= 1:
        return ranked_bank.copy()
    return np.vstack([ranked_bank[-1:], ranked_bank[:-1]])


def seed_dir_sort_key(path: Path) -> tuple[int, int | str]:
    suffix = path.name.removeprefix("seed_")
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, path.name)


def load_static_discrete_metrics(
    seed_dir: Path,
) -> tuple[dict[str, Any] | None, str | None, Path | None]:
    validation_path = seed_dir / "validation_metrics.json"
    validation_payload = load_json_if_dict(validation_path)
    if validation_payload is not None:
        validation_metrics = validation_payload.get("discrete")
        if isinstance(validation_metrics, dict):
            return (
                validation_metrics,
                "validation_metrics.json:discrete",
                validation_path,
            )

    params_path = seed_dir / "discrete_best_params.json"
    params_payload = load_json_if_dict(params_path)
    if params_payload is not None:
        metrics = params_payload.get("metrics")
        if isinstance(metrics, dict):
            return metrics, "discrete_best_params.json:metrics", params_path
        best_candidate = params_payload.get("best_candidate")
        if isinstance(best_candidate, dict) and isinstance(
            best_candidate.get("metrics"), dict
        ):
            return (
                best_candidate["metrics"],
                "discrete_best_params.json:best_candidate.metrics",
                params_path,
            )

    return None, None, None


def load_json_if_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path.resolve()}.")
    return payload


def load_static_gamma_vector(path: Path) -> np.ndarray:
    gammas = np.asarray(np.load(path.resolve()), dtype=np.float64)
    if gammas.ndim == 2:
        if gammas.shape[0] != 1:
            raise ValueError(
                "Expected 1D or single-row static gammas in "
                f"{path.resolve()}; got shape {gammas.shape}."
            )
        gammas = gammas[0]
    if gammas.ndim != 1:
        raise ValueError(
            "Expected 1D or single-row static gammas in "
            f"{path.resolve()}; got shape {gammas.shape}."
        )
    if not np.all(np.isfinite(gammas)):
        raise ValueError(
            f"Static gamma artifact contains non-finite values: {path.resolve()}"
        )
    return gammas


def find_circuits(circuits_dir: Path, selected_p_values: list[float]) -> list[Path]:
    wanted = {str(p) for p in selected_p_values}
    matched = []
    for path in sorted(
        circuits_dir.glob("circuit=bicycle_bivariate_144_12_12_memory_Z,*.stim")
    ):
        metadata = parse_name_metadata(path)
        if metadata.get("error_rate") in wanted:
            matched.append(path)

    matched = sorted(matched, key=lambda path: parse_name_metadata(path)["p"])
    found = {parse_name_metadata(path)["error_rate"] for path in matched}
    missing = sorted(float(p) for p in (wanted - found))
    if missing:
        raise FileNotFoundError(
            f"Missing 144_12_12 memory_Z circuits for p values: {missing} in {circuits_dir}."
        )
    return matched


def select_num_errors(p: float, errors_by_p: dict[float, int]) -> int:
    key = round(float(p), 3)
    if key not in errors_by_p:
        raise KeyError(f"No error budget configured for p={p}.")
    return int(errors_by_p[key])


def build_tasks(circuit_paths: list[Path]) -> list[sinter.Task]:
    tasks: list[sinter.Task] = []
    for circuit_path in circuit_paths:
        full_circuit = stim.Circuit.from_file(circuit_path)
        filtered_circuit = _filter_detectors_by_basis(full_circuit, BASIS_FILTER)
        dem = filtered_circuit.detector_error_model()
        metadata = parse_name_metadata(circuit_path)
        metadata["basis_filter_applied"] = BASIS_FILTER
        tasks.append(
            sinter.Task(
                circuit=filtered_circuit,
                detector_error_model=dem,
                json_metadata=metadata,
                collection_options=sinter.CollectionOptions(
                    max_errors=select_num_errors(metadata["p"], ERRORS_BY_P),
                    max_shots=MAX_SHOTS_SAFETY_CAP,
                ),
            )
        )
    return tasks


class RelayBPIterationSampler(Sampler):
    """Custom sinter sampler that aggregates iteration totals per batch."""

    def __init__(self, *, decoder: SinterDecoder_RelayBP):
        self.decoder = decoder

    def compiled_sampler_for_task(self, task: sinter.Task) -> CompiledSampler:
        return CompiledRelayBPIterationSampler(decoder=self.decoder, task=task)


class CompiledRelayBPIterationSampler(CompiledSampler):
    def __init__(self, *, decoder: SinterDecoder_RelayBP, task: sinter.Task):
        self.task = task
        self.decoder = decoder
        self.check_matrices = CheckMatrices.from_dem(
            task.detector_error_model,
            decomposed_hyperedges=decoder.decomposed_hyperedges,
            prune_decided_errors=decoder.prune_decided_errors,
            threshold=decoder.threshold,
        )
        self.observable_decoder = decoder.build_observable_decoder(self.check_matrices)
        self.stim_sampler = task.circuit.compile_detector_sampler()
        self.num_detectors = task.circuit.num_detectors
        self.num_observables = task.circuit.num_observables
        self.postselection_mask = task.postselection_mask
        self.postselected_observables_mask = None
        if task.postselected_observables_mask is not None:
            self.postselected_observables_mask = np.unpackbits(
                task.postselected_observables_mask,
                bitorder="little",
            )[: self.num_observables].astype(np.uint8)

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets_packed, actual_obs_packed = self.stim_sampler.sample(
            shots=suggested_shots,
            bit_packed=True,
            separate_observables=True,
        )
        num_sampled_shots = int(dets_packed.shape[0])

        if self.postselection_mask is not None:
            discarded_flags = np.any(dets_packed & self.postselection_mask, axis=1)
            num_discards_1 = int(np.count_nonzero(discarded_flags))
            if num_discards_1:
                kept_flags = ~discarded_flags
                dets_packed = dets_packed[kept_flags, :]
                actual_obs_packed = actual_obs_packed[kept_flags, :]
        else:
            num_discards_1 = 0

        num_kept_after_det = int(dets_packed.shape[0])
        if num_kept_after_det == 0:
            return sinter.AnonTaskStats(
                shots=num_sampled_shots,
                errors=0,
                discards=num_discards_1,
                seconds=time.monotonic() - t0,
                custom_counts=collections.Counter(),
            )

        syndromes = np.unpackbits(dets_packed, axis=1, bitorder="little")[
            :, : self.num_detectors
        ].astype(np.uint8)
        actual_obs = np.unpackbits(actual_obs_packed, axis=1, bitorder="little")[
            :, : self.num_observables
        ].astype(np.uint8)

        if self.check_matrices.syndrome_bias is not None:
            syndromes = (syndromes + self.check_matrices.syndrome_bias) % 2

        detailed_results = self.observable_decoder.decode_observables_detailed_batch(
            syndromes,
            parallel=self.decoder.parallel,
            progress_bar=self.decoder.show_progress,
            leave_progress_bar_on_finish=self.decoder.leave_progress_bar_on_finish,
        )

        predicted_obs = np.asarray(
            [
                np.asarray(detail.observables, dtype=np.uint8)
                for detail in detailed_results
            ],
            dtype=np.uint8,
        )

        if self.check_matrices.observables_bias is not None:
            predicted_obs = (predicted_obs + self.check_matrices.observables_bias) % 2

        kept_for_error = np.ones(num_kept_after_det, dtype=bool)
        if self.postselected_observables_mask is not None:
            mismatches = actual_obs ^ predicted_obs
            discard_obs = np.any(
                mismatches & self.postselected_observables_mask, axis=1
            )
            kept_for_error &= ~discard_obs
            num_discards_2 = int(np.count_nonzero(discard_obs))
        else:
            num_discards_2 = 0

        if np.any(kept_for_error):
            logical_errors = np.any(
                actual_obs[kept_for_error] != predicted_obs[kept_for_error],
                axis=1,
            )
            num_errors = int(np.count_nonzero(logical_errors))
            iteration_sum = int(
                sum(
                    int(detailed_results[idx].iterations)
                    for idx, keep in enumerate(kept_for_error)
                    if keep
                )
            )
            decoded_shots = int(np.count_nonzero(kept_for_error))
        else:
            num_errors = 0
            iteration_sum = 0
            decoded_shots = 0

        return sinter.AnonTaskStats(
            shots=num_sampled_shots,
            errors=num_errors,
            discards=num_discards_1 + num_discards_2,
            seconds=time.monotonic() - t0,
            custom_counts=collections.Counter(
                {
                    "decoded_shots": decoded_shots,
                    "iteration_sum": iteration_sum,
                }
            ),
        )


def build_custom_decoders(
    *,
    static_discrete_gammas: np.ndarray | None = None,
    static_continuous_gammas: np.ndarray | None = None,
    static_discrete_assignment_bank: np.ndarray | None = None,
    message_mix_bernoulli: (
        tuple[float, float, float, float, float, float] | None
    ) = None,
    message_mix_second_previous_bernoulli: tuple[float, float, float] | None = None,
) -> dict[str, Sampler]:
    common = dict(
        alpha=RELAY_ALPHA,
        gamma0=RELAY_GAMMA0,
        pre_iter=RELAY_PRE_ITER,
        num_sets=RELAY_NUM_SETS,
        set_max_iter=RELAY_SET_MAX_ITER,
        gamma_dist_interval=RELAY_GAMMA_DIST_INTERVAL,
        relay_posteriors=RELAY_POSTERIORS,
        stop_nconv=RELAY_STOP_NCONV,
        stopping_criterion="nconv",
        logging=False,
        show_progress=False,
        leave_progress_bar_on_finish=False,
    )
    decoders: dict[str, Sampler] = {
        "relay-bp5-r601-no-damping": RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                seed=RELAY_BASE_SEED,
                decoder_label="relay-bp5-r601-no-damping",
            )
        ),
        "relay-bp5-r601-damp085-095": RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                c_damp_dist_interval=RELAY_DAMPING_INTERVAL,
                seed=RELAY_BASE_SEED + 1,
                decoder_label="relay-bp5-r601-damp085-095",
            )
        ),
        "relay-bp5-r601-discrete-memory": RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                gamma_bernoulli=RELAY_BERNOULLI_GAMMA,
                seed=RELAY_BASE_SEED + 2,
                decoder_label="relay-bp5-r601-discrete-memory",
            )
        ),
        "relay-bp5-r601-discrete-memory-damp0p9": RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                gamma_bernoulli=RELAY_BERNOULLI_GAMMA,
                c_damp=RELAY_FIXED_DAMPING,
                seed=RELAY_BASE_SEED + 3,
                decoder_label="relay-bp5-r601-discrete-memory-damp0p9",
            )
        ),
        "relay-bp5-r601-discrete-memory-damp085-095": RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                gamma_bernoulli=RELAY_BERNOULLI_GAMMA,
                c_damp_dist_interval=RELAY_DAMPING_INTERVAL,
                seed=RELAY_BASE_SEED + 4,
                decoder_label="relay-bp5-r601-discrete-memory-damp085-095",
            )
        ),
        "relay-bp5-r601-damp0p9": RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                c_damp=RELAY_FIXED_DAMPING,
                seed=RELAY_BASE_SEED + 5,
                decoder_label="relay-bp5-r601-damp0p9",
            )
        ),
    }
    if static_discrete_gammas is not None:
        decoders[STATIC_DISCRETE_DECODER_NAME] = RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                explicit_gammas=static_discrete_gammas,
                seed=RELAY_BASE_SEED + 6,
                decoder_label=STATIC_DISCRETE_DECODER_NAME,
            )
        )
    if static_continuous_gammas is not None:
        decoders[STATIC_CONTINUOUS_DECODER_NAME] = RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                explicit_gammas=static_continuous_gammas,
                seed=RELAY_BASE_SEED + 7,
                decoder_label=STATIC_CONTINUOUS_DECODER_NAME,
            )
        )
    if static_discrete_assignment_bank is not None:
        decoders[STATIC_DISCRETE_TOPK_DAMP_DECODER_NAME] = RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                explicit_gammas=static_discrete_assignment_bank,
                c_damp=RELAY_FIXED_DAMPING,
                seed=RELAY_BASE_SEED + 8,
                decoder_label=STATIC_DISCRETE_TOPK_DAMP_DECODER_NAME,
            )
        )
    if message_mix_bernoulli is not None:
        message_mix_no_gamma = {
            **common,
            "gamma0": None,
            "message_mix_bernoulli": message_mix_bernoulli,
            "message_mix_second_previous_bernoulli": (
                message_mix_second_previous_bernoulli
            ),
            "seed": RELAY_BASE_SEED + 9,
            "decoder_label": MESSAGE_MIX_NO_GAMMA_DECODER_NAME,
        }
        decoders[MESSAGE_MIX_NO_GAMMA_DECODER_NAME] = RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(**message_mix_no_gamma)
        )
        message_mix_gamma0125 = {
            **common,
            "gamma0": RELAY_GAMMA0,
            "gamma_dist_interval": (RELAY_GAMMA0, RELAY_GAMMA0),
            "message_mix_bernoulli": message_mix_bernoulli,
            "message_mix_second_previous_bernoulli": (
                message_mix_second_previous_bernoulli
            ),
            "seed": RELAY_BASE_SEED + 10,
            "decoder_label": MESSAGE_MIX_GAMMA0125_DECODER_NAME,
        }
        decoders[MESSAGE_MIX_GAMMA0125_DECODER_NAME] = RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(**message_mix_gamma0125)
        )
    return decoders


def samples_to_df(samples: list[sinter.TaskStats]) -> pd.DataFrame:
    rows = []
    for stat in samples:
        circuit_name = stat.json_metadata["circuit"]
        detail_shots = int(stat.custom_counts.get("decoded_shots", 0))
        detail_iteration_sum = int(stat.custom_counts.get("iteration_sum", 0))
        rows.append(
            {
                "decoder": stat.decoder,
                "circuit": circuit_name,
                "memory": circuit_name.rsplit("_", 1)[-1],
                "p": float(stat.json_metadata["p"]),
                "shots": int(stat.shots),
                "errors": int(stat.errors),
                "seconds": float(stat.seconds),
                "basis_filter_applied": stat.json_metadata.get(
                    "basis_filter_applied", "none"
                ),
                "detail_shots": detail_shots,
                "detail_iteration_sum": detail_iteration_sum,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "decoder",
                "circuit",
                "memory",
                "p",
                "shots",
                "errors",
                "seconds",
                "basis_filter_applied",
                "detail_shots",
                "detail_iteration_sum",
                "ler",
                "stderr",
                "mean_iterations",
                "hit_target",
                "shot_cap_reached",
            ]
        )

    df = pd.DataFrame(rows)
    summary_df = (
        df.groupby(
            ["decoder", "circuit", "memory", "p", "basis_filter_applied"],
            as_index=False,
        )
        .agg(
            shots=("shots", "sum"),
            errors=("errors", "sum"),
            seconds=("seconds", "sum"),
            detail_shots=("detail_shots", "sum"),
            detail_iteration_sum=("detail_iteration_sum", "sum"),
        )
        .sort_values(["decoder", "p"])
        .reset_index(drop=True)
    )
    summary_df["ler"] = summary_df["errors"] / summary_df["shots"]
    summary_df["stderr"] = (
        (summary_df["ler"] * (1.0 - summary_df["ler"]) / summary_df["shots"]).clip(
            lower=0.0
        )
    ) ** 0.5
    summary_df["mean_iterations"] = np.where(
        summary_df["detail_shots"] > 0,
        summary_df["detail_iteration_sum"] / summary_df["detail_shots"],
        np.nan,
    )
    summary_df["hit_target"] = summary_df["errors"] >= summary_df["p"].map(ERRORS_BY_P)
    summary_df["shot_cap_reached"] = summary_df["shots"] >= MAX_SHOTS_SAFETY_CAP
    return summary_df


def plot_logical_error_rate(
    summary_df: pd.DataFrame,
    *,
    decoder_sequence: list[str],
    save_path: Path,
) -> None:
    if summary_df.empty:
        print(f"[warn] No samples available to plot for {save_path.name}.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    available = [
        name for name in decoder_sequence if name in set(summary_df["decoder"])
    ]
    for decoder_name in available:
        group = summary_df[summary_df["decoder"] == decoder_name].sort_values("p")
        ax.plot(group["p"], group["ler"], marker="o", linewidth=1.8, label=decoder_name)

    ax.set_title("Relay-BP-5 logical error performance (Z-basis filtered)")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_mean_iterations(
    summary_df: pd.DataFrame,
    *,
    decoder_sequence: list[str],
    save_path: Path,
) -> None:
    if summary_df.empty:
        print(f"[warn] No samples available to plot for {save_path.name}.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    available = [
        name for name in decoder_sequence if name in set(summary_df["decoder"])
    ]
    for decoder_name in available:
        group = summary_df[summary_df["decoder"] == decoder_name].sort_values("p")
        ax.plot(
            group["p"],
            group["mean_iterations"],
            marker="o",
            linewidth=1.8,
            label=decoder_name,
        )

    ax.set_title("Relay-BP-5 mean iterations per decoded shot")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("mean iterations")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_combined_summary(
    summary_df: pd.DataFrame,
    *,
    decoder_sequence: list[str],
    save_path: Path,
) -> None:
    if summary_df.empty:
        print(f"[warn] No samples available to plot for {save_path.name}.")
        return

    fig, (ax_ler, ax_iter) = plt.subplots(1, 2, figsize=(12, 5))
    available = [
        name for name in decoder_sequence if name in set(summary_df["decoder"])
    ]
    for decoder_name in available:
        group = summary_df[summary_df["decoder"] == decoder_name].sort_values("p")
        ax_ler.plot(
            group["p"], group["ler"], marker="o", linewidth=1.8, label=decoder_name
        )
        ax_iter.plot(
            group["p"],
            group["mean_iterations"],
            marker="o",
            linewidth=1.8,
            label=decoder_name,
        )

    ax_ler.set_title("Logical error rate")
    ax_ler.set_xlabel("physical error rate p")
    ax_ler.set_ylabel("logical error rate")
    ax_ler.set_yscale("log")
    ax_ler.grid(True, which="both", alpha=0.3)

    ax_iter.set_title("Mean iterations")
    ax_iter.set_xlabel("physical error rate p")
    ax_iter.set_ylabel("mean iterations")
    ax_iter.set_yscale("log")
    ax_iter.grid(True, which="both", alpha=0.3)
    ax_iter.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def build_setup_dict(
    *,
    repo_root: Path,
    output_dir: Path,
    plots_dir: Path,
    matched_circuits: list[Path],
    selected_p_values: list[float],
    decoder_sequence: list[str],
    num_workers: int,
    max_batch_size: int,
    static_discrete_gammas_metadata: dict[str, Any] | None,
    static_continuous_gammas_metadata: dict[str, Any] | None,
    static_discrete_assignment_bank_metadata: dict[str, Any] | None,
    message_mix_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "matched_circuits": [str(path) for path in matched_circuits],
        "selected_p_values": [float(p) for p in selected_p_values],
        "errors_by_p": {str(k): int(v) for k, v in ERRORS_BY_P.items()},
        "max_shots_safety_cap": int(MAX_SHOTS_SAFETY_CAP),
        "basis_filter": BASIS_FILTER,
        "relay_alpha": float(RELAY_ALPHA),
        "relay_gamma0": float(RELAY_GAMMA0),
        "relay_pre_iter": int(RELAY_PRE_ITER),
        "relay_num_sets": int(RELAY_NUM_SETS),
        "relay_total_legs": int(RELAY_NUM_SETS + 1),
        "relay_set_max_iter": int(RELAY_SET_MAX_ITER),
        "relay_gamma_dist_interval": list(RELAY_GAMMA_DIST_INTERVAL),
        "relay_stop_nconv": int(RELAY_STOP_NCONV),
        "relay_posteriors": bool(RELAY_POSTERIORS),
        "relay_fixed_damping": float(RELAY_FIXED_DAMPING),
        "relay_damping_interval": list(RELAY_DAMPING_INTERVAL),
        "relay_bernoulli_gamma": list(RELAY_BERNOULLI_GAMMA),
        "static_discrete_gammas": static_discrete_gammas_metadata,
        "static_continuous_gammas": static_continuous_gammas_metadata,
        "static_discrete_assignment_bank": static_discrete_assignment_bank_metadata,
        "message_mix": message_mix_metadata,
        "decoder_sequence": list(decoder_sequence),
        "num_workers": int(num_workers),
        "max_batch_size": int(max_batch_size),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
    }


def validate_resume_setup(setup_json_path: Path, current_setup: dict[str, Any]) -> None:
    if not setup_json_path.exists():
        return
    existing_setup = json.loads(setup_json_path.read_text(encoding="utf-8"))
    fields_to_compare = [
        "selected_p_values",
        "errors_by_p",
        "max_shots_safety_cap",
        "basis_filter",
        "relay_alpha",
        "relay_gamma0",
        "relay_pre_iter",
        "relay_num_sets",
        "relay_set_max_iter",
        "relay_gamma_dist_interval",
        "relay_stop_nconv",
        "relay_posteriors",
        "relay_damping_interval",
        "max_batch_size",
    ]
    mismatches = [
        field
        for field in fields_to_compare
        if existing_setup.get(field) != current_setup.get(field)
    ]
    existing_decoder_sequence = list(existing_setup.get("decoder_sequence", []))
    current_decoder_sequence = list(current_setup.get("decoder_sequence", []))
    decoder_sequence_compatible = (
        existing_decoder_sequence == current_decoder_sequence
        or current_decoder_sequence[: len(existing_decoder_sequence)]
        == existing_decoder_sequence
    )
    if not decoder_sequence_compatible:
        mismatches.append("decoder_sequence")

    if "relay_bernoulli_gamma" in existing_setup and existing_setup.get(
        "relay_bernoulli_gamma"
    ) != current_setup.get("relay_bernoulli_gamma"):
        mismatches.append("relay_bernoulli_gamma")

    if "relay_fixed_damping" in existing_setup and existing_setup.get(
        "relay_fixed_damping"
    ) != current_setup.get("relay_fixed_damping"):
        mismatches.append("relay_fixed_damping")

    for field in [
        "static_discrete_gammas",
        "static_continuous_gammas",
        "static_discrete_assignment_bank",
        "message_mix",
    ]:
        if existing_setup.get(field) != current_setup.get(field):
            mismatches.append(field)

    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise ValueError(
            "Existing performance outputs were created with different settings: "
            f"{mismatch_text}. Use --reset-data or choose a different output directory."
        )


def resolve_selected_decoder_steps(
    decoder_sequence: list[str],
    decoder_indices: list[int] | None,
) -> list[tuple[int, str]]:
    if decoder_indices is None:
        return [(index + 1, name) for index, name in enumerate(decoder_sequence)]

    total = len(decoder_sequence)
    invalid = [index for index in decoder_indices if index < 1 or index > total]
    if invalid:
        raise ValueError(
            f"Decoder indices out of range: {invalid}. Valid one-based indices are 1..{total}."
        )

    selected_steps: list[tuple[int, str]] = []
    seen: set[int] = set()
    for index in decoder_indices:
        if index in seen:
            continue
        seen.add(index)
        selected_steps.append((index, decoder_sequence[index - 1]))
    return selected_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing tests/testdata/bicycle_bivariate. Auto-detected by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for csv outputs and plots/. Defaults under examples/notebook_data/.",
    )
    parser.add_argument(
        "--reset-data",
        action="store_true",
        help="Delete prior csv/json outputs before running.",
    )
    parser.add_argument(
        "--print-progress",
        action="store_true",
        default=False,
        help="Enable sinter progress output.",
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=None,
        help="Subset or reordered decoder names to run.",
    )
    parser.add_argument(
        "--decoder-indices",
        nargs="+",
        type=int,
        default=DEFAULT_DECODER_INDICES,
        help=(
            "Optional one-based positions into the decoder list to actually run. "
            "Example: --decoder-indices 2 runs only the damping variant."
        ),
    )
    parser.add_argument(
        "--p-values",
        type=str,
        default=None,
        help="Optional comma-separated p values. Defaults to 0.001,0.002,0.003,0.004,0.005.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="Maximum sinter batch size per worker. Lower this if detailed decoding uses too much memory.",
    )
    parser.add_argument(
        "--static-discrete-gammas-path",
        type=Path,
        default=None,
        help="Optional .npy static discrete gamma artifact to add as a comparison decoder.",
    )
    parser.add_argument(
        "--static-continuous-gammas-path",
        type=Path,
        default=None,
        help="Optional .npy static continuous gamma artifact to add as a comparison decoder.",
    )
    parser.add_argument(
        "--static-discrete-assignment-bank-base",
        type=Path,
        default=None,
        help=(
            "Optional training output base containing seed_*/discrete_best_static_gammas.npy "
            "artifacts. The top static discrete assignments are stacked and rotated as a "
            "damped comparison decoder."
        ),
    )
    parser.add_argument(
        "--static-discrete-assignment-bank-top-k",
        type=int,
        default=DEFAULT_STATIC_DISCRETE_ASSIGNMENT_BANK_TOP_K,
        help="Number of ranked static discrete assignments to stack for the rotating bank.",
    )
    parser.add_argument(
        "--message-mix-best-params-path",
        type=Path,
        default=None,
        help=(
            "Optional best_params.json or training output base to append the "
            f"{MESSAGE_MIX_NO_GAMMA_DECODER_NAME} comparison decoder."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = (
        args.repo_root.resolve()
        if args.repo_root
        else find_repo_root(Path.cwd().resolve())
    )
    circuits_dir = repo_root / "tests" / "testdata" / "bicycle_bivariate"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root
        / "examples"
        / "notebook_data"
        / "bicycle_bivariate_144_12_12_relay_bp5_performance_cluster_Z_basis"
    )
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_stem = "bicycle_bivariate_144_12_12_relay_bp5_performance_cluster_Z_basis"
    resume_csv = output_dir / f"{save_stem}_resume.csv"
    summary_csv = output_dir / f"{save_stem}_summary.csv"
    flat_summary_csv = output_dir / f"{save_stem}_flat_summary.csv"
    setup_json = output_dir / f"{save_stem}_setup.json"

    if args.reset_data:
        for path in [resume_csv, summary_csv, flat_summary_csv, setup_json]:
            if path.exists():
                path.unlink()

    static_discrete_gammas, static_discrete_gammas_metadata = load_static_gammas(
        args.static_discrete_gammas_path
    )
    static_continuous_gammas, static_continuous_gammas_metadata = load_static_gammas(
        args.static_continuous_gammas_path
    )
    (
        static_discrete_assignment_bank,
        static_discrete_assignment_bank_metadata,
    ) = load_static_discrete_assignment_bank(
        args.static_discrete_assignment_bank_base,
        top_k=args.static_discrete_assignment_bank_top_k,
    )
    (
        message_mix_bernoulli,
        message_mix_second_previous_bernoulli,
        message_mix_metadata,
    ) = load_message_mix_best_params(args.message_mix_best_params_path)

    custom_decoders = build_custom_decoders(
        static_discrete_gammas=static_discrete_gammas,
        static_continuous_gammas=static_continuous_gammas,
        static_discrete_assignment_bank=static_discrete_assignment_bank,
        message_mix_bernoulli=message_mix_bernoulli,
        message_mix_second_previous_bernoulli=(
            message_mix_second_previous_bernoulli
        ),
    )
    default_decoder_sequence = list(DEFAULT_DECODER_SEQUENCE)
    if static_discrete_gammas is not None:
        default_decoder_sequence.append(STATIC_DISCRETE_DECODER_NAME)
    if static_continuous_gammas is not None:
        default_decoder_sequence.append(STATIC_CONTINUOUS_DECODER_NAME)
    if static_discrete_assignment_bank is not None:
        default_decoder_sequence.append(STATIC_DISCRETE_TOPK_DAMP_DECODER_NAME)
    if message_mix_bernoulli is not None:
        default_decoder_sequence.append(MESSAGE_MIX_NO_GAMMA_DECODER_NAME)
        default_decoder_sequence.append(MESSAGE_MIX_GAMMA0125_DECODER_NAME)
    decoder_sequence = args.decoders or default_decoder_sequence
    unknown = [name for name in decoder_sequence if name not in custom_decoders]
    if unknown:
        raise KeyError(f"Unknown decoders requested: {unknown}")
    selected_decoder_steps = resolve_selected_decoder_steps(
        decoder_sequence,
        args.decoder_indices,
    )
    selected_p_values = parse_float_csv(args.p_values) or list(SELECTED_P_VALUES)
    max_batch_size = max(1, int(args.max_batch_size))

    matched_circuits = find_circuits(circuits_dir, selected_p_values)
    tasks = build_tasks(matched_circuits)
    matched_metadata_df = pd.DataFrame(
        [parse_name_metadata(path) for path in matched_circuits]
    )
    matched_metadata_df["basis_filter_applied"] = BASIS_FILTER
    matched_metadata_df.to_csv(output_dir / "matched_circuits.csv", index=False)

    num_workers = resolve_num_workers()
    setup = build_setup_dict(
        repo_root=repo_root,
        output_dir=output_dir,
        plots_dir=plots_dir,
        matched_circuits=matched_circuits,
        selected_p_values=selected_p_values,
        decoder_sequence=decoder_sequence,
        num_workers=num_workers,
        max_batch_size=max_batch_size,
        static_discrete_gammas_metadata=static_discrete_gammas_metadata,
        static_continuous_gammas_metadata=static_continuous_gammas_metadata,
        static_discrete_assignment_bank_metadata=static_discrete_assignment_bank_metadata,
        message_mix_metadata=message_mix_metadata,
    )
    validate_resume_setup(setup_json, setup)
    setup_json.write_text(json.dumps(setup, indent=2, sort_keys=True), encoding="utf-8")

    print(f"repo_root   = {repo_root}")
    print(f"circuits_dir= {circuits_dir}")
    print(f"output_dir  = {output_dir}")
    print(f"plots_dir   = {plots_dir}")
    print(f"num_workers = {num_workers}")
    print(f"max_batch_size = {max_batch_size}")
    print("selected_p_values =", selected_p_values)
    print("errors_by_p =", ERRORS_BY_P)
    print("max_shots safety cap =", MAX_SHOTS_SAFETY_CAP)
    print("basis_filter =", BASIS_FILTER)
    print("relay_fixed_damping =", RELAY_FIXED_DAMPING)
    print("relay_bernoulli_gamma =", RELAY_BERNOULLI_GAMMA)
    print("static_discrete_gammas =", static_discrete_gammas_metadata)
    print("static_continuous_gammas =", static_continuous_gammas_metadata)
    print("static_discrete_assignment_bank =", static_discrete_assignment_bank_metadata)
    print("message_mix =", message_mix_metadata)
    print("decoders =", decoder_sequence)
    if args.decoder_indices is None:
        print("selected_decoder_indices = all")
    else:
        print("selected_decoder_indices =", args.decoder_indices)

    samples: list[sinter.TaskStats] = []
    summary_df = pd.DataFrame()

    for step_index, decoder_name in selected_decoder_steps:
        print(
            f"\n=== Running decoder {step_index}/{len(decoder_sequence)}: {decoder_name} ==="
        )
        samples = sinter.collect(
            tasks=tasks,
            decoders=[decoder_name],
            custom_decoders={decoder_name: custom_decoders[decoder_name]},
            num_workers=num_workers,
            print_progress=args.print_progress,
            save_resume_filepath=resume_csv,
            max_batch_size=max_batch_size,
        )

        write_stats(samples, summary_csv)
        summary_df = samples_to_df(samples)
        summary_df.to_csv(flat_summary_csv, index=False)

        plot_logical_error_rate(
            summary_df,
            decoder_sequence=decoder_sequence,
            save_path=plots_dir / f"step_{step_index:02d}_logical_error_rate.png",
        )
        plot_mean_iterations(
            summary_df,
            decoder_sequence=decoder_sequence,
            save_path=plots_dir / f"step_{step_index:02d}_mean_iterations.png",
        )
        plot_combined_summary(
            summary_df,
            decoder_sequence=decoder_sequence,
            save_path=plots_dir / f"step_{step_index:02d}_combined.png",
        )

    plot_logical_error_rate(
        summary_df,
        decoder_sequence=decoder_sequence,
        save_path=plots_dir / "logical_error_rate.png",
    )
    plot_mean_iterations(
        summary_df,
        decoder_sequence=decoder_sequence,
        save_path=plots_dir / "mean_iterations.png",
    )
    plot_combined_summary(
        summary_df,
        decoder_sequence=decoder_sequence,
        save_path=plots_dir / "combined_summary.png",
    )

    print("\nDone.")
    print(f"resume_csv       = {resume_csv}")
    print(f"summary_csv      = {summary_csv}")
    print(f"flat_summary_csv = {flat_summary_csv}")
    print(f"plots_dir        = {plots_dir}")


if __name__ == "__main__":
    main()
