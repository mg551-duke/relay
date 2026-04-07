# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import csv
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sinter  # type: ignore
import stim  # type: ignore

from ..noise import detect_data_qubits
from .decoders import sinter_decoders_from_specs
from .utils import write_stats


@dataclass(frozen=True)
class SinterFolderRunResult:
    matched_circuits: list[Path]
    decoder_names: list[str]
    samples: list[sinter.TaskStats]
    resume_csv: Path
    summary_csv: Path
    flat_csv: Path
    details_dir: Path | None


def parse_name_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"stim_path": str(path)}
    for part in path.stem.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        metadata[key] = value
    if "error_rate" in metadata:
        try:
            metadata["p"] = float(metadata["error_rate"])
        except ValueError:
            pass
    if "distance" in metadata:
        try:
            metadata["d"] = int(metadata["distance"])
        except ValueError:
            pass
    if "rounds" in metadata:
        try:
            metadata["r"] = int(metadata["rounds"])
        except ValueError:
            pass
    return metadata


def _normalize_basis_filter(basis_filter: str) -> str:
    normalized = basis_filter.strip().upper()
    if normalized in ("", "NONE"):
        return "none"
    if normalized not in ("X", "Z"):
        raise ValueError(
            f"Unsupported basis filter '{basis_filter}'. Expected one of: none, X, Z."
        )
    return normalized


def _filter_detectors_by_basis(circuit: stim.Circuit, basis: str) -> stim.Circuit:
    """Drop detectors that do not respond to the requested data-qubit basis."""

    pauli_error = "Z" if basis == "X" else "X"
    circuit = circuit.flattened()

    noiseless_circuit = circuit.without_noise()
    sampler = noiseless_circuit.compile_detector_sampler()
    reference_detectors, _ = sampler.sample(1, separate_observables=True)
    reference_detectors = reference_detectors[0, :]
    num_detectors = len(reference_detectors)

    detector_is_sensitive = np.full(num_detectors, False, dtype=bool)
    data_qubits = detect_data_qubits(noiseless_circuit)
    to_test = list(data_qubits)
    to_test_set = set(to_test)

    inst_idx = 0
    while to_test:
        for qubit in to_test:
            injected_circuit = stim.Circuit()
            injected_circuit += noiseless_circuit
            injected_circuit.insert(
                inst_idx,
                stim.CircuitInstruction(pauli_error + "_ERROR", [qubit], [1.0]),
            )
            injected_sampler = injected_circuit.compile_detector_sampler()
            injected_detectors, _ = injected_sampler.sample(1, separate_observables=True)
            injected_detectors = injected_detectors[0, :]
            detectors_flipped = np.where(reference_detectors != injected_detectors)
            detector_is_sensitive[detectors_flipped] = True

        to_test = []
        for inst in noiseless_circuit[inst_idx:]:
            inst_idx += 1
            if inst.name.startswith("R") or inst.name.startswith("M"):
                to_test = list(to_test_set)
                break

    filtered_circuit = stim.Circuit()
    detector_idx = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            keep = detector_is_sensitive[detector_idx]
            detector_idx += 1
            if not keep:
                continue
        filtered_circuit.append(inst)
    return filtered_circuit


def find_circuits(
    circuits_dir: Path,
    filename_contains: str,
    max_circuits: int | None = None,
) -> list[Path]:
    paths = sorted(circuits_dir.glob("*.stim"))
    needles = [needle.strip() for needle in filename_contains.split(",") if needle.strip()]
    if needles:
        paths = [path for path in paths if all(needle in path.name for needle in needles)]
    if max_circuits is not None:
        paths = paths[:max_circuits]
    if not paths:
        raise FileNotFoundError(
            f"No .stim files matched in {circuits_dir!s} with filter {filename_contains!r}"
        )
    return paths


def load_decoder_specs(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        specs = json.load(f)
    if not isinstance(specs, list):
        raise TypeError("Decoder config must contain a JSON list of decoder specs.")
    return specs


def build_tasks(
    circuit_paths: list[Path],
    max_shots: int,
    basis_filter: str = "none",
) -> list[sinter.Task]:
    normalized_basis_filter = _normalize_basis_filter(basis_filter)
    tasks: list[sinter.Task] = []
    for circuit_path in circuit_paths:
        circuit = stim.Circuit.from_file(circuit_path)
        if normalized_basis_filter != "none":
            circuit = _filter_detectors_by_basis(circuit, normalized_basis_filter)
        dem = circuit.detector_error_model()
        metadata = parse_name_metadata(circuit_path)
        metadata["basis_filter_applied"] = normalized_basis_filter
        tasks.append(
            sinter.Task(
                circuit=circuit,
                detector_error_model=dem,
                json_metadata=metadata,
                collection_options=sinter.CollectionOptions(max_shots=max_shots),
            )
        )
    return tasks


def write_flat_summary(samples: list[sinter.TaskStats], out_csv: Path) -> None:
    fieldnames = [
        "decoder",
        "shots",
        "errors",
        "logical_error_rate",
        "seconds",
        "strong_id",
        "circuit",
        "p",
        "distance",
        "rounds",
        "basis",
        "basis_filter_applied",
        "noise_model",
        "stim_path",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stat in samples:
            metadata = dict(getattr(stat, "json_metadata", {}) or {})
            writer.writerow(
                {
                    "decoder": stat.decoder,
                    "shots": stat.shots,
                    "errors": stat.errors,
                    "logical_error_rate": (stat.errors / stat.shots) if stat.shots else 0.0,
                    "seconds": getattr(stat, "seconds", ""),
                    "strong_id": getattr(stat, "strong_id", ""),
                    "circuit": metadata.get("circuit", ""),
                    "p": metadata.get("p", ""),
                    "distance": metadata.get("distance", metadata.get("d", "")),
                    "rounds": metadata.get("rounds", metadata.get("r", "")),
                    "basis": metadata.get("basis", ""),
                    "basis_filter_applied": metadata.get("basis_filter_applied", "none"),
                    "noise_model": metadata.get("noise_model", ""),
                    "stim_path": metadata.get("stim_path", ""),
                }
            )


def run_sinter_folder_benchmark(
    *,
    circuits_dir: Path,
    decoder_config: Path,
    outdir: Path,
    filename_contains: str = "",
    max_shots: int = 10_000,
    num_workers: int | None = None,
    save_name: str = "stim_sinter",
    basis_filter: str = "none",
    max_circuits: int | None = None,
    print_progress: bool = False,
    include_decode_result: bool = True,
) -> SinterFolderRunResult:
    outdir.mkdir(parents=True, exist_ok=True)
    details_dir = outdir / "details" if include_decode_result else None
    if details_dir is not None:
        details_dir.mkdir(parents=True, exist_ok=True)

    specs = load_decoder_specs(decoder_config)
    custom_decoders = sinter_decoders_from_specs(
        specs,
        include_decode_result=include_decode_result,
        details_dir=str(details_dir) if details_dir is not None else None,
    )
    decoder_names = list(custom_decoders.keys())

    circuit_paths = find_circuits(circuits_dir, filename_contains, max_circuits=max_circuits)
    tasks = build_tasks(circuit_paths, max_shots, basis_filter=basis_filter)

    actual_num_workers = (
        num_workers
        if num_workers is not None
        else max(1, multiprocessing.cpu_count() // 2)
    )
    resume_csv = outdir / f"{save_name}_resume.csv"
    summary_csv = outdir / f"{save_name}_summary.csv"
    flat_csv = outdir / f"{save_name}_flat_summary.csv"

    samples = sinter.collect(
        tasks=tasks,
        decoders=decoder_names,
        custom_decoders=custom_decoders,
        num_workers=actual_num_workers,
        print_progress=print_progress,
        save_resume_filepath=resume_csv,
    )

    write_stats(samples, summary_csv)
    write_flat_summary(samples, flat_csv)

    return SinterFolderRunResult(
        matched_circuits=circuit_paths,
        decoder_names=decoder_names,
        samples=samples,
        resume_csv=resume_csv,
        summary_csv=summary_csv,
        flat_csv=flat_csv,
        details_dir=details_dir,
    )
