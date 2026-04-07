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

import hashlib
import json
import os
import pathlib
import uuid
from typing import Any

import numpy as np
import relay_bp
import sinter
import stim
from sinter import CompiledDecoder, Decoder

from .check_matrices import CheckMatrices


class SinterCompiledDecoder_BP(CompiledDecoder):
    """Compiled decoder used by Sinter.

    The default fast path mirrors the original repository and only returns the
    observable predictions needed by Sinter. When detailed results are enabled,
    the wrapper switches to `decode_observables_detailed_batch` and persists the
    richer per-shot data exposed by the native Rust decoders.
    """

    def __init__(
        self,
        observable_decoder: relay_bp.ObservableDecoderRunner,
        check_matrices: CheckMatrices,
        parallel: bool = False,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        decoder_label: str = "decoder",
        record_details: bool = False,
        details_dir: str | None = None,
        dem_hash: str | None = None,
    ):
        self.observable_decoder = observable_decoder
        self.parallel = parallel
        self.check_matrices = check_matrices
        self.show_progress = show_progress
        self.leave_progress_bar_on_finish = leave_progress_bar_on_finish
        self.decoder_label = decoder_label
        self.record_details = record_details
        self.details_dir = pathlib.Path(details_dir) if details_dir is not None else None
        self.dem_hash = dem_hash or "unknown_dem"
        self._detail_path: pathlib.Path | None = None

    def _ensure_detail_path(self) -> pathlib.Path | None:
        if self.details_dir is None:
            return None
        if self._detail_path is None:
            self.details_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{self.decoder_label}_pid{os.getpid()}_{self.dem_hash}_"
                f"{uuid.uuid4().hex[:8]}.jsonl"
            )
            self._detail_path = self.details_dir / filename
        return self._detail_path

    def _serialize_detail(self, detail: Any) -> dict[str, Any]:
        physical = detail.physical_decode_result
        record: dict[str, Any] = {
            "decoder": self.decoder_label,
            "dem_hash": self.dem_hash,
            "converged": bool(detail.converged),
            "iterations": int(detail.iterations),
            "observables": np.asarray(detail.observables, dtype=np.uint8).tolist(),
        }
        if physical is None:
            record.update(
                {
                    "fallback_used": None,
                    "converged_stage_index": None,
                    "stage_names": [],
                    "stage_iterations": [],
                    "stage_converged": [],
                    "posterior_norm": None,
                }
            )
            return record

        posterior = np.asarray(physical.posterior_ratios, dtype=np.float64)
        record.update(
            {
                "physical_success": bool(physical.success),
                "physical_iterations": int(physical.iterations),
                "decoding": np.asarray(physical.decoding, dtype=np.uint8).tolist(),
                "decoded_detectors": np.asarray(physical.decoded_detectors, dtype=np.uint8).tolist(),
                "fallback_used": physical.fallback_used,
                "converged_stage_index": physical.converged_stage_index,
                "stage_names": list(physical.stage_names),
                "stage_iterations": list(physical.stage_iterations),
                "stage_converged": list(physical.stage_converged),
                "posterior_norm": float(np.linalg.norm(posterior)) if posterior.size else 0.0,
            }
        )
        return record

    def _write_detail_records(self, detailed_results: list[Any]) -> None:
        detail_path = self._ensure_detail_path()
        if detail_path is None:
            return
        with open(detail_path, "a", encoding="utf-8") as f:
            for detail in detailed_results:
                f.write(json.dumps(self._serialize_detail(detail), sort_keys=True))
                f.write("\n")

    def _decode_predictions(self, syndromes: np.ndarray) -> np.ndarray:
        if self.record_details or self.details_dir is not None:
            detailed_results = self.observable_decoder.decode_observables_detailed_batch(
                syndromes,
                parallel=self.parallel,
                progress_bar=self.show_progress,
                leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
            )
            predictions = np.asarray(
                [np.asarray(detail.observables, dtype=np.uint8) for detail in detailed_results],
                dtype=np.uint8,
            )
            self._write_detail_records(detailed_results)
            return predictions

        return self.observable_decoder.decode_observables_batch(
            syndromes,
            parallel=self.parallel,
            progress_bar=self.show_progress,
            leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
        )

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        syndromes = np.unpackbits(
            bit_packed_detection_event_data, bitorder="little", axis=1
        ).astype(np.uint8)

        if self.check_matrices.syndrome_bias is not None:
            syndromes = (syndromes + self.check_matrices.syndrome_bias) % 2

        predictions = self._decode_predictions(syndromes)

        if self.check_matrices.observables_bias is not None:
            predictions = (predictions + self.check_matrices.observables_bias) % 2

        return np.packbits(predictions, axis=1, bitorder="little")


class SinterDecoder_BaseBP(Decoder):
    def __init__(
        self,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        include_decode_result: bool = False,
        details_dir: str | None = None,
        decoder_label: str | None = None,
    ):
        self.parallel = parallel
        self.decomposed_hyperedges = decomposed_hyperedges
        self.prune_decided_errors = prune_decided_errors
        self.threshold = threshold
        self.show_progress = show_progress
        self.leave_progress_bar_on_finish = leave_progress_bar_on_finish
        self.include_decode_result = include_decode_result
        self.details_dir = details_dir
        self.decoder_label = decoder_label or self.__class__.__name__

    def build_observable_decoder(
        self, dem: stim.DetectorErrorModel
    ) -> relay_bp.ObservableDecoderRunner:
        raise NotImplementedError("Not yet implemented")

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> CompiledDecoder:
        check_matrices = CheckMatrices.from_dem(
            dem,
            decomposed_hyperedges=self.decomposed_hyperedges,
            prune_decided_errors=self.prune_decided_errors,
            threshold=self.threshold,
        )
        observable_decoder_runner = self.build_observable_decoder(check_matrices)
        dem_hash = hashlib.sha256(str(dem).encode("utf-8")).hexdigest()[:16]
        return SinterCompiledDecoder_BP(
            observable_decoder_runner,
            check_matrices=check_matrices,
            parallel=self.parallel,
            show_progress=self.show_progress,
            leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
            decoder_label=self.decoder_label,
            record_details=self.include_decode_result,
            details_dir=self.details_dir,
            dem_hash=dem_hash,
        )

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        dem = stim.DetectorErrorModel.from_file(dem_path)
        check_matrices = CheckMatrices.from_dem(
            dem,
            decomposed_hyperedges=self.decomposed_hyperedges,
            prune_decided_errors=self.prune_decided_errors,
            threshold=self.threshold,
        )
        observable_decoder = self.build_observable_decoder(check_matrices)
        compiled = SinterCompiledDecoder_BP(
            observable_decoder,
            check_matrices=check_matrices,
            parallel=self.parallel,
            show_progress=self.show_progress,
            leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
            decoder_label=self.decoder_label,
            record_details=self.include_decode_result,
            details_dir=self.details_dir,
            dem_hash=hashlib.sha256(str(dem).encode("utf-8")).hexdigest()[:16],
        )

        syndromes = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format="b8",
            num_detectors=dem.num_detectors,
            bit_packed=False,
        ).astype(np.uint8)

        if check_matrices.syndrome_bias is not None:
            syndromes = (syndromes + check_matrices.syndrome_bias) % 2

        predictions = compiled._decode_predictions(syndromes)

        if check_matrices.observables_bias is not None:
            predictions = (predictions + check_matrices.observables_bias) % 2

        stim.write_shot_data_file(
            data=np.packbits(predictions, axis=1, bitorder="little"),
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=dem.num_observables,
        )


class SinterDecoder_RelayBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        alpha: float | None = None,
        gamma0: float = 0.1,
        pre_iter: int = 60,
        num_sets: int = 60,
        set_max_iter: int = 60,
        gamma_dist_interval: tuple[float, float] = (-0.24, 0.66),
        explicit_gammas: np.ndarray | None = None,
        stop_nconv: int = 5,
        stopping_criterion: str = "nconv",
        logging: bool = False,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        include_decode_result: bool = False,
        details_dir: str | None = None,
        decoder_label: str | None = None,
    ):
        self.alpha = alpha
        self.gamma0 = gamma0
        self.pre_iter = pre_iter
        self.num_sets = num_sets
        self.set_max_iter = set_max_iter
        self.gamma_dist_interval = tuple(gamma_dist_interval)
        self.explicit_gammas = explicit_gammas
        self.stop_nconv = stop_nconv
        self.stopping_criterion = stopping_criterion
        self.logging = logging
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            show_progress=show_progress,
            leave_progress_bar_on_finish=leave_progress_bar_on_finish,
            include_decode_result=include_decode_result,
            details_dir=details_dir,
            decoder_label=decoder_label,
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.RelayDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            alpha=None if self.alpha == 0.0 else self.alpha,
            gamma0=self.gamma0,
            pre_iter=self.pre_iter,
            num_sets=self.num_sets,
            set_max_iter=self.set_max_iter,
            gamma_dist_interval=self.gamma_dist_interval,
            explicit_gammas=self.explicit_gammas,
            stop_nconv=self.stop_nconv,
            stopping_criterion=self.stopping_criterion,
            logging=self.logging,
        )
        return relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=self.include_decode_result or self.details_dir is not None,
        )


class SinterDecoder_MemBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        max_iter: int = 100,
        alpha: float | None = None,
        gamma0: float = 0.1,
        c_damp: float | None = None,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        include_decode_result: bool = False,
        details_dir: str | None = None,
        decoder_label: str | None = None,
    ):
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma0 = gamma0
        self.c_damp = c_damp
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            show_progress=show_progress,
            leave_progress_bar_on_finish=leave_progress_bar_on_finish,
            include_decode_result=include_decode_result,
            details_dir=details_dir,
            decoder_label=decoder_label,
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.MinSumBPDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            max_iter=self.max_iter,
            alpha=None if self.alpha == 0.0 else self.alpha,
            c_damp=self.c_damp,
            gamma0=self.gamma0,
        )
        return relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=self.include_decode_result or self.details_dir is not None,
        )


class SinterDecoder_MSLBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        max_iter: int = 100,
        alpha: float | None = None,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        include_decode_result: bool = False,
        details_dir: str | None = None,
        decoder_label: str | None = None,
    ):
        self.max_iter = max_iter
        self.alpha = alpha
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            show_progress=show_progress,
            leave_progress_bar_on_finish=leave_progress_bar_on_finish,
            include_decode_result=include_decode_result,
            details_dir=details_dir,
            decoder_label=decoder_label,
        )

    def build_observable_decoder(
        self,
        check_matrices: CheckMatrices,
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.MinSumBPDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            max_iter=self.max_iter,
            alpha=None if self.alpha == 0.0 else self.alpha,
            c_damp=None,
            gamma0=None,
        )
        return relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=self.include_decode_result or self.details_dir is not None,
        )


class SinterDecoder_DampedBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        max_iter: int = 100,
        alpha: float | None = None,
        c_damp: float = 0.5,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        include_decode_result: bool = False,
        details_dir: str | None = None,
        decoder_label: str | None = None,
    ):
        self.max_iter = max_iter
        self.alpha = alpha
        self.c_damp = c_damp
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            show_progress=show_progress,
            leave_progress_bar_on_finish=leave_progress_bar_on_finish,
            include_decode_result=include_decode_result,
            details_dir=details_dir,
            decoder_label=decoder_label,
        )

    def build_observable_decoder(
        self,
        check_matrices: CheckMatrices,
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.MinSumBPDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            max_iter=self.max_iter,
            alpha=None if self.alpha == 0.0 else self.alpha,
            c_damp=self.c_damp,
            gamma0=None,
        )
        return relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=self.include_decode_result or self.details_dir is not None,
        )


class SinterDecoder_BPGD(SinterDecoder_BaseBP):
    def __init__(
        self,
        pre_iter: int = 0,
        initial_decimation_percentage: float = 0.0,
        r_low: int = 5,
        t_low: int = 3,
        r_high: int = 1000,
        t_high: int = 5,
        alpha: float | None = None,
        c_damp: float | None = None,
        fallback_kind: str = "none",
        fallback_gamma0: float = 0.1,
        fallback_pre_iter: int = 80,
        fallback_num_sets: int = 60,
        fallback_set_max_iter: int = 60,
        fallback_gamma_dist_interval: tuple[float, float] = (-0.24, 0.66),
        fallback_explicit_gammas: np.ndarray | None = None,
        fallback_stop_nconv: int = 5,
        fallback_stopping_criterion: str = "nconv",
        fallback_logging: bool = False,
        fallback_seed: int = 0,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        include_decode_result: bool = False,
        details_dir: str | None = None,
        decoder_label: str | None = None,
    ):
        self.pre_iter = pre_iter
        self.initial_decimation_percentage = initial_decimation_percentage
        self.r_low = r_low
        self.t_low = t_low
        self.r_high = r_high
        self.t_high = t_high
        self.alpha = alpha
        self.c_damp = c_damp
        self.fallback_kind = fallback_kind
        self.fallback_gamma0 = fallback_gamma0
        self.fallback_pre_iter = fallback_pre_iter
        self.fallback_num_sets = fallback_num_sets
        self.fallback_set_max_iter = fallback_set_max_iter
        self.fallback_gamma_dist_interval = tuple(fallback_gamma_dist_interval)
        self.fallback_explicit_gammas = fallback_explicit_gammas
        self.fallback_stop_nconv = fallback_stop_nconv
        self.fallback_stopping_criterion = fallback_stopping_criterion
        self.fallback_logging = fallback_logging
        self.fallback_seed = fallback_seed
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            show_progress=show_progress,
            leave_progress_bar_on_finish=leave_progress_bar_on_finish,
            include_decode_result=include_decode_result,
            details_dir=details_dir,
            decoder_label=decoder_label,
        )

    def build_observable_decoder(
        self,
        check_matrices: CheckMatrices,
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.BPGDDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            pre_iter=self.pre_iter,
            initial_decimation_percentage=self.initial_decimation_percentage,
            r_low=self.r_low,
            t_low=self.t_low,
            r_high=self.r_high,
            t_high=self.t_high,
            alpha=None if self.alpha == 0.0 else self.alpha,
            c_damp=self.c_damp,
            fallback_kind=self.fallback_kind,
            fallback_gamma0=self.fallback_gamma0,
            fallback_pre_iter=self.fallback_pre_iter,
            fallback_num_sets=self.fallback_num_sets,
            fallback_set_max_iter=self.fallback_set_max_iter,
            fallback_gamma_dist_interval=self.fallback_gamma_dist_interval,
            fallback_explicit_gammas=self.fallback_explicit_gammas,
            fallback_stop_nconv=self.fallback_stop_nconv,
            fallback_stopping_criterion=self.fallback_stopping_criterion,
            fallback_logging=self.fallback_logging,
            fallback_seed=self.fallback_seed,
        )
        return relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=True,
        )


def _common_decoder_kwargs(decoder_kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    common_keys = {
        "parallel",
        "decomposed_hyperedges",
        "prune_decided_errors",
        "threshold",
        "show_progress",
        "leave_progress_bar_on_finish",
        "include_decode_result",
        "details_dir",
    }
    common = {key: decoder_kwargs[key] for key in common_keys if key in decoder_kwargs}
    specific = {key: value for key, value in decoder_kwargs.items() if key not in common_keys}
    return common, specific


def _canonical_spec_kind(kind: str) -> str:
    return kind


def decoder_from_spec(spec: dict[str, Any], **decoder_kwargs: Any) -> Decoder:
    """Instantiate a Sinter decoder from a JSON-style stage spec.

    The helper intentionally mirrors the naming and staged decoder structure
    discussed in the project chat. It accepts the experiment JSON used in the
    surrounding scripts and turns it into a Sinter custom decoder object.
    """
    name = spec["name"]
    stages = spec["stages"]
    common, _ = _common_decoder_kwargs(decoder_kwargs)

    if len(stages) == 1:
        kind = _canonical_spec_kind(stages[0]["kind"])
        params = dict(stages[0].get("params", {}))
        if kind == "msl-bp":
            return SinterDecoder_MSLBP(decoder_label=name, **params, **common)
        if kind == "damped_BP":
            return SinterDecoder_DampedBP(decoder_label=name, **params, **common)
        if kind == "mem-bp":
            return SinterDecoder_MemBP(decoder_label=name, **params, **common)
        if kind == "relay-bp":
            return SinterDecoder_RelayBP(decoder_label=name, **params, **common)
        if kind == "BPGD":
            params = {
                "pre_iter": int(params.get("pre_iter", 0)),
                "initial_decimation_percentage": float(
                    params.get("initial_decimation_percentage", 0.0)
                ),
                "r_low": int(params.get("r_low", params.get("R_low", 5))),
                "t_low": int(params.get("t_low", params.get("T_low", 3))),
                "r_high": int(params.get("r_high", params.get("R_high", 1000))),
                "t_high": int(params.get("t_high", params.get("T_high", 5))),
                "alpha": params.get("alpha", None),
                "c_damp": params.get("c_damp", None),
            }
            return SinterDecoder_BPGD(decoder_label=name, **params, **common)
        raise ValueError(f"Unsupported single-stage decoder kind: {kind}")

    if (
        len(stages) == 2
        and _canonical_spec_kind(stages[0]["kind"]) == "BPGD"
        and _canonical_spec_kind(stages[1]["kind"]) == "relay-bp"
    ):
        params = dict(stages[0].get("params", {}))
        fallback = dict(stages[1].get("params", {}))
        return SinterDecoder_BPGD(
            decoder_label=name,
            fallback_kind="relay-bp",
            fallback_gamma0=float(fallback.get("gamma0", 0.1)),
            fallback_pre_iter=int(fallback.get("pre_iter", 80)),
            fallback_num_sets=int(fallback.get("num_sets", 60)),
            fallback_set_max_iter=int(fallback.get("set_max_iter", 60)),
            fallback_gamma_dist_interval=tuple(
                fallback.get("gamma_dist_interval", (-0.24, 0.66))
            ),
            fallback_explicit_gammas=fallback.get("explicit_gammas", None),
            fallback_stop_nconv=int(fallback.get("stop_nconv", 5)),
            fallback_stopping_criterion=str(
                fallback.get("stopping_criterion", "nconv")
            ),
            fallback_logging=bool(fallback.get("logging", False)),
            fallback_seed=int(fallback.get("seed", 0)),
            pre_iter=int(params.get("pre_iter", 0)),
            initial_decimation_percentage=float(
                params.get("initial_decimation_percentage", 0.0)
            ),
            r_low=int(params.get("r_low", params.get("R_low", 5))),
            t_low=int(params.get("t_low", params.get("T_low", 3))),
            r_high=int(params.get("r_high", params.get("R_high", 1000))),
            t_high=int(params.get("t_high", params.get("T_high", 5))),
            alpha=params.get("alpha", None),
            c_damp=params.get("c_damp", None),
            **common,
        )

    if len(stages) == 2 and _canonical_spec_kind(stages[0]["kind"]) == "BPGD":
        raise ValueError(
            "Native staged BPGD specs only support a relay-bp fallback stage; "
            f"got {stages[1]['kind']!r} in {name!r}."
        )

    raise ValueError(f"Unsupported staged decoder spec: {spec}")


def sinter_decoders_from_specs(
    specs: list[dict[str, Any]], **decoder_kwargs: Any
) -> dict[str, Decoder]:
    return {
        spec["name"]: decoder_from_spec(spec, **decoder_kwargs)
        for spec in specs
    }


def sinter_decoders(**decoder_kwargs: dict) -> dict[str, Decoder]:
    common, specific = _common_decoder_kwargs(decoder_kwargs)

    max_iter = int(specific.get("max_iter", 100))
    alpha = specific.get("alpha", None)
    gamma0 = float(specific.get("gamma0", 0.1))
    c_damp = specific.get("c_damp", 0.5)

    relay_kwargs = {
        "alpha": alpha,
        "gamma0": gamma0,
        "pre_iter": int(specific.get("pre_iter", 60)),
        "num_sets": int(specific.get("num_sets", 60)),
        "set_max_iter": int(specific.get("set_max_iter", 60)),
        "gamma_dist_interval": tuple(
            specific.get("gamma_dist_interval", (-0.24, 0.66))
        ),
        "explicit_gammas": specific.get("explicit_gammas", None),
        "stop_nconv": int(specific.get("stop_nconv", 5)),
        "stopping_criterion": str(specific.get("stopping_criterion", "nconv")),
        "logging": bool(specific.get("logging", False)),
    }

    bpgd_kwargs = {
        "pre_iter": int(specific.get("pre_iter", 0)),
        "initial_decimation_percentage": float(
            specific.get("initial_decimation_percentage", 0.0)
        ),
        "r_low": int(specific.get("r_low", specific.get("R_low", 5))),
        "t_low": int(specific.get("t_low", specific.get("T_low", 3))),
        "r_high": int(specific.get("r_high", specific.get("R_high", 1000))),
        "t_high": int(specific.get("t_high", specific.get("T_high", 5))),
        "alpha": alpha,
        "c_damp": specific.get("c_damp", None),
    }

    return {
        "relay-bp": SinterDecoder_RelayBP(decoder_label="relay-bp", **relay_kwargs, **common),
        "mem-bp": SinterDecoder_MemBP(
            decoder_label="mem-bp",
            max_iter=max_iter,
            alpha=alpha,
            gamma0=gamma0,
            **common,
        ),
        "msl-bp": SinterDecoder_MSLBP(
            decoder_label="msl-bp",
            max_iter=max_iter,
            alpha=alpha,
            **common,
        ),
        "damped_BP": SinterDecoder_DampedBP(
            decoder_label="damped_BP",
            max_iter=max_iter,
            alpha=alpha,
            c_damp=float(c_damp),
            **common,
        ),
        "BPGD": SinterDecoder_BPGD(
            decoder_label="BPGD",
            **bpgd_kwargs,
            **common,
        ),
        "BPGD-relay-bp": SinterDecoder_BPGD(
            decoder_label="BPGD-relay-bp",
            fallback_kind="relay-bp",
            fallback_gamma0=gamma0,
            fallback_pre_iter=relay_kwargs["pre_iter"],
            fallback_num_sets=relay_kwargs["num_sets"],
            fallback_set_max_iter=relay_kwargs["set_max_iter"],
            fallback_gamma_dist_interval=relay_kwargs["gamma_dist_interval"],
            fallback_explicit_gammas=relay_kwargs["explicit_gammas"],
            fallback_stop_nconv=relay_kwargs["stop_nconv"],
            fallback_stopping_criterion=relay_kwargs["stopping_criterion"],
            fallback_logging=relay_kwargs["logging"],
            **bpgd_kwargs,
            **common,
        ),
    }
