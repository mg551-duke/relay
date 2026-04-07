# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json
import shutil
from pathlib import Path

import pytest
import sinter
import stim

from relay_bp.stim import (
    decoder_from_spec,
    run_sinter_folder_benchmark,
    sinter_decoders_from_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sinter_decoders_from_native_surface_and_bivariate_specs():
    surface_specs = json.loads(
        (REPO_ROOT / "configs" / "decoder_specs_surface.json").read_text(encoding="utf-8")
    )
    bivariate_specs = json.loads(
        (REPO_ROOT / "configs" / "decoder_specs_bivariate.json").read_text(encoding="utf-8")
    )

    surface_decoders = sinter_decoders_from_specs(surface_specs)
    bivariate_decoders = sinter_decoders_from_specs(bivariate_specs)

    expected_names = {
        "msl-bp_no_decimation",
        "damped_BP_no_decimation",
        "mem-bp_no_decimation",
        "relay-bp_no_decimation",
        "BPGD_Rl_5_Tl_3_Rh_1000_Th_5",
        "BPGD-relay-bp_Rl_5_Tl_3_Rh_1000_Th_5",
    }
    assert set(surface_decoders) == expected_names
    assert set(bivariate_decoders) == expected_names


def test_decoder_from_spec_rejects_nonrelay_bpgd_fallback():
    with pytest.raises(ValueError, match="only support a relay-bp fallback stage"):
        decoder_from_spec(
            {
                "name": "bad_bpgd_regular",
                "stages": [
                    {"kind": "BPGD", "params": {"R_low": 5, "T_low": 3}},
                    {"kind": "msl-bp", "params": {"max_iter": 100}},
                ],
            }
        )


def test_decoder_from_spec_preserves_bpgd_warmup_and_initial_high_decimation():
    decoder = decoder_from_spec(
        {
            "name": "warm_bpgd",
            "stages": [
                {
                    "kind": "BPGD",
                    "params": {
                        "pre_iter": 7,
                        "initial_decimation_percentage": 12.5,
                        "R_low": 3,
                        "T_low": 2,
                        "R_high": 9,
                        "T_high": 4,
                    },
                }
            ],
        }
    )

    assert decoder.pre_iter == 7
    assert decoder.initial_decimation_percentage == 12.5
    assert decoder.r_low == 3
    assert decoder.t_low == 2
    assert decoder.r_high == 9
    assert decoder.t_high == 4


def test_run_sinter_folder_benchmark_smoke(monkeypatch):
    def fake_collect(
        *,
        tasks,
        decoders,
        custom_decoders,
        num_workers,
        print_progress,
        save_resume_filepath,
    ):
        task_list = list(tasks)
        assert len(task_list) == 1
        assert decoders == ["BPGD-relay-bp_smoke"]
        stat = sinter.TaskStats(
            strong_id="stub-strong-id",
            decoder=decoders[0],
            json_metadata=task_list[0].json_metadata,
            shots=5,
            errors=0,
            seconds=0.01,
        )
        resume_path = Path(save_resume_filepath)
        resume_path.write_text(
            sinter.CSV_HEADER + "\n" + stat.to_csv_line() + "\n",
            encoding="utf-8",
        )
        details_dir = Path(custom_decoders[decoders[0]].details_dir)
        details_dir.mkdir(parents=True, exist_ok=True)
        (details_dir / "stub.jsonl").write_text(
            json.dumps({"decoder": decoders[0], "shots": stat.shots}) + "\n",
            encoding="utf-8",
        )
        return [stat]

    monkeypatch.setattr("relay_bp.stim.sinter.runner.sinter.collect", fake_collect)

    temp_path = REPO_ROOT / "tests" / "_tmp_native_bpgd_smoke"
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True)
    try:
        circuits_dir = temp_path / "circuits"
        circuits_dir.mkdir()
        circuit_path = (
            circuits_dir
            / "circuit=rotated_surface_code_memory_X,distance=3,rounds=3,error_rate=0.0001,noise_model=uniform_circuit,basis=CX.stim"
        )
        circuit = stim.Circuit.generated(
            rounds=3,
            distance=3,
            after_clifford_depolarization=0.0001,
            code_task="surface_code:rotated_memory_x",
        )
        circuit.to_file(circuit_path)

        config_path = temp_path / "decoder_specs.json"
        config_path.write_text(
            json.dumps(
                [
                    {
                        "name": "BPGD-relay-bp_smoke",
                        "stages": [
                            {
                                "kind": "BPGD",
                                "params": {
                                    "R_low": 0,
                                    "T_low": 1,
                                    "R_high": 0,
                                    "T_high": 1,
                                    "alpha": None,
                                    "c_damp": None,
                                },
                            },
                            {
                                "kind": "relay-bp",
                                "params": {
                                    "alpha": None,
                                    "gamma0": 0.1,
                                    "pre_iter": 40,
                                    "num_sets": 10,
                                    "set_max_iter": 20,
                                    "gamma_dist_interval": [-0.24, 0.66],
                                    "stop_nconv": 3,
                                    "stopping_criterion": "nconv",
                                    "logging": False,
                                },
                            },
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )

        outdir = temp_path / "results"
        result = run_sinter_folder_benchmark(
            circuits_dir=circuits_dir,
            decoder_config=config_path,
            outdir=outdir,
            filename_contains="rotated_surface_code_memory_X",
            max_shots=5,
            num_workers=1,
            save_name="smoke",
            print_progress=False,
            include_decode_result=True,
        )

        assert len(result.matched_circuits) == 1
        assert result.decoder_names == ["BPGD-relay-bp_smoke"]
        assert len(result.samples) == 1
        assert result.resume_csv.exists()
        assert result.summary_csv.exists()
        assert result.flat_csv.exists()
        assert result.details_dir is not None
        assert result.details_dir.exists()
        detail_files = sorted(result.details_dir.glob("*.jsonl"))
        assert detail_files
        assert detail_files[0].read_text(encoding="utf-8").strip()
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)
