#!/usr/bin/env python3
"""Run the native bivariate bicycle-code disordered-damping Sinter benchmark."""

from __future__ import annotations

from pathlib import Path

from run_stim_sinter_benchmarks import build_parser, run_from_args


def main() -> None:
    parser = build_parser(
        __doc__,
        default_circuits_dir=Path("tests/testdata/bicycle_bivariate"),
        default_decoder_config=Path("configs/decoder_specs_bivariate_disordered_damping.json"),
        default_outdir=Path("results_bivariate_disordered_damping_sinter"),
        default_filename_contains="bicycle_bivariate",
        default_max_shots=1_000,
        default_save_name="bivariate_disordered_damping",
    )
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
