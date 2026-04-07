#!/usr/bin/env python3
"""Run native Sinter decoder benchmarks on a folder of Stim circuits."""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

from relay_bp.stim import run_sinter_folder_benchmark


DEFAULT_NUM_WORKERS = max(1, multiprocessing.cpu_count() // 2)


def build_parser(
    description: str,
    *,
    default_circuits_dir: Path | None = None,
    default_decoder_config: Path | None = None,
    default_outdir: Path | None = None,
    default_filename_contains: str = "",
    default_max_shots: int = 10_000,
    default_save_name: str = "stim_sinter",
    default_basis_filter: str = "none",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--circuits-dir",
        type=Path,
        required=default_circuits_dir is None,
        default=default_circuits_dir,
        help="Directory containing the target .stim files.",
    )
    parser.add_argument(
        "--decoder-config",
        type=Path,
        required=default_decoder_config is None,
        default=default_decoder_config,
        help="JSON file describing the registered decoders to run.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=default_outdir is None,
        default=default_outdir,
        help="Directory where resumable CSVs, summaries, and detail JSONL files are written.",
    )
    parser.add_argument(
        "--filename-contains",
        type=str,
        default=default_filename_contains,
        help="Comma-separated substrings that must all appear in selected .stim filenames.",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=default_max_shots,
        help="Maximum number of Sinter shots per circuit/decoder task.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of Sinter worker processes.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=default_save_name,
        help="Prefix used for the output files.",
    )
    parser.add_argument(
        "--basis-filter",
        type=str,
        default=default_basis_filter,
        help="Optional detector basis filter: none, X, or Z.",
    )
    parser.add_argument(
        "--max-circuits",
        type=int,
        default=None,
        help="Optional cap on the number of matching .stim circuits to run.",
    )
    parser.add_argument(
        "--print-progress",
        action="store_true",
        help="Show Sinter progress bars while collecting.",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> None:
    result = run_sinter_folder_benchmark(
        circuits_dir=args.circuits_dir,
        decoder_config=args.decoder_config,
        outdir=args.outdir,
        filename_contains=args.filename_contains,
        max_shots=args.max_shots,
        num_workers=args.num_workers,
        save_name=args.save_name,
        basis_filter=args.basis_filter,
        max_circuits=args.max_circuits,
        print_progress=args.print_progress,
        include_decode_result=True,
    )

    print(f"Matched circuits: {len(result.matched_circuits)}")
    print(f"Decoders: {result.decoder_names}")
    print(f"Resume CSV: {result.resume_csv}")
    print(f"Summary CSV: {result.summary_csv}")
    print(f"Flat summary CSV: {result.flat_csv}")
    if result.details_dir is not None:
        print(f"Detailed shot records: {result.details_dir}")


def main() -> None:
    parser = build_parser(__doc__)
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
