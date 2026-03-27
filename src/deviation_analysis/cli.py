"""Command-line interface for the deviation analysis pipeline.

Usage:
    python -m deviation_analysis.cli batch --data-dir src/demos/m
    python -m deviation_analysis.cli single scan.csv toolpath.csv --stl model.stl
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with 'batch' and 'single' subcommands.
    """
    parser = argparse.ArgumentParser(
        description="3D Print Deviation Analysis Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline mode")

    # --- batch subcommand ---
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run full batch analysis across all configured methods",
    )
    batch_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("src/demos/m"),
        help="Directory containing scan CSVs and STL file",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/deviation_analysis/output"),
        help="Directory for output figures and tables",
    )
    batch_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching; recompute all stages",
    )

    # --- single subcommand ---
    single_parser = subparsers.add_parser(
        "single",
        help="Process a single scan file",
    )
    single_parser.add_argument(
        "scan_csv",
        type=Path,
        help="Path to scan cycle CSV file",
    )
    single_parser.add_argument(
        "toolpath_csv",
        type=Path,
        help="Path to scan trajectory CSV file",
    )
    single_parser.add_argument(
        "--stl",
        type=Path,
        required=True,
        help="Path to CAD reference STL file",
    )
    single_parser.add_argument(
        "--scan-speed",
        type=float,
        default=5.0,
        help="Scan speed in mm/s (default: 5.0)",
    )

    return parser


def main() -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "batch":
        from deviation_analysis.batch import run_batch
        from deviation_analysis.config import PipelineConfig

        config = PipelineConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
        run_batch(config)

    elif args.command == "single":
        from deviation_analysis.batch import process_single_scan
        from deviation_analysis.config import PipelineConfig, ScanConfig

        config = PipelineConfig(
            data_dir=args.scan_csv.parent,
            scan=ScanConfig(scan_speed=args.scan_speed),
        )
        distances, metrics = process_single_scan(
            scan_csv=args.scan_csv,
            toolpath_csv=args.toolpath_csv,
            stl_path=args.stl,
            config=config,
        )
        print(f"RMSD:  {metrics.rmsd:.4f} mm")
        print(f"MSD:   {metrics.msd:.4f} mm")
        print(f"MAE:   {metrics.mae:.4f} mm")
        print(f"95th%: {metrics.percentile_abs:.4f} mm")
        print(f"Excess volume:  {metrics.excess_volume_mm3:.4f} mm³")
        print(f"Deficit volume: {metrics.deficit_volume_mm3:.4f} mm³")
        print(f"Points analyzed: {metrics.n_points}")


if __name__ == "__main__":
    main()
