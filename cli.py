#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-Line Interface for Policy Analysis System
================================================
Provides command-line access to the policy analysis and feasibility scoring system
with configurable parameters for parallel processing, device selection, and output control.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from log_config import configure_logging

LOGGER = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser with all supported flags."""

    parser = argparse.ArgumentParser(
        description="Policy Analysis System - Feasibility Scoring and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --input ./documents --outdir ./results
  python cli.py --input ./documents --workers 8 --device cuda --precision float32
  python cli.py --input ./documents --topk 10 --umbral 0.75 --max-segmentos 1000
        """,
    )

    # Input/Output paths
    parser.add_argument(
        "--input",
        type=str,
        default=".",
        help="Input directory path containing documents to analyze (default: current directory)",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="output",
        help='Output directory path for results (default: "output")',
    )

    # Parallel processing configuration
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help=f"Number of parallel workers for processing (default: {min(os.cpu_count() or 1, 8)})",
    )

    # Device selection for computation
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps"],
        help="Computation device selection (default: auto-detect)",
    )

    # Numerical precision settings
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Numerical precision for calculations (default: float32)",
    )

    # Top-k search results
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of top-k search results to return (default: 10)",
    )

    # Threshold values
    parser.add_argument(
        "--umbral",
        type=float,
        default=0.5,
        help="Threshold value for similarity/confidence filtering (default: 0.5)",
    )

    # Maximum segments limit
    parser.add_argument(
        "--max-segmentos",
        type=int,
        default=1000,
        help="Maximum number of text segments to process (default: 1000)",
    )

    # Processing mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="feasibility",
        choices=["feasibility", "embedding", "demo"],
        help="Processing mode to execute (default: feasibility)",
    )

    # Additional options
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file (overrides command-line options)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without executing processing",
    )

    return parser


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.exception(f"Error loading config file {config_path}")
        raise ValueError(f"Invalid configuration file: {config_path}") from exc


def validate_args(args: argparse.Namespace) -> None:
    """Validate and adjust parsed arguments."""

    # Validate input path exists
    input_path = Path(args.input)
    if not input_path.exists():
        LOGGER.error(f"Input path '{args.input}' does not exist")
        raise ValueError(f"Input path '{args.input}' does not exist")

    # Create output directory if it doesn't exist
    output_path = Path(args.outdir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.exception(f"Error creating output directory '{args.outdir}'")
        raise ValueError(f"Unable to create output directory '{args.outdir}'") from exc

    # Validate workers count
    if args.workers < 1:
        LOGGER.error(f"Workers count must be at least 1 (received {args.workers})")
        raise ValueError("Workers count must be at least 1")

    # Validate topk value
    if args.topk < 1:
        LOGGER.error(f"topk value must be at least 1 (received {args.topk})")
        raise ValueError("topk value must be at least 1")

    # Validate umbral range
    if not 0.0 <= args.umbral <= 1.0:
        LOGGER.error(
            f"umbral value must be between 0.0 and 1.0 (received {args.umbral})"
        )
        raise ValueError("umbral value must be between 0.0 and 1.0")

    # Validate max_segmentos
    if args.max_segmentos < 1:
        LOGGER.error(
            f"max-segmentos value must be at least 1 (received {args.max_segmentos})"
        )
        raise ValueError("max-segmentos value must be at least 1")


def get_device_config(device_arg: str) -> str:
    """Determine the optimal device configuration."""
    if device_arg == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    else:
        return device_arg


def setup_logging(verbose: bool = False):
    """Setup logging configuration based on verbosity level."""
    log_level = "DEBUG" if verbose else None
    configure_logging(log_level)

    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not any(
        isinstance(handler, logging.FileHandler) for handler in root_logger.handlers
    ):
        file_handler = logging.FileHandler("policy_analysis.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    if verbose:
        LOGGER.debug("Verbose logging enabled")


def run_feasibility_mode(args: argparse.Namespace) -> int:
    """Execute feasibility scoring mode."""
    try:
        from feasibility_scorer import FeasibilityScorer
    except ImportError as exc:
        LOGGER.exception("Required module not available for feasibility mode")
        return 1

    LOGGER.info("Running feasibility analysis")
    LOGGER.info(
        f"Configuration: input={args.input}, outdir={args.outdir}, workers={args.workers}, device={args.device}, precision={args.precision}, topk={args.topk}, umbral={args.umbral}, max_segmentos={args.max_segmentos}"
    )

    scorer = FeasibilityScorer()
    scorer.configure_parallel(
        enable_parallel=(args.workers or 0) > 1,
        n_jobs=args.workers,
        backend="loky",
    )

    input_path = Path(args.input)
    text_files = []
    for ext in ["*.txt", "*.md", "*.pdf"]:
        text_files.extend(input_path.glob(ext))

    if not text_files:
        LOGGER.warning(f"No text files found in {args.input}")
        return 1

    LOGGER.info(f"Found {len(text_files)} files to process")

    indicators = []
    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as input_file:
                content = input_file.read()
        except OSError as exc:
            LOGGER.warning(f"Could not read {file_path}: {exc}")
            continue

        segments = [
            segment.strip() for segment in content.split("\n") if segment.strip()
        ]
        indicators.extend(segments[: args.max_segmentos])

    if not indicators:
        LOGGER.warning("No content found to analyze")
        return 1

    LOGGER.info(f"Analyzing {len(indicators)} indicators")

    try:
        if args.workers > 1:
            results = scorer.batch_score(
                indicators[: args.max_segmentos], compare_backends=args.verbose
            )
        else:
            results = [
                scorer.calculate_feasibility_score(ind)
                for ind in indicators[: args.max_segmentos]
            ]
    except Exception:
        LOGGER.exception("Failed to score indicators")
        return 1

    filtered_results = [
        (ind, result)
        for ind, result in zip(indicators, results)
        if result.feasibility_score >= args.umbral
    ]

    filtered_results.sort(key=lambda item: item[1].feasibility_score, reverse=True)
    top_results = filtered_results[: args.topk]

    output_file = Path(args.outdir) / "feasibility_report.json"
    report_data = {
        "config": {
            "input": args.input,
            "workers": args.workers,
            "device": args.device,
            "precision": args.precision,
            "topk": args.topk,
            "umbral": args.umbral,
            "max_segmentos": args.max_segmentos,
        },
        "summary": {
            "total_indicators": len(indicators),
            "processed_indicators": min(len(indicators), args.max_segmentos),
            "passed_threshold": len(filtered_results),
            "top_k_results": len(top_results),
        },
        "results": [
            {
                "text": text[:200] + ("..." if len(text) > 200 else ""),
                "score": result.feasibility_score,
                "quality_tier": result.quality_tier,
                "components": [c.value for c in result.components_detected],
                "quantitative_baseline": result.has_quantitative_baseline,
                "quantitative_target": result.has_quantitative_target,
            }
            for text, result in top_results
        ],
    }

    try:
        with open(output_file, "w", encoding="utf-8") as report_file:
            json.dump(report_data, report_file, indent=2, ensure_ascii=False)
    except OSError:
        LOGGER.exception(f"Failed to write feasibility report to {output_file}")
        return 1

    LOGGER.info(f"Analysis complete. Results saved to {output_file}")
    if top_results:
        LOGGER.info(
            f"Top {len(top_results)} results (threshold >= {args.umbral})"
        )
        for index, (text, result) in enumerate(top_results, start=1):
            display_text = text[:100] + ("..." if len(text) > 100 else "")
            LOGGER.info(
                f"{index}. Score: {result.feasibility_score:.3f} | {result.quality_tier} | {display_text}"
            )

    return 0


def run_embedding_mode(args: argparse.Namespace) -> int:
    """Execute embedding model mode."""
    try:
        from embedding_model import create_embedding_model

        LOGGER.info("Running embedding analysis")

        # Get device configuration
        device = get_device_config(args.device)

        # Create embedding model with CLI parameters
        model = create_embedding_model(
            device=device, precision=args.precision, enable_cache=True
        )

        # Process input files
        input_path = Path(args.input)
        documents = []

        for file_path in input_path.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as source_file:
                    content = source_file.read().strip()
                    if content:
                        documents.append(
                            {
                                "file": file_path.name,
                                "content": content[
                                    : args.max_segmentos
                                ],  # Limit content length
                            }
                        )
            except OSError as exc:
                LOGGER.warning(f"Could not read {file_path}: {exc}")

        if not documents:
            LOGGER.warning("No documents found to process")
            return 1

        LOGGER.info(f"Processing {len(documents)} documents")

        # Generate embeddings
        texts = [doc["content"] for doc in documents]
        embeddings = model.encode(texts)

        LOGGER.info(f"Generated embeddings with shape {embeddings.shape}")

        # Save results
        output_file = Path(args.outdir) / "embeddings.npy"
        metadata_file = Path(args.outdir) / "embeddings_metadata.json"

        import numpy as np

        np.save(output_file, embeddings)

        metadata = {
            "config": {
                "device": device,
                "precision": args.precision,
                "max_segmentos": args.max_segmentos,
            },
            "documents": [doc["file"] for doc in documents],
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
        }

        with open(metadata_file, "w", encoding="utf-8") as metadata_handle:
            json.dump(metadata, metadata_handle, indent=2)

        LOGGER.info(f"Embeddings saved to {output_file}")
        LOGGER.info(f"Metadata saved to {metadata_file}")

        return 0

    except ImportError:
        LOGGER.exception("Required module not available for embedding mode")
        return 1
    except Exception:
        LOGGER.exception("Error in embedding mode")
        return 1


def run_demo_mode(args: argparse.Namespace) -> int:
    """Execute demo mode."""
    try:
        # Pass CLI parameters through environment variables to maintain compatibility
        os.environ["CLI_WORKERS"] = str(args.workers)
        os.environ["CLI_DEVICE"] = args.device
        os.environ["CLI_OUTPUT_DIR"] = args.outdir

        LOGGER.info(
            f"Running demo mode with workers={args.workers}, device={args.device}, output={args.outdir}"
        )

        # Import and run demo with environment configuration
        import demo

        demo.main()

        return 0

    except ImportError:
        LOGGER.exception("Demo module not available")
        return 1
    except Exception:
        LOGGER.exception("Error in demo mode")
        return 1


def main():
    """Main entry point for the CLI application."""

    # Create and parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Load configuration file if provided
    if args.config:
        try:
            config = load_config_file(args.config)
        except ValueError as exc:
            LOGGER.error(f"Failed to load configuration file: {exc}")
            return 1
        # Override command line args with config file values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Setup logging
    setup_logging(args.verbose)

    # Validate arguments
    try:
        validate_args(args)
    except ValueError as exc:
        LOGGER.error(f"Invalid CLI configuration: {exc}")
        return 1

    # Show configuration and exit if dry-run
    if args.dry_run:
        LOGGER.info("Configuration (dry-run mode):")
        LOGGER.info(f"  Input directory: {args.input}")
        LOGGER.info(f"  Output directory: {args.outdir}")
        LOGGER.info(f"  Workers: {args.workers}")
        LOGGER.info(f"  Device: {get_device_config(args.device)}")
        LOGGER.info(f"  Precision: {args.precision}")
        LOGGER.info(f"  Top-k: {args.topk}")
        LOGGER.info(f"  Umbral: {args.umbral}")
        LOGGER.info(f"  Max segments: {args.max_segmentos}")
        LOGGER.info(f"  Mode: {args.mode}")
        LOGGER.info(f"  Verbose: {args.verbose}")
        return 0

    # Execute the selected mode
    if args.mode == "feasibility":
        return run_feasibility_mode(args)
    elif args.mode == "embedding":
        return run_embedding_mode(args)
    elif args.mode == "demo":
        return run_demo_mode(args)
    else:
        LOGGER.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
