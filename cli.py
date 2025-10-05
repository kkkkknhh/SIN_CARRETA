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
        choices=["feasibility", "decatalogo", "embedding", "demo"],
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
        LOGGER.exception("Error loading config file %s", config_path)
        raise ValueError(f"Invalid configuration file: {config_path}") from exc


def validate_args(args: argparse.Namespace) -> None:
    """Validate and adjust parsed arguments."""

    # Validate input path exists
    input_path = Path(args.input)
    if not input_path.exists():
        LOGGER.error("Input path '%s' does not exist", args.input)
        raise ValueError(f"Input path '{args.input}' does not exist")

    # Create output directory if it doesn't exist
    output_path = Path(args.outdir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.exception("Error creating output directory '%s'", args.outdir)
        raise ValueError(f"Unable to create output directory '{args.outdir}'") from exc

    # Validate workers count
    if args.workers < 1:
        LOGGER.error("Workers count must be at least 1 (received %s)", args.workers)
        raise ValueError("Workers count must be at least 1")

    # Validate topk value
    if args.topk < 1:
        LOGGER.error("topk value must be at least 1 (received %s)", args.topk)
        raise ValueError("topk value must be at least 1")

    # Validate umbral range
    if not 0.0 <= args.umbral <= 1.0:
        LOGGER.error(
            "umbral value must be between 0.0 and 1.0 (received %s)", args.umbral
        )
        raise ValueError("umbral value must be between 0.0 and 1.0")

    # Validate max_segmentos
    if args.max_segmentos < 1:
        LOGGER.error(
            "max-segmentos value must be at least 1 (received %s)",
            args.max_segmentos,
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
        "Configuration: input=%s, outdir=%s, workers=%s, device=%s, precision=%s, topk=%s, umbral=%s, max_segmentos=%s",
        args.input,
        args.outdir,
        args.workers,
        args.device,
        args.precision,
        args.topk,
        args.umbral,
        args.max_segmentos,
    )

    scorer = FeasibilityScorer(
        enable_parallel=args.workers > 1, n_jobs=args.workers, backend="loky"
    )

    input_path = Path(args.input)
    text_files = []
    for ext in ["*.txt", "*.md", "*.pdf"]:
        text_files.extend(input_path.glob(ext))

    if not text_files:
        LOGGER.warning("No text files found in %s", args.input)
        return 1

    LOGGER.info("Found %s files to process", len(text_files))

    indicators = []
    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as input_file:
                content = input_file.read()
        except OSError as exc:
            LOGGER.warning("Could not read %s: %s", file_path, exc)
            continue

        segments = [
            segment.strip() for segment in content.split("\n") if segment.strip()
        ]
        indicators.extend(segments[: args.max_segmentos])

    if not indicators:
        LOGGER.warning("No content found to analyze")
        return 1

    LOGGER.info("Analyzing %s indicators", len(indicators))

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
        LOGGER.exception("Failed to write feasibility report to %s", output_file)
        return 1

    LOGGER.info("Analysis complete. Results saved to %s", output_file)
    if top_results:
        LOGGER.info(
            "Top %s results (threshold >= %s)",
            len(top_results),
            args.umbral,
        )
        for index, (text, result) in enumerate(top_results, start=1):
            display_text = text[:100] + ("..." if len(text) > 100 else "")
            LOGGER.info(
                "%s. Score: %.3f | %s | %s",
                index,
                result.feasibility_score,
                result.quality_tier,
                display_text,
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
                LOGGER.warning("Could not read %s: %s", file_path, exc)

        if not documents:
            LOGGER.warning("No documents found to process")
            return 1

        LOGGER.info("Processing %s documents", len(documents))

        # Generate embeddings
        texts = [doc["content"] for doc in documents]
        embeddings = model.encode(texts)

        LOGGER.info("Generated embeddings with shape %s", embeddings.shape)

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

        LOGGER.info("Embeddings saved to %s", output_file)
        LOGGER.info("Metadata saved to %s", metadata_file)

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
            "Running demo mode with workers=%s, device=%s, output=%s",
            args.workers,
            args.device,
            args.outdir,
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


def run_decatalogo_mode(args: argparse.Namespace) -> int:
    """Execute Decatalogo evaluation mode."""
    os.environ["CLI_WORKERS"] = str(args.workers)
    os.environ["CLI_DEVICE"] = args.device
    os.environ["CLI_PRECISION"] = args.precision
    os.environ["CLI_TOPK"] = str(args.topk)
    os.environ["CLI_UMBRAL"] = str(args.umbral)
    os.environ["CLI_MAX_SEGMENTOS"] = str(args.max_segmentos)
    os.environ["CLI_INPUT_DIR"] = args.input
    os.environ["CLI_OUTPUT_DIR"] = args.outdir

    try:
        from Decatalogo_evaluador import IndustrialDecatalogoEvaluatorFull
    except ImportError:
        LOGGER.exception("Decatalogo evaluator module not available")
        return 1

    LOGGER.info(
        "Running Decatalogo evaluation with input=%s output=%s", args.input, args.outdir
    )

    evaluator = IndustrialDecatalogoEvaluatorFull()

    input_path = Path(args.input)
    text_files = list(input_path.glob("*.txt"))

    if not text_files:
        LOGGER.warning("No text files found in %s", args.input)
        return 1

    LOGGER.info("Found %s files to evaluate", len(text_files))

    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as source_file:
                content = source_file.read()

            for punto_id in range(1, 11):
                result = evaluator.evaluar_punto_completo(content, punto_id)
                output_file = (
                    Path(args.outdir)
                    / f"decatalogo_punto_{punto_id}_{file_path.stem}.json"
                )
                with open(output_file, "w", encoding="utf-8") as output_handle:
                    json.dump(
                        {
                            "punto_id": result.punto_id,
                            "nombre_punto": result.nombre_punto,
                            "puntaje_agregado": result.puntaje_agregado_punto,
                            "evaluaciones_dimensiones": [
                                {
                                    "dimension": ed.dimension,
                                    "puntaje": ed.puntaje_dimension,
                                    "preguntas_evaluadas": len(
                                        ed.evaluaciones_preguntas
                                    ),
                                }
                                for ed in result.evaluaciones_dimensiones
                            ],
                        },
                        output_handle,
                        indent=2,
                        ensure_ascii=False,
                    )

            LOGGER.info("Processed %s", file_path.name)
        except OSError as exc:
            LOGGER.warning("Error processing %s: %s", file_path, exc)
        except Exception:
            LOGGER.exception("Unexpected error processing %s", file_path)

    LOGGER.info("Decatalogo evaluation complete. Results in %s", args.outdir)
    return 0


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
            LOGGER.error("Failed to load configuration file: %s", exc)
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
        LOGGER.error("Invalid CLI configuration: %s", exc)
        return 1

    # Show configuration and exit if dry-run
    if args.dry_run:
        LOGGER.info("Configuration (dry-run mode):")
        LOGGER.info("  Input directory: %s", args.input)
        LOGGER.info("  Output directory: %s", args.outdir)
        LOGGER.info("  Workers: %s", args.workers)
        LOGGER.info("  Device: %s", get_device_config(args.device))
        LOGGER.info("  Precision: %s", args.precision)
        LOGGER.info("  Top-k: %s", args.topk)
        LOGGER.info("  Umbral: %s", args.umbral)
        LOGGER.info("  Max segments: %s", args.max_segmentos)
        LOGGER.info("  Mode: %s", args.mode)
        LOGGER.info("  Verbose: %s", args.verbose)
        return 0

    # Execute the selected mode
    if args.mode == "feasibility":
        return run_feasibility_mode(args)
    elif args.mode == "embedding":
        return run_embedding_mode(args)
    elif args.mode == "demo":
        return run_demo_mode(args)
    elif args.mode == "decatalogo":
        return run_decatalogo_mode(args)
    else:
        LOGGER.error("Unknown mode: %s", args.mode)
        return 1


if __name__ == "__main__":
    sys.exit(main())
