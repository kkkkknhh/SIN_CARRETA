#!/usr/bin/env python3
"""
MINIMINIMOON System Runner

Command-line utility to run the integrated MINIMINIMOON system on
plan documents for comprehensive analysis and scoring.

Usage:
    python run_system.py analyze path/to/plan.txt
    python run_system.py batch path/to/plans_directory/
    python run_system.py interactive

Options:
    analyze: Process a single plan document
    batch: Process multiple documents in a directory
    interactive: Run in interactive mode with manual input
"""

import argparse
import json
import os
import sys
import time

from miniminimoon_system import MINIMINIMOONSystem


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="MINIMINIMOON System - Comprehensive Plan Analysis Tool"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single plan document")
    analyze_parser.add_argument("file_path", help="Path to the plan document")
    analyze_parser.add_argument("--output", "-o", help="Output file path (JSON)")
    analyze_parser.add_argument("--config", "-c", help="Path to configuration file")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process multiple documents in a directory")
    batch_parser.add_argument("directory", help="Directory containing plan documents")
    batch_parser.add_argument("--output", "-o", help="Output directory for results")
    batch_parser.add_argument("--config", "-c", help="Path to configuration file")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    interactive_parser.add_argument("--config", "-c", help="Path to configuration file")
    
    return parser


def analyze_single_plan(args):
    """Analyze a single plan document."""
    print(f"Analyzing plan: {args.file_path}")
    start_time = time.time()
    
    # Initialize system
    system = MINIMINIMOONSystem(args.config)
    
    # Analyze plan
    results = system.analyze_plan(args.file_path)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    results["processing_time"] = processing_time
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")
    else:
        # Print summary to console
        print_analysis_summary(results)
    
    print(f"Analysis completed in {processing_time:.2f} seconds")


def batch_process_plans(args):
    """Process multiple plans in a directory."""
    print(f"Batch processing plans in: {args.directory}")
    start_time = time.time()
    
    # Initialize system
    system = MINIMINIMOONSystem(args.config)
    
    # Process plans
    results = system.batch_process_plans(args.directory)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    if isinstance(results, dict):
        results["processing_time"] = processing_time
    
    # Output results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, "batch_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    else:
        # Print summary to console
        print_batch_summary(results)
    
    print(f"Batch processing completed in {processing_time:.2f} seconds")


def run_interactive_mode(args):
    """Run the system in interactive mode."""
    print("MINIMINIMOON System - Interactive Mode")
    print("=====================================")
    
    # Initialize system
    system = MINIMINIMOONSystem(args.config)
    
    while True:
        print("\nOptions:")
        print("1. Analyze text")
        print("2. Evaluate against a dimension")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            text = input_multiline_text("Enter text to analyze (end with Ctrl+D or empty line):")
            
            if text:
                print("\nAnalyzing text...")
                results = system.analyze_text(text)
                print_text_analysis(results)
            else:
                print("No text entered.")
        
        elif choice == "2":
            text = input_multiline_text("Enter text to evaluate (end with Ctrl+D or empty line):")
            
            if text:
                try:
                    dimension = int(input("\nEnter dimension ID (1-10): "))
                    if 1 <= dimension <= 10:
                        print(f"\nEvaluating against dimension {dimension}...")
                        results = system.evaluate_decalogo_dimension(text, dimension)
                        print_dimension_evaluation(results)
                    else:
                        print("Dimension ID must be between 1 and 10.")
                except ValueError:
                    print("Invalid dimension ID. Please enter a number.")
            else:
                print("No text entered.")
        
        elif choice == "3":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")


def input_multiline_text(prompt):
    """Get multiline text input from user."""
    print(prompt)
    lines = []
    
    try:
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
    except EOFError:
        pass
    
    return "\n".join(lines)


def print_analysis_summary(results):
    """Print a summary of the analysis results."""
    print("\n===== ANALYSIS SUMMARY =====")
    print(f"Plan: {results.get('plan_name', 'Unknown')}")
    
    if "plan_evaluation" in results:
        eval_results = results["plan_evaluation"]
        print(f"\nGlobal Score: {eval_results.get('global_score', 'N/A')}")
        print(f"Alignment Level: {eval_results.get('alignment_level', 'N/A')}")
        
        if "dimension_scores" in eval_results:
            print("\nDimension Scores:")
            for dim, score in sorted(eval_results["dimension_scores"].items()):
                print(f"  Dimension {dim}: {score:.2f}")
        
        if "global_recommendations" in eval_results:
            print("\nKey Recommendations:")
            for i, rec in enumerate(eval_results["global_recommendations"][:3], 1):
                print(f"  {i}. {rec}")
    
    if "teoria_cambio_validation" in results:
        tcv = results["teoria_cambio_validation"]
        print("\nTheory of Change:")
        print(f"  Valid: {tcv.get('is_valid', False)}")
        print(f"  Causal Coefficient: {tcv.get('causal_coefficient', 0):.2f}")
        print(f"  Monte Carlo p-value: {tcv.get('monte_carlo', {}).get('p_value', 0):.4f}")
    
    if "text_analysis" in results:
        ta = results["text_analysis"]
        
        if "contradictions" in ta:
            print(f"\nContradictions: {ta['contradictions'].get('total', 0)}")
            print(f"  Risk Level: {ta['contradictions'].get('risk_level', 'Unknown')}")
        
        if "responsibilities" in ta:
            print(f"\nResponsible Entities: {len(ta['responsibilities'])}")
        
        if "feasibility" in ta:
            print(f"\nFeasibility Score: {ta['feasibility'].get('score', 0):.2f}")


def print_batch_summary(results):
    """Print a summary of batch processing results."""
    print("\n===== BATCH PROCESSING SUMMARY =====")
    
    if "_summary" in results:
        summary = results["_summary"]
        print(f"Total Plans: {summary['total_plans']}")
        print(f"Processed Plans: {summary['processed_plans']}")
        print(f"Average Score: {summary['average_score']:.2f}")
        print(f"Score Range: {summary['min_score']:.2f} - {summary['max_score']:.2f}")
        
        # Print top 3 plans by score
        top_plans = sorted(
            [(name, r["global_score"]) for name, r in results.items() 
             if isinstance(r, dict) and "global_score" in r],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        print("\nTop Performing Plans:")
        for i, (name, score) in enumerate(top_plans, 1):
            print(f"  {i}. {name}: {score:.2f}")
    
    # Print errors if any
    errors = [(name, r["error"]) for name, r in results.items() 
              if isinstance(r, dict) and "error" in r]
    
    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for name, error in errors[:3]:
            print(f"  {name}: {error}")
        
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more")


def print_text_analysis(results):
    """Print text analysis results."""
    print("\n===== TEXT ANALYSIS RESULTS =====")
    
    if "contradictions" in results:
        contra = results["contradictions"]
        print(f"Contradictions: {contra.get('total', 0)}")
        print(f"Risk Level: {contra.get('risk_level', 'Unknown')}")
        
        if contra.get("matches"):
            print("\nTop Contradiction:")
            match = contra["matches"][0]
            print(f"  Text: {match.get('text', '')}")
            print(f"  Connector: {match.get('connector', '')}")
            print(f"  Risk: {match.get('risk_level', '')}")
            print(f"  Confidence: {match.get('confidence', 0):.2f}")
    
    if "responsibilities" in results:
        print(f"\nResponsible Entities: {len(results['responsibilities'])}")
        for i, resp in enumerate(results["responsibilities"][:3], 1):
            print(f"  {i}. {resp.get('text', '')} ({resp.get('type', '')})")
    
    if "feasibility" in results:
        feas = results["feasibility"]
        print(f"\nFeasibility Score: {feas.get('score', 0):.2f}")
        print(f"Has Baseline: {feas.get('has_baseline', False)}")
        print(f"Has Target: {feas.get('has_target', False)}")
        print(f"Has Timeframe: {feas.get('has_timeframe', False)}")
        
        if "detailed_matches" in feas:
            print("\nKey Indicators:")
            for i, match in enumerate(feas["detailed_matches"][:3], 1):
                print(f"  {i}. {match.get('text', '')} ({match.get('type', '')})")
    
    if "monetary" in results:
        print(f"\nMonetary Expressions: {len(results['monetary'])}")
        for i, mon in enumerate(results["monetary"][:3], 1):
            print(f"  {i}. {mon.get('text', '')} = {mon.get('value', '')} {mon.get('currency', '')}")


def print_dimension_evaluation(results):
    """Print dimension evaluation results."""
    print("\n===== DIMENSION EVALUATION RESULTS =====")
    print(f"Dimension: {results.get('dimension_id', 'Unknown')}")
    print(f"Score: {results.get('score', 0):.2f}")
    print(f"Classification: {results.get('classification', 'Unknown')}")
    
    if "dimensional_scores" in results:
        print("\nDimensional Scores:")
        for dim, score in results["dimensional_scores"].items():
            print(f"  {dim}: {score:.2f}")
    
    if "explanation" in results:
        print("\nExplanation:")
        explanation = results["explanation"]
        if len(explanation) > 500:
            explanation = explanation[:497] + "..."
        print(f"  {explanation}")
    
    if "gaps" in results and results["gaps"]:
        print("\nIdentified Gaps:")
        for i, gap in enumerate(results["gaps"][:3], 1):
            print(f"  {i}. {gap}")
    
    if "recommendations" in results and results["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(results["recommendations"][:3], 1):
            print(f"  {i}. {rec}")
    
    if "strengths" in results and results["strengths"]:
        print("\nStrengths:")
        for i, strength in enumerate(results["strengths"][:3], 1):
            print(f"  {i}. {strength}")
    
    if "risks" in results and results["risks"]:
        print("\nRisks:")
        for i, risk in enumerate(results["risks"][:3], 1):
            print(f"  {i}. {risk}")


def main():
    """Main entry point."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyze_single_plan(args)
    elif args.command == "batch":
        batch_process_plans(args)
    elif args.command == "interactive":
        run_interactive_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
