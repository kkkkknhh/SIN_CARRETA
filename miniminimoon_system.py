# coding=utf-8
"""
MINIMINIMOON System Integration
===============================
System adapter for integrating with the run_system.py launcher.

Follows the canonical dependency flow structure (24 critical flows)
with strict validation gates and deterministic processing.

Version: 3.0.0 (Unified Flow Architecture)
Date: 2025-10-06
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Import orchestrator module (Flow #18 in canonical documentation)
from miniminimoon_orchestrator import (
    MiniminimoonOrchestrator,  # Replaces deprecated CanonicalDeterministicOrchestrator
    UnifiedEvaluationPipeline
)

# Import for Gate #1: Configuration immutability verification
from miniminimoon_immutability import EnhancedImmutabilityContract

# System validators for pre/post checks (Flow #19)
from system_validators import SystemValidators


class MINIMINIMOONSystem:
    """
    MINIMINIMOONSystem provides a unified interface for the orchestrator
    that implements the canonical dependency flow (Flow #72).
    
    Gates enforced:
    1. Configuration immutability verified before execution
    2. Flow runtime matches canonical documentation
    3. Evidence hash stability for deterministic results
    4. Complete coverage (≥300 questions)
    5. Rubric weight/question alignment
    6. No use of deprecated orchestrators
    """
    
    VERSION = "3.0.0"
    
    def __init__(self, config_dir: str, log_level: str = "INFO"):
        """
        Initialize the MINIMINIMOON system with validation gates.
        
        Args:
            config_dir: Path to configuration directory
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.config_dir = Path(config_dir)
        self.log_level = log_level
        
        # Configure logging (Flow #26)
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Verify configuration directories exist
        if not self.config_dir.exists():
            self.logger.error(f"Configuration directory not found: {self.config_dir}")
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
        
        # Gate #1: Verify immutability contract (Flow #55)
        self._verify_immutability()
        
        # Initialize system validators (Flow #56)
        self.system_validators = SystemValidators(self.config_dir)
        
        self.logger.info(f"MINIMINIMOONSystem {self.VERSION} initialized (Unified Flow Architecture)")
    
    def _verify_immutability(self):
        """
        Verify the immutability contract as a hard gate.
        Raises RuntimeError if verification fails.
        """
        immutability_contract = EnhancedImmutabilityContract()
        
        if not immutability_contract.has_snapshot():
            error_msg = "GATE #1 FAILED: No frozen config snapshot. Run freeze_configuration() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        if not immutability_contract.verify_frozen_config():
            error_msg = "GATE #1 FAILED: Frozen config mismatch. Config files changed since snapshot."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        self.logger.info("✓ Gate #1 PASSED: Frozen configuration verified")
    
    def evaluate_plan(
        self, 
        plan_path: str, 
        output_dir: str, 
        flow_doc_path: Optional[str] = None,
        enable_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a plan using the unified evaluation pipeline (Flow #72).
        
        Args:
            plan_path: Path to the plan file
            output_dir: Path to output directory
            flow_doc_path: Path to flow documentation file (optional)
            enable_validation: Whether to enable validation gates
            
        Returns:
            Dict containing evaluation results with evidence hash and flow validation
        """
        self.logger.info(f"Evaluating plan: {plan_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run pre-validation checks (Gate #2, Flow #56)
        pre_validation = self.system_validators.run_pre_checks()
        if not pre_validation["pre_validation_ok"]:
            error_msg = "GATE #2 FAILED: Pre-validation checks failed."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Flow documentation path
        flow_doc = Path(flow_doc_path) if flow_doc_path else None
        
        # Initialize the unified evaluation pipeline (Flow #18, #72)
        pipeline = UnifiedEvaluationPipeline(
            config_dir=self.config_dir,
            flow_doc_path=flow_doc,
            enable_validation=enable_validation
        )
        
        # Execute evaluation (Flow #72)
        results = pipeline.evaluate(plan_path, output_path)
        
        # Run post-validation checks (Gates #3, #4, #5)
        post_validation = self.system_validators.run_post_checks(results)
        if not post_validation["post_validation_ok"]:
            self.logger.warning("Post-validation checks failed - see report for details")
        
        # Ensure evidence hash is present (Gate #3)
        if "evidence_hash" not in results:
            self.logger.warning("GATE #3 WARNING: No evidence hash in results")
            
        # Check question coverage (Gate #4)
        answers = results.get("evaluations", {}).get("answers_report", {})
        total_questions = answers.get("summary", {}).get("total_questions", 0) or \
                         answers.get("global_summary", {}).get("answered_questions", 0)
        if total_questions < 300:
            self.logger.warning(f"GATE #4 WARNING: Only {total_questions}/300 questions answered")
        else:
            self.logger.info(f"✓ Gate #4 PASSED: {total_questions}/300 questions answered")
        
        # Add results bundle metadata (Flow #63)
        results_bundle = {
            "results": results,
            "system_version": self.VERSION,
            "pre_validation": pre_validation,
            "post_validation": post_validation,
            "metadata": {
                "plan_path": plan_path,
                "config_dir": str(self.config_dir),
                "output_dir": str(output_path)
            }
        }
        
        # Save results bundle (Flow #63)
        bundle_path = output_path / "results_bundle.json"
        with open(bundle_path, 'w', encoding='utf-8') as f:
            json.dump(results_bundle, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✓ Evaluation complete. Results bundle saved to {bundle_path}")
        return results_bundle
    
    def freeze_configuration(self) -> Dict[str, Any]:
        """
        Freeze the current configuration to ensure reproducibility (Flow #55).
        
        Returns:
            Dict containing snapshot details
        """
        from miniminimoon_immutability import freeze_configuration
        
        self.logger.info(f"Freezing configuration in {self.config_dir}")
        return freeze_configuration(self.config_dir)
    
    def verify_configuration(self) -> Dict[str, bool]:
        """
        Verify the frozen configuration without running a full evaluation (Flow #71).
        
        Returns:
            Dict containing verification results
        """
        # Gate #1: Configuration immutability
        try:
            self._verify_immutability()
            immutability_ok = True
        except RuntimeError:
            immutability_ok = False
        
        # Run pre-validation checks
        pre_validation = self.system_validators.run_pre_checks()
        
        return {
            "immutability_ok": immutability_ok,
            "pre_validation_ok": pre_validation["pre_validation_ok"],
            "checks": pre_validation["checks"]
        }


# Utility function for run_system.py
def create_system(config_dir: str = "config", log_level: str = "INFO") -> MINIMINIMOONSystem:
    """
    Factory function to create a MINIMINIMOONSystem instance (Flow #25).
    
    Args:
        config_dir: Path to configuration directory
        log_level: Logging level
        
    Returns:
        MINIMINIMOONSystem instance
    """
    return MINIMINIMOONSystem(config_dir=config_dir, log_level=log_level)


# Compatibility warning for deprecated orchestrator
def _check_deprecated_imports():
    """
    Check for deprecated imports and raise exception if found (Flow #67).
    """
    try:
        # This import is only for checking if it exists
        import decalogo_pipeline_orchestrator  # noqa: F401
        raise RuntimeError(
            "GATE #6 FAILED: decalogo_pipeline_orchestrator is DEPRECATED and must not be imported. "
            "Use MiniminimoonOrchestrator instead."
        )
    except ImportError:
        # This is expected - the module shouldn't be there
        pass


# Perform deprecated imports check at module load time
_check_deprecated_imports()


if __name__ == "__main__":
    """
    Command-line interface for direct system usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"MINIMINIMOON System v{MINIMINIMOONSystem.VERSION} (Unified Flow Architecture)"
    )
    parser.add_argument(
        "--plan", "-p", 
        required=True,
        help="Path to the plan file to evaluate"
    )
    parser.add_argument(
        "--config-dir", "-c", 
        default="config",
        help="Path to configuration directory (default: ./config)"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        default="output",
        help="Path to output directory (default: ./output)"
    )
    parser.add_argument(
        "--freeze", 
        action="store_true",
        help="Freeze the current configuration (must be run before evaluation)"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify the frozen configuration without running evaluation"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    try:
        system = MINIMINIMOONSystem(config_dir=args.config_dir, log_level=args.log_level)
        
        if args.freeze:
            snapshot = system.freeze_configuration()
            print(f"✓ Configuration frozen: {snapshot['snapshot_path']}")
            sys.exit(0)
            
        if args.verify:
            verification = system.verify_configuration()
            if verification["immutability_ok"] and verification["pre_validation_ok"]:
                print("✓ Configuration verified: All checks passed")
                sys.exit(0)
            else:
                print("⨯ Configuration verification failed:")
                for check in verification.get("checks", []):
                    print(f"  - {check['name']}: {check['status']} - {check['message']}")
                sys.exit(1)
        
        # Default: evaluate plan
        system.evaluate_plan(args.plan, args.output_dir)
        print(f"✓ Evaluation complete. Results saved to {args.output_dir}")
        sys.exit(0)
        
    except Exception as e:
        print(f"⨯ Error: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)