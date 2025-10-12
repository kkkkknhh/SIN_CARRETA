#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Evaluation Pipeline
============================

Single entry point for complete PDM evaluation that:
1. Runs the canonical MINIMINIMOON pipeline once
2. Produces a frozen EvidenceRegistry
3. Invokes both Decálogo and Questionnaire evaluators
4. Both evaluators consume the SAME evidence registry
5. Returns unified, deterministic results
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import core orchestrator
from miniminimoon_orchestrator import MINIMINIMOONOrchestrator

# Import evaluators
try:
    from questionnaire_engine import QuestionnaireEngine
    QUESTIONNAIRE_AVAILABLE = True
except ImportError:
    QUESTIONNAIRE_AVAILABLE = False
    logging.warning("questionnaire_engine not available")

# Import validators
from system_validators import SystemHealthValidator
from evidence_registry import EvidenceRegistry

logger = logging.getLogger(__name__)


class UnifiedEvaluationPipeline:
    """
    Unified pipeline that orchestrates:
    - Canonical flow execution (11 nodes)
    - Evidence registry building
    - Decálogo evaluation
    - Questionnaire evaluation (300 questions)
    - Post-execution validation
    """

    def __init__(self, 
                 repo_root: Optional[str] = None,
                 rubric_path: Optional[str] = None,
                 config_path: Optional[str] = "system_configuration.json"):
        """
        Initialize unified pipeline.

        Args:
            repo_root: Repository root directory
            rubric_path: Path to rubric scoring JSON file
            config_path: Path to system configuration file
        """
        logger.info("Initializing Unified Evaluation Pipeline...")

        self.repo_root = repo_root if repo_root else "."
        self.rubric_path = rubric_path if rubric_path else "RUBRIC_SCORING.json"

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize orchestrator
        self.orchestrator = MINIMINIMOONOrchestrator(config_path)

        # Initialize system validator
        self.system_validator = SystemHealthValidator(self.repo_root)

        # Initialize evaluators (lazy loading)
        self.decalogo_evaluator = None
        self.questionnaire_evaluator = None
        
        # Warm-up flag for thread-safe one-time initialization
        self._warmup_done = False
        self._warmup_lock = __import__('threading').Lock()

        logger.info("✅ Unified Evaluation Pipeline initialized")

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        if not Path(config_path).exists():
            logger.warning("Config file not found: %s, using defaults", config_path)
            return {
                "evaluators": {
                    "decalogo": {"enabled": True},
                    "questionnaire": {"enabled": True}
                },
                "parallel_processing": {
                    "enabled": True,
                    "max_workers": 4,
                    "components": ["questionnaire_engine"]
                }
            }

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error loading config: %s", e)
            return {}

    def _ensure_warmup(self):
        """
        Thread-safe warm-up of models before batch processing.
        Ensures embedding model and questionnaire engine are preloaded.
        Called once before processing the first document in a batch.
        
        Thread-safety: Uses double-checked locking pattern with _warmup_lock
        to ensure warm-up executes exactly once across parallel workers.
        
        Validates:
        - Orchestrator warm_up() method is invoked
        - Embedding model connection pool is accessible
        - Singleton model instance is loaded into memory
        """
        if self._warmup_done:
            return
            
        with self._warmup_lock:
            if self._warmup_done:  # Double-check pattern
                return
                
            logger.info("🔥 Warming up models (embedding + questionnaire)...")
            try:
                # Invoke orchestrator warm_up() method
                if hasattr(self.orchestrator, 'warm_up'):
                    self.orchestrator.warm_up()
                elif hasattr(self.orchestrator, 'warmup_models'):
                    # Fallback for compatibility
                    self.orchestrator.warmup_models()
                    
                # Verify connection pool state
                if hasattr(self.orchestrator, '_get_shared_embedding_model'):
                    model = self.orchestrator._get_shared_embedding_model()
                    logger.info("✅ Embedding model connection pool validated: %s", type(model).__name__)
                
                self._warmup_done = True
                logger.info("✅ Warm-up complete - ready for parallel batch processing")
                
            except Exception as e:
                logger.warning("⚠️ Warm-up encountered issue (non-fatal): %s", e)
                # Still mark as done to avoid repeated failures
                self._warmup_done = True

    def evaluate(
        self,
        pdm_path: str,
        municipality: str = "",
        department: str = "",
        export_json: bool = True,
        output_dir: Optional[str] = "output"
    ) -> Dict[str, Any]:
        """
        Execute complete evaluation pipeline.

        Args:
            pdm_path: Path to PDM document
            municipality: Municipality name
            department: Department name
            export_json: Whether to export results to JSON
            output_dir: Output directory for results

        Returns:
            Complete evaluation results with both evaluators
        """
        logger.info("="*80)
        logger.info("UNIFIED EVALUATION: %s", pdm_path)
        logger.info("="*80)

        start_time = datetime.now()

        # WARM-UP: Preload models before batch processing (thread-safe, one-time)
        self._ensure_warmup()

        # PRE-EXECUTION VALIDATION
        logger.info("→ Running pre-execution validation...")
        pre_valid, pre_report = self.system_validator.validate_pre_execution()

        if not pre_valid:
            logger.warning("⚠️ Pre-execution validation had warnings, continuing...")

        # STEP 1: RUN CANONICAL PIPELINE
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Canonical Pipeline Execution")
        logger.info("="*80)

        try:
            pipeline_results = self.orchestrator.process_plan(pdm_path)
        except Exception as e:
            logger.error("Pipeline execution failed: %s", e)
            return {
                "status": "pipeline_error",
                "error": str(e),
                "timestamp": start_time.isoformat()
            }

        if "error" in pipeline_results:
            logger.error("Pipeline returned error: %s", pipeline_results['error'])
            return {
                "status": "pipeline_error",
                "pipeline_results": pipeline_results,
                "timestamp": start_time.isoformat()
            }

        # Get evidence registry reference
        evidence_registry = self.orchestrator.evidence_registry

        logger.info("✅ Pipeline completed: %s evidence items", len(evidence_registry))

        # STEP 2: DECÁLOGO EVALUATION
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Decálogo Evaluation")
        logger.info("="*80)

        decalogo_results = {}
        if self.config.get("evaluators", {}).get("decalogo", {}).get("enabled", True):
            try:
                decalogo_results = self._run_decalogo_evaluation(
                    pdm_path, evidence_registry, municipality, department
                )
                logger.info("✅ Decálogo evaluation completed")
            except Exception as e:
                logger.error("Decálogo evaluation failed: %s", e)
                decalogo_results = {"error": str(e)}
        else:
            logger.info("→ Decálogo evaluation disabled in config")

        # STEP 3: QUESTIONNAIRE EVALUATION
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Questionnaire Evaluation (300 questions)")
        logger.info("="*80)

        questionnaire_results = {}
        if self.config.get("evaluators", {}).get("questionnaire", {}).get("enabled", True):
            if QUESTIONNAIRE_AVAILABLE:
                try:
                    questionnaire_results = self._run_questionnaire_evaluation(
                        pdm_path, evidence_registry
                    )
                    logger.info("✅ Questionnaire evaluation completed")
                except Exception as e:
                    logger.error("Questionnaire evaluation failed: %s", e)
                    questionnaire_results = {"error": str(e)}
            else:
                logger.warning("⚠️ Questionnaire evaluator not available")
                questionnaire_results = {"error": "Evaluator not available"}
        else:
            logger.info("→ Questionnaire evaluation disabled in config")

        # STEP 4: POST-EXECUTION VALIDATION
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Post-Execution Validation")
        logger.info("="*80)

        # Build complete results for validation
        complete_results = {
            "plan_path": pdm_path,
            "executed_nodes": pipeline_results.get("executed_nodes", []),
            "evidence_registry": {
                "statistics": evidence_registry.get_statistics()
            },
            "evaluations": {
                "decalogo": decalogo_results,
                "questionnaire": questionnaire_results
            },
            "immutability_proof": pipeline_results.get("immutability_proof", {})
        }

        post_valid, post_report = self.system_validator.validate_post_execution(complete_results)

        if post_valid:
            logger.info("✅ Post-execution validation passed")
        else:
            logger.warning("⚠️ Post-execution validation warnings detected")

        # BUILD FINAL RESULTS
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        final_results = {
            "status": "success",
            "metadata": {
                "pdm_path": pdm_path,
                "municipality": municipality,
                "department": department,
                "execution_start": start_time.isoformat(),
                "execution_end": end_time.isoformat(),
                "execution_time_seconds": execution_time
            },
            "pipeline": {
                "executed_nodes": pipeline_results.get("executed_nodes", []),
                "node_results": pipeline_results,
                "execution_summary": pipeline_results.get("execution_summary", {})
            },
            "evidence_registry": {
                "statistics": evidence_registry.get_statistics(),
                "deterministic_hash": evidence_registry.deterministic_hash()
            },
            "evaluations": {
                "decalogo": decalogo_results,
                "questionnaire": questionnaire_results
            },
            "validation": {
                "pre_execution": pre_report,
                "post_execution": post_report
            },
            "immutability_proof": pipeline_results.get("immutability_proof", {})
        }

        # EXPORT RESULTS
        if export_json and output_dir:
            self._export_results(final_results, output_dir, evidence_registry)

        logger.info("\n" + "="*80)
        logger.info("✅ UNIFIED EVALUATION COMPLETED")
        logger.info("   Total time: %.2fs", execution_time)
        logger.info("   Evidence items: %s", len(evidence_registry))
        logger.info("   Evidence hash: %s...", evidence_registry.deterministic_hash()[:16])
        logger.info("="*80)

        return final_results

    @staticmethod
    def _run_decalogo_evaluation(
        _pdm_path: str,
        evidence_registry: EvidenceRegistry,
        municipality: str,
        department: str
    ) -> Dict[str, Any]:
        """
        Run Decálogo evaluation consuming evidence registry.

        Args:
            _pdm_path: Path to PDM document
            evidence_registry: Frozen evidence registry from pipeline
            municipality: Municipality name
            department: Department name

        Returns:
            Decálogo evaluation results
        """
        logger.info("Running Decálogo evaluation with EvidenceRegistry...")

        try:
            # Use central path resolver
            from repo_paths import get_decalogo_path
            decalogo_json = get_decalogo_path()

            with open(decalogo_json, 'r', encoding='utf-8') as f:
                decalogo_data = json.load(f)

            questions = decalogo_data.get("questions", [])
            logger.info("→ Loaded %s questions from %s", len(questions), decalogo_json.name)

            # Evaluate each question using evidence from registry
            dimension_results = {}
            all_scores = []

            for question in questions:
                qid = question.get("id")
                dimension = question.get("dimension")

                # Get evidence for this question
                evidence_list = evidence_registry.for_question(qid)

                # Score based on evidence availability and confidence
                if evidence_list:
                    avg_confidence = sum(e.confidence for e in evidence_list) / len(evidence_list)
                    score = 3.0 if avg_confidence > 0.7 else 2.0 if avg_confidence > 0.4 else 1.0
                else:
                    score = 0.0

                all_scores.append(score)

                # Aggregate by dimension
                if dimension not in dimension_results:
                    dimension_results[dimension] = {
                        "scores": [],
                        "questions_evaluated": 0
                    }

                dimension_results[dimension]["scores"].append(score)
                dimension_results[dimension]["questions_evaluated"] += 1

            # Calculate dimension averages
            dimension_summary = {}
            for dim, data in dimension_results.items():
                scores = data["scores"]
                max_score = len(scores) * 3.0
                percentage = (sum(scores) / max_score * 100) if max_score > 0 else 0.0

                dimension_summary[dim] = {
                    "dimension_name": f"Dimensión {dim}",
                    "score": round(sum(scores), 2),
                    "max_score": max_score,
                    "percentage": round(percentage, 1),
                    "questions_evaluated": data["questions_evaluated"]
                }

            # Calculate overall score
            total_score = sum(all_scores)
            max_possible = len(all_scores) * 3.0
            overall_percentage = (total_score / max_possible * 100) if max_possible > 0 else 0.0

            # Classify band
            if overall_percentage >= 85:
                band = "EXCELENTE"
            elif overall_percentage >= 70:
                band = "BUENO"
            elif overall_percentage >= 55:
                band = "SATISFACTORIO"
            elif overall_percentage >= 40:
                band = "INSUFICIENTE"
            else:
                band = "DEFICIENTE"

            logger.info("✅ Decálogo: %s questions evaluated", len(questions))
            logger.info("   Overall score: %.1f%% (%s)", overall_percentage, band)

            return {
                "status": "completed",
                "total_questions": len(questions),
                "answered_questions": len(questions),
                "scores_by_dimension": dimension_summary,
                "overall_score": round(total_score, 2),
                "overall_max": max_possible,
                "overall_percentage": round(overall_percentage, 1),
                "score_band": band,
                "evidence_consumed": True,
                "evidence_hash_verified": evidence_registry.deterministic_hash(),
                "municipality": municipality,
                "department": department
            }

        except Exception as e:
            logger.error("Decálogo evaluation failed: %s", e, exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "evidence_consumed": False
            }

    def _run_questionnaire_evaluation(
        self,
        _pdm_path: str,
        evidence_registry: EvidenceRegistry
    ) -> Dict[str, Any]:
        """
        Run questionnaire evaluation (300 questions) consuming evidence registry.

        Args:
            _pdm_path: Path to PDM document
            evidence_registry: Frozen evidence registry

        Returns:
            Questionnaire evaluation results
        """
        logger.info("Running Questionnaire evaluation with EvidenceRegistry...")

        try:
            # Initialize questionnaire engine
            self.questionnaire_evaluator = QuestionnaireEngine()

            # Get parallel execution config
            parallel_config = self.config.get("parallel_processing", {})
            max_workers = parallel_config.get("max_workers", 4)
            parallel_enabled = parallel_config.get("enabled", True) and \
                              "questionnaire_engine" in parallel_config.get("components", [])

            # Execute evaluation consuming evidence registry
            if parallel_enabled:
                logger.info("→ Using parallel execution (max_workers=%s)", max_workers)
                results = self.questionnaire_evaluator.execute_full_evaluation_parallel(
                    evidence_registry=evidence_registry,
                    municipality=self.config.get("municipality", ""),
                    department=self.config.get("department", ""),
                    max_workers=max_workers
                )
            else:
                logger.info("→ Using sequential execution")
                results = self.questionnaire_evaluator.execute_full_evaluation_parallel(
                    evidence_registry=evidence_registry,
                    municipality="",
                    department="",
                    max_workers=1
                )

            logger.info("✅ Questionnaire: %s questions evaluated", results['metadata']['total_evaluations'])

            return {
                "status": "completed",
                "total_questions": results['metadata']['total_evaluations'],
                "answered_questions": results['questionnaire_structure']['questions_evaluated'],
                "execution_time_seconds": results['metadata']['execution_time_seconds'],
                "parallel_execution": parallel_enabled,
                "evidence_consumed": True,
                "evidence_hash": results['metadata']['evidence_hash'],
                "overall_score": results['results']['global'].score_percentage,
                "classification": results['results']['global'].classification.name,
                "detailed_results": results
            }

        except Exception as e:
            logger.error("Questionnaire evaluation failed: %s", e, exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "evidence_consumed": False
            }

    @staticmethod
    def _export_results(
        results: Dict[str, Any],
        output_dir: str,
        evidence_registry: EvidenceRegistry
    ):
        """Export results to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_name = Path(results["metadata"]["pdm_path"]).stem

        # Export main results
        results_file = output_path / f"{plan_name}_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("→ Exported results to %s", results_file)

        # Export evidence registry
        evidence_file = output_path / f"{plan_name}_evidence_{timestamp}.json"
        evidence_registry.export_to_json(str(evidence_file))
        logger.info("→ Exported evidence registry to %s", evidence_file)


def main():
    """CLI entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python unified_evaluation_pipeline.py <pdm_path> [municipality] [department]")
        sys.exit(1)

    pdm_path = sys.argv[1]
    municipality = sys.argv[2] if len(sys.argv) > 2 else ""
    department = sys.argv[3] if len(sys.argv) > 3 else ""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run evaluation
    pipeline = UnifiedEvaluationPipeline()
    results = pipeline.evaluate(
        pdm_path=pdm_path,
        municipality=municipality,
        department=department,
        export_json=True
    )

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Status: {results['status']}")
    print(f"Execution time: {results['metadata']['execution_time_seconds']:.2f}s")
    print(f"Evidence items: {results['evidence_registry']['statistics']['total_evidence']}")
    print(f"Evidence hash: {results['evidence_registry']['deterministic_hash']}")
    print("="*80)


if __name__ == "__main__":
    main()
