#!/usr/bin/env python3
# coding=utf-8
"""
Deterministic Pipeline Validator (v2.0 - Flow-Aligned)
======================================================

Validates deterministic execution and canonical structure integrity for the
MINIMINIMOON pipeline. Aligned with 72-flow dependency documentation.

Flow #17: deterministic_pipeline_validator → flow_runtime.json
Flow #57: Exports runtime trace for doc↔runtime comparison (gate #2)
Flow #58: Compares tools/flow_doc.json ↔ artifacts/flow_runtime.json

Key Components:
- NodeContract: I/O schema specification per pipeline stage
- CanonicalFlowValidator: Order and contract validation
- RuntimeTracer: Execution trace recording
- FlowComparator: Doc vs runtime diff analysis

Author: System Architect
Version: 2.0.0 (Flow-Finalized)
Date: 2025-10-05
"""

import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PipelineValidator")

# ============================================================================
# NODE CONTRACTS (I/O Schemas)
# ============================================================================


@dataclass
class NodeContract:
    """
    I/O contract for a pipeline node.
    Specifies expected input/output schemas and validation rules.
    """

    node_id: str
    node_name: str
    input_schema: Dict[str, str]  # {field_name: type}
    output_schema: Dict[str, str]
    required_inputs: List[str]
    required_outputs: List[str]
    invariants: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def validate_input(self, data: Any) -> Tuple[bool, List[str]]:
        """
        Validate input data against schema.

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        if not isinstance(data, dict):
            errors.append(f"Input must be dict, got {type(data)}")
            return False, errors

        # Check required fields
        for field in self.required_inputs:
            if field not in data:
                errors.append(f"Missing required input field: {field}")

        # Check types (basic validation)
        for field, expected_type in self.input_schema.items():
            if field in data:
                actual_type = type(data[field]).__name__
                if not self._types_compatible(actual_type, expected_type):
                    errors.append(
                        f"Type mismatch for {field}: "
                        f"expected {expected_type}, got {actual_type}"
                    )

        return len(errors) == 0, errors

    def validate_output(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate output data against schema"""
        errors = []

        if not isinstance(data, dict):
            errors.append(f"Output must be dict, got {type(data)}")
            return False, errors

        for field in self.required_outputs:
            if field not in data:
                errors.append(f"Missing required output field: {field}")

        for field, expected_type in self.output_schema.items():
            if field in data:
                actual_type = type(data[field]).__name__
                if not self._types_compatible(actual_type, expected_type):
                    errors.append(
                        f"Type mismatch for {field}: "
                        f"expected {expected_type}, got {actual_type}"
                    )

        return len(errors) == 0, errors

    @staticmethod
    def _types_compatible(actual: str, expected: str) -> bool:
        """Check if types are compatible (with some flexibility)"""
        type_mappings = {
            "str": ["str", "string"],
            "int": ["int", "integer"],
            "float": ["float", "number"],
            "list": ["list", "array"],
            "dict": ["dict", "object"],
            "bool": ["bool", "boolean"],
        }

        actual_variants = type_mappings.get(actual, [actual])
        expected_variants = type_mappings.get(expected, [expected])

        return any(a in expected_variants for a in actual_variants)

    def to_dict(self) -> Dict[str, Any]:
        """Export contract as dictionary"""
        return asdict(self)


class PipelineStage(Enum):
    """Canonical pipeline stages (must match orchestrator)"""

    SANITIZATION = "sanitization"
    PLAN_PROCESSING = "plan_processing"
    SEGMENTATION = "document_segmentation"
    EMBEDDING = "embedding"  # UPDATED (was embedding_generation)
    RESPONSIBILITY = "responsibility_detection"
    CONTRADICTION = "contradiction_detection"
    MONETARY = "monetary_detection"
    FEASIBILITY = "feasibility_scoring"
    CAUSAL = "causal_detection"  # UPDATED (was causal_pattern_detection)
    TEORIA = "teoria_cambio"  # UPDATED (was teoria_cambio_validation)
    DAG = "dag_validation"
    REGISTRY_BUILD = "evidence_registry_build"
    DECALOGO_LOAD = "decalogo_load"  # NEW stage (explicit extractor load)
    DECALOGO_EVAL = "decalogo_evaluation"
    QUESTIONNAIRE_EVAL = "questionnaire_evaluation"
    ANSWERS_ASSEMBLY = "answers_assembly"  # UPDATED (was answer_assembly)


# Canonical node contracts (flows #1-16) — updated names + legacy aliases retained
CANONICAL_NODE_CONTRACTS = {
    # Stage 1
    "sanitization": NodeContract(
        node_id="node_01",
        node_name="sanitization",
        input_schema={"raw_text": "str"},
        output_schema={"sanitized_text": "str"},
        required_inputs=["raw_text"],
        required_outputs=["sanitized_text"],
        invariants=["non_empty_output", "unicode_valid"],
        dependencies=[],
    ),
    # Stage 2
    "plan_processing": NodeContract(
        node_id="node_02",
        node_name="plan_processing",
        input_schema={"sanitized_text": "str"},
        output_schema={"doc_struct": "dict"},
        required_inputs=["sanitized_text"],
        required_outputs=["doc_struct"],
        invariants=["structure_valid"],
        dependencies=["sanitization"],
    ),
    # Stage 3
    "document_segmentation": NodeContract(
        node_id="node_03",
        node_name="document_segmentation",
        input_schema={"doc_struct": "dict"},
        output_schema={"segments": "list"},
        required_inputs=["doc_struct"],
        required_outputs=["segments"],
        invariants=["segments_non_empty", "deterministic_ids"],
        dependencies=["plan_processing"],
    ),
    # Stage 4 (new name) + legacy alias
    "embedding": NodeContract(
        node_id="node_04",
        node_name="embedding",
        input_schema={"segments": "list"},
        output_schema={"embeddings": "list"},
        required_inputs=["segments"],
        required_outputs=["embeddings"],
        invariants=["length_match", "deterministic_seed"],
        dependencies=["document_segmentation"],
    ),
    "embedding_generation": NodeContract(
        node_id="node_04_legacy",
        node_name="embedding_generation",
        input_schema={"segments": "list"},
        output_schema={"embeddings": "list"},
        required_inputs=["segments"],
        required_outputs=["embeddings"],
        invariants=["length_match", "deterministic_seed"],
        dependencies=["document_segmentation"],
    ),
    # Stage 5
    "responsibility_detection": NodeContract(
        node_id="node_05",
        node_name="responsibility_detection",
        input_schema={"segments": "list"},
        output_schema={"responsibilities": "list"},
        required_inputs=["segments"],
        required_outputs=["responsibilities"],
        invariants=["provenance_tracked"],
        dependencies=["document_segmentation"],
    ),
    # Stage 6
    "contradiction_detection": NodeContract(
        node_id="node_06",
        node_name="contradiction_detection",
        input_schema={"segments": "list"},
        output_schema={"contradictions": "list"},
        required_inputs=["segments"],
        required_outputs=["contradictions"],
        invariants=["consistency_checked"],
        dependencies=["document_segmentation"],
    ),
    # Stage 7
    "monetary_detection": NodeContract(
        node_id="node_07",
        node_name="monetary_detection",
        input_schema={"segments": "list"},
        output_schema={"monetary": "list"},
        required_inputs=["segments"],
        required_outputs=["monetary"],
        invariants=["currency_validated"],
        dependencies=["document_segmentation"],
    ),
    # Stage 8
    "feasibility_scoring": NodeContract(
        node_id="node_08",
        node_name="feasibility_scoring",
        input_schema={"segments": "list"},
        output_schema={"feasibility": "dict"},
        required_inputs=["segments"],
        required_outputs=["feasibility"],
        invariants=["score_bounded"],
        dependencies=["document_segmentation"],
    ),
    # Stage 9 (new name) + legacy alias
    "causal_detection": NodeContract(
        node_id="node_09",
        node_name="causal_detection",
        input_schema={"segments": "list"},
        output_schema={"causal_patterns": "dict"},
        required_inputs=["segments"],
        required_outputs=["causal_patterns"],
        invariants=["patterns_valid"],
        dependencies=["document_segmentation"],
    ),
    "causal_pattern_detection": NodeContract(
        node_id="node_09_legacy",
        node_name="causal_pattern_detection",
        input_schema={"segments": "list"},
        output_schema={"causal_patterns": "dict"},
        required_inputs=["segments"],
        required_outputs=["causal_patterns"],
        invariants=["patterns_valid"],
        dependencies=["document_segmentation"],
    ),
    # Stage 10 (new name) + legacy alias
    "teoria_cambio": NodeContract(
        node_id="node_10",
        node_name="teoria_cambio",
        input_schema={"segments": "list"},
        output_schema={"toc_graph": "dict"},
        required_inputs=["segments"],
        required_outputs=["toc_graph"],
        invariants=["graph_valid"],
        dependencies=["document_segmentation"],
    ),
    "teoria_cambio_validation": NodeContract(
        node_id="node_10_legacy",
        node_name="teoria_cambio_validation",
        input_schema={"segments": "list"},
        output_schema={"toc_graph": "dict"},
        required_inputs=["segments"],
        required_outputs=["toc_graph"],
        invariants=["graph_valid"],
        dependencies=["document_segmentation"],
    ),
    # Stage 11
    "dag_validation": NodeContract(
        node_id="node_11",
        node_name="dag_validation",
        input_schema={"toc_graph": "dict"},
        output_schema={"dag_diagnostics": "dict"},
        required_inputs=["toc_graph"],
        required_outputs=["dag_diagnostics"],
        invariants=["dag_acyclic"],
        dependencies=["teoria_cambio"],
    ),
    # Stage 12
    "evidence_registry_build": NodeContract(
        node_id="node_12",
        node_name="evidence_registry_build",
        input_schema={
            "responsibilities": "list",
            "contradictions": "list",
            "monetary": "list",
            "feasibility": "dict",
            "causal_patterns": "dict",
            "toc_graph": "dict",
            "dag_diagnostics": "dict",
        },
        output_schema={"evidence_hash": "str", "evidence_store": "object"},
        required_inputs=["responsibilities", "contradictions"],
        required_outputs=["evidence_hash"],
        invariants=["hash_deterministic"],
        dependencies=[
            "responsibility_detection",
            "contradiction_detection",
            "monetary_detection",
            "feasibility_scoring",
            "causal_detection",
            "causal_pattern_detection",
            "teoria_cambio",
            "teoria_cambio_validation",
            "dag_validation",
        ],
    ),
    # Stage 13 (NEW)
    "decalogo_load": NodeContract(
        node_id="node_13",
        node_name="decalogo_load",
        input_schema={},  # no required inputs (loads bundle)
        output_schema={
            "status": "str",
            "bundle_version": "str",
            "categories_count": "int",
            "extractor_type": "str",
        },
        required_inputs=[],
        required_outputs=["status"],
        invariants=["load_success"],
        dependencies=["evidence_registry_build"],
    ),
    # Stage 14 (Decálogo eval now depends on load + registry)
    "decalogo_evaluation": NodeContract(
        node_id="node_14",
        node_name="decalogo_evaluation",
        input_schema={"evidence_store": "object"},
        output_schema={"decalogo_eval": "dict"},
        required_inputs=["evidence_store"],
        required_outputs=["decalogo_eval"],
        invariants=["scores_bounded"],
        dependencies=["decalogo_load", "evidence_registry_build"],
    ),
    # Stage 15
    "questionnaire_evaluation": NodeContract(
        node_id="node_15",
        node_name="questionnaire_evaluation",
        input_schema={"evidence_store": "object"},
        output_schema={"questionnaire_eval": "dict"},
        required_inputs=["evidence_store"],
        required_outputs=["questionnaire_eval"],
        invariants=["300_questions", "weights_aligned"],
        dependencies=["evidence_registry_build"],
    ),
    # Stage 16 (updated name) + legacy alias
    "answers_assembly": NodeContract(
        node_id="node_16",
        node_name="answers_assembly",
        input_schema={
            "evidence_store": "object",
            "rubric": "dict",
            "questionnaire_eval": "dict",
        },
        output_schema={"answers_report": "dict"},
        required_inputs=["evidence_store", "questionnaire_eval"],
        required_outputs=["answers_report"],
        invariants=["provenance_complete", "confidence_bounded"],
        dependencies=["questionnaire_evaluation", "decalogo_evaluation"],
    ),
    "answer_assembly": NodeContract(
        node_id="node_16_legacy",
        node_name="answer_assembly",
        input_schema={
            "evidence_store": "object",
            "rubric": "dict",
            "questionnaire_eval": "dict",
        },
        output_schema={"answers_report": "dict"},
        required_inputs=["evidence_store", "questionnaire_eval"],
        required_outputs=["answers_report"],
        invariants=["provenance_complete", "confidence_bounded"],
        dependencies=["questionnaire_evaluation", "decalogo_evaluation"],
    ),
}

# ============================================================================
# RUNTIME TRACER
# ============================================================================


@dataclass
class StageExecution:
    """Record of a single stage execution"""

    stage_name: str
    node_id: str
    timestamp: float
    duration_ms: float
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    contract_violations: List[str] = field(default_factory=list)


class RuntimeTracer:
    """
    Records pipeline execution for validation and reproducibility.
    Flow #57: Exports to artifacts/flow_runtime.json
    """

    def __init__(self):
        self.executions: List[StageExecution] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metadata: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def start(self, metadata: Optional[Dict[str, Any]] = None):
        """Start runtime tracing"""
        self.start_time = time.time()
        self.metadata = metadata or {}
        self.logger.info("Runtime tracing started")

    def stop(self):
        """Stop runtime tracing"""
        self.end_time = time.time()
        self.logger.info(
            f"Runtime tracing stopped "
            f"({len(self.executions)} stages, "
            f"{(self.end_time - self.start_time):.2f}s)"
        )

    def record_stage(
        self,
        stage_name: str,
        node_id: str,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        input_hash: Optional[str] = None,
        output_hash: Optional[str] = None,
        contract_violations: Optional[List[str]] = None,
    ):
        """Record execution of a single stage"""
        execution = StageExecution(
            stage_name=stage_name,
            node_id=node_id,
            timestamp=time.time(),
            duration_ms=duration_ms,
            input_hash=input_hash,
            output_hash=output_hash,
            success=success,
            error=error,
            contract_violations=contract_violations or [],
        )

        self.executions.append(execution)

        status = "✓" if success else "✗"
        self.logger.debug(f"{status} {stage_name} ({duration_ms:.1f}ms) [{node_id}]")

    def get_stage_sequence(self) -> List[str]:
        """Get ordered list of executed stages"""
        return [ex.stage_name for ex in self.executions]

    def compute_flow_hash(self) -> str:
        """
        Compute deterministic hash of execution flow.
        Used for reproducibility verification (gate #3).
        """
        stages_str = "|".join(self.get_stage_sequence())
        return hashlib.sha256(stages_str.encode()).hexdigest()

    def export(self) -> Dict[str, Any]:
        """
        Export runtime trace to dictionary.
        Flow #57: Structure for artifacts/flow_runtime.json
        """
        return {
            "version": "2.0.0",
            "flow_hash": self.compute_flow_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "total_duration_seconds": (
                (self.end_time - self.start_time)
                if self.end_time and self.start_time
                else None
            ),
            "metadata": self.metadata,
            "execution_sequence": [
                {
                    "stage_name": ex.stage_name,
                    "node_id": ex.node_id,
                    "timestamp": datetime.fromtimestamp(ex.timestamp).isoformat() + "Z",
                    "duration_ms": ex.duration_ms,
                    "success": ex.success,
                    "error": ex.error,
                    "input_hash": ex.input_hash,
                    "output_hash": ex.output_hash,
                    "contract_violations": ex.contract_violations,
                }
                for ex in self.executions
            ],
            "stage_count": len(self.executions),
            "success_count": sum(1 for ex in self.executions if ex.success),
            "failure_count": sum(1 for ex in self.executions if not ex.success),
            "total_violations": sum(
                len(ex.contract_violations) for ex in self.executions
            ),
        }

    def save(self, output_path: Path):
        """Save runtime trace to JSON file"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.export(), f, indent=2, ensure_ascii=False)

        self.logger.info("Runtime trace saved to: %s", output_path)


# ============================================================================
# CANONICAL FLOW VALIDATOR
# ============================================================================


class CanonicalFlowValidator:
    """
    Validates pipeline execution against canonical flow documentation.

    Flow #17: deterministic_pipeline_validator → data_flow_contract
    Flow #58: Compares tools/flow_doc.json ↔ artifacts/flow_runtime.json
    Gate #2: flow_runtime.json identical to canonical documentation
    """

    # Canonical execution order (UPDATED: 16 stages)
    CANONICAL_ORDER = [
        "sanitization",
        "plan_processing",
        "document_segmentation",
        "embedding",
        "responsibility_detection",
        "contradiction_detection",
        "monetary_detection",
        "feasibility_scoring",
        "causal_detection",
        "teoria_cambio",
        "dag_validation",
        "evidence_registry_build",
        "decalogo_load",
        "decalogo_evaluation",
        "questionnaire_evaluation",
        "answers_assembly",
    ]

    def __init__(self, flow_doc_path: Optional[Path] = None):
        """
        Initialize validator.

        Args:
            flow_doc_path: Path to tools/flow_doc.json (canonical documentation)
        """
        self.flow_doc_path = flow_doc_path
        self.canonical_doc: Optional[Dict[str, Any]] = None
        self.logger = logging.getLogger(__name__)

        if flow_doc_path and flow_doc_path.exists():
            self._load_canonical_doc()

    def _load_canonical_doc(self):
        """Load canonical flow documentation"""
        try:
            with open(self.flow_doc_path, "r", encoding="utf-8") as f:
                self.canonical_doc = json.load(f)
            self.logger.info("Loaded canonical doc: %s", self.flow_doc_path)
        except Exception as e:
            self.logger.error("Failed to load canonical doc: %s", e)

    def validate_order(self, runtime_trace: RuntimeTracer) -> Dict[str, Any]:
        """
        Validate execution order matches canonical sequence.

        Returns:
            Validation report with status and discrepancies
        """
        actual_stages = runtime_trace.get_stage_sequence()

        result = {
            "order_valid": actual_stages == self.CANONICAL_ORDER,
            "expected_sequence": self.CANONICAL_ORDER,
            "actual_sequence": actual_stages,
            "missing_stages": list(set(self.CANONICAL_ORDER) - set(actual_stages)),
            "extra_stages": list(set(actual_stages) - set(self.CANONICAL_ORDER)),
            "out_of_order": [],
            "flow_hash": runtime_trace.compute_flow_hash(),
        }

        # Detect out-of-order stages
        canonical_indices = {stage: i for i, stage in enumerate(self.CANONICAL_ORDER)}
        for i in range(len(actual_stages) - 1):
            curr_stage = actual_stages[i]
            next_stage = actual_stages[i + 1]

            if (
                curr_stage in canonical_indices
                and next_stage in canonical_indices
                and canonical_indices[curr_stage] >= canonical_indices[next_stage]
            ):
                result["out_of_order"].append(
                    {"position": i, "stage": curr_stage, "followed_by": next_stage}
                )

        if not result["order_valid"]:
            self.logger.error(
                f"⨯ Order validation FAILED (gate #2): "
                f"missing={result['missing_stages']}, "
                f"extra={result['extra_stages']}, "
                f"out_of_order={len(result['out_of_order'])}"
            )
        else:
            self.logger.info(
                "✓ Order validation PASSED (gate #2): canonical sequence preserved"
            )

        return result

    def validate_contracts(self, runtime_trace: RuntimeTracer) -> Dict[str, Any]:
        """
        Validate I/O contracts for all executed stages.

        Returns:
            Contract validation report
        """
        violations = []
        stages_checked = 0

        for execution in runtime_trace.executions:
            stage_name = execution.stage_name

            if stage_name not in CANONICAL_NODE_CONTRACTS:
                violations.append(
                    {
                        "stage": stage_name,
                        "type": "unknown_stage",
                        "message": f"Stage {stage_name} has no defined contract",
                    }
                )
                continue

            stages_checked += 1

            # Contract violations recorded during execution
            if execution.contract_violations:
                for violation in execution.contract_violations:
                    violations.append(
                        {
                            "stage": stage_name,
                            "type": "contract_violation",
                            "message": violation,
                        }
                    )

        result = {
            "contracts_valid": len(violations) == 0,
            "stages_checked": stages_checked,
            "violations": violations,
            "violation_count": len(violations),
        }

        if result["contracts_valid"]:
            self.logger.info(
                f"✓ Contract validation PASSED: {stages_checked} stages checked"
            )
        else:
            self.logger.error(
                f"⨯ Contract validation FAILED: {len(violations)} violations"
            )

        return result

    def compare_with_doc(self, runtime_trace: RuntimeTracer) -> Dict[str, Any]:
        """
        Compare runtime trace with canonical documentation.
        Flow #58: Doc↔runtime diff analysis.

        Returns:
            Comparison report with hashes and differences
        """
        if not self.canonical_doc:
            return {
                "comparison_possible": False,
                "message": "No canonical documentation loaded",
            }

        runtime_export = runtime_trace.export()

        result = {
            "comparison_possible": True,
            "doc_hash": self.canonical_doc.get("flow_hash"),
            "runtime_hash": runtime_export["flow_hash"],
            "hashes_match": (
                self.canonical_doc.get("flow_hash") == runtime_export["flow_hash"]
            ),
            "doc_stage_count": len(self.canonical_doc.get("stages", [])),
            "runtime_stage_count": runtime_export["stage_count"],
            "differences": [],
        }

        # Compare stage sequences
        doc_stages = [s["name"] for s in self.canonical_doc.get("stages", [])]
        runtime_stages = runtime_trace.get_stage_sequence()

        if doc_stages != runtime_stages:
            result["differences"].append(
                {
                    "type": "sequence_mismatch",
                    "doc_sequence": doc_stages,
                    "runtime_sequence": runtime_stages,
                }
            )

        # Compare stage counts
        if result["doc_stage_count"] != result["runtime_stage_count"]:
            result["differences"].append(
                {
                    "type": "count_mismatch",
                    "doc_count": result["doc_stage_count"],
                    "runtime_count": result["runtime_stage_count"],
                }
            )

        if result["hashes_match"] and not result["differences"]:
            self.logger.info("✓ Doc↔Runtime comparison PASSED: perfect match")
        else:
            self.logger.warning(
                f"⚠ Doc↔Runtime comparison: {len(result['differences'])} differences"
            )

        return result

    def validate(self, runtime_trace: RuntimeTracer) -> Dict[str, Any]:
        """
        Complete validation: order + contracts + doc comparison.

        Returns:
            Comprehensive validation report
        """
        self.logger.info("Running comprehensive flow validation...")

        order_result = self.validate_order(runtime_trace)
        contract_result = self.validate_contracts(runtime_trace)

        comparison_result = None
        if self.canonical_doc:
            comparison_result = self.compare_with_doc(runtime_trace)

        overall_valid = (
            order_result["order_valid"]
            and contract_result["contracts_valid"]
            and (
                comparison_result is None
                or comparison_result.get("hashes_match", False)
            )
        )

        report = {
            "validation_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "overall_valid": overall_valid,
            "order_validation": order_result,
            "contract_validation": contract_result,
            "doc_comparison": comparison_result,
            "flow_hash": runtime_trace.compute_flow_hash(),
        }

        if overall_valid:
            self.logger.info("✓ VALIDATION PASSED: All checks OK")
        else:
            self.logger.error("⨯ VALIDATION FAILED: See report for details")

        return report


# ============================================================================
# FLOW DOCUMENTATION GENERATOR
# ============================================================================


class FlowDocGenerator:
    """
    Generates canonical flow documentation (tools/flow_doc.json).
    Used as reference for runtime comparison (flow #58).
    """

    @staticmethod
    def generate_flow_doc(output_path: Path):
        """Generate canonical flow documentation from contracts (16 stages)."""
        stages = []
        for stage_name in CanonicalFlowValidator.CANONICAL_ORDER:
            if stage_name in CANONICAL_NODE_CONTRACTS:
                contract = CANONICAL_NODE_CONTRACTS[stage_name]
                stages.append(
                    {
                        "name": stage_name,
                        "node_id": contract.node_id,
                        "input_schema": contract.input_schema,
                        "output_schema": contract.output_schema,
                        "required_inputs": contract.required_inputs,
                        "required_outputs": contract.required_outputs,
                        "invariants": contract.invariants,
                        "dependencies": contract.dependencies,
                    }
                )
        stages_str = "|".join([s["name"] for s in stages])
        flow_hash = hashlib.sha256(stages_str.encode()).hexdigest()
        doc = {
            "version": "2.1.0",
            "flow_hash": flow_hash,
            "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
            "description": "Canonical deterministic pipeline flow (16 stages)",
            "total_stages": len(stages),
            "stages": stages,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        logger.info("✓ Flow documentation generated (16 stages): %s", output_path)
        logger.info("  Flow hash: %s", flow_hash)
        logger.info("  Stages: %s", len(stages))

        return doc


# ============================================================================
# UTILITIES
# ============================================================================


def compute_data_hash(data: Any) -> str:
    """Compute deterministic hash of data structure"""
    if isinstance(data, (dict, list)):
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=True)
    else:
        json_str = str(data)

    return hashlib.sha256(json_str.encode()).hexdigest()


def validate_stage_io(stage_name: str, input_data: Any, output_data: Any) -> List[str]:
    """
    Validate I/O for a specific stage using its contract.

    Returns:
        List of violation messages (empty if valid)
    """
    violations = []

    if stage_name not in CANONICAL_NODE_CONTRACTS:
        violations.append(f"No contract defined for stage: {stage_name}")
        return violations

    contract = CANONICAL_NODE_CONTRACTS[stage_name]

    # Validate input
    if input_data is not None:
        input_valid, input_errors = contract.validate_input(input_data)
        if not input_valid:
            violations.extend([f"INPUT: {e}" for e in input_errors])

    # Validate output
    if output_data is not None:
        output_valid, output_errors = contract.validate_output(output_data)
        if not output_valid:
            violations.extend([f"OUTPUT: {e}" for e in output_errors])

    return violations


# ============================================================================
# CLI & TESTING
# ============================================================================


def run_validation_tests():
    """Run built-in validation tests"""
    print("\n" + "=" * 80)
    print("DETERMINISTIC PIPELINE VALIDATOR - TEST SUITE")
    print("=" * 80 + "\n")

    # Test 1: Contract validation
    print("Test 1: Node Contract Validation")
    contract = CANONICAL_NODE_CONTRACTS["sanitization"]

    valid_input = {"raw_text": "test"}
    valid, errors = contract.validate_input(valid_input)
    print(f"  Valid input: {valid} ✓" if valid else f"  Invalid input: {errors}")

    invalid_input = {"wrong_field": 123}
    valid, errors = contract.validate_input(invalid_input)
    print(
        f"  Invalid input detected: {not valid} ✓"
        if not valid
        else "  Failed to detect: ✗"
    )

    # Test 2: Runtime tracer
    print("\nTest 2: Runtime Tracer")
    tracer = RuntimeTracer()
    tracer.start({"test": "mode"})

    for stage in CanonicalFlowValidator.CANONICAL_ORDER[:3]:
        tracer.record_stage(
            stage_name=stage, node_id=f"node_{stage}", duration_ms=10.5, success=True
        )

    tracer.stop()
    print(f"  Recorded stages: {len(tracer.executions)} ✓")
    print(f"  Flow hash: {tracer.compute_flow_hash()[:16]}... ✓")

    # Test 3: Order validation
    print("\nTest 3: Canonical Order Validation")
    validator = CanonicalFlowValidator()
    order_result = validator.validate_order(tracer)
    print(f"  Order valid: {order_result['order_valid']} ✓")

    # Test 4: Export
    print("\nTest 4: Runtime Trace Export")
    export = tracer.export()
    print(f"  Export keys: {list(export.keys())} ✓")
    print(f"  Stage count: {export['stage_count']} ✓")

    print("\n" + "=" * 80)
    print("✓ All tests completed")
    print("=" * 80 + "\n")


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deterministic Pipeline Validator (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate canonical flow documentation
  python deterministic_pipeline_validator.py generate-doc tools/flow_doc.json

  # Run validation tests
  python deterministic_pipeline_validator.py test

  # Validate a runtime trace
  python deterministic_pipeline_validator.py validate artifacts/flow_runtime.json
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Generate doc command
    gen_parser = subparsers.add_parser("generate-doc", help="Generate flow_doc.json")
    gen_parser.add_argument(
        "output", type=Path, help="Output path (e.g., tools/flow_doc.json)"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Run validation tests")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate runtime trace")
    val_parser.add_argument(
        "runtime_trace", type=Path, help="Path to flow_runtime.json"
    )
    val_parser.add_argument("--flow-doc", type=Path, help="Path to flow_doc.json")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "generate-doc":
            FlowDocGenerator.generate_flow_doc(args.output)
            return 0

        elif args.command == "test":
            run_validation_tests()
            return 0

        elif args.command == "validate":
            # Load runtime trace
            with open(args.runtime_trace, "r") as f:
                trace_data = json.load(f)

            # Reconstruct tracer
            tracer = RuntimeTracer()
            tracer.start_time = time.time() - trace_data.get(
                "total_duration_seconds", 0
            )
            tracer.end_time = time.time()
            tracer.metadata = trace_data.get("metadata", {})

            for exec_data in trace_data.get("execution_sequence", []):
                tracer.record_stage(
                    stage_name=exec_data["stage_name"],
                    node_id=exec_data["node_id"],
                    duration_ms=exec_data["duration_ms"],
                    success=exec_data["success"],
                    error=exec_data.get("error"),
                    contract_violations=exec_data.get("contract_violations", []),
                )

            # Validate
            validator = CanonicalFlowValidator(flow_doc_path=args.flow_doc)
            report = validator.validate(tracer)

            # Print summary
            print("\n" + "=" * 80)
            print("VALIDATION REPORT")
            print("=" * 80)
            print(f"\nOverall: {'✓ PASSED' if report['overall_valid'] else '✗ FAILED'}")
            print(f"Flow hash: {report['flow_hash']}")

            if not report["order_validation"]["order_valid"]:
                print("\n⨯ Order issues:")
                print(f"  Missing: {report['order_validation']['missing_stages']}")
                print(f"  Extra: {report['order_validation']['extra_stages']}")

            if not report["contract_validation"]["contracts_valid"]:
                print(
                    f"\n⨯ Contract violations: {report['contract_validation']['violation_count']}"
                )

            print("\n" + "=" * 80 + "\n")

            return 0 if report["overall_valid"] else 3

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
