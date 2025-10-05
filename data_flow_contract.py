#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Flow Contract Validator
=============================

Defines and enforces contracts for each node in the canonical MINIMINIMOON pipeline.
Ensures that each component receives valid inputs and produces valid outputs,
maintaining data integrity throughout the flow.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Valid data types in the pipeline"""
    RAW_TEXT = "raw_text"
    SANITIZED_TEXT = "sanitized_text"
    SEGMENTS = "segments"
    EMBEDDINGS = "embeddings"
    ENTITIES = "entities"
    CONTRADICTIONS = "contradictions"
    MONETARY_VALUES = "monetary_values"
    FEASIBILITY_SCORES = "feasibility_scores"
    CAUSAL_PATTERNS = "causal_patterns"
    TEORIA_CAMBIO = "teoria_cambio"
    DAG_STRUCTURE = "dag_structure"
    METADATA = "metadata"


@dataclass
class NodeContract:
    """
    Contract specification for a pipeline node.

    Defines what a node expects as input and what it promises to produce as output.
    """
    node_name: str
    dependencies: List[str] = field(default_factory=list)  # Nodes that must run before
    required_inputs: List[DataType] = field(default_factory=list)
    produces: List[DataType] = field(default_factory=list)
    validators: Dict[str, Callable] = field(default_factory=dict)  # Custom validators
    optional_inputs: List[DataType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate_inputs(self, available_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate that required inputs are available.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check required inputs
        for required in self.required_inputs:
            if required.value not in available_data:
                errors.append(f"Missing required input: {required.value}")

        # Run custom validators
        for validator_name, validator_func in self.validators.items():
            try:
                is_valid, msg = validator_func(available_data)
                if not is_valid:
                    errors.append(f"{validator_name}: {msg}")
            except Exception as e:
                errors.append(f"{validator_name} failed: {str(e)}")

        return len(errors) == 0, errors

    def validate_outputs(self, outputs: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate that produced outputs match contract.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        for expected in self.produces:
            if expected.value not in outputs:
                errors.append(f"Missing expected output: {expected.value}")

        return len(errors) == 0, errors


class CanonicalFlowValidator:
    """
    Validator for the canonical MINIMINIMOON pipeline flow.

    Maintains contracts for all nodes and validates the entire flow.
    """

    def __init__(self):
        """Initialize with canonical node contracts"""
        self.contracts: Dict[str, NodeContract] = {}
        self.execution_order: List[str] = []
        self._build_canonical_contracts()

        logger.info("CanonicalFlowValidator initialized")

    def _build_canonical_contracts(self):
        """Build contracts for all canonical pipeline nodes"""

        # 1. Sanitization
        self.contracts["sanitization"] = NodeContract(
            node_name="sanitization",
            dependencies=[],
            required_inputs=[DataType.RAW_TEXT],
            produces=[DataType.SANITIZED_TEXT],
            validators={
                "non_empty": lambda d: (
                    len(d.get("raw_text", "")) > 0,
                    "Input text cannot be empty"
                )
            }
        )

        # 2. Plan Processing
        self.contracts["plan_processing"] = NodeContract(
            node_name="plan_processing",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.METADATA],
            validators={
                "text_length": lambda d: (
                    len(d.get("sanitized_text", "")) >= 100,
                    "Sanitized text too short for processing"
                )
            }
        )

        # 3. Document Segmentation
        self.contracts["document_segmentation"] = NodeContract(
            node_name="document_segmentation",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.SEGMENTS],
            validators={
                "valid_text": lambda d: (
                    isinstance(d.get("sanitized_text"), str),
                    "Sanitized text must be string"
                )
            }
        )

        # 4. Embedding
        self.contracts["embedding"] = NodeContract(
            node_name="embedding",
            dependencies=["document_segmentation"],
            required_inputs=[DataType.SEGMENTS],
            produces=[DataType.EMBEDDINGS],
            validators={
                "segments_list": lambda d: (
                    isinstance(d.get("segments"), list) and len(d.get("segments", [])) > 0,
                    "Segments must be non-empty list"
                )
            }
        )

        # 5. Responsibility Detection
        self.contracts["responsibility_detection"] = NodeContract(
            node_name="responsibility_detection",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.ENTITIES],
            validators={
                "text_present": lambda d: (
                    len(d.get("sanitized_text", "")) > 50,
                    "Text too short for entity detection"
                )
            }
        )

        # 6. Contradiction Detection
        self.contracts["contradiction_detection"] = NodeContract(
            node_name="contradiction_detection",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.CONTRADICTIONS],
        )

        # 7. Monetary Detection
        self.contracts["monetary_detection"] = NodeContract(
            node_name="monetary_detection",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.MONETARY_VALUES],
        )

        # 8. Feasibility Scoring
        self.contracts["feasibility_scoring"] = NodeContract(
            node_name="feasibility_scoring",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.FEASIBILITY_SCORES],
        )

        # 9. Causal Pattern Detection
        self.contracts["causal_detection"] = NodeContract(
            node_name="causal_detection",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.CAUSAL_PATTERNS],
        )

        # 10. Theory of Change
        self.contracts["teoria_cambio"] = NodeContract(
            node_name="teoria_cambio",
            dependencies=["responsibility_detection", "causal_detection", "monetary_detection"],
            required_inputs=[
                DataType.SANITIZED_TEXT,
                DataType.ENTITIES,
                DataType.CAUSAL_PATTERNS,
                DataType.MONETARY_VALUES
            ],
            produces=[DataType.TEORIA_CAMBIO],
        )

        # 11. DAG Validation
        self.contracts["dag_validation"] = NodeContract(
            node_name="dag_validation",
            dependencies=["teoria_cambio"],
            required_inputs=[DataType.TEORIA_CAMBIO],
            produces=[DataType.DAG_STRUCTURE],
        )

        # Define execution order (canonical)
        self.execution_order = [
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
            "dag_validation"
        ]

        logger.info(f"Built contracts for {len(self.contracts)} nodes")

    def validate_node_execution(
        self,
        node_name: str,
        available_data: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Validate a node's execution (inputs and optionally outputs).

        Args:
            node_name: Name of the node
            available_data: Data available before node execution
            outputs: Data produced by node (optional, for post-validation)

        Returns:
            (is_valid, report_dict)
        """
        if node_name not in self.contracts:
            return False, {
                "valid": False,
                "node": node_name,
                "error": f"Unknown node: {node_name}"
            }

        contract = self.contracts[node_name]
        report = {
            "node": node_name,
            "valid": True,
            "input_validation": {},
            "output_validation": {},
            "errors": []
        }

        # Validate inputs
        input_valid, input_errors = contract.validate_inputs(available_data)
        report["input_validation"] = {
            "valid": input_valid,
            "errors": input_errors
        }

        if not input_valid:
            report["valid"] = False
            report["errors"].extend(input_errors)

        # Validate outputs if provided
        if outputs is not None:
            output_valid, output_errors = contract.validate_outputs(outputs)
            report["output_validation"] = {
                "valid": output_valid,
                "errors": output_errors
            }

            if not output_valid:
                report["valid"] = False
                report["errors"].extend(output_errors)

        return report["valid"], report

    def validate_flow_dependencies(self, executed_nodes: List[str]) -> tuple[bool, List[str]]:
        """
        Validate that all dependencies were satisfied.

        Args:
            executed_nodes: List of nodes that were executed in order

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        executed_set = set(executed_nodes)

        for node_name in executed_nodes:
            if node_name not in self.contracts:
                errors.append(f"Unknown node executed: {node_name}")
                continue

            contract = self.contracts[node_name]
            for dep in contract.dependencies:
                if dep not in executed_set:
                    errors.append(f"Node {node_name} executed without dependency {dep}")

        return len(errors) == 0, errors

    def validate_canonical_order(self, executed_nodes: List[str]) -> tuple[bool, List[str]]:
        """
        Validate that nodes were executed in canonical order.

        Args:
            executed_nodes: List of nodes in execution order

        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []

        # Check if all canonical nodes were executed
        canonical_set = set(self.execution_order)
        executed_set = set(executed_nodes)

        missing = canonical_set - executed_set
        if missing:
            warnings.append(f"Missing canonical nodes: {missing}")

        extra = executed_set - canonical_set
        if extra:
            warnings.append(f"Extra non-canonical nodes: {extra}")

        # Check order for nodes that are in canonical order
        canonical_indices = {}
        for i, node in enumerate(self.execution_order):
            canonical_indices[node] = i

        last_canonical_index = -1
        for node in executed_nodes:
            if node in canonical_indices:
                current_index = canonical_indices[node]
                if current_index < last_canonical_index:
                    warnings.append(
                        f"Node {node} executed out of canonical order "
                        f"(expected around position {current_index}, but came after {last_canonical_index})"
                    )
                last_canonical_index = max(last_canonical_index, current_index)

        return len(warnings) == 0, warnings

    def get_execution_plan(self) -> List[str]:
        """Get the canonical execution order"""
        return self.execution_order.copy()

    def get_contract(self, node_name: str) -> Optional[NodeContract]:
        """Get contract for a specific node"""
        return self.contracts.get(node_name)

    def generate_flow_report(self, executed_nodes: List[str], node_reports: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive flow validation report.

        Args:
            executed_nodes: Nodes that were executed
            node_reports: Individual validation reports for each node

        Returns:
            Complete flow validation report
        """
        dep_valid, dep_errors = self.validate_flow_dependencies(executed_nodes)
        order_valid, order_warnings = self.validate_canonical_order(executed_nodes)

        all_nodes_valid = all(
            report.get("valid", False)
            for report in node_reports.values()
        )

        return {
            "flow_valid": dep_valid and all_nodes_valid,
            "canonical_order_followed": order_valid,
            "dependencies_satisfied": dep_valid,
            "dependency_errors": dep_errors,
            "order_warnings": order_warnings,
            "executed_nodes": executed_nodes,
            "canonical_order": self.execution_order,
            "node_reports": node_reports,
            "summary": {
                "total_nodes": len(executed_nodes),
                "valid_nodes": sum(1 for r in node_reports.values() if r.get("valid", False)),
                "invalid_nodes": sum(1 for r in node_reports.values() if not r.get("valid", True)),
            }
        }

