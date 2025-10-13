#!/usr/bin/env python3
"""
Connection Stability Analyzer
==============================
Validates inter-node data flows, tracks retry/backoff metrics,
verifies cardinality constraints for all 72 flow contracts.

Integrates with diagnostic instrumentation to provide
per-connection verdicts with detailed contract violation reports.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Valid data types in the pipeline (from data_flow_contract.py)"""

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
    DECATALOGO_EVIDENCIA = "decatalogo_evidencia"
    DECATALOGO_DIMENSION = "decatalogo_dimension"
    DECATALOGO_CLUSTER = "decatalogo_cluster"
    ONTOLOGIA_PATTERNS = "ontologia_patterns"
    ADVANCED_EMBEDDINGS = "advanced_embeddings"
    CAUSAL_COEFFICIENTS = "causal_coefficients"


class VerdictStatus(Enum):
    """Connection stability verdicts"""

    STABLE = "stable"
    UNSTABLE = "unstable"
    SUITABLE = "suitable"
    UNSUITABLE = "unsuitable"


@dataclass
class ConnectionMetrics:
    """Tracks retry rates, backoff patterns, and error rates per connection"""

    connection_id: str
    retry_count: int = 0
    total_attempts: int = 0
    backoff_delays: List[float] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0
    schema_mismatches: int = 0
    type_incompatibilities: int = 0
    cardinality_violations: int = 0
    last_error_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_latency_ms: float = 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        if self.total_attempts == 0:
            return 0.0
        return self.error_count / self.total_attempts

    @property
    def avg_backoff_delay(self) -> float:
        """Calculate average backoff delay"""
        if not self.backoff_delays:
            return 0.0
        return sum(self.backoff_delays) / len(self.backoff_delays)

    @property
    def max_backoff_delay(self) -> float:
        """Get maximum backoff delay"""
        return max(self.backoff_delays) if self.backoff_delays else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count


@dataclass
class SchemaMismatch:
    """Captures concrete examples of schema mismatches"""

    connection_id: str
    timestamp: float
    expected_schema: Dict[str, Any]
    actual_schema: Dict[str, Any]
    missing_fields: List[str] = field(default_factory=list)
    extra_fields: List[str] = field(default_factory=list)
    type_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    example_data: Optional[Dict[str, Any]] = None


@dataclass
class FlowSpecification:
    """Specification for a data flow contract"""

    flow_id: str
    source: str
    target: str
    flow_type: str
    cardinality: str
    input_schema: Dict[str, DataType]
    output_schema: Dict[str, DataType]
    description: str = ""
    performance_budget_ms: Optional[float] = None

    def validate_cardinality(self, actual_count: int) -> Tuple[bool, str]:
        """Validate actual cardinality matches specification"""
        if self.cardinality == "1:1":
            if actual_count != 1:
                return False, f"Expected 1:1 cardinality, got {actual_count}"
        elif self.cardinality == "1:N":
            if actual_count < 1:
                return False, f"Expected 1:N cardinality, got {actual_count}"
        return True, ""


@dataclass
class ConnectionVerdict:
    """Per-connection verdict with detailed analysis"""

    connection_id: str
    is_stable: bool
    is_suitable: bool
    stability_score: float
    suitability_score: float
    metrics: ConnectionMetrics
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    schema_mismatches: List[SchemaMismatch] = field(default_factory=list)

    @property
    def verdict_status(self) -> str:
        """Get combined verdict status"""
        if self.is_stable and self.is_suitable:
            return "STABLE_SUITABLE"
        elif self.is_stable and not self.is_suitable:
            return "STABLE_UNSUITABLE"
        elif not self.is_stable and self.is_suitable:
            return "UNSTABLE_SUITABLE"
        else:
            return "UNSTABLE_UNSUITABLE"


class ConnectionStabilityAnalyzer:
    """
    Validates all inter-node data flows by checking schema/type compatibility,
    measuring retry rates and backoff patterns, and verifying cardinality constraints.
    """

    def __init__(self, flow_doc_path: str = "tools/flow_doc.json"):
        """Initialize analyzer with flow specifications"""
        self.flow_doc_path = Path(flow_doc_path)
        self.flow_specifications: Dict[str, FlowSpecification] = {}
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        self.schema_mismatches: Dict[str, List[SchemaMismatch]] = defaultdict(list)
        self.verdicts: Dict[str, ConnectionVerdict] = {}

        self.error_rate_threshold = 0.1
        self.retry_rate_threshold = 0.3
        self.max_backoff_threshold_ms = 5000

        self._load_flow_specifications()

    def _load_flow_specifications(self):
        """Load flow specifications from tools/flow_doc.json and DEPENDENCY_FLOWS.md"""
        logger.info("Loading flow specifications from %s", self.flow_doc_path)

        if self.flow_doc_path.exists():
            try:
                with open(self.flow_doc_path, "r") as f:
                    flow_doc = json.load(f)
                logger.info("Loaded flow_doc.json: %s", flow_doc)
            except Exception as e:
                logger.warning("Failed to load flow_doc.json: %s", e)

        dependency_flows_path = Path("DEPENDENCY_FLOWS.md")
        if dependency_flows_path.exists():
            self._parse_dependency_flows(dependency_flows_path)

        self._load_contract_specifications()

        logger.info("Loaded %s flow specifications", len(self.flow_specifications))

    def _parse_dependency_flows(self, path: Path):
        """Parse DEPENDENCY_FLOWS.md to extract flow specifications"""
        try:
            content = path.read_text()
            lines = content.split("\n")

            current_flow_id = 0
            for line in lines:
                if line.startswith("### "):
                    parts = line.split(".")
                    if len(parts) >= 2:
                        flow_parts = parts[1].strip().split("→")
                        if len(flow_parts) == 2:
                            source = flow_parts[0].strip()
                            target = flow_parts[1].strip()
                            current_flow_id += 1

                            flow_id = f"flow_{current_flow_id:03d}_{source}_{target}"
                            self.flow_specifications[flow_id] = FlowSpecification(
                                flow_id=flow_id,
                                source=source,
                                target=target,
                                flow_type="data",
                                cardinality="1:N",
                                input_schema={"data": DataType.METADATA},
                                output_schema={"data": DataType.METADATA},
                                description=f"{source} depends on {target}",
                            )

            logger.info("Parsed %s flows from DEPENDENCY_FLOWS.md", current_flow_id)
        except Exception as e:
            logger.error("Failed to parse DEPENDENCY_FLOWS.md: %s", e)

    def _load_contract_specifications(self):
        """Load contract specifications from data_flow_contract module"""
        try:
            from data_flow_contract import CanonicalFlowValidator

            validator = CanonicalFlowValidator()
            contracts = validator.contracts

            for node_name, contract in contracts.items():
                for dep in contract.dependencies:
                    flow_id = f"contract_{dep}_to_{node_name}"

                    input_schema = {dt.value: dt for dt in contract.required_inputs}
                    output_schema = {dt.value: dt for dt in contract.produces}

                    self.flow_specifications[flow_id] = FlowSpecification(
                        flow_id=flow_id,
                        source=dep,
                        target=node_name,
                        flow_type="data",
                        cardinality="1:1",
                        input_schema=input_schema,
                        output_schema=output_schema,
                        description=f"Contract: {dep} → {node_name}",
                        performance_budget_ms=contract.metadata.get(
                            "performance_budget_ms"
                        ),
                    )

            logger.info("Loaded %s contract specifications", len(contracts))
        except Exception as e:
            logger.warning("Failed to load contract specifications: %s", e)

    def get_or_create_metrics(self, connection_id: str) -> ConnectionMetrics:
        """Get or create metrics for a connection"""
        if connection_id not in self.connection_metrics:
            self.connection_metrics[connection_id] = ConnectionMetrics(
                connection_id=connection_id
            )
        return self.connection_metrics[connection_id]

    def track_retry_attempt(self, connection_id: str, backoff_delay: float):
        """Track a retry attempt with backoff delay"""
        metrics = self.get_or_create_metrics(connection_id)
        metrics.retry_count += 1
        metrics.total_attempts += 1
        metrics.backoff_delays.append(backoff_delay)
        logger.debug(
            "Retry tracked for %s: delay=%sms, total_retries=%s",
            connection_id,
            backoff_delay,
            metrics.retry_count,
        )

    def track_attempt(self, connection_id: str, success: bool, latency_ms: float = 0.0):
        """Track a connection attempt"""
        metrics = self.get_or_create_metrics(connection_id)
        metrics.total_attempts += 1

        if success:
            metrics.success_count += 1
            metrics.last_success_time = time.time()
            metrics.total_latency_ms += latency_ms
        else:
            metrics.error_count += 1
            metrics.last_error_time = time.time()

    def validate_interface(
        self, source: str, target: str, data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate schema/type compatibility at interface boundary.

        Returns:
            (is_valid, list_of_errors)
        """
        connection_id = f"{source}_to_{target}"
        errors = []

        flow_spec = None
        for spec in self.flow_specifications.values():
            if spec.source == source and spec.target == target:
                flow_spec = spec
                break

        if not flow_spec:
            logger.warning("No flow specification found for %s", connection_id)
            return True, []

        for field_name, expected_type in flow_spec.output_schema.items():
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")
                self.get_or_create_metrics(connection_id).schema_mismatches += 1
                continue

            actual_value = data[field_name]
            if not self._validate_type(actual_value, expected_type):
                errors.append(
                    f"Type mismatch for {field_name}: expected {expected_type.value}, got {type(actual_value).__name__}"
                )
                self.get_or_create_metrics(connection_id).type_incompatibilities += 1

        if errors:
            self.capture_schema_mismatch(
                connection_id=connection_id,
                expected_schema={
                    k: v.value for k, v in flow_spec.output_schema.items()
                },
                actual_schema={k: type(v).__name__ for k, v in data.items()},
                example_data=data,
            )

        return len(errors) == 0, errors

    @staticmethod
    def _validate_type(value: Any, expected_type: DataType) -> bool:
        """Validate value matches expected data type"""
        type_validators = {
            DataType.RAW_TEXT: lambda v: isinstance(v, str),
            DataType.SANITIZED_TEXT: lambda v: isinstance(v, str),
            DataType.SEGMENTS: lambda v: isinstance(v, list),
            DataType.EMBEDDINGS: lambda v: isinstance(v, (list, dict)),
            DataType.ENTITIES: lambda v: isinstance(v, list),
            DataType.CONTRADICTIONS: lambda v: isinstance(v, list),
            DataType.MONETARY_VALUES: lambda v: isinstance(v, list),
            DataType.FEASIBILITY_SCORES: lambda v: isinstance(v, dict),
            DataType.CAUSAL_PATTERNS: lambda v: isinstance(v, dict),
            DataType.TEORIA_CAMBIO: lambda v: isinstance(v, dict),
            DataType.DAG_STRUCTURE: lambda v: isinstance(v, dict),
            DataType.METADATA: lambda v: isinstance(v, dict),
        }

        validator = type_validators.get(expected_type)
        if validator:
            return validator(value)
        return True

    def verify_cardinality(
        self, connection_id: str, actual_count: int
    ) -> Tuple[bool, str]:
        """
        Verify cardinality constraint matches documented expectation.

        Returns:
            (is_valid, error_message)
        """
        flow_spec = None
        for spec in self.flow_specifications.values():
            if (
                spec.flow_id == connection_id
                or f"{spec.source}_to_{spec.target}" == connection_id
            ):
                flow_spec = spec
                break

        if not flow_spec:
            return True, ""

        is_valid, error_msg = flow_spec.validate_cardinality(actual_count)

        if not is_valid:
            self.get_or_create_metrics(connection_id).cardinality_violations += 1
            logger.warning("Cardinality violation for %s: %s", connection_id, error_msg)

        return is_valid, error_msg

    def capture_schema_mismatch(
        self,
        connection_id: str,
        expected_schema: Dict[str, Any],
        actual_schema: Dict[str, Any],
        example_data: Optional[Dict[str, Any]] = None,
    ):
        """Capture concrete example of schema mismatch"""
        expected_fields = set(expected_schema.keys())
        actual_fields = set(actual_schema.keys())

        missing_fields = list(expected_fields - actual_fields)
        extra_fields = list(actual_fields - expected_fields)

        type_mismatches = []
        for field in expected_fields & actual_fields:
            if expected_schema[field] != actual_schema[field]:
                type_mismatches.append(
                    {
                        "field": field,
                        "expected": expected_schema[field],
                        "actual": actual_schema[field],
                    }
                )

        mismatch = SchemaMismatch(
            connection_id=connection_id,
            timestamp=time.time(),
            expected_schema=expected_schema,
            actual_schema=actual_schema,
            missing_fields=missing_fields,
            extra_fields=extra_fields,
            type_mismatches=type_mismatches,
            example_data=example_data,
        )

        self.schema_mismatches[connection_id].append(mismatch)

        metrics = self.get_or_create_metrics(connection_id)
        metrics.schema_mismatches += 1

        logger.error(
            "Schema mismatch captured for %s: %s missing, %s type errors",
            connection_id,
            len(missing_fields),
            len(type_mismatches),
        )

    def analyze_connection_stability(self, connection_id: str) -> Dict[str, Any]:
        """
        Analyze connection stability metrics.

        Returns:
            Dictionary with stability analysis
        """
        metrics = self.get_or_create_metrics(connection_id)

        analysis = {
            "connection_id": connection_id,
            "total_attempts": metrics.total_attempts,
            "success_count": metrics.success_count,
            "error_count": metrics.error_count,
            "retry_count": metrics.retry_count,
            "error_rate": metrics.error_rate,
            "retry_rate": metrics.retry_count / metrics.total_attempts
            if metrics.total_attempts > 0
            else 0.0,
            "avg_backoff_delay_ms": metrics.avg_backoff_delay,
            "max_backoff_delay_ms": metrics.max_backoff_delay,
            "avg_latency_ms": metrics.avg_latency_ms,
            "schema_mismatches": metrics.schema_mismatches,
            "type_incompatibilities": metrics.type_incompatibilities,
            "cardinality_violations": metrics.cardinality_violations,
            "is_stable": self._is_connection_stable(metrics),
            "stability_issues": self._identify_stability_issues(metrics),
        }

        return analysis

    def _is_connection_stable(self, metrics: ConnectionMetrics) -> bool:
        """Determine if connection is stable based on metrics"""
        if metrics.total_attempts == 0:
            return True

        if metrics.error_rate > self.error_rate_threshold:
            return False

        retry_rate = metrics.retry_count / metrics.total_attempts
        if retry_rate > self.retry_rate_threshold:
            return False

        if metrics.max_backoff_delay > self.max_backoff_threshold_ms:
            return False

        return True

    def _identify_stability_issues(self, metrics: ConnectionMetrics) -> List[str]:
        """Identify specific stability issues"""
        issues = []

        if metrics.error_rate > self.error_rate_threshold:
            issues.append(
                f"High error rate: {metrics.error_rate:.2%} > {self.error_rate_threshold:.2%}"
            )

        retry_rate = (
            metrics.retry_count / metrics.total_attempts
            if metrics.total_attempts > 0
            else 0.0
        )
        if retry_rate > self.retry_rate_threshold:
            issues.append(
                f"High retry rate: {retry_rate:.2%} > {self.retry_rate_threshold:.2%}"
            )

        if metrics.max_backoff_delay > self.max_backoff_threshold_ms:
            issues.append(
                f"Excessive backoff delay: {metrics.max_backoff_delay:.0f}ms > {self.max_backoff_threshold_ms:.0f}ms"
            )

        if metrics.schema_mismatches > 0:
            issues.append(f"Schema mismatches: {metrics.schema_mismatches}")

        if metrics.type_incompatibilities > 0:
            issues.append(f"Type incompatibilities: {metrics.type_incompatibilities}")

        if metrics.cardinality_violations > 0:
            issues.append(f"Cardinality violations: {metrics.cardinality_violations}")

        return issues

    def generate_verdict(self, connection_id: str) -> ConnectionVerdict:
        """
        Generate per-connection verdict with detailed analysis.

        Returns:
            ConnectionVerdict with stability and suitability assessment
        """
        metrics = self.get_or_create_metrics(connection_id)
        analysis = self.analyze_connection_stability(connection_id)

        is_stable = analysis["is_stable"]
        stability_issues = analysis["stability_issues"]

        stability_score = 1.0
        if metrics.total_attempts > 0:
            stability_score -= metrics.error_rate * 0.5
            retry_rate = metrics.retry_count / metrics.total_attempts
            stability_score -= retry_rate * 0.3
            if metrics.schema_mismatches > 0:
                stability_score -= 0.2
        stability_score = max(0.0, min(1.0, stability_score))

        suitability_score = 1.0
        if metrics.schema_mismatches > 0:
            suitability_score -= 0.4
        if metrics.type_incompatibilities > 0:
            suitability_score -= 0.3
        if metrics.cardinality_violations > 0:
            suitability_score -= 0.3
        suitability_score = max(0.0, min(1.0, suitability_score))

        is_suitable = suitability_score >= 0.7

        violations = []
        if metrics.schema_mismatches > 0:
            violations.append(f"{metrics.schema_mismatches} schema mismatches detected")
        if metrics.type_incompatibilities > 0:
            violations.append(
                f"{metrics.type_incompatibilities} type incompatibilities detected"
            )
        if metrics.cardinality_violations > 0:
            violations.append(
                f"{metrics.cardinality_violations} cardinality violations detected"
            )
        violations.extend(stability_issues)

        recommendations = []
        if metrics.error_rate > self.error_rate_threshold:
            recommendations.append(
                "Implement circuit breaker pattern for fault tolerance"
            )
        if metrics.retry_count > metrics.total_attempts * 0.2:
            recommendations.append("Review retry strategy and backoff configuration")
        if metrics.schema_mismatches > 0:
            recommendations.append("Validate data contracts at compile time")
        if metrics.avg_backoff_delay > 1000:
            recommendations.append("Optimize backoff delays to reduce latency")

        verdict = ConnectionVerdict(
            connection_id=connection_id,
            is_stable=is_stable,
            is_suitable=is_suitable,
            stability_score=stability_score,
            suitability_score=suitability_score,
            metrics=metrics,
            violations=violations,
            recommendations=recommendations,
            schema_mismatches=self.schema_mismatches.get(connection_id, []),
        )

        self.verdicts[connection_id] = verdict
        return verdict

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report with all contract violations.

        Returns:
            Detailed report dictionary
        """
        for connection_id in self.connection_metrics:
            if connection_id not in self.verdicts:
                self.generate_verdict(connection_id)

        total_connections = len(self.connection_metrics)
        stable_connections = sum(1 for v in self.verdicts.values() if v.is_stable)
        suitable_connections = sum(1 for v in self.verdicts.values() if v.is_suitable)

        total_violations = sum(len(v.violations) for v in self.verdicts.values())
        total_schema_mismatches = sum(
            len(mismatches) for mismatches in self.schema_mismatches.values()
        )

        stable_suitable = []
        stable_unsuitable = []
        unstable_suitable = []
        unstable_unsuitable = []

        for connection_id, verdict in self.verdicts.items():
            if verdict.is_stable and verdict.is_suitable:
                stable_suitable.append(connection_id)
            elif verdict.is_stable and not verdict.is_suitable:
                stable_unsuitable.append(connection_id)
            elif not verdict.is_stable and verdict.is_suitable:
                unstable_suitable.append(connection_id)
            else:
                unstable_unsuitable.append(connection_id)

        report = {
            "summary": {
                "total_connections": total_connections,
                "total_flows_specified": len(self.flow_specifications),
                "stable_connections": stable_connections,
                "suitable_connections": suitable_connections,
                "total_violations": total_violations,
                "total_schema_mismatches": total_schema_mismatches,
                "stability_rate": stable_connections / total_connections
                if total_connections > 0
                else 0.0,
                "suitability_rate": suitable_connections / total_connections
                if total_connections > 0
                else 0.0,
            },
            "connection_categories": {
                "stable_suitable": stable_suitable,
                "stable_unsuitable": stable_unsuitable,
                "unstable_suitable": unstable_suitable,
                "unstable_unsuitable": unstable_unsuitable,
            },
            "verdicts": {
                connection_id: {
                    "status": verdict.verdict_status,
                    "stability_score": verdict.stability_score,
                    "suitability_score": verdict.suitability_score,
                    "violations": verdict.violations,
                    "recommendations": verdict.recommendations,
                    "metrics": {
                        "total_attempts": verdict.metrics.total_attempts,
                        "error_rate": verdict.metrics.error_rate,
                        "retry_count": verdict.metrics.retry_count,
                        "schema_mismatches": verdict.metrics.schema_mismatches,
                        "type_incompatibilities": verdict.metrics.type_incompatibilities,
                        "cardinality_violations": verdict.metrics.cardinality_violations,
                    },
                }
                for connection_id, verdict in self.verdicts.items()
            },
            "schema_mismatches": {
                connection_id: [
                    {
                        "timestamp": mismatch.timestamp,
                        "missing_fields": mismatch.missing_fields,
                        "extra_fields": mismatch.extra_fields,
                        "type_mismatches": mismatch.type_mismatches,
                    }
                    for mismatch in mismatches
                ]
                for connection_id, mismatches in self.schema_mismatches.items()
            },
            "flow_specifications": {
                flow_id: {
                    "source": spec.source,
                    "target": spec.target,
                    "cardinality": spec.cardinality,
                    "flow_type": spec.flow_type,
                    "description": spec.description,
                }
                for flow_id, spec in self.flow_specifications.items()
            },
            "timestamp": time.time(),
        }

        return report

    def export_report(self, output_path: str = "connection_stability_report.json"):
        """Export report to JSON file"""
        report = self.generate_report()
        output_file = Path(output_path)

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Connection stability report exported to %s", output_file)
        return report


def create_connection_stability_analyzer(
    flow_doc_path: str = "tools/flow_doc.json",
) -> ConnectionStabilityAnalyzer:
    """Factory function to create connection stability analyzer"""
    return ConnectionStabilityAnalyzer(flow_doc_path=flow_doc_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = create_connection_stability_analyzer()
    stability_report = analyzer.generate_report()
    print(json.dumps(stability_report, indent=2))
