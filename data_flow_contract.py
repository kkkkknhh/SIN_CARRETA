#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Flow Contract Validator
=============================

Defines and enforces contracts for each node in the canonical MINIMINIMOON pipeline.
Ensures that each component receives valid inputs and produces valid outputs,
maintaining data integrity throughout the flow.

Performance Optimizations:
- Memoization cache for validation results with input hash-based keys
- LRU eviction policy with configurable size limits
- Cache invalidation based on contract version changes
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
    # **NUEVOS TIPOS PARA DECATALOGO_PRINCIPAL.PY**
    DECATALOGO_EVIDENCIA = "decatalogo_evidencia"
    DECATALOGO_DIMENSION = "decatalogo_dimension"
    DECATALOGO_CLUSTER = "decatalogo_cluster"
    ONTOLOGIA_PATTERNS = "ontologia_patterns"
    ADVANCED_EMBEDDINGS = "advanced_embeddings"
    CAUSAL_COEFFICIENTS = "causal_coefficients"


class ValidationCache:
    """
    LRU cache for validation results with hash-based memoization.

    Reduces validation overhead by caching results based on input hashes.
    Implements size-based eviction to prevent memory growth.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._version = "1.0"

    def _compute_hash(self, data: Dict[str, Any], node_name: str) -> str:
        """Compute stable hash for input data"""
        try:
            # Sort keys for deterministic hashing
            stable_repr = json.dumps(
                {k: self._hashable_value(v) for k, v in sorted(data.items())},
                sort_keys=True,
                default=str,
            )
            hash_input = f"{node_name}:{self._version}:{stable_repr}"
            return hashlib.sha256(hash_input.encode()).hexdigest()
        except Exception:
            return None

    @staticmethod
    def _hashable_value(value: Any) -> Any:
        """Convert value to hashable representation"""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return str(value[:100])  # Limit list size for hashing
        elif isinstance(value, dict):
            return str({k: str(v)[:100] for k, v in list(value.items())[:10]})
        else:
            return str(type(value))

    def get(
        self, data: Dict[str, Any], node_name: str
    ) -> Optional[Tuple[bool, Dict[str, Any]]]:
        """Get cached validation result"""
        cache_key = self._compute_hash(data, node_name)
        if cache_key is None:
            return None

        if cache_key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            cached = self.cache[cache_key]
            return cached["result"], cached["report"]

        self.misses += 1
        return None

    def put(
        self, data: Dict[str, Any], node_name: str, result: bool, report: Dict[str, Any]
    ):
        """Store validation result in cache"""
        cache_key = self._compute_hash(data, node_name)
        if cache_key is None:
            return

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            self.cache.popitem(last=False)

        self.cache[cache_key] = {
            "result": result,
            "report": report,
            "timestamp": time.time(),
        }

    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


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
    Validates the canonical flow of the MINIMINIMOON pipeline.

    Ensures all data contracts are met at each node transition.
    **EXTENDED WITH DECATALOGO_PRINCIPAL.PY VALIDATION**
    """

    def __init__(self):
        self.contracts = self._define_contracts()
        self.validation_cache = ValidationCache()
        self._version = "2.0-decatalogo-integrated"

    @staticmethod
    def _define_contracts() -> Dict[str, NodeContract]:
        """
        Define contracts for each node in the canonical flow.

        **INCLUDES DECATALOGO_PRINCIPAL.PY NODE CONTRACT**
        """
        contracts = {}

        # 1. Sanitization
        contracts["sanitization"] = NodeContract(
            node_name="sanitization",
            dependencies=[],
            required_inputs=[DataType.RAW_TEXT],
            produces=[DataType.SANITIZED_TEXT],
            validators={
                "non_empty": lambda d: (
                    len(d.get("raw_text", "")) > 0,
                    "Input text cannot be empty",
                )
            },
        )

        # 2. Plan Processing
        contracts["plan_processing"] = NodeContract(
            node_name="plan_processing",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.METADATA],
            validators={
                "text_length": lambda d: (
                    len(d.get("sanitized_text", "")) >= 100,
                    "Sanitized text too short for processing",
                )
            },
        )

        # 3. Document Segmentation
        contracts["document_segmentation"] = NodeContract(
            node_name="document_segmentation",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.SEGMENTS],
            validators={
                "valid_text": lambda d: (
                    isinstance(d.get("sanitized_text"), str),
                    "Sanitized text must be string",
                )
            },
        )

        # 4. Embedding
        contracts["embedding"] = NodeContract(
            node_name="embedding",
            dependencies=["document_segmentation"],
            required_inputs=[DataType.SEGMENTS],
            produces=[DataType.EMBEDDINGS],
            validators={
                "segments_list": lambda d: (
                    isinstance(d.get("segments"), list)
                    and len(d.get("segments", [])) > 0,
                    "Segments must be non-empty list",
                )
            },
        )

        # 5. Responsibility Detection
        contracts["responsibility_detection"] = NodeContract(
            node_name="responsibility_detection",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.ENTITIES],
            validators={
                "text_present": lambda d: (
                    len(d.get("sanitized_text", "")) > 50,
                    "Text too short for entity detection",
                )
            },
        )

        # 6. Contradiction Detection
        contracts["contradiction_detection"] = NodeContract(
            node_name="contradiction_detection",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.CONTRADICTIONS],
        )

        # 7. Monetary Detection
        contracts["monetary_detection"] = NodeContract(
            node_name="monetary_detection",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.MONETARY_VALUES],
        )

        # 8. Feasibility Scoring
        contracts["feasibility_scoring"] = NodeContract(
            node_name="feasibility_scoring",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.FEASIBILITY_SCORES],
        )

        # 9. Causal Pattern Detection
        contracts["causal_detection"] = NodeContract(
            node_name="causal_detection",
            dependencies=["sanitization"],
            required_inputs=[DataType.SANITIZED_TEXT],
            produces=[DataType.CAUSAL_PATTERNS],
        )

        # 10. Theory of Change
        contracts["teoria_cambio"] = NodeContract(
            node_name="teoria_cambio",
            dependencies=[
                "responsibility_detection",
                "causal_detection",
                "monetary_detection",
            ],
            required_inputs=[
                DataType.SANITIZED_TEXT,
                DataType.ENTITIES,
                DataType.CAUSAL_PATTERNS,
                DataType.MONETARY_VALUES,
            ],
            produces=[DataType.TEORIA_CAMBIO],
        )

        # 11. DAG Validation
        contracts["dag_validation"] = NodeContract(
            node_name="dag_validation",
            dependencies=["teoria_cambio"],
            required_inputs=[DataType.TEORIA_CAMBIO],
            produces=[DataType.DAG_STRUCTURE],
        )

        # **CONTRATO CRÍTICO: DECATALOGO_EVALUATION**
        contracts["decatalogo_evaluation"] = NodeContract(
            node_name="decatalogo_evaluation",
            required_inputs={
                "plan_text": DataType.SANITIZED_TEXT,
                "segments": DataType.SEGMENTS,
                "responsibilities": DataType.ENTITIES,
                "monetary": DataType.MONETARY_VALUES,
                "feasibility": DataType.FEASIBILITY_SCORES,
                "teoria_cambio": DataType.TEORIA_CAMBIO,
                "causal_patterns": DataType.CAUSAL_PATTERNS,
            },
            required_outputs={
                "evaluacion_por_dimension": DataType.DECATALOGO_DIMENSION,
                "evidencias_globales": DataType.DECATALOGO_EVIDENCIA,
                "metricas_globales": DataType.METADATA,
                "analisis_clusters": DataType.DECATALOGO_CLUSTER,
                "interdependencias_globales": DataType.DAG_STRUCTURE,
            },
            validation_rules=[
                lambda data: "plan_text" in data and len(data["plan_text"]) > 100,
                lambda data: "segments" in data and isinstance(data["segments"], list),
                lambda data: "responsibilities" in data
                and isinstance(data["responsibilities"], list),
                lambda data: "monetary" in data and isinstance(data["monetary"], list),
                lambda data: "feasibility" in data
                and isinstance(data["feasibility"], dict),
                lambda data: "teoria_cambio" in data
                and isinstance(data["teoria_cambio"], dict),
                lambda data: "causal_patterns" in data
                and isinstance(data["causal_patterns"], dict),
            ],
            output_validation_rules=[
                lambda data: "evaluacion_por_dimension" in data
                and isinstance(data["evaluacion_por_dimension"], dict),
                lambda data: "metricas_globales" in data
                and "coherencia_promedio" in data["metricas_globales"],
                lambda data: "metricas_globales" in data
                and "kpi_promedio" in data["metricas_globales"],
                lambda data: "metricas_globales" in data
                and "evidencias_totales" in data["metricas_globales"],
                lambda data: "analisis_clusters" in data
                and isinstance(data["analisis_clusters"], dict),
                lambda data: "cobertura_cuestionario_industrial" in data,
                lambda data: data.get("metricas_globales", {}).get(
                    "dimensiones_evaluadas", 0
                )
                > 0,
            ],
            dependencies=[
                "sanitization",
                "segmentation",
                "responsibility_detection",
                "monetary_detection",
                "feasibility_scoring",
                "teoria_cambio",
                "causal_patterns",
            ],
            performance_budget_ms=30000,  # 30 segundos para evaluación completa
            description="Evaluación avanzada con ExtractorEvidenciaIndustrialAvanzado para 300 preguntas del decálogo industrial",
        )

        # **CONTRATO: DECATALOGO_EXTRACTOR (componente interno)**
        contracts["decatalogo_extractor_init"] = NodeContract(
            node_name="decatalogo_extractor_init",
            required_inputs={
                "documentos": DataType.SEGMENTS,
                "nombre_plan": DataType.METADATA,
            },
            required_outputs={
                "embeddings_doc": DataType.ADVANCED_EMBEDDINGS,
                "embeddings_metadata": DataType.METADATA,
                "estructura_documental": DataType.METADATA,
                "ontologia": DataType.ONTOLOGIA_PATTERNS,
            },
            validation_rules=[
                lambda data: "documentos" in data
                and isinstance(data["documentos"], list),
                lambda data: len(data.get("documentos", [])) > 0,
                lambda data: all(
                    isinstance(doc, tuple) and len(doc) == 2
                    for doc in data.get("documentos", [])
                ),
            ],
            output_validation_rules=[
                lambda data: "embeddings_doc" in data,
                lambda data: "embeddings_metadata" in data
                and isinstance(data["embeddings_metadata"], list),
                lambda data: "estructura_documental" in data
                and isinstance(data["estructura_documental"], dict),
            ],
            dependencies=["segmentation"],
            performance_budget_ms=10000,
            description="Inicialización del ExtractorEvidenciaIndustrialAvanzado con precomputación de embeddings y análisis estructural",
        )

        # **CONTRATO: BUSQUEDA DE EVIDENCIA AVANZADA**
        contracts["decatalogo_evidencia_busqueda"] = NodeContract(
            node_name="decatalogo_evidencia_busqueda",
            required_inputs={
                "query": DataType.RAW_TEXT,
                "conceptos_clave": DataType.METADATA,
                "embeddings_doc": DataType.ADVANCED_EMBEDDINGS,
            },
            required_outputs={
                "evidencias": DataType.DECATALOGO_EVIDENCIA,
                "scores": DataType.METADATA,
            },
            validation_rules=[
                lambda data: "query" in data
                and isinstance(data["query"], str)
                and len(data["query"]) > 0,
                lambda data: "conceptos_clave" in data
                and isinstance(data["conceptos_clave"], list),
                lambda data: "embeddings_doc" in data,
            ],
            output_validation_rules=[
                lambda data: "evidencias" in data
                and isinstance(data["evidencias"], list),
                lambda data: all(
                    "score_final" in e for e in data.get("evidencias", [])
                ),
                lambda data: all(
                    "similitud_semantica" in e for e in data.get("evidencias", [])
                ),
                lambda data: all(
                    "densidad_causal_agregada" in e for e in data.get("evidencias", [])
                ),
                lambda data: all(
                    0 <= e.get("score_final", 0) <= 1
                    for e in data.get("evidencias", [])
                ),
            ],
            dependencies=["decatalogo_extractor_init"],
            performance_budget_ms=5000,
            description="Búsqueda avanzada de evidencia causal con scoring multi-criterio",
        )

        return contracts

    @staticmethod
    def validate_decatalogo_integration(
        orchestrator_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validación especializada para la integración de Decatalogo_principal.py

        Args:
            orchestrator_results: Resultados completos del orquestador

        Returns:
            Reporte de validación con estado y detalles
        """
        validation_report = {
            "status": "passed",
            "timestamp": time.time(),
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        try:
            decatalogo_results = orchestrator_results.get("decatalogo_evaluation", {})

            # Check 1: Estructura básica
            if not isinstance(decatalogo_results, dict):
                validation_report["errors"].append(
                    "decatalogo_evaluation debe ser un diccionario"
                )
                validation_report["status"] = "failed"
                return validation_report

            validation_report["checks"]["estructura_basica"] = "passed"

            # Check 2: Metadatos
            metadata = decatalogo_results.get("metadata", {})
            required_metadata = ["plan_evaluado", "fecha_evaluacion", "version_sistema"]
            for metadata_field in required_metadata:
                if metadata_field not in metadata:
                    validation_report["errors"].append(
                        f"Metadata faltante: {metadata_field}"
                    )
                    validation_report["status"] = "failed"

            validation_report["checks"]["metadata"] = (
                "passed" if not validation_report["errors"] else "failed"
            )

            # Check 3: Métricas globales
            metricas = decatalogo_results.get("metricas_globales", {})
            required_metrics = [
                "coherencia_promedio",
                "kpi_promedio",
                "evidencias_totales",
                "dimensiones_evaluadas",
                "cobertura_preguntas",
            ]
            for metric in required_metrics:
                if metric not in metricas:
                    validation_report["errors"].append(f"Métrica faltante: {metric}")
                    validation_report["status"] = "failed"

            # Validar rangos de métricas
            if "coherencia_promedio" in metricas:
                if not (0 <= metricas["coherencia_promedio"] <= 1):
                    validation_report["warnings"].append(
                        "coherencia_promedio fuera de rango [0,1]"
                    )

            if "kpi_promedio" in metricas:
                if not (0 <= metricas["kpi_promedio"] <= 1):
                    validation_report["warnings"].append(
                        "kpi_promedio fuera de rango [0,1]"
                    )

            validation_report["checks"]["metricas_globales"] = (
                "passed" if not validation_report["errors"] else "failed"
            )

            # Check 4: Evaluación por dimensión
            evaluacion_dims = decatalogo_results.get("evaluacion_por_dimension", {})
            if not isinstance(evaluacion_dims, dict):
                validation_report["errors"].append(
                    "evaluacion_por_dimension debe ser un diccionario"
                )
                validation_report["status"] = "failed"
            else:
                # Validar estructura de cada dimensión
                for dim_nombre, dim_data in evaluacion_dims.items():
                    required_fields = [
                        "dimension_id",
                        "coherencia",
                        "kpis",
                        "evidencias_encontradas",
                    ]
                    for req_field in required_fields:
                        if req_field not in dim_data:
                            validation_report["warnings"].append(
                                f"Dimensión '{dim_nombre}' faltante campo: {req_field}"
                            )

            validation_report["checks"]["evaluacion_dimensiones"] = "passed"

            # Check 5: Clusters
            clusters = decatalogo_results.get("analisis_clusters", {})
            if not isinstance(clusters, dict):
                validation_report["warnings"].append(
                    "analisis_clusters debe ser un diccionario"
                )
            else:
                validation_report["checks"]["analisis_clusters"] = "passed"

            # Check 6: Cobertura de preguntas
            cobertura = decatalogo_results.get("cobertura_cuestionario_industrial", {})
            if "porcentaje_cobertura" in cobertura:
                porcentaje = cobertura["porcentaje_cobertura"]
                if porcentaje < 30:
                    validation_report["warnings"].append(
                        f"Cobertura baja del cuestionario: {porcentaje:.1f}%"
                    )
                validation_report["checks"]["cobertura_preguntas"] = "passed"
            else:
                validation_report["errors"].append("Falta porcentaje_cobertura")
                validation_report["status"] = "failed"

            # Check 7: Integración de componentes
            integracion = decatalogo_results.get("integracion_componentes", {})
            if integracion:
                validation_report["checks"]["integracion_componentes"] = "passed"
            else:
                validation_report["warnings"].append("Falta integracion_componentes")

            # Summary
            validation_report["summary"] = {
                "total_checks": len(validation_report["checks"]),
                "passed_checks": sum(
                    1 for v in validation_report["checks"].values() if v == "passed"
                ),
                "total_errors": len(validation_report["errors"]),
                "total_warnings": len(validation_report["warnings"]),
            }

        except Exception as e:
            validation_report["status"] = "error"
            validation_report["errors"].append(f"Error durante validación: {str(e)}")

        return validation_report

    def validate_node_execution(
        self,
        node_name: str,
        available_data: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Validate a node's execution (inputs and optionally outputs).

        Performance-optimized with memoization cache to reduce validation overhead.

        Args:
            node_name: Name of the node
            available_data: Data available before node execution
            outputs: Data produced by node (optional, for post-validation)
            use_cache: Use cached results if available

        Returns:
            (is_valid, report_dict)
        """
        # Try cache first if enabled
        if use_cache and self.validation_cache is not None and outputs is None:
            cached = self.validation_cache.get(available_data, node_name)
            if cached is not None:
                is_valid, report = cached
                report["cached"] = True
                return is_valid, report

        if node_name not in self.contracts:
            return False, {
                "valid": False,
                "node": node_name,
                "error": f"Unknown node: {node_name}",
                "cached": False,
            }

        contract = self.contracts[node_name]
        report = {
            "node": node_name,
            "valid": True,
            "input_validation": {},
            "output_validation": {},
            "errors": [],
            "cached": False,
        }

        # Validate inputs
        input_valid, input_errors = contract.validate_inputs(available_data)
        report["input_validation"] = {"valid": input_valid, "errors": input_errors}

        if not input_valid:
            report["valid"] = False
            report["errors"].extend(input_errors)

        # Validate outputs if provided
        if outputs is not None:
            output_valid, output_errors = contract.validate_outputs(outputs)
            report["output_validation"] = {
                "valid": output_valid,
                "errors": output_errors,
            }

            if not output_valid:
                report["valid"] = False
                report["errors"].extend(output_errors)

        # Cache result if only input validation (outputs=None)
        if use_cache and self.validation_cache is not None and outputs is None:
            self.validation_cache.put(
                available_data, node_name, report["valid"], report
            )

        return report["valid"], report

    def validate_flow_dependencies(
        self, executed_nodes: List[str]
    ) -> tuple[bool, List[str]]:
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

    def validate_canonical_order(
        self, executed_nodes: List[str]
    ) -> tuple[bool, List[str]]:
        """
        Validate that nodes were executed in canonical order.

        Args:
            executed_nodes: List of nodes in execution order

        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []

        # Check if all canonical nodes were executed
        canonical_set = set(self.get_execution_plan())
        executed_set = set(executed_nodes)

        missing = canonical_set - executed_set
        if missing:
            warnings.append(f"Missing canonical nodes: {missing}")

        extra = executed_set - canonical_set
        if extra:
            warnings.append(f"Extra non-canonical nodes: {extra}")

        # Check order for nodes that are in canonical order
        canonical_indices = {}
        for i, node in enumerate(self.get_execution_plan()):
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

    def clear_cache(self):
        """Clear validation cache"""
        if self.validation_cache is not None:
            self.validation_cache.clear()
            logger.info("Validation cache cleared")

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache performance statistics"""
        if self.validation_cache is not None:
            return self.validation_cache.get_stats()
        return None

    def generate_flow_report(
        self, executed_nodes: List[str], node_reports: Dict[str, Dict]
    ) -> Dict[str, Any]:
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
            report.get("valid", False) for report in node_reports.values()
        )

        report = {
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
                "valid_nodes": sum(
                    1 for r in node_reports.values() if r.get("valid", False)
                ),
                "invalid_nodes": sum(
                    1 for r in node_reports.values() if not r.get("valid", True)
                ),
            },
        }

        # Add cache stats if available
        cache_stats = self.get_cache_stats()
        if cache_stats is not None:
            report["cache_stats"] = cache_stats

        return report
