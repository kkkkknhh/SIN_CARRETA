# Module Documentation
Generated: 2025-10-05T10:52:36.043923

## Core Modules

### feasibility_scorer
Feasibility Scoring Module

Evaluates the presence and quality of baselines, targets, and timeframes in plans,
which are essential for answering key DECALOGO questions:
- DE-1 Q3: "Do outcomes have ba...

- **Path**: `feasibility_scorer.py`
- **Size**: 2021 lines
- **Imported by**: 13 modules
- **Classes**: ComponentType, DetectionResult, IndicatorScore, SafeWriteResult, FeasibilityScorer
- **Functions**: 40 functions

### embedding_model
Industrial-Grade Embedding Model Framework
==========================================
Advanced semantic embedding system with enterprise-level features:
- Multi-modal embedding pipeline with dynamic m...

- **Path**: `embedding_model.py`
- **Size**: 2169 lines
- **Imported by**: 7 modules
- **Classes**: ProductionLogger, EmbeddingModelError, ModelInitializationError, EmbeddingComputationError, ModelConfiguration
- **Functions**: 51 functions

### log_config
Centralised logging configuration for the MINIMINIMOON project.

- **Path**: `log_config.py`
- **Size**: 109 lines
- **Imported by**: 7 modules
- **Classes**: JsonFormatter
- **Functions**: 3 functions

### monetary_detector
Monetary Detection Module

Extracts financial commitments and monetary values from plan documents,
essential for budget planning evaluation (DE-2) and resource adequacy
assessment (DE-4).

Features:
-...

- **Path**: `monetary_detector.py`
- **Size**: 815 lines
- **Imported by**: 5 modules
- **Classes**: MonetaryCategory, FinancialTimeframe, MonetaryMatch, MonetaryAnalysis, MonetaryDetector
- **Functions**: 17 functions

### causal_pattern_detector
- **Path**: `causal_pattern_detector.py`
- **Size**: 372 lines
- **Imported by**: 5 modules
- **Classes**: PDETCausalPatternDetector
- **Functions**: 19 functions

### dag_validation
Deterministic Monte Carlo Sampling for Advanced DAG Validation
==============================================================

Enhanced version with sophisticated statistical testing, multiple validat...

- **Path**: `dag_validation.py`
- **Size**: 1695 lines
- **Imported by**: 5 modules
- **Classes**: GraphType, StatisticalTest, AdvancedGraphNode, MonteCarloAdvancedResult, HypothesisTestResult
- **Functions**: 56 functions

### teoria_cambio
Teoria de Cambio (Theory of Change) Module

This module implements causal graph construction and validation for development plans,
enabling the evaluation of logical intervention frameworks and causal...

- **Path**: `teoria_cambio.py`
- **Size**: 942 lines
- **Imported by**: 5 modules
- **Classes**: CausalElementType, LogicModelQuality, CausalElement, LogicModelValidationResult, TeoriaCambio
- **Functions**: 20 functions

### questionnaire_engine
AUTHORITATIVE QUESTIONNAIRE ENGINE v2.0 - COMPLETE IMPLEMENTATION

- **Path**: `questionnaire_engine.py`
- **Size**: 1940 lines
- **Imported by**: 4 modules
- **Classes**: ScoringModality, ScoreBand, QuestionnaireStructure, ThematicPoint, SearchPattern
- **Functions**: 20 functions

### text_processor
Text Processing Module

Provides robust Unicode normalization and text processing utilities
for consistent handling of Spanish text in development plans.

Features:
- Unicode normalization (NFC, NFKC)...

- **Path**: `text_processor.py`
- **Size**: 262 lines
- **Imported by**: 4 modules
- **Classes**: TextProcessor
- **Functions**: 6 functions

### plan_processor
Plan Processor Module

Processes plan documents to extract structured information aligned with
the DECALOGO evaluation framework, ensuring all necessary evidence for
answering evaluation questions is ...

- **Path**: `plan_processor.py`
- **Size**: 789 lines
- **Imported by**: 4 modules
- **Classes**: PlanProcessor
- **Functions**: 22 functions

### responsibility_detector
Responsibility Detection Module

Detects and classifies entities responsible for plan implementation,
with specific focus on answering DE-1 Q2: "Are institutional responsibilities clearly defined?"

F...

- **Path**: `responsibility_detector.py`
- **Size**: 542 lines
- **Imported by**: 4 modules
- **Classes**: EntityType, ResponsibilityEntity, ResponsibilityDetector
- **Functions**: 10 functions

### evidence_registry
Evidence Registry - Central Immutable Evidence Store
====================================================

Provides a canonical, immutable, deterministic registry for all evidence
produced by MINIMINI...

- **Path**: `evidence_registry.py`
- **Size**: 316 lines
- **Imported by**: 3 modules
- **Classes**: CanonicalEvidence, EvidenceRegistry
- **Functions**: 16 functions

### device_config
PyTorch Device Configuration Module
==================================
Centralized device management for PyTorch operations across the application.

- **Path**: `device_config.py`
- **Size**: 224 lines
- **Imported by**: 3 modules
- **Classes**: DeviceConfig
- **Functions**: 14 functions

### document_segmenter
Document Segmenter Module

Segments plan documents into logical units (objectives, strategies, etc.)
to enable precise analysis and alignment with DECALOGO questions.

Features:
- Multiple segmentatio...

- **Path**: `document_segmenter.py`
- **Size**: 1102 lines
- **Imported by**: 3 modules
- **Classes**: SegmentationType, SectionType, DocumentSegment, DocumentSegmenter
- **Functions**: 25 functions

### contradiction_detector
Contradiction Detection Module

Detecta contradicciones en texto en español buscando conectores adversativos
cercanos a indicadores de metas, verbos de acción y objetivos cuantitativos.

Esta versión ...

- **Path**: `contradiction_detector.py`
- **Size**: 390 lines
- **Imported by**: 3 modules
- **Classes**: RiskLevel, ContradictionMatch, ContradictionAnalysis, ContradictionDetector
- **Functions**: 9 functions

### plan_sanitizer
Plan Sanitizer Module

Provides robust text cleaning and normalization for plan documents,
ensuring that all key elements needed for DECALOGO evaluation are preserved.

Features:
- Unicode normalizati...

- **Path**: `plan_sanitizer.py`
- **Size**: 357 lines
- **Imported by**: 3 modules
- **Classes**: PlanSanitizer
- **Functions**: 10 functions

### data_flow_contract
Data Flow Contract Validator
=============================

Defines and enforces contracts for each node in the canonical MINIMINIMOON pipeline.
Ensures that each component receives valid inputs and p...

- **Path**: `data_flow_contract.py`
- **Size**: 418 lines
- **Imported by**: 2 modules
- **Classes**: DataType, NodeContract, CanonicalFlowValidator
- **Functions**: 10 functions

### miniminimoon_orchestrator
MINIMINIMOON Orchestrator
=========================

Central orchestrator that coordinates all components in the canonical flow of the MINIMINIMOON system.
This module manages the execution sequence, ...

- **Path**: `miniminimoon_orchestrator.py`
- **Size**: 725 lines
- **Imported by**: 2 modules
- **Classes**: ExecutionContext, MINIMINIMOONOrchestrator
- **Functions**: 26 functions

### circuit_breaker
Circuit Breaker Pattern Implementation for Fault Recovery
=========================================================
Addresses partial recovery scenarios identified in fault injection tests:
- network_...

- **Path**: `circuit_breaker.py`
- **Size**: 518 lines
- **Imported by**: 2 modules
- **Classes**: CircuitState, CircuitBreakerConfig, CircuitMetrics, CircuitBreaker, CircuitBreakerError
- **Functions**: 29 functions

### spacy_loader
SpaCy Model Loader Module

Provides robust loading of spaCy models with automatic download capabilities
and graceful degradation when models are unavailable.

Features:
- Automatic model download with...

- **Path**: `spacy_loader.py`
- **Size**: 394 lines
- **Imported by**: 2 modules
- **Classes**: SpacyModelLoader, SafeSpacyProcessor, DegradedDoc, DegradedToken
- **Functions**: 18 functions

### jsonschema
- **Path**: `jsonschema.py`
- **Size**: 28 lines
- **Imported by**: 2 modules
- **Classes**: RefResolver, _DummyValidator
- **Functions**: 5 functions

### json_utils
JSON Utilities with NaN/Inf Handling
====================================

Comprehensive utilities for safe JSON serialization with automatic cleaning
of special float values (NaN, Infinity, -Infinity...

- **Path**: `json_utils.py`
- **Size**: 149 lines
- **Imported by**: 2 modules
- **Functions**: 4 functions

### miniminimoon_immutability
MINIMINIMOON Immutability Contract
==================================

Provides mechanisms to freeze and verify the integrity of the MINIMINIMOON integration.
This module ensures that critical compone...

- **Path**: `miniminimoon_immutability.py`
- **Size**: 555 lines
- **Imported by**: 1 modules
- **Classes**: ImmutabilityContract
- **Functions**: 12 functions

### Decatalogo_principal
Sistema Integral de Evaluación de Cadenas de Valor en Planes de Desarrollo Municipal
Versión: 9.0 – Marco Teórico-Institucional con Análisis Causal Multinivel, Frontier AI Capabilities,
Mathematical I...

- **Path**: `Decatalogo_principal.py`
- **Size**: 3094 lines
- **Imported by**: 1 modules
- **Classes**: AdvancedDeviceConfig, MathematicalInnovations, NivelAnalisis, TipoCadenaValor, TipoEvidencia
- **Functions**: 61 functions

### text_truncation_logger
Text Truncation Logging Utilities
Provides text truncation mechanisms for logging systems that limit logged text content
to a maximum length and replace full text with hash references, page numbers, a...

- **Path**: `text_truncation_logger.py`
- **Size**: 249 lines
- **Imported by**: 1 modules
- **Classes**: TextReference, TextTruncationLogger
- **Functions**: 16 functions

### system_validators
- **Path**: `system_validators.py`
- **Size**: 1 lines
- **Imported by**: 1 modules

### deterministic_pipeline_validator
Comprehensive Deterministic Pipeline Validator with Micro-Level Characterization
================================================================================
A sophisticated test suite for validat...

- **Path**: `deterministic_pipeline_validator.py`
- **Size**: 1434 lines
- **Imported by**: 1 modules
- **Classes**: TestSeverity, ContractType, TestResult, DependencyFlow, StateTransition
- **Functions**: 78 functions

### safe_io
Robust disk persistence helpers with explicit fallbacks.

- **Path**: `safe_io.py`
- **Size**: 184 lines
- **Imported by**: 1 modules
- **Classes**: SafeWriteResult
- **Functions**: 13 functions

### decalogo_loader
DECALOGO_INDUSTRIAL Template Loading Module

Provides atomic file operations with fallback template loading for DECALOGO components.
Ensures reliable access to decalogo templates even in restricted en...

- **Path**: `decalogo_loader.py`
- **Size**: 411 lines
- **Imported by**: 1 modules
- **Functions**: 7 functions

### memory_watchdog
Memory Monitoring Watchdog for Plan Processing Workers

This module implements a memory monitoring system using psutil that tracks RSS memory usage
during plan processing and terminates workers that e...

- **Path**: `memory_watchdog.py`
- **Size**: 541 lines
- **Imported by**: 1 modules
- **Classes**: TerminationReason, MemoryUsage, WatchdogEvent, MemoryWatchdog, PlanProcessingWatchdog
- **Functions**: 27 functions

### pattern_detector
Pattern Detection Module for Policy Indicator Analysis
======================================================

Advanced pattern detection system for identifying baseline values, targets,
and timeframe...

- **Path**: `factibilidad/pattern_detector.py`
- **Size**: 384 lines
- **Imported by**: 1 modules
- **Classes**: PatternMatch, PatternDetector
- **Functions**: 8 functions

### decalogo_loader_adapter
Adaptador para cargar los decálogos limpios y validar sus contratos.

- **Path**: `pdm_contra/bridges/decalogo_loader_adapter.py`
- **Size**: 89 lines
- **Imported by**: 1 modules
- **Classes**: CanonicalDecalogoBundle
- **Functions**: 5 functions

### prompt_maestro
Carga el Prompt Maestro para evaluación causal de PDM.

- **Path**: `pdm_contra/prompts/prompt_maestro.py`
- **Size**: 39 lines
- **Imported by**: 1 modules
- **Classes**: PromptLoadError
- **Functions**: 1 functions

## Test Modules

- `annotated_examples_test` - 286 lines
- `conftest` - 12 lines
- `demo_signal_test` - 247 lines
- `pdm_tests_examples` - 489 lines
- `run_all_tests` - 24 lines
- `run_tests` - 412 lines
- `test_autoload_in_apps` - 22 lines
- `test_basic_signal` - 60 lines
- `test_causal_pattern_detector` - 368 lines
- `test_competence_map` - 18 lines
- `test_contradiction_detector` - 367 lines
- `test_crosswalk_isomorphism` - 50 lines
- `test_dag_validation` - 228 lines
- `test_debug_demo` - 21 lines
- `test_decalogo_loader` - 159 lines
- `test_deterministic_normalization` - 25 lines
- `test_deterministic_seeding` - 371 lines
- `test_device_config` - 78 lines
- `test_device_cuda` - 44 lines
- `test_document_embedding_mapper` - 409 lines
- `test_document_segmenter` - 422 lines
- `test_embedding_device` - 69 lines
- `test_embedding_model` - 379 lines
- `test_embedding_model_import` - 38 lines
- `test_evidence_quality` - 59 lines
- `test_factibilidad` - 216 lines
- `test_feasibility_scorer` - 1160 lines
- `test_heap_functionality` - 130 lines
- `test_high_priority_fixes` - 1 lines
- `test_info_demo` - 21 lines
- `test_invalid_demo` - 20 lines
- `test_json_utils` - 267 lines
- `test_loader_compat` - 21 lines
- `test_log_config` - 136 lines
- `test_memory_watchdog` - 484 lines
- `test_monetary_detector` - 323 lines
- `test_parallel` - 74 lines
- `test_plan_processor` - 449 lines
- `test_plan_processor_basic` - 39 lines
- `test_plan_sanitizer` - 465 lines
- `test_prompt_maestro` - 33 lines
- `test_refined_scoring` - 274 lines
- `test_responsibility_detector` - 150 lines
- `test_safe_io` - 69 lines
- `test_schema_validation` - 51 lines
- `test_signal_handling` - 276 lines
- `test_sota_embedding_requirements` - 61 lines
- `test_spacy_loader` - 208 lines
- `test_teamcity_setup` - 426 lines
- `test_teoria_cambio` - 321 lines
- `test_text_truncation_logger` - 369 lines
- `test_unicode_normalization` - 153 lines
- `test_unicode_only` - 135 lines
- `test_with_joblib` - 91 lines
- `test_zero_evidence` - 124 lines
- `unicode_test_samples` - 89 lines

## Demo/Example Modules

- `demo` - Demonstration script showing the feasibility scorer in action....
- `demo_document_mapper` - Demo script for DocumentEmbeddingMapper with duplicate handling....
- `demo_document_segmentation` - No description...
- `demo_heap_functionality` - Demo de la nueva funcionalidad --max-segmentos con selección global top-k usando heap...
- `demo_plan_sanitizer` - Demonstration of plan name sanitization and JSON key standardization functionality....
- `demo_unicode_comparison` - INDUSTRIAL UNICODE NORMALIZATION ANALYSIS FRAMEWORK
================================================...
- `example_monetary_usage` - Example usage of the MonetaryDetector for Spanish text processing.

This module demonstrates how to ...
- `example_teoria_cambio` - Ejemplo de uso de la validación de Teoría de Cambio
Demuestra las capacidades de validación de orden...
- `example_usage` - Example usage of the embedding model....
- `integration_example` - INTEGRATION EXAMPLE: How to use the Authoritative Questionnaire Engine
=============================...

