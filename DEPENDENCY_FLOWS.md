# Dependency Flows Documentation

*Generated: 2025-10-05 11:01:35*

Total flows documented: 66

## Critical Flows

### 1. cli → embedding_model

- **Type**: data
- **Cardinality**: 1:N
- **Description**: cli depends on embedding_model
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 2. dag_validation → json_utils

- **Type**: utility
- **Cardinality**: 1:N
- **Description**: dag_validation depends on json_utils
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 3. dag_validation → log_config

- **Type**: configuration
- **Cardinality**: 1:N
- **Description**: dag_validation depends on log_config
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 4. decalogo_pipeline_orchestrator → monetary_detector

- **Type**: data
- **Cardinality**: 1:N
- **Description**: decalogo_pipeline_orchestrator depends on monetary_detector
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 5. decalogo_pipeline_orchestrator → causal_pattern_detector

- **Type**: data
- **Cardinality**: 1:N
- **Description**: decalogo_pipeline_orchestrator depends on causal_pattern_detector
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 6. decalogo_pipeline_orchestrator → teoria_cambio

- **Type**: data
- **Cardinality**: 1:N
- **Description**: decalogo_pipeline_orchestrator depends on teoria_cambio
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 7. decalogo_pipeline_orchestrator → feasibility_scorer

- **Type**: data
- **Cardinality**: 1:N
- **Description**: decalogo_pipeline_orchestrator depends on feasibility_scorer
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 8. decalogo_pipeline_orchestrator → responsibility_detector

- **Type**: data
- **Cardinality**: 1:N
- **Description**: decalogo_pipeline_orchestrator depends on responsibility_detector
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 9. decalogo_pipeline_orchestrator → contradiction_detector

- **Type**: data
- **Cardinality**: 1:N
- **Description**: decalogo_pipeline_orchestrator depends on contradiction_detector
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 10. decalogo_pipeline_orchestrator → document_segmenter

- **Type**: data
- **Cardinality**: 1:N
- **Description**: decalogo_pipeline_orchestrator depends on document_segmenter
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 11. example_usage → embedding_model

- **Type**: data
- **Cardinality**: 1:N
- **Description**: example_usage depends on embedding_model
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 12. integrated_evaluation_system → miniminimoon_orchestrator

- **Type**: data
- **Cardinality**: 1:N
- **Description**: integrated_evaluation_system depends on miniminimoon_orchestrator
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 13. integrated_evaluation_system → questionnaire_engine

- **Type**: data
- **Cardinality**: 1:N
- **Description**: integrated_evaluation_system depends on questionnaire_engine
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 14. integration_example → questionnaire_engine

- **Type**: data
- **Cardinality**: 1:N
- **Description**: integration_example depends on questionnaire_engine
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 15. miniminimoon_orchestrator → dag_validation

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on dag_validation
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 16. miniminimoon_orchestrator → evidence_registry

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on evidence_registry
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 17. miniminimoon_orchestrator → plan_processor

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on plan_processor
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 18. miniminimoon_orchestrator → causal_pattern_detector

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on causal_pattern_detector
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 19. miniminimoon_orchestrator → document_segmenter

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on document_segmenter
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 20. miniminimoon_orchestrator → miniminimoon_immutability

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on miniminimoon_immutability
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 21. miniminimoon_orchestrator → plan_sanitizer

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on plan_sanitizer
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 22. miniminimoon_orchestrator → responsibility_detector

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on responsibility_detector
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 23. miniminimoon_orchestrator → data_flow_contract

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on data_flow_contract
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 24. miniminimoon_orchestrator → embedding_model

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on embedding_model
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 25. miniminimoon_orchestrator → monetary_detector

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on monetary_detector
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 26. miniminimoon_orchestrator → spacy_loader

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on spacy_loader
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 27. miniminimoon_orchestrator → teoria_cambio

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on teoria_cambio
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 28. miniminimoon_orchestrator → feasibility_scorer

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on feasibility_scorer
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 29. miniminimoon_orchestrator → questionnaire_engine

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on questionnaire_engine
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 30. miniminimoon_orchestrator → contradiction_detector

- **Type**: data
- **Cardinality**: 1:N
- **Description**: miniminimoon_orchestrator depends on contradiction_detector
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 31. teoria_cambio → dag_validation

- **Type**: data
- **Cardinality**: 1:N
- **Description**: teoria_cambio depends on dag_validation
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 32. unified_evaluation_pipeline → miniminimoon_orchestrator

- **Type**: data
- **Cardinality**: 1:N
- **Description**: unified_evaluation_pipeline depends on miniminimoon_orchestrator
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 33. unified_evaluation_pipeline → system_validators

- **Type**: control
- **Cardinality**: 1:N
- **Description**: unified_evaluation_pipeline depends on system_validators
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 34. unified_evaluation_pipeline → questionnaire_engine

- **Type**: data
- **Cardinality**: 1:N
- **Description**: unified_evaluation_pipeline depends on questionnaire_engine
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 35. unified_evaluation_pipeline → evidence_registry

- **Type**: data
- **Cardinality**: 1:N
- **Description**: unified_evaluation_pipeline depends on evidence_registry
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 36. verify_coverage_metric → embedding_model

- **Type**: data
- **Cardinality**: 1:N
- **Description**: verify_coverage_metric depends on embedding_model
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 37. verify_coverage_metric → dag_validation

- **Type**: data
- **Cardinality**: 1:N
- **Description**: verify_coverage_metric depends on dag_validation
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 38. verify_coverage_metric → evidence_registry

- **Type**: data
- **Cardinality**: 1:N
- **Description**: verify_coverage_metric depends on evidence_registry
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

### 39. verify_reproducibility → dag_validation

- **Type**: data
- **Cardinality**: 1:N
- **Description**: verify_reproducibility depends on dag_validation
- **Input**: Dict[str, Any]
- **Output**: Dict[str, Any]

## Standard Flows

<details>
<summary>Click to expand all standard flows</summary>

- annotated_examples_test → causal_pattern_detector (data)
- cli → demo (data)
- cli → feasibility_scorer (data)
- cli → log_config (configuration)
- debug_causal_patterns → causal_pattern_detector (data)
- demo → feasibility_scorer (data)
- demo_performance_optimizations → data_flow_contract (data)
- demo_performance_optimizations → performance_test_suite (data)
- demo_plan_sanitizer → plan_sanitizer (data)
- demo_signal_test → log_config (configuration)
- example_monetary_usage → monetary_detector (data)
- example_teoria_cambio → teoria_cambio (data)
- investigate_fault_recovery → circuit_breaker (data)
- memory_watchdog → log_config (configuration)
- plan_processor → text_processor (data)
- plan_sanitizer → text_processor (data)
- profile_contract_validation → deterministic_pipeline_validator (control)
- run_tests → feasibility_scorer (data)
- validate_teoria_cambio → log_config (configuration)
- validate_teoria_cambio → teoria_cambio (data)
- verify_coverage_metric → circuit_breaker (data)
- verify_coverage_metric → data_flow_contract (data)
- verify_coverage_metric → feasibility_scorer (data)
- verify_coverage_metric → monetary_detector (data)
- verify_coverage_metric → plan_processor (data)
- verify_coverage_metric → responsibility_detector (data)
- verify_coverage_metric → text_processor (data)

</details>
