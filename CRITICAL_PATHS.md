# Critical Paths Documentation

*Generated: 2025-10-05 11:01:35*

## Overview

This document describes the 5 critical execution paths in the system. These paths represent the most important data flows and should be thoroughly tested and monitored.

## Path 1: unified_evaluation_pipeline → miniminimoon_orchestrator → dag_validation

### Flow Description

1. **unified_evaluation_pipeline** processes data and passes to **miniminimoon_orchestrator**
2. **miniminimoon_orchestrator** processes data and passes to **dag_validation**

### Testing Requirements

- [ ] Unit tests for each component
- [ ] Integration test for complete path
- [ ] Performance benchmark
- [ ] Error handling validation
- [ ] Contract compliance check

### Monitoring Points

- unified_evaluation_pipeline: Execution time, error rate, throughput
- miniminimoon_orchestrator: Execution time, error rate, throughput
- dag_validation: Execution time, error rate, throughput

## Path 2: decalogo_pipeline_orchestrator → decalogo_loader → system_validators

### Flow Description

1. **decalogo_pipeline_orchestrator** processes data and passes to **decalogo_loader**
2. **decalogo_loader** processes data and passes to **system_validators**

### Testing Requirements

- [ ] Unit tests for each component
- [ ] Integration test for complete path
- [ ] Performance benchmark
- [ ] Error handling validation
- [ ] Contract compliance check

### Monitoring Points

- decalogo_pipeline_orchestrator: Execution time, error rate, throughput
- decalogo_loader: Execution time, error rate, throughput
- system_validators: Execution time, error rate, throughput

## Path 3: embedding_model → spacy_loader → device_config

### Flow Description

1. **embedding_model** processes data and passes to **spacy_loader**
2. **spacy_loader** processes data and passes to **device_config**

### Testing Requirements

- [ ] Unit tests for each component
- [ ] Integration test for complete path
- [ ] Performance benchmark
- [ ] Error handling validation
- [ ] Contract compliance check

### Monitoring Points

- embedding_model: Execution time, error rate, throughput
- spacy_loader: Execution time, error rate, throughput
- device_config: Execution time, error rate, throughput

## Path 4: questionnaire_engine → evidence_registry → system_validators

### Flow Description

1. **questionnaire_engine** processes data and passes to **evidence_registry**
2. **evidence_registry** processes data and passes to **system_validators**

### Testing Requirements

- [ ] Unit tests for each component
- [ ] Integration test for complete path
- [ ] Performance benchmark
- [ ] Error handling validation
- [ ] Contract compliance check

### Monitoring Points

- questionnaire_engine: Execution time, error rate, throughput
- evidence_registry: Execution time, error rate, throughput
- system_validators: Execution time, error rate, throughput

## Path 5: plan_processor → plan_sanitizer → json_utils

### Flow Description

1. **plan_processor** processes data and passes to **plan_sanitizer**
2. **plan_sanitizer** processes data and passes to **json_utils**

### Testing Requirements

- [ ] Unit tests for each component
- [ ] Integration test for complete path
- [ ] Performance benchmark
- [ ] Error handling validation
- [ ] Contract compliance check

### Monitoring Points

- plan_processor: Execution time, error rate, throughput
- plan_sanitizer: Execution time, error rate, throughput
- json_utils: Execution time, error rate, throughput

