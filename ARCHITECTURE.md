# System Architecture Documentation
*Generated: 2025-10-05 11:01:35*

## Overview

This document describes the architecture of the MINIMINIMOON system, including 66 dependency flows, 5 critical paths, and detailed component interactions.

## System Components

### Core Components

- **causal_pattern_detector**
  - Incoming dependencies: 4
  - Outgoing dependencies: 0
  - Type: Critical

- **cli**
  - Incoming dependencies: 0
  - Outgoing dependencies: 4
  - Type: Critical

- **contradiction_detector**
  - Incoming dependencies: 2
  - Outgoing dependencies: 0
  - Type: Critical

- **dag_validation**
  - Incoming dependencies: 4
  - Outgoing dependencies: 2
  - Type: Critical

- **data_flow_contract**
  - Incoming dependencies: 3
  - Outgoing dependencies: 0
  - Type: Critical

- **decalogo_pipeline_orchestrator**
  - Incoming dependencies: 0
  - Outgoing dependencies: 7
  - Type: Critical

- **document_segmenter**
  - Incoming dependencies: 2
  - Outgoing dependencies: 0
  - Type: Critical

- **embedding_model**
  - Incoming dependencies: 4
  - Outgoing dependencies: 0
  - Type: Critical

- **evidence_registry**
  - Incoming dependencies: 3
  - Outgoing dependencies: 0
  - Type: Critical

- **example_usage**
  - Incoming dependencies: 0
  - Outgoing dependencies: 1
  - Type: Critical

- **feasibility_scorer**
  - Incoming dependencies: 6
  - Outgoing dependencies: 0
  - Type: Critical

- **integrated_evaluation_system**
  - Incoming dependencies: 0
  - Outgoing dependencies: 2
  - Type: Critical

- **integration_example**
  - Incoming dependencies: 0
  - Outgoing dependencies: 1
  - Type: Critical

- **json_utils**
  - Incoming dependencies: 1
  - Outgoing dependencies: 0
  - Type: Critical

- **log_config**
  - Incoming dependencies: 5
  - Outgoing dependencies: 0
  - Type: Critical

- **miniminimoon_immutability**
  - Incoming dependencies: 1
  - Outgoing dependencies: 0
  - Type: Critical

- **miniminimoon_orchestrator**
  - Incoming dependencies: 2
  - Outgoing dependencies: 16
  - Type: Critical

- **monetary_detector**
  - Incoming dependencies: 4
  - Outgoing dependencies: 0
  - Type: Critical

- **plan_processor**
  - Incoming dependencies: 2
  - Outgoing dependencies: 1
  - Type: Critical

- **plan_sanitizer**
  - Incoming dependencies: 2
  - Outgoing dependencies: 1
  - Type: Critical

- **questionnaire_engine**
  - Incoming dependencies: 4
  - Outgoing dependencies: 0
  - Type: Critical

- **responsibility_detector**
  - Incoming dependencies: 3
  - Outgoing dependencies: 0
  - Type: Critical

- **spacy_loader**
  - Incoming dependencies: 1
  - Outgoing dependencies: 0
  - Type: Critical

- **system_validators**
  - Incoming dependencies: 1
  - Outgoing dependencies: 0
  - Type: Critical

- **teoria_cambio**
  - Incoming dependencies: 4
  - Outgoing dependencies: 1
  - Type: Critical

- **unified_evaluation_pipeline**
  - Incoming dependencies: 0
  - Outgoing dependencies: 4
  - Type: Critical

- **verify_coverage_metric**
  - Incoming dependencies: 0
  - Outgoing dependencies: 10
  - Type: Critical

- **verify_reproducibility**
  - Incoming dependencies: 0
  - Outgoing dependencies: 1
  - Type: Critical


## Dependency Statistics

- Total dependency flows: 66
- Critical flows: 39
- Unique modules: 45
- Critical paths identified: 5

## Flow Types Distribution

- Configuration: 5 flows
- Control: 2 flows
- Data: 58 flows
- Utility: 1 flows

---

See also:
- [Dependency Flows](DEPENDENCY_FLOWS.md)
- [Critical Paths](CRITICAL_PATHS.md)
- [Data Contracts](DATA_CONTRACTS.md)
- [Component Diagram](COMPONENT_DIAGRAM.md)
