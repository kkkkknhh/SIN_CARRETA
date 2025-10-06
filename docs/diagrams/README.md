# Visual Architecture Diagrams - MINIMINIMOON System

**Version**: 2.0.0  
**Date**: October 6, 2025  
**Status**: ✅ Complete - 7 Advanced Diagrams

---

## Overview

This directory contains **SEVEN hyper-modern, futuristic neo-punk architectural diagrams** that provide a comprehensive visual narrative of the MINIMINIMOON unified evaluation architecture. Each diagram uses consistent color schemes, clear node labels, and directional arrows with cardinality annotations to illustrate different aspects of the system.

---

## Diagram Index

### 1. System Architecture (`01_system_architecture.dot`)
**Type**: High-level overview  
**Purpose**: Entry point to artifact generation flow  
**Key Components**:
- CLI Interface (miniminimoon_cli.py)
- Unified Evaluation Pipeline (pre/post validation)
- Canonical Orchestrator (15-stage pipeline)
- Artifact Generation

**Cardinality**: 1:1 linear flow with side validators

**Cross-References**:
- FLUJOS_CRITICOS_GARANTIZADOS.md - Flow #18
- ARCHITECTURE.md - Core Components

---

### 2. Evidence Data Flow (`02_data_flow.dot`)
**Type**: Data flow with fan-in/fan-out  
**Purpose**: Evidence lifecycle from detection to evaluation  
**Key Components**:
- 7 Parallel Detectors (Stages 5-11)
- Evidence Registry (Stage 12 - FAN-IN N:1)
- 2 Evaluators (Stages 13-14)
- Answer Assembler (Stage 15)

**Cardinality**: N:1 fan-in at registry, 1:1 to evaluators

**Cross-References**:
- FLUJOS_CRITICOS_GARANTIZADOS.md - Flows #5-#15
- ARCHITECTURE.md - Evidence Registry component

---

### 3. Validation Gates (`03_validation_gates.dot`)
**Type**: Control flow with decision points  
**Purpose**: Pre/post-execution validation gates  
**Key Components**:
- Pre-Execution Gates (1, 2, 6)
  - Gate #1: Freeze verification
  - Gate #2: Flow order validation
  - Gate #6: No deprecated modules
- Pipeline Execution (15 stages)
- Post-Execution Gates (3, 4, 5)
  - Gate #3: Determinism (hash stability)
  - Gate #4: Coverage (≥300 questions)
  - Gate #5: Rubric alignment

**Cardinality**: Sequential with parallel failure paths

**Cross-References**:
- FLUJOS_CRITICOS_GARANTIZADOS.md - Section 2 (Gates)
- README.md - Gates de Aceptación

---

### 4. CI/CD Pipeline (`04_cicd_pipeline.dot`)
**Type**: Sequential workflow with branching  
**Purpose**: Complete build and deployment workflow  
**Key Stages**:
1. Setup (checkout, Python, spaCy)
2. Freeze Verification (Gate #1)
3. Build (py_compile)
4. Lint (PEP 8)
5. Triple-Run Reproducibility (Gate #3)
6. Unit Tests
7. Integration Tests (72 flows)
8. Performance Gate (p95 < budget+10%)
9. Artifact Archival (30-day retention)

**Cardinality**: 1:1 sequential with failure exits

**Cross-References**:
- FLUJOS_CRITICOS_GARANTIZADOS.md - Section 9 (Determinismo)
- README.md - Contribución > CI/CD Pipeline

---

### 5. 15-Stage Pipeline (`05_15_stage_pipeline.dot`)
**Type**: Detailed sequential flow  
**Purpose**: Complete stage-by-stage pipeline visualization  
**Key Phases**:
- **Phase 1**: Processing (Stages 1-11)
  - Sanitization → Plan Processing → Segmentation → Embeddings
  - 7 Parallel Detectors (Responsibility, Contradiction, Monetary, Feasibility, Causal, ToC, DAG)
- **Phase 2**: Evidence Registry (Stage 12 - FAN-IN)
- **Phase 3**: Evaluation (Stages 13-14)
  - Decálogo Evaluator
  - Questionnaire Engine (300 questions)
- **Phase 4**: Assembly (Stage 15)

**Cardinality**: 1:N at Stage 4 (fan-out), N:1 at Stage 12 (fan-in), 1:1 final assembly

**Cross-References**:
- FLUJOS_CRITICOS_GARANTIZADOS.md - Section 1 (15 Flujos)
- ARCHITECTURE.md - System Components

---

### 6. Contract Validation (`06_contract_validation.dot`)
**Type**: Left-right validation flow  
**Purpose**: Data contract validation layers  
**Key Components**:
- Type Validation (schema conformance)
- Mathematical Invariants
  - PERMUTATION_INVARIANCE
  - MONOTONICITY
  - IDEMPOTENCE
  - CONSERVATION
- Semantic Validation (domain rules)
- Validation Cache (37% speedup)

**Performance Metrics**:
- Type check: ~2ms
- Invariant check: ~0.15ms
- Semantic check: ~1ms
- Cache hit: ~0.1ms
- **Target p95**: <5ms

**Cross-References**:
- DATA_CONTRACTS.md - Contract details
- README.md - Performance y Optimizaciones

---

### 7. Deployment & Monitoring (`07_deployment_monitoring.dot`)
**Type**: Infrastructure flow with feedback loops  
**Purpose**: Canary deployment with observability  
**Key Components**:
- Traffic Router (progressive rollout: 5%→25%→100%)
- Baseline v1.0 (current production)
- Canary v2.0 (new deployment)
- OpenTelemetry Tracing (28 flows, 11 components)
- Metrics Collector (error rate, latency, availability)
- SLO Monitor (thresholds):
  - Availability: 99.5%
  - P95 Latency: 200ms
  - Error Rate: 0.1%
  - Performance Regression: 10%
  - Fault Recovery: 1.5s p99
- Decision Engine (promote/rollback)

**Cardinality**: 1:N at router, N:1 at decision engine

**Cross-References**:
- DEPLOYMENT_INFRASTRUCTURE.md - Complete deployment docs
- README.md - Deployment Infrastructure

---

## Design Principles

### Color Scheme (Hyper Modern Neo-Punk)

```
Magenta (#ff00ff)  → CLI, Entry Points, Critical Gates
Cyan (#00ffff)     → Core Processing, Evidence Registry
Yellow (#ffff00)   → Evaluation, SLO Monitoring
Green (#00ff88)    → Success States, Validation Passed
Red (#ff0000)      → Failure States, Rollback Actions
Blue (#00d4ff)     → Orchestration, Components
```

### Typography
- **Font**: JetBrains Mono (monospace for technical aesthetic)
- **Size**: 9-12pt for nodes, 8-9pt for edges
- **Style**: Bold for titles, regular for descriptions

### Edge Annotations
All edges labeled with cardinality:
- **1:1**: One-to-one relationship
- **1:N**: One-to-many (fan-out)
- **N:1**: Many-to-one (fan-in)

Example: `Detectors → Registry [label="N:1", color="#00ffff"]`

### Graph Layouts
- **TB (Top-Bottom)**: Sequential flows (pipelines, CI/CD, validation gates)
- **LR (Left-Right)**: Data validation, contract checking
- **splines**: `ortho` for clean orthogonal edges, `polyline` for complex flows

### Background
- **Dark theme**: #0a0e27 (deep navy) for high contrast
- **High DPI**: 300 DPI for publication-quality output

---

## Generating Images

### Requirements

**System**:
- Graphviz 2.40+: `brew install graphviz` (macOS) or `apt-get install graphviz` (Linux)
- Python 3.7+

**Python (optional)**:
```bash
pip install graphviz
```

### Generation Script

```bash
cd docs/diagrams
python3 generate_images.py
```

This will generate high-resolution PNG files (300 DPI) from all `.dot` source files.

### Manual Generation

```bash
cd docs/diagrams
dot -Tpng -Gdpi=300 01_system_architecture.dot -o 01_system_architecture.png
dot -Tpng -Gdpi=300 02_data_flow.dot -o 02_data_flow.png
# ... repeat for all 7 diagrams
```

### Output Files

```
01_system_architecture.png      # ~500KB, 3000×2000px
02_data_flow.png                # ~600KB, 3500×2500px
03_validation_gates.png         # ~700KB, 3000×3000px
04_cicd_pipeline.png            # ~800KB, 3500×4000px
05_15_stage_pipeline.png        # ~900KB, 4000×3500px
06_contract_validation.png      # ~450KB, 3500×2000px
07_deployment_monitoring.png    # ~750KB, 3500×3000px
```

---

## Integration with Documentation

These diagrams are embedded in the main README.md in the **Visual Architecture Diagrams** section. Each diagram includes:

1. **Location**: Path to PNG file
2. **Mermaid Preview**: Simplified inline version for GitHub rendering
3. **Description**: Detailed explanation of components and flows
4. **References**: Cross-references to detailed documentation files

### Example Integration

```markdown
### 1️⃣ High-Level System Architecture

**Location**: `docs/diagrams/01_system_architecture.png`

[Mermaid diagram inline preview]

**Description**: Shows CLI → Unified Pipeline → Orchestrator → Artifacts flow...

**References**: 
- [FLUJOS_CRITICOS_GARANTIZADOS.md](FLUJOS_CRITICOS_GARANTIZADOS.md) - Flow #18
- [ARCHITECTURE.md](ARCHITECTURE.md) - Core Components
```

---

## Maintenance

### Adding New Diagrams

1. Create `.dot` file following naming convention: `NN_descriptive_name.dot`
2. Use consistent color scheme and typography
3. Add cardinality labels to all edges
4. Test generation: `dot -Tpng -Gdpi=300 NN_descriptive_name.dot -o NN_descriptive_name.png`
5. Update this README with diagram details
6. Add reference in main README.md Visual Architecture section

### Updating Existing Diagrams

1. Edit `.dot` source file
2. Regenerate PNG: `python3 generate_images.py`
3. Verify output quality (300 DPI, clear labels)
4. Update descriptions if components changed
5. Check cross-references are still valid

---

## Cross-Reference Matrix

| Diagram | FLUJOS_CRITICOS | ARCHITECTURE | README | DEPLOYMENT | DATA_CONTRACTS |
|---------|----------------|--------------|--------|------------|----------------|
| 01_system_architecture | Flow #18 | Core Components | ✓ | - | - |
| 02_data_flow | Flows #5-#15 | Evidence Registry | ✓ | - | - |
| 03_validation_gates | Section 2 (Gates) | - | Gates section | - | - |
| 04_cicd_pipeline | Section 9 | - | CI/CD section | - | - |
| 05_15_stage_pipeline | Section 1 (15 Flujos) | System Components | ✓ | - | - |
| 06_contract_validation | - | - | Performance section | - | ✓ |
| 07_deployment_monitoring | - | - | Deployment section | ✓ | - |

---

## Diagram Statistics

```
Total DOT files: 7
Total lines: 972
Average complexity: 139 lines/diagram
Total size (DOT): ~60KB
Total size (PNG): ~4.7MB (estimated at 300 DPI)
```

**Breakdown by diagram**:
- 01_system_architecture: 86 lines, 5.1KB
- 02_data_flow: 117 lines, 7.2KB
- 03_validation_gates: 137 lines, 8.5KB
- 04_cicd_pipeline: 176 lines, 11KB
- 05_15_stage_pipeline: 206 lines, 11KB
- 06_contract_validation: 114 lines, 7.3KB
- 07_deployment_monitoring: 136 lines, 8.7KB

---

## License

These diagrams are part of the MINIMINIMOON project and follow the same license as the main project.

---

## Contact

For questions or suggestions about the visual architecture:
- See main README.md for project contacts
- Open an issue in the project repository
- Refer to ARCHITECTURE.md for detailed technical documentation

---

**Last Updated**: October 6, 2025  
**Diagram Version**: 2.0.0  
**Status**: ✅ Complete and Verified
