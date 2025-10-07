# INTEGRATION FLOW - Unified Evaluation Pipeline Execution Path

## Overview

This document traces the complete evaluation execution path from CLI commands through the unified_evaluation_pipeline facade, into miniminimoon_orchestrator for evidence generation, through AnswerAssembler for question-level response compilation using RUBRIC_SCORING.json weights, writing outputs to the artifacts/ directory, and finally through system_validators for validation with strict exit codes.

## Architecture Diagram

```
CLI (miniminimoon_cli.py)
    ↓
[Phase 1] Pre-Execution Validation (determinism_guard.py + system_validators.py)
    ↓
[Phase 2] Unified Pipeline Facade (unified_evaluation_pipeline.py)
    ↓
[Phase 3] Evidence Generation (miniminimoon_orchestrator.py)
    ├─→ 11 Canonical Pipeline Nodes
    └─→ EvidenceRegistry Building
    ↓
[Phase 4] Answer Assembly (answer_assembler.py)
    ├─→ Load RUBRIC_SCORING.json weights
    └─→ Compile question-level responses
    ↓
[Phase 5] Artifact Generation
    ├─→ artifacts/answers_report.json
    ├─→ artifacts/flow_runtime.json
    └─→ artifacts/evidence_registry.json
    ↓
[Phase 6] Post-Execution Validation (system_validators.py)
    ├─→ Coverage validation (≥300 questions)
    └─→ tools/rubric_check.py (1:1 alignment)
    ↓
[Phase 7] Traceability Matrix (trace_matrix.py)
    └─→ artifacts/module_to_questions_matrix.csv
```

## CLI Entry Points

**File**: `miniminimoon_cli.py`

Available commands:
- `freeze`: Freezes configuration (EnhancedImmutabilityContract snapshot)
- `evaluate`: Executes complete evaluation pipeline with pre/post validation
- `verify`: Validates post-execution artifacts (order, contracts, coverage)
- `rubric-check`: Validates 1:1 question-weight alignment via tools/rubric_check.py
- `trace-matrix`: Generates module→question provenance matrix CSV

**Exit Codes**:
- `0`: Success
- `1`: Internal error (exception)
- `2`: Validation failure (gates)
- `3`: Rubric mismatch (rubric_check.py specific)

## Phase 1: Pre-Execution Validation

**Components**: `system_validators.py`, `determinism_guard.py`

**Execution Flow**:

1. **Seed Freezing** (`determinism_guard.py`):
```python
from determinism_guard import enforce_determinism

# Called at pipeline start
enforce_determinism(seed=42, strict=True)
# Seeds: Python random, NumPy, PyTorch (if available)
# Guarantees: Reproducible embeddings, sampling, tie-breaking
```

2. **Pre-Execution Validation** (`system_validators.py`):
```python
validator = SystemHealthValidator(repo_root)
pre_valid, pre_report = validator.validate_pre_execution()

# Checks:
# - EnhancedImmutabilityContract.verify_frozen_config()
# - RUBRIC_SCORING.json exists
# - tools/flow_doc.json exists (canonical order)
```

**Gates Enforced**:
- Gate #1: Immutability freeze verification
- Deterministic seed application across all RNGs

## Phase 2: Unified Pipeline Facade

**Component**: `unified_evaluation_pipeline.py`

**Entry Point**:
```python
from unified_evaluation_pipeline import UnifiedEvaluationPipeline

pipeline = UnifiedEvaluationPipeline(
    repo_root=".",
    rubric_path="RUBRIC_SCORING.json",
    config_path="system_configuration.json"
)

results = pipeline.evaluate(
    pdm_path="plan.pdf",
    municipality="Example",
    department="Department",
    export_json=True,
    output_dir="artifacts"
)
```

**Orchestration**:
1. Warm-up models (embedding model preloading, thread-safe singleton)
2. Run pre-execution validation
3. Invoke miniminimoon_orchestrator for evidence generation
4. Run Decálogo evaluation (if enabled)
5. Run Questionnaire evaluation (300 questions)
6. Run post-execution validation
7. Export results to artifacts/

**Key Methods**:
- `_ensure_warmup()`: Thread-safe model initialization
- `_run_decalogo_evaluation()`: Consumes EvidenceRegistry
- `_run_questionnaire_evaluation()`: Parallel execution with EvidenceRegistry
- `_export_results()`: Writes artifacts/answers_report.json, flow_runtime.json

## Phase 3: Evidence Generation

**Component**: `miniminimoon_orchestrator.py`

**Canonical Pipeline (11 Nodes)**:

```python
orchestrator = MINIMINIMOONOrchestrator(config_path)
pipeline_results = orchestrator.process_plan(pdm_path)
evidence_registry = orchestrator.evidence_registry
```

**Canonical Order** (Gate #2):
1. `sanitization` - PlanSanitizer
2. `plan_processing` - PlanProcessor  
3. `document_segmentation` - DocumentSegmenter
4. `embedding` - EmbeddingModel (singleton pool)
5. `responsibility_detection` - ResponsibilityDetector
6. `contradiction_detection` - ContradictionDetector
7. `monetary_detection` - MonetaryDetector
8. `feasibility_scoring` - FeasibilityScorer
9. `causal_detection` - CausalPatternDetector
10. `teoria_cambio` - TeoriaCambioValidator
11. `dag_validation` - DAGValidator

**EvidenceRegistry Building**:
```python
class EvidenceRegistry:
    def register(self, entry: EvidenceEntry) -> str:
        # Thread-safe registration with RLock
        # Builds stage_index and segment_index
        # Returns evidence_id
        
    def deterministic_hash(self) -> str:
        # SHA-256 of all evidence hashes (sorted)
        # Guarantees: Identical across triple runs
```

**Evidence Entry Schema**:
```python
@dataclass
class EvidenceEntry:
    evidence_id: str  # Format: "{module}::{type}::{hash}"
    stage: str
    content: Any
    source_segment_ids: List[str]
    confidence: float
    timestamp: str
    metadata: Dict[str, Any]
```

**Immutability Proof**:
- Frozen configuration snapshot (SHA-256)
- Deterministic flow hash (stage order verification)
- Evidence registry hash (content verification)

## Phase 4: Answer Assembly

**Component**: `answer_assembler.py`

**RUBRIC_SCORING.json Loading** (Single Source of Truth):

```python
class AnswerAssembler:
    def __init__(self, rubric_path="RUBRIC_SCORING.json"):
        self.rubric_config = self._load_json_config(rubric_path)
        
        # Load weights section (per-question mappings)
        self.weights = self.rubric_config.get("weights", {})
        
        # Load questions section (scoring modalities)
        self.question_templates = self._parse_question_templates()
        
        # Validate 1:1 alignment (Gate #5)
        self._validate_rubric_coverage()
```

**Question-Level Response Compilation**:

```python
def assemble(self, question_id, dimension, relevant_evidence_ids, 
             raw_score, reasoning) -> Answer:
    # 1. Look up rubric weight from RUBRIC_SCORING.json
    weight = self.weights.get(question_id)
    if weight is None:
        raise KeyError(f"Gate #5 failure: {question_id} missing weight")
    
    # 2. Retrieve evidence from registry
    evidence_entries = [self.registry.get(eid) for eid in relevant_evidence_ids]
    
    # 3. Calculate confidence (Bayesian)
    confidence = self._calculate_confidence(evidence_entries, raw_score)
    
    # 4. Extract supporting quotes
    quotes = self._extract_quotes(evidence_entries)
    
    # 5. Generate reasoning
    reasoning = self._generate_reasoning(dimension, evidence_entries, raw_score)
    
    # 6. Identify caveats
    caveats = self._identify_caveats(evidence_entries, raw_score)
    
    return Answer(
        question_id=question_id,
        dimension=dimension,
        evidence_ids=relevant_evidence_ids,
        confidence=confidence,
        score=raw_score,
        reasoning=reasoning,
        rubric_weight=weight,  # From RUBRIC_SCORING.json
        supporting_quotes=quotes,
        caveats=caveats
    )
```

**Weights Configuration**:
- **Source**: `RUBRIC_SCORING.json` "weights" section
- **Schema**: `{"D1-Q1-P01": 0.033, "D1-Q2-P01": 0.033, ...}`
- **Coverage**: All 300 questions (D1-D6, Q1-Q5, P01-P10)
- **Validation**: Gate #5 enforces strict 1:1 alignment

## Phase 5: Artifact Generation

**Output Directory**: `artifacts/`

**Key Artifacts**:

1. **answers_report.json**:
```json
{
  "metadata": {
    "system_version": "3.0",
    "evaluation_timestamp": "2025-01-XX...",
    "rubric_file": "RUBRIC_SCORING.json"
  },
  "global_summary": {
    "global_percentage": 78.5,
    "score_band": "BUENO",
    "total_questions": 300,
    "answered_questions": 300,
    "questions_with_evidence": 287
  },
  "question_answers": [
    {
      "question_id": "D1-Q1-P01",
      "dimension": "D1",
      "raw_score": 2.5,
      "rubric_weight": 0.033,
      "evidence_ids": ["responsibility::assignment::a3f9c2e1"],
      "confidence": 0.85,
      "rationale": "Strong evidence from responsibility detection..."
    }
  ]
}
```

2. **flow_runtime.json**:
```json
{
  "flow_hash": "sha256_of_stage_order",
  "stages": [
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
  ],
  "stage_count": 11,
  "duration_seconds": 45.3
}
```

3. **evidence_registry.json**:
```json
{
  "evidence_count": 542,
  "deterministic_hash": "a1b2c3d4e5f6...",
  "evidence": {
    "responsibility::assignment::hash1": {...},
    "monetary::budget::hash2": {...}
  }
}
```

## Phase 6: Post-Execution Validation

**Component**: `system_validators.py`

**Post-Execution Checks**:

```python
validator = SystemHealthValidator(repo_root)
post_valid, post_report = validator.validate_post_execution(
    artifacts_dir="artifacts",
    check_rubric_strict=True
)
```

**Validation Gates**:

1. **Artifact Existence**:
   - `artifacts/flow_runtime.json` exists
   - `artifacts/answers_report.json` exists

2. **Canonical Order Verification** (Gate #2):
```python
# Compare tools/flow_doc.json vs flow_runtime.json
doc_order = flow_doc.get("canonical_order", [])
rt_order = runtime_trace.get("order", [])
ok_order = (doc_order == rt_order)  # Exact match required
```

3. **Coverage Validation**:
```python
total_questions = answers.get("summary", {}).get("total_questions", 0)
ok_coverage = total_questions >= 300  # Must cover all 300 questions
```

4. **Rubric 1:1 Alignment** (Gate #5):
```python
# Invoke tools/rubric_check.py as subprocess
result = subprocess.run(
    [sys.executable, "tools/rubric_check.py", 
     "artifacts/answers_report.json", "RUBRIC_SCORING.json"],
    capture_output=True
)

# Handle exit codes:
# 0 = success (1:1 alignment verified)
# 2 = missing input files
# 3 = mismatch (missing_weights or extra_weights detected)
```

**tools/rubric_check.py Exit Codes**:

```python
def check_rubric_alignment(answers_path, rubric_path):
    # Load files
    answers = json.load(answers_path)
    rubric = json.load(rubric_path)
    
    # Extract IDs
    answer_ids = {a["question_id"] for a in answers["answers"]}
    weight_ids = set(rubric["weights"].keys())
    
    # Check alignment
    missing = [qid for qid in answer_ids if qid not in weight_ids]
    extra = [qid for qid in weight_ids if qid not in answer_ids]
    
    if missing or extra:
        print(json.dumps({
            "ok": False,
            "missing_in_rubric": missing[:10],
            "extra_in_rubric": extra[:10]
        }))
        return 3  # Mismatch exit code
    
    return 0  # Success
```

**Exit Codes Summary**:
- `0`: All validations passed
- `2`: Validation failure (missing artifacts, coverage < 300)
- `3`: Rubric mismatch (1:1 alignment failed)

## Phase 7: Traceability Matrix Generation

**Component**: `trace_matrix.py`

**Purpose**: Generate module→question provenance matrix for auditing evidence coverage.

**Execution**:
```bash
python tools/trace_matrix.py
# Input: artifacts/answers_report.json
# Output: artifacts/module_to_questions_matrix.csv
```

**Matrix Generation**:

```python
def build_traceability_matrix(answers):
    rows = []
    for ans in answers:
        question_id = ans["question_id"]
        confidence = ans["confidence"]
        score = ans["score"]
        
        for evidence_id in ans["evidence_ids"]:
            # Extract module from evidence_id
            # Format: "module::type::hash"
            module = evidence_id.split("::", 1)[0]
            
            rows.append({
                "module": module,
                "question_id": question_id,
                "evidence_id": evidence_id,
                "confidence": confidence,
                "score": score
            })
    
    return rows
```

**Output CSV Schema**:
```csv
module,question_id,evidence_id,confidence,score
responsibility_detector,D1-Q1-P01,responsibility::assignment::a3f9,0.85,2.5
monetary_detector,D3-Q2-P05,monetary::budget::b4e2,0.72,2.0
feasibility_scorer,D5-Q3-P08,feasibility::score::c9d1,0.68,1.5
```

**Use Cases**:
- Audit which detectors contribute to which questions
- Identify under-utilized or over-utilized modules
- Verify evidence provenance chain
- Debug coverage gaps

**Exit Codes**:
- `0`: Success (CSV generated)
- `1`: Runtime error
- `2`: Missing input (answers_report.json not found)
- `3`: Malformed data (schema validation failed)

## Deterministic Reproducibility Guarantees

### Seed Freezing (`determinism_guard.py`)

**Enforcement**:
```python
from determinism_guard import enforce_determinism, verify_determinism

# Called at pipeline initialization
state = enforce_determinism(seed=42, strict=True)
# Seeds:
# - Python random.seed(42)
# - NumPy np.random.seed(42)
# - PyTorch torch.manual_seed(42)
# - PyTorch CUDA torch.cuda.manual_seed_all(42)

# Verification (optional, for testing)
verify_result = verify_determinism(seed=42, n_samples=100)
assert verify_result["deterministic"] == True
```

**Components Requiring Determinism**:
1. **EmbeddingModel**: Dropout, random projection, batching order
2. **DocumentSegmenter**: Sampling for large documents
3. **FeasibilityScorer**: Tie-breaking in ranking
4. **TeoriaCambio**: Graph construction with non-deterministic ordering

### Triple-Run Consistency

**Test Protocol**:
```bash
# Run 1
python miniminimoon_cli.py evaluate plan.pdf --output-dir artifacts/run1
hash1=$(jq -r '.evidence_registry.deterministic_hash' artifacts/run1/evidence_registry.json)

# Run 2
python miniminimoon_cli.py evaluate plan.pdf --output-dir artifacts/run2
hash2=$(jq -r '.evidence_registry.deterministic_hash' artifacts/run2/evidence_registry.json)

# Run 3
python miniminimoon_cli.py evaluate plan.pdf --output-dir artifacts/run3
hash3=$(jq -r '.evidence_registry.deterministic_hash' artifacts/run3/evidence_registry.json)

# Verify consistency
[ "$hash1" == "$hash2" ] && [ "$hash2" == "$hash3" ] && echo "✓ Deterministic"
```

**Guaranteed Invariants**:
- **Evidence Hash**: Same SHA-256 across all 3 runs
- **Flow Hash**: Same canonical order execution
- **Answer Scores**: Identical raw_score values (±0.001 tolerance for floating-point)
- **Coverage**: Exactly 300/300 questions every run

### Evidence Hash Computation

**Implementation** (`EvidenceRegistry.deterministic_hash()`):
```python
def deterministic_hash(self) -> str:
    # 1. Sort evidence IDs alphabetically
    sorted_eids = sorted(self._evidence.keys())
    
    # 2. Hash each evidence entry content
    hash_inputs = [self._evidence[eid].to_hash() for eid in sorted_eids]
    
    # 3. Combine hashes with deterministic separator
    combined = "|".join(hash_inputs)
    
    # 4. Compute final SHA-256
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()
```

**Properties**:
- **Order-independent**: Sorting ensures consistent ordering
- **Content-sensitive**: Any evidence change → different hash
- **Collision-resistant**: SHA-256 provides 256-bit security
- **Reproducible**: Same evidence → same hash across runs

## Complete Integration Flow Summary

### Execution Sequence

```
1. CLI Command
   └─→ miniminimoon_cli.py evaluate plan.pdf

2. Pre-Execution Validation
   ├─→ determinism_guard.enforce_determinism(seed=42)
   └─→ system_validators.validate_pre_execution()
       ├─→ Check immutability freeze
       ├─→ Verify RUBRIC_SCORING.json exists
       └─→ Verify tools/flow_doc.json exists

3. Unified Pipeline Orchestration
   └─→ unified_evaluation_pipeline.evaluate()
       ├─→ _ensure_warmup() [thread-safe model loading]
       └─→ miniminimoon_orchestrator.process_plan()

4. Evidence Generation (11 Canonical Nodes)
   └─→ miniminimoon_orchestrator.process_plan()
       ├─→ Node 1-11: Detectors & validators
       ├─→ EvidenceRegistry.register() [thread-safe]
       └─→ Immutability proof generation

5. Answer Assembly
   └─→ answer_assembler.assemble()
       ├─→ Load RUBRIC_SCORING.json weights
       ├─→ Retrieve evidence from registry
       ├─→ Calculate confidence (Bayesian)
       ├─→ Generate rationale
       └─→ Return Answer with rubric_weight

6. Artifact Generation
   ├─→ artifacts/answers_report.json
   ├─→ artifacts/flow_runtime.json
   └─→ artifacts/evidence_registry.json

7. Post-Execution Validation
   └─→ system_validators.validate_post_execution()
       ├─→ Verify artifact existence
       ├─→ Check canonical order (Gate #2)
       ├─→ Verify coverage ≥ 300 questions
       └─→ Invoke tools/rubric_check.py (Gate #5)
           └─→ Exit codes: 0=success, 2=missing, 3=mismatch

8. Traceability Matrix
   └─→ trace_matrix.py
       ├─→ Parse answers_report.json
       ├─→ Extract module from evidence_id
       └─→ Write module_to_questions_matrix.csv
```

### Quality Gates

| Gate | Component | Check | Exit Code on Failure |
|------|-----------|-------|---------------------|
| #1 | system_validators | Immutability freeze | 2 |
| #2 | system_validators | Canonical order match | 2 |
| #3 | miniminimoon_orchestrator | Evidence registry building | 1 |
| #4 | answer_assembler | RUBRIC_SCORING.json loading | 1 |
| #5 | answer_assembler + rubric_check.py | 1:1 question-weight alignment | 3 |
| #6 | system_validators | Coverage ≥ 300 questions | 2 |

### Exit Code Semantics

| Code | Meaning | Context |
|------|---------|---------|
| 0 | Success | All validations passed |
| 1 | Internal error | Exception, missing dependencies, I/O failure |
| 2 | Validation failure | Missing artifacts, coverage < 300, order mismatch |
| 3 | Rubric mismatch | tools/rubric_check.py detected missing_weights or extra_weights |

### Artifact Dependencies

```
RUBRIC_SCORING.json (input)
    ↓
[Evidence Generation]
    ↓
artifacts/evidence_registry.json
    ├─→ deterministic_hash [used for triple-run verification]
    └─→ evidence entries [consumed by AnswerAssembler]
    ↓
artifacts/answers_report.json
    ├─→ question_answers[] [300 entries]
    ├─→ evidence_ids per question [consumed by trace_matrix.py]
    └─→ rubric_weight per question [from RUBRIC_SCORING.json]
    ↓
artifacts/flow_runtime.json
    ├─→ stages[] [canonical order]
    └─→ flow_hash [verified against tools/flow_doc.json]
    ↓
artifacts/module_to_questions_matrix.csv
    └─→ module,question_id,evidence_id,confidence,score [audit trail]
```

## Validation Chain

### Pre-Execution
1. Freeze verification → deterministic configuration
2. Seed enforcement → reproducible RNGs
3. File existence → RUBRIC_SCORING.json, flow_doc.json

### Post-Execution
1. Artifact existence → answers_report.json, flow_runtime.json
2. Order verification → flow_doc.json ↔ flow_runtime.json exact match
3. Coverage validation → total_questions ≥ 300
4. Rubric alignment → tools/rubric_check.py subprocess
5. Traceability → module_to_questions_matrix.csv generation

### Deterministic Reproducibility
1. Evidence hash → SHA-256 of sorted evidence
2. Flow hash → SHA-256 of stage order
3. Triple-run test → hash1 == hash2 == hash3
4. Seed verification → verify_determinism() after enforce

## Tool Contribution Summary

| Tool | Phase | Contribution | Output |
|------|-------|--------------|--------|
| miniminimoon_cli.py | 0 | CLI entry point | Command routing |
| determinism_guard.py | 1 | Seed freezing | Reproducible RNGs |
| system_validators.py | 1,6 | Pre/post validation | Gate enforcement |
| unified_evaluation_pipeline.py | 2 | Orchestration facade | Pipeline coordination |
| miniminimoon_orchestrator.py | 3 | Evidence generation | EvidenceRegistry + 11 nodes |
| answer_assembler.py | 4 | Question-level compilation | Answer objects with weights |
| RUBRIC_SCORING.json | 4 | Weight configuration | Per-question weights |
| trace_matrix.py | 7 | Provenance tracking | module_to_questions_matrix.csv |
| tools/rubric_check.py | 6 | 1:1 alignment validation | Exit codes 0/2/3 |

## Coverage Verification (300/300)

**Mechanism**:
```python
# In system_validators.py
answers = json.load("artifacts/answers_report.json")
total = answers["summary"]["total_questions"]
assert total >= 300, f"Coverage failed: {total}/300"
```

**RUBRIC_SCORING.json Structure**:
- **Dimensions**: D1-D6 (6 dimensions)
- **Questions per dimension**: Q1-Q5 (5 questions)
- **Points per question**: P01-P10 (10 points)
- **Total**: 6 × 5 × 10 = 300 questions
- **Weight per question**: 1.0 / 300 ≈ 0.00333

**Validation**:
- Every question ID in answers_report.json must exist in RUBRIC_SCORING.json weights
- Every weight in RUBRIC_SCORING.json must correspond to a question in answers_report.json
- No missing weights, no extra weights (strict 1:1)

## Evidence Hash Consistency (Triple-Run)

**Test Command**:
```bash
for i in {1..3}; do
  python miniminimoon_cli.py evaluate plan.pdf --output-dir "artifacts/run$i"
  hash=$(jq -r '.evidence_registry.deterministic_hash' "artifacts/run$i/evidence_registry.json")
  echo "Run $i: $hash"
done
```

**Expected Output**:
```
Run 1: a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890
Run 2: a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890
Run 3: a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890
✓ Deterministic: All hashes identical
```

**Failure Cases**:
- Different hashes → Non-deterministic RNG, missing seed enforcement
- Different stage counts → Pipeline order violation
- Different scores → Evidence generation instability

## Conclusion

This integration flow ensures:
- **Determinism**: Reproducible results via seed freezing
- **Traceability**: Evidence provenance via trace_matrix.py
- **Coverage**: 300/300 questions via system_validators.py
- **Alignment**: 1:1 question-weight mapping via tools/rubric_check.py
- **Immutability**: SHA-256 verification via miniminimoon_immutability
- **Quality Gates**: Strict validation with semantic exit codes

All components work together to guarantee deterministic, auditable, and reproducible evaluation execution.
