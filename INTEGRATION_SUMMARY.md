# AnswerAssembler Integration Summary

## Modifications to miniminimoon_orchestrator.py

### 1. Pipeline Stage Addition
- **Added**: `PipelineStage.ANSWER_ASSEMBLY = "answers_assembly"` enum value
- **Position**: Flow #15 - executes after `questionnaire_evaluation` completes
- **Location**: Line 247 in PipelineStage enum definition

### 2. AnswerAssembler Instantiation
- **Import**: `from answer_assembler import AnswerAssembler as ExternalAnswerAssembler` (line 75)
- **Initialization**: In `CanonicalDeterministicOrchestrator.__init__()` (lines 806-809)
  ```python
  self.external_answer_assembler = ExternalAnswerAssembler(
      rubric_path=str(rubric_path),
      decalogo_path=str(decalogo_path)
  )
  ```
- **Timing**: Instantiated after `QuestionnaireEngine`, receives populated `EvidenceRegistry` and `RUBRIC_SCORING.json` weights section

### 3. Answer Assembly Implementation
- **Method**: `_assemble_answers(self, evaluation_inputs: Dict[str, Any])` (lines 946-1041)
- **Key Features**:
  - Loads `RUBRIC_SCORING.json` to extract weights section
  - Creates `RegistryAdapter` to bridge `EvidenceRegistry` with external assembler interface
  - Converts questionnaire evaluation results to expected format
  - Calls `ExternalAnswerAssembler.assemble()` with:
    - Populated `EvidenceRegistry` (adaptor wrapping internal registry)
    - `evaluation_results` containing question scores
  - Logs passage of evidence count and weights count for traceability

### 4. Answer Object Structure
Each assembled answer contains:
- `question_id`: Unique question identifier
- `evidence_ids`: List of source evidence IDs
- `confidence`: Calculated confidence score (0.0-1.0)
- `score`: Raw question score
- `rationale`: Generated reasoning text
- `rubric_weight`: Weight from RUBRIC_SCORING.json weights section
- `scoring_modality`: Scoring type (TYPE_A through TYPE_F)
- `dimension`: Decálogo dimension (D1-D6)
- `supporting_quotes`: Extracted evidence quotes
- `caveats`: Quality warnings

### 5. Evidence Registry Registration
- **Implementation**: Lines 993-1030 in `_assemble_answers()`
- **Provenance Metadata**:
  ```python
  answer_metadata = {
      "question_id": qa['question_id'],
      "dimension": qa.get('dimension', ''),
      "score": qa.get('raw_score', 0.0),
      "evidence_count": qa.get('evidence_count', 0),
      "question_unique_id": qa['question_id'],
      "source_evidence_ids": evidence_ids,  # Links to source evidence
      "rubric_weight": weights.get(qa['question_id'], 0.0),
      "confidence": qa.get('confidence', 0.0),
      "rationale": qa.get('rationale', ''),
      "scoring_modality": qa.get('scoring_modality', 'UNKNOWN'),
      "assembled_by": "AnswerAssembler",
      "assembler_version": "4.0",
      "provenance": {
          "stage": PipelineStage.ANSWER_ASSEMBLY.value,
          "linked_evidence_ids": evidence_ids,
          "question_engine_score": qa.get('raw_score', 0.0),
          "final_confidence": qa.get('confidence', 0.0)
      }
  }
  ```
- **EvidenceEntry Creation**: Each answer registered with:
  - `evidence_id`: `f"answer_{qa['question_id']}"`
  - `stage`: `PipelineStage.ANSWER_ASSEMBLY.value`
  - `content`: Complete question answer object
  - `source_segment_ids`: Links to evidence_ids
  - `confidence`: Assembled confidence score
  - `metadata`: Full provenance structure above

### 6. Artifact Serialization

#### answers_report.json
- **Path**: `artifacts/answers_report.json`
- **Content**: Complete answers collection with:
  - `metadata`: System version, timestamps, config file paths
  - `global_summary`: Overall scoring, bands, statistics
  - `point_summaries`: Aggregated scores by thematic point
  - `question_answers`: Full list of 300 assembled answers
- **Encoding**: Deterministic JSON with `sort_keys=True`, `ensure_ascii=False`
- **Implementation**: Lines 1298-1302

#### answers_sample.json
- **Path**: `artifacts/answers_sample.json`
- **Content**: Sample subset containing:
  - `metadata`: System metadata
  - `global_summary`: Overall statistics
  - `sample_question_answers`: First 10 answers, sorted by `question_id`
- **Encoding**: Deterministic JSON with `sort_keys=True`, `ensure_ascii=False`
- **Implementation**: Lines 1304-1313
- **Sorting**: Sample answers sorted by `question_id` to ensure deterministic ordering

### 7. Flow Runtime Capture

#### flow_runtime.json
- **Path**: `artifacts/flow_runtime.json`
- **Content**: Execution trace with:
  - `stages`: Ordered list of all 15 executed stages including `answers_assembly`
  - `flow_hash`: SHA-256 hash of stage sequence
  - `stage_timestamps`: Deterministically sorted timestamps by stage name
  - `stage_count`: Total stage count (should be 15)
  - `duration_seconds`: Total pipeline execution time
  - `evidence_hash`: Deterministic hash of complete evidence registry
  - `validation`: Flow validation results against canonical order
  - `orchestrator_version`: Pipeline version
  - `errors`: Any stage errors (empty on success)
- **Encoding**: Deterministic JSON with `sort_keys=True`, `ensure_ascii=False`
- **Implementation**: Lines 1312-1318, method `_generate_flow_runtime_metadata()` (lines 1323-1338)
- **Validation**: Stage order validated against `CanonicalFlowValidator.CANONICAL_ORDER`

### 8. Canonical Flow Documentation

#### tools/flow_doc.json
- **Updated**: Added `answers_assembly` as 15th stage
- **Canonical Order**:
  ```json
  {
    "canonical_order": [
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
      "decalogo_evaluation",
      "questionnaire_evaluation",
      "answers_assembly"
    ],
    "flow_hash": "8d6e9c4f2a1b3e7d5c8a9f2e6b4d1a3c7f9e2b8d5a1c4e7f9a2d6b3c8e1f4a7b"
  }
  ```
- **Position**: `answers_assembly` is the 15th and final stage

### 9. Validation and Logging
- **Pre-assembly**: Logs evidence count and weights count passed to assembler
- **Post-assembly**: Logs registration of answers with provenance metadata
- **GATE #4 Check**: Validates coverage of 300 questions
- **Artifact Export**: Logs each artifact export with deterministic encoding confirmation

## Key Design Decisions

1. **External Assembler**: Used `answer_assembler.py` as-is, wrapped `EvidenceRegistry` with adapter
2. **Provenance First**: Each answer includes full provenance linking back to source evidence
3. **Deterministic JSON**: All artifacts use `sort_keys=True` for reproducible serialization
4. **Flow Validation**: Runtime trace validated against canonical order in `tools/flow_doc.json`
5. **Sample Subset**: Provides quick inspection with 10 representative answers, sorted deterministically
6. **Metadata Enrichment**: Answers include assembler version, modality, confidence, and rationale

## Execution Flow

```
Flow #14: questionnaire_evaluation
    ↓
Flow #15: answers_assembly
    ├─ Load RUBRIC_SCORING.json weights
    ├─ Create RegistryAdapter(evidence_registry)
    ├─ Call ExternalAnswerAssembler.assemble()
    ├─ Register 300 answers to EvidenceRegistry with provenance
    └─ Return final_report
    ↓
export_artifacts()
    ├─ answers_report.json (complete, deterministic)
    ├─ answers_sample.json (10 samples, deterministic)
    └─ flow_runtime.json (15 stages, deterministic)
```

## Compliance

- ✅ AnswerAssembler instantiated after questionnaire_engine completes
- ✅ Populated EvidenceRegistry passed to assembler
- ✅ RUBRIC_SCORING.json weights section passed to assembler
- ✅ Answer objects contain evidence_ids, confidence, rationale, score
- ✅ Answers registered back to EvidenceRegistry with provenance
- ✅ answers_report.json serialized with deterministic JSON (sorted keys)
- ✅ answers_sample.json serialized with deterministic JSON (sorted keys)
- ✅ flow_runtime.json captures all 15 stages including answers_assembly
- ✅ Execution order matches canonical sequence in tools/flow_doc.json
