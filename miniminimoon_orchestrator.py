# coding=utf-8
import json
import hashlib
import random
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed



class ThreadSafeLRUCache:
    """Thread-safe LRU cache with TTL."""

    def __init__(self, max_size: int = 128, ttl_seconds: int = 1800):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._lock = threading.RLock()
        self._store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()

    def _evict_if_needed(self):
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def set(self, key: str, value: Any):
        with self._lock:
            now = time.time()
            if key in self._store:
                self._store.pop(key, None)
            self._store[key] = (now, value)
            self._evict_if_needed()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            ts, value = item
            if (time.time() - ts) > self.ttl:
                self._store.pop(key, None)
                return None
            self._store.pop(key)
            self._store[key] = (ts, value)
            return value

    def has(self, key: str) -> bool:
        return self.get(key) is not None


class EmbeddingModelPool:
    """Thread-safe singleton connection pool for embedding model."""
    _instance_lock = threading.Lock()
    _model_instance: Optional[Any] = None

    @classmethod
    def get_model(cls) -> Any:
        if cls._model_instance is not None:
            return cls._model_instance
        with cls._instance_lock:
            if cls._model_instance is None:
                from embedding_model import IndustrialEmbeddingModel as EmbeddingModel
                cls._model_instance = EmbeddingModel()
        return cls._model_instance


@dataclass
class EvidenceEntry:
    evidence_id: str
    stage: str
    content: Any
    source_segment_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_hash(self) -> str:
        content_str = json.dumps(self.content, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:16]


class EvidenceRegistry:
    def __init__(self):
        self._evidence: Dict[str, EvidenceEntry] = {}
        self._stage_index: Dict[str, List[str]] = {}
        self._segment_index: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

    def register(self, entry: EvidenceEntry) -> str:
        with self._lock:
            eid = entry.evidence_id
            if eid in self._evidence:
                self.logger.warning(f"Evidence {eid} already exists, overwriting")
            self._evidence[eid] = entry
            if entry.stage not in self._stage_index:
                self._stage_index[entry.stage] = []
            self._stage_index[entry.stage].append(eid)
            for seg_id in entry.source_segment_ids:
                if seg_id not in self._segment_index:
                    self._segment_index[seg_id] = []
                self._segment_index[seg_id].append(eid)
            return eid

    def get(self, evidence_id: str) -> Optional[EvidenceEntry]:
        return self._evidence.get(evidence_id)

    def get_by_stage(self, stage: str) -> List[EvidenceEntry]:
        eids = self._stage_index.get(stage, [])
        return [self._evidence[eid] for eid in eids]

    def deterministic_hash(self) -> str:
        sorted_eids = sorted(self._evidence.keys())
        hash_inputs = [self._evidence[eid].to_hash() for eid in sorted_eids]
        combined = "|".join(hash_inputs)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def export(self, path: Path):
        data = {
            "evidence_count": len(self._evidence),
            "deterministic_hash": self.deterministic_hash(),
            "evidence": {eid: asdict(entry) for eid, entry in self._evidence.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class RuntimeTracer:
    def __init__(self):
        self.stages: List[str] = []
        self.stage_timestamps: Dict[str, float] = {}
        self.stage_errors: Dict[str, str] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        self.start_time = datetime.utcnow().timestamp()

    def stop(self):
        self.end_time = datetime.utcnow().timestamp()

    def record_stage(self, stage_name: str, success: bool = True, error: str = ""):
        self.stages.append(stage_name)
        self.stage_timestamps[stage_name] = datetime.utcnow().timestamp()
        if not success:
            self.stage_errors[stage_name] = error

    def get_stages(self) -> List[str]:
        return self.stages

    def compute_flow_hash(self) -> str:
        stages_str = "|".join(self.stages)
        return hashlib.sha256(stages_str.encode('utf-8')).hexdigest()

    def export(self) -> dict:
        return {
            "flow_hash": self.compute_flow_hash(),
            "stages": self.stages,
            "stage_count": len(self.stages),
            "stage_timestamps": self.stage_timestamps,
            "errors": self.stage_errors,
            "duration_seconds": (self.end_time - self.start_time) if self.end_time else None
        }


class PipelineStage(Enum):
    SANITIZATION = "sanitization"
    PLAN_PROCESSING = "plan_processing"
    SEGMENTATION = "document_segmentation"
    EMBEDDING = "embedding"
    RESPONSIBILITY = "responsibility_detection"
    CONTRADICTION = "contradiction_detection"
    MONETARY = "monetary_detection"
    FEASIBILITY = "feasibility_scoring"
    CAUSAL = "causal_detection"
    TEORIA = "teoria_cambio"
    DAG = "dag_validation"
    REGISTRY_BUILD = "evidence_registry_build"
    DECALOGO_EVAL = "decalogo_evaluation"
    QUESTIONNAIRE_EVAL = "questionnaire_evaluation"
    ANSWER_ASSEMBLY = "answers_assembly"


class AnswerAssembler:
    def __init__(self, rubric_path: Path, evidence_registry: EvidenceRegistry):
        self.rubric = self._load_rubric(rubric_path)
        self.registry = evidence_registry
        self.logger = logging.getLogger(__name__)
        self.questions = self.rubric.get("questions", {})
        self.weights = self.rubric.get("weights", {})
        self._validate_rubric_coverage()

    def _load_rubric(self, path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            rubric = json.load(f)
        if "questions" not in rubric:
            raise ValueError(f"Rubric missing 'questions' section: {path}")
        if "weights" not in rubric:
            raise ValueError(f"Rubric missing 'weights' section: {path}")
        return rubric

    def _validate_rubric_coverage(self):
        if isinstance(self.questions, list):
            questions = set(q.get("id") for q in self.questions if "id" in q)
        else:
            questions = set(self.questions.keys())

        weights = set(self.weights.keys())
        missing = questions - weights
        extra = weights - questions

        if missing or extra:
            error_parts = []
            if missing:
                sample = sorted(missing)[:10]
                error_parts.append(f"missing weights for {len(missing)} questions: {sample}")
            if extra:
                sample = sorted(extra)[:10]
                error_parts.append(f"extra weights for {len(extra)} non-existent questions: {sample}")
            raise ValueError("Rubric validation FAILED: " + "; ".join(error_parts))

        self.logger.info(f"✓ Rubric validated: {len(questions)} questions with {len(weights)} weights")

    def assemble(self, evidence_registry: Any, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        question_scores = evaluation_results.get("question_scores", [])
        question_answers = []

        for qs in question_scores:
            question_unique_id = qs.get("question_unique_id", "")
            score = qs.get("score", 0.0)

            dimension = ""
            if question_unique_id.startswith("D") and "-" in question_unique_id:
                dimension = question_unique_id.split("-")[0]

            evidence_list = evidence_registry.get_evidence_for_question(question_unique_id)
            evidence_ids = [
                e.metadata.get("evidence_id", "")
                for e in evidence_list
                if hasattr(e, "metadata")
            ]

            weight = self.weights.get(question_unique_id, 0.0)
            if weight == 0.0 and question_unique_id:
                self.logger.warning(f"No rubric weight found for question '{question_unique_id}'")

            confidence = self._calculate_confidence_from_evidence(evidence_list, score)
            reasoning = self._generate_reasoning(dimension, evidence_list, score)
            caveats = self._identify_caveats_from_evidence(evidence_list, score)
            quotes = self._extract_quotes_from_evidence(evidence_list)

            question_answer = {
                "question_id": question_unique_id,
                "dimension": dimension,
                "raw_score": score,
                "rubric_weight": weight,
                "confidence": confidence,
                "evidence_ids": evidence_ids,
                "evidence_count": len(evidence_ids),
                "rationale": reasoning,
                "supporting_quotes": quotes,
                "caveats": caveats,
                "scoring_modality": self._get_scoring_modality(question_unique_id)
            }

            question_answers.append(question_answer)

        total_weight = sum(qa["rubric_weight"] for qa in question_answers)
        weighted_score = sum(qa["raw_score"] * qa["rubric_weight"] for qa in question_answers)

        global_summary = {
            "answered_questions": len(question_answers),
            "total_questions": 300,
            "total_weight": total_weight,
            "weighted_score": weighted_score,
            "average_confidence": (
                sum(qa["confidence"] for qa in question_answers) / len(question_answers)
                if question_answers else 0.0
            )
        }

        return {
            "question_answers": question_answers,
            "global_summary": global_summary
        }

    def _get_scoring_modality(self, question_unique_id: str) -> str:
        if isinstance(self.questions, list):
            for q in self.questions:
                if q.get("id") == question_unique_id:
                    return q.get("scoring_modality", "UNKNOWN")
        elif isinstance(self.questions, dict):
            q = self.questions.get(question_unique_id, {})
            return q.get("scoring_modality", "UNKNOWN")
        return "UNKNOWN"

    def _calculate_confidence_from_evidence(self, evidence_list: List[Any], score: float) -> float:
        if not evidence_list:
            return 0.3

        avg_evidence_conf = sum(
            e.confidence if hasattr(e, "confidence") else 0.5
            for e in evidence_list
        ) / len(evidence_list)

        evidence_factor = min(len(evidence_list) / 3.0, 1.0)
        extremity = abs(score - 0.5) * 2
        extremity_penalty = 0.85 if (extremity > 0.7 and len(evidence_list) < 2) else 1.0
        confidence = avg_evidence_conf * evidence_factor * extremity_penalty
        return round(min(confidence, 1.0), 2)

    def _extract_quotes_from_evidence(self, evidence_list: List[Any], max_quotes: int = 3) -> List[str]:
        quotes = []
        for e in evidence_list[:max_quotes]:
            if hasattr(e, "metadata"):
                text = e.metadata.get("text", "")
                if text:
                    if len(text) > 150:
                        text = text[:147] + "..."
                    quotes.append(text)
        return quotes

    def _identify_caveats_from_evidence(self, evidence_list: List[Any], score: float) -> List[str]:
        caveats = []
        if len(evidence_list) == 0:
            caveats.append("No supporting evidence found")
        elif len(evidence_list) == 1:
            caveats.append("Based on single evidence source")

        low_conf_evidence = [
            e for e in evidence_list
            if hasattr(e, "confidence") and e.confidence < 0.5
        ]
        if low_conf_evidence:
            caveats.append(f"{len(low_conf_evidence)} low-confidence evidence pieces")

        if score > 0.8 and len(evidence_list) < 2:
            caveats.append("High score with limited evidence—verify manually")

        return caveats

    def _generate_reasoning(self, dimension: str, evidence: List[Any], score: float) -> str:
        if not evidence:
            return f"No evidence found for {dimension}. Score reflects absence of required information."

        evidence_types = []
        for e in evidence:
            if hasattr(e, 'stage'):
                evidence_types.append(e.stage)
            elif hasattr(e, 'metadata') and isinstance(e.metadata, dict):
                stage = e.metadata.get('stage', 'unknown')
                evidence_types.append(stage)

        evidence_types = sorted(set(evidence_types))
        evidence_summary = ", ".join(evidence_types[:3]) if evidence_types else "mixed sources"

        if score > 0.7:
            return f"Strong evidence from {evidence_summary} supports high compliance in {dimension}."
        elif score > 0.4:
            return f"Partial evidence from {evidence_summary} indicates moderate compliance in {dimension}."
        else:
            return f"Limited evidence from {evidence_summary} suggests low compliance in {dimension}."


class CanonicalDeterministicOrchestrator:
    VERSION = "2.1.0"

    def __init__(
            self,
            config_dir: Path,
            enable_validation: bool = True,
            flow_doc_path: Optional[Path] = None,
            log_level: str = "INFO",
            intermediate_cache_ttl: int = 900,
            document_cache_ttl: int = 900,
            intermediate_cache_size: int = 64,
            document_cache_size: int = 16,
            enable_document_result_cache: bool = True
    ):
        self.config_dir = Path(config_dir)
        self.enable_validation = enable_validation
        self.flow_doc_path = flow_doc_path
        self.enable_document_result_cache = enable_document_result_cache

        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self._set_deterministic_seeds()
        self.evidence_registry = EvidenceRegistry()

        self.intermediate_cache = ThreadSafeLRUCache(
            max_size=intermediate_cache_size,
            ttl_seconds=intermediate_cache_ttl
        )
        self.document_result_cache = ThreadSafeLRUCache(
            max_size=document_cache_size,
            ttl_seconds=document_cache_ttl
        )

        self._init_pipeline_components()
        self._init_evaluators()

        if self.enable_validation:
            self.runtime_tracer = RuntimeTracer()

        self.warmup_models()
        self.logger.info(f"CanonicalDeterministicOrchestrator {self.VERSION} initialized")

    def _set_deterministic_seeds(self):
        SEED = 42
        random.seed(SEED)
        self.logger.info("Deterministic seeds set")

    def _init_pipeline_components(self):
        from plan_sanitizer import PlanSanitizer
        from plan_processor import PlanProcessor
        from document_segmenter import DocumentSegmenter
        from responsibility_detector import ResponsibilityDetector
        from contradiction_detector import ContradictionDetector
        from monetary_detector import MonetaryDetector
        from feasibility_scorer import FeasibilityScorer
        from causal_pattern_detector import CausalPatternDetector
        from teoria_cambio import TeoriaCambioValidator
        from dag_validation import DAGValidator

        self.plan_sanitizer = PlanSanitizer()
        self.plan_processor = PlanProcessor()
        self.document_segmenter = DocumentSegmenter()
        self.embedding_model = EmbeddingModelPool.get_model()
        self.responsibility_detector = ResponsibilityDetector()
        self.contradiction_detector = ContradictionDetector()
        self.monetary_detector = MonetaryDetector()
        self.feasibility_scorer = FeasibilityScorer()
        self.causal_pattern_detector = CausalPatternDetector()
        self.teoria_cambio_validator = TeoriaCambioValidator()
        self.dag_validator = DAGValidator()

    def _init_evaluators(self):
        from questionnaire_engine import QuestionnaireEngine
        from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado, BUNDLE

        rubric_path = self.config_dir / "RUBRIC_SCORING.json"
        self.questionnaire_engine = QuestionnaireEngine(
            evidence_registry=self.evidence_registry,
            rubric_path=rubric_path
        )
        self.external_answer_assembler = AnswerAssembler(
            rubric_path=rubric_path,
            evidence_registry=self.evidence_registry
        )
        self.decatalogo_extractor = ExtractorEvidenciaIndustrialAvanzado(BUNDLE)

    def warmup_models(self):
        self.logger.info("Warming up models...")
        try:
            model = self.embedding_model
            model.encode(["warmup embedding sentinel"])
            self.logger.info("✅ Embedding model warmed")
        except Exception as e:
            self.logger.warning(f"Embedding warmup failed: {e}")
        self.logger.info("✅ Models warmed successfully")

    def _encode_segments_dynamic(self, segment_texts: List[str]) -> List[Any]:
        if not segment_texts:
            return []
        results: List[Any] = []
        idx = 0
        while idx < len(segment_texts):
            batch_size = 64 if (len(segment_texts) - idx) >= 64 else 32
            batch = segment_texts[idx: idx + batch_size]
            try:
                batch_embeddings = self.embedding_model.encode(batch)
                results.extend(batch_embeddings)
            except Exception as e:
                self.logger.error(f"Embedding batch failed: {e}")
                raise
            idx += batch_size
        return results

    def _parallel_questionnaire_evaluation(self) -> Dict[str, Any]:
        engine = self.questionnaire_engine
        question_ids: List[str] = []

        if hasattr(engine, "get_question_ids"):
            try:
                question_ids = list(engine.get_question_ids())
            except Exception:
                question_ids = []
        if not question_ids and hasattr(engine, "questions"):
            q_attr = getattr(engine, "questions")
            if isinstance(q_attr, dict):
                question_ids = list(q_attr.keys())
            elif isinstance(q_attr, list):
                question_ids = q_attr

        if not question_ids or not hasattr(engine, "evaluate_question"):
            return engine.evaluate()

        self.logger.info(f"Parallel questionnaire evaluation over {len(question_ids)} questions")
        results_map: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {
                executor.submit(engine.evaluate_question, qid): qid
                for qid in question_ids
            }
            for future in as_completed(future_map):
                qid = future_map[future]
                try:
                    res = future.result()
                except Exception as e:
                    self.logger.error(f"Question {qid} evaluation failed: {e}")
                    res = {"question_id": qid, "error": str(e), "score": 0.0}
                results_map[qid] = res

        ordered = [results_map[qid] for qid in sorted(results_map.keys())]
        return {
            "question_results": ordered,
            "question_count": len(ordered),
            "metadata": {"parallel": True, "max_workers": 4}
        }

    def _build_evidence_registry(self, all_inputs: Dict[str, Any]):
        self.logger.info("Building evidence registry...")

        def register_evidence(stage: PipelineStage, items: List[Any], id_prefix: str):
            if not isinstance(items, list):
                return
            for item in items:
                try:
                    item_str = json.dumps(item, sort_keys=True, default=str)
                    evidence_id = f"{id_prefix}_{hashlib.sha1(item_str.encode()).hexdigest()[:10]}"
                    entry = EvidenceEntry(
                        evidence_id=evidence_id,
                        stage=stage.value,
                        content=item,
                        source_segment_ids=[],
                        confidence=item.get('confidence', 0.8) if isinstance(item, dict) else 0.8
                    )
                    self.evidence_registry.register(entry)
                except (TypeError, AttributeError) as e:
                    self.logger.warning(f"Could not process item in stage {stage.value}: {e}")

        register_evidence(PipelineStage.RESPONSIBILITY, all_inputs.get('responsibilities', []), 'resp')
        register_evidence(PipelineStage.MONETARY, all_inputs.get('monetary', []), 'money')

        feasibility_report = all_inputs.get('feasibility')
        if isinstance(feasibility_report, dict):
            register_evidence(PipelineStage.FEASIBILITY, feasibility_report.get('indicators', []), 'feas')

        self.logger.info(f"Evidence registry built with {len(self.evidence_registry._evidence)} entries")
        return {"status": "built", "entries": len(self.evidence_registry._evidence)}

    def _execute_decalogo_evaluation(self, evidence_registry: EvidenceRegistry) -> Dict[str, Any]:
        self.logger.info("Executing Decálogo evaluation...")
        evaluation = self.decatalogo_extractor.evaluate_from_evidence(evidence_registry)
        self.logger.info("Decálogo evaluation completed")
        return evaluation

    def _assemble_answers(self, evaluation_inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Assembling final answers report...")
        questionnaire_eval = evaluation_inputs.get('questionnaire_eval', {})

        rubric_path = self.config_dir / "RUBRIC_SCORING.json"
        with open(rubric_path, 'r', encoding='utf-8') as f:
            rubric = json.load(f)
        weights = rubric.get("weights", {})

        question_scores = []
        if "question_results" in questionnaire_eval:
            for result in questionnaire_eval["question_results"]:
                raw_id = result.get("question_id", "")
                standardized_id = self._standardize_question_id(raw_id)

                if standardized_id not in weights:
                    self.logger.warning(f"Question ID '{standardized_id}' not found in rubric weights")

                question_scores.append({
                    "question_unique_id": standardized_id,
                    "score": result.get("score", 0.0)
                })

        evaluation_results = {"question_scores": question_scores}

        class RegistryAdapter:
            def __init__(self, registry: EvidenceRegistry):
                self.registry = registry

            def get_evidence_for_question(self, question_unique_id: str):
                matching = []
                for entry in self.registry._evidence.values():
                    stored_id = entry.metadata.get("question_unique_id", "")
                    if self._standardize_question_id(stored_id) == question_unique_id:
                        from collections import namedtuple
                        MockEvidence = namedtuple('MockEvidence', ['confidence', 'metadata'])
                        matching.append(MockEvidence(
                            confidence=entry.confidence,
                            metadata=entry.metadata
                        ))
                return matching

            def _standardize_question_id(self, raw_id: str) -> str:
                import re
                if not raw_id:
                    return raw_id
                if re.match(r'^D\d+-Q\d+$', raw_id):
                    return raw_id
                dim_match = re.search(r'D(\d+)', raw_id, re.IGNORECASE)
                q_match = re.search(r'Q(\d+)', raw_id, re.IGNORECASE)
                if dim_match and q_match:
                    return f"D{dim_match.group(1)}-Q{q_match.group(1)}"
                return raw_id

        registry_adapter = RegistryAdapter(self.evidence_registry)
        final_report = self.external_answer_assembler.assemble(registry_adapter, evaluation_results)

        total_questions = final_report.get("global_summary", {}).get("answered_questions", 0)
        if total_questions < 300:
            self.logger.warning(f"Coverage is {total_questions}/300 questions")
        else:
            self.logger.info(f"✓ Coverage is {total_questions}/300 questions")

        return final_report

    def _standardize_question_id(self, raw_id: str) -> str:
        import re
        if not raw_id:
            return raw_id
        if re.match(r'^D\d+-Q\d+$', raw_id):
            return raw_id
        dim_match = re.search(r'D(\d+)', raw_id, re.IGNORECASE)
        q_match = re.search(r'Q(\d+)', raw_id, re.IGNORECASE)
        if dim_match and q_match:
            return f"D{dim_match.group(1)}-Q{q_match.group(1)}"
        return raw_id

    def process_plan_deterministic(self, plan_path: str) -> Dict[str, Any]:
        plan_path = Path(plan_path)
        if not plan_path.exists():
            raise FileNotFoundError(f"Plan not found: {plan_path}")

        self.logger.info(f"Starting canonical pipeline for: {plan_path.name}")
        start_time = datetime.utcnow()

        if self.enable_validation:
            self.runtime_tracer.start()

        results = {
            "plan_path": str(plan_path),
            "orchestrator_version": self.VERSION,
            "start_time": start_time.isoformat(),
            "stages_completed": [],
            "evaluations": {},
            "runtime_stats": {}
        }

        with open(plan_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        sanitized_text = self._run_stage(
            PipelineStage.SANITIZATION,
            lambda: self.plan_sanitizer.sanitize_text(raw_text),
            results["stages_completed"]
        )

        doc_hash = hashlib.sha256(sanitized_text.encode('utf-8')).hexdigest()
        results["document_hash"] = doc_hash

        cached_full = self.document_result_cache.get(
            f"docres:{doc_hash}") if self.enable_document_result_cache else None
        if cached_full:
            self.logger.info("Document-level cached result HIT")
            if self.enable_validation:
                for stage in PipelineStage:
                    self.runtime_tracer.record_stage(stage.value, success=True)
                self.runtime_tracer.stop()
            cached_full["cache_hit"] = True
            return cached_full

        _ = self._run_stage(
            PipelineStage.PLAN_PROCESSING,
            lambda: self.plan_processor.process(sanitized_text),
            results["stages_completed"]
        )

        cache_key_segments = f"{doc_hash}:segments"
        segments = self.intermediate_cache.get(cache_key_segments)
        if segments:
            self.logger.info("Segments cache HIT")
            self._run_stage(PipelineStage.SEGMENTATION, lambda: segments, results["stages_completed"])
        else:
            segments = self._run_stage(
                PipelineStage.SEGMENTATION,
                lambda: self.document_segmenter.segment(sanitized_text),
                results["stages_completed"]
            )
            self.intermediate_cache.set(cache_key_segments, segments)

        segment_texts = [getattr(s, "text", str(s)) for s in segments]

        cache_key_embeddings = f"{doc_hash}:embeddings"
        embeddings = self.intermediate_cache.get(cache_key_embeddings)
        if embeddings:
            self.logger.info("Embeddings cache HIT")
            self._run_stage(PipelineStage.EMBEDDING, lambda: embeddings, results["stages_completed"])
        else:
            embeddings = self._run_stage(
                PipelineStage.EMBEDDING,
                lambda: self._encode_segments_dynamic(segment_texts),
                results["stages_completed"]
            )
            self.intermediate_cache.set(cache_key_embeddings, embeddings)

        responsibilities = self._run_stage(
            PipelineStage.RESPONSIBILITY,
            lambda: self.responsibility_detector.detect_entities(sanitized_text),
            results["stages_completed"]
        )

        contradictions = self._run_stage(
            PipelineStage.CONTRADICTION,
            lambda: self.contradiction_detector.detect_contradictions(sanitized_text),
            results["stages_completed"]
        )

        monetary = self._run_stage(
            PipelineStage.MONETARY,
            lambda: self.monetary_detector.detect(sanitized_text, plan_name=plan_path.name),
            results["stages_completed"]
        )

        feasibility = self._run_stage(
            PipelineStage.FEASIBILITY,
            lambda: self.feasibility_scorer.evaluate_plan_feasibility(sanitized_text),
            results["stages_completed"]
        )

        causal_patterns = self._run_stage(
            PipelineStage.CAUSAL,
            lambda: self.causal_pattern_detector.detect_patterns(sanitized_text, plan_name=plan_path.name),
            results["stages_completed"]
        )

        all_detector_outputs = {
            "segments": segments,
            "embeddings": embeddings,
            "responsibilities": responsibilities,
            "contradictions": contradictions,
            "monetary": monetary,
            "feasibility": feasibility,
            "causal_patterns": causal_patterns
        }

        self._run_stage(
            PipelineStage.REGISTRY_BUILD,
            lambda: self._build_evidence_registry(all_detector_outputs),
            results["stages_completed"]
        )

        decalogo_eval = self._run_stage(
            PipelineStage.DECALOGO_EVAL,
            lambda: self._execute_decalogo_evaluation(self.evidence_registry),
            results["stages_completed"]
        )
        results["evaluations"]["decalogo"] = decalogo_eval

        questionnaire_eval = self._run_stage(
            PipelineStage.QUESTIONNAIRE_EVAL,
            lambda: self._parallel_questionnaire_evaluation(),
            results["stages_completed"]
        )
        results["evaluations"]["questionnaire"] = questionnaire_eval

        answer_assembly_input = {
            "decalogo_eval": decalogo_eval,
            "questionnaire_eval": questionnaire_eval
        }
        answers_report = self._run_stage(
            PipelineStage.ANSWER_ASSEMBLY,
            lambda: self._assemble_answers(answer_assembly_input),
            results["stages_completed"]
        )
        results["evaluations"]["answers_report"] = answers_report

        if self.enable_validation:
            self.runtime_tracer.stop()

        results["evidence_hash"] = self.evidence_registry.deterministic_hash()

        end_time = datetime.utcnow()
        results["end_time"] = end_time.isoformat()
        results["runtime_stats"] = {
            "duration_seconds": (end_time - start_time).total_seconds(),
            "stages_count": len(results["stages_completed"]),
            "evidence_entries": len(self.evidence_registry._evidence)
        }

        self.logger.info(f"Pipeline completed in {results['runtime_stats']['duration_seconds']:.1f}s")

        if self.enable_document_result_cache:
            self.document_result_cache.set(f"docres:{doc_hash}", results)

        return results

    def _run_stage(self, stage: PipelineStage, func: Callable, stages_list: List[str]) -> Any:
        stage_name = stage.value
        self.logger.info(f"→ {stage_name}")
        try:
            result = func()
            if self.enable_validation:
                self.runtime_tracer.record_stage(stage_name, success=True)
            stages_list.append(stage_name)
            return result
        except Exception as e:
            self.logger.error(f"Stage {stage_name} FAILED: {e}")
            if self.enable_validation:
                self.runtime_tracer.record_stage(stage_name, success=False, error=str(e))
            raise

    def export_artifacts(self, output_dir: Path, pipeline_results: Dict[str, Any] = None):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        evidence_path = output_dir / "evidence_registry.json"
        self.evidence_registry.export(evidence_path)

        if pipeline_results and "evaluations" in pipeline_results:
            answers_report = pipeline_results["evaluations"].get("answers_report")
            if answers_report:
                answers_path = output_dir / "answers_report.json"
                with open(answers_path, 'w', encoding='utf-8') as f:
                    json.dump(answers_report, f, indent=2, ensure_ascii=False, sort_keys=True)

                sample_path = output_dir / "answers_sample.json"
                answers_sample = {
                    "metadata": answers_report.get("metadata", {}),
                    "global_summary": answers_report.get("global_summary", {}),
                    "sample_question_answers": sorted(
                        answers_report.get("question_answers", [])[:10],
                        key=lambda x: x.get("question_id", "")
                    )
                }
                with open(sample_path, 'w', encoding='utf-8') as f:
                    json.dump(answers_sample, f, indent=2, ensure_ascii=False, sort_keys=True)

        self.logger.info(f"Artifacts exported to: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Canonical Deterministic Orchestrator")
    parser.add_argument("--plan", "-p", required=True, help="Path to the plan file to evaluate")
    parser.add_argument("--config-dir", "-c", default="config", help="Configuration directory")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--log-level", default="INFO", help="Set logging level")

    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        orchestrator = CanonicalDeterministicOrchestrator(config_dir=config_dir)
        results = orchestrator.process_plan_deterministic(args.plan)
        orchestrator.export_artifacts(output_dir, results)
        print(f"Evaluation completed successfully. Results saved to {output_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()