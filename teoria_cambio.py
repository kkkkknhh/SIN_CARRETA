# coding=utf-8
"""
INDUSTRIAL-GRADE TEORÍA DE CAMBIO v5.0.0 (self-contained, P–D–Q canonical)
- Deterministic, thread-safe, dependency-graceful.
- Embeddings pluggables: SentenceTransformer → TF-IDF fallback.
- Extractor híbrido: spaCy (si existe) → regex fallback.
- Evidencia con IDs canónicos P{1..10}-D{1..6}-Q{k} y rubric_key D#-Q#.
- Interfaz drop-in para el orquestador: TeoriaCambioValidator.verificar_marco_logico_completo(...)

Compatibilidad:
- El orquestador invoca: from teoria_cambio import TeoriaCambioValidator
- Y llama: TeoriaCambioValidator().verificar_marco_logico_completo(segments)
"""

from __future__ import annotations
import hashlib, json, logging, re, sys, threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np

# -----------------------------
# Configuración y logging
# -----------------------------
SEED = 42
np.random.seed(SEED)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("teoria_cambio")

# -----------------------------
# Convenciones y constantes
# -----------------------------
RUBRIC_HINTS: Dict[str, List[str]] = {
    "P1": ["violencias basadas en género", "participación política", "autonomía económica"],
    "P2": ["homicidios", "desplazamiento forzado", "economías ilegales"],
    "P3": ["deforestación", "calidad del agua", "emisiones GEI", "PNACC"],
    "P4": ["educación", "salud", "WASH", "vivienda", "empleo"],
    "P5": ["RUV", "reparación integral", "memoria", "PAO"],
    "P6": ["cobertura escolar", "embarazo adolescente", "trabajo infantil", "SRPA"],
    "P7": ["catastro", "formalización", "restitución", "consulta previa"],
    "P8": ["amenazas", "homicidios de líderes", "UNP", "participación ciudadana"],
    "P9": ["hacinamiento", "salud intramural", "INPEC", "defensoría"],
    "P10": ["flujos en tránsito", "regularización", "ACNUR", "antixenofobia"],
}
DIMENSIONS = ("D1", "D2", "D3", "D4", "D5", "D6")
Q_PER_DIM = 5

CAUSAL_VERBS = {
    "generar", "producir", "crear", "entregar",
    "lograr", "alcanzar", "conseguir", "mejorar", "reducir", "aumentar",
    "impactar", "transformar", "cambiar", "afectar",
}

# -----------------------------
# Validadores de IDs canónicos
# -----------------------------
_P_PAT = re.compile(r"^P(10|[1-9])$")
_D_PAT = re.compile(r"^D[1-6]$")
_Q_PAT = re.compile(r"^Q[1-9]\d*$")

def assert_policy(pid: str) -> None:
    if not _P_PAT.match(pid): raise ValueError(f"Policy ID inválido (P1..P10): {pid}")

def assert_dim(did: str) -> None:
    if not _D_PAT.match(did): raise ValueError(f"Dimension ID inválido (D1..D6): {did}")

def assert_q(qid: str) -> None:
    if not _Q_PAT.match(qid): raise ValueError(f"Question ID inválido (Q1..): {qid}")

def make_question_uid(policy: str, dim: str, q: int) -> str:
    pid = policy if policy.startswith("P") else f"P{policy}"
    did = dim if dim.startswith("D") else f"D{dim}"
    assert_policy(pid); assert_dim(did); assert_q(f"Q{q}")
    return f"{pid}-{did}-Q{q}"

def to_rubric_key(question_uid: str) -> str:
    m_dim = re.search(r"(D[1-6])", question_uid)
    m_q   = re.search(r"(Q[0-9]+)", question_uid)
    if not (m_dim and m_q): raise ValueError(f"UID no estandarizable: {question_uid}")
    return f"{m_dim.group(1)}-{m_q.group(1)}"

# -----------------------------
# Utilidades deterministas
# -----------------------------
class DeterministicHasher:
    @staticmethod
    def canonical(obj: Any) -> bytes:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    @classmethod
    def sha256(cls, obj: Any) -> str:
        return hashlib.sha256(cls.canonical(obj)).hexdigest()

class ThreadSafeLRUCache:
    def __init__(self, capacity: int = 512):
        self.capacity = capacity; self._store: Dict[Any, Any] = {}; self._order: List[Any] = []; self._lock = threading.RLock()
    def get(self, key: Any, default=None):
        with self._lock:
            if key in self._store:
                if key in self._order: self._order.remove(key)
                self._order.append(key); return self._store[key]
            return default
    def set(self, key: Any, value: Any):
        with self._lock:
            if key in self._store: self._order.remove(key)
            elif len(self._store) >= self.capacity:
                oldest = self._order.pop(0); self._store.pop(oldest, None)
            self._store[key] = value; self._order.append(key)

# -----------------------------
# Embeddings (estrategia pluggable)
# -----------------------------
class EmbeddingBackend:
    def encode(self, texts: Sequence[str]) -> np.ndarray: raise NotImplementedError
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return a_n @ b_n.T

class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts)))

class TfidfBackend(EmbeddingBackend):
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), norm="l2")
        seed = [" ".join(v) for v in RUBRIC_HINTS.values()]
        self.vec.fit(list(seed)+["educación salud empleo vivienda agua infancia seguridad ambiente género víctimas indicadores base"])
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.vec.transform(list(texts)).astype(np.float64).toarray()

def build_embedding_backend() -> EmbeddingBackend:
    try:
        import sentence_transformers  # noqa: F401
        log.info("Embeddings: SentenceTransformer"); return SentenceTransformerBackend()
    except Exception:
        log.info("Embeddings: TF-IDF fallback"); from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        return TfidfBackend()

# -----------------------------
# Extractores causales
# -----------------------------
@dataclass
class CausalElement:
    id: str; text: str; element_type: str; policy_area: str; confidence: float = 1.0

class CausalExtractor:  # iface
    def extract(self, text_block: str) -> List[CausalElement]: raise NotImplementedError

class SpacyExtractor(CausalExtractor):
    def __init__(self):
        import spacy
        self.nlp = None
        for name in ("es_core_news_md","es_core_news_sm","es_core_news_lg"):
            try:
                self.nlp = spacy.load(name, disable=["ner"]); break
            except Exception: pass
        if not self.nlp: raise RuntimeError("spaCy no disponible")
    def extract(self, text_block: str) -> List[CausalElement]:
        doc = self.nlp(text_block); elements: List[CausalElement] = []; k=0
        for sent in doc.sents:
            for t in sent:
                if t.pos_ == "VERB" and t.lemma_.lower() in CAUSAL_VERBS:
                    effect=None
                    for ch in t.children:
                        if ch.dep_ in ("obj","dobj","xcomp","ccomp","attr"): effect = ch.subtree.text; break
                    if not effect: effect = sent.text
                    elements.append(CausalElement(id=f"elem_{k}", text=effect.strip(),
                                             element_type=classify_by_verb(t.lemma_.lower()),
                                             policy_area="P4", confidence=0.87)); k+=1
        return elements

class RegexExtractor(CausalExtractor):
    _PAT = re.compile(r"\b(?:generar|producir|crear|entregar|lograr|alcanzar|conseguir|mejorar|reducir|aumentar|impactar|transformar|cambiar|afectar)\b\s+([^\.;]+)", re.IGNORECASE)
    def extract(self, text_block: str) -> List[CausalElement]:
        elements: List[CausalElement] = []
        for k, m in enumerate(self._PAT.finditer(text_block)):
            verb = m.group(0).split()[0].lower(); effect = m.group(1).strip()
            elements.append(CausalElement(id=f"fallback_{k}", text=effect,
                                     element_type=classify_by_verb(verb),
                                     policy_area="P4", confidence=0.6))
        return elements

def build_extractor() -> CausalExtractor:
    try:
        import spacy  # noqa: F401
        return SpacyExtractor()
    except Exception:
        log.info("Extractor: regex fallback"); return RegexExtractor()

# -----------------------------
# Auxiliares
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+")
def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and len(s.strip()) > 20]
    return sents or ([text.strip()] if text.strip() else [])
def classify_by_verb(v: str) -> str:
    if v in {"generar","producir","crear","entregar"}: return "output"
    if v in {"lograr","alcanzar","conseguir","mejorar","reducir","aumentar"}: return "outcome"
    if v in {"impactar","transformar","cambiar","afectar"}: return "impact"
    return "output"

# -----------------------------
# Tipos de evidencia / reporte
# -----------------------------
@dataclass
class EvidenceEntry:
    evidence_id: str
    question_unique_id: str  # P?-D?-Q?
    content: Dict[str, Any]
    confidence: float
    stage: str = "teoria_cambio"

@dataclass
class TocReport:
    toc_graph: Dict[str, Any]
    industrial_validation: Dict[str, Any]
    immutability: Dict[str, Any]

# -----------------------------
# Motor principal
# -----------------------------
class TeoriaCambioIndustrial:
    def __init__(self, relevance_threshold: float = 0.35):
        self.embed = build_embedding_backend()
        self.extractor = build_extractor()
        self.relevance_threshold = relevance_threshold
        self._cache = ThreadSafeLRUCache(1024)
        self._policy_hint_embs = self._precompute_hint_embeddings()
        self._prompt_embs = self._precompute_prompt_embeddings()

    # precómputos
    def _precompute_hint_embeddings(self) -> Dict[str, np.ndarray]:
        hints_flat = {p: " ".join(hints) for p, hints in RUBRIC_HINTS.items()}
        E = self.embed.encode(list(hints_flat.values()))
        return dict(zip(hints_flat.keys(), E))
    def _precompute_prompt_embeddings(self) -> Dict[str, np.ndarray]:
        prompts: Dict[str, str] = {}
        for p in RUBRIC_HINTS.keys():
            for d in DIMENSIONS:
                for q in range(1, Q_PER_DIM+1):
                    uid = make_question_uid(p, d, q)
                    prompts[uid] = f"{p} {d} pregunta {q}"
        E = self.embed.encode(list(prompts.values()))
        return dict(zip(prompts.keys(), E))

    # segmentación P#
    def segment_text_by_policy(self, text: str) -> Dict[str, List[str]]:
        sentences = split_sentences(text)
        if not sentences: return {"P4":[text]}
        S = self.embed.encode(sentences)
        keys = list(self._policy_hint_embs.keys())
        P = np.vstack([self._policy_hint_embs[k] for k in keys])
        sims = self.embed.cosine(S, P)
        active = {p for p, col in zip(keys, sims.T) if col.max() >= self.relevance_threshold}
        assigns = sims.argmax(axis=1)
        segs: Dict[str, List[str]] = {}
        for sent, idx in zip(sentences, assigns):
            pol = keys[idx]
            if pol in active: segs.setdefault(pol, []).append(sent)
        return segs or {"P4": sentences}

    def _extract_for_policy(self, block: str, policy: str) -> List[CausalElement]:
        assert_policy(policy)
        key = ("extract", DeterministicHasher.sha256({"t": block, "p": policy}))
        cached = self._cache.get(key)
        if cached is not None: return cached
        elems = self.extractor.extract(block)
        for e in elems: e.policy_area = policy
        self._cache.set(key, elems); return elems

    def _score_block(self, policy: str, elems: List[CausalElement]) -> List[EvidenceEntry]:
        evs: List[EvidenceEntry] = []; texts = [e.text for e in elems]
        E = self.embed.encode(texts) if elems else np.zeros((0,1))
        for d in DIMENSIONS:
            for q in range(1, Q_PER_DIM+1):
                uid = make_question_uid(policy, d, q)
                p = self._prompt_embs[uid]
                if elems:
                    sims = self.embed.cosine(p.reshape(1,-1), E)[0]
                    score = float(np.max(sims)); top_idx = int(np.argmax(sims)); top_txt = elems[top_idx].text
                else:
                    score = 0.0; top_txt = None
                evs.append(EvidenceEntry(
                    evidence_id=f"toc_{uid}",
                    question_unique_id=uid,
                    content={
                        "policy": policy, "dimension": d, "question": q,
                        "score": score, "elements_found": len(elems),
                        "top_example": top_txt, "rubric_key": to_rubric_key(uid),
                    },
                    confidence=score))
        return evs

    # API principal (acepta texto o lista de segmentos)
    def verificar_marco_logico_completo(self, text_or_segments: Any) -> Dict[str, Any]:
        if isinstance(text_or_segments, str):
            corpus_text = text_or_segments
        elif isinstance(text_or_segments, list):
            # Segment objects o strings; extraemos .text si existe
            parts = []
            for s in text_or_segments:
                parts.append(getattr(s, "text", str(s)))
            corpus_text = " ".join(parts)
        else:
            corpus_text = str(text_or_segments)

        segments = self.segment_text_by_policy(corpus_text)
        all_evidence: List[EvidenceEntry] = []
        for policy, sents in segments.items():
            block = ". ".join(sents)
            elems = self._extract_for_policy(block, policy)
            all_evidence.extend(self._score_block(policy, elems))

        # sanity de UID
        for ev in all_evidence:
            if not re.match(r"^P(10|[1-9])\-D[1-6]\-Q[0-9]+$", ev.question_unique_id):
                raise AssertionError(f"UID no canónico: {ev.question_unique_id}")

        metrics_payload = [asdict(e) for e in all_evidence]
        toc_graph = {"policy_areas": list(segments.keys())}
        summary = {"success": True, "counts": {"policies": len(segments), "evidences": len(all_evidence)}, "metrics": metrics_payload}
        immut = {"version":"5.0.0","timestamp_utc": datetime.now(timezone.utc).isoformat(),
                 "hash": DeterministicHasher.sha256({"toc_graph": toc_graph, "summary": summary})}

        # El orquestador espera un dict (no un dataclass) aquí.
        return {"toc_graph": toc_graph, "industrial_validation": summary, "immutability": immut}

# Fachada esperada por el orquestador
TeoriaCambioValidator = TeoriaCambioIndustrial

# CLI de prueba manual
if __name__ == "__main__":
    test_text = " ".join(sys.argv[1:]).strip() or (
        "El plan generará empleo juvenil y mejorará la cobertura escolar para reducir el embarazo adolescente; "
        "además, impactará la calidad del agua y reducirá la deforestación mediante incentivos.")
    tc = TeoriaCambioValidator()
    result = tc.verificar_marco_logico_completo(test_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
