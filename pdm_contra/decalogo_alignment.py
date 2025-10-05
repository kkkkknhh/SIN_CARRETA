"""Herramientas para normalizar y alinear los decálogos en formato canónico.

Este módulo implementa una alineación determinista basada en el esquema
común definido para el ecosistema PDM. Dado que las fuentes originales
presentan daños estructurales (JSON truncados, signos residuales de parches
sin aplicar y faltantes de información), la estrategia se centra en:

* Recuperar toda la evidencia disponible de los archivos de entrada.
* Normalizar cadenas usando Unicode NFKC, espacios y tildes estandarizadas.
* Construir una estructura canónica de clusters → puntos → preguntas.
* Generar placeholders explícitos con `status` = "evidencia_insuficiente"
  cuando la fuente carece de datos verificables.
* Producir un crosswalk 3-vías y reportes de auditoría.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

AUDIT_SEPARATOR = "\n---\n"


@dataclass
class AuditIssue:
    """Representa una observación detectada durante la auditoría."""

    source: str
    category: str
    message: str

    def as_markdown(self) -> str:
        return f"* **{self.source}** · _{self.category}_ · {self.message}"


@dataclass
class QuestionSpec:
    q_id: str
    q_code: str
    q_label: str
    aliases: List[str]
    refs: List[str]
    q_type: str = "evidence"
    q_weight: float = 1.0
    status: str = ""

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "q_id": self.q_id,
            "q_code": self.q_code,
            "q_label": self.q_label,
            "q_type": self.q_type,
            "q_weight": self.q_weight,
        }
        if self.aliases:
            payload["aliases"] = self.aliases
        if self.refs:
            payload["refs"] = self.refs
        if self.status:
            payload["status"] = self.status
        return payload


@dataclass
class PointSpec:
    point_id: str
    point_code: str
    point_label: str
    point_order: int
    questions: List[QuestionSpec]
    aliases: List[str] = field(default_factory=list)
    status: str = ""

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "point_id": self.point_id,
            "point_code": self.point_code,
            "point_label": self.point_label,
            "point_order": self.point_order,
            "questions": [q.to_dict() for q in self.questions],
        }
        if self.aliases:
            payload["aliases"] = self.aliases
        if self.status:
            payload["status"] = self.status
        return payload


@dataclass
class ClusterSpec:
    cluster_id: str
    cluster_code: str
    cluster_label: str
    cluster_order: int
    points: List[PointSpec]
    aliases: List[str] = field(default_factory=list)
    status: str = ""

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "cluster_id": self.cluster_id,
            "cluster_code": self.cluster_code,
            "cluster_label": self.cluster_label,
            "cluster_order": self.cluster_order,
            "points": [p.to_dict() for p in self.points],
        }
        if self.aliases:
            payload["aliases"] = self.aliases
        if self.status:
            payload["status"] = self.status
        return payload


def _nfkc_trim(text: str) -> str:
    normal = unicodedata.normalize("NFKC", text or "")
    normal = normal.replace("\u00a0", " ")
    normal = re.sub(r"\s+", " ", normal)
    return normal.strip()


def _sanitize_code(prefix: str, raw: str, fallback_index: int) -> str:
    slug = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if not slug:
        slug = f"{prefix}{fallback_index:02d}"
    return f"{prefix}-{slug}"


@dataclass
class CanonicalBundle:
    clusters: List[ClusterSpec]
    audit: List[AuditIssue]
    crosswalk: Dict[str, List[Dict[str, str]]]


def _clean_json_text(path: Path, add_braces: bool = False) -> str:
    raw = path.read_text(encoding="utf-8")
    text = raw.strip()
    if add_braces and not text.startswith("{"):
        text = "{" + text
    if add_braces and not text.endswith("}"):
        text = text + "}"
    return text


def _load_decalogo_full(path: Path, audit: List[AuditIssue]) -> Dict[str, object]:
    text = _clean_json_text(path, add_braces=True)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        audit.append(
            AuditIssue(
                source="decalogo-full",
                category="json_parse_error",
                message=f"No fue posible parsear {path.name}: {exc}",
            )
        )
        raise
    return data


def _load_decalogo_industrial(
        path: Path, audit: List[AuditIssue]
) -> List[Dict[str, object]]:
    raw = path.read_text(encoding="utf-8")
    cleaned_lines = []
    for line in raw.splitlines():
        cleaned_lines.append(line.lstrip("+ "))
    cleaned_text = "\n".join(cleaned_lines).strip()
    try:
        data = json.loads(cleaned_text)
        if not isinstance(data, list):
            raise TypeError("El decálogo industrial debe ser una lista")
        return data
    except Exception as exc:  # pragma: no cover - errores severos
        audit.append(
            AuditIssue(
                source="decalogo-industrial",
                category="json_parse_error",
                message=f"No fue posible parsear {path.name}: {exc}",
            )
        )
        raise


def _load_dnp_standards(
        path: Path, audit: List[AuditIssue]
) -> Optional[Dict[str, object]]:
    raw = path.read_text(encoding="utf-8")
    trimmed = raw.strip()
    # El archivo está truncado; eliminamos la sección incompleta de emergent themes.
    sentinel = '    "circular_economy": {'
    if sentinel in trimmed:
        trimmed = trimmed.split(sentinel)[0].rstrip()
        trimmed += "\n  }\n}\n"
        audit.append(
            AuditIssue(
                source="dnp-standards",
                category="truncated_section",
                message="Se removió la sección 'circular_economy' por datos incompletos.",
            )
        )
    try:
        data = json.loads(trimmed)
    except json.JSONDecodeError as exc:
        audit.append(
            AuditIssue(
                source="dnp-standards",
                category="json_parse_error",
                message=f"No fue posible parsear {path.name}: {exc}",
            )
        )
        return None
    return data


def _group_questions_by_point(
        full_data: Dict[str, object], audit: List[AuditIssue]
) -> Dict[str, Dict[str, object]]:
    questions = full_data.get("questions")
    if not isinstance(questions, list):
        audit.append(
            AuditIssue(
                source="decalogo-full",
                category="structure_error",
                message="Campo 'questions' ausente o mal tipificado.",
            )
        )
        return {}
    grouped: Dict[str, Dict[str, object]] = {}
    for entry in questions:
        point_code = _nfkc_trim(str(entry.get("point_code", "")))
        point_title = (
                _nfkc_trim(str(entry.get("point_title", ""))
                           ) or "evidencia_insuficiente"
        )
        if point_code not in grouped:
            grouped[point_code] = {
                "title": point_title,
                "questions": [],
            }
        grouped[point_code]["questions"].append(entry)
    return grouped


def _build_canonical_clusters(
        grouped: Dict[str, Dict[str, object]],
) -> List[ClusterSpec]:
    clusters: List[ClusterSpec] = []
    for idx, (point_code, payload) in enumerate(
            sorted(grouped.items(), key=lambda kv: kv[0])
    ):
        order = idx
        label = _nfkc_trim(payload["title"]) or "evidencia_insuficiente"
        cluster_code = _sanitize_code("CL", point_code, idx + 1)
        point_code_canonical = _sanitize_code("PT", point_code, idx + 1)
        cluster_id = f"cluster_{idx + 1:02d}"
        point_id = f"point_{idx + 1:02d}"
        questions_specs: List[QuestionSpec] = []
        for q_idx, entry in enumerate(
                sorted(payload["questions"], key=lambda e: str(e.get("id", "")))
        ):
            raw_id = _nfkc_trim(
                str(entry.get("id", f"Q{idx + 1:02d}{q_idx + 1:02d}")))
            q_code = _sanitize_code(
                "Q", f"{raw_id}-{idx + 1}-{q_idx + 1}", q_idx + 1)
            prompt = _nfkc_trim(
                str(entry.get("prompt", "evidencia_insuficiente")))
            hints = entry.get("hints") or []
            if not isinstance(hints, list):
                hints = []
            aliases = sorted(
                {_nfkc_trim(raw_id)} | {_nfkc_trim(h)
                                        for h in hints if _nfkc_trim(h)}
            )
            refs = [hint for hint in (_nfkc_trim(h) for h in hints) if hint]
            questions_specs.append(
                QuestionSpec(
                    q_id=raw_id,
                    q_code=q_code,
                    q_label=prompt,
                    aliases=aliases,
                    refs=refs,
                    status="evidencia_insuficiente_tipo",
                )
            )
        clusters.append(
            ClusterSpec(
                cluster_id=cluster_id,
                cluster_code=cluster_code,
                cluster_label=label,
                cluster_order=order,
                points=[
                    PointSpec(
                        point_id=point_id,
                        point_code=point_code_canonical,
                        point_label=label,
                        point_order=0,
                        questions=questions_specs,
                        aliases=[label],
                    )
                ],
                aliases=[label],
            )
        )
    return clusters


def _placeholder_clusters_from_canonical(
        canonical: List[ClusterSpec],
        domain_prefix: str,
        status_suffix: str,
) -> List[ClusterSpec]:
    placeholders: List[ClusterSpec] = []
    for cluster in canonical:
        order = cluster.cluster_order
        cluster_code = _sanitize_code(
            "CL", f"{domain_prefix}{order + 1:02d}", order + 1
        )
        point_code = _sanitize_code(
            "PT", f"{domain_prefix}{order + 1:02d}", order + 1)
        placeholder_question = QuestionSpec(
            q_id=f"{domain_prefix.lower()}_q_{order + 1:02d}",
            q_code=_sanitize_code(
                "Q", f"{domain_prefix}{order + 1:02d}", order + 1),
            q_label="evidencia_insuficiente",
            aliases=[],
            refs=[],
            status=status_suffix,
        )
        placeholders.append(
            ClusterSpec(
                cluster_id=f"{domain_prefix.lower()}_cluster_{order + 1:02d}",
                cluster_code=cluster_code,
                cluster_label="evidencia_insuficiente",
                cluster_order=order,
                points=[
                    PointSpec(
                        point_id=f"{domain_prefix.lower()}_point_{order + 1:02d}",
                        point_code=point_code,
                        point_label="evidencia_insuficiente",
                        point_order=0,
                        questions=[placeholder_question],
                        status=status_suffix,
                    )
                ],
                status=status_suffix,
            )
        )
    return placeholders


def _build_crosswalk(
        canonical_clusters: List[ClusterSpec],
        full_clusters: List[ClusterSpec],
        industrial_clusters: List[ClusterSpec],
        dnp_clusters: List[ClusterSpec],
) -> Dict[str, List[Dict[str, str]]]:
    crosswalk = {"clusters": [], "points": [], "questions": []}
    for idx, canon_cluster in enumerate(canonical_clusters):
        full_cluster = full_clusters[idx]
        industrial_cluster = industrial_clusters[idx]
        dnp_cluster = dnp_clusters[idx]
        crosswalk["clusters"].append(
            {
                "canonical_code": canon_cluster.cluster_code,
                "full": full_cluster.cluster_code,
                "industrial": industrial_cluster.cluster_code,
                "dnp": dnp_cluster.cluster_code,
                "note": canon_cluster.cluster_label,
            }
        )
        crosswalk["points"].append(
            {
                "canonical_code": canon_cluster.points[0].point_code,
                "full": full_cluster.points[0].point_code,
                "industrial": industrial_cluster.points[0].point_code,
                "dnp": dnp_cluster.points[0].point_code,
            }
        )
        crosswalk["questions"].append(
            {
                "canonical_code": canon_cluster.points[0].questions[0].q_code,
                "full": full_cluster.points[0].questions[0].q_code,
                "industrial": industrial_cluster.points[0].questions[0].q_code,
                "dnp": dnp_cluster.points[0].questions[0].q_code,
            }
        )
    return crosswalk


def align_decalogos(
        full_path: Path,
        industrial_path: Path,
        dnp_path: Path,
        out_dir: Path,
) -> CanonicalBundle:
    audit: List[AuditIssue] = []
    full_data = _load_decalogo_full(full_path, audit)
    grouped = _group_questions_by_point(full_data, audit)
    if not grouped:
        raise ValueError(
            "No se encontraron preguntas para construir el canónico")
    canonical_clusters = _build_canonical_clusters(grouped)
    full_clusters = canonical_clusters

    industrial_raw = _load_decalogo_industrial(industrial_path, audit)
    if industrial_raw:
        audit.append(
            AuditIssue(
                source="decalogo-industrial",
                category="info",
                message="Los indicadores industriales carecen de preguntas explícitas; se crean placeholders.",
            )
        )
    industrial_clusters = _placeholder_clusters_from_canonical(
        canonical_clusters, "IND", "faltante_en_industrial"
    )

    dnp_data = _load_dnp_standards(dnp_path, audit)
    if dnp_data is None:
        audit.append(
            AuditIssue(
                source="dnp-standards",
                category="evidencia_insuficiente",
                message="No hay datos estructurados; se crean placeholders.",
            )
        )
    else:
        audit.append(
            AuditIssue(
                source="dnp-standards",
                category="info",
                message="El archivo fue truncado y sólo se dispone de metadatos parciales.",
            )
        )
    dnp_clusters = _placeholder_clusters_from_canonical(
        canonical_clusters, "DNP", "faltante_en_dnp"
    )

    crosswalk = _build_crosswalk(
        canonical_clusters, full_clusters, industrial_clusters, dnp_clusters
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_clean_outputs(
        out_dir, canonical_clusters, industrial_clusters, dnp_clusters, crosswalk
    )
    _write_reports(out_dir.parent, audit)

    return CanonicalBundle(
        clusters=canonical_clusters, audit=audit, crosswalk=crosswalk
    )


def _write_clean_outputs(
        out_dir: Path,
        full_clusters: List[ClusterSpec],
        industrial_clusters: List[ClusterSpec],
        dnp_clusters: List[ClusterSpec],
        crosswalk: Dict[str, List[Dict[str, str]]],
        version: str = "1.0.0",
) -> None:
    def dump_json(path: Path, payload: Dict[str, object]) -> None:
        text = json.dumps(payload, ensure_ascii=False,
                          indent=2, sort_keys=True)
        path.write_text(text + "\n", encoding="utf-8")

    full_payload = {
        "version": version,
        "domain": "PDM",
        "clusters": [c.to_dict() for c in full_clusters],
        "crosswalk": crosswalk,
        "subclusters": [],
    }
    industrial_payload = {
        "version": version,
        "domain": "Industrial",
        "clusters": [c.to_dict() for c in industrial_clusters],
        "crosswalk": crosswalk,
        "kpi_templates": [],
    }
    dnp_payload = {
        "version": version,
        "domain": "DNP",
        "clusters": [c.to_dict() for c in dnp_clusters],
        "crosswalk": crosswalk,
        "norm_refs": [],
        "competence_map": {
            "municipal": "evidencia_insuficiente",
            "departamental": "evidencia_insuficiente",
            "nacional": "evidencia_insuficiente",
        },
    }

    dump_json(out_dir / "decalogo-full.v1.0.0.clean.json", full_payload)
    dump_json(out_dir / "decalogo-industrial.v1.0.0.clean.json",
              industrial_payload)
    dump_json(out_dir / "dnp-standards.v1.0.0.clean.json", dnp_payload)
    dump_json(out_dir / "crosswalk.v1.0.0.json",
              {"version": version, **crosswalk})

    # Symlinks para versiones latest.
    for target, link_name in [
        ("decalogo-full.v1.0.0.clean.json", "decalogo-full.latest.clean.json"),
        (
                "decalogo-industrial.v1.0.0.clean.json",
                "decalogo-industrial.latest.clean.json",
        ),
        ("dnp-standards.v1.0.0.clean.json", "dnp-standards.latest.clean.json"),
        ("crosswalk.v1.0.0.json", "crosswalk.latest.json"),
    ]:
        link_path = out_dir / link_name
        target_path = Path(target)
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target_path)


def _write_reports(root: Path, audit: List[AuditIssue]) -> None:
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    audit_lines = ["# Auditoría de decálogos", ""]
    if not audit:
        audit_lines.append("Sin observaciones")
    else:
        audit_lines.extend(issue.as_markdown() for issue in audit)

    (reports_dir / "decalogo_audit.md").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )

    changelog_lines = [
        "# Changelog de normalización",
        "",
        "- Normalización inicial versión 1.0.0",
    ]
    (reports_dir / "decalogo_changelog.md").write_text(
        "\n".join(changelog_lines) + "\n", encoding="utf-8"
    )
