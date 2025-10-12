#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate canonical decalogo_bundle.json from existing bundle with P-D-Q notation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def extract_rubric_key(question_unique_id: str) -> str:
    """Extract D#-Q# from P#-D#-Q#."""
    parts = question_unique_id.split("-")
    if len(parts) == 3:
        return f"{parts[1]}-{parts[2]}"
    raise ValueError(f"Invalid question_unique_id format: {question_unique_id}")


def generate_canonical_bundle(source_path: Path, output_path: Path) -> None:
    """Generate canonical bundle from source."""
    print(f"Loading source bundle from: {source_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        source = json.load(f)

    # Initialize canonical structure
    canonical = {
        "version": "5.0.0-canonical",
        "policies": [f"P{i}" for i in range(1, 11)],
        "dimensions": [f"D{i}" for i in range(1, 7)],
        "questions": [],
        "lexicon": {},
    }

    # Extract questions with canonical format
    for q in source.get("questions", []):
        qid = q.get("id")
        if not qid:
            continue

        canonical_q = {
            "question_unique_id": qid,
            "rubric_key": extract_rubric_key(qid),
            "text": q.get("prompt", ""),
            "dimension": q.get("dimension", ""),
            "question_no": q.get("question_no", 0),
            "point_code": q.get("point_code", ""),
            "point_title": q.get("point_title", ""),
            "hints": q.get("hints", []),
        }
        canonical["questions"].append(canonical_q)

    # Build lexicon from policy hints
    _policy_titles = {
        "P1": "Derechos de las mujeres e igualdad de género",
        "P2": "Prevención de la violencia y protección frente al conflicto",
        "P3": "Ambiente sano, cambio climático, prevención y atención a desastres",
        "P4": "Derechos económicos, sociales y culturales",
        "P5": "Derechos de las víctimas y construcción de paz",
        "P6": "Derecho al buen futuro de la niñez, adolescencia, juventud",
        "P7": "Tierras y territorios",
        "P8": "Líderes y defensores de derechos humanos",
        "P9": "Crisis de derechos de personas privadas de la libertad",
        "P10": "Migración transfronteriza",
    }

    # Extract unique hints per policy
    for pid in canonical["policies"]:
        hints_set = set()
        for q in canonical["questions"]:
            if q["point_code"] == pid:
                hints_set.update(q.get("hints", []))
        canonical["lexicon"][pid] = sorted(list(hints_set))

    # Add metadata
    canonical["metadata"] = {
        "total_questions": len(canonical["questions"]),
        "policies_count": len(canonical["policies"]),
        "dimensions_count": len(canonical["dimensions"]),
        "generated_from": str(source_path.name),
        "schema_version": "decalogo_bundle.schema.json",
    }

    print(f"Generated bundle with {len(canonical['questions'])} questions")
    print(f"Writing to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(canonical, f, ensure_ascii=False, indent=2, sort_keys=True)

    print("✓ Canonical bundle generated successfully")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    source = repo_root / "bundles" / "decalogo-industrial.latest.clean.json"
    output = repo_root / "bundles" / "decalogo_bundle.json"

    if not source.exists():
        print(f"ERROR: Source bundle not found: {source}")
        exit(1)

    generate_canonical_bundle(source, output)
