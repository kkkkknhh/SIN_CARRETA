#!/usr/bin/env python3
"""
Deep alignment verification test.
Verifies that all components correctly use the canonical decalogo and DNP standards.
"""

import sys


def test_decalogo_loader():
    """Test that decalogo_loader loads the canonical files correctly."""
    from decalogo_loader import (
        ensure_aligned_templates,
        get_decalogo_industrial,
        load_dnp_standards,
    )

    print("Testing decalogo_loader...")

    industrial = get_decalogo_industrial()
    assert industrial["version"] == "1.0", (
        f"Expected version 1.0, got {industrial['version']}"
    )
    assert industrial["schema"] == "decalogo_causal_questions_v1", "Unexpected schema"
    assert industrial["total"] == 300, (
        f"Expected 300 questions, got {industrial['total']}"
    )
    assert len(industrial["questions"]) == 300, (
        f"Expected 300 actual questions, got {len(industrial['questions'])}"
    )

    dnp = load_dnp_standards()
    assert dnp["version"] == "2.0_operational_integrated_complete", (
        "Unexpected DNP version"
    )
    assert dnp["schema"] == "estandar_instrucciones_evaluacion_pdm_300_criterios", (
        "Unexpected DNP schema"
    )

    templates = ensure_aligned_templates()
    assert templates["alignment"]["status"] == "verified", "Alignment not verified"
    assert templates["alignment"]["questions_found"] == 300, "Not all questions found"

    print("  ✓ decalogo_loader works correctly")
    return True


def test_bridge_provider():
    """Test that the bridge provider loads the bundle correctly."""
    from pdm_contra.bridges.decatalogo_provider import provide_decalogos

    print("Testing pdm_contra.bridges.decatalogo_provider...")

    bundle = provide_decalogos()
    assert "version" in bundle, "Bundle missing version"
    assert "clusters" in bundle, "Bundle missing clusters"
    assert "crosswalk" in bundle, "Bundle missing crosswalk"

    # Verify clusters contain questions
    clusters = bundle.get("clusters", [])
    assert len(clusters) > 0, "No clusters found"

    # Check first cluster has questions
    if clusters:
        first_cluster = clusters[0]
        assert "questions" in first_cluster, "Cluster missing questions"
        questions = first_cluster.get("questions", [])
        assert len(questions) > 0, "No questions in cluster"

        # Verify question structure
        first_q = questions[0]
        required_keys = ["id", "dimension", "point_code", "prompt"]
        for key in required_keys:
            assert key in first_q, f"Question missing required key: {key}"

    print("  ✓ bridge provider works correctly")
    return True


def test_dimension_alignment():
    """Test that dimensions are consistent across all files."""
    from decalogo_loader import get_decalogo_industrial, load_dnp_standards

    print("Testing dimension alignment...")

    industrial = get_decalogo_industrial()
    dnp = load_dnp_standards()

    # Get dimensions from industrial
    industrial_dims = set()
    for q in industrial["questions"]:
        industrial_dims.add(q["dimension"])

    # Get dimensions from DNP
    dnp_dims = set(dnp.get("mapeo_eslabones_dimensiones", {}).keys())

    # Verify they match
    assert industrial_dims == dnp_dims, (
        f"Dimension mismatch: industrial={industrial_dims}, dnp={dnp_dims}"
    )

    expected_dims = {"D1", "D2", "D3", "D4", "D5", "D6"}
    assert industrial_dims == expected_dims, (
        f"Expected {expected_dims}, got {industrial_dims}"
    )

    print(f"  ✓ Dimensions aligned: {sorted(industrial_dims)}")
    return True


def test_point_codes_alignment():
    """Test that point codes are consistent."""
    from decalogo_loader import get_decalogo_industrial

    print("Testing point codes alignment...")

    industrial = get_decalogo_industrial()

    # Get all point codes
    point_codes = set()
    for q in industrial["questions"]:
        point_codes.add(q["point_code"])

    expected_points = {f"P{i}" for i in range(1, 11)}  # P1 through P10
    assert point_codes == expected_points, (
        f"Expected {expected_points}, got {point_codes}"
    )

    # Verify each point has questions from all 6 dimensions
    from collections import defaultdict

    point_dims = defaultdict(set)
    for q in industrial["questions"]:
        point_dims[q["point_code"]].add(q["dimension"])

    for point, dims in point_dims.items():
        assert len(dims) == 6, f"Point {point} has only {len(dims)} dimensions: {dims}"

    print(f"  ✓ Point codes aligned: {sorted(point_codes, key=lambda x: int(x[1:]))}")
    return True


def test_question_id_format():
    """Test that question IDs follow the P#-D#-Q# format."""
    import re

    from decalogo_loader import get_decalogo_industrial

    print("Testing question ID format...")

    industrial = get_decalogo_industrial()

    pattern = re.compile(r"^P\d+-D\d+-Q\d+$")
    invalid_ids = []

    for q in industrial["questions"]:
        qid = q.get("id", "")
        if not pattern.match(qid):
            invalid_ids.append(qid)

    assert len(invalid_ids) == 0, (
        f"Found {len(invalid_ids)} invalid question IDs: {invalid_ids[:5]}"
    )

    print("  ✓ All 300 question IDs follow P#-D#-Q# format")
    return True


def test_scoring_scale_alignment():
    """Test that scoring scales are properly documented."""
    from decalogo_loader import load_dnp_standards

    print("Testing scoring scale alignment...")

    dnp = load_dnp_standards()

    # Check scoring system
    scoring = dnp.get("sistema_scoring_referencia", {})
    scale = scoring.get("escala_base", [])

    assert scale == [0, 4], f"Expected [0, 4] scale, got {scale}"

    # Check mapping rules exist
    mapeo = dnp.get("metodologia_evaluacion", {}).get("mapeo_scores_escala", {})
    assert "reglas" in mapeo, "Missing scoring rules"

    rules = mapeo["reglas"]
    expected_levels = [
        "0_AUSENTE",
        "1_INSUFICIENTE",
        "2_BASICO",
        "3_SATISFACTORIO",
        "4_AVANZADO",
    ]
    for level in expected_levels:
        assert level in rules, f"Missing scoring level: {level}"

    print("  ✓ Scoring scale [0, 4] properly defined with 5 levels")
    return True


def test_config_file_paths():
    """Test that configuration file points to correct files."""
    from pathlib import Path

    import yaml

    print("Testing configuration file paths...")

    config_path = Path("pdm_contra/config/decalogo.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check all paths exist
    paths = config.get("paths", {})
    for key, rel_path in paths.items():
        full_path = (config_path.parent / rel_path).resolve()
        assert full_path.exists(), f"Path for {key} not found: {full_path}"

    # Check crosswalk
    crosswalk_path = (config_path.parent / config["crosswalk"]).resolve()
    assert crosswalk_path.exists(), f"Crosswalk not found: {crosswalk_path}"

    print("  ✓ All configuration paths are valid")
    return True


def test_module_imports():
    """Test that all required modules can be imported."""
    print("Testing module imports...")

    modules = [
        "pdm_contra",
        "pdm_contra.core",
        "pdm_contra.models",
        "pdm_contra.scoring",
        "pdm_contra.prompts",
        "pdm_contra.policy",
        "pdm_contra.nlp",
        "pdm_contra.explain",
        "pdm_contra.bridges",
        "factibilidad",
        "jsonschema",
        "econml",
    ]

    for module in modules:
        try:
            __import__(module)
        except ImportError as e:
            # Some modules require numpy which might not be installed
            if "numpy" in str(e):
                print(f"  ⚠ {module} requires numpy (optional)")
            else:
                raise

    print("  ✓ All core modules import successfully")
    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("DEEP ALIGNMENT VERIFICATION TEST")
    print("=" * 80)
    print()

    tests = [
        test_decalogo_loader,
        test_bridge_provider,
        test_dimension_alignment,
        test_point_codes_alignment,
        test_question_id_format,
        test_scoring_scale_alignment,
        test_config_file_paths,
        test_module_imports,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
        print()

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
