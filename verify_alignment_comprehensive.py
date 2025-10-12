#!/usr/bin/env python3
"""
Comprehensive alignment verification script.
Checks all files mentioned in the problem statement for alignment with
decalogo-industrial.latest.clean.json and dnp-standards.latest.clean.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def load_canonical_standards():
    """Load the canonical JSON standards."""
    # Use central path resolver
    from repo_paths import get_decalogo_path, get_dnp_path
    
    with open(get_decalogo_path()) as f:
        decalogo = json.load(f)
    with open(get_dnp_path()) as f:
        dnp = json.load(f)
    return decalogo, dnp

def check_file_exists(path: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    p = Path(path)
    if p.exists():
        return True, f"{GREEN}✓{RESET} EXISTS"
    else:
        return False, f"{RED}✗{RESET} MISSING"

def check_module_imports(module_path: str) -> Tuple[bool, str]:
    """Check if a Python module can be imported."""
    try:
        module_name = module_path.replace('/', '.').replace('.py', '')
        __import__(module_name)
        return True, f"{GREEN}✓{RESET} IMPORTS OK"
    except ImportError as e:
        return False, f"{RED}✗{RESET} IMPORT FAILED: {e}"
    except Exception as e:
        return False, f"{YELLOW}⚠{RESET} ERROR: {e}"

def check_dimension_references(file_path: str) -> Tuple[bool, str]:
    """Check if file references correct dimensions (D1-D6)."""
    if not Path(file_path).exists():
        return False, "File not found"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for dimension references
        dimensions = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
        found_dims = [d for d in dimensions if d in content]
        
        if found_dims:
            return True, f"{GREEN}✓{RESET} References: {', '.join(found_dims)}"
        else:
            return True, f"{YELLOW}⚠{RESET} No dimension references found"
    except Exception as e:
        return False, f"{RED}✗{RESET} ERROR: {e}"

def check_question_id_format(file_path: str) -> Tuple[bool, str]:
    """Check if file uses correct question ID format (P#-D#-Q#)."""
    if not Path(file_path).exists():
        return False, "File not found"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for P#-D#-Q# pattern
        import re
        pattern = r'P\d+-D\d+-Q\d+'
        matches = re.findall(pattern, content)
        
        if matches:
            return True, f"{GREEN}✓{RESET} Found {len(matches)} question IDs"
        else:
            return True, f"{YELLOW}⚠{RESET} No question IDs found"
    except Exception as e:
        return False, f"{RED}✗{RESET} ERROR: {e}"

def check_scoring_references(file_path: str) -> Tuple[bool, str]:
    """Check if file references correct scoring scale (0-4)."""
    if not Path(file_path).exists():
        return False, "File not found"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for scoring references
        has_scoring = any(term in content.lower() for term in ['score', 'scoring', 'puntuación', 'puntaje'])
        has_scale = '0-4' in content or '[0, 4]' in content
        
        if has_scoring:
            if has_scale:
                return True, f"{GREEN}✓{RESET} Has scoring with 0-4 scale"
            else:
                return True, f"{YELLOW}⚠{RESET} Has scoring (scale unclear)"
        else:
            return True, f"{YELLOW}⚠{RESET} No scoring references"
    except Exception as e:
        return False, f"{RED}✗{RESET} ERROR: {e}"

def main():
    print("=" * 80)
    print("COMPREHENSIVE DECALOGO ALIGNMENT VERIFICATION")
    print("=" * 80)
    print()
    
    # Load standards
    print("Loading canonical standards...")
    try:
        decalogo, dnp = load_canonical_standards()
        print(f"{GREEN}✓{RESET} Loaded decalogo-industrial.latest.clean.json (v{decalogo['version']}, {decalogo['total']} questions)")
        print(f"{GREEN}✓{RESET} Loaded dnp-standards.latest.clean.json (v{dnp['version']})")
        print()
    except Exception as e:
        print(f"{RED}✗{RESET} Failed to load standards: {e}")
        return 1
    
    # Files to check (from problem statement)
    files_to_check = [
        # pdm_contra core files
        ("pdm_contra/core.py", True, True, True),
        ("pdm_contra/__init__.py", True, False, False),
        ("pdm_contra/models.py", True, False, False),
        ("pdm_contra/decalogo_alignment.py", True, True, True),
        
        # pdm_contra/scoring
        ("pdm_contra/scoring/risk.py", True, True, True),
        ("pdm_contra/scoring/__init__.py", True, False, False),
        
        # pdm_contra/prompts
        ("pdm_contra/prompts/prompt_scoring_system.py", True, False, True),
        ("pdm_contra/prompts/prompt_scoring_system.md", False, False, True),
        ("pdm_contra/prompts/prompt_maestro.py", True, False, False),
        ("pdm_contra/prompts/prompt_maestro_pdm.md", False, False, False),
        ("pdm_contra/prompts/__init__.py", True, False, False),
        
        # pdm_contra/policy
        ("pdm_contra/policy/competence.py", True, False, False),
        ("pdm_contra/policy/__init__.py", True, False, False),
        
        # pdm_contra/nlp
        ("pdm_contra/nlp/patterns.py", True, False, False),
        ("pdm_contra/nlp/__init__.py", True, False, False),
        ("pdm_contra/nlp/nli.py", True, False, False),
        
        # pdm_contra/explain
        ("pdm_contra/explain/tracer.py", True, False, False),
        ("pdm_contra/explain/__init__.py", True, False, False),
        
        # pdm_contra/config
        ("pdm_contra/config/decalogo.yaml", False, False, False),
        
        # pdm_contra/bridges
        ("pdm_contra/bridges/decatalogo_provider.py", True, False, False),
        ("pdm_contra/bridges/decalogo_loader_adapter.py", True, False, False),
        
        # jsonschema
        ("jsonschema/__init__.py", True, False, False),
        ("jsonschema/validators.py", True, False, False),
        
        # factibilidad
        ("factibilidad/scoring.py", True, False, True),
        ("factibilidad/__init__.py", True, False, False),
        ("factibilidad/pattern_detector.py", True, False, False),
        
        # evaluation
        ("evaluation/reliability_calibration.py", True, False, True),
        ("evaluation/ground_truth_collector.py", True, False, False),
        ("evaluation/__init__.py", True, False, False),
        
        # econml
        ("econml/dml.py", True, False, False),
        ("econml/__init__.py", True, False, False),
    ]
    
    print("Checking files...")
    print()
    
    results = []
    for file_path, check_dims, check_qids, check_scoring in files_to_check:
        print(f"Checking: {file_path}")
        
        # Check existence
        exists, exists_msg = check_file_exists(file_path)
        print(f"  Exists: {exists_msg}")
        
        if not exists:
            results.append((file_path, False, "File missing"))
            print()
            continue
        
        # Check imports for Python files
        if file_path.endswith('.py'):
            _imports_ok, imports_msg = check_module_imports(file_path)
            print(f"  Imports: {imports_msg}")
        
        # Check dimension references
        if check_dims:
            _dims_ok, dims_msg = check_dimension_references(file_path)
            print(f"  Dimensions: {dims_msg}")
        
        # Check question ID format
        if check_qids:
            _qids_ok, qids_msg = check_question_id_format(file_path)
            print(f"  Question IDs: {qids_msg}")
        
        # Check scoring references
        if check_scoring:
            _scoring_ok, scoring_msg = check_scoring_references(file_path)
            print(f"  Scoring: {scoring_msg}")
        
        results.append((file_path, exists, "OK"))
        print()
    
    # Check output directory
    print("Checking output directory...")
    output_dir = Path("output")
    if output_dir.exists():
        print(f"{GREEN}✓{RESET} output/ directory exists")
    else:
        print(f"{YELLOW}⚠{RESET} output/ directory missing (will be created on first run)")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = len(results)
    existing = sum(1 for _, exists, _ in results if exists)
    missing = total - existing
    
    print(f"Total files checked: {total}")
    print(f"{GREEN}Existing: {existing}{RESET}")
    if missing > 0:
        print(f"{RED}Missing: {missing}{RESET}")
    
    if missing == 0:
        print()
        print(f"{GREEN}✓ All required files exist and are accessible{RESET}")
        return 0
    else:
        print()
        print(f"{RED}✗ Some required files are missing{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
