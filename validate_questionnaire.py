#!/usr/bin/env python3
"""
JSON Validator for 300-Question Questionnaire
Ensures structural integrity and prevents corruption during AI processing
"""

import json
import sys
from pathlib import Path

def validate_questionnaire_structure(data):
    """Validate the complete 300-question structure"""
    errors = []
    
    # Check basic structure
    required_fields = ['version', 'schema', 'total', 'questions']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Check total count
    if data.get('total') != 300:
        errors.append(f"Total should be 300, got {data.get('total')}")
    
    questions = data.get('questions', [])
    if len(questions) != 300:
        errors.append(f"Questions array should have 300 items, got {len(questions)}")
    
    # Validate structure: 10 domains (P1-P10) × 6 dimensions (D1-D6) × 5 questions each
    domains = set()
    dimensions = set()
    question_counts = {}
    
    for i, q in enumerate(questions):
        # Check required question fields
        required_q_fields = ['id', 'dimension', 'question_no', 'point_code', 'point_title', 'prompt', 'hints']
        for field in required_q_fields:
            if field not in q:
                errors.append(f"Question {i+1}: Missing field '{field}'")
        
        # Track domains and dimensions
        point_code = q.get('point_code', '')
        dimension = q.get('dimension', '')
        domains.add(point_code)
        dimensions.add(dimension)
        
        # Count questions per domain-dimension combination
        key = f"{point_code}-{dimension}"
        question_counts[key] = question_counts.get(key, 0) + 1
    
    # Validate domain structure (P1-P10)
    expected_domains = {f"P{i}" for i in range(1, 11)}
    if domains != expected_domains:
        missing = expected_domains - domains
        extra = domains - expected_domains
        if missing:
            errors.append(f"Missing domains: {sorted(missing)}")
        if extra:
            errors.append(f"Unexpected domains: {sorted(extra)}")
    
    # Validate dimension structure (D1-D6)
    expected_dimensions = {f"D{i}" for i in range(1, 7)}
    if dimensions != expected_dimensions:
        missing = expected_dimensions - dimensions
        extra = dimensions - expected_dimensions
        if missing:
            errors.append(f"Missing dimensions: {sorted(missing)}")
        if extra:
            errors.append(f"Unexpected dimensions: {sorted(extra)}")
    
    # Validate question distribution (5 questions per domain-dimension)
    for domain in expected_domains:
        for dim in expected_dimensions:
            key = f"{domain}-{dim}"
            count = question_counts.get(key, 0)
            if count != 5:
                errors.append(f"Domain {domain} Dimension {dim}: Expected 5 questions, got {count}")
    
    return errors

def validate_json_file(file_path):
    """Validate a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ JSON syntax is VALID for {file_path}")
        
        # Validate questionnaire structure
        errors = validate_questionnaire_structure(data)
        
        if errors:
            print(f"\n❌ STRUCTURAL ERRORS FOUND ({len(errors)}):")
            for error in errors:
                print(f"  • {error}")
            return False
        else:
            print("✅ QUESTIONNAIRE STRUCTURE IS PERFECT")
            print(f"  • Total questions: {len(data.get('questions', []))}")
            print(f"  • Domains: P1-P10 (10 domains)")
            print(f"  • Dimensions: D1-D6 (6 dimensions)")
            print(f"  • Questions per domain: 30 (6 × 5)")
            return True
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON SYNTAX ERROR in {file_path}:")
        print(f"  • {e}")
        return False
    except FileNotFoundError:
        print(f"❌ FILE NOT FOUND: {file_path}")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_questionnaire.py <json_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    is_valid = validate_json_file(file_path)
    
    if not is_valid:
        sys.exit(1)
    
    print(f"\n🎉 VALIDATION COMPLETE: {file_path} is ready for automated processing!")