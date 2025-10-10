#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for decalogo-industrial.latest.clean.json and dnp-standards.latest.clean.json

This script validates that:
1. Both JSON files are in the correct location (repository root)
2. All Python modules that reference these files can find them correctly
3. The paths resolve correctly from all locations in the codebase

Per issue requirements: Ensure the files are in the right location and verify all
invocations match the available version in the current path.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class FileLocationValidator:
    """Validator for JSON file locations and references."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent.resolve()
        self.industrial_file = "decalogo-industrial.latest.clean.json"
        self.dnp_file = "dnp-standards.latest.clean.json"
        self.errors = []
        self.warnings = []
        
    def validate_file_exists(self, filename: str) -> bool:
        """Validate that a JSON file exists in the repository root."""
        file_path = self.repo_root / filename
        if not file_path.exists():
            self.errors.append(f"❌ {filename} not found at {file_path}")
            return False
        print(f"✓ {filename} found at {file_path}")
        return True
    
    def validate_json_structure(self, filename: str) -> bool:
        """Validate that a JSON file has valid structure."""
        file_path = self.repo_root / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check basic structure
            if filename == self.industrial_file:
                if not isinstance(data, dict):
                    self.errors.append(f"❌ {filename} should be a dict")
                    return False
                if 'questions' not in data:
                    self.errors.append(f"❌ {filename} missing 'questions' key")
                    return False
                questions = data.get('questions', [])
                print(f"✓ {filename} has valid structure with {len(questions)} questions")
                return True
            
            elif filename == self.dnp_file:
                if not isinstance(data, dict):
                    self.errors.append(f"❌ {filename} should be a dict")
                    return False
                print(f"✓ {filename} has valid structure")
                return True
                
        except json.JSONDecodeError as e:
            self.errors.append(f"❌ {filename} has invalid JSON: {e}")
            return False
        except Exception as e:
            self.errors.append(f"❌ Error reading {filename}: {e}")
            return False
    
    def validate_module_imports(self) -> bool:
        """Validate that modules can import and load the files."""
        try:
            # Test decalogo_loader.py
            from decalogo_loader import get_decalogo_industrial, load_dnp_standards
            
            industrial = get_decalogo_industrial()
            if not industrial or 'questions' not in industrial:
                self.errors.append("❌ decalogo_loader.get_decalogo_industrial() failed")
                return False
            print(f"✓ decalogo_loader.get_decalogo_industrial() works ({len(industrial['questions'])} questions)")
            
            dnp = load_dnp_standards()
            if not dnp:
                self.errors.append("❌ decalogo_loader.load_dnp_standards() failed")
                return False
            print(f"✓ decalogo_loader.load_dnp_standards() works")
            
            return True
            
        except ImportError as e:
            self.errors.append(f"❌ Import error: {e}")
            return False
        except Exception as e:
            self.errors.append(f"❌ Module validation error: {e}")
            return False
    
    def validate_config_paths(self) -> bool:
        """Validate pdm_contra/config/decalogo.yaml paths."""
        try:
            import yaml
            
            config_path = self.repo_root / "pdm_contra" / "config" / "decalogo.yaml"
            if not config_path.exists():
                self.warnings.append(f"⚠️  Config file not found: {config_path}")
                return True  # Not critical
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            config_dir = config_path.parent
            paths_config = config.get('paths', {})
            
            for key, rel_path in paths_config.items():
                resolved = (config_dir / rel_path).resolve()
                if not resolved.exists():
                    self.errors.append(f"❌ Config path '{key}' resolves to non-existent file: {resolved}")
                    return False
                print(f"✓ Config path '{key}' resolves correctly to {resolved.name}")
            
            return True
            
        except ImportError:
            self.warnings.append("⚠️  PyYAML not installed, skipping config validation")
            return True
        except Exception as e:
            self.errors.append(f"❌ Config validation error: {e}")
            return False
    
    def validate_test_paths(self) -> List[Tuple[str, bool, str]]:
        """Validate paths used in test files."""
        test_results = []
        
        # Check test files that directly reference the JSON files
        test_files = [
            'test_decalogo_loader.py',
            'test_decalogo_alignment_fix.py',
            'test_dnp_standards_json.py',
            'verify_decalogo_alignment.py'
        ]
        
        for test_file in test_files:
            test_path = self.repo_root / test_file
            if test_path.exists():
                # Verify the test file references the correct location
                content = test_path.read_text(encoding='utf-8')
                
                # Check if it uses Path(__file__).parent or repo_root patterns
                if "Path(__file__).parent" in content or "repo_root" in content:
                    test_results.append((test_file, True, "Uses correct path pattern"))
                    print(f"✓ {test_file} uses correct path pattern")
                else:
                    test_results.append((test_file, False, "May have hardcoded paths"))
                    self.warnings.append(f"⚠️  {test_file} may have hardcoded paths")
            else:
                test_results.append((test_file, False, "File not found"))
        
        return test_results
    
    def run_validation(self) -> bool:
        """Run all validations."""
        print("=" * 80)
        print("JSON FILE LOCATION VALIDATION")
        print("=" * 80)
        print()
        print(f"Repository root: {self.repo_root}")
        print()
        
        print("1. Validating file existence...")
        print("-" * 80)
        exists_industrial = self.validate_file_exists(self.industrial_file)
        exists_dnp = self.validate_file_exists(self.dnp_file)
        print()
        
        if not (exists_industrial and exists_dnp):
            print("❌ Critical files missing!")
            return False
        
        print("2. Validating JSON structure...")
        print("-" * 80)
        valid_industrial = self.validate_json_structure(self.industrial_file)
        valid_dnp = self.validate_json_structure(self.dnp_file)
        print()
        
        print("3. Validating module imports...")
        print("-" * 80)
        imports_ok = self.validate_module_imports()
        print()
        
        print("4. Validating config paths...")
        print("-" * 80)
        config_ok = self.validate_config_paths()
        print()
        
        print("5. Validating test file paths...")
        print("-" * 80)
        test_results = self.validate_test_paths()
        print()
        
        # Print summary
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if not self.errors:
            print("\n✅ ALL VALIDATIONS PASSED")
            print("\nThe JSON files are in the correct location and all references resolve correctly.")
            print(f"\nCanonical locations:")
            print(f"  - {self.industrial_file}: {self.repo_root / self.industrial_file}")
            print(f"  - {self.dnp_file}: {self.repo_root / self.dnp_file}")
            return True
        else:
            print("\n❌ VALIDATION FAILED")
            return False


def main():
    """Main entry point."""
    validator = FileLocationValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
