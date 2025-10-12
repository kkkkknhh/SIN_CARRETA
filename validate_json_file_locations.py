#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for canonical JSON artifacts

This script validates that:
1. Both JSON files are in the correct canonical locations
2. All Python modules that reference these files can find them correctly
3. The paths resolve correctly from all locations in the codebase

Canonical locations:
- /bundles/decalogo-industrial.latest.clean.json
- /standards/dnp-standards.latest.clean.json
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple


class FileLocationValidator:
    """Validator for JSON file locations and references."""
    
    def __init__(self):
        # Use central path resolver
        from repo_paths import get_decalogo_path, get_dnp_path, REPO_ROOT
        
        self.repo_root = REPO_ROOT
        self.decalogo_path = get_decalogo_path()
        self.dnp_path = get_dnp_path()
        self.errors = []
        self.warnings = []
        
    def validate_file_exists(self, filepath: Path, name: str) -> bool:
        """Validate that a JSON file exists at the canonical location."""
        if not filepath.exists():
            self.errors.append(f"❌ {name} not found at {filepath}")
            return False
        print(f"✓ {name} found at {filepath}")
        return True
    
    def validate_json_structure(self, filepath: Path, name: str) -> bool:
        """Validate that a JSON file has valid structure."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check basic structure
            if "decalogo" in name.lower():
                if not isinstance(data, dict):
                    self.errors.append(f"❌ {name} should be a dict")
                    return False
                if 'questions' not in data:
                    self.errors.append(f"❌ {name} missing 'questions' key")
                    return False
                questions = data.get('questions', [])
                print(f"✓ {name} has valid structure with {len(questions)} questions")
                return True
            
            elif "dnp" in name.lower():
                if not isinstance(data, dict):
                    self.errors.append(f"❌ {name} should be a dict")
                    return False
                print(f"✓ {name} has valid structure")
                return True
                
        except json.JSONDecodeError as e:
            self.errors.append(f"❌ {name} has invalid JSON: {e}")
            return False
        except Exception as e:
            self.errors.append(f"❌ Error reading {name}: {e}")
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
        exists_decalogo = self.validate_file_exists(self.decalogo_path, "decalogo-industrial")
        exists_dnp = self.validate_file_exists(self.dnp_path, "dnp-standards")
        print()
        
        if not (exists_decalogo and exists_dnp):
            print("❌ Critical files missing!")
            return False
        
        print("2. Validating JSON structure...")
        print("-" * 80)
        valid_decalogo = self.validate_json_structure(self.decalogo_path, "decalogo-industrial")
        valid_dnp = self.validate_json_structure(self.dnp_path, "dnp-standards")
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
            print("\nThe JSON files are in the correct canonical locations and all references resolve correctly.")
            print(f"\nCanonical locations:")
            print(f"  - Decalogo: {self.decalogo_path}")
            print(f"  - DNP: {self.dnp_path}")
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
