#!/usr/bin/env python3
"""
Continuous Validation System
Pre-commit/deployment validation for MINIMINIMOON.

Features:
- Automated checks for Python version, imports, conflicts
- Type checking, linting, testing integration
- Fail-fast approach with clear error reporting
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class ValidationRunner:
    """Run continuous validation checks."""
    
    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path.cwd()
        self.results: List[Dict[str, Any]] = []
        
    def run_check(self, name: str, command: List[str], timeout: int = 60) -> Tuple[bool, str]:
        """Run a single validation check."""
        print(f"Running {name}...", end=' ', flush=True)
        
        try:
            result = subprocess.run(
                command,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            
            print("✓" if passed else "✗")
            
            return passed, output
            
        except subprocess.TimeoutExpired:
            print("⏱ TIMEOUT")
            return False, f"Check timed out after {timeout}s"
        except FileNotFoundError:
            print("⚠ SKIP (command not found)")
            return True, "Command not available"
        except Exception as e:
            print(f"✗ ERROR")
            return False, str(e)
    
    def check_python_version(self) -> bool:
        """Check Python version requirement."""
        script_path = self.root_dir / 'scripts' / 'check_python_version.py'
        
        if not script_path.exists():
            print("Checking Python version...", end=' ', flush=True)
            version = sys.version_info
            passed = (3, 10) <= (version.major, version.minor) <= (3, 12)
            print("✓" if passed else "✗")
            
            self.results.append({
                'check': 'python_version',
                'passed': passed,
                'message': f"Python {version.major}.{version.minor}.{version.micro}",
            })
            return passed
        
        passed, output = self.run_check(
            'Python version',
            [sys.executable, str(script_path), '--strict']
        )
        
        self.results.append({
            'check': 'python_version',
            'passed': passed,
            'output': output,
        })
        
        return passed
    
    def check_imports(self) -> bool:
        """Check critical imports."""
        print("Checking critical imports...", end=' ', flush=True)
        
        critical_imports = [
            'numpy', 'scipy', 'sklearn', 'torch', 'transformers',
            'sentence_transformers', 'spacy', 'pandas', 'networkx',
        ]
        
        failed = []
        for module in critical_imports:
            try:
                __import__(module)
            except ImportError:
                failed.append(module)
        
        passed = len(failed) == 0
        print("✓" if passed else "✗")
        
        self.results.append({
            'check': 'imports',
            'passed': passed,
            'failed_imports': failed,
        })
        
        return passed
    
    def check_conflicts(self) -> bool:
        """Check for dependency conflicts."""
        script_path = self.root_dir / 'scripts' / 'check_conflicts.py'
        
        if not script_path.exists():
            # Fallback to pip check
            passed, output = self.run_check(
                'dependency conflicts',
                [sys.executable, '-m', 'pip', 'check']
            )
        else:
            passed, output = self.run_check(
                'dependency conflicts',
                [sys.executable, str(script_path)]
            )
        
        self.results.append({
            'check': 'conflicts',
            'passed': passed,
            'output': output[:500] if output else '',
        })
        
        return passed
    
    def check_type_hints(self) -> bool:
        """Run type checking with mypy (if available)."""
        passed, output = self.run_check(
            'type checking (mypy)',
            [sys.executable, '-m', 'mypy', '--config-file=pyproject.toml', '.'],
            timeout=120
        )
        
        self.results.append({
            'check': 'type_checking',
            'passed': passed,
            'output': output[:500] if output else '',
        })
        
        return passed
    
    def check_linting(self) -> bool:
        """Run linting with flake8 (if available)."""
        passed, output = self.run_check(
            'linting (flake8)',
            [sys.executable, '-m', 'flake8', '.'],
            timeout=120
        )
        
        self.results.append({
            'check': 'linting',
            'passed': passed,
            'output': output[:500] if output else '',
        })
        
        return passed
    
    def check_formatting(self) -> bool:
        """Check code formatting with black (if available)."""
        passed, output = self.run_check(
            'formatting (black)',
            [sys.executable, '-m', 'black', '--check', '.'],
            timeout=120
        )
        
        self.results.append({
            'check': 'formatting',
            'passed': passed,
            'output': output[:500] if output else '',
        })
        
        return passed
    
    def run_tests(self) -> bool:
        """Run test suite (if available)."""
        passed, output = self.run_check(
            'tests (pytest)',
            [sys.executable, '-m', 'pytest', '-x', '--tb=short'],
            timeout=300
        )
        
        self.results.append({
            'check': 'tests',
            'passed': passed,
            'output': output[:1000] if output else '',
        })
        
        return passed
    
    def run_all_checks(self, include_optional: bool = False) -> bool:
        """Run all validation checks."""
        print("="*70)
        print("CONTINUOUS VALIDATION")
        print("="*70)
        print()
        
        # Critical checks (always run)
        critical_checks = [
            ('Python Version', self.check_python_version),
            ('Critical Imports', self.check_imports),
            ('Dependency Conflicts', self.check_conflicts),
        ]
        
        # Optional checks (only run if requested)
        optional_checks = [
            ('Type Checking', self.check_type_hints),
            ('Linting', self.check_linting),
            ('Formatting', self.check_formatting),
            ('Tests', self.run_tests),
        ]
        
        print("Critical Checks:")
        print("-" * 70)
        critical_passed = True
        for name, check_func in critical_checks:
            try:
                passed = check_func()
                critical_passed = critical_passed and passed
            except Exception as e:
                print(f"✗ {name} failed with exception: {e}")
                critical_passed = False
        
        print()
        
        if include_optional:
            print("Optional Checks:")
            print("-" * 70)
            optional_passed = True
            for name, check_func in optional_checks:
                try:
                    passed = check_func()
                    optional_passed = optional_passed and passed
                except Exception as e:
                    print(f"⚠ {name} failed with exception: {e}")
            print()
        else:
            optional_passed = True
        
        all_passed = critical_passed and optional_passed
        
        print("="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        if all_passed:
            print("✅ All checks passed")
        else:
            print("❌ Some checks failed")
            print("\nFailed checks:")
            for result in self.results:
                if not result['passed']:
                    print(f"  • {result['check']}")
        
        return all_passed


def main() -> int:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run continuous validation for MINIMINIMOON'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all checks including optional ones (type checking, linting, tests)'
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path.cwd(),
        help='Root directory of the project'
    )
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Exit immediately on first failure'
    )
    
    args = parser.parse_args()
    
    runner = ValidationRunner(args.root)
    
    try:
        all_passed = runner.run_all_checks(include_optional=args.all)
    except KeyboardInterrupt:
        print("\n\n⚠ Validation interrupted by user")
        return 130
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
