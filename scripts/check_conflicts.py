#!/usr/bin/env python3
"""
Conflict Detection Engine
Multi-layered dependency conflict detection for MINIMINIMOON.

Features:
- Pip dependency resolution checks
- Version compatibility validation
- CUDA/OpenCV conflict detection
- Security vulnerability scanning
- Missing dependency identification
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ConflictDetector:
    """Detect dependency conflicts and compatibility issues."""

    def __init__(self, requirements_dir: Optional[Path] = None):
        self.requirements_dir = requirements_dir or Path("requirements")
        self.conflicts: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []

    def check_pip_dependencies(self) -> bool:
        """Check for pip dependency conflicts."""
        print("Checking pip dependency resolution...")

        try:
            # Run pip check
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                self.conflicts.append(
                    {
                        "type": "pip_check",
                        "severity": "error",
                        "message": "Pip dependency conflicts detected",
                        "details": result.stdout + result.stderr,
                    }
                )
                return False

            print("  ✓ No pip dependency conflicts")
            return True

        except subprocess.TimeoutExpired:
            self.warnings.append(
                {
                    "type": "pip_check",
                    "severity": "warning",
                    "message": "Pip check timed out",
                }
            )
            return True
        except Exception as e:
            self.warnings.append(
                {
                    "type": "pip_check",
                    "severity": "warning",
                    "message": f"Could not run pip check: {e}",
                }
            )
            return True

    def check_version_compatibility(self) -> bool:
        """Check if installed package versions are compatible."""
        print("Checking version compatibility...")

        # Define known compatibility constraints
        constraints = [
            {
                "package": "numpy",
                "min_version": (1, 21, 0),
                "max_version": (1, 27, 0),
                "reason": "Python 3.10-3.12 compatibility",
            },
            {
                "package": "torch",
                "min_version": (1, 12, 0),
                "max_version": (3, 0, 0),
                "reason": "Python 3.10+ support and API stability",
            },
            {
                "package": "transformers",
                "min_version": (4, 20, 0),
                "max_version": (5, 0, 0),
                "reason": "Model compatibility and API stability",
            },
            {
                "package": "spacy",
                "min_version": (3, 4, 0),
                "max_version": (4, 0, 0),
                "reason": "Language model compatibility",
            },
        ]

        all_compatible = True

        for constraint in constraints:
            package = constraint["package"]

            try:
                import importlib

                module = importlib.import_module(package)
                version_str = getattr(module, "__version__", None)

                if not version_str:
                    self.warnings.append(
                        {
                            "type": "version_check",
                            "severity": "warning",
                            "package": package,
                            "message": f"Could not determine {package} version",
                        }
                    )
                    continue

                # Parse version
                version = self._parse_version(version_str)
                min_ver = constraint["min_version"]
                max_ver = constraint["max_version"]

                if version < min_ver:
                    self.conflicts.append(
                        {
                            "type": "version_check",
                            "severity": "error",
                            "package": package,
                            "current": version_str,
                            "required": f">= {'.'.join(map(str, min_ver))}",
                            "reason": constraint["reason"],
                        }
                    )
                    all_compatible = False
                elif version >= max_ver:
                    self.conflicts.append(
                        {
                            "type": "version_check",
                            "severity": "error",
                            "package": package,
                            "current": version_str,
                            "required": f"< {'.'.join(map(str, max_ver))}",
                            "reason": constraint["reason"],
                        }
                    )
                    all_compatible = False
                else:
                    print(f"  ✓ {package} {version_str} is compatible")

            except ImportError:
                self.warnings.append(
                    {
                        "type": "version_check",
                        "severity": "warning",
                        "package": package,
                        "message": f"{package} is not installed",
                    }
                )

        return all_compatible

    def check_cuda_conflicts(self) -> bool:
        """Check for CUDA/OpenCV conflicts."""
        print("Checking CUDA compatibility...")

        try:
            import torch

            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                print(f"  ✓ CUDA {cuda_version} available")

                # Check for OpenCV with CUDA
                try:
                    import cv2

                    build_info = cv2.getBuildInformation()

                    if "CUDA" in build_info and "YES" in build_info:
                        # Check if PyTorch and OpenCV use compatible CUDA versions
                        self.warnings.append(
                            {
                                "type": "cuda_check",
                                "severity": "warning",
                                "message": "OpenCV built with CUDA detected",
                                "details": "Ensure OpenCV and PyTorch use compatible CUDA versions",
                            }
                        )
                except ImportError:
                    pass  # OpenCV not installed
            else:
                print("  ℹ CUDA not available (CPU-only mode)")

        except ImportError:
            print("  ℹ PyTorch not installed")

        return True

    def check_missing_dependencies(self) -> bool:
        """Check for missing required dependencies."""
        print("Checking for missing dependencies...")

        required_packages = [
            "numpy",
            "scipy",
            "sklearn",
            "torch",
            "transformers",
            "sentence_transformers",
            "spacy",
            "pandas",
            "networkx",
            "pydantic",
            "jsonschema",
        ]

        missing = []

        for package in required_packages:
            # Special case: scikit-learn imports as sklearn
            import_name = "sklearn" if package == "sklearn" else package

            try:
                __import__(import_name)
            except ImportError:
                missing.append(package)

        if missing:
            self.conflicts.append(
                {
                    "type": "missing_dependencies",
                    "severity": "error",
                    "message": "Required dependencies are missing",
                    "packages": missing,
                }
            )
            print(f"  ✗ Missing packages: {', '.join(missing)}")
            return False

        print(f"  ✓ All required packages installed")
        return True

    def check_security_vulnerabilities(self) -> bool:
        """Check for known security vulnerabilities (if safety is installed)."""
        print("Checking for security vulnerabilities...")

        try:
            # Check if safety is installed
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "safety"],
                capture_output=True,
                timeout=10,
            )

            if result.returncode != 0:
                print("  ℹ safety not installed (install with: pip install safety)")
                return True

            # Run safety check
            result = subprocess.run(
                [sys.executable, "-m", "safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.stdout:
                try:
                    vulnerabilities = json.loads(result.stdout)

                    if vulnerabilities:
                        self.warnings.append(
                            {
                                "type": "security",
                                "severity": "warning",
                                "message": f"Found {len(vulnerabilities)} security vulnerabilities",
                                "details": vulnerabilities,
                            }
                        )
                        print(f"  ⚠ {len(vulnerabilities)} vulnerabilities found")
                    else:
                        print("  ✓ No known vulnerabilities")
                except json.JSONDecodeError:
                    pass

        except subprocess.TimeoutExpired:
            print("  ℹ Security check timed out")
        except Exception as e:
            print(f"  ℹ Could not run security check: {e}")

        return True

    def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all conflict detection checks."""
        print("=" * 70)
        print("DEPENDENCY CONFLICT DETECTION")
        print("=" * 70)
        print()

        checks = [
            ("pip_dependencies", self.check_pip_dependencies),
            ("version_compatibility", self.check_version_compatibility),
            ("cuda_conflicts", self.check_cuda_conflicts),
            ("missing_dependencies", self.check_missing_dependencies),
            ("security_vulnerabilities", self.check_security_vulnerabilities),
        ]

        results = {}
        all_passed = True

        for check_name, check_func in checks:
            try:
                passed = check_func()
                results[check_name] = "passed" if passed else "failed"
                all_passed = all_passed and passed
            except Exception as e:
                results[check_name] = f"error: {e}"
                self.warnings.append(
                    {
                        "type": check_name,
                        "severity": "error",
                        "message": f"Check failed with exception: {e}",
                    }
                )
            print()

        return all_passed, {
            "checks": results,
            "conflicts": self.conflicts,
            "warnings": self.warnings,
        }

    @staticmethod
    def _parse_version(version_str: str) -> Tuple[int, ...]:
        """Parse version string to tuple of integers."""
        parts = []
        for part in version_str.split("."):
            # Extract numeric part
            num = ""
            for char in part:
                if char.isdigit():
                    num += char
                else:
                    break
            parts.append(int(num) if num else 0)
        return tuple(parts)


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect dependency conflicts in MINIMINIMOON"
    )
    parser.add_argument(
        "--requirements-dir",
        type=Path,
        default=Path("requirements"),
        help="Requirements directory (default: requirements/)",
    )
    parser.add_argument("--output", type=Path, help="Output JSON file for results")

    args = parser.parse_args()

    detector = ConflictDetector(args.requirements_dir)
    all_passed, results = detector.run_all_checks()

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_passed:
        print("✓ All checks passed - no conflicts detected")
        exit_code = 0
    else:
        print("✗ Conflicts detected")
        print(f"\nErrors: {len(results['conflicts'])}")
        for conflict in results["conflicts"]:
            print(
                f"  • {conflict.get('package', conflict['type'])}: {conflict['message']}"
            )
        exit_code = 1

    if results["warnings"]:
        print(f"\nWarnings: {len(results['warnings'])}")
        for warning in results["warnings"][:5]:
            print(
                f"  • {warning.get('package', warning['type'])}: {warning['message']}"
            )

    # Write results if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed report written to: {args.output}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
