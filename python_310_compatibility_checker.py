"""
Python 3.10 Compatibility Checker
Validates library imports and identifies incompatibilities across the system.
Special attention to NumPy and critical dependencies.
"""

import importlib
import logging
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Result of an import test."""

    module_name: str
    success: bool
    version: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class Python310CompatibilityChecker:
    """Comprehensive Python 3.10 compatibility checker."""

    # Critical modules with special attention to NumPy
    CRITICAL_MODULES = [
        "numpy",
        "scipy",
        "scikit-learn",
        "sentence_transformers",
        "transformers",
        "torch",
        "spacy",
        "opentelemetry",
        "pytest",
        "pydantic",
    ]

    # System modules to validate
    SYSTEM_MODULES = [
        "embedding_model",
        "responsibility_detector",
        "decalogo_loader",
        "spacy_loader",
        "canary_deployment",
        "opentelemetry_instrumentation",
        "slo_monitoring",
        "determinism_verifier",
    ]

    # Known incompatible version patterns
    INCOMPATIBLE_PATTERNS = {
        "numpy": {
            "min_version": "1.21.0",
            "max_version": "1.25.0",
            "issues": [
                "Matrix API changes in 1.25+",
                "Deprecated random number generation",
            ],
        },
        "scikit-learn": {
            "min_version": "1.0.0",
            "max_version": "1.4.0",
            "issues": ["API changes in sklearn 1.4+"],
        },
        "torch": {
            "min_version": "1.12.0",
            "max_version": "2.1.0",
            "issues": ["Python 3.10 support added in 1.12+"],
        },
    }

    def __init__(self):
        self.results: List[ImportResult] = []
        self.python_version = sys.version_info

    def validate_python_version(self) -> bool:
        """Validate Python 3.10 is being used."""
        if self.python_version.major != 3 or self.python_version.minor != 10:
            logger.error(
                "Python 3.10 required, found %s.%s",
                self.python_version.major,
                self.python_version.minor,
            )
            return False
        logger.info(
            "✓ Python %s.%s.%s detected",
            self.python_version.major,
            self.python_version.minor,
            self.python_version.micro,
        )
        return True

    @staticmethod
    def test_import(module_name: str) -> ImportResult:
        """Test importing a module and capture version info."""
        try:
            # Capture warnings during import
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                module = importlib.import_module(module_name)

                # Get version if available
                version = None
                for attr in ["__version__", "version", "VERSION"]:
                    if hasattr(module, attr):
                        version = getattr(module, attr)
                        break

                # Convert warnings to strings
                warning_messages = [str(warning.message) for warning in w]

                return ImportResult(
                    module_name=module_name,
                    success=True,
                    version=str(version) if version else None,
                    warnings=warning_messages,
                )

        except ImportError as e:
            return ImportResult(
                module_name=module_name,
                success=False,
                error_message=f"ImportError: {str(e)}",
            )
        except Exception as e:
            return ImportResult(
                module_name=module_name,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    @staticmethod
    def check_numpy_compatibility() -> Dict[str, any]:
        """Special NumPy compatibility checks for Python 3.10."""
        try:
            import numpy as np

            compatibility_report = {
                "version": np.__version__,
                "python_310_compatible": True,
                "issues": [],
                "recommendations": [],
            }

            # Check version compatibility
            version_parts = np.__version__.split(".")
            major, minor = int(version_parts[0]), int(version_parts[1])

            if major < 1 or (major == 1 and minor < 21):
                compatibility_report["python_310_compatible"] = False
                compatibility_report["issues"].append(
                    f"NumPy {np.__version__} lacks Python 3.10 support"
                )
                compatibility_report["recommendations"].append(
                    "Upgrade to NumPy >= 1.21.0"
                )

            if major > 1 or (major == 1 and minor >= 25):
                compatibility_report["issues"].append(
                    f"NumPy {np.__version__} may have breaking changes"
                )
                compatibility_report["recommendations"].append(
                    "Consider NumPy < 1.25.0 for stability"
                )

            # Test critical NumPy operations
            try:
                # Test random number generation (changed in recent versions)
                rng = np.random.default_rng(42)
                test_array = rng.random(10)

                # Test matrix operations
                matrix = np.array([[1, 2], [3, 4]])
                result = matrix @ matrix.T

                compatibility_report["core_operations"] = "OK"

            except Exception as e:
                compatibility_report["python_310_compatible"] = False
                compatibility_report["issues"].append(
                    f"NumPy core operations failed: {str(e)}"
                )

            return compatibility_report

        except ImportError:
            return {
                "version": None,
                "python_310_compatible": False,
                "issues": ["NumPy not installed"],
                "recommendations": ["Install NumPy >= 1.21.0, < 1.25.0"],
            }

    def run_compatibility_check(self) -> Dict[str, any]:
        """Run comprehensive compatibility check."""
        logger.info("Starting Python 3.10 compatibility check...")

        # Validate Python version
        python_ok = self.validate_python_version()

        # Test critical modules
        logger.info("Testing critical dependencies...")
        for module in self.CRITICAL_MODULES:
            result = self.test_import(module)
            self.results.append(result)

            if result.success:
                logger.info("✓ %s v%s", module, result.version or "unknown")
                if result.warnings:
                    for warning in result.warnings:
                        logger.warning("  ⚠ %s", warning)
            else:
                logger.error("✗ %s: %s", module, result.error_message)

        # Test system modules
        logger.info("Testing system modules...")
        for module in self.SYSTEM_MODULES:
            result = self.test_import(module)
            self.results.append(result)

            if result.success:
                logger.info("✓ %s", module)
            else:
                logger.warning("⚠ %s: %s", module, result.error_message)

        # Special NumPy check
        numpy_report = self.check_numpy_compatibility()

        # Generate summary
        successful_imports = sum(1 for r in self.results if r.success)
        total_imports = len(self.results)

        summary = {
            "python_version_ok": python_ok,
            "successful_imports": successful_imports,
            "total_imports": total_imports,
            "import_success_rate": successful_imports / total_imports
            if total_imports > 0
            else 0,
            "numpy_compatibility": numpy_report,
            "detailed_results": self.results,
            "overall_compatible": python_ok
            and numpy_report["python_310_compatible"]
            and successful_imports >= len(self.CRITICAL_MODULES),
        }

        return summary

    def generate_report(self, summary: Dict[str, any]) -> str:
        """Generate human-readable compatibility report."""
        report_lines = [
            "=" * 60,
            "PYTHON 3.10 COMPATIBILITY REPORT",
            "=" * 60,
            "",
            f"Python Version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro} {'✓' if summary['python_version_ok'] else '✗'}",
            f"Import Success Rate: {summary['import_success_rate']:.1%} ({summary['successful_imports']}/{summary['total_imports']})",
            f"Overall Compatible: {'✓' if summary['overall_compatible'] else '✗'}",
            "",
            "NUMPY COMPATIBILITY:",
            f"  Version: {summary['numpy_compatibility']['version'] or 'Not installed'}",
            f"  Python 3.10 Compatible: {'✓' if summary['numpy_compatibility']['python_310_compatible'] else '✗'}",
        ]

        if summary["numpy_compatibility"]["issues"]:
            report_lines.append("  Issues:")
            for issue in summary["numpy_compatibility"]["issues"]:
                report_lines.append(f"    - {issue}")

        if summary["numpy_compatibility"]["recommendations"]:
            report_lines.append("  Recommendations:")
            for rec in summary["numpy_compatibility"]["recommendations"]:
                report_lines.append(f"    - {rec}")

        report_lines.extend(
            [
                "",
                "DETAILED IMPORT RESULTS:",
            ]
        )

        for result in summary["detailed_results"]:
            status = "✓" if result.success else "✗"
            version_info = f" v{result.version}" if result.version else ""
            report_lines.append(f"  {status} {result.module_name}{version_info}")

            if not result.success:
                report_lines.append(f"      Error: {result.error_message}")
            elif result.warnings:
                for warning in result.warnings:
                    report_lines.append(f"      Warning: {warning}")

        return "\n".join(report_lines)


def main():
    """Run compatibility check and display results."""
    checker = Python310CompatibilityChecker()
    summary = checker.run_compatibility_check()
    report = checker.generate_report(summary)

    print(report)

    # Exit with error code if not compatible
    if not summary["overall_compatible"]:
        sys.exit(1)

    print("\n✓ System is Python 3.10 compatible!")


if __name__ == "__main__":
    main()
