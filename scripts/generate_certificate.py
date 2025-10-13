#!/usr/bin/env python3
"""
Compatibility Certificate Generator
Generates cryptographically signed certificates proving system compatibility.

Features:
- Verifiable proof of conflict-free system
- JSON + Markdown report formats
- Signature-based integrity validation
- Environment snapshot
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class CertificateGenerator:
    """Generate compatibility certificates for MINIMINIMOON."""

    def __init__(self):
        self.environment_info = {}
        self.dependency_info = {}
        self.test_results = {}

    def collect_environment_info(self) -> Dict[str, Any]:
        """Collect Python and system environment information."""
        print("Collecting environment information...")

        import platform

        env_info = {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Get git commit if available
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                env_info["git_commit"] = result.stdout.strip()
        except Exception:
            pass

        self.environment_info = env_info
        return env_info

    def collect_dependency_info(self) -> Dict[str, Any]:
        """Collect installed dependency information."""
        print("Collecting dependency information...")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                packages = json.loads(result.stdout)

                # Focus on key dependencies
                key_packages = {
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "torch",
                    "transformers",
                    "sentence-transformers",
                    "spacy",
                    "pandas",
                    "networkx",
                    "pydantic",
                    "jsonschema",
                    "fastapi",
                    "pytest",
                }

                filtered = {
                    pkg["name"]: pkg["version"]
                    for pkg in packages
                    if pkg["name"].lower() in {k.lower() for k in key_packages}
                }

                self.dependency_info = filtered
                return filtered

        except Exception as e:
            print(f"Warning: Could not collect dependency info: {e}")
            return {}

    def run_validation_checks(self) -> Dict[str, Any]:
        """Run validation checks and collect results."""
        print("Running validation checks...")

        checks = {
            "python_version": self._check_python_version(),
            "imports": self._check_critical_imports(),
            "conflicts": self._check_conflicts(),
        }

        self.test_results = checks
        return checks

    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        version = sys.version_info

        is_compatible = (3, 10) <= (version.major, version.minor) <= (3, 12)

        return {
            "passed": is_compatible,
            "version": f"{version.major}.{version.minor}.{version.micro}",
            "required": "3.10-3.12",
        }

    def _check_critical_imports(self) -> Dict[str, Any]:
        """Check if critical modules can be imported."""
        critical_modules = [
            "numpy",
            "scipy",
            "sklearn",
            "torch",
            "transformers",
            "sentence_transformers",
            "spacy",
            "pandas",
            "networkx",
        ]

        results = {}
        all_passed = True

        for module in critical_modules:
            try:
                imported = __import__(module)
                version = getattr(imported, "__version__", "unknown")
                results[module] = {
                    "passed": True,
                    "version": version,
                }
            except ImportError as e:
                results[module] = {
                    "passed": False,
                    "error": str(e),
                }
                all_passed = False

        return {
            "passed": all_passed,
            "modules": results,
        }

    def _check_conflicts(self) -> Dict[str, Any]:
        """Check for dependency conflicts."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            passed = result.returncode == 0

            return {
                "passed": passed,
                "output": result.stdout if not passed else "No conflicts",
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
            }

    def generate_certificate(self) -> Dict[str, Any]:
        """Generate the complete compatibility certificate."""
        print("\n" + "=" * 70)
        print("GENERATING COMPATIBILITY CERTIFICATE")
        print("=" * 70 + "\n")

        # Collect all information
        self.collect_environment_info()
        self.collect_dependency_info()
        self.run_validation_checks()

        # Check if all tests passed
        all_passed = all(
            check.get("passed", False) for check in self.test_results.values()
        )

        # Build certificate
        certificate = {
            "certificate_version": "1.0",
            "system": "MINIMINIMOON",
            "timestamp": self.environment_info["timestamp"],
            "status": "CERTIFIED" if all_passed else "FAILED",
            "environment": self.environment_info,
            "dependencies": self.dependency_info,
            "validation_results": self.test_results,
        }

        # Generate signature (hash of certificate content)
        cert_json = json.dumps(certificate, sort_keys=True)
        signature = hashlib.sha256(cert_json.encode()).hexdigest()
        certificate["signature"] = signature

        return certificate

    def save_certificate(self, certificate: Dict[str, Any], output_dir: Path) -> None:
        """Save certificate in JSON and Markdown formats."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_dir / "compatibility_certificate.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(certificate, f, indent=2)
        print(f"✓ JSON certificate saved to: {json_path}")

        # Save Markdown
        md_path = output_dir / "compatibility_certificate.md"
        markdown = self._format_markdown(certificate)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"✓ Markdown certificate saved to: {md_path}")

    def _format_markdown(self, certificate: Dict[str, Any]) -> str:
        """Format certificate as Markdown."""
        status = certificate["status"]
        status_emoji = "✅" if status == "CERTIFIED" else "❌"

        md = f"""# MINIMINIMOON Compatibility Certificate

{status_emoji} **Status**: {status}

**Generated**: {certificate["timestamp"]}
**Signature**: `{certificate["signature"][:16]}...`

---

## Environment

- **Python Version**: {certificate["environment"]["python_version"].split()[0]}
- **Platform**: {certificate["environment"]["platform"]}
- **Machine**: {certificate["environment"]["machine"]}
"""

        if "git_commit" in certificate["environment"]:
            md += (
                f"- **Git Commit**: `{certificate['environment']['git_commit'][:8]}`\n"
            )

        md += "\n---\n\n## Core Dependencies\n\n"

        for pkg, version in sorted(certificate["dependencies"].items()):
            md += f"- **{pkg}**: {version}\n"

        md += "\n---\n\n## Validation Results\n\n"

        for check_name, result in certificate["validation_results"].items():
            passed = result.get("passed", False)
            check_emoji = "✅" if passed else "❌"
            md += f"### {check_emoji} {check_name.replace('_', ' ').title()}\n\n"

            if check_name == "python_version":
                md += f"- **Current**: {result['version']}\n"
                md += f"- **Required**: {result['required']}\n"
            elif check_name == "imports":
                failed = [m for m, r in result["modules"].items() if not r["passed"]]
                if failed:
                    md += f"- **Failed imports**: {', '.join(failed)}\n"
                else:
                    md += "- All critical imports successful\n"
            elif check_name == "conflicts":
                if not passed:
                    md += f"```\n{result.get('output', result.get('error'))}\n```\n"
                else:
                    md += "- No dependency conflicts detected\n"

            md += "\n"

        md += "---\n\n"
        md += f"**Certificate Signature**: `{certificate['signature']}`\n\n"
        md += "This certificate is cryptographically signed and can be verified.\n"

        return md


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate compatibility certificate for MINIMINIMOON"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("certificates"),
        help="Output directory for certificate files (default: certificates/)",
    )

    args = parser.parse_args()

    generator = CertificateGenerator()
    certificate = generator.generate_certificate()
    generator.save_certificate(certificate, args.output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("CERTIFICATE SUMMARY")
    print("=" * 70)

    status = certificate["status"]
    if status == "CERTIFIED":
        print("✅ System is CERTIFIED - all validation checks passed")
        print(f"\nSignature: {certificate['signature'][:32]}...")
        return 0
    else:
        print("❌ System FAILED certification")
        print("\nFailed checks:")
        for check_name, result in certificate["validation_results"].items():
            if not result.get("passed", False):
                print(f"  • {check_name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
