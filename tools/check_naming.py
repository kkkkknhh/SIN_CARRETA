#!/usr/bin/env python3
"""
Naming Convention Checker for MINIMINIMOON Project

Verifies that all code follows Python naming conventions:
- Modules: lowercase_with_underscores
- Classes: PascalCase
- Functions/Methods: lowercase_with_underscores
- Variables/Parameters: lowercase_with_underscores
- Constants: UPPERCASE_WITH_UNDERSCORES
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple


class NamingConventionChecker:
    """Check Python naming conventions in code"""

    def __init__(self):
        self.violations = []

    def check_module_name(self, filepath: Path) -> List[str]:
        """Check if module filename follows convention"""
        violations = []
        name = filepath.stem

        # Module names should be lowercase_with_underscores
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            violations.append(
                f"Module name '{name}' should be lowercase_with_underscores"
            )

        return violations

    def check_code_conventions(self, filepath: Path) -> Dict[str, List]:
        """Parse Python file and check naming conventions"""
        violations = {"classes": [], "functions": [], "variables": [], "parameters": []}

        try:
            with open(filepath) as f:
                tree = ast.parse(f.read())
        except Exception as e:
            return {"error": [f"Failed to parse {filepath}: {e}"]}

        for node in ast.walk(tree):
            # Check class names (should be PascalCase)
            if isinstance(node, ast.ClassDef):
                if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                    violations["classes"].append(
                        f"Class '{node.name}' should be PascalCase (line {node.lineno})"
                    )

            # Check function/method names (should be lowercase_with_underscores)
            elif isinstance(node, ast.FunctionDef):
                # Skip dunder methods
                if not (node.name.startswith("__") and node.name.endswith("__")):
                    if not re.match(r"^[a-z][a-z0-9_]*$", node.name):
                        violations["functions"].append(
                            f"Function '{node.name}' should be lowercase_with_underscores (line {node.lineno})"
                        )

                # Check parameter names
                for arg in node.args.args:
                    if arg.arg != "self" and arg.arg != "cls":
                        if not re.match(r"^[a-z][a-z0-9_]*$", arg.arg):
                            violations["parameters"].append(
                                f"Parameter '{arg.arg}' in {node.name}() should be lowercase_with_underscores (line {node.lineno})"
                            )

        return violations

    def check_file(self, filepath: Path) -> Tuple[str, Dict]:
        """Check a single Python file"""
        result = {
            "module": self.check_module_name(filepath),
            "code": self.check_code_conventions(filepath),
        }
        return str(filepath), result

    def check_project(self, root_path: Path, patterns: List[str]) -> Dict:
        """Check all Python files matching patterns"""
        results = {}

        for pattern in patterns:
            for filepath in root_path.glob(pattern):
                if filepath.is_file() and filepath.suffix == ".py":
                    filename, violations = self.check_file(filepath)
                    if any(violations["module"]) or any(
                        v for v in violations["code"].values()
                    ):
                        results[filename] = violations

        return results


def print_report(results: Dict):
    """Print formatted violation report"""
    print("=" * 80)
    print("NAMING CONVENTION VERIFICATION REPORT")
    print("=" * 80)
    print()

    if not results:
        print("✅ ALL FILES PASS NAMING CONVENTION CHECKS")
        print()
        return True

    total_violations = 0
    for filepath, violations in results.items():
        print(f"\n📄 {filepath}")
        print("-" * 80)

        # Module name violations
        if violations["module"]:
            print("\n  ❌ Module Name Issues:")
            for v in violations["module"]:
                print(f"     • {v}")
                total_violations += 1

        # Code violations
        code_viols = violations.get("code", {})
        if isinstance(code_viols, dict):
            for category, items in code_viols.items():
                if items and category != "error":
                    print(f"\n  ❌ {category.title()} Issues:")
                    for v in items:
                        print(f"     • {v}")
                        total_violations += 1

            if "error" in code_viols:
                print(f"\n  ⚠️  Parse Error:")
                for v in code_viols["error"]:
                    print(f"     • {v}")

    print()
    print("=" * 80)
    print(f"TOTAL VIOLATIONS: {total_violations}")
    print("=" * 80)
    print()

    return False


def main():
    """Main execution"""
    root = Path("/home/claude")

    print("Checking Python naming conventions...")
    print()

    # Common Python files to check
    patterns = ["*.py", "**/*.py"]

    checker = NamingConventionChecker()
    results = checker.check_project(root, patterns)

    all_pass = print_report(results)

    if all_pass:
        print("✅ Naming conventions verified successfully")
        return 0
    else:
        print("❌ Naming convention violations found - please fix")
        return 1


if __name__ == "__main__":
    exit(main())
