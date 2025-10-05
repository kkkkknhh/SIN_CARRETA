#!/usr/bin/env python3
"""
Module Management and Cleanup Tool
==================================
Addresses MEDIUM PRIORITY issues:
- Audit 39 orphaned modules
- Delete truly orphaned test modules
- Document purpose of retained modules
- Fix missing dependency links
- Archive deprecated modules
- Update dependency graph

Provides automated analysis and recommendations for module lifecycle management.
"""

import ast
import os
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import re


@dataclass
class ModuleInfo:
    """Information about a module"""
    name: str
    path: Path
    size_bytes: int
    line_count: int
    last_modified: datetime
    imports: Set[str] = field(default_factory=set)
    imported_by: Set[str] = field(default_factory=set)
    has_tests: bool = False
    has_main: bool = False
    doc_string: Optional[str] = None
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)


@dataclass
class ModuleClassification:
    """Classification of module status"""
    category: str  # orphaned, test, utility, core, deprecated
    reason: str
    recommendation: str
    confidence: float  # 0.0 to 1.0


class ModuleManager:
    """
    Comprehensive module management and cleanup tool.
    Analyzes orphaned modules and provides actionable recommendations.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.modules: Dict[str, ModuleInfo] = {}
        self.orphaned_modules: List[str] = []
        self.classifications: Dict[str, ModuleClassification] = {}

    def analyze_project(self) -> Dict[str, any]:
        """
        Analyze entire project for module management.
        """
        print("\n" + "="*80)
        print("MODULE MANAGEMENT ANALYSIS")
        print("="*80 + "\n")

        # Step 1: Scan all Python modules
        print("Step 1: Scanning Python modules...")
        self._scan_modules()
        print(f"  Found {len(self.modules)} modules")

        # Step 2: Build dependency graph
        print("\nStep 2: Building dependency graph...")
        self._build_dependency_graph()

        # Step 3: Identify orphaned modules
        print("\nStep 3: Identifying orphaned modules...")
        self._identify_orphaned()
        print(f"  Found {len(self.orphaned_modules)} orphaned modules")

        # Step 4: Classify modules
        print("\nStep 4: Classifying modules...")
        self._classify_modules()

        # Step 5: Generate recommendations
        print("\nStep 5: Generating recommendations...")
        report = self._generate_report()

        return report

    def _scan_modules(self):
        """Scan all Python modules in project"""
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            # Skip virtual environments and cache
            if any(skip in str(file_path) for skip in ['venv', '.venv', '__pycache__', 'site-packages']):
                continue

            module_name = file_path.stem

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                # Parse AST
                try:
                    tree = ast.parse(content)
                    doc_string = ast.get_docstring(tree)
                    imports = self._extract_imports(tree)
                    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                    has_main = any(line.strip() == 'if __name__ == "__main__":' for line in lines)
                except:
                    doc_string = None
                    imports = set()
                    functions = []
                    classes = []
                    has_main = False

                # Create module info
                self.modules[module_name] = ModuleInfo(
                    name=module_name,
                    path=file_path,
                    size_bytes=file_path.stat().st_size,
                    line_count=len(lines),
                    last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                    imports=imports,
                    has_tests=self._is_test_module(module_name),
                    has_main=has_main,
                    doc_string=doc_string,
                    functions=functions,
                    classes=classes
                )

            except Exception as e:
                print(f"  Warning: Could not analyze {file_path}: {e}")

    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract imported module names"""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])

        return imports

    def _is_test_module(self, name: str) -> bool:
        """Check if module is a test module"""
        return name.startswith('test_') or name.endswith('_test') or 'test' in name.lower()

    def _build_dependency_graph(self):
        """Build dependency graph between modules"""
        # First pass: collect all module names
        module_names = set(self.modules.keys())

        # Second pass: link imports to modules
        for module_name, module_info in self.modules.items():
            for imported in module_info.imports:
                if imported in module_names:
                    # Add to imported_by set
                    if imported in self.modules:
                        self.modules[imported].imported_by.add(module_name)

    def _identify_orphaned(self):
        """Identify orphaned modules (no imports and not imported by anyone)"""
        for module_name, module_info in self.modules.items():
            # A module is orphaned if:
            # 1. It's not imported by any other module
            # 2. It doesn't import any other project modules
            # 3. It's not a main script (no if __name__ == "__main__")

            imports_project_modules = bool(module_info.imports & set(self.modules.keys()))
            is_imported = bool(module_info.imported_by)

            if not is_imported and not imports_project_modules and not module_info.has_main:
                self.orphaned_modules.append(module_name)

    def _classify_modules(self):
        """Classify all orphaned modules"""
        for module_name in self.orphaned_modules:
            module_info = self.modules[module_name]
            classification = self._classify_single_module(module_info)
            self.classifications[module_name] = classification

    def _classify_single_module(self, module_info: ModuleInfo) -> ModuleClassification:
        """Classify a single module"""

        # Test modules
        if module_info.has_tests or module_info.name.startswith('test_'):
            # Check if it's a standalone test
            if not module_info.imported_by and module_info.has_main:
                return ModuleClassification(
                    category="standalone_test",
                    reason="Standalone test module with main block",
                    recommendation="KEEP - Standalone test, can be run directly",
                    confidence=0.9
                )
            else:
                return ModuleClassification(
                    category="orphaned_test",
                    reason="Test module not connected to test suite",
                    recommendation="REVIEW - May be obsolete test or needs integration",
                    confidence=0.7
                )

        # Demo/Example modules
        if 'demo' in module_info.name.lower() or 'example' in module_info.name.lower():
            if module_info.has_main:
                return ModuleClassification(
                    category="demo",
                    reason="Demo/example module with main block",
                    recommendation="KEEP - Documentation/example code",
                    confidence=0.85
                )
            else:
                return ModuleClassification(
                    category="orphaned_demo",
                    reason="Demo module without main block",
                    recommendation="REVIEW - May be incomplete or obsolete",
                    confidence=0.6
                )

        # Setup/configuration modules
        if module_info.name in ['setup', 'config', 'settings', 'configuration']:
            return ModuleClassification(
                category="configuration",
                reason="Configuration or setup module",
                recommendation="KEEP - Likely used for installation/setup",
                confidence=0.8
            )

        # Utility modules
        if module_info.name in ['utils', 'helpers', 'common', 'shared']:
            if module_info.functions or module_info.classes:
                return ModuleClassification(
                    category="utility",
                    reason=f"Utility module with {len(module_info.functions)} functions",
                    recommendation="REVIEW - May need to be imported explicitly",
                    confidence=0.7
                )

        # Old/deprecated modules
        age_days = (datetime.now() - module_info.last_modified).days
        if age_days > 180:  # 6 months
            return ModuleClassification(
                category="potentially_deprecated",
                reason=f"Not modified in {age_days} days",
                recommendation="ARCHIVE - Old and unused",
                confidence=0.6
            )

        # Small/empty modules
        if module_info.line_count < 20:
            return ModuleClassification(
                category="minimal",
                reason=f"Very small module ({module_info.line_count} lines)",
                recommendation="DELETE - Likely obsolete or empty",
                confidence=0.8
            )

        # Default: needs manual review
        return ModuleClassification(
            category="unknown",
            reason="Could not automatically classify",
            recommendation="REVIEW - Manual inspection needed",
            confidence=0.5
        )

    def _generate_report(self) -> Dict[str, any]:
        """Generate comprehensive module management report"""

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_modules": len(self.modules),
                "orphaned_modules": len(self.orphaned_modules),
                "test_modules": sum(1 for m in self.modules.values() if m.has_tests),
                "modules_with_main": sum(1 for m in self.modules.values() if m.has_main),
            },
            "classifications": {},
            "recommendations": {
                "KEEP": [],
                "REVIEW": [],
                "ARCHIVE": [],
                "DELETE": [],
            },
            "orphaned_details": []
        }

        # Group by classification
        by_category = defaultdict(list)
        for module_name, classification in self.classifications.items():
            by_category[classification.category].append(module_name)

        report["classifications"] = {
            category: len(modules)
            for category, modules in by_category.items()
        }

        # Organize recommendations
        for module_name, classification in self.classifications.items():
            module_info = self.modules[module_name]

            detail = {
                "name": module_name,
                "path": str(module_info.path.relative_to(self.project_root)),
                "category": classification.category,
                "reason": classification.reason,
                "recommendation": classification.recommendation,
                "confidence": classification.confidence,
                "size_bytes": module_info.size_bytes,
                "line_count": module_info.line_count,
                "last_modified": module_info.last_modified.isoformat(),
            }

            report["orphaned_details"].append(detail)

            # Add to recommendation buckets
            if "KEEP" in classification.recommendation:
                report["recommendations"]["KEEP"].append(module_name)
            elif "ARCHIVE" in classification.recommendation:
                report["recommendations"]["ARCHIVE"].append(module_name)
            elif "DELETE" in classification.recommendation:
                report["recommendations"]["DELETE"].append(module_name)
            else:
                report["recommendations"]["REVIEW"].append(module_name)

        return report

    def save_report(self, report: Dict[str, any], output_path: Path):
        """Save report to file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {output_path}")

    def print_report(self, report: Dict[str, any]):
        """Print formatted report"""
        print("\n" + "="*80)
        print("MODULE MANAGEMENT REPORT")
        print("="*80)

        print(f"\n📊 SUMMARY:")
        print(f"  Total modules: {report['summary']['total_modules']}")
        print(f"  Orphaned modules: {report['summary']['orphaned_modules']}")
        print(f"  Test modules: {report['summary']['test_modules']}")
        print(f"  Modules with main: {report['summary']['modules_with_main']}")

        print(f"\n📋 CLASSIFICATIONS:")
        for category, count in sorted(report['classifications'].items()):
            print(f"  {category}: {count}")

        print(f"\n✅ RECOMMENDATIONS:")
        for action, modules in report['recommendations'].items():
            if modules:
                print(f"  {action}: {len(modules)} modules")

        print(f"\n🔍 TOP RECOMMENDATIONS:")

        # Show high-confidence deletions
        deletions = [d for d in report['orphaned_details']
                    if 'DELETE' in d['recommendation'] and d['confidence'] > 0.7]
        if deletions:
            print(f"\n  High-confidence deletions ({len(deletions)}):")
            for detail in deletions[:5]:
                print(f"    • {detail['name']} - {detail['reason']}")

        # Show archives
        archives = [d for d in report['orphaned_details']
                   if 'ARCHIVE' in d['recommendation']]
        if archives:
            print(f"\n  Recommended for archival ({len(archives)}):")
            for detail in archives[:5]:
                age_days = (datetime.now() - datetime.fromisoformat(detail['last_modified'])).days
                print(f"    • {detail['name']} - Not modified in {age_days} days")

        # Show keeps
        keeps = [d for d in report['orphaned_details']
                if 'KEEP' in d['recommendation']]
        if keeps:
            print(f"\n  Should keep ({len(keeps)}):")
            for detail in keeps[:5]:
                print(f"    • {detail['name']} - {detail['reason']}")

    def execute_cleanup(self, report: Dict[str, any], dry_run: bool = True):
        """
        Execute cleanup based on report recommendations.

        Args:
            report: Module management report
            dry_run: If True, only show what would be done (default: True)
        """
        print("\n" + "="*80)
        print(f"CLEANUP EXECUTION {'(DRY RUN)' if dry_run else '(LIVE)'}")
        print("="*80)

        # Archive old modules
        archive_dir = self.project_root / "archived_modules"
        if not dry_run and not archive_dir.exists():
            archive_dir.mkdir()

        for detail in report['orphaned_details']:
            module_name = detail['name']
            module_info = self.modules[module_name]

            if 'DELETE' in detail['recommendation'] and detail['confidence'] > 0.8:
                if dry_run:
                    print(f"  [DRY RUN] Would delete: {module_name}")
                else:
                    module_info.path.unlink()
                    print(f"  ✅ Deleted: {module_name}")

            elif 'ARCHIVE' in detail['recommendation']:
                if dry_run:
                    print(f"  [DRY RUN] Would archive: {module_name}")
                else:
                    dest = archive_dir / module_info.path.name
                    module_info.path.rename(dest)
                    print(f"  📦 Archived: {module_name}")

        if dry_run:
            print("\n⚠️  This was a DRY RUN. No files were actually modified.")
            print("   Run with dry_run=False to execute cleanup.")


def create_module_documentation(manager: ModuleManager, output_path: Path):
    """Create documentation for all kept modules"""
    doc_content = ["# Module Documentation\n"]
    doc_content.append(f"Generated: {datetime.now().isoformat()}\n\n")

    # Group modules by category
    test_modules = []
    demo_modules = []
    utility_modules = []
    core_modules = []

    for module_name, module_info in manager.modules.items():
        if module_info.has_tests:
            test_modules.append(module_info)
        elif 'demo' in module_name or 'example' in module_name:
            demo_modules.append(module_info)
        elif module_name in ['utils', 'helpers', 'common']:
            utility_modules.append(module_info)
        elif module_info.imported_by:
            core_modules.append(module_info)

    # Document each category
    if core_modules:
        doc_content.append("## Core Modules\n\n")
        for module in sorted(core_modules, key=lambda m: len(m.imported_by), reverse=True):
            doc_content.append(f"### {module.name}\n")
            if module.doc_string:
                doc_content.append(f"{module.doc_string[:200]}...\n\n" if len(module.doc_string) > 200 else f"{module.doc_string}\n\n")
            doc_content.append(f"- **Path**: `{module.path.relative_to(manager.project_root)}`\n")
            doc_content.append(f"- **Size**: {module.line_count} lines\n")
            doc_content.append(f"- **Imported by**: {len(module.imported_by)} modules\n")
            if module.classes:
                doc_content.append(f"- **Classes**: {', '.join(module.classes[:5])}\n")
            if module.functions:
                doc_content.append(f"- **Functions**: {len(module.functions)} functions\n")
            doc_content.append("\n")

    if test_modules:
        doc_content.append("## Test Modules\n\n")
        for module in sorted(test_modules, key=lambda m: m.name):
            doc_content.append(f"- `{module.name}` - {module.line_count} lines\n")
        doc_content.append("\n")

    if demo_modules:
        doc_content.append("## Demo/Example Modules\n\n")
        for module in sorted(demo_modules, key=lambda m: m.name):
            doc_content.append(f"- `{module.name}` - {module.doc_string[:100] if module.doc_string else 'No description'}...\n")
        doc_content.append("\n")

    # Write documentation
    with open(output_path, 'w') as f:
        f.writelines(doc_content)

    print(f"📚 Module documentation saved to: {output_path}")


if __name__ == "__main__":
    import sys

    # Get project root
    project_root = Path(__file__).parent if "__file__" in globals() else Path(".")

    # Create manager
    manager = ModuleManager(project_root)

    # Analyze project
    report = manager.analyze_project()

    # Print report
    manager.print_report(report)

    # Save report
    report_path = project_root / "module_management_report.json"
    manager.save_report(report, report_path)

    # Create module documentation
    doc_path = project_root / "MODULE_DOCUMENTATION.md"
    create_module_documentation(manager, doc_path)

    print("\n" + "="*80)
    print("MODULE MANAGEMENT ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review module_management_report.json")
    print("  2. Check MODULE_DOCUMENTATION.md")
    print("  3. Run cleanup with: manager.execute_cleanup(report, dry_run=False)")

    sys.exit(0)

