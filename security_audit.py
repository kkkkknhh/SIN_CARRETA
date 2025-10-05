#!/usr/bin/env python3
"""
Security Audit Module
====================
Static analysis tools to detect dangerous patterns and ensure code safety.
Prevents future introduction of unsafe practices.
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum


class SecurityLevel(Enum):
    """Security issue severity levels"""
    CRITICAL = "CRITICAL"  # Immediate security risk
    HIGH = "HIGH"          # Major security concern
    MEDIUM = "MEDIUM"      # Moderate risk
    LOW = "LOW"            # Minor security issue
    INFO = "INFO"          # Informational


@dataclass
class SecurityIssue:
    """Represents a security issue found in code"""
    level: SecurityLevel
    issue_type: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str


class SecurityAuditor:
    """
    Static analysis security auditor for Python code.
    Detects dangerous patterns and provides safe alternatives.
    """

    # Dangerous patterns to detect
    DANGEROUS_FUNCTIONS = {
        'eval': 'Use ast.literal_eval() for safe literal evaluation',
        'exec': 'Use function mapping or importlib for dynamic execution',
        'compile': 'Avoid dynamic code compilation when possible',
        '__import__': 'Use importlib.import_module() instead',
        'pickle.loads': 'Use JSON or safer serialization formats',
        'pickle.load': 'Use JSON or safer serialization formats',
        'yaml.load': 'Use yaml.safe_load() instead',
        'subprocess.call': 'Use subprocess.run() with shell=False',
        'os.system': 'Use subprocess.run() instead',
    }

    DANGEROUS_IMPORTS = {
        'pickle': 'Consider safer alternatives like JSON',
        'marshal': 'Avoid marshal for untrusted data',
        'shelve': 'Be cautious with shelve, prefer database',
    }

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues: List[SecurityIssue] = []

    def audit_project(self) -> Dict[str, Any]:
        """
        Audit entire project for security issues.

        Returns:
            Dictionary with audit results
        """
        print(f"\n🔍 Starting security audit of {self.project_root}")

        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            # Skip virtual environments and cache directories
            if any(skip in str(file_path) for skip in ['venv', '.venv', '__pycache__', 'site-packages']):
                continue

            self.audit_file(file_path)

        return self.generate_report()

    def audit_file(self, file_path: Path):
        """Audit a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Parse AST
            try:
                tree = ast.parse(content, filename=str(file_path))
                self._check_ast(tree, file_path, lines)
            except SyntaxError as e:
                # File has syntax errors, skip AST analysis
                pass

            # Pattern-based checks (regex)
            self._check_patterns(content, file_path, lines)

        except Exception as e:
            print(f"Warning: Could not audit {file_path}: {e}")

    def _check_ast(self, tree: ast.AST, file_path: Path, lines: List[str]):
        """Check AST for dangerous patterns"""

        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)

                if func_name in self.DANGEROUS_FUNCTIONS:
                    self.issues.append(SecurityIssue(
                        level=SecurityLevel.CRITICAL if func_name in ['eval', 'exec'] else SecurityLevel.HIGH,
                        issue_type=f"Dangerous function: {func_name}",
                        description=f"Use of {func_name}() detected",
                        file_path=str(file_path),
                        line_number=node.lineno if hasattr(node, 'lineno') else 0,
                        code_snippet=lines[node.lineno - 1].strip() if hasattr(node, 'lineno') and node.lineno <= len(lines) else "",
                        recommendation=self.DANGEROUS_FUNCTIONS[func_name]
                    ))

            # Check for dangerous imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.DANGEROUS_IMPORTS:
                        self.issues.append(SecurityIssue(
                            level=SecurityLevel.MEDIUM,
                            issue_type=f"Risky import: {alias.name}",
                            description=f"Import of {alias.name} module",
                            file_path=str(file_path),
                            line_number=node.lineno if hasattr(node, 'lineno') else 0,
                            code_snippet=lines[node.lineno - 1].strip() if hasattr(node, 'lineno') and node.lineno <= len(lines) else "",
                            recommendation=self.DANGEROUS_IMPORTS[alias.name]
                        ))

            elif isinstance(node, ast.ImportFrom):
                if node.module in self.DANGEROUS_IMPORTS:
                    self.issues.append(SecurityIssue(
                        level=SecurityLevel.MEDIUM,
                        issue_type=f"Risky import: {node.module}",
                        description=f"Import from {node.module} module",
                        file_path=str(file_path),
                        line_number=node.lineno if hasattr(node, 'lineno') else 0,
                        code_snippet=lines[node.lineno - 1].strip() if hasattr(node, 'lineno') and node.lineno <= len(lines) else "",
                        recommendation=self.DANGEROUS_IMPORTS[node.module]
                    ))

    def _check_patterns(self, content: str, file_path: Path, lines: List[str]):
        """Check for dangerous patterns using regex"""

        # Check for SQL injection vulnerabilities
        sql_pattern = r'execute\s*\(\s*["\'].*%s.*["\']'
        for i, line in enumerate(lines, 1):
            if re.search(sql_pattern, line):
                self.issues.append(SecurityIssue(
                    level=SecurityLevel.HIGH,
                    issue_type="Potential SQL injection",
                    description="String formatting in SQL query detected",
                    file_path=str(file_path),
                    line_number=i,
                    code_snippet=line.strip(),
                    recommendation="Use parameterized queries instead"
                ))

        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'secret[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded secret key"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, desc in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip if it's clearly a placeholder
                    if any(placeholder in line.lower() for placeholder in ['your_', 'example', 'placeholder', 'xxx', '***']):
                        continue

                    self.issues.append(SecurityIssue(
                        level=SecurityLevel.HIGH,
                        issue_type="Hardcoded secret",
                        description=desc,
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip()[:80] + "..." if len(line.strip()) > 80 else line.strip(),
                        recommendation="Use environment variables or secure vault"
                    ))

        # Check for shell=True in subprocess
        if 'shell=True' in content:
            for i, line in enumerate(lines, 1):
                if 'shell=True' in line:
                    self.issues.append(SecurityIssue(
                        level=SecurityLevel.HIGH,
                        issue_type="Shell injection risk",
                        description="subprocess called with shell=True",
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        recommendation="Use shell=False and pass command as list"
                    ))

    def _get_function_name(self, node: ast.AST) -> str:
        """Extract function name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_function_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return ""

    def generate_report(self) -> Dict[str, Any]:
        """Generate security audit report"""

        # Group issues by severity
        by_severity = {}
        for level in SecurityLevel:
            by_severity[level.value] = [
                issue for issue in self.issues
                if issue.level == level
            ]

        # Group issues by type
        by_type = {}
        for issue in self.issues:
            if issue.issue_type not in by_type:
                by_type[issue.issue_type] = []
            by_type[issue.issue_type].append(issue)

        report = {
            "total_issues": len(self.issues),
            "by_severity": {
                level: len(issues)
                for level, issues in by_severity.items()
            },
            "by_type": {
                issue_type: len(issues)
                for issue_type, issues in by_type.items()
            },
            "critical_issues": [
                {
                    "type": issue.issue_type,
                    "file": issue.file_path,
                    "line": issue.line_number,
                    "description": issue.description,
                    "recommendation": issue.recommendation
                }
                for issue in self.issues
                if issue.level in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]
            ]
        }

        return report

    def print_report(self):
        """Print formatted security report"""
        report = self.generate_report()

        print("\n" + "="*80)
        print("SECURITY AUDIT REPORT")
        print("="*80)

        print(f"\nTotal Issues Found: {report['total_issues']}")

        print("\nBy Severity:")
        for level, count in report['by_severity'].items():
            if count > 0:
                emoji = "🔴" if level == "CRITICAL" else "🟠" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🟢"
                print(f"  {emoji} {level}: {count}")

        if report['critical_issues']:
            print("\n" + "="*80)
            print("CRITICAL AND HIGH SEVERITY ISSUES")
            print("="*80)

            for i, issue in enumerate(report['critical_issues'], 1):
                print(f"\n{i}. {issue['type']}")
                print(f"   File: {issue['file']}:{issue['line']}")
                print(f"   Description: {issue['description']}")
                print(f"   ✅ Recommendation: {issue['recommendation']}")

        if report['total_issues'] == 0:
            print("\n✅ No security issues detected!")

        print("\n" + "="*80)


class SafeAlternatives:
    """
    Provides safe alternatives to dangerous operations.
    """

    @staticmethod
    def safe_literal_eval(expression: str) -> Any:
        """
        Safe alternative to eval() for literal expressions.
        Only evaluates Python literals (strings, numbers, tuples, lists, dicts, booleans, None).
        """
        import ast
        return ast.literal_eval(expression)

    @staticmethod
    def safe_dynamic_import(module_name: str):
        """Safe alternative to __import__()"""
        import importlib
        return importlib.import_module(module_name)

    @staticmethod
    def safe_function_mapping(function_name: str, function_map: Dict[str, callable]):
        """
        Safe alternative to dynamic execution using function mapping.

        Example:
            function_map = {
                'process': process_function,
                'validate': validate_function,
            }
            result = safe_function_mapping('process', function_map)
        """
        if function_name not in function_map:
            raise ValueError(f"Unknown function: {function_name}")

        return function_map[function_name]

    @staticmethod
    def safe_subprocess_run(command: List[str], **kwargs):
        """Safe subprocess execution without shell injection"""
        import subprocess
        # Force shell=False
        kwargs['shell'] = False
        # Capture output by default
        if 'capture_output' not in kwargs:
            kwargs['capture_output'] = True

        return subprocess.run(command, **kwargs)


def create_pre_commit_hook():
    """
    Create a pre-commit hook for security scanning.
    """
    hook_content = """#!/bin/bash
# Pre-commit security scan

echo "Running security audit..."
python3 security_audit.py --quick

if [ $? -ne 0 ]; then
    echo "❌ Security issues detected. Commit blocked."
    echo "Run 'python3 security_audit.py' for details."
    exit 1
fi

echo "✅ Security audit passed"
"""

    hook_path = Path(".git/hooks/pre-commit")
    if hook_path.parent.exists():
        with open(hook_path, 'w') as f:
            f.write(hook_content)
        os.chmod(hook_path, 0o755)
        print(f"✅ Pre-commit hook created at {hook_path}")
    else:
        print("ℹ️  No .git directory found. Hook not created.")


if __name__ == "__main__":
    import sys

    # Determine project root
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])
    else:
        project_root = Path(__file__).parent

    # Run security audit
    auditor = SecurityAuditor(project_root)
    auditor.audit_project()
    auditor.print_report()

    # Exit with error code if critical issues found
    report = auditor.generate_report()
    critical_count = report['by_severity'].get('CRITICAL', 0) + report['by_severity'].get('HIGH', 0)

    if critical_count > 0:
        print(f"\n⚠️  Found {critical_count} critical/high severity issues")
        sys.exit(1)
    else:
        print("\n✅ Security audit completed successfully")
        sys.exit(0)

