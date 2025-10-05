#!/usr/bin/env python3
"""
Audit eval() Function Calls
============================
Catalogs all eval() function calls in the codebase, documenting their purpose
and whether ast.literal_eval or dispatch tables could replace them.
"""

import ast
import os
from pathlib import Path
from typing import List, Dict, Any
import json


class EvalCallDetector(ast.NodeVisitor):
    """AST visitor to detect eval() calls"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.eval_calls = []
    
    def visit_Call(self, node):
        """Visit call nodes and check if they're eval()"""
        # Check if it's a direct eval() call
        if isinstance(node.func, ast.Name) and node.func.id == 'eval':
            context = self._get_context(node)
            self.eval_calls.append({
                "line": node.lineno,
                "col": node.col_offset,
                "context": context,
                "args": len(node.args)
            })
        
        # Continue visiting child nodes
        self.generic_visit(node)
    
    def _get_context(self, node):
        """Try to get the context around the eval() call"""
        # This is simplified - in a real implementation, you'd
        # need to track more context
        return "eval() call detected"


def scan_file_for_eval(filepath: Path) -> List[Dict[str, Any]]:
    """Scan a single Python file for eval() calls"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content, filename=str(filepath))
        
        # Visit nodes
        detector = EvalCallDetector(str(filepath))
        detector.visit(tree)
        
        # Also do a simple text search for 'eval('
        lines = content.split('\n')
        text_matches = []
        for i, line in enumerate(lines, 1):
            if 'eval(' in line and not line.strip().startswith('#'):
                # Skip if it's .eval() (method call)
                if '.eval(' not in line:
                    text_matches.append({
                        "line": i,
                        "content": line.strip(),
                        "is_comment": False
                    })
        
        return detector.eval_calls, text_matches
        
    except SyntaxError as e:
        return [], []
    except Exception as e:
        return [], []


def analyze_eval_call(filepath: str, line_num: int, line_content: str) -> Dict[str, Any]:
    """Analyze a specific eval() call and suggest alternatives"""
    analysis = {
        "file": filepath,
        "line": line_num,
        "code": line_content,
        "purpose": "Unknown",
        "risk_level": "HIGH",
        "can_use_literal_eval": False,
        "can_use_dispatch_table": False,
        "recommendations": []
    }
    
    # Heuristic analysis
    lower_content = line_content.lower()
    
    # Check if it might be evaluating literals
    if any(x in lower_content for x in ['json', 'literal', 'dict', 'list', 'tuple']):
        analysis["purpose"] = "Likely parsing literal data structures"
        analysis["can_use_literal_eval"] = True
        analysis["recommendations"].append(
            "SAFE ALTERNATIVE: Use ast.literal_eval() instead of eval() for literal evaluation"
        )
        analysis["risk_level"] = "MEDIUM"
    
    # Check if it might be dynamic function dispatch
    if any(x in lower_content for x in ['function', 'method', 'call', 'dispatch']):
        analysis["purpose"] = "Possible dynamic function dispatch"
        analysis["can_use_dispatch_table"] = True
        analysis["recommendations"].append(
            "SAFE ALTERNATIVE: Use a dispatch table (dict of functions) instead of eval()"
        )
        analysis["risk_level"] = "CRITICAL"
    
    # Check if it's in a string context
    if 'str' in lower_content or 'format' in lower_content:
        analysis["purpose"] = "String evaluation or formatting"
        analysis["recommendations"].append(
            "SAFE ALTERNATIVE: Use string formatting or template engines"
        )
        analysis["risk_level"] = "HIGH"
    
    # Check if it's for configuration
    if 'config' in lower_content or 'settings' in lower_content:
        analysis["purpose"] = "Configuration value parsing"
        analysis["can_use_literal_eval"] = True
        analysis["recommendations"].append(
            "SAFE ALTERNATIVE: Use json.loads() or configparser for configuration"
        )
        analysis["risk_level"] = "HIGH"
    
    # Default recommendations if none added
    if not analysis["recommendations"]:
        analysis["recommendations"] = [
            "REVIEW REQUIRED: Manual inspection needed to determine safe alternative",
            "CONSIDER: Refactoring to avoid dynamic code execution",
            "SECURITY: eval() allows arbitrary code execution - high security risk"
        ]
    
    return analysis


def scan_codebase(root_path: str = ".") -> Dict[str, Any]:
    """Scan entire codebase for eval() calls"""
    print("Scanning codebase for eval() calls...")
    print("=" * 80)
    
    root = Path(root_path)
    all_findings = []
    files_scanned = 0
    files_with_eval = 0
    
    # Get all Python files
    python_files = list(root.glob("**/*.py"))
    
    # Exclude venv and other common directories
    python_files = [
        f for f in python_files 
        if not any(part in f.parts for part in ['venv', '.venv', 'site-packages', '.git', '__pycache__'])
    ]
    
    for filepath in python_files:
        files_scanned += 1
        ast_calls, text_matches = scan_file_for_eval(filepath)
        
        if text_matches:
            files_with_eval += 1
            for match in text_matches:
                analysis = analyze_eval_call(
                    str(filepath.relative_to(root)),
                    match["line"],
                    match["content"]
                )
                all_findings.append(analysis)
    
    # Summary statistics
    summary = {
        "files_scanned": files_scanned,
        "files_with_eval": files_with_eval,
        "total_eval_calls": len(all_findings),
        "risk_breakdown": {
            "CRITICAL": len([f for f in all_findings if f["risk_level"] == "CRITICAL"]),
            "HIGH": len([f for f in all_findings if f["risk_level"] == "HIGH"]),
            "MEDIUM": len([f for f in all_findings if f["risk_level"] == "MEDIUM"]),
            "LOW": len([f for f in all_findings if f["risk_level"] == "LOW"])
        },
        "can_use_literal_eval": len([f for f in all_findings if f["can_use_literal_eval"]]),
        "can_use_dispatch_table": len([f for f in all_findings if f["can_use_dispatch_table"]])
    }
    
    return {
        "summary": summary,
        "findings": all_findings
    }


def print_report(results: Dict[str, Any]):
    """Print human-readable report"""
    summary = results["summary"]
    findings = results["findings"]
    
    print("\n" + "=" * 80)
    print("EVAL() AUDIT REPORT")
    print("=" * 80)
    
    print(f"\nFiles scanned: {summary['files_scanned']}")
    print(f"Files with eval(): {summary['files_with_eval']}")
    print(f"Total eval() calls found: {summary['total_eval_calls']}")
    
    print("\nRisk Level Breakdown:")
    print("─" * 60)
    for level, count in summary["risk_breakdown"].items():
        if count > 0:
            print(f"  {level:10s}: {count:3d} calls")
    
    print(f"\nSafe Alternative Applicability:")
    print("─" * 60)
    print(f"  Can use ast.literal_eval(): {summary['can_use_literal_eval']} calls")
    print(f"  Can use dispatch table:     {summary['can_use_dispatch_table']} calls")
    
    if findings:
        print("\n" + "=" * 80)
        print("DETAILED FINDINGS")
        print("=" * 80)
        
        # Group by file
        by_file = {}
        for finding in findings:
            file = finding["file"]
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(finding)
        
        for file, file_findings in sorted(by_file.items()):
            print(f"\n📄 {file}")
            print("─" * 80)
            
            for finding in file_findings:
                print(f"\n  Line {finding['line']}: [{finding['risk_level']}]")
                print(f"  Code: {finding['code'][:100]}")
                print(f"  Purpose: {finding['purpose']}")
                
                if finding['can_use_literal_eval']:
                    print(f"  ✅ Can use ast.literal_eval()")
                if finding['can_use_dispatch_table']:
                    print(f"  ✅ Can use dispatch table")
                
                print(f"  Recommendations:")
                for rec in finding['recommendations']:
                    print(f"    • {rec}")
    else:
        print("\n✅ No eval() calls found in codebase!")
    
    print("\n" + "=" * 80)


def generate_safe_alternatives_guide():
    """Generate a guide for safe alternatives to eval()"""
    guide = """
# Safe Alternatives to eval()

## 1. ast.literal_eval() - For Literal Data Structures
Use when you need to parse Python literals (strings, numbers, tuples, lists, dicts, booleans, None).

```python
# UNSAFE
data = eval("{'key': 'value'}")

# SAFE
import ast
data = ast.literal_eval("{'key': 'value'}")
```

## 2. json.loads() - For JSON Data
Use when parsing JSON data.

```python
# UNSAFE
data = eval(json_string)

# SAFE
import json
data = json.loads(json_string)
```

## 3. Dispatch Tables - For Dynamic Function Calls
Use when you need to call different functions based on input.

```python
# UNSAFE
result = eval(f"{function_name}()")

# SAFE
DISPATCH_TABLE = {
    'function1': function1,
    'function2': function2,
}
result = DISPATCH_TABLE[function_name]()
```

## 4. getattr() - For Dynamic Attribute Access
Use when accessing object attributes dynamically.

```python
# UNSAFE
value = eval(f"obj.{attr_name}")

# SAFE
value = getattr(obj, attr_name)
```

## 5. operator module - For Mathematical Operations
Use for safe mathematical operations.

```python
# UNSAFE
result = eval(f"{a} {op} {b}")

# SAFE
import operator
ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
result = ops[op](a, b)
```

## Security Considerations
- eval() executes arbitrary code and is a major security risk
- Always validate and sanitize input before any dynamic execution
- Use the principle of least privilege - don't use eval() when safer alternatives exist
- Consider using a restricted execution environment if dynamic code execution is absolutely necessary
"""
    return guide


def main():
    """Main execution"""
    # Scan codebase
    results = scan_codebase(".")
    
    # Print report
    print_report(results)
    
    # Save results to JSON
    with open("eval_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Audit results saved to eval_audit_results.json")
    
    # Save safe alternatives guide
    guide = generate_safe_alternatives_guide()
    with open("safe_alternatives_to_eval.md", "w") as f:
        f.write(guide)
    
    print("✅ Safe alternatives guide saved to safe_alternatives_to_eval.md")


if __name__ == "__main__":
    main()
