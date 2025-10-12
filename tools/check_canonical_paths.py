#!/usr/bin/env python3
"""
Canonical Path Checker

Scans the repository for non-canonical references to decalogo/DNP JSON files.
This script is used in CI to prevent regression to non-standard paths.

Exit codes:
  0 - All paths are canonical
  1 - Non-canonical references found
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any

# Repository root
ROOT = Path(__file__).resolve().parents[1]

# Patterns to detect non-canonical references
# These match any variant that is NOT in the canonical location
BAD_DEC = re.compile(
    r'(?i)(?:^|[/"\'\s])(?!bundles/)(?:[./]*)?'
    r'(?:deca[ln]ogo|decálogo)'
    r'(?:[-_.]?industrial)?'
    r'(?:[-_.]?latest)?'
    r'(?:[-_.]?clean)?'
    r'\.json',
    re.MULTILINE
)

BAD_DNP = re.compile(
    r'(?i)(?:^|[/"\'\s])(?!standards/)(?:[./]*)?'
    r'dnp[-_.]?standa?rds'
    r'(?:[-_.]?industrial)?'
    r'(?:[-_.]?latest)?'
    r'(?:[-_.]?clean)?'
    r'\.json',
    re.MULTILINE
)

# Files to exclude from scanning
EXCLUDE_PATTERNS = [
    '.git',
    '__pycache__',
    '.venv',
    'node_modules',
    '.pytest_cache',
    'tools/check_canonical_paths.py',  # Don't flag ourselves
    'repo_paths.py',  # Allow references in the resolver itself
]

# File extensions to scan
SCAN_EXTENSIONS = {'.py', '.ts', '.js', '.json', '.yaml', '.yml', '.md', '.sh', '.ini', '.toml', '.txt'}


def should_scan(path: Path) -> bool:
    """Check if a path should be scanned."""
    # Skip excluded patterns
    for pattern in EXCLUDE_PATTERNS:
        if pattern in str(path):
            return False
    
    # Only scan specific extensions
    return path.suffix in SCAN_EXTENSIONS


def scan_file(path: Path) -> List[Dict[str, Any]]:
    """
    Scan a file for non-canonical path references.
    
    Returns:
        List of findings with file, line, match, and kind
    """
    findings = []
    
    try:
        content = path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        # Skip files that can't be read
        return findings
    
    # Split into lines for better reporting
    lines = content.split('\n')
    
    for line_no, line in enumerate(lines, start=1):
        # Check for non-canonical decalogo references
        for match in BAD_DEC.finditer(line):
            matched_text = match.group(0).strip()
            # Skip if this is actually pointing to the canonical path
            if 'bundles/decalogo-industrial.latest.clean.json' in line:
                continue
            # Skip documentation about old paths
            if 'was:' in line.lower() or 'old:' in line.lower() or 'before:' in line.lower():
                continue
                
            findings.append({
                'file': str(path.relative_to(ROOT)),
                'line': line_no,
                'match': matched_text,
                'kind': 'DECALOGO',
                'context': line.strip()[:100]
            })
        
        # Check for non-canonical DNP references
        for match in BAD_DNP.finditer(line):
            matched_text = match.group(0).strip()
            # Skip if this is actually pointing to the canonical path
            if 'standards/dnp-standards.latest.clean.json' in line:
                continue
            # Skip documentation about old paths
            if 'was:' in line.lower() or 'old:' in line.lower() or 'before:' in line.lower():
                continue
                
            findings.append({
                'file': str(path.relative_to(ROOT)),
                'line': line_no,
                'match': matched_text,
                'kind': 'DNP',
                'context': line.strip()[:100]
            })
    
    return findings


def scan_repository() -> List[Dict[str, Any]]:
    """
    Scan the entire repository for non-canonical references.
    
    Returns:
        List of all findings
    """
    all_findings = []
    
    for path in ROOT.rglob('*'):
        if path.is_file() and should_scan(path):
            findings = scan_file(path)
            all_findings.extend(findings)
    
    return all_findings


def main():
    """Main entry point."""
    print("=" * 80)
    print("Canonical Path Checker")
    print("=" * 80)
    print(f"Scanning: {ROOT}")
    print()
    
    findings = scan_repository()
    
    if findings:
        print(f"❌ Found {len(findings)} non-canonical reference(s):")
        print()
        
        # Group by file
        by_file = {}
        for finding in findings:
            file = finding['file']
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(finding)
        
        for file, file_findings in sorted(by_file.items()):
            print(f"📁 {file}")
            for finding in file_findings:
                print(f"  Line {finding['line']}: [{finding['kind']}] {finding['match']}")
                print(f"    Context: {finding['context']}")
            print()
        
        print("=" * 80)
        print(f"❌ FAILED: {len(findings)} non-canonical references found")
        print()
        print("Expected paths:")
        print("  - /bundles/decalogo-industrial.latest.clean.json")
        print("  - /standards/dnp-standards.latest.clean.json")
        print()
        sys.exit(1)
    else:
        print("✅ OK: All paths are canonical")
        print()
        print("Verified paths:")
        print("  - /bundles/decalogo-industrial.latest.clean.json")
        print("  - /standards/dnp-standards.latest.clean.json")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
