#!/usr/bin/env python3
"""
Import Analysis System
Comprehensive static analysis of all Python imports in MINIMINIMOON.

Features:
- Detect stdlib vs third-party modules
- Identify optional vs required imports
- Build dependency graphs
- Detect circular dependencies
- Generate detailed JSON reports with version information
"""

import ast
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any


# Standard library modules (Python 3.10+)
STDLIB_MODULES = {
    'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'copy',
    'csv', 'dataclasses', 'datetime', 'decimal', 'enum', 'functools',
    'hashlib', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
    'os', 'pathlib', 'pickle', 're', 'shutil', 'signal', 'statistics',
    'string', 'subprocess', 'sys', 'tempfile', 'textwrap', 'threading',
    'time', 'traceback', 'typing', 'unittest', 'uuid', 'warnings', 'weakref',
    'xml', 'zipfile', '__future__',
}


class ImportAnalyzer:
    """Analyze Python imports across the codebase."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.imports: Dict[str, Dict[str, Any]] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.circular_deps: List[List[str]] = []
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze imports in a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            return {
                'error': f'Syntax error: {e}',
                'imports': [],
                'from_imports': [],
            }
        except Exception as e:
            return {
                'error': f'Parse error: {e}',
                'imports': [],
                'from_imports': [],
            }
        
        imports = []
        from_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno,
                    })
                    
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    from_imports.append({
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'level': node.level,
                        'line': node.lineno,
                    })
        
        return {
            'imports': imports,
            'from_imports': from_imports,
        }
    
    def classify_import(self, module_name: str) -> Tuple[str, bool]:
        """
        Classify an import as stdlib, third-party, or local.
        
        Returns:
            Tuple of (classification, is_optional)
        """
        if not module_name:
            return 'local', False
        
        # Check if it's a relative import or local module
        if module_name.startswith('.'):
            return 'local', False
        
        # Get root module name
        root_module = module_name.split('.')[0]
        
        # Check if it's stdlib
        if root_module in STDLIB_MODULES:
            return 'stdlib', False
        
        # Check if it's a local module
        local_modules = {
            'embedding_model', 'responsibility_detector', 'contradiction_detector',
            'decalogo_loader', 'spacy_loader', 'plan_sanitizer', 'plan_processor',
            'document_segmenter', 'feasibility_scorer', 'monetary_detector',
            'causal_pattern_detector', 'teoria_cambio', 'dag_validation',
            'miniminimoon_orchestrator', 'evidence_registry', 'answer_assembler',
            'pdm_contra', 'factibilidad', 'evaluation', 'sin_carreta',
        }
        
        if root_module in local_modules:
            return 'local', False
        
        # Check if module is installed
        try:
            spec = importlib.util.find_spec(root_module)
            if spec is None:
                return 'third-party', True  # Not installed = optional
            
            # Try to import and get version
            module = importlib.import_module(root_module)
            has_version = hasattr(module, '__version__')
            
            return 'third-party', False
        except (ImportError, ModuleNotFoundError):
            return 'third-party', True  # Import error = optional
        except Exception:
            return 'third-party', False
    
    def get_module_version(self, module_name: str) -> Optional[str]:
        """Get installed version of a module."""
        try:
            module = importlib.import_module(module_name)
            return getattr(module, '__version__', None)
        except Exception:
            return None
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze all Python files in the codebase."""
        python_files = list(self.root_dir.rglob('*.py'))
        
        # Filter out virtual environments and build directories
        python_files = [
            f for f in python_files
            if not any(part.startswith('.') or part in {'build', 'dist', '__pycache__'}
                      for part in f.parts)
        ]
        
        all_imports = defaultdict(lambda: {
            'files': [],
            'classification': None,
            'is_optional': False,
            'version': None,
        })
        
        for py_file in python_files:
            rel_path = py_file.relative_to(self.root_dir)
            result = self.analyze_file(py_file)
            
            if 'error' in result:
                print(f"Warning: {rel_path}: {result['error']}", file=sys.stderr)
                continue
            
            # Process direct imports
            for imp in result['imports']:
                module = imp['module']
                all_imports[module]['files'].append({
                    'path': str(rel_path),
                    'line': imp['line'],
                    'type': 'import',
                })
                
                # Build dependency graph
                self.dependency_graph[str(rel_path)].add(module)
            
            # Process from imports
            for imp in result['from_imports']:
                module = imp['module']
                if module:  # Skip relative imports without module
                    all_imports[module]['files'].append({
                        'path': str(rel_path),
                        'line': imp['line'],
                        'type': 'from',
                        'name': imp['name'],
                    })
                    
                    # Build dependency graph
                    self.dependency_graph[str(rel_path)].add(module)
        
        # Classify all imports
        for module, data in all_imports.items():
            classification, is_optional = self.classify_import(module)
            data['classification'] = classification
            data['is_optional'] = is_optional
            
            if classification == 'third-party':
                root_module = module.split('.')[0]
                data['version'] = self.get_module_version(root_module)
        
        # Detect circular dependencies
        self.circular_deps = self._detect_cycles()
        
        return {
            'total_files': len(python_files),
            'imports': dict(all_imports),
            'dependency_graph': {k: list(v) for k, v in self.dependency_graph.items()},
            'circular_dependencies': self.circular_deps,
            'summary': self._generate_summary(all_imports),
        }
    
    def _detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            rec_stack.remove(node)
        
        for node in self.dependency_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def _generate_summary(self, all_imports: Dict) -> Dict[str, Any]:
        """Generate summary statistics."""
        by_classification = defaultdict(int)
        required_third_party = []
        optional_third_party = []
        
        for module, data in all_imports.items():
            classification = data['classification']
            by_classification[classification] += 1
            
            if classification == 'third-party':
                if data['is_optional']:
                    optional_third_party.append(module)
                else:
                    required_third_party.append(module)
        
        return {
            'by_classification': dict(by_classification),
            'required_third_party': sorted(set(required_third_party)),
            'optional_third_party': sorted(set(optional_third_party)),
            'circular_dependencies_count': len(self.circular_deps),
        }


def main() -> int:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze Python imports in MINIMINIMOON'
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path.cwd(),
        help='Root directory to analyze (default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('import_analysis.json'),
        help='Output JSON file (default: import_analysis.json)'
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing imports in: {args.root}")
    analyzer = ImportAnalyzer(args.root)
    results = analyzer.analyze_codebase()
    
    # Write results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = results['summary']
    print(f"\n{'='*70}")
    print("IMPORT ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total Python files analyzed: {results['total_files']}")
    print(f"\nImports by classification:")
    for classification, count in summary['by_classification'].items():
        print(f"  {classification:15s}: {count:4d}")
    
    print(f"\nRequired third-party packages: {len(summary['required_third_party'])}")
    for pkg in summary['required_third_party'][:10]:
        version = results['imports'][pkg].get('version', 'unknown')
        print(f"  • {pkg:30s} {version}")
    
    if len(summary['required_third_party']) > 10:
        print(f"  ... and {len(summary['required_third_party']) - 10} more")
    
    if summary['optional_third_party']:
        print(f"\nOptional/missing packages: {len(summary['optional_third_party'])}")
        for pkg in summary['optional_third_party'][:5]:
            print(f"  • {pkg}")
    
    if summary['circular_dependencies_count'] > 0:
        print(f"\n⚠ WARNING: Found {summary['circular_dependencies_count']} circular dependencies")
        for cycle in results['circular_dependencies'][:3]:
            print(f"  • {' → '.join(cycle)}")
    
    print(f"\nDetailed report written to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
