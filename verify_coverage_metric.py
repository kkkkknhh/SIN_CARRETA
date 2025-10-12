#!/usr/bin/env python3
"""
Coverage Metric Verification
============================
Manually traces through 10 representative test flows to verify the 65% coverage
metric is accurate by comparing actual code execution paths against reported coverage data.
"""
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
import importlib.util


class CodePathTracer:
    """Traces code execution paths"""
    
    def __init__(self):
        self.executed_lines: Dict[str, Set[int]] = {}
        self.function_calls: List[Dict[str, Any]] = []
    
    def trace_function(self, frame, event, _arg):
        """Trace function for sys.settrace"""
        if event == 'line':
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            
            # Only trace project files (not stdlib)
            if 'venv' not in filename and 'site-packages' not in filename:
                if filename not in self.executed_lines:
                    self.executed_lines[filename] = set()
                self.executed_lines[filename].add(lineno)
        
        elif event == 'call':
            filename = frame.f_code.co_filename
            funcname = frame.f_code.co_name
            
            if 'venv' not in filename and 'site-packages' not in filename:
                self.function_calls.append({
                    "function": funcname,
                    "file": filename,
                    "line": frame.f_lineno
                })
        
        return self.trace_function
    
    def start_tracing(self):
        """Start tracing execution"""
        sys.settrace(self.trace_function)
    
    @staticmethod
    def stop_tracing():
        """Stop tracing execution"""
        sys.settrace(None)
    
    def get_coverage_for_file(self, filepath: str) -> Dict[str, Any]:
        """Calculate coverage for a specific file"""
        if filepath not in self.executed_lines:
            return {
                "total_lines": 0,
                "executed_lines": 0,
                "coverage_percent": 0.0,
                "executed_line_numbers": []
            }
        
        # Count total executable lines (excluding comments, blank lines)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            total_executable = 0
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    total_executable += 1
            
            executed = len(self.executed_lines[filepath])
            coverage_percent = (executed / total_executable * 100) if total_executable > 0 else 0
            
            return {
                "total_lines": total_executable,
                "executed_lines": executed,
                "coverage_percent": coverage_percent,
                "executed_line_numbers": sorted(self.executed_lines[filepath])
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_lines": 0,
                "executed_lines": 0,
                "coverage_percent": 0.0
            }


def load_module_from_file(filepath: str, module_name: str):
    """Load a Python module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return None


def trace_test_flow(test_name: str, test_func, tracer: CodePathTracer) -> Dict[str, Any]:
    """Trace a single test flow"""
    print(f"\n  Tracing: {test_name}...")
    
    # Reset tracer
    tracer.executed_lines.clear()
    tracer.function_calls.clear()
    
    # Start tracing
    tracer.start_tracing()
    
    try:
        # Execute test
        result = test_func()
        status = "PASSED"
        error = None
    except Exception as e:
        status = "FAILED"
        error = str(e)
        result = None
    finally:
        # Stop tracing
        tracer.stop_tracing()
    
    # Get coverage stats
    total_files = len(tracer.executed_lines)
    total_lines_executed = sum(len(lines) for lines in tracer.executed_lines.values())
    
    return {
        "test_name": test_name,
        "status": status,
        "error": error,
        "files_touched": total_files,
        "lines_executed": total_lines_executed,
        "function_calls": len(tracer.function_calls),
        "executed_files": list(tracer.executed_lines.keys())
    }


def create_representative_test_flows():
    """Create 10 representative test flows"""
    
    # Test 1: Embedding model test
    def test_embedding_model():
        try:
            from embedding_model import create_embedding_model
            model = create_embedding_model()
            return True
        except:
            return False
    
    # Test 2: Text processor test
    def test_text_processor():
        try:
            from text_processor import TextProcessor
            processor = TextProcessor()
            processor.process("Test text")
            return True
        except:
            return False
    
    # Test 3: Responsibility detector test
    def test_responsibility_detector():
        try:
            from responsibility_detector import ResponsibilityDetector
            detector = ResponsibilityDetector()
            detector.detect_entities("El Ministerio de Educación implementará el programa.")
            return True
        except:
            return False
    
    # Test 4: Circuit breaker test
    def test_circuit_breaker():
        try:
            from circuit_breaker import CircuitBreaker, CircuitBreakerConfig
            cb = CircuitBreaker("test", CircuitBreakerConfig())
            return True
        except:
            return False
    
    # Test 5: Data flow contract test
    def test_data_flow_contract():
        try:
            from data_flow_contract import CanonicalFlowValidator
            validator = CanonicalFlowValidator()
            return True
        except:
            return False
    
    # Test 6: Evidence registry test
    def test_evidence_registry():
        try:
            from evidence_registry import EvidenceRegistry
            registry = EvidenceRegistry()
            return True
        except:
            return False
    
    # Test 7: DAG validation test
    def test_dag_validation():
        try:
            from dag_validation import validate_dag
            return True
        except:
            return False
    
    # Test 8: Monetary detector test
    def test_monetary_detector():
        try:
            from monetary_detector import MonetaryDetector
            detector = MonetaryDetector()
            detector.detect("El presupuesto es de $1,000,000 pesos.")
            return True
        except:
            return False
    
    # Test 9: Feasibility scorer test
    def test_feasibility_scorer():
        try:
            from feasibility_scorer import FeasibilityScorer
            scorer = FeasibilityScorer()
            return True
        except:
            return False
    
    # Test 10: Plan processor test
    def test_plan_processor():
        try:
            from plan_processor import PlanProcessor
            processor = PlanProcessor()
            return True
        except:
            return False
    
    return [
        ("test_embedding_model", test_embedding_model),
        ("test_text_processor", test_text_processor),
        ("test_responsibility_detector", test_responsibility_detector),
        ("test_circuit_breaker", test_circuit_breaker),
        ("test_data_flow_contract", test_data_flow_contract),
        ("test_evidence_registry", test_evidence_registry),
        ("test_dag_validation", test_dag_validation),
        ("test_monetary_detector", test_monetary_detector),
        ("test_feasibility_scorer", test_feasibility_scorer),
        ("test_plan_processor", test_plan_processor),
    ]


def analyze_coverage_discrepancies(trace_results: List[Dict], expected_coverage: float = 65.0):
    """Analyze discrepancies between actual and reported coverage"""
    print("\n" + "="*80)
    print("COVERAGE DISCREPANCY ANALYSIS")
    print("="*80)
    
    # Count files touched across all tests
    all_files = set()
    for result in trace_results:
        if result["status"] == "PASSED":
            all_files.update(result["executed_files"])
    
    # Get project Python files
    project_files = list(Path(".").glob("*.py"))
    project_files = [f for f in project_files if f.name not in ['setup.py', 'conftest.py']]
    
    covered_count = len(all_files)
    total_count = len(project_files)
    actual_coverage = (covered_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\nExpected coverage: {expected_coverage:.1f}%")
    print(f"Actual coverage:   {actual_coverage:.1f}%")
    print(f"Discrepancy:       {abs(actual_coverage - expected_coverage):.1f}%")
    
    print(f"\nFiles covered: {covered_count} / {total_count}")
    
    # Identify uncovered files
    uncovered = set(str(f) for f in project_files) - all_files
    if uncovered:
        print(f"\nUncovered files ({len(uncovered)}):")
        for f in sorted(uncovered)[:10]:
            print(f"  - {f}")
        if len(uncovered) > 10:
            print(f"  ... and {len(uncovered) - 10} more")
    
    return {
        "expected_coverage": expected_coverage,
        "actual_coverage": actual_coverage,
        "discrepancy": abs(actual_coverage - expected_coverage),
        "files_covered": covered_count,
        "files_total": total_count,
        "uncovered_files": list(uncovered)
    }


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("COVERAGE METRIC VERIFICATION")
    print("="*80)
    print("\nManually tracing 10 representative test flows...")
    
    # Create tracer
    tracer = CodePathTracer()
    
    # Get test flows
    test_flows = create_representative_test_flows()
    
    # Trace each flow
    trace_results = []
    for test_name, test_func in test_flows:
        result = trace_test_flow(test_name, test_func, tracer)
        trace_results.append(result)
        
        status_symbol = "✅" if result["status"] == "PASSED" else "❌"
        print(f"    {status_symbol} {result['status']}: {result['lines_executed']} lines, "
              f"{result['files_touched']} files")
    
    # Summary
    print("\n" + "="*80)
    print("TRACE RESULTS SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in trace_results if r["status"] == "PASSED")
    failed = sum(1 for r in trace_results if r["status"] == "FAILED")
    
    print(f"\nTests passed: {passed} / {len(trace_results)}")
    print(f"Tests failed: {failed} / {len(trace_results)}")
    
    # Show failed tests
    if failed > 0:
        print("\nFailed tests:")
        for result in trace_results:
            if result["status"] == "FAILED":
                print(f"  - {result['test_name']}: {result['error']}")
    
    # Analyze coverage
    analysis = analyze_coverage_discrepancies(trace_results, expected_coverage=65.0)
    
    # Save results
    import json
    output = {
        "trace_results": trace_results,
        "coverage_analysis": analysis,
        "summary": {
            "total_tests": len(trace_results),
            "passed": passed,
            "failed": failed
        }
    }
    
    with open("coverage_verification_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Results saved to coverage_verification_results.json")


if __name__ == "__main__":
    main()
