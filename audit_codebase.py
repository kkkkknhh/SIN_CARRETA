#!/usr/bin/env python3
"""
Comprehensive codebase audit against 66 documented flows.
Identifies files for deletion that are neither flow nodes nor supporting infrastructure.
"""

import os
import glob

# From DEPENDENCY_FLOWS.md (66 flows total)
FLOW_NODES = {
    # Direct flow nodes from critical and standard flows
    'cli',
    'embedding_model',
    'dag_validation',
    'json_utils',
    'log_config',
    'decalogo_pipeline_orchestrator',  # DEPRECATED but documented
    'monetary_detector',
    'causal_pattern_detector',
    'teoria_cambio',
    'feasibility_scorer',
    'responsibility_detector',
    'contradiction_detector',
    'document_segmenter',
    'example_usage',
    'integrated_evaluation_system',
    'miniminimoon_orchestrator',
    'questionnaire_engine',
    'integration_example',
    'evidence_registry',
    'plan_processor',
    'miniminimoon_immutability',
    'plan_sanitizer',
    'data_flow_contract',
    'spacy_loader',
    'unified_evaluation_pipeline',
    'system_validators',
    'verify_coverage_metric',
    'verify_reproducibility',
    'annotated_examples_test',
    'demo',
    'debug_causal_patterns',
    'demo_performance_optimizations',
    'performance_test_suite',
    'demo_plan_sanitizer',
    'demo_signal_test',
    'example_monetary_usage',
    'example_teoria_cambio',
    'investigate_fault_recovery',
    'circuit_breaker',
    'memory_watchdog',
    'text_processor',
    'profile_contract_validation',
    'deterministic_pipeline_validator',
    'run_tests',
    'validate_teoria_cambio',
    # From critical paths
    'decalogo_loader',
    'device_config',
}

# Supporting infrastructure (configuration, utilities, testing, CLI tools)
SUPPORTING_INFRASTRUCTURE = {
    # Configuration
    'log_config',
    'freeze_config',
    'device_config',
    'json_utils',
    
    # Utilities
    'utils',
    'text_processor',
    'safe_io',
    
    # Verification and validation
    'miniminimoon_immutability',
    'data_flow_contract',
    'system_validators',
    'deterministic_pipeline_validator',
    'mathematical_invariant_guards',
    'determinism_guard',
    
    # Resilience
    'circuit_breaker',
    'memory_watchdog',
    'resilience_system',
    
    # Testing infrastructure
    'performance_test_suite',
    
    # Deployment infrastructure
    'canary_deployment',
    'opentelemetry_instrumentation',
    'slo_monitoring',
    
    # CLI tools (referenced in AGENTS.md)
    'miniminimoon_cli',
    
    # Setup and configuration
    'setup',
}

# Test files that validate documented flows
def is_valid_test_file(filename):
    """Check if test file validates a documented flow or infrastructure."""
    if not filename.startswith('test_'):
        return False
    
    base_name = filename[5:-3]  # Remove 'test_' prefix and '.py' suffix
    
    # Direct test for flow node
    if base_name in FLOW_NODES:
        return True
    
    # Direct test for supporting infrastructure
    if base_name in SUPPORTING_INFRASTRUCTURE:
        return True
    
    # Known valid test patterns
    valid_test_patterns = {
        'test_canonical_flow_integration',
        'test_canonical_integration',
        'test_critical_flows',
        'test_deployment_integration',
        'test_e2e_unified_pipeline',
        'test_e2e_unified_pipeline_mock',
        'test_batch_infrastructure',
        'test_batch_processor',
        'test_batch_optimizer',
        'test_batch_validators',
        'test_batch_load',
        'test_evidence_quality',
        'test_high_priority_fixes',
        'test_enhanced_orchestrator',
        'test_orchestrator_instrumentation',
        'test_orchestrator_modifications',
        'test_orchestrator_syntax',
        'test_basic_signal',
        'test_signal_handling',
        'test_deterministic_seeding',
        'test_answer_assembler',
        'test_answer_assembler_integration',
        'test_answer_assembler_refactor',
        'test_zero_evidence',
    }
    
    return filename[:-3] in valid_test_patterns

# Validation scripts referenced in AGENTS.md or flow documentation
VALIDATION_SCRIPTS = {
    'validate',
    'validate_teoria_cambio',
    'validate_canonical_integration',
    'validate_decalogo_alignment',
    'validate_performance_changes',
    'validate_questionnaire',
    'validate_batch_tests',
    'verify_coverage_metric',
    'verify_critical_flows',
    'verify_reproducibility',
    'verify_installation',
    'verify_orchestrator_changes',
    'rubric_check',
}

# Demo scripts explicitly referenced in documentation
DEMO_SCRIPTS = {
    'demo',
    'demo_performance_optimizations',
    'demo_plan_sanitizer',
    'demo_signal_test',
    'demo_unicode_comparison',
    'demo_document_segmentation',
    'demo_document_mapper',
    'demo_heap_functionality',
    'demo_questionnaire_driven_system',
    'example_usage',
    'example_teoria_cambio',
    'example_monetary_usage',
    'deployment_example',
}

# Additional utility scripts
UTILITY_SCRIPTS = {
    'dependency_doc_generator',
    'trace_matrix',
    'text_truncation_logger',
    'unicode_test_samples',
}

def categorize_python_files():
    """Categorize all Python files in the repository."""
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    
    valid_files = set()
    to_delete = []
    
    for filename in sorted(py_files):
        base_name = filename[:-3]  # Remove .py
        
        # Check if it's a flow node
        if base_name in FLOW_NODES:
            valid_files.add(filename)
            continue
        
        # Check if it's supporting infrastructure
        if base_name in SUPPORTING_INFRASTRUCTURE:
            valid_files.add(filename)
            continue
        
        # Check if it's a valid test file
        if is_valid_test_file(filename):
            valid_files.add(filename)
            continue
        
        # Check if it's a validation script
        if base_name in VALIDATION_SCRIPTS:
            valid_files.add(filename)
            continue
        
        # Check if it's a demo script
        if base_name in DEMO_SCRIPTS:
            valid_files.add(filename)
            continue
        
        # Check if it's a utility script
        if base_name in UTILITY_SCRIPTS:
            valid_files.add(filename)
            continue
        
        # If we get here, it's a candidate for deletion
        to_delete.append(filename)
    
    return valid_files, to_delete

def main():
    valid_files, to_delete = categorize_python_files()
    
    print("=" * 80)
    print("CODEBASE AUDIT REPORT")
    print("=" * 80)
    print()
    
    print(f"Total Python files analyzed: {len(valid_files) + len(to_delete)}")
    print(f"Valid files (flow nodes + infrastructure + tests): {len(valid_files)}")
    print(f"Files to delete (orphaned/deprecated): {len(to_delete)}")
    print()
    
    if to_delete:
        print("=" * 80)
        print("FILES MARKED FOR DELETION")
        print("=" * 80)
        print()
        for filename in sorted(to_delete):
            print(f"  - {filename}")
        print()
    
    print("=" * 80)
    print("DELETION CANDIDATES ANALYSIS")
    print("=" * 80)
    print()
    
    categories = {
        'Standalone examples': [],
        'Deprecated orchestrators': [],
        'PDM modules (not in flows)': [],
        'Analysis/audit scripts': [],
        'Batch processing (not in flows)': [],
        'Econml/Sinkhorn (not in flows)': [],
        'Orchestrator variants': [],
        'Run scripts': [],
        'Other': [],
    }
    
    for filename in to_delete:
        if 'pdm_' in filename:
            categories['PDM modules (not in flows)'].append(filename)
        elif 'batch_' in filename:
            categories['Batch processing (not in flows)'].append(filename)
        elif 'orchestrator' in filename and filename != 'decalogo_pipeline_orchestrator.py':
            categories['Orchestrator variants'].append(filename)
        elif filename.startswith('run_'):
            categories['Run scripts'].append(filename)
        elif 'audit' in filename or 'analyze' in filename:
            categories['Analysis/audit scripts'].append(filename)
        elif 'sinkhorn' in filename:
            categories['Econml/Sinkhorn (not in flows)'].append(filename)
        elif 'decatalogo' in filename.lower():
            categories['Deprecated orchestrators'].append(filename)
        elif filename.startswith('example_') or filename.startswith('demo_'):
            categories['Standalone examples'].append(filename)
        else:
            categories['Other'].append(filename)
    
    for category, files in categories.items():
        if files:
            print(f"\n{category}:")
            for f in sorted(files):
                print(f"  - {f}")
    
    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("The following files should be DELETED as they are not part of the")
    print("66 documented flows or supporting infrastructure:")
    print()
    for filename in sorted(to_delete):
        print(f"rm {filename}")
    
    return to_delete

if __name__ == '__main__':
    files_to_delete = main()
    
    # Write to file for easy review
    with open('files_to_delete.txt', 'w') as f:
        for filename in sorted(files_to_delete):
            f.write(f"{filename}\n")
    
    print()
    print("List written to: files_to_delete.txt")
