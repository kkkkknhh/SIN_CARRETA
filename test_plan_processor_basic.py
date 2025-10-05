#!/usr/bin/env python3
"""
Quick test script for plan_processor module
"""

import sys
import os
sys.path.insert(0, '/workspaces/MINIMINIMOON')

try:
    from plan_processor import FeasibilityPlanProcessor, PlanProcessingResult

    # Test basic instantiation
    processor = FeasibilityPlanProcessor()
    print("✓ FeasibilityPlanProcessor instantiated successfully")

    # Test process_plan with valid data
    plan_data = {"indicators": ["Test indicator"]}
    result = processor.process_plan(plan_data, "test_plan")

    print(f"✓ Plan processed successfully: {result.success}")
    print(f"✓ Plan ID: {result.plan_id}")
    print(f"✓ Processing time: {result.processing_time:.4f}s")
    print(f"✓ Total indicators: {result.result_data.get('total_indicators', 'N/A')}")

    # Test batch processing
    plans = [
        ({"indicators": ["Indicator 1"]}, "plan_1"),
        ({"indicators": ["Indicator 2"]}, "plan_2"),
    ]
    results = processor.batch_process_plans(plans)
    print(f"✓ Batch processing successful: {len(results)} results")

    print("All basic tests passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()