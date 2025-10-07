"""
Test CI Batch Workflow - Verifies full batch load test workflow
Simulates CI environment and validates all artifacts are generated correctly
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path


def test_batch_workflow():
    """Run complete batch load test workflow and verify artifacts"""
    
    # Clean up any existing artifacts
    for f in ['processing_times.json', 'throughput_report.json', 
              'latency_distribution.json', 'queue_depth.json',
              'memory_profile.json', 'worker_resource_utilization.json']:
        if Path(f).exists():
            Path(f).unlink()
    
    if Path('batch_metrics').exists():
        import shutil
        shutil.rmtree('batch_metrics')
    
    print("✅ Clean environment")
    
    # Run batch load test
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'test_batch_load.py', '-v'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Batch load test failed:\n{result.stdout}\n{result.stderr}"
    print("✅ Batch load test passed")
    
    # Run stress test
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'test_stress_test.py', '-v'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Stress test failed:\n{result.stdout}\n{result.stderr}"
    print("✅ Stress test passed")
    
    # Verify artifacts exist
    expected_files = [
        'processing_times.json',
        'throughput_report.json',
        'latency_distribution.json',
        'queue_depth.json',
        'memory_profile.json',
        'worker_resource_utilization.json'
    ]
    
    for filename in expected_files:
        assert Path(filename).exists(), f"Missing artifact: {filename}"
        
        # Validate JSON structure
        with open(filename) as f:
            data = json.load(f)
            assert isinstance(data, dict), f"{filename} is not a valid JSON object"
            print(f"✅ Artifact exists and is valid JSON: {filename}")
    
    # Simulate CI artifact collection
    Path('batch_metrics').mkdir(exist_ok=True)
    
    for filename in expected_files:
        subprocess.run(['cp', '-f', filename, 'batch_metrics/'], check=False)
    
    # Verify batch_metrics directory
    batch_metrics_files = list(Path('batch_metrics').glob('*.json'))
    assert len(batch_metrics_files) >= 6, \
        f"Expected at least 6 files in batch_metrics/, found {len(batch_metrics_files)}"
    print(f"✅ Batch metrics directory contains {len(batch_metrics_files)} files")
    
    # Validate throughput meets threshold
    with open('throughput_report.json') as f:
        throughput = json.load(f)
        assert throughput['metrics']['passed'], \
            f"Throughput test failed: {throughput['metrics']['throughput']:.2f} docs/hour < 170"
        print(f"✅ Throughput: {throughput['metrics']['throughput']:.2f} docs/hour (>= 170)")
    
    # Validate memory growth is within threshold
    with open('memory_profile.json') as f:
        memory = json.load(f)
        growth = memory['memory_stats']['memory_growth_percent']
        assert not memory['memory_stats']['memory_leak_detected'], \
            f"Memory leak detected: {growth:.2f}% growth > 20%"
        print(f"✅ Memory growth: {growth:.2f}% (<= 20%)")
    
    print("\n" + "="*60)
    print("🎉 BATCH LOAD TEST WORKFLOW VALIDATION COMPLETE")
    print("="*60)
    print("\nArtifacts generated:")
    for f in batch_metrics_files:
        print(f"  - {f.name}")


if __name__ == '__main__':
    test_batch_workflow()
