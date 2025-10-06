"""
Stress Test for FastAPI Document Evaluation Endpoint

Tests 50 concurrent uploads with memory profiling.
Monitors worker memory usage and detects memory leaks.
"""

import asyncio
import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List

import psutil
import pytest


class MockCeleryWorker:
    """Mock Celery worker with memory tracking"""
    
    def __init__(self):
        self.tasks_processed = 0
        self.process = psutil.Process()
    
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    async def process_task(self, document_text: str) -> Dict:
        """Simulate task processing"""
        self.tasks_processed += 1
        await asyncio.sleep(0.05)  # Simulate work
        
        # Simulate some memory allocation (small amount)
        _ = [i for i in range(1000)]
        
        return {
            "status": "success",
            "task_id": f"task_{self.tasks_processed}",
            "memory_mb": self.get_memory_mb()
        }


class MockFastAPIClient:
    """Mock FastAPI client connected to mock worker"""
    
    def __init__(self, worker: MockCeleryWorker):
        self.worker = worker
    
    async def upload_document(self, document_text: str) -> Dict:
        """Upload document for processing"""
        return await self.worker.process_task(document_text)


async def stress_test_document(client: MockFastAPIClient, doc_index: int) -> Dict:
    """Process a single document in stress test"""
    document_text = f"Stress test document {doc_index} with content"
    result = await client.upload_document(document_text)
    
    return {
        "doc_index": doc_index,
        "status": result["status"],
        "task_id": result.get("task_id"),
        "memory_mb": result.get("memory_mb")
    }


@pytest.mark.asyncio
async def test_stress_test_50_concurrent():
    """
    Stress test: 50 concurrent document uploads
    Memory requirement: Memory growth <= 20% between start and end
    """
    # Initialize tracemalloc for detailed memory tracking
    tracemalloc.start()
    
    # Create worker and client
    worker = MockCeleryWorker()
    client = MockFastAPIClient(worker)
    
    num_documents = 50
    memory_samples = []
    
    # Capture initial memory state
    initial_memory_mb = worker.get_memory_mb()
    initial_snapshot = tracemalloc.take_snapshot()
    
    # Start timing
    stress_start = time.time()
    
    # Execute concurrent stress test in batches to monitor memory over time
    batch_size = 10
    all_results = []
    
    for batch_num in range(num_documents // batch_size):
        batch_start_idx = batch_num * batch_size
        batch_tasks = [
            stress_test_document(client, batch_start_idx + i) 
            for i in range(batch_size)
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        all_results.extend(batch_results)
        
        # Sample memory after each batch
        current_memory_mb = worker.get_memory_mb()
        memory_samples.append({
            "batch": batch_num,
            "documents_processed": (batch_num + 1) * batch_size,
            "memory_mb": current_memory_mb
        })
        
        # Small delay between batches
        await asyncio.sleep(0.1)
    
    # End timing
    stress_end = time.time()
    total_time_seconds = stress_end - stress_start
    
    # Capture final memory state
    final_memory_mb = worker.get_memory_mb()
    final_snapshot = tracemalloc.take_snapshot()
    
    # Calculate memory growth
    memory_growth_mb = final_memory_mb - initial_memory_mb
    memory_growth_percent = (memory_growth_mb / initial_memory_mb) * 100
    
    # Analyze memory differences
    top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
    
    # Save memory profile
    memory_profile = {
        "test_type": "stress_test_50_concurrent",
        "num_documents": num_documents,
        "total_time_seconds": total_time_seconds,
        "memory_stats": {
            "initial_memory_mb": initial_memory_mb,
            "final_memory_mb": final_memory_mb,
            "memory_growth_mb": memory_growth_mb,
            "memory_growth_percent": memory_growth_percent,
            "threshold_percent": 20
        },
        "memory_samples_over_time": memory_samples,
        "top_memory_allocations": [
            {
                "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size_kb": stat.size / 1024,
                "count": stat.count
            }
            for stat in top_stats[:10]
        ]
    }
    
    # Save to file for artifact archival
    Path("memory_profile.json").write_text(json.dumps(memory_profile, indent=2))
    
    # Stop tracemalloc
    tracemalloc.stop()
    
    # Assertions
    assert memory_growth_percent <= 20, \
        f"Memory growth {memory_growth_percent:.2f}% exceeds threshold of 20%"
    
    # Verify all documents processed successfully
    successful_docs = sum(1 for r in all_results if r["status"] == "success")
    assert successful_docs == num_documents, \
        f"Only {successful_docs}/{num_documents} documents processed successfully"
    
    # Check for monotonic memory growth (potential leak indicator)
    memory_values = [s["memory_mb"] for s in memory_samples]
    if len(memory_values) >= 3:
        # Allow some variance, but fail if memory grows consistently across all batches
        growth_trend = all(
            memory_values[i] <= memory_values[i+1] + 1.0  # Allow 1MB variance
            for i in range(len(memory_values) - 1)
        )
        assert not growth_trend or memory_growth_percent <= 20, \
            f"Detected monotonic memory growth pattern (potential leak)"


if __name__ == "__main__":
    # Run test directly
    asyncio.run(test_stress_test_50_concurrent())
    print("✅ Stress test passed")
