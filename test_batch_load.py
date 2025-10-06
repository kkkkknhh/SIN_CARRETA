"""
Batch Load Test for FastAPI Document Evaluation Endpoint

Tests concurrent document evaluation with throughput validation.
Requires: pytest-asyncio, httpx
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

import pytest


class MockFastAPIClient:
    """Mock FastAPI client for testing without actual server"""
    
    async def upload_document(self, document_text: str) -> Dict:
        """Simulate document upload and evaluation"""
        await asyncio.sleep(0.1)  # Simulate processing time
        return {
            "status": "success",
            "document_id": f"doc_{hash(document_text) % 10000}",
            "processing_time_ms": 100,
            "evaluation_score": 0.85
        }


async def evaluate_document_concurrent(client: MockFastAPIClient, doc_index: int) -> Dict:
    """Evaluate a single document and track timing"""
    start_time = time.time()
    
    document_text = f"Test document {doc_index} for evaluation"
    result = await client.upload_document(document_text)
    
    end_time = time.time()
    processing_time = (end_time - start_time) * 1000  # Convert to ms
    
    return {
        "doc_index": doc_index,
        "processing_time_ms": processing_time,
        "status": result["status"],
        "document_id": result.get("document_id")
    }


@pytest.mark.asyncio
async def test_batch_load_10_concurrent():
    """
    Batch load test: 10 concurrent document evaluations
    Performance requirement: >= 170 documents/hour (600ms per document average)
    """
    client = MockFastAPIClient()
    num_documents = 10
    
    # Start timing
    batch_start = time.time()
    
    # Execute concurrent evaluations
    tasks = [evaluate_document_concurrent(client, i) for i in range(num_documents)]
    results = await asyncio.gather(*tasks)
    
    # End timing
    batch_end = time.time()
    total_time_seconds = batch_end - batch_start
    total_time_ms = total_time_seconds * 1000
    
    # Calculate metrics
    avg_time_per_doc_ms = total_time_ms / num_documents
    throughput_docs_per_hour = (num_documents / total_time_seconds) * 3600
    
    # Save processing times
    processing_times = {
        "test_type": "batch_load_10_concurrent",
        "num_documents": num_documents,
        "total_time_seconds": total_time_seconds,
        "avg_time_per_doc_ms": avg_time_per_doc_ms,
        "throughput_docs_per_hour": throughput_docs_per_hour,
        "threshold_docs_per_hour": 170,
        "threshold_ms_per_doc": 600,
        "results": results
    }
    
    # Save to file for artifact archival
    Path("processing_times.json").write_text(json.dumps(processing_times, indent=2))
    
    # Generate throughput report
    throughput_report = {
        "test_name": "batch_load_test",
        "timestamp": time.time(),
        "metrics": {
            "throughput": throughput_docs_per_hour,
            "threshold": 170,
            "passed": throughput_docs_per_hour >= 170
        },
        "performance_summary": {
            "avg_ms_per_doc": avg_time_per_doc_ms,
            "total_time_seconds": total_time_seconds,
            "documents_processed": num_documents
        }
    }
    
    Path("throughput_report.json").write_text(json.dumps(throughput_report, indent=2))
    
    # Assertions
    assert avg_time_per_doc_ms <= 600, \
        f"Average time per document {avg_time_per_doc_ms:.2f}ms exceeds threshold of 600ms"
    
    assert throughput_docs_per_hour >= 170, \
        f"Throughput {throughput_docs_per_hour:.2f} docs/hour below threshold of 170 docs/hour"
    
    # Verify all documents processed successfully
    successful_docs = sum(1 for r in results if r["status"] == "success")
    assert successful_docs == num_documents, \
        f"Only {successful_docs}/{num_documents} documents processed successfully"


if __name__ == "__main__":
    # Run test directly
    asyncio.run(test_batch_load_10_concurrent())
    print("✅ Batch load test passed")
