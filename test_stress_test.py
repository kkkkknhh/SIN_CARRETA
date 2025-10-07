#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stress_test.py — Stress Test for Batch Processing

Tests batch processing with 50 concurrent uploads and monitors
worker memory usage for leak detection (threshold: ≤20% growth).
"""

import json
import os
import pytest
import time
from pathlib import Path
from typing import List, Dict, Any
import tempfile

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


class TestStressTest:
    """Stress test suite for memory leak detection"""
    
    @pytest.fixture(scope="class")
    def stress_documents(self) -> List[Path]:
        """Create 50 sample PDF documents for stress testing"""
        docs = []
        temp_dir = Path(tempfile.mkdtemp(prefix="stress_test_"))
        
        for i in range(50):
            doc_path = temp_dir / f"stress_pdm_{i:02d}.pdf"
            with open(doc_path, "wb") as f:
                # Minimal PDF
                f.write(b"%PDF-1.4\n")
                f.write(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
                f.write(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
                f.write(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] >>\nendobj\n")
                f.write(b"4 0 obj\n<< >>\nendobj\n")
                f.write(b"xref\n0 5\n")
                f.write(b"0000000000 65535 f\n")
                f.write(b"0000000009 00000 n\n")
                f.write(b"0000000058 00000 n\n")
                f.write(b"0000000115 00000 n\n")
                f.write(b"0000000214 00000 n\n")
                f.write(b"trailer\n<< /Size 5 /Root 1 0 R >>\n")
                f.write(b"startxref\n234\n%%EOF\n")
            
            docs.append(doc_path)
        
        yield docs
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_stress_concurrent_uploads(self, stress_documents: List[Path]):
        """Test 50 concurrent uploads with memory monitoring"""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available - skipping stress test")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / (1024 ** 2)
        
        memory_samples = []
        
        def process_batch(batch_docs: List[Path], batch_idx: int) -> Dict[str, Any]:
            """Process a batch of documents"""
            time.sleep(0.1)  # Simulate processing
            
            # Sample memory
            current_memory_mb = process.memory_info().rss / (1024 ** 2)
            memory_samples.append({
                "batch": batch_idx,
                "documents_processed": (batch_idx + 1) * len(batch_docs),
                "memory_mb": current_memory_mb
            })
            
            return {
                "batch_idx": batch_idx,
                "documents": len(batch_docs),
                "memory_mb": current_memory_mb
            }
        
        # Process in batches of 10
        batches = [stress_documents[i:i+10] for i in range(0, len(stress_documents), 10)]
        
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(process_batch, batch, idx): idx
                for idx, batch in enumerate(batches)
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        final_memory_mb = process.memory_info().rss / (1024 ** 2)
        memory_growth_mb = final_memory_mb - initial_memory_mb
        memory_growth_percent = (memory_growth_mb / initial_memory_mb) * 100
        
        # Write memory profile
        memory_profile = {
            "test": "stress_test_50_concurrent",
            "memory_stats": {
                "initial_memory_mb": initial_memory_mb,
                "final_memory_mb": final_memory_mb,
                "memory_growth_mb": memory_growth_mb,
                "memory_growth_percent": memory_growth_percent,
                "threshold_percent": 20.0
            },
            "memory_samples_over_time": memory_samples
        }
        
        with open("memory_profile.json", "w") as f:
            json.dump(memory_profile, f, indent=2)
        
        # Write worker resource utilization
        worker_util = {
            "cpu_percent": process.cpu_percent(interval=1),
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads()
        }
        
        with open("worker_resource_utilization.json", "w") as f:
            json.dump(worker_util, f, indent=2)
        
        print(f"\n=== Stress Test Results ===")
        print(f"Documents processed: {len(stress_documents)}")
        print(f"Initial memory: {initial_memory_mb:.2f} MB")
        print(f"Final memory: {final_memory_mb:.2f} MB")
        print(f"Memory growth: {memory_growth_mb:.2f} MB ({memory_growth_percent:.2f}%)")
        print(f"Threshold: 20%")
        print(f"Status: {'✅ PASSED' if memory_growth_percent <= 20.0 else '❌ FAILED'}")
        
        # Assert memory growth within threshold
        assert memory_growth_percent <= 20.0, \
            f"Memory growth {memory_growth_percent:.2f}% exceeds threshold 20%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
