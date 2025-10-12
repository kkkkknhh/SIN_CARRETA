#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
celery_tasks.py — Celery Task Definitions for Document Processing

Defines Celery tasks that invoke unified_evaluation_pipeline for each document
and store results in Redis with artifacts persisted to filesystem.
"""

import json
import logging
import time
import traceback
from typing import Dict, Any
from celery import Celery, Task
from prometheus_client import Counter, Histogram

from unified_evaluation_pipeline import UnifiedEvaluationPipeline
from batch_processor import BatchJobManager

logger = logging.getLogger(__name__)

# Initialize Celery app
app = Celery("pdm_evaluation")
app.config_from_object("celeryconfig")

# Initialize batch job manager
batch_manager = BatchJobManager()

# Prometheus metrics for Celery tasks
documents_processed_total = Counter(
    "documents_processed_total",
    "Total documents processed by workers",
    ["status"]
)

batch_document_processing_latency_seconds = Histogram(
    "batch_document_processing_latency_seconds",
    "Document processing latency in seconds",
    buckets=[1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120]
)


class DocumentProcessingTask(Task):
    """Custom task class with retry logic and error handling"""
    
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True


@app.task(
    bind=True,
    base=DocumentProcessingTask,
    name="celery_tasks.process_document_task",
    queue="pdm_evaluation_queue"
)
def process_document_task(
    self,
    job_id: str,
    document_path: str,
    document_index: int,
    _config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single document using unified_evaluation_pipeline.
    
    Args:
        job_id: Unique job identifier
        document_path: Path to PDF document
        document_index: Document index in batch
        _config: Processing configuration dictionary
        
    Returns:
        Processing results dictionary
    """
    start_time = time.time()
    
    logger.info("Processing document %s for job %s: %s", document_index, job_id, document_path)
    
    try:
        # Transition job to processing state (first document only)
        if document_index == 0:
            batch_manager.transition_to_processing(job_id)
        
        # Initialize unified evaluation pipeline
        pipeline = UnifiedEvaluationPipeline()
        
        # Execute pipeline (simplified call - uses orchestrator run_evaluation)
        results = pipeline.run_evaluation(
            document_path=document_path,
            output_dir=str(batch_manager.artifacts_base_dir / job_id / f"{job_id}_doc_{document_index}")
        )
        
        processing_time = time.time() - start_time
        
        # Update progress
        job_data = batch_manager.get_job_data(job_id)
        if job_data:
            total_docs = job_data.get("document_count", 1)
            completed = document_index + 1
            batch_manager.update_progress(
                job_id=job_id,
                current_step=f"Processing document {completed}/{total_docs}",
                completed_steps=completed,
                total_steps=total_docs
            )
        
        # Store individual document artifacts
        document_id = f"{job_id}_doc_{document_index}"
        artifacts_dir = batch_manager.artifacts_base_dir / job_id / document_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = artifacts_dir / "evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save evidence registry
        if "evidence_registry" in results:
            evidence_file = artifacts_dir / "evidence_registry.json"
            with open(evidence_file, "w", encoding="utf-8") as f:
                json.dump(results["evidence_registry"], f, indent=2, ensure_ascii=False)
        
        # Save coverage report
        if "coverage_report" in results:
            coverage_file = artifacts_dir / "coverage_report.json"
            with open(coverage_file, "w", encoding="utf-8") as f:
                json.dump(results["coverage_report"], f, indent=2, ensure_ascii=False)
        
        # Update metrics
        documents_processed_total.labels(status="success").inc()
        batch_document_processing_latency_seconds.observe(processing_time)
        
        logger.info("Successfully processed document %s for job %s in %.2fs", document_index, job_id, processing_time)
        
        return {
            "status": "success",
            "document_id": document_id,
            "document_index": document_index,
            "processing_time": processing_time,
            "results_path": str(results_file)
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Document processing failed: {str(e)}\n{traceback.format_exc()}"
        
        logger.error("Failed to process document %s for job %s: %s", document_index, job_id, error_msg)
        
        # Update metrics
        documents_processed_total.labels(status="error").inc()
        batch_document_processing_latency_seconds.observe(processing_time)
        
        # If this is the last retry, mark job as failed
        if self.request.retries >= self.max_retries:
            batch_manager.transition_to_failed(job_id, error_msg)
        
        # Re-raise for Celery retry mechanism
        raise


@app.task(
    bind=True,
    name="celery_tasks.aggregate_batch_results_task",
    queue="pdm_evaluation_queue"
)
def aggregate_batch_results_task(self, job_id: str) -> Dict[str, Any]:
    """
    Aggregate results from all documents in a batch job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Aggregated results dictionary
    """
    logger.info("Aggregating batch results for job %s", job_id)
    
    try:
        job_data = batch_manager.get_job_data(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")
        
        document_count = job_data.get("document_count", 0)
        
        # Collect results from all documents
        aggregated_results = {
            "job_id": job_id,
            "document_count": document_count,
            "documents": [],
            "summary": {
                "total_questions_coverage": 0,
                "total_evidence_items": 0,
                "processing_time_total": 0.0
            }
        }
        
        job_artifacts_dir = batch_manager.artifacts_base_dir / job_id
        
        for doc_idx in range(document_count):
            document_id = f"{job_id}_doc_{doc_idx}"
            doc_artifacts_dir = job_artifacts_dir / document_id
            
            results_file = doc_artifacts_dir / "evaluation_results.json"
            if results_file.exists():
                with open(results_file, "r", encoding="utf-8") as f:
                    doc_results = json.load(f)
                    aggregated_results["documents"].append({
                        "document_id": document_id,
                        "document_index": doc_idx,
                        "results": doc_results
                    })
                    
                    # Update summary
                    if "coverage_report" in doc_results:
                        coverage = doc_results["coverage_report"].get("summary", {}).get("total_questions", 0)
                        aggregated_results["summary"]["total_questions_coverage"] += coverage
                    
                    if "evidence_registry" in doc_results:
                        evidence_count = len(doc_results["evidence_registry"].get("evidence", []))
                        aggregated_results["summary"]["total_evidence_items"] += evidence_count
        
        # Store aggregated results
        batch_manager.store_artifacts(job_id, aggregated_results)
        
        # Transition to completed
        batch_manager.transition_to_completed(job_id, aggregated_results)
        
        logger.info("Successfully aggregated results for job %s", job_id)
        
        return aggregated_results
        
    except Exception as e:
        error_msg = f"Batch aggregation failed: {str(e)}\n{traceback.format_exc()}"
        logger.error("Failed to aggregate results for job %s: %s", job_id, error_msg)
        batch_manager.transition_to_failed(job_id, error_msg)
        raise
