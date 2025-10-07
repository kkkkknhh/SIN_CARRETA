#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
celeryconfig.py — Celery Configuration for Batch Processing

Configures Celery workers with:
- concurrency=8 for parallel document processing
- prefetch_multiplier=4 for optimal task prefetching
- Redis backend for task queue and results storage
"""

import os

# Broker and Backend Configuration
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Task Configuration
task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]
timezone = "UTC"
enable_utc = True

# Worker Configuration
worker_concurrency = int(os.getenv("CELERY_WORKER_CONCURRENCY", "8"))
worker_prefetch_multiplier = int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "4"))
worker_max_tasks_per_child = int(os.getenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", "100"))

# Task routing
task_routes = {
    "celery_tasks.process_document_task": {"queue": "pdm_evaluation_queue"}
}

# Task execution
task_acks_late = True
task_reject_on_worker_lost = True
task_time_limit = 600  # 10 minutes hard limit
task_soft_time_limit = 540  # 9 minutes soft limit

# Result backend settings
result_expires = 86400  # 24 hours

# Worker heartbeat (for monitoring)
worker_send_task_events = True
task_send_sent_event = True

# Logging
worker_log_format = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
worker_task_log_format = "[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s"

# Monitoring
worker_redirect_stdouts_level = "INFO"
