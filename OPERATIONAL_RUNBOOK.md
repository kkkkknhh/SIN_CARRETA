# MINIMINIMOON Operational Runbook

## Overview

This runbook provides step-by-step troubleshooting procedures for the 6 fault scenarios identified in Phase 4 planning. Each scenario includes symptoms, diagnosis steps, remediation procedures, and prevention strategies to ensure rapid recovery and minimal downtime for the MINIMINIMOON orchestrator pipeline.

**Target Recovery Times:**
- network_failure: 0.967s
- disk_full: 0.891s
- cpu_throttling: 0.777s

**Pipeline Components Affected:**
All 11 canonical pipeline nodes can be impacted by these faults:
1. Sanitization
2. Plan Processing
3. Document Segmentation
4. Embedding
5. Responsibility Detection
6. Contradiction Detection
7. Monetary Detection
8. Feasibility Scoring
9. Causal Detection
10. Theory of Change (TeoriaCambio)
11. DAG Validation

---

## Fault Scenario 1: network_failure

### Description
Complete network connectivity loss affecting external API calls, model downloads, and remote service dependencies. This is a critical failure that can halt the entire pipeline if not handled properly.

### Symptoms
- Connection timeout errors in logs
- Failed API calls to external services
- Embedding model download failures
- spaCy model download failures
- HTTP 503 or connection refused errors
- Trace IDs showing repeated retry attempts
- Metrics showing spike in error counters for embedding and responsibility_detection components

### Diagnosis Steps

1. **Check network connectivity:**
   ```bash
   ping 8.8.8.8
   ping google.com
   curl -I https://huggingface.co
   ```

2. **Verify DNS resolution:**
   ```bash
   nslookup huggingface.co
   dig google.com
   ```

3. **Check firewall rules:**
   ```bash
   sudo iptables -L -n
   # Check if outbound HTTPS (443) is allowed
   ```

4. **Review orchestrator logs:**
   ```bash
   grep "network\|timeout\|connection" logs/*.log
   # Look for trace IDs with multiple retry attempts
   ```

5. **Check health endpoint:**
   ```python
   orchestrator.get_health()
   # Look for unhealthy embedding and responsibility_detection components
   ```

6. **Review Prometheus metrics:**
   ```bash
   curl http://localhost:8000/metrics | grep error
   # Check error counters for embedding and responsibility_detection
   ```

### Remediation

1. **Immediate Actions (0-30s):**
   - Activate cached model fallback mode
   - Enable offline mode if configured
   - Queue incoming requests for later processing

2. **Short-term Recovery (30s-5min):**
   ```python
   # Verify cached models are available
   ls -lh ~/.cache/huggingface/
   ls -lh ~/.cache/spacy/
   
   # Restart orchestrator in offline mode
   export TRANSFORMERS_OFFLINE=1
   export HF_DATASETS_OFFLINE=1
   python miniminimoon_orchestrator.py
   ```

3. **Network Restoration:**
   ```bash
   # Restart network service
   sudo systemctl restart NetworkManager
   
   # Check routes
   ip route show
   
   # Verify connectivity restored
   ping 8.8.8.8
   ```

4. **Post-Recovery Verification:**
   - Check health endpoint returns all components healthy
   - Process test plan through pipeline
   - Verify trace IDs show no retry attempts
   - Check metrics show error counters stopped increasing

### Prevention

1. **Pre-cache all models:**
   ```python
   # Run during initialization/deployment
   python -c "from embedding_model import IndustrialEmbeddingModel; IndustrialEmbeddingModel()"
   python -m spacy download es_core_news_sm
   ```

2. **Implement circuit breakers:**
   - Add circuit breaker pattern for external calls
   - Configure timeout thresholds (5s, 10s, 30s)
   - Implement exponential backoff (already in place: 1s, 2s, 4s)

3. **Set up redundant connections:**
   - Configure multiple model mirror URLs
   - Use local model registry when possible
   - Implement model version pinning

4. **Monitoring:**
   - Alert on error_counter > 10 for any component
   - Alert on p99 latency > 10s for embedding component
   - Dashboard showing network connectivity status

---

## Fault Scenario 2: disk_full

### Description
Disk space exhaustion preventing file writes, cache updates, and intermediate result storage. Can cause silent failures or cascading errors across pipeline components.

### Symptoms
- "No space left on device" errors in logs
- Failed to write cache files
- Unable to store intermediate outputs
- Evidence registry write failures
- Execution context save failures
- Trace IDs showing OSError or IOError exceptions
- Metrics showing failed writes across multiple components

### Diagnosis Steps

1. **Check disk usage:**
   ```bash
   df -h
   du -sh /* | sort -h
   du -sh ~/.cache/* | sort -h
   ```

2. **Identify large files/directories:**
   ```bash
   find / -type f -size +1G 2>/dev/null
   find ~/.cache -type f -size +100M
   ```

3. **Check inode usage:**
   ```bash
   df -i
   # Inode exhaustion can occur even with free space
   ```

4. **Review orchestrator logs:**
   ```bash
   grep -i "space\|disk\|write\|OSError\|IOError" logs/*.log
   ```

5. **Check write permissions:**
   ```bash
   ls -la ~/.cache/
   touch /tmp/test_write && rm /tmp/test_write
   ```

6. **Monitor in real-time:**
   ```bash
   watch -n 1 df -h
   ```

### Remediation

1. **Immediate Actions (0-30s):**
   - Stop orchestrator to prevent cascading failures
   - Enable streaming mode (no intermediate file writes)
   - Activate compression for in-memory storage

2. **Emergency Cleanup (30s-2min):**
   ```bash
   # Clear old logs (keep last 7 days)
   find logs/ -type f -mtime +7 -delete
   
   # Clear embedding cache (will regenerate)
   rm -rf ~/.cache/huggingface/transformers/
   
   # Clear spaCy model cache (will re-download)
   rm -rf ~/.cache/spacy/
   
   # Clear temporary files
   rm -rf /tmp/miniminimoon_*
   
   # Clear old execution contexts
   find . -name "execution_context_*.json" -mtime +1 -delete
   ```

3. **Comprehensive Cleanup (2min-10min):**
   ```bash
   # Package manager cleanup
   sudo apt-get clean
   sudo apt-get autoremove
   
   # Docker cleanup (if applicable)
   docker system prune -af
   
   # Clear Python cache
   find . -type d -name __pycache__ -exec rm -rf {} +
   find . -type f -name "*.pyc" -delete
   ```

4. **Verify Recovery:**
   ```bash
   df -h
   # Ensure > 20% free space before restart
   
   # Restart orchestrator
   python miniminimoon_orchestrator.py test_plan.txt
   ```

### Prevention

1. **Disk monitoring:**
   ```bash
   # Set up alerts for disk usage > 80%
   # Alert on disk usage > 90% (critical)
   ```

2. **Automatic cleanup cron job:**
   ```bash
   # Add to crontab
   0 2 * * * find ~/logs -type f -mtime +7 -delete
   0 3 * * 0 find ~/.cache -type f -atime +30 -delete
   ```

3. **Configure log rotation:**
   ```bash
   # /etc/logrotate.d/miniminimoon
   /path/to/logs/*.log {
       daily
       rotate 7
       compress
       delaycompress
       missingok
       notifempty
   }
   ```

4. **Implement streaming mode:**
   - Process data in streaming fashion without writing to disk
   - Use in-memory intermediate storage with compression
   - Configure max memory limits

5. **Capacity planning:**
   - Monitor disk usage trends
   - Plan for 3x growth headroom
   - Implement automatic archival to S3/object storage

---

## Fault Scenario 3: cpu_throttling

### Description
CPU resource exhaustion or throttling causing degraded performance across all pipeline components, especially compute-intensive operations like embedding generation and DAG validation.

### Symptoms
- High CPU utilization (>90%) sustained
- Increased latency across all components
- p95 latency > 5s, p99 latency > 10s
- Embedding generation taking >10s per batch
- Monte Carlo validation taking >30s
- Trace IDs showing successful execution but slow timing
- Health checks taking >100ms (exceeding threshold)
- Worker processes stuck or unresponsive

### Diagnosis Steps

1. **Check CPU usage:**
   ```bash
   top -bn1 | head -20
   htop
   mpstat 1 5
   ```

2. **Identify CPU hogs:**
   ```bash
   ps aux --sort=-%cpu | head -10
   pidstat -u 1 5
   ```

3. **Check for CPU throttling:**
   ```bash
   # Check CPU frequency scaling
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
   
   # Check thermal throttling
   sensors
   cat /sys/class/thermal/thermal_zone*/temp
   ```

4. **Review orchestrator metrics:**
   ```python
   # Check latency percentiles
   summary = orchestrator.context.get_execution_summary()
   print(f"P95: {summary['latency_p95']:.3f}s")
   print(f"P99: {summary['latency_p99']:.3f}s")
   ```

5. **Check component-specific timing:**
   ```bash
   grep "Completed.*in" logs/*.log | awk '{print $NF}' | sort -n
   # Look for timing > 5s
   ```

6. **Check parallel processing:**
   ```bash
   pstree -p | grep python
   # Verify expected number of worker processes
   ```

### Remediation

1. **Immediate Actions (0-30s):**
   - Reduce parallel workers
   - Enable CPU throttling mode
   - Defer non-critical tasks

2. **Worker Reduction (30s-2min):**
   ```python
   # Update configuration
   config = {
       "parallel_processing": False,  # Disable parallel processing
       "embedding_batch_size": 8,     # Reduce from 32
       "monte_carlo_iterations": 1000 # Reduce from 5000
   }
   
   # Restart with reduced configuration
   orchestrator = MINIMINIMOONOrchestrator(config_path="low_cpu_config.json")
   ```

3. **Process Priority Adjustment:**
   ```bash
   # Increase nice value (lower priority)
   renice +10 $(pgrep -f miniminimoon_orchestrator)
   
   # Set CPU affinity to specific cores
   taskset -cp 0-1 $(pgrep -f miniminimoon_orchestrator)
   ```

4. **Thermal Management:**
   ```bash
   # If thermal throttling detected
   # Clean fans, improve ventilation
   # Reduce clock speed temporarily
   sudo cpupower frequency-set --max 2.0GHz
   ```

5. **Task Deferral:**
   - Queue low-priority plans for off-peak processing
   - Skip DAG validation for simple plans
   - Use cached embeddings aggressively

### Prevention

1. **CPU monitoring and alerting:**
   ```bash
   # Alert on CPU > 80% for 5min
   # Alert on p99 latency > 5s
   # Alert on health check > 100ms
   ```

2. **Optimize compute-intensive operations:**
   ```python
   # Use batch processing efficiently
   # Enable embedding cache
   # Reduce Monte Carlo iterations for non-critical validation
   # Pre-compute common patterns
   ```

3. **Load balancing:**
   - Distribute work across multiple instances
   - Use task queue (Celery, RQ) for async processing
   - Implement rate limiting on input

4. **Capacity planning:**
   - Monitor CPU trends over time
   - Plan for 2x peak capacity
   - Use auto-scaling in cloud environments

5. **Code optimization:**
   - Profile CPU hotspots
   - Optimize NumPy operations
   - Use compiled extensions for critical paths
   - Consider GPU acceleration for embeddings

---

## Fault Scenario 4: network_failure (Partial Recovery)

### Description
Intermittent network connectivity with partial service availability. Some model downloads succeed while others fail. Requires graceful degradation and selective fallback.

### Symptoms
- Intermittent connection timeouts
- Some components succeed, others fail within same execution
- Trace IDs showing mixed success/retry patterns
- Embedding model loads but spaCy download fails (or vice versa)
- Error counters increasing but not for all components
- p99 latency > 2s due to retries

### Diagnosis Steps

1. **Check connection stability:**
   ```bash
   ping -c 100 8.8.8.8 | grep "packet loss"
   mtr huggingface.co
   ```

2. **Test specific endpoints:**
   ```bash
   for i in {1..10}; do
       curl -w "%{http_code}\n" -o /dev/null -s https://huggingface.co
       sleep 1
   done
   ```

3. **Review mixed success patterns:**
   ```bash
   grep "succeeded on retry\|failed after" logs/*.log
   # Look for inconsistent patterns
   ```

4. **Check component-specific failures:**
   ```python
   health = orchestrator.get_health()
   for comp, status in health['components'].items():
       if status['status'] == 'unhealthy':
           print(f"Unhealthy: {comp}")
   ```

5. **Network quality metrics:**
   ```bash
   # Check latency variance
   ping -c 50 8.8.8.8 | tail -1
   # High stddev indicates instability
   ```

### Remediation

1. **Immediate Actions (0-30s):**
   - Enable hybrid mode (cache + network)
   - Increase retry count to 5
   - Increase backoff delay to 2s base

2. **Selective Fallback (30s-2min):**
   ```python
   # Configure per-component fallback
   config = {
       "embedding_model": {
           "use_cache_first": True,
           "network_timeout": 10
       },
       "spacy_loader": {
           "degraded_mode": True,  # Use basic processing
           "skip_ner": False       # Keep NER if model loaded
       }
   }
   ```

3. **Request Queuing:**
   ```python
   # Queue failed requests for retry during stable period
   from collections import deque
   retry_queue = deque()
   
   # Process during stable connectivity
   while retry_queue:
       plan = retry_queue.popleft()
       try:
           orchestrator.process_plan(plan)
       except NetworkError:
           retry_queue.append(plan)  # Re-queue
   ```

4. **Monitor Recovery:**
   ```bash
   watch -n 5 'curl -w "%{http_code}\n" -o /dev/null -s https://huggingface.co'
   # Wait for consistent 200s before full mode
   ```

### Prevention

1. **Connection quality monitoring:**
   - Track packet loss percentage
   - Monitor connection latency variance
   - Alert on jitter > 50ms

2. **Implement adaptive retry:**
   ```python
   # Increase retries based on success rate
   if success_rate < 0.7:
       max_retries = 5
       base_delay = 2.0
   ```

3. **Use connection pooling:**
   - Reuse HTTP connections
   - Implement keep-alive
   - Pre-warm connections

4. **Fallback chain:**
   - Primary: Network download
   - Secondary: Local cache
   - Tertiary: Degraded mode

---

## Fault Scenario 5: disk_full (Partial Recovery)

### Description
Disk space critically low but not completely exhausted. Some writes succeed while others fail. Requires selective cleanup and prioritized storage.

### Symptoms
- Intermittent write failures
- Some intermediate outputs saved, others skipped
- Evidence registry partially written
- Execution context save succeeds but cache writes fail
- Disk usage 85-95%
- Trace IDs showing mixed write success
- Some components complete, others fail on storage

### Diagnosis Steps

1. **Check exact disk usage:**
   ```bash
   df -h | grep -E "(Filesystem|/$|/home)"
   df -i | grep -E "(Filesystem|/$|/home)"
   ```

2. **Identify recent growth:**
   ```bash
   du -sh /*/ 2>/dev/null | sort -h | tail -5
   find ~/.cache -type f -mmin -60 | xargs du -sh
   ```

3. **Check write patterns:**
   ```bash
   grep -E "write|save|store" logs/*.log | grep -i "error\|failed"
   ```

4. **Monitor real-time growth:**
   ```bash
   watch -n 2 'df -h | grep -E "/$|/home"'
   ```

5. **Check application-specific storage:**
   ```bash
   du -sh ~/.cache/huggingface ~/.cache/spacy logs/ intermediate_outputs/
   ```

### Remediation

1. **Immediate Actions (0-30s):**
   - Enable in-memory storage mode
   - Disable non-critical writes
   - Enable aggressive compression

2. **Prioritized Cleanup (30s-2min):**
   ```bash
   # Priority 1: Clear temporary files (safest)
   rm -rf /tmp/*
   rm -rf intermediate_outputs/*.tmp
   
   # Priority 2: Clear old logs
   find logs/ -type f -mtime +2 -delete
   
   # Priority 3: Clear embedding cache (will regenerate)
   du -sh ~/.cache/huggingface/
   # If large (>5GB), selectively clear old models
   find ~/.cache/huggingface/ -type f -atime +7 -delete
   
   # Priority 4: Compress results
   gzip logs/*.log
   gzip results/*.json
   ```

3. **Selective Storage (2min-5min):**
   ```python
   # Configure minimal storage mode
   config = {
       "cache_embeddings": False,  # Don't cache, recompute
       "save_intermediate": False,  # Don't save intermediate outputs
       "compress_results": True,    # Compress final results
       "log_level": "WARNING"       # Reduce log verbosity
   }
   ```

4. **Verify Stability:**
   ```bash
   df -h
   # Ensure > 15% free before continuing
   ```

### Prevention

1. **Tiered storage strategy:**
   - Critical: Final results, evidence registry
   - Important: Execution context, error logs
   - Optional: Intermediate outputs, debug logs
   - Cacheable: Embeddings, model caches

2. **Automatic compression:**
   ```python
   import gzip
   import json
   
   def save_compressed(data, filename):
       with gzip.open(f"{filename}.gz", 'wt') as f:
           json.dump(data, f)
   ```

3. **Storage quotas:**
   - Limit cache size to 10GB
   - Limit logs to 5GB
   - Limit intermediate outputs to 5GB
   - Implement LRU eviction

4. **Monitoring and alerts:**
   - Alert at 80% disk usage
   - Critical alert at 90%
   - Auto-cleanup at 85%

---

## Fault Scenario 6: cpu_throttling (Partial Recovery)

### Description
CPU under heavy load but not completely saturated. Some components complete quickly while others experience delays. Requires intelligent task scheduling and resource allocation.

### Symptoms
- CPU usage fluctuating 70-95%
- Variable latency across components
- Some components fast (p50 < 1s) others slow (p99 > 5s)
- Embedding generation inconsistent timing
- DAG validation sometimes quick, sometimes slow
- Health checks occasionally > 100ms
- Trace IDs showing high timing variance

### Diagnosis Steps

1. **Check CPU load over time:**
   ```bash
   uptime
   sar -u 1 10
   mpstat -P ALL 1 5
   ```

2. **Identify load patterns:**
   ```bash
   pidstat -u -p $(pgrep -f miniminimoon) 1 10
   # Look for CPU% fluctuations
   ```

3. **Check component timing variance:**
   ```python
   summary = orchestrator.context.get_execution_summary()
   for comp, time in summary['component_times'].items():
       print(f"{comp}: {time:.3f}s")
   # Look for high variance
   ```

4. **Check context switching:**
   ```bash
   vmstat 1 5
   # High 'cs' (context switches) indicates CPU contention
   ```

5. **Check I/O wait:**
   ```bash
   iostat -x 1 5
   # High %iowait may be masking as CPU issue
   ```

### Remediation

1. **Immediate Actions (0-30s):**
   - Reduce batch sizes
   - Enable adaptive processing
   - Throttle request rate

2. **Adaptive Configuration (30s-2min):**
   ```python
   # Monitor CPU and adjust dynamically
   import psutil
   
   cpu_percent = psutil.cpu_percent(interval=1)
   
   if cpu_percent > 85:
       config = {
           "parallel_processing": False,
           "embedding_batch_size": 8,
           "monte_carlo_iterations": 1000
       }
   elif cpu_percent > 70:
       config = {
           "parallel_processing": True,
           "embedding_batch_size": 16,
           "monte_carlo_iterations": 2500
       }
   else:
       config = {
           "parallel_processing": True,
           "embedding_batch_size": 32,
           "monte_carlo_iterations": 5000
       }
   ```

3. **Task Prioritization:**
   ```python
   # Process critical components first
   priority_order = [
       "sanitization",
       "plan_processing",
       "document_segmentation",
       "embedding",  # CPU intensive
       # ... defer others if CPU high
   ]
   ```

4. **Yield to Other Processes:**
   ```python
   import time
   
   # Between components, check CPU and yield if high
   if psutil.cpu_percent() > 90:
       time.sleep(0.5)  # Let other processes run
   ```

### Prevention

1. **Implement CPU-aware scheduling:**
   ```python
   class CPUAwareOrchestrator:
       def should_throttle(self):
           return psutil.cpu_percent() > 85
       
       def adjust_batch_size(self):
           cpu = psutil.cpu_percent()
           if cpu > 85:
               return 8
           elif cpu > 70:
               return 16
           return 32
   ```

2. **Use process nice values:**
   ```bash
   # Start with lower priority
   nice -n 10 python miniminimoon_orchestrator.py
   ```

3. **Implement rate limiting:**
   ```python
   from time import sleep
   
   # Limit to 10 plans per minute under high CPU
   if psutil.cpu_percent() > 80:
       sleep(6)  # Rate limit
   ```

4. **Optimize hot paths:**
   - Profile CPU usage per component
   - Optimize embedding batch processing
   - Cache expensive computations
   - Use NumPy vectorization

5. **Monitoring:**
   - Track CPU usage per component
   - Alert on CPU > 85% for > 2 minutes
   - Alert on p99 latency degradation > 20%
   - Dashboard with CPU usage trends

---

## General Troubleshooting Workflow

### Step 1: Identify the Fault (0-30s)

```bash
# Quick health check
curl http://localhost:8000/health

# Check recent errors
tail -100 logs/orchestrator.log | grep ERROR

# Check metrics
curl http://localhost:8000/metrics | grep -E "error|latency"

# System resources
df -h && free -h && uptime
```

### Step 2: Determine Severity (30s-1min)

- **Critical**: Pipeline completely halted, health check fails
- **Major**: >50% error rate, p99 latency > 10s
- **Minor**: <20% error rate, p99 latency > 5s
- **Warning**: Degraded performance but functional

### Step 3: Apply Remediation (1min-5min)

Follow specific scenario playbook based on fault type identified.

### Step 4: Verify Recovery (5min-10min)

```python
# Test with sample plan
orchestrator = MINIMINIMOONOrchestrator()
result = orchestrator.process_plan("test_plan.txt")

# Verify success
assert "error" not in result
assert len(result["executed_nodes"]) == 11

# Check metrics normalized
health = orchestrator.get_health()
assert health["overall_status"] == "healthy"
```

### Step 5: Post-Mortem

1. Document incident timeline
2. Review trace IDs for affected requests
3. Analyze root cause
4. Update prevention measures
5. Update alert thresholds if needed

---

## Emergency Contacts

- **System Admin**: [Contact info]
- **Network Team**: [Contact info]
- **DevOps On-Call**: [Contact info]
- **Escalation Path**: [Define escalation chain]

---

## Appendix: Quick Reference Commands

### Health Check
```bash
python -c "from miniminimoon_orchestrator import MINIMINIMOONOrchestrator; o = MINIMINIMOONOrchestrator(); print(o.get_health())"
```

### Metrics Export
```bash
python -c "from miniminimoon_orchestrator import MINIMINIMOONOrchestrator; o = MINIMINIMOONOrchestrator(); print(o.get_metrics())"
```

### Component Status
```bash
grep -E "initialized|success|error" logs/*.log | tail -20
```

### Trace ID Lookup
```bash
grep "TRACE_ID" logs/*.log
```

### Resource Summary
```bash
echo "=== Disk ===" && df -h && echo "=== Memory ===" && free -h && echo "=== CPU ===" && uptime
```

---

**Last Updated**: 2024
**Version**: 1.0
**Maintainer**: MINIMINIMOON DevOps Team
