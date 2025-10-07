# Batch Processing Guide

Complete setup and operational guide for the DECALOGO batch processing system.

## Table of Contents

1. [Installation](#installation)
2. [Worker Configuration](#worker-configuration)
3. [Nginx Reverse Proxy Setup](#nginx-reverse-proxy-setup)
4. [Horizontal Scaling](#horizontal-scaling)
5. [API Authentication](#api-authentication)
6. [Rate Limiting](#rate-limiting)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning Matrix](#performance-tuning-matrix)

---

## Installation

### Redis Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Verify installation
redis-cli ping
# Should return: PONG
```

**CentOS/RHEL:**
```bash
sudo yum install epel-release -y
sudo yum install redis -y
sudo systemctl enable redis
sudo systemctl start redis

# Verify installation
redis-cli ping
```

**macOS:**
```bash
brew install redis
brew services start redis

# Verify installation
redis-cli ping
```

### Configure Redis for Production

Edit `/etc/redis/redis.conf`:

```conf
# Bind to all interfaces (or specific IP)
bind 0.0.0.0

# Set password
requirepass your_secure_password_here

# Enable persistence (AOF + RDB)
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec

save 900 1
save 300 10
save 60 10000

# Set max memory and eviction policy
maxmemory 2gb
maxmemory-policy allkeys-lru

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG ""
```

Restart Redis:
```bash
sudo systemctl restart redis-server
```

### Celery and Dependencies Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install celery[redis]==5.3.4
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install redis==5.0.1
pip install python-multipart==0.0.6
pip install pydantic==2.5.0
pip install python-jose[cryptography]==3.3.0
pip install passlib[bcrypt]==1.7.4
pip install aiofiles==23.2.1
```

---

## Worker Configuration

### Create Worker Module (`batch_worker.py`)

```python
from celery import Celery
from kombu import Queue

# Initialize Celery
app = Celery(
    'decalogo_batch',
    broker='redis://:your_password@localhost:6379/0',
    backend='redis://:your_password@localhost:6379/1'
)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 minutes soft limit
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=100,
    result_expires=2592000,  # 30 days
    task_default_queue='decalogo_batch_tasks',
    task_default_priority=5,
)

# Define queues
app.conf.task_queues = (
    Queue('decalogo_batch_tasks', routing_key='task.#'),
    Queue('high_priority', routing_key='priority.high'),
    Queue('low_priority', routing_key='priority.low'),
)

@app.task(bind=True)
def process_document(self, document_id, document_content, options):
    """Process a single document through the evaluation pipeline"""
    # Implementation here
    pass
```

### Start Worker

**Standard Configuration (8 threads per worker):**
```bash
celery -A batch_worker worker \
    --concurrency=8 \
    --prefetch-multiplier=4 \
    --max-tasks-per-child=100 \
    --loglevel=info \
    --logfile=logs/worker_%n%I.log \
    --pidfile=run/worker_%n.pid \
    --hostname=worker1@%h
```

**High Performance Configuration (16 threads):**
```bash
celery -A batch_worker worker \
    --concurrency=16 \
    --prefetch-multiplier=2 \
    --max-tasks-per-child=50 \
    --loglevel=warning \
    --logfile=logs/worker_%n%I.log \
    --hostname=worker1@%h
```

**With Priority Queues:**
```bash
celery -A batch_worker worker \
    --concurrency=8 \
    --prefetch-multiplier=4 \
    --queues=high_priority,decalogo_batch_tasks,low_priority \
    --loglevel=info
```

### Create Systemd Service

Create `/etc/systemd/system/celery-worker@.service`:

```ini
[Unit]
Description=Celery Worker %i
After=network.target redis-server.service

[Service]
Type=forking
User=celery
Group=celery
WorkingDirectory=/opt/decalogo
Environment="PATH=/opt/decalogo/venv/bin"
ExecStart=/opt/decalogo/venv/bin/celery -A batch_worker worker \
    --concurrency=8 \
    --prefetch-multiplier=4 \
    --max-tasks-per-child=100 \
    --logfile=/var/log/celery/worker_%i.log \
    --pidfile=/var/run/celery/worker_%i.pid \
    --hostname=worker%i@%h \
    --detach
ExecStop=/opt/decalogo/venv/bin/celery -A batch_worker control shutdown
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start workers:
```bash
sudo systemctl enable celery-worker@{1..4}
sudo systemctl start celery-worker@{1..4}
```

---

## Nginx Reverse Proxy Setup

### Install Nginx

**Ubuntu/Debian:**
```bash
sudo apt-get install nginx -y
```

**CentOS/RHEL:**
```bash
sudo yum install nginx -y
```

### Configure Nginx

Create `/etc/nginx/sites-available/decalogo-batch`:

```nginx
# Rate limiting zones
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
limit_req_zone $binary_remote_addr zone=upload_limit:10m rate=10r/m;

# Upstream FastAPI servers
upstream fastapi_backend {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s backup;
}

server {
    listen 80;
    server_name batch.decalogo.example.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name batch.decalogo.example.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/decalogo.crt;
    ssl_certificate_key /etc/ssl/private/decalogo.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Logging
    access_log /var/log/nginx/decalogo-batch-access.log;
    error_log /var/log/nginx/decalogo-batch-error.log;
    
    # Max upload size
    client_max_body_size 100M;
    client_body_timeout 300s;
    
    # Upload endpoint (stricter rate limit)
    location /api/v1/batch/upload {
        limit_req zone=upload_limit burst=5 nodelay;
        
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_read_timeout 300s;
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
    }
    
    # Other API endpoints
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_read_timeout 60s;
        proxy_connect_timeout 10s;
    }
    
    # Health check (no rate limit)
    location /api/v1/health {
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        access_log off;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/decalogo-batch /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Horizontal Scaling

### Adding Workers on the Same Machine

```bash
# Start additional workers
celery -A batch_worker worker \
    --concurrency=8 \
    --prefetch-multiplier=4 \
    --hostname=worker5@localhost \
    --loglevel=info &

celery -A batch_worker worker \
    --concurrency=8 \
    --prefetch-multiplier=4 \
    --hostname=worker6@localhost \
    --loglevel=info &
```

### Adding Workers on New Machines

**On the new machine:**

1. **Install dependencies:**
```bash
sudo apt-get install python3 python3-venv redis-tools -y
```

2. **Clone repository and setup:**
```bash
git clone https://github.com/your-org/decalogo.git /opt/decalogo
cd /opt/decalogo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure worker to connect to main Redis:**

Edit `batch_worker.py`:
```python
app = Celery(
    'decalogo_batch',
    broker='redis://:your_password@redis-server.example.com:6379/0',
    backend='redis://:your_password@redis-server.example.com:6379/1'
)
```

4. **Start worker:**
```bash
celery -A batch_worker worker \
    --concurrency=8 \
    --prefetch-multiplier=4 \
    --hostname=worker1@machine2 \
    --loglevel=info
```

### Auto-Scaling with Systemd

Create `/usr/local/bin/celery-autoscale.sh`:

```bash
#!/bin/bash

QUEUE_DEPTH=$(redis-cli -a your_password llen decalogo_batch_tasks)
ACTIVE_WORKERS=$(celery -A batch_worker inspect active_queues | grep -c "worker")

# Scale up if queue depth > 500 and workers < 20
if [ $QUEUE_DEPTH -gt 500 ] && [ $ACTIVE_WORKERS -lt 20 ]; then
    NEXT_ID=$((ACTIVE_WORKERS + 1))
    sudo systemctl start celery-worker@$NEXT_ID
    echo "Scaled up to $NEXT_ID workers"
fi

# Scale down if queue depth < 50 and workers > 2
if [ $QUEUE_DEPTH -lt 50 ] && [ $ACTIVE_WORKERS -gt 2 ]; then
    sudo systemctl stop celery-worker@$ACTIVE_WORKERS
    echo "Scaled down to $((ACTIVE_WORKERS - 1)) workers"
fi
```

Add cron job:
```bash
*/5 * * * * /usr/local/bin/celery-autoscale.sh >> /var/log/celery-autoscale.log 2>&1
```

---

## API Authentication

### Token-Based Authentication Setup

**Generate secret key:**
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Configure FastAPI (`api_server.py`):**

```python
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib

app = FastAPI()
security = HTTPBearer()

# Store tokens in Redis or database
VALID_TOKENS = {
    "token_abc123def456": {"user": "admin", "tier": "premium"},
    "token_xyz789ghi012": {"user": "standard_user", "tier": "standard"},
}

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in VALID_TOKENS:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return VALID_TOKENS[token]

@app.post("/api/v1/batch/upload")
async def upload_batch(
    documents: list,
    user_info: dict = Depends(verify_token)
):
    # Check tier limits
    if user_info["tier"] == "standard" and len(documents) > 50:
        raise HTTPException(status_code=403, detail="Standard tier limited to 50 documents per batch")
    
    # Process upload
    return {"job_id": "batch_123", "status": "queued"}
```

### Token Generation Script

Create `generate_token.py`:

```python
import secrets
import json
from datetime import datetime, timedelta

def generate_token(user, tier="standard", expires_days=365):
    token = f"token_{secrets.token_urlsafe(32)}"
    expires = (datetime.now() + timedelta(days=expires_days)).isoformat()
    
    token_data = {
        "token": token,
        "user": user,
        "tier": tier,
        "created": datetime.now().isoformat(),
        "expires": expires
    }
    
    print(json.dumps(token_data, indent=2))
    return token

if __name__ == "__main__":
    generate_token("client_name", tier="premium")
```

Usage:
```bash
python3 generate_token.py
```

---

## Rate Limiting

### Application-Level Rate Limiting

**Install dependencies:**
```bash
pip install slowapi==0.1.9
```

**Configure in FastAPI:**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/batch/upload")
@limiter.limit("10/minute")
async def upload_batch(request: Request, user_info: dict = Depends(verify_token)):
    # Upload logic
    pass

@app.get("/api/v1/batch/status/{job_id}")
@limiter.limit("100/minute")
async def check_status(request: Request, job_id: str):
    # Status check logic
    pass
```

### Redis-Based Rate Limiting

```python
import redis
import time

redis_client = redis.Redis(host='localhost', port=6379, db=2, password='your_password')

def check_rate_limit(user_id: str, limit: int = 100, window: int = 60):
    key = f"rate_limit:{user_id}:{int(time.time() / window)}"
    current = redis_client.incr(key)
    
    if current == 1:
        redis_client.expire(key, window)
    
    if current > limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {window} seconds.",
            headers={"Retry-After": str(window)}
        )
    
    return current
```

---

## Troubleshooting

### Worker Crash Recovery

**Symptoms:**
- Workers disappearing from Celery inspect
- Tasks stuck in "started" state
- Log files showing segmentation faults or OOM errors

**Investigation:**

1. **Check worker logs:**
```bash
tail -f /var/log/celery/worker_1.log
```

2. **Inspect system resources:**
```bash
# Memory usage
free -h
top -u celery

# Disk space
df -h

# Check for OOM killer
dmesg | grep -i "killed process"
journalctl -xe | grep -i oom
```

3. **Check Celery worker status:**
```bash
celery -A batch_worker inspect active
celery -A batch_worker inspect registered
celery -A batch_worker inspect stats
```

**Solutions:**

1. **Increase memory limits:**
   - Reduce `--concurrency` from 8 to 4
   - Reduce `worker_prefetch_multiplier` from 4 to 2
   - Add swap space

2. **Enable automatic restart:**
```ini
# In systemd service file
Restart=always
RestartSec=10
```

3. **Implement health checks:**
```python
@app.task(bind=True)
def health_check(self):
    return {"status": "healthy", "worker": self.request.hostname}

# Run periodically
from celery.schedules import crontab
app.conf.beat_schedule = {
    'health-check': {
        'task': 'batch_worker.health_check',
        'schedule': crontab(minute='*/5'),
    },
}
```

### Redis Queue Backpressure Mitigation

**Symptoms:**
- Queue depth > 5000 tasks
- Redis memory usage > 80%
- Slow task enqueue times

**Investigation:**

```bash
# Check queue depth
redis-cli -a your_password llen decalogo_batch_tasks

# Monitor Redis memory
redis-cli -a your_password info memory

# Check slow queries
redis-cli -a your_password slowlog get 10
```

**Solutions:**

1. **Add more workers:**
```bash
# Start 4 additional workers
for i in {5..8}; do
    systemctl start celery-worker@$i
done
```

2. **Increase Redis memory:**
```conf
# Edit /etc/redis/redis.conf
maxmemory 4gb
```

3. **Implement queue priorities:**
```python
# High priority tasks
process_document.apply_async(args=[doc], priority=9, queue='high_priority')

# Normal priority
process_document.apply_async(args=[doc], priority=5, queue='decalogo_batch_tasks')

# Low priority
process_document.apply_async(args=[doc], priority=1, queue='low_priority')
```

4. **Reject new tasks when queue is full:**
```python
MAX_QUEUE_DEPTH = 10000

def enqueue_batch(documents):
    queue_depth = redis_client.llen('decalogo_batch_tasks')
    
    if queue_depth > MAX_QUEUE_DEPTH:
        raise HTTPException(
            status_code=503,
            detail="Queue capacity exceeded, please retry later"
        )
    
    # Enqueue tasks
    for doc in documents:
        process_document.apply_async(args=[doc])
```

### Timeout Adjustment Strategies

**Symptoms:**
- Tasks timing out before completion
- `SoftTimeLimitExceeded` or `TimeLimitExceeded` errors
- Partial results for large documents

**Investigation:**

```bash
# Check task execution times
celery -A batch_worker inspect stats | grep -A 5 "task-runtime"

# Monitor long-running tasks
celery -A batch_worker inspect active | grep -B 3 "time_start"
```

**Solutions:**

1. **Increase task time limits:**
```python
app.conf.update(
    task_time_limit=7200,  # 2 hours hard limit
    task_soft_time_limit=6900,  # 1h 55m soft limit
)
```

2. **Per-task timeout configuration:**
```python
@app.task(bind=True, time_limit=3600, soft_time_limit=3300)
def process_large_document(self, doc):
    # Processing logic
    pass
```

3. **Implement chunking for large documents:**
```python
@app.task
def process_document_chunk(chunk_id, chunk_content):
    # Process chunk
    return result

@app.task
def process_large_document(document_id, document_content):
    # Split into chunks
    chunks = split_into_chunks(document_content, chunk_size=50000)
    
    # Process chunks in parallel
    job = group(process_document_chunk.s(i, chunk) for i, chunk in enumerate(chunks))
    result = job.apply_async()
    
    # Aggregate results
    return aggregate_results(result.get())
```

4. **Graceful timeout handling:**
```python
from celery.exceptions import SoftTimeLimitExceeded

@app.task(bind=True, soft_time_limit=3300)
def process_document(self, doc_id, doc_content):
    try:
        # Processing logic
        return evaluate_document(doc_content)
    except SoftTimeLimitExceeded:
        # Save partial results
        save_partial_results(doc_id, partial_data)
        # Re-queue with higher timeout
        self.retry(countdown=60, max_retries=1, time_limit=7200)
```

---

## Performance Tuning Matrix

Recommended worker counts and concurrency settings for different hardware profiles to achieve target throughput.

### Hardware Profiles and Recommendations

| CPU Cores | RAM (GB) | Workers | Concurrency | Prefetch | Max Tasks/Child | Expected Throughput (docs/hour) | Notes |
|-----------|----------|---------|-------------|----------|-----------------|----------------------------------|-------|
| 2 | 4 | 1 | 4 | 2 | 100 | 30-40 | Minimal setup, suitable for testing |
| 4 | 8 | 2 | 4 | 4 | 100 | 60-80 | Small deployment |
| 8 | 16 | 4 | 8 | 4 | 100 | **150-180** | **Recommended for 170 docs/hour target** |
| 16 | 32 | 4 | 16 | 4 | 50 | 250-300 | High performance, single machine |
| 16 | 32 | 8 | 8 | 4 | 100 | 300-350 | High throughput, balanced |
| 32 | 64 | 8 | 16 | 2 | 50 | 500-600 | Maximum single-machine performance |

### Multi-Machine Deployment

| Total CPU Cores | Machines | Workers per Machine | Concurrency | Expected Throughput (docs/hour) | Cost Efficiency |
|-----------------|----------|---------------------|-------------|----------------------------------|-----------------|
| 16 | 2×8 | 2 | 8 | 150-180 | High |
| 32 | 4×8 | 2 | 8 | 300-360 | High |
| 64 | 4×16 | 4 | 8 | 600-720 | Medium |
| 128 | 8×16 | 4 | 8 | 1200-1440 | Medium |

### Tuning Parameters Explained

**Concurrency:**
- Number of threads/processes per worker
- Higher = more parallelism, but more memory usage
- Formula: `concurrency = CPU_cores_per_worker * 2` (thread pool)
- Recommended: 4-8 for I/O-bound tasks, 8-16 for CPU-bound

**Prefetch Multiplier:**
- Number of tasks to prefetch per worker thread
- Higher = better throughput, but risks task loss on worker crash
- Formula: `total_prefetch = concurrency * prefetch_multiplier`
- Recommended: 4 for reliable processing, 2 for critical tasks

**Max Tasks Per Child:**
- Worker restarts after processing this many tasks
- Prevents memory leaks from accumulating
- Lower = more restarts, higher = better performance but risk of leaks
- Recommended: 100 for production, 50 for memory-intensive tasks

### Performance Optimization Tips

1. **For Maximum Throughput:**
   - Increase concurrency to 16
   - Use prefetch_multiplier=2
   - Deploy across multiple machines
   - Use SSD storage for artifacts

2. **For Maximum Reliability:**
   - Keep concurrency at 4-8
   - Use prefetch_multiplier=2
   - Set max_tasks_per_child=50
   - Enable task_acks_late=True

3. **For Cost Efficiency:**
   - Use 8-core machines with 2 workers @ concurrency=8
   - Target 150-180 docs/hour per machine
   - Scale horizontally only when needed

4. **For Low Latency (Quick Response):**
   - Increase worker count
   - Reduce prefetch_multiplier to 1
   - Use priority queues
   - Co-locate workers with Redis

### Monitoring Recommendations

```bash
# Monitor throughput
watch -n 5 'celery -A batch_worker inspect stats | grep total'

# Monitor queue depth
watch -n 5 'redis-cli -a your_password llen decalogo_batch_tasks'

# Monitor resource usage
htop -u celery

# Monitor Redis memory
watch -n 5 'redis-cli -a your_password info memory | grep used_memory_human'
```

### Performance Benchmarking

Run this script to measure your actual throughput:

```python
import time
from celery import group
from batch_worker import process_document

def benchmark_throughput(num_docs=100):
    start_time = time.time()
    
    # Create test documents
    docs = [{"id": f"doc_{i}", "content": "test content"} for i in range(num_docs)]
    
    # Submit batch
    job = group(process_document.s(doc["id"], doc["content"]) for doc in docs)
    result = job.apply_async()
    
    # Wait for completion
    result.get()
    
    elapsed = time.time() - start_time
    throughput = (num_docs / elapsed) * 3600
    
    print(f"Processed {num_docs} documents in {elapsed:.2f} seconds")
    print(f"Throughput: {throughput:.2f} documents/hour")

if __name__ == "__main__":
    benchmark_throughput(100)
```

---

## Quick Start Checklist

- [ ] Install Redis and verify with `redis-cli ping`
- [ ] Install Python dependencies with pip
- [ ] Configure `batch_worker.py` with Redis connection
- [ ] Start 4 workers with concurrency=8
- [ ] Install and configure Nginx reverse proxy
- [ ] Generate API authentication tokens
- [ ] Test with sample upload to `/api/v1/batch/upload`
- [ ] Monitor worker status with `celery inspect`
- [ ] Verify throughput meets 170 docs/hour target
- [ ] Set up systemd services for auto-start
- [ ] Configure monitoring and alerting

---

## API Usage Examples

### Python Client Examples

#### Example 1: Single Document Upload with Status Tracking

```python
#!/usr/bin/env python3
"""
Single document upload with polling for completion.
"""
import requests
import time
import json
from typing import Dict, Any

API_BASE_URL = "https://batch.decalogo.example.com/api/v1"
AUTH_TOKEN = "your_token_here"

def upload_document(document: Dict[str, Any]) -> str:
    """Upload a single document and return job_id"""
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": [document],
        "options": {
            "priority": "normal",
            "include_evidence": True,
            "include_questionnaire": True
        }
    }
    
    response = requests.post(
        f"{API_BASE_URL}/batch/upload",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code == 202:
        job_data = response.json()
        print(f"✓ Job submitted: {job_data['job_id']}")
        return job_data['job_id']
    elif response.status_code == 401:
        raise Exception("Authentication failed. Check your token.")
    elif response.status_code == 429:
        retry_after = response.headers.get('Retry-After', 60)
        raise Exception(f"Rate limit exceeded. Retry after {retry_after} seconds.")
    else:
        raise Exception(f"Upload failed: {response.status_code} - {response.text}")

def check_status(job_id: str) -> Dict[str, Any]:
    """Check job status"""
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    
    response = requests.get(
        f"{API_BASE_URL}/batch/status/{job_id}",
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        raise Exception(f"Job {job_id} not found or expired")
    else:
        raise Exception(f"Status check failed: {response.status_code}")

def get_results(job_id: str) -> Dict[str, Any]:
    """Retrieve job results"""
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    
    response = requests.get(
        f"{API_BASE_URL}/batch/results/{job_id}",
        headers=headers,
        params={"include_evidence": True, "format": "json"},
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 202:
        status_data = response.json()
        print(f"Job still processing: {status_data['progress']}%")
        return None
    else:
        raise Exception(f"Failed to get results: {response.status_code}")

def wait_for_completion(job_id: str, poll_interval: int = 10, timeout: int = 3600) -> Dict[str, Any]:
    """Poll job status until completion"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = check_status(job_id)
        
        if status['status'] == 'completed':
            print(f"✓ Job completed in {time.time() - start_time:.1f} seconds")
            return get_results(job_id)
        elif status['status'] == 'failed':
            raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
        elif status['status'] == 'cancelled':
            raise Exception("Job was cancelled")
        else:
            progress = status.get('progress', {}).get('percent_complete', 0)
            print(f"Processing... {progress:.1f}% complete")
            time.sleep(poll_interval)
    
    raise Exception(f"Job timed out after {timeout} seconds")

def main():
    # Sample document
    document = {
        "id": "doc-001",
        "title": "Plan de Desarrollo Municipal Florencia 2024",
        "content": open("plan_florencia.txt", "r", encoding="utf-8").read(),
        "metadata": {
            "municipality": "Florencia",
            "department": "Caquetá",
            "year": 2024
        }
    }
    
    print("Uploading document...")
    job_id = upload_document(document)
    
    print("Waiting for completion...")
    results = wait_for_completion(job_id, poll_interval=5)
    
    print("\n=== Results ===")
    doc_result = results['documents'][0]
    print(f"Document: {doc_result['title']}")
    print(f"Status: {doc_result['status']}")
    print(f"Processing time: {doc_result['processing_time_seconds']:.1f}s")
    print(f"\nDecálogo score: {doc_result['scores']['decalogo']['total_score']:.1f}/100")
    print(f"Questionnaire score: {doc_result['scores']['questionnaire']['total_score']:.1f}/60")
    
    # Save results
    with open(f"results_{job_id}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to results_{job_id}.json")

if __name__ == "__main__":
    main()
```

#### Example 2: Batch Upload with Progress Tracking

```python
#!/usr/bin/env python3
"""
Batch document upload with real-time progress tracking.
"""
import requests
import time
import json
import glob
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE_URL = "https://batch.decalogo.example.com/api/v1"
AUTH_TOKEN = "your_token_here"

def upload_batch(documents: List[Dict[str, Any]], priority: str = "normal") -> str:
    """Upload multiple documents as a batch"""
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": documents,
        "options": {
            "priority": priority,
            "include_evidence": True,
            "include_questionnaire": True,
            "webhook_url": "https://your-app.com/webhook/batch-complete"
        }
    }
    
    response = requests.post(
        f"{API_BASE_URL}/batch/upload",
        headers=headers,
        json=payload,
        timeout=60
    )
    
    if response.status_code == 202:
        return response.json()['job_id']
    else:
        raise Exception(f"Batch upload failed: {response.status_code} - {response.text}")

def track_progress(job_id: str, update_interval: int = 10):
    """Track and display job progress"""
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    
    print(f"Tracking job: {job_id}")
    print("=" * 60)
    
    while True:
        response = requests.get(
            f"{API_BASE_URL}/batch/status/{job_id}",
            headers=headers,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"Error checking status: {response.status_code}")
            break
        
        status = response.json()
        
        if status['status'] == 'completed':
            print("\n✓ Batch processing completed!")
            break
        elif status['status'] in ['failed', 'cancelled']:
            print(f"\n✗ Job {status['status']}")
            break
        
        progress = status['progress']
        percent = progress['percent_complete']
        completed = progress['completed_documents']
        total = progress['total_documents']
        failed = progress['failed_documents']
        
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r[{bar}] {percent:.1f}% | {completed}/{total} docs | {failed} failed", end='')
        
        if status['status'] == 'processing':
            time.sleep(update_interval)
        else:
            time.sleep(update_interval)
    
    print("\n" + "=" * 60)

def load_documents_from_directory(directory: str, max_docs: int = 100) -> List[Dict[str, Any]]:
    """Load documents from text files in a directory"""
    documents = []
    
    for i, filepath in enumerate(glob.glob(f"{directory}/*.txt")[:max_docs]):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = filepath.split('/')[-1].replace('.txt', '')
        documents.append({
            "id": f"doc-{i+1:03d}",
            "title": filename.replace('_', ' ').title(),
            "content": content,
            "metadata": {
                "source_file": filepath
            }
        })
    
    return documents

def main():
    # Load documents
    print("Loading documents from directory...")
    documents = load_documents_from_directory("./plans", max_docs=50)
    print(f"Loaded {len(documents)} documents")
    
    # Upload batch
    print("Uploading batch...")
    job_id = upload_batch(documents, priority="high")
    print(f"Job ID: {job_id}")
    
    # Track progress
    track_progress(job_id, update_interval=5)
    
    # Get results
    print("Fetching results...")
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    response = requests.get(
        f"{API_BASE_URL}/batch/results/{job_id}",
        headers=headers,
        timeout=30
    )
    
    if response.status_code == 200:
        results = response.json()
        
        print("\n=== Summary ===")
        summary = results['summary']
        print(f"Average Decálogo score: {summary['avg_decalogo_score']:.1f}")
        print(f"Average Questionnaire score: {summary['avg_questionnaire_score']:.1f}")
        print(f"Total violations: {summary['total_violations']}")
        print(f"Documents with warnings: {summary['documents_with_warnings']}")
        
        # Save results
        output_file = f"batch_results_{job_id}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
    else:
        print(f"Failed to retrieve results: {response.status_code}")

if __name__ == "__main__":
    main()
```

#### Example 3: Async Batch Processing with Webhook

```python
#!/usr/bin/env python3
"""
Asynchronous batch processing with webhook callback.
"""
import requests
import json
from flask import Flask, request, jsonify

API_BASE_URL = "https://batch.decalogo.example.com/api/v1"
AUTH_TOKEN = "your_token_here"

# Flask app to receive webhook callbacks
app = Flask(__name__)

def upload_batch_async(documents, webhook_url):
    """Upload batch with webhook callback"""
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": documents,
        "options": {
            "priority": "normal",
            "include_evidence": True,
            "include_questionnaire": True,
            "webhook_url": webhook_url
        }
    }
    
    response = requests.post(
        f"{API_BASE_URL}/batch/upload",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 202:
        job_data = response.json()
        print(f"Job submitted: {job_data['job_id']}")
        print(f"Webhook will be called at: {webhook_url}")
        return job_data['job_id']
    else:
        raise Exception(f"Upload failed: {response.status_code}")

@app.route('/webhook/batch-complete', methods=['POST'])
def webhook_handler():
    """Handle batch completion webhook"""
    data = request.json
    
    job_id = data.get('job_id')
    status = data.get('status')
    
    print(f"\n=== Webhook Received ===")
    print(f"Job ID: {job_id}")
    print(f"Status: {status}")
    
    if status == 'completed':
        # Fetch full results
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        response = requests.get(
            f"{API_BASE_URL}/batch/results/{job_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            results = response.json()
            
            # Process results (e.g., store in database, send notifications)
            print(f"Documents processed: {results['document_count']}")
            print(f"Average score: {results['summary']['avg_decalogo_score']:.1f}")
            
            # Save to file
            with open(f"results_{job_id}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to results_{job_id}.json")
    
    return jsonify({"status": "received"}), 200

if __name__ == "__main__":
    # Start Flask server to receive webhooks
    print("Starting webhook server on port 5000...")
    print("Make sure to configure your public webhook URL")
    app.run(host='0.0.0.0', port=5000)
```

### cURL Examples

#### Upload Single Document

```bash
#!/bin/bash
# upload_document.sh

API_BASE="https://batch.decalogo.example.com/api/v1"
TOKEN="your_token_here"

curl -X POST "${API_BASE}/batch/upload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "doc-001",
        "title": "Plan de Desarrollo Municipal 2024",
        "content": "Contenido del plan...",
        "metadata": {
          "municipality": "Florencia",
          "department": "Caquetá",
          "year": 2024
        }
      }
    ],
    "options": {
      "priority": "normal",
      "include_evidence": true,
      "include_questionnaire": true
    }
  }' \
  | jq '.'

# Save job_id
JOB_ID=$(curl -s -X POST "${API_BASE}/batch/upload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d @document.json \
  | jq -r '.job_id')

echo "Job ID: ${JOB_ID}"
```

#### Check Job Status

```bash
#!/bin/bash
# check_status.sh

API_BASE="https://batch.decalogo.example.com/api/v1"
TOKEN="your_token_here"
JOB_ID="$1"

if [ -z "$JOB_ID" ]; then
  echo "Usage: $0 <job_id>"
  exit 1
fi

curl -X GET "${API_BASE}/batch/status/${JOB_ID}" \
  -H "Authorization: Bearer ${TOKEN}" \
  | jq '{
      status: .status,
      progress: .progress.percent_complete,
      completed: .progress.completed_documents,
      total: .progress.total_documents
    }'
```

#### Get Results

```bash
#!/bin/bash
# get_results.sh

API_BASE="https://batch.decalogo.example.com/api/v1"
TOKEN="your_token_here"
JOB_ID="$1"

if [ -z "$JOB_ID" ]; then
  echo "Usage: $0 <job_id>"
  exit 1
fi

curl -X GET "${API_BASE}/batch/results/${JOB_ID}?include_evidence=true&format=json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -o "results_${JOB_ID}.json"

echo "Results saved to results_${JOB_ID}.json"

# Display summary
jq '.summary' "results_${JOB_ID}.json"
```

#### Cancel Job

```bash
#!/bin/bash
# cancel_job.sh

API_BASE="https://batch.decalogo.example.com/api/v1"
TOKEN="your_token_here"
JOB_ID="$1"

if [ -z "$JOB_ID" ]; then
  echo "Usage: $0 <job_id>"
  exit 1
fi

curl -X DELETE "${API_BASE}/batch/${JOB_ID}" \
  -H "Authorization: Bearer ${TOKEN}" \
  | jq '.'
```

#### Health Check

```bash
#!/bin/bash
# health_check.sh

API_BASE="https://batch.decalogo.example.com/api/v1"

curl -X GET "${API_BASE}/health" | jq '{
  status: .status,
  components: .components | to_entries | map({name: .key, status: .value.status}),
  queue_length: .metrics.jobs_in_queue,
  throughput: .metrics.throughput_docs_per_hour
}'
```

#### Get Metrics

```bash
#!/bin/bash
# get_metrics.sh

API_BASE="https://batch.decalogo.example.com/api/v1"
TOKEN="your_token_here"

curl -X GET "${API_BASE}/metrics" \
  -H "Authorization: Bearer ${TOKEN}" \
  | grep -E "^(decalogo_batch_|# HELP|# TYPE)"
```

#### Complete Workflow Script

```bash
#!/bin/bash
# complete_workflow.sh - Upload, track, and download results

set -e

API_BASE="https://batch.decalogo.example.com/api/v1"
TOKEN="your_token_here"
DOCUMENT_FILE="$1"

if [ -z "$DOCUMENT_FILE" ]; then
  echo "Usage: $0 <document.json>"
  exit 1
fi

echo "=== Uploading document ==="
JOB_ID=$(curl -s -X POST "${API_BASE}/batch/upload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d @"${DOCUMENT_FILE}" \
  | jq -r '.job_id')

echo "Job ID: ${JOB_ID}"

echo -e "\n=== Tracking progress ==="
while true; do
  STATUS=$(curl -s -X GET "${API_BASE}/batch/status/${JOB_ID}" \
    -H "Authorization: Bearer ${TOKEN}")
  
  STATE=$(echo "$STATUS" | jq -r '.status')
  PROGRESS=$(echo "$STATUS" | jq -r '.progress.percent_complete // 0')
  
  echo -ne "\rStatus: ${STATE} | Progress: ${PROGRESS}%"
  
  if [ "$STATE" == "completed" ]; then
    echo -e "\n✓ Job completed!"
    break
  elif [ "$STATE" == "failed" ] || [ "$STATE" == "cancelled" ]; then
    echo -e "\n✗ Job ${STATE}"
    exit 1
  fi
  
  sleep 5
done

echo -e "\n=== Downloading results ==="
curl -s -X GET "${API_BASE}/batch/results/${JOB_ID}" \
  -H "Authorization: Bearer ${TOKEN}" \
  -o "results_${JOB_ID}.json"

echo "Results saved to: results_${JOB_ID}.json"

echo -e "\n=== Summary ==="
jq '.summary' "results_${JOB_ID}.json"
```

## Advanced Troubleshooting

### Worker Memory Leaks

**Symptoms:**
- Worker memory usage gradually increasing over time
- OOM kills after hours/days of operation
- Degraded performance after processing many documents

**Investigation:**

```bash
# Track memory usage over time
while true; do
  date >> worker_memory.log
  ps aux | grep celery | grep -v grep >> worker_memory.log
  sleep 300  # Every 5 minutes
done

# Analyze memory growth
awk '{if ($6 ~ /^[0-9]+$/) print $6}' worker_memory.log | \
  awk '{sum+=$1; count++} END {print "Avg RSS:", sum/count/1024, "MB"}'
```

**Solutions:**

1. **Reduce max_tasks_per_child:**
```python
app.conf.worker_max_tasks_per_child = 50  # Restart worker after 50 tasks
```

2. **Enable memory profiling:**
```python
import tracemalloc

@app.task(bind=True)
def process_document(self, doc_id, doc_content):
    tracemalloc.start()
    
    try:
        result = evaluate_document(doc_content)
        return result
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"Memory usage: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")
```

3. **Explicit garbage collection:**
```python
import gc

@app.task
def process_document(doc_id, doc_content):
    result = evaluate_document(doc_content)
    gc.collect()  # Force garbage collection
    return result
```

### Redis Connection Pool Exhaustion

**Symptoms:**
- `ConnectionError: Too many connections`
- Tasks failing to enqueue
- Slow Redis operations

**Investigation:**

```bash
# Check current connections
redis-cli -a your_password CLIENT LIST | wc -l

# Monitor connection count
watch -n 2 'redis-cli -a your_password CLIENT LIST | wc -l'

# Check connection pool settings
redis-cli -a your_password CONFIG GET maxclients
```

**Solutions:**

1. **Increase Redis max clients:**
```bash
redis-cli -a your_password CONFIG SET maxclients 10000
```

Edit `/etc/redis/redis.conf`:
```conf
maxclients 10000
```

2. **Configure Celery connection pool:**
```python
app.conf.broker_pool_limit = 50  # Increase pool size
app.conf.broker_connection_max_retries = 3
app.conf.broker_connection_retry_on_startup = True
```

3. **Enable connection reuse:**
```python
from kombu import Connection

# Reuse connections
app.conf.broker_transport_options = {
    'max_connections': 100,
    'socket_timeout': 10,
    'socket_connect_timeout': 10,
}
```

### Task Result Loss

**Symptoms:**
- Results not available when querying
- `KeyError` when accessing result.get()
- Inconsistent result retrieval

**Investigation:**

```bash
# Check result backend keys
redis-cli -a your_password KEYS "celery-task-meta-*" | wc -l

# Check TTL on results
redis-cli -a your_password TTL "celery-task-meta-<task_id>"

# Monitor result expirations
redis-cli -a your_password --scan --pattern "celery-task-meta-*" | \
  while read key; do
    ttl=$(redis-cli -a your_password TTL "$key")
    echo "$key: $ttl seconds"
  done
```

**Solutions:**

1. **Increase result expiration:**
```python
app.conf.result_expires = 2592000  # 30 days instead of default 24 hours
```

2. **Store results in persistent storage:**
```python
import json

@app.task(bind=True)
def process_document(self, doc_id, doc_content):
    result = evaluate_document(doc_content)
    
    # Store in persistent storage (S3, database, etc.)
    with open(f"/var/results/{self.request.id}.json", 'w') as f:
        json.dump(result, f)
    
    return result
```

3. **Use result backend with persistence:**
```python
# Use database backend instead of Redis
app.conf.result_backend = 'db+postgresql://user:pass@localhost/celery_results'
```

## Additional Resources

- [Celery Documentation](https://docs.celeryq.dev/)
- [Redis Documentation](https://redis.io/documentation)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [DEPLOYMENT_INFRASTRUCTURE.md](./DEPLOYMENT_INFRASTRUCTURE.md) - Architecture details
- [Postman Collection](./batch_processing_postman_collection.json) - Pre-configured API requests

For issues and questions, see the Troubleshooting section above or file an issue on GitHub.
