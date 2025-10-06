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

## Additional Resources

- [Celery Documentation](https://docs.celeryq.dev/)
- [Redis Documentation](https://redis.io/documentation)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [DEPLOYMENT_INFRASTRUCTURE.md](./DEPLOYMENT_INFRASTRUCTURE.md) - Architecture details

For issues and questions, see the Troubleshooting section above or file an issue on GitHub.
