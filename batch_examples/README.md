# DECALOGO Batch Processing Client Examples

This directory contains three complete client implementations for interacting with the DECALOGO batch processing API.

## Client Implementations

### 1. Python Client (`python_client.py`)

Full-featured Python client using the `requests` library.

**Features:**
- Upload documents from JSON files or directories
- Poll job status with progress indicators
- Retrieve results in JSON or CSV format
- Health checks and error handling
- Complete workflow automation

**Installation:**
```bash
pip install requests
```

**Usage Examples:**

```bash
# Upload documents from JSON file
python3 python_client.py --token abc123 --file documents.json

# Upload text files from directory
python3 python_client.py --token abc123 --dir /path/to/documents/

# Check job status
python3 python_client.py --token abc123 --status batch_20240115_abc123

# Retrieve results
python3 python_client.py --token abc123 --results batch_20240115_abc123 --output results.json

# High priority batch with custom polling
python3 python_client.py --token abc123 --file docs.json --priority high --poll-interval 5

# Upload without waiting for completion
python3 python_client.py --token abc123 --file docs.json --no-wait

# Health check
python3 python_client.py --token abc123 --health

# Custom API URL
python3 python_client.py --url https://batch.decalogo.example.com --token abc123 --file docs.json
```

**Document JSON Format:**
```json
{
  "documents": [
    {
      "id": "doc_001",
      "content": "Plan text content...",
      "metadata": {
        "title": "Plan Nacional",
        "year": 2024
      }
    }
  ],
  "options": {
    "priority": "normal",
    "include_evidence": true
  }
}
```

### 2. Shell Script Client (`shell_client.sh`)

Bash script using `curl` and `jq` for command-line workflows.

**Requirements:**
```bash
# Ubuntu/Debian
sudo apt-get install curl jq

# macOS
brew install curl jq
```

**Usage Examples:**

```bash
# Make executable
chmod +x shell_client.sh

# Create sample documents file
./shell_client.sh sample

# Upload documents
./shell_client.sh upload sample_documents.json abc123

# Check job status
./shell_client.sh status batch_20240115_abc123 abc123

# Wait for completion
./shell_client.sh wait batch_20240115_abc123 abc123

# Retrieve results
./shell_client.sh results batch_20240115_abc123 abc123 json results.json

# Complete workflow (upload + wait + results)
./shell_client.sh workflow sample_documents.json abc123 results.json

# Health check
./shell_client.sh health abc123

# Custom API URL
export API_URL=https://batch.decalogo.example.com
./shell_client.sh upload documents.json abc123
```

**Shell Script in CI/CD:**
```bash
# GitLab CI example
deploy_and_evaluate:
  script:
    - ./shell_client.sh upload production_plans.json $API_TOKEN
    - JOB_ID=$(cat .last_job_id)
    - ./shell_client.sh wait $JOB_ID $API_TOKEN
    - ./shell_client.sh results $JOB_ID $API_TOKEN json evaluation_results.json
    - python3 check_results.py evaluation_results.json
```

### 3. Postman Collection (`postman_collection.json`)

Complete Postman collection with pre-configured requests for all endpoints.

**Features:**
- Pre-request scripts for authentication
- Test scripts to validate responses
- Auto-save job_id for subsequent requests
- Collection variables for easy configuration
- Example requests for all use cases

**Setup:**

1. **Import Collection:**
   - Open Postman
   - Click "Import" → "Upload Files"
   - Select `postman_collection.json`

2. **Configure Variables:**
   - Click collection → "Variables" tab
   - Set `base_url`: `http://localhost:8000` or your API URL
   - Set `api_token`: Your authentication token

3. **Run Requests:**
   - Start with "Health Check" to verify connectivity
   - Use "Upload Documents" to submit a batch
   - Job ID is automatically saved to `{{job_id}}`
   - Use "Check Job Status" to monitor progress
   - Use "Get Results" when job completes

**Collection Variables:**
- `base_url`: API base URL
- `api_token`: Your authentication token
- `job_id`: Auto-populated after upload

**Available Requests:**
- Health Check
- Upload Documents - Small Batch (3 docs)
- Upload Documents - High Priority (1 urgent doc)
- Upload Documents - Large Batch (5 docs)
- Check Job Status
- Get Results - JSON
- Get Results - CSV
- Get Results - With Traces

**Running Collection with Newman (CLI):**
```bash
# Install Newman
npm install -g newman

# Run collection
newman run postman_collection.json \
  --env-var "base_url=http://localhost:8000" \
  --env-var "api_token=abc123"

# Run specific request
newman run postman_collection.json \
  --folder "Health Check" \
  --env-var "api_token=abc123"
```

## Sample Documents

### Minimal Example
```json
{
  "documents": [
    {
      "id": "doc_001",
      "content": "Plan text here..."
    }
  ]
}
```

### Complete Example
```json
{
  "documents": [
    {
      "id": "plan_nacional_2024",
      "content": "Plan Nacional de Desarrollo 2024-2028. Diagnóstico: La pobreza afecta al 42% de la población. Causalidad: La falta de acceso a educación y salud perpetúa el ciclo de pobreza. Objetivos: Reducir la pobreza al 30% mediante programas de educación, salud y empleo. Indicadores: Tasa de pobreza, cobertura educativa, acceso a salud. Presupuesto: $1,000 millones USD. Cronograma: 4 años.",
      "metadata": {
        "title": "Plan Nacional de Desarrollo",
        "year": 2024,
        "sector": "Social",
        "country": "Colombia",
        "author": "Ministerio de Planeación"
      }
    }
  ],
  "options": {
    "priority": "normal",
    "include_evidence": true,
    "include_traces": false,
    "webhook_url": "https://example.com/webhook"
  }
}
```

## API Endpoints Reference

### Upload Documents
```
POST /api/v1/batch/upload
Authorization: Bearer <token>
Content-Type: application/json

Response: 202 Accepted
{
  "job_id": "batch_20240115_abc123",
  "status": "queued",
  "document_count": 15,
  "estimated_completion_seconds": 300
}
```

### Check Status
```
GET /api/v1/batch/status/{job_id}
Authorization: Bearer <token>

Response: 200 OK
{
  "job_id": "batch_20240115_abc123",
  "status": "processing",
  "progress_percent": 53.3,
  "documents_completed": 8,
  "documents_failed": 1
}
```

### Get Results
```
GET /api/v1/batch/results/{job_id}?format=json&include_evidence=true
Authorization: Bearer <token>

Response: 200 OK
{
  "job_id": "batch_20240115_abc123",
  "status": "completed",
  "summary": {
    "average_score": 7.8,
    "total_evidence_extracted": 142
  },
  "results": [...]
}
```

### Health Check
```
GET /api/v1/health

Response: 200 OK
{
  "status": "healthy",
  "components": {
    "api": {"status": "healthy"},
    "redis": {"status": "healthy"},
    "workers": {"status": "healthy"}
  }
}
```

## Error Handling

### Rate Limit Exceeded (429)
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit of 100 requests per minute exceeded",
  "retry_after_seconds": 45
}
```

**Handling:**
- Wait for `retry_after_seconds`
- Implement exponential backoff
- Reduce request frequency

### Authentication Failed (401)
```json
{
  "error": "unauthorized",
  "message": "Invalid or missing authentication token"
}
```

**Handling:**
- Verify token is correct
- Check token hasn't expired
- Request new token if needed

### Service Unavailable (503)
```json
{
  "error": "service_unavailable",
  "message": "Queue capacity exceeded, please retry later"
}
```

**Handling:**
- Wait before retrying (use exponential backoff)
- Check system status with health endpoint
- Contact administrator if persistent

## Performance Tips

### Batch Size Optimization
- **Small batches (1-10 docs)**: Low latency, higher overhead
- **Medium batches (11-50 docs)**: Balanced performance
- **Large batches (51-100 docs)**: Maximum throughput, higher latency

### Polling Strategy
- **Fast polling (5s)**: For urgent jobs, higher load on API
- **Standard polling (10-15s)**: Recommended for most jobs
- **Slow polling (30s+)**: For large batches, lower API load

### Priority Levels
- **High**: Emergency responses, critical evaluations (10% capacity)
- **Normal**: Regular operations (80% capacity)
- **Low**: Batch processing, analytics (10% capacity)

## Integration Examples

### Python Integration in Application
```python
from batch_examples.python_client import DecalogoBatchClient

# Initialize client
client = DecalogoBatchClient(
    base_url="https://batch.decalogo.example.com",
    token="your_token_here"
)

# Process documents
documents = load_documents_from_database()
results = client.process_batch(documents, wait=True)

# Store results
save_results_to_database(results)
```

### Shell Script in Cron Job
```bash
#!/bin/bash
# Daily evaluation of new plans

# Load environment
source /opt/decalogo/.env

# Export new plans
python3 export_plans.py > daily_plans.json

# Submit batch
./shell_client.sh workflow daily_plans.json $API_TOKEN daily_results.json

# Import results
python3 import_results.py daily_results.json

# Send notification
mail -s "Daily Evaluation Complete" admin@example.com < daily_results.json
```

### Postman/Newman in CI Pipeline
```yaml
# GitHub Actions
name: Evaluate Plans
on: [push]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Postman Collection
        run: |
          npm install -g newman
          newman run batch_examples/postman_collection.json \
            --env-var "base_url=${{ secrets.API_URL }}" \
            --env-var "api_token=${{ secrets.API_TOKEN }}" \
            --reporters cli,json \
            --reporter-json-export results.json
      - name: Check Results
        run: python3 validate_results.py results.json
```

## Troubleshooting

### Connection Refused
```
Error: Failed to connect to API
```
**Solution:**
- Verify API URL is correct
- Check network connectivity
- Ensure API server is running
- Check firewall rules

### Timeout Errors
```
Error: Request timeout after 300s
```
**Solution:**
- Increase timeout value
- Check API server health
- Verify worker availability
- Reduce batch size

### Invalid Token
```
Error: 401 Unauthorized
```
**Solution:**
- Verify token is correct
- Check token format (should be plain token, not "Bearer token")
- Request new token if expired
- Check token has required permissions

### Job Not Found
```
Error: 404 Not Found - Job ID not found
```
**Solution:**
- Verify job ID is correct
- Check if job was deleted (>30 day retention)
- Ensure using same API instance where job was created

## Additional Resources

- **API Documentation**: See [DEPLOYMENT_INFRASTRUCTURE.md](../DEPLOYMENT_INFRASTRUCTURE.md)
- **Setup Guide**: See [BATCH_PROCESSING_GUIDE.md](../BATCH_PROCESSING_GUIDE.md)
- **Architecture**: See batch processing architecture section in deployment docs
- **Performance Tuning**: See performance tuning matrix in processing guide

## Support

For issues, questions, or feature requests:
1. Check troubleshooting section above
2. Review API documentation
3. File an issue on GitHub
4. Contact system administrator
