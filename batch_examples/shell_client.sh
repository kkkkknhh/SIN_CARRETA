#!/bin/bash
# Shell script client for DECALOGO batch processing API using curl
# 
# Usage:
#   ./shell_client.sh upload documents.json YOUR_TOKEN
#   ./shell_client.sh status JOB_ID YOUR_TOKEN
#   ./shell_client.sh results JOB_ID YOUR_TOKEN
#   ./shell_client.sh health YOUR_TOKEN

set -e

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
POLL_INTERVAL=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
error() {
    echo -e "${RED}✗ Error: $1${NC}" >&2
    exit 1
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

info() {
    echo "ℹ $1"
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    error "jq is required but not installed. Install with: sudo apt-get install jq"
fi

# Function to upload documents
upload_documents() {
    local file=$1
    local token=$2
    
    if [ ! -f "$file" ]; then
        error "File not found: $file"
    fi
    
    info "Uploading documents from $file..."
    
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Authorization: Bearer $token" \
        -H "Content-Type: application/json" \
        -d @"$file" \
        "$API_URL/api/v1/batch/upload")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -eq 202 ]; then
        job_id=$(echo "$body" | jq -r '.job_id')
        doc_count=$(echo "$body" | jq -r '.document_count')
        estimated_time=$(echo "$body" | jq -r '.estimated_completion_seconds')
        
        success "Upload successful"
        echo "  Job ID: $job_id"
        echo "  Documents: $doc_count"
        echo "  Estimated completion: ${estimated_time}s"
        echo ""
        echo "  Status URL: $API_URL/api/v1/batch/status/$job_id"
        echo "  Results URL: $API_URL/api/v1/batch/results/$job_id"
        
        # Save job ID to file for convenience
        echo "$job_id" > .last_job_id
        
        echo "$job_id"
    elif [ "$http_code" -eq 401 ]; then
        error "Authentication failed. Check your token."
    elif [ "$http_code" -eq 429 ]; then
        retry_after=$(echo "$body" | jq -r '.retry_after_seconds // 60')
        error "Rate limit exceeded. Retry after ${retry_after}s"
    else
        error "Upload failed with HTTP $http_code: $(echo "$body" | jq -r '.message // .detail // "Unknown error"')"
    fi
}

# Function to check job status
check_status() {
    local job_id=$1
    local token=$2
    
    response=$(curl -s -w "\n%{http_code}" \
        -H "Authorization: Bearer $token" \
        "$API_URL/api/v1/batch/status/$job_id")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -eq 200 ]; then
        echo "$body"
    elif [ "$http_code" -eq 404 ]; then
        error "Job not found: $job_id"
    else
        error "Status check failed with HTTP $http_code"
    fi
}

# Function to wait for job completion
wait_for_completion() {
    local job_id=$1
    local token=$2
    
    info "Waiting for job $job_id to complete..."
    echo -n "Status: "
    
    while true; do
        status_json=$(check_status "$job_id" "$token")
        status=$(echo "$status_json" | jq -r '.status')
        
        case $status in
            queued)
                echo -n "Q"
                ;;
            processing)
                progress=$(echo "$status_json" | jq -r '.progress_percent // 0')
                docs_completed=$(echo "$status_json" | jq -r '.documents_completed // 0')
                docs_total=$(echo "$status_json" | jq -r '.document_count // 0')
                echo -ne "\rProcessing: ${progress}% ($docs_completed/$docs_total docs)  "
                ;;
            completed)
                echo ""
                success "Job completed!"
                
                docs_completed=$(echo "$status_json" | jq -r '.documents_completed')
                docs_failed=$(echo "$status_json" | jq -r '.documents_failed')
                total_time=$(echo "$status_json" | jq -r '.total_processing_seconds // 0')
                
                echo "  Documents completed: $docs_completed"
                echo "  Documents failed: $docs_failed"
                echo "  Total time: ${total_time}s"
                
                return 0
                ;;
            failed)
                echo ""
                error=$(echo "$status_json" | jq -r '.error // "Unknown error"')
                error "Job failed: $error"
                ;;
            *)
                echo ""
                warning "Unknown status: $status"
                ;;
        esac
        
        sleep $POLL_INTERVAL
    done
}

# Function to retrieve results
get_results() {
    local job_id=$1
    local token=$2
    local format=${3:-json}
    local output_file=$4
    
    info "Retrieving results for job $job_id..."
    
    response=$(curl -s -w "\n%{http_code}" \
        -H "Authorization: Bearer $token" \
        "$API_URL/api/v1/batch/results/$job_id?format=$format&include_evidence=true")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -eq 200 ]; then
        if [ -n "$output_file" ]; then
            echo "$body" > "$output_file"
            success "Results saved to $output_file"
        else
            echo "$body" | jq '.'
        fi
        
        # Print summary
        if [ "$format" = "json" ]; then
            echo ""
            echo "=========================================="
            echo "RESULTS SUMMARY"
            echo "=========================================="
            echo "$body" | jq -r '
                "Total documents: \(.document_count)",
                "Completed: \(.documents_completed)",
                "Failed: \(.documents_failed)",
                "Average score: \(.summary.average_score)",
                "Median score: \(.summary.median_score)",
                "Total evidence: \(.summary.total_evidence_extracted)",
                "Contradictions: \(.summary.total_contradictions)",
                "Avg processing time: \(.summary.average_processing_time_seconds)s"
            '
            echo "=========================================="
        fi
    elif [ "$http_code" -eq 202 ]; then
        warning "Job still processing. Wait for completion first."
    elif [ "$http_code" -eq 404 ]; then
        error "Job not found: $job_id"
    else
        error "Failed to retrieve results with HTTP $http_code"
    fi
}

# Function to check API health
check_health() {
    local token=$1
    
    response=$(curl -s -w "\n%{http_code}" \
        -H "Authorization: Bearer $token" \
        "$API_URL/api/v1/health")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -eq 200 ]; then
        echo "$body" | jq '.'
        
        status=$(echo "$body" | jq -r '.status')
        if [ "$status" = "healthy" ]; then
            success "API is healthy"
        else
            warning "API status: $status"
        fi
    else
        error "Health check failed with HTTP $http_code"
    fi
}

# Function to create sample documents file
create_sample() {
    local output_file=${1:-sample_documents.json}
    
    cat > "$output_file" <<'EOF'
{
  "documents": [
    {
      "id": "doc_001",
      "content": "Plan Nacional de Desarrollo Social 2024-2028. Diagnóstico: La pobreza afecta al 42% de la población. Causalidad: La falta de acceso a educación y salud perpetúa el ciclo de pobreza. Objetivos: Reducir la pobreza al 30% en 4 años mediante programas de educación, salud y empleo. Indicadores: Tasa de pobreza, cobertura educativa, acceso a salud.",
      "metadata": {
        "title": "Plan Nacional de Desarrollo Social",
        "year": 2024,
        "sector": "Social"
      }
    },
    {
      "id": "doc_002",
      "content": "Programa de Infraestructura Vial 2024. Diagnóstico: El 60% de las vías rurales están en mal estado, limitando el acceso a mercados. Objetivos: Rehabilitar 5000 km de vías rurales para mejorar la conectividad y el comercio. Presupuesto: $500 millones USD. Cronograma: 3 años. Beneficiarios: 2 millones de personas en zonas rurales.",
      "metadata": {
        "title": "Programa de Infraestructura Vial",
        "year": 2024,
        "sector": "Infraestructura"
      }
    },
    {
      "id": "doc_003",
      "content": "Plan de Salud Pública 2024-2030. Diagnóstico: Alta incidencia de enfermedades prevenibles por falta de vacunación. Causalidad: La baja cobertura de vacunación se debe a limitaciones de acceso en zonas remotas. Estrategia: Implementar brigadas móviles de vacunación en 200 municipios. Meta: Alcanzar 95% de cobertura de vacunación infantil. Recursos: Personal médico, vehículos, vacunas.",
      "metadata": {
        "title": "Plan de Salud Pública",
        "year": 2024,
        "sector": "Salud"
      }
    }
  ],
  "options": {
    "priority": "normal",
    "include_evidence": true,
    "include_traces": false
  }
}
EOF
    
    success "Sample documents file created: $output_file"
    info "Upload with: ./shell_client.sh upload $output_file YOUR_TOKEN"
}

# Main script
main() {
    local command=$1
    
    case $command in
        upload)
            if [ $# -lt 3 ]; then
                error "Usage: $0 upload <file> <token>"
            fi
            upload_documents "$2" "$3"
            ;;
        status)
            if [ $# -lt 3 ]; then
                error "Usage: $0 status <job_id> <token>"
            fi
            check_status "$2" "$3" | jq '.'
            ;;
        wait)
            if [ $# -lt 3 ]; then
                error "Usage: $0 wait <job_id> <token>"
            fi
            wait_for_completion "$2" "$3"
            ;;
        results)
            if [ $# -lt 3 ]; then
                error "Usage: $0 results <job_id> <token> [format] [output_file]"
            fi
            get_results "$2" "$3" "${4:-json}" "$5"
            ;;
        health)
            if [ $# -lt 2 ]; then
                error "Usage: $0 health <token>"
            fi
            check_health "$2"
            ;;
        workflow)
            # Complete workflow: upload, wait, get results
            if [ $# -lt 3 ]; then
                error "Usage: $0 workflow <file> <token> [output_file]"
            fi
            job_id=$(upload_documents "$2" "$3")
            echo ""
            wait_for_completion "$job_id" "$3"
            echo ""
            get_results "$job_id" "$3" "json" "$4"
            ;;
        sample)
            create_sample "${2:-sample_documents.json}"
            ;;
        *)
            echo "DECALOGO Batch Processing Shell Client"
            echo ""
            echo "Usage: $0 <command> [arguments]"
            echo ""
            echo "Commands:"
            echo "  upload <file> <token>              Upload documents from JSON file"
            echo "  status <job_id> <token>            Check job status"
            echo "  wait <job_id> <token>              Wait for job completion"
            echo "  results <job_id> <token> [format] [output_file]"
            echo "                                     Retrieve results (format: json|csv)"
            echo "  health <token>                     Check API health"
            echo "  workflow <file> <token> [output]   Complete workflow (upload, wait, results)"
            echo "  sample [file]                      Create sample documents file"
            echo ""
            echo "Environment Variables:"
            echo "  API_URL                            API base URL (default: http://localhost:8000)"
            echo ""
            echo "Examples:"
            echo "  $0 sample                          # Create sample_documents.json"
            echo "  $0 upload sample_documents.json abc123"
            echo "  $0 status batch_20240115_abc123 abc123"
            echo "  $0 results batch_20240115_abc123 abc123 json results.json"
            echo "  $0 workflow sample_documents.json abc123 results.json"
            echo ""
            exit 1
            ;;
    esac
}

main "$@"
