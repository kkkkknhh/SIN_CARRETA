#!/usr/bin/env python3
"""
Python client for DECALOGO batch processing API

Usage:
    python3 python_client.py --token YOUR_TOKEN --file documents.json
    python3 python_client.py --token YOUR_TOKEN --dir /path/to/documents/
"""

import requests
import json
import time
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

class DecalogoBatchClient:
    """Client for interacting with the DECALOGO batch processing API"""
    
    def __init__(self, base_url: str, token: str, timeout: int = 300):
        """
        Initialize the batch client
        
        Args:
            base_url: Base URL of the API (e.g., https://batch.decalogo.example.com)
            token: Authentication token
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        })
    
    def upload_documents(self, documents: List[Dict], options: Optional[Dict] = None) -> Dict:
        """
        Upload documents for batch processing
        
        Args:
            documents: List of document dictionaries with 'id', 'content', and optional 'metadata'
            options: Optional processing options (priority, include_evidence, etc.)
        
        Returns:
            Response dictionary with job_id and status
        """
        url = f"{self.base_url}/api/v1/batch/upload"
        
        payload = {
            "documents": documents,
            "options": options or {}
        }
        
        print(f"Uploading {len(documents)} documents...")
        
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            print(f"✓ Upload successful. Job ID: {result['job_id']}")
            return result
        except requests.exceptions.HTTPError as e:
            print(f"✗ HTTP Error: {e}")
            if e.response.status_code == 429:
                retry_after = e.response.headers.get('Retry-After', 60)
                print(f"  Rate limit exceeded. Retry after {retry_after} seconds.")
            elif e.response.status_code == 401:
                print("  Authentication failed. Check your token.")
            else:
                print(f"  Response: {e.response.text}")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
            sys.exit(1)
    
    def check_status(self, job_id: str) -> Dict:
        """
        Check the status of a batch job
        
        Args:
            job_id: Job identifier returned from upload
        
        Returns:
            Status dictionary with progress information
        """
        url = f"{self.base_url}/api/v1/batch/status/{job_id}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"✗ Job not found: {job_id}")
            else:
                print(f"✗ Status check failed: {e}")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
            sys.exit(1)
    
    def wait_for_completion(self, job_id: str, poll_interval: int = 10) -> Dict:
        """
        Poll job status until completion
        
        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks
        
        Returns:
            Final status dictionary
        """
        print(f"\nWaiting for job {job_id} to complete...")
        print("Status: ", end='', flush=True)
        
        while True:
            status = self.check_status(job_id)
            
            if status['status'] == 'queued':
                print("Q", end='', flush=True)
            elif status['status'] == 'processing':
                progress = status.get('progress_percent', 0)
                docs_completed = status.get('documents_completed', 0)
                docs_total = status.get('document_count', 0)
                print(f"\rProcessing: {progress:.1f}% ({docs_completed}/{docs_total} docs)", end='', flush=True)
            elif status['status'] == 'completed':
                print("\n✓ Job completed!")
                print(f"  Documents completed: {status['documents_completed']}")
                print(f"  Documents failed: {status['documents_failed']}")
                print(f"  Total time: {status.get('total_processing_seconds', 0):.1f}s")
                return status
            elif status['status'] == 'failed':
                print("\n✗ Job failed!")
                print(f"  Error: {status.get('error', 'Unknown error')}")
                sys.exit(1)
            
            time.sleep(poll_interval)
    
    def get_results(self, job_id: str, format: str = 'json', 
                   include_evidence: bool = True, include_traces: bool = False) -> Dict:
        """
        Retrieve results for a completed job
        
        Args:
            job_id: Job identifier
            format: Result format ('json' or 'csv')
            include_evidence: Include evidence in results
            include_traces: Include trace information
        
        Returns:
            Results dictionary
        """
        url = f"{self.base_url}/api/v1/batch/results/{job_id}"
        params = {
            'format': format,
            'include_evidence': str(include_evidence).lower(),
            'include_traces': str(include_traces).lower()
        }
        
        print(f"\nRetrieving results for job {job_id}...")
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            if format == 'json':
                results = response.json()
                print("✓ Results retrieved successfully")
                return results
            else:
                # CSV format
                print("✓ CSV results retrieved")
                return {'csv': response.text}
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 202:
                print("✗ Job still processing. Wait for completion first.")
            elif e.response.status_code == 404:
                print(f"✗ Job not found: {job_id}")
            else:
                print(f"✗ Failed to retrieve results: {e}")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
            sys.exit(1)
    
    def health_check(self) -> Dict:
        """Check API health status"""
        url = f"{self.base_url}/api/v1/health"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"✗ Health check failed: {e}")
            sys.exit(1)
    
    def process_batch(self, documents: List[Dict], options: Optional[Dict] = None,
                     wait: bool = True, poll_interval: int = 10) -> Dict:
        """
        Complete workflow: upload, wait, and retrieve results
        
        Args:
            documents: List of documents to process
            options: Processing options
            wait: Wait for completion before returning
            poll_interval: Seconds between status checks
        
        Returns:
            Results dictionary
        """
        # Upload
        upload_response = self.upload_documents(documents, options)
        job_id = upload_response['job_id']
        
        if not wait:
            return upload_response
        
        # Wait for completion
        self.wait_for_completion(job_id, poll_interval)
        
        # Retrieve results
        results = self.get_results(job_id)
        
        return results


def load_documents_from_file(file_path: str) -> List[Dict]:
    """Load documents from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif 'documents' in data:
            return data['documents']
        else:
            return [data]


def load_documents_from_directory(dir_path: str) -> List[Dict]:
    """Load documents from text files in a directory"""
    documents = []
    path = Path(dir_path)
    
    for file_path in path.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'id': file_path.stem,
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'size': len(content)
                }
            })
    
    return documents


def save_results(results: Dict, output_path: str, format: str = 'json'):
    """Save results to a file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        if format == 'json':
            json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            f.write(results.get('csv', ''))
    
    print(f"✓ Results saved to {output_path}")


def print_summary(results: Dict):
    """Print a summary of the results"""
    summary = results.get('summary', {})
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total documents: {results.get('document_count', 0)}")
    print(f"Completed: {results.get('documents_completed', 0)}")
    print(f"Failed: {results.get('documents_failed', 0)}")
    print(f"Average score: {summary.get('average_score', 0):.2f}")
    print(f"Median score: {summary.get('median_score', 0):.2f}")
    print(f"Total evidence: {summary.get('total_evidence_extracted', 0)}")
    print(f"Contradictions: {summary.get('total_contradictions', 0)}")
    print(f"Avg processing time: {summary.get('average_processing_time_seconds', 0):.1f}s")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='DECALOGO Batch Processing Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload documents from a JSON file
  python3 python_client.py --token abc123 --file documents.json

  # Upload documents from a directory
  python3 python_client.py --token abc123 --dir /path/to/docs

  # Check job status
  python3 python_client.py --token abc123 --status batch_20240115_abc123

  # Retrieve results
  python3 python_client.py --token abc123 --results batch_20240115_abc123

  # High priority batch
  python3 python_client.py --token abc123 --file docs.json --priority high
        """
    )
    
    parser.add_argument('--url', default='http://localhost:8000',
                       help='API base URL (default: http://localhost:8000)')
    parser.add_argument('--token', required=True,
                       help='Authentication token')
    parser.add_argument('--file', help='JSON file with documents')
    parser.add_argument('--dir', help='Directory with text files')
    parser.add_argument('--status', help='Check status of job ID')
    parser.add_argument('--results', help='Retrieve results for job ID')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--priority', choices=['high', 'normal', 'low'], default='normal',
                       help='Processing priority (default: normal)')
    parser.add_argument('--no-wait', action='store_true',
                       help='Do not wait for completion')
    parser.add_argument('--poll-interval', type=int, default=10,
                       help='Seconds between status checks (default: 10)')
    parser.add_argument('--health', action='store_true',
                       help='Check API health')
    
    args = parser.parse_args()
    
    # Initialize client
    client = DecalogoBatchClient(args.url, args.token)
    
    # Health check
    if args.health:
        health = client.health_check()
        print(json.dumps(health, indent=2))
        return
    
    # Check status
    if args.status:
        status = client.check_status(args.status)
        print(json.dumps(status, indent=2))
        return
    
    # Retrieve results
    if args.results:
        results = client.get_results(args.results, format=args.format)
        
        if args.output:
            save_results(results, args.output, args.format)
        else:
            if args.format == 'json':
                print(json.dumps(results, indent=2))
                print_summary(results)
            else:
                print(results.get('csv', ''))
        return
    
    # Upload documents
    if args.file:
        documents = load_documents_from_file(args.file)
    elif args.dir:
        documents = load_documents_from_directory(args.dir)
    else:
        parser.error('Must specify --file, --dir, --status, --results, or --health')
    
    if not documents:
        print("✗ No documents found")
        sys.exit(1)
    
    print(f"Loaded {len(documents)} documents")
    
    # Process batch
    options = {
        'priority': args.priority,
        'include_evidence': True,
        'include_traces': False
    }
    
    results = client.process_batch(
        documents,
        options=options,
        wait=not args.no_wait,
        poll_interval=args.poll_interval
    )
    
    # Save or print results
    if not args.no_wait:
        if args.output:
            save_results(results, args.output, args.format)
        else:
            print(json.dumps(results, indent=2))
            print_summary(results)
    else:
        print(f"\nJob ID: {results['job_id']}")
        print(f"Check status with: python3 python_client.py --token {args.token} --status {results['job_id']}")


if __name__ == '__main__':
    main()
