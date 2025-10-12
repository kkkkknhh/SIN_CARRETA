"""
Example client demonstrating usage of DECALOGO PDM Evaluation API.

Shows how to:
- Upload documents
- Check job status
- Retrieve results
- Monitor system health
"""

import json
import time
from pathlib import Path

import requests


class DecalogoAPIClient:
    """Client for interacting with DECALOGO PDM Evaluation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def upload_documents(
        self,
        files: list,
        municipality: str,
        department: str,
        year: int,
        period: str,
        requester_name: str = None,
        requester_email: str = None,
        notes: str = None,
        enable_causal_analysis: bool = True,
        enable_contradiction_detection: bool = True,
        enable_monetary_analysis: bool = True,
        enable_teoria_cambio: bool = True,
        strictness_level: str = "standard",
    ) -> dict:
        """
        Upload PDM documents for evaluation.
        
        Args:
            files: List of file paths or file-like objects
            municipality: Municipality name
            department: Department name
            year: PDM year
            period: PDM period (e.g., '2024-2027')
            requester_name: Optional requester name
            requester_email: Optional requester email
            notes: Optional notes
            enable_causal_analysis: Enable causal pattern detection
            enable_contradiction_detection: Enable contradiction detection
            enable_monetary_analysis: Enable monetary value detection
            enable_teoria_cambio: Enable Theory of Change analysis
            strictness_level: Validation strictness (lenient, standard, strict)
        
        Returns:
            Response dict with job_id, status, document_count
        """
        url = f"{self.base_url}/upload"
        
        file_objects = []
        for f in files:
            if isinstance(f, (str, Path)):
                file_objects.append(("files", open(f, "rb")))
            else:
                file_objects.append(("files", f))
        
        data = {
            "municipality": municipality,
            "department": department,
            "year": year,
            "period": period,
            "enable_causal_analysis": enable_causal_analysis,
            "enable_contradiction_detection": enable_contradiction_detection,
            "enable_monetary_analysis": enable_monetary_analysis,
            "enable_teoria_cambio": enable_teoria_cambio,
            "strictness_level": strictness_level,
        }
        
        if requester_name:
            data["requester_name"] = requester_name
        if requester_email:
            data["requester_email"] = requester_email
        if notes:
            data["notes"] = notes
        
        try:
            response = self.session.post(url, files=file_objects, data=data)
            response.raise_for_status()
            return response.json()
        finally:
            for _, f in file_objects:
                if hasattr(f, "close"):
                    f.close()
    
    def get_status(self, job_id: str) -> dict:
        """
        Get current status of a job.
        
        Args:
            job_id: Unique job identifier
        
        Returns:
            Job status dict with state, progress, and metadata
        """
        url = f"{self.base_url}/status/{job_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_results(self, job_id: str, format: str = "json", output_path: str = None) -> bytes:
        """
        Retrieve evaluation results.
        
        Args:
            job_id: Unique job identifier
            format: Result format ('json', 'pdf', or 'zip')
            output_path: Optional path to save results
        
        Returns:
            Result content as bytes
        """
        url = f"{self.base_url}/results/{job_id}"
        params = {"format": format}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        if output_path:
            Path(output_path).write_bytes(response.content)
        
        return response.content
    
    def check_health(self) -> dict:
        """
        Check system health.
        
        Returns:
            Health status dict with component checks
        """
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, job_id: str, timeout: int = 3600, poll_interval: int = 5) -> dict:
        """
        Wait for job to complete, polling status periodically.
        
        Args:
            job_id: Unique job identifier
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between status checks
        
        Returns:
            Final job status
        
        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails
        """
        start_time = time.time()
        
        while True:
            status = self.get_status(job_id)
            
            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                error = status.get("error_message", "Unknown error")
                raise RuntimeError(f"Job failed: {error}")
            
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job did not complete within {timeout} seconds")
            
            time.sleep(poll_interval)


def example_usage():
    """Demonstrate API usage."""
    client = DecalogoAPIClient("http://localhost:8000")
    
    print("Checking system health...")
    health = client.check_health()
    print(f"Status: {health['status']}")
    print(f"Workers available: {health['workers_available']}")
    print(f"Queue size: {health['queue_size']}")
    print()
    
    print("Uploading PDM document...")
    upload_response = client.upload_documents(
        files=["FLORENCIA - PLAN DE DESARROLLO.pdf"],
        municipality="Florencia",
        department="Caquetá",
        year=2024,
        period="2024-2027",
        requester_name="Demo User",
        requester_email="demo@example.com",
        notes="Example evaluation request",
    )
    
    job_id = upload_response["job_id"]
    print(f"Job created: {job_id}")
    print(f"Document count: {upload_response['document_count']}")
    print()
    
    print("Monitoring job status...")
    try:
        final_status = client.wait_for_completion(job_id, timeout=3600)
        print("Job completed!")
        print(f"Progress: {final_status['progress']['progress_percentage']}%")
        print()
        
        print("Retrieving results...")
        results_json = client.get_results(job_id, format="json", output_path=f"results_{job_id}.json")
        print(f"Results saved to results_{job_id}.json")
        
        results = json.loads(results_json)
        print(f"Evaluation score: {results.get('overall_score', 'N/A')}")
        
    except TimeoutError as e:
        print(f"Timeout: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_usage()
