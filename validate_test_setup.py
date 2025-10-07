#!/usr/bin/env python3
"""Quick validation of test infrastructure setup"""

import sys
import json
from pathlib import Path

def validate_setup():
    """Validate all required components for unified flow certification"""
    repo_root = Path(__file__).parent
    
    print("="*80)
    print("VALIDATING TEST INFRASTRUCTURE SETUP")
    print("="*80)
    print()
    
    errors = []
    
    # Check required files
    required_files = {
        "Test file": "tests/test_unified_flow_certification.py",
        "Mock script": "test_mock_execution.py",
        "CLI": "miniminimoon_cli.py",
        "Validators": "system_validators.py",
        "Flow doc": "tools/flow_doc.json",
        "Rubric check": "tools/rubric_check.py",
        "Test plan": "data/florencia_plan_texto.txt",
        "Rubric": "RUBRIC_SCORING.json"
    }
    
    for name, path in required_files.items():
        full_path = repo_root / path
        if full_path.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (MISSING)")
            errors.append(f"Missing {name}: {path}")
    
    print()
    
    # Check artifacts from mock run
    artifacts_dir = repo_root / "artifacts"
    if artifacts_dir.exists():
        print(f"✅ Artifacts directory exists")
        
        expected_artifacts = [
            "answers_report.json",
            "flow_runtime.json",
            "coverage_report.json",
            "evidence_registry.json"
        ]
        
        for artifact in expected_artifacts:
            artifact_path = artifacts_dir / artifact
            if artifact_path.exists():
                size_kb = artifact_path.stat().st_size / 1024
                print(f"   ✅ {artifact} ({size_kb:.1f} KB)")
            else:
                print(f"   ❌ {artifact} (MISSING)")
                errors.append(f"Missing artifact: {artifact}")
    else:
        print(f"⚠️  Artifacts directory not found (will be created)")
    
    print()
    
    # Test rubric_check.py
    if (repo_root / "tools/rubric_check.py").exists() and (artifacts_dir / "answers_report.json").exists():
        print("Testing rubric_check.py...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "tools/rubric_check.py", "artifacts/answers_report.json", "RUBRIC_SCORING.json"],
            capture_output=True,
            text=True,
            cwd=str(repo_root)
        )
        
        if result.returncode == 0:
            print(f"✅ rubric_check.py executed successfully")
            try:
                output = json.loads(result.stdout)
                print(f"   Result: {output.get('message', 'OK')}")
            except:
                pass
        else:
            print(f"⚠️  rubric_check.py returned exit code {result.returncode}")
            errors.append(f"rubric_check.py validation issue")
    
    print()
    print("="*80)
    
    if errors:
        print(f"❌ VALIDATION FAILED: {len(errors)} errors")
        for error in errors:
            print(f"   • {error}")
        return False
    else:
        print("✅ ALL CHECKS PASSED")
        print()
        print("Infrastructure is ready for unified flow certification test execution.")
        return True

if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)
