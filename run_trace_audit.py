#!/usr/bin/env python3.10
"""
Simplified trace audit runner
"""

import subprocess
import sys
from pathlib import Path

print("Running orchestrator trace audit...")
print()

# Run the test
result = subprocess.run(
    [sys.executable, "test_orchestrator_trace.py"], capture_output=False, timeout=120
)

sys.exit(result.returncode)
