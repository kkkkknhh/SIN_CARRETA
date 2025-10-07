#!/usr/bin/env python3
from miniminimoon_immutability import EnhancedImmutabilityContract
from pathlib import Path
import sys

try:
    contract = EnhancedImmutabilityContract(config_dir=Path("."))
    snapshot = contract.freeze_configuration()
    print("\n✅ Configuration frozen successfully!")
    print(f"   Snapshot hash: {snapshot['snapshot_hash'][:16]}...")
    print(f"   Files: {snapshot['file_count']}")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
