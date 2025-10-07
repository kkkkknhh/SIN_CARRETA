#!/usr/bin/env python3
"""Tool to freeze configuration for immutability contract"""
from miniminimoon_immutability import EnhancedImmutabilityContract
from pathlib import Path

contract = EnhancedImmutabilityContract(config_dir=Path("."))
snapshot = contract.freeze_configuration()
print(f"✅ Configuration frozen: {snapshot['snapshot_hash'][:16]}...")
