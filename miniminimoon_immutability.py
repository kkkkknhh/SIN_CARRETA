# coding=utf-8
"""
MINIMINIMOON Enhanced Immutability Contract (v2.0)
==================================================

Configuration freezing and verification system for deterministic pipeline execution.

Flow #16: miniminimoon_orchestrator → miniminimoon_immutability (control gate)
Flow #55: freeze_configuration() → .immutability_snapshot.json
Gate #1: verify_frozen_config() == True before any pipeline execution

Architecture:
- Focuses on configuration files (JSON), not Python code
- Uses SHA-256 for file integrity
- Simple, fast verification
- Single snapshot file: .immutability_snapshot.json

Author: System Architect
Version: 2.0.0 (Flow-Aligned)
Date: 2025-10-05
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MINIMINIMOONImmutability")


class EnhancedImmutabilityContract:
    """
    Configuration immutability contract for the MINIMINIMOON pipeline.

    Responsibilities:
    1. Create SHA-256 snapshots of critical config files
    2. Verify config hasn't changed since snapshot
    3. Enforce gate #1: no execution without valid frozen config

    Critical files monitored:
    - DECALOGO_FULL.json
    - decalogo_industrial.json
    - dnp-standards.latest.clean.json
    - RUBRIC_SCORING.json
    """

    # Snapshot file location
    SNAPSHOT_FILE = ".immutability_snapshot.json"

    # Critical configuration files (must exist and be frozen)
    CRITICAL_CONFIGS = [
        "DECALOGO_FULL.json",
        "decalogo_industrial.json",
        "dnp-standards.latest.clean.json",
        "RUBRIC_SCORING.json"
    ]

    # Optional configuration files (snapshot if present)
    OPTIONAL_CONFIGS = [
        "decalogo_contexto_avanzado.json",
        "ontologia_politicas.json",
        "teoria_cambio_config.json"
    ]

    VERSION = "2.0.0"

    def __init__(self, config_dir: Optional[Path] = None, snapshot_path: Optional[Path] = None):
        """
        Initialize immutability contract.

        Args:
            config_dir: Directory containing config files (default: ./config/)
            snapshot_path: Path to snapshot file (default: ./.immutability_snapshot.json)
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./config/")
        self.snapshot_path = Path(snapshot_path) if snapshot_path else Path(self.SNAPSHOT_FILE)
        self.logger = logging.getLogger(__name__)

        self.logger.debug(f"Immutability contract initialized (config_dir={self.config_dir})")

    def _compute_file_hash(self, filepath: Path) -> str:
        """
        Compute SHA-256 hash of a file.

        Args:
            filepath: Path to file

        Returns:
            Hex-encoded SHA-256 hash
        """
        sha256 = hashlib.sha256()

        try:
            with open(filepath, 'rb') as f:
                # Read in chunks for memory efficiency
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)

            return sha256.hexdigest()

        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error hashing file {filepath}: {e}")

    def _get_config_files(self) -> Dict[str, Path]:
        """
        Discover all config files to snapshot.

        Returns:
            Dictionary mapping filename → absolute path
        """
        files = {}

        # Critical configs (must exist)
        for filename in self.CRITICAL_CONFIGS:
            filepath = self.config_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Critical config missing: {filepath}. "
                    f"Cannot freeze without all critical configs."
                )
            files[filename] = filepath

        # Optional configs (snapshot if present)
        for filename in self.OPTIONAL_CONFIGS:
            filepath = self.config_dir / filename
            if filepath.exists():
                files[filename] = filepath
                self.logger.debug(f"Found optional config: {filename}")

        return files

    def freeze_configuration(self) -> Dict[str, Any]:
        """
        Create immutability snapshot of all configuration files.

        Flow #55: freeze_configuration() → .immutability_snapshot.json

        Process:
        1. Discover all config files
        2. Compute SHA-256 hash of each
        3. Save snapshot with metadata

        Returns:
            Snapshot metadata including hash and timestamp
        """
        self.logger.info("🔒 Freezing configuration (creating immutability snapshot)...")

        # Get all config files
        config_files = self._get_config_files()

        # Compute hashes
        file_hashes = {}
        for filename, filepath in config_files.items():
            try:
                file_hash = self._compute_file_hash(filepath)
                file_hashes[filename] = {
                    "sha256": file_hash,
                    "path": str(filepath.absolute()),
                    "size_bytes": filepath.stat().st_size
                }
                self.logger.debug(f"  {filename}: {file_hash[:12]}...")
            except Exception as e:
                raise RuntimeError(f"Failed to hash {filename}: {e}")

        # Compute global snapshot hash (hash of all hashes)
        combined_hashes = "|".join(
            f"{filename}:{data['sha256']}"
            for filename, data in sorted(file_hashes.items())
        )
        snapshot_hash = hashlib.sha256(combined_hashes.encode()).hexdigest()

        # Create snapshot record
        snapshot = {
            "version": self.VERSION,
            "snapshot_timestamp": datetime.utcnow().isoformat() + "Z",
            "snapshot_hash": snapshot_hash,
            "config_dir": str(self.config_dir.absolute()),
            "files": file_hashes,
            "critical_configs": self.CRITICAL_CONFIGS,
            "file_count": len(file_hashes)
        }

        # Save snapshot to file
        try:
            with open(self.snapshot_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"✓ Configuration frozen: {snapshot['snapshot_hash'][:16]}... "
                f"({len(file_hashes)} files)"
            )
            self.logger.info(f"  Snapshot saved to: {self.snapshot_path.absolute()}")

            return snapshot

        except Exception as e:
            raise RuntimeError(f"Failed to save snapshot: {e}")

    def has_snapshot(self) -> bool:
        """
        Check if a frozen configuration snapshot exists.

        Returns:
            True if snapshot file exists, False otherwise
        """
        exists = self.snapshot_path.exists()

        if exists:
            self.logger.debug(f"Snapshot found: {self.snapshot_path}")
        else:
            self.logger.debug(f"No snapshot found at: {self.snapshot_path}")

        return exists

    def load_snapshot(self) -> Dict[str, Any]:
        """
        Load frozen configuration snapshot.

        Returns:
            Snapshot data

        Raises:
            FileNotFoundError: If snapshot doesn't exist
            ValueError: If snapshot is invalid
        """
        if not self.has_snapshot():
            raise FileNotFoundError(
                f"No frozen config snapshot found at {self.snapshot_path}. "
                f"Run freeze_configuration() first."
            )

        try:
            with open(self.snapshot_path, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)

            # Validate snapshot structure
            required_keys = ["version", "snapshot_hash", "files"]
            missing_keys = [k for k in required_keys if k not in snapshot]

            if missing_keys:
                raise ValueError(
                    f"Invalid snapshot (missing keys: {missing_keys}). "
                    f"Re-run freeze_configuration()."
                )

            return snapshot

        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt snapshot file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading snapshot: {e}")

    def verify_frozen_config(self) -> bool:
        """
        Verify that current config matches frozen snapshot.

        Gate #1: This must return True before pipeline execution.
        Flow #16: Called by miniminimoon_orchestrator.__init__()

        Process:
        1. Load snapshot
        2. Compute current hashes of all files
        3. Compare with snapshot hashes

        Returns:
            True if config matches snapshot, False if any mismatch
        """
        self.logger.info("🔍 Verifying frozen configuration...")

        # Load snapshot
        try:
            snapshot = self.load_snapshot()
        except FileNotFoundError:
            self.logger.error("⨯ No snapshot found (gate #1 prerequisite)")
            return False
        except Exception as e:
            self.logger.error(f"⨯ Failed to load snapshot: {e}")
            return False

        # Get current file hashes
        mismatches = []
        missing_files = []

        for filename, snapshot_data in snapshot["files"].items():
            expected_hash = snapshot_data["sha256"]
            filepath = self.config_dir / filename

            # Check file exists
            if not filepath.exists():
                missing_files.append(filename)
                self.logger.error(f"⨯ Config file missing: {filename}")
                continue

            # Compute current hash
            try:
                current_hash = self._compute_file_hash(filepath)

                if current_hash != expected_hash:
                    mismatches.append({
                        "file": filename,
                        "expected": expected_hash[:12] + "...",
                        "current": current_hash[:12] + "..."
                    })
                    self.logger.error(
                        f"⨯ Config mismatch: {filename} "
                        f"(expected {expected_hash[:12]}..., got {current_hash[:12]}...)"
                    )
                else:
                    self.logger.debug(f"  ✓ {filename}: hash matches")

            except Exception as e:
                self.logger.error(f"⨯ Error verifying {filename}: {e}")
                mismatches.append({
                    "file": filename,
                    "error": str(e)
                })

        # Report results
        if mismatches or missing_files:
            self.logger.error(
                f"⨯ GATE #1 FAILED: Config verification failed "
                f"({len(mismatches)} mismatches, {len(missing_files)} missing)"
            )

            if mismatches:
                self.logger.error("  Mismatched files:")
                for m in mismatches:
                    self.logger.error(f"    - {m['file']}")

            if missing_files:
                self.logger.error("  Missing files:")
                for f in missing_files:
                    self.logger.error(f"    - {f}")

            return False

        else:
            self.logger.info(
                f"✓ GATE #1 PASSED: Frozen config verified "
                f"({len(snapshot['files'])} files match snapshot)"
            )
            return True

    def get_snapshot_info(self) -> Dict[str, Any]:
        """
        Get information about the current snapshot without verification.

        Returns:
            Snapshot metadata
        """
        if not self.has_snapshot():
            return {
                "exists": False,
                "message": "No snapshot found"
            }

        try:
            snapshot = self.load_snapshot()
            return {
                "exists": True,
                "version": snapshot.get("version"),
                "snapshot_hash": snapshot.get("snapshot_hash"),
                "timestamp": snapshot.get("snapshot_timestamp"),
                "file_count": snapshot.get("file_count"),
                "files": list(snapshot.get("files", {}).keys())
            }
        except Exception as e:
            return {
                "exists": True,
                "error": str(e),
                "message": "Snapshot exists but is invalid"
            }

    def diff_configs(self) -> Dict[str, Any]:
        """
        Show differences between current config and snapshot.

        Useful for debugging why verification fails.

        Returns:
            Detailed diff report
        """
        if not self.has_snapshot():
            return {
                "status": "no_snapshot",
                "message": "No snapshot exists"
            }

        try:
            snapshot = self.load_snapshot()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load snapshot: {e}"
            }

        diff_report = {
            "status": "diff_computed",
            "snapshot_hash": snapshot["snapshot_hash"],
            "snapshot_timestamp": snapshot.get("snapshot_timestamp"),
            "files": {}
        }

        for filename, snapshot_data in snapshot["files"].items():
            expected_hash = snapshot_data["sha256"]
            filepath = self.config_dir / filename

            if not filepath.exists():
                diff_report["files"][filename] = {
                    "status": "missing",
                    "expected_hash": expected_hash[:16] + "..."
                }
            else:
                try:
                    current_hash = self._compute_file_hash(filepath)

                    if current_hash == expected_hash:
                        diff_report["files"][filename] = {
                            "status": "match",
                            "hash": current_hash[:16] + "..."
                        }
                    else:
                        diff_report["files"][filename] = {
                            "status": "mismatch",
                            "expected_hash": expected_hash[:16] + "...",
                            "current_hash": current_hash[:16] + "...",
                            "expected_size": snapshot_data.get("size_bytes"),
                            "current_size": filepath.stat().st_size
                        }
                except Exception as e:
                    diff_report["files"][filename] = {
                        "status": "error",
                        "error": str(e)
                    }

        # Summary
        statuses = [f["status"] for f in diff_report["files"].values()]
        diff_report["summary"] = {
            "total_files": len(diff_report["files"]),
            "matching": statuses.count("match"),
            "mismatched": statuses.count("mismatch"),
            "missing": statuses.count("missing"),
            "errors": statuses.count("error")
        }

        return diff_report

    def verify_decatalogo_integration(self) -> Dict[str, Any]:
        """
        Verify DECATALOGO_PRINCIPAL.PY integration is correct.

        Simplified version focusing on critical checks only.

        Returns:
            Verification report
        """
        self.logger.info("🔍 Verifying Decatalogo_principal integration...")

        report = {
            "module": "Decatalogo_principal",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": {},
            "status": "unknown"
        }

        try:
            # Check 1: Module importable
            try:
                import Decatalogo_principal
                report["checks"]["importable"] = "PASS"
            except ImportError as e:
                report["checks"]["importable"] = "FAIL"
                report["status"] = "FAIL"
                report["error"] = f"Cannot import: {e}"
                return report

            # Check 2: Critical function present
            if hasattr(Decatalogo_principal, "obtener_decalogo_contexto_avanzado"):
                report["checks"]["context_function"] = "PASS"
            else:
                report["checks"]["context_function"] = "FAIL"
                report["status"] = "FAIL"
                report["missing_function"] = "obtener_decalogo_contexto_avanzado"

            # Check 3: Critical class present
            if hasattr(Decatalogo_principal, "ExtractorEvidenciaIndustrialAvanzado"):
                report["checks"]["extractor_class"] = "PASS"
            else:
                report["checks"]["extractor_class"] = "FAIL"
                report["status"] = "FAIL"
                report["missing_class"] = "ExtractorEvidenciaIndustrialAvanzado"

            # Final status
            if all(v == "PASS" for v in report["checks"].values()):
                report["status"] = "PASS"
                self.logger.info("✓ Decatalogo_principal integration verified")
            else:
                if report["status"] == "unknown":
                    report["status"] = "PARTIAL"
                self.logger.warning(f"⚠ Decatalogo_principal verification: {report['status']}")

        except Exception as e:
            report["status"] = "ERROR"
            report["error"] = str(e)
            self.logger.error(f"✗ Error verifying Decatalogo_principal: {e}")

        return report


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI entry point for immutability contract operations"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="MINIMINIMOON Immutability Contract Tool (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Freeze configuration (creates snapshot)
  python miniminimoon_immutability.py freeze ./config/

  # Verify current config matches snapshot
  python miniminimoon_immutability.py verify ./config/

  # Show snapshot info
  python miniminimoon_immutability.py info ./config/

  # Show config differences
  python miniminimoon_immutability.py diff ./config/

  # Verify Decatalogo integration
  python miniminimoon_immutability.py check-decatalogo
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Freeze command
    freeze_parser = subparsers.add_parser("freeze", help="Freeze configuration")
    freeze_parser.add_argument("config_dir", type=Path, help="Config directory")
    freeze_parser.add_argument("--snapshot", type=Path, help="Snapshot file path")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify frozen config")
    verify_parser.add_argument("config_dir", type=Path, help="Config directory")
    verify_parser.add_argument("--snapshot", type=Path, help="Snapshot file path")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show snapshot info")
    info_parser.add_argument("config_dir", type=Path, help="Config directory")
    info_parser.add_argument("--snapshot", type=Path, help="Snapshot file path")

    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Show config differences")
    diff_parser.add_argument("config_dir", type=Path, help="Config directory")
    diff_parser.add_argument("--snapshot", type=Path, help="Snapshot file path")

    # Check Decatalogo command
    check_parser = subparsers.add_parser("check-decatalogo", help="Verify Decatalogo integration")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == "freeze":
            contract = EnhancedImmutabilityContract(
                config_dir=args.config_dir,
                snapshot_path=args.snapshot
            )
            snapshot = contract.freeze_configuration()

            print("\n✓ Configuration frozen successfully")
            print(f"  Snapshot hash: {snapshot['snapshot_hash']}")
            print(f"  Files: {snapshot['file_count']}")
            print(f"  Timestamp: {snapshot['snapshot_timestamp']}")
            sys.exit(0)

        elif args.command == "verify":
            contract = EnhancedImmutabilityContract(
                config_dir=args.config_dir,
                snapshot_path=args.snapshot
            )

            is_valid = contract.verify_frozen_config()

            if is_valid:
                print("\n✓ Verification PASSED (gate #1)")
                sys.exit(0)
            else:
                print("\n✗ Verification FAILED (gate #1)")
                print("\nRun 'diff' command to see what changed.")
                sys.exit(3)

        elif args.command == "info":
            contract = EnhancedImmutabilityContract(
                config_dir=args.config_dir,
                snapshot_path=args.snapshot
            )

            info = contract.get_snapshot_info()

            if info["exists"]:
                print("\nSnapshot Information:")
                print(f"  Version: {info.get('version', 'unknown')}")
                print(f"  Snapshot hash: {info.get('snapshot_hash', 'unknown')}")
                print(f"  Timestamp: {info.get('timestamp', 'unknown')}")
                print(f"  File count: {info.get('file_count', 0)}")
                print(f"\n  Files:")
                for filename in info.get('files', []):
                    print(f"    - {filename}")
                sys.exit(0)
            else:
                print("\n✗ No snapshot found")
                sys.exit(3)

        elif args.command == "diff":
            contract = EnhancedImmutabilityContract(
                config_dir=args.config_dir,
                snapshot_path=args.snapshot
            )

            diff = contract.diff_configs()

            if diff["status"] == "no_snapshot":
                print("\n✗ No snapshot exists. Run 'freeze' first.")
                sys.exit(3)

            print("\nConfiguration Diff:")
            print(f"  Snapshot: {diff['snapshot_hash'][:16]}...")
            print(f"  Timestamp: {diff.get('snapshot_timestamp', 'unknown')}")
            print(f"\n  Summary:")
            print(f"    Total files: {diff['summary']['total_files']}")
            print(f"    Matching: {diff['summary']['matching']}")
            print(f"    Mismatched: {diff['summary']['mismatched']}")
            print(f"    Missing: {diff['summary']['missing']}")

            if diff['summary']['mismatched'] > 0 or diff['summary']['missing'] > 0:
                print(f"\n  Differences:")
                for filename, file_info in diff["files"].items():
                    if file_info["status"] != "match":
                        print(f"    {filename}: {file_info['status']}")
                        if file_info["status"] == "mismatch":
                            print(f"      Expected: {file_info['expected_hash']}")
                            print(f"      Current:  {file_info['current_hash']}")
                sys.exit(3)
            else:
                print("\n✓ All files match snapshot")
                sys.exit(0)

        elif args.command == "check-decatalogo":
            contract = EnhancedImmutabilityContract()
            report = contract.verify_decatalogo_integration()

            print("\nDecatalogo_principal Verification:")
            print(f"  Status: {report['status']}")
            print(f"\n  Checks:")
            for check, result in report["checks"].items():
                symbol = "✓" if result == "PASS" else "✗"
                print(f"    {symbol} {check}: {result}")

            if report["status"] == "PASS":
                sys.exit(0)
            else:
                if "error" in report:
                    print(f"\n  Error: {report['error']}")
                sys.exit(3)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()