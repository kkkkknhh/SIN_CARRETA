# coding=utf-8
# coding=utf-8
import re
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from enum import Enum

# Module-level logger (instrumentable, audit-aligned)
import logging
logger = logging.getLogger("feasibility_scorer")
logger.setLevel(logging.INFO)

class ComponentType(Enum):
    BASELINE = "baseline"
    TARGET = "target"
    TIMEFRAME = "timeframe"
    UNIT = "unit"
    RESPONSIBLE = "responsible"
    DATE = "date"

class ComponentDict(dict):
    """
    Dictionary subclass that behaves like a dict but iterates over detection results.
    This allows backward compatibility with code that expects a dict, while also
    supporting iteration over detection results for tests.
    """
    def __iter__(self):
        """Iterate over all detection results (flattened)."""
        for detection_list in self.values():
            for detection in detection_list:
                yield detection

    def __getitem__(self, key):
        """Support dict-style access."""
        return super().__getitem__(key)

@dataclass
class DetectionResult:
    """
    Detection result for a single component.
    Attributes:
        text: Matched text
        component_type: ComponentType
        numeric_value: Extracted numerical value if applicable
        unit: Unit of measurement if applicable
        metadata: Additional metadata
        matched_text: Alias for text attribute (for backward compatibility)
    """
    text: str
    component_type: ComponentType
    numeric_value: Optional[float] = None
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def matched_text(self) -> str:
        """Alias for text attribute."""
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "component_type": self.component_type.value,
            "numeric_value": self.numeric_value,
            "unit": self.unit,
            "metadata": self.metadata,
            "matched_text": self.matched_text,
        }

@dataclass
class IndicatorScore:
    """
    Evaluation score for an indicator.
    Attributes:
        text: Source text
        has_baseline: Baseline detected
        has_target: Target detected
        has_timeframe: Time horizon detected
        has_quantitative_target: Quantitative target detected
        has_unit: Unit detected
        has_responsible: Responsible entity detected
        smart_score: Score for SMART criteria (0-1)
        feasibility_score: Overall feasibility score (0-1)
        components: Detected components related to this indicator
        quality_tier: Quality tier classification (set by calculate_feasibility_score)
        components_detected: List of component types detected (for backward compatibility)
        detailed_matches: List of detection results (for backward compatibility)
        has_quantitative_baseline: Whether the baseline is quantitative
    """
    text: str
    has_baseline: bool = False
    has_target: bool = False
    has_timeframe: bool = False
    has_quantitative_target: bool = False
    has_unit: bool = False
    has_responsible: bool = False
    smart_score: float = 0.0
    feasibility_score: float = 0.0
    components: List[DetectionResult] = field(default_factory=list)
    quality_tier: str = ""
    components_detected: List[ComponentType] = field(default_factory=list)
    detailed_matches: List[DetectionResult] = field(default_factory=list)
    has_quantitative_baseline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "has_baseline": self.has_baseline,
            "has_target": self.has_target,
            "has_timeframe": self.has_timeframe,
            "has_quantitative_target": self.has_quantitative_target,
            "has_unit": self.has_unit,
            "has_responsible": self.has_responsible,
            "smart_score": self.smart_score,
            "feasibility_score": self.feasibility_score,
            "components": [c.to_dict() for c in self.components],
            "quality_tier": self.quality_tier,
            "components_detected": [ct.value for ct in self.components_detected],
            "detailed_matches": [c.to_dict() for c in self.detailed_matches],
            "has_quantitative_baseline": self.has_quantitative_baseline,
        }


@dataclass(frozen=True)
class FeasibilityConfig:
    """Immutable configuration contract for :class:`FeasibilityScorer`."""

    enable_parallel: bool = True
    n_jobs: int = 8
    backend: str = "loky"
    seed: int = 42
    evidence_registry: Optional[Any] = None
    backend_options: Dict[str, Any] = field(default_factory=dict)

class FeasibilityScorer:
    """Industrial-grade feasibility scorer with auditable configuration."""

    ALLOWED_BACKENDS = ("loky", "threading", "ray", "dask")
    MIN_JOBS = 1
    MAX_JOBS = 64

    def __init__(
        self,
        config: Optional[FeasibilityConfig] = None,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Instantiate the scorer with an explicit configuration contract."""

        self.logger = logger or logging.getLogger("feasibility_scorer")
        self.logger.setLevel(logging.INFO)
        self._config_history: List[Dict[str, Any]] = []
        self.backend_options: Dict[str, Any] = {}
        self._apply_config(config or FeasibilityConfig(), event="initialized")
        self._initialize_patterns()

    @property
    def current_config(self) -> FeasibilityConfig:
        """Return the active configuration snapshot."""

        return self._config

    @property
    def configuration_history(self) -> List[Dict[str, Any]]:
        """Expose immutable configuration change history for auditing."""

        return list(self._config_history)

    def describe_configuration(self) -> Dict[str, Any]:
        """Return the current configuration as a serializable dictionary."""

        return asdict(self._config)

    def log_configuration(self, event: str = "snapshot", extra: Optional[Dict[str, Any]] = None) -> None:
        """Emit an explicit log entry with the current configuration."""

        self._trace_config(event, extra)

    def configure(self, **kwargs: Any) -> None:
        """Dynamically reconfigure the scorer using explicit keyword overrides."""

        allowed_keys = set(FeasibilityConfig.__dataclass_fields__)
        unknown = set(kwargs) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")

        new_config = replace(self._config, **kwargs)
        self._apply_config(new_config, event="reconfigured", extra={"changes": kwargs})

    def attach_evidence_registry(self, evidence_registry: Any) -> None:
        """Attach or swap the evidence registry with runtime type validation."""

        if not hasattr(evidence_registry, "register"):
            raise TypeError("Evidence registry must expose a 'register' method")
        self.configure(evidence_registry=evidence_registry)

    def set_seed(self, seed: int) -> None:
        """Update the deterministic seed and reseed all managed generators."""

        self.configure(seed=seed)

    @contextmanager
    def seed_context(self, seed: Optional[int] = None):
        """Context manager that temporarily overrides RNG state with audit logs."""

        active_seed = self.seed if seed is None else seed
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        self._trace_config("seed_context_enter", {"seed": active_seed})
        self._seed_random_generators(active_seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            self._trace_config("seed_context_exit", {"seed": active_seed})

    @staticmethod
    def _seed_random_generators(seed: int) -> None:
        """Seed Python and NumPy random generators in a single place."""

        np.random.seed(seed)
        random.seed(seed)

    def _apply_config(
        self,
        config: FeasibilityConfig,
        *,
        event: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized_config, adjustments = self._normalize_config(config)
        self._config = normalized_config
        self.enable_parallel = normalized_config.enable_parallel
        self.n_jobs = normalized_config.n_jobs
        self.backend = normalized_config.backend
        self.seed = normalized_config.seed
        self.evidence_registry = normalized_config.evidence_registry
        self.backend_options = dict(normalized_config.backend_options)
        self._seed_random_generators(self.seed)

        trace_extra: Dict[str, Any] = {}
        if extra:
            trace_extra.update(extra)
        if adjustments:
            trace_extra["adjustments"] = adjustments
        self._trace_config(event, trace_extra or None)

    def _normalize_config(
        self, config: FeasibilityConfig
    ) -> tuple[FeasibilityConfig, Dict[str, Any]]:
        adjustments: Dict[str, Any] = {}
        n_jobs = max(self.MIN_JOBS, min(config.n_jobs, self.MAX_JOBS))
        if n_jobs != config.n_jobs:
            adjustments["n_jobs"] = {"requested": config.n_jobs, "applied": n_jobs}

        backend = config.backend
        if backend not in self.ALLOWED_BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Allowed: {self.ALLOWED_BACKENDS}")

        if config.evidence_registry is not None and backend in {"loky", "ray", "dask"}:
            backend = "threading"
            adjustments["backend"] = {
                "reason": "evidence registry not shareable across process/distributed boundaries",
                "applied": backend,
            }

        normalized = replace(config, n_jobs=n_jobs, backend=backend)
        return normalized, adjustments

    def _select_backend(self, backend: str):
        if backend == "threading":
            return self._execute_thread_pool
        if backend == "loky":
            return self._execute_process_pool
        if backend == "ray":
            return self._execute_ray_backend
        if backend == "dask":
            return self._execute_dask_backend
        return None

    def _execute_thread_pool(
        self, indicators: List[str], evidencia_soporte_list: List[Optional[int]]
    ) -> List[IndicatorScore]:
        from concurrent.futures import ThreadPoolExecutor

        self.logger.info(
            "Executing feasibility scoring with threading backend (n_jobs=%s)",
            self.n_jobs,
        )

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(
                    self.calculate_feasibility_score,
                    indicator,
                    evidencia_soporte=evidencia,
                )
                for indicator, evidencia in zip(indicators, evidencia_soporte_list)
            ]
            return [future.result() for future in futures]

    def _execute_process_pool(
        self, indicators: List[str], evidencia_soporte_list: List[Optional[int]]
    ) -> List[IndicatorScore]:
        try:
            from concurrent.futures import ProcessPoolExecutor
        except ImportError:  # pragma: no cover
            self.logger.warning("ProcessPoolExecutor unavailable; falling back to threading")
            return self._execute_thread_pool(indicators, evidencia_soporte_list)

        self.logger.info(
            "Executing feasibility scoring with process backend 'loky' (n_jobs=%s)",
            self.n_jobs,
        )

        config_dict = asdict(self._config)
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(
                    _score_indicator_with_config,
                    config_dict,
                    indicator,
                    evidencia,
                )
                for indicator, evidencia in zip(indicators, evidencia_soporte_list)
            ]
            return [future.result() for future in futures]

    def _execute_ray_backend(
        self, indicators: List[str], evidencia_soporte_list: List[Optional[int]]
    ) -> List[IndicatorScore]:
        try:
            import ray
        except ImportError:
            self.logger.warning("Ray backend not available; falling back to threading")
            return self._execute_thread_pool(indicators, evidencia_soporte_list)

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, **self.backend_options.get("ray_init", {}))

        self.logger.info(
            "Executing feasibility scoring with Ray backend (n_jobs=%s)",
            self.n_jobs,
        )

        config_dict = asdict(self._config)

        @ray.remote
        def _ray_worker(indicator_text: str, evidencia: Optional[int]):
            return _score_indicator_with_config(config_dict, indicator_text, evidencia)

        tasks = [
            _ray_worker.remote(indicator, evidencia)
            for indicator, evidencia in zip(indicators, evidencia_soporte_list)
        ]
        results = ray.get(tasks)
        if self.backend_options.get("ray_shutdown_on_exit", True):
            ray.shutdown()
        return results

    def _execute_dask_backend(
        self, indicators: List[str], evidencia_soporte_list: List[Optional[int]]
    ) -> List[IndicatorScore]:
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError:
            self.logger.warning("Dask backend not available; falling back to threading")
            return self._execute_thread_pool(indicators, evidencia_soporte_list)

        cluster_kwargs = self.backend_options.get("dask_cluster", {})
        self.logger.info(
            "Executing feasibility scoring with Dask backend (n_jobs=%s)",
            self.n_jobs,
        )

        cluster = LocalCluster(n_workers=self.n_jobs, **cluster_kwargs)
        client = Client(cluster)
        config_dict = asdict(self._config)
        futures = [
            client.submit(
                _score_indicator_with_config,
                config_dict,
                indicator,
                evidencia,
            )
            for indicator, evidencia in zip(indicators, evidencia_soporte_list)
        ]
        results = client.gather(futures)
        client.close()
        cluster.close()
        return results

    def _trace_config(self, event: str, extra: Optional[Dict[str, Any]] = None) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            "config": asdict(self._config),
        }
        if extra:
            record["extra"] = extra
        self._config_history.append(record)
        self.logger.info("[FeasibilityScorer] %s | %s", event, record)

    def _initialize_patterns(self) -> None:
        # Baseline patterns (both quantitative and qualitative)
        self.baseline_patterns = [
            # Quantitative patterns (with numbers)
            r'(?i)l[ií]nea\s+base\s+de\s+(\d+(?:[.,]\d+)?(?:\s*%)?)',  # línea base de X%
            r'(?i)l[ií]nea\s+base\s+muestra\s+(\d+(?:[.,]\d+)?(?:\s*%)?)',  # línea base muestra X%
            r'(?i)l[ií]nea\s+(?:de\s+)?base\s*[:=]?\s*(\d+(?:[.,]\d+)?(?:\s*%)?)',
            r'(?i)(?:valor|medici[óo]n)\s+(?:inicial|actual|presente)\s+(\d+(?:[.,]\d+)?(?:\s*%)?)',
            r'(?i)(?:valor|medici[óo]n)\s+(?:inicial|actual|presente)\s*[:=]?\s*(\d+(?:[.,]\d+)?(?:\s*%)?)',
            r'(?i)(?:actualmente|al presente|al inicio|hoy)[^.]*?(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:situaci[óo]n actual|escenario base)[^.]*?(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:indicador|valor)[^.]*?(\d{4})[^.]*?(?:fue|era|correspondi[óo] a)\s*(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)baseline\s+(?:of\s+)?(\d+(?:[.,]\d+)?(?:\s*%)?)',  # English pattern
            r'(?i)current\s+level\s+(\d+(?:[.,]\d+)?(?:\s*%)?)',  # current level X
            r'(?i)starting\s+(?:point|level|value)\s+(\d+(?:[.,]\d+)?(?:\s*%)?)',  # starting point X
            # Qualitative patterns (no numbers)
            r'(?i)(?:partir\s+de\s+|desde\s+)(?:la\s+)?l[ií]nea\s+base\b',  # partir de la línea base
            r'(?i)(?:partir\s+de\s+|desde\s+)(?:la\s+)?situaci[óo]n\s+actual\b',  # desde situación actual
            r'(?i)starting\s+point\b',  # starting point (qualitative)
        ]

        # Target patterns (both quantitative and qualitative)
        self.target_patterns = [
            # Quantitative patterns (with numbers)
            r'(?i)meta\s+de\s+(\d+(?:[.,]\d+)?(?:\s*%)?)',  # meta de X%
            r'(?i)meta\s*[:=]?\s*(\d+(?:[.,]\d+)?(?:\s*%)?)',
            r'(?i)objetivo\s+(\d+(?:[.,]\d+)?(?:\s*%)?)',  # objetivo X%
            r'(?i)(?:alcanzar|lograr|conseguir|obtener|llegar a)\s*(?:un|una|el|la)?[^.]*?(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:valor|nivel)\s+(?:esperado|objetivo|deseado)\s*[:=]?\s*(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:aumentar|incrementar|crecer|elevar)[^.]*?(?:hasta|a)\s*(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:reducir|disminuir|bajar|decrecer)[^.]*?(?:hasta|a)\s*(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:para|to)\s+(?:el\s+)?objetivo\s+(?:de\s+)?(\d+(?:[.,]\d+)?(?:\s*%)?)',  # para objetivo X
            r'(?i)(?:target|goal)\s+(?:of\s+)?(\d+(?:[.,]\d+)?(?:\s*%)?)',  # English pattern
            r'(?i)to\s+(?:target|goal)\s+(?:of\s+)?(\d+(?:[.,]\d+)?(?:\s*%)?)',  # to target/goal of X%
            r'(?i)to\s+(?:a\s+)?goal\s+(?:of\s+)?(\d+(?:[.,]\d+)?(?:\s*%)?)',  # to goal X
            r'(?i)(?:hasta|to)\s+(?:el\s+)?(?:objetivo|target|goal)\s+(\d+(?:[.,]\d+)?(?:\s*%)?)',  # hasta objetivo X%
            # Qualitative patterns (no numbers)
            r'(?i)(?:alcanzar|lograr|hasta)\s+(?:el\s+)?(?:objetivo|prop[óo]sito)\b',  # alcanzar el objetivo
            r'(?i)(?:mejorar|improve)\b',  # mejorar (generic improvement)
            r'(?i)(?:para|to)\s+(?:el\s+)?(?:objetivo|aim)\b',  # para el objetivo
            r'(?i)la\s+meta\s+es\s+',  # la meta es
            r'(?i)el\s+objetivo\s+es\s+',  # el objetivo es
        ]

        # Time horizon patterns (examples)
        self.time_horizon_patterns = [
            r'(?i)(?:para el año|by year)\s*(20\d{2})',
            r'(?i)(?:en|by)\s*(?:el|the)?\s*(?:trimestre|quarter)\s*(?:[1-4])',
            r'(?i)(?:en|by)\s*(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|january|february|march|april|may|june|july|august|september|october|november|december)',
        ]


_WORKER_SCORERS: Dict[str, FeasibilityScorer] = {}


def _score_indicator_with_config(
    config_dict: Dict[str, Any],
    indicator: str,
    evidencia: Optional[int],
) -> IndicatorScore:
    """Helper for distributed backends to score indicators with cached scorers."""

    cache_key = repr(sorted(config_dict.items()))
    scorer = _WORKER_SCORERS.get(cache_key)
    if scorer is None:
        scorer = FeasibilityScorer(FeasibilityConfig(**config_dict))
        _WORKER_SCORERS[cache_key] = scorer
    return scorer.calculate_feasibility_score(indicator, evidencia_soporte=evidencia)

    def detect_components(self, text: str) -> ComponentDict:
        """
        Detect components in the text.
        Returns:
            Dictionary mapping component types to lists of detection results.
            When iterated, returns all detection results as a flat list.
        """
        if not text:
            return ComponentDict({ct: [] for ct in ComponentType})

        results = {
            ComponentType.BASELINE: self._detect_baselines(text),
            ComponentType.TARGET: self._detect_targets(text),
            ComponentType.TIMEFRAME: self._detect_timeframes(text),
            ComponentType.UNIT: self._detect_units(text),
            ComponentType.RESPONSIBLE: self._detect_responsibles(text),
            ComponentType.DATE: self._detect_dates(text),
        }

        return ComponentDict(results)

    # Placeholder detection methods (implementations required)
    def _detect_baselines(self, text: str) -> List[DetectionResult]:
        matches = []
        for pattern in self.baseline_patterns:
            for m in re.finditer(pattern, text):
                numeric_val = self._extract_numeric(m)
                matches.append(DetectionResult(
                    text=m.group(0),
                    component_type=ComponentType.BASELINE,
                    numeric_value=numeric_val,
                    unit="",
                    metadata={"pattern": pattern}
                ))
        return matches

    def _detect_targets(self, text: str) -> List[DetectionResult]:
        matches = []
        for pattern in self.target_patterns:
            for m in re.finditer(pattern, text):
                numeric_val = self._extract_numeric(m)
                matches.append(DetectionResult(
                    text=m.group(0),
                    component_type=ComponentType.TARGET,
                    numeric_value=numeric_val,
                    unit="",
                    metadata={"pattern": pattern}
                ))
        return matches

    def _detect_timeframes(self, text: str) -> List[DetectionResult]:
        matches = []
        for pattern in self.time_horizon_patterns:
            for m in re.finditer(pattern, text):
                matches.append(DetectionResult(
                    text=m.group(0),
                    component_type=ComponentType.TIMEFRAME,
                    numeric_value=None,
                    unit="",
                    metadata={"pattern": pattern}
                ))
        return matches

    @staticmethod
    def _detect_units(text: str) -> List[DetectionResult]:
        # Add unit detection logic if needed
        return []

    @staticmethod
    def _detect_responsibles(text: str) -> List[DetectionResult]:
        # Add responsible entity detection logic if needed
        return []

    @staticmethod
    def _detect_dates(text: str) -> List[DetectionResult]:
        # Add date detection logic if needed
        return []

    @staticmethod
    def _extract_numeric(match) -> Optional[float]:
        for group in match.groups():
            try:
                val = float(str(group).replace(',', '.').replace('%', ''))
                return val
            except Exception:
                continue
        return None

    def evaluate_indicator(self, text: str) -> IndicatorScore:
        """
        Evaluate a single indicator.
        """
        components_dict = self.detect_components(text)
        baselines = components_dict[ComponentType.BASELINE]
        targets = components_dict[ComponentType.TARGET]
        timeframes = components_dict[ComponentType.TIMEFRAME]
        units = components_dict[ComponentType.UNIT]
        responsibles = components_dict[ComponentType.RESPONSIBLE]

        has_baseline = len(baselines) > 0
        has_target = len(targets) > 0
        has_timeframe = len(timeframes) > 0
        has_quantitative_target = any(t.numeric_value is not None for t in targets)
        has_unit = len(units) > 0
        has_responsible = len(responsibles) > 0

        smart_score = (
            (1.0 if has_baseline else 0.0) * 0.2 +
            (1.0 if has_target else 0.0) * 0.2 +
            (1.0 if has_timeframe else 0.0) * 0.2 +
            (1.0 if has_quantitative_target else 0.0) * 0.2 +
            (1.0 if has_unit else 0.0) * 0.1 +
            (1.0 if has_responsible else 0.0) * 0.1
        )

        feasibility_score = self._calculate_feasibility_score(
            has_baseline, has_target, has_timeframe, has_quantitative_target, has_unit, has_responsible
        )

        components = []
        for ct in ComponentType:
            components.extend(components_dict[ct])

        return IndicatorScore(
            text=text,
            has_baseline=has_baseline,
            has_target=has_target,
            has_timeframe=has_timeframe,
            has_quantitative_target=has_quantitative_target,
            has_unit=has_unit,
            has_responsible=has_responsible,
            smart_score=smart_score,
            feasibility_score=feasibility_score,
            components=components,
        )

    @staticmethod
    def _calculate_feasibility_score(
        has_baseline: bool, has_target: bool, has_timeframe: bool,
        has_quantitative_target: bool, has_unit: bool, has_responsible: bool
    ) -> float:
        """
        Returns:
            Score between 0 and 1 representing overall feasibility
        """
        # MANDATORY: Both baseline and target must be present
        if not has_baseline or not has_target:
            return 0.0

        # Essential components check (baseline, target, timeframe)
        essential_components = (
            (1.0 if has_baseline else 0.0) * 0.4 +
            (1.0 if has_target else 0.0) * 0.4 +
            (1.0 if has_timeframe else 0.0) * 0.2
        )
        # Quantitative boost
        quantitative_boost = 0.2 if has_quantitative_target else 0.0
        unit_boost = 0.1 if has_unit else 0.0
        responsible_boost = 0.1 if has_responsible else 0.0
        score = essential_components + quantitative_boost + unit_boost + responsible_boost
        return min(score, 1.0)

    def batch_score(
        self,
        indicators: List[str],
        compare_backends: bool = False,
        use_parallel: Optional[bool] = None,
        evidencia_soporte_list: Optional[List[int]] = None,
    ) -> List[IndicatorScore]:
        """Process a batch of indicators honouring the configured parallel backend."""

        if evidencia_soporte_list is None:
            evidencia_soporte_list = [None] * len(indicators)

        if len(evidencia_soporte_list) != len(indicators):
            evidencia_soporte_list = (
                evidencia_soporte_list
                + [None] * (len(indicators) - len(evidencia_soporte_list))
            )

        should_parallel = self.enable_parallel if use_parallel is None else use_parallel
        should_parallel = should_parallel and self.n_jobs > 1 and len(indicators) > 1

        if should_parallel:
            executor = self._select_backend(self.backend)
            if executor is not None:
                return executor(indicators, evidencia_soporte_list)
            self.logger.warning(
                "Parallel execution requested but backend '%s' unavailable; falling back to sequential",
                self.backend,
            )

        results = []
        for indicator, evidencia_soporte in zip(indicators, evidencia_soporte_list):
            results.append(
                self.calculate_feasibility_score(
                    indicator, evidencia_soporte=evidencia_soporte
                )
            )
        return results

    def calculate_feasibility_score(self, text: str, evidencia_soporte: Optional[int] = None) -> IndicatorScore:
        """
        Wrapper method for evaluate_indicator() for backward compatibility.

        Args:
            text: Text to evaluate
            evidencia_soporte: Number of supporting evidences (0 = override to insufficient)

        Returns:
            IndicatorScore with evaluation results
        """
        if evidencia_soporte is not None and evidencia_soporte == 0:
            return IndicatorScore(
                text=text,
                has_baseline=False,
                has_target=False,
                has_timeframe=False,
                has_quantitative_target=False,
                has_unit=False,
                has_responsible=False,
                smart_score=0.0,
                feasibility_score=0.0,
                components=[],
                quality_tier="REQUIERE MAYOR EVIDENCIA",
                components_detected=[],
                detailed_matches=[],
                has_quantitative_baseline=False
            )

        result = self.evaluate_indicator(text)

        # Add quality_tier for compatibility
        if result.feasibility_score == 0.0:
            result.quality_tier = "insufficient"
        elif result.feasibility_score < 0.3:
            result.quality_tier = "insufficient"
        elif result.feasibility_score < 0.5:
            result.quality_tier = "low"
        elif result.feasibility_score < 0.7:
            result.quality_tier = "medium"
        else:
            result.quality_tier = "high"

        # Add components_detected for compatibility
        all_components = []
        components_dict = self.detect_components(text)
        for component_list in components_dict.values():
            all_components.extend(component_list)
        result.components_detected = [c.component_type for c in all_components]

        # Add detailed_matches for compatibility
        result.detailed_matches = all_components

        # Add quantitative detection for compatibility
        has_quantitative_baseline = False
        for baseline in components_dict[ComponentType.BASELINE]:
            if baseline.numeric_value is not None:
                has_quantitative_baseline = True
                break
        result.has_quantitative_baseline = has_quantitative_baseline

        return result

    def batch_score_with_monitoring(self, indicators: List[str]) -> Any:
        """
        Batch score indicators with monitoring metrics.
        Returns an object with:
            - scores: List[IndicatorScore]
            - total_indicators: int
            - duracion_segundos: float
            - planes_por_minuto: float
            - execution_time: str
        """
        import time
        from dataclasses import dataclass

        @dataclass
        class MonitoringResult:
            scores: List[IndicatorScore]
            total_indicators: int
            duracion_segundos: float
            planes_por_minuto: float
            execution_time: str

        start_time = time.time()
        scores = self.batch_score(indicators)
        duration = time.time() - start_time

        rate = (len(indicators) / duration * 60) if duration > 0 else 0

        return MonitoringResult(
            scores=scores,
            total_indicators=len(indicators),
            duracion_segundos=duration,
            planes_por_minuto=rate,
            execution_time=f"{duration:.2f}s"
        )

    def _create_picklable_copy(self):
        """Create a picklable copy of the scorer (without logger)."""
        import copy
        scorer_copy = copy.copy(self)
        if hasattr(scorer_copy, 'logger'):
            delattr(scorer_copy, 'logger')
        return scorer_copy

    def get_detection_rules_documentation(self) -> str:
        """
        Generate comprehensive documentation of detection rules.
        Returns:
            Markdown-formatted documentation string
        """
        doc = []
        doc.append("# Feasibility Scorer Detection Rules Documentation\n")
        doc.append("\n## Overview\n")
        doc.append("This scorer evaluates plan indicators for feasibility by detecting key components.\n")

        doc.append("\n## Spanish Pattern Recognition\n")
        doc.append("\n### Baseline Patterns\n")
        for pattern in self.baseline_patterns[:3]:
            doc.append(f"- `{pattern}`\n")

        doc.append("\n### Target Patterns\n")
        for pattern in self.target_patterns[:3]:
            doc.append(f"- `{pattern}`\n")

        doc.append("\n### Time Horizon Patterns\n")
        for pattern in self.time_horizon_patterns[:3]:
            doc.append(f"- `{pattern}`\n")

        doc.append("\n## Quality Tiers\n")
        doc.append("- **High** (0.7-1.0): Complete information with quantitative elements\n")
        doc.append("- **Medium** (0.5-0.7): Basic components with some quantification\n")
        doc.append("- **Low** (0.3-0.5): Minimal components\n")
        doc.append("- **Insufficient** (0.0-0.3): Missing critical components\n")

        doc.append("\n## Examples\n")
        doc.append("\n### High Quality\n")
        doc.append('- "Incrementar la línea base de 65% a una meta de 85% para el año 2025"\n')

        doc.append("\n### Medium Quality\n")
        doc.append('- "Mejorar desde la situación inicial hasta el objetivo propuesto"\n')

        doc.append("\n### Low Quality\n")
        doc.append('- "Partir de la línea base para alcanzar el objetivo"\n')

        doc.append("\n### Insufficient\n")
        doc.append('- "Aumentar el acceso a servicios de salud en la región"\n')

        return "".join(doc)

    def calcular_calidad_evidencia(self, text: str) -> float:
        """
        Calculate evidence quality score based on content richness.

        Args:
            text: Text to evaluate

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not text or not text.strip():
            return 0.0

        text = self._normalize_text(text)
        score = 0.0

        # Monetary value detection (0.3 points)
        monetary_patterns = [
            r'(?i)COP\s*\$?\s*[\d.,]+\s*(?:millones?|mil|billones?)',
            r'(?i)\$\s*[\d.,]+\s*(?:millones?|mil|USD|pesos)',
            r'(?i)[\d.,]+\s*(?:millones?|mil)\s*(?:pesos|dólares|dollars)',
        ]
        for pattern in monetary_patterns:
            if re.search(pattern, text):
                score += 0.15
                break

        # Temporal indicators (0.25 points total)
        if re.search(r'(?i)20\d{2}', text):  # Years
            score += 0.10
        if re.search(r'(?i)(?:Q[1-4]|trimestre|quarter)', text):  # Quarters
            score += 0.08
        if re.search(r'(?i)(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|january|february|march|april|may|june|july|august|september|october|november|december)', text):  # Months
            score += 0.07

        # Measurement terminology (0.25 points total)
        if re.search(r'(?i)(?:línea\s+base|baseline|valor\s+inicial)', text):
            score += 0.10
        if re.search(r'(?i)(?:meta|objetivo|target|goal)', text):
            score += 0.08
        if re.search(r'(?i)(?:indicador|KPI|métrica|monitoreo|evaluación)', text):
            score += 0.07

        # Periodicity (0.05 points)
        if re.search(r'(?i)(?:periodicidad|frecuencia|frequency|mensual|trimestral|anual|quarterly|monthly)', text):
            score += 0.05

        # Structure penalty for titles without values
        title_markers = [r'^[\s]*[•#\-]', r'^[A-Z\s]{10,}:']
        is_title = any(re.match(pattern, text) for pattern in title_markers)
        has_values = bool(re.search(r'\d', text))

        if is_title and not has_values:
            score *= 0.5  # 50% penalty

        # Cap at 1.0
        return min(1.0, score)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize Unicode text for consistent pattern matching."""
        import unicodedata
        return unicodedata.normalize("NFKC", text)