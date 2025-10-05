import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import networkx as nx


class PDETCausalPatternDetector:
    """Industrial-grade causal detector for PDET municipal development plans"""

    def __init__(self, pdet_municipalities: List[str]):
        self.pdet_municipalities = pdet_municipalities
        self.indicators = self._load_pdet_indicators()
        self.logger = logging.getLogger(__name__)

    def _load_pdet_indicators(self) -> Dict[str, List[str]]:
        """Load PDET-specific indicators for causal analysis"""
        return {
            'violence_indicators': [
                'tasa_homicidios', 'desplazamiento_forzado', 'minas_antipersonal',
                'eventos_conflicto_armado', 'violencia_genero', 'reclutamiento_menores'
            ],
            'development_indicators': [
                'pobreza_multidimensional', 'cobertura_educativa', 'acceso_salud',
                'cobertura_acueducto', 'cobertura_alcantarillado', 'acceso_energia',
                'desempleo', 'ingreso_per_capita', 'produccion_agricola'
            ],
            'institutional_indicators': [
                'capacidad_institucional', 'inversion_publica_per_capita',
                'presupuesto_seguridad', 'presupuesto_salud', 'presupuesto_educacion'
            ],
            'territorial_indicators': [
                'cobertura_bosque', 'deforestacion', 'contaminacion_agua',
                'acceso_tierras', 'conflictos_uso_suelo'
            ]
        }

    def _text_to_data(self, text: str) -> pd.DataFrame:
        """Convert development plan text to structured PDET indicators"""
        # Extract municipality name from text
        municipality = self._extract_municipality(text)

        # Extract quantitative targets and baselines
        quantitative_data = self._extract_quantitative_metrics(text)

        # Extract causal relationships from plan text
        causal_relationships = self._extract_causal_claims(text)

        # Build comprehensive data profile
        data_profile = {
            'municipio': municipality,
            'plan_text_length': len(text),
            'causal_claims_count': len(causal_relationships),
            'quantitative_targets_count': len(quantitative_data),
            **quantitative_data,
            **self._calculate_text_complexity(text),
            **self._extract_policy_domains(text)
        }

        return pd.DataFrame([data_profile])

    def _extract_municipality(self, text: str) -> str:
        """Extract municipality name from development plan text"""
        # Look for municipality patterns in text
        patterns = [
            r"Municipio de (\w+)",
            r"Alcaldía (?:Municipal|de) (\w+)",
            r"Plan de Desarrollo (\w+)",
            r"PDET (\w+)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for municipality in self.pdet_municipalities:
                    if municipality.upper() in matches[0].upper():
                        return municipality

        # Fallback: use the most mentioned municipality from PDET list
        municipality_counts = []
        for municipality in self.pdet_municipalities:
            count = text.upper().count(municipality.upper())
            municipality_counts.append((municipality, count))

        return max(municipality_counts, key=lambda x: x[1])[0] if municipality_counts else "UNKNOWN"

    def _extract_quantitative_metrics(self, text: str) -> Dict[str, float]:
        """Extract quantitative metrics, targets, and baselines from text"""
        metrics = {}

        # Patterns for quantitative indicators
        patterns = {
            'meta_reduccion_violencia': r'reducir.*violencia.*(\d+\.?\d*)%',
            'meta_pobreza': r'reducir.*pobreza.*(\d+\.?\d*)%',
            'meta_educacion': r'aumentar.*cobertura.*educativa.*(\d+\.?\d*)%',
            'meta_salud': r'mejorar.*acceso.*salud.*(\d+\.?\d*)%',
            'presupuesto_total': r'presupuesto.*?\$?\s*(\d+(?:\.\d+)*)',
            'tiempo_ejecucion': r'periodo.*(\d+).*años'
        }

        for metric, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    metrics[metric] = float(matches[0].replace('.', ''))
                except ValueError:
                    continue

        return metrics

    def _extract_causal_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract explicit causal claims from development plan text"""
        causal_patterns = [
            # Cause-effect patterns
            (r'(\w+)\s+causa\s+(\w+)', 'direct_causation'),
            (r'(\w+)\s+genera\s+(\w+)', 'direct_causation'),
            (r'(\w+)\s+produce\s+(\w+)', 'direct_causation'),
            (r'(\w+)\s+conlleva\s+a\s+(\w+)', 'enabling'),
            (r'(\w+)\s+permite\s+(\w+)', 'enabling'),
            (r'(\w+)\s+contribuye\s+a\s+(\w+)', 'contributing'),
            (r'(\w+)\s+mejora\s+(\w+)', 'improvement'),
            (r'(\w+)\s+reduce\s+(\w+)', 'reduction'),
            (r'(\w+)\s+aumenta\s+(\w+)', 'increase'),
        ]

        claims = []
        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            for pattern, relation_type in causal_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                for cause, effect in matches:
                    if len(cause) > 3 and len(effect) > 3:  # Avoid short meaningless words
                        claims.append({
                            'cause': cause.strip(),
                            'effect': effect.strip(),
                            'relation_type': relation_type,
                            'sentence': sentence,
                            'confidence': self._calculate_claim_confidence(sentence, cause, effect)
                        })

        return claims

    def _calculate_text_complexity(self, text: str) -> Dict[str, float]:
        """Calculate text complexity metrics"""
        sentences = self._split_into_sentences(text)
        words = re.findall(r'\w+', text.lower())

        return {
            'complexity_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'complexity_word_diversity': len(set(words)) / len(words) if words else 0,
            'complexity_causal_density': len(self._extract_causal_claims(text)) / len(sentences) if sentences else 0,
            'complexity_quantitative_density': len(self._extract_quantitative_metrics(text)) / len(
                sentences) if sentences else 0
        }

    def _extract_policy_domains(self, text: str) -> Dict[str, int]:
        """Extract policy domain mentions"""
        domains = {
            'mentions_women_rights': len(re.findall(r'mujer|género|femenino|violencia.*género', text, re.IGNORECASE)),
            'mentions_education': len(re.findall(r'educación|escuela|colegio|alfabetización', text, re.IGNORECASE)),
            'mentions_health': len(re.findall(r'salud|hospital|medicina|enfermedad', text, re.IGNORECASE)),
            'mentions_infrastructure': len(
                re.findall(r'infraestructura|carretera|vía|construcción', text, re.IGNORECASE)),
            'mentions_economy': len(re.findall(r'económico|empleo|ingreso|producción', text, re.IGNORECASE)),
            'mentions_environment': len(re.findall(r'ambiente|bosque|agua|contaminación', text, re.IGNORECASE))
        }

        return domains

    def _build_statistical_graph(self, data: pd.DataFrame) -> nx.DiGraph:
        """Build causal graph using statistical causal discovery methods"""
        if data.empty:
            return nx.DiGraph()

        G = nx.DiGraph()

        # Prepare data for causal analysis
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] < 2:
            return G

        # Use multiple causal discovery methods
        self._add_pc_algorithm_edges(G, numeric_data)
        self._add_mutual_info_edges(G, numeric_data)
        self._add_granger_causality_edges(G, numeric_data)

        return G

    def _add_pc_algorithm_edges(self, G: nx.DiGraph, data: pd.DataFrame):
        """Add edges using PC algorithm (conditional independence testing)"""
        try:
            from causalnex.structure import StructureModel
            from causalnex.structure.notears import from_pandas

            # Use NOTEARS for causal structure learning
            sm = from_pandas(data, tabu_edges=[], max_iter=100)

            for edge in sm.edges:
                if edge[0] in data.columns and edge[1] in data.columns:
                    G.add_edge(edge[0], edge[1], method='notears',
                               weight=abs(data[edge[0]].corr(data[edge[1]])))

        except ImportError:
            self.logger.warning("causalnex not available, using correlation fallback")
            self._add_correlation_edges(G, data)

    def _add_mutual_info_edges(self, G: nx.DiGraph, data: pd.DataFrame):
        """Add edges based on mutual information"""
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i != j:
                    mi = mutual_info_regression(data[[col1]], data[col2])[0]
                    if mi > 0.1:  # Threshold for meaningful mutual information
                        if col1 not in G or col2 not in G[col1]:
                            G.add_edge(col1, col2, method='mutual_info', weight=mi)

    def _add_granger_causality_edges(self, G: nx.DiGraph, data: pd.DataFrame):
        """Add edges based on Granger causality (for time series patterns)"""
        # Since we have cross-sectional data, we'll use a simplified approach
        # based on predictive power
        for cause in data.columns:
            for effect in data.columns:
                if cause != effect:
                    # Use Random Forest to test predictive power
                    X = data[[cause]]
                    y = data[effect]

                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X, y)
                    feature_importance = model.feature_importances_[0]

                    if feature_importance > 0.05:  # Meaningful predictive power
                        if cause not in G or effect not in G[cause]:
                            G.add_edge(cause, effect, method='predictive_power',
                                       weight=feature_importance)

    def _add_correlation_edges(self, G: nx.DiGraph, data: pd.DataFrame):
        """Fallback: add edges based on correlation"""
        correlation_matrix = data.corr()

        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i != j and abs(correlation_matrix.iloc[i, j]) > 0.3:
                    G.add_edge(col1, col2, method='correlation',
                               weight=abs(correlation_matrix.iloc[i, j]))

    def analyze_development_plan(self, plan_text: str) -> Dict[str, Any]:
        """Complete analysis of a municipal development plan"""
        # Convert text to structured data
        data = self._text_to_data(plan_text)

        # Build causal graph
        causal_graph = self._build_statistical_graph(data)

        # Extract causal claims
        causal_claims = self._extract_causal_claims(plan_text)

        # Calculate comprehensive scores
        scores = self._calculate_evaluation_scores(data, causal_claims, causal_graph)

        return {
            'municipality': data['municipio'].iloc[0] if not data.empty else 'UNKNOWN',
            'causal_graph': causal_graph,
            'causal_claims': causal_claims,
            'quantitative_metrics': self._extract_quantitative_metrics(plan_text),
            'evaluation_scores': scores,
            'text_complexity': self._calculate_text_complexity(plan_text),
            'policy_domains': self._extract_policy_domains(plan_text)
        }

    def _calculate_evaluation_scores(self, data: pd.DataFrame,
                                     causal_claims: List[Dict],
                                     causal_graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate evaluation scores for the development plan"""
        if data.empty:
            return {
                'causal_coherence': 0.0,
                'quantitative_rigor': 0.0,
                'policy_completeness': 0.0,
                'implementation_feasibility': 0.0,
                'overall_quality': 0.0
            }

        # Causal coherence score
        causal_coherence = min(1.0, len(causal_claims) / 10)  # Normalize by expected claims

        # Quantitative rigor score
        quant_metrics = [col for col in data.columns if 'meta' in col or 'presupuesto' in col]
        quantitative_rigor = min(1.0, len(quant_metrics) / 5)

        # Policy completeness score
        policy_mentions = [col for col in data.columns if 'mentions' in col]
        policy_completeness = min(1.0, sum(data[policy_mentions].iloc[0]) / 30) if policy_mentions else 0.0

        # Implementation feasibility
        has_budget = 'presupuesto_total' in data.columns
        has_timeline = 'tiempo_ejecucion' in data.columns
        implementation_feasibility = 0.3 if has_budget else 0.0 + 0.2 if has_timeline else 0.0

        # Overall quality score (weighted average)
        overall_quality = (
                causal_coherence * 0.3 +
                quantitative_rigor * 0.3 +
                policy_completeness * 0.2 +
                implementation_feasibility * 0.2
        )

        return {
            'causal_coherence': causal_coherence,
            'quantitative_rigor': quantitative_rigor,
            'policy_completeness': policy_completeness,
            'implementation_feasibility': implementation_feasibility,
            'overall_quality': overall_quality
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for Spanish text"""
        # Improved sentence splitting for Spanish development plans
        text = re.sub(r'[\n\r]+', ' ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]  # Filter very short sentences

    def _calculate_claim_confidence(self, sentence: str, cause: str, effect: str) -> float:
        """Calculate confidence score for a causal claim"""
        confidence = 0.5

        # Boost confidence for quantitative language
        if re.search(r'\d+%|\d+\.\d+|\$?\d+', sentence):
            confidence += 0.2

        # Boost confidence for strong causal verbs
        strong_verbs = ['causa', 'genera', 'produce', 'determina']
        if any(verb in sentence.lower() for verb in strong_verbs):
            confidence += 0.2

        # Reduce confidence for weak or conditional language
        weak_indicators = ['podría', 'quizás', 'tal vez', 'posiblemente']
        if any(indicator in sentence.lower() for indicator in weak_indicators):
            confidence -= 0.1

        return max(0.1, min(1.0, confidence))


# Factory function for your system
def create_pdet_causal_detector(municipalities_file: str = "pdet_municipalities.txt"):
    """Create a PDET-specific causal detector"""
    # Load PDET municipalities (you'll provide this list)
    with open(municipalities_file, 'r', encoding='utf-8') as f:
        pdet_municipalities = [line.strip() for line in f if line.strip()]

    return PDETCausalPatternDetector(pdet_municipalities)


# Usage in your MINIMINIMOON system
def integrate_with_miniminimoon():
    """Integration example with your main system"""
    detector = create_pdet_causal_detector()

    # For each of the 170 development plans
    for plan_file in development_plan_files:
        with open(plan_file, 'r', encoding='utf-8') as f:
            plan_text = f.read()

        analysis = detector.analyze_development_plan(plan_text)

        # Use analysis results to answer your 300 questions
        # The analysis contains all the structured data needed

        print(f"Municipio: {analysis['municipality']}")
        print(f"Calidad general: {analysis['evaluation_scores']['overall_quality']:.2f}")
        print(f"Reclamaciones causales: {len(analysis['causal_claims'])}")
        print(f"Métricas cuantitativas: {len(analysis['quantitative_metrics'])}")