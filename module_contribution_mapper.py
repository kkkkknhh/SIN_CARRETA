#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Contribution Mapper - MINIMINIMOON v2.0
================================================

Maps which modules contribute to answering each of the 300 questions,
with contribution percentages. This provides transparency on how each
module's evidence flows into the final answer.

CANONICAL QUESTION STRUCTURE:
- 300 questions total (D1-D6 × P1-P10 × 5 questions each)
- Format: P#-D#-Q# (e.g., P1-D1-Q1)

MODULE CATEGORIES:
1. CORE DETECTORS (Stages 1-11): Base evidence extraction
2. PDM_CONTRA: Advanced contradiction, competence, risk analysis
3. FACTIBILIDAD: Feasibility patterns and scoring
4. EVALUATION: Reliability calibration
5. DOCTORAL: Argumentation quality

CONTRIBUTION MODEL:
- Each question has primary (50%+) and supporting (10-40%) modules
- Percentages ensure traceability: "70% from responsibility, 20% from monetary, 10% from feasibility"
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModuleCategory(str, Enum):
    """Categories of evidence-generating modules"""
    
    # Core pipeline detectors (Stages 1-11)
    SANITIZATION = "sanitization"
    PLAN_PROCESSING = "plan_processing"
    SEGMENTATION = "segmentation"
    EMBEDDING = "embedding"
    RESPONSIBILITY = "responsibility_detection"
    CONTRADICTION = "contradiction_detection"
    MONETARY = "monetary_detection"
    FEASIBILITY = "feasibility_scoring"
    CAUSAL = "causal_detection"
    TEORIA = "teoria_cambio"
    DAG = "dag_validation"
    
    # Enrichment modules (Stage 14)
    PDM_CONTRA_CORE = "pdm_contra_core"
    PDM_CONTRA_RISK = "pdm_contra_risk"
    PDM_CONTRA_PATTERNS = "pdm_contra_patterns"
    PDM_CONTRA_NLI = "pdm_contra_nli"
    PDM_CONTRA_COMPETENCE = "pdm_contra_competence"
    PDM_CONTRA_TRACER = "pdm_contra_tracer"
    FACTIBILIDAD_PATTERNS = "factibilidad_patterns"
    RELIABILITY_CALIBRATION = "reliability_calibration"
    
    # Answer generation
    DOCTORAL_ARGUMENTATION = "doctoral_argumentation"


@dataclass
class ModuleContribution:
    """Represents a module's contribution to answering a question"""
    
    module: ModuleCategory
    contribution_percentage: float  # 0-100
    evidence_types: List[str] = field(default_factory=list)
    description: str = ""
    
    def __post_init__(self):
        """Validate contribution percentage"""
        if not 0 <= self.contribution_percentage <= 100:
            raise ValueError(f"Contribution must be 0-100, got {self.contribution_percentage}")


@dataclass
class QuestionContributionMap:
    """Maps which modules contribute to a specific question"""
    
    question_id: str  # Format: P#-D#-Q# or D#-Q#
    dimension: int  # 1-6
    point: Optional[int]  # 1-10 (for P#-D#-Q# format)
    question_number: int  # Question number within dimension
    contributions: List[ModuleContribution] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate total contributions sum to ~100%"""
        total = sum(c.contribution_percentage for c in self.contributions)
        if self.contributions and not (95 <= total <= 105):
            logger.warning(
                f"Question {self.question_id} contributions sum to {total}% (expected ~100%)"
            )
    
    def get_primary_module(self) -> Optional[ModuleContribution]:
        """Get the module with highest contribution"""
        if not self.contributions:
            return None
        return max(self.contributions, key=lambda c: c.contribution_percentage)
    
    def get_supporting_modules(self) -> List[ModuleContribution]:
        """Get all modules except the primary one"""
        primary = self.get_primary_module()
        if not primary:
            return []
        return [c for c in self.contributions if c.module != primary.module]


class ModuleContributionMapper:
    """
    Strategic mapper that defines which modules contribute to each question.
    
    This is the "choreographer" that provides clarity on:
    - Which module answers which question
    - With what percentage contribution
    - Which evidence types are used
    
    Example:
        Question P1-D1-Q1 (Responsibility identification):
        - responsibility_detection: 50% (primary)
        - pdm_contra_competence: 25% (competence validation)
        - factibilidad_patterns: 15% (feasibility markers)
        - reliability_calibration: 10% (confidence adjustment)
    """
    
    def __init__(self, mapping_file: Optional[Path] = None):
        """
        Initialize mapper with optional pre-defined mapping file.
        
        Args:
            mapping_file: Path to JSON file with pre-defined mappings
        """
        self.mapping_file = mapping_file
        self.question_maps: Dict[str, QuestionContributionMap] = {}
        
        if mapping_file and mapping_file.exists():
            self._load_mappings(mapping_file)
        else:
            self._initialize_default_mappings()
    
    def _load_mappings(self, path: Path):
        """Load pre-defined mappings from JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for q_data in data.get('questions', []):
                question_id = q_data['question_id']
                contributions = [
                    ModuleContribution(
                        module=ModuleCategory(c['module']),
                        contribution_percentage=c['percentage'],
                        evidence_types=c.get('evidence_types', []),
                        description=c.get('description', '')
                    )
                    for c in q_data['contributions']
                ]
                
                self.question_maps[question_id] = QuestionContributionMap(
                    question_id=question_id,
                    dimension=q_data['dimension'],
                    point=q_data.get('point'),
                    question_number=q_data['question_number'],
                    contributions=contributions
                )
            
            logger.info(f"Loaded {len(self.question_maps)} question mappings from {path}")
        
        except Exception as e:
            logger.error(f"Error loading mappings from {path}: {e}")
            self._initialize_default_mappings()
    
    def _initialize_default_mappings(self):
        """
        Initialize default contribution mappings based on question structure.
        
        CANONICAL MAPPING STRATEGY:
        - D1 (INSUMOS): Heavy on responsibility, monetary, feasibility
        - D2 (ACTIVIDADES): Heavy on causal, teoria, feasibility
        - D3 (PRODUCTOS): Heavy on feasibility, monetary, patterns
        - D4 (RESULTADOS): Heavy on teoria, causal, dag
        - D5 (IMPACTOS): Heavy on causal, teoria, contradiction
        - D6 (CAUSALIDAD): Heavy on causal, teoria, dag, patterns
        
        ALL DIMENSIONS benefit from:
        - pdm_contra modules (competence, risk, nli, patterns)
        - factibilidad_patterns
        - reliability_calibration
        """
        logger.info("Initializing default module contribution mappings...")
        
        # Define dimension-specific primary modules
        dimension_primary_modules = {
            1: [  # D1 - INSUMOS
                (ModuleCategory.RESPONSIBILITY, 30),
                (ModuleCategory.MONETARY, 20),
                (ModuleCategory.FEASIBILITY, 15),
            ],
            2: [  # D2 - ACTIVIDADES
                (ModuleCategory.CAUSAL, 25),
                (ModuleCategory.TEORIA, 20),
                (ModuleCategory.FEASIBILITY, 15),
            ],
            3: [  # D3 - PRODUCTOS
                (ModuleCategory.FEASIBILITY, 30),
                (ModuleCategory.MONETARY, 20),
                (ModuleCategory.PDM_CONTRA_PATTERNS, 10),
            ],
            4: [  # D4 - RESULTADOS
                (ModuleCategory.TEORIA, 30),
                (ModuleCategory.CAUSAL, 25),
                (ModuleCategory.DAG, 10),
            ],
            5: [  # D5 - IMPACTOS
                (ModuleCategory.CAUSAL, 30),
                (ModuleCategory.TEORIA, 25),
                (ModuleCategory.CONTRADICTION, 10),
            ],
            6: [  # D6 - CAUSALIDAD
                (ModuleCategory.CAUSAL, 30),
                (ModuleCategory.TEORIA, 25),
                (ModuleCategory.DAG, 15),
            ],
        }
        
        # Common supporting modules (applied to all questions)
        common_supporting = [
            (ModuleCategory.PDM_CONTRA_COMPETENCE, 8),
            (ModuleCategory.PDM_CONTRA_RISK, 5),
            (ModuleCategory.PDM_CONTRA_NLI, 4),
            (ModuleCategory.FACTIBILIDAD_PATTERNS, 8),
            (ModuleCategory.RELIABILITY_CALIBRATION, 5),
        ]
        
        # Generate mappings for all 300 questions (D1-D6 × P1-P10 × Q1-Q50)
        for dim in range(1, 7):  # D1-D6
            for point in range(1, 11):  # P1-P10
                for q_num in range(1, 51):  # Q1-Q50 (30 questions per D-P combo, +20 buffer)
                    question_id = f"P{point}-D{dim}-Q{q_num}"
                    rubric_id = f"D{dim}-Q{q_num}"
                    
                    # Get dimension-specific modules
                    primary_modules = dimension_primary_modules.get(dim, [])
                    
                    # Build contributions list
                    contributions = []
                    
                    # Add primary modules
                    for module, percentage in primary_modules:
                        contributions.append(
                            ModuleContribution(
                                module=module,
                                contribution_percentage=percentage,
                                evidence_types=[f"{module.value}_evidence"],
                                description=f"Primary evidence from {module.value}"
                            )
                        )
                    
                    # Add common supporting modules
                    for module, percentage in common_supporting:
                        contributions.append(
                            ModuleContribution(
                                module=module,
                                contribution_percentage=percentage,
                                evidence_types=[f"{module.value}_evidence"],
                                description=f"Supporting evidence from {module.value}"
                            )
                        )
                    
                    # Normalize to 100%
                    total = sum(c.contribution_percentage for c in contributions)
                    if total > 0:
                        for contrib in contributions:
                            contrib.contribution_percentage = (
                                contrib.contribution_percentage / total * 100
                            )
                    
                    # Create mapping
                    self.question_maps[rubric_id] = QuestionContributionMap(
                        question_id=rubric_id,
                        dimension=dim,
                        point=point,
                        question_number=q_num,
                        contributions=contributions
                    )
        
        logger.info(f"Initialized {len(self.question_maps)} default question mappings")
    
    def get_question_mapping(self, question_id: str) -> Optional[QuestionContributionMap]:
        """
        Get contribution mapping for a specific question.
        
        Args:
            question_id: Question ID in P#-D#-Q# or D#-Q# format
        
        Returns:
            QuestionContributionMap or None if not found
        """
        # Try direct lookup
        if question_id in self.question_maps:
            return self.question_maps[question_id]
        
        # Try converting P#-D#-Q# to D#-Q#
        import re
        match = re.search(r'D(\d+)-Q(\d+)', question_id)
        if match:
            rubric_id = f"D{match.group(1)}-Q{match.group(2)}"
            return self.question_maps.get(rubric_id)
        
        return None
    
    def get_modules_for_question(self, question_id: str) -> List[ModuleCategory]:
        """Get list of all modules that contribute to a question"""
        mapping = self.get_question_mapping(question_id)
        if not mapping:
            return []
        return [c.module for c in mapping.contributions]
    
    def get_module_contribution(
        self, question_id: str, module: ModuleCategory
    ) -> Optional[ModuleContribution]:
        """Get specific module's contribution to a question"""
        mapping = self.get_question_mapping(question_id)
        if not mapping:
            return None
        
        for contrib in mapping.contributions:
            if contrib.module == module:
                return contrib
        
        return None
    
    def export_mappings(self, output_path: Path):
        """Export mappings to JSON file for inspection and editing"""
        data = {
            'metadata': {
                'version': '1.0',
                'total_questions': len(self.question_maps),
                'description': 'Module contribution mappings for MINIMINIMOON evaluation'
            },
            'questions': []
        }
        
        for q_id, q_map in sorted(self.question_maps.items()):
            data['questions'].append({
                'question_id': q_map.question_id,
                'dimension': q_map.dimension,
                'point': q_map.point,
                'question_number': q_map.question_number,
                'contributions': [
                    {
                        'module': c.module.value,
                        'percentage': round(c.contribution_percentage, 2),
                        'evidence_types': c.evidence_types,
                        'description': c.description
                    }
                    for c in q_map.contributions
                ]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(self.question_maps)} mappings to {output_path}")
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """Get summary statistics about module usage across all questions"""
        module_usage = {}
        
        for q_map in self.question_maps.values():
            for contrib in q_map.contributions:
                module = contrib.module.value
                if module not in module_usage:
                    module_usage[module] = {
                        'count': 0,
                        'total_percentage': 0.0,
                        'avg_percentage': 0.0,
                        'max_percentage': 0.0,
                        'min_percentage': 100.0
                    }
                
                stats = module_usage[module]
                stats['count'] += 1
                stats['total_percentage'] += contrib.contribution_percentage
                stats['max_percentage'] = max(
                    stats['max_percentage'], contrib.contribution_percentage
                )
                stats['min_percentage'] = min(
                    stats['min_percentage'], contrib.contribution_percentage
                )
        
        # Calculate averages
        for module, stats in module_usage.items():
            if stats['count'] > 0:
                stats['avg_percentage'] = stats['total_percentage'] / stats['count']
        
        return {
            'total_questions': len(self.question_maps),
            'module_usage': module_usage
        }


def create_default_mapper(output_path: Optional[Path] = None) -> ModuleContributionMapper:
    """
    Create and optionally export default module contribution mapper.
    
    Args:
        output_path: Optional path to export mappings as JSON
    
    Returns:
        Configured ModuleContributionMapper
    """
    mapper = ModuleContributionMapper()
    
    if output_path:
        mapper.export_mappings(output_path)
    
    return mapper


if __name__ == "__main__":
    # Example usage: create and export default mappings
    mapper = create_default_mapper(
        output_path=Path("artifacts/module_contribution_mappings.json")
    )
    
    # Print statistics
    stats = mapper.get_summary_statistics()
    print(f"\n=== Module Contribution Statistics ===")
    print(f"Total questions: {stats['total_questions']}")
    print(f"\nModule Usage:")
    for module, data in sorted(
        stats['module_usage'].items(),
        key=lambda x: x[1]['avg_percentage'],
        reverse=True
    ):
        print(
            f"  {module:35s} → "
            f"Used in {data['count']:3d} questions, "
            f"Avg: {data['avg_percentage']:5.1f}%, "
            f"Range: [{data['min_percentage']:5.1f}% - {data['max_percentage']:5.1f}%]"
        )
    
    # Show example question mapping
    example_q = "D1-Q1"
    mapping = mapper.get_question_mapping(example_q)
    if mapping:
        print(f"\n=== Example: {example_q} ===")
        primary = mapping.get_primary_module()
        print(f"Primary: {primary.module.value} ({primary.contribution_percentage:.1f}%)")
        print(f"Supporting modules:")
        for contrib in mapping.get_supporting_modules():
            print(f"  - {contrib.module.value}: {contrib.contribution_percentage:.1f}%")
