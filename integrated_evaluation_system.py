#!/usr/bin/env python3
"""
INTEGRATED EVALUATION SYSTEM
============================

Sistema que integra el MINIMINIMOON Orchestrator con el Questionnaire Engine
para proporcionar evaluación completa de PDMs.

Este sistema:
1. Usa MINIMINIMOONOrchestrator para procesar el documento (11 pasos)
2. Usa QuestionnaireEngine para evaluar 300 preguntas estructuradas
3. Combina ambos resultados en un reporte unificado
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Importar componentes existentes
from miniminimoon_orchestrator import MINIMINIMOONOrchestrator
from questionnaire_engine import QuestionnaireEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("IntegratedEvaluationSystem")


class IntegratedEvaluationSystem:
    """
    Sistema integrado que une MINIMINIMOON con el Questionnaire Engine.
    
    Proporciona evaluación completa de PDMs combinando:
    - Análisis profundo del MINIMINIMOON (responsabilidades, contradicciones, etc.)
    - Evaluación estructurada de 300 preguntas del Questionnaire Engine
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializar el sistema integrado
        
        Args:
            config_path: Ruta opcional al archivo de configuración
        """
        logger.info("Inicializando Sistema de Evaluación Integrado...")
        
        # Inicializar componente MINIMINIMOON
        logger.info("  → Cargando MINIMINIMOON Orchestrator...")
        self.orchestrator = MINIMINIMOONOrchestrator(config_path)
        
        # Inicializar componente Questionnaire
        logger.info("  → Cargando Questionnaire Engine...")
        self.questionnaire_engine = QuestionnaireEngine()
        
        logger.info("✅ Sistema Integrado inicializado correctamente")
    
    def evaluate_pdm_complete(
        self, 
        pdm_path: str,
        municipality: str = "",
        department: str = "",
        export_json: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluación completa de un PDM usando ambos sistemas
        
        Args:
            pdm_path: Ruta al archivo del PDM
            municipality: Nombre del municipio
            department: Nombre del departamento
            export_json: Si exportar resultados a JSON
            output_dir: Directorio para guardar resultados
            
        Returns:
            Diccionario con resultados completos de ambos sistemas
        """
        logger.info("="*80)
        logger.info(f"INICIANDO EVALUACIÓN COMPLETA: {pdm_path}")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # ========================================
        # PASO 1: Procesamiento con MINIMINIMOON
        # ========================================
        logger.info("\n📊 PASO 1: Procesamiento MINIMINIMOON (11 componentes)")
        logger.info("-"*80)
        
        orchestrator_results = self.orchestrator.process_plan(pdm_path)
        
        if "error" in orchestrator_results:
            logger.error(f"❌ Error en MINIMINIMOON: {orchestrator_results['error']}")
            return {
                "status": "error",
                "error": orchestrator_results["error"],
                "timestamp": datetime.now().isoformat()
            }
        
        logger.info("✅ MINIMINIMOON completado exitosamente")
        self._log_orchestrator_summary(orchestrator_results)
        
        # ========================================
        # PASO 2: Evaluación de 300 Preguntas
        # ========================================
        logger.info("\n📋 PASO 2: Evaluación de 300 Preguntas Estructuradas")
        logger.info("-"*80)
        
        questionnaire_results = self.questionnaire_engine.execute_full_evaluation(
            orchestrator_results=orchestrator_results,  # ← Pasar resultados, NO el path
            municipality=municipality,
            department=department
        )
        
        logger.info("✅ Evaluación de 300 preguntas completada")
        self._log_questionnaire_summary(questionnaire_results)
        
        # ========================================
        # PASO 3: Integración de Resultados
        # ========================================
        logger.info("\n🔗 PASO 3: Integrando Resultados")
        logger.info("-"*80)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        integrated_results = {
            "metadata": {
                "pdm_path": pdm_path,
                "municipality": municipality,
                "department": department,
                "evaluation_timestamp": end_time.isoformat(),
                "total_processing_time_seconds": processing_time,
                "system_version": "IntegratedSystem-v1.0"
            },
            
            # Resultados del MINIMINIMOON Orchestrator
            "miniminimoon": {
                "responsibilities": orchestrator_results.get("responsibilities", []),
                "contradictions": orchestrator_results.get("contradictions", {}),
                "monetary": orchestrator_results.get("monetary", []),
                "feasibility": orchestrator_results.get("feasibility", {}),
                "causal_patterns": orchestrator_results.get("causal_patterns", []),
                "teoria_cambio": orchestrator_results.get("teoria_cambio", {}),
                "evaluation": orchestrator_results.get("evaluation", {}),
                "execution_summary": orchestrator_results.get("execution_summary", {})
            },
            
            # Resultados del Questionnaire Engine
            "questionnaire_300": {
                "global_summary": questionnaire_results.get("global_summary", {}),
                "dimension_summary": questionnaire_results.get("dimension_summary", {}),
                "thematic_points": questionnaire_results.get("thematic_points", []),
                "total_questions_evaluated": len(questionnaire_results.get("evaluation_matrix", []))
            },
            
            # Resumen Ejecutivo Combinado
            "executive_summary": self._generate_executive_summary(
                orchestrator_results, 
                questionnaire_results
            )
        }
        
        # ========================================
        # PASO 4: Exportar Resultados (opcional)
        # ========================================
        if export_json:
            output_path = self._export_results(
                integrated_results, 
                pdm_path, 
                output_dir
            )
            integrated_results["metadata"]["export_path"] = str(output_path)
            logger.info(f"💾 Resultados exportados a: {output_path}")
        
        logger.info("\n" + "="*80)
        logger.info("🎉 EVALUACIÓN COMPLETA FINALIZADA")
        logger.info("="*80)
        
        return integrated_results
    
    def _log_orchestrator_summary(self, results: Dict[str, Any]):
        """Log resumen de resultados del orchestrator"""
        logger.info("\n  Resumen MINIMINIMOON:")
        logger.info(f"    • Responsabilidades detectadas: {len(results.get('responsibilities', []))}")
        logger.info(f"    • Contradicciones detectadas: {results.get('contradictions', {}).get('total', 0)}")
        logger.info(f"    • Valores monetarios detectados: {len(results.get('monetary', []))}")
        logger.info(f"    • Patrones causales detectados: {len(results.get('causal_patterns', []))}")
        logger.info(f"    • Teoría de cambio válida: {results.get('teoria_cambio', {}).get('is_valid', False)}")
        
        eval_score = results.get('evaluation', {}).get('global_score', 0)
        logger.info(f"    • Score global Decálogo: {eval_score:.2f}")
    
    def _log_questionnaire_summary(self, results: Dict[str, Any]):
        """Log resumen de resultados del questionnaire"""
        summary = results.get("global_summary", {})
        logger.info("\n  Resumen Questionnaire Engine:")
        logger.info(f"    • Total preguntas evaluadas: {len(results.get('evaluation_matrix', []))}")
        logger.info(f"    • Score global: {summary.get('score_percentage', 0):.1f}%")
        logger.info(f"    • Clasificación: {summary.get('classification', 'N/A')}")
        logger.info(f"    • Puntos temáticos evaluados: {summary.get('points_evaluated', 0)}")
    
    def _generate_executive_summary(
        self, 
        orchestrator_results: Dict[str, Any],
        questionnaire_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generar resumen ejecutivo combinado
        
        Args:
            orchestrator_results: Resultados del orchestrator
            questionnaire_results: Resultados del questionnaire
            
        Returns:
            Diccionario con resumen ejecutivo
        """
        
        # Score del Questionnaire Engine (0-100)
        q_score = questionnaire_results.get("global_summary", {}).get("score_percentage", 0)
        
        # Score del MINIMINIMOON Decálogo (0-100)
        m_score = orchestrator_results.get("evaluation", {}).get("global_score", 0) * 100
        
        # Score combinado (promedio ponderado)
        combined_score = (q_score * 0.6) + (m_score * 0.4)  # 60% questionnaire, 40% miniminimoon
        
        # Clasificación combinada
        if combined_score >= 85:
            classification = "EXCELENTE"
            recommendation = "PDM cumple con estándares avanzados. Implementación directa recomendada."
        elif combined_score >= 70:
            classification = "BUENO"
            recommendation = "PDM sólido con vacíos menores. Realizar ajustes específicos antes de implementación."
        elif combined_score >= 55:
            classification = "SATISFACTORIO"
            recommendation = "PDM cumple mínimos. Requiere mejoras sustanciales en dimensiones débiles."
        elif combined_score >= 40:
            classification = "INSUFICIENTE"
            recommendation = "PDM con vacíos críticos. Reformulación sustancial requerida."
        else:
            classification = "DEFICIENTE"
            recommendation = "PDM no cumple estándares básicos. Rediseño completo necesario."
        
        # Identificar fortalezas y debilidades
        dim_scores = questionnaire_results.get("dimension_summary", {})
        strongest_dim = max(dim_scores.items(), key=lambda x: x[1]) if dim_scores else ("N/A", 0)
        weakest_dim = min(dim_scores.items(), key=lambda x: x[1]) if dim_scores else ("N/A", 0)
        
        return {
            "combined_score": round(combined_score, 1),
            "classification": classification,
            "recommendation": recommendation,
            "scores": {
                "questionnaire_300": round(q_score, 1),
                "miniminimoon_decalogo": round(m_score, 1),
                "combined": round(combined_score, 1)
            },
            "key_findings": {
                "strongest_dimension": {
                    "id": strongest_dim[0],
                    "score": round(strongest_dim[1], 1)
                },
                "weakest_dimension": {
                    "id": weakest_dim[0],
                    "score": round(weakest_dim[1], 1)
                },
                "contradictions_detected": orchestrator_results.get("contradictions", {}).get("total", 0),
                "budget_defined": len(orchestrator_results.get("monetary", [])) > 0,
                "theory_of_change_valid": orchestrator_results.get("teoria_cambio", {}).get("is_valid", False)
            },
            "priority_actions": self._generate_priority_actions(
                orchestrator_results, 
                questionnaire_results,
                weakest_dim[0]
            )
        }
    
    def _generate_priority_actions(
        self, 
        orchestrator_results: Dict[str, Any],
        questionnaire_results: Dict[str, Any],
        weakest_dimension: str
    ) -> list:
        """Generar acciones prioritarias basadas en resultados"""
        actions = []
        
        # Acción 1: Basada en dimensión más débil
        actions.append(f"Prioridad 1: Fortalecer dimensión {weakest_dimension}")
        
        # Acción 2: Basada en contradicciones
        contradictions = orchestrator_results.get("contradictions", {})
        if contradictions.get("total", 0) > 5:
            actions.append(f"Prioridad 2: Resolver {contradictions['total']} contradicciones detectadas")
        
        # Acción 3: Basada en presupuesto
        if len(orchestrator_results.get("monetary", [])) == 0:
            actions.append("Prioridad 3: Definir asignación presupuestal para programas")
        
        # Acción 4: Basada en teoría de cambio
        teoria = orchestrator_results.get("teoria_cambio", {})
        if not teoria.get("is_valid", False):
            actions.append("Prioridad 4: Desarrollar teoría de cambio explícita con validación causal")
        
        return actions[:5]  # Máximo 5 acciones
    
    def _export_results(
        self, 
        results: Dict[str, Any], 
        pdm_path: str,
        output_dir: Optional[str] = None
    ) -> Path:
        """Exportar resultados a archivo JSON"""
        
        if output_dir is None:
            output_dir = Path.cwd() / "evaluation_results"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre de archivo
        pdm_name = Path(pdm_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{pdm_name}_evaluation_{timestamp}.json"
        output_path = output_dir / output_filename
        
        # Exportar
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        return output_path


# ============================================================================
# FUNCIÓN DE UTILIDAD PARA USO RÁPIDO
# ============================================================================

def evaluate_pdm(
    pdm_path: str,
    municipality: str = "",
    department: str = "",
    config_path: Optional[str] = None,
    export_results: bool = True
) -> Dict[str, Any]:
    """
    Función de conveniencia para evaluación rápida de PDM
    
    Args:
        pdm_path: Ruta al archivo del PDM
        municipality: Nombre del municipio
        department: Nombre del departamento
        config_path: Ruta opcional a configuración
        export_results: Si exportar resultados a JSON
        
    Returns:
        Diccionario con resultados completos
        
    Ejemplo:
        >>> results = evaluate_pdm("plan_anori_2024-2027.txt", "Anorí", "Antioquia")
        >>> print(f"Score: {results['executive_summary']['combined_score']}")
    """
    system = IntegratedEvaluationSystem(config_path)
    return system.evaluate_pdm_complete(
        pdm_path=pdm_path,
        municipality=municipality,
        department=department,
        export_json=export_results
    )


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python integrated_evaluation_system.py <ruta_pdm> [municipio] [departamento]")
        print("\nEjemplo:")
        print("  python integrated_evaluation_system.py plan_anori.txt Anorí Antioquia")
        sys.exit(1)
    
    pdm_path = sys.argv[1]
    municipality = sys.argv[2] if len(sys.argv) > 2 else ""
    department = sys.argv[3] if len(sys.argv) > 3 else ""
    
    print("\n" + "="*80)
    print("SISTEMA DE EVALUACIÓN INTEGRADO - MINIMINIMOON + QUESTIONNAIRE ENGINE")
    print("="*80 + "\n")
    
    results = evaluate_pdm(
        pdm_path=pdm_path,
        municipality=municipality,
        department=department,
        export_results=True
    )
    
    # Mostrar resumen
    summary = results.get("executive_summary", {})
    print("\n" + "="*80)
    print("RESUMEN EJECUTIVO")
    print("="*80)
    print(f"Score Combinado: {summary.get('combined_score', 0):.1f}%")
    print(f"Clasificación: {summary.get('classification', 'N/A')}")
    print(f"\nRecomendación:")
    print(f"  {summary.get('recommendation', 'N/A')}")
    print(f"\nAcciones Prioritarias:")
    for action in summary.get('priority_actions', []):
        print(f"  • {action}")
    print("="*80 + "\n")
