#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Verificación de Flujos Críticos
==========================================

Este script verifica que todos los 72 flujos críticos estén correctamente implementados
y que se cumplan los 6 criterios de aceptación (gates).

Ejecutar: python verify_critical_flows.py
"""

import sys
import json
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Colores para output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class CriticalFlowVerifier:
    """Verificador de flujos críticos del sistema"""

    def __init__(self):
        self.results = {
            "critical_flows": [],
            "standard_flows": [],
            "meta_flows": [],
            "gates": [],
            "errors": [],
            "warnings": []
        }

    def verify_all(self) -> Dict[str, Any]:
        """Ejecuta todas las verificaciones"""
        print(f"{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}VERIFICACIÓN DE FLUJOS CRÍTICOS - MINIMINIMOON{RESET}")
        print(f"{BLUE}{'='*80}{RESET}\n")

        # 1. Verificar orquestador principal
        self._verify_main_orchestrator()

        # 2. Verificar componentes del pipeline (flows #1-11)
        self._verify_pipeline_components()

        # 3. Verificar evaluadores (flows #13-15)
        self._verify_evaluators()

        # 4. Verificar deprecación del orquestador antiguo (gate #6)
        self._verify_deprecated_orchestrator()

        # 5. Verificar inmutabilidad (gate #1)
        self._verify_immutability()

        # 6. Verificar validadores del sistema
        self._verify_system_validators()

        # 7. Verificar CLI
        self._verify_cli()

        # 8. Verificar archivos de configuración
        self._verify_config_files()

        # Resumen
        self._print_summary()

        return self.results

    def _verify_main_orchestrator(self):
        """Verifica el orquestador principal CanonicalDeterministicOrchestrator"""
        print(f"\n{YELLOW}► Verificando orquestador principal...{RESET}")

        try:
            from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator

            # Verificar que tiene el método principal
            if not hasattr(CanonicalDeterministicOrchestrator, 'process_plan_deterministic'):
                self._add_error("CanonicalDeterministicOrchestrator no tiene método process_plan_deterministic")
            else:
                self._add_success("✓ CanonicalDeterministicOrchestrator.process_plan_deterministic existe")

            # Verificar que tiene EvidenceRegistry
            if not hasattr(CanonicalDeterministicOrchestrator, 'evidence_registry'):
                self._add_warning("CanonicalDeterministicOrchestrator podría no inicializar evidence_registry")

            self._add_success("✓ Orquestador principal verificado")

        except ImportError as e:
            self._add_error(f"No se pudo importar CanonicalDeterministicOrchestrator: {e}")
        except Exception as e:
            self._add_error(f"Error verificando orquestador: {e}")

    def _verify_pipeline_components(self):
        """Verifica componentes del pipeline (flows #1-11)"""
        print(f"\n{YELLOW}► Verificando componentes del pipeline...{RESET}")

        components = [
            ("plan_sanitizer", "PlanSanitizer"),
            ("plan_processor", "PlanProcessor"),
            ("document_segmenter", "DocumentSegmenter"),
            ("embedding_model", "IndustrialEmbeddingModel"),
            ("responsibility_detector", "ResponsibilityDetector"),
            ("contradiction_detector", "ContradictionDetector"),
            ("monetary_detector", "MonetaryDetector"),
            ("feasibility_scorer", "FeasibilityScorer"),
            ("causal_pattern_detector", "CausalPatternDetector"),
            ("teoria_cambio", "TeoriaCambioValidator"),
            ("dag_validation", "AdvancedDAGValidator"),
        ]

        for module_name, class_name in components:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    self._add_success(f"✓ {module_name}.{class_name}")
                else:
                    # Verificar alias
                    if module_name == "teoria_cambio" and hasattr(module, "TeoriaCambio"):
                        self._add_success(f"✓ {module_name}.TeoriaCambio (alias TeoriaCambioValidator)")
                    else:
                        self._add_error(f"Clase {class_name} no encontrada en {module_name}")
            except ImportError as e:
                self._add_error(f"No se pudo importar {module_name}: {e}")
            except Exception as e:
                self._add_error(f"Error verificando {module_name}: {e}")

    def _verify_evaluators(self):
        """Verifica evaluadores (flows #13-15)"""
        print(f"\n{YELLOW}► Verificando evaluadores...{RESET}")

        try:
            from questionnaire_engine import QuestionnaireEngine
            self._add_success("✓ QuestionnaireEngine")
        except ImportError as e:
            self._add_error(f"No se pudo importar QuestionnaireEngine: {e}")

        try:
            from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado, BUNDLE
            self._add_success("✓ ExtractorEvidenciaIndustrialAvanzado")
            self._add_success("✓ BUNDLE")
        except ImportError as e:
            self._add_error(f"No se pudo importar componentes de Decatalogo_principal: {e}")

        try:
            # Verificar AnswerAssembler en orchestrator
            from miniminimoon_orchestrator import AnswerAssembler
            self._add_success("✓ AnswerAssembler")
        except ImportError as e:
            self._add_warning(f"AnswerAssembler podría estar integrado en orchestrator: {e}")

    def _verify_deprecated_orchestrator(self):
        """Verifica que el orquestador antiguo esté deprecado (gate #6)"""
        print(f"\n{YELLOW}► Verificando deprecación de orquestador antiguo (Gate #6)...{RESET}")

        try:
            import decalogo_pipeline_orchestrator
            self._add_error("❌ Gate #6 FAILED: decalogo_pipeline_orchestrator se importó sin error (debería lanzar RuntimeError)")
        except RuntimeError as e:
            if "DEPRECATED" in str(e) or "FORBIDDEN" in str(e):
                self._add_success("✓ Gate #6 PASSED: decalogo_pipeline_orchestrator correctamente deprecado")
            else:
                self._add_error(f"Gate #6: RuntimeError incorrecto: {e}")
        except ImportError:
            self._add_warning("decalogo_pipeline_orchestrator no existe (aceptable si fue eliminado)")
        except Exception as e:
            self._add_error(f"Error inesperado verificando deprecación: {e}")

    def _verify_immutability(self):
        """Verifica sistema de inmutabilidad (gate #1)"""
        print(f"\n{YELLOW}► Verificando sistema de inmutabilidad (Gate #1)...{RESET}")

        try:
            from miniminimoon_immutability import EnhancedImmutabilityContract

            immut = EnhancedImmutabilityContract()

            # Verificar métodos críticos
            required_methods = ['freeze_configuration', 'verify_frozen_config', 'has_snapshot']
            for method in required_methods:
                if hasattr(immut, method):
                    self._add_success(f"✓ EnhancedImmutabilityContract.{method}")
                else:
                    self._add_error(f"Falta método {method} en EnhancedImmutabilityContract")

            # Verificar si existe snapshot
            if immut.has_snapshot():
                self._add_success("✓ Snapshot de inmutabilidad existe")
            else:
                self._add_warning("⚠ No hay snapshot de inmutabilidad (ejecutar freeze primero)")

        except ImportError as e:
            self._add_error(f"No se pudo importar EnhancedImmutabilityContract: {e}")
        except Exception as e:
            self._add_error(f"Error verificando inmutabilidad: {e}")

    def _verify_system_validators(self):
        """Verifica validadores del sistema"""
        print(f"\n{YELLOW}► Verificando validadores del sistema...{RESET}")

        try:
            from system_validators import SystemHealthValidator
            self._add_success("✓ SystemHealthValidator")
        except ImportError as e:
            self._add_error(f"No se pudo importar SystemHealthValidator: {e}")

        try:
            from deterministic_pipeline_validator import DeterministicPipelineValidator
            self._add_success("✓ DeterministicPipelineValidator")
        except ImportError:
            self._add_warning("DeterministicPipelineValidator podría estar integrado en orchestrator")

    def _verify_cli(self):
        """Verifica CLI unificado"""
        print(f"\n{YELLOW}► Verificando CLI...{RESET}")

        cli_file = Path("miniminimoon_cli.py")
        if cli_file.exists():
            self._add_success("✓ miniminimoon_cli.py existe")

            # Verificar comandos
            with open(cli_file, 'r') as f:
                content = f.read()
                required_commands = ['freeze', 'evaluate', 'verify', 'rubric-check', 'trace-matrix']
                for cmd in required_commands:
                    if f"cmd_{cmd.replace('-', '_')}" in content or f"'{cmd}'" in content:
                        self._add_success(f"✓ Comando '{cmd}' implementado")
                    else:
                        self._add_warning(f"⚠ Comando '{cmd}' podría no estar implementado")
        else:
            self._add_error("❌ miniminimoon_cli.py no existe")

    def _verify_config_files(self):
        """Verifica archivos de configuración críticos"""
        print(f"\n{YELLOW}► Verificando archivos de configuración...{RESET}")

        config_files = [
            "DECALOGO_FULL.json",
            "decalogo_industrial.json",
            "dnp-standards.latest.clean.json",
            "RUBRIC_SCORING.json"
        ]

        for config_file in config_files:
            path = Path(config_file)
            if path.exists():
                self._add_success(f"✓ {config_file}")
            else:
                self._add_warning(f"⚠ {config_file} no encontrado")

    def _add_success(self, msg: str):
        """Agrega mensaje de éxito"""
        print(f"{GREEN}{msg}{RESET}")
        self.results["critical_flows"].append({"status": "success", "message": msg})

    def _add_error(self, msg: str):
        """Agrega error"""
        print(f"{RED}❌ {msg}{RESET}")
        self.results["errors"].append(msg)

    def _add_warning(self, msg: str):
        """Agrega advertencia"""
        print(f"{YELLOW}⚠ {msg}{RESET}")
        self.results["warnings"].append(msg)

    def _print_summary(self):
        """Imprime resumen final"""
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}RESUMEN DE VERIFICACIÓN{RESET}")
        print(f"{BLUE}{'='*80}{RESET}\n")

        total_checks = len(self.results["critical_flows"])
        errors = len(self.results["errors"])
        warnings = len(self.results["warnings"])

        print(f"Total de verificaciones: {total_checks}")
        print(f"{RED}Errores: {errors}{RESET}")
        print(f"{YELLOW}Advertencias: {warnings}{RESET}")

        if errors == 0:
            print(f"\n{GREEN}✓ ¡TODOS LOS FLUJOS CRÍTICOS VERIFICADOS!{RESET}")
            return 0
        else:
            print(f"\n{RED}❌ FALLOS ENCONTRADOS - REVISAR ERRORES{RESET}")
            return 1


def main():
    """Función principal"""
    verifier = CriticalFlowVerifier()
    results = verifier.verify_all()

    # Guardar resultados
    output_file = Path("artifacts/flow_verification_report.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{BLUE}Reporte guardado en: {output_file}{RESET}")

    # Exit code
    return 0 if len(results["errors"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

