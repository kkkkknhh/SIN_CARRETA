# coding=utf-8
"""
EXAMPLE USAGE — Diagnostic Runner
==================================
Demonstrates how to use diagnostic_runner.py to profile the 
MiniMinimoonOrchestrator pipeline with comprehensive instrumentation.
"""

import logging
from pathlib import Path
from diagnostic_runner import DiagnosticRunner, run_diagnostic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample plan text for testing
SAMPLE_PLAN = """
Programa de Desarrollo Industrial Sustentable 2024-2026

CONTEXTO:
El sector industrial de la región enfrenta desafíos de competitividad y 
sostenibilidad que requieren intervención coordinada.

OBJETIVO GENERAL:
Incrementar la productividad industrial en 15% y reducir emisiones en 20% 
mediante capacitación técnica, modernización de procesos y adopción de 
tecnologías limpias.

PRESUPUESTO:
- Total: $8,500,000 MXN
- Capacitación: $2,000,000 MXN
- Infraestructura: $4,500,000 MXN
- Tecnología limpia: $2,000,000 MXN

PLAZO DE EJECUCIÓN:
24 meses (enero 2024 - diciembre 2025)

RESPONSABLES:
- Secretaría de Economía (coordinación general)
- Cámara Nacional de la Industria de Transformación (CANACINTRA)
- Instituto Tecnológico Regional (capacitación)
- Secretaría de Medio Ambiente (certificación ambiental)

ACTIVIDADES PRINCIPALES:

1. DIAGNÓSTICO Y PLANEACIÓN (Meses 1-3)
   - Evaluación de capacidades industriales actuales
   - Identificación de brechas tecnológicas
   - Diseño de programa de capacitación técnica
   - Establecimiento de línea base de emisiones

2. CAPACITACIÓN TÉCNICA (Meses 4-18)
   - Cursos en gestión de calidad ISO 9001
   - Formación en producción más limpia
   - Certificación en mantenimiento preventivo
   - Talleres de eficiencia energética

3. MODERNIZACIÓN DE INFRAESTRUCTURA (Meses 6-20)
   - Mejora de conectividad en parques industriales
   - Instalación de sistemas de tratamiento de agua
   - Construcción de centros de acopio de residuos
   - Implementación de energías renovables

4. ADOPCIÓN TECNOLÓGICA (Meses 10-22)
   - Subsidios para adquisición de maquinaria eficiente
   - Implementación de sistemas de gestión ambiental
   - Digitalización de procesos productivos
   - Certificaciones ambientales

5. MONITOREO Y EVALUACIÓN (Meses 1-24)
   - Sistema de indicadores de desempeño
   - Evaluaciones trimestrales de avance
   - Medición de impacto ambiental
   - Ajustes correctivos basados en resultados

INDICADORES DE ÉXITO:
- Productividad industrial: incremento del 15%
- Reducción de emisiones: 20%
- Empresas capacitadas: 150 empresas
- Certificaciones ISO obtenidas: 30 empresas
- Empleos generados: 200 nuevos empleos

TEORÍA DEL CAMBIO:
Si aumentamos las capacidades técnicas del personal industrial Y 
modernizamos la infraestructura productiva Y promovemos tecnologías limpias,
ENTONCES aumentará la productividad Y se reducirán las emisiones,
PORQUE la eficiencia operativa y ambiental son mutuamente reforzantes
cuando se implementan de manera integrada.
"""


def example_basic_usage():
    """Example 1: Basic diagnostic run with convenience function."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Diagnostic Run")
    print("=" * 80)
    
    results = run_diagnostic(
        input_text=SAMPLE_PLAN,
        plan_id="example_basic",
        config_dir=Path("config"),
        output_dir=Path("diagnostic_output")
    )
    
    print("\n✓ Diagnostic completed")
    print(f"  Total stages: {len(results['stage_details'])}")
    print(f"  Total time: {results['diagnostic_metrics']['total_wall_time_ms']:.2f} ms")
    print(f"  Peak memory: {results['diagnostic_metrics']['peak_memory_mb']:.2f} MB")


def example_advanced_usage():
    """Example 2: Advanced usage with custom configuration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Advanced Diagnostic Run")
    print("=" * 80)
    
    # Create runner with custom config
    runner = DiagnosticRunner(
        config_dir=Path("config")
    )
    
    # Run diagnostics
    results = runner.run_with_diagnostics(
        input_text=SAMPLE_PLAN,
        plan_id="example_advanced",
        rubric_path=Path("config/rubrica_v3.json")
    )
    
    # Generate detailed report
    report = runner.generate_report()
    print(report)
    
    # Export metrics to JSON
    output_dir = Path("diagnostic_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.export_metrics_json(output_dir / "metrics_advanced.json")
    
    print("\n✓ Advanced diagnostic completed")
    print("  Report generated")
    print(f"  Metrics exported to: {output_dir / 'metrics_advanced.json'}")


def example_analyze_bottlenecks():
    """Example 3: Identify performance bottlenecks."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Bottleneck Analysis")
    print("=" * 80)
    
    runner = DiagnosticRunner(config_dir=Path("config"))
    results = runner.run_with_diagnostics(
        input_text=SAMPLE_PLAN,
        plan_id="example_bottleneck"
    )
    
    # Analyze stage timings
    stage_details = results['stage_details']
    
    # Sort stages by wall time
    sorted_stages = sorted(
        stage_details.items(),
        key=lambda x: x[1]['wall_time_ms'],
        reverse=True
    )
    
    print("\nTop 5 Slowest Stages:")
    print("-" * 80)
    total_time = results['diagnostic_metrics']['total_wall_time_ms']
    
    for i, (stage_name, metrics) in enumerate(sorted_stages[:5], 1):
        wall_time = metrics['wall_time_ms']
        percentage = (wall_time / total_time) * 100 if total_time > 0 else 0
        
        print(f"{i}. {stage_name}")
        print(f"   Wall Time: {wall_time:>10.2f} ms ({percentage:>5.1f}%)")
        print(f"   CPU Time:  {metrics['cpu_time_ms']:>10.2f} ms")
        print(f"   Memory:    {metrics['peak_memory_mb']:>10.2f} MB")
        print(f"   I/O Wait:  {metrics['io_wait_ms']:>10.2f} ms")
        print()


def example_contract_validation():
    """Example 4: Contract validation analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Contract Validation")
    print("=" * 80)
    
    runner = DiagnosticRunner(config_dir=Path("config"))
    results = runner.run_with_diagnostics(
        input_text=SAMPLE_PLAN,
        plan_id="example_contracts"
    )
    
    # Check for contract violations
    metrics = results['diagnostic_metrics']
    violations = metrics['contract_violations']
    
    print("\nContract Validation Summary:")
    print(f"  Total Stages: {metrics['stages_passed'] + metrics['stages_failed']}")
    print(f"  Stages Passed: {metrics['stages_passed']}")
    print(f"  Contract Violations: {violations}")
    
    if violations > 0:
        print("\n⚠ Contract Violations Detected:")
        stage_details = results['stage_details']
        for stage_name, stage_metrics in stage_details.items():
            if not stage_metrics['contract_valid']:
                print(f"\n  Stage: {stage_name}")
                for error in stage_metrics['contract_errors']:
                    print(f"    - {error}")
    else:
        print("\n✓ All contract validations passed")


def example_compare_runs():
    """Example 5: Compare multiple runs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Compare Multiple Runs")
    print("=" * 80)
    
    # Run diagnostics multiple times
    runner = DiagnosticRunner(config_dir=Path("config"))
    
    runs = []
    for i in range(3):
        print(f"\nExecuting run {i+1}/3...")
        results = runner.run_with_diagnostics(
            input_text=SAMPLE_PLAN,
            plan_id=f"compare_run_{i+1}"
        )
        runs.append(results['diagnostic_metrics'])
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    metrics_to_compare = [
        ('total_wall_time_ms', 'Total Wall Time (ms)'),
        ('total_cpu_time_ms', 'Total CPU Time (ms)'),
        ('peak_memory_mb', 'Peak Memory (MB)'),
    ]
    
    for metric_key, metric_label in metrics_to_compare:
        values = [run[metric_key] for run in runs]
        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        
        print(f"\n{metric_label}:")
        print(f"  Average: {avg:>10.2f}")
        print(f"  Min:     {min_val:>10.2f}")
        print(f"  Max:     {max_val:>10.2f}")
        print(f"  Range:   {max_val - min_val:>10.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DIAGNOSTIC RUNNER EXAMPLES")
    print("=" * 80)
    
    try:
        # Run all examples
        # Note: These will fail if orchestrator dependencies are not available
        # example_basic_usage()
        # example_advanced_usage()
        # example_analyze_bottlenecks()
        # example_contract_validation()
        # example_compare_runs()
        
        print("\n" + "=" * 80)
        print("EXAMPLES OVERVIEW (not executed)")
        print("=" * 80)
        print("\nAvailable examples:")
        print("  1. example_basic_usage()        - Simple diagnostic run")
        print("  2. example_advanced_usage()     - Custom configuration")
        print("  3. example_analyze_bottlenecks() - Performance analysis")
        print("  4. example_contract_validation() - Contract checking")
        print("  5. example_compare_runs()       - Multiple run comparison")
        print("\nUncomment function calls to run examples.")
        
    except Exception as e:
        print(f"\n⚠ Example failed: {e}")
        print("This is expected if orchestrator dependencies are not available.")
