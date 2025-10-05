#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script para verificar el manejo de señales y terminación graciosa
del Sistema Industrial de Evaluación de Políticas Públicas.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def create_test_pdf_content():
    """Crea contenido de prueba que simule un plan de desarrollo"""
    return """
PLAN DE DESARROLLO MUNICIPAL DE PRUEBA

1. PREVENCIÓN DE LA VIOLENCIA Y PROTECCIÓN
Este plan incluye estrategias para:
- Reducir los índices de violencia
- Fortalecer la seguridad ciudadana
- Proteger a defensores de derechos humanos

2. RECURSOS Y CAPACIDADES
- Presupuesto asignado: $1,000,000
- Personal capacitado: 50 funcionarios
- Infraestructura disponible

3. INDICADORES DE IMPACTO
- Reducción del 20% en homicidios
- Aumento del 15% en percepción de seguridad
- Implementación de 5 sistemas de alerta temprana

4. TEORÍA DE CAMBIO
Si invertimos en seguridad y prevención, entonces
reduciremos la violencia y mejoraremos la calidad de vida.

5. CADENA DE VALOR
Insumos → Procesos → Productos → Resultados → Impactos

6. EQUIDAD DE GÉNERO
Políticas inclusivas para garantizar participación femenina
en todos los niveles del desarrollo municipal.

7. DESARROLLO ECONÓMICO
Fomento de emprendimientos locales y generación de empleo
digno para toda la población.

8. SOSTENIBILIDAD AMBIENTAL
Protección de recursos naturales y mitigación del cambio
climático a través de políticas ambientales.

9. INFRAESTRUCTURA
Mejoramiento de vías, conectividad digital y servicios
públicos básicos para toda la comunidad.

10. GOBERNANZA
Fortalecimiento de la participación ciudadana y
transparencia en la gestión pública municipal.
""".strip()


def test_signal_handling():
    """Test principal para manejo de señales"""
    print("🧪 Iniciando test de manejo de señales...")

    # Crear directorio temporal
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Crear algunos archivos de prueba (como texto plano que simule PDFs)
        for i in range(3):
            content = create_test_pdf_content()
            # Por simplicidad en el test, creamos archivos .txt que contengan el contenido
            # En un entorno real tendrían que ser PDFs válidos
            txt_path = temp_path / f"plan_desarrollo_test_{i + 1}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)

        print("⚠️  NOTA: Este test usa archivos .txt en lugar de .pdf para simplicidad")

        print(f"📁 Archivos de prueba creados en: {temp_path}")
        print(f"📄 Archivos: {list(temp_path.glob('*.txt'))}")
        print("ℹ️  Para test real con PDFs, use archivos PDF válidos")

        # Ejecutar el programa principal en un subprocess
        cmd = [sys.executable, "Decatalogo_principal.py", str(temp_path)]

        print(f"🚀 Ejecutando comando: {' '.join(cmd)}")

        try:
            # Iniciar el proceso
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
            )

            print(f"📊 Proceso iniciado con PID: {process.pid}")

            # Esperar un momento para que inicie el procesamiento
            time.sleep(5)

            # Enviar SIGINT (Ctrl+C)
            print("🚨 Enviando SIGINT para probar terminación graciosa...")
            process.send_signal(signal.SIGINT)

            # Esperar a que termine
            stdout, stderr = process.communicate(timeout=30)

            print("📤 STDOUT:")
            print(stdout)
            print("\n📤 STDERR:")
            print(stderr)

            # Verificar que se crearon archivos de dump de emergencia
            output_dir = Path("resultados_evaluacion_industrial")
            if output_dir.exists():
                dump_files = list(output_dir.glob(
                    "dump_emergencia_monitoreo_*.json"))

                if dump_files:
                    print(f"✅ Dump de emergencia encontrado: {dump_files}")

                    # Verificar contenido del dump
                    dump_path = dump_files[0]
                    with open(dump_path, "r", encoding="utf-8") as f:
                        dump_data = json.load(f)

                    print("📊 Contenido del dump de emergencia:")
                    print(
                        f"  - Sistema interrumpido: {dump_data.get('sistema_interrumpido', 'N/A')}"
                    )
                    print(
                        f"  - Ejecuciones completadas: {dump_data.get('ejecuciones_completadas', 0)}"
                    )
                    print(
                        f"  - Ejecuciones fallidas: {dump_data.get('ejecuciones_fallidas', 0)}"
                    )
                    print(
                        f"  - Trabajadores activos: {len(dump_data.get('trabajadores_activos', []))}"
                    )

                    if "estadisticas_parciales" in dump_data:
                        stats = dump_data["estadisticas_parciales"]
                        if "mensaje" not in stats:
                            print(
                                f"  - Puntaje promedio parcial: {stats.get('puntaje_promedio', 'N/A'):.1f}"
                            )
                            print(
                                f"  - Planes completados: {stats.get('total_completados', 0)}"
                            )

                    print("✅ Test de manejo de señales EXITOSO")
                    return True
                else:
                    print("❌ No se encontró dump de emergencia")
                    return False
            else:
                print("❌ No se creó directorio de resultados")
                return False

        except subprocess.TimeoutExpired:
            print("⏰ Timeout - Terminando proceso...")
            process.kill()
            return False
        except Exception as e:
            print(f"❌ Error durante test: {e}")
            return False


def test_atexit_handling():
    """Test para verificar el handler atexit"""
    print("\n🧪 Iniciando test de handler atexit...")

    # Crear un script que termine inesperadamente
    test_script = """
import sys
import os
sys.path.insert(0, os.getcwd())

from Decatalogo_principal import SistemaMonitoreoIndustrial, _sistema_monitoreo_global, _output_dir_global, atexit_handler
from pathlib import Path
import atexit

# Simular configuración global
_output_dir_global = Path("resultados_evaluacion_industrial")
_output_dir_global.mkdir(exist_ok=True)

sistema = SistemaMonitoreoIndustrial()
sistema.iniciar_monitoreo()

# Registrar algunas ejecuciones simuladas
sistema.registrar_ejecucion("plan_test_1", {"status": "completed", "puntaje_promedio": 85.5})
sistema.registrar_ejecucion("plan_test_2", {"status": "failed", "error": "Error simulado"})

# Configurar variable global
import Decatalogo_principal
Decatalogo_principal._sistema_monitoreo_global = sistema

# Registrar atexit handler
atexit.register(atexit_handler)

# Terminar inesperadamente
print("Simulando terminación inesperada...")
sys.exit(0)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        script_path = f.name

    try:
        # Ejecutar script
        result = subprocess.run(
            [sys.executable, script_path], capture_output=True, text=True, timeout=30, 
        check=True)

        print(f"📤 Output del test atexit: {result.stdout}")
        print(f"📤 Errors del test atexit: {result.stderr}")

        # Verificar que se creó dump de emergencia
        output_dir = Path("resultados_evaluacion_industrial")
        dump_files = list(output_dir.glob("dump_emergencia_monitoreo_*.json"))

        if dump_files:
            print("✅ Test de handler atexit EXITOSO")
            return True
        else:
            print("❌ No se encontró dump de emergencia del test atexit")
            return False

    except Exception as e:
        print(f"❌ Error en test atexit: {e}")
        return False
    finally:
        # Limpiar
        if os.path.exists(script_path):
            os.unlink(script_path)


if __name__ == "__main__":
    print("🏭 Test de Sistema de Manejo de Señales - Evaluación de Políticas Públicas")
    print("=" * 80)

    # Este test no requiere dependencias especiales ya que usa archivos de texto

    # Ejecutar tests
    signal_test_ok = test_signal_handling()
    atexit_test_ok = test_atexit_handling()

    print("\n" + "=" * 80)
    print("📊 RESULTADOS DE TESTS:")
    print(
        f"  🚨 Test manejo de señales: {'✅ EXITOSO' if signal_test_ok else '❌ FALLIDO'}"
    )
    print(
        f"  🔄 Test handler atexit: {'✅ EXITOSO' if atexit_test_ok else '❌ FALLIDO'}"
    )

    if signal_test_ok and atexit_test_ok:
        print(
            "\n🎉 TODOS LOS TESTS EXITOSOS - Sistema de manejo de señales funcionando correctamente"
        )
        sys.exit(0)
    else:
        print("\n❌ ALGUNOS TESTS FALLARON - Revisar implementación")
        sys.exit(1)
