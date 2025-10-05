#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo simplificado para mostrar el funcionamiento del manejo de señales
sin dependencias pesadas como spaCy, sentence-transformers, etc.
"""

import atexit
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from log_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


# ==================== SISTEMA DE MONITOREO SIMPLIFICADO ====================
class SistemaMonitoreoDemo:
    """Versión demo del sistema de monitoreo para testing"""

    def __init__(self):
        self.logger = LOGGER
        self.ejecuciones = []
        self.tiempo_inicio = None
        self.trabajadores_activos = set()
        self.lock = threading.RLock()
        self.interrumpido = False

    def iniciar_monitoreo(self):
        self.tiempo_inicio = datetime.now()
        self.logger.info("🚀 Sistema de monitoreo iniciado: %s", self.tiempo_inicio)

    def registrar_trabajador(self, trabajador_id: str):
        with self.lock:
            self.trabajadores_activos.add(trabajador_id)
            self.logger.info("➕ Trabajador registrado: %s", trabajador_id)

    def desregistrar_trabajador(self, trabajador_id: str):
        with self.lock:
            self.trabajadores_activos.discard(trabajador_id)
            self.logger.info("➖ Trabajador desregistrado: %s", trabajador_id)

    def terminar_trabajadores(self):
        with self.lock:
            self.interrumpido = True
            trabajadores_copia = self.trabajadores_activos.copy()
        self.logger.warning("🛑 Terminando %s trabajadores...", len(trabajadores_copia))

    def registrar_ejecucion(self, nombre: str, resultado: dict):
        with self.lock:
            ejecucion = {
                "nombre": nombre,
                "resultado": resultado,
                "timestamp": datetime.now().isoformat(),
            }
            self.ejecuciones.append(ejecucion)
            self.logger.info(
                "📝 Ejecución registrada: %s - %s",
                nombre,
                resultado.get("status", "unknown"),
            )

    def generar_dump_emergencia(self, output_dir: Path) -> Path:
        with self.lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dump_path = output_dir / f"dump_emergencia_demo_{timestamp}.json"

            estado = {
                "timestamp_dump": datetime.now().isoformat(),
                "sistema_interrumpido": self.interrumpido,
                "trabajadores_activos": list(self.trabajadores_activos),
                "tiempo_inicio": (
                    self.tiempo_inicio.isoformat() if self.tiempo_inicio else None
                ),
                "total_ejecuciones": len(self.ejecuciones),
                "ejecuciones": self.ejecuciones,
            }

            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(estado, f, indent=2, ensure_ascii=False)

            self.logger.info("💾 Dump de emergencia guardado: %s", dump_path)
            return dump_path


# ==================== MANEJO DE SEÑALES ====================
_sistema_monitoreo_global = None
_output_dir_global = None
_signal_handler_lock = threading.RLock()


def signal_handler(signum, frame):
    """Manejador de señales demo"""
    global _sistema_monitoreo_global, _output_dir_global

    with _signal_handler_lock:
        LOGGER.warning(
            "🚨 SEÑAL %s RECIBIDA - Iniciando terminación graciosa...", signum
        )

        if _sistema_monitoreo_global:
            try:
                _sistema_monitoreo_global.terminar_trabajadores()

                if _output_dir_global:
                    dump_path = _sistema_monitoreo_global.generar_dump_emergencia(
                        _output_dir_global
                    )
                    LOGGER.info("📊 Estado guardado en: %s", dump_path)

            except Exception:  # pragma: no cover - defensive signal logging
                LOGGER.exception("❌ Error durante terminación")

        LOGGER.info("🛑 Terminación graciosa completada")
        sys.exit(1)


def atexit_handler():
    """Handler para terminación inesperada"""
    global _sistema_monitoreo_global, _output_dir_global

    with _signal_handler_lock:
        LOGGER.warning("🚨 Terminación inesperada detectada...")

        if _sistema_monitoreo_global and _output_dir_global:
            try:
                dump_path = _sistema_monitoreo_global.generar_dump_emergencia(
                    _output_dir_global
                )
                LOGGER.info("💾 Estado de emergencia guardado: %s", dump_path)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception("❌ Error en atexit handler")


def simular_trabajo(trabajador_id: str, duracion: int):
    """Simula trabajo de procesamiento"""
    global _sistema_monitoreo_global

    if _sistema_monitoreo_global:
        _sistema_monitoreo_global.registrar_trabajador(trabajador_id)

    try:
        LOGGER.info("🔄 %s iniciando trabajo por %ss...", trabajador_id, duracion)

        for i in range(duracion):
            time.sleep(1)
            if _sistema_monitoreo_global and _sistema_monitoreo_global.interrumpido:
                LOGGER.warning(
                    "⚠️  %s detectó interrupción, terminando...", trabajador_id
                )
                break
            LOGGER.info("🔄 %s trabajando... (%s/%s)", trabajador_id, i + 1, duracion)

        # Simular resultado
        resultado = {
            "status": (
                "completed"
                if not (
                    _sistema_monitoreo_global and _sistema_monitoreo_global.interrumpido
                )
                else "interrupted"
            ),
            "puntaje": (
                85.5
                if not (
                    _sistema_monitoreo_global and _sistema_monitoreo_global.interrumpido
                )
                else 0
            ),
        }

        if _sistema_monitoreo_global:
            _sistema_monitoreo_global.registrar_ejecucion(
                f"plan_{trabajador_id}", resultado
            )

        LOGGER.info("✅ %s completado", trabajador_id)

    except Exception as error:
        LOGGER.exception("❌ Error en %s", trabajador_id)
        if _sistema_monitoreo_global:
            _sistema_monitoreo_global.registrar_ejecucion(
                f"plan_{trabajador_id}",
                {"status": "failed", "error": str(error)},
            )
    finally:
        if _sistema_monitoreo_global:
            _sistema_monitoreo_global.desregistrar_trabajador(trabajador_id)


def main():
    """Función principal demo"""
    global _sistema_monitoreo_global, _output_dir_global

    LOGGER.info("🏭 DEMO: Sistema de Manejo de Señales")
    LOGGER.info("%s", "=" * 50)
    LOGGER.info("Presiona Ctrl+C para probar la terminación graciosa")
    LOGGER.info("%s", "=" * 50)

    # Configurar output
    output_dir = Path("demo_resultados")
    output_dir.mkdir(exist_ok=True)
    _output_dir_global = output_dir

    # Inicializar monitoreo
    sistema = SistemaMonitoreoDemo()
    _sistema_monitoreo_global = sistema
    sistema.iniciar_monitoreo()

    # Configurar handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(atexit_handler)

    # Simular procesamiento con threads
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=simular_trabajo, args=(f"worker_{i + 1}", 10), daemon=True
        )
        threads.append(thread)
        thread.start()

    try:
        # Esperar a que terminen los threads
        for thread in threads:
            thread.join()

        LOGGER.info("✅ Procesamiento completado normalmente")

    except KeyboardInterrupt:
        # El signal handler se encarga de esto
        pass

    LOGGER.info("📊 Resultados disponibles en: %s", output_dir)


if __name__ == "__main__":
    main()
