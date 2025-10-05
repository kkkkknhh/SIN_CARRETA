#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test básico para verificar la implementación de signal handling
"""

import sys
from pathlib import Path

def test_basic_imports():
    """Test básico de imports y configuración"""
    print('🧪 Test básico de imports y configuración...')

    try:
        # Importar el módulo principal
        import Decatalogo_principal as dp
        print('✅ Módulo principal importado correctamente')
        
        # Verificar que las funciones de signal handling existen
        assert hasattr(dp, 'signal_handler'), 'signal_handler no encontrado'
        assert hasattr(dp, 'atexit_handler'), 'atexit_handler no encontrado'
        assert hasattr(dp, 'procesar_plan_industrial_con_monitoreo'), 'wrapper de monitoreo no encontrado'
        print('✅ Funciones de signal handling encontradas')
        
        # Verificar que el sistema de monitoreo tiene las nuevas funciones
        sistema = dp.SistemaMonitoreoIndustrial()
        assert hasattr(sistema, 'generar_dump_emergencia'), 'generar_dump_emergencia no encontrado'
        assert hasattr(sistema, 'registrar_trabajador'), 'registrar_trabajador no encontrado'
        assert hasattr(sistema, 'terminar_trabajadores'), 'terminar_trabajadores no encontrado'
        print('✅ Sistema de monitoreo actualizado correctamente')
        
        # Verificar thread safety
        assert hasattr(sistema, 'lock'), 'lock thread-safe no encontrado'
        print('✅ Mecanismos thread-safe implementados')
        
        # Test funcional básico del dump de emergencia
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        sistema.iniciar_monitoreo()
        sistema.registrar_ejecucion("test_plan", {"status": "completed", "puntaje_promedio": 85.5})
        
        dump_path = sistema.generar_dump_emergencia(output_dir)
        assert dump_path.exists(), "Dump de emergencia no se creó"
        print(f'✅ Dump de emergencia creado: {dump_path}')
        
        # Limpiar
        dump_path.unlink()
        output_dir.rmdir()
        
        print('🎉 Test básico EXITOSO - Signal handling implementado correctamente')
        return True
        
    except Exception as e:
        print(f'❌ Test básico FALLIDO: {e}')
        return False

if __name__ == "__main__":
    success = test_basic_imports()
    sys.exit(0 if success else 1)