#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo de la nueva funcionalidad --max-segmentos con selección global top-k usando heap
"""

def demo_max_segmentos():
    """Demo de cómo usar la nueva funcionalidad --max-segmentos"""
    print("🎯 DEMO: Nueva funcionalidad --max-segmentos")
    print("=" * 60)
    
    print("\n📖 ¿Qué hace --max-segmentos?")
    print("   • Limita el número total de segmentos de texto procesados")
    print("   • Usa un algoritmo de heap para selección global top-k")
    print("   • Procesa documentos en batches para optimizar memoria")
    print("   • Garantiza que los segmentos finales representan los mejores matches")
    print("   • Funciona a través de múltiples documentos, no solo dentro de cada uno")
    
    print("\n🚀 Ejemplos de uso:")
    print("   python Decatalogo_principal.py ./planes/")
    print("   python Decatalogo_principal.py ./planes/ --max-segmentos 1000")
    print("   python Decatalogo_principal.py ./plan.pdf --max-segmentos 500 --batch-size 64")
    
    print("\n⚙️  Cómo funciona internamente:")
    print("   1. Se generan queries semánticas basadas en las dimensiones del decálogo")
    print("   2. Se procesan en batches para optimizar el uso de memoria") 
    print("   3. Se usa un min-heap para mantener solo los top-k segmentos globalmente")
    print("   4. Se reemplazan dinámicamente segmentos con scores bajos por mejores")
    print("   5. Se ordenan los resultados finales por score descendente")
    
    print("\n💡 Beneficios:")
    print("   ✅ Reduce el uso de memoria al procesar corpus grandes")
    print("   ✅ Mejora la calidad del análisis al enfocar en los mejores segmentos")
    print("   ✅ Acelera el procesamiento al evitar análisis de segmentos irrelevantes")
    print("   ✅ Funciona de manera distribuida entre múltiples documentos")
    print("   ✅ Implementación eficiente O(n log k) usando estructuras de heap")
    
    print("\n🎛️ Parámetros:")
    print("   --max-segmentos: Número máximo de segmentos (default: sin límite)")
    print("   --batch-size: Tamaño de batch para embeddings (default: 32)")
    
    print("\n📊 Ejemplo de output:")
    print("   🔍 Aplicando selección global de top-1000 segmentos con batch_size=32")
    print("   ✅ Segmentos filtrados: 1000 de 5847 originales")
    print("   📊 SEGMENTOS PROCESADOS: 1000/5847")

if __name__ == "__main__":
    demo_max_segmentos()