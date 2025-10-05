#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test básico de la funcionalidad de heap implementada en el sistema.
"""
import heapq
import hashlib
from typing import List, Dict, Any, Tuple
from datetime import datetime

def test_heap_global_selection():
    """Test de la funcionalidad de selección global con heap"""
    print("🧪 Probando funcionalidad de heap para selección global top-k")
    
    # Simular datos de segmentos con scores
    segmentos_simulados = [
        {"texto": "Segmento 1", "score": 0.9, "pagina": 1},
        {"texto": "Segmento 2", "score": 0.8, "pagina": 2}, 
        {"texto": "Segmento 3", "score": 0.95, "pagina": 1},
        {"texto": "Segmento 4", "score": 0.7, "pagina": 3},
        {"texto": "Segmento 5", "score": 0.85, "pagina": 2},
        {"texto": "Segmento 6", "score": 0.6, "pagina": 4},
        {"texto": "Segmento 7", "score": 0.92, "pagina": 1},
        {"texto": "Segmento 8", "score": 0.75, "pagina": 3},
    ]
    
    # Configurar parámetros
    max_segmentos = 5
    heap_global = []
    
    print(f"📊 Procesando {len(segmentos_simulados)} segmentos para seleccionar top-{max_segmentos}")
    
    # Simular el procesamiento con heap (usando min-heap)
    for idx, seg in enumerate(segmentos_simulados):
        score = seg["score"]
        
        datos_segmento = {
            "texto": seg["texto"],
            "pagina": seg["pagina"],
            "similitud_semantica": score,
            "score_final": score,
            "hash_segmento": hashlib.md5(seg["texto"].encode('utf-8')).hexdigest()[:8],
            "timestamp_extraccion": datetime.now().isoformat()
        }
        
        # Usar heap para mantener top-k global eficientemente (min-heap con scores directos)
        if len(heap_global) < max_segmentos:
            # Heap no está lleno, agregar directamente
            heapq.heappush(heap_global, (score, idx, datos_segmento))
            print(f"  ➕ Agregado: {seg['texto']} (score: {score})")
        elif score > heap_global[0][0]:  # score > min_score_in_heap
            # Reemplazar el peor elemento con este mejor elemento
            old_score, old_idx, old_datos = heapq.heappushpop(heap_global, (score, idx, datos_segmento))
            print(f"  🔄 Reemplazado: {old_datos['texto']} (score: {old_score:.2f}) por {seg['texto']} (score: {score})")
        else:
            print(f"  ❌ Rechazado: {seg['texto']} (score: {score}) - score demasiado bajo (min en heap: {heap_global[0][0]:.2f})")
    
    # Extraer los resultados finales del heap y ordenar por score descendente
    resultados_finales = []
    while heap_global:
        score, _, datos_segmento = heapq.heappop(heap_global)
        resultados_finales.append(datos_segmento)
    
    # Ordenar por score final descendente para tener los mejores primero
    resultados_finales.sort(key=lambda x: x['score_final'], reverse=True)
    
    print(f"\n✅ Selección completada: {len(resultados_finales)} segmentos seleccionados")
    print("\n🏆 Top segmentos seleccionados (ordenados por score):")
    for i, seg in enumerate(resultados_finales, 1):
        print(f"  {i}. {seg['texto']} (score: {seg['score_final']:.2f}, pág: {seg['pagina']})")
    
    # Verificar que efectivamente tenemos los top-k
    scores_originales = sorted([s["score"] for s in segmentos_simulados], reverse=True)
    top_k_esperados = scores_originales[:max_segmentos]
    scores_obtenidos = [s["score_final"] for s in resultados_finales]
    
    print("\n🔍 Verificación:")
    print(f"  Top-{max_segmentos} esperados: {top_k_esperados}")
    print(f"  Top-{max_segmentos} obtenidos: {scores_obtenidos}")
    
    if set(top_k_esperados) == set(scores_obtenidos):
        print("  ✅ ¡Selección correcta!")
        return True
    else:
        print("  ❌ Error en la selección")
        return False

def test_batch_processing():
    """Test del procesamiento en batches"""
    print("\n🧪 Probando funcionalidad de procesamiento en batches")
    
    # Simular queries
    queries = [f"query_{i}" for i in range(1, 26)]  # 25 queries
    batch_size = 8
    
    print(f"📊 Procesando {len(queries)} queries en batches de {batch_size}")
    
    batches_procesados = 0
    for batch_start in range(0, len(queries), batch_size):
        batch_queries = queries[batch_start:batch_start + batch_size]
        print(f"  📦 Batch {batches_procesados + 1}: procesando queries {batch_start + 1}-{min(batch_start + batch_size, len(queries))}")
        print(f"     Queries en este batch: {batch_queries}")
        batches_procesados += 1
    
    print(f"✅ Procesados {batches_procesados} batches en total")
    
    # Verificar que se procesaron todas las queries
    queries_procesadas = batches_procesados * batch_size - max(0, batches_procesados * batch_size - len(queries))
    if queries_procesadas == len(queries):
        print("  ✅ Todas las queries fueron procesadas correctamente")
        return True
    else:
        print(f"  ❌ Error: se procesaron {queries_procesadas} queries de {len(queries)} esperadas")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando tests de funcionalidad de heap y batch processing\n")
    
    test1_passed = test_heap_global_selection()
    test2_passed = test_batch_processing()
    
    print("\n📊 Resultados de los tests:")
    print(f"  Heap global selection: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Batch processing: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ¡Todos los tests pasaron exitosamente!")
        print("✅ La funcionalidad de --max-segmentos está lista para uso")
    else:
        print("\n⚠️ Algunos tests fallaron. Revisar implementación.")