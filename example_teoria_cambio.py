#!/usr/bin/env python3
"""
Ejemplo de uso de la validación de Teoría de Cambio
Demuestra las capacidades de validación de orden causal, detección de caminos completos
y generación de sugerencias para completar la cadena causal.
"""

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available. Creating mock implementation...")

if not NETWORKX_AVAILABLE:
    # Mock implementation for NetworkX DiGraph
    class MockDiGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = {}

        def add_node(self, node, **attrs):
            self._nodes[node] = attrs

        def add_edge(self, from_node, to_node):
            if from_node not in self._edges:
                self._edges[from_node] = []
            self._edges[from_node].append(to_node)

        def nodes(self):
            return self._nodes.keys()

        def successors(self, node):
            return self._edges.get(node, [])

        def has_path(self, source, target):
            if source == target:
                return True
            visited = set()
            stack = [source]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                if current == target:
                    return True

                for neighbor in self.successors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)
            return False

        def shortest_path(self, source, target):
            if not self.has_path(source, target):
                raise ValueError(f"No path from {source} to {target}")

            visited = set()
            queue = [(source, [source])]

            while queue:
                current, path = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                if current == target:
                    return path

                for neighbor in self.successors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

            raise ValueError(f"No path from {source} to {target}")

        def has_edge(self, from_node, to_node):
            return to_node in self._edges.get(from_node, [])

    # Replace networkx with mock
    class MockNetworkX:
        DiGraph = MockDiGraph

        @staticmethod
        def has_path(graph, source, target):
            return graph.has_path(source, target)

        @staticmethod
        def shortest_path(graph, source, target):
            return graph.shortest_path(source, target)

    nx = MockNetworkX()

from teoria_cambio import CategoriaCausal, TeoriaCambio


def ejemplo_teoria_cambio_completa():
    """Ejemplo de una teoría de cambio bien estructurada"""
    print("=== EJEMPLO: Teoría de Cambio Completa ===")

    tc = TeoriaCambio()

    # Crear grafo con estructura causal correcta
    grafo = nx.DiGraph()

    # INSUMOS
    grafo.add_node("recursos_financieros", tipo="insumo")
    grafo.add_node("equipo_tecnico", tipo="insumo")

    # PROCESOS
    grafo.add_node("capacitacion_docentes", tipo="proceso")
    grafo.add_node("desarrollo_materiales", tipo="proceso")

    # PRODUCTOS
    grafo.add_node("manual_pedagogico", tipo="producto")
    grafo.add_node("docentes_capacitados", tipo="producto")

    # RESULTADOS
    grafo.add_node("mejora_ensenanza", tipo="resultado")
    grafo.add_node("mayor_participacion", tipo="resultado")

    # IMPACTOS
    grafo.add_node("aumento_rendimiento_academico", tipo="impacto")

    # Crear conexiones causales válidas
    grafo.add_edge("recursos_financieros", "capacitacion_docentes")
    grafo.add_edge("equipo_tecnico", "desarrollo_materiales")
    grafo.add_edge("capacitacion_docentes", "docentes_capacitados")
    grafo.add_edge("desarrollo_materiales", "manual_pedagogico")
    grafo.add_edge("docentes_capacitados", "mejora_ensenanza")
    grafo.add_edge("manual_pedagogico", "mejora_ensenanza")
    grafo.add_edge("mejora_ensenanza", "aumento_rendimiento_academico")

    # Ejecutar validación completa
    resultado = tc.validacion_completa(grafo)

    print(f"Teoría de cambio válida: {resultado.es_valida}")
    print(f"Violaciones de orden: {len(resultado.violaciones_orden)}")
    print(f"Caminos completos encontrados: {len(resultado.caminos_completos)}")
    print(
        f"Categorías faltantes: {[cat.name for cat in resultado.categorias_faltantes]}"
    )

    if resultado.caminos_completos:
        print("Primer camino completo:", " → ".join(
            resultado.caminos_completos[0]))

    print()


def ejemplo_teoria_cambio_problematica():
    """Ejemplo de una teoría de cambio con problemas de validación"""
    print("=== EJEMPLO: Teoría de Cambio con Problemas ===")

    tc = TeoriaCambio()

    # Crear grafo con problemas de validación
    grafo = nx.DiGraph()

    # Solo algunas categorías
    grafo.add_node("recursos", tipo="insumo")
    grafo.add_node("impacto_educativo", tipo="impacto")
    grafo.add_node("producto_intermedio", tipo="producto")

    # Conexión que viola el orden causal (INSUMOS → IMPACTOS directamente)
    grafo.add_edge("recursos", "impacto_educativo")
    grafo.add_edge("recursos", "producto_intermedio")

    # Ejecutar validación completa
    resultado = tc.validacion_completa(grafo)

    print(f"Teoría de cambio válida: {resultado.es_valida}")
    print(f"Violaciones de orden: {len(resultado.violaciones_orden)}")

    if resultado.violaciones_orden:
        for violacion in resultado.violaciones_orden:
            print(
                f"  - Conexión inválida: {violacion['categoria_origen']} → {violacion['categoria_destino']}"
            )

    print(f"Caminos completos: {len(resultado.caminos_completos)}")
    print(
        f"Categorías faltantes: {[cat.name for cat in resultado.categorias_faltantes]}"
    )

    print("Sugerencias de mejora:")
    for i, sugerencia in enumerate(resultado.sugerencias, 1):
        print(f"  {i}. {sugerencia}")

    print()


def ejemplo_analisis_detallado():
    """Ejemplo de análisis detallado de una teoría de cambio"""
    print("=== EJEMPLO: Análisis Detallado ===")

    tc = TeoriaCambio()

    # Crear teoría de cambio parcialmente desarrollada
    grafo = nx.DiGraph()

    grafo.add_node("insumo_presupuesto", tipo="insumo")
    grafo.add_node("proceso_investigacion", tipo="proceso")
    grafo.add_node("resultado_conocimiento", tipo="resultado")
    # Falta PRODUCTOS e IMPACTOS

    grafo.add_edge("insumo_presupuesto", "proceso_investigacion")
    grafo.add_edge("proceso_investigacion", "resultado_conocimiento")
    # Falta conexión a IMPACTOS

    # Análisis por componentes
    print("--- Validación de Orden Causal ---")
    orden = tc.validar_orden_causal(grafo)
    print(f"Orden válido: {orden.es_valida}")

    print("\n--- Detección de Caminos Completos ---")
    caminos = tc.detectar_caminos_completos(grafo)
    print(f"Caminos completos: {caminos.es_valida}")
    print(f"Número de caminos: {len(caminos.caminos_completos)}")

    print("\n--- Generación de Sugerencias ---")
    sugerencias = tc.generar_sugerencias(grafo)
    print(
        f"Categorías faltantes: {[cat.name for cat in sugerencias.categorias_faltantes]}"
    )
    print("Recomendaciones:")
    for sugerencia in sugerencias.sugerencias:
        print(f"  • {sugerencia}")

    print()


def ejemplo_validacion_individual():
    """Ejemplo de validaciones individuales por categoría"""
    print("=== EJEMPLO: Validación por Categorías ===")

    tc = TeoriaCambio()
    grafo = nx.DiGraph()

    # Crear nodos con nombres descriptivos
    nodos_test = [
        ("fondos_insumos", CategoriaCausal.INSUMOS),
        ("talleres_proceso", CategoriaCausal.PROCESOS),
        ("certificados_producto", CategoriaCausal.PRODUCTOS),
        ("competencias_resultado", CategoriaCausal.RESULTADOS),
        ("desarrollo_impacto", CategoriaCausal.IMPACTOS),
    ]

    for nodo, categoria_esperada in nodos_test:
        grafo.add_node(nodo)
        categoria_detectada = tc._obtener_categoria_nodo(nodo, grafo)
        print(
            f"{nodo}: {categoria_detectada.name} {'✓' if categoria_detectada == categoria_esperada else '✗'}"
        )

    print("\n--- Test de Conexiones Válidas ---")
    conexiones_test = [
        (CategoriaCausal.INSUMOS, CategoriaCausal.PROCESOS, True),
        (CategoriaCausal.INSUMOS, CategoriaCausal.PRODUCTOS, True),  # Salto permitido
        (CategoriaCausal.INSUMOS, CategoriaCausal.IMPACTOS, False),  # Salto excesivo
        (CategoriaCausal.PROCESOS, CategoriaCausal.RESULTADOS, True),
        (CategoriaCausal.PRODUCTOS, CategoriaCausal.IMPACTOS, True),
    ]

    for origen, destino, esperado in conexiones_test:
        resultado = tc._es_conexion_valida(origen, destino)
        status = "✓" if resultado == esperado else "✗"
        print(f"{origen.name} → {destino.name}: {resultado} {status}")

    print()


if __name__ == "__main__":
    print("DEMOSTRACIÓN DEL SISTEMA DE VALIDACIÓN DE TEORÍA DE CAMBIO")
    print("=" * 60)
    print()

    ejemplo_teoria_cambio_completa()
    ejemplo_teoria_cambio_problematica()
    ejemplo_analisis_detallado()
    ejemplo_validacion_individual()

    print("Demostración completada. El sistema permite:")
    print("1. Validar orden causal INSUMOS→PROCESOS→PRODUCTOS→RESULTADOS→IMPACTOS")
    print("2. Detectar caminos causales completos")
    print("3. Generar sugerencias específicas para completar la teoría de cambio")
    print("4. Identificar categorías faltantes y conexiones problemáticas")
