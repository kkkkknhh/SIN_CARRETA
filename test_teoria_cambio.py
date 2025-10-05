import unittest

import networkx as nx

from teoria_cambio import CategoriaCausal, TeoriaCambio


class TestTeoriaCambioValidacion(unittest.TestCase):
    @staticmethod
    def test_validacion_orden_causal_valido():
        """Test de validación con orden causal correcto"""
        tc = TeoriaCambio()

        # Crear grafo con orden correcto
        grafo = nx.DiGraph()
        grafo.add_node("insumos_recursos", tipo="insumo")
        grafo.add_node("proceso_capacitacion", tipo="proceso")
        grafo.add_node("producto_manual", tipo="producto")
        grafo.add_node("resultado_conocimiento", tipo="resultado")
        grafo.add_node("impactos_cambio", tipo="impacto")

        # Conexiones válidas (máximo salto de 1 categoría)
        grafo.add_edge("insumos_recursos", "proceso_capacitacion")
        grafo.add_edge("proceso_capacitacion", "producto_manual")
        grafo.add_edge("producto_manual", "resultado_conocimiento")
        grafo.add_edge("resultado_conocimiento", "impactos_cambio")

        resultado = tc.validar_orden_causal(grafo)

        assert resultado.es_valida
        assert len(resultado.violaciones_orden) == 0

    @staticmethod
    def test_validacion_orden_causal_invalido():
        """Test de validación con violaciones de orden causal"""
        tc = TeoriaCambio()

        grafo = nx.DiGraph()
        grafo.add_node("insumos", tipo="insumo")
        grafo.add_node("impactos", tipo="impacto")

        # Conexión inválida: INSUMOS directamente a IMPACTOS (salto de 3 categorías)
        grafo.add_edge("insumos", "impactos")

        resultado = tc.validar_orden_causal(grafo)

        assert not resultado.es_valida
        assert len(resultado.violaciones_orden) == 1
        assert resultado.violaciones_orden[0]["categoria_origen"] == "INSUMOS"
        assert resultado.violaciones_orden[0]["categoria_destino"] == "IMPACTOS"

    @staticmethod
    def test_detectar_caminos_completos_existentes():
        """Test de detección de caminos completos válidos"""
        tc = TeoriaCambio()

        grafo = nx.DiGraph()
        # Crear cadena causal completa
        grafo.add_node("insumos", tipo="insumo")
        grafo.add_node("proceso", tipo="proceso")
        grafo.add_node("producto", tipo="producto")
        grafo.add_node("resultado", tipo="resultado")
        grafo.add_node("impactos", tipo="impacto")

        grafo.add_edge("insumos", "proceso")
        grafo.add_edge("proceso", "producto")
        grafo.add_edge("producto", "resultado")
        grafo.add_edge("resultado", "impactos")

        resultado = tc.detectar_caminos_completos(grafo)

        assert resultado.es_valida
        assert len(resultado.caminos_completos) >= 1
        # Verificar que el camino va de insumos a impactos
        camino = resultado.caminos_completos[0]
        assert camino[0] == "insumos"
        assert camino[-1] == "impactos"

    @staticmethod
    def test_detectar_caminos_incompletos():
        """Test cuando no hay caminos completos disponibles"""
        tc = TeoriaCambio()

        grafo = nx.DiGraph()
        grafo.add_node("insumos", tipo="insumo")
        grafo.add_node("proceso", tipo="proceso")
        grafo.add_node("impactos", tipo="impacto")

        # Solo conexión parcial, no hay camino completo
        grafo.add_edge("insumos", "proceso")
        # Falta conexión de proceso a impactos

        resultado = tc.detectar_caminos_completos(grafo)

        assert not resultado.es_valida
        assert len(resultado.caminos_completos) == 0

    @staticmethod
    def test_generar_sugerencias_categorias_faltantes():
        """Test de generación de sugerencias para categorías faltantes"""
        tc = TeoriaCambio()

        grafo = nx.DiGraph()
        # Solo agregar algunas categorías
        grafo.add_node("insumos", tipo="insumo")
        grafo.add_node("impactos", tipo="impacto")

        resultado = tc.generar_sugerencias(grafo)

        categorias_faltantes_nombres = [
            cat.name for cat in resultado.categorias_faltantes
        ]
        assert "PROCESOS" in categorias_faltantes_nombres
        assert "PRODUCTOS" in categorias_faltantes_nombres
        assert "RESULTADOS" in categorias_faltantes_nombres

        # Verificar que se generaron sugerencias apropiadas
        assert len(resultado.sugerencias) > 0
        sugerencias_texto = " ".join(resultado.sugerencias)
        assert "PROCESOS" in sugerencias_texto
        assert "PRODUCTOS" in sugerencias_texto

    @staticmethod
    def test_generar_sugerencias_conexiones_faltantes():
        """Test de sugerencias para conexiones faltantes entre categorías"""
        tc = TeoriaCambio()

        grafo = nx.DiGraph()
        grafo.add_node("insumos", tipo="insumo")
        grafo.add_node("proceso", tipo="proceso")
        grafo.add_node("impactos", tipo="impacto")

        # Crear nodos sin conexiones entre categorías adyacentes

        resultado = tc.generar_sugerencias(grafo)

        # Debe sugerir conexiones entre categorías adyacentes
        assert len(resultado.sugerencias) > 0

    @staticmethod
    def test_validacion_completa_integracion():
        """Test de validación completa integrando todas las funcionalidades"""
        tc = TeoriaCambio()

        # Crear teoría de cambio incompleta
        grafo = nx.DiGraph()
        grafo.add_node("insumos", tipo="insumo")
        grafo.add_node("producto", tipo="producto")
        grafo.add_node("impactos", tipo="impacto")

        # Conexión que viola el orden (salto excesivo)
        grafo.add_edge("insumos", "impactos")

        resultado = tc.validacion_completa(grafo)

        # La validación debe fallar por múltiples razones
        assert not resultado.es_valida
        assert len(resultado.violaciones_orden) > 0
        assert len(resultado.categorias_faltantes) > 0
        assert len(resultado.sugerencias) > 0

    @staticmethod
    def test_obtener_categoria_nodo_por_nombre():
        """Test de identificación de categorías por nombre del nodo"""
        tc = TeoriaCambio()
        grafo = nx.DiGraph()

        grafo.add_node("recursos_insumos", tipo="test")
        grafo.add_node("actividad_proceso", tipo="test")
        grafo.add_node("entregable_producto", tipo="test")
        grafo.add_node("cambio_resultado", tipo="test")
        grafo.add_node("transformacion_impacto", tipo="test")

        assert (
            tc._obtener_categoria_nodo("recursos_insumos", grafo)
            == CategoriaCausal.INSUMOS
        )
        assert (
            tc._obtener_categoria_nodo("actividad_proceso", grafo)
            == CategoriaCausal.PROCESOS
        )
        assert (
            tc._obtener_categoria_nodo("entregable_producto", grafo)
            == CategoriaCausal.PRODUCTOS
        )
        assert (
            tc._obtener_categoria_nodo("cambio_resultado", grafo)
            == CategoriaCausal.RESULTADOS
        )
        assert (
            tc._obtener_categoria_nodo("transformacion_impacto", grafo)
            == CategoriaCausal.IMPACTOS
        )

    @staticmethod
    def test_obtener_categoria_nodo_por_posicion():
        """Test de inferencia de categorías por posición topológica"""
        tc = TeoriaCambio()
        grafo = nx.DiGraph()

        # Nodo sin predecesores (debe ser INSUMOS)
        grafo.add_node("nodo_inicial")
        # Nodo sin sucesores (debe ser IMPACTOS)
        grafo.add_node("nodo_final")
        # Nodo intermedio
        grafo.add_node("nodo_medio")

        grafo.add_edge("nodo_inicial", "nodo_medio")
        grafo.add_edge("nodo_medio", "nodo_final")

        assert (
            tc._obtener_categoria_nodo(
                "nodo_inicial", grafo) == CategoriaCausal.INSUMOS
        )
        assert (
            tc._obtener_categoria_nodo(
                "nodo_final", grafo) == CategoriaCausal.IMPACTOS
        )
        assert (
            tc._obtener_categoria_nodo(
                "nodo_medio", grafo) == CategoriaCausal.PRODUCTOS
        )

    @staticmethod
    def test_es_conexion_valida():
        """Test de validación de conexiones entre categorías"""
        tc = TeoriaCambio()

        # Conexiones válidas (diferencia de 1 o 2)
        assert tc._es_conexion_valida(
            CategoriaCausal.INSUMOS, CategoriaCausal.PROCESOS
        )  # diff = 1
        assert tc._es_conexion_valida(
            CategoriaCausal.INSUMOS, CategoriaCausal.PRODUCTOS
        )  # diff = 2
        assert tc._es_conexion_valida(
            CategoriaCausal.PROCESOS, CategoriaCausal.RESULTADOS
        )  # diff = 2

        # Conexiones inválidas (diferencia > 2 o <= 0)
        assert not tc._es_conexion_valida(
            CategoriaCausal.INSUMOS, CategoriaCausal.RESULTADOS
        )  # diff = 3
        assert not tc._es_conexion_valida(
            CategoriaCausal.INSUMOS, CategoriaCausal.IMPACTOS
        )  # diff = 4
        assert not tc._es_conexion_valida(
            CategoriaCausal.PROCESOS, CategoriaCausal.INSUMOS
        )  # diff = -1

    @staticmethod
    def test_es_camino_completo():
        """Test de verificación de caminos completos"""
        tc = TeoriaCambio()
        grafo = nx.DiGraph()

        # Crear nodos con nombres que identifiquen claramente sus categorías
        nodos = [
            "insumos",
            "proceso_actividad",
            "producto_entregable",
            "resultado_cambio",
            "impactos",
        ]
        for nodo in nodos:
            grafo.add_node(nodo)

        # Camino completo
        camino_completo = [
            "insumos",
            "proceso_actividad",
            "producto_entregable",
            "resultado_cambio",
            "impactos",
        ]
        assert tc._es_camino_completo(camino_completo, grafo)

        # Camino incompleto (no termina en IMPACTOS)
        camino_incompleto = ["insumos",
                             "proceso_actividad", "producto_entregable"]
        assert not tc._es_camino_completo(camino_incompleto, grafo)

        # Camino que no empieza en INSUMOS
        camino_sin_inicio = ["proceso_actividad",
                             "producto_entregable", "impactos"]
        assert not tc._es_camino_completo(camino_sin_inicio, grafo)

    @staticmethod
    def test_obtener_nodos_por_categoria():
        """Test de obtención de nodos por categoría específica"""
        tc = TeoriaCambio()
        grafo = nx.DiGraph()

        grafo.add_node("recurso_insumo1", tipo="insumo")
        grafo.add_node("recurso_insumo2", tipo="insumo")
        grafo.add_node("actividad_proceso1", tipo="proceso")
        grafo.add_node("transformacion_impacto", tipo="impacto")

        nodos_insumos = tc._obtener_nodos_por_categoria(
            grafo, CategoriaCausal.INSUMOS)
        nodos_procesos = tc._obtener_nodos_por_categoria(
            grafo, CategoriaCausal.PROCESOS
        )
        nodos_impactos = tc._obtener_nodos_por_categoria(
            grafo, CategoriaCausal.IMPACTOS
        )

        assert len(nodos_insumos) == 2
        assert "recurso_insumo1" in nodos_insumos
        assert "recurso_insumo2" in nodos_insumos

        assert len(nodos_procesos) == 1
        assert "actividad_proceso1" in nodos_procesos

        assert len(nodos_impactos) == 1
        assert "transformacion_impacto" in nodos_impactos


if __name__ == "__main__":
    unittest.main()
