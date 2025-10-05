#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sistema de Evaluación Causal de Planes de Desarrollo Municipal (PDM)

Este módulo implementa un sistema automatizado para evaluar la solidez del diseño 
causal de Planes de Desarrollo Municipal aplicando un cuestionario estandarizado 
de 30 preguntas a 10 puntos temáticos.
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configuración del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Cuestionario base completo con 30 preguntas
BASE_QUESTIONNAIRE = [
    # DIMENSIÓN D1: DIAGNÓSTICO Y RECURSOS (Q1-Q5)
    {
        "id": "D1-Q1",
        "dimension": "D1",
        "descripcion": "Línea base cuantitativa",
        "pregunta": "¿El diagnóstico presenta líneas base cuantitativas para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D1-Q2",
        "dimension": "D1",
        "descripcion": "Cuantificación de magnitud",
        "pregunta": "¿Se cuantifica la magnitud del problema y se reconocen vacíos de información para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D1-Q3",
        "dimension": "D1",
        "descripcion": "Asignación presupuestal",
        "pregunta": "¿Los recursos del Plan Plurianual de Inversiones están asignados explícitamente a {PUNTO_TEMATICO}?"
    },
    {
        "id": "D1-Q4",
        "dimension": "D1",
        "descripcion": "Capacidades institucionales",
        "pregunta": "¿Se describen las capacidades institucionales necesarias para implementar las acciones en {PUNTO_TEMATICO}?"
    },
    {
        "id": "D1-Q5",
        "dimension": "D1",
        "descripcion": "Coherencia recursos-objetivos",
        "pregunta": "¿Existe presupuesto asignado y productos definidos para {PUNTO_TEMATICO}?"
    },
    
    # DIMENSIÓN D2: DISEÑO DE INTERVENCIÓN (Q6-Q10)
    {
        "id": "D2-Q6",
        "dimension": "D2",
        "descripcion": "Formalización de actividades",
        "pregunta": "¿Las actividades/productos para {PUNTO_TEMATICO} están formalizadas en tablas estructuradas?"
    },
    {
        "id": "D2-Q7",
        "dimension": "D2",
        "descripcion": "Población diana especificada",
        "pregunta": "¿Se especifica la población diana de las actividades en {PUNTO_TEMATICO}?"
    },
    {
        "id": "D2-Q8",
        "dimension": "D2",
        "descripcion": "Correspondencia problema-producto",
        "pregunta": "¿Los problemas identificados en {PUNTO_TEMATICO} tienen productos/actividades correspondientes?"
    },
    {
        "id": "D2-Q9",
        "dimension": "D2",
        "descripcion": "Causalidad explícita",
        "pregunta": "¿Se explicita la relación causal entre actividades, productos e indicadores de resultado para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D2-Q10",
        "dimension": "D2",
        "descripcion": "Cobertura poblacional",
        "pregunta": "¿La cobertura poblacional de las actividades garantiza el acceso equitativo a {PUNTO_TEMATICO}?"
    },
    
    # DIMENSIÓN D3: EJE TRANSVERSAL (Q11-Q15)
    {
        "id": "D3-Q11",
        "dimension": "D3",
        "descripcion": "Enfoque diferencial",
        "pregunta": "¿Se identifican grupos diferenciales vulnerables en relación con {PUNTO_TEMATICO}?"
    },
    {
        "id": "D3-Q12",
        "dimension": "D3",
        "descripcion": "Participación ciudadana",
        "pregunta": "¿Se contempla participación ciudadana en el diseño o ejecución de acciones para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D3-Q13",
        "dimension": "D3",
        "descripcion": "Género y diversidad",
        "pregunta": "¿Se incorpora enfoque de género y diversidad en el abordaje de {PUNTO_TEMATICO}?"
    },
    {
        "id": "D3-Q14",
        "dimension": "D3",
        "descripcion": "Cambio climático",
        "pregunta": "¿Se consideran riesgos y oportunidades asociados al cambio climático para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D3-Q15",
        "dimension": "D3",
        "descripcion": "Gobierno digital",
        "pregunta": "¿Se prevé el uso de tecnologías digitales en la ejecución de acciones para {PUNTO_TEMATICO}?"
    },
    
    # DIMENSIÓN D4: IMPLEMENTACIÓN (Q16-Q20)
    {
        "id": "D4-Q16",
        "dimension": "D4",
        "descripcion": "Cronograma de ejecución",
        "pregunta": "¿Se establece cronograma claro de ejecución para las actividades de {PUNTO_TEMATICO}?"
    },
    {
        "id": "D4-Q17",
        "dimension": "D4",
        "descripcion": "Responsables identificados",
        "pregunta": "¿Se identifican claramente los responsables de la ejecución de actividades para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D4-Q18",
        "dimension": "D4",
        "descripcion": "Mecanismos de seguimiento",
        "pregunta": "¿Se establecen mecanismos de seguimiento y control para las acciones de {PUNTO_TEMATICO}?"
    },
    {
        "id": "D4-Q19",
        "dimension": "D4",
        "descripcion": "Coordinación interinstitucional",
        "pregunta": "¿Se identifican mecanismos de coordinación interinstitucional para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D4-Q20",
        "dimension": "D4",
        "descripcion": "Riesgos identificados",
        "pregunta": "¿Se identifican riesgos potenciales en la ejecución de acciones para {PUNTO_TEMATICO}?"
    },
    
    # DIMENSIÓN D5: RESULTADOS (Q21-Q25)
    {
        "id": "D5-Q21",
        "dimension": "D5",
        "descripcion": "Indicadores de resultado",
        "pregunta": "¿Se definen indicadores de resultado claros y medibles para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D5-Q22",
        "dimension": "D5",
        "descripcion": "Metas cuantificadas",
        "pregunta": "¿Las metas definidas para {PUNTO_TEMATICO} son cuantificables y verificables?"
    },
    {
        "id": "D5-Q23",
        "dimension": "D5",
        "descripcion": "Líneas base de resultado",
        "pregunta": "¿Se establecen líneas base para los indicadores de resultado de {PUNTO_TEMATICO}?"
    },
    {
        "id": "D5-Q24",
        "dimension": "D5",
        "descripcion": "Medios de verificación",
        "pregunta": "¿Se especifican medios de verificación para los indicadores de {PUNTO_TEMATICO}?"
    },
    {
        "id": "D5-Q25",
        "dimension": "D5",
        "descripcion": "Análisis de impacto",
        "pregunta": "¿Se incluye análisis de impacto esperado para las acciones de {PUNTO_TEMATICO}?"
    },
    
    # DIMENSIÓN D6: SOSTENIBILIDAD (Q26-Q30)
    {
        "id": "D6-Q26",
        "dimension": "D6",
        "descripcion": "Sostenibilidad financiera",
        "pregunta": "¿Se identifican fuentes de financiación sostenibles para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D6-Q27",
        "dimension": "D6",
        "descripcion": "Sostenibilidad institucional",
        "pregunta": "¿Se establecen mecanismos de sostenibilidad institucional para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D6-Q28",
        "dimension": "D6",
        "descripcion": "Sostenibilidad ambiental",
        "pregunta": "¿Se consideran aspectos de sostenibilidad ambiental para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D6-Q29",
        "dimension": "D6",
        "descripcion": "Lecciones aprendidas",
        "pregunta": "¿Se incorporan lecciones aprendidas de planes anteriores para {PUNTO_TEMATICO}?"
    },
    {
        "id": "D6-Q30",
        "dimension": "D6",
        "descripcion": "Evaluación ex-post",
        "pregunta": "¿Se prevén evaluaciones ex-post para las acciones de {PUNTO_TEMATICO}?"
    }
]

class PDMEvaluator:
    """
    Clase principal para evaluar la solidez del diseño causal de un Plan de Desarrollo Municipal.
    """
    
    def __init__(self, pdm_document: str):
        """
        Inicializa el evaluador con el documento PDM.
        
        Args:
            pdm_document: Ruta al documento PDM o contenido del documento
        """
        self.pdm_document = pdm_document
        self.document_content = self._load_document()
        self.results = {}
        
    def _load_document(self) -> str:
        """
        Carga el contenido del documento PDM.
        
        Returns:
            Contenido del documento como string
        """
        # En una implementación completa, aquí se procesaría el PDF
        # Por ahora retornamos un string vacío
        return ""
        
    def evaluate_thematic_point(self, point: Dict[str, Any], questionnaire: List[Dict] = BASE_QUESTIONNAIRE) -> Dict[str, Any]:
        """
        Evalúa un punto temático aplicando el cuestionario.
        
        Args:
            point: Diccionario con información del punto temático
            questionnaire: Lista de preguntas del cuestionario
            
        Returns:
            Resultados de la evaluación
        """
        results = {
            "punto_tematico_id": point["id"],
            "punto_tematico_nombre": point["nombre"],
            "evaluaciones": []
        }
        
        for question in questionnaire:
            # Reemplazar variable en la pregunta
            question_text = question["pregunta"].format(PUNTO_TEMATICO=point["nombre"])
            
            # Evaluar pregunta
            evaluation = self._evaluate_question(question, point, question_text)
            results["evaluaciones"].append(evaluation)
            
        return results
    
    def _evaluate_question(self, question: Dict[str, Any], point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa una pregunta individual del cuestionario.
        
        Args:
            question: Diccionario con información de la pregunta
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta parametrizado
            
        Returns:
            Resultado de la evaluación de la pregunta
        """
        # Esta es una implementación básica que necesita ser expandida
        # según los patrones de búsqueda específicos de cada pregunta
        
        evaluation = {
            "pregunta_id": question["id"],
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {},
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Implementar lógica de evaluación específica según el tipo de pregunta
        if question["id"] == "D1-Q1":
            evaluation = self._evaluate_d1_q1(point, question_text)
        elif question["id"] == "D1-Q2":
            evaluation = self._evaluate_d1_q2(point, question_text)
        elif question["id"] == "D1-Q3":
            evaluation = self._evaluate_d1_q3(point, question_text)
        elif question["id"] == "D1-Q4":
            evaluation = self._evaluate_d1_q4(point, question_text)
        elif question["id"] == "D1-Q5":
            evaluation = self._evaluate_d1_q5(point, question_text)
        elif question["id"] == "D2-Q6":
            evaluation = self._evaluate_d2_q6(point, question_text)
        elif question["id"] == "D2-Q7":
            evaluation = self._evaluate_d2_q7(point, question_text)
        elif question["id"] == "D2-Q8":
            evaluation = self._evaluate_d2_q8(point, question_text)
        elif question["id"] == "D2-Q9":
            evaluation = self._evaluate_d2_q9(point, question_text)
        elif question["id"] == "D2-Q10":
            evaluation = self._evaluate_d2_q10(point, question_text)
        elif question["id"] == "D3-Q11":
            evaluation = self._evaluate_d3_q11(point, question_text)
        elif question["id"] == "D3-Q12":
            evaluation = self._evaluate_d3_q12(point, question_text)
        elif question["id"] == "D3-Q13":
            evaluation = self._evaluate_d3_q13(point, question_text)
        elif question["id"] == "D3-Q14":
            evaluation = self._evaluate_d3_q14(point, question_text)
        elif question["id"] == "D3-Q15":
            evaluation = self._evaluate_d3_q15(point, question_text)
        elif question["id"] == "D4-Q16":
            evaluation = self._evaluate_d4_q16(point, question_text)
        elif question["id"] == "D4-Q17":
            evaluation = self._evaluate_d4_q17(point, question_text)
        elif question["id"] == "D4-Q18":
            evaluation = self._evaluate_d4_q18(point, question_text)
        elif question["id"] == "D4-Q19":
            evaluation = self._evaluate_d4_q19(point, question_text)
        elif question["id"] == "D4-Q20":
            evaluation = self._evaluate_d4_q20(point, question_text)
        elif question["id"] == "D5-Q21":
            evaluation = self._evaluate_d5_q21(point, question_text)
        elif question["id"] == "D5-Q22":
            evaluation = self._evaluate_d5_q22(point, question_text)
        elif question["id"] == "D5-Q23":
            evaluation = self._evaluate_d5_q23(point, question_text)
        elif question["id"] == "D5-Q24":
            evaluation = self._evaluate_d5_q24(point, question_text)
        elif question["id"] == "D5-Q25":
            evaluation = self._evaluate_d5_q25(point, question_text)
        elif question["id"] == "D6-Q26":
            evaluation = self._evaluate_d6_q26(point, question_text)
        elif question["id"] == "D6-Q27":
            evaluation = self._evaluate_d6_q27(point, question_text)
        elif question["id"] == "D6-Q28":
            evaluation = self._evaluate_d6_q28(point, question_text)
        elif question["id"] == "D6-Q29":
            evaluation = self._evaluate_d6_q29(point, question_text)
        elif question["id"] == "D6-Q30":
            evaluation = self._evaluate_d6_q30(point, question_text)
        else:
            # Pregunta no implementada
            evaluation["recomendacion"] = "Pregunta no implementada en esta versión"
            
        return evaluation
    
    def _evaluate_d1_q1(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D1-Q1: Línea base cuantitativa.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D1-Q1",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "valor_numerico": False,
                "año": False,
                "fuente": False,
                "serie_temporal": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Buscar en el contenido del documento usando patrones
        # Esta es una implementación simplificada
        
        # Patrones de búsqueda
        patterns = {
            "valor_numerico": r"\d+[.,]?\d*\s*(%|casos|personas|tasa|porcentaje|índice)",
            "año": r"(20\d{2}|periodo|año|vigencia)",
            "fuente": r"(fuente:|según|DANE|Ministerio|Secretaría|Encuesta|Censo|SISBEN|SIVIGILA|\(20\d{2}\))",
            "serie_temporal": r"(20\d{2}.{0,50}20\d{2}|serie|histórico|evolución|tendencia)"
        }
        
        # Buscar elementos en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de diagnóstico",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (0.75 puntos por elemento, máximo 3)
        evaluation["score"] = (elementos_encontrados / 4) * 3
        
        # Generar recomendación
        if elementos_encontrados < 4:
            evaluation["recomendacion"] = "Incluir datos históricos (2019-2023) para identificar tendencias"
        
        return evaluation
    
    def _evaluate_d1_q2(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D1-Q2: Cuantificación de magnitud.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D1-Q2",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "poblacion_afectada": False,
                "brecha_deficit": False,
                "vacios_info": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "poblacion_afectada": r"(\d+\s*(personas|habitantes|casos|familias|mujeres|niños)|población.*\d+|afectados.*\d+)",
            "brecha_deficit": r"(brecha|déficit|diferencia|faltante|carencia|necesidad insatisfecha).{0,30}\d+",
            "vacios_info": r"(sin datos|no.*disponible|vacío|falta.*información|se requiere.*datos|limitación.*información|no se cuenta con)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de diagnóstico",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d1_q3(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D1-Q3: Asignación presupuestal.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D1-Q3",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "presupuesto_total": False,
                "desglose": False,
                "trazabilidad": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "presupuesto_total": r"\$\s*\d+([.,]\d+)?\s*(millones|miles de millones|mil|COP|pesos)",
            "desglose": r"(20\d{2}.*\$|Producto.*\$|Meta.*\$|anual|vigencia.*presupuesto|por año)",
            "trazabilidad": r"(Programa.{0,50}(inversión|presupuesto|recursos)|{}.{0,50}\$)".format(
                "|".join(point.get("programas_relevantes", [])))
        }
        
        # Buscar en secciones relacionadas con presupuesto
        context = self._get_budget_context()
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección presupuestal",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d1_q4(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D1-Q4: Capacidades institucionales.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D1-Q4",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "recursos_humanos": False,
                "infraestructura": False,
                "procesos_instancias": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "recursos_humanos": r"(profesionales|técnicos|funcionarios|personal|equipo|contratación|psicólogo|trabajador social|profesional).{0,50}\d+|se requiere.*personal|brecha.*talento",
            "infraestructura": r"(infraestructura|equipamiento|sede|oficina|espacios|dotación|vehículos|adecuación)",
            "procesos_instancias": r"(Secretaría|Comisaría|Comité|Mesa|Consejo|Sistema|procedimiento|protocolo|ruta|proceso de)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de capacidad institucional",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d1_q5(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D1-Q5: Coherencia recursos-objetivos.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D1-Q5",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {},
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Evaluar presupuesto (de Q3) y productos
        # Esta es una implementación simplificada
        presupuesto = self._get_budget_for_point(point)
        num_productos = self._count_products_for_point(point)
        
        # Regla de scoring
        if presupuesto > 0 and num_productos > 0:
            evaluation["score"] = 3
        elif presupuesto > 0:
            evaluation["score"] = 2
        else:
            evaluation["score"] = 0
            
        # Agregar evidencia
        evaluation["evidencia"].append({
            "texto": f"Presupuesto: ${presupuesto}, Productos definidos: {num_productos}",
            "ubicacion": "sección presupuestal y plan de acción",
            "elemento_verificado": ["presupuesto", "productos"]
        })
        
        return evaluation
    
    def _evaluate_d2_q6(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D2-Q6: Formalización de actividades.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D2-Q6",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "tabla_productos": False,
                "meta_cantidad": False,
                "unidad_medida": False,
                "responsable": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        expected_headers = [
            r"(Producto|Actividad|Indicador de producto)",
            r"(Meta|Cantidad|Número)",
            r"(Unidad|Medida)",
            r"(Responsable|Dependencia|Secretaría)"
        ]
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        # Verificar si hay tabla de productos
        if self._has_product_table(context):
            evaluation["elementos_encontrados"]["tabla_productos"] = True
            columnas_encontradas = 1  # Contar la tabla como una columna encontrada
            
            # Verificar otras columnas
            for i, pattern in enumerate(expected_headers[1:], 1):  # Saltar la primera que ya verificamos
                if re.search(pattern, context, re.IGNORECASE):
                    key = ["meta_cantidad", "unidad_medida", "responsable"][i-1]
                    evaluation["elementos_encontrados"][key] = True
                    columnas_encontradas += 1
                    # Agregar evidencia
                    matches = re.finditer(pattern, context, re.IGNORECASE)
                    for match in matches:
                        evaluation["evidencia"].append({
                            "texto": match.group(),
                            "ubicacion": "tabla de productos",
                            "elemento_verificado": [key]
                        })
                else:
                    key = ["meta_cantidad", "unidad_medida", "responsable"][i-1]
                    evaluation["elementos_faltantes"].append(key)
            
            # Calcular score (0.75 puntos por elemento, máximo 3)
            evaluation["score"] = (columnas_encontradas / 4) * 3
        else:
            evaluation["elementos_faltantes"] = ["tabla_productos", "meta_cantidad", "unidad_medida", "responsable"]
            evaluation["recomendacion"] = "Incluir tabla estructurada de productos con todas las columnas requeridas"
        
        return evaluation
    
    def _evaluate_d2_q7(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D2-Q7: Población diana especificada.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D2-Q7",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "poblacion_objetivo": False,
                "cuantificacion": False,
                "focalizacion": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "poblacion_objetivo": r"(mujeres|niños|niñas|adolescentes|jóvenes|víctimas|familias|comunidad|población|adultos mayores|personas con discapacidad)",
            "cuantificacion": r"\d+\s*(personas|beneficiarios|familias|atenciones|servicios|participantes)",
            "focalizacion": r"(zona rural|urbano|cabecera|prioridad|vulnerable|focalización|criterios|selección|población objetivo)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "descripción de productos",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d2_q8(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D2-Q8: Correspondencia problema-producto.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D2-Q8",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {},
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Método de verificación:
        # 1. Extraer problemas del diagnóstico
        # 2. Extraer productos de tabla de productos
        # 3. Calcular similaridad semántica
        # 4. Contar problemas con productos relacionados
        
        # Esta es una implementación simplificada
        problemas = self._extract_problems(point)
        productos = self._extract_products(point)
        
        # Calcular cobertura (en una implementación real usar embeddings semánticos)
        total_problemas = len(problemas)
        problemas_con_producto = self._count_problems_with_products(problemas, productos)
        
        if total_problemas > 0:
            ratio_cobertura = problemas_con_producto / total_problemas
        else:
            ratio_cobertura = 0
            
        # Regla de scoring
        if ratio_cobertura >= 0.80:
            evaluation["score"] = 3
        elif ratio_cobertura >= 0.50:
            evaluation["score"] = 2
        elif ratio_cobertura >= 0.30:
            evaluation["score"] = 1
        else:
            evaluation["score"] = 0
            
        # Agregar evidencia
        evaluation["evidencia"].append({
            "texto": f"Problemas identificados: {total_problemas}, con productos relacionados: {problemas_con_producto}",
            "ubicacion": "sección de diagnóstico y plan de acción",
            "elemento_verificado": ["problemas", "productos"]
        })
        
        return evaluation
    
    def _evaluate_d2_q9(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D2-Q9: Causalidad explícita.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D2-Q9",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "cadena_causal": False,
                "logica_intervencion": False,
                "teoria_cambio": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "cadena_causal": r"(cadena causal|causalidad|relación causal|lógicamente|porque|debido a)",
            "logica_intervencion": r"(lógica de intervención|teoría de cambio|hipótesis|supuesto)",
            "teoria_cambio": r"(teoría de cambio|modelo de cambio|cómo se espera|espera que|se prevé)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de diseño de intervención",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d2_q10(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D2-Q10: Cobertura poblacional.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D2-Q10",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "equidad_acceso": False,
                "cobertura_integral": False,
                "exclusion_mitigada": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "equidad_acceso": r"(equidad|acceso|igualdad|equitativo|justo|oportunidad)",
            "cobertura_integral": r"(cobertura|integral|completo|total|amplio|universal)",
            "exclusion_mitigada": r"(exclusión|mitigar|reducir|evitar|superar|inclusión)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de diseño de intervención",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d3_q11(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D3-Q11: Enfoque diferencial.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D3-Q11",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "grupos_diferenciales": False,
                "enfoque_diferencial": False,
                "vulnerabilidad_reconocida": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "grupos_diferenciales": r"(grupos diferenciales|pueblos indígenas|afrodescendientes|raizales|palenqueros|rom|gitano|indígena|afro|raizal|palenquero)",
            "enfoque_diferencial": r"(enfoque diferencial|diferencial|especial|particular|adaptado)",
            "vulnerabilidad_reconocida": r"(vulnerabilidad|riesgo|débil|frágil|situación especial|protección|salvaguarda)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección transversal",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d3_q12(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D3-Q12: Participación ciudadana.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D3-Q12",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "participacion_ciudadana": False,
                "instancia_participativa": False,
                "mecanismo_consulta": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "participacion_ciudadana": r"(participación ciudadana|ciudadanía|comunidad|participar|involucrar|consultar)",
            "instancia_participativa": r"(instancia|espacio|foro|encuentro|reunión|taller|charla)",
            "mecanismo_consulta": r"(mecanismo|consulta|diálogo|debate|discusión|interacción)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección transversal",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d3_q13(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D3-Q13: Género y diversidad.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D3-Q13",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "genero_incorporado": False,
                "diversidad_incluida": False,
                "perspectiva_genero": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "genero_incorporado": r"(género|masculino|femenino|identidad|expresión|rol de género)",
            "diversidad_incluida": r"(diversidad|LGBTI|sexualidad|orientación|identidad|expresión)",
            "perspectiva_genero": r"(perspectiva de género|enfoque de género|género y|género en)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección transversal",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d3_q14(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D3-Q14: Cambio climático.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D3-Q14",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "riesgos_climaticos": False,
                "oportunidades_verdes": False,
                "adaptacion_mitigacion": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "riesgos_climaticos": r"(riesgo.*climático|cambio climático|calentamiento global|emisiones|contaminación)",
            "oportunidades_verdes": r"(oportunidad.*verde|sostenible|ecológico|ambiental|verde|renovable)",
            "adaptacion_mitigacion": r"(adaptación|mitigación|resiliencia|resistencia|preparación|prevención)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección transversal",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d3_q15(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D3-Q15: Gobierno digital.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D3-Q15",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "tecnologias_digitales": False,
                "gobierno_digital": False,
                "transformacion_digital": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "tecnologias_digitales": r"(tecnología digital|digital|tecnología|TIC|internet|web|aplicación|app)",
            "gobierno_digital": r"(gobierno digital|digitalización|digitalizar|tecnología|plataforma|sistema)",
            "transformacion_digital": r"(transformación digital|digitalización|modernización|tecnología|innovación)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección transversal",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d4_q16(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D4-Q16: Cronograma de ejecución.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D4-Q16",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "cronograma_ejecucion": False,
                "fechas_definidas": False,
                "secuencia_actividades": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "cronograma_ejecucion": r"(cronograma|calendario|planificación temporal|temporalidad|tiempo|plazo)",
            "fechas_definidas": r"(fecha|inicio|fin|comienzo|terminación|duración|periodo|vigencia)",
            "secuencia_actividades": r"(secuencia|orden|sucesión|primero|después|luego|posteriormente|orden cronológico)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de implementación",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d4_q17(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D4-Q17: Responsables identificados.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D4-Q17",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "responsables_identificados": False,
                "entidades_involucradas": False,
                "roles_definidos": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "responsables_identificados": r"(responsable|encargado|coordinador|director|líder|jefe)",
            "entidades_involucradas": r"(entidad|secretaría|dependencia|unidad|departamento|oficina|institución)",
            "roles_definidos": r"(rol|función|tarea|responsabilidad|obligación|deber|funciones)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de implementación",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d4_q18(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D4-Q18: Mecanismos de seguimiento.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D4-Q18",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "mecanismos_seguimiento": False,
                "control_actividades": False,
                "indicadores_monitoreo": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "mecanismos_seguimiento": r"(seguimiento|monitoreo|control|verificación|evaluación|revisión)",
            "control_actividades": r"(control|supervisión|vigilancia|verificación|chequeo|revisión)",
            "indicadores_monitoreo": r"(indicador|medidor|métrica|variable|parámetro|criterio)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de implementación",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d4_q19(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D4-Q19: Coordinación interinstitucional.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D4-Q19",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "coordinacion_interinstitucional": False,
                "articulacion_entidades": False,
                "mecanismos_coordinacion": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "coordinacion_interinstitucional": r"(coordinación|articulación|colaboración|cooperación|trabajo conjunto|interinstitucional)",
            "articulacion_entidades": r"(articulación|interacción|relación|vínculo|conexión|intercambio)",
            "mecanismos_coordinacion": r"(mecanismo|instrumento|procedimiento|protocolo|acuerdo|alianza)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de implementación",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d4_q20(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D4-Q20: Riesgos identificados.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D4-Q20",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "riesgos_identificados": False,
                "analisis_riesgos": False,
                "mitigacion_riesgos": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "riesgos_identificados": r"(riesgo|amenaza|vulnerabilidad|debilidad|problema potencial|factor crítico)",
            "analisis_riesgos": r"(análisis|evaluación|estudio|identificación|diagnóstico|revisión)",
            "mitigacion_riesgos": r"(mitigación|prevención|reducción|eliminación|control|manejo|gestión)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de implementación",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d5_q21(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D5-Q21: Indicadores de resultado.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D5-Q21",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "indicadores_resultado": False,
                "medibles_claros": False,
                "vinculados_objetivos": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "indicadores_resultado": r"(indicador|medida|métrica|variable|parámetro|criterio)",
            "medibles_claros": r"(medible|cuantificable|verificable|observable|calculable|mensurable)",
            "vinculados_objetivos": r"(vinculado|relacionado|asociado|conectado|ligado|enlazado)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de resultados",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d5_q22(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D5-Q22: Metas cuantificadas.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D5-Q22",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "metas_cuantificadas": False,
                "verificables": False,
                "temporalizadas": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "metas_cuantificadas": r"(meta|objetivo|propósito|resultado esperado|logro|meta cuantitativa)",
            "verificables": r"(verificable|comprobable|demostrable|constatable|observable|evaluable)",
            "temporalizadas": r"(temporalizado|cronificado|programado|plazo|fecha|periodo|vigencia)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de resultados",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d5_q23(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D5-Q23: Líneas base de resultado.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D5-Q23",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "lineas_base_resultado": False,
                "punto_referencia": False,
                "medicion_inicial": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "lineas_base_resultado": r"(línea base|base inicial|punto de partida|situación inicial|estado inicial)",
            "punto_referencia": r"(referencia|punto de comparación|comparador|benchmark|estándar|parámetro)",
            "medicion_inicial": r"(medición|evaluación|diagnóstico|análisis|estudio|levantamiento)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de resultados",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d5_q24(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D5-Q24: Medios de verificación.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D5-Q24",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "medios_verificacion": False,
                "fuentes_informacion": False,
                "metodos_recoleccion": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "medios_verificacion": r"(medio de verificación|verificación|comprobación|demostración|constatación|evidencia)",
            "fuentes_informacion": r"(fuente|origen|procedencia|referencia|base de datos|repositorio|sistema)",
            "metodos_recoleccion": r"(método|método de recolección|recolección|recopilación|captura|registro)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de resultados",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d5_q25(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D5-Q25: Análisis de impacto.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D5-Q25",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "analisis_impacto": False,
                "efectos_esperados": False,
                "beneficiarios_potenciales": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "analisis_impacto": r"(análisis de impacto|impacto|efecto|consecuencia|resultado|resultado esperado)",
            "efectos_esperados": r"(efecto esperado|resultado esperado|impacto esperado|beneficio|ventaja|mejora)",
            "beneficiarios_potenciales": r"(beneficiario|afectado|receptor|destinatario|usuario|población)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de resultados",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d6_q26(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D6-Q26: Sostenibilidad financiera.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D6-Q26",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "sostenibilidad_financiera": False,
                "fuentes_financiacion": False,
                "modelo_sostenible": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "sostenibilidad_financiera": r"(sostenibilidad financiera|financiamiento|recursos|presupuesto|inversión|costo)",
            "fuentes_financiacion": r"(fuente de financiación|recurso|financiador|financia|patrocinador|apoyo económico)",
            "modelo_sostenible": r"(modelo sostenible|sostenibilidad|duradero|permanente|continuidad|mantenimiento)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de sostenibilidad",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d6_q27(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D6-Q27: Sostenibilidad institucional.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D6-Q27",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "sostenibilidad_institucional": False,
                "capacidades_institucionales": False,
                "estructuras_sostenibles": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "sostenibilidad_institucional": r"(sostenibilidad institucional|institucionalidad|estructura|organización|capacidad institucional)",
            "capacidades_institucionales": r"(capacidad institucional|fortalecimiento|desarrollo institucional|institucional|organizacional)",
            "estructuras_sostenibles": r"(estructura sostenible|modelo sostenible|sistema sostenible|organización sostenible)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de sostenibilidad",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d6_q28(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D6-Q28: Sostenibilidad ambiental.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D6-Q28",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "sostenibilidad_ambiental": False,
                "impacto_ambiental": False,
                "practicas_sostenibles": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "sostenibilidad_ambiental": r"(sostenibilidad ambiental|medio ambiente|ambiente|ecología|ecosistema|naturaleza)",
            "impacto_ambiental": r"(impacto ambiental|efecto ambiental|consecuencia ambiental|daño ambiental|afectación ambiental)",
            "practicas_sostenibles": r"(práctica sostenible|práctica ecológica|práctica ambiental|sostenible|ecológico|verde)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de sostenibilidad",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d6_q29(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D6-Q29: Lecciones aprendidas.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D6-Q29",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "lecciones_aprendidas": False,
                "experiencias_previas": False,
                "mejora_continua": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "lecciones_aprendidas": r"(lección aprendida|experiencia|aprendizaje|conocimiento adquirido|sabiduría|enseñanza)",
            "experiencias_previas": r"(experiencia previa|experiencia anterior|plan anterior|proceso anterior|ejercicio anterior)",
            "mejora_continua": r"(mejora continua|mejoramiento|optimización|perfeccionamiento|evolución|desarrollo)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de sostenibilidad",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _evaluate_d6_q30(self, point: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        """
        Evalúa la pregunta D6-Q30: Evaluación ex-post.
        
        Args:
            point: Diccionario con información del punto temático
            question_text: Texto de la pregunta
            
        Returns:
            Resultado de la evaluación
        """
        evaluation = {
            "pregunta_id": "D6-Q30",
            "pregunta": question_text,
            "score": 0,
            "elementos_encontrados": {
                "evaluacion_expost": False,
                "evaluacion_posterior": False,
                "seguimiento_largo_plazo": False
            },
            "evidencia": [],
            "elementos_faltantes": [],
            "recomendacion": ""
        }
        
        # Patrones de búsqueda
        patterns = {
            "evaluacion_expost": r"(evaluación ex-post|evaluación posterior|evaluación ex post|evaluación después|evaluación final)",
            "evaluacion_posterior": r"(evaluación posterior|evaluación siguiente|evaluación futura|evaluación después|evaluación al final)",
            "seguimiento_largo_plazo": r"(seguimiento a largo plazo|evaluación a largo plazo|monitoreo continuo|evaluación permanente|evaluación duradera)"
        }
        
        # Buscar en el contexto del punto temático
        context = self._get_context_for_point(point)
        
        elementos_encontrados = 0
        for key, pattern in patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                evaluation["elementos_encontrados"][key] = True
                elementos_encontrados += 1
                # Agregar evidencia
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    evaluation["evidencia"].append({
                        "texto": match.group(),
                        "ubicacion": "sección de sostenibilidad",
                        "elemento_verificado": [key]
                    })
            else:
                evaluation["elementos_faltantes"].append(key)
        
        # Calcular score (1 punto por elemento, máximo 3)
        evaluation["score"] = elementos_encontrados
        
        return evaluation
    
    def _get_context_for_point(self, point: Dict[str, Any]) -> str:
        """
        Obtiene el contexto relevante para un punto temático.
        
        Args:
            point: Diccionario con información del punto temático
            
        Returns:
            Texto con el contexto relevante
        """
        # En una implementación completa, esto buscaría en las secciones
        # específicas del documento identificadas para el punto temático
        keywords = "|".join(point.get("keywords", []))
        # Buscar en el contenido del documento usando las keywords
        # Esta es una simplificación
        return self.document_content
    
    def _get_budget_context(self) -> str:
        """
        Obtiene el contexto relacionado con el presupuesto.
        
        Returns:
            Texto con el contexto presupuestal
        """
        # En una implementación completa, esto buscaría en las secciones
        # específicas del documento relacionadas con el presupuesto
        return self.document_content
    
    def _get_budget_for_point(self, point: Dict[str, Any]) -> float:
        """
        Obtiene el presupuesto asignado a un punto temático.
        
        Args:
            point: Diccionario con información del punto temático
            
        Returns:
            Valor del presupuesto
        """
        # En una implementación completa, esto extraería el valor del presupuesto
        # del documento para el punto temático específico
        return 0.0
    
    def _count_products_for_point(self, point: Dict[str, Any]) -> int:
        """
        Cuenta el número de productos definidos para un punto temático.
        
        Args:
            point: Diccionario con información del punto temático
            
        Returns:
            Número de productos definidos
        """
        # En una implementación completa, esto contaría los productos definidos
        # en el documento para el punto temático específico
        return 0
    
    def _has_product_table(self, context: str) -> bool:
        """
        Verifica si hay una tabla de productos en el contexto.
        
        Args:
            context: Texto con el contexto a analizar
            
        Returns:
            True si se encuentra una tabla de productos, False en caso contrario
        """
        # En una implementación completa, esto verificaría la existencia
        # de una tabla estructurada de productos
        return False
    
    def _extract_problems(self, point: Dict[str, Any]) -> List[str]:
        """
        Extrae los problemas identificados en el diagnóstico para un punto temático.
        
        Args:
            point: Diccionario con información del punto temático
            
        Returns:
            Lista de problemas identificados
        """
        # En una implementación completa, esto extraería los problemas
        # del diagnóstico del documento
        return []
    
    def _extract_products(self, point: Dict[str, Any]) -> List[str]:
        """
        Extrae los productos definidos para un punto temático.
        
        Args:
            point: Diccionario con información del punto temático
            
        Returns:
            Lista de productos definidos
        """
        # En una implementación completa, esto extraería los productos
        # definidos en el documento
        return []
    
    def _count_problems_with_products(self, problems: List[str], products: List[str]) -> int:
        """
        Cuenta cuántos problemas tienen al menos un producto relacionado.
        
        Args:
            problems: Lista de problemas identificados
            products: Lista de productos definidos
            
        Returns:
            Número de problemas con productos relacionados
        """
        # En una implementación completa, esto calcularía la similaridad
        # semántica entre problemas y productos
        return 0
    
    def run_evaluation(self, thematic_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ejecuta la evaluación completa para todos los puntos temáticos.
        
        Args:
            thematic_points: Lista de puntos temáticos a evaluar
            
        Returns:
            Resultados completos de la evaluación
        """
        results = {
            "evaluacion_completa": {
                "puntos_tematicos_evaluados": []
            }
        }
        
        for point in thematic_points:
            logger.info(f"Evaluando punto temático: {point['nombre']}")
            point_results = self.evaluate_thematic_point(point)
            results["evaluacion_completa"]["puntos_tematicos_evaluados"].append(point_results)
            
        return results


def create_sample_evaluation():
    """
    Crea una evaluación de ejemplo con datos de prueba.
    """
    # Ejemplo de parámetros de entrada
    params = {
        "documento_pdm": "ruta/al/pdm.pdf",
        "municipio": "Anorí",
        "departamento": "Antioquia",
        "periodo": "2024-2027",
        "puntos_tematicos": [
            {
                "id": "P1",
                "nombre": "Derechos de las mujeres e igualdad de género",
                "programas_relevantes": ["Por las Mujeres", "Equidad de Género"],
                "keywords": ["mujer", "género", "violencia de género", "autonomía económica", "participación"],
                "seccion_pdm": ["páginas 45-52", "tablas 8-10"]
            },
            {
                "id": "P2",
                "nombre": "Prevención de la violencia y protección frente al conflicto",
                "programas_relevantes": ["Paz y Convivencia", "Víctimas"],
                "keywords": ["conflicto", "violencia", "paz", "convivencia", "seguridad"],
                "seccion_pdm": ["páginas 53-60"]
            }
        ]
    }
    
    # Crear evaluador
    evaluator = PDMEvaluator(params["documento_pdm"])
    
    # Ejecutar evaluación
    results = evaluator.run_evaluation(params["puntos_tematicos"])
    
    return results


if __name__ == "__main__":
    # Crear y mostrar una evaluación de ejemplo
    sample_results = create_sample_evaluation()
    print(json.dumps(sample_results, indent=2, ensure_ascii=False))