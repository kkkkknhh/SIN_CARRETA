# PROMPT MAESTRO PARA INTEGRACIÓN EN PIPELINE AUTOMATIZADO

## ESTRUCTURA DEL PROMPT

```yaml
VERSION: 1.0
TIPO: Evaluación Causal de Planes de Desarrollo Municipal
METODOLOGIA: Cuestionario de 30 preguntas × 10 puntos temáticos (300 evaluaciones)
OUTPUT: JSON estructurado con scores, evidencia y metadata
```

---

## I. INSTRUCCIONES GENERALES PARA EL SISTEMA

### **OBJETIVO**
Evaluar la solidez del diseño causal de un Plan de Desarrollo Municipal (PDM) aplicando 30 preguntas estandarizadas a cada uno de los 10 Puntos Temáticos priorizados. Cada pregunta busca verificar la presencia objetiva de elementos específicos en el documento.

### **PARÁMETROS DE ENTRADA**

```json
{
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
    // ... P3 a P10
  ]
}
```

### **MODO DE EJECUCIÓN**

```python
# Para cada punto temático (P1 a P10):
for punto_tematico in puntos_tematicos:
    # Aplicar las 30 preguntas (Q1 a Q30) agrupadas en 6 dimensiones (D1 a D6)
    resultados = evaluar_punto_tematico(
        pdm=documento_pdm,
        punto=punto_tematico,
        cuestionario=CUESTIONARIO_30_PREGUNTAS
    )
    
    # Guardar resultados
    guardar_resultados(punto_tematico.id, resultados)
```

### **PRINCIPIOS DE EVALUACIÓN**

1. **Objetividad**: Verificar presencia/ausencia de elementos, NO interpretar calidad subjetiva
2. **Evidencia obligatoria**: Cada score debe acompañarse de cita textual del PDM
3. **Contextualización**: Las búsquedas deben filtrarse por las secciones y programas relevantes de cada punto temático
4. **Manejo de ausencias**: Si un elemento no se encuentra, score = 0 (no asumir presencia implícita)
5. **N/A explícito**: Si un punto temático no aplica al municipio, marcar como N/A y excluir del cálculo global

---

## II. CUESTIONARIO BASE (30 PREGUNTAS PARAMETRIZABLES)

### **NOTA CRÍTICA PARA EL SISTEMA**
Las siguientes 30 preguntas se aplican a CADA uno de los 10 puntos temáticos. La variable `{PUNTO_TEMATICO}` debe ser sustituida dinámicamente por el nombre del punto temático en evaluación.

**Ejemplo de parametrización:**
- Si `PUNTO_TEMATICO = "Derechos de las mujeres e igualdad de género"`
- Entonces Q1 se convierte en: "¿El diagnóstico presenta líneas base con fuentes para **Derechos de las mujeres e igualdad de género**?"

---

## **DIMENSIÓN D1: DIAGNÓSTICO Y RECURSOS (Q1-Q5)**

### **D1-Q1: LÍNEA BASE CUANTITATIVA**

**Pregunta parametrizable:**
```
¿El diagnóstico presenta líneas base cuantitativas para {PUNTO_TEMATICO}?
```

**Elementos a verificar** (0.75 puntos cada uno, máximo 3):
1. **Valor numérico** de línea base presente
2. **Año/período** de medición explícito  
3. **Fuente** de datos identificada
4. **Serie temporal** (≥2 puntos en el tiempo) o comparador externo

**Patrones de búsqueda:**
```json
{
  "buscar_en": "secciones de diagnóstico relacionadas con {KEYWORDS_PUNTO_TEMATICO}",
  "patrones": {
    "valor_numerico": "\\d+[.,]?\\d*\\s*(%|casos|personas|tasa|porcentaje|índice)",
    "año": "(20\\d{2}|periodo|año|vigencia)",
    "fuente": "(fuente:|según|DANE|Ministerio|Secretaría|Encuesta|Censo|SISBEN|SIVIGILA|\\(20\\d{2}\\))",
    "serie_temporal": "(20\\d{2}.{0,50}20\\d{2}|serie|histórico|evolución|tendencia)"
  },
  "contexto": "Buscar dentro de ±500 palabras de {KEYWORDS_PUNTO_TEMATICO}"
}
```

**Regla de scoring:**
```python
elementos_encontrados = sum([
    1 if encontrado(patron["valor_numerico"]) else 0,
    1 if encontrado(patron["año"]) else 0,
    1 if encontrado(patron["fuente"]) else 0,
    1 if encontrado(patron["serie_temporal"]) else 0
])
score_q1 = (elementos_encontrados / 4) * 3
```

**Output esperado:**
```json
{
  "pregunta_id": "P1-D1-Q1",
  "punto_tematico": "Derechos de las mujeres e igualdad de género",
  "score": 2.25,
  "elementos_encontrados": {
    "valor_numerico": true,
    "año": true,
    "fuente": true,
    "serie_temporal": false
  },
  "evidencia": [
    {
      "texto": "Tasa de desempleo mujeres: 12.4% (2021). Fuente: GEIH",
      "ubicacion": "página 47, párrafo 3",
      "elemento_verificado": ["valor_numerico", "año", "fuente"]
    }
  ],
  "elementos_faltantes": ["serie_temporal"],
  "recomendacion": "Incluir datos históricos (2019-2023) para identificar tendencias"
}
```

---

### **D1-Q2: CUANTIFICACIÓN DE MAGNITUD**

**Pregunta parametrizable:**
```
¿Se cuantifica la magnitud del problema y se reconocen vacíos de información para {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1 punto cada uno, máximo 3):
1. **Población afectada** cuantificada (N, %, tasa)
2. **Brecha o déficit** calculado explícitamente
3. **Vacíos de información** reconocidos

**Patrones de búsqueda:**
```json
{
  "buscar_en": "secciones de diagnóstico + análisis situacional",
  "patrones": {
    "poblacion_afectada": "(\\d+\\s*(personas|habitantes|casos|familias|mujeres|niños)|población.*\\d+|afectados.*\\d+)",
    "brecha_deficit": "(brecha|déficit|diferencia|faltante|carencia|necesidad insatisfecha).{0,30}\\d+",
    "vacios_info": "(sin datos|no.*disponible|vacío|falta.*información|se requiere.*datos|limitación.*información|no se cuenta con)"
  }
}
```

**Regla de scoring:**
```python
score_q2 = sum([
    1 if encontrado("poblacion_afectada") else 0,
    1 if encontrado("brecha_deficit") else 0,
    1 if encontrado("vacios_info") else 0
])
```

---

### **D1-Q3: ASIGNACIÓN PRESUPUESTAL**

**Pregunta parametrizable:**
```
¿Los recursos del Plan Plurianual de Inversiones están asignados explícitamente a {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1 punto cada uno, máximo 3):
1. **Presupuesto total** del programa identificado (cifra + moneda)
2. **Desglose temporal o por producto** presente
3. **Trazabilidad programa-presupuesto** verificable

**Patrones de búsqueda:**
```json
{
  "buscar_en": "tablas de inversión + plan indicativo + sección presupuestal",
  "patrones": {
    "presupuesto_total": "\\$\\s*\\d+([.,]\\d+)?\\s*(millones|miles de millones|mil|COP|pesos)",
    "desglose": "(20\\d{2}.*\\$|Producto.*\\$|Meta.*\\$|anual|vigencia.*presupuesto|por año)",
    "trazabilidad": "(Programa.{0,50}(inversión|presupuesto|recursos)|{NOMBRE_PROGRAMA}.{0,50}\\$)"
  },
  "verificacion_tabular": "Buscar en tabla donde columna='Programa' contiene {NOMBRE_PROGRAMA} y existe columna con valores monetarios"
}
```

**Regla de scoring:**
```python
score_q3 = sum([
    1 if presupuesto_total > 0 else 0,
    1 if existe_desglose() else 0,
    1 if programa_en_tabla_presupuestal() else 0
])
```

---

### **D1-Q4: CAPACIDADES INSTITUCIONALES**

**Pregunta parametrizable:**
```
¿Se describen las capacidades institucionales necesarias para implementar las acciones en {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1 punto cada uno, máximo 3):
1. **Recursos humanos** mencionados (cantidad, perfil, o brecha)
2. **Infraestructura/equipamiento** mencionado
3. **Procesos o instancias institucionales** descritos

**Patrones de búsqueda:**
```json
{
  "buscar_en": "sección de capacidad institucional + descripción de programas",
  "patrones": {
    "recursos_humanos": "(profesionales|técnicos|funcionarios|personal|equipo|contratación|psicólogo|trabajador social|profesional).{0,50}\\d+|se requiere.*personal|brecha.*talento",
    "infraestructura": "(infraestructura|equipamiento|sede|oficina|espacios|dotación|vehículos|adecuación)",
    "procesos_instancias": "(Secretaría|Comisaría|Comité|Mesa|Consejo|Sistema|procedimiento|protocolo|ruta|proceso de)"
  }
}
```

---

### **D1-Q5: COHERENCIA RECURSOS-OBJETIVOS**

**Pregunta parametrizable:**
```
¿Existe presupuesto asignado y productos definidos para {PUNTO_TEMATICO}?
```

**Elementos a verificar** (regla lógica):
- Presupuesto > 0 AND existen productos definidos → 3 puntos
- Presupuesto > 0 BUT sin productos → 2 puntos  
- Presupuesto = 0 → 0 puntos

**Patrones de búsqueda:**
```json
{
  "verificacion_presupuesto": "Extraer valor monetario de Q3",
  "verificacion_productos": "Contar filas en tabla de productos/indicadores de producto para {PROGRAMA_RELEVANTE}"
}
```

**Regla de scoring:**
```python
if presupuesto > 0 and num_productos > 0:
    score_q5 = 3
elif presupuesto > 0:
    score_q5 = 2
else:
    score_q5 = 0
```

---

## **DIMENSIÓN D2: DISEÑO DE INTERVENCIÓN (Q6-Q10)**

### **D2-Q6: FORMALIZACIÓN DE ACTIVIDADES**

**Pregunta parametrizable:**
```
¿Las actividades/productos para {PUNTO_TEMATICO} están formalizadas en tablas estructuradas?
```

**Elementos a verificar** (0.75 puntos cada uno, máximo 3):
1. **Tabla de productos** identificada
2. Columna **"Meta/Cantidad"** presente
3. Columna **"Unidad de medida"** presente
4. Columna **"Responsable/Dependencia"** presente

**Patrones de búsqueda:**
```json
{
  "buscar_en": "tablas en sección de programas o plan de acción",
  "deteccion_tabla": {
    "encabezados_esperados": [
      "(Producto|Actividad|Indicador de producto)",
      "(Meta|Cantidad|Número)",
      "(Unidad|Medida)",
      "(Responsable|Dependencia|Secretaría)"
    ]
  },
  "filtro": "Tabla debe estar en sección de {PROGRAMA_RELEVANTE}"
}
```

**Regla de scoring:**
```python
columnas_encontradas = contar_columnas_en_tabla_productos({PROGRAMA_RELEVANTE})
score_q6 = (columnas_encontradas / 4) * 3
```

---

### **D2-Q7: POBLACIÓN DIANA ESPECIFICADA**

**Pregunta parametrizable:**
```
¿Se especifica la población diana de las actividades en {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1 punto cada uno, máximo 3):
1. **Población objetivo** nombrada (grupo específico)
2. **Cuantificación** de beneficiarios (meta numérica)
3. **Criterios de focalización** mencionados

**Patrones de búsqueda:**
```json
{
  "buscar_en": "descripción de productos + tabla de metas",
  "patrones": {
    "poblacion_objetivo": "(mujeres|niños|niñas|adolescentes|jóvenes|víctimas|familias|comunidad|población|adultos mayores|personas con discapacidad)",
    "cuantificacion": "\\d+\\s*(personas|beneficiarios|familias|atenciones|servicios|participantes)",
    "focalizacion": "(zona rural|urbano|cabecera|prioridad|vulnerable|focalización|criterios|selección|población objetivo)"
  }
}
```

---

### **D2-Q8: CORRESPONDENCIA PROBLEMA-PRODUCTO**

**Pregunta parametrizable:**
```
¿Los problemas identificados en {PUNTO_TEMATICO} tienen productos/actividades correspondientes?
```

**Método de verificación:**
1. Extraer problemas del diagnóstico (sección "Situación actual" o "Problemática")
2. Extraer productos de tabla de productos
3. Calcular similaridad semántica entre cada problema y conjunto de productos
4. Contar cuántos problemas tienen al menos 1 producto relacionado (similaridad >0.6)

**Patrón de búsqueda:**
```json
{
  "paso1_extraer_problemas": {
    "buscar_en": "sección de diagnóstico de {PUNTO_TEMATICO}",
    "patrones": "(problema:|limitación:|dificultad:|situación actual|déficit|brecha|necesidad)"
  },
  "paso2_extraer_productos": {
    "buscar_en": "tabla de productos de {PROGRAMA_RELEVANTE}",
    "extraer": "columna 'Producto' o 'Indicador de producto'"
  },
  "paso3_calcular_cobertura": "usar embeddings semánticos (sentence-transformers)"
}
```

**Regla de scoring:**
```python
ratio_cobertura = problemas_con_producto / total_problemas

if ratio_cobertura >= 0.80:
    score_q8 = 3
elif ratio_cobertura >= 0.50:
    score_q8 = 2
elif ratio_cobertura >= 0.30:
    score_q8 = 1
else:
    score_q8 = 0
```

---

### **D2-Q9: RECONOCIMIENTO DE RIESGOS**

**Pregunta parametrizable:**
```
¿Se reconocen limitaciones, riesgos o factores externos para {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1.5 puntos cada uno, máximo 3):
1. **Mención de riesgos/limitaciones** explícita
2. **Factores externos** reconocidos

**Patrones de búsqueda:**
```json
{
  "buscar_en": "cualquier sección del programa",
  "patrones": {
    "riesgos_explicitos": "(riesgo|limitación|restricción|dificultad|cuello de botella|matriz.*riesgo|desafío|obstáculo)",
    "factores_externos": "(depende|articulación|coordinación|transversal|nivel nacional|competencia de|requiere apoyo|sujeto a)"
  }
}
```

**Regla de scoring:**
```python
score_q9 = (elementos_encontrados / 2) * 3
```

---

### **D2-Q10: ARTICULACIÓN ENTRE ACTIVIDADES**

**Pregunta parametrizable:**
```
¿Se menciona articulación o complementariedad entre actividades de {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1.5 puntos cada uno, máximo 3):
1. **Términos de integración** presentes
2. **Referencia cruzada** a otros programas

**Patrones de búsqueda:**
```json
{
  "buscar_en": "descripción del programa o productos",
  "patrones": {
    "integracion": "(articulación|complementa|sinergia|coordinación|integra|transversal|en conjunto|simultáneamente)",
    "referencia_cruzada": "(programa de|articulado con|en el marco de|junto con|además de)"
  }
}
```

---

## **DIMENSIÓN D3: PRODUCTOS (Q11-Q15)**

### **D3-Q11: MENCIÓN DE ESTÁNDARES**

**Pregunta parametrizable:**
```
¿Se mencionan estándares técnicos o protocolos para los productos de {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1.5 puntos cada uno, máximo 3):
1. **Normas/estándares/protocolos** referenciados
2. **Supervisión/verificación** mencionada

**Patrones de búsqueda:**
```json
{
  "patrones": {
    "estandares": "(norma|estándar|protocolo|lineamiento|guía|directriz|NTC|ISO|Resolución|Decreto|Ley|según el Ministerio)",
    "control_calidad": "(certificación|acreditación|supervisión|verificación|control de calidad|seguimiento|auditoría)"
  }
}
```

---

### **D3-Q12: PROPORCIONALIDAD META-PROBLEMA**

**Pregunta parametrizable:**
```
¿La meta de productos es proporcional a la magnitud del problema en {PUNTO_TEMATICO}?
```

**Método de verificación:**
1. Extraer magnitud del problema de Q2 (población afectada)
2. Extraer meta de producto principal de tabla
3. Calcular ratio de cobertura

**Regla de scoring:**
```python
ratio_cobertura = meta_producto / magnitud_problema

if ratio_cobertura >= 0.50:
    score_q12 = 3
elif ratio_cobertura >= 0.20:
    score_q12 = 2
elif ratio_cobertura >= 0.05:
    score_q12 = 1
else:
    score_q12 = 0
```

---

### **D3-Q13: CUANTIFICACIÓN DE BENEFICIARIOS**

**Pregunta parametrizable:**
```
¿Las metas de productos en {PUNTO_TEMATICO} están cuantificadas?
```

**Elementos a verificar** (1.5 puntos cada uno, máximo 3):
1. **Meta numérica** presente
2. **Desagregación** de meta (temporal, espacial, por grupo)

**Patrones de búsqueda:**
```json
{
  "buscar_en": "tabla de productos, columna 'Meta'",
  "patrones": {
    "meta_numerica": "\\d+",
    "desagregacion": "(\\d+.{0,20}(rural|urbano|2024|2025|2026|2027|hombres|mujeres|año|anual))"
  }
}
```

---

### **D3-Q14: DEPENDENCIA RESPONSABLE ASIGNADA**

**Pregunta parametrizable:**
```
¿Cada producto en {PUNTO_TEMATICO} tiene una dependencia responsable asignada?
```

**Método de verificación:**
- Contar productos con valor no vacío en columna "Responsable"

**Regla de scoring:**
```python
ratio = productos_con_responsable / total_productos

if ratio >= 0.90:
    score_q14 = 3
elif ratio >= 0.70:
    score_q14 = 2
elif ratio >= 0.40:
    score_q14 = 1
else:
    score_q14 = 0
```

---

### **D3-Q15: JUSTIFICACIÓN CAUSAL**

**Pregunta parametrizable:**
```
¿Los productos de {PUNTO_TEMATICO} tienen justificación causal (relación con resultados esperados)?
```

**Método de verificación:**
- Buscar en ±200 palabras de cada producto términos: "para", "con el fin de", "contribuye a", "permite", "lograr"

**Regla de scoring:**
```python
ratio = productos_con_terminos_causales / total_productos

if ratio >= 0.70:
    score_q15 = 3
elif ratio >= 0.40:
    score_q15 = 2
elif ratio >= 0.20:
    score_q15 = 1
else:
    score_q15 = 0
```

---

## **DIMENSIÓN D4: RESULTADOS (Q16-Q20)**

### **D4-Q16: INDICADOR DE RESULTADO PRESENTE**

**Pregunta parametrizable:**
```
¿Existe un indicador de resultado formalizado para {PUNTO_TEMATICO}?
```

**Elementos a verificar** (0.75 puntos cada uno, máximo 3):
1. **Nombre del indicador** de resultado
2. **Línea base** numérica
3. **Meta cuatrienio** numérica
4. **Fuente** del indicador

**Patrones de búsqueda:**
```json
{
  "buscar_en": "tabla de indicadores de resultado o sección 'Indicadores'",
  "patrones": {
    "nombre_indicador": "(Tasa|Porcentaje|Índice|Cobertura|Prevalencia|Incidencia).{0,100}(resultado|impacto)",
    "linea_base": "(línea base|LB|valor inicial|20(21|22|23)).*\\d+",
    "meta": "(meta|valor esperado|20(27|28)).*\\d+",
    "fuente": "(fuente|según|medición|DANE|Ministerio|Secretaría)"
  }
}
```

---

### **D4-Q17: DIFERENCIACIÓN PRODUCTO-RESULTADO**

**Pregunta parametrizable:**
```
¿El indicador de resultado de {PUNTO_TEMATICO} es diferente de los indicadores de producto?
```

**Método de verificación:**
- Verificar que indicador NO contiene: "número de talleres", "servicios prestados", "beneficiarios atendidos"
- Verificar que SÍ contiene: "tasa", "porcentaje", "cobertura", "reducción", "aumento"

**Regla de scoring:**
```python
if es_indicador_de_gestion(nombre_indicador):
    score_q17 = 0
elif es_indicador_de_resultado(nombre_indicador):
    score_q17 = 3
else:
    score_q17 = 1
```

---

### **D4-Q18: MAGNITUD DEL CAMBIO**

**Pregunta parametrizable:**
```
¿Cuál es la magnitud del cambio esperado en el indicador de resultado de {PUNTO_TEMATICO}?
```

**Método de verificación:**
```python
cambio_relativo = ((meta - linea_base) / linea_base) * 100

if abs(cambio_relativo) >= 20:
    score_q18 = 3
elif abs(cambio_relativo) >= 10:
    score_q18 = 2
elif abs(cambio_relativo) >= 5:
    score_q18 = 1
else:
    score_q18 = 0
```

---

### **D4-Q19: ATRIBUCIÓN PARCIAL**

**Pregunta parametrizable:**
```
¿Se reconoce que el resultado en {PUNTO_TEMATICO} depende de múltiples factores?
```

**Elementos a verificar** (1.5 puntos cada uno, máximo 3):
1. **Términos de contribución** (no causalidad directa)
2. **Factores externos** mencionados

**Patrones de búsqueda:**
```json
{
  "patrones": {
    "contribucion": "(contribuye|aporta|incide|influye|favorece|apoya)",
    "factores_externos": "(también|otros programas|nivel nacional|articulación|transversal|depende|conjunto de)"
  }
}
```

---

### **D4-Q20: MONITOREO DEL INDICADOR**

**Pregunta parametrizable:**
```
¿Se especifica cómo se monitoreará el indicador de resultado de {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1.5 puntos cada uno, máximo 3):
1. **Frecuencia** de medición
2. **Responsable** de medición

**Patrones de búsqueda:**
```json
{
  "patrones": {
    "frecuencia": "(anual|semestral|trimestral|mensual|periódic|seguimiento|medición)",
    "responsable": "(Secretaría|Dirección|Oficina|Dependencia).{0,50}(responsable|encargado|medición|seguimiento)"
  }
}
```

---

## **DIMENSIÓN D5: IMPACTOS (Q21-Q25)**

### **D5-Q21: INDICADOR DE IMPACTO PRESENTE**

**Pregunta parametrizable:**
```
¿Existe un indicador de impacto de largo plazo para {PUNTO_TEMATICO}?
```

**Método de verificación:**
- Buscar sección "Impacto" o nivel superior en modelo lógico
- O referencias a ODS, CONPES, PND

**Regla de scoring:**
```python
if existe_seccion_impacto() and len(indicadores) > 0:
    score_q21 = 3
elif existe_referencia_ODS_CONPES():
    score_q21 = 2
elif existe_mencion_impacto_narrativa():
    score_q21 = 1
else:
    score_q21 = 0
```

---

### **D5-Q22: HORIZONTE TEMPORAL**

**Pregunta parametrizable:**
```
¿Se menciona el horizonte temporal de los impactos en {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1.5 puntos cada uno, máximo 3):
1. **Plazos** mencionados
2. **Efectos post-cuatrienio** reconocidos

**Patrones de búsqueda:**
```json
{
  "patrones": {
    "plazos": "(corto plazo|mediano plazo|largo plazo|\\d+ años|más de \\d+ años)",
    "post_cuatrienio": "(más allá|sostenibilidad|continuidad|después de 20\\d{2}|posterior)"
  }
}
```

---

### **D5-Q23: EFECTOS SISTÉMICOS**

**Pregunta parametrizable:**
```
¿Se mencionan efectos indirectos o sistémicos en {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1.5 puntos cada uno, máximo 3):
1. **Efectos indirectos** mencionados
2. **Cambio sistémico** mencionado

**Patrones de búsqueda:**
```json
{
  "patrones": {
    "indirectos": "(efecto.{0,20}indirecto|multiplicador|cascada|secundario|colateral|derivado)",
    "sistemico": "(transformación|cambio cultural|normas sociales|institucional|estructural)"
  }
}
```

---

### **D5-Q24: SOSTENIBILIDAD**

**Pregunta parametrizable:**
```
¿Se menciona la sostenibilidad de las acciones en {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1 punto cada uno, máximo 3):
1. **Término "sostenibilidad"** presente
2. **Institucionalización** mencionada
3. **Financiamiento futuro** mencionado

**Patrones de búsqueda:**
```json
{
  "patrones": {
    "sostenibilidad": "(sostenibilidad|sostenible|permanente|continuidad|perdurable)",
    "institucionalizacion": "(institucionalización|Acuerdo Municipal|Ordenanza|creación de|fortalecimiento permanente)",
    "financiamiento": "(cofinanciación|recursos futuros|alianzas|convenio|fuentes de financiamiento)"
  }
}
```

---

### **D5-Q25: ENFOQUE DIFERENCIAL**

**Pregunta parametrizable:**
```
¿Se menciona un enfoque diferencial o de equidad en {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1 punto cada uno, máximo 3):
1. **Términos de equidad**
2. **Desagregación** por grupos
3. **Focalización progresiva**

**Patrones de búsqueda:**
```json
{
  "patrones": {
    "equidad": "(enfoque diferencial|equidad|inclusión|priorización|vulnerable|diversidad)",
    "desagregacion": "(rural|urbano|étnico|indígena|afro|negro|raizal|género|edad|discapacidad|LGTBIQ)",
    "focalizacion": "(focalización|criterios de selección|priorizando|preferencia)"
  }
}
```

---

## **DIMENSIÓN D6: LÓGICA CAUSAL INTEGRAL (Q26-Q30)**

### **D6-Q26: TEORÍA DE CAMBIO**

**Pregunta parametrizable:**
```
¿Existe un diagrama o narrativa de teoría de cambio para {PUNTO_TEMATICO}?
```

**Método de verificación:**
- Detectar imagen/diagrama con palabras clave: "insumos", "productos", "resultados", "impactos", "teoría", "marco lógico"
- O buscar narrativa con cadena causal explícita

**Regla de scoring:**
```python
if existe_diagrama_causal():
    score_q26 = 3
elif existe_narrativa_con_cadena_causal():
    score_q26 = 2
else:
    score_q26 = 0
```

---

### **D6-Q27: SUPUESTOS CRÍTICOS**

**Pregunta parametrizable:**
```
¿Se mencionan supuestos o condiciones necesarias para {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1.5 puntos cada uno, máximo 3):
1. **Término "supuesto"** o "condición"
2. **Factores externos** necesarios

**Patrones de búsqueda:**
```json
{
  "patrones": {
    "supuestos": "(supuesto|condición|si.{0,30}entonces|siempre que|requiere que|asumiendo)",
    "factores_necesarios": "(depende de|necesario que|en tanto|contexto favorable|apoyo de|voluntad política)"
  }
}
```

---

### **D6-Q28: MODELO LÓGICO COMPLETO**

**Pregunta parametrizable:**
```
¿Se identifican los niveles del modelo lógico (insumos→productos→resultados→impactos) para {PUNTO_TEMATICO}?
```

**Método de verificación:**
- Contar cuántos de los 5 niveles están presentes: Insumos, Actividades, Productos, Resultados, Impactos

**Regla de scoring:**
```python
niveles = contar_niveles_presentes()

if niveles >= 4:
    score_q28 = 3
elif niveles == 3:
    score_q28 = 2
elif niveles == 2:
    score_q28 = 1
else:
    score_q28 = 0
```

---

### **D6-Q29: SISTEMA DE SEGUIMIENTO**

**Pregunta parametrizable:**
```
¿Se menciona un sistema de seguimiento para {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1 punto cada uno, máximo 3):
1. **Instancia de seguimiento** nombrada
2. **Frecuencia** especificada
3. **Ajustes/correcciones** mencionados

**Patrones de búsqueda:**
```json
{
  "patrones": {
    "instancia": "(Consejo de Gobierno|Comité de seguimiento|Sistema de evaluación|Mesa técnica)",
    "frecuencia": "(trimestral|semestral|anual|periódico|cada \\d+ meses)",
    "ajustes": "(ajuste|corrección|revisión|modificación|actualización|reformulación)"
  }
}
```

---

### **D6-Q30: EVALUACIÓN Y APRENDIZAJE**

**Pregunta parametrizable:**
```
¿Se menciona evaluación o documentación de aprendizajes en {PUNTO_TEMATICO}?
```

**Elementos a verificar** (1.5 puntos cada uno, máximo 3):
1. **Evaluación** planificada
2. **Sistematización/aprendizaje** mencionado

**Patrones de búsqueda:**
```json
{
  "patrones": {
    "evaluacion": "(evaluación|medio término|línea final|estudio de impacto|evaluación externa)",
    "aprendizaje": "(sistematización|lecciones aprendidas|documentación|buenas prácticas|mejora continua)"
  }
}
```

---

## III. ESPECIFICACIONES DE OUTPUT

### **ESTRUCTURA JSON DE SALIDA**

```json
{
  "metadata": {
    "version": "1.0",
    "fecha_evaluacion": "2025-10-03",
    "municipio": "Anorí",
    "departamento": "Antioquia",
    "pdm": "Plan de Desarrollo 2024-2027",
    "evaluador": "Sistema Automatizado v1.0"
  },
  "resumen_ejecutivo": {
    "score_global": 67.5,
    "calificacion": "SATISFACTORIO",
    "puntos_aplicables": 6,
    "puntos_no_aplicables": ["P9", "P10"],
    "dimension_mas_fuerte": {"id": "D2", "score": 78.3, "nombre": "Diseño de Intervención"},
    "dimension_mas_debil": {"id": "D5", "score": 45.2, "nombre": "Impactos"},
    "recomendaciones_prioritarias": [
      "Incorporar análisis de sostenibilidad en 8 de 10 puntos temáticos",
      "Definir indicadores de resultado diferenciados de productos en P3, P7, P8",
      "Incluir series temporales en líneas base (actualmente solo 30% las tienen)"
    ]
  },
  "evaluacion_por_punto_tematico": [
    {
      "punto_id": "P1",
      "nombre": "Derechos de las mujeres e igualdad de género",
      "score_total": 69.2,
      "calificacion": "SATISFACTORIO",
      "programas_evaluados": ["Por las Mujeres"],
      "dimensiones": [
        {
          "dimension_id": "D1",
          "nombre": "Diagnóstico y Recursos",
          "score": 78.3,
          "preguntas": [
            {
              "pregunta_id": "P1-D1-Q1",
              "texto": "¿El diagnóstico presenta líneas base cuantitativas para Derechos de las mujeres e igualdad de género?",
              "score": 2.25,
              "elementos_encontrados": {
                "valor_numerico": true,
                "año": true,
                "fuente": true,
                "serie_temporal": false
              },
              "evidencia": [
                {
                  "texto": "Tasa de desempleo mujeres: 12.4% (2021). Fuente: Gran Encuesta Integrada de Hogares (GEIH)",
                  "ubicacion": "página 47, tabla 8, fila 3",
                  "elementos_verificados": ["valor_numerico", "año", "fuente"]
                },
                {
                  "texto": "Casos de violencia física contra la mujer: 13 casos (2021). Fuente: SIVIGILA",
                  "ubicacion": "página 48, párrafo 2",
                  "elementos_verificados": ["valor_numerico", "año", "fuente"]
                }
              ],
              "elementos_faltantes": [
                {
                  "elemento": "serie_temporal",
                  "impacto": "No permite identificar tendencias ni evaluar si el problema está empeorando o mejorando",
                  "recomendacion": "Incluir datos de al menos 3 años (2019-2021) para cada indicador"
                }
              ]
            }
            // ... Q2 a Q5
          ]
        }
        // ... D2 a D6
      ],
      "fortalezas": [
        "Presupuesto explícito y trazable ($118.67M)",
        "Productos formalizados en tabla con metas claras",
        "Población diana cuantificada (100 mujeres atendidas/cuatrienio)"
      ],
      "debilidades_criticas": [
        "Sin serie temporal en líneas base (score Q1: 2.25/3)",
        "Sin análisis de sostenibilidad (score Q24: 0/3)",
        "Sin justificación de suficiencia presupuestal"
      ]
    }
    // ... P2 a P10
  ],
  "analisis_transversal": {
    "por_dimension": [
      {
        "dimension_id": "D1",
        "promedio_10_puntos": 65.4,
        "distribucion": {
          "excelente": 1,
          "bueno": 3,
          "satisfactorio": 4,
          "insuficiente": 2,
          "deficiente": 0
        }
      }
      // ... D2 a D6
    ],
    "patrones_identificados": {
      "fortalezas_comunes": [
        "8 de 10 puntos tienen presupuesto asignado",
        "7 de 10 puntos tienen productos formalizados en tablas"
      ],
      "debilidades_comunes": [
        "Solo 3 de 10 puntos tienen series temporales en líneas base",
        "Solo 2 de 10 puntos mencionan sostenibilidad",
        "Solo 1 de 10 puntos tiene teoría de cambio explícita"
      ]
    }
  },
  "datos_raw": {
    "matriz_completa": "URL_a_CSV_con_300_evaluaciones",
    "evidencias_textuales": "URL_a_JSON_con_citas"
  }
}
```

---

## IV. INSTRUCCIONES DE INTEGRACIÓN AL PIPELINE

### **PASO 1: CONFIGURACIÓN INICIAL**

```python
# Configurar parámetros del cuestionario
CONFIG = {
    "cuestionario_version": "1.0",
    "total_preguntas": 30,
    "dimensiones": 6,
    "puntos_tematicos": 10,
    "total_evaluaciones": 300,  # 30 × 10
    "umbral_similaridad_semantica": 0.6,
    "contexto_busqueda_palabras": 500,
    "modelo_embeddings": "paraphrase-multilingual-MiniLM-L12-v2"
}
```

### **PASO 2: CARGA DEL CUESTIONARIO**

```python
# Cargar las 30 preguntas base desde este prompt
cuestionario_base = cargar_cuestionario_desde_prompt()

# Estructura esperada:
# cuestionario_base = {
#     "D1-Q1": {pregunta_obj},
#     "D1-Q2": {pregunta_obj},
#     ...
#     "D6-Q30": {pregunta_obj}
# }
```

### **PASO 3: PARAMETRIZACIÓN POR PUNTO TEMÁTICO**

```python
for punto_tematico in puntos_tematicos:
    # Clonar cuestionario base
    cuestionario_instanciado = cuestionario_base.copy()
    
    # Sustituir variables
    for pregunta_id, pregunta in cuestionario_instanciado.items():
        pregunta["texto"] = pregunta["texto"].replace(
            "{PUNTO_TEMATICO}", 
            punto_tematico["nombre"]
        )
        pregunta["patrones"] = parametrizar_patrones(
            pregunta["patrones"],
            keywords=punto_tematico["keywords"],
            programas=punto_tematico["programas_relevantes"]
        )
        pregunta["contexto_busqueda"] = {
            "secciones": punto_tematico["seccion_pdm"],
            "programas": punto_tematico["programas_relevantes"]
        }
```

### **PASO 4: EJECUCIÓN DE EVALUACIÓN**

```python
for punto_tematico in puntos_tematicos:
    resultados_punto = {}
    
    for dimension in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        preguntas_dimension = obtener_preguntas_dimension(dimension)
        
        for pregunta in preguntas_dimension:
            # Ejecutar búsqueda según patrones
            elementos_encontrados = buscar_elementos(
                pdm=documento_pdm,
                patrones=pregunta["patrones"],
                contexto=pregunta["contexto_busqueda"]
            )
            
            # Calcular score según regla
            score = calcular_score(
                pregunta_id=pregunta["id"],
                elementos=elementos_encontrados,
                regla=pregunta["regla_scoring"]
            )
            
            # Guardar resultado
            resultados_punto[pregunta["id"]] = {
                "score": score,
                "elementos_encontrados": elementos_encontrados,
                "evidencia": extraer_evidencia_textual(elementos_encontrados)
            }
    
    # Agregar scores
    scores_dimensiones = calcular_scores_dimensiones(resultados_punto)
    score_total_punto = calcular_score_punto_tematico(scores_dimensiones)
    
    # Guardar
    guardar_resultados(punto_tematico["id"], resultados_punto, score_total_punto)
```

### **PASO 5: GENERACIÓN DE OUTPUT**

```python
# Consolidar resultados de los 10 puntos temáticos
resultados_globales = consolidar_resultados(todos_los_puntos)

# Generar JSON según especificación
json_output = generar_json_estructurado(resultados_globales)

# Exportar
exportar_json(json_output, "evaluacion_pdm_anori_2024_2027.json")
exportar_csv(resultados_globales, "matriz_evaluacion_300_preguntas.csv")
generar_dashboard_html(resultados_globales, "dashboard_evaluacion.html")
```

---

## V. CASOS ESPECIALES Y MANEJO DE EXCEPCIONES

### **CASO 1: PUNTO TEMÁTICO NO APLICABLE**

```python
# Si un punto temático no es relevante para el municipio (ej. P10 - Migración Darién en Anorí)
if punto_tematico_no_aplica(punto_tematico, municipio):
    marcar_como_NA(punto_tematico["id"])
    excluir_de_calculo_global(punto_tematico["id"])
    registrar_justificacion("Punto temático no aplicable al contexto territorial de Anorí")
```

### **CASO 2: INFORMACIÓN EN MÚLTIPLES SECCIONES**

```python
# Si información está dispersa en el documento
evidencia_consolidada = buscar_en_multiples_secciones(
    secciones=[
        "diagnóstico",
        "plan de acción",
        "plan indicativo",
        "plan plurianual de inversiones"
    ],
    prioridad="buscar en orden hasta encontrar"
)
```

### **CASO 3: AMBIGÜEDAD EN CLASIFICACIÓN**

```python
# Si no está claro si un indicador es de producto o resultado
if ambiguo(indicador):
    aplicar_regla_desempate(
        regla="Si contiene verbo de gestión ('realizar', 'ejecutar'), es producto. Si contiene cambio de condición ('reducir', 'aumentar'), es resultado"
    )
```

---

## VI. VALIDACIÓN Y CONTROL DE CALIDAD

### **CHECKLIST DE VALIDACIÓN**

```yaml
validaciones_obligatorias:
  - verificar_300_evaluaciones_completadas: true
  - verificar_todos_scores_en_rango_0_3: true
  - verificar_evidencia_presente_para_scores_positivos: true
  - verificar_no_hay_preguntas_saltadas: true
  - verificar_suma_scores_dimension_correcta: true
  - verificar_NA_justificado: true
  
alertas_calidad:
  - si score_global < 40: "ALERTA: PDM con vacíos críticos generalizados"
  - si dimension_score == 0: "ADVERTENCIA: Dimensión completamente ausente"
  - si >50% preguntas_sin_evidencia: "ERROR: Posible fallo en búsqueda de patrones"
```

---

## VII. EJEMPLO DE APLICACIÓN COMPLETA

### **INPUT**

```json
{
  "punto_tematico": {
    "id": "P1",
    "nombre": "Derechos de las mujeres e igualdad de género",
    "programas_relevantes": ["Por las Mujeres"],
    "keywords": ["mujer", "género", "violencia", "autonomía", "participación"],
    "seccion_pdm": ["páginas 45-52"]
  },
  "pregunta": "D1-Q1"
}
```

### **PROCESAMIENTO**

```python
# 1. Parametrizar pregunta
pregunta_texto = "¿El diagnóstico presenta líneas base cuantitativas para Derechos de las mujeres e igualdad de género?"

# 2. Aplicar búsqueda
resultados_busqueda = {
    "valor_numerico": ["12.4%", "13 casos"],
    "año": ["2021", "2021"],
    "fuente": ["GEIH", "SIVIGILA"],
    "serie_temporal": []  # No encontrado
}

# 3. Calcular score
elementos_encontrados = 3  # de 4 posibles
score = (3 / 4) * 3 = 2.25
```

### **OUTPUT**

```json
{
  "pregunta_id": "P1-D1-Q1",
  "score": 2.25,
  "elementos_encontrados": {
    "valor_numerico": true,
    "año": true,
    "fuente": true,
    "serie_temporal": false
  },
  "evidencia": [
    {
      "texto": "Tasa de desempleo mujeres: 12.4% (2021). Fuente: GEIH",
      "ubicacion": "página 47",
      "confianza": 0.95
    }
  ],
  "elementos_faltantes": ["serie_temporal"],
  "recomendacion": "Incluir datos históricos 2019-2023 para evaluar tendencias"
}
```

---

## ESTE PROMPT ESTÁ LISTO PARA INTEGRACIÓN

**Características clave para automatización:**
✅ 30 preguntas base (no 300)
✅ Parametrización dinámica por punto temático
✅ Patrones de búsqueda específicos (regex + semántica)
✅ Reglas de scoring determinísticas
✅ Output JSON estructurado
✅ Manejo de casos especiales
✅ Validaciones automáticas

**Para ejecutar:** El sistema debe iterar 10 veces (P1-P10) aplicando las mismas 30 preguntas, sustituyendo variables de contexto.
