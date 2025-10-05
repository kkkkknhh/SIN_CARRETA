# PROMPT SISTEMA DE SCORING - ESPECIFICACIÓN TÉCNICA COMPLETA

## METADATA DEL SISTEMA

```yaml
VERSION: 1.0
TIPO: Sistema de Puntuación para Evaluación Causal de PDMs
ESCALA_BASE: 0-3 puntos por pregunta
NIVELES_AGREGACION: 4 (Pregunta → Dimensión → Punto Temático → Global)
TOTAL_EVALUACIONES: 300 (30 preguntas × 10 puntos temáticos)
PRECISION_NUMERICA: 2 decimales
```

---

## I. TAXONOMÍA DE MODALIDADES DE SCORING

### **CLASIFICACIÓN DE LAS 30 PREGUNTAS POR MODALIDAD**

| Modalidad | Preguntas que la usan | Total | Fórmula Base |
|-----------|----------------------|-------|--------------|
| **TIPO A: Conteo Simple (4 elementos)** | Q1, Q6, Q16 | 3 | `(encontrados/4) × 3` |
| **TIPO B: Conteo Simple (3 elementos)** | Q2, Q3, Q4, Q11, Q13, Q24, Q25, Q29 | 8 | `encontrados` (máx 3) |
| **TIPO C: Conteo Binario (2 elementos)** | Q9, Q10, Q19, Q20, Q22, Q23, Q27, Q30 | 8 | `(encontrados/2) × 3` |
| **TIPO D: Ratio Cuantitativo** | Q12, Q14, Q15 | 3 | `f(ratio)` con umbrales |
| **TIPO E: Regla Lógica Simple** | Q5, Q17, Q18, Q21, Q26, Q28 | 6 | `if-then-else` |
| **TIPO F: Análisis Semántico** | Q8 | 1 | Similaridad coseno |
| **N/A: Verificación Agregada** | Q7 (usa suma de elementos de otros tipos) | 1 | Híbrido |

---

## II. ESPECIFICACIONES DETALLADAS POR MODALIDAD

### **MODALIDAD TIPO A: CONTEO SIMPLE (4 ELEMENTOS) → SCORE 0-3**

**Preguntas:** Q1, Q6, Q16

**Fórmula:**
```python
score = (elementos_encontrados / 4) * 3
```

**Tabla de conversión exacta:**
| Elementos Encontrados | Cálculo | Score |
|-----------------------|---------|-------|
| 0 de 4 | (0/4) × 3 | 0.00 |
| 1 de 4 | (1/4) × 3 | 0.75 |
| 2 de 4 | (2/4) × 3 | 1.50 |
| 3 de 4 | (3/4) × 3 | 2.25 |
| 4 de 4 | (4/4) × 3 | 3.00 |

**Ejemplo completo - Q1:**

```python
# ENTRADA
pregunta_id = "P1-D1-Q1"
elementos_esperados = {
    "valor_numerico": False,
    "año": False,
    "fuente": False,
    "serie_temporal": False
}

# BÚSQUEDA
texto_pdm = "Tasa de desempleo mujeres: 12.4% (2021). Fuente: Gran Encuesta Integrada de Hogares (GEIH)"

# EVALUACIÓN
if re.search(r'\d+[.,]?\d*\s*%', texto_pdm):
    elementos_esperados["valor_numerico"] = True  # Encuentra "12.4%"

if re.search(r'20\d{2}', texto_pdm):
    elementos_esperados["año"] = True  # Encuentra "2021"

if re.search(r'(fuente:|Fuente|GEIH)', texto_pdm, re.IGNORECASE):
    elementos_esperados["fuente"] = True  # Encuentra "Fuente: ... GEIH"

if re.search(r'20\d{2}.{0,50}20\d{2}', texto_pdm):
    elementos_esperados["serie_temporal"] = False  # No encuentra 2 años

# CONTEO
elementos_encontrados = sum(elementos_esperados.values())  # = 3

# SCORING
score_q1 = (elementos_encontrados / 4) * 3
score_q1 = (3 / 4) * 3 = 2.25

# OUTPUT
{
    "pregunta_id": "P1-D1-Q1",
    "score": 2.25,
    "elementos_esperados": 4,
    "elementos_encontrados": 3,
    "elementos_faltantes": ["serie_temporal"],
    "detalle": {
        "valor_numerico": True,
        "año": True,
        "fuente": True,
        "serie_temporal": False
    }
}
```

---

### **MODALIDAD TIPO B: CONTEO SIMPLE (3 ELEMENTOS) → SCORE 0-3**

**Preguntas:** Q2, Q3, Q4, Q11, Q13, Q24, Q25, Q29

**Fórmula:**
```python
score = min(elementos_encontrados, 3)  # Cada elemento vale 1 punto, máximo 3
```

**Tabla de conversión exacta:**
| Elementos Encontrados | Score |
|-----------------------|-------|
| 0 de 3 | 0 |
| 1 de 3 | 1 |
| 2 de 3 | 2 |
| 3 de 3 | 3 |

**Ejemplo completo - Q2:**

```python
# ENTRADA
pregunta_id = "P1-D1-Q2"
elementos_esperados = ["poblacion_afectada", "brecha_deficit", "vacios_info"]

# BÚSQUEDA
texto_pdm = """
El municipio tiene una brecha de desempleo de 3.3 puntos porcentuales entre 
mujeres y hombres. Se requiere información sobre discriminación a población LGTBIQ+.
"""

# EVALUACIÓN
encontrados = []

if re.search(r'(brecha|déficit).{0,30}\d+', texto_pdm):
    encontrados.append("brecha_deficit")  # Encuentra "brecha ... 3.3 puntos"

if re.search(r'(sin datos|se requiere.*información|vacío)', texto_pdm):
    encontrados.append("vacios_info")  # Encuentra "se requiere información"

if re.search(r'\d+\s*(personas|habitantes|casos)', texto_pdm):
    # No encuentra población cuantificada
    pass

# SCORING
score_q2 = len(encontrados)  # = 2
score_q2 = 2

# OUTPUT
{
    "pregunta_id": "P1-D1-Q2",
    "score": 2,
    "elementos_esperados": 3,
    "elementos_encontrados": 2,
    "elementos_faltantes": ["poblacion_afectada"],
    "detalle": {
        "poblacion_afectada": False,
        "brecha_deficit": True,
        "vacios_info": True
    }
}
```

---

### **MODALIDAD TIPO C: CONTEO BINARIO (2 ELEMENTOS) → SCORE 0-3**

**Preguntas:** Q9, Q10, Q19, Q20, Q22, Q23, Q27, Q30

**Fórmula:**
```python
score = (elementos_encontrados / 2) * 3
```

**Tabla de conversión exacta:**
| Elementos Encontrados | Cálculo | Score |
|-----------------------|---------|-------|
| 0 de 2 | (0/2) × 3 | 0.0 |
| 1 de 2 | (1/2) × 3 | 1.5 |
| 2 de 2 | (2/2) × 3 | 3.0 |

**Ejemplo completo - Q9:**

```python
# ENTRADA
pregunta_id = "P1-D2-Q9"
elementos_esperados = ["riesgos_explicitos", "factores_externos"]

# BÚSQUEDA
texto_pdm = """
El programa presenta limitaciones en su capacidad operativa. 
Los resultados dependen de la articulación con el nivel nacional.
"""

# EVALUACIÓN
encontrados = 0

if re.search(r'(riesgo|limitación|restricción|dificultad)', texto_pdm):
    encontrados += 1  # Encuentra "limitaciones"

if re.search(r'(depende|articulación|coordinación|nivel nacional)', texto_pdm):
    encontrados += 1  # Encuentra "dependen ... articulación ... nivel nacional"

# SCORING
score_q9 = (encontrados / 2) * 3
score_q9 = (2 / 2) * 3 = 3.0

# OUTPUT
{
    "pregunta_id": "P1-D2-Q9",
    "score": 3.0,
    "elementos_esperados": 2,
    "elementos_encontrados": 2,
    "detalle": {
        "riesgos_explicitos": True,
        "factores_externos": True
    }
}
```

---

### **MODALIDAD TIPO D: RATIO CUANTITATIVO → SCORE 0-3**

**Preguntas:** Q12, Q14, Q15

**Fórmula general:**
```python
ratio = valor_observado / valor_total
score = funcion_escalada(ratio)
```

#### **D.1 - Q12: Proporcionalidad Meta-Problema**

**Fórmula específica:**
```python
ratio_cobertura = meta_producto / magnitud_problema

if ratio_cobertura >= 0.50:      # Cobertura ≥50%
    score_q12 = 3
elif ratio_cobertura >= 0.20:    # Cobertura 20-49%
    score_q12 = 2
elif ratio_cobertura >= 0.05:    # Cobertura 5-19%
    score_q12 = 1
else:                             # Cobertura <5%
    score_q12 = 0
```

**Tabla de umbrales:**
| Ratio de Cobertura | Interpretación | Score |
|--------------------|----------------|-------|
| ≥ 0.50 (50%+) | Cobertura alta | 3 |
| 0.20 - 0.49 | Cobertura media | 2 |
| 0.05 - 0.19 | Cobertura baja | 1 |
| < 0.05 (5%) | Cobertura marginal | 0 |

**Ejemplo completo:**

```python
# ENTRADA
pregunta_id = "P1-D3-Q12"

# DATOS EXTRAÍDOS
magnitud_problema = 500  # Extraído de Q2: "500 casos de violencia/año"
meta_producto = 100      # Extraído de tabla: "100 atenciones cuatrienio"

# AJUSTE TEMPORAL (si es necesario)
# El problema es anual (500/año), la meta es cuatrienio
magnitud_problema_cuatrienio = 500 * 4  # = 2000 casos en 4 años

# CÁLCULO
ratio_cobertura = meta_producto / magnitud_problema_cuatrienio
ratio_cobertura = 100 / 2000 = 0.05  # = 5%

# SCORING
if ratio_cobertura >= 0.50:
    score_q12 = 3
elif ratio_cobertura >= 0.20:
    score_q12 = 2
elif ratio_cobertura >= 0.05:
    score_q12 = 1
else:
    score_q12 = 0

# resultado: ratio = 0.05 → score = 1

# OUTPUT
{
    "pregunta_id": "P1-D3-Q12",
    "score": 1,
    "ratio_cobertura": 0.05,
    "magnitud_problema": 2000,
    "meta_producto": 100,
    "interpretacion": "Cobertura del 5% - baja pero no marginal"
}
```

#### **D.2 - Q14: Productos con Responsable Asignado**

**Fórmula específica:**
```python
ratio = productos_con_responsable / total_productos

if ratio >= 0.90:      # ≥90% tienen responsable
    score_q14 = 3
elif ratio >= 0.70:    # 70-89%
    score_q14 = 2
elif ratio >= 0.40:    # 40-69%
    score_q14 = 1
else:                  # <40%
    score_q14 = 0
```

**Ejemplo completo:**

```python
# ENTRADA
pregunta_id = "P1-D3-Q14"

# DATOS EXTRAÍDOS DE TABLA
tabla_productos = [
    {"producto": "Servicio de orientación", "responsable": "Secretaría de Desarrollo Social"},
    {"producto": "Talleres de formación", "responsable": "Secretaría de Desarrollo Social"},
    {"producto": "Estrategias de sensibilización", "responsable": ""},  # Vacío
    {"producto": "Fortalecimiento institucional", "responsable": "Secretaría de Planeación"}
]

# CONTEO
total_productos = len(tabla_productos)  # = 4
productos_con_responsable = sum(1 for p in tabla_productos if p["responsable"] != "")  # = 3

# CÁLCULO
ratio = productos_con_responsable / total_productos
ratio = 3 / 4 = 0.75  # = 75%

# SCORING
if ratio >= 0.90:
    score_q14 = 3
elif ratio >= 0.70:
    score_q14 = 2
elif ratio >= 0.40:
    score_q14 = 1
else:
    score_q14 = 0

# resultado: ratio = 0.75 → score = 2

# OUTPUT
{
    "pregunta_id": "P1-D3-Q14",
    "score": 2,
    "ratio": 0.75,
    "total_productos": 4,
    "productos_con_responsable": 3,
    "productos_sin_responsable": ["Estrategias de sensibilización"]
}
```

#### **D.3 - Q15: Productos con Justificación Causal**

**Fórmula específica:**
```python
ratio = productos_con_justificacion / total_productos

if ratio >= 0.70:      # ≥70% justificados
    score_q15 = 3
elif ratio >= 0.40:    # 40-69%
    score_q15 = 2
elif ratio >= 0.20:    # 20-39%
    score_q15 = 1
else:                  # <20%
    score_q15 = 0
```

**Ejemplo completo:**

```python
# ENTRADA
pregunta_id = "P1-D3-Q15"

# DATOS
productos = [
    "Servicio de orientación",
    "Talleres de formación",
    "Estrategias de sensibilización"
]

# BÚSQUEDA EN CONTEXTO (±200 palabras de cada producto)
contextos = {
    "Servicio de orientación": "...para reducir la reincidencia de violencia...",
    "Talleres de formación": "...que permitan generar autonomía económica...",
    "Estrategias de sensibilización": "..."  # Sin términos causales cercanos
}

# EVALUACIÓN
terminos_causales = r'(para|con el fin de|contribuye|permite|lograr|reducir|aumentar)'
productos_con_justificacion = 0

for producto, contexto in contextos.items():
    if re.search(terminos_causales, contexto):
        productos_con_justificacion += 1

# productos_con_justificacion = 2 (los primeros 2 tienen términos causales)

# CÁLCULO
ratio = productos_con_justificacion / len(productos)
ratio = 2 / 3 = 0.67  # = 67%

# SCORING
if ratio >= 0.70:
    score_q15 = 3
elif ratio >= 0.40:
    score_q15 = 2
elif ratio >= 0.20:
    score_q15 = 1
else:
    score_q15 = 0

# resultado: ratio = 0.67 → score = 2

# OUTPUT
{
    "pregunta_id": "P1-D3-Q15",
    "score": 2,
    "ratio": 0.67,
    "total_productos": 3,
    "productos_con_justificacion": 2,
    "productos_sin_justificacion": ["Estrategias de sensibilización"]
}
```

---

### **MODALIDAD TIPO E: REGLA LÓGICA SIMPLE → SCORE 0, 1, 2, o 3**

**Preguntas:** Q5, Q17, Q18, Q21, Q26, Q28

Cada pregunta tiene su propia regla lógica específica.

#### **E.1 - Q5: Coherencia Recursos-Productos**

**Regla:**
```python
if presupuesto > 0 and num_productos > 0:
    score_q5 = 3
elif presupuesto > 0 and num_productos == 0:
    score_q5 = 2
elif presupuesto == 0:
    score_q5 = 0
else:
    score_q5 = 0  # Caso no contemplado
```

**Ejemplo:**
```python
# ENTRADA
presupuesto = 118670000  # De Q3
num_productos = 4         # De Q6

# EVALUACIÓN
if presupuesto > 0 and num_productos > 0:
    score_q5 = 3

# OUTPUT
{
    "pregunta_id": "P1-D1-Q5",
    "score": 3,
    "presupuesto": 118670000,
    "num_productos": 4,
    "evaluacion": "Existe presupuesto Y productos definidos"
}
```

#### **E.2 - Q17: Diferenciación Producto-Resultado**

**Regla:**
```python
# Listas de términos
terminos_gestion = ["número de", "cantidad de", "talleres realizados", 
                    "servicios prestados", "personas capacitadas", "atenciones"]
terminos_resultado = ["tasa", "porcentaje", "cobertura", "reducción", 
                      "aumento", "disminución", "prevalencia", "incidencia"]

indicador_resultado = "Tasa de violencia física contra mujeres"

# Evaluación
es_gestion = any(term in indicador_resultado.lower() for term in terminos_gestion)
es_resultado = any(term in indicador_resultado.lower() for term in terminos_resultado)

if es_gestion:
    score_q17 = 0
elif es_resultado:
    score_q17 = 3
else:
    score_q17 = 1  # Ambiguo
```

**Ejemplo:**
```python
# CASO 1: Indicador correcto
indicador = "Tasa de violencia física contra mujeres"
# Contiene "Tasa" → es_resultado = True → score = 3

# CASO 2: Indicador incorrecto
indicador = "Número de mujeres atendidas"
# Contiene "Número de" → es_gestion = True → score = 0

# CASO 3: Indicador ambiguo
indicador = "Situación de las mujeres del municipio"
# No contiene términos claros → score = 1
```

#### **E.3 - Q18: Magnitud del Cambio Esperado**

**Regla:**
```python
cambio_absoluto = meta - linea_base
cambio_relativo = abs((meta - linea_base) / linea_base * 100)

if cambio_relativo >= 20:      # ≥20% de cambio
    score_q18 = 3
elif cambio_relativo >= 10:    # 10-19%
    score_q18 = 2
elif cambio_relativo >= 5:     # 5-9%
    score_q18 = 1
else:                           # <5%
    score_q18 = 0
```

**Ejemplo:**
```python
# ENTRADA
linea_base = 13    # Casos de violencia (2021)
meta = 8           # Meta 2027

# CÁLCULO
cambio_absoluto = 8 - 13 = -5
cambio_relativo = abs((8 - 13) / 13 * 100) = abs(-38.46) = 38.46%

# SCORING
if 38.46 >= 20:
    score_q18 = 3

# OUTPUT
{
    "pregunta_id": "P1-D4-Q18",
    "score": 3,
    "linea_base": 13,
    "meta": 8,
    "cambio_absoluto": -5,
    "cambio_relativo_porcentaje": 38.46,
    "direccion": "reducción"
}
```

**CASOS ESPECIALES:**

```python
# CASO A: Meta = Línea base (sin cambio)
linea_base = 100
meta = 100
cambio_relativo = 0%
score = 0

# CASO B: Cambio en dirección incorrecta (empeora)
# Si el indicador es "casos de violencia" (menor es mejor)
linea_base = 13
meta = 20  # Aumenta (malo)
cambio_relativo = 53.8%
# Aún así score = 3 (magnitud alta), pero registrar alerta

# CASO C: Línea base = 0 (división por cero)
linea_base = 0
meta = 10
cambio_relativo = infinito → usar cambio absoluto
if cambio_absoluto >= 10:
    score = 3
```

#### **E.4 - Q21: Indicador de Impacto Presente**

**Regla:**
```python
tiene_seccion_impacto = False
tiene_indicador_impacto = False
tiene_referencia_ODS = False
tiene_mencion_narrativa = False

# Búsquedas
if "Impacto" in secciones_documento or "Componente" in secciones_documento:
    tiene_seccion_impacto = True
    if len(indicadores_en_seccion) > 0:
        tiene_indicador_impacto = True

if re.search(r'ODS|Objetivo.*Desarrollo Sostenible|CONPES|PND', documento):
    tiene_referencia_ODS = True

if re.search(r'impacto de largo plazo|impacto socioeconómico', documento):
    tiene_mencion_narrativa = True

# Scoring
if tiene_seccion_impacto and tiene_indicador_impacto:
    score_q21 = 3
elif tiene_referencia_ODS:
    score_q21 = 2
elif tiene_mencion_narrativa:
    score_q21 = 1
else:
    score_q21 = 0
```

#### **E.5 - Q26: Teoría de Cambio Presente**

**Regla:**
```python
tiene_diagrama = False
tiene_narrativa_causal = False

# Detectar imagen/diagrama
if detectar_imagen_con_keywords(["insumos", "productos", "resultados", "impactos"]):
    tiene_diagrama = True

# Detectar narrativa
if re.search(r'teoría de cambio|marco lógico|cadena causal|modelo de intervención', documento):
    tiene_narrativa_causal = True

# Scoring
if tiene_diagrama:
    score_q26 = 3
elif tiene_narrativa_causal:
    score_q26 = 2
else:
    score_q26 = 0
```

#### **E.6 - Q28: Modelo Lógico Completo**

**Regla:**
```python
niveles_esperados = ["Insumos", "Actividades", "Productos", "Resultados", "Impactos"]
niveles_encontrados = []

for nivel in niveles_esperados:
    if re.search(nivel, documento, re.IGNORECASE):
        niveles_encontrados.append(nivel)

num_niveles = len(niveles_encontrados)

if num_niveles >= 4:      # 4-5 niveles presentes
    score_q28 = 3
elif num_niveles == 3:    # 3 niveles
    score_q28 = 2
elif num_niveles == 2:    # 2 niveles
    score_q28 = 1
else:                     # 0-1 niveles
    score_q28 = 0
```

---

### **MODALIDAD TIPO F: ANÁLISIS SEMÁNTICO → SCORE 0-3**

**Pregunta:** Q8 únicamente

**Método:** Similaridad coseno entre embeddings de problemas y productos

**Fórmula:**
```python
# 1. Extraer problemas del diagnóstico
problemas = extraer_problemas_diagnostico(seccion_diagnostico)

# 2. Extraer productos
productos = extraer_productos_tabla(tabla_productos)

# 3. Generar embeddings
embeddings_problemas = embedder.encode(problemas)
embeddings_productos = embedder.encode(productos)

# 4. Calcular matriz de similaridad
matriz_similaridad = cosine_similarity(embeddings_problemas, embeddings_productos)

# 5. Para cada problema, verificar si tiene al menos 1 producto relacionado
UMBRAL_SIMILARIDAD = 0.6
problemas_con_producto = 0

for i, problema in enumerate(problemas):
    max_similaridad = max(matriz_similaridad[i])
    if max_similaridad >= UMBRAL_SIMILARIDAD:
        problemas_con_producto += 1

# 6. Calcular ratio
ratio_cobertura = problemas_con_producto / len(problemas)

# 7. Scoring
if ratio_cobertura >= 0.80:
    score_q8 = 3
elif ratio_cobertura >= 0.50:
    score_q8 = 2
elif ratio_cobertura >= 0.30:
    score_q8 = 1
else:
    score_q8 = 0
```

**Ejemplo completo:**

```python
# ENTRADA
problemas = [
    "Alta tasa de violencia de género (13 casos/año)",
    "Baja autonomía económica de las mujeres (tasa desempleo 12.4%)",
    "Escasa participación política de las mujeres"
]

productos = [
    "Servicio de orientación en casos de violencia de género",
    "Mujeres formadas en habilidades empresariales",
    "Estrategias de transformación de imaginarios de género"
]

# GENERACIÓN DE EMBEDDINGS (simulado)
# embeddings_problemas = [[0.2, 0.8, 0.1], [0.7, 0.3, 0.5], [0.1, 0.2, 0.9]]
# embeddings_productos = [[0.3, 0.7, 0.2], [0.8, 0.2, 0.4], [0.2, 0.3, 0.8]]

# MATRIZ DE SIMILARIDAD (simulada)
matriz_similaridad = [
    [0.95, 0.30, 0.25],  # Problema 1 vs Productos 1, 2, 3
    [0.25, 0.88, 0.35],  # Problema 2 vs Productos 1, 2, 3
    [0.20, 0.30, 0.45]   # Problema 3 vs Productos 1, 2, 3
]

# EVALUACIÓN
UMBRAL = 0.6
problemas_con_producto = 0

# Problema 1: max = 0.95 ≥ 0.6 → ✓
# Problema 2: max = 0.88 ≥ 0.6 → ✓
# Problema 3: max = 0.45 < 0.6 → ✗

problemas_con_producto = 2

# RATIO
ratio_cobertura = 2 / 3 = 0.67  # = 67%

# SCORING
if 0.67 >= 0.80:
    score = 3
elif 0.67 >= 0.50:
    score = 2  # ← RESULTADO
elif 0.67 >= 0.30:
    score = 1
else:
    score = 0

# OUTPUT
{
    "pregunta_id": "P1-D2-Q8",
    "score": 2,
    "ratio_cobertura": 0.67,
    "total_problemas": 3,
    "problemas_con_producto": 2,
    "problemas_sin_producto": ["Escasa participación política de las mujeres"],
    "detalle_matching": [
        {
            "problema": "Alta tasa de violencia de género",
            "producto_mejor_match": "Servicio de orientación en casos de violencia de género",
            "similaridad": 0.95
        },
        {
            "problema": "Baja autonomía económica de las mujeres",
            "producto_mejor_match": "Mujeres formadas en habilidades empresariales",
            "similaridad": 0.88
        },
        {
            "problema": "Escasa participación política de las mujeres",
            "producto_mejor_match": "Estrategias de transformación de imaginarios de género",
            "similaridad": 0.45  # Bajo umbral
        }
    ]
}
```

---

### **MODALIDAD HÍBRIDA: Q7**

**Pregunta Q7** combina conteo simple de 3 elementos

**Fórmula:**
```python
score_q7 = min(elementos_encontrados, 3)
```

Similar a TIPO B (Conteo Simple 3 elementos)

---

## III. AGREGACIÓN DE SCORES

### **NIVEL 1: SCORE POR PREGUNTA (Ya especificado arriba)**

Rango: 0.00 - 3.00
Precisión: 2 decimales

---

### **NIVEL 2: SCORE POR DIMENSIÓN**

**Fórmula:**
```python
score_dimension = (suma_scores_preguntas / puntos_maximos_dimension) * 100
```

Donde:
- `suma_scores_preguntas` = suma de los scores de las 5 preguntas de la dimensión
- `puntos_maximos_dimension` = 15 (5 preguntas × 3 puntos máximo)

**Rango:** 0.00 - 100.00 (porcentaje)
**Precisión:** 1 decimal

**Ejemplo completo - Dimensión D1:**

```python
# SCORES DE LAS 5 PREGUNTAS
scores_d1 = {
    "Q1": 2.25,
    "Q2": 2.00,
    "Q3": 3.00,
    "Q4": 1.50,
    "Q5": 3.00
}

# SUMA
suma = sum(scores_d1.values())
suma = 2.25 + 2.00 + 3.00 + 1.50 + 3.00 = 11.75

# SCORING
puntos_maximos = 15  # 5 preguntas × 3 puntos
score_d1 = (suma / puntos_maximos) * 100
score_d1 = (11.75 / 15) * 100 = 78.333...

# REDONDEO
score_d1 = round(78.333, 1) = 78.3%

# OUTPUT
{
    "dimension_id": "D1",
    "nombre": "Diagnóstico y Recursos",
    "score_porcentaje": 78.3,
    "puntos_obtenidos": 11.75,
    "puntos_maximos": 15.0,
    "preguntas": scores_d1
}
```

---

### **NIVEL 3: SCORE POR PUNTO TEMÁTICO**

**Fórmula:**
```python
score_punto_tematico = suma_scores_dimensiones / num_dimensiones
```

Donde:
- `suma_scores_dimensiones` = suma de los 6 scores de dimensiones (ya en %)
- `num_dimensiones` = 6

**Rango:** 0.00 - 100.00 (porcentaje)
**Precisión:** 1 decimal

**Ejemplo completo - Punto Temático P1:**

```python
# SCORES DE LAS 6 DIMENSIONES (en %)
scores_dimensiones = {
    "D1": 78.3,
    "D2": 85.0,
    "D3": 66.7,
    "D4": 70.0,
    "D5": 55.0,
    "D6": 60.0
}

# PROMEDIO
suma = sum(scores_dimensiones.values())
suma = 78.3 + 85.0 + 66.7 + 70.0 + 55.0 + 60.0 = 415.0

score_p1 = suma / 6
score_p1 = 415.0 / 6 = 69.166...

# REDONDEO
score_p1 = round(69.166, 1) = 69.2%

# OUTPUT
{
    "punto_tematico_id": "P1",
    "nombre": "Derechos de las mujeres e igualdad de género",
    "score_porcentaje": 69.2,
    "dimensiones": scores_dimensiones,
    "calificacion": "SATISFACTORIO"
}
```

---

### **NIVEL 4: SCORE GLOBAL DEL PDM**

**Opción A: Promedio Simple (incluye todos los puntos temáticos)**

```python
score_global = suma_scores_puntos_tematicos / 10
```

**Opción B: Promedio Ajustado (excluye N/A)**

```python
puntos_aplicables = [p for p in puntos_tematicos if p.score != "N/A"]
score_global = sum(p.score for p in puntos_aplicables) / len(puntos_aplicables)
```

**Ejemplo - Opción B (recomendada):**

```python
# SCORES DE PUNTOS TEMÁTICOS
scores_puntos = {
    "P1": 69.2,
    "P2": 62.5,
    "P3": 45.8,
    "P4": 72.0,
    "P5": 50.3,
    "P6": 80.5,
    "P7": 30.2,
    "P8": 40.1,
    "P9": "N/A",  # No aplica (sin centro carcelario en Anorí)
    "P10": "N/A"  # No aplica (Darién geográficamente lejano)
}

# FILTRAR APLICABLES
puntos_aplicables = {k: v for k, v in scores_puntos.items() if v != "N/A"}
# puntos_aplicables = P1 a P8 (8 puntos)

# SUMA
suma = sum(puntos_aplicables.values())
suma = 69.2 + 62.5 + 45.8 + 72.0 + 50.3 + 80.5 + 30.2 + 40.1 = 450.6

# PROMEDIO
num_aplicables = len(puntos_aplicables)  # = 8
score_global = suma / num_aplicables
score_global = 450.6 / 8 = 56.325

# REDONDEO
score_global = round(56.325, 1) = 56.3%

# OUTPUT
{
    "score_global": 56.3,
    "calificacion": "SATISFACTORIO",
    "puntos_evaluados": 8,
    "puntos_no_aplicables": 2,
    "puntos_aplicables": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
    "puntos_NA": ["P9", "P10"]
}
```

---

## IV. TABLA MAESTRA DE INTERPRETACIÓN

### **INTERPRETACIÓN DE SCORES (0-100%)**

| Rango (%) | Calificación | Color | Semáforo | Interpretación |
|-----------|--------------|-------|----------|----------------|
| **85-100** | EXCELENTE | 🟢 Verde | 🟢🟢🟢 | Diseño causal robusto. Cumple estándares avanzados de planificación basada en evidencia. |
| **70-84** | BUENO | 🟡 Amarillo | 🟢🟢⚪ | Diseño sólido con vacíos menores. Requiere ajustes específicos identificados. |
| **55-69** | SATISFACTORIO | 🟠 Naranja | 🟢⚪⚪ | Cumple requisitos mínimos. Requiere mejoras sustanciales en dimensiones débiles. |
| **40-54** | INSUFICIENTE | 🔴 Rojo | ⚪⚪⚪ | Vacíos críticos en diseño causal. Alta probabilidad de fallo en implementación. |
| **0-39** | DEFICIENTE | ⚫ Negro | ❌❌❌ | Ausencia de diseño causal verificable. Intervención no fundamentada. |

---

## V. CASOS ESPECIALES Y MANEJO DE EXCEPCIONES

### **CASO 1: Información No Encontrada (Missing Data)**

```python
# Si no se encuentra información para un elemento
elemento_encontrado = None

# SCORING
if elemento_encontrado is None:
    elemento_valor = False  # Se cuenta como NO encontrado
    score_parcial = 0

# NOTA: La ausencia de evidencia NO es evidencia de ausencia, 
# pero para scoring automatizado, se penaliza la no evidencia explícita
```

### **CASO 2: Punto Temático No Aplicable (N/A)**

```python
# Criterios para marcar N/A:
# 1. No hay programas relevantes para el punto temático
# 2. El diagnóstico explícitamente dice que no aplica
# 3. Características territoriales hacen irrelevante el tema

def es_punto_no_aplicable(punto_tematico, municipio):
    # Ejemplo: P10 (Migración Darién) en Anorí
    if punto_tematico.id == "P10" and municipio.departamento != "Chocó":
        return True, "Municipio no limita con la ruta migratoria del Darién"
    
    # Ejemplo: P9 (Privados de libertad) si no hay centro carcelario
    if punto_tematico.id == "P9" and not municipio.tiene_centro_carcelario:
        return True, "No hay centro de reclusión en el municipio"
    
    return False, None

# SCORING
if es_no_aplicable:
    score_punto_tematico = "N/A"
    excluir_de_calculo_global = True
```

### **CASO 3: División por Cero**

```python
# En cálculos de ratio (Q12, Q14, Q15, Q8)
if denominador == 0:
    # Opción A: Asignar score = 0
    score = 0
    nota = "Denominador igual a cero - no se puede calcular ratio"
    
    # Opción B: Marcar como N/A
    score = "N/A"
    nota = "No hay base para calcular ratio (total = 0)"

# Ejemplo: Q12 si magnitud_problema = 0
if magnitud_problema == 0:
    score_q12 = "N/A"
    justificacion = "No se cuantificó la magnitud del problema en diagnóstico"
```

### **CASO 4: Cambio Relativo con Línea Base = 0 (Q18)**

```python
# Si línea base = 0, no se puede calcular cambio relativo
linea_base = 0
meta = 50

# Usar cambio absoluto en su lugar
cambio_absoluto = meta - linea_base  # = 50

# Regla alternativa
if cambio_absoluto >= 20:
    score_q18 = 3
elif cambio_absoluto >= 10:
    score_q18 = 2
elif cambio_absoluto >= 5:
    score_q18 = 1
else:
    score_q18 = 0
```

### **CASO 5: Información Contradictoria**

```python
# Si el PDM tiene información contradictoria en diferentes secciones
# Ejemplo: Diagnóstico dice "500 casos" pero tabla dice "300 casos"

# REGLA: Priorizar información de tablas sobre narrativa
magnitud_problema = valor_en_tabla if valor_en_tabla else valor_en_narrativa

# Registrar alerta
alertas.append({
    "tipo": "contradiccion",
    "pregunta": "Q2",
    "mensaje": "Valores diferentes en diagnóstico (500) y tabla (300). Se usó valor de tabla."
})
```

### **CASO 6: Múltiples Programas para un Punto Temático**

```python
# Si un punto temático tiene múltiples programas
punto_tematico = {
    "id": "P4",
    "programas": ["Educación de Calidad", "Salud para Todos", "Cultura Viva"]
}

# REGLA: Agregar evidencia de TODOS los programas
evidencia_consolidada = []
for programa in punto_tematico.programas:
    evidencia_consolidada.extend(buscar_evidencia(programa))

# Scoring se basa en la consolidación
# Si al menos 1 programa cumple el criterio, se cuenta como encontrado
```

---

## VI. VALIDACIONES Y CONTROLES DE CALIDAD

### **VALIDACIONES OBLIGATORIAS POST-SCORING**

```python
def validar_resultados(resultados):
    errores = []
    advertencias = []
    
    # V1: Todos los scores están en rango válido
    for pregunta, score in resultados.items():
        if score != "N/A" and not (0 <= score <= 3):
            errores.append(f"{pregunta}: score {score} fuera de rango [0-3]")
    
    # V2: Suma de scores de dimensión es correcta
    for dimension in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        preguntas_dim = obtener_preguntas(dimension)
        suma_manual = sum(resultados[p] for p in preguntas_dim if resultados[p] != "N/A")
        suma_registrada = resultados[f"{dimension}_suma"]
        
        if abs(suma_manual - suma_registrada) > 0.01:  # Tolerancia para redondeo
            errores.append(f"{dimension}: suma incorrecta {suma_registrada} vs {suma_manual}")
    
    # V3: Score de dimensión en rango [0-100]
    for dimension in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        score_dim = resultados[f"{dimension}_porcentaje"]
        if not (0 <= score_dim <= 100):
            errores.append(f"{dimension}: porcentaje {score_dim} fuera de rango [0-100]")
    
    # V4: Todas las preguntas fueron evaluadas
    total_preguntas = 300  # 30 × 10
    preguntas_evaluadas = sum(1 for k, v in resultados.items() if k.startswith("P") and "-Q" in k)
    
    if preguntas_evaluadas < total_preguntas:
        advertencias.append(f"Solo {preguntas_evaluadas} de {total_preguntas} preguntas evaluadas")
    
    # V5: Evidencia presente para scores > 0
    for pregunta, score in resultados.items():
        if score > 0 and not resultados.get(f"{pregunta}_evidencia"):
            advertencias.append(f"{pregunta}: score {score} sin evidencia textual")
    
    return {
        "valido": len(errores) == 0,
        "errores": errores,
        "advertencias": advertencias
    }
```

### **ALERTAS DE CALIDAD**

```python
def generar_alertas_calidad(resultados):
    alertas = []
    
    # A1: Score global muy bajo
    if resultados["score_global"] < 40:
        alertas.append({
            "nivel": "CRITICO",
            "mensaje": "Score global < 40% indica vacíos críticos generalizados en el PDM"
        })
    
    # A2: Dimensión completamente ausente
    for dim in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        if resultados[f"{dim}_porcentaje"] == 0:
            alertas.append({
                "nivel": "CRITICO",
                "mensaje": f"Dimensión {dim} completamente ausente (0%)"
            })
    
    # A3: Más de 50% de preguntas sin evidencia
    total_preguntas = 30
    preguntas_sin_evidencia = sum(1 for q in range(1, 31) 
                                   if resultados.get(f"Q{q}_evidencia") == [])
    
    if preguntas_sin_evidencia / total_preguntas > 0.5:
        alertas.append({
            "nivel": "ADVERTENCIA",
            "mensaje": f"{preguntas_sin_evidencia}/{total_preguntas} preguntas sin evidencia. Posible fallo en búsqueda."
        })
    
    # A4: Coherencia interna (ej. Q3 tiene presupuesto pero Q5 dice que no)
    if resultados["Q3_score"] > 0 and resultados["Q5_score"] == 0:
        alertas.append({
            "nivel": "INCONSISTENCIA",
            "mensaje": "Q3 indica presupuesto presente pero Q5 indica ausencia de recursos"
        })
    
    return alertas
```

---

## VII. EJEMPLO COMPLETO DE CÁLCULO END-TO-END

### **ENTRADA: Evaluación de P1 (Derechos de las mujeres)**

```python
# SCORES DE LAS 30 PREGUNTAS (simplificado, solo algunas)
scores_p1 = {
    # Dimensión D1
    "P1-D1-Q1": 2.25,  # Línea base (3 de 4 elementos)
    "P1-D1-Q2": 2.00,  # Magnitud (2 de 3 elementos)
    "P1-D1-Q3": 3.00,  # Presupuesto (3 de 3 elementos)
    "P1-D1-Q4": 1.50,  # Capacidades (1 de 2 elementos = 1.5)
    "P1-D1-Q5": 3.00,  # Coherencia (presupuesto Y productos)
    
    # Dimensión D2
    "P1-D2-Q6": 2.25,  # Tabla productos (3 de 4 columnas)
    "P1-D2-Q7": 3.00,  # Población diana (3 de 3)
    "P1-D2-Q8": 2.00,  # Correspondencia (2 de 3 problemas)
    "P1-D2-Q9": 3.00,  # Riesgos (2 de 2 elementos)
    "P1-D2-Q10": 1.50, # Articulación (1 de 2 elementos)
    
    # Dimensión D3
    "P1-D3-Q11": 1.50, # Estándares (1 de 2 elementos)
    "P1-D3-Q12": 1.00, # Proporcionalidad (ratio 5%)
    "P1-D3-Q13": 3.00, # Cuantificación (2 de 2 elementos)
    "P1-D3-Q14": 2.00, # Responsables (75% productos)
    "P1-D3-Q15": 2.00, # Justificación causal (67%)
    
    # Dimensión D4
    "P1-D4-Q16": 3.00, # Indicador resultado (4 de 4)
    "P1-D4-Q17": 3.00, # Diferenciación (es resultado)
    "P1-D4-Q18": 3.00, # Magnitud cambio (38%)
    "P1-D4-Q19": 1.50, # Atribución (1 de 2)
    "P1-D4-Q20": 1.50, # Monitoreo (1 de 2)
    
    # Dimensión D5
    "P1-D5-Q21": 1.00, # Indicador impacto (solo narrativa)
    "P1-D5-Q22": 0.00, # Horizonte temporal (0 de 2)
    "P1-D5-Q23": 1.50, # Efectos sistémicos (1 de 2)
    "P1-D5-Q24": 0.00, # Sostenibilidad (0 de 3)
    "P1-D5-Q25": 2.00, # Enfoque diferencial (2 de 3)
    
    # Dimensión D6
    "P1-D6-Q26": 0.00, # Teoría de cambio (ausente)
    "P1-D6-Q27": 1.50, # Supuestos (1 de 2)
    "P1-D6-Q28": 2.00, # Modelo lógico (3 niveles)
    "P1-D6-Q29": 3.00, # Sistema seguimiento (3 de 3)
    "P1-D6-Q30": 0.00  # Evaluación (0 de 2)
}
```

### **PASO 1: Calcular Scores de Dimensiones**

```python
# D1: Diagnóstico y Recursos
suma_d1 = 2.25 + 2.00 + 3.00 + 1.50 + 3.00 = 11.75
score_d1 = (11.75 / 15) * 100 = 78.3%

# D2: Diseño de Intervención
suma_d2 = 2.25 + 3.00 + 2.00 + 3.00 + 1.50 = 11.75
score_d2 = (11.75 / 15) * 100 = 78.3%

# D3: Productos
suma_d3 = 1.50 + 1.00 + 3.00 + 2.00 + 2.00 = 9.50
score_d3 = (9.50 / 15) * 100 = 63.3%

# D4: Resultados
suma_d4 = 3.00 + 3.00 + 3.00 + 1.50 + 1.50 = 12.00
score_d4 = (12.00 / 15) * 100 = 80.0%

# D5: Impactos
suma_d5 = 1.00 + 0.00 + 1.50 + 0.00 + 2.00 = 4.50
score_d5 = (4.50 / 15) * 100 = 30.0%

# D6: Lógica Causal
suma_d6 = 0.00 + 1.50 + 2.00 + 3.00 + 0.00 = 6.50
score_d6 = (6.50 / 15) * 100 = 43.3%
```

### **PASO 2: Calcular Score de Punto Temático P1**

```python
suma_dimensiones = 78.3 + 78.3 + 63.3 + 80.0 + 30.0 + 43.3 = 373.2
score_p1 = 373.2 / 6 = 62.2%
```

### **PASO 3: Interpretar**

```python
if score_p1 >= 85:
    calificacion = "EXCELENTE"
elif score_p1 >= 70:
    calificacion = "BUENO"
elif score_p1 >= 55:
    calificacion = "SATISFACTORIO"  # ← RESULTADO
elif score_p1 >= 40:
    calificacion = "INSUFICIENTE"
else:
    calificacion = "DEFICIENTE"
```

### **OUTPUT FINAL**

```json
{
  "punto_tematico_id": "P1",
  "nombre": "Derechos de las mujeres e igualdad de género",
  "score_total": 62.2,
  "calificacion": "SATISFACTORIO",
  "color": "🟠",
  "dimensiones": {
    "D1": {"score": 78.3, "puntos": "11.75/15"},
    "D2": {"score": 78.3, "puntos": "11.75/15"},
    "D3": {"score": 63.3, "puntos": "9.50/15"},
    "D4": {"score": 80.0, "puntos": "12.00/15"},
    "D5": {"score": 30.0, "puntos": "4.50/15"},
    "D6": {"score": 43.3, "puntos": "6.50/15"}
  },
  "fortalezas": [
    "D4 (Resultados): 80% - Indicadores de resultado bien definidos",
    "D1-D2 (Diagnóstico y Diseño): 78.3% - Bases sólidas"
  ],
  "debilidades_criticas": [
    "D5 (Impactos): 30% - Sin análisis de sostenibilidad ni horizontes temporales",
    "D6 (Lógica Causal): 43.3% - Sin teoría de cambio explícita"
  ],
  "recomendaciones": [
    "Prioridad 1: Desarrollar estrategia de sostenibilidad (Q24: 0/3)",
    "Prioridad 2: Elaborar diagrama de teoría de cambio (Q26: 0/3)",
    "Prioridad 3: Especificar horizontes temporales de impacto (Q22: 0/3)"
  ]
}
```

---

## VIII. FÓRMULAS RESUMIDAS - REFERENCIA RÁPIDA

```python
# ========================================
# NIVEL PREGUNTA (0-3 puntos)
# ========================================

# Tipo A (4 elementos):
score = (encontrados / 4) * 3

# Tipo B (3 elementos):
score = min(encontrados, 3)

# Tipo C (2 elementos):
score = (encontrados / 2) * 3

# Tipo D (ratio):
score = funcion_escalada(ratio)  # Ver tablas específicas

# Tipo E (lógica):
score = if-then-else  # Ver reglas específicas

# Tipo F (semántico):
score = funcion_escalada(ratio_cobertura_problemas)

# ========================================
# NIVEL DIMENSIÓN (0-100%)
# ========================================

score_dimension = (suma_5_preguntas / 15) * 100

# ========================================
# NIVEL PUNTO TEMÁTICO (0-100%)
# ========================================

score_punto_tematico = suma_6_dimensiones / 6

# ========================================
# NIVEL GLOBAL (0-100%)
# ========================================

# Opción A: Todos los puntos
score_global = suma_10_puntos / 10

# Opción B: Excluir N/A (RECOMENDADO)
score_global = suma_puntos_aplicables / num_aplicables

# ========================================
# INTERPRETACIÓN
# ========================================

if score >= 85:  return "EXCELENTE"
elif score >= 70: return "BUENO"
elif score >= 55: return "SATISFACTORIO"
elif score >= 40: return "INSUFICIENTE"
else: return "DEFICIENTE"
```

---

## IX. CHECKLIST DE IMPLEMENTACIÓN

```yaml
implementacion_scoring:
  paso_1_configuracion:
    - [ ] Definir precisión numérica (2 decimales para preguntas, 1 para dimensiones)
    - [ ] Configurar umbrales de scoring según tablas
    - [ ] Cargar modelo de embeddings para Q8
  
  paso_2_evaluacion_preguntas:
    - [ ] Aplicar modalidad correcta a cada pregunta
    - [ ] Extraer evidencia textual para cada elemento encontrado
    - [ ] Guardar coordenadas (página, párrafo) de evidencia
    - [ ] Registrar elementos faltantes
  
  paso_3_agregacion:
    - [ ] Sumar scores de preguntas para cada dimensión
    - [ ] Calcular porcentaje de dimensión
    - [ ] Promediar dimensiones para punto temático
    - [ ] Promediar puntos temáticos (excluir N/A)
  
  paso_4_validacion:
    - [ ] Verificar todos los scores en rango válido
    - [ ] Verificar sumas correctas
    - [ ] Verificar evidencia presente para scores > 0
    - [ ] Generar alertas de calidad
  
  paso_5_output:
    - [ ] Generar JSON estructurado
    - [ ] Crear matriz CSV de 300 evaluaciones
    - [ ] Generar dashboard HTML
    - [ ] Compilar recomendaciones priorizadas
```

---

**ESTE PROMPT DEFINE COMPLETAMENTE EL SISTEMA DE SCORING SIN AMBIGÜEDADES**

✅ **Modalidades de scoring claramente diferenciadas**  
✅ **Fórmulas matemáticas explícitas**  
✅ **Ejemplos numéricos completos paso a paso**  
✅ **Reglas de agregación en 4 niveles**  
✅ **Manejo de casos especiales y excepciones**  
✅ **Validaciones y controles de calidad**  
✅ **Tabla de interpretación estandarizada**
