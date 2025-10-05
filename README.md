# MINIMINIMOON

Sistema de Evaluación Causal de Planes de Desarrollo Municipal (PDM)

## Descripción

Este proyecto implementa un sistema automatizado para evaluar la solidez del diseño causal de Planes de Desarrollo Municipal aplicando un cuestionario estandarizado de 30 preguntas a 10 puntos temáticos.

## Componentes principales

### Evaluador PDM
Implementación del sistema de evaluación automatizada de Planes de Desarrollo Municipal.

- Archivo principal: `pdm_evaluator.py`
- Contiene la clase `PDMEvaluator` con métodos para evaluar cada dimensión del cuestionario
- Incluye el cuestionario base completo con 30 preguntas organizadas en 6 dimensiones

## Estructura del cuestionario

El cuestionario está organizado en 6 dimensiones:

1. **D1: Diagnóstico y Recursos** (Q1-Q5)
2. **D2: Diseño de Intervención** (Q6-Q10)
3. **D3: Eje Transversal** (Q11-Q15)
4. **D4: Implementación** (Q16-Q20)
5. **D5: Resultados** (Q21-Q25)
6. **D6: Sostenibilidad** (Q26-Q30)

## Modo de ejecución

Para cada punto temático (P1 a P10):
- Aplicar las 30 preguntas (Q1 a Q30) agrupadas en 6 dimensiones (D1 a D6)
- Evaluar la presencia objetiva de elementos específicos en el documento
- Generar resultados estructurados con scores, evidencia y metadata

## Principios de evaluación

1. **Objetividad**: Verificar presencia/ausencia de elementos, NO interpretar calidad subjetiva
2. **Evidencia obligatoria**: Cada score debe acompañarse de cita textual del PDM
3. **Contextualización**: Las búsquedas deben filtrarse por las secciones y programas relevantes de cada punto temático
4. **Manejo de ausencias**: Si un elemento no se encuentra, score = 0 (no asumir presencia implícita)
5. **N/A explícito**: Si un punto temático no aplica al municipio, marcar como N/A y excluir del cálculo global

## Uso

Para utilizar el sistema de evaluación:

```python
from pdm_evaluator import PDMEvaluator

# Crear instancia del evaluador
evaluator = PDMEvaluator("ruta/al/pdm.pdf")

# Definir puntos temáticos
thematic_points = [
    {
        "id": "P1",
        "nombre": "Derechos de las mujeres e igualdad de género",
        "programas_relevantes": ["Por las Mujeres", "Equidad de Género"],
        "keywords": ["mujer", "género", "violencia de género", "autonomía económica", "participación"],
        "seccion_pdm": ["páginas 45-52", "tablas 8-10"]
    }
    # ... más puntos temáticos
]

# Ejecutar evaluación
results = evaluator.run_evaluation(thematic_points)

# Guardar resultados
import json
with open('resultados.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

# Multi-Component Python Suite

Esta suite reúne módulos para análisis documental, detección de contradicciones, embeddings, validación de grafos y detección de responsabilidades en textos en español. El objetivo es ofrecer cimientos reutilizables para proyectos de planeación territorial y analítica de políticas públicas.

## Panorama general del sistema
- **Motor de contradicciones (`pdm_contra/core.py`)**: orquesta coincidencias léxicas, inferencia NLI española, validación de competencias y análisis de riesgo en un flujo híbrido, complementado con generación de explicaciones trazables.
- **Proveedor de decálogos (`pdm_contra/bridges/decatalogo_provider.py`)**: encapsula la carga de plantillas normalizadas a partir de un paquete YAML autocontenible, asegurando rutas relativas y validaciones estrictas.
- **Módulo de detección de responsabilidades (`responsibility_detector.py`)**: combina NER de spaCy con patrones jerárquicos para priorizar entidades gubernamentales y roles institucionales.
- **Modelo de embeddings con fallback (`embedding_model.py`)**: inicia SentenceTransformer multilingüe con retroceso automático y ajustes específicos por modelo.
- **Patrones de factibilidad (`factibilidad/`)**: evalúa la presencia y proximidad de indicadores de línea base, metas y plazos.
- **Validación de DAG (`dag_validation.py`)**: ejecuta muestreos deterministas para verificar aciclicidad en grafos causales.

## Investigación de cambios recientes
Una revisión de las últimas integraciones resalta tres transformaciones estructurales:
1. **Replanteamiento del motor `ContradictionDetector`**: la clase central ahora inicializa selectores de patrones adversativos, detectores NLI ligeros/pesados, validadores de competencia sectorial y un `RiskScorer`, con opción de codificador SentenceTransformer dependiendo del modo de operación.
2. **Capa de puentes configurables**: el nuevo proveedor de decálogos permite declarar bundles completos vía YAML, validando rutas absolutas/relativas y habilitando `autoload` para despliegues reproducibles.
3. **Narrativas explicativas centralizadas**: `ExplanationTracer` consolida hallazgos en reportes humanamente legibles, registrando trazas temporales para auditoría y comunicación ejecutiva.

Estos cambios sientan las bases para análisis más auditables y facilitan la integración de catálogos externos sin manipulación manual.

## Componentes destacados
### Detección de responsabilidades
- **Integración spaCy**: usa el modelo `es_core_news_sm` para entidades PERSON y ORG.
- **Jerarquía de confianza**: prioriza instituciones gubernamentales y cargos oficiales para puntajes de factibilidad más robustos.
- **Uso básico**:
  ```python
  from responsibility_detector import ResponsibilityDetector

  detector = ResponsibilityDetector()
  texto = "La Alcaldía Municipal coordinará con la Secretaría de Salud el programa de vacunación."
  resultado = detector.calculate_responsibility_score(texto)
  ```

### Embeddings resilientes
- **Fallback automático**: intenta `multilingual-e5-base` y retrocede a `all-MiniLM-L6-v2` cuando se activa el modo liviano o ante fallos de red.
- **Cómputo de similitud**: expone utilidades para codificar lotes y evaluar similitud coseno.

### Factibilidad y planeación
- **Detección de patrones**: reconoce vocabulario de línea base, metas y plazos con ventanas de proximidad configurables.
- **Puntaje compuesto**: combina densidad, clusters y cobertura de patrones para priorizar textos accionables.

### Validación de grafos causales
- **Monte Carlo determinista**: usa semillas reproducibles para evaluar aciclicidad en estructuras PDM.
- **Interpretación estadística**: reporta `p-values` y cobertura empírica para contextualizar resultados.

## Decálogos y decatalogos
El repositorio incluye un bundle curado de decálogos normativos y técnicos que alimenta tanto a los evaluadores industriales
como al motor de contradicciones:

- **Proveedor central (`pdm_contra/bridges/decatalogo_provider.py`)**: lee `pdm_contra/config/decalogo.yaml`, resuelve rutas
  relativas y valida que existan las versiones limpias empaquetadas en `out/` (`decalogo-full.latest.clean.json`,
  `decalogo-industrial.latest.clean.json`, `dnp-standards.latest.clean.json`, además del `crosswalk.latest.json`).
- **Contratos y validación**: `pdm_contra/bridges/decalogo_loader_adapter.py` aplica los esquemas JSON incluidos en `schemas/`
  para asegurar integridad de dominios, clusters y crosswalks antes de exponer el bundle como diccionario Python.
- **Consumidores principales**: `Decatalogo_principal.py` construye el contexto industrial completo y `Decatalogo_evaluador.py`
  orquesta la evaluación de evidencias integrando contradicciones, responsabilidades, patrones de factibilidad y señales
  monetarias.
- **Uso rápido**:
  ```python
  from pdm_contra.bridges.decatalogo_provider import provide_decalogos

  bundle = provide_decalogos()
  print(bundle["version"], bundle["domains"])  # Versionado coherente en los tres catálogos
  ```

Los archivos fuente originales (`DECALOGO_FULL.json`, `decalogo_industrial.json`, `DNP_STANDARDS.json`) se conservan en la
raíz para trazabilidad y pueden regenerarse a los formatos limpios siguiendo los pipelines del directorio `out/` cuando se
actualicen las matrices normativas.

## Guía de inicio paso a paso

1. **Verifica los prerrequisitos del sistema**
   - Asegúrate de contar con **Python 3.8 o superior** instalado. Compruébalo ejecutando `python3 --version` en tu terminal.
   - Confirma que tienes disponible `pip` (normalmente se instala junto con Python) mediante `python3 -m pip --version`.
   - Si necesitas clonar el repositorio, valida que `git` esté disponible con `git --version`.

2. **Clona el repositorio (opcional si ya lo tienes localmente)**
   ```bash
   git clone https://github.com/<tu-organizacion>/MINIMINIMOON.git
   cd MINIMINIMOON
   ```

3. **Crea y activa un entorno virtual aislado**
   - En sistemas Unix (Linux/macOS):
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - En Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   Una vez activado el entorno virtual, el prompt mostrará un prefijo `venv`. Manténlo activo mientras trabajes con este proyecto.

4. **Actualiza `pip` dentro del entorno virtual (recomendado)**
   ```bash
   python -m pip install --upgrade pip
   ```

5. **Instala las dependencias del proyecto**
   ```bash
   pip install -r requirements.txt
   ```

6. **Descarga el modelo de spaCy necesario para la detección de responsabilidades**
   ```bash
   python -m spacy download es_core_news_sm
   ```
   Si estás detrás de un proxy o sin acceso a internet, deja documentado el incidente y ejecuta el proyecto en modo degradado (algunos módulos continuarán funcionando, pero la detección de responsabilidades requerirá el modelo descargado previamente).

7. **Comprueba que la instalación fue correcta**
   Ejecuta una prueba de humo mínima para verificar que los módulos críticos se importan sin errores:
   ```bash
   python -c "import embedding_model, responsibility_detector, dag_validation; print('Componentes importados correctamente')"
   ```

Con estos pasos completados ya puedes explorar los módulos individuales o ejecutar las suites de pruebas descritas a continuación.

## Pruebas
```bash
python3 -m pytest test_embedding_model.py -v         # Embeddings
python3 -m pytest test_responsibility_detector.py -v  # Responsabilidad
python3 test_factibilidad.py                          # Patrones de factibilidad
python3 test_dag_validation.py 2>/dev/null || echo "DAG validation opcional"
python3 validate.py 2>/dev/null || echo "Suite integral opcional"
```

## Ejemplos rápidos
```python
from embedding_model import create_embedding_model

modelo = create_embedding_model()
info = modelo.get_model_info()
print(info["model_name"], info["embedding_dimension"], info["is_fallback"])

sentencias = ["Hola mundo", "Embeddings resilientes"]
embeddings = modelo.encode(sentencias)
```

```python
from factibilidad import FactibilidadScorer

scorer = FactibilidadScorer(proximity_window=500)
texto = "La línea base actual es 100 hogares. Buscamos 250 hogares a diciembre de 2024."
print(scorer.score_text(texto))
```

## Stack de visualizaciones minimalistas (inspirado en Omar Rayo)
Para reportar hallazgos con una estética geométrica y contrastada reminiscentes de Omar Rayo se propone el siguiente stack avanzado:

1. **Modelado declarativo con Vega-Lite/Altair**: define gramáticas de gráficos utilizando combinaciones de blanco, negro y acentos cromáticos puntuales para enfatizar patrones y contradicciones.
2. **Renderizado interactivo con Plotly Dash o Panel**: integra gráficos responsivos alimentados por los análisis de `pdm_contra`, permitiendo resaltar segmentos críticos con trazos ortogonales y sombras suaves.
3. **Diseño tipográfico y espacial**: aplicar grids modulares y tipografías sans-serif sobrias (p. ej. Inter, Work Sans) con amplio espaciado negativo para reforzar la abstracción geométrica.
4. **Sistema de anotaciones**: superponer overlays discretos que conecten hallazgos de riesgo con trazos lineales inspirados en grabados de Rayo, facilitando narrativas ejecutivas.
5. **Automatización de paletas**: centralizar la paleta en utilidades reutilizables (hex #000000, #FFFFFF, acentos carmín/amarillo) y exponer toggles de accesibilidad (alto contraste, modos daltónicos).

Este enfoque convierte los análisis numéricos en tableros interpretables sin sacrificar rigor, alineando la identidad visual con el minimalismo geométrico característico.
