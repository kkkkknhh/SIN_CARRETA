# SOLUCIÓN: Problema de Espacio en Disco

## 🚨 Problema
Error durante instalación: `[Errno 28] No space left on device`

## ✅ Soluciones (en orden de prioridad)

### Opción 1: Liberar Espacio en Disco (Recomendado)

```bash
# Ver espacio disponible
df -h

# Limpiar caché de Homebrew (si usas Homebrew)
brew cleanup -s

# Limpiar caché de pip
pip cache purge

# Limpiar archivos temporales de macOS
sudo rm -rf /private/var/log/asl/*.asl
sudo rm -rf /Library/Caches/*
rm -rf ~/Library/Caches/*

# Vaciar papelera
rm -rf ~/.Trash/*

# Ver qué ocupa más espacio
du -sh ~/Music/MINIMINIMOON-main/*
```

### Opción 2: Instalar Dependencias Mínimas

Crea `requirements-minimal.txt` con solo las dependencias críticas:

```bash
# Core necesario para ejecutar el sistema
numpy>=1.21.0,<2.0.0
torch>=2.0.0,<3.0.0
sentence-transformers>=2.2.0
pandas>=1.3.0,<3.0.0
networkx>=2.6.0,<4.0.0
spacy>=3.5.0
pyyaml>=6.0.0
```

Luego instalar:
```bash
pip install -r requirements-minimal.txt
python -m spacy download es_core_news_sm  # Solo modelo pequeño
```

### Opción 3: Usar Instalación en Otro Disco

```bash
# Crear entorno virtual en otro disco con más espacio
python3 -m venv /Volumes/OtroDisco/miniminimoon_venv

# Activar
source /Volumes/OtroDisco/miniminimoon_venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 4: Instalar Solo lo Necesario Paso a Paso

```bash
# Activar entorno existente
source venv/bin/activate

# Instalar en grupos pequeños
pip install numpy scipy scikit-learn
pip install pandas
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU only, más ligero
pip install sentence-transformers
pip install spacy
pip install networkx matplotlib
pip install pytest

# Modelos de Spacy (solo esencial)
python -m spacy download es_core_news_sm
```

### Opción 5: Usar CPU-only Torch (Más Ligero)

```bash
# Desinstalar torch si existe
pip uninstall torch

# Instalar versión CPU (mucho más pequeña ~200MB vs ~2GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 🔍 Verificar Espacio Necesario

Espacio estimado requerido:
- **Entorno virtual completo**: ~3-4 GB
- **Entorno mínimo (CPU-only)**: ~1-1.5 GB
- **Modelos de Spacy**: ~100-500 MB

## 📊 Diagnóstico Actual

```bash
# Ver cuánto espacio tienes
df -h | grep /Users

# Ver tamaño del proyecto actual
du -sh ~/Music/MINIMINIMOON-main/

# Ver tamaño del venv
du -sh ~/Music/MINIMINIMOON-main/venv/
```

## ✅ Próximos Pasos

1. **Liberar espacio** usando las opciones arriba
2. **Reintentar instalación**:
   ```bash
   cd ~/Music/MINIMINIMOON-main
   
   # Limpiar venv incompleto
   rm -rf venv
   
   # Ejecutar setup nuevamente
   bash setup_environment.sh
   ```

3. **Verificar instalación**:
   ```bash
   python test_critical_flows.py
   ```

## 🎯 Instalación Mínima Funcional

Si el espacio es muy limitado, esta es la configuración mínima:

```bash
# Crear venv
python3 -m venv venv
source venv/bin/activate

# Instalar core
pip install numpy pandas networkx pyyaml pytest

# Instalar torch CPU-only (ligero)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Instalar NLP básico
pip install spacy
python -m spacy download es_core_news_sm

# Listo para tests básicos
python test_critical_flows.py
```

Esta configuración mínima debería ocupar **menos de 1.5 GB**.

