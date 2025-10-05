#!/bin/bash
set -e

VENV_PY="$HOME/.venvs/policy311/bin/python"
echo "Configurando DataSpell en: $(pwd)"
echo "Para usar: $VENV_PY"

# Backup
zip -qr ".idea_backup_$(date +%s).zip" ".idea" 2>/dev/null || true

# Limpiar configuraciones tóxicas
if [ -f ".idea/workspace.xml" ]; then
  sed -i '' 's/ADD_CONTENT_ROOTS" value="true"/ADD_CONTENT_ROOTS" value="false"/g' .idea/workspace.xml
  sed -i '' 's/ADD_SOURCE_ROOTS" value="true"/ADD_SOURCE_ROOTS" value="false"/g' .idea/workspace.xml
  echo "✓ workspace.xml limpio"
fi

# Limpiar variables de entorno tóxicas en todas las configuraciones
find .idea -name "*.xml" -exec sed -i '' '/PYTHONHOME/d; /PYTHONPATH/d' {} \; 2>/dev/null || true
echo "✓ Variables PYTHONHOME/PYTHONPATH eliminadas"

echo "Listo. Abre DataSpell y configura el intérprete a: $VENV_PY"
