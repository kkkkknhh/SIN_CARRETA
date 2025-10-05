#!/bin/bash

# Eliminar configuraciones de inspección Python 2.7
find .idea -name "*.xml" -exec sed -i '' '/<inspection_tool class="Python.*Compatibility"/,/<\/inspection_tool>/d' {} \; 2>/dev/null
find .idea -name "*.xml" -exec sed -i '' '/Python.*2\.7.*compatibility/d' {} \; 2>/dev/null
find .idea -name "*.xml" -exec sed -i '' '/languageLevel.*2\.7/d' {} \; 2>/dev/null

# Forzar Python 3.11 en misc.xml
if [ -f ".idea/misc.xml" ]; then
  sed -i '' 's/languageLevel="[^"]*"/languageLevel="JDK_11"/g' .idea/misc.xml
fi

# Eliminar perfiles de inspección completos si existen
rm -rf .idea/inspectionProfiles/ 2>/dev/null

echo "Eliminadas todas las inspecciones Python 2.7"
