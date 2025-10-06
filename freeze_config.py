#!/usr/bin/env python3
"""
Script para ejecutar freeze_configuration sin importar el orquestador completo
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime

def freeze_configuration_minimal():
    """Crea snapshot de inmutabilidad de archivos críticos"""

    config_files = [
        'DECALOGO_FULL.json',
        'decalogo_industrial.json',
        'dnp-standards.latest.clean.json',
        'RUBRIC_SCORING.json'
    ]

    snapshot = {
        'version': '2.0',
        'timestamp': datetime.utcnow().isoformat(),
        'verify': True,
        'files': {}
    }

    for filename in config_files:
        filepath = Path(filename)
        if filepath.exists():
            with open(filepath, 'rb') as f:
                content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
                snapshot['files'][filename] = {
                    'sha256': file_hash,
                    'size': len(content)
                }
            print(f"✓ {filename}: {file_hash[:16]}...")
        else:
            print(f"⚠ {filename}: NO ENCONTRADO")

    # Calcular hash del snapshot
    snapshot_str = json.dumps(snapshot, sort_keys=True)
    snapshot_hash = hashlib.sha256(snapshot_str.encode()).hexdigest()
    snapshot['snapshot_hash'] = snapshot_hash

    # Guardar snapshot
    with open('.immutability_snapshot.json', 'w') as f:
        json.dump(snapshot, f, indent=2)

    print(f"\n✓ Configuration frozen: {snapshot_hash[:16]}...")
    print(f"  Files: {list(snapshot['files'].keys())}")
    return snapshot

if __name__ == '__main__':
    freeze_configuration_minimal()

