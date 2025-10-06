import nltk
import ssl

# Solución para el error [SSL: CERTIFICATE_VERIFY_FAILED] en macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Descargando datos de NLTK (punkt, stopwords, wordnet, averaged_perceptron_tagger)...")

packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for package in packages:
    try:
        print(f"Descargando {package}...")
        nltk.download(package)
        print(f"{package} descargado exitosamente.")
    except Exception as e:
        print(f"Error descargando {package}: {e}")

print("\nProceso de descarga de datos de NLTK completado.")

