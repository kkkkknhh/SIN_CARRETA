from setuptools import setup, find_packages

setup(
    name="responsibility-detector",
    version="1.0.0",
    description="Responsibility detection using spaCy NER and lexical rules",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.4.0",
        "spacy-transformers>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ]
    },
    python_requires=">=3.7",
)
