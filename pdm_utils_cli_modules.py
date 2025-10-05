"""
Document ingestion module for PDM files.
Supports PDF, DOCX, HTML, and plain text.
"""

import json
import logging
import re
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import html2text
import typer
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from packaging.version import parse
from pypdf import PdfReader
from rich.console import Console
from rich.table import Table

from pdm_contra.core import ContradictionDetector
from pdm_contra.ingest.loader import PDMLoader
from pdm_contra.utils.guard_novelty import enforce_novelty

logger = logging.getLogger(__name__)


class PDMLoader:
    """
    Loader for PDM documents in various formats.
    """

    def __init__(self):
        """Initialize document loader."""
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True

    def load(self, path: Path) -> Dict[str, Any]:
        """
        Load document from file path.

        Args:
            path: Path to document file

        Returns:
            Dictionary with document text and metadata
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(path)
        elif suffix in [".docx", ".doc"]:
            return self._load_docx(path)
        elif suffix in [".html", ".htm"]:
            return self._load_html(path)
        elif suffix in [".txt", ".md"]:
            return self._load_text(path)
        else:
            # Try as text file
            logger.warning(f"Unknown file type {suffix}, attempting text load")
            return self._load_text(path)

    def _load_pdf(self, path: Path) -> Dict[str, Any]:
        """Load PDF document."""
        try:
            reader = PdfReader(path)

            # Extract text from all pages
            text_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_pages.append({"page": i + 1, "text": page_text})

            # Combine text
            full_text = "\n\n".join(p["text"] for p in text_pages)

            # Extract metadata
            metadata = {
                "format": "pdf",
                "pages": len(reader.pages),
                "file": str(path),
                "title": reader.metadata.title if reader.metadata else None,
                "author": reader.metadata.author if reader.metadata else None,
            }

            # Try to extract sections
            sections = self._extract_sections(full_text)

            return {
                "text": full_text,
                "pages": text_pages,
                "sections": sections,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Error loading PDF {path}: {e}")
            raise

    def _load_docx(self, path: Path) -> Dict[str, Any]:
        """Load DOCX document."""
        try:
            doc = DocxDocument(path)

            # Extract paragraphs with formatting info
            paragraphs = []
            current_section = ""
            sections = {}

            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue

                # Check if it's a heading
                if para.style and "Heading" in para.style.name:
                    current_section = text
                    sections[current_section] = []

                paragraphs.append(text)

                if current_section:
                    sections[current_section].append(text)

            # Combine text
            full_text = "\n\n".join(paragraphs)

            # Metadata
            metadata = {
                "format": "docx",
                "file": str(path),
                "paragraphs": len(paragraphs),
                "sections_found": len(sections),
            }

            # Extract core properties if available
            if hasattr(doc, "core_properties"):
                props = doc.core_properties
                metadata.update(
                    {
                        "title": props.title,
                        "author": props.author,
                        "created": str(props.created) if props.created else None,
                    }
                )

            return {
                "text": full_text,
                "paragraphs": paragraphs,
                "sections": sections if sections else self._extract_sections(full_text),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Error loading DOCX {path}: {e}")
            raise

    def _load_html(self, path: Path) -> Dict[str, Any]:
        """Load HTML document."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()

            # Convert to text
            text = self.html_converter.handle(str(soup))

            # Try to extract sections from headers
            sections = {}
            for header in soup.find_all(["h1", "h2", "h3"]):
                section_name = header.get_text().strip()
                sections[section_name] = []

                # Get content until next header
                for sibling in header.find_next_siblings():
                    if sibling.name in ["h1", "h2", "h3"]:
                        break
                    sections[section_name].append(sibling.get_text().strip())

            metadata = {
                "format": "html",
                "file": str(path),
                "title": soup.title.string if soup.title else None,
            }

            return {
                "text": text,
                "sections": sections if sections else self._extract_sections(text),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Error loading HTML {path}: {e}")
            raise

    def _load_text(self, path: Path) -> Dict[str, Any]:
        """Load plain text document."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            metadata = {"format": "text", "file": str(path), "size": len(text)}

            sections = self._extract_sections(text)

            return {"text": text, "sections": sections, "metadata": metadata}

        except Exception as e:
            logger.error(f"Error loading text file {path}: {e}")
            raise

    @staticmethod
    def _extract_sections(text: str) -> Dict[str, str]:
        """
        Extract sections from text using common PDM patterns.

        Args:
            text: Full document text

        Returns:
            Dictionary mapping section names to content
        """
        sections = {}

        # Common PDM section patterns
        patterns = {
            "presentacion": r"(?i)(presentaci[óo]n|introducci[óo]n)",
            "diagnostico": r"(?i)(diagn[óo]stico|an[áa]lisis\s+situacional|contexto)",
            "vision": r"(?i)(visi[óo]n\s+(?:de\s+desarrollo)?)",
            "mision": r"(?i)(misi[óo]n\s+(?:institucional)?)",
            "objetivos": r"(?i)(objetivos?\s+(?:estrat[ée]gicos?|generales?)?)",
            "ejes": r"(?i)(ejes?\s+(?:estrat[ée]gicos?|tem[áa]ticos?)?)",
            "programas": r"(?i)(programas?\s+(?:y\s+proyectos?)?)",
            "metas": r"(?i)(metas?\s+(?:de\s+resultado|de\s+producto)?)",
            "indicadores": r"(?i)(indicadores?\s+(?:de\s+gesti[óo]n|de\s+impacto)?)",
            "presupuesto": r"(?i)(presupuesto|plan\s+plurianual|recursos)",
            "seguimiento": r"(?i)(seguimiento|evaluaci[óo]n|monitoreo)",
        }

        # Find all section headers
        found_sections = []
        for section_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                found_sections.append(
                    {
                        "type": section_type,
                        "start": match.start(),
                        "end": match.end(),
                        "header": match.group(),
                    }
                )

        # Sort by position
        found_sections.sort(key=lambda x: x["start"])

        # Extract content between sections
        for i, section in enumerate(found_sections):
            # Content starts after header
            content_start = section["end"]

            # Content ends at next section or document end
            if i < len(found_sections) - 1:
                content_end = found_sections[i + 1]["start"]
            else:
                content_end = len(text)

            # Extract and clean content
            content = text[content_start:content_end].strip()

            # Limit content length
            if len(content) > 5000:
                content = content[:5000] + "..."

            if content:
                sections[section["type"]] = content

        return sections


# pdm_contra/explain/tracer.py
"""
Explanation and traceability module for PDM analysis.
"""

logger = logging.getLogger(__name__)


class ExplanationTracer:
    """
    Generate human-readable explanations and audit traces.
    """

    def __init__(self, language: str = "es"):
        """Initialize explanation tracer."""
        self.language = language
        self.trace_log = []

    def generate_explanations(
        self,
        contradictions: List[Any],
        competence_issues: List[Any],
        agenda_gaps: List[Any],
    ) -> List[str]:
        """
        Generate explanations for all findings.

        Args:
            contradictions: List of contradictions found
            competence_issues: List of competence issues
            agenda_gaps: List of agenda alignment gaps

        Returns:
            List of human-readable explanations
        """
        explanations = []

        # Explain contradictions
        if contradictions:
            explanations.append(self._explain_contradictions(contradictions))

        # Explain competence issues
        if competence_issues:
            explanations.append(self._explain_competences(competence_issues))

        # Explain agenda gaps
        if agenda_gaps:
            explanations.append(self._explain_agenda(agenda_gaps))

        # Add summary
        if any([contradictions, competence_issues, agenda_gaps]):
            explanations.append(
                self._generate_summary(
                    len(contradictions), len(
                        competence_issues), len(agenda_gaps)
                )
            )

        return explanations

    @staticmethod
    def _explain_contradictions(contradictions: List[Any]) -> str:
        """Generate explanation for contradictions."""
        high_risk = [
            c
            for c in contradictions
            if hasattr(c, "risk_level") and "high" in str(c.risk_level).lower()
        ]

        explanation = (
            f"Se detectaron {len(contradictions)} contradicciones en el documento:\n"
        )

        if high_risk:
            explanation += (
                f"- {len(high_risk)} de alto riesgo que requieren atención inmediata\n"
            )

        # Group by type if possible
        types = {}
        for c in contradictions:
            if hasattr(c, "type"):
                type_str = str(c.type)
                types[type_str] = types.get(type_str, 0) + 1

        if types:
            explanation += "Tipos de contradicciones encontradas:\n"
            for type_name, count in types.items():
                explanation += f"  • {type_name}: {count} casos\n"

        return explanation

    @staticmethod
    def _explain_competences(issues: List[Any]) -> str:
        """Generate explanation for competence issues."""
        explanation = f"Se identificaron {len(issues)} problemas de competencias:\n"

        # Group by sector
        sectors = {}
        for issue in issues:
            if isinstance(issue, dict):
                sector = issue.get("sector", "general")
            else:
                sector = getattr(issue, "sector", "general")
            sectors[sector] = sectors.get(sector, 0) + 1

        if sectors:
            explanation += "Distribución por sector:\n"
            for sector, count in sectors.items():
                explanation += f"  • {sector.title()}: {count} casos\n"

        # Identify overreach cases
        overreach = [
            i
            for i in issues
            if (isinstance(i, dict) and "overreach" in i.get("type", ""))
            or (hasattr(i, "competence_type") and "overreach" in str(i.competence_type))
        ]

        if overreach:
            explanation += (
                f"\n⚠️ {len(overreach)} casos de posible extralimitación de funciones\n"
            )

        return explanation

    @staticmethod
    def _explain_agenda(gaps: List[Any]) -> str:
        """Generate explanation for agenda gaps."""
        explanation = (
            f"Se encontraron {len(gaps)} brechas en la alineación de agenda:\n"
        )

        # Group by severity
        severities = {"low": 0, "medium": 0, "high": 0}
        for gap in gaps:
            if isinstance(gap, dict):
                severity = gap.get("severity", "medium")
            else:
                severity = getattr(gap, "severity", "medium")
            severities[severity] = severities.get(severity, 0) + 1

        if any(severities.values()):
            explanation += "Severidad de las brechas:\n"
            if severities["high"] > 0:
                explanation += (
                    f"  • Alta: {severities['high']} (requieren acción inmediata)\n"
                )
            if severities["medium"] > 0:
                explanation += f"  • Media: {severities['medium']}\n"
            if severities["low"] > 0:
                explanation += f"  • Baja: {severities['low']}\n"

        return explanation

    @staticmethod
    def _generate_summary(
        n_contradictions: int, n_competences: int, n_agenda: int
    ) -> str:
        """Generate overall summary."""
        total = n_contradictions + n_competences + n_agenda

        summary = "\n📊 RESUMEN EJECUTIVO:\n"
        summary += f"Total de hallazgos: {total}\n"

        if total == 0:
            summary += (
                "✅ No se encontraron problemas significativos en el PDM analizado."
            )
        elif total <= 5:
            summary += "⚠️ Se encontraron algunos problemas que requieren revisión."
        elif total <= 15:
            summary += "⚠️ Se identificaron múltiples áreas de mejora importantes."
        else:
            summary += "🚨 El PDM requiere revisión sustancial antes de su aprobación."

        # Recommendations
        summary += "\n\nRECOMENDACIONES PRINCIPALES:\n"

        if n_contradictions > 0:
            summary += "1. Revisar y reconciliar las contradicciones identificadas\n"
        if n_competences > 0:
            summary += "2. Ajustar acciones a las competencias municipales\n"
        if n_agenda > 0:
            summary += "3. Fortalecer la cadena de alineación estratégica\n"

        return summary

    def add_trace(
        self, action: str, details: Dict[str, Any], timestamp: Optional[datetime] = None
    ):
        """
        Add entry to trace log for audit purposes.

        Args:
            action: Action performed
            details: Details of the action
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.trace_log.append(
            {"timestamp": timestamp.isoformat(), "action": action,
             "details": details}
        )

    def get_trace_report(self) -> str:
        """Get formatted trace report."""
        report = "REGISTRO DE TRAZABILIDAD\n"
        report += "=" * 50 + "\n"

        for entry in self.trace_log:
            report += f"\n[{entry['timestamp']}] {entry['action']}\n"
            for key, value in entry["details"].items():
                report += f"  {key}: {value}\n"

        return report


# pdm_contra/utils/guard_novelty.py
"""
Dependency version guard to ensure use of novel libraries.
"""

logger = logging.getLogger(__name__)

# Required package versions (2024+)
REQUIRED_PACKAGES = {
    "sentence-transformers": "3.3.0",  # Nov 2024
    "transformers": "4.46.0",  # Oct 2024
    "torch": "2.5.0",  # Oct 2024
    "typer": "0.12.5",  # Aug 2024
    "pydantic": "2.10.0",  # Nov 2024
    "polars": "1.15.0",  # Nov 2024
    "pypdf": "5.1.0",  # Oct 2024
    "python-docx": "1.1.2",  # Jun 2024
    "mapie": "0.9.0",  # Sep 2024
    "scikit-learn": "1.5.0",  # May 2024
    "numpy": "1.26.0",  # Sep 2023 (stable/required)
    "rich": "13.9.0",  # Oct 2024
}


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required novel dependencies are installed.

    Returns:
        Tuple of (all_valid, list_of_issues)
    """
    issues = []
    all_valid = True

    for package, min_version in REQUIRED_PACKAGES.items():
        try:
            installed_version = version(package)
            if parse(installed_version) < parse(min_version):
                issues.append(
                    f"❌ {package}: installed {installed_version}, "
                    f"required >={min_version}"
                )
                all_valid = False
            else:
                logger.info(f"✅ {package} v{installed_version} OK")

        except PackageNotFoundError:
            issues.append(
                f"❌ {package}: NOT INSTALLED (required >={min_version})")
            all_valid = False

    return all_valid, issues


def enforce_novelty():
    """
    Enforce novelty requirements, exit if not met.
    """
    print("\n" + "=" * 60)
    print("PDM CONTRADICTION DETECTOR - Verificación de Dependencias")
    print("=" * 60)

    all_valid, issues = check_dependencies()

    if not all_valid:
        print("\n⚠️  DEPENDENCIAS NO VÁLIDAS:")
        for issue in issues:
            print(f"  {issue}")

        print("\n📦 Para instalar las versiones correctas:")
        print("  pip install -U pdm_contra")
        print("  # o")
        print("  pip install -r requirements.txt")
        print("\n" + "=" * 60)

        sys.exit(1)
    else:
        print("\n✅ Todas las dependencias cumplen los requisitos de novedad (2024+)")
        print("=" * 60 + "\n")


# pdm_contra/cli.py
"""
Command-line interface for PDM contradiction detection.
"""

# Initialize Typer app
app = typer.Typer(
    name="pdm-contradict",
    help="Detección de contradicciones en Planes de Desarrollo Municipal",
    add_completion=False,
)

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.command()
def scan(
    files: List[Path] = typer.Argument(
        ..., help="Archivos PDM a analizar (PDF, DOCX, HTML, TXT)"
    ),
    competence_matrix: Optional[Path] = typer.Option(
        None, "--matrix", "-m", help="Archivo JSON con matriz de competencias"
    ),
    sectors: Optional[List[str]] = typer.Option(
        None, "--sector", "-s", help="Sectores a analizar"
    ),
    light_mode: bool = typer.Option(
        False, "--light", "-l", help="Usar modo ligero (modelos más pequeños)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Archivo de salida para resultados"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Salida detallada"),
):
    """
    Escanear PDM en busca de contradicciones.
    """
    # Check dependencies
    enforce_novelty()

    # Default sectors if not provided
    if not sectors:
        sectors = [
            "salud",
            "educacion",
            "agua",
            "ambiente",
            "movilidad",
            "seguridad",
            "cultura",
            "vivienda",
        ]

    # Initialize components
    console.print(
        "\n[bold blue]Inicializando detector de contradicciones...[/bold blue]"
    )
    detector = ContradictionDetector(
        competence_matrix_path=competence_matrix, language="es", mode_light=light_mode
    )

    loader = PDMLoader()

    # Process each file
    all_results = []

    for file_path in files:
        if not file_path.exists():
            console.print(f"[red]❌ Archivo no encontrado: {file_path}[/red]")
            continue

        console.print(f"\n[cyan]📄 Procesando: {file_path.name}[/cyan]")

        try:
            # Load document
            with console.status("Cargando documento..."):
                doc_data = loader.load(file_path)

            # Extract text and structure
            text = doc_data["text"]
            pdm_structure = {
                "sections": doc_data.get("sections", {}),
                "metadata": doc_data.get("metadata", {}),
            }

            # Run analysis
            with console.status("Analizando contradicciones..."):
                start_time = datetime.now()
                analysis = detector.detect_contradictions(
                    text=text, sectors=sectors, pdm_structure=pdm_structure
                )
                processing_time = (datetime.now() - start_time).total_seconds()

            # Add metadata
            analysis.processing_time_seconds = processing_time
            analysis.file_analyzed = str(file_path)

            # Store results
            all_results.append(analysis)

            # Display summary
            _display_summary(analysis, verbose)

        except Exception as e:
            console.print(f"[red]❌ Error procesando {file_path}: {e}[/red]")
            if verbose:
                logger.exception("Error details:")

    # Save results if output specified
    if output and all_results:
        _save_results(all_results, output)
        console.print(f"\n[green]✅ Resultados guardados en: {output}[/green]")

    # Final summary
    if all_results:
        _display_final_summary(all_results)


@app.command()
def report(
    results_file: Path = typer.Argument(
        ..., help="Archivo JSON con resultados de análisis"
    ),
    format: str = typer.Option(
        "markdown", "--format", "-f", help="Formato del reporte (markdown/html/json)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Archivo de salida"
    ),
):
    """
    Generar reporte a partir de resultados de análisis.
    """
    if not results_file.exists():
        console.print(f"[red]❌ Archivo no encontrado: {results_file}[/red]")
        raise typer.Exit(1)

    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    # Generate report based on format
    if format == "markdown":
        report_content = _generate_markdown_report(results_data)
    elif format == "html":
        report_content = _generate_html_report(results_data)
    else:  # json
        report_content = json.dumps(results_data, indent=2, ensure_ascii=False)

    # Save or display
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(report_content)
        console.print(f"[green]✅ Reporte guardado en: {output}[/green]")
    else:
        console.print(report_content)


@app.command()
def explain(
    results_file: Path = typer.Argument(...,
                                        help="Archivo JSON con resultados"),
    item_id: Optional[str] = typer.Option(
        None, "--id", "-i", help="ID específico del hallazgo a explicar"
    ),
):
    """
    Explicar hallazgos específicos del análisis.
    """
    if not results_file.exists():
        console.print(f"[red]❌ Archivo no encontrado: {results_file}[/red]")
        raise typer.Exit(1)

    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if item_id:
        # Find and explain specific item
        item = _find_item_by_id(results, item_id)
        if item:
            _explain_item(item)
        else:
            console.print(
                f"[red]No se encontró el item con ID: {item_id}[/red]")
    else:
        # Explain all high-risk items
        high_risk_items = _get_high_risk_items(results)
        if high_risk_items:
            console.print(
                "[bold]Explicación de hallazgos de alto riesgo:[/bold]\n")
            for item in high_risk_items[:5]:  # Limit to top 5
                _explain_item(item)
                console.print("-" * 60)
        else:
            console.print(
                "[green]No se encontraron hallazgos de alto riesgo[/green]")


def _display_summary(analysis, verbose: bool):
    """Display analysis summary in console."""
    # Create summary table
    table = Table(title="Resumen del Análisis", show_header=True)
    table.add_column("Categoría", style="cyan")
    table.add_column("Cantidad", justify="right", style="yellow")
    table.add_column("Riesgo", style="red")

    table.add_row(
        "Contradicciones",
        str(analysis.total_contradictions),
        _risk_emoji(analysis.risk_level),
    )
    table.add_row(
        "Problemas de Competencias", str(analysis.total_competence_issues), ""
    )
    table.add_row("Brechas de Agenda", str(analysis.total_agenda_gaps), "")

    console.print(table)

    # Risk score
    risk_color = (
        "red"
        if analysis.risk_score > 0.7
        else "yellow"
        if analysis.risk_score > 0.4
        else "green"
    )
    console.print(
        f"\n[{risk_color}]Puntuación de Riesgo: {analysis.risk_score:.2f}[/{risk_color}]"
    )

    # Confidence intervals
    if analysis.confidence_intervals:
        ci = analysis.confidence_intervals.get("overall", [0, 1])
        console.print(
            f"Intervalo de Confianza (90%): [{ci[0]:.2f}, {ci[1]:.2f}]")

    # Explanations
    if verbose and analysis.explanations:
        console.print("\n[bold]Explicaciones:[/bold]")
        for exp in analysis.explanations[:3]:  # Show first 3
            console.print(f"  • {exp}")


def _risk_emoji(risk_level) -> str:
    """Get emoji for risk level."""
    risk_map = {
        "LOW": "✅",
        "MEDIUM": "⚠️",
        "MEDIUM_HIGH": "⚠️⚠️",
        "HIGH": "🔴",
        "CRITICAL": "🚨",
    }
    return risk_map.get(str(risk_level).upper(), "❓")


def _save_results(results: List, output_path: Path):
    """Save analysis results to file."""
    # Convert to dict format
    results_dict = {
        "analysis_date": datetime.now().isoformat(),
        "total_files": len(results),
        "results": [],
    }

    for r in results:
        # Convert Pydantic model to dict
        if hasattr(r, "dict"):
            result_dict = r.dict()
        else:
            result_dict = dict(r)
        results_dict["results"].append(result_dict)

    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)


def _display_final_summary(results: List):
    """Display final summary of all analyses."""
    console.print("\n" + "=" * 60)
    console.print("[bold green]RESUMEN FINAL[/bold green]")
    console.print("=" * 60)

    total_issues = sum(
        r.total_contradictions + r.total_competence_issues + r.total_agenda_gaps
        for r in results
    )

    high_risk_count = sum(1 for r in results if r.risk_score > 0.7)

    console.print(f"📊 Archivos analizados: {len(results)}")
    console.print(f"⚠️  Total de hallazgos: {total_issues}")
    console.print(f"🔴 Documentos de alto riesgo: {high_risk_count}")


def _generate_markdown_report(results_data: dict) -> str:
    """Generate Markdown report from results."""
    report = "# Reporte de Análisis PDM\n\n"
    report += f"**Fecha:** {results_data.get('analysis_date', 'N/A')}\n\n"
    report += f"**Archivos analizados:** {results_data.get('total_files', 0)}\n\n"

    report += "## Resumen Ejecutivo\n\n"

    for i, result in enumerate(results_data.get("results", []), 1):
        report += f"### Documento {i}: {result.get('file_analyzed', 'N/A')}\n\n"
        report += f"- **Contradicciones:** {result.get('total_contradictions', 0)}\n"
        report += f"- **Problemas de Competencias:** {result.get('total_competence_issues', 0)}\n"
        report += f"- **Brechas de Agenda:** {result.get('total_agenda_gaps', 0)}\n"
        report += f"- **Riesgo:** {result.get('risk_score', 0):.2f} ({result.get('risk_level', 'N/A')})\n\n"

    return report


def _generate_html_report(results_data: dict) -> str:
    """Generate HTML report from results."""
    # Simplified HTML report
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Reporte PDM</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .summary { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .high-risk { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Reporte de Análisis PDM</h1>
"""

    html += f"<p><strong>Fecha:</strong> {results_data.get('analysis_date', 'N/A')}</p>"
    html += f"<p><strong>Archivos:</strong> {results_data.get('total_files', 0)}</p>"

    html += "</body></html>"
    return html


def _find_item_by_id(results: dict, item_id: str) -> Optional[dict]:
    """Find specific item by ID in results."""
    # Implementation depends on result structure
    return None


def _get_high_risk_items(results: dict) -> List[dict]:
    """Get high risk items from results."""
    items = []
    # Extract high risk items from results
    return items


def _explain_item(item: dict):
    """Display detailed explanation of an item."""
    console.print(f"[bold]Item: {item.get('id', 'N/A')}[/bold]")
    console.print(f"Tipo: {item.get('type', 'N/A')}")
    console.print(f"Explicación: {item.get('explanation', 'N/A')}")


if __name__ == "__main__":
    app()
