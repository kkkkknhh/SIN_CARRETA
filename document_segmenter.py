"""
Document Segmenter Module

Segments plan documents into logical units (objectives, strategies, etc.)
to enable precise analysis and alignment with DECALOGO questions.

Features:
- Multiple segmentation strategies (paragraph, section, semantic)
- Section type detection
- Logical unit identification
- Cross-reference preservation
- Context preservation
- Direct alignment with DECALOGO dimensions
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationType(Enum):
    """Type of segmentation strategy to use."""
    PARAGRAPH = "paragraph"
    SECTION = "section"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"


class SectionType(Enum):
    """Types of sections in a plan document relevant to DECALOGO."""
    DIAGNOSTIC = "diagnostic"           # DE-2: Thematic Inclusion
    VISION = "vision"                   # DE-1: Logic Intervention Framework
    OBJECTIVE = "objective"             # DE-1: Logic Intervention Framework
    STRATEGY = "strategy"               # DE-1: Logic Intervention Framework
    INDICATOR = "indicator"             # DE-4: Results Orientation
    RESPONSIBILITY = "responsibility"   # DE-1: Logic Intervention Framework (Q2)
    PARTICIPATION = "participation"     # DE-3: Participation and Governance
    MONITORING = "monitoring"           # DE-4: Results Orientation
    BUDGET = "budget"                   # DE-2: Thematic Inclusion
    TIMELINE = "timeline"               # DE-4: Results Orientation
    OTHER = "other"                     # General content


@dataclass
class DocumentSegment:
    """
    A segment of a document with metadata for DECALOGO evaluation.
    
    Attributes:
        text: The text content of the segment
        start_pos: Start position in the original document
        end_pos: End position in the original document
        segment_type: Type of segment (paragraph, section, etc.)
        section_type: Type of section content (diagnostic, objective, etc.)
        metadata: Additional metadata about the segment
        decalogo_dimensions: DECALOGO dimensions this segment is relevant to
        parent_id: ID of parent segment (for hierarchical structure)
        segment_id: Unique identifier for the segment
    """
    text: str
    start_pos: int
    end_pos: int
    segment_type: SegmentationType
    section_type: SectionType = SectionType.OTHER
    metadata: Dict[str, Any] = field(default_factory=dict)
    decalogo_dimensions: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    segment_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        if self.segment_id is None:
            # Generate a unique ID based on content and position
            import hashlib
            content_hash = hashlib.md5(self.text[:100].encode()).hexdigest()[:8]
            self.segment_id = f"{self.start_pos}_{self.end_pos}_{content_hash}"
        
        # Auto-assign relevant DECALOGO dimensions based on section type
        if not self.decalogo_dimensions:
            self.decalogo_dimensions = self._infer_decalogo_dimensions()
    
    def _infer_decalogo_dimensions(self) -> List[str]:
        """Infer which DECALOGO dimensions this segment is relevant to."""
        section_to_dimension = {
            SectionType.DIAGNOSTIC: ["DE-2"],
            SectionType.VISION: ["DE-1"],
            SectionType.OBJECTIVE: ["DE-1"],
            SectionType.STRATEGY: ["DE-1"],
            SectionType.INDICATOR: ["DE-1", "DE-4"],
            SectionType.RESPONSIBILITY: ["DE-1"],  # Specifically for DE-1 Q2
            SectionType.PARTICIPATION: ["DE-3"],
            SectionType.MONITORING: ["DE-4"],
            SectionType.BUDGET: ["DE-2"],
            SectionType.TIMELINE: ["DE-4"],
            SectionType.OTHER: [],
        }
        return section_to_dimension.get(self.section_type, [])


class DocumentSegmenter:
    """
    Segments a document into logical units for DECALOGO analysis.
    
    Attributes:
        segmentation_type: Type of segmentation to perform
        min_segment_length: Minimum length for a valid segment
        max_segment_length: Maximum length for a segment before splitting
        preserve_context: Whether to include surrounding context in segments
        section_patterns: Regex patterns for identifying section types
    """
    
    def __init__(
        self,
        segmentation_type: SegmentationType = SegmentationType.SECTION,
        min_segment_length: int = 50,
        max_segment_length: int = 1000,
        preserve_context: bool = True,
    ):
        """
        Initialize the document segmenter.
        
        Args:
            segmentation_type: Type of segmentation to perform
            min_segment_length: Minimum length for a valid segment
            max_segment_length: Maximum length for a segment before splitting
            preserve_context: Whether to include surrounding context in segments
        """
        self.segmentation_type = segmentation_type
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.preserve_context = preserve_context
        
        # Section identification patterns - aligned with DECALOGO dimensions
        self.section_patterns = {
            # DE-2: Thematic Inclusion (Diagnostic)
            SectionType.DIAGNOSTIC: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:diagn[óo]stico|antecedentes|contexto|situaci[óo]n actual)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:problem[áa]tica|necesidades|demandas)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:caracterizaci[óo]n|perfil)"
            ],
            
            # DE-1: Logical Intervention Framework (Vision)
            SectionType.VISION: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:visi[óo]n|misi[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:escenario(?:s)? deseado(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:futuro(?:s)? deseado(?:s)?)"
            ],
            
            # DE-1: Logical Intervention Framework (Objectives)
            SectionType.OBJECTIVE: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:objetivo(?:s)?|prop[óo]sito(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:finalidad(?:es)?|meta(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:logro(?:s)? esperado(?:s)?)"
            ],
            
            # DE-1: Logical Intervention Framework (Strategies)
            SectionType.STRATEGY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:estrategia(?:s)?|l[íi]nea(?:s)? (?:de)? acci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:programa(?:s)?|proyecto(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:iniciativa(?:s)?|actividad(?:es)?)"
            ],
            
            # DE-4: Results Orientation (Indicators)
            SectionType.INDICATOR: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:indicador(?:es)?|medici[óo]n(?:es)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:m[ée]trica(?:s)?|valor(?:es)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:l[íi]nea(?:s)? base)"
            ],
            
            # DE-1: Logical Intervention Framework (Q2 - Responsibilities)
            SectionType.RESPONSIBILITY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:responsable(?:s)?|encargado(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:entidad(?:es)? (?:responsable|ejecutora)(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:actor(?:es)? (?:responsable|institucional)(?:s)?)"
            ],
            
            # DE-3: Participation and Governance
            SectionType.PARTICIPATION: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:participaci[óo]n|gobernanza|concertaci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:mesa(?:s)? (?:t[ée]cnica(?:s)?|participativa(?:s)?))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:di[áa]logo(?:s)?|consulta(?:s)?)"
            ],
            
            # DE-4: Results Orientation (Monitoring)
            SectionType.MONITORING: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:seguimiento|monitoreo|evaluaci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:control|supervisi[óo]n|vigilancia)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:tablero(?:s)? de (?:control|mando))"
            ],
            
            # DE-2: Thematic Inclusion (Budget)
            SectionType.BUDGET: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:presupuesto|recursos (?:financieros|econ[óo]micos))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:financiaci[óo]n|inversi[óo]n|gasto(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:costeo|asignaci[óo]n (?:presupuestal|de recursos))"
            ],
            
            # DE-4: Results Orientation (Timeline)
            SectionType.TIMELINE: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:cronograma|calendario|plazos)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:tiempos|periodicidad|fechas)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:hitos|milestones|fases)"
            ],
        }
        
        # Compile all section patterns for efficiency
        self.compiled_patterns = {}
        for section_type, patterns in self.section_patterns.items():
            self.compiled_patterns[section_type] = [re.compile(pattern) for pattern in patterns]
    
    def segment_text(self, text: str) -> List[DocumentSegment]:
        """
        Segment the given text into logical units.
        
        Args:
            text: Text to segment
            
        Returns:
            List of DocumentSegment objects
        """
        if not text:
            return []
            
        # Choose segmentation strategy
        if self.segmentation_type == SegmentationType.PARAGRAPH:
            return self._segment_by_paragraph(text)
        elif self.segmentation_type == SegmentationType.SECTION:
            return self._segment_by_section(text)
        elif self.segmentation_type == SegmentationType.SENTENCE:
            return self._segment_by_sentence(text)
        elif self.segmentation_type == SegmentationType.SEMANTIC:
            return self._segment_by_semantic_units(text)
        else:
            # Default to paragraph segmentation
            return self._segment_by_paragraph(text)
    
    def _segment_by_paragraph(self, text: str) -> List[DocumentSegment]:
        """
        Segment text by paragraphs.
        
        Args:
            text: Text to segment
            
        Returns:
            List of DocumentSegment objects
        """
        segments = []
        
        # Split by double line breaks (typical paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        position = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            
            # Skip empty paragraphs
            if not paragraph or len(paragraph) < self.min_segment_length:
                position += len(paragraph) + 2  # +2 for the newlines
                continue
            
            # Find the original position in the text
            start_pos = text.find(paragraph, position)
            if start_pos == -1:  # Fallback if exact match not found
                start_pos = position
                
            end_pos = start_pos + len(paragraph)
            position = end_pos
            
            # Identify section type
            section_type = self._identify_section_type(paragraph)
            
            # Create segment
            segment = DocumentSegment(
                text=paragraph,
                start_pos=start_pos,
                end_pos=end_pos,
                segment_type=SegmentationType.PARAGRAPH,
                section_type=section_type,
            )
            
            segments.append(segment)
        
        return segments
    
    def _segment_by_section(self, text: str) -> List[DocumentSegment]:
        """
        Segment text by document sections.
        
        Args:
            text: Text to segment
            
        Returns:
            List of DocumentSegment objects
        """
        segments = []
        
        # Find all potential section headers
        all_matches = []
        for section_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    all_matches.append((match.start(), match.end(), section_type))
        
        # Sort matches by position
        all_matches.sort(key=lambda x: x[0])
        
        # Create segments from sections
        for i in range(len(all_matches)):
            start_pos, header_end_pos, section_type = all_matches[i]
            
            # Find the end of this section (start of next section or end of text)
            if i < len(all_matches) - 1:
                end_pos = all_matches[i + 1][0]
            else:
                end_pos = len(text)
            
            # Extract section content (excluding header)
            section_text = text[header_end_pos:end_pos].strip()
            
            # Skip if section is too short
            if len(section_text) < self.min_segment_length:
                continue
            
            # Split if section is too long
            if len(section_text) > self.max_segment_length:
                subsegments = self._split_long_segment(section_text, start_pos=header_end_pos)
                
                # Set parent-child relationship
                parent_id = f"section_{start_pos}_{header_end_pos}"
                for subsegment in subsegments:
                    subsegment.section_type = section_type
                    subsegment.parent_id = parent_id
                
                segments.extend(subsegments)
            else:
                # Create section segment
                segment = DocumentSegment(
                    text=section_text,
                    start_pos=header_end_pos,
                    end_pos=end_pos,
                    segment_type=SegmentationType.SECTION,
                    section_type=section_type,
                )
                segments.append(segment)
            
            # Also include the header as a separate segment for context
            if self.preserve_context:
                header_text = text[start_pos:header_end_pos].strip()
                header_segment = DocumentSegment(
                    text=header_text,
                    start_pos=start_pos,
                    end_pos=header_end_pos,
                    segment_type=SegmentationType.SECTION,
                    section_type=section_type,
                    metadata={"is_header": True},
                )
                segments.append(header_segment)
        
        # If no sections found, fall back to paragraph segmentation
        if not segments:
            return self._segment_by_paragraph(text)
        
        return segments
    
    def _segment_by_sentence(self, text: str) -> List[DocumentSegment]:
        """
        Segment text by sentences.
        
        Args:
            text: Text to segment
            
        Returns:
            List of DocumentSegment objects
        """
        segments = []
        
        # Simple sentence splitting with regex (for Spanish text)
        # More sophisticated NLP-based sentence splitting could be used here
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])')
        
        position = 0
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Find paragraph position
            para_start = text.find(paragraph, position)
            if para_start == -1:
                para_start = position
            
            # Split paragraph into sentences
            sentences = sentence_pattern.split(paragraph)
            
            sent_position = para_start
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Skip very short sentences
                if not sentence or len(sentence) < self.min_segment_length:
                    sent_position += len(sentence) + 1  # +1 for the space
                    continue
                
                # Find the original position in the text
                sent_start = text.find(sentence, sent_position)
                if sent_start == -1:  # Fallback if exact match not found
                    sent_start = sent_position
                    
                sent_end = sent_start + len(sentence)
                sent_position = sent_end
                
                # Identify section type - for sentences, check the paragraph context
                section_type = self._identify_section_type(paragraph)
                
                # Create segment
                segment = DocumentSegment(
                    text=sentence,
                    start_pos=sent_start,
                    end_pos=sent_end,
                    segment_type=SegmentationType.SENTENCE,
                    section_type=section_type,
                )
                
                segments.append(segment)
            
            position = para_start + len(paragraph)
        
        return segments
    
    def _segment_by_semantic_units(self, text: str) -> List[DocumentSegment]:
        """
        Segment text by semantic units (combining sections, paragraphs and other cues).
        
        Args:
            text: Text to segment
            
        Returns:
            List of DocumentSegment objects
        """
        # Start with section segmentation
        section_segments = self._segment_by_section(text)
        
        # Process each section for further semantic segmentation
        final_segments = []
        
        for segment in section_segments:
            # If segment is a header or already small enough, keep as is
            if segment.metadata.get("is_header", False) or len(segment.text) <= self.max_segment_length:
                final_segments.append(segment)
                continue
            
            # Further segment by semantic units
            subsegments = []
            
            # Check for tables, lists, and other structured content
            if self._contains_table(segment.text):
                table_segments = self._extract_tables(segment.text, segment.start_pos)
                subsegments.extend(table_segments)
            elif self._contains_list(segment.text):
                list_segments = self._extract_lists(segment.text, segment.start_pos)
                subsegments.extend(list_segments)
            else:
                # Fall back to paragraph segmentation for this section
                paragraph_segments = self._segment_by_paragraph(segment.text)
                
                # Adjust positions to be relative to the original text
                for para_segment in paragraph_segments:
                    para_segment.start_pos += segment.start_pos
                    para_segment.end_pos += segment.start_pos
                    para_segment.section_type = segment.section_type
                
                subsegments.extend(paragraph_segments)
            
            # Preserve parent-child relationship
            for subsegment in subsegments:
                subsegment.parent_id = segment.segment_id
                
            final_segments.extend(subsegments)
        
        return final_segments
    
    def _split_long_segment(self, text: str, start_pos: int = 0) -> List[DocumentSegment]:
        """
        Split a long segment into smaller segments.
        
        Args:
            text: Text to split
            start_pos: Starting position in the original document
            
        Returns:
            List of DocumentSegment objects
        """
        segments = []
        
        # Try to split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        if len(paragraphs) > 1:
            position = start_pos
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph or len(paragraph) < self.min_segment_length:
                    continue
                
                # Find paragraph position
                para_start = text.find(paragraph, position - start_pos)
                if para_start == -1:
                    para_start = position - start_pos
                else:
                    para_start += start_pos
                
                para_end = para_start + len(paragraph)
                
                # Identify section type
                section_type = self._identify_section_type(paragraph)
                
                # Create segment
                segment = DocumentSegment(
                    text=paragraph,
                    start_pos=para_start,
                    end_pos=para_end,
                    segment_type=SegmentationType.PARAGRAPH,
                    section_type=section_type,
                )
                
                segments.append(segment)
                position = para_end
        else:
            # If no paragraph breaks, split by size
            chunk_size = self.max_segment_length
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size].strip()
                if not chunk:
                    continue
                
                chunk_start = start_pos + i
                chunk_end = chunk_start + len(chunk)
                
                # Try to find a sentence boundary to split on
                if i > 0:
                    sentence_boundary = re.search(r'[.!?]\s+', chunk[:100])
                    if sentence_boundary:
                        # Adjust to split on sentence boundary
                        boundary_pos = sentence_boundary.end()
                        prev_segment = segments[-1]
                        prev_segment.text += " " + chunk[:boundary_pos]
                        prev_segment.end_pos += boundary_pos
                        
                        chunk = chunk[boundary_pos:].strip()
                        chunk_start += boundary_pos
                
                # Identify section type
                section_type = self._identify_section_type(chunk)
                
                # Create segment
                segment = DocumentSegment(
                    text=chunk,
                    start_pos=chunk_start,
                    end_pos=chunk_end,
                    segment_type=SegmentationType.PARAGRAPH,
                    section_type=section_type,
                )
                
                segments.append(segment)
        
        return segments
    
    def _identify_section_type(self, text: str) -> SectionType:
        """
        Identify the type of section based on text content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Section type
        """
        # Check for exact section headers
        for section_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return section_type
        
        # If no exact match, check for content indicators
        lower_text = text.lower()
        
        # Keywords for each section type
        keywords = {
            SectionType.DIAGNOSTIC: ["diagnóstico", "situación", "problemática", "contexto", "antecedentes"],
            SectionType.VISION: ["visión", "misión", "futuro", "escenario"],
            SectionType.OBJECTIVE: ["objetivo", "propósito", "finalidad", "meta"],
            SectionType.STRATEGY: ["estrategia", "línea", "acción", "programa", "proyecto"],
            SectionType.INDICATOR: ["indicador", "medición", "métrica", "línea base"],
            SectionType.RESPONSIBILITY: ["responsable", "encargado", "entidad", "actor"],
            SectionType.PARTICIPATION: ["participación", "gobernanza", "concertación", "mesa", "diálogo"],
            SectionType.MONITORING: ["seguimiento", "monitoreo", "evaluación", "control"],
            SectionType.BUDGET: ["presupuesto", "recursos", "financiación", "inversión"],
            SectionType.TIMELINE: ["cronograma", "calendario", "plazo", "tiempo"],
        }
        
        # Count keyword occurrences
        counts = {section_type: 0 for section_type in SectionType}
        
        for section_type, words in keywords.items():
            for word in words:
                counts[section_type] += lower_text.count(word)
        
        # Get section type with highest keyword count, if any
        max_count = max(counts.values())
        if max_count > 0:
            for section_type, count in counts.items():
                if count == max_count:
                    return section_type
        
        # Default to OTHER if no patterns or keywords match
        return SectionType.OTHER
    
    def _contains_table(self, text: str) -> bool:
        """Check if text contains a table."""
        # Look for table-like patterns
        table_patterns = [
            r"\|\s*[\w\s]+\s*\|",  # | Column | Column |
            r"\+[-+]+\+",          # +----+----+
            r"┌[─┬]+┐",            # ┌────┬────┐
            r"\n\s*\|.*\|.*\|",    # Multiple | pipe characters on one line
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _extract_tables(self, text: str, start_offset: int) -> List[DocumentSegment]:
        """Extract table segments from text."""
        segments = []
        
        # Find potential table boundaries
        table_start_patterns = [
            r"\n\s*\+[-+]+\+",
            r"\n\s*\|",
            r"\n\s*┌[─┬]+┐",
        ]
        
        table_end_patterns = [
            r"\+[-+]+\+\s*\n",
            r"\|\s*\n(?!\s*\|)",
            r"└[─┴]+┘\s*\n",
        ]
        
        # Find all table starts and ends
        starts = []
        ends = []
        
        for pattern in table_start_patterns:
            for match in re.finditer(pattern, text):
                starts.append(match.start())
        
        for pattern in table_end_patterns:
            for match in re.finditer(pattern, text):
                ends.append(match.end())
        
        # Match starts and ends
        if starts and ends:
            starts.sort()
            ends.sort()
            
            table_regions = []
            start_idx = 0
            end_idx = 0
            
            while start_idx < len(starts) and end_idx < len(ends):
                if starts[start_idx] < ends[end_idx]:
                    table_regions.append((starts[start_idx], ends[end_idx]))
                    start_idx += 1
                    end_idx += 1
                else:
                    end_idx += 1
            
            # Create segments for each table
            for table_start, table_end in table_regions:
                table_text = text[table_start:table_end].strip()
                
                # Skip if too short
                if len(table_text) < self.min_segment_length:
                    continue
                
                segment = DocumentSegment(
                    text=table_text,
                    start_pos=start_offset + table_start,
                    end_pos=start_offset + table_end,
                    segment_type=SegmentationType.SECTION,
                    section_type=SectionType.INDICATOR,  # Most tables are indicators
                    metadata={"content_type": "table"},
                )
                segments.append(segment)
        
        return segments
    
    def _contains_list(self, text: str) -> bool:
        """Check if text contains a list."""
        # Look for list-like patterns
        list_patterns = [
            r"(?m)^\s*[\*\-•]\s+\w+",      # Bullet lists
            r"(?m)^\s*\d+[\.\)]\s+\w+",    # Numbered lists
            r"(?m)^\s*[a-z][\.\)]\s+\w+"   # Alphabetic lists
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, text)
            if len(matches) >= 2:  # At least 2 list items
                return True
        
        return False
    
    def _extract_lists(self, text: str, start_offset: int) -> List[DocumentSegment]:
        """Extract list segments from text."""
        segments = []
        
        # Find list items
        list_patterns = [
            r"(?m)^\s*([\*\-•]\s+.+?)(?=\n\s*[\*\-•]\s+|\n\s*\n|$)",      # Bullet lists
            r"(?m)^\s*(\d+[\.\)]\s+.+?)(?=\n\s*\d+[\.\)]\s+|\n\s*\n|$)",  # Numbered lists
            r"(?m)^\s*([a-z][\.\)]\s+.+?)(?=\n\s*[a-z][\.\)]\s+|\n\s*\n|$)"  # Alphabetic lists
        ]
        
        for pattern in list_patterns:
            matches = list(re.finditer(pattern, text))
            if len(matches) >= 2:  # At least 2 list items
                # Find the boundaries of the list
                list_start = matches[0].start()
                list_end = matches[-1].end()
                
                list_text = text[list_start:list_end].strip()
                
                # Identify section type based on content
                section_type = self._identify_section_type(list_text)
                
                segment = DocumentSegment(
                    text=list_text,
                    start_pos=start_offset + list_start,
                    end_pos=start_offset + list_end,
                    segment_type=SegmentationType.SECTION,
                    section_type=section_type,
                    metadata={"content_type": "list"},
                )
                segments.append(segment)
        
        return segments
    
    def segment_file(
        self, 
        input_path: Union[str, Path], 
        segmentation_type: Optional[SegmentationType] = None
    ) -> List[DocumentSegment]:
        """
        Segment a file containing text.
        
        Args:
            input_path: Path to input file
            segmentation_type: Optional override for segmentation type
            
        Returns:
            List of DocumentSegment objects
        """
        input_path = Path(input_path)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try alternate encodings if utf-8 fails
            try:
                with open(input_path, 'r', encoding='latin-1') as f:
                    text = f.read()
            except Exception as e:
                logger.error(f"Failed to read file with latin-1 encoding: {e}")
                raise
        
        # Use provided segmentation type or instance default
        current_type = segmentation_type or self.segmentation_type
        original_type = self.segmentation_type
        
        try:
            # Temporarily set segmentation type if override provided
            if segmentation_type:
                self.segmentation_type = segmentation_type
                
            # Segment the text
            segments = self.segment_text(text)
            
            logger.info(f"Segmented file {input_path} into {len(segments)} segments")
            return segments
        finally:
            # Restore original segmentation type
            if segmentation_type:
                self.segmentation_type = original_type


# Factory function to create a document segmenter with configuration
def create_document_segmenter(
    segmentation_type: str = "section",
    min_segment_length: int = 50,
    max_segment_length: int = 1000,
    preserve_context: bool = True,
) -> DocumentSegmenter:
    """
    Create a document segmenter with specified configuration.
    
    Args:
        segmentation_type: Type of segmentation ("paragraph", "section", "sentence", "semantic")
        min_segment_length: Minimum length for a valid segment
        max_segment_length: Maximum length for a segment before splitting
        preserve_context: Whether to preserve context around segments
        
    Returns:
        Configured DocumentSegmenter instance
    """
    # Convert string to enum
    seg_type = {
        "paragraph": SegmentationType.PARAGRAPH,
        "section": SegmentationType.SECTION,
        "sentence": SegmentationType.SENTENCE,
        "semantic": SegmentationType.SEMANTIC,
    }.get(segmentation_type.lower(), SegmentationType.SECTION)
    
    return DocumentSegmenter(
        segmentation_type=seg_type,
        min_segment_length=min_segment_length,
        max_segment_length=max_segment_length,
        preserve_context=preserve_context,
    )


# Simple usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    segmenter = create_document_segmenter(segmentation_type="semantic")
    
    # Example text with different section types
    sample_text = """
    PLAN DE DESARROLLO MUNICIPAL 2020-2023
    
    1. DIAGNÓSTICO
    
    La situación actual del municipio presenta desafíos en educación y salud.
    
    La tasa de analfabetismo es del 8.5%.
    La cobertura en salud alcanza solo el 75% de la población.
    
    2. OBJETIVOS ESTRATÉGICOS
    
    2.1 EDUCACIÓN
    
    Aumentar la cobertura educativa en todos los niveles.
    Reducir la deserción escolar.
    
    Indicador: Tasa de escolaridad
    Línea Base: 85%
    Meta: 95%
    
    Responsable: Secretaría de Educación Municipal
    
    3. MECANISMOS DE PARTICIPACIÓN
    
    Se realizaron 5 mesas técnicas con la comunidad:
    * Mesa 1: Sector educativo
    * Mesa 2: Sector salud
    * Mesa 3: Sector productivo
    * Mesa 4: Jóvenes
    * Mesa 5: Tercera edad
    
    4. SEGUIMIENTO Y MONITOREO
    
    El plan contará con un tablero de control para seguimiento trimestral.
    
    +----------------+------------+--------------+
    | Indicador      | Frecuencia | Responsable  |
    +----------------+------------+--------------+
    | Cobertura      | Trimestral | Sec. Educ.   |
    | Deserción      | Semestral  | Sec. Educ.   |
    | Mortalidad     | Mensual    | Sec. Salud   |
    +----------------+------------+--------------+
    """
    
    segments = segmenter.segment_text(sample_text)
    
    print(f"Segmented into {len(segments)} parts\n")
    
    for i, segment in enumerate(segments):
        print(f"Segment {i+1} - Type: {segment.section_type.name}, DECALOGO: {segment.decalogo_dimensions}")
        print(f"  {segment.text[:60]}...")
        print()

    def _post_process_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Post-process segments to ensure quality and consistency (maintain original logic)"""

        # Remove empty segments
        segments = [seg for seg in segments if seg["text"].strip()]

        # Merge small segments (below min_segment_chars) with previous one
        if self.min_segment_chars > 0:
            merged_segments = []
            current_segment = None

            for seg in segments:
                if current_segment is None:
                    current_segment = seg
                else:
                    combined_text = current_segment["text"] + " " + seg["text"]
                    if len(combined_text) < self.min_segment_chars:
                        current_segment["text"] = combined_text
                        current_segment["sentences"].extend(seg["sentences"])
                    else:
                        merged_segments.append(current_segment)
                        current_segment = seg

            if current_segment is not None:
                merged_segments.append(current_segment)

            segments = merged_segments

        # Final consistency check (maintain original logic)
        segments = [seg for seg in segments if len(seg["text"].strip()) >= 10]

        return segments

    def _emergency_fallback_segmentation(self, text: str) -> List[Dict[str, Any]]:
        """Emergency fallback segmentation using simple character-based chunking"""
        if not text or not text.strip():
            return []

        segments = []
        words = text.split()
        current_words = []
        current_length = 0
        target_length = (self.target_char_min + self.target_char_max) // 2

        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_length + word_len > target_length and current_words:
                segment_text = " ".join(current_words)
                if len(segment_text) >= self.min_segment_chars:
                    segments.append(
                        self._create_segment_dict(segment_text, [], "emergency_fallback")
                    )
                current_words = [word]
                current_length = len(word)
            else:
                current_words.append(word)
                current_length += word_len

        if current_words:
            segment_text = " ".join(current_words)
            segments.append(
                self._create_segment_dict(segment_text, [], "emergency_fallback")
            )

        return segments

    def _estimate_sentence_count(self, text: str) -> int:
        """Estimate sentence count using simple heuristics"""
        if not text:
            return 0

        # Count sentence-ending punctuation
        sentence_enders = re.findall(r'[.!?]+', text)
        estimated_sentences = len(sentence_enders)

        # Fallback: estimate based on text length (rough approximation)
        if estimated_sentences == 0:
            # Average sentence length is about 15-20 words
            word_count = len(text.split())
            estimated_sentences = max(1, word_count // 18)

        return max(1, estimated_sentences)

    def _estimate_semantic_coherence(self, text: str) -> float:
        """Basic semantic coherence estimation using lexical overlap"""
        if not text or len(text) < 50:
            return 0.5

        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 10:
            return 0.5

        # Simple coherence based on word repetition
        word_counts = Counter(words)
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        unique_words = len(word_counts)

        # Normalize to 0-1 range
        coherence = min(1.0, repeated_words / max(unique_words, 1) * 2)
        return coherence

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate basic readability score using Flesch-Kincaid approximation"""
        if not text or len(text) < 50:
            return 0.0

        words = text.split()
        sentences = self._estimate_sentence_count(text)

        if len(words) == 0 or sentences == 0:
            return 0.0

        avg_words_per_sentence = len(words) / sentences
        avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)

        # Simplified Flesch Reading Ease formula
        readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)

        # Normalize to 0-1 scale (Flesch scores typically 0-100)
        return max(0.0, min(1.0, readability / 100.0))

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"

        if word[0] in vowels:
            count += 1

        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1

        if word.endswith("e"):
            count -= 1

        return max(1, count)

    def _calculate_lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (unique words / total words)"""
        if not text:
            return 0.0

        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 5:
            return 0.0

        unique_words = len(set(words))
        return unique_words / len(words)

    def _calculate_syntactic_complexity(self, text: str) -> float:
        """Calculate syntactic complexity based on sentence structure"""
        if not text:
            return 0.0

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.0

        # Average sentence length in words
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Complexity based on sentence length variation and average length
        lengths = [len(s.split()) for s in sentences]
        if len(lengths) > 1:
            length_std = statistics.stdev(lengths)
            complexity = min(1.0, (avg_sentence_length / 20.0) + (length_std / 10.0))
        else:
            complexity = min(1.0, avg_sentence_length / 20.0)

        return complexity


def audit_performance_hotspots() -> Dict[str, List[str]]:
    """Resumen estático de posibles hotspots de rendimiento y efectos laterales."""

    return {
        "bottlenecks": [
            "IndustrialSemanticAnalyzer.analyze_comprehensive_coherence: combina múltiples analizadores secuenciales (lexical, transformer, tópicos) que procesan texto completo en cada invocación.",
            "DocumentSegmenter.segment_document: ejecuta pipelines de spaCy, clustering y cálculo de métricas por segmento, costoso en colecciones extensas.",
        ],
        "side_effects": [
            "IndustrialSemanticAnalyzer._initialize_models: descarga modelos externos y mantiene referencias en caché compartida.",
            "IndustrialSemanticAnalyzer.analyze_comprehensive_coherence: muta _coherence_cache con resultados memoizados.",
        ],
        "vectorization_opportunities": [
            "IndustrialSemanticAnalyzer._compute_lexical_coherence: podría reemplazar contadores Python por operaciones NumPy para textos largos.",
            "IndustrialSemanticAnalyzer._compute_topic_coherence: admite paralelización segura sobre ventanas de términos cuando HAS_ADVANCED_ML es True.",
        ],
    }
