"""
Abstract base class and error types for SAKSHAM model parsers.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional
from ..canonical.model import CanonicalModel


class ModelParseError(Exception):
    """Raised when parsing of LP or MPS fails due to syntax or semantic errors."""
    def __init__(
        self,
        message: str,
        line_num: Optional[int] = None,
        line_text: Optional[str] = None,
        file_path: Optional[str] = None
    ):
        self.message = message
        self.line_num = line_num
        self.line_text = line_text
        self.file_path = file_path
        
        detail_parts = [message]
        if line_num is not None:
            detail_parts.append(f"at line {line_num}")
        if line_text:
            detail_parts.append(f"content: '{line_text.strip()}'")
        if file_path:
            detail_parts.append(f"in file '{file_path}'")
            
        full_msg = " | ".join(detail_parts)
        super().__init__(full_msg)


def detect_model_format(content_or_filename: str) -> str:
    """
    Detects whether an input is in CPLEX LP or MPS format.
    
    Args:
        content_or_filename: Either the raw string content or a filename/path.
        
    Returns:
        'lp' or 'mps'
        
    Raises:
        ModelParseError if format cannot be determined.
    """
    sample = content_or_filename.strip()
    
    # Check filename extension first if short string with extension
    if "\n" not in sample and ("." in sample):
        ext = sample.lower().split(".")[-1]
        if ext == "lp":
            return "lp"
        if ext == "mps":
            return "mps"

    # Analyze textual content
    lines = [line.strip().upper() for line in sample.splitlines() if line.strip() and not line.strip().startswith("\\")]
    if not lines:
        raise ModelParseError("Cannot detect model format: input is empty.")

    first_lines = lines[:5]
    for line in first_lines:
        if line.startswith("NAME"):
            return "mps"
        if line in ["MINIMIZE", "MAXIMIZE", "MIN", "MAX"] or line.startswith("MIN") or line.startswith("MAX"):
            return "lp"

    # Secondary checks
    joined = " ".join(first_lines)
    if "ROWS" in joined and "COLUMNS" in sample.upper():
        return "mps"
    if "SUBJECT TO" in sample.upper() or "SUCH THAT" in sample.upper() or "ST" in first_lines:
        return "lp"

    raise ModelParseError(
        "Unknown model format. Could not identify as CPLEX LP (starting with MIN/MAX) or MPS (starting with NAME)."
    )


class AbstractModelParser(ABC):
    """Base interface for all optimization problem parsers."""

    @abstractmethod
    def parse_file(self, file_path: Union[str, Path]) -> CanonicalModel:
        """Parse problem file into CanonicalModel."""
        pass

    @abstractmethod
    def parse_string(self, content: str, model_name: Optional[str] = None) -> CanonicalModel:
        """Parse problem from text content into CanonicalModel."""
        pass
