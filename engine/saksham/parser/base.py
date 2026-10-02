"""
Abstract base class for LP/MPS model parsers.
Full implementation scheduled for Phase 2.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from ..canonical.model import CanonicalModel


class AbstractModelParser(ABC):
    """Base interface for all optimization problem parsers."""

    @abstractmethod
    def parse_file(self, file_path: Union[str, Path]) -> CanonicalModel:
        """Parse problem file into CanonicalModel."""
        pass

    @abstractmethod
    def parse_string(self, content: str) -> CanonicalModel:
        """Parse problem from text content into CanonicalModel."""
        pass
