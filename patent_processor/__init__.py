"""
Patent Processor - module for processing patent data from ZIP archives.

Main components:
- PatentProcessor: Main processor for lazy patent processing from ZIP archives
- NodeExporter: Export chemistry nodes to JSON
- PatentDocument, ChemistryNode: Data models
"""

from .patent_processor import PatentProcessor
from .data_structures import PatentDocument, ChemistryNode
from .node_exporter import NodeExporter

__all__ = [
    'PatentProcessor',
    'PatentDocument', 
    'ChemistryNode',
    'NodeExporter'
]

__version__ = "0.1.0"
