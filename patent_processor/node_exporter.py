"""
Export chemistry nodes to various formats
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from .data_structures import ChemistryNode


class NodeExporter:
    """Class for exporting chemistry nodes to JSON format"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_chemistry_as_single_file(self, chemistry_nodes: List[ChemistryNode], 
                                      output_dir: str, filename: str = "chemistry_nodes.json",
                                      patent_metadata: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """
        Export all ChemistryNode instances to a single JSON file
        
        Args:
            chemistry_nodes: List of chemistry nodes to export
            output_dir: Directory for saving the file
            filename: Filename (default chemistry_nodes.json)
            patent_metadata: Additional patent metadata to include
            
        Returns:
            Dict: Dictionary with the created file
        """
        if not chemistry_nodes:
            self.logger.warning("No chemistry nodes to export")
            return {"json": []}
        
        # Create directory if it does not exist
        full_output_dir = Path(output_dir)
        full_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert all ChemistryNode instances to a serializable format
        serialized_nodes = []
        for i, node in enumerate(chemistry_nodes):
            # ChemistryNode now contains only simple types; use model_dump() directly
            serialized_node = node.model_dump()
            serialized_nodes.append(serialized_node)
        
        # Build final data structure
        export_data = {
            "chemistry_nodes": serialized_nodes,
            "total_count": len(serialized_nodes)
        }
        
        # Add patent metadata
        if patent_metadata:
            export_data.update(patent_metadata)
        
        # Save file
        file_path = full_output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
        
        self.logger.debug(f"Successfully exported {len(serialized_nodes)} chemistry nodes to single file: {file_path}")
        return {"json": [str(file_path)]}
    
    async def async_export_chemistry_as_single_file(self, chemistry_nodes: List[ChemistryNode], 
                                                  output_dir: str, filename: str = "chemistry_nodes.json",
                                                  patent_metadata: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """
        Asynchronous version of exporting all ChemistryNode instances to a single JSON file
        
        Args:
            chemistry_nodes: List of chemistry nodes to export
            output_dir: Directory for saving the file
            filename: Filename (default chemistry_nodes.json)
            patent_metadata: Additional patent metadata to include
            
        Returns:
            Dict: Dictionary with the created file
        """
        # Run synchronous work in a separate thread
        return await asyncio.to_thread(
            self.export_chemistry_as_single_file, chemistry_nodes, output_dir, filename, patent_metadata
        )
    
    def _make_safe_filename(self, name: str) -> str:
        """Create a safe filename"""
        if not name:
            return "unnamed"
        
        # Replace unsafe characters
        safe_name = str(name)
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            safe_name = safe_name.replace(char, '_')
        
        # Limit length
        safe_name = safe_name[:50]
        
        return safe_name.strip('_')
