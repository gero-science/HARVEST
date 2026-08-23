"""
Extract chemistry nodes from XML patent documents
"""
import logging
import asyncio
from typing import List
from lxml import etree
from .data_structures import PatentDocument, ChemistryNode


class ChemistryExtractor:
    """Extracts chemistry nodes (<chemistry>) from an XML document."""

    def __init__(self, num_preceding_siblings: int = 3, num_following_siblings: int = 1):
        """
        Initialize the extractor.
        
        Args:
            num_preceding_siblings: Number of preceding sibling elements to collect for context
            num_following_siblings: Number of following sibling elements to collect for context
        """
        self.logger = logging.getLogger(__name__)
        self.num_preceding_siblings = num_preceding_siblings
        self.num_following_siblings = num_following_siblings

    async def async_extract_all_nodes(self, document: PatentDocument) -> List[ChemistryNode]:
        """
        Asynchronously extract all chemistry nodes from the document.
        
        Args:
            document: Patent document to process
            
        Returns:
            List[ChemistryNode]: List of found chemistry nodes
        """
        if document.xml_root is None:
            self.logger.warning(f"No XML root for patent {document.patent_id}")
            return []
        
        self.logger.debug(f"Starting chemistry extraction for patent {document.patent_id}")
        
        # Run the main work in an executor to avoid blocking
        return await asyncio.to_thread(self._extract_nodes_sync, document)
    
    def _extract_clean_text(self, element) -> str:
        """
        Extract only clean text from an element, excluding content of
        non-text elements according to the USPTO DTD.
        
        Args:
            element: XML element to extract text from
            
        Returns:
            str: Clean text without tables, instructions, and tags
        """
        if element is None:
            return ""
        
        # Elements that do NOT contain text per DTD (must be excluded)
        excluded_tags = {
            'img',                    # images (EMPTY)
            'chemistry',              # chemical structures (contains only img or chem+img) 
            'maths',                  # mathematical formulas (contains only img or math+img)
            'tables',                 # tables
            'table',                  # table element (EMPTY)
            'table-external-doc',     # external table reference
            'us-chemistry'            # external chemical structures
        }
        
        text_parts = []
        
        # Recursively traverse the element
        for item in element.iter():
            # Skip ProcessingInstruction and Comment
            if isinstance(item, (etree._ProcessingInstruction, etree._Comment)):
                continue
                
            # Skip excluded tags
            if hasattr(item, 'tag') and item.tag in excluded_tags:
                continue
                
            # Add element text (but not tail, to avoid duplication)
            if hasattr(item, 'text') and item.text:
                text_parts.append(item.text)
        
        # Join and normalize
        full_text = ''.join(text_parts)
        # Normalize whitespace
        import re
        full_text = re.sub(r'\s+', ' ', full_text)
        return full_text.strip()
    
    def _extract_nodes_sync(self, document: PatentDocument) -> List[ChemistryNode]:
        """
        Synchronous method for extracting chemistry nodes.
        Runs in a separate thread via asyncio.to_thread.
        """
        nodes = []
        
        try:
            # 1. Find all <chemistry> elements only in the <description> section
            chemistry_elements = document.xml_root.xpath('//description//chemistry')
            
            for i, chemistry_element in enumerate(chemistry_elements):
                # 2. Get the parent element (e.g., <p>)
                parent_element = chemistry_element.getparent()
                
                if parent_element is None:
                    self.logger.warning(f"Chemistry element {chemistry_element.get('id', f'chem_{i+1}')} has no parent")
                    continue
                
                # 3. Create a node with text content
                # Extract all text from the parent element (including sub-elements)
                element_text = ''.join(parent_element.itertext()).strip()
                
                # NEW: Extract num attribute for linking with MOL files (Iteration 2)
                chem_num = chemistry_element.get('num')
                
                node = ChemistryNode(
                    chemistry_id=chemistry_element.get('id', f'chem_{i+1}'),
                    element_text=element_text,
                    chem_num=chem_num  # New field for Iteration 2
                )

                # 4. Collect preceding siblings of the PARENT element (text only)
                current = parent_element
                for _ in range(self.num_preceding_siblings):
                    prev_sibling = current.getprevious()
                    if prev_sibling is not None:
                        # Use function to extract clean text
                        sibling_text = self._extract_clean_text(prev_sibling)
                        if sibling_text:  # Add only non-empty texts
                            node.preceding_siblings_text.insert(0, sibling_text)
                        current = prev_sibling
                    else:
                        break
                
                # 5. Collect following siblings of the PARENT element (text only)
                current = parent_element
                for _ in range(self.num_following_siblings):
                    next_sibling = current.getnext()
                    if next_sibling is not None:
                        # Use function to extract clean text
                        sibling_text = self._extract_clean_text(next_sibling)
                        if sibling_text:  # Add only non-empty texts
                            node.following_siblings_text.append(sibling_text)
                        current = next_sibling
                    else:
                        break
                        
                nodes.append(node)
            
            self.logger.debug(f"Found {len(nodes)} chemistry nodes in {document.patent_id}")
            
        except Exception as e:
            self.logger.error(f"Error extracting chemistry nodes from {document.patent_id}: {e}")
            document.parsing_errors.append(f"Chemistry extraction error: {e}")
        
        return nodes

    def extract_all_nodes(self, document: PatentDocument) -> List[ChemistryNode]:
        """
        Synchronous version of chemistry node extraction for backward compatibility.
        
        Args:
            document: Patent document to process
            
        Returns:
            List[ChemistryNode]: List of found chemistry nodes
        """
        return self._extract_nodes_sync(document)
