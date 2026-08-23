"""
Data structures for XML Patent Processor
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional
from lxml import etree


class ChemistryNode(BaseModel):
    """Represents a chemical structure from a patent with textual context"""
    
    chemistry_id: Optional[str] = None  # ID from the 'id' attribute
    
    # --- Text context fields ---
    # Text content of the element containing <chemistry>
    element_text: str = ""
    # List of preceding sibling element texts
    preceding_siblings_text: List[str] = Field(default_factory=list)
    # List of following sibling element texts
    following_siblings_text: List[str] = Field(default_factory=list)
    
    # --- SMILES fields (Iteration 2) ---
    chem_num: Optional[str] = None      # Number from the num attribute
    smiles: Optional[str] = None        # Extracted SMILES string
    mol_filename: Optional[str] = None  # Associated MOL filename
    inchikey: Optional[str] = None      # Standard InChI key for the chemical structure
    molecular_weight: Optional[float] = None  # Molecular weight in g/mol (Da)
    is_encoding_error: Optional[bool] = None  # Whether an encoding error occurred during zip extraction (True/False)
    is_scaffold: Optional[bool] = False  # Marker for incomplete molecules (scaffolds) with attachment points


class PatentDocument(BaseModel):
    """Represents a parsed patent document"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    file_path: str
    patent_id: Optional[str] = None
    
    # Fields from Parquet file
    publication_number: Optional[str] = None
    publication_date: Optional[int] = None
    filing_date: Optional[int] = None
    grant_date: Optional[int] = None
    priority_date: Optional[int] = None
    patent_title: Optional[str] = None
    abstract: Optional[str] = None
    claims: Optional[str] = None
    description: Optional[str] = None
    
    # Service fields
    description_html: Optional[str] = Field(default=None, exclude=True)  # Excluded from serialization
    
    # Extracted structured data
    chemistry_nodes: List[ChemistryNode] = Field(default_factory=list)
    
    # CDX structures from the patent archive: {chem_num -> {smiles, inchikey}}
    # Separate source from ChemistryNode.smiles, exported as smiles_cdx / inchi_key_cdx
    cdx_data: Dict[str, Dict[str, Optional[str]]] = Field(default_factory=dict)
    
    # Parsing metadata
    xml_root: Optional[etree._Element] = None
    parsing_errors: List[str] = Field(default_factory=list)
