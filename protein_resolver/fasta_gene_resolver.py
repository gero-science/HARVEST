#!/usr/bin/env python3
"""
FastaGeneResolver - direct protein resolver by gene + species from FASTA.

This module implements simple protein lookup directly via GN (gene name)
and OS (organism/species) fields from UniProt FASTA headers.

Does not require id_mapping files - works with FASTA only.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


class FastaGeneResolver:
    """
    Resolver for looking up proteins by gene + species directly from UniProt FASTA.
    
    Implements the Singleton pattern for efficient memory use.
    Supports any organisms present in the FASTA file.
    
    UniProt header format:
    >sp|P31946|1433B_HUMAN 14-3-3 protein beta/alpha OS=Homo sapiens OX=9606 GN=YWHAB PE=1 SV=1
    
    Indexed by: (gene_name.lower(), species.lower()) -> SeqRecord
    """
    
    _instance: Optional['FastaGeneResolver'] = None
    _initialized: bool = False
    
    # Labels that previously silently mapped to Homo sapiens — no longer remapped.
    _NONSPECIFIC_SPECIES = {
        "mammalian", "mammals", "mammal",
        "primate", "primates",
        "vertebrate", "vertebrates",
        "eukaryote", "eukaryotes",
        "unknown", "unspecified", "not specified",
        "human_default",
    }
    
    def __init__(self):
        """Private constructor. Use FastaGeneResolver.create() to instantiate."""
        if FastaGeneResolver._instance is not None:
            raise RuntimeError("FastaGeneResolver already created. Use FastaGeneResolver.create()")
        
        self.logger = logging.getLogger(__name__)
        # Primary index: (gene, species) -> SeqRecord
        self._index: Dict[Tuple[str, str], SeqRecord] = {}
        # Secondary index by accession for extract_accession_from_id
        self._records_by_accession: Dict[str, SeqRecord] = {}
    
    @classmethod
    async def create(cls, fasta_path: Union[str, Path]) -> 'FastaGeneResolver':
        """
        Async factory method for creating FastaGeneResolver (Singleton).
        
        Args:
            fasta_path: Path to uniprot_sprot.fasta
        
        Returns:
            Initialized FastaGeneResolver instance
        """
        if cls._instance is not None and cls._initialized:
            cls._instance.logger.debug("Returning existing FastaGeneResolver instance")
            return cls._instance
        
        if cls._instance is None:
            cls._instance = cls()
        
        if not cls._initialized:
            await cls._instance._async_init(fasta_path)
            cls._initialized = True
        
        return cls._instance
    
    async def _async_init(self, fasta_path: Union[str, Path]):
        """Async initialization - load FASTA in a background thread."""
        self.logger.info(f"Starting FastaGeneResolver load from {fasta_path}...")
        
        try:
            await asyncio.to_thread(self._load_fasta, Path(fasta_path))
            self.logger.info(
                f"FastaGeneResolver initialized: {len(self._index)} (gene, species) records"
            )
        except Exception as e:
            self.logger.error(f"Error initializing FastaGeneResolver: {e}")
            raise
    
    def _load_fasta(self, fasta_path: Path):
        """
        Load FASTA file and build (gene, species) -> SeqRecord index.
        
        Runs in a separate thread for non-blocking operation.
        """
        if not fasta_path.exists():
            self.logger.warning(f"File {fasta_path} not found.")
            return
        
        records_loaded = 0
        indexed_count = 0
        
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            records_loaded += 1
            
            # Store by accession for compatibility
            accession = self._extract_accession(record.id)
            if accession:
                self._records_by_accession[accession] = record
            
            # Parse header to extract gene and species
            gene, species = self._parse_header(record.description)
            
            if gene and species:
                key = (gene.lower(), species.lower())
                # Do not overwrite if key already exists (first entry wins)
                if key not in self._index:
                    self._index[key] = record
                    indexed_count += 1
            
            # Log progress every 100k records
            if records_loaded % 100000 == 0:
                self.logger.debug(f"Processed {records_loaded} records, indexed {indexed_count}")
        
        self.logger.info(f"Loaded {records_loaded} records, indexed {indexed_count}")
    
    def _parse_header(self, description: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract gene name (GN) and species (OS) from a FASTA record description.
        
        Example input:
        sp|P31946|1433B_HUMAN 14-3-3 protein beta/alpha OS=Homo sapiens OX=9606 GN=YWHAB PE=1 SV=1
        
        Returns:
            Tuple[gene_name, species] or (None, None) if parsing fails
        """
        gene = None
        species = None
        
        # Extract OS= (organism/species) - everything until the next field (OX=, GN=, PE=, SV=)
        os_match = re.search(r'OS=(.+?)(?:\s+(?:OX|GN|PE|SV)=|$)', description)
        if os_match:
            species = os_match.group(1).strip()
        
        # Extract GN= (gene name)
        gn_match = re.search(r'GN=([^\s]+)', description)
        if gn_match:
            gene = gn_match.group(1).strip()
        
        return gene, species
    
    def _extract_accession(self, seq_id: str) -> Optional[str]:
        """Extract Accession ID from a string like 'sp|P01112|HRAS_HUMAN'."""
        if '|' in seq_id:
            parts = seq_id.split('|')
            if len(parts) >= 2:
                return parts[1]
        return seq_id
    
    def _normalize_species(self, species: str) -> str:
        """
        Normalize species by extracting the scientific name from complex strings.
        
        Examples:
        - "hek-293 cells (human)" → "homo sapiens"
        - "human (cho k1 cells)" → "homo sapiens"
        - "mammalian" / "unknown" → left as-is (no silent Homo sapiens default)
        """
        species_lower = species.lower().strip()
        
        # Non-specific labels are not remapped to human anymore
        if species_lower in self._NONSPECIFIC_SPECIES:
            return species_lower
        
        # Extract "human" from complex strings like "hek-293 cells (human)"
        if "(human)" in species_lower or "human (" in species_lower or species_lower == "human":
            return "homo sapiens"
        if "(mouse)" in species_lower or "mouse (" in species_lower or species_lower == "mouse":
            return "mus musculus"
        if "(rat)" in species_lower or "rat (" in species_lower or species_lower == "rat":
            return "rattus norvegicus"
        
        return species_lower
    
    def resolve(self, gene: str, species: str) -> Optional[SeqRecord]:
        """
        Resolve gene + species to a SeqRecord.
        
        Lookup strategy:
        1. Direct lookup by (gene, species)
        2. Lookup with normalized species (explicit human/mouse/rat aliases only)
        
        Does **not** fall back to Homo sapiens when the requested species is missing.
        
        Args:
            gene: Gene name (e.g. "EGFR", "Maoa")
            species: Scientific organism name (e.g. "Homo sapiens", "Rattus norvegicus")
        
        Returns:
            SeqRecord with sequence or None if not found
        """
        if not gene or not species:
            return None
        
        gene_lower = gene.lower()
        species_lower = species.lower()
        
        # Attempt 1: Direct lookup
        key = (gene_lower, species_lower)
        record = self._index.get(key)
        
        if record:
            self.logger.debug(f"Found {gene} ({species}): {len(record.seq)} aa")
            return record
        
        # Attempt 2: Normalized species (explicit aliases only)
        normalized_species = self._normalize_species(species)
        if normalized_species != species_lower:
            key = (gene_lower, normalized_species)
            record = self._index.get(key)
            if record:
                self.logger.debug(f"Found {gene} ({species} → {normalized_species}): {len(record.seq)} aa")
                return record
        
        self.logger.debug(f"Not found: {gene} ({species})")
        return None
    
    def extract_accession_from_id(self, seq_id: str) -> Optional[str]:
        """Extract Accession ID from a string like 'sp|P01112|HRAS_HUMAN'."""
        return self._extract_accession(seq_id)
    
    def extract_uniprot_from_id(self, seq_id: str) -> Optional[str]:
        """Extract UniProt ID from a string like 'sp|P01112|HRAS_HUMAN'."""
        if '|' in seq_id:
            parts = seq_id.split('|')
            if len(parts) >= 3:
                return parts[2]
        return seq_id
    
    def get_stats(self) -> Dict[str, int]:
        """Return statistics for loaded data."""
        return {
            "indexed_gene_species_pairs": len(self._index),
            "total_records": len(self._records_by_accession),
            "initialized": self._initialized
        }
