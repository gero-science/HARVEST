"""
Data source for patents from ZIP archives.
Handles retrieval and delivery of data from ZIP archives containing XML and MOL files.
"""

import os
import sys
import logging
import zipfile
import tempfile
from typing import Iterator, Optional, Dict, Any
from pathlib import Path
import re

from lxml import etree

# RDKit import for MOL file processing (Iteration 2)
try:
    from rdkit import Chem
    from rdkit.Chem import inchi
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. SMILES extraction will be disabled.")

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from chemistry_rdkit import get_molecular_weight  # type: ignore
    CHEM_UTILS_AVAILABLE = True
except ImportError:
    CHEM_UTILS_AVAILABLE = False
    logging.debug("chem_utils not available, falling back to RDKit Descriptors")

# Import functions for scaffold detection
try:
    from .mol_processor import is_scaffold_mol_file, has_dummy_atoms
    MOL_PROCESSOR_AVAILABLE = True
except ImportError:
    MOL_PROCESSOR_AVAILABLE = False
    logging.warning("mol_processor not available. Scaffold detection will be disabled.")

from .cdx_extractor import extract_chem_num_from_cdx_filename, parse_cdx_bytes
from .data_structures import PatentDocument
from .xml_patent_parser import XMLPatentParser


class ZipPatentSource:
    """
    Data source from ZIP archives.
    Reads patents from ZIP archives containing XML and MOL files.
    """
    
    def __init__(self, zip_file_path: str, batch_size: int = 1000, parse_cdx: bool = True):
        """
        Initialize the data source.
        
        Args:
            zip_file_path: Path to the ZIP file.
            batch_size: Batch size for processing (not used in the current implementation, kept for compatibility).
            parse_cdx: Parse CDX files from the same archive alongside MOL files.
        """
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(f"ZIP file not found: {zip_file_path}")
            
        self.zip_file_path = zip_file_path
        self.batch_size = batch_size
        self.parse_cdx = parse_cdx
        self.parser = XMLPatentParser()
        self.logger = logging.getLogger(__name__)
        
        # Cache for SMILES data (Iteration 2): patent_id -> {chem_num -> {filename, smiles, is_scaffold?}}
        # For scaffolds, store only filename and is_scaffold=True (without smiles)
        self.smiles_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Cache for CDX data: patent_id -> {chem_num -> {smiles, inchikey}}
        self.cdx_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
    def _extract_patent_id_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract patent ID from filename.
        Example: US20100003239A1-20100107.xml -> US20100003239A1
        """
        # Remove extension and extract ID up to the first date hyphen
        basename = Path(filename).stem
        # Pattern for patent ID (before date in -YYYYMMDD format)
        match = re.match(r'^([A-Z]{2}\d+[A-Z]\d+)', basename)
        return match.group(1) if match else None
        
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from filename.
        Example: US20100003239A1-20100107.xml -> {patent_id: US20100003239A1, date: 20100107}
        """
        basename = Path(filename).stem
        metadata = {}
        
        # Extract patent_id and date
        match = re.match(r'^([A-Z]{2}\d+[A-Z]\d+)-(\d{8})', basename)
        if match:
            metadata['patent_id'] = match.group(1)
            metadata['publication_date'] = match.group(2)
            
        return metadata
        
    def _extract_chem_num_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract chemical compound number from MOL filename.
        Example: US20100003239A1-20100107-C00001.MOL -> 00001
        """
        match = re.search(r'-C(\d+)\.MOL$', filename, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _is_valid_single_molecule(self, mol, mol_filename: str) -> tuple[bool, Optional[str]]:
        """
        Check whether the molecule is a valid single structure.
        
        Returns:
            tuple[bool, Optional[str]]: (is_valid, rejection_reason)
        """
        # Check 1: Fragment count (strictly = 1)
        frags = Chem.GetMolFrags(mol, asMols=False)
        if len(frags) > 1:
            frag_sizes = sorted([len(f) for f in frags], reverse=True)
            return False, f"multiple fragments ({len(frags)}), sizes: {frag_sizes}"
        
        # Check 2: Query atoms
        for atom in mol.GetAtoms():
            if atom.HasQuery():
                return False, f"query atom at index {atom.GetIdx()}"
        
        # Check 3: Query bonds
        for bond in mol.GetBonds():
            if bond.HasQuery():
                return False, f"query bond between atoms {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}"
        
        # Check 4: Dummy atoms (atomic number 0 or *)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                return False, f"dummy atom (*) at index {atom.GetIdx()}"
        
        return True, None
    
    def _validate_and_convert_mol(self, mol_content: str, mol_filename: str) -> tuple[Optional[str], Optional[str], Optional[float]]:
        """
        Validate a MOL file, attempt to fix errors, and convert to SMILES, InChIKey, and molecular weight.
        Uses a two-stage loading strategy for improved reliability.
        
        Args:
            mol_content: MOL file content
            mol_filename: MOL filename for logging
            
        Returns:
            tuple[Optional[str], Optional[str], Optional[float]]: (SMILES, InChIKey, MolecularWeight) or (None, None, None)
        """
        if not RDKIT_AVAILABLE:
            self.logger.debug(f"RDKit not available, skipping MOL file: {mol_filename}")
            return None, None, None
            
        try:
            # Stage 1: Standard loading with sanitizer
            try:
                mol = Chem.MolFromMolBlock(mol_content, sanitize=True)
            except Exception as e:
                self.logger.debug(f"Primary loading failed for {mol_filename}: {e}")
                mol = None
            
            # Stage 2: If the first attempt fails, try without sanitizer and fix manually
            if mol is None:
                self.logger.debug(f"Attempting fallback loading for {mol_filename}")
                try:
                    mol = Chem.MolFromMolBlock(mol_content, sanitize=False)
                    if mol is None:
                        self.logger.debug(f"RDKit failed to parse MOL file: {mol_filename}")
                        return None, None, None
                    
                    # Attempt to fix the structure
                    try:
                        Chem.Kekulize(mol, clearAromaticFlags=True)
                        self.logger.debug(f"Successfully kekulized {mol_filename}")
                    except Exception as e:
                        # Kekulization errors may occur; not always critical
                        self.logger.debug(f"Kekulization warning for {mol_filename}: {e}")
                        pass
                    
                    # Run full sanitizer with problem fixing
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
                    self.logger.debug(f"Successfully sanitized {mol_filename} after manual fixes")
                    
                except Exception as e:
                    self.logger.debug(f"Fallback loading and sanitization failed for {mol_filename}: {e}")
                    return None, None, None

            # Stage 2.5: Validity check (single molecule, no query/dummy)
            is_valid, rejection_reason = self._is_valid_single_molecule(mol, mol_filename)
            if not is_valid:
                self.logger.debug(f"Rejected MOL file {mol_filename}: {rejection_reason}")
                return None, None, None

            # Stage 3: Generate SMILES, InChIKey, and molecular weight
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                if not smiles:
                    self.logger.warning(f"Empty SMILES generated for {mol_filename}")
                    return None, None, None
                    
                inchikey = inchi.MolToInchiKey(mol)
                if not inchikey:
                    self.logger.warning(f"Empty InChIKey generated for {mol_filename}")
                    return None, None, None
                
                # Calculate molecular weight in g/mol
                try:
                    if CHEM_UTILS_AVAILABLE:
                        # Use utility (average molecular weight)
                        mol_weight = get_molecular_weight(mol, exact=False)
                    else:
                        # Fallback: direct RDKit call
                        from rdkit.Chem import Descriptors
                        mol_weight = Descriptors.MolWt(mol)

                    if mol_weight:
                        self.logger.debug(f"Molecular weight calculated for {mol_filename}: {mol_weight:.4f} g/mol")
                except Exception as e:
                    self.logger.warning(f"Failed to calculate molecular weight for {mol_filename}: {e}")
                    mol_weight = None
                    
                mw_str = f"{mol_weight:.4f}" if mol_weight else "N/A"
                self.logger.debug(f"Successfully converted MOL: {mol_filename} -> SMILES: {smiles}, InChIKey: {inchikey}, MW: {mw_str}")
                return smiles, inchikey, mol_weight
                
            except Exception as e:
                self.logger.error(f"Error generating SMILES/InChIKey for {mol_filename}: {e}")
                return None, None, None
                
        except Exception as e:
            self.logger.error(f"Unexpected error processing MOL file {mol_filename}: {e}")
            return None, None, None

    @staticmethod
    def read_text_different_encodings(file_text):
        """
        Returns text from file, and is_encoding_error flag
        """
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                return file_text.read().decode(enc), False
            except Exception:
                continue
        return file_text.read().decode('utf-8', errors="ignore"), True

    def _process_mol_files(self, zip_file: zipfile.ZipFile, patent_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Process all MOL files for the given patent in the ZIP archive.
        Skips scaffolds (incomplete molecules) and stores information about them in smiles_cache with is_scaffold marker.

        Args:
            zip_file: Open ZIP archive
            patent_id: Patent ID
            
        Returns:
            Dict: Dictionary {chem_num -> {filename, smiles?}} or {chem_num -> {filename, is_scaffold: True}}
        """
        mol_data = {}
        
        # Counters for aggregated statistics
        mol_stats = {'valid': 0, 'scaffolds': 0, 'rejected': 0, 'failed': 0, 'empty': 0}
        
        # Find MOL files for this patent
        mol_pattern = f"{patent_id}"  # Pattern for searching patent files
        
        mol_files = [f for f in zip_file.namelist() 
                    if f.lower().endswith('.mol') and mol_pattern in f]
        
        self.logger.debug(f"Found {len(mol_files)} MOL files for patent {patent_id}")
        
        for mol_filename in mol_files:
            try:
                # Extract compound number
                chem_num = self._extract_chem_num_from_filename(mol_filename)
                if not chem_num:
                    self.logger.warning(f"Cannot extract chem_num from filename: {mol_filename}")
                    continue
                
                # Read MOL file content
                with zip_file.open(mol_filename) as mol_file:
                    mol_content, is_encoding_error = self.read_text_different_encodings(mol_file)

                if not mol_content or not mol_content.strip():
                    self.logger.debug(f"Empty MOL file: {mol_filename}")
                    mol_stats['empty'] += 1
                    continue
                
                # Check whether the molecule is a scaffold (incomplete molecule)
                is_scaffold = False
                if MOL_PROCESSOR_AVAILABLE:
                    try:
                        # Create temporary file for scaffold check
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as temp_file:
                            temp_file.write(mol_content)
                            temp_file_path = temp_file.name

                        try:
                            # Check whether the file is a scaffold
                            is_scaffold = is_scaffold_mol_file(temp_file_path)

                            # Additional check via RDKit (if available)
                            if not is_scaffold and RDKIT_AVAILABLE:
                                try:
                                    mol = Chem.MolFromMolBlock(mol_content, sanitize=False)
                                    if mol is not None:
                                        is_scaffold = has_dummy_atoms(mol)
                                except Exception:
                                    pass
                        finally:
                            # Remove temporary file
                            if os.path.exists(temp_file_path):
                                os.unlink(temp_file_path)
                    except Exception as e:
                        self.logger.debug(f"Error checking scaffold for {mol_filename}: {e}")

                # If scaffold - skip processing, but store information in cache
                if is_scaffold:
                    self.logger.debug(f"Skipping scaffold (incomplete molecule) MOL file: {mol_filename} -> chem_num: {chem_num}")
                    mol_data[chem_num] = {
                        'filename': mol_filename,
                        'is_scaffold': True,
                        'is_encoding_error': is_encoding_error,
                    }
                    mol_stats['scaffolds'] += 1
                    continue

                # Validate and convert to SMILES, InChIKey, and molecular weight
                smiles, inchikey, mol_weight = self._validate_and_convert_mol(mol_content, mol_filename)
                if smiles and inchikey:
                    mol_data[chem_num] = {
                        'filename': mol_filename,
                        'smiles': smiles,
                        'inchikey': inchikey,
                        'mol_weight': mol_weight,
                        'is_encoding_error': is_encoding_error,
                    }
                    mw_str = f"{mol_weight:.4f}" if mol_weight else "N/A"
                    self.logger.debug(f"Processed MOL file: {mol_filename} -> chem_num: {chem_num}, SMILES: {smiles[:50]}..., mol weight: {mw_str}")
                    mol_stats['valid'] += 1
                else:
                    self.logger.debug(f"Failed to process MOL file: {mol_filename} -> chem_num: {chem_num}")
                    mol_stats['rejected'] += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to process MOL file {mol_filename}: {e}")
                mol_stats['failed'] += 1
                continue
        
        # Aggregated message about MOL file processing results
        if mol_files:
            self.logger.info(
                f"MOL processing for {patent_id}: {mol_stats['valid']} valid, "
                f"{mol_stats['scaffolds']} scaffolds, {mol_stats['rejected']} rejected, "
                f"{mol_stats['empty']} empty, {mol_stats['failed']} failed"
            )
        
        return mol_data
    
    def _process_cdx_files(self, zip_file: zipfile.ZipFile, patent_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Process all CDX files for the given patent in the ZIP archive.
        Runs next to MOL processing while the archive is already open, so the final
        export does not need a second pass over ZIP files.

        Args:
            zip_file: Open ZIP archive
            patent_id: Patent ID

        Returns:
            Dict: Dictionary {chem_num -> {smiles, inchikey}}. Compounds that could
                  not be converted are kept with None values.
        """
        cdx_data: Dict[str, Dict[str, Any]] = {}

        cdx_stats = {'valid': 0, 'unusable': 0, 'failed': 0}

        cdx_files = [f for f in zip_file.namelist()
                     if f.lower().endswith('.cdx') and patent_id in f]

        self.logger.debug(f"Found {len(cdx_files)} CDX files for patent {patent_id}")

        for cdx_filename in cdx_files:
            try:
                chem_num = extract_chem_num_from_cdx_filename(cdx_filename)
                if not chem_num:
                    self.logger.warning(f"Cannot extract chem_num from filename: {cdx_filename}")
                    continue

                cdx_bytes = zip_file.read(cdx_filename)
                if not cdx_bytes:
                    self.logger.debug(f"Empty CDX file: {cdx_filename}")
                    cdx_stats['unusable'] += 1
                    continue

                smiles, inchikey = parse_cdx_bytes(cdx_bytes)
                if smiles:
                    cdx_stats['valid'] += 1
                    self.logger.debug(f"Processed CDX file: {cdx_filename} -> chem_num: {chem_num}, SMILES: {smiles[:50]}...")
                else:
                    # Scaffolds and unparsable structures: recorded, but without structure
                    cdx_stats['unusable'] += 1

                compound = {'smiles': smiles, 'inchikey': inchikey}
                for key in self._cdx_cache_keys(chem_num):
                    cdx_data[key] = compound

            except Exception as e:
                self.logger.error(f"Failed to process CDX file {cdx_filename}: {e}")
                cdx_stats['failed'] += 1
                continue

        if cdx_files:
            self.logger.info(
                f"CDX processing for {patent_id}: {cdx_stats['valid']} valid, "
                f"{cdx_stats['unusable']} without structure, {cdx_stats['failed']} failed"
            )

        return cdx_data
    
    @staticmethod
    def _cdx_cache_keys(chem_num: str) -> list[str]:
        """
        Build cache keys for a compound number extracted from a CDX filename.

        Downstream lookup uses the number as written in chemical_id, which appears both
        zero-padded (CHEM-US-00013) and bare (CHEM-US-13), so store the entry under both.
        """
        keys = [chem_num]
        stripped = chem_num.lstrip('0') or '0'
        if stripped != chem_num:
            keys.append(stripped)
        return keys
    
    def get_smiles_for_patent(self, patent_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Return SMILES data and scaffold information for the specified patent from cache.
        
        Args:
            patent_id: Patent ID
            
        Returns:
            Dict or None: {chem_num -> {filename, smiles?} or {filename, is_scaffold: True}}
        """
        return self.smiles_cache.get(patent_id)

    def get_cdx_for_patent(self, patent_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Return CDX structure data for the specified patent from cache.
        
        Args:
            patent_id: Patent ID
            
        Returns:
            Dict or None: {chem_num -> {smiles, inchikey}}
        """
        return self.cdx_cache.get(patent_id)

    @staticmethod
    def extract_patent_title(xml_root: Optional[etree._Element]):
        """
        Extract title <invention-title id="d0e43">HETEROARYLOXYCARBOCYCLYL COMPOUNDS AS PDE10 INHIBITORS</invention-title>    """
        if xml_root is None:
            return ""

        try:
            title = str(xml_root.xpath('normalize-space(string(//invention-title))'))
            return title

        except Exception:
            return None
        
    def iter_patents(self) -> Iterator[PatentDocument]:
        """
        Iterate over patents from the ZIP file.
        
        Yields:
            PatentDocument: Patent document object.
        """
        try:
            self.logger.info(f"Starting to read ZIP file: {self.zip_file_path}")
            
            total_processed = 0
            total_errors = 0
            total_mol_processed = 0
            total_cdx_processed = 0
            
            with zipfile.ZipFile(self.zip_file_path, 'r') as zip_file:
                # Get list of all files in the archive
                file_list = zip_file.namelist()
                xml_files = [f for f in file_list if f.lower().endswith('.xml')]
                
                self.logger.info(f"Found {len(xml_files)} XML files in ZIP archive")
                
                if not xml_files:
                    self.logger.warning("No XML files found in ZIP archive")
                    return
                
                for xml_filename in xml_files:
                    try:
                        # Read XML content
                        with zip_file.open(xml_filename) as xml_file:
                            xml_content = xml_file.read().decode('utf-8')
                        
                        if not xml_content or not xml_content.strip():
                            self.logger.warning(f"Empty XML content in file: {xml_filename}")
                            continue
                        
                        # Extract metadata from filename
                        metadata = self._extract_metadata_from_filename(xml_filename)
                        patent_id = metadata.get('patent_id')
                        
                        if not patent_id:
                            # Fallback: use filename as patent_id
                            patent_id = Path(xml_filename).stem
                        
                        # NEW STEP: Process MOL files for this patent (Iteration 2)
                        mol_data = self._process_mol_files(zip_file, patent_id)
                        if mol_data:
                            self.smiles_cache[patent_id] = mol_data
                            total_mol_processed += len(mol_data)
                            self.logger.debug(f"Cached {len(mol_data)} SMILES entries for patent {patent_id}")
                        
                        # Process CDX files from the same archive
                        if self.parse_cdx:
                            cdx_data = self._process_cdx_files(zip_file, patent_id)
                            if cdx_data:
                                self.cdx_cache[patent_id] = cdx_data
                                total_cdx_processed += len(cdx_data)
                                self.logger.debug(f"Cached {len(cdx_data)} CDX entries for patent {patent_id}")
                        
                        # Parse XML content
                        xml_root, parse_error = self.parser.parse(xml_content)

                        patent_title = self.extract_patent_title(xml_root)
                        if not patent_title:
                            self.logger.warning(f"Could not extract patent title for {self.zip_file_path}")
                        
                        # Create PatentDocument
                        document = PatentDocument(
                            file_path=f"zip://{self.zip_file_path}#{xml_filename}",
                            patent_id=patent_id,
                            publication_number=patent_id,  # Usually the same in ZIP patents
                            publication_date=metadata.get('publication_date'),
                            patent_title=patent_title,
                            abstract=None,  # Will be extracted from XML if needed
                            description_html=xml_content,
                            xml_root=xml_root,
                            parsing_errors=[parse_error] if parse_error else []
                        )
                        
                        if parse_error:
                            total_errors += 1
                            self.logger.warning(f"XML parsing error in patent {patent_id}: {parse_error}")
                        
                        total_processed += 1
                        yield document
                        
                    except Exception as e:
                        total_errors += 1
                        self.logger.error(f"Failed to process XML file {xml_filename}: {e}")
                        
                        # Create document with error, but continue processing
                        patent_id = self._extract_patent_id_from_filename(xml_filename) or Path(xml_filename).stem
                        yield PatentDocument(
                            file_path=f"zip://{self.zip_file_path}#{xml_filename}",
                            patent_id=patent_id,
                            publication_number=patent_id,
                            parsing_errors=[f"Failed to process XML file: {e}"]
                        )
                
                self.logger.info(f"Completed ZIP processing: {total_processed} documents processed, "
                               f"{total_mol_processed} SMILES entries cached, "
                               f"{total_cdx_processed} CDX entries cached, {total_errors} errors")
                
        except Exception as e:
            self.logger.critical(f"Critical error reading ZIP file {self.zip_file_path}: {e}")
            raise
    
    def get_total_count(self) -> Optional[int]:
        """
        Return the total number of XML files in the ZIP archive.
        
        Returns:
            int: Number of XML files or None on error.
        """
        try:
            with zipfile.ZipFile(self.zip_file_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                xml_count = len([f for f in file_list if f.lower().endswith('.xml')])
                return xml_count
        except Exception as e:
            self.logger.error(f"Error counting files in ZIP: {e}")
            return None
