"""
Shared CDX parsing helpers.

Used both by the extraction stage (patent_processor.zip_source, which reads CDX
files from the patent ZIP while it is already open) and by the standalone
the CDX parsing path.
"""

import logging
import os
import re
import sys
from typing import Optional, Tuple

# Add project root to path for imports of root-level modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rdkit import Chem

    from cdx_reader import read_cdx_molecule
    from chemistry_rdkit import is_scaffold

    CDX_PARSING_AVAILABLE = True
except ImportError:
    CDX_PARSING_AVAILABLE = False
    logging.debug("CDX parsing dependencies not available (rdkit/pycdxml)")

# Unknown ChemDraw fragment labels and unfixable valences are expected for
# scaffold-heavy patents: such compounds are simply recorded without a structure.
# Both loggers are only reachable through CDX parsing, and a single patent can hold
# hundreds of CDX files, so keep them out of the pipeline log. Same approach
# cdx_reader already uses for pycdxml.
logging.getLogger('cdx_reader').setLevel(logging.CRITICAL)


CDX_FILENAME_CHEM_NUM_PATTERN = re.compile(r"-C(\d+)\.CDX$", re.IGNORECASE)


def extract_chem_num_from_cdx_filename(filename: str) -> Optional[str]:
    """
    Extract chemical compound number from a CDX filename.
    Example: US20100003239A1-20100107-C00001.CDX -> 00001
    """
    if not filename:
        return None

    match = CDX_FILENAME_CHEM_NUM_PATTERN.search(filename)
    return match.group(1) if match else None


def parse_cdx_bytes(cdx_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert CDX file bytes to SMILES and InChIKey.

    Scaffolds (structures with attachment points) are rejected the same way as
    scaffold MOL files: they must not be used as full molecules.

    Returns:
        Tuple (smiles, inchikey) or (None, None)
    """
    if not CDX_PARSING_AVAILABLE:
        return None, None

    try:
        mol = read_cdx_molecule(cdx_bytes, as_single=True)
        if mol is None or is_scaffold(mol):
            return None, None

        smiles = Chem.MolToSmiles(mol, canonical=True)
        inchikey = Chem.MolToInchiKey(mol)
        return smiles, inchikey
    except Exception:
        return None, None
