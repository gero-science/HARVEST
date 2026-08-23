"""Basic RDKit helpers: SMILES parsing, InChI Key, molecular weight."""

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert a SMILES string to an RDKit molecule."""
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.debug(f"Failed to parse SMILES: {smiles}")
        return mol
    except Exception as e:
        logging.debug(f"Error parsing SMILES {smiles}: {e}")
        return None


def mol_to_inchi_key(mol: Chem.Mol) -> Optional[str]:
    """Generate an InChIKey from an RDKit molecule."""
    if mol is None:
        return None

    try:
        inchi_key = Chem.MolToInchiKey(mol)
        return inchi_key if inchi_key else None
    except Exception as e:
        logging.debug(f"Error generating InChI Key: {e}")
        return None


def get_molecular_weight(mol: Chem.Mol, exact: bool = False) -> Optional[float]:
    """Return molecular weight in Da for an RDKit molecule."""
    if mol is None:
        return None

    try:
        if exact:
            mw = rdMolDescriptors.CalcExactMolWt(mol)
        else:
            mw = Descriptors.MolWt(mol)

        return float(mw) if mw else None
    except Exception as e:
        logging.debug(f"Error calculating molecular weight: {e}")
        return None


def smiles_to_inchi_key(smiles: str) -> Optional[str]:
    """Convert a SMILES string to an InChIKey."""
    mol = smiles_to_mol(smiles)
    return mol_to_inchi_key(mol) if mol else None


def smiles_to_molecular_weight(smiles: str, exact: bool = False) -> Optional[float]:
    """Calculate molecular weight from a SMILES string."""
    mol = smiles_to_mol(smiles)
    return get_molecular_weight(mol, exact=exact) if mol else None


def is_scaffold(mol: Chem.Mol) -> bool:
    """Check if a molecule is a scaffold (has attachment-point or placeholder atoms).

    Detects atoms with atomic number 0 (dummy/wildcard), 99 (Einsteinium,
    used as R-group placeholder), or 100 (Fermium, same convention).
    """
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in (0, 99, 100):
            return True
    return False
