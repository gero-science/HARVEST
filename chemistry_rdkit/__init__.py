"""Neutral RDKit helpers shared by patent parsing and BindingDB export.

This package contains local structure utilities: SMILES parsing,
InChI Key computation, molecular-weight calculation, SMILES
standardisation, scaffold detection, and IUPAC name normalization.
"""

from .core import (
    get_molecular_weight,
    is_scaffold,
    mol_to_inchi_key,
    smiles_to_inchi_key,
    smiles_to_mol,
    smiles_to_molecular_weight,
)
from .iupac import (
    fix_balanced,
    fix_balanced_by_adding,
    fix_balanced_by_removing,
    fix_balanced_by_removing_variants,
    is_balanced,
    normalize_iupac_spelling,
)
from .standardize import (
    get_heavy_atom_count,
    standardize_smiles,
)

__all__ = [
    "fix_balanced",
    "fix_balanced_by_adding",
    "fix_balanced_by_removing",
    "fix_balanced_by_removing_variants",
    "get_heavy_atom_count",
    "get_molecular_weight",
    "is_balanced",
    "is_scaffold",
    "mol_to_inchi_key",
    "normalize_iupac_spelling",
    "smiles_to_inchi_key",
    "smiles_to_mol",
    "smiles_to_molecular_weight",
    "standardize_smiles",
]
