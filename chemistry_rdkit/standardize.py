"""SMILES standardisation and heavy-atom counting.

Standardisation pipeline (tautomer-canonical, uncharged, largest fragment)
ported from ``final_postprocessing/add_clean_best.py`` so every script in
the repo uses the same logic.
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem


def standardize_smiles(smiles: str) -> Optional[str]:
    """Standardise a SMILES string.

    Pipeline: parse → cleanup → largest fragment → uncharge →
    canonical tautomer → round-trip check.

    Returns the canonical SMILES or *None* on failure.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        smiles = smiles.replace("[N]", "[NH]").replace("[O]", "[OH]")
        smiles = smiles.replace("[C-]#[N+]", "N#C").replace("[2C]", "C")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)

        try:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        except RuntimeError:
            return None

        if mol is None:
            return None

        s = Chem.MolToSmiles(mol)
        return s if Chem.MolFromSmiles(s) is not None else None
    except Exception:
        return None


def get_heavy_atom_count(smiles: str) -> Optional[int]:
    """Return the number of heavy atoms for a SMILES string."""
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol else None
    except Exception:
        return None
