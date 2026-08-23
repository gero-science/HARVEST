import logging

import shelve
from pyopsin.pyopsin import PyOpsin
from typing import Tuple, Optional

from rdkit import Chem, RDLogger

from chemistry_rdkit import (
    smiles_to_molecular_weight,
)


class LigandInfoExtractor:
    def __init__(
        self,
        opsin: Optional[PyOpsin],
        cache: shelve.DbfilenameShelf,
        smiles_cache: shelve.DbfilenameShelf,
    ):
        self.opsin = opsin  # Can be None to disable IUPAC→SMILES conversion
        self.cache = cache
        self.smiles_cache = smiles_cache
        RDLogger.DisableLog('rdApp.*')  # https://github.com/rdkit/rdkit/issues/2683

    def get_inchi_key_by_smiles(self, smiles: str) -> Optional[str]:
        """
        Get InChI key from a SMILES string.

        Priority:
        1. Cache
        2. RDKit (local)
        """
        if smiles in self.smiles_cache:
            cached_inchi_key = self.smiles_cache[smiles]
            # Validate cached data
            if isinstance(cached_inchi_key, str):
                return cached_inchi_key
            else:
                cache_path = getattr(self.cache, 'cache_path', 'unknown_cache_path')
                logging.error(f"Invalid cache data types for SMILES {smiles}. Remove {cache_path} and rerun.")
                exit(1)
        try:
            mol = Chem.MolFromSmiles(smiles)
            inchi = Chem.MolToInchiKey(mol)
            if inchi:
                # Cache the RDKit result
                self.smiles_cache[smiles] = inchi
                return inchi
        except Exception:
            pass
        return None

    def extract(self, name: str) -> Tuple[Optional[str], Optional[str]]:
        if name in self.cache:
            cached_data = self.cache[name]
            # Validate cached data format
            if isinstance(cached_data, tuple) and len(cached_data) == 2:
                inchi_key, smiles = cached_data
                # Additional validation: ensure all elements are strings or None
                if all(isinstance(x, (str, type(None))) for x in cached_data):
                    return inchi_key, smiles
                else:
                    cache_path = getattr(self.cache, 'cache_path', 'unknown_cache_path')
                    logging.error(
                        f"Invalid cache data types for {name}. Remove {cache_path} and rerun.")
                    exit(1)
            else:
                cache_path = getattr(self.cache, 'cache_path', 'unknown_cache_path')
                logging.error(
                    f"Invalid cache data types for {name}. Remove {cache_path} and rerun.")
                exit(1)
        
        # If opsin is disabled, skip IUPAC→SMILES conversion
        if self.opsin is None:
            return None, None
        
        smiles = self.opsin.to_smiles(name.replace('cis-', '').replace('trans-', ''))
        if len(smiles) > 0 and smiles[0] is not None:
            inchi_key = self.get_inchi_key_by_smiles(smiles[0])
            if inchi_key:
                self.cache[name] = inchi_key, smiles[0]
                return inchi_key, smiles[0]
        inchi_key, smiles = None, None
        self.cache[name] = inchi_key, smiles
        return inchi_key, smiles

    def get_molecular_weight(self, smiles: str) -> Optional[float]:
        """
        Get molecular weight (g/mol or Da) from a SMILES string.
        
        Args:
            smiles: Molecule SMILES string
            
        Returns:
            Molecular weight in g/mol (Da), or None if calculation failed
        """
        if not smiles:
            return None
        
        # Use the chem_utils helper
        mw = smiles_to_molecular_weight(smiles, exact=False)
        if mw is None:
            logging.warning(f"Failed to calculate molecular weight for SMILES: {smiles}")
        
        return mw
