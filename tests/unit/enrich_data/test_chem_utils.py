"""Tests for enrich_data/chem_utils.py — RDKit wrappers."""
import pytest
from rdkit import Chem

from enrich_data.chem_utils import (
    get_molecular_weight,
    mol_to_inchi_key,
    smiles_to_inchi_key,
    smiles_to_mol,
    smiles_to_molecular_weight,
)


# Known ground-truth values, generated directly from RDKit. If RDKit upgrades
# silently change these, we want to know — they propagate into MW conversions.
KNOWN_MOLECULES = [
    # (smiles, average_mw, exact_mw, inchi_key)
    ("O", 18.015, 18.01056, "XLYOFNOQVPJJNP-UHFFFAOYSA-N"),
    ("CCO", 46.069, 46.04186, "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
    ("c1ccccc1", 78.114, 78.04695, "UHOVQNZJYSORNB-UHFFFAOYSA-N"),
    # aspirin
    ("CC(=O)Oc1ccccc1C(=O)O", 180.159, 180.04226, "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"),
]


# ---------------------------------------------------------------------------
# smiles_to_mol
# ---------------------------------------------------------------------------
class TestSmilesToMol:
    @pytest.mark.parametrize("smiles", [s for s, *_ in KNOWN_MOLECULES])
    def test_valid_smiles(self, smiles):
        mol = smiles_to_mol(smiles)
        assert mol is not None
        assert isinstance(mol, Chem.Mol)

    @pytest.mark.parametrize("smiles", ["", None])
    def test_falsy_inputs_return_none(self, smiles):
        assert smiles_to_mol(smiles) is None

    @pytest.mark.parametrize("smiles", ["@@@invalid", "ZZZZZ", "C(C("])
    def test_invalid_returns_none_no_exception(self, smiles):
        # RDKit may print to stderr but the wrapper must swallow and return None.
        assert smiles_to_mol(smiles) is None


# ---------------------------------------------------------------------------
# mol_to_inchi_key
# ---------------------------------------------------------------------------
class TestMolToInchiKey:
    def test_none_input(self):
        assert mol_to_inchi_key(None) is None

    @pytest.mark.parametrize("smiles,_avg,_exact,inchi", KNOWN_MOLECULES)
    def test_known_molecules(self, smiles, _avg, _exact, inchi):
        assert mol_to_inchi_key(Chem.MolFromSmiles(smiles)) == inchi


# ---------------------------------------------------------------------------
# get_molecular_weight
# ---------------------------------------------------------------------------
class TestGetMolecularWeight:
    def test_none_input(self):
        assert get_molecular_weight(None) is None

    @pytest.mark.parametrize("smiles,avg_mw,_exact,_inchi", KNOWN_MOLECULES)
    def test_average_weight(self, smiles, avg_mw, _exact, _inchi):
        result = get_molecular_weight(Chem.MolFromSmiles(smiles), exact=False)
        assert result == pytest.approx(avg_mw, rel=1e-3)

    @pytest.mark.parametrize("smiles,_avg,exact_mw,_inchi", KNOWN_MOLECULES)
    def test_exact_weight(self, smiles, _avg, exact_mw, _inchi):
        result = get_molecular_weight(Chem.MolFromSmiles(smiles), exact=True)
        assert result == pytest.approx(exact_mw, rel=1e-4)

    def test_exact_differs_from_average_for_benzene(self):
        # Sanity: exact (78.047) and average (78.114) must be distinct for at
        # least one common molecule, proving the `exact` flag matters.
        mol = Chem.MolFromSmiles("c1ccccc1")
        avg = get_molecular_weight(mol, exact=False)
        exact = get_molecular_weight(mol, exact=True)
        assert avg != exact
        assert abs(avg - exact) > 0.05


# ---------------------------------------------------------------------------
# smiles_to_inchi_key
# ---------------------------------------------------------------------------
class TestSmilesToInchiKey:
    @pytest.mark.parametrize("smiles,_avg,_exact,inchi", KNOWN_MOLECULES)
    def test_known_molecules(self, smiles, _avg, _exact, inchi):
        assert smiles_to_inchi_key(smiles) == inchi

    @pytest.mark.parametrize("smiles", ["", None, "@@@invalid"])
    def test_invalid_returns_none(self, smiles):
        assert smiles_to_inchi_key(smiles) is None


# ---------------------------------------------------------------------------
# smiles_to_molecular_weight
# ---------------------------------------------------------------------------
class TestSmilesToMolecularWeight:
    @pytest.mark.parametrize("smiles,avg_mw,_exact,_inchi", KNOWN_MOLECULES)
    def test_average(self, smiles, avg_mw, _exact, _inchi):
        assert smiles_to_molecular_weight(smiles) == pytest.approx(avg_mw, rel=1e-3)

    @pytest.mark.parametrize("smiles,_avg,exact_mw,_inchi", KNOWN_MOLECULES)
    def test_exact(self, smiles, _avg, exact_mw, _inchi):
        assert smiles_to_molecular_weight(smiles, exact=True) == pytest.approx(
            exact_mw, rel=1e-4
        )

    @pytest.mark.parametrize("smiles", ["", None, "@@@invalid"])
    def test_invalid_returns_none(self, smiles):
        assert smiles_to_molecular_weight(smiles) is None
