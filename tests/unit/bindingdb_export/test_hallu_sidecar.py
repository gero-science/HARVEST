"""Tests for hallu sidecar loading and filtering in bindingdb_export."""

import json
import os
import tempfile

import pytest

from bindingdb_export.artifacts import is_hallucinated, load_hallu_flags, mask_hallucinated_fields
from bindingdb_export.rows import select_ligand_smiles


# ---------------------------------------------------------------------------
# load_hallu_flags
# ---------------------------------------------------------------------------


class TestLoadHalluFlags:
    def test_returns_none_when_absent(self, tmp_path):
        """No _hallu.json → None (annotation not run)."""
        patent_dir = tmp_path / "US12345A1"
        patent_dir.mkdir()
        assert load_hallu_flags(str(patent_dir)) is None

    def test_loads_valid_sidecar(self, tmp_path):
        """A well-formed _hallu.json is returned as a dict."""
        patent_dir = tmp_path / "US12345A1"
        patent_dir.mkdir()
        payload = {
            "detector_version": "1.0",
            "flagged_compounds": {"CHEM-US-00001": ["chem_id"]},
            "flagged_iupac": ["made-up-iupac"],
            "flagged_values": [],
        }
        (patent_dir / "US12345A1_hallu.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        result = load_hallu_flags(str(patent_dir))
        assert result == payload

    def test_returns_none_on_malformed_json(self, tmp_path):
        """Corrupt JSON → None with a warning, not a crash."""
        patent_dir = tmp_path / "US12345A1"
        patent_dir.mkdir()
        (patent_dir / "US12345A1_hallu.json").write_text(
            "{bad json", encoding="utf-8"
        )
        result = load_hallu_flags(str(patent_dir))
        assert result is None

    def test_empty_sidecar(self, tmp_path):
        """An empty sidecar (no flags) is a valid dict."""
        patent_dir = tmp_path / "US12345A1"
        patent_dir.mkdir()
        payload = {
            "detector_version": "1.0",
            "flagged_compounds": {},
            "flagged_iupac": [],
            "flagged_values": [],
        }
        (patent_dir / "US12345A1_hallu.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        result = load_hallu_flags(str(patent_dir))
        assert result is not None
        assert result["flagged_compounds"] == {}


# ---------------------------------------------------------------------------
# is_hallucinated  (backward-compat wrapper)
# ---------------------------------------------------------------------------


class TestIsHallucinated:
    FLAGS = {
        "detector_version": "1.0",
        "flagged_compounds": {
            "CHEM-US-00001": ["chem_id"],
            "CHEM-US-00099": ["iupac", "mismatch"],
        },
        "flagged_iupac": [
            "made-up-iupac-name",
            "US12345A1-20200101-C00999.TIF",
        ],
        "flagged_values": [],
    }

    def test_chem_id_match(self):
        binding = {"chemical_id": "CHEM-US-00001", "compound_IUPAC_name": ""}
        assert is_hallucinated(binding, self.FLAGS) is True

    def test_chem_id_no_match(self):
        binding = {"chemical_id": "CHEM-US-00050", "compound_IUPAC_name": ""}
        assert is_hallucinated(binding, self.FLAGS) is False

    def test_iupac_match(self):
        binding = {
            "chemical_id": "CHEM-US-00050",
            "compound_IUPAC_name": "made-up-iupac-name",
        }
        assert is_hallucinated(binding, self.FLAGS) is True

    def test_tif_as_iupac_match(self):
        binding = {
            "chemical_id": "",
            "compound_IUPAC_name": "US12345A1-20200101-C00999.TIF",
        }
        assert is_hallucinated(binding, self.FLAGS) is True

    def test_no_match(self):
        binding = {
            "chemical_id": "CHEM-US-00050",
            "compound_IUPAC_name": "real-iupac-name",
        }
        assert is_hallucinated(binding, self.FLAGS) is False

    def test_empty_binding(self):
        binding = {}
        assert is_hallucinated(binding, self.FLAGS) is False

    def test_empty_flags(self):
        empty_flags = {
            "flagged_compounds": {},
            "flagged_iupac": [],
            "flagged_values": [],
        }
        binding = {"chemical_id": "CHEM-US-00001", "compound_IUPAC_name": "x"}
        assert is_hallucinated(binding, empty_flags) is False

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace in binding values should not prevent matching."""
        binding = {
            "chemical_id": " CHEM-US-00001 ",
            "compound_IUPAC_name": "",
        }
        assert is_hallucinated(binding, self.FLAGS) is True

    def test_none_values_safe(self):
        """None in chemical_id/compound_IUPAC_name should not crash."""
        binding = {"chemical_id": None, "compound_IUPAC_name": None}
        assert is_hallucinated(binding, self.FLAGS) is False


# ---------------------------------------------------------------------------
# mask_hallucinated_fields
# ---------------------------------------------------------------------------


class TestMaskHallucinatedFields:
    """Test field-level masking instead of whole-binding drop."""

    FLAGS = {
        "detector_version": "1.0",
        "flagged_compounds": {"CHEM-US-00001": ["chem_id"]},
        "flagged_iupac": ["made-up-iupac-name"],
        "flagged_values": [
            {"compound": "A26", "value": "0.019 to 0.026", "binding_metric": "IC50"},
        ],
    }

    def test_masks_chemical_id(self):
        binding = {
            "chemical_id": "CHEM-US-00001",
            "compound_IUPAC_name": "real-name",
            "value": 100.0,
        }
        masked = mask_hallucinated_fields(binding, self.FLAGS)
        assert masked == {"chemical_id"}
        assert binding["chemical_id"] == ""
        # Other fields untouched
        assert binding["compound_IUPAC_name"] == "real-name"
        assert binding["value"] == 100.0

    def test_masks_iupac(self):
        binding = {
            "chemical_id": "CHEM-US-00050",
            "compound_IUPAC_name": "made-up-iupac-name",
            "value": 100.0,
        }
        masked = mask_hallucinated_fields(binding, self.FLAGS)
        assert masked == {"compound_IUPAC_name"}
        assert binding["compound_IUPAC_name"] == ""
        assert binding["chemical_id"] == "CHEM-US-00050"

    def test_masks_value(self):
        binding = {
            "chemical_id": "",
            "compound_IUPAC_name": "",
            "compound": "A26",
            "original_value": "0.019 to 0.026",
            "binding_metric": "IC50",
            "value": 0.022,
        }
        masked = mask_hallucinated_fields(binding, self.FLAGS)
        assert masked == {"value"}
        assert binding["value"] is None

    def test_value_match_requires_compound(self):
        """Value match checks compound too — different compound not masked."""
        binding = {
            "chemical_id": "",
            "compound_IUPAC_name": "",
            "compound": "B99",  # different compound
            "original_value": "0.019 to 0.026",
            "binding_metric": "IC50",
            "value": 0.022,
        }
        masked = mask_hallucinated_fields(binding, self.FLAGS)
        assert masked == set()
        assert binding["value"] == 0.022

    def test_multiple_masks(self):
        """Both chem_id and IUPAC flagged → both masked, binding survives."""
        binding = {
            "chemical_id": "CHEM-US-00001",
            "compound_IUPAC_name": "made-up-iupac-name",
            "compound": "A26",
            "original_value": "0.019 to 0.026",
            "binding_metric": "IC50",
            "value": 0.022,
        }
        masked = mask_hallucinated_fields(binding, self.FLAGS)
        assert masked == {"chemical_id", "compound_IUPAC_name", "value"}
        assert binding["chemical_id"] == ""
        assert binding["compound_IUPAC_name"] == ""
        assert binding["value"] is None

    def test_dry_run_no_mutation(self):
        """dry_run=True reports what would be masked without changing anything."""
        binding = {
            "chemical_id": "CHEM-US-00001",
            "compound_IUPAC_name": "made-up-iupac-name",
            "value": 100.0,
        }
        masked = mask_hallucinated_fields(binding, self.FLAGS, dry_run=True)
        assert masked == {"chemical_id", "compound_IUPAC_name"}
        # Binding unchanged
        assert binding["chemical_id"] == "CHEM-US-00001"
        assert binding["compound_IUPAC_name"] == "made-up-iupac-name"

    def test_clean_binding_no_mask(self):
        binding = {
            "chemical_id": "CHEM-US-00050",
            "compound_IUPAC_name": "real-name",
            "compound": "X1",
            "original_value": "5.0",
            "value": 5.0,
        }
        masked = mask_hallucinated_fields(binding, self.FLAGS)
        assert masked == set()

    def test_no_original_value_skips_value_check(self):
        """Bindings without original_value (pre-normalization) skip value match."""
        binding = {
            "chemical_id": "",
            "compound_IUPAC_name": "",
            "compound": "A26",
            "value": 0.022,
            # no original_value key
        }
        masked = mask_hallucinated_fields(binding, self.FLAGS)
        assert "value" not in masked


# ---------------------------------------------------------------------------
# select_ligand_smiles
# ---------------------------------------------------------------------------


class TestSelectLigandSmiles:
    """Test hallu-aware SMILES source selection at build time."""

    # Two compounds with different InChI Key connectivity
    OPSIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
    OPSIN_INCHI = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    CDX_SMILES_SAME = "CC(=O)Oc1ccccc1C(O)=O"  # same compound, different notation
    CDX_INCHI_SAME = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"  # same connectivity
    CDX_SMILES_DIFF = "c1ccc(cc1)O"
    CDX_INCHI_DIFF = "ISWSIDIOOBJBQZ-UHFFFAOYSA-N"  # different connectivity

    def _row(self, *, opsin="", opsin_inchi="", cdx="", cdx_inchi="",
             source="py2opsin", hallu=None):
        r = {
            "Ligand SMILES": opsin,
            "Ligand InChI Key": opsin_inchi,
            "smiles_cdx": cdx,
            "inchi_key_cdx": cdx_inchi,
            "smiles_source": source,
        }
        if hallu is not None:
            r["_hallu_masked"] = hallu
        return r

    def test_opsin_and_cdx_same_picks_opsin(self):
        """Both available, same connectivity → OPSIN selected."""
        row = self._row(
            opsin=self.OPSIN_SMILES, opsin_inchi=self.OPSIN_INCHI,
            cdx=self.CDX_SMILES_SAME, cdx_inchi=self.CDX_INCHI_SAME,
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == self.OPSIN_SMILES
        assert inchi == self.OPSIN_INCHI
        assert src == "py2opsin"

    def test_opsin_and_cdx_differ_picks_cdx(self):
        """Both available, different connectivity → CDX selected."""
        row = self._row(
            opsin=self.OPSIN_SMILES, opsin_inchi=self.OPSIN_INCHI,
            cdx=self.CDX_SMILES_DIFF, cdx_inchi=self.CDX_INCHI_DIFF,
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == self.CDX_SMILES_DIFF
        assert inchi == self.CDX_INCHI_DIFF
        assert src == "cdx_over_opsin"

    def test_opsin_tainted_falls_back_to_cdx(self):
        """IUPAC flagged → OPSIN tainted, CDX used."""
        row = self._row(
            opsin=self.OPSIN_SMILES, opsin_inchi=self.OPSIN_INCHI,
            cdx=self.CDX_SMILES_DIFF, cdx_inchi=self.CDX_INCHI_DIFF,
            hallu={"compound_IUPAC_name"},
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == self.CDX_SMILES_DIFF
        assert inchi == self.CDX_INCHI_DIFF
        assert src == "cdx_fallback"

    def test_cdx_tainted_keeps_opsin(self):
        """chem_id flagged → CDX tainted, OPSIN used."""
        row = self._row(
            opsin=self.OPSIN_SMILES, opsin_inchi=self.OPSIN_INCHI,
            cdx=self.CDX_SMILES_DIFF, cdx_inchi=self.CDX_INCHI_DIFF,
            hallu={"chemical_id"},
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == self.OPSIN_SMILES
        assert inchi == self.OPSIN_INCHI
        assert src == "py2opsin"

    def test_both_tainted_returns_empty(self):
        """Both IUPAC and chem_id flagged → no reliable source."""
        row = self._row(
            opsin=self.OPSIN_SMILES, opsin_inchi=self.OPSIN_INCHI,
            cdx=self.CDX_SMILES_DIFF, cdx_inchi=self.CDX_INCHI_DIFF,
            hallu={"compound_IUPAC_name", "chemical_id"},
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == ""
        assert inchi == ""

    def test_only_opsin_available(self):
        """No CDX SMILES → OPSIN used."""
        row = self._row(
            opsin=self.OPSIN_SMILES, opsin_inchi=self.OPSIN_INCHI,
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == self.OPSIN_SMILES
        assert src == "py2opsin"

    def test_only_cdx_available(self):
        """No OPSIN SMILES → CDX used as fallback."""
        row = self._row(
            cdx=self.CDX_SMILES_DIFF, cdx_inchi=self.CDX_INCHI_DIFF,
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == self.CDX_SMILES_DIFF
        assert src == "cdx_fallback"

    def test_table_chemistry_tag_passthrough(self):
        """Non-OPSIN source (table_chemistry_tag) passes through unchanged."""
        row = self._row(
            opsin=self.CDX_SMILES_DIFF, opsin_inchi=self.CDX_INCHI_DIFF,
            cdx=self.CDX_SMILES_DIFF, cdx_inchi=self.CDX_INCHI_DIFF,
            source="table_chemistry_tag",
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == self.CDX_SMILES_DIFF
        assert src == "table_chemistry_tag"

    def test_table_chemistry_tag_prefers_cdx(self):
        """Non-OPSIN source prefers CDX SMILES over Ligand SMILES when both differ."""
        row = self._row(
            opsin=self.OPSIN_SMILES, opsin_inchi=self.OPSIN_INCHI,
            cdx=self.CDX_SMILES_DIFF, cdx_inchi=self.CDX_INCHI_DIFF,
            source="table_chemistry_tag",
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == self.CDX_SMILES_DIFF
        assert inchi == self.CDX_INCHI_DIFF
        assert src == "table_chemistry_tag"

    def test_table_chemistry_tag_fallback_to_ligand(self):
        """Non-OPSIN source falls back to Ligand SMILES when CDX is empty."""
        row = self._row(
            opsin=self.OPSIN_SMILES, opsin_inchi=self.OPSIN_INCHI,
            cdx="", cdx_inchi="",
            source="table_chemistry_tag",
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == self.OPSIN_SMILES
        assert inchi == self.OPSIN_INCHI
        assert src == "table_chemistry_tag"

    def test_table_chemistry_tag_cdx_tainted(self):
        """CDX source + chem_id flagged → empty (no OPSIN alternative)."""
        row = self._row(
            opsin=self.CDX_SMILES_DIFF, opsin_inchi=self.CDX_INCHI_DIFF,
            source="table_chemistry_tag",
            hallu={"chemical_id"},
        )
        smiles, inchi, src = select_ligand_smiles(row)
        assert smiles == ""
        assert inchi == ""


# ---------------------------------------------------------------------------
# patent_worker integration
# ---------------------------------------------------------------------------


class TestPatentWorkerHalluIntegration:
    """Verify that process_patent_worker respects hallu sidecar flags."""

    def _make_patent_dir(self, tmp_path, patent_id, bindings, hallu_flags=None):
        """Create a minimal patent directory with _resolved.json and optional _hallu.json."""
        patent_dir = tmp_path / patent_id
        patent_dir.mkdir()
        (patent_dir / f"{patent_id}_resolved.json").write_text(
            json.dumps(bindings), encoding="utf-8"
        )
        if hallu_flags is not None:
            (patent_dir / f"{patent_id}_hallu.json").write_text(
                json.dumps(hallu_flags), encoding="utf-8"
            )
        return str(patent_dir)

    def test_hallu_flag_masks_binding(self, tmp_path, monkeypatch):
        """A binding whose chemical_id is flagged gets its chem_id blanked."""
        bindings = [
            {
                "compound": "Compound-1",
                "chemical_id": "CHEM-US-00001",
                "compound_IUPAC_name": "",
                "agent2_resolution_source": "skipped_enriched_by_agent1",
                "binding_metric": "IC50",
                "value": "100",
                "unit": "nM",
                "protein_target_name": "TargetX",
                "organism": "human",
            },
            {
                "compound": "Compound-2",
                "chemical_id": "CHEM-US-00050",
                "compound_IUPAC_name": "",
                "agent2_resolution_source": "skipped_enriched_by_agent1",
                "binding_metric": "IC50",
                "value": "200",
                "unit": "nM",
                "protein_target_name": "TargetY",
                "organism": "human",
            },
        ]
        flags = {
            "detector_version": "1.0",
            "flagged_compounds": {"CHEM-US-00001": ["chem_id"]},
            "flagged_iupac": [],
            "flagged_values": [],
        }

        patent_dir = self._make_patent_dir(
            tmp_path, "US99999A1", bindings, hallu_flags=flags
        )

        from bindingdb_export.artifacts import (
            load_hallu_flags,
            load_resolved_bindings,
            mask_hallucinated_fields,
        )
        from bindingdb_export.rows import normalize_accepted_bindings
        from bindingdb_export.patent_worker import RESOLVED_PATTERNS

        loaded = load_resolved_bindings(patent_dir, RESOLVED_PATTERNS)
        assert len(loaded) == 2

        normalized = normalize_accepted_bindings(loaded)
        assert len(normalized) == 2

        hallu = load_hallu_flags(patent_dir)
        assert hallu is not None

        for b in normalized:
            mask_hallucinated_fields(b, hallu)

        # Both bindings survive — but first one has chemical_id blanked
        assert len(normalized) == 2
        assert normalized[0]["chemical_id"] == ""
        assert normalized[1]["chemical_id"] == "CHEM-US-00050"

    def test_no_hallu_file_keeps_all(self, tmp_path):
        """Without _hallu.json, all bindings survive."""
        bindings = [
            {
                "compound": "Compound-1",
                "chemical_id": "CHEM-US-00001",
                "compound_IUPAC_name": "",
                "agent2_resolution_source": "skipped_enriched_by_agent1",
                "binding_metric": "IC50",
                "value": "100",
                "unit": "nM",
                "protein_target_name": "TargetX",
                "organism": "human",
            },
        ]
        patent_dir = self._make_patent_dir(
            tmp_path, "US88888A1", bindings, hallu_flags=None
        )

        from bindingdb_export.artifacts import load_hallu_flags, load_resolved_bindings
        from bindingdb_export.patent_worker import RESOLVED_PATTERNS

        loaded = load_resolved_bindings(patent_dir, RESOLVED_PATTERNS)
        hallu = load_hallu_flags(patent_dir)
        assert hallu is None
        # No filtering — all bindings survive
        assert len(loaded) == 1
