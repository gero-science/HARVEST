"""Tests for the species provenance policy of the BindingDB export.

The species of a row is decided downstream, from the organism label that Stage 1
extracted from the patent text. Ambiguous human annotations are written as
``organism = "human (default)"`` instead of being silently labelled
``human``, and no protein agent has to be re-run for the label to appear.
"""

from types import SimpleNamespace

from bindingdb_export.artifacts import load_stage1_organisms
from bindingdb_export.enrichment import process_one_patent_bindings
from bindingdb_export.organisms import (
    HUMAN_DEFAULT_FLAG,
    HUMAN_DEFAULT_ORGANISM,
    HUMAN_FALLBACK_FLAG,
    INFERRED_FROM_TARGET,
    SPECIES_STATED,
    UNSTATED_SPECIES,
    align_human_default_scientific,
    classify_species_source,
    is_expression_host_label,
    is_nonspecific_organism,
    lookup_protein_data,
    normalize_organism,
    strip_human_protein_parts,
)
from protein_resolver.fasta_gene_resolver import FastaGeneResolver


class _FakeLigandExtractor:
    def get_inchi_key_by_smiles(self, smiles):
        return "FAKEINCHIKEY-A"

    def extract(self, name):
        return None, None

    def get_molecular_weight(self, smiles):
        return None


def _binding(**overrides):
    binding = {
        "molecule_name": "Ethanol",
        "molecule_smiles": "CCO",
        "molecule_inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "protein_target_name": "EGFR",
        "assay_id": "ASSAY_001",
        "value": 10,
        "unit": "nM",
    }
    binding.update(overrides)
    return binding


def _protein_entry(**overrides):
    entry = {
        "sequence": "MKTAY",
        "accession": "P00533",
        "uniprot_id": "EGFR_HUMAN",
        "gene": "EGFR",
        "organism_scientific": "Homo sapiens",
    }
    entry.update(overrides)
    return entry


class TestNonspecificOrganisms:
    def test_labels_without_a_species(self):
        for label in (None, "", "  ", "unknown", "Unspecified", "not specified", "n/a"):
            assert is_nonspecific_organism(label) is True

    def test_broad_taxon_labels_are_not_a_species(self):
        # These dominate the real corpus and used to resolve to Homo sapiens.
        for label in ("mammalian", "Mammal", "primate", "vertebrates", "eukaryote"):
            assert is_nonspecific_organism(label) is True

    def test_named_species_kept(self):
        for label in ("human", "Rat", "guinea pig", "Mus musculus", "HIV-1"):
            assert is_nonspecific_organism(label) is False


class TestNormalizeOrganism:
    def test_missing_is_not_rewritten_to_human(self):
        assert normalize_organism(None) == "unspecified"
        assert normalize_organism("") == "unspecified"
        assert normalize_organism("unknown") == "unspecified"
        assert normalize_organism("Unspecified species") == "unspecified"

    def test_explicit_labels_lowercased(self):
        assert normalize_organism(" Human ") == "human"
        assert normalize_organism("Mus musculus") == "mus musculus"
        assert normalize_organism("mammalian") == "mammalian"


class TestClassifySpeciesSource:
    def test_stated_species(self):
        assert classify_species_source("rat", "Rattus norvegicus", "EGFR_RAT") == SPECIES_STATED
        assert classify_species_source("human", "Homo sapiens", "EGFR_HUMAN") == SPECIES_STATED

    def test_assumed_human_is_flagged(self):
        assert classify_species_source(None, "Homo sapiens", "EGFR_HUMAN") == HUMAN_DEFAULT_FLAG
        assert classify_species_source("unspecified", "Homo sapiens", None) == HUMAN_DEFAULT_FLAG
        assert classify_species_source("mammalian", "Homo sapiens", "CCR3_HUMAN") == HUMAN_DEFAULT_FLAG

    def test_human_sequence_without_human_label_still_assumed_human(self):
        assert classify_species_source("unspecified", None, "EGFR_HUMAN") == HUMAN_DEFAULT_FLAG

    def test_non_human_species_from_target_is_not_a_human_default(self):
        # Viral / bacterial targets carry their organism in the target name.
        assert (
            classify_species_source("unspecified", "Hepatitis C virus", "POLG_HCVJ")
            == INFERRED_FROM_TARGET
        )
        assert (
            classify_species_source("unspecified", "Escherichia coli", None)
            == INFERRED_FROM_TARGET
        )

    def test_virus_named_after_its_host_is_not_human(self):
        for species in ("Human cytomegalovirus", "Human immunodeficiency virus 1"):
            assert classify_species_source("unspecified", species, None) == INFERRED_FROM_TARGET

    def test_no_species_anywhere(self):
        assert classify_species_source("unspecified", None, None) == UNSTATED_SPECIES
        assert classify_species_source(None, "", "") == UNSTATED_SPECIES


class TestAlignHumanDefaultScientific:
    def test_expression_host_species_becomes_human(self):
        assert (
            align_human_default_scientific("Cricetulus griseus", "IGF1R_HUMAN")
            == "Homo sapiens"
        )
        assert (
            align_human_default_scientific(
                "Mus musculus × Rattus norvegicus hybrid", "KCNQ2_HUMAN;KCNQ3_HUMAN"
            )
            == "Homo sapiens"
        )

    def test_viral_scientific_name_is_kept(self):
        assert (
            align_human_default_scientific(
                "Human immunodeficiency virus 1", "ATTY_HUMAN"
            )
            == "Human immunodeficiency virus 1"
        )

    def test_already_human_is_unchanged(self):
        assert align_human_default_scientific("Homo sapiens", "EGFR_HUMAN") == "Homo sapiens"

    def test_no_human_uniprot_is_unchanged(self):
        assert align_human_default_scientific("Cricetulus griseus", None) == "Cricetulus griseus"

    def test_human_sequence_for_nonhuman_target_is_flagged(self):
        assert (
            classify_species_source("rabbit", "Oryctolagus cuniculus", "ACE_HUMAN")
            == HUMAN_FALLBACK_FLAG
        )

    def test_complex_with_any_human_component_flagged(self):
        assert (
            classify_species_source("bovine", "Bos taurus", "PRKACA_HUMAN;PRKAR1A_BOVIN")
            == HUMAN_FALLBACK_FLAG
        )

    def test_cell_line_label_counts_as_human(self):
        assert (
            classify_species_source("hek-293 cells (human)", "Homo sapiens", "EGFR_HUMAN")
            == SPECIES_STATED
        )

    def test_expression_host_label_is_an_assumption_not_a_mismatch(self):
        # CHO names where the target was expressed, not which species it is.
        assert (
            classify_species_source("chinese hamster ovary", "Cricetulus griseus", "IGF1R_HUMAN")
            == HUMAN_DEFAULT_FLAG
        )
        assert (
            classify_species_source("mouse/rat (SH-SY5Y hybrid)", "Mus musculus", "KCNQ2_HUMAN")
            == HUMAN_DEFAULT_FLAG
        )


class TestExpressionHostLabels:
    def test_host_only_labels(self):
        for label in ("CHO", "cho-k1", "CHO cells", "chinese hamster ovary (CHO)",
                      "HEK-293", "hek 293 cells", "COS-7", "SH-SY5Y", "mouse/rat (sh-sy5y hybrid)"):
            assert is_expression_host_label(label) is True

    def test_labels_that_still_name_a_species(self):
        # cyno-CHO is a cynomolgus receptor in CHO cells, dog kidney is MDCK.
        for label in ("cyno-CHO", "chinese hamster", "dog kidney", "rat", "Trichoderma reesei", ""):
            assert is_expression_host_label(label) is False


class TestStripHumanProteinParts:
    def test_complex_loses_only_its_human_subunit(self):
        gene, uniprot_id, accession, sequence = strip_human_protein_parts(
            "Gabra1;Gabrq;Gabrb2",
            "GBRA1_RAT;GBRT_HUMAN;GBRB2_RAT",
            "P62813;Q9UN88;P63138",
            "AAA;HHH;BBB",
        )

        assert gene == "Gabra1;Gabrb2"
        assert uniprot_id == "GBRA1_RAT;GBRB2_RAT"
        assert accession == "P62813;P63138"
        assert sequence == "AAA;BBB"

    def test_single_target_keeps_gene_and_loses_sequence(self):
        assert strip_human_protein_parts("TRPM8", "TRPM8_HUMAN", "Q7Z2W7", "MKT") == (
            "TRPM8",
            "",
            "",
            "",
        )

    def test_nothing_to_strip(self):
        unchanged = ("EGFR", "EGFR_RAT", "P00533", "MKTAY")
        assert strip_human_protein_parts(*unchanged) == unchanged

    def test_misaligned_lists_drop_the_whole_annotation(self):
        # Without parallel lists there is no safe way to pick the human parts.
        assert strip_human_protein_parts("A;B", "A_HUMAN;B_RAT", "P1", "AAA") == ("A;B", "", "", "")


class TestLookupProteinData:
    def test_named_species_never_matches_another_species(self):
        entries = {"human": _protein_entry()}
        assert lookup_protein_data(entries, "rat") is None

    def test_nonspecific_label_reuses_legacy_keys(self):
        # Legacy artifacts stored targets without a stated species under "human".
        entries = {"human": _protein_entry()}
        assert lookup_protein_data(entries, "unspecified") is entries["human"]
        assert lookup_protein_data(entries, "mammalian") is entries["human"]
        assert lookup_protein_data(entries, None) is entries["human"]

    def test_exact_key_wins_over_legacy_keys(self):
        entries = {"human": _protein_entry(), "mammalian": _protein_entry(gene="CCR3")}
        assert lookup_protein_data(entries, "mammalian")["gene"] == "CCR3"

    def test_empty_map(self):
        assert lookup_protein_data({}, "human") is None


class TestStage1OrganismLoading:
    def _write_stage1(self, tmp_path, rows):
        header = (
            "assay_id\tprotein_target_name\tis_complex\tprotein_modification\t"
            "assay_description\treasoning\tassay\torganism\textreme_conditions"
        )
        lines = [header] + ["\t".join(row) for row in rows]
        (tmp_path / "US1_agent1_stage1_targets.tsv").write_text("\n".join(lines) + "\n")

    def test_reads_organism_by_assay_id(self, tmp_path):
        self._write_stage1(
            tmp_path,
            [
                ["ASSAY_001", "EGFR", "N", "", "desc", "p-1", "Binding", "unspecified", "N"],
                ["ASSAY_002", "MAO-A", "N", "", "desc", "p-2", "Binding", "Rat", "N"],
            ],
        )
        assert load_stage1_organisms(str(tmp_path)) == {
            "ASSAY_001": "unspecified",
            "ASSAY_002": "Rat",
        }

    def test_misaligned_rows_are_skipped(self, tmp_path):
        self._write_stage1(tmp_path, [["ASSAY_001", "EGFR", "N"]])
        assert load_stage1_organisms(str(tmp_path)) == {}

    def test_missing_file_and_directory(self, tmp_path):
        assert load_stage1_organisms(str(tmp_path)) == {}
        assert load_stage1_organisms(str(tmp_path / "absent")) == {}


class TestEnrichmentSpeciesProvenance:
    def test_stage1_label_overrides_legacy_row_organism(self):
        # Legacy runs wrote "human" into the row; Stage 1 said nothing.
        rows = process_one_patent_bindings(
            [_binding(organism="human")],
            _FakeLigandExtractor(),
            {"EGFR": {"human": _protein_entry()}},
            stage1_organisms={"ASSAY_001": "unspecified"},
        )

        assert len(rows) == 1
        row = rows[0]
        assert row["organism"] == HUMAN_DEFAULT_ORGANISM
        assert "species_source" not in row
        # The sequence is kept so the row stays usable and auditable.
        assert row["Sequence"] == "MKTAY"
        assert row["organism_scientific"] == "Homo sapiens"

    def test_row_label_used_without_stage1_tsv(self):
        rows = process_one_patent_bindings(
            [_binding(organism="rat")],
            _FakeLigandExtractor(),
            {"EGFR": {"rat": _protein_entry(uniprot_id="EGFR_RAT", organism_scientific="Rattus norvegicus")}},
        )

        assert rows[0]["organism"] == "rat"
        assert "species_source" not in rows[0]

    def test_stated_species_never_keeps_a_human_sequence(self):
        rows = process_one_patent_bindings(
            [_binding(organism="rabbit")],
            _FakeLigandExtractor(),
            {"EGFR": {"rabbit": _protein_entry(organism_scientific="Oryctolagus cuniculus")}},
        )

        # No rabbit ortholog is attached, so the human one is dropped instead of
        # being passed off as the stated species.
        assert rows[0]["organism"] == "rabbit"
        assert "species_source" not in rows[0]
        assert rows[0]["gene"] == "EGFR"
        assert rows[0]["protein_sequence"] == ""
        assert rows[0]["protein_uniprot_id"] == ""
        assert "Sequence" not in rows[0]

    def test_complex_keeps_its_non_human_subunits(self):
        entry = _protein_entry(
            gene="Gabra1;Gabrq",
            uniprot_id="GBRA1_RAT;GBRT_HUMAN",
            accession="P62813;Q9UN88",
            sequence="AAA;HHH",
            organism_scientific="Rattus norvegicus",
        )
        rows = process_one_patent_bindings(
            [_binding(protein_target_name="GABA-A receptor", organism="rat")],
            _FakeLigandExtractor(),
            {"GABA-A receptor": {"rat": entry}},
        )

        assert rows[0]["organism"] == "rat"
        assert "species_source" not in rows[0]
        assert rows[0]["gene"] == "Gabra1"
        assert rows[0]["Sequence"] == "AAA"
        assert rows[0]["UniProt ID"] == "GBRA1_RAT"

    def test_expression_host_label_keeps_the_human_sequence(self):
        rows = process_one_patent_bindings(
            [_binding(protein_target_name="IGF-1R", organism="chinese hamster ovary")],
            _FakeLigandExtractor(),
            {
                "IGF-1R": {
                    "chinese hamster ovary": _protein_entry(
                        gene="Igf1r", uniprot_id="IGF1R_HUMAN", organism_scientific="Cricetulus griseus"
                    )
                }
            },
        )

        assert rows[0]["organism"] == HUMAN_DEFAULT_ORGANISM
        assert "species_source" not in rows[0]
        assert rows[0]["Sequence"] == "MKTAY"
        assert rows[0]["organism_scientific"] == "Homo sapiens"

    def test_assumed_human_keeps_a_viral_scientific_name(self):
        rows = process_one_patent_bindings(
            [_binding(protein_target_name="Tat", organism="unspecified")],
            _FakeLigandExtractor(),
            {
                "Tat": {
                    "unspecified": _protein_entry(
                        gene="tat",
                        uniprot_id="ATTY_HUMAN",
                        organism_scientific="Human immunodeficiency virus 1",
                    )
                }
            },
        )

        assert rows[0]["organism"] == HUMAN_DEFAULT_ORGANISM
        assert rows[0]["organism_scientific"] == "Human immunodeficiency virus 1"
        assert rows[0]["UniProt ID"] == "ATTY_HUMAN"

    def test_row_without_protein_match_is_still_classified(self):
        rows = process_one_patent_bindings(
            [_binding(organism="mammalian")],
            _FakeLigandExtractor(),
            {},
        )

        assert rows[0]["organism"] == "mammalian"
        assert "species_source" not in rows[0]
        assert rows[0]["protein_sequence"] is None

    def test_viral_target_without_stated_organism_is_not_humanized(self):
        rows = process_one_patent_bindings(
            [_binding(protein_target_name="NS5B", organism="unspecified")],
            _FakeLigandExtractor(),
            {
                "NS5B": {
                    "unspecified": _protein_entry(
                        uniprot_id="POLG_HCVJ", organism_scientific="Hepatitis C virus"
                    )
                }
            },
        )

        assert rows[0]["organism"] == "unspecified"
        assert "species_source" not in rows[0]
        assert rows[0]["organism_scientific"] == "Hepatitis C virus"


class TestFastaGeneResolverNoHumanFallback:
    """The resolver must not invent a human ortholog for another species."""

    def test_nonspecific_not_mapped_to_human(self):
        # Bypass singleton constructor by calling unbound method
        resolver = object.__new__(FastaGeneResolver)
        assert resolver._normalize_species("mammalian") == "mammalian"
        assert resolver._normalize_species("unknown") == "unknown"
        assert resolver._normalize_species("human") == "homo sapiens"
        assert resolver._normalize_species("hek-293 cells (human)") == "homo sapiens"

    def test_resolve_does_not_fallback_to_human(self):
        resolver = object.__new__(FastaGeneResolver)
        resolver.logger = SimpleNamespace(debug=lambda *a, **k: None)
        human_rec = SimpleNamespace(id="sp|P00533|EGFR_HUMAN", seq="MKTAY")
        resolver._index = {("egfr", "homo sapiens"): human_rec}

        assert resolver.resolve("EGFR", "Homo sapiens") is human_rec
        assert resolver.resolve("EGFR", "Rattus norvegicus") is None
        assert resolver.resolve("EGFR", "mammalian") is None
        assert resolver.resolve("EGFR", "unknown") is None
