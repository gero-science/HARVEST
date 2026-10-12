"""Tests for the protein postprocessor: Entry name fallback and request binding.

The export joins single_proteins.json on the requested protein name and organism,
so every result has to come back under the request it answers, whatever form
the LLM echoed it in.
"""

import asyncio
import json
import logging

import pytest

from bindingdb_export.organisms import lookup_protein_data
from bindingdb_export.proteins import load_single_proteins
from protein_postprocessor.config_postprocess import ConfigProteinLLM
from protein_postprocessor.protein_postprocessor import ProteinPostprocessor, _name_aliases
from protein_resolver.fasta_gene_resolver import FastaGeneResolver

FASTA = """\
>sp|P54760|EPHB4_HUMAN Ephrin type-B receptor 4 OS=Homo sapiens OX=9606 GN=EPHB4 PE=1 SV=2
MELRVLLCWASLAAAL
>sp|P27487|DPP4_HUMAN Dipeptidyl peptidase 4 OS=Homo sapiens OX=9606 GN=DPP4 PE=1 SV=2
MKTPWKVLLGLLGAAA
>sp|P00533|EGFR_HUMAN Epidermal growth factor receptor OS=Homo sapiens OX=9606 GN=EGFR PE=1 SV=2
MRPSGTAGAALLALLA
>sp|Q01279|EGFR_MOUSE Epidermal growth factor receptor OS=Mus musculus OX=10090 GN=Egfr PE=1 SV=1
MRPSGTARTTLLVLLT
>sp|P0DTD1|R1AB_SARS2 Replicase polyprotein 1ab OS=Severe acute respiratory syndrome coronavirus 2 OX=2697049 GN=rep PE=1 SV=1
SGFRKMAFPSGKVEGC
"""


@pytest.fixture
def resolver(tmp_path):
    fasta = tmp_path / "uniprot_sprot.fasta"
    fasta.write_text(FASTA)
    # Bypass the singleton constructor
    res = object.__new__(FastaGeneResolver)
    res.logger = logging.getLogger("test_fasta_gene_resolver")
    res._index = {}
    res._records_by_accession = {}
    res._records_by_entry_name = {}
    res._load_fasta(fasta)
    return res


class _ScriptedLLM:
    """Returns the given responses in order and records the messages it was sent."""

    def __init__(self, *responses):
        self.responses = [json.dumps(r) if not isinstance(r, str) else r for r in responses]
        self.calls = []

    async def async_call_llm(self, user_prompt=None, system_prompt=None, messages=None, **kwargs):
        self.calls.append(messages)
        return (self.responses.pop(0) if self.responses else None), None


def _postprocessor(resolver, llm):
    pp = object.__new__(ProteinPostprocessor)
    pp.config = ConfigProteinLLM()
    pp.llm = llm
    pp.resolver = resolver
    pp.zip_test_dir = None
    return pp


def _single(name, organism, gene=None, species=None, sequence=None):
    return {
        "protein_name": name,
        "organism": organism,
        "organism_scientific": species,
        "gene": gene,
        "sequence": sequence,
        "accession": None,
        "uniprot_id": None,
    }


class TestPrompts:
    def test_unstated_organism_rule_is_present(self):
        assert "assume Homo sapiens" in ConfigProteinLLM.SYSTEM_PROMPT
        assert "assume Homo sapiens" in ConfigProteinLLM.SYSTEM_PROMPT_COMPLEX

    def test_entry_name_prompts_format(self):
        for template in (ConfigProteinLLM.ENTRY_NAME_PROMPT, ConfigProteinLLM.ENTRY_NAME_PROMPT_COMPLEX):
            text = template.format(unresolved_list="- nsp5 (species: SARS-CoV-2)")
            assert "- nsp5 (species: SARS-CoV-2)" in text
            assert '{"gene": "...", "species": "...", "entry_name": "..."}' in text


class TestResolveByEntryName:
    def test_case_insensitive_lookup(self, resolver):
        record = resolver.resolve_by_entry_name("r1ab_sars2", "SARS-CoV-2")
        assert record.id == "sp|P0DTD1|R1AB_SARS2"

    def test_unknown_entry_name(self, resolver):
        assert resolver.resolve_by_entry_name("NOPE_HUMAN", "Homo sapiens") is None
        assert resolver.resolve_by_entry_name("", "Homo sapiens") is None

    def test_model_species_entry_must_match_requested_species(self, resolver):
        assert resolver.resolve_by_entry_name("EGFR_MOUSE", "Homo sapiens") is None
        assert resolver.resolve_by_entry_name("EGFR_HUMAN", "Rattus norvegicus") is None
        assert resolver.resolve_by_entry_name("EGFR_HUMAN", "unspecified") is None
        assert resolver.resolve_by_entry_name("EGFR_MOUSE", "Mus musculus").id == "sp|Q01279|EGFR_MOUSE"
        assert resolver.resolve_by_entry_name("EGFR_HUMAN", "human").id == "sp|P00533|EGFR_HUMAN"

    def test_entry_accepted_without_species(self, resolver):
        assert resolver.resolve_by_entry_name("EGFR_MOUSE").id == "sp|Q01279|EGFR_MOUSE"

    def test_stats_count_entry_names(self, resolver):
        assert resolver.get_stats()["indexed_entry_names"] == 5


class TestEntryNameFallback:
    def test_single_protein_resolved_by_entry_name(self, resolver):
        llm = _ScriptedLLM(
            [{"protein": "SARS2 protease", "organism": "unspecified", "gene": "nsp5", "species": "SARS-CoV-2"}],
            [{"gene": "nsp5", "species": "SARS-CoV-2", "entry_name": "R1AB_SARS2"}],
        )
        pp = _postprocessor(resolver, llm)

        result = asyncio.run(pp.postprocess([{"protein_name": "SARS2 protease", "organism": "unspecified"}]))

        assert [r["uniprot_id"] for r in result] == ["R1AB_SARS2"]
        assert result[0]["accession"] == "P0DTD1"
        followup = llm.calls[1]
        assert [m["role"] for m in followup] == ["system", "user", "assistant", "user"]
        assert "- SARS2 protease (gene: nsp5, species: SARS-CoV-2)" in followup[-1]["content"]

    def test_entry_name_matching_ignores_case_and_spacing(self, resolver):
        llm = _ScriptedLLM(
            [{"protein": "SARS2 protease", "organism": "unspecified", "gene": "nsp5", "species": "SARS-CoV-2"}],
            [{"gene": "NSP5", "species": " sars-cov-2 ", "entry_name": "R1AB_SARS2"}],
        )
        result = asyncio.run(
            _postprocessor(resolver, llm).postprocess([{"protein_name": "SARS2 protease", "organism": "unspecified"}])
        )
        assert result[0]["uniprot_id"] == "R1AB_SARS2"

    def test_no_followup_when_everything_resolves(self, resolver):
        llm = _ScriptedLLM(
            [{"protein": "EphB4", "organism": "unspecified", "gene": "EPHB4", "species": "Homo sapiens"}],
        )
        result = asyncio.run(
            _postprocessor(resolver, llm).postprocess([{"protein_name": "EphB4", "organism": "unspecified"}])
        )
        assert result[0]["uniprot_id"] == "EPHB4_HUMAN"
        assert len(llm.calls) == 1

    def test_empty_followup_keeps_first_step_results(self, resolver):
        llm = _ScriptedLLM(
            [{"protein": "SARS2 protease", "organism": "unspecified", "gene": "nsp5", "species": "SARS-CoV-2"}],
        )
        result = asyncio.run(
            _postprocessor(resolver, llm).postprocess([{"protein_name": "SARS2 protease", "organism": "unspecified"}])
        )
        assert result[0]["gene"] == "nsp5"
        assert result[0]["sequence"] is None

    def test_empty_first_response_raises(self, resolver):
        pp = _postprocessor(resolver, _ScriptedLLM())
        with pytest.raises(RuntimeError):
            asyncio.run(pp.postprocess([{"protein_name": "EphB4", "organism": "unspecified"}]))
        with pytest.raises(RuntimeError):
            asyncio.run(pp.postprocess_complex([{"complex_name": "X", "organism": "human"}]))

    def test_complex_gene_resolved_by_entry_name(self, resolver):
        llm = _ScriptedLLM(
            [{
                "complex_name": "SARS2 replicase",
                "organism": "unspecified",
                "species": "SARS-CoV-2",
                "genes": ["nsp5", "nsp12"],
            }],
            [
                {"gene": "nsp5", "species": "SARS-CoV-2", "entry_name": "R1AB_SARS2"},
                {"gene": "nsp12", "species": "SARS-CoV-2", "entry_name": None},
            ],
        )
        result = asyncio.run(
            _postprocessor(resolver, llm).postprocess_complex(
                [{"complex_name": "SARS2 replicase", "organism": "unspecified"}]
            )
        )
        proteins = result[0]["proteins"]
        assert [p["uniprot_id"] for p in proteins] == ["R1AB_SARS2", None]
        assert "- nsp12 (species: SARS-CoV-2)" in llm.calls[1][-1]["content"]


class TestBindToRequests:
    def _bind(self, resolver, results, requests):
        return _postprocessor(resolver, None)._bind_to_requests(results, requests, "protein_name")

    def test_aliases(self):
        assert _name_aliases("neutral sphingomyelinase 2 (nSMase2)") == {
            "neutral sphingomyelinase 2 (nsmase2)",
            "neutral sphingomyelinase 2",
            "nsmase2",
        }
        assert _name_aliases("EphB4") == {"ephb4"}

    def test_reworded_organism_takes_request_organism(self, resolver):
        results = [_single("EphB4", "human (assumed)", "EPHB4", "Homo sapiens", "MELR")]
        bound = self._bind(resolver, results, [{"protein_name": "EphB4", "organism": "unspecified"}])
        assert [(r["protein_name"], r["organism"], r["sequence"]) for r in bound] == [
            ("EphB4", "unspecified", "MELR")
        ]

    def test_short_alias_binds_to_full_request_name(self, resolver):
        results = [_single("nSMase2", "Human", "SMPD3", "Homo sapiens", "MVLY")]
        requests = [{"protein_name": "neutral sphingomyelinase 2 (nSMase2)", "organism": "Human"}]
        bound = self._bind(resolver, results, requests)
        assert [r["protein_name"] for r in bound] == ["neutral sphingomyelinase 2 (nSMase2)"]

    def test_split_name_keeps_best_resolved_result(self, resolver):
        results = [
            _single("dipeptidylpeptidase-IV", "Human", None, "Homo sapiens"),
            _single("DPP-IV", "Human", "DPP4", "Homo sapiens", "MKTP"),
        ]
        requests = [{"protein_name": "dipeptidylpeptidase-IV (DPP-IV)", "organism": "Human"}]
        bound = self._bind(resolver, results, requests)
        assert [(r["protein_name"], r["gene"]) for r in bound] == [("dipeptidylpeptidase-IV (DPP-IV)", "DPP4")]

    def test_backslashes_dropped_from_request_name(self, resolver):
        results = [_single("PI3Kα", "human", "PIK3CA", "Homo sapiens")]
        bound = self._bind(resolver, results, [{"protein_name": "PI3K\\α", "organism": "human"}])
        assert bound[0]["protein_name"] == "PI3Kα"

    def test_same_name_bound_by_organism(self, resolver):
        results = [
            _single("EGFR", "rat", "Egfr", "Rattus norvegicus"),
            _single("EGFR", "human", "EGFR", "Homo sapiens"),
        ]
        requests = [
            {"protein_name": "EGFR", "organism": "Human"},
            {"protein_name": "EGFR", "organism": "Rat"},
        ]
        bound = self._bind(resolver, results, requests)
        assert [(r["organism"], r["gene"]) for r in bound] == [("Human", "EGFR"), ("Rat", "Egfr")]

    def test_alias_shared_by_several_requests_stays_unbound(self, resolver):
        results = [
            _single("EGFR (L858R)", "human", "EGFR", "Homo sapiens"),
            _single("EGFR", "human", "EGFR", "Homo sapiens"),
        ]
        requests = [
            {"protein_name": "EGFR (L858R)", "organism": "human"},
            {"protein_name": "EGFR (T790M)", "organism": "human"},
        ]
        bound = self._bind(resolver, results, requests)
        assert [r["protein_name"] for r in bound] == ["EGFR (L858R)", "EGFR"]

    def test_unmatched_result_kept_unless_it_shadows_a_bound_one(self, resolver):
        results = [
            _single("EphB4", "unspecified", "EPHB4", "Homo sapiens", "MELR"),
            _single("Mystery kinase", "human"),
        ]
        bound = self._bind(resolver, results, [{"protein_name": "EphB4", "organism": "unspecified"}])
        assert [r["protein_name"] for r in bound] == ["EphB4", "Mystery kinase"]


@pytest.mark.regression
class TestExportJoinAfterBinding:
    """Kostya's benchmark: rows lost because the LLM reworded the requested key."""

    def test_reworded_answers_reach_the_export(self, resolver, tmp_path):
        llm = _ScriptedLLM([
            {"protein": "EphB4", "organism": "human (assumed)", "gene": "EPHB4", "species": "Homo sapiens"},
            {"protein": "dipeptidylpeptidase-IV", "organism": "human", "gene": None, "species": "Homo sapiens"},
            {"protein": "DPP-IV", "organism": "human", "gene": "DPP4", "species": "Homo sapiens"},
        ])
        requests = [
            {"protein_name": "EphB4", "organism": "unspecified"},
            {"protein_name": "dipeptidylpeptidase-IV (DPP-IV)", "organism": "Human"},
        ]
        proteins = asyncio.run(_postprocessor(resolver, llm).postprocess(requests))
        (tmp_path / "single_proteins.json").write_text(json.dumps({"patent_id": "US1", "proteins": proteins}))

        resolved = load_single_proteins(str(tmp_path))

        assert lookup_protein_data(resolved["EphB4"], "unspecified")["uniprot_id"] == "EPHB4_HUMAN"
        assert lookup_protein_data(resolved["dipeptidylpeptidase-IV (DPP-IV)"], "Human")["uniprot_id"] == "DPP4_HUMAN"
