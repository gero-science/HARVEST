import asyncio
import logging
from types import SimpleNamespace

from pipeline.accounting import empty_agent1_usage
from pipeline.agent1_processing import (
    build_agent1_error_detail,
    build_agent1_error_payload,
    enrich_agent1_data_from_document,
    process_agent1_task,
)


def chem_node(chem_num, smiles="CCO", inchikey=None, is_scaffold=False):
    return SimpleNamespace(
        chem_num=chem_num,
        smiles=smiles,
        inchikey=inchikey,
        is_scaffold=is_scaffold,
    )


def task(**overrides):
    base = {
        "patent_id": "USUNITTESTA1",
        "chunk": "<xml />",
        "total_chunks_in_patent": 1,
        "patent_text": "patent text",
        "document": SimpleNamespace(chemistry_nodes=[]),
        "zip_file_path": "/tmp/USUNITTESTA1.zip",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeSimpleAgent:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def async_process_document(self, chunk, output_dir=None):
        self.calls.append((chunk, output_dir))
        return self.result


def usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.25):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
    }


def test_enrich_agent1_data_from_document_exact_match_adds_smiles_and_inchikey():
    rows = [{"chemical_id": "12", "molecule_name": "Example 12"}]
    document = SimpleNamespace(chemistry_nodes=[chem_node("12", smiles="CCO", inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N")])

    count = enrich_agent1_data_from_document(rows, document)

    assert count == 1
    assert rows[0]["molecule_smiles"] == "CCO"
    assert rows[0]["molecule_inchikey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert rows[0]["smiles_source"] == "table_chemistry_tag"


def test_enrich_agent1_data_from_document_normalized_numeric_id_matches_current_behavior():
    rows = [{"chemical_id": "00012"}]
    document = SimpleNamespace(chemistry_nodes=[chem_node("12", smiles="CCN")])

    count = enrich_agent1_data_from_document(rows, document)

    assert count == 1
    assert rows[0]["molecule_smiles"] == "CCN"
    assert "molecule_inchikey" not in rows[0]
    assert rows[0]["smiles_source"] == "table_chemistry_tag"


def test_enrich_agent1_data_from_document_skips_scaffold_nodes_and_missing_document():
    rows = [{"chemical_id": "12"}]
    document = SimpleNamespace(chemistry_nodes=[chem_node("12", smiles="CCO", is_scaffold=True)])

    assert enrich_agent1_data_from_document(rows, None) == 0
    assert enrich_agent1_data_from_document(rows, document) == 0
    assert "molecule_smiles" not in rows[0]


def test_process_agent1_task_preserves_queue_tuple_and_usage_contract(tmp_path):
    final_data = [{"chemical_id": "00012", "compound": "Example 12"}]
    document = SimpleNamespace(chemistry_nodes=[chem_node("12", smiles="CCO")])
    result = {
        "final_data": final_data,
        "debug_artifacts": [
            {"usage_stats": usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.25)},
            {"usage_stats_list": [
                usage(prompt_tokens=20, completion_tokens=7, total_tokens=27, cost=0.5)
            ]},
        ],
    }
    fake_agent = FakeSimpleAgent(result)
    agent_task = task(document=document, total_chunks_in_patent=3)

    payload = asyncio.run(process_agent1_task(fake_agent, agent_task, output_dir=tmp_path))

    assert fake_agent.calls == [("<xml />", tmp_path)]
    assert payload[0] == "USUNITTESTA1"
    assert payload[1] is final_data
    assert payload[1][0]["molecule_smiles"] == "CCO"
    assert payload[2] == 3
    assert payload[3] == "patent text"
    assert payload[4]["prompt_tokens"] == 30
    assert payload[4]["completion_tokens"] == 12
    assert payload[4]["total_tokens"] == 42
    assert payload[4]["request_count"] == 2
    assert payload[5] is document
    assert payload[6] == {}
    assert payload[7] is None
    assert payload[8] == "/tmp/USUNITTESTA1.zip"


def test_build_agent1_error_detail_uses_current_classification_and_no_traceback_without_debug():
    previous_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)
    try:
        error_type, detail = build_agent1_error_detail(task(), ValueError("json parse failed"))
    finally:
        logging.getLogger().setLevel(previous_level)

    assert error_type == "parse_errors"
    assert detail["agent"] == "agent1"
    assert detail["patent_id"] == "USUNITTESTA1"
    assert detail["error_type"] == "parse_errors"
    assert detail["error_class"] == "ValueError"
    assert detail["error_message"] == "json parse failed"
    assert detail["traceback"] is None


def test_build_agent1_error_payload_preserves_metadata_and_sends_per_payload_increment():
    error_detail = {"error_type": "parse_errors"}
    document = SimpleNamespace(chemistry_nodes=[])
    agent_task = task(document=document, total_chunks_in_patent=2)

    payload = build_agent1_error_payload(agent_task, "parse_errors", error_detail)

    assert payload == (
        "USUNITTESTA1",
        [],
        2,
        "patent text",
        empty_agent1_usage(),
        document,
        {"parse_errors": 1},
        error_detail,
        "/tmp/USUNITTESTA1.zip",
    )
    assert payload[6] == {"parse_errors": 1}
