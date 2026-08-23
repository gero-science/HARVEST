"""Tag extracted rows with the provenance the BindingDB export filters on.

Rows the extractor already enriched with a structure are marked
``skipped_enriched_by_agent1``; that value is the allow-list key in
``bindingdb_export/rows.py``, so only those rows reach the final Parquet.
Anything without ``molecule_smiles`` is reported as unresolved.

The name is historical: this used to choose between the legacy ``alias_to_name``
resolver ("Agent 2") and an extractor-only bypass. The resolver has been removed
-- everything it resolved was discarded by the export filter anyway -- so the
bypass is now the only path.
"""

ACCEPTED_EXTRACTOR_SOURCE = "skipped_enriched_by_agent1"
UNRESOLVED_MISSING_SMILES_REASON = "missing_molecule_smiles"


def empty_agent2_usage():
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "request_count": 0,
        "token_per_request": [],
        "max_tokens_per_request": 0,
        "completion_tokens_per_request": [],
        "max_completion_tokens_per_request": 0,
        "pyopsin_filtered_count": 0,
    }


def bypass_agent2_alias_resolution(measures_list):
    resolved = []
    unresolved = []

    for item in measures_list:
        row = dict(item)
        if row.get("molecule_smiles"):
            row["agent2_resolution_source"] = ACCEPTED_EXTRACTOR_SOURCE
            if not row.get("original_alias"):
                row["original_alias"] = row.get("molecule_name") or row.get("compound")
            resolved.append(row)
            continue

        row["agent2_resolution_source"] = "unresolved"
        row["unresolved_reason"] = UNRESOLVED_MISSING_SMILES_REASON
        unresolved.append(row)

    return resolved, unresolved, 0.0, empty_agent2_usage()


async def resolve_aliases_for_patent(measures_list):
    """Kept async because every caller awaits it."""
    return bypass_agent2_alias_resolution(measures_list)
