"""Artifact writing helpers for pipeline workers."""

import asyncio
import json
import time
from pathlib import Path

CDX_RESULTS_FILENAME = "cdx_results.json"


def build_cdx_results_payload(patent_id, cdx_data):
    """Build the cdx_results.json payload consumed by the export stage."""
    return {
        "patent_id": patent_id,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "compounds": cdx_data,
    }


async def save_patent_result_artifacts(
    patent_id,
    resolved,
    unresolved,
    output_dir,
    processed_patents_file,
    to_thread=asyncio.to_thread,
    cdx_data=None,
):
    """Save per-patent result JSON files and append the processed marker."""
    output_dir_patent = Path(output_dir) / patent_id
    await to_thread(output_dir_patent.mkdir, parents=True, exist_ok=True)

    if resolved:
        resolved_file = output_dir_patent / f"{patent_id}_resolved.json"
        await to_thread(
            lambda: resolved_file.write_text(
                json.dumps(resolved, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        )

    if unresolved:
        unresolved_file = output_dir_patent / f"{patent_id}_unresolved.json"
        await to_thread(
            lambda: unresolved_file.write_text(
                json.dumps(unresolved, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        )

    if cdx_data:
        cdx_file = output_dir_patent / CDX_RESULTS_FILENAME
        payload = build_cdx_results_payload(patent_id, cdx_data)
        await to_thread(
            lambda: cdx_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        )

    def append_to_file():
        with open(processed_patents_file, "a", encoding="utf-8") as f:
            f.write(f"{patent_id}\n")

    await to_thread(append_to_file)
