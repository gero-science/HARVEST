"""Debug artifact writers for per-patent pipeline data."""

import asyncio
import logging
from pathlib import Path

from patent_processor import NodeExporter


async def save_debug_data(patent_id, document, output_dir, exporter_factory=NodeExporter):
    """
    Save optional debug chemistry artifacts for one patent.
    """
    try:
        output_dir_patent = Path(output_dir) / patent_id
        debug_tasks = []

        exporter = exporter_factory()

        if hasattr(document, "chemistry_nodes") and document.chemistry_nodes:
            debug_tasks.append(
                exporter.async_export_chemistry_as_single_file(
                    chemistry_nodes=document.chemistry_nodes,
                    output_dir=str(output_dir_patent / "chemistry"),
                    filename="debug_chemistry.json",
                )
            )
            logging.debug(f"Debug: Found {len(document.chemistry_nodes)} chemistry nodes for patent {patent_id}")
        else:
            logging.debug(f"Debug: No chemistry data found for patent {patent_id}")

        if debug_tasks:
            results = await asyncio.gather(*debug_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.warning(f"Debug save error for patent {patent_id}, task {i}: {result}")
                else:
                    logging.debug(f"Debug data saved for patent {patent_id}, task {i}")
        else:
            logging.debug(f"Debug: No debug data to save for patent {patent_id}")

    except Exception as e:
        logging.warning(f"Debug mode failed for patent {patent_id}: {e}")
