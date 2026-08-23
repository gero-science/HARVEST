"""Producer module for generating chunks from patent documents."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigPipeline
from patent_processor import PatentProcessor as XMLProcessor, PatentDocument


@dataclass
class ChunkProcessingTask:
    """Task for processing one patent chunk."""
    patent_id: str
    patent_text: str
    chunk: Any
    total_chunks_in_patent: int
    document: PatentDocument  # Parsed patent, carried through to Agent 2
    zip_file_path: str = ""  # Path to the ZIP file from which the patent was extracted


async def producer_chunks(queue_agent1, input_paths, limit, already_processed):
    """
    Function 1: Asynchronous chunk producer.
    Reads files and enqueues chunks for processing.
    
    Args:
        queue_agent1: Async queue for passing tasks to Agent 1
        input_paths: List of paths to patent ZIP archives
        limit: Cap on the number of documents to process (None = no limit)
        already_processed: Set of already processed patent IDs
    """
    total_documents_processed = 0
    
    skipped_count = 0
    
    for file_index, input_path in enumerate(input_paths, 1):
        # Extract patent_id from the filename before parsing the ZIP
        patent_id_from_filename = Path(input_path).stem
        
        # Skip already processed patents without parsing the ZIP
        if patent_id_from_filename in already_processed:
            skipped_count += 1
            if skipped_count <= 10 or skipped_count % 100 == 0:
                logging.debug(f"Skipping already processed patent: {patent_id_from_filename} ({skipped_count} skipped)")
            continue
        
        logging.info(f"Processing file {file_index}/{len(input_paths)}: {input_path}")
        
        try:
            xml_processor = XMLProcessor(
                input_path,  # Path to the ZIP file
                batch_size=100,
            )
            
            # Apply the limit across all input files
            remaining_limit = None
            if limit is not None:
                remaining_limit = limit - total_documents_processed
                if remaining_limit <= 0:
                    logging.info(f"Limit of {limit} reached. Stopping task collection.")
                    break
            
            documents_collected = 0
            # KEY CHANGE: use the async document iterator
            async for doc in xml_processor.async_iter_documents():
                # Check the limit
                if remaining_limit is not None and documents_collected >= remaining_limit:
                    break
                    
                # Skip already processed patents
                if doc.patent_id in already_processed:
                    logging.debug(f"Skipping already processed patent: {doc.patent_id}")
                    continue
                
                # Treat the whole patent as one chunk for the extractor
                # Require XML before processing
                if doc.xml_root is None:
                    logging.debug(f"[{doc.patent_id}] No XML root. Skipping.")
                    continue
                
                logging.debug(f"[{doc.patent_id}] Entire patent will be processed as one chunk")
                
                # Save the full patent text (can also be done asynchronously)
                patent_text = ""
                if doc.xml_root is not None:
                    patent_text = await asyncio.to_thread(lambda: " ".join(doc.xml_root.itertext()))
                
                # Create a single task for the whole patent
                task = ChunkProcessingTask(
                    patent_id=doc.patent_id,
                    patent_text=patent_text,
                    chunk=doc,  # Pass the full PatentDocument as the "chunk"
                    total_chunks_in_patent=1,  # Always one chunk
                    document=doc,
                    zip_file_path=input_path  # Preserve the source ZIP path
                )
                await queue_agent1.put(task)
                
                documents_collected += 1
                total_documents_processed += 1
                logging.debug(f"[{doc.patent_id}] Added to queue as one chunk.")
        
        except Exception as e:
            logging.error(f"Error processing file {input_path}: {e}", exc_info=True)
            continue
    
    # Send shutdown signals to all workers
    for _ in range(ConfigPipeline.MAX_WORKERS):
        await queue_agent1.put(None)
    
    logging.info(f"Producer finished. New documents: {total_documents_processed}, skipped (already processed): {skipped_count}")


