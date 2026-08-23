"""Workers module for chunk processing and result aggregation."""

import asyncio
import logging
import time
from pathlib import Path

from llm.extractor import BioactivityExtractor
from llm.call_llm import LLM, save_global_failed_requests_log

from .accounting import (
    add_usage_to_global_agent,
    create_global_stats,
    finalize_total_usage,
)
from .agent1_processing import (
    build_agent1_error_detail,
    build_agent1_error_payload,
    enrich_agent1_data_from_document,
    process_agent1_task,
)
from .agent2_orchestration import process_patent_agent2 as process_patent_agent2_impl
from .agent2_processing import append_error_detail
from .aggregator_state import AggregatorState, parse_agent1_payload
from .artifacts import save_patent_result_artifacts
from .debug_artifacts import save_debug_data as save_debug_data_impl
from .error_handling import create_empty_error_stats
from .finalization import (
    print_final_statistics,
    save_error_logs,
)


async def worker_agent1(queue_in, queue_out, config_agent1, debug_mode=False, output_dir=None):
    """
    Function 2: Asynchronous chunk worker (Agent 1).
    Uses fully async methods for maximum throughput.
    
    Args:
        queue_in: Input queue with ChunkProcessingTask items
        queue_out: Output queue for passing results to the aggregator
        config_agent1: Configuration for Agent 1
        debug_mode: Debug mode
        output_dir: Directory for saving results (including LLM responses)
    """
    # Create one LLM instance for this worker
    llm = LLM.from_config(config_agent1, logging.getLogger(__name__), debug_mode, output_dir)
    extractor = BioactivityExtractor(config_agent1, llm)
    
    # Per-worker error counters
    worker_error_stats = create_empty_error_stats()
    
    try:
        while True:
            task = await queue_in.get()
            if task is None:  # Shutdown signal
                break
                
            try:
                await queue_out.put(await process_agent1_task(extractor, task, output_dir=output_dir))
                
            except Exception as e:
                error_type, error_detail = build_agent1_error_detail(task, e)
                worker_error_stats[error_type] += 1
                
                logging.error(f"[Agent1] Error processing chunk for {task.patent_id}: {type(e).__name__}: {e}")
                logging.debug(f"[Agent1] Classified as {error_type}")
                await queue_out.put(build_agent1_error_payload(task, worker_error_stats, error_detail))
    finally:
        # Close the aiohttp session when the worker exits
        await llm.close()


async def save_patent_results(
    patent_id,
    resolved,
    unresolved,
    output_dir,
    processed_patents_file,
    cdx_data=None,
):
    """
    Save patent processing results to files.
    
    Args:
        patent_id: Patent ID
        resolved: List of resolved aliases
        unresolved: List of unresolved aliases
        output_dir: Directory for saving results
        processed_patents_file: File containing processed patent IDs
        cdx_data: CDX structures for cdx_results.json ({chem_num -> {smiles, inchikey}})
    """
    return await save_patent_result_artifacts(
        patent_id,
        resolved,
        unresolved,
        output_dir,
        processed_patents_file,
        to_thread=asyncio.to_thread,
        cdx_data=cdx_data,
    )


async def save_debug_data(patent_id, document, output_dir):
    """Compatibility wrapper for moved debug artifact writer."""
    return await save_debug_data_impl(patent_id, document, output_dir)


async def process_patent_agent2(
    patent_id, 
    measures_list, 
    patent_text, 
    document,
    output_dir,
    processed_patents_file,
    global_stats,
    detailed_error_log,
    agent1_usage_for_patent,
    debug_mode=False,
):
    """Compatibility wrapper for the moved Agent 2 orchestration."""
    return await process_patent_agent2_impl(
        patent_id,
        measures_list,
        patent_text,
        document,
        output_dir,
        processed_patents_file,
        global_stats,
        detailed_error_log,
        agent1_usage_for_patent,
        debug_mode,
    )


async def worker_agent2_aggregator(
        queue_in, 
        output_dir, 
        start_time, 
        debug_mode: bool = False,
):
    """
    Function 3: Asynchronous aggregator.
    Collects chunks into patents and runs Agent 2 in parallel for each patent.
    
    Args:
        queue_in: Input queue with results from Agent 1
        output_dir: Directory for saving results
        start_time: Pipeline start time
        debug_mode: Debug mode
    """
    state = AggregatorState()
    processed_patents_file = Path(output_dir) / "processed_patents.txt"
    
    # Active Agent 2 tasks
    agent2_tasks = []
    
    # Global error counters
    global_error_stats = {
        "agent1_errors": create_empty_error_stats(),
        "agent2_errors": create_empty_error_stats()
    }
    
    # File for incremental error logging (do not accumulate in memory)
    error_log_file = Path(output_dir) / "debug_error_log.jsonl"
    detailed_error_count = 0
    
    # Global statistics
    global_stats = create_global_stats()
    
    while True:
        patent_data = await queue_in.get()
        if patent_data is None:  # Shutdown signal
            break
        
        payload = parse_agent1_payload(patent_data)
        patent_id = payload.patent_id
        
        # Aggregate errors from Agent 1 workers
        for error_type, count in payload.worker_errors.items():
            if count > 0:
                global_error_stats["agent1_errors"][error_type] += count
                logging.debug(f"[Agent1] Aggregated error {error_type}: +{count} for patent {patent_id}")
        
        # Append errors immediately to file (do not accumulate in memory)
        if payload.error_detail:
            append_error_detail(error_log_file, payload.error_detail)
            detailed_error_count += 1
            logging.debug(f"[ErrorLog] Added error from Agent1: {payload.error_detail['error_type']} for patent {patent_id}")
        
        state.ingest_payload(payload)
        
        # Check whether all chunks for this patent have arrived
        if state.is_complete(patent_id):
            completed = state.pop_completed_patent(patent_id)
            measures_list = completed.measures_list
            
            # Update global Agent 1 statistics
            add_usage_to_global_agent(global_stats, "agent1", completed.agent1_usage)
            
            # Update bioactivity data count
            global_stats["data_counts"]["bioactivity_data_found"] += len(measures_list)
            
            logging.info(
                f"Patent {patent_id}: received all "
                f"{completed.received_chunks}/{completed.total_chunks_in_patent} chunks. "
                f"Total data points: {len(measures_list)}"
            )
            
            if measures_list:  # If data is present
                if completed.failed_chunks:
                    # Partial extraction: kept, because the chunks that did
                    # answer carry real data, but the patent is marked
                    # processed below and will not be retried.
                    logging.warning(
                        f"Patent {patent_id}: {completed.failed_chunks} of "
                        f"{completed.received_chunks} chunks failed; continuing with "
                        "partial data."
                    )
                # Start async patent processing
                task = asyncio.create_task(
                    process_patent_agent2(
                        patent_id, measures_list, completed.patent_text,
                        completed.document,
                        output_dir, processed_patents_file,
                        global_stats,
                        error_log_file,
                        completed.agent1_usage,
                        debug_mode,
                    )
                )
                # Store patent_id on the task for tracking
                task.patent_id = patent_id
                agent2_tasks.append(task)
                logging.info(f"Patent {patent_id} sent for PARALLEL processing by Agent 2. Active tasks: {len(agent2_tasks)}")
            elif completed.failed_chunks:
                # Empty because Agent 1 failed on every chunk, not because the
                # patent holds no bioactivity data. Marking it processed would
                # make --resume skip it forever, so a transient outage or an
                # expired API key would silently drop patents from the corpus.
                # Leave it unmarked: the next run retries it.
                logging.error(
                    f"Patent {patent_id} produced no data after "
                    f"{completed.failed_chunks}/{completed.received_chunks} chunks failed; "
                    "not marking it processed so it is retried on the next run."
                )
            else:
                # Every chunk answered and there was genuinely nothing to
                # extract, so this patent really is done.
                def append_empty_to_file():
                    with open(processed_patents_file, "a", encoding='utf-8') as f:
                        f.write(f"{patent_id}\n")

                await asyncio.to_thread(append_empty_to_file)
                logging.info(f"Patent {patent_id} processed with no data.")
    
    # Collect incomplete patents from pending_data (missing chunks)
    incomplete_patents = state.build_incomplete_patents()
    for incomplete in incomplete_patents:
        logging.warning(f"Patent {incomplete['patent_id']} incomplete: {incomplete['reason'].lower()}")
    
    # Wait for all Agent 2 tasks to finish
    if agent2_tasks:
        logging.info(f"Waiting for {len(agent2_tasks)} asynchronous Agent 2 tasks to finish...")
        results = await asyncio.gather(*agent2_tasks, return_exceptions=True)
        
        # Check task results for errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # If a task failed, extract patent_id from the task
                task_info = agent2_tasks[i]
                patent_id = getattr(task_info, 'patent_id', f"unknown_{i}")
                
                incomplete_patents.append({
                    'patent_id': patent_id,
                    'zip_file_path': '',
                    'reason': f'Error during Agent 2 processing: {str(result)}',
                    'stage': 'agent2_processing_error'
                })
                logging.error(f"Patent {patent_id} incomplete due to Agent 2 error: {result}")
        
        logging.info("All Agent 2 tasks completed!")
    
    # Save detailed error logs in debug mode
    await _save_error_logs(output_dir, error_log_file, detailed_error_count, debug_mode)
    
    # Save global failed LLM request logs in debug mode
    if debug_mode:
        save_global_failed_requests_log(output_dir)
    
    # Compute final statistics
    finalize_total_usage(global_stats)
    
    # Print final statistics
    _print_final_statistics(global_stats, start_time)
    
    # SAVE EXTENDED STATISTICS TO FILES
    global_stats["error_analysis"] = global_error_stats
    
    from .statistics import save_statistics_to_files
    total_time = time.time() - start_time
    await save_statistics_to_files(output_dir, global_stats, total_time, start_time, debug_mode)
    
    # Return the list of incomplete patents
    return incomplete_patents



async def _save_error_logs(output_dir, error_log_file, error_count, debug_mode):
    """Compatibility wrapper for moved error log finalization."""
    return await save_error_logs(output_dir, error_log_file, error_count, debug_mode)


def _print_final_statistics(global_stats, start_time):
    """Compatibility wrapper for moved final statistics logging."""
    return print_final_statistics(global_stats, start_time)
