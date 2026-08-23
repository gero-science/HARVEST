"""Core pipeline module with the main run_pipeline_async function."""

import asyncio
import logging
import time
from pathlib import Path

from .config import ConfigPipeline
from llm.call_llm import clear_global_failed_requests_log

from .producer import producer_chunks
from .workers import worker_agent1, worker_agent2_aggregator


async def run_pipeline_async(
        input_paths: list[str],
        output_dir: str,
        limit: int,
        resume: bool,
        debug_mode: bool = False,
):
    """
    Main asynchronous entry point.
    Starts the fully asynchronous 3-component pipeline.
    
    Args:
        input_paths: List of paths to patent ZIP archives
        output_dir: Directory for saving results
        limit: Cap on the number of patents to process (None = no limit)
        resume: Resume mode (shows additional progress information)
        debug_mode: Debug mode (saves intermediate data)
    """
    # Record pipeline start time
    start_time = time.time()
    
    # Clear the global failed LLM request log for a fresh run
    if debug_mode:
        clear_global_failed_requests_log()
    
    logging.info("Starting FULLY ASYNCHRONOUS pipeline")
    logging.info(f"Files to process: {len(input_paths)}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Maximum workers: {ConfigPipeline.MAX_WORKERS}")
    
    # Create the output directory asynchronously
    output_path = Path(output_dir)
    await asyncio.to_thread(output_path.mkdir, parents=True, exist_ok=True)
    
    # Load already processed patents asynchronously
    already_processed = set()
    processed_file = Path(output_dir) / "processed_patents.txt"
    if await asyncio.to_thread(processed_file.exists):
        content = await asyncio.to_thread(processed_file.read_text, encoding='utf-8')
        already_processed = set(line.strip() for line in content.splitlines() if line.strip())
        logging.info(f"Loaded {len(already_processed)} already processed patents.")
    
    if resume:
        logging.info("Resume mode enabled.")
    
    # Imported here, not at module scope: llm.config reads LLM_KEY and friends
    # from the environment eagerly, and ``pipeline/__init__.py`` imports this
    # module. A top-level import would therefore make *every* pipeline entry
    # point require API credentials, including the LLM-free
    # ``--stages export,postprocess`` rebuild of an existing results directory.
    from llm.config import ConfigExtractor

    # Create configurations
    config_agent1 = ConfigExtractor()
    
    # Create asynchronous queues
    queue_agent1 = asyncio.Queue(maxsize=ConfigPipeline.QUEUE_SIZE)
    queue_agent2 = asyncio.Queue(maxsize=ConfigPipeline.QUEUE_SIZE)
    
    # Start all components in parallel
    tasks = []
    
    # 1 producer (chunk generator)
    tasks.append(asyncio.create_task(
        producer_chunks(queue_agent1, input_paths, limit, already_processed)
    ))
    
    # N workers for Agent 1 (chunk processors)
    for i in range(ConfigPipeline.MAX_WORKERS):
        tasks.append(asyncio.create_task(
            worker_agent1(queue_agent1, queue_agent2, config_agent1, debug_mode, output_dir)
        ))
    
    # 1 aggregator (Agent 2)
    aggregator_task = asyncio.create_task(
        worker_agent2_aggregator(
            queue_agent2,
            output_dir,
            start_time,
            debug_mode,
        )
    )
    tasks.append(aggregator_task)
    
    # Helper to shut down the aggregator cleanly
    async def wait_and_stop_aggregator():
        # Wait for all Agent 1 workers to finish
        for i in range(len(tasks) - 1):  # All tasks except the aggregator
            await tasks[i]

        # Send shutdown signal to the aggregator
        await queue_agent2.put(None)

        # Wait for the aggregator to finish and return its result
        incomplete_patents = await aggregator_task
        return incomplete_patents
    
    # Wait for all components to finish
    logging.info("Starting all asynchronous pipeline components...")
    incomplete_patents = await wait_and_stop_aggregator()
    
    # Compute total runtime
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    
    logging.info("Asynchronous pipeline completed!")
    logging.info(f"Total elapsed time: {hours:02d}:{minutes:02d}:{seconds:06.3f}")
    
    # Return the list of incomplete patents
    return incomplete_patents if incomplete_patents else []
