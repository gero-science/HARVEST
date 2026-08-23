"""Top-level BindingDB Parquet export orchestration."""

import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pyarrow.parquet as pq

from enrich_data.ligand_info_extractor import LigandInfoExtractor

from .artifacts import get_patent_directories, load_patent_number_dict
from .batches import merge_parquet_batches, write_binding_batch
from .cache import ThreadSafeCache
from .patent_worker import process_patent_worker


def _default_cache_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "cache")


def _manifest_path(batch_file: str) -> str:
    """Sidecar recording which patents a batch covers."""
    return batch_file[: -len(".parquet")] + ".json"


def write_batch_manifest(batch_file: str, patent_dirs: list[str], batch_size: int) -> None:
    """Record a batch's membership so a later run can tell what it covers."""
    payload = {"batch_size": batch_size, "patents": sorted(patent_dirs)}
    with open(_manifest_path(batch_file), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _discard_batch(batch_file: str) -> None:
    for path in (batch_file, _manifest_path(batch_file)):
        try:
            os.remove(path)
        except OSError:
            pass


def reusable_batches(
    batches_dir: str, requested_dirs: list[str], force: bool
) -> tuple[set[int], set[str]]:
    """Find batches that can be kept, and the patents they already cover.

    Resume used to key on the batch index alone, which says nothing about the
    batch's contents: adding a patent silently reused a stale batch and dropped
    the new work from the output. A batch is reusable only when its Parquet
    still reads and every patent it covers is still requested; anything else is
    deleted so it gets recomputed.
    """
    if force or not os.path.exists(batches_dir):
        return set(), set()

    requested = set(requested_dirs)
    reusable: set[int] = set()
    covered: set[str] = set()

    for name in sorted(os.listdir(batches_dir)):
        if not (name.startswith("batch_") and name.endswith(".parquet")):
            continue
        batch_file = os.path.join(batches_dir, name)

        try:
            index = int(name.split("_")[1].split(".")[0])
            pq.read_table(batch_file)
            with open(_manifest_path(batch_file), encoding="utf-8") as f:
                members = set(json.load(f)["patents"])
        except Exception:
            # Unreadable, or written before manifests existed: cannot be trusted.
            _discard_batch(batch_file)
            continue

        if members and members <= requested:
            reusable.add(index)
            covered |= members
        else:
            _discard_batch(batch_file)

    return reusable, covered


def run_bindingdb_processing(
    input_dir: str,
    output_file: str,
    workers: int = 8,
    cache_dir: Optional[str] = None,
    patent_dict_file: Optional[str] = None,
    batch_size: int = 200,
    force: bool = False,
    use_opsin: bool = False,
    structure_source: str = "cdx",
) -> tuple[bool, str, float]:
    """
    Main Parquet table assembly function.

    Writes in batches of batch_size patents for fault tolerance.
    After interruption, processing can resume from the last completed batch.
    """
    start_time = time.time()

    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        if not os.path.isdir(input_dir):
            return False, f"Input directory does not exist: {input_dir}", 0.0

        patent_dict = load_patent_number_dict(patent_dict_file)

        if not cache_dir:
            cache_dir = _default_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        batches_dir = output_file + ".batches"

        logging.info("Scanning for patent directories...")
        patent_dirs = get_patent_directories(input_dir)
        logging.info(f"Found {len(patent_dirs)} patent directories")

        if not patent_dirs:
            return False, "No patent directories found", time.time() - start_time

        if force:
            if os.path.exists(batches_dir):
                shutil.rmtree(batches_dir)
                logging.info("Batches directory removed (--force)")
            if os.path.exists(output_file):
                os.remove(output_file)
                logging.info("Output file removed (--force)")

        os.makedirs(batches_dir, exist_ok=True)

        existing_batches, covered_dirs = reusable_batches(batches_dir, patent_dirs, force)

        if existing_batches:
            logging.info(
                f"Reusing {len(existing_batches)} completed batches "
                f"covering {len(covered_dirs)} patents"
            )

        # Only export what no reusable batch already covers, and number the new
        # batches after the highest reused one so both merge into the output.
        remaining = [d for d in patent_dirs if d not in covered_dirs]
        next_index = max(existing_batches, default=-1) + 1
        batches_to_process = [
            (next_index + offset, remaining[start:start + batch_size])
            for offset, start in enumerate(range(0, len(remaining), batch_size))
        ]
        num_batches = len(existing_batches) + len(batches_to_process)

        if covered_dirs:
            logging.info(
                f"{len(remaining)} patents to export, {len(covered_dirs)} already covered"
            )

        logging.info("Opening shared caches...")
        inchi_key_cache = ThreadSafeCache(os.path.join(cache_dir, "inchi_key_cache.db"), "inchi_key")
        smiles_cache = ThreadSafeCache(os.path.join(cache_dir, "smiles_cache.db"), "smiles")

        try:
            opsin = None
            if use_opsin:
                from pyopsin.pyopsin import PyOpsin
                opsin = PyOpsin()
                logging.info("PyOpsin enabled (--use-opsin)")
            else:
                logging.info("PyOpsin disabled")

            ligand_extractor = LigandInfoExtractor(
                opsin,
                inchi_key_cache,
                smiles_cache,
            )

            logging.info("Caches opened successfully")
            logging.info(f"Using {workers} workers, batch size {batch_size}")

            total_written = 0
            total_skipped = 0
            total_errors = 0
            # Exact, not len(batches) * batch_size: the last batch may be partial.
            total_patents_processed = len(covered_dirs)

            for batch_idx, batch_patents in batches_to_process:
                batch_file = os.path.join(batches_dir, f"batch_{batch_idx:04d}.parquet")
                batch_bindings = []
                batch_errors = 0

                logging.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_patents)} patents)...")

                worker_args = [(patent_dir, ligand_extractor, structure_source) for patent_dir in batch_patents]

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(process_patent_worker, args): args[0]
                        for args in worker_args
                    }

                    for i, future in enumerate(as_completed(futures), 1):
                        patent_dir = futures[future]

                        try:
                            patent_name, updated_bindings, error = future.result()

                            if error:
                                batch_errors += 1
                                logging.error(f"  [{i}/{len(batch_patents)}] {error}")
                            elif updated_bindings:
                                batch_bindings.extend(updated_bindings)
                                logging.debug(
                                    f"  [{i}/{len(batch_patents)}] {os.path.basename(patent_name)}: "
                                    f"{len(updated_bindings)} bindings"
                                )
                        except Exception as e:
                            batch_errors += 1
                            logging.error(f"  [{i}/{len(batch_patents)}] Unexpected error: {e}")

                batch_written, batch_skipped = write_binding_batch(batch_bindings, batch_file, patent_dict)
                # A batch with no rows writes no Parquet, so it gets no manifest
                # either: reuse is driven by the Parquet files on disk.
                if os.path.exists(batch_file):
                    write_batch_manifest(batch_file, batch_patents, batch_size)

                total_written += batch_written
                total_skipped += batch_skipped
                total_errors += batch_errors
                total_patents_processed += len(batch_patents)

                logging.info(
                    f"Batch {batch_idx + 1}/{num_batches} complete: "
                    f"{batch_written} rows written, {batch_errors} errors "
                    f"(total: {total_written} rows, {total_patents_processed} patents)"
                )

            if batches_to_process or existing_batches:
                logging.info("Merging batches into final file...")
                final_rows = merge_parquet_batches(batches_dir, output_file)
                logging.info(f"Written {final_rows} rows to {output_file}")
            else:
                final_rows = 0

            if final_rows == 0:
                return False, "No valid bindings to write", time.time() - start_time

        finally:
            logging.info("Closing caches...")
            inchi_key_cache.close()
            smiles_cache.close()
            logging.info("Caches closed")

        elapsed_time = time.time() - start_time
        message = f"Successfully processed {total_patents_processed} patents, {final_rows} bindings written, {total_errors} errors"

        logging.info(f"Processing complete: {message}")
        logging.info(f"Output: {output_file}")
        logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        return True, message, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"Error during processing: {str(e)}"
        # exception() rather than error(): the message alone gives no location,
        # and this handler covers the whole export, so without a traceback a
        # failure deep in row conversion is indistinguishable from one in I/O.
        logging.exception(error_msg)
        return False, error_msg, elapsed_time
