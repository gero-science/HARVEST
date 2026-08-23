"""Pipeline utilities: statistics aggregation, file reading, logging setup."""

import asyncio
import csv
import logging
import os
import subprocess
from pathlib import Path


def aggregate_usage_stats(usage_list: list[dict]) -> dict:
    """Aggregate a list of usage statistics into a single dictionary."""
    if not usage_list:
        return {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0,
            "request_count": 0, "token_per_request": [], "max_tokens_per_request": 0,
            "completion_tokens_per_request": [], "max_completion_tokens_per_request": 0
        }
    
    token_per_request = []
    completion_tokens_per_request = []
    for u in usage_list:
        if u and u.get("total_tokens", 0) > 0:
            token_per_request.append(u.get("total_tokens", 0))
            completion_tokens_per_request.append(u.get("completion_tokens", 0))
    
    # DEBUG: log aggregation details
    logging.debug(f"[aggregate_usage_stats] Processing {len(usage_list)} usage entries")
    logging.debug(f"[aggregate_usage_stats] Valid requests: {len(token_per_request)}")
    logging.debug(f"[aggregate_usage_stats] Completion tokens per request: {completion_tokens_per_request[:5]}...")  # First 5
    if completion_tokens_per_request:
        logging.debug(f"[aggregate_usage_stats] Max completion tokens: {max(completion_tokens_per_request)}")
    
    aggregated = {
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_list if u),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_list if u),
        "total_tokens": sum(u.get("total_tokens", 0) for u in usage_list if u),
        "cost": sum(u.get("cost", 0) for u in usage_list if u),
        "request_count": len([u for u in usage_list if u]),  # Count all valid requests, not only those with tokens
        "token_per_request": token_per_request,  # Keep for detailed statistics
        "max_tokens_per_request": max(token_per_request) if token_per_request else 0,
        "completion_tokens_per_request": completion_tokens_per_request,  # Keep for detailed statistics
        "max_completion_tokens_per_request": max(completion_tokens_per_request) if completion_tokens_per_request else 0
    }
    return aggregated


def read_file_list(file_list_path: str) -> list[str]:
    """
    Read a list of files from a text file.
    """
    files = []
    if not os.path.exists(file_list_path):
        raise FileNotFoundError(f"File list not found: {file_list_path}")
    
    with open(file_list_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            
            if not line.lower().endswith('.zip'):
                logging.warning(f"Line {line_num}: file {line} does not have a .zip extension. Skipping.")
                continue
                
            if not os.path.exists(line):
                logging.warning(f"Line {line_num}: file {line} not found. Skipping.")
                continue
                
            files.append(line)
    
    logging.info(f"Read {len(files)} valid ZIP files from the list.")
    return files


async def async_read_file_list(file_list_path: str) -> list[str]:
    """
    Async version of reading a file list from a text file.
    """
    def _read_sync():
        return read_file_list(file_list_path)
    
    return await asyncio.to_thread(_read_sync)


def setup_logging(log_level: str, output_dir: str = None, log_filename: str = "pipeline.log"):
    """
    Configure logging with console and file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        output_dir: Directory for log files. If set, logs are written to a file
                    with rotation (10MB x 5 files = 50MB max)
        log_filename: Log file name (default: pipeline.log)
    """
    from logging.handlers import RotatingFileHandler
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid logging level: {log_level}')
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    
    # Console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File output (if output directory is provided)
    if output_dir:
        log_file = Path(output_dir) / log_filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # RotatingFileHandler: 10MB per file, up to 5 files (50MB max)
        # Auto-flush on each write to preserve logs on interruption
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def save_git_commit_hash(output_dir: str) -> None:
    """
    Save the current Git commit hash to git_commit_hash.txt in output_dir.
    Used only in debug mode.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up one level (from pipeline/ to repo root)
        
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_root,
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        git_hash_file = os.path.join(output_dir, 'git_commit_hash.txt')
        with open(git_hash_file, 'w', encoding='utf-8') as f:
            f.write(git_hash)
        logging.debug(f"Commit hash saved to file: {git_hash_file} (commit: {git_hash[:8]})")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.debug(f"Failed to get Git commit hash: {e}")


def save_incomplete_patents(
    output_dir: str,
    incomplete_patents: list[dict]
) -> None:
    """
    Save the list of incomplete patents to a CSV file.
    
    Args:
        output_dir: Directory for saving results
        incomplete_patents: List of dicts describing incomplete patents
                           Each dict must contain: patent_id, zip_file_path, reason, stage
    """
    if not incomplete_patents:
        return
    
    csv_file = Path(output_dir) / "incomplete_patents.csv"
    file_exists = csv_file.exists()
    
    try:
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['zip_file_path', 'patent_id', 'reason', 'stage']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for patent_info in incomplete_patents:
                writer.writerow({
                    'zip_file_path': patent_info.get('zip_file_path', ''),
                    'patent_id': patent_info.get('patent_id', ''),
                    'reason': patent_info.get('reason', 'Unknown'),
                    'stage': patent_info.get('stage', 'Unknown')
                })
        
        logging.warning(f"Saved {len(incomplete_patents)} incomplete patents to {csv_file}")
    except Exception as e:
        logging.error(f"Error saving incomplete patents: {e}")


