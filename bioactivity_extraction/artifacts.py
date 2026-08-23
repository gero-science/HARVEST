"""Stage artifact writing helpers for the bioactivity extraction pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def save_stage_responses(
    patent_id: str,
    stage1_response: str,
    stage2_responses: List[str],
    stage3_responses: List[str],
    stage3_compounds_cleaned: Optional[List[Dict]] = None,
    output_dir: Optional[str] = None,
    stage2_failed: Optional[List] = None,
    stage3_failed: Optional[List] = None,
    logger=None,
) -> None:
    """
    Save TSV responses from the three extraction stages to separate files.
    """
    try:
        if output_dir:
            patent_dir = Path(output_dir) / patent_id
            patent_dir.mkdir(parents=True, exist_ok=True)
            stage1_file = patent_dir / f"{patent_id}_agent1_stage1_targets.tsv"
            stage2_file = patent_dir / f"{patent_id}_agent1_stage2_bioactivity.tsv"
            stage3_file = patent_dir / f"{patent_id}_agent1_stage3_compounds.tsv"
        else:
            logs_dir = Path("llm_responses")
            logs_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stage1_file = logs_dir / f"{patent_id}_{timestamp}_stage1.tsv"
            stage2_file = logs_dir / f"{patent_id}_{timestamp}_stage2.tsv"
            stage3_file = logs_dir / f"{patent_id}_{timestamp}_stage3.tsv"

        with open(stage1_file, "w", encoding="utf-8") as f:
            f.write(stage1_response)
        if logger:
            logger.info(f"Stage 1 response saved to: {stage1_file}")

        with open(stage2_file, "w", encoding="utf-8") as f:
            for idx, response in enumerate(stage2_responses):
                if idx > 0:
                    f.write("\n")
                f.write(response)
        if logger:
            if len(stage2_responses) > 1:
                logger.info(f"Stage 2 response saved to: {stage2_file} ({len(stage2_responses)} parts)")
            else:
                logger.info(f"Stage 2 response saved to: {stage2_file}")

        with open(stage3_file, "w", encoding="utf-8") as f:
            for idx, response in enumerate(stage3_responses):
                if idx > 0:
                    f.write("\n")
                f.write(response)
        if logger:
            if len(stage3_responses) > 1:
                logger.info(f"Stage 3 response saved to: {stage3_file} ({len(stage3_responses)} parts)")
            else:
                logger.info(f"Stage 3 response saved to: {stage3_file}")

        if stage3_compounds_cleaned is not None:
            if output_dir:
                stage3_cleaned_file = patent_dir / f"{patent_id}_agent1_stage3_compounds_cleaned.tsv"
            else:
                stage3_cleaned_file = logs_dir / f"{patent_id}_{timestamp}_stage3_cleaned.tsv"

            if stage3_compounds_cleaned:
                headers = list(stage3_compounds_cleaned[0].keys())

                with open(stage3_cleaned_file, "w", encoding="utf-8") as f:
                    f.write("\t".join(headers) + "\n")

                    for item in stage3_compounds_cleaned:
                        row = [str(item.get(h, "")) for h in headers]
                        f.write("\t".join(row) + "\n")

                if logger:
                    logger.info(f"Stage 3 cleaned response saved to: {stage3_cleaned_file}")
            else:
                with open(stage3_cleaned_file, "w", encoding="utf-8") as f:
                    f.write("compound\tcompound_IUPAC_name\tchemical_id\n")
                if logger:
                    logger.info(f"Stage 3 cleaned response saved (empty) to: {stage3_cleaned_file}")

        if stage2_failed:
            stage2_failed_file = (
                patent_dir / f"{patent_id}_agent1_stage2_failed.tsv"
                if output_dir
                else logs_dir / f"{patent_id}_{timestamp}_stage2_failed.tsv"
            )
            with open(stage2_failed_file, "w", encoding="utf-8") as f:
                f.write("# Malformed/Invalid responses from Stage 2\n")
                f.write("# These lines failed format validation\n\n")
                for failed_item in stage2_failed:
                    f.write(f"# Continuation {failed_item.get('continuation_num')}\n")
                    f.write(f"# Reason: {failed_item.get('reason')}\n")
                    if "invalid_lines" in failed_item:
                        for line in failed_item["invalid_lines"]:
                            f.write(f"{line}\n")
                    elif "full_response" in failed_item:
                        f.write(f"{failed_item['full_response']}\n")
                    f.write("\n")
            if logger:
                logger.info(f"Stage 2 failed responses saved to: {stage2_failed_file}")

        if stage3_failed:
            stage3_failed_file = (
                patent_dir / f"{patent_id}_agent1_stage3_failed.tsv"
                if output_dir
                else logs_dir / f"{patent_id}_{timestamp}_stage3_failed.tsv"
            )
            with open(stage3_failed_file, "w", encoding="utf-8") as f:
                f.write("# Malformed/Invalid responses from Stage 3\n")
                f.write("# These lines failed format validation\n\n")
                for failed_item in stage3_failed:
                    f.write(f"# Continuation {failed_item.get('continuation_num')}\n")
                    f.write(f"# Reason: {failed_item.get('reason')}\n")
                    if "invalid_lines" in failed_item:
                        for line in failed_item["invalid_lines"]:
                            f.write(f"{line}\n")
                    elif "full_response" in failed_item:
                        f.write(f"{failed_item['full_response']}\n")
                    f.write("\n")
            if logger:
                logger.info(f"Stage 3 failed responses saved to: {stage3_failed_file}")

    except Exception as e:
        if logger:
            logger.warning(f"Failed to save stage responses: {e}")
