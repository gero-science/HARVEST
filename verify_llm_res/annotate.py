"""Annotate per-patent LLM results with hallucination flags.

Runs ``hallucination_detector.process_single_patent`` against patent XML sources
and writes a sidecar ``{patent_id}_hallu.json`` next to each ``_resolved.json``.
The sidecar is read during the BindingDB Parquet export so that hallucinated rows
are dropped at build time rather than in a post-hoc filter.

Standalone usage::

    python -m verify_llm_res \\
        --results-dir path/to/llm_results \\
        --source-dir  path/to/patent_zips \\
        --workers 16
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

HALLU_FILENAME_SUFFIX = "_hallu.json"
DETECTOR_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Convert detector stats → sidecar JSON
# ---------------------------------------------------------------------------

def patent_stats_to_hallu_json(
    patent_stats: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert ``process_single_patent`` output to ``_hallu.json`` content.

    Returns a dict ready for ``json.dump``.  When *patent_stats* is ``None``
    (no TSV found, ZIP missing, etc.) a clean sidecar is returned.
    """
    flagged_compounds: Dict[str, list[str]] = {}
    flagged_iupac: list[str] = []
    flagged_values: list[Dict[str, str]] = []

    if patent_stats is None:
        return {
            "detector_version": DETECTOR_VERSION,
            "flagged_compounds": flagged_compounds,
            "flagged_iupac": flagged_iupac,
            "flagged_values": flagged_values,
        }

    details = patent_stats.get("hallucination_details", {})

    # Stage 3 — chem_id
    for item in details.get("chem_id", []):
        cid = item.get("chem_id", "")
        if cid:
            flagged_compounds.setdefault(cid, [])
            if "chem_id" not in flagged_compounds[cid]:
                flagged_compounds[cid].append("chem_id")

    # Stage 3 — mismatch (keyed by chem_id)
    for item in details.get("mismatch", []):
        cid = item.get("chem_id", "")
        if cid:
            flagged_compounds.setdefault(cid, [])
            if "mismatch" not in flagged_compounds[cid]:
                flagged_compounds[cid].append("mismatch")

    # Stage 3 — iupac
    for item in details.get("iupac", []):
        iupac = item.get("iupac", "")
        if iupac and iupac not in flagged_iupac:
            flagged_iupac.append(iupac)

    # Stage 3 — tif (TIF filenames are stored in compound_IUPAC_name)
    for item in details.get("tif", []):
        tif = item.get("tif", "")
        if tif and tif not in flagged_iupac:
            flagged_iupac.append(tif)

    # Stage 2 — value
    s2_details = patent_stats.get("stage2", {}).get("hallucination_details", {})
    for item in s2_details.get("value", []):
        flagged_values.append({
            "compound": item.get("compound", ""),
            "value": item.get("value", ""),
            "binding_metric": item.get("binding_metric", ""),
        })

    return {
        "detector_version": DETECTOR_VERSION,
        "flagged_compounds": flagged_compounds,
        "flagged_iupac": flagged_iupac,
        "flagged_values": flagged_values,
    }


# ---------------------------------------------------------------------------
# Per-patent worker (runs inside ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _annotate_one_patent(
    patent_dir: Path,
    source_path: Path,
    force: bool,
) -> tuple[str, bool, Optional[str]]:
    """Check one patent and write its ``_hallu.json``.

    Returns ``(patent_id, wrote_file, error_or_None)``.
    """
    patent_id = patent_dir.name
    hallu_path = patent_dir / f"{patent_id}{HALLU_FILENAME_SUFFIX}"

    if hallu_path.exists() and not force:
        return patent_id, False, None

    try:
        # Lazy import — each worker process loads the detector independently.
        from verify_llm_res.hallucination_detector import process_single_patent

        patent_stats = process_single_patent(patent_dir, source_path)
        sidecar = patent_stats_to_hallu_json(patent_stats)
        hallu_path.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return patent_id, True, None
    except Exception as exc:
        return patent_id, False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Sidecar inspection
# ---------------------------------------------------------------------------

def _has_hallu_flags(patent_dir: Path) -> bool:
    """Return True if the existing ``_hallu.json`` has any non-empty flags."""
    hallu_path = patent_dir / f"{patent_dir.name}{HALLU_FILENAME_SUFFIX}"
    if not hallu_path.exists():
        return False
    try:
        data = json.loads(hallu_path.read_text(encoding="utf-8"))
        return bool(
            data.get("flagged_compounds")
            or data.get("flagged_iupac")
            or data.get("flagged_values")
        )
    except (json.JSONDecodeError, OSError):
        return False


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def annotate_all(
    results_dir: str | Path,
    source_dir: str | Path,
    *,
    workers: int = 16,
    force: bool = False,
    force_hallu: bool = False,
) -> Dict[str, Any]:
    """Annotate every patent directory under *results_dir*.

    Returns a summary dict with keys ``total``, ``annotated``, ``skipped``,
    ``failed``.
    """
    results_path = Path(results_dir)
    source_path = Path(source_dir)

    patent_dirs = sorted(
        p for p in results_path.iterdir()
        if p.is_dir() and (p / f"{p.name}_resolved.json").exists()
    )

    if not patent_dirs:
        logging.warning(f"No patent directories found in {results_dir}")
        return {"total": 0, "annotated": 0, "skipped": 0, "failed": 0}

    # --force-hallu: only re-run patents whose sidecar has flags
    if force_hallu and not force:
        flagged = [d for d in patent_dirs if _has_hallu_flags(d)]
        logging.info(
            f"--force-hallu: {len(flagged):,} of {len(patent_dirs):,} patents "
            f"have non-empty flags — re-running those"
        )
        patent_dirs = flagged
        force = True  # override skip for these

    logging.info(
        f"Annotating {len(patent_dirs)} patents with {workers} workers "
        f"(force={force})"
    )

    annotated = 0
    skipped = 0
    failed = 0
    errors: list[tuple[str, str]] = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _annotate_one_patent, patent_dir, source_path, force
                ): patent_dir
                for patent_dir in patent_dirs
            }
            for future in as_completed(futures):
                patent_id, wrote, error = future.result()
                if error:
                    failed += 1
                    errors.append((patent_id, error))
                    logging.error(f"  {patent_id}: {error}")
                elif wrote:
                    annotated += 1
                else:
                    skipped += 1
    else:
        for patent_dir in patent_dirs:
            patent_id, wrote, error = _annotate_one_patent(
                patent_dir, source_path, force
            )
            if error:
                failed += 1
                errors.append((patent_id, error))
                logging.error(f"  {patent_id}: {error}")
            elif wrote:
                annotated += 1
            else:
                skipped += 1

    summary = {
        "total": len(patent_dirs),
        "annotated": annotated,
        "skipped": skipped,
        "failed": failed,
    }

    print(f"\nAnnotation complete:")
    print(f"  Total patents: {summary['total']:,}")
    print(f"  Annotated:     {summary['annotated']:,}")
    print(f"  Skipped:       {summary['skipped']:,} (already annotated)")
    print(f"  Failed:        {summary['failed']:,}")

    if errors:
        print(f"\nFirst failures:")
        for pid, err in errors[:5]:
            print(f"  {pid}: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    return summary
