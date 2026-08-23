"""Artifact loading helpers for BindingDB export."""

import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd


CDX_RESULTS_FILENAME = "cdx_results.json"
MOL_RESULTS_FILENAME = "mol_results.json"
HALLU_SUFFIX = "_hallu.json"
STAGE1_TARGETS_SUFFIX = "_agent1_stage1_targets.tsv"


def get_patent_directories(input_dir: str) -> list[str]:
    """Return the list of patent directories, in a stable order.

    Sorted because the export batches this list and reuses completed batches
    across runs; os.walk order is filesystem-dependent, so without sorting the
    same corpus could be split differently on each run.
    """
    patent_dirs = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_resolved.json"):
                patent_dirs.append(root)
                break

    return sorted(patent_dirs)


def load_resolved_bindings(patent_dir: str, patterns: list[str]) -> list[Dict[str, Any]]:
    bindings = []

    for root, dirs, files in os.walk(patent_dir):
        for file in files:
            if any(file.endswith(pat) for pat in patterns):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for dat in data:
                                dat["patent_number"] = os.path.basename(patent_dir)
                            bindings.extend(data)
                except Exception as e:
                    logging.error(f"Error loading {file_path}: {e}")

    return bindings


def load_stage1_organisms(patent_dir: str) -> Dict[str, str]:
    """Map ``assay_id`` to the organism exactly as Stage 1 extracted it.

    Runs before the species audit overwrote the per-row organism with ``human``
    during in-pipeline protein enrichment, so the Stage 1 TSV is the only place
    where "the text named no species" survives for those results.

    Returns:
        Dict {assay_id: organism}, empty when no Stage 1 TSV is present.
    """
    try:
        entries = sorted(os.listdir(patent_dir))
    except OSError:
        return {}

    organisms: Dict[str, str] = {}
    for name in entries:
        if not name.endswith(STAGE1_TARGETS_SUFFIX):
            continue

        path = os.path.join(patent_dir, name)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline().rstrip("\n").split("\t")
                if "assay_id" not in header or "organism" not in header:
                    continue

                assay_index = header.index("assay_id")
                organism_index = header.index("organism")
                for line in f:
                    fields = line.rstrip("\n").split("\t")
                    # A wrong field count means the row is misaligned (e.g. a tab
                    # inside a description); fall back to the row's own label.
                    if len(fields) != len(header):
                        continue
                    assay_id = fields[assay_index].strip()
                    if assay_id and assay_id not in organisms:
                        organisms[assay_id] = fields[organism_index].strip()
        except Exception as e:
            logging.debug(f"Error loading {path}: {e}")

    return organisms


def load_cdx_results(patent_dir: str) -> Dict[str, Dict]:
    """
    Load CDX parsing results from cdx_results.json.

    Returns:
        Dict {chem_num: {"smiles": str, "inchikey": str}}
    """
    cdx_path = os.path.join(patent_dir, CDX_RESULTS_FILENAME)
    if not os.path.exists(cdx_path):
        return {}

    try:
        with open(cdx_path, "r") as f:
            data = json.load(f)
        return data.get("compounds", {})
    except Exception as e:
        logging.debug(f"Error loading cdx_results.json from {patent_dir}: {e}")
        return {}


def load_mol_results(patent_dir: str) -> Dict[str, Dict]:
    """Load MOL parsing results from mol_results.json.

    Same format as :func:`load_cdx_results` but sourced from MOL files
    instead of CDX structural drawings.

    Returns:
        Dict {chem_num: {"smiles": str, "inchikey": str}}
    """
    mol_path = os.path.join(patent_dir, MOL_RESULTS_FILENAME)
    if not os.path.exists(mol_path):
        return {}

    try:
        with open(mol_path, "r") as f:
            data = json.load(f)
        return data.get("compounds", {})
    except Exception as e:
        logging.debug(f"Error loading mol_results.json from {patent_dir}: {e}")
        return {}


def load_hallu_flags(patent_dir: str) -> Optional[Dict[str, Any]]:
    """Load hallucination sidecar flags for a patent, if present.

    Returns the parsed ``_hallu.json`` content, or ``None`` when the file is
    absent (annotation step not run for this patent).
    """
    patent_id = os.path.basename(patent_dir)
    hallu_path = os.path.join(patent_dir, f"{patent_id}{HALLU_SUFFIX}")
    if not os.path.exists(hallu_path):
        return None
    try:
        with open(hallu_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Error loading hallu flags from {hallu_path}: {e}")
        return None


def is_hallucinated(binding: Dict[str, Any], hallu_flags: Dict[str, Any]) -> bool:
    """Check whether a single binding is flagged as hallucinated.

    .. deprecated::
        Use :func:`mask_hallucinated_fields` instead — it masks individual
        fields rather than dropping the entire binding.
    """
    return bool(mask_hallucinated_fields(binding, hallu_flags, dry_run=True))


def mask_hallucinated_fields(
    binding: Dict[str, Any],
    hallu_flags: Dict[str, Any],
    *,
    dry_run: bool = False,
) -> set[str]:
    """Mask hallucinated fields on *binding* and return the set of masked field names.

    Checks three categories:

    * ``chemical_id`` — against ``flagged_compounds`` (chem_id / mismatch).
    * ``compound_IUPAC_name`` — against ``flagged_iupac`` (iupac / tif).
    * ``value`` — against ``flagged_values``, matched on the raw
      ``original_value`` string together with ``compound``.
      ``original_value`` is preserved by ``process_row()`` before nM
      conversion, so the match uses the same string the detector checked.

    Masked fields are set to ``""`` (IUPAC, chem_id) or ``None`` (value).
    When *dry_run* is True the binding is not modified — only the set of
    field names that *would* be masked is returned.
    """
    masked: set[str] = set()
    flagged_compounds = hallu_flags.get("flagged_compounds", {})
    flagged_iupac = hallu_flags.get("flagged_iupac", [])
    flagged_values = hallu_flags.get("flagged_values", [])

    # --- chemical_id ---
    chem_id = (binding.get("chemical_id") or "").strip()
    if chem_id and chem_id in flagged_compounds:
        masked.add("chemical_id")

    # --- compound_IUPAC_name ---
    iupac = (binding.get("compound_IUPAC_name") or "").strip()
    if iupac and iupac in flagged_iupac:
        masked.add("compound_IUPAC_name")

    # --- value (matched on pre-normalization string) ---
    if flagged_values:
        original_value = (binding.get("original_value") or "").strip()
        compound = (binding.get("compound") or "").strip()
        if original_value:
            for fv in flagged_values:
                if (fv.get("value", "").strip() == original_value
                        and fv.get("compound", "").strip() == compound):
                    masked.add("value")
                    break

    # Apply masks
    if masked and not dry_run:
        if "chemical_id" in masked:
            binding["chemical_id"] = ""
        if "compound_IUPAC_name" in masked:
            binding["compound_IUPAC_name"] = ""
        if "value" in masked:
            binding["value"] = None

    return masked


_DEFAULT_PATENT_MAPPING = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "curated_data",
    "patent_mapping.csv",
)


def load_patent_number_dict(patent_dict_file: Optional[str] = None) -> Dict[str, int]:
    """Load the application_number → patent_number mapping.

    Accepts CSV (``patent_number,application_number``) or JSONL with the same
    two columns.  Defaults to ``curated_data/patent_mapping.csv`` which is
    shipped in the repository.
    """
    if not patent_dict_file:
        if os.path.exists(_DEFAULT_PATENT_MAPPING):
            patent_dict_file = _DEFAULT_PATENT_MAPPING
        else:
            # Legacy fallback paths (pre-release layouts)
            for path in ("data/bdb_100_dict.json", "./output/patent_number_dict.json"):
                if os.path.exists(path):
                    patent_dict_file = path
                    break

    if not patent_dict_file or not os.path.exists(patent_dict_file):
        return {}

    try:
        if patent_dict_file.endswith(".csv"):
            df = pd.read_csv(patent_dict_file, dtype=str)
        else:
            df = pd.read_json(patent_dict_file, lines=True, dtype=str)

        def _to_numeric(val):
            """Strip country prefix (e.g. 'US') so the value stays int-compatible."""
            s = str(val).strip()
            digits = s.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            try:
                return int(digits) if digits else s
            except ValueError:
                return s

        patent_dict = {
            str(k).strip(): _to_numeric(v)
            for k, v in zip(df["application_number"], df["patent_number"])
        }
        logging.info(f"Loaded {len(patent_dict)} patent number mappings from {patent_dict_file}")
        return patent_dict
    except Exception as e:
        logging.debug(f"Failed to load patent dictionary: {e}")
        return {}
