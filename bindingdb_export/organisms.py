"""Organism normalization and species provenance for BindingDB export."""

import re
from typing import Any, Dict, List, Optional, Tuple

SPECIES_STATED = "stated"
HUMAN_DEFAULT_FLAG = "human_default"
INFERRED_FROM_TARGET = "inferred_from_target"
UNSTATED_SPECIES = "unstated"
HUMAN_FALLBACK_FLAG = "human_fallback"
UNSPECIFIED_ORGANISM = "unspecified"
# Written to the organism column instead of a separate species_source flag.
HUMAN_DEFAULT_ORGANISM = "human (default)"
LEGACY_HUMAN_DEFAULT_ORGANISM = "homo sapiens (default)"
HOMO_SAPIENS_SCIENTIFIC = "Homo sapiens"

# Labels that name no species: the patent text never said which organism was
# assayed. Everything here is treated as "species not stated", so the export
# flags the row instead of silently calling it human.
NONSPECIFIC_ORGANISMS = frozenset({
    "",
    "h",
    "unknown",
    "unspecified",
    "not specified",
    "not stated",
    "n/a",
    "na",
    "none",
    "null",
    "mammalian",
    "mammal",
    "mammals",
    "primate",
    "primates",
    "vertebrate",
    "vertebrates",
    "eukaryote",
    "eukaryotes",
})

# Organism keys that legacy runs wrote into single_proteins.json /
# protein_complexes.json for targets without a stated species.
LEGACY_NONSPECIFIC_KEYS = (UNSPECIFIED_ORGANISM, "human")

_HUMAN_LABELS = frozenset({"human", "homo sapiens", "hsa"})

# Labels that name the cell line a target was expressed in rather than the
# species of the target itself. ``chinese hamster ovary`` is CHO, while
# ``chinese hamster`` is the animal, so the patterns are anchored instead of
# matched as substrings. Labels that also name a species, such as ``cyno-CHO``
# or ``dog kidney``, are deliberately left out.
EXPRESSION_HOST_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^cho(?:[ -]?k1)?(?: cells?)?$",
        r"^chinese hamster ovary\b",
        r"^hek[ -]?293\b",
        r"^cos[ -]?7\b",
        r"^hela\b",
        r"^jurkat\b",
        r"^sf-?9\b",
        r"^3t3\b",
        r"^u-?2 ?os\b",
        r"\bsh-?sy5y\b",
    )
)

# Separator used by the protein postprocessor for complexes: gene, UniProt ID,
# accession and sequence are parallel lists of the same length.
PROTEIN_PART_SEPARATOR = ";"


def is_expression_host_label(organism: Optional[str]) -> bool:
    """True when the label names an expression system, not the target species."""
    if not organism:
        return False
    organism_lower = str(organism).lower().strip()
    return any(pattern.search(organism_lower) for pattern in EXPRESSION_HOST_PATTERNS)


def split_protein_field(value: Optional[str]) -> List[str]:
    """Split a ``;``-joined protein field into its components."""
    if not value:
        return []
    return [part.strip() for part in str(value).split(PROTEIN_PART_SEPARATOR) if part.strip()]


def is_human_uniprot_part(part: Optional[str]) -> bool:
    """True for a UniProt mnemonic of a Homo sapiens entry."""
    return bool(part) and str(part).strip().lower().endswith("_human")


def strip_human_protein_parts(
    gene: Optional[str],
    uniprot_id: Optional[str],
    accession: Optional[str],
    sequence: Optional[str],
) -> Tuple[str, str, str, str]:
    """Drop the Homo sapiens components of a protein annotation.

    Used when the text named a non-human species but a human sequence was
    attached. Components are removed from all parallel lists at once. When no
    component survives, only the sequence, accession and UniProt ID are dropped:
    ``gene`` still identifies the target and keeps the row exportable.
    """
    gene_parts = split_protein_field(gene)
    uniprot_parts = split_protein_field(uniprot_id)
    accession_parts = split_protein_field(accession)
    sequence_parts = split_protein_field(sequence)

    keep = [i for i, part in enumerate(uniprot_parts) if not is_human_uniprot_part(part)]
    if not uniprot_parts or len(keep) == len(uniprot_parts):
        return gene or "", uniprot_id or "", accession or "", sequence or ""

    aligned = len(accession_parts) == len(uniprot_parts) and len(sequence_parts) == len(uniprot_parts)
    if not keep or not aligned:
        return gene or "", "", "", ""

    join = PROTEIN_PART_SEPARATOR.join
    gene_out = join(gene_parts[i] for i in keep) if len(gene_parts) == len(uniprot_parts) else (gene or "")
    return (
        gene_out,
        join(uniprot_parts[i] for i in keep),
        join(accession_parts[i] for i in keep),
        join(sequence_parts[i] for i in keep),
    )


def normalize_organism(organism: Optional[str]) -> str:
    """Normalize an organism label for join keys and Parquet output.

    Missing / unknown / unspecified become ``unspecified``; they are never
    rewritten to ``human``.
    """
    if not organism:
        return UNSPECIFIED_ORGANISM
    organism_lower = organism.lower().strip()
    if organism_lower in ["unknown", "", "h", "H"] or "unspecified" in organism_lower:
        return UNSPECIFIED_ORGANISM
    return organism_lower


def is_nonspecific_organism(organism: Optional[str]) -> bool:
    """True when the label identifies no species (species not stated in text)."""
    if organism is None:
        return True
    organism_lower = str(organism).lower().strip()
    if not organism_lower:
        return True
    if "unspecified" in organism_lower or "not specified" in organism_lower:
        return True
    return organism_lower in NONSPECIFIC_ORGANISMS


def _organism_claims_human(organism: Optional[str]) -> bool:
    """True when the text label points at human material.

    Substring matching is intended here: labels look like ``recombinant human``
    or ``hek-293 cells (human)``.
    """
    if not organism:
        return False
    organism_lower = str(organism).lower().strip()
    return "human" in organism_lower or "homo sapiens" in organism_lower


def _species_is_human(organism_scientific: Optional[str]) -> bool:
    """True only for Homo sapiens itself, not for ``Human cytomegalovirus``."""
    if not organism_scientific:
        return False
    species_lower = str(organism_scientific).lower().strip()
    return species_lower in _HUMAN_LABELS or species_lower.startswith("homo sapiens")


def _has_human_sequence(uniprot_id: Optional[str]) -> bool:
    """True when any UniProt mnemonic in the field is a Homo sapiens entry."""
    if not uniprot_id:
        return False
    return any(part.strip().lower().endswith("_human") for part in str(uniprot_id).split(";"))


_PATHOGEN_MARKERS = ("virus", "phage", "bacteri", "archaea")


def scientific_name_is_pathogen(organism_scientific: Optional[str]) -> bool:
    """True for a viral / bacterial scientific name, not for a mammalian host."""
    if not organism_scientific:
        return False
    lower = str(organism_scientific).lower()
    return any(marker in lower for marker in _PATHOGEN_MARKERS)


def align_human_default_scientific(
    organism_scientific: Optional[str],
    uniprot_id: Optional[str],
) -> Optional[str]:
    """Point ``organism_scientific`` at Homo sapiens when the protein is human.

    Expression-host lookups often leave ``Cricetulus griseus`` (CHO) or a
    hybrid cell-line species on a ``*_HUMAN`` sequence. That name describes the
    host, not the target, so it is replaced. A virus or bacterium in the
    scientific name is kept: the UniProt may be a bad human hit, and the
    target is still the pathogen.
    """
    if not _has_human_sequence(uniprot_id):
        return organism_scientific
    if scientific_name_is_pathogen(organism_scientific):
        return organism_scientific
    return HOMO_SAPIENS_SCIENTIFIC


def classify_species_source(
    organism: Optional[str],
    organism_scientific: Optional[str] = None,
    uniprot_id: Optional[str] = None,
) -> str:
    """Classify where the species of a row came from.

    Used internally during export. ``human_default`` is written to the table as
    ``organism = "human (default)"`` rather than a separate column.

    - ``stated``: the patent text named the species.
    - ``human_default``: the text named no species and the row is annotated as
      human, so being human is an assumption rather than evidence.
    - ``inferred_from_target``: the text named no species, but the resolved
      species is not human (e.g. a viral or bacterial protein whose organism
      follows from the target name).
    - ``unstated``: no species in the text and none resolved.
    - ``human_fallback``: the text named a non-human species, but the attached
      sequence is a human one (legacy ``FastaGeneResolver`` ortholog fallback).

    A label naming only an expression system (``CHO``, ``HEK-293``) states no
    target species, so a human sequence under such a label is an assumption
    rather than a mismatch.
    """
    claims_human = _organism_claims_human(organism) or _species_is_human(organism_scientific)

    if not is_nonspecific_organism(organism):
        if _has_human_sequence(uniprot_id) and not claims_human:
            if is_expression_host_label(organism):
                return HUMAN_DEFAULT_FLAG
            return HUMAN_FALLBACK_FLAG
        return SPECIES_STATED

    if _species_is_human(organism_scientific) or _has_human_sequence(uniprot_id):
        return HUMAN_DEFAULT_FLAG
    if organism_scientific and str(organism_scientific).strip():
        return INFERRED_FROM_TARGET
    return UNSTATED_SPECIES


def lookup_protein_data(
    entries_by_organism: Dict[str, Any],
    organism: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Find the protein record for an organism label.

    For labels without a species the legacy organism keys are also tried,
    because runs before the species audit stored those targets under
    ``unspecified`` or ``human``. Labels that do name a species are never
    matched against another species.
    """
    if not entries_by_organism:
        return None

    keys = [normalize_organism(organism)]
    if is_nonspecific_organism(organism):
        keys.extend(key for key in LEGACY_NONSPECIFIC_KEYS if key not in keys)

    for key in keys:
        entry = entries_by_organism.get(key)
        if entry is not None:
            return entry

    return None
