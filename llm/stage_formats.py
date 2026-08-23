"""
Centralized TSV format definitions for all patent processing stages.

This module contains a single definition of the data structure for each stage
to avoid duplication and inconsistencies in the code.
"""

from typing import List, Tuple


class StageFormats:
    """TSV formats for all data extraction stages."""

    # ============================================================================
    # STAGE 1: Extract protein targets and assay descriptions
    # ============================================================================
    STAGE1_COLUMNS: List[str] = [
        "assay_id",              # Unique assay identifier
        "protein_target_name",   # Protein target name
        "is_complex",            # Protein complex flag (Y/N)
        "protein_modification",  # Protein modification or substrate/reporter
        "assay_description",     # Assay description
        "reasoning",             # XML paragraph ID where the assay was found
        "assay",                 # Assay type
        "organism",              # Source organism for the protein
        "extreme_conditions"     # Extreme conditions (Y/N/U)
    ]

    STAGE1_HEADER: str = "\t".join(STAGE1_COLUMNS)
    STAGE1_COLUMN_COUNT: int = len(STAGE1_COLUMNS)

    # Normalized patterns for header filtering (lowercase, without _ and -)
    STAGE1_PATTERN: Tuple[str, ...] = tuple(
        col.lower().replace('_', '').replace('-', '')
        for col in STAGE1_COLUMNS
    )

    # ============================================================================
    # STAGE 2: Extract bioactivities
    # ============================================================================
    STAGE2_COLUMNS: List[str] = [
        "compound",         # Compound identifier (alias)
        "assay_id",         # Reference to assay from Stage 1
        "binding_metric",   # Metric type (IC50, Ki, Kd, EC50, etc.)
        "value",            # Numeric value
        "unit",             # Measurement units
        "reasoning"         # XML paragraph ID where the data was found
    ]

    STAGE2_HEADER: str = "\t".join(STAGE2_COLUMNS)
    STAGE2_COLUMN_COUNT: int = len(STAGE2_COLUMNS)

    STAGE2_PATTERN: Tuple[str, ...] = tuple(
        col.lower().replace('_', '').replace('-', '')
        for col in STAGE2_COLUMNS
    )

    # ============================================================================
    # STAGE 3: Compound mapping
    # ============================================================================
    STAGE3_COLUMNS: List[str] = [
        "compound",                    # Compound identifier (alias)
        "compound_IUPAC_identifier",   # IUPAC name or TIF filename
        "reasoning",                   # Rationale
        "chemical_id"                  # Chemical ID from chemistry tags
    ]

    STAGE3_HEADER: str = "\t".join(STAGE3_COLUMNS)
    STAGE3_COLUMN_COUNT: int = len(STAGE3_COLUMNS)

    STAGE3_PATTERN: Tuple[str, ...] = tuple(
        col.lower().replace('_', '').replace('-', '')
        for col in STAGE3_COLUMNS
    )

    # ============================================================================
    # All patterns for header filtering
    # ============================================================================
    ALL_HEADER_PATTERNS: List[Tuple[str, ...]] = [
        STAGE1_PATTERN,
        STAGE2_PATTERN,
        STAGE3_PATTERN,
    ]

    @classmethod
    def get_stage_info(cls, stage_num: int) -> dict:
        """
        Get Stage format information.

        Args:
            stage_num: Stage number (1, 2, or 3)

        Returns:
            dict with keys: columns, header, count, pattern
        """
        if stage_num == 1:
            return {
                'columns': cls.STAGE1_COLUMNS,
                'header': cls.STAGE1_HEADER,
                'count': cls.STAGE1_COLUMN_COUNT,
                'pattern': cls.STAGE1_PATTERN
            }
        elif stage_num == 2:
            return {
                'columns': cls.STAGE2_COLUMNS,
                'header': cls.STAGE2_HEADER,
                'count': cls.STAGE2_COLUMN_COUNT,
                'pattern': cls.STAGE2_PATTERN
            }
        elif stage_num == 3:
            return {
                'columns': cls.STAGE3_COLUMNS,
                'header': cls.STAGE3_HEADER,
                'count': cls.STAGE3_COLUMN_COUNT,
                'pattern': cls.STAGE3_PATTERN
            }
        else:
            raise ValueError(f"Invalid stage number: {stage_num}. Must be 1, 2, or 3.")


# Backward compatibility - export constants directly
STAGE1_COLUMNS = StageFormats.STAGE1_COLUMNS
STAGE1_HEADER = StageFormats.STAGE1_HEADER
STAGE1_COLUMN_COUNT = StageFormats.STAGE1_COLUMN_COUNT

STAGE2_COLUMNS = StageFormats.STAGE2_COLUMNS
STAGE2_HEADER = StageFormats.STAGE2_HEADER
STAGE2_COLUMN_COUNT = StageFormats.STAGE2_COLUMN_COUNT

STAGE3_COLUMNS = StageFormats.STAGE3_COLUMNS
STAGE3_HEADER = StageFormats.STAGE3_HEADER
STAGE3_COLUMN_COUNT = StageFormats.STAGE3_COLUMN_COUNT

ALL_HEADER_PATTERNS = StageFormats.ALL_HEADER_PATTERNS
