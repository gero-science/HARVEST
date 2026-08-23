"""Verification and hallucination detection for LLM extraction results.

This package provides:

- ``hallucination_detector`` — Stage 2/3 TSV vs patent XML checks
- ``annotate`` — batch annotation that writes per-patent ``_hallu.json`` sidecars

Standalone scripts (``check_tsv_*``, ``compare_with_mapping``, etc.) live in
this directory but are not imported by default.
"""

from .annotate import annotate_all, patent_stats_to_hallu_json

__all__ = ["annotate_all", "patent_stats_to_hallu_json"]
