"""Optional Parquet post-processing after BindingDB export.

Use ``python -m final_postprocessing`` to run enrichment + cleaning filters
in one command. ``add_final_structure`` is intentionally separate.
"""

from .run import STEPS, run_final_postprocessing

__all__ = ["STEPS", "run_final_postprocessing"]
