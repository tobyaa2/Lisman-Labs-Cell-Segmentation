"""cell_counter — cell counting & X-gal stain classification from a matched
filtered/unfiltered microscopy image pair.

See ``cell_counter_plan.md`` for the design rationale.
"""
from .config import Config, load_config, __version__
from .models import CellRecord
from .pipeline import Result, run_from_paths, run_pipeline, write_outputs

__all__ = [
    "Config", "load_config", "CellRecord", "Result",
    "run_pipeline", "run_from_paths", "write_outputs", "__version__",
]
