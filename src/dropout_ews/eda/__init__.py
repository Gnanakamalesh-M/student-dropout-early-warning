"""Exploratory data analysis.

Analysis logic lives here as importable, tested functions rather than inside
notebook cells. Notebooks under ``notebooks/`` are thin drivers that call these
and render the output.

The reason is practical: logic written in cells cannot be unit-tested, tends to
drift from the pipeline it is supposed to describe, and quietly rots into an
unreproducible artifact. Findings that shape the feature allowlist need to be
reproducible on demand.
"""
