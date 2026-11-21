"""PER analysis of the sensitivity (exp1) and IAA odor-pair (exp2/exp3) experiments.

- ``config``: paths, threshold, windows, cohorts, palettes, exclusion switches.
- ``io``: ``load_trials()`` -> the tidy one-row-per-scored-trial table (the contract).
- ``qc``: per-fly QC flags and report; ``apply_exclusions``.
- ``stats`` / ``plots``: analyses on the tidy table.

Submodules are imported explicitly (``from per_analysis import io``) so that importing
the package does not read the data.
"""

__all__ = ["config", "io", "qc"]
