"""Contract tests for docs/Detailed_Pipeline_Walkthrough.ipynb.

These lightweight tests verify that the notebook has the required structure
without actually executing any cells or needing heavy compute resources.
"""

from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "docs" / "Detailed_Pipeline_Walkthrough.ipynb"


@pytest.fixture(scope="module")
def notebook():
    """Load the walkthrough notebook once for all tests."""
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    return nbformat.read(str(NOTEBOOK_PATH), as_version=4)


def _markdown_sources(nb) -> list[str]:
    """Return the combined source text of all markdown cells."""
    return [cell.source for cell in nb.cells if cell.cell_type == "markdown"]


def _code_sources(nb) -> list[str]:
    """Return the combined source text of all code cells."""
    return [cell.source for cell in nb.cells if cell.cell_type == "code"]


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestRequiredSections:
    """Verify that all required section headings exist in the notebook."""

    REQUIRED_HEADINGS = [
        "Parameters",
        "Environment Validation",
        "Config",
        "Dry-Run",
        "Run the Pipeline",
        "Output Summary",
        "Backups",
        "Troubleshooting",
    ]

    def test_all_required_headings_present(self, notebook):
        all_md = "\n".join(_markdown_sources(notebook))
        for heading in self.REQUIRED_HEADINGS:
            assert heading.lower() in all_md.lower(), (
                f"Required section heading '{heading}' not found in notebook markdown"
            )

    def test_has_both_cell_types(self, notebook):
        md_count = sum(1 for c in notebook.cells if c.cell_type == "markdown")
        code_count = sum(1 for c in notebook.cells if c.cell_type == "code")
        assert md_count >= 5, f"Expected >= 5 markdown cells, got {md_count}"
        assert code_count >= 5, f"Expected >= 5 code cells, got {code_count}"


class TestSafetyWarnings:
    """Verify that the notebook contains safety warnings."""

    def test_safety_section_exists(self, notebook):
        all_md = "\n".join(_markdown_sources(notebook))
        assert "safety" in all_md.lower(), "Notebook must contain a safety section"

    def test_delete_source_warning(self, notebook):
        all_sources = "\n".join(
            c.source for c in notebook.cells
        )
        assert "delete_source_after_render" in all_sources, (
            "Notebook must warn about delete_source_after_render"
        )

    def test_dry_run_default(self, notebook):
        code = "\n".join(_code_sources(notebook))
        assert "DRY_RUN_ONLY = True" in code, (
            "DRY_RUN_ONLY must default to True for safety"
        )

    def test_backups_off_by_default(self, notebook):
        code = "\n".join(_code_sources(notebook))
        assert "RUN_BACKUPS = False" in code, (
            "RUN_BACKUPS must default to False"
        )


class TestPipelineReference:
    """Verify that the notebook references the actual pipeline execution."""

    def test_references_make_run(self, notebook):
        all_sources = "\n".join(c.source for c in notebook.cells)
        assert "make run" in all_sources.lower(), (
            "Notebook must reference 'make run' as the pipeline command"
        )

    def test_references_run_workflows(self, notebook):
        code = "\n".join(_code_sources(notebook))
        assert "run_workflows" in code, (
            "Notebook must reference scripts/pipeline/run_workflows.py"
        )

    def test_references_config_yaml(self, notebook):
        code = "\n".join(_code_sources(notebook))
        assert "config.yaml" in code or "CONFIG_PATH" in code, (
            "Notebook must reference config.yaml"
        )


class TestShellRunnerHelper:
    """Verify that the shell runner helper handles failures properly."""

    def test_has_run_shell_command(self, notebook):
        code = "\n".join(_code_sources(notebook))
        assert "def run_shell_command" in code, (
            "Notebook must define run_shell_command helper"
        )

    def test_runner_captures_exit_code(self, notebook):
        code = "\n".join(_code_sources(notebook))
        assert "returncode" in code, (
            "Shell runner must check return code"
        )

    def test_runner_shows_tail_on_failure(self, notebook):
        code = "\n".join(_code_sources(notebook))
        # The runner should show last N lines on failure
        assert "50" in code and "last" in code.lower(), (
            "Shell runner must show last 50 lines on failure"
        )
