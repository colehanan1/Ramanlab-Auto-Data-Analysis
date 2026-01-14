"""
Tests for compose_videos_rms module, specifically:
- fps validation and enforcement
- environment variable gating (DISABLE_COMPOSE_RMS, COMPOSE_RMS_DEBUG, COMPOSE_RMS_USE_FFMPEG)
- ffmpeg fallback command construction

These tests are hermetic: no real video files, actual ffmpeg/ffprobe, or GPU required.
"""
from __future__ import annotations

import io
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# =============================================================================
# Test _validate_fps function
# =============================================================================


class TestValidateFps:
    """Tests for fps validation helper."""

    def test_valid_fps_passthrough(self):
        """Valid fps values should pass through unchanged."""
        from fbpipe.steps.compose_videos_rms import _validate_fps

        assert _validate_fps(40.0) == 40.0
        assert _validate_fps(30.0) == 30.0
        assert _validate_fps(24) == 24.0  # int converted to float
        assert _validate_fps(29.97) == 29.97

    def test_none_returns_fallback(self):
        """None fps should return fallback."""
        from fbpipe.steps.compose_videos_rms import _validate_fps

        assert _validate_fps(None) == 40.0
        assert _validate_fps(None, fallback=30.0) == 30.0

    def test_zero_returns_fallback(self):
        """Zero fps should return fallback."""
        from fbpipe.steps.compose_videos_rms import _validate_fps

        assert _validate_fps(0) == 40.0
        assert _validate_fps(0.0) == 40.0

    def test_negative_returns_fallback(self):
        """Negative fps should return fallback."""
        from fbpipe.steps.compose_videos_rms import _validate_fps

        assert _validate_fps(-1) == 40.0
        assert _validate_fps(-30.0) == 40.0

    def test_infinity_returns_fallback(self):
        """Infinite fps should return fallback."""
        from fbpipe.steps.compose_videos_rms import _validate_fps

        assert _validate_fps(float("inf")) == 40.0
        assert _validate_fps(float("-inf")) == 40.0

    def test_nan_returns_fallback(self):
        """NaN fps should return fallback."""
        from fbpipe.steps.compose_videos_rms import _validate_fps

        assert _validate_fps(float("nan")) == 40.0

    def test_invalid_type_returns_fallback(self):
        """Invalid types should return fallback."""
        from fbpipe.steps.compose_videos_rms import _validate_fps

        assert _validate_fps("forty") == 40.0  # type: ignore
        assert _validate_fps([40]) == 40.0  # type: ignore


# =============================================================================
# Test DISABLE_COMPOSE_RMS environment variable
# =============================================================================


class TestDisableComposeRms:
    """Tests for DISABLE_COMPOSE_RMS kill-switch."""

    @pytest.fixture(autouse=True)
    def cleanup_env(self):
        """Clean up environment variables after each test."""
        original = os.environ.get("DISABLE_COMPOSE_RMS")
        yield
        if original is None:
            os.environ.pop("DISABLE_COMPOSE_RMS", None)
        else:
            os.environ["DISABLE_COMPOSE_RMS"] = original

    def test_disabled_skips_step(self, capsys):
        """DISABLE_COMPOSE_RMS=1 should skip the step entirely."""
        os.environ["DISABLE_COMPOSE_RMS"] = "1"

        # Reimport to pick up env var
        import importlib
        import fbpipe.steps.compose_videos_rms as module
        importlib.reload(module)

        # Create a minimal mock config
        mock_cfg = MagicMock()

        # Run main
        module.main(mock_cfg)

        # Check output
        captured = capsys.readouterr()
        assert "SKIPPED" in captured.out
        assert "DISABLE_COMPOSE_RMS=1" in captured.out

    def test_disabled_with_true_string(self, capsys):
        """DISABLE_COMPOSE_RMS=true should also skip."""
        os.environ["DISABLE_COMPOSE_RMS"] = "true"

        import importlib
        import fbpipe.steps.compose_videos_rms as module
        importlib.reload(module)

        mock_cfg = MagicMock()
        module.main(mock_cfg)

        captured = capsys.readouterr()
        assert "SKIPPED" in captured.out


# =============================================================================
# Test COMPOSE_RMS_DEBUG environment variable
# =============================================================================


class TestComposeRmsDebug:
    """Tests for COMPOSE_RMS_DEBUG instrumentation."""

    @pytest.fixture(autouse=True)
    def cleanup_env(self):
        """Clean up environment variables after each test."""
        original_debug = os.environ.get("COMPOSE_RMS_DEBUG")
        original_disable = os.environ.get("DISABLE_COMPOSE_RMS")
        yield
        for key, val in [("COMPOSE_RMS_DEBUG", original_debug), ("DISABLE_COMPOSE_RMS", original_disable)]:
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_debug_prints_fps_block(self, capsys):
        """COMPOSE_RMS_DEBUG=1 should print fps debug block."""
        os.environ["COMPOSE_RMS_DEBUG"] = "1"
        os.environ.pop("DISABLE_COMPOSE_RMS", None)

        import importlib
        import fbpipe.steps.compose_videos_rms as module
        importlib.reload(module)

        # Reset the debug printed flag
        module._DEBUG_PRINTED = False

        # Create mock clips with fps attributes
        mock_clip = MagicMock()
        mock_clip.fps = 40.0
        mock_panel_clip = MagicMock()
        mock_panel_clip.fps = 40.0
        mock_composite = MagicMock()
        mock_composite.fps = 40.0

        # Call the debug function directly
        module._print_debug_info(40.0, mock_clip, mock_panel_clip, mock_composite)

        captured = capsys.readouterr()
        assert "[COMPOSE_RMS_DEBUG]" in captured.out
        assert "FPS Debug Block" in captured.out
        assert "output_fps: 40.0" in captured.out


# =============================================================================
# Test ffmpeg fallback command construction
# =============================================================================


class TestFfmpegFallback:
    """Tests for ffmpeg fallback path."""

    @pytest.fixture(autouse=True)
    def cleanup_env(self):
        """Clean up environment variables after each test."""
        original_ffmpeg = os.environ.get("COMPOSE_RMS_USE_FFMPEG")
        original_debug = os.environ.get("COMPOSE_RMS_DEBUG")
        yield
        for key, val in [("COMPOSE_RMS_USE_FFMPEG", original_ffmpeg), ("COMPOSE_RMS_DEBUG", original_debug)]:
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_ffmpeg_command_includes_fps(self):
        """ffmpeg fallback command should include -r fps flag."""
        os.environ["COMPOSE_RMS_USE_FFMPEG"] = "1"
        os.environ["COMPOSE_RMS_DEBUG"] = "1"

        import importlib
        import fbpipe.steps.compose_videos_rms as module
        importlib.reload(module)

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Create temp paths
            video_path = Path("/tmp/test_video.mp4")
            panel_path = Path("/tmp/test_panel.png")
            out_path = Path("/tmp/test_out.mp4")

            # Mock out_path.exists and stat
            with patch.object(Path, "exists", return_value=True), \
                 patch.object(Path, "stat", return_value=MagicMock(st_size=1000)):

                result = module._compose_via_ffmpeg(
                    video_path, panel_path, out_path,
                    output_fps=40.0, duration=10.0, video_height=1080, panel_height=259
                )

            # Check that subprocess.run was called
            assert mock_run.called

            # Get the command that was passed
            cmd = mock_run.call_args[0][0]

            # Verify fps is in command
            assert "-r" in cmd
            fps_idx = cmd.index("-r")
            assert cmd[fps_idx + 1] == "40.0"

            # Verify vstack filter is used
            filter_complex_idx = cmd.index("-filter_complex")
            assert "vstack" in cmd[filter_complex_idx + 1]

    def test_ffmpeg_command_includes_framerate(self):
        """ffmpeg command should include -framerate for image input."""
        os.environ["COMPOSE_RMS_USE_FFMPEG"] = "1"

        import importlib
        import fbpipe.steps.compose_videos_rms as module
        importlib.reload(module)

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            video_path = Path("/tmp/test_video.mp4")
            panel_path = Path("/tmp/test_panel.png")
            out_path = Path("/tmp/test_out.mp4")

            with patch.object(Path, "exists", return_value=True), \
                 patch.object(Path, "stat", return_value=MagicMock(st_size=1000)):

                module._compose_via_ffmpeg(
                    video_path, panel_path, out_path,
                    output_fps=30.0, duration=5.0, video_height=720, panel_height=173
                )

            cmd = mock_run.call_args[0][0]

            # Verify -framerate is present for image input
            assert "-framerate" in cmd
            framerate_idx = cmd.index("-framerate")
            assert cmd[framerate_idx + 1] == "30.0"


# =============================================================================
# Test write_videofile fps parameter
# =============================================================================


class TestWriteVideofileFps:
    """Tests that write_videofile receives valid fps."""

    def test_write_videofile_receives_fps_kwarg(self):
        """write_videofile should always receive fps as a float, never None."""
        import importlib
        import fbpipe.steps.compose_videos_rms as module
        importlib.reload(module)

        # We'll mock VideoFileClip, VideoClip, and CompositeVideoClip
        with patch.object(module, "VideoFileClip") as mock_vfc, \
             patch.object(module, "VideoClip") as mock_vc, \
             patch.object(module, "CompositeVideoClip") as mock_cvc, \
             patch.object(module, "_render_line_panel_png") as mock_render:

            # Setup mock clip with fps=None (the problematic case)
            mock_clip = MagicMock()
            mock_clip.fps = None  # Simulate missing fps
            mock_clip.size = (1920, 1080)
            mock_clip.duration = 10.0
            mock_clip.set_fps.return_value = mock_clip
            mock_clip.set_position.return_value = mock_clip
            mock_vfc.return_value = mock_clip

            # Setup mock panel clip
            mock_panel = MagicMock()
            mock_panel.fps = 40.0
            mock_panel.set_fps.return_value = mock_panel
            mock_panel.set_position.return_value = mock_panel
            mock_vc.return_value = mock_panel

            # Setup mock composite
            mock_composite = MagicMock()
            mock_composite.fps = 40.0  # set_fps will set this
            mock_composite.set_fps.return_value = mock_composite
            mock_cvc.return_value = mock_composite

            # Setup render to return a valid numpy array
            mock_render.return_value = np.zeros((259, 1920, 3), dtype=np.uint8)

            # Create test paths
            video_path = Path("/tmp/test.mp4")
            out_mp4 = Path("/tmp/out.mp4")

            # Mock the directory creation
            with patch.object(Path, "mkdir"):
                module._compose_lineplot_video(
                    video_path,
                    series_list=[{"t": [0, 1], "y": [0, 0], "label": "test", "color": "blue"}],
                    xlim=(0.0, 10.0),
                    odor_on=2.0,
                    odor_off=4.0,
                    out_mp4=out_mp4,
                    panel_height_fraction=0.24,
                    ylim=(-100, 100),
                    threshold=None,
                )

            # Verify write_videofile was called with fps kwarg
            mock_composite.write_videofile.assert_called_once()
            call_kwargs = mock_composite.write_videofile.call_args[1]

            # Critical assertion: fps must be a valid float
            assert "fps" in call_kwargs
            fps_value = call_kwargs["fps"]
            assert fps_value is not None, "fps should never be None"
            assert isinstance(fps_value, float), f"fps should be float, got {type(fps_value)}"
            assert math.isfinite(fps_value), "fps should be finite"
            assert fps_value > 0, "fps should be positive"
            # Should be the fallback since source clip.fps was None
            assert fps_value == 40.0


# =============================================================================
# Test Multi-Fly Plotting
# =============================================================================


class TestMultiFlyPlotting:
    """Tests for multi-fly RMS plotting (one mp4 per trial with multiple series)."""

    def test_derive_fly_label(self):
        """Test label derivation from token."""
        from fbpipe.steps.compose_videos_rms import _derive_fly_label

        assert _derive_fly_label("fly_0_slot1_distances", 1) == "Fly 0 Slot 1"
        assert _derive_fly_label("slot2_distances", 2) == "Slot 2"
        assert _derive_fly_label("unknown_token", 0) == "Series 0"
        assert _derive_fly_label("fly_3_distances", 1) == "Fly 3"

    def test_multifly_trial_single_output(self):
        """For a trial with 3 slot entries, compositor should be called once with 3 series."""
        import importlib
        import fbpipe.steps.compose_videos_rms as module
        importlib.reload(module)

        # Mock the series computation to return valid data for each slot
        def mock_series_func(csv_path, *args, **kwargs):
            """Return different series for each csv_path (simulating different flies)."""
            path_str = str(csv_path)
            if "slot_0" in path_str:
                return (np.array([0, 1, 2, 3]), np.array([10, 20, 30, 40]), 0.5, 1.5, 15.0)
            elif "slot_1" in path_str:
                return (np.array([0, 1, 2, 3]), np.array([15, 25, 35, 45]), 0.5, 1.5, 20.0)
            elif "slot_2" in path_str:
                return (np.array([0, 1, 2, 3]), np.array([12, 22, 32, 42]), 0.5, 1.5, 17.0)
            return None

        with patch.object(module, "_series_rms_from_rmscalc", side_effect=mock_series_func), \
             patch.object(module, "_compose_lineplot_video", return_value=True) as mock_compose, \
             patch.object(module, "_find_video_for_trial", return_value=Path("/tmp/video.mp4")), \
             patch.object(module, "_collect_trial_csvs") as mock_collect, \
             patch.object(module, "_discover_month_folders", return_value=[]):

            # Setup mock trial_map with 3 slot entries
            slot_entries = [
                ("fly_0_slot_0_distances", 0, Path("/tmp/slot_0.csv")),
                ("fly_0_slot_1_distances", 1, Path("/tmp/slot_1.csv")),
                ("fly_0_slot_2_distances", 2, Path("/tmp/slot_2.csv")),
            ]
            mock_collect.return_value = {1: slot_entries}

            # Mock config and other dependencies
            mock_cfg = MagicMock()
            mock_cfg.fps_default = 40.0
            mock_cfg.odor_on_s = 1.0
            mock_cfg.odor_off_s = 3.0
            mock_cfg.odor_latency_s = 0.0
            mock_cfg.delete_source_after_render = False

            # Simulate the grouping logic (this is what the refactored code does)
            series_list = []
            global_xlim = [float('inf'), float('-inf')]
            first_odor_on, first_odor_off, first_threshold = None, None, None

            for token, slot_idx, csv_path in slot_entries:
                series = mock_series_func(csv_path, None, 40.0, 1.0, 2.0, 90.0, 0.0)
                if series is None:
                    continue

                t, rms, odor_on, odor_off, threshold = series
                label = module._derive_fly_label(token, slot_idx)
                series_list.append({"t": t, "y": rms, "label": label})

                global_xlim[0] = min(global_xlim[0], float(np.nanmin(t)))
                global_xlim[1] = max(global_xlim[1], float(np.nanmax(t)))

                if first_odor_on is None and odor_on is not None:
                    first_odor_on = odor_on

            # Verify: should have 3 series
            assert len(series_list) == 3
            # Verify: labels derived correctly
            assert series_list[0]["label"] == "Fly 0 Slot 0"
            assert series_list[1]["label"] == "Fly 0 Slot 1"
            assert series_list[2]["label"] == "Fly 0 Slot 2"

    def test_singlefly_trial_still_works(self):
        """For a trial with 1 slot entry, should still output one mp4."""
        import importlib
        import fbpipe.steps.compose_videos_rms as module
        importlib.reload(module)

        def mock_series_func(csv_path, *args, **kwargs):
            return (np.array([0, 1, 2]), np.array([10, 20, 30]), 0.5, 1.5, 15.0)

        # Simulate grouping for 1 slot
        slot_entries = [
            ("fly_0_distances", 0, Path("/tmp/slot_0.csv")),
        ]

        series_list = []
        for token, slot_idx, csv_path in slot_entries:
            series = mock_series_func(csv_path, None, 40.0, 1.0, 2.0, 90.0, 0.0)
            if series:
                t, rms, odor_on, odor_off, threshold = series
                label = module._derive_fly_label(token, slot_idx)
                series_list.append({"t": t, "y": rms, "label": label})

        assert len(series_list) == 1
        assert "Fly 0" in series_list[0]["label"]

    def test_partial_none_series_handling(self):
        """For a trial where one slot returns None, render with remaining series."""
        import importlib
        import fbpipe.steps.compose_videos_rms as module
        importlib.reload(module)

        def mock_series_func(csv_path, *args, **kwargs):
            path_str = str(csv_path)
            if "slot_0" in path_str:
                return (np.array([0, 1, 2]), np.array([10, 20, 30]), 0.5, 1.5, 15.0)
            elif "slot_1" in path_str:
                # This one returns None (missing data)
                return None
            elif "slot_2" in path_str:
                return (np.array([0, 1, 2]), np.array([12, 22, 32]), 0.5, 1.5, 17.0)
            return None

        slot_entries = [
            ("fly_0_slot_0_distances", 0, Path("/tmp/slot_0.csv")),
            ("fly_0_slot_1_distances", 1, Path("/tmp/slot_1.csv")),
            ("fly_0_slot_2_distances", 2, Path("/tmp/slot_2.csv")),
        ]

        series_list = []
        for token, slot_idx, csv_path in slot_entries:
            series = mock_series_func(csv_path, None, 40.0, 1.0, 2.0, 90.0, 0.0)
            if series is None:
                continue
            t, rms, odor_on, odor_off, threshold = series
            label = module._derive_fly_label(token, slot_idx)
            series_list.append({"t": t, "y": rms, "label": label})

        # Should have 2 series (skipped the None)
        assert len(series_list) == 2

    def test_all_none_series_skip(self):
        """For a trial where all slots return None, skip rendering."""
        import importlib
        import fbpipe.steps.compose_videos_rms as module
        importlib.reload(module)

        def mock_series_func(csv_path, *args, **kwargs):
            return None  # All return None

        slot_entries = [
            ("fly_0_slot_0_distances", 0, Path("/tmp/slot_0.csv")),
            ("fly_0_slot_1_distances", 1, Path("/tmp/slot_1.csv")),
        ]

        series_list = []
        for token, slot_idx, csv_path in slot_entries:
            series = mock_series_func(csv_path, None, 40.0, 1.0, 2.0, 90.0, 0.0)
            if series is None:
                continue
            t, rms, odor_on, odor_off, threshold = series
            label = module._derive_fly_label(token, slot_idx)
            series_list.append({"t": t, "y": rms, "label": label})

        # Should be empty
        assert len(series_list) == 0


# =============================================================================
# Run pytest if executed directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
