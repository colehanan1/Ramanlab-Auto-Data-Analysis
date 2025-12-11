#!/usr/bin/env python3
"""
Weekly Olfactory Training Trace Analysis Pipeline

This script processes behavioral training data from Drosophila olfactory experiments
and generates publication-quality weekly aggregation plots.

Author: Ramanlab Auto-Data-Analysis
Date: 2024-12-11
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weekly_training_traces.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def parse_date_from_fly(fly_str: str) -> Optional[datetime]:
    """
    Extract date from fly identifier string.

    Args:
        fly_str: Fly identifier (e.g., 'november_19_batch_1', 'december_03_batch_1_rig_2')

    Returns:
        datetime object or None if parsing fails
    """
    match = re.match(r'([a-z]+)_(\d+)_', fly_str)
    if match:
        month_str = match.group(1)
        day = int(match.group(2))

        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        month = month_map.get(month_str)
        if month:
            return datetime(2024, month, day)
    return None


def assign_weekly_groups(dates: pd.Series, days_per_group: int = 7) -> pd.Series:
    """
    Assign weekly group numbers using fixed 7-day intervals.

    Args:
        dates: Series of datetime objects
        days_per_group: Number of days per group (default: 7)

    Returns:
        Series of group numbers (1, 2, 3, ...)
    """
    if len(dates.dropna()) == 0:
        return pd.Series(dtype=int)

    # Find the minimum date to use as reference point
    min_date = dates.min()

    # Calculate group number based on days elapsed from min_date
    # Group 1: days 0-6, Group 2: days 7-13, Group 3: days 14-20, etc.
    def get_group(date):
        if pd.isna(date):
            return None
        days_elapsed = (date - min_date).days
        group_number = (days_elapsed // days_per_group) + 1
        return group_number

    return dates.apply(get_group)



def load_and_filter_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and filter to experimental datasets only.

    Args:
        csv_path: Path to the training CSV file

    Returns:
        Filtered DataFrame containing only experimental data with date and week columns
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} total rows")

    # Filter to experimental datasets (those starting with 'opto')
    exp_df = df[df['dataset'].str.startswith('opto')].copy()
    logger.info(f"Filtered to {len(exp_df)} experimental rows")

    # Exclude non-reactive flies
    non_reactive_count = (exp_df['non_reactive_flag'] == 1.0).sum()
    exp_df = exp_df[exp_df['non_reactive_flag'] == 0.0].copy()
    logger.info(f"Excluded {non_reactive_count} non-reactive trials ({100*non_reactive_count/(len(exp_df)+non_reactive_count):.1f}%)")
    logger.info(f"Retained {len(exp_df)} reactive trials")

    # Log control datasets excluded
    control_datasets = df[df['dataset'].str.contains('control')]['dataset'].unique()
    logger.info(f"Excluded control datasets: {', '.join(control_datasets)}")

    # Log experimental datasets included
    exp_datasets = exp_df['dataset'].unique()
    logger.info(f"Included experimental datasets: {', '.join(sorted(exp_datasets))}")

    # Extract dates
    exp_df['date'] = exp_df['fly'].apply(parse_date_from_fly)

    # Remove rows where date parsing failed
    rows_before = len(exp_df)
    exp_df = exp_df.dropna(subset=['date'])
    rows_after = len(exp_df)
    if rows_before != rows_after:
        logger.warning(f"Dropped {rows_before - rows_after} rows due to date parsing failure")

    # Assign weekly groups using exact 7-day intervals
    exp_df['week'] = assign_weekly_groups(exp_df['date'], days_per_group=7)
    logger.info(f"Assigned {exp_df['week'].nunique()} weekly groups using exact 7-day intervals")

    return exp_df


def extract_envelope_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract envelope response column names (dir_val_X).

    Args:
        df: DataFrame containing envelope data

    Returns:
        List of column names for envelope time-series
    """
    dir_cols = [col for col in df.columns if col.startswith('dir_val_')]
    logger.info(f"Identified {len(dir_cols)} envelope timepoint columns")
    return dir_cols


def aggregate_trial_by_week(
    df: pd.DataFrame,
    week: int,
    trial_number: int,
    dir_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Aggregate envelope traces for a specific trial and week across all flies.

    Args:
        df: DataFrame containing experimental data
        week: Week number to filter
        trial_number: Training trial number (1, 2, 3, 4, 6, or 8)
        dir_cols: List of envelope column names

    Returns:
        Tuple of (mean_trace, sem_trace, n_flies)
        - mean_trace: Average envelope response across flies
        - sem_trace: Standard error of the mean
        - n_flies: Number of flies contributing to this average
    """
    trial_label = f'training_{trial_number}'

    # Filter to specific week and trial
    filtered = df[(df['week'] == week) & (df['trial_label'] == trial_label)]
    n_flies = len(filtered)

    if n_flies == 0:
        logger.warning(f"Week {week}, Trial {trial_number}: No data found")
        return np.full(len(dir_cols), np.nan), np.full(len(dir_cols), np.nan), 0

    # Extract envelope traces
    traces = filtered[dir_cols].values  # Shape: (n_flies, n_timepoints)

    # Compute mean and SEM across flies (axis=0)
    mean_trace = np.nanmean(traces, axis=0)
    sem_trace = np.nanstd(traces, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(traces), axis=0))

    logger.debug(f"Week {week}, Trial {trial_number}: Aggregated {n_flies} flies")

    return mean_trace, sem_trace, n_flies


def aggregate_grand_average(
    df: pd.DataFrame,
    week: int,
    trial_numbers: List[int],
    dir_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute grand average across all specified trials for a week.

    Args:
        df: DataFrame containing experimental data
        week: Week number to filter
        trial_numbers: List of trial numbers to include (e.g., [1,2,3,4,6,8])
        dir_cols: List of envelope column names

    Returns:
        Tuple of (mean_trace, sem_trace, n_observations)
    """
    # Filter to specified week and trials
    trial_labels = [f'training_{n}' for n in trial_numbers]
    filtered = df[(df['week'] == week) & (df['trial_label'].isin(trial_labels))]

    n_observations = len(filtered)

    if n_observations == 0:
        logger.warning(f"Week {week}: No data for grand average")
        return np.full(len(dir_cols), np.nan), np.full(len(dir_cols), np.nan), 0

    # Extract all envelope traces (across flies and trials)
    traces = filtered[dir_cols].values

    # Compute mean and SEM
    mean_trace = np.nanmean(traces, axis=0)
    sem_trace = np.nanstd(traces, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(traces), axis=0))

    logger.debug(f"Week {week}: Grand average from {n_observations} total observations")

    return mean_trace, sem_trace, n_observations


def create_weekly_figure(
    df: pd.DataFrame,
    week: int,
    dir_cols: List[str],
    fps: float = 40.0,
    output_dir: Path = Path('/home/ramanlab/Documents/cole/Results/Opto/Weekly-Training-Envelopes')
) -> str:
    """
    Create vertical figure showing individual trial averages and grand average for one week.

    Args:
        df: DataFrame containing experimental data
        week: Week number to visualize
        dir_cols: List of envelope column names
        fps: Frames per second (for time axis conversion)
        output_dir: Directory to save output figure

    Returns:
        Path to saved figure file
    """
    trial_numbers = [1, 2, 3, 4, 6, 8]

    # Get date range for this week
    week_df = df[df['week'] == week]
    dates = week_df['date'].unique()
    start_date = min(dates).strftime('%Y-%m-%d')
    end_date = max(dates).strftime('%Y-%m-%d')

    # Calculate max number of flies across all trials (for title)
    max_flies = 0
    for trial_num in trial_numbers:
        trial_label = f'training_{trial_num}'
        filtered = df[(df['week'] == week) & (df['trial_label'] == trial_label)]
        n_flies = len(filtered)
        max_flies = max(max_flies, n_flies)

    logger.info(f"Creating figure for Group {week} ({start_date} to {end_date}, max {max_flies} flies)")

    # Create time axis (in seconds) and limit to 0-90 seconds
    time_axis = np.arange(len(dir_cols)) / fps
    time_mask = time_axis <= 90
    time_axis_plot = time_axis[time_mask]

    # Set up figure with vertical layout (7 rows, 1 column)
    # 6 individual trials + 1 grand average
    fig, axes = plt.subplots(7, 1, figsize=(12, 18))
    fig.suptitle(f'Weekly Training Traces - Group {week} ({start_date} to {end_date})\n'
                 f'Max {max_flies} flies', fontsize=16, fontweight='bold')

    # Color palette (colorblind-friendly)
    colors = sns.color_palette('colorblind', 7)

    # Plot individual trials vertically
    for idx, trial_num in enumerate(trial_numbers):
        ax = axes[idx]

        # Aggregate data for this trial
        mean_trace, sem_trace, n_flies = aggregate_trial_by_week(df, week, trial_num, dir_cols)

        if n_flies > 0:
            # Apply time mask to limit to 0-90 seconds
            mean_trace_plot = mean_trace[time_mask]
            sem_trace_plot = sem_trace[time_mask]

            # Add red background for odor presentation period
            if trial_num in [1, 2, 3]:
                # Trials 1, 2, 3: odor on from 32-62 seconds
                ax.axvspan(32, 62, color='red', alpha=0.15, zorder=0)
            else:  # trial_num in [4, 6, 8]
                # Trials 4, 6, 8: odor on from 32-67 seconds
                ax.axvspan(32, 67, color='red', alpha=0.15, zorder=0)

            # Plot mean line
            ax.plot(time_axis_plot, mean_trace_plot, color=colors[idx], linewidth=2, alpha=0.8,
                   label=f'n={n_flies}', zorder=2)

            # Plot SEM shaded region
            ax.fill_between(time_axis_plot, mean_trace_plot - sem_trace_plot,
                           mean_trace_plot + sem_trace_plot,
                           color=colors[idx], alpha=0.3, zorder=1)

            ax.set_title(f'Training Trial {trial_num}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Envelope Response', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, zorder=0)
            ax.set_xlim(0, 90)

            # Only show x-label on bottom plot
            if idx < len(trial_numbers) - 1:
                ax.set_xticklabels([])
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, color='gray')
            ax.set_title(f'Training Trial {trial_num}', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 90)

    # Plot grand average in last subplot
    ax_grand = axes[6]
    mean_trace, sem_trace, n_obs = aggregate_grand_average(df, week, trial_numbers, dir_cols)

    if n_obs > 0:
        # Apply time mask
        mean_trace_plot = mean_trace[time_mask]
        sem_trace_plot = sem_trace[time_mask]

        # Add red background for odor presentation (32-67s for grand average)
        ax_grand.axvspan(32, 67, color='red', alpha=0.15, zorder=0)

        ax_grand.plot(time_axis_plot, mean_trace_plot, color='black', linewidth=2.5, alpha=0.8,
                     label=f'n={n_obs} observations', zorder=2)
        ax_grand.fill_between(time_axis_plot, mean_trace_plot - sem_trace_plot,
                             mean_trace_plot + sem_trace_plot,
                             color='black', alpha=0.2, zorder=1)
        ax_grand.set_title('Grand Average (All Trials)', fontsize=12, fontweight='bold')
        ax_grand.set_xlabel('Time (s)', fontsize=10)
        ax_grand.set_ylabel('Envelope Response', fontsize=10)
        ax_grand.legend(loc='upper right', fontsize=8)
        ax_grand.grid(True, alpha=0.3, zorder=0)
        ax_grand.set_xlim(0, 90)

    # Add x-label to second-to-last plot (Trial 8)
    axes[5].set_xlabel('Time (s)', fontsize=10)

    # Use tight_layout with rect to leave space at top for suptitle
    # rect = (left, bottom, right, top) as fractions of figure
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Leave 4% at top for title

    # Save figure
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'weekly_training_traces_group{week}_{start_date}_to_{end_date}.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure: {filepath}")
        plt.close()
        return str(filepath)
    else:
        plt.show()
        return None


def create_all_weeks_combined_plot(
    df: pd.DataFrame,
    dir_cols: List[str],
    fps: float = 40.0,
    output_dir: Path = Path('/home/ramanlab/Documents/cole/Results/Opto/Weekly-Training-Envelopes')
) -> str:
    """
    Create a single plot showing grand averages for all weeks overlaid on one figure.

    Args:
        df: DataFrame containing experimental data
        dir_cols: List of envelope column names
        fps: Frames per second (for time axis conversion)
        output_dir: Directory to save output figure

    Returns:
        Path to saved figure file
    """
    trial_numbers = [1, 2, 3, 4, 6, 8]
    weeks = sorted(df['week'].unique())

    logger.info(f"Creating combined plot for all {len(weeks)} groups")

    # Create time axis (in seconds) and limit to 0-90 seconds
    time_axis = np.arange(len(dir_cols)) / fps
    time_mask = time_axis <= 90
    time_axis_plot = time_axis[time_mask]

    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Create color spectrum from green to blue
    # Using a colormap that goes through green -> cyan -> blue
    cmap = plt.cm.get_cmap('viridis', len(weeks))
    colors = [cmap(i) for i in range(len(weeks))]

    # Plot each week's grand average
    for idx, week in enumerate(weeks):
        # Get date range for this week
        week_df = df[df['week'] == week]
        dates = week_df['date'].unique()
        start_date = min(dates).strftime('%m/%d')
        end_date = max(dates).strftime('%m/%d')
        n_unique_flies = week_df['fly'].nunique()

        # Compute grand average for this week
        mean_trace, sem_trace, n_obs = aggregate_grand_average(df, week, trial_numbers, dir_cols)

        if n_obs > 0:
            # Apply time mask
            mean_trace_plot = mean_trace[time_mask]
            sem_trace_plot = sem_trace[time_mask]

            # Plot mean line with label showing week info
            label = f'Group {int(week)} ({start_date}-{end_date}, n={n_unique_flies} flies)'
            ax.plot(time_axis_plot, mean_trace_plot, color=colors[idx],
                   linewidth=2.5, alpha=0.85, label=label, zorder=2)

            # Plot SEM shaded region (lighter)
            ax.fill_between(time_axis_plot, mean_trace_plot - sem_trace_plot,
                           mean_trace_plot + sem_trace_plot,
                           color=colors[idx], alpha=0.15, zorder=1)

    # Add red background for odor presentation (32-67s for grand average)
    ax.axvspan(32, 67, color='red', alpha=0.1, zorder=0, label='Odor ON')

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Envelope Response', fontsize=14, fontweight='bold')
    ax.set_title('Grand Average Training Traces - All Groups Combined',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 90)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)

    # Add legend outside plot area on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
             fontsize=10, framealpha=0.95, edgecolor='black')

    plt.tight_layout()

    # Save figure
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = 'weekly_training_traces_all_groups_combined.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined figure: {filepath}")
        plt.close()
        return str(filepath)
    else:
        plt.show()
        return None


def extract_odor_period_response(
    trace: np.ndarray,
    fps: float = 40.0,
    odor_start_sec: float = 32.0,
    odor_end_sec: float = 67.0
) -> Tuple[float, float]:
    """
    Extract mean and peak response during odor presentation period.

    Args:
        trace: Envelope trace array
        fps: Frames per second
        odor_start_sec: Start of odor period in seconds
        odor_end_sec: End of odor period in seconds

    Returns:
        Tuple of (mean_response, peak_response) during odor period
    """
    start_idx = int(odor_start_sec * fps)
    end_idx = int(odor_end_sec * fps)

    if end_idx > len(trace):
        end_idx = len(trace)

    odor_period = trace[start_idx:end_idx]

    if len(odor_period) == 0:
        return np.nan, np.nan

    mean_resp = np.nanmean(odor_period)
    peak_resp = np.nanmax(odor_period)

    return mean_resp, peak_resp


def perform_statistical_analysis(
    df: pd.DataFrame,
    dir_cols: List[str],
    output_dir: Path,
    fps: float = 40.0
) -> str:
    """
    Perform statistical analysis to detect changes in response amplitude across weeks.

    Args:
        df: DataFrame containing experimental data
        dir_cols: List of envelope column names
        output_dir: Directory to save output files
        fps: Frames per second

    Returns:
        Path to saved statistical report
    """
    trial_numbers = [1, 2, 3, 4, 6, 8]
    weeks = sorted(df['week'].unique())

    logger.info("=" * 80)
    logger.info("STATISTICAL ANALYSIS: Response Amplitude Across Weeks")
    logger.info("=" * 80)

    # Extract odor period responses for each week
    week_responses = {}

    for week in weeks:
        responses = []
        trial_labels = [f'training_{n}' for n in trial_numbers]
        filtered = df[(df['week'] == week) & (df['trial_label'].isin(trial_labels))]

        for _, row in filtered.iterrows():
            trace = row[dir_cols].values
            mean_resp, peak_resp = extract_odor_period_response(trace, fps)
            if not np.isnan(mean_resp):
                responses.append({
                    'week': week,
                    'mean_response': mean_resp,
                    'peak_response': peak_resp,
                    'fly': row['fly'],
                    'trial': row['trial_label']
                })

        week_responses[week] = responses
        logger.info(f"Week {week}: {len(responses)} observations")

    # Create DataFrame for analysis
    all_responses = []
    for week, responses in week_responses.items():
        all_responses.extend(responses)

    resp_df = pd.DataFrame(all_responses)

    if len(resp_df) == 0:
        logger.warning("No response data found for statistical analysis")
        return None

    # Statistical tests
    report_path = output_dir / 'statistical_analysis_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL ANALYSIS: Response Amplitude Changes Across Weeks\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Descriptive statistics by week
        f.write("DESCRIPTIVE STATISTICS BY WEEK\n")
        f.write("-" * 80 + "\n")
        for week in weeks:
            week_data = resp_df[resp_df['week'] == week]
            if len(week_data) > 0:
                f.write(f"\nWeek {int(week)}:\n")
                f.write(f"  n observations: {len(week_data)}\n")
                f.write(f"  Mean response: {week_data['mean_response'].mean():.4f} ± {week_data['mean_response'].std():.4f}\n")
                f.write(f"  Peak response: {week_data['peak_response'].mean():.4f} ± {week_data['peak_response'].std():.4f}\n")
                f.write(f"  Unique flies: {week_data['fly'].nunique()}\n")

        # ANOVA test (if 3+ weeks)
        if len(weeks) >= 3:
            f.write("\n\nANOVA TEST: Comparing Mean Response Across All Weeks\n")
            f.write("-" * 80 + "\n")

            groups = [resp_df[resp_df['week'] == w]['mean_response'].values for w in weeks]
            # Remove empty groups
            groups = [g for g in groups if len(g) > 0]

            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                f.write(f"F-statistic: {f_stat:.4f}\n")
                f.write(f"p-value: {p_value:.6f}\n")

                if p_value < 0.05:
                    f.write(f"Result: SIGNIFICANT difference detected (p < 0.05)\n")
                else:
                    f.write(f"Result: No significant difference (p >= 0.05)\n")

        # Pairwise comparisons with Week 1
        if len(weeks) > 1:
            week1_data = resp_df[resp_df['week'] == weeks[0]]['mean_response'].values

            f.write("\n\nPAIRWISE T-TESTS: Comparing Each Week to Week 1\n")
            f.write("-" * 80 + "\n")
            f.write(f"Week 1 baseline: n={len(week1_data)}, mean={np.mean(week1_data):.4f}\n\n")

            for week in weeks[1:]:
                week_data = resp_df[resp_df['week'] == week]['mean_response'].values

                if len(week_data) > 0:
                    t_stat, p_value = stats.ttest_ind(week1_data, week_data)
                    mean_diff = np.mean(week_data) - np.mean(week1_data)
                    pct_change = (mean_diff / np.mean(week1_data)) * 100

                    f.write(f"Week {int(week)} vs Week 1:\n")
                    f.write(f"  n={len(week_data)}, mean={np.mean(week_data):.4f}\n")
                    f.write(f"  Difference: {mean_diff:.4f} ({pct_change:+.1f}%)\n")
                    f.write(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}\n")

                    if p_value < 0.05:
                        direction = "DECREASE" if mean_diff < 0 else "INCREASE"
                        f.write(f"  Result: SIGNIFICANT {direction} (p < 0.05)\n")
                    else:
                        f.write(f"  Result: No significant difference (p >= 0.05)\n")
                    f.write("\n")

        # Trend analysis (linear regression)
        f.write("\n\nTREND ANALYSIS: Linear Regression of Response vs Week\n")
        f.write("-" * 80 + "\n")

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            resp_df['week'], resp_df['mean_response']
        )

        f.write(f"Slope: {slope:.6f} (response units per week)\n")
        f.write(f"R-squared: {r_value**2:.4f}\n")
        f.write(f"p-value: {p_value:.6f}\n")

        if p_value < 0.05:
            direction = "decreasing" if slope < 0 else "increasing"
            f.write(f"Result: SIGNIFICANT {direction} trend over time (p < 0.05)\n")
        else:
            f.write(f"Result: No significant linear trend (p >= 0.05)\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Statistical analysis report saved: {report_path}")

    # Create visualization
    create_response_amplitude_plot(resp_df, weeks, output_dir)

    return str(report_path)


def create_response_amplitude_plot(
    resp_df: pd.DataFrame,
    weeks: List[int],
    output_dir: Path
) -> None:
    """
    Create visualization of response amplitude changes across weeks.

    Args:
        resp_df: DataFrame with response data
        weeks: List of week numbers
        output_dir: Directory to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Mean response over weeks (box plot)
    ax1 = axes[0]

    box_data = [resp_df[resp_df['week'] == w]['mean_response'].values for w in weeks]
    positions = list(range(len(weeks)))

    bp = ax1.boxplot(box_data, positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    # Color boxes with gradient
    cmap = plt.cm.get_cmap('viridis', len(weeks))
    for patch, idx in zip(bp['boxes'], range(len(weeks))):
        patch.set_facecolor(cmap(idx))
        patch.set_alpha(0.6)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'Week {int(w)}' for w in weeks])
    ax1.set_ylabel('Mean Response (Odor Period)', fontsize=12, fontweight='bold')
    ax1.set_title('Response Amplitude Across Weeks\n(Box = IQR, Red Diamond = Mean)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add sample sizes
    for i, week in enumerate(weeks):
        n = len(resp_df[resp_df['week'] == week])
        ax1.text(i, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02,
                f'n={n}', ha='center', fontsize=9)

    # Plot 2: Individual observations with trend line
    ax2 = axes[1]

    # Scatter plot with jitter
    for week in weeks:
        week_data = resp_df[resp_df['week'] == week]
        jitter = np.random.normal(0, 0.1, len(week_data))
        ax2.scatter(week_data['week'].values + jitter,
                   week_data['mean_response'].values,
                   alpha=0.5, s=50, label=f'Week {int(week)}')

    # Add trend line
    slope, intercept, r_value, p_value, _ = stats.linregress(
        resp_df['week'], resp_df['mean_response']
    )

    trend_x = np.array([weeks[0], weeks[-1]])
    trend_y = slope * trend_x + intercept

    ax2.plot(trend_x, trend_y, 'r--', linewidth=2, alpha=0.8,
            label=f'Trend: y={slope:.4f}x+{intercept:.4f}\nR²={r_value**2:.3f}, p={p_value:.4f}')

    ax2.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Response (Odor Period)', fontsize=12, fontweight='bold')
    ax2.set_title('Individual Observations with Linear Trend',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = output_dir / 'response_amplitude_analysis.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Response amplitude plot saved: {filepath}")


def perform_pca_analysis(
    df: pd.DataFrame,
    dir_cols: List[str],
    output_dir: Path,
    fps: float = 40.0
) -> str:
    """
    Perform PCA on weekly grand averages for weeks with >1 fly.

    Args:
        df: DataFrame containing experimental data
        dir_cols: List of envelope column names
        output_dir: Directory to save output files
        fps: Frames per second

    Returns:
        Path to saved PCA report
    """
    trial_numbers = [1, 2, 3, 4, 6, 8]
    weeks = sorted(df['week'].unique())

    logger.info("=" * 80)
    logger.info("PCA ANALYSIS: Dimensionality Reduction of Weekly Responses")
    logger.info("=" * 80)

    # Filter to odor period only (32-67 seconds)
    odor_start_idx = int(32 * fps)
    odor_end_idx = int(67 * fps)
    odor_cols = dir_cols[odor_start_idx:odor_end_idx]

    # Collect data for PCA (only weeks with >1 fly)
    pca_data = []
    pca_labels = []

    for week in weeks:
        trial_labels = [f'training_{n}' for n in trial_numbers]
        filtered = df[(df['week'] == week) & (df['trial_label'].isin(trial_labels))]

        n_flies = len(filtered)

        if n_flies > 1:  # Only include weeks with >1 fly
            logger.info(f"Week {week}: Including {n_flies} observations in PCA")

            for _, row in filtered.iterrows():
                trace_odor = row[odor_cols].values

                # Only include if no NaN values
                if not np.any(np.isnan(trace_odor)):
                    pca_data.append(trace_odor)
                    pca_labels.append(int(week))
        else:
            logger.info(f"Week {week}: Skipping (only {n_flies} fly)")

    if len(pca_data) < 2:
        logger.warning("Not enough data for PCA analysis (need at least 2 observations)")
        return None

    # Convert to array
    X = np.array(pca_data)
    y = np.array(pca_labels)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=min(10, X_scaled.shape[0], X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Create report
    report_path = output_dir / 'pca_analysis_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PCA ANALYSIS: Dimensionality Reduction of Weekly Training Responses\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("DATA SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total observations: {len(X)}\n")
        f.write(f"Features (time points during odor): {X.shape[1]}\n")
        f.write(f"Weeks included: {sorted(set(y))}\n\n")

        for week in sorted(set(y)):
            count = np.sum(y == week)
            f.write(f"  Week {week}: {count} observations\n")

        f.write("\n\nPCA RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of components: {pca.n_components_}\n\n")

        f.write("Explained variance ratio by component:\n")
        cumulative_var = 0
        for i, var in enumerate(pca.explained_variance_ratio_):
            cumulative_var += var
            f.write(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%) - Cumulative: {cumulative_var:.4f} ({cumulative_var*100:.2f}%)\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"PCA analysis report saved: {report_path}")

    # Create PCA visualization
    create_pca_plot(X_pca, y, pca, weeks, output_dir)

    return str(report_path)


def create_pca_plot(
    X_pca: np.ndarray,
    labels: np.ndarray,
    pca: PCA,
    weeks: List[int],
    output_dir: Path
) -> None:
    """
    Create PCA visualization plots.

    Args:
        X_pca: PCA-transformed data
        labels: Week labels for each observation
        pca: Fitted PCA object
        weeks: List of all week numbers
        output_dir: Directory to save figure
    """
    fig = plt.figure(figsize=(16, 6))

    # Plot 1: PC1 vs PC2 scatter
    ax1 = fig.add_subplot(131)

    unique_weeks = sorted(set(labels))
    cmap = plt.cm.get_cmap('viridis', len(unique_weeks))

    for idx, week in enumerate(unique_weeks):
        mask = labels == week
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[cmap(idx)], s=100, alpha=0.7,
                   label=f'Week {int(week)}', edgecolors='black', linewidth=0.5)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   fontsize=11, fontweight='bold')
    ax1.set_title('PCA: PC1 vs PC2', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scree plot
    ax2 = fig.add_subplot(132)

    n_components = len(pca.explained_variance_ratio_)
    ax2.bar(range(1, n_components + 1), pca.explained_variance_ratio_,
           alpha=0.7, color='steelblue', edgecolor='black')
    ax2.plot(range(1, n_components + 1),
            np.cumsum(pca.explained_variance_ratio_),
            'ro-', linewidth=2, markersize=8, label='Cumulative')

    ax2.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Explained Variance Ratio', fontsize=11, fontweight='bold')
    ax2.set_title('Scree Plot', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: PC1 vs PC3
    if X_pca.shape[1] >= 3:
        ax3 = fig.add_subplot(133)

        for idx, week in enumerate(unique_weeks):
            mask = labels == week
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 2],
                       c=[cmap(idx)], s=100, alpha=0.7,
                       label=f'Week {int(week)}', edgecolors='black', linewidth=0.5)

        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                       fontsize=11, fontweight='bold')
        ax3.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
                       fontsize=11, fontweight='bold')
        ax3.set_title('PCA: PC1 vs PC3', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = output_dir / 'pca_analysis_plot.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"PCA plot saved: {filepath}")


def generate_validation_report(
    df: pd.DataFrame,
    output_files: List[str],
    output_dir: Path,
    total_exp_rows: int,
    non_reactive_excluded: int
) -> str:
    """
    Generate summary report documenting processing results.

    Args:
        df: Processed DataFrame (after filtering)
        output_files: List of generated figure file paths
        output_dir: Directory where report will be saved
        total_exp_rows: Total experimental rows before filtering
        non_reactive_excluded: Number of non-reactive trials excluded

    Returns:
        Path to saved report file
    """
    report_path = output_dir / 'weekly_summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("WEEKLY OLFACTORY TRAINING TRACE ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Dataset summary
        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total experimental rows (before filtering): {total_exp_rows}\n")
        f.write(f"Non-reactive trials excluded: {non_reactive_excluded} ({100*non_reactive_excluded/total_exp_rows:.1f}%)\n")
        f.write(f"Reactive trials retained: {len(df)} ({100*len(df)/total_exp_rows:.1f}%)\n\n")
        f.write(f"Experimental datasets included:\n")
        for dataset in sorted(df['dataset'].unique()):
            count = len(df[df['dataset'] == dataset])
            f.write(f"  - {dataset}: {count} rows\n")
        f.write("\n")

        # Date range
        f.write("DATE RANGE\n")
        f.write("-" * 80 + "\n")
        f.write(f"First date: {df['date'].min().strftime('%Y-%m-%d')}\n")
        f.write(f"Last date: {df['date'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"Span: {(df['date'].max() - df['date'].min()).days} days\n\n")

        # Weekly breakdown
        f.write("GROUP BREAKDOWN (Exact 7-day intervals)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of groups processed: {df['week'].nunique()}\n\n")

        for week in sorted(df['week'].unique()):
            week_df = df[df['week'] == week]
            dates = week_df['date'].unique()
            f.write(f"Group {int(week)}:\n")
            f.write(f"  Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}\n")
            f.write(f"  Unique fly identifiers: {week_df['fly'].nunique()}\n")
            f.write(f"  Total observations: {len(week_df)}\n")
            f.write(f"  Datasets: {', '.join(sorted(week_df['dataset'].unique()))}\n")

            # Trial distribution
            f.write(f"  Trial distribution:\n")
            for trial in sorted(week_df['trial_label'].unique()):
                count = len(week_df[week_df['trial_label'] == trial])
                f.write(f"    {trial}: {count} flies\n")
            f.write("\n")

        # Output files
        f.write("GENERATED FIGURES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total figures generated: {len(output_files)}\n\n")
        for filepath in sorted(output_files):
            f.write(f"  {Path(filepath).name}\n")
        f.write("\n")

        # Data quality metrics
        f.write("DATA QUALITY METRICS\n")
        f.write("-" * 80 + "\n")
        dir_cols = [col for col in df.columns if col.startswith('dir_val_')]
        envelope_data = df[dir_cols]
        total_values = envelope_data.size
        nan_values = envelope_data.isna().sum().sum()
        f.write(f"Total envelope data points: {total_values:,}\n")
        f.write(f"NaN values: {nan_values:,} ({100*nan_values/total_values:.2f}%)\n")
        f.write(f"Valid values: {total_values - nan_values:,} ({100*(total_values-nan_values)/total_values:.2f}%)\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Validation report saved: {report_path}")
    return str(report_path)


def main():
    """Main execution function."""
    # Configuration
    CSV_PATH = '/home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide_training.csv'
    OUTPUT_DIR = Path('/home/ramanlab/Documents/cole/Results/Opto/Weekly-Training-Envelopes')

    logger.info("=" * 80)
    logger.info("WEEKLY OLFACTORY TRAINING TRACE ANALYSIS - STARTING")
    logger.info("=" * 80)

    # Step 1: Load CSV and get initial experimental row count
    raw_df = pd.read_csv(CSV_PATH)
    exp_df_unfiltered = raw_df[raw_df['dataset'].str.startswith('opto')]
    total_exp_rows = len(exp_df_unfiltered)
    non_reactive_count = (exp_df_unfiltered['non_reactive_flag'] == 1.0).sum()

    # Step 2: Load and filter data (excludes non-reactive)
    df = load_and_filter_data(CSV_PATH)

    # Step 3: Extract envelope columns
    dir_cols = extract_envelope_columns(df)

    # Step 4: Get FPS from data
    fps = df['fps'].iloc[0]
    logger.info(f"Using FPS: {fps}")

    # Step 5: Generate figures for each group (exact 7-day intervals)
    weeks = sorted(df['week'].unique())
    logger.info(f"Processing {len(weeks)} groups (7-day intervals): {weeks}")

    output_files = []
    for week in weeks:
        filepath = create_weekly_figure(df, int(week), dir_cols, fps, OUTPUT_DIR)
        if filepath:
            output_files.append(filepath)

    # Step 6: Generate combined plot with all weeks overlaid
    combined_filepath = create_all_weeks_combined_plot(df, dir_cols, fps, OUTPUT_DIR)
    if combined_filepath:
        output_files.append(combined_filepath)

    # Step 7: Generate validation report
    report_path = generate_validation_report(df, output_files, OUTPUT_DIR, total_exp_rows, non_reactive_count)

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Figures saved to: {OUTPUT_DIR}")
    logger.info(f"Summary report: {report_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
