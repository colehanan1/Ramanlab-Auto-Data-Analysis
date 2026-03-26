#!/usr/bin/env python3
"""Generate publication-grade analysis pipeline figure."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# ═══ Global Style ═══════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,
    'axes.linewidth': 2.5,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.major.size': 7,
    'ytick.major.size': 7,
    'legend.fontsize': 16,
    'figure.dpi': 300,
})

# ═══ Color Palette ══════════════════════════════════════════════
DARK       = '#1B2631'
BLUE_FILL  = '#D4E6F1'
BLUE_EDGE  = '#2471A3'
GRN_FILL   = '#D4EFDF'
GRN_EDGE   = '#1E8449'
CORAL_FILL = '#FADBD8'
CORAL_EDGE = '#CB4335'

# ═══ Load Real Data ═════════════════════════════════════════════
DATA_PATH = (
    '/home/ramanlab/Documents/cole/Data/flys/EB-Training/'
    'february_12_batch_1/RMS_calculations/'
    'updated_february_12_batch_1_testing_1_fly1_distances.csv'
)
df = pd.read_csv(DATA_PATH)

eye_x_col  = df['x_class0'].to_numpy(dtype=float)
eye_y_col  = df['y_class0'].to_numpy(dtype=float)
prob_x_col = df['x_class1'].to_numpy(dtype=float)
prob_y_col = df['y_class1'].to_numpy(dtype=float)

d_px = np.sqrt((prob_x_col - eye_x_col)**2 + (prob_y_col - eye_y_col)**2)

angle_pct = df['angle_centered_pct'].to_numpy(dtype=float)
angle_pct = np.clip(angle_pct, -100.0, 100.0)
multiplier = np.where(
    angle_pct <= 0.0,
    1.0,
    1.0 + np.log1p((angle_pct / 100.0) * (np.e - 1.0)),
)

d_px_weighted = d_px * multiplier
finite = d_px_weighted[np.isfinite(d_px_weighted)]
gmin, gmax = float(np.min(finite)), float(np.max(finite))
if gmax > gmin:
    combined_pct = 100.0 * (d_px_weighted - gmin) / (gmax - gmin)
else:
    combined_pct = np.zeros_like(d_px_weighted)
combined_pct = np.clip(combined_pct, 0, 100)

window = 10
kernel = np.ones(window) / window
combined_smooth = np.convolve(combined_pct, kernel, mode='same')

n_frames = len(combined_smooth)
frames = np.arange(n_frames)


# ═══ Figure & Axes ══════════════════════════════════════════════
fig = plt.figure(figsize=(24, 23), facecolor='white')
fig.suptitle('Data Analysis Pipeline: YOLO Coordinates to Behavioral Metrics',
             fontsize=26, fontweight='bold', color=DARK, y=0.988)

# Shifted up to make room for variable table at bottom
ax_flow = fig.add_axes([0.04, 0.26, 0.44, 0.69])
ax_flow.set_xlim(-2, 11)
ax_flow.set_ylim(-0.5, 24)
ax_flow.axis('off')

ax_dist = fig.add_axes([0.56, 0.64, 0.40, 0.29])
ax_wind = fig.add_axes([0.56, 0.26, 0.40, 0.28])

# Table axes at bottom — spanning full width
ax_tbl = fig.add_axes([0.04, 0.01, 0.92, 0.23])
ax_tbl.axis('off')


# ═══ Flowchart Helpers ══════════════════════════════════════════
def draw_step(ax, cx, cy, w, h, title, lines, fill, edge):
    """Draw a rounded step box with title and content."""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.2",
        facecolor=fill, edgecolor=edge, linewidth=3, zorder=2,
    )
    ax.add_patch(rect)

    title_y = cy + h / 2 - 0.58
    ax.text(cx, title_y, title, ha='center', va='center',
            fontsize=18, fontweight='bold', color=DARK, zorder=3)

    sep_y = title_y - 0.40
    ax.plot([cx - w / 2 + 0.5, cx + w / 2 - 0.5], [sep_y, sep_y],
            color=edge, linewidth=1.2, alpha=0.4, zorder=3)

    spacing = 0.50
    start_y = sep_y - 0.52
    for i, line in enumerate(lines):
        ax.text(cx, start_y - i * spacing, line, ha='center', va='center',
                fontsize=16, color='#2C3E50', zorder=3)


def draw_arrow(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='-|>', color=DARK,
                                lw=3, mutation_scale=25),
                zorder=1)


# ═══ Step Definitions ═══════════════════════════════════════════
CX = 5.0
W  = 9.0

steps = [
    (22.1, 1.8,
     'Step 1:  YOLO Detection',
     ['Eye (x, y)  &  Proboscis (x, y)'],
     BLUE_FILL, BLUE_EDGE),

    (18.7, 2.8,
     'Step 2:  Basic Geometry',
     ['dx = prob_x \u2212 eye_x',
      'dy = prob_y \u2212 eye_y',
      'r  = \u221A(dx\u00B2 + dy\u00B2)'],
     BLUE_FILL, BLUE_EDGE),

    (14.6, 3.4,
     'Step 3:  Per-Fly Statistics',
     ['w_est = max(x) \u2212 min(x)',
      'h_est = max(y) \u2212 min(y)',
      'diag_est = \u221A(w_est\u00B2 + h_est\u00B2)',
      'r_p01, r_p99  (robust range)'],
     GRN_FILL, GRN_EDGE),

    (10.2, 3.4,
     'Step 4:  Normalized Features',
     ['r_pct = 100 \u00D7 (r \u2212 r_p01) / (r_p99 \u2212 r_p01)',
      'speed(t) = \u0394r / \u0394t',
      'sin(\u03B8) = dy / r',
      'r_z = (r \u2212 r_mean) / r_std'],
     GRN_FILL, GRN_EDGE),

    (5.8, 3.4,
     'Step 5:  Behavioral Windows',
     ['Before :   frames < 1260  (~31 s @ 40 fps)',
      'During :   1260 \u2264 frames < 2460  (~30 s)',
      'After :      frames \u2265 2460',
      'Early :     frames 1260\u20131300  (~1 s)'],
     CORAL_FILL, CORAL_EDGE),

    (2.0, 2.2,
     'Step 6:  Summary Statistics',
     ['r_before_mean,  r_during_mean',
      'fast_speed = (r_early \u2212 r_before) / 1 s'],
     CORAL_FILL, CORAL_EDGE),
]

for i, (cy, h, title, lines, fill, edge) in enumerate(steps):
    draw_step(ax_flow, CX, cy, W, h, title, lines, fill, edge)
    if i < len(steps) - 1:
        y_from = cy - h / 2
        y_to   = steps[i + 1][0] + steps[i + 1][1] / 2
        draw_arrow(ax_flow, CX, y_from, y_to)


# ═══ Section Side Brackets ══════════════════════════════════════
bracket_x = 0.0
sec_defs = [
    (steps[0][0] + steps[0][1]/2 + 0.15, steps[1][0] - steps[1][1]/2 - 0.15,
     'Per Frame', BLUE_EDGE),
    (steps[2][0] + steps[2][1]/2 + 0.15, steps[3][0] - steps[3][1]/2 - 0.15,
     'Per Fly', GRN_EDGE),
    (steps[4][0] + steps[4][1]/2 + 0.15, steps[5][0] - steps[5][1]/2 - 0.15,
     'Behavioral', CORAL_EDGE),
]
for y_top, y_bot, label, color in sec_defs:
    ax_flow.plot([bracket_x, bracket_x], [y_bot, y_top],
                 color=color, linewidth=6, solid_capstyle='round',
                 alpha=0.6, zorder=0)
    ax_flow.text(bracket_x - 0.65, (y_top + y_bot) / 2, label,
                 fontsize=13, fontweight='bold', color=color,
                 ha='center', va='center', rotation=90)

# Transition dots
for i_gap in [1, 3]:
    cy1, h1 = steps[i_gap][0], steps[i_gap][1]
    cy2, h2 = steps[i_gap + 1][0], steps[i_gap + 1][1]
    mid_y = (cy1 - h1 / 2 + cy2 + h2 / 2) / 2
    circle = plt.Circle((CX, mid_y), 0.25,
                         facecolor=CORAL_EDGE, edgecolor='white',
                         linewidth=2.5, zorder=5)
    ax_flow.add_patch(circle)


# ═══ Right Top: Distance Calculation Example ════════════════════
ax_dist.set_title('Example: Eye\u2013Proboscis Distance', fontsize=20,
                   fontweight='bold', color=DARK, pad=14)

prob_pt = (25, 40)
eye_pt  = (82, 80)

ax_dist.plot([prob_pt[0], eye_pt[0]], [prob_pt[1], prob_pt[1]], ':',
             color=BLUE_EDGE, linewidth=3, alpha=0.55)
ax_dist.plot([eye_pt[0], eye_pt[0]], [prob_pt[1], eye_pt[1]], ':',
             color=BLUE_EDGE, linewidth=3, alpha=0.55)

cs = 4.5
ax_dist.plot([eye_pt[0] - cs, eye_pt[0] - cs, eye_pt[0]],
             [prob_pt[1], prob_pt[1] + cs, prob_pt[1] + cs],
             color=BLUE_EDGE, linewidth=2.5, alpha=0.55)

ax_dist.plot([prob_pt[0], eye_pt[0]], [prob_pt[1], eye_pt[1]], '--',
             color='#5D6D7E', linewidth=3, zorder=2)

ax_dist.text((prob_pt[0] + eye_pt[0]) / 2, prob_pt[1] - 5, 'dx = 57',
             ha='center', va='top', fontsize=17,
             fontweight='bold', color=BLUE_EDGE)
ax_dist.text(eye_pt[0] + 4, (prob_pt[1] + eye_pt[1]) / 2, 'dy = 40',
             ha='left', va='center', fontsize=17,
             fontweight='bold', color=BLUE_EDGE)

angle = np.degrees(np.arctan2(eye_pt[1] - prob_pt[1], eye_pt[0] - prob_pt[0]))
ax_dist.text((prob_pt[0] + eye_pt[0]) / 2 - 8, (prob_pt[1] + eye_pt[1]) / 2 + 5,
             'r \u2248 70', fontsize=19, fontweight='bold',
             color=CORAL_EDGE, rotation=angle, ha='center', va='bottom')

ax_dist.scatter(*eye_pt,  s=600, c=BLUE_EDGE,  edgecolors='white',
                linewidth=3, zorder=4, label='Eye')
ax_dist.scatter(*prob_pt, s=600, c=CORAL_EDGE, edgecolors='white',
                linewidth=3, zorder=4, marker='s', label='Proboscis')

ax_dist.set_xlabel('X (pixels)', fontweight='bold')
ax_dist.set_ylabel('Y (pixels)', fontweight='bold')
ax_dist.set_xlim(0, 110)
ax_dist.set_ylim(10, 100)
ax_dist.legend(loc='upper left', frameon=True, fancybox=True,
               edgecolor='#BDC3C7', framealpha=0.95, fontsize=16,
               handletextpad=0.5, markerscale=1.0)
ax_dist.grid(True, alpha=0.25, linewidth=1.2)
for s in ax_dist.spines.values():
    s.set_linewidth(2.5)


# ═══ Right Bottom: Behavioral Windows (REAL DATA) ═══════════════
ax_wind.set_title(
    'Example: Eye\u2013Proboscis Distance Trace for a Single Fly',
    fontsize=18, fontweight='bold', color=DARK, pad=14,
)

BEFORE_END = 1260
DURING_END = 2460

ax_wind.axvspan(0,          BEFORE_END, alpha=0.18, color=BLUE_FILL, zorder=0)
ax_wind.axvspan(BEFORE_END, DURING_END, alpha=0.18, color=CORAL_FILL, zorder=0)
ax_wind.axvspan(DURING_END, n_frames,   alpha=0.18, color=GRN_FILL,  zorder=0)

for xv in [BEFORE_END, DURING_END]:
    ax_wind.axvline(xv, color='#7F8C8D', linewidth=2.5,
                    linestyle='--', alpha=0.6)

ax_wind.plot(frames, combined_smooth, color=DARK, linewidth=2.5, alpha=0.85)

label_y = combined_smooth.max() * 1.05 + 5
label_y = min(label_y, 95)
ax_wind.text(BEFORE_END / 2,               label_y, 'Before\n(Baseline)',
             ha='center', va='center', fontsize=15, fontweight='bold',
             color=BLUE_EDGE)
ax_wind.text((BEFORE_END + DURING_END) / 2, label_y, 'During\n(Odor)',
             ha='center', va='center', fontsize=15, fontweight='bold',
             color=CORAL_EDGE)
ax_wind.text((DURING_END + n_frames) / 2,   label_y, 'After',
             ha='center', va='center', fontsize=15, fontweight='bold',
             color=GRN_EDGE)

ax_wind.text(BEFORE_END / 2,                3, '~31 s',
             ha='center', fontsize=13, color='#95A5A6', fontstyle='italic')
ax_wind.text((BEFORE_END + DURING_END) / 2, 3, '~30 s',
             ha='center', fontsize=13, color='#95A5A6', fontstyle='italic')

ax_wind.set_xlabel('Frame Number', fontweight='bold')
ax_wind.set_ylabel('Max Distance \u00D7 Angle %', fontweight='bold')
ax_wind.set_xlim(0, n_frames)
y_max = min(combined_smooth.max() * 1.15 + 8, 105)
ax_wind.set_ylim(0, y_max)
ax_wind.grid(True, alpha=0.25, linewidth=1.2, axis='y')
for s in ax_wind.spines.values():
    s.set_linewidth(2.5)


# ═══ Variable Reference Table ═══════════════════════════════════
ax_tbl.set_xlim(0, 1)
ax_tbl.set_ylim(0, 1)

ax_tbl.text(0.5, 0.97, 'Variable Reference',
            ha='center', va='top', fontsize=22, fontweight='bold', color=DARK)

# Table data: (variable, description, units)
table_data = [
    ['eye_x, eye_y',       'Eye center coordinates (YOLO detection)',                   'pixels'],
    ['prob_x, prob_y',     'Proboscis tip coordinates (YOLO detection)',                 'pixels'],
    ['dx, dy',             'Horizontal / vertical offset (proboscis \u2212 eye)',        'pixels'],
    ['r',                  'Euclidean eye-to-proboscis distance',                        'pixels'],
    ['w_est, h_est',       'Fly bounding width / height across all frames',             'pixels'],
    ['diag_est',           'Fly diagonal size: \u221A(w_est\u00B2 + h_est\u00B2)',      'pixels'],
    ['r_p01, r_p99',       '1st / 99th percentile of r  (per-fly robust range)',        'pixels'],
    ['r_pct',              'Distance normalized to robust range: 0\u2013100',            '%'],
    ['speed(t)',            'Frame-to-frame rate of change in distance',                 'px / frame'],
    ['\u03B8',             'Angle of proboscis relative to eye',                         'degrees'],
    ['r_z',                'Z-scored distance: (r \u2212 r_mean) / r_std',              'unitless'],
    ['r_before_mean',      'Mean distance during Before (baseline) window',             'pixels'],
    ['r_during_mean',      'Mean distance during During (odor) window',                 'pixels'],
    ['r_early',            'Mean distance in Early window (first ~1 s of odor)',        'pixels'],
    ['fast_speed',         'Initial response speed: (r_early \u2212 r_before) / 1 s',  'px / s'],
]

col_headers = ['Variable', 'Description', 'Units']
col_x = [0.02, 0.22, 0.88]
col_align = ['left', 'left', 'left']
header_y = 0.86
row_h = 0.052

# Header background
ax_tbl.fill_between([0.01, 0.99], header_y - 0.025, header_y + 0.028,
                     color=DARK, alpha=0.85, zorder=0)

# Draw header row
for j, hdr in enumerate(col_headers):
    ax_tbl.text(col_x[j], header_y, hdr, ha=col_align[j], va='center',
                fontsize=16, fontweight='bold', color='white')

# Separator line below header
ax_tbl.plot([0.01, 0.99], [header_y - 0.028, header_y - 0.028],
            color=DARK, linewidth=1.5)

# Draw data rows
for i, row in enumerate(table_data):
    y = header_y - 0.058 - i * row_h

    # Alternating row background
    if i % 2 == 0:
        ax_tbl.fill_between([0.01, 0.99], y - row_h / 2 + 0.004, y + row_h / 2 - 0.004,
                             color='#F2F3F4', alpha=0.6, zorder=0)

    # Variable name in bold
    ax_tbl.text(col_x[0], y, row[0], ha='left', va='center',
                fontsize=15, fontweight='bold', color=DARK)
    # Description
    ax_tbl.text(col_x[1], y, row[1], ha='left', va='center',
                fontsize=15, color='#2C3E50')
    # Units
    ax_tbl.text(col_x[2], y, row[2], ha='left', va='center',
                fontsize=15, color='#5D6D7E', fontstyle='italic')

# Bottom border
bottom_y = header_y - 0.058 - (len(table_data) - 1) * row_h - row_h / 2
ax_tbl.plot([0.01, 0.99], [bottom_y + 0.004, bottom_y + 0.004],
            color=DARK, linewidth=1.0, alpha=0.3)


# ═══ Save ═══════════════════════════════════════════════════════
out = '/home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis/analysis_pipeline_figure.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {out}')
