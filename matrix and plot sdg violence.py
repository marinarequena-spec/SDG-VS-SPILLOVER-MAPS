# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:24:49 2025

@author: marina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.colors import TwoSlopeNorm

# ─── 1) Load & clean headers ─────────────────────────────────────────────────
df = pd.read_excel('SDR2024-data.xlsx', sheet_name='data')

# strip normal spaces, NBSPs, BOMs, etc.
df.columns = (
    df.columns
      .str.replace('\u00A0', ' ')      # NBSP → space
      .str.replace('\ufeff', '')       # BOM
      .str.strip()
)

# ─── 2) Numeric columns you care about ───────────────────────────────────────
num_cols = [
    "Spillovers Score", "SDG Index", "SDG1.Poverty", "SDG2.Hunger", "SDG3.Health",
    "SDG4.Education", "SDG5.Equality", "SDG6.Water", "SDG7.Energy", "SDG8.Growth",
    "SDG9.Industry", "SDG10.Inequality", "SDG11.Cities",
    "SDG12.Consumption", "SDG13.Climate",
    "SDG14.Oceans", "SDG15.Land", "SDG16.Peace", "SDG17.Partnerships"
]

# ─── 3) Build a DataFrame that *includes* Country + those numerics ──────────
df_num = df[['Country'] + num_cols].copy()

# coerce all SDG columns to numeric
df_num[num_cols] = df_num[num_cols].apply(pd.to_numeric, errors='coerce')

# ─── 4) Drop any row with a missing value ────────────────────────────────────
df_num = df_num.dropna(subset=num_cols)

# ─── 5) Remove outliers (|z| > 3 in *any* numeric column) ───────────────────
z = (df_num[num_cols] - df_num[num_cols].mean()) / df_num[num_cols].std(ddof=0)
mask = (z.abs() <= 3).all(axis=1)
data_clean = df_num.loc[mask].copy()

print(f"After dropna & outlier removal: N = {len(data_clean)} rows")

# ─── 6) (Optional) Build your correlation matrices ──────────────────────────
r_mat = pd.DataFrame(index=num_cols, columns=num_cols, dtype=float)
p_mat = pd.DataFrame(index=num_cols, columns=num_cols, dtype=float)
n_mat = pd.DataFrame(index=num_cols, columns=num_cols, dtype=int)
# for x in num_cols:
#     for y in num_cols:
#         r, p = pearsonr(data_clean[x], data_clean[y])
#         r_mat.at[x,y], p_mat.at[x,y] = r, p
#         n_mat.at[x,y] = len(data_clean[[x,y]])

# ─── 7) Plotting setup ───────────────────────────────────────────────────────
highlight_countries = [
    "China", "Djibouti", "Cuba", "United States",
    "Angola", "Finland", "Sweden", "Denmark",
    "South Sudan", "Central African Republic",
    "Chad","Afghanistan"
]

# pick a discrete colormap
if len(highlight_countries) <= 10:
    base = plt.colormaps['tab10']
else:
    base = plt.colormaps['tab20']
cmap = base.resampled(len(highlight_countries))
colors = {c: cmap(i) for i,c in enumerate(highlight_countries)}

# define your four panels
plots = [
    ("SDG Index",        "SDG12.Consumption", "a) SDG Index vs SDG12"),
    ("SDG Index",        "SDG13.Climate",                "b) SDG Index vs SDG13"),
    ("Spillovers Score", "SDG12.Consumption", "c) Spillovers vs SDG12"),
    ("Spillovers Score", "SDG13.Climate",                "d) Spillovers vs SDG13"),
]

# ─── 8) Draw the 2×2 figure ─────────────────────────────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
for ax, (xcol, ycol, title) in zip(axs.ravel(), plots):
    ax.set_facecolor('#f0f0f0')
    # all points in pale olive
    ax.scatter(
        data_clean[xcol], data_clean[ycol],
        color='olive', alpha=0.5, s=40
    )
    # overlay each highlighted country
    for country in highlight_countries:
        sub = data_clean[
            data_clean['Country'].str.lower() == country.lower()
        ]
        for _, row in sub.iterrows():
            x, y = row[xcol], row[ycol]
            ax.scatter(
                x, y,
                color=colors[country], s=200,
                edgecolor='black', linewidth=1.2, alpha=0.9
            )
            ax.text(
                x, y, country,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color=colors[country],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)
            )
    ax.set_xlabel(xcol, fontsize=12)
    ax.set_ylabel(ycol, fontsize=12)
    ax.set_title(title, fontsize=14)

plt.suptitle(
    'SDG & Spillover Relationships vs SDG12 & SDG13',
    fontsize=16, y=1.02
)
plt.tight_layout()
plt.show()

# What countries were in the original df but not in our cleaned set?
all_ctry = set(df['Country'].str.strip().str.lower())
clean_ctry = set(data_clean['Country'].str.strip().str.lower())
dropped = all_ctry - clean_ctry
print("Dropped by dropna/outlier removal:", dropped)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# 1. Load and clean headers
df = pd.read_excel('SDR2024-data.xlsx', sheet_name='data')
df.columns = df.columns.str.strip()

# 2. Explicitly list your 18 columns in order
cols = [
    "Spillovers Score", "SDG Index", "SDG1.Poverty", "SDG2.Hunger", "SDG3.Health",
    "SDG4.Education", "SDG5.Equality", "SDG6.Water", "SDG7.Energy", "SDG8.Growth",
    "SDG9.Industry", "SDG10.Inequality", "SDG11.Cities", "SDG12.Consumption", "SDG13.Climate",
    "SDG14.Oceans", "SDG15.Land", "SDG16.Peace", "SDG17.Partnerships"
]

# 3. Slice and coerce to numeric
data = df[cols].apply(pd.to_numeric, errors='coerce')

# 4. Drop any row with a missing value
data = data.dropna()

# 5. Remove outliers (|z-score| > 3 in any column)
z = (data - data.mean()) / data.std(ddof=0)
data_clean = data[(z.abs() <= 3).all(axis=1)].copy()

print(f"After dropna & outlier removal: N = {len(data_clean)} rows")

# 6. Prepare empty result matrices
r_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
p_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
n_mat = pd.DataFrame(index=cols, columns=cols, dtype=int)

# 7. Compute Pearson’s r, p-value, and N (constant across all pairs)
N = len(data_clean)
for x in cols:
    for y in cols:
        if N > 1:
            r, p = pearsonr(data_clean[x], data_clean[y])
        else:
            r, p = np.nan, np.nan
        r_mat.at[x, y] = r
        p_mat.at[x, y] = p
        n_mat.at[x, y] = N

# 8. Save to Excel: three sheets “correlation” (r), “sig” (p), “n” (counts)
with pd.ExcelWriter('correlations_with_stats.xlsx') as writer:
    r_mat.to_excel(writer, sheet_name='correlation')
    p_mat.to_excel(writer, sheet_name='sig')
    n_mat.to_excel(writer, sheet_name='n')

print("Saved [correlation, sig, n] sheets to 'correlations_with_stats.xlsx'")

# 9. Plot the heatmap with grey at r=0
#    - blue for r=-1, grey for r=0, red for r=+1
cmap = LinearSegmentedColormap.from_list(
    'blue_grey_red',
    ['#2166ac', '#bababa', '#b2182b']
)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

fig, ax = plt.subplots(figsize=(12, 10))
heat = ax.imshow(r_mat.values.astype(float), cmap=cmap, norm=norm)

# Annotate each cell with the r‐value
for i in range(len(cols)):
    for j in range(len(cols)):
        val = r_mat.iat[i, j]
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=10)

# Ticks and labels
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(cols)))
ax.set_xticklabels(cols, rotation=90, fontsize=15)
ax.set_yticklabels(cols, fontsize=15)

# Colorbar
cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Pearson r', fontsize=18)

ax.set_title('Pearson Correlation Matrix SDGs & Spillovers', fontsize=18)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load & clean
df = pd.read_excel('SDR2024-data.xlsx', sheet_name='data')
df.columns = df.columns.str.strip()

# 2. Scores
scores = df[['Spillovers Score', 'SDG Index']].apply(pd.to_numeric, errors='coerce')

# 3. Identify which rows are the four income groups
income_acronyms = ['LIC','LMC','UMC','HIC']
is_income_group = df['Country'].isin(income_acronyms)

# 4. Drop missing / outliers exactly as before
valid = scores.dropna().index
scores = scores.loc[valid]
is_income_group = is_income_group.loc[valid]

z = (scores - scores.mean()) / scores.std(ddof=0)
mask = (z.abs() <= 3).all(axis=1)
scores = scores[mask]
is_income_group = is_income_group[mask]

# 5. Base‐layer: everything in grey, no legend entry
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(
    scores['Spillovers Score'],
    scores['SDG Index'],
    color='grey',
    alpha=0.3,
    s=40,
    label='_nolegend_'    # suppress this from the legend
)

# 6. Now overlay each income group in its own colour + legend entry
mapping = {
    'LIC': 'Low-income',
    'LMC': 'Lower-middle-income',
    'UMC': 'Upper-middle-income',
    'HIC': 'High-income'
}
palette = plt.cm.tab10.colors
for i, acr in enumerate(income_acronyms):
    sel = (df['Country'] == acr) & mask & (valid.to_series())
    ax.scatter(
        scores.loc[sel, 'Spillovers Score'],
        scores.loc[sel, 'SDG Index'],
        label=mapping[acr],
        s=80,
        edgecolors='w',
        color=palette[i],
        alpha=0.9
    )

# 7. Median quadrants
x_med = scores['Spillovers Score'].median()
y_med = scores['SDG Index'].median()
ax.axvline(x_med, color='gray', linestyle='--', linewidth=1)
ax.axhline(y_med, color='gray', linestyle='--', linewidth=1)

# 8. Quadrant labels
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
dx, dy = (x1-x0)*0.05, (y1-y0)*0.05

ax.text(x0+dx, y1-dy, 'High SDG\nLow Spillover',
        ha='left',  va='top',    weight='bold')
ax.text(x1-dx, y1-dy, 'High SDG\nHigh Spillover',
        ha='right', va='top',    weight='bold')
ax.text(x0+dx, y0+dy, 'Low SDG\nLow Spillover',
        ha='left',  va='bottom', weight='bold')
ax.text(x1-dx, y0+dy, 'Low SDG\nHigh Spillover',
        ha='right', va='bottom', weight='bold')

# 9. Styling & legend
ax.set_xlabel('International Spillovers Score', fontsize=14)
ax.set_ylabel('SDG Index',                fontsize=14)
ax.set_title('SDG Index vs. International Spillovers by Income Group',
             fontsize=16, pad=20)
ax.grid(alpha=0.3)
ax.legend(title='Income Group')
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# 1. Load and clean headers
df = pd.read_excel('SDR2024-data.xlsx', sheet_name='data')
df.columns = df.columns.str.strip()

# 2. Explicitly list your 18 columns in order
cols = [
    "Spillovers Score", "SDG Index", "SDG1.Poverty", "SDG2.Hunger", "SDG3.Health",
    "SDG4.Education", "SDG5.Equality", "SDG6.Water", "SDG7.Energy", "SDG8.Growth",
    "SDG9.Industry", "SDG10.Inequality", "SDG11.Cities", "SDG12.Consumption", "SDG13.Climate",
    "SDG14.Oceans", "SDG15.Land", "SDG16.Peace", "SDG17.Partnerships"
]

# 3. Slice and coerce to numeric
data = df[cols].apply(pd.to_numeric, errors='coerce')

# 4. Drop any row with a missing value
data_clean = data.dropna()

# 5. Remove outliers (|z-score| > 3 in any column)
z = (data_clean - data_clean.mean()) / data_clean.std(ddof=0)
data_clean = data_clean[(z.abs() <= 3).all(axis=1)].copy()

print(f"After dropna & outlier removal: N = {len(data_clean)} rows")

# 6. Prepare empty result matrices
r_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
p_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
n_mat = pd.DataFrame(index=cols, columns=cols, dtype=int)

# 7. Compute Pearson’s r, p-value, and N (constant across all pairs)
N = len(data_clean)
for x in cols:
    for y in cols:
        if N > 1:
            r, p = pearsonr(data_clean[x], data_clean[y])
        else:
            r, p = np.nan, np.nan
        r_mat.at[x, y] = r
        p_mat.at[x, y] = p
        n_mat.at[x, y] = N

# 8. Save to Excel: three sheets “correlation” (r), “sig” (p), “n” (counts)
with pd.ExcelWriter('correlations_with_stats.xlsx') as writer:
    r_mat.to_excel(writer, sheet_name='correlation')
    p_mat.to_excel(writer, sheet_name='sig')
    n_mat.to_excel(writer, sheet_name='n')

print("Saved [correlation, sig, n] sheets to 'correlations_with_stats.xlsx'")

# 9. Plot the heatmap with grey at r=0
cmap = LinearSegmentedColormap.from_list('blue_grey_red',
                                         ['#2166ac', '#bababa', '#b2182b'])
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

fig, ax = plt.subplots(figsize=(12, 10))
heat = ax.imshow(r_mat.values.astype(float), cmap=cmap, norm=norm)

for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{r_mat.iat[i, j]:.2f}", ha='center', va='center', fontsize=10)

ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(cols)))
ax.set_xticklabels(cols, rotation=90, fontsize=8)
ax.set_yticklabels(cols, fontsize=8)

cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Pearson r', fontsize=12)

ax.set_title('Pearson Correlation Matrix SDGs & Spillovers', fontsize=14)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 10. FOUR SCATTER PLOTS IN ONE FIGURE
# -----------------------------------------------------------------------------
fig2, axs = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: SDG Index vs SDG12.Consumption
axs[0, 0].scatter(data_clean['SDG Index'], data_clean['SDG12.Consumption'], alpha=0.6)
axs[0, 0].set_xlabel('SDG Index')
axs[0, 0].set_ylabel('SDG12.Consumption')
axs[0, 0].set_title('SDG Index vs SDG12.Consumption')

# Top-right: SDG Index vs SDG13.Climate
axs[0, 1].scatter(data_clean['SDG Index'], data_clean['SDG13.Climate'], alpha=0.6)
axs[0, 1].set_xlabel('SDG Index')
axs[0, 1].set_ylabel('SDG13.Climate')
axs[0, 1].set_title('SDG Index vs SDG13.Climate')

# Bottom-left: Spillovers Score vs SDG12.Consumption
axs[1, 0].scatter(data_clean['Spillovers Score'], data_clean['SDG12.Consumption'], alpha=0.6)
axs[1, 0].set_xlabel('Spillovers Score')
axs[1, 0].set_ylabel('SDG12.Consumption')
axs[1, 0].set_title('Spillovers Score vs SDG12.Consumption')

# Bottom-right: Spillovers Score vs SDG13.Climate
axs[1, 1].scatter(data_clean['Spillovers Score'], data_clean['SDG13.Climate'], alpha=0.6)
axs[1, 1].set_xlabel('Spillovers Score')
axs[1, 1].set_ylabel('SDG13.Climate')
axs[1, 1].set_title('Spillovers Score vs SDG13.Climate')

plt.suptitle('Key Pairwise SDG & Spillover Relationships', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# assume df (with a 'Country' column) and data_clean (numeric columns) exist,
# and that data_clean.index matches df.index

import matplotlib.pyplot as plt

# 1) choose which countries to highlight (now including Chad)
highlight_countries = [
    "China", "Djibouti", "Cuba", "United States", "Finland", "Sweden", "Denmark",
    "South Sudan", "Central African Republic",
    "Chad"
]

# pick a distinct color for each
palette = plt.cm.tab10.colors
colors = {c: palette[i] for i, c in enumerate(highlight_countries)}

# 2) define your four (x, y, title) tuples
plots = [
    ("SDG Index",           "SDG12.Responsible_Consumption", "a)SDG Index vs SDG12.Responsible_Consumption"),
    ("SDG Index",           "SDG13.Climate",     "b)SDG Index vs SDG13.Climate"),
    ("Spillovers Score",    "SDG12.Responsible_Consumption", "c)Spillovers vs SDG12.Responsible_Consumption"),
    ("Spillovers Score",    "SDG13.Climate",     "d)Spillovers vs SDG13.Climate"),
]

# 3) make the 2×2 figure
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
for ax, (xcol, ycol, title) in zip(axs.ravel(), plots):
    # grey background
    ax.set_facecolor('#f0f0f0')
    # all points in pale grey
    ax.scatter(
        data_clean[xcol],
        data_clean[ycol],
        color='olive',
        alpha=0.5,
        s=40
    )
    # now overlay each highlight country
    for country in highlight_countries:
        # find the matching row(s)
        mask = df['Country'].str.lower() == country.lower()
        idxs = [i for i in df.index[mask] if i in data_clean.index]
        for idx in idxs:
            x = data_clean.at[idx, xcol]
            y = data_clean.at[idx, ycol]
            ax.scatter(
                x, y,
                color=colors[country],
                s=200,
                edgecolor='black',
                linewidth=1.2,
                alpha=0.9
            )
            ax.text(
                x, y, country,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color=colors[country],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)
            )
    # labels & title
    ax.set_xlabel(xcol, fontsize=12)
    ax.set_ylabel(ycol, fontsize=12)
    ax.set_title(title, fontsize=14)

plt.suptitle('SDG & Spillover Relationships VS SDG12.Responsible_Consmuption & SDG13.Climate', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# 0) make sure Country has no leading/trailing spaces
data_clean['Country'] = data_clean['Country'].str.strip()

# 1) choose which countries to highlight
highlight_countries = [
    "China", "Djibouti", "Cuba", "United States", "Finland", "Sweden",
    "Denmark", "South Sudan", "Central African Republic", "Chad"
]

# 2) pick a distinct color for each
palette = plt.cm.tab10.colors
colors = {c: palette[i] for i, c in enumerate(highlight_countries)}

# 3) define your four (x, y, title) tuples
plots = [
    ("SDG Index",        "SDG12.Responsible_Consumption", "a) SDG Index vs SDG12"),
    ("SDG Index",        "SDG13.Climate",                "b) SDG Index vs SDG13"),
    ("Spillovers Score", "SDG12.Responsible_Consumption","c) Spillovers vs SDG12"),
    ("Spillovers Score", "SDG13.Climate",                "d) Spillovers vs SDG13"),
]

# 4) make the 2×2 figure
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for ax, (xcol, ycol, title) in zip(axs.ravel(), plots):
    ax.set_facecolor('#f0f0f0')
    # all points in pale olive
    ax.scatter(
        data_clean[xcol], data_clean[ycol],
        color='olive', alpha=0.5, s=40
    )
    
    # now overlay each highlight country from data_clean
    for country in highlight_countries:
        sub = data_clean[
            data_clean['Country'].str.lower() == country.lower()
        ]
        for _, row in sub.iterrows():
            x, y = row[xcol], row[ycol]
            ax.scatter(
                x, y,
                color=colors[country],
                s=200,
                edgecolor='black',
                linewidth=1.2,
                alpha=0.9
            )
            ax.text(
                x, y, country,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color=colors[country],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)
            )

    ax.set_xlabel(xcol, fontsize=12)
    ax.set_ylabel(ycol, fontsize=12)
    ax.set_title(title, fontsize=14)

plt.suptitle(
    'SDG & Spillover Relationships vs SDG12 & SDG13',
    fontsize=16, y=1.02
)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# ─── 0) load your data and clean the column‐names ─────────────────────────────
DATA_PATH = "path/to/your/data.csv"
data_clean = pd.read_csv(DATA_PATH)

# Strip whitespace from all the column headers so 'Country' really is 'Country'
data_clean.columns = data_clean.columns.str.strip()

# ─── 1) choose which countries to highlight ───────────────────────────────────
highlight_countries = [
    "China", "Djibouti", "Cuba", "United States",
    "Finland", "Sweden", "Denmark",
    "South Sudan", "Central African Republic",
    "Chad"
]

# ─── 2) pick a distinct color for each ────────────────────────────────────────
palette = plt.cm.tab10.colors
colors = {c: palette[i] for i, c in enumerate(highlight_countries)}

# ─── 3) define your four (x, y, title) tuples ────────────────────────────────
plots = [
    ("SDG Index",        "SDG12.Responsible_Consumption", "a) SDG Index vs SDG12"),
    ("SDG Index",        "SDG13.Climate",                "b) SDG Index vs SDG13"),
    ("Spillovers Score", "SDG12.Responsible_Consumption","c) Spillovers vs SDG12"),
    ("Spillovers Score", "SDG13.Climate",                "d) Spillovers vs SDG13"),
]

# ─── 4) make the 2×2 figure ───────────────────────────────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
for ax, (xcol, ycol, title) in zip(axs.ravel(), plots):
    ax.set_facecolor('#f0f0f0')
    # all points in pale olive
    ax.scatter(
        data_clean[xcol],
        data_clean[ycol],
        color='olive',
        alpha=0.5,
        s=40
    )

    # now overlay each highlight country (from data_clean, not df!)
    for country in highlight_countries:
        mask = data_clean['Country'].str.lower() == country.lower()
        for _, row in data_clean[mask].iterrows():
            x, y = row[xcol], row[ycol]
            ax.scatter(
                x, y,
                color=colors[country],
                s=200,
                edgecolor='black',
                linewidth=1.2,
                alpha=0.9
            )
            ax.text(
                x, y, country,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color=colors[country],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)
            )

    ax.set_xlabel(xcol, fontsize=12)
    ax.set_ylabel(ycol, fontsize=12)
    ax.set_title(title, fontsize=14)

plt.suptitle(
    'SDG & Spillover Relationships vs SDG12 & SDG13',
    fontsize=16, y=1.02
)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# 1. Load & clean
df = pd.read_excel('SDR2024-data.xlsx', sheet_name='data')
df.columns = df.columns.str.strip()

# 2. Numeric scores & mask
scores = df[['Spillovers Score','SDG Index']].apply(pd.to_numeric, errors='coerce')
valid = scores.dropna().index
scores = scores.loc[valid]
id_series = df.loc[valid, 'id']
z = (scores - scores.mean())/scores.std(ddof=0)
ok = (z.abs()<=3).all(axis=1)
scores = scores[ok]
id_series = id_series[ok]

# 3. Build base scatter + medians
fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(scores['Spillovers Score'],
           scores['SDG Index'],
           color='lightgrey', alpha=0.4, s=50, label='_nolegend_')

# income‐group overlays
highlight_ids   = ['_LIC','_LMIC','_UMIC','_HIC']
highlight_names = {'_LIC':'Low-income',
                   '_LMIC':'Lower-middle',
                   '_UMIC':'Upper-middle',
                   '_HIC':'High-income'}
palette = plt.cm.Set1.colors

for i, hid in enumerate(highlight_ids):
    sel = id_series==hid
    ax.scatter(scores.loc[sel,'Spillovers Score'],
               scores.loc[sel,'SDG Index'],
               s=100, edgecolor='white',
               color=palette[i], alpha=0.9,
               label=highlight_names[hid])

# median lines
x_med = scores['Spillovers Score'].median()
y_med = scores['SDG Index'].median()
ax.axvline(x_med, color='gray', linestyle='--', linewidth=1)
ax.axhline(y_med, color='gray', linestyle='--', linewidth=1)

# quadrant labels
x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
dx, dy = (x1-x0)*0.05, (y1-y0)*0.05
ax.text(x0+dx, y1-dy, 'High SDG\nNegative Spillover',
        ha='left', va='top', weight='bold')
ax.text(x1-dx, y1-dy, 'High SDG\nPositive Spillover',
        ha='right', va='top', weight='bold')
ax.text(x0+dx, y0+dy, 'Low SDG\nNegative Spillover',
        ha='left', va='bottom', weight='bold')
ax.text(x1-dx, y0+dy, 'Low SDG\nPositive Spillover',
        ha='right', va='bottom', weight='bold')

# 4. Reposition legend outside plot
ax.legend(title='Income Group',
          loc='upper left',
          bbox_to_anchor=(1.02,1),
          borderaxespad=0.5,
          frameon=True,
          edgecolor='gray')

# 5. Highlight individual countries by quadrant
high_sdg_neg = ['Country A','Country B']
high_sdg_pos = ['Country C','Country D']
low_sdg_neg  = ['Country E']
low_sdg_pos  = ['Country F','Country G']

def highlight(countries, mk, tk):
    sel = df['Country'].isin(countries) & ok
    xs = scores.loc[sel,'Spillovers Score']
    ys = scores.loc[sel,'SDG Index']
    ax.scatter(xs, ys, **mk)
    for x,y,name in zip(xs,ys,df.loc[sel,'Country']):
        ax.text(x,y,name, **tk)

# larger colored markers + labels
highlight(high_sdg_neg,
          mk={'s':200,'edgecolor':'k','color':'red','alpha':0.9},
          tk={'ha':'right','va':'bottom','fontsize':10,'weight':'bold'})
highlight(high_sdg_pos,
          mk={'s':200,'edgecolor':'k','color':'green','alpha':0.9},
          tk={'ha':'left','va':'bottom','fontsize':10,'weight':'bold'})
highlight(low_sdg_neg,
          mk={'s':200,'edgecolor':'k','color':'blue','alpha':0.9},
          tk={'ha':'right','va':'top','fontsize':10,'weight':'bold'})
highlight(low_sdg_pos,
          mk={'s':200,'edgecolor':'k','color':'orange','alpha':0.9},
          tk={'ha':'left','va':'top','fontsize':10,'weight':'bold'})

# final touches
ax.set_xlabel('International Spillovers Score', fontsize=14)
ax.set_ylabel('SDG Index',               fontsize=14)
ax.set_title('SDG vs Spillovers by Income Group', fontsize=16, pad=20)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load & clean
df = pd.read_excel('SDR2024-data.xlsx', sheet_name='data')
df.columns = df.columns.str.strip()

# 2. Numeric scores & initial mask
scores = df[['Spillovers Score', 'SDG Index']].apply(pd.to_numeric, errors='coerce')
valid_idx = scores.dropna().index
scores = scores.loc[valid_idx]
id_series = df.loc[valid_idx, 'id']

# 3. Remove outliers (|z| > 3)
z = (scores - scores.mean()) / scores.std(ddof=0)
ok = (z.abs() <= 3).all(axis=1)
scores = scores[ok]
id_series = id_series[ok]

# 4. Compute medians
x_med = scores['Spillovers Score'].median()
y_med = scores['SDG Index'].median()

# 5. Set up figure
fig, ax = plt.subplots(figsize=(10, 8))

# 6. Plot all points in grey
ax.scatter(
    scores['Spillovers Score'],
    scores['SDG Index'],
    color='lightgrey',
    alpha=0.4,
    s=50,
    label='_nolegend_'
)

# 7. Median lines
ax.axvline(x_med, color='gray', linestyle='--', linewidth=1)
ax.axhline(y_med, color='gray', linestyle='--', linewidth=1)

# 8. Quadrant labels
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
dx, dy = (x1 - x0) * 0.05, (y1 - y0) * 0.05

ax.text(x0 + dx, y1 - dy, 'High SDG\nNegative Spillover',
        ha='left',  va='top',    weight='bold')
ax.text(x1 - dx, y1 - dy, 'High SDG\nPositive Spillover',
        ha='right', va='top',    weight='bold')
ax.text(x0 + dx, y0 + dy, 'Low SDG\nNegative Spillover',
        ha='left',  va='bottom', weight='bold')
ax.text(x1 - dx, y0 + dy, 'Low SDG\nPositive Spillover',
        ha='right', va='bottom', weight='bold')

# 9. Income‐group summary: big dots + on‐plot labels
highlight_ids   = ['_LIC', '_LMIC', '_UMIC', '_HIC']
highlight_names = {
    '_LIC':  'Low-income Countries',
    '_LMIC': 'Lower-middle-income Countries',
    '_UMIC': 'Upper-middle-income Countries',
    '_HIC':  'High-income Countries'
}
palette = plt.cm.Set1.colors

for i, hid in enumerate(highlight_ids):
    sel = id_series == hid
    xg = scores.loc[sel, 'Spillovers Score']
    yg = scores.loc[sel, 'SDG Index']
    # big dot
    ax.scatter(
        xg, yg,
        s=300,
        edgecolor='white',
        color=palette[i],
        alpha=0.9,
        label=highlight_names[hid]
    )
    # label next to dot
    for x, y in zip(xg, yg):
        ax.annotate(
            highlight_names[hid],
            xy=(x, y),
            xytext=(10, 5),
            textcoords='offset points',
            ha='left', va='bottom',
            fontsize=12, fontweight='bold',
            color=palette[i]
        )

# 10. Move legend outside
ax.legend(
    title='Income Group',
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.5,
    frameon=True,
    edgecolor='gray'
)

# 11. Compute distance from the median
scores['dist'] = np.hypot(
    scores['Spillovers Score'] - x_med,
    scores['SDG Index']          - y_med
)

# 12. Define quadrant masks
q1 = (scores['Spillovers Score'] > x_med)  & (scores['SDG Index'] > y_med)
q2 = (scores['Spillovers Score'] <= x_med) & (scores['SDG Index'] > y_med)
q3 = (scores['Spillovers Score'] <= x_med) & (scores['SDG Index'] <= y_med)
q4 = (scores['Spillovers Score'] > x_med)  & (scores['SDG Index'] <= y_med)
quad_masks  = [q1, q2, q3, q4]
quad_colors = ['red', 'green', 'blue', 'orange']

# 13. Label top-2 “extreme” countries in each quadrant
for mask, color in zip(quad_masks, quad_colors):
    top2 = scores[mask].nlargest(2, 'dist')
    for idx, row in top2.iterrows():
        x, y = row['Spillovers Score'], row['SDG Index']
        country = df.loc[idx, 'Country']  # adjust if your country column is named differently
        ax.scatter(
            x, y,
            s=250,
            edgecolor='black',
            color=color,
            alpha=0.8
        )
        ax.text(
            x, y, country,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
        )

# 14. Final formatting
ax.set_xlabel('International Spillovers Score', fontsize=14)
ax.set_ylabel('SDG Index',                fontsize=14)
ax.set_title('SDG Index vs. International Spillovers by Income Group',
             fontsize=16, pad=20)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load & clean
df = pd.read_excel('SDR2024-data.xlsx', sheet_name='data')
df.columns = df.columns.str.strip()

# 2. Numeric scores & initial mask
scores = df[['Spillovers Score', 'SDG Index']].apply(pd.to_numeric, errors='coerce')
valid_idx = scores.dropna().index
scores = scores.loc[valid_idx]
id_series = df.loc[valid_idx, 'id']

# 3. Remove outliers (|z| > 3)
z = (scores - scores.mean()) / scores.std(ddof=0)
ok = (z.abs() <= 3).all(axis=1)
scores = scores[ok]
id_series = id_series[ok]

# 4. Compute medians
x_med = scores['Spillovers Score'].median()
y_med = scores['SDG Index'].median()

# 5. Set up figure
fig, ax = plt.subplots(figsize=(10, 8))

# 6. Plot all points in grey
ax.scatter(
    scores['Spillovers Score'],
    scores['SDG Index'],
    color='lightgrey',
    alpha=0.4,
    s=50
)

# 7. Median lines
ax.axvline(x_med, color='gray', linestyle='--', linewidth=1)
ax.axhline(y_med, color='gray', linestyle='--', linewidth=1)

# 8. Quadrant labels
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
dx, dy = (x1 - x0) * 0.05, (y1 - y0) * 0.05

ax.text(x0 + dx, y1 - dy, 'High SDG\nNegative Spillover',
        ha='left',  va='top',    weight='bold')
ax.text(x1 - dx, y1 - dy, 'High SDG\nPositive Spillover',
        ha='right', va='top',    weight='bold')
ax.text(x0 + dx, y0 + dy, 'Low SDG\nNegative Spillover',
        ha='left',  va='bottom', weight='bold')
ax.text(x1 - dx, y0 + dy, 'Low SDG\nPositive Spillover',
        ha='right', va='bottom', weight='bold')

# 9. Income-group summary: big dots + on-plot labels
highlight_ids   = ['_LIC', '_LMIC', '_UMIC', '_HIC']
highlight_names = {
    '_LIC':  'Low-income\nCountries',
    '_LMIC': 'Lower-middle-income\nCountries',
    '_UMIC': 'Upper-middle-income\nCountries',
    '_HIC':  'High-income\nCountries'
}
palette = plt.cm.Set1.colors

for i, hid in enumerate(highlight_ids):
    sel = id_series == hid
    xg = scores.loc[sel, 'Spillovers Score']
    yg = scores.loc[sel, 'SDG Index']
    # big dot
    ax.scatter(
        xg, yg,
        s=300,
        edgecolor='white',
        color=palette[i],
        alpha=0.9
    )
    # label next to dot
    for x, y in zip(xg, yg):
        ax.annotate(
            highlight_names[hid],
            xy=(x, y),
            xytext=(10, 5),
            textcoords='offset points',
            ha='left', va='bottom',
            fontsize=12, fontweight='bold',
            color=palette[i]
        )

# 10. Compute distance from the median
scores['dist'] = np.hypot(
    scores['Spillovers Score'] - x_med,
    scores['SDG Index']          - y_med
)

# 11. Define quadrant masks
q1 = (scores['Spillovers Score'] > x_med)  & (scores['SDG Index'] > y_med)
q2 = (scores['Spillovers Score'] <= x_med) & (scores['SDG Index'] > y_med)
q3 = (scores['Spillovers Score'] <= x_med) & (scores['SDG Index'] <= y_med)
q4 = (scores['Spillovers Score'] > x_med)  & (scores['SDG Index'] <= y_med)
quad_masks  = [q1, q2, q3, q4]
quad_colors = ['red', 'green', 'blue', 'orange']

# 12. Label top-2 “extreme” countries in each quadrant
for mask, color in zip(quad_masks, quad_colors):
    top2 = scores[mask].nlargest(2, 'dist')
    for idx, row in top2.iterrows():
        x, y = row['Spillovers Score'], row['SDG Index']
        country = df.loc[idx, 'Country']  # adjust if named differently
        ax.scatter(
            x, y,
            s=250,
            edgecolor='black',
            color=color,
            alpha=0.8
        )
        ax.text(
            x, y, country,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
        )

# 13. Final formatting
ax.set_xlabel('International Spillovers Score', fontsize=14)
ax.set_ylabel('SDG Index',                fontsize=14)
ax.set_title('SDG Index vs. International Spillovers by Income Group',
             fontsize=16, pad=20)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load & clean
df = pd.read_excel('SDR2024-data.xlsx', sheet_name='data')
df.columns = df.columns.str.strip()

# 2. Numeric scores & initial mask
scores = df[['Spillovers Score', 'SDG Index']].apply(pd.to_numeric, errors='coerce')
valid_idx = scores.dropna().index
scores = scores.loc[valid_idx]
id_series = df.loc[valid_idx, 'id']

# 3. Remove outliers (|z| > 3)
z = (scores - scores.mean()) / scores.std(ddof=0)
ok = (z.abs() <= 3).all(axis=1)
scores = scores[ok]
id_series = id_series[ok]

# 4. Compute medians
x_med = scores['Spillovers Score'].median()
y_med = scores['SDG Index'].median()

# 5. Set up figure
fig, ax = plt.subplots(figsize=(10, 8))

# 6. Plot all points in grey
ax.scatter(
    scores['Spillovers Score'],
    scores['SDG Index'],
    color='lightgrey',
    alpha=0.4,
    s=50
)

# 7. Median lines
ax.axvline(x_med, color='gray', linestyle='--', linewidth=1)
ax.axhline(y_med, color='gray', linestyle='--', linewidth=1)

# 8. Quadrant labels
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
dx, dy = (x1 - x0) * 0.05, (y1 - y0) * 0.05

ax.text(x0 + dx, y1 - dy, 'High SDG\nNegative Spillover',
        ha='left',  va='top',    weight='bold')
ax.text(x1 - dx, y1 - dy, 'High SDG\nPositive Spillover',
        ha='right', va='top',    weight='bold')
ax.text(x0 + dx, y0 + dy, 'Low SDG\nNegative Spillover',
        ha='left',  va='bottom', weight='bold')
ax.text(x1 - dx, y0 + dy, 'Low SDG\nPositive Spillover',
        ha='right', va='bottom', weight='bold')

# 9. Income-group summary: big dots + on-plot labels
highlight_ids   = ['_LIC', '_LMIC', '_UMIC', '_HIC']
highlight_names = {
    '_LIC':  'Low-income\nCountries',
    '_LMIC': 'Lower-middle-income\nCountries',
    '_UMIC': 'Upper-middle-income\nCountries',
    '_HIC':  'High-income\nCountries'
}
palette = plt.cm.Set1.colors

for i, hid in enumerate(highlight_ids):
    sel = id_series == hid
    xg = scores.loc[sel, 'Spillovers Score']
    yg = scores.loc[sel, 'SDG Index']
    # big dot
    ax.scatter(
        xg, yg,
        s=300,
        edgecolor='white',
        color=palette[i],
        alpha=0.9
    )
    # label next to dot
    for x, y in zip(xg, yg):
        ax.annotate(
            highlight_names[hid],
            xy=(x, y),
            xytext=(10, 5),
            textcoords='offset points',
            ha='left', va='bottom',
            fontsize=12, fontweight='bold',
            color=palette[i]
        )

# 10. Compute distance from the median
scores['dist'] = np.hypot(
    scores['Spillovers Score'] - x_med,
    scores['SDG Index']          - y_med
)

# 11. Define quadrant masks
q1 = (scores['Spillovers Score'] > x_med)  & (scores['SDG Index'] > y_med)
q2 = (scores['Spillovers Score'] <= x_med) & (scores['SDG Index'] > y_med)
q3 = (scores['Spillovers Score'] <= x_med) & (scores['SDG Index'] <= y_med)
q4 = (scores['Spillovers Score'] > x_med)  & (scores['SDG Index'] <= y_med)
quad_masks  = [q1, q2, q3, q4]
quad_colors = ['red', 'green', 'blue', 'orange']

# 12. Label top-2 “extreme” countries in each quadrant
for mask, color in zip(quad_masks, quad_colors):
    top2 = scores[mask].nlargest(2, 'dist')
    for idx, row in top2.iterrows():
        x, y = row['Spillovers Score'], row['SDG Index']
        country = df.loc[idx, 'Country']  # adjust if named differently
        ax.scatter(
            x, y,
            s=250,
            edgecolor='black',
            color=color,
            alpha=0.8
        )
        ax.text(
            x, y, country,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
        )

# 13. Final formatting
ax.set_xlabel('International Spillovers Score', fontsize=14)
ax.set_ylabel('SDG Index',                fontsize=14)
ax.set_title('SDG Index vs. International Spillovers by Income Group',
             fontsize=16, pad=20)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load & clean
df = pd.read_excel('SDR2024-data.xlsx', sheet_name='data')
df.columns = df.columns.str.strip()
df['Country'] = df['Country'].str.strip()   # ensure no stray spaces

# 2. Numeric scores & initial mask
scores = df[['Spillovers Score', 'SDG Index']].apply(pd.to_numeric, errors='coerce')
valid_idx = scores.dropna().index
scores = scores.loc[valid_idx].copy()
id_series = df.loc[valid_idx, 'id']

# 3. Remove outliers (|z| > 3)
z = (scores - scores.mean()) / scores.std(ddof=0)
ok = (z.abs() <= 3).all(axis=1)
scores = scores[ok].copy()
id_series = id_series[ok]

# 4. Compute medians
x_med = scores['Spillovers Score'].median()
y_med = scores['SDG Index'].median()

# 5. Set up figure
fig, ax = plt.subplots(figsize=(10, 8))

# 6. Plot all points in grey
ax.scatter(
    scores['Spillovers Score'],
    scores['SDG Index'],
    color='lightgrey',
    alpha=0.4,
    s=50
)

# 7. Median lines
ax.axvline(x_med, color='gray', linestyle='--', linewidth=1)
ax.axhline(y_med, color='gray', linestyle='--', linewidth=1)

# 8. Quadrant labels
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
dx, dy = (x1-x0)*0.05, (y1-y0)*0.05

ax.text(x0+dx, y1-dy, 'High SDG\nNegative Spillover', ha='left',  va='top',    weight='bold')
ax.text(x1-dx, y1-dy, 'High SDG\nPositive Spillover', ha='right', va='top',    weight='bold')
ax.text(x0+dx, y0+dy, 'Low SDG\nNegative Spillover',  ha='left',  va='bottom', weight='bold')
ax.text(x1-dx, y0+dy, 'Low SDG\nPositive Spillover',  ha='right', va='bottom', weight='bold')

# 9. Income-group summary: big dots + on-plot labels (no legend)
highlight_ids   = ['_LIC', '_LMIC', '_UMIC', '_HIC']
highlight_names = {
    '_LIC':  'Low-income\nCountries',
    '_LMIC': 'Lower-middle-income\nCountries',
    '_UMIC': 'Upper-middle-income\nCountries',
    '_HIC':  'High-income\nCountries'
}
palette = plt.cm.Set1.colors

for i, hid in enumerate(highlight_ids):
    sel = (id_series == hid)
    xg = scores.loc[sel, 'Spillovers Score']
    yg = scores.loc[sel, 'SDG Index']
    ax.scatter(xg, yg, s=300, edgecolor='white', color=palette[i], alpha=0.9)
    for x, y in zip(xg, yg):
        ax.annotate(
            highlight_names[hid],
            xy=(x, y),
            xytext=(10, 5),
            textcoords='offset points',
            ha='left', va='bottom',
            fontsize=12, fontweight='bold',
            color=palette[i]
        )

# 10. Compute distance from the median
scores['dist'] = np.hypot(
    scores['Spillovers Score'] - x_med,
    scores['SDG Index']          - y_med
)

# 11. Define quadrant masks & colors
q1 = (scores['Spillovers Score'] > x_med)  & (scores['SDG Index'] > y_med)
q2 = (scores['Spillovers Score'] <= x_med) & (scores['SDG Index'] > y_med)
q3 = (scores['Spillovers Score'] <= x_med) & (scores['SDG Index'] <= y_med)
q4 = (scores['Spillovers Score'] > x_med)  & (scores['SDG Index'] <= y_med)
quad_masks  = [q1, q2, q3, q4]
quad_colors = ['red', 'green', 'blue', 'orange']

# 12. Label countries per quadrant
# Q1: manually keep Cuba + add China, skip Uruguay
for country in ['Cuba', 'China']:
    sel = (df['Country'].str.lower() == country.lower()) & q1
    sel = sel & df.index.isin(scores.index)   # ensure it's in our scored set
    if sel.any():
        idx = sel[sel].index[0]
        x, y = scores.at[idx, 'Spillovers Score'], scores.at[idx, 'SDG Index']
        ax.scatter(x, y, s=250, edgecolor='black', color=quad_colors[0], alpha=0.8)
        ax.text(
            x, y, country,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
        )

# Q2–Q4: pick top-2 by distance, excluding Argentina
for mask, color in zip(quad_masks[1:], quad_colors[1:]):
    candidates = scores[mask].copy()
    # drop Argentina
    is_arg = df.loc[candidates.index, 'Country'].str.lower() == 'argentina'
    candidates = candidates.loc[~is_arg]
    top2 = candidates.nlargest(2, 'dist')
    for idx, row in top2.iterrows():
        x, y = row['Spillovers Score'], row['SDG Index']
        country = df.at[idx, 'Country']
        ax.scatter(x, y, s=250, edgecolor='black', color=color, alpha=0.8)
        ax.text(
            x, y, country,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
        )

# 13. Explicitly highlight Djibouti (if present)
dj_sel = df['Country'].str.lower() == 'djibouti'
dj_sel = dj_sel & df.index.isin(scores.index)
if dj_sel.any():
    idx = dj_sel[dj_sel].index[0]
    x, y = scores.at[idx, 'Spillovers Score'], scores.at[idx, 'SDG Index']
    ax.scatter(x, y, s=300, edgecolor='black', color='purple', alpha=0.9)
    ax.text(
        x, y, 'Djibouti',
        ha='center', va='center',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)
    )

# 14. Final formatting
ax.set_xlabel('International Spillovers Score', fontsize=14)
ax.set_ylabel('SDG Index',                fontsize=14)
ax.set_title('SDG Index vs. International Spillovers by Income Group',
             fontsize=16, pad=20)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
