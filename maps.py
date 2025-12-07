import os
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ---------------------------------------------------------------------
# CONFIG FLAGS
# ---------------------------------------------------------------------
AUTO_OPEN_HTML = True         # Set to False if browser opening is slow
ENABLE_PNG_EXPORT = True      # Set to False if PNG export is too slow / problematic

# ---------------------------------------------------------------------
# Helper for timing
# ---------------------------------------------------------------------
def step(msg, t0):
    """Print step message with elapsed time."""
    t1 = time.time()
    print(f"{msg}  (elapsed: {t1 - t0:.2f} s)")
    return t1

print("Working directory:", os.getcwd())
t0 = time.time()

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
print("\n[STEP 1] Reading Excel file...")
df = pd.read_excel("spill over & index_updated.xlsx", sheet_name="Sheet1")
t0 = step("✔ Excel loaded", t0)
print("   Rows:", len(df), "   Columns:", list(df.columns))

# ---------------------------------------------------------------------
# 2. Build combined figure: two heatmaps
# ---------------------------------------------------------------------
print("\n[STEP 2] Building combined figure...")
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
    subplot_titles=["Spill-over effects score2", "2024 SDG Index Score"],
)

# LEFT: Spill-over
fig.add_trace(
    go.Choropleth(
        locations=df["Country Name"],
        locationmode="country names",
        z=df["Spill-over effects score"],
        coloraxis="coloraxis",  # LEFT colour axis
        hovertemplate="<b>%{location}</b><br>Spill-over: %{z:.1f}<extra></extra>",
    ),
    row=1, col=1,
)

# RIGHT: SDG Index
fig.add_trace(
    go.Choropleth(
        locations=df["Country Name"],
        locationmode="country names",
        z=df["2024 SDG Index Score"],
        coloraxis="coloraxis2",  # RIGHT colour axis
        hovertemplate="<b>%{location}</b><br>SDG Index 2024: %{z:.1f}<extra></extra>",
    ),
    row=1, col=2,
)

# Layout: SAME colourscale + SAME numeric range (0–100)
fig.update_layout(
    title_text="Spill-over effects vs 2024 SDG Index Score",
    margin=dict(l=0, r=0, t=40, b=0),

    # Left colour axis: Spill-over
    coloraxis=dict(
        colorscale="Viridis",   # same palette on both maps
        cmin=0,                 # common numeric scale
        cmax=100,
        colorbar=dict(
            title="Spill-over (0–100)",
            x=0.45,            # between the two plots
            y=0.5,
            len=0.8,
        ),
    ),

    # Right colour axis: SDG Index
    coloraxis2=dict(
        colorscale="Viridis",   # same palette
        cmin=0,                 # same numeric scale
        cmax=100,
        colorbar=dict(
            title="SDG Index (0–100)",
            x=1.02,             # to the right of the second map
            y=0.5,
            len=0.8,
        ),
    ),

    # Geographic layout for both subplots
    geo=dict(
        projection_type="natural earth",
        showcoastlines=False,
        showcountries=True,
        lataxis_range=[-60, 85],
    ),
    geo2=dict(
        projection_type="natural earth",
        showcoastlines=False,
        showcountries=True,
        lataxis_range=[-60, 85],
    ),
)

t0 = step("✔ Combined figure built", t0)

# ---------------------------------------------------------------------
# 3. Save interactive HTML
# ---------------------------------------------------------------------
print("\n[STEP 3] Saving HTML...")
html_file = "spillover_sdg_world_heatmaps_same_scale.html"
pio.write_html(fig, html_file, auto_open=AUTO_OPEN_HTML)
t0 = step(f"✔ HTML saved to: {os.path.abspath(html_file)}", t0)
if AUTO_OPEN_HTML:
    print("   (If it 'hangs' here, the delay is your browser starting.)")

# ---------------------------------------------------------------------
# 4. Save PNGs (requires kaleido)
# ---------------------------------------------------------------------
if ENABLE_PNG_EXPORT:
    print("\n[STEP 4] Attempting PNG export (this can be slow the first time)...")
    try:
        # Combined PNG with both maps
        png_combined = "spillover_sdg_world_heatmaps_same_scale.png"
        pio.write_image(fig, png_combined, width=1800, height=800, scale=2)
        t0 = step(f"✔ Combined PNG saved to: {os.path.abspath(png_combined)}", t0)

        # Optional: separate PNGs
        print("[STEP 4a] Exporting separate Spill-over PNG...")
        fig_spill = go.Figure(
            go.Choropleth(
                locations=df["Country Name"],
                locationmode="country names",
                z=df["Spill-over effects score"],
                colorscale="Viridis",
                cmin=0,
                cmax=100,
                colorbar_title="Spill-over (0–100)",
                hovertemplate="<b>%{location}</b><br>Spill-over: %{z:.1f}<extra></extra>",
            )
        )
        fig_spill.update_layout(
            title_text="Spill-over effects score",
            geo=dict(
                projection_type="natural earth",
                showcoastlines=False,
                showcountries=True,
                lataxis_range=[-60, 85],
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        png_spill = "spillover_heatmap_same_scale.png"
        pio.write_image(fig_spill, png_spill, width=900, height=600, scale=2)
        t0 = step(f"✔ Spill-over PNG saved to: {os.path.abspath(png_spill)}", t0)

        print("[STEP 4b] Exporting separate SDG PNG...")
        fig_sdg = go.Figure(
            go.Choropleth(
                locations=df["Country Name"],
                locationmode="country names",
                z=df["2024 SDG Index Score"],
                colorscale="Viridis",
                cmin=0,
                cmax=100,
                colorbar_title="SDG Index (0–100)",
                hovertemplate="<b>%{location}</b><br>SDG Index 2024: %{z:.1f}<extra></extra>",
            )
        )
        fig_sdg.update_layout(
            title_text="2024 SDG Index Score",
            geo=dict(
                projection_type="natural earth",
                showcoastlines=False,
                showcountries=True,
                lataxis_range=[-60, 85],
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        png_sdg = "sdg_index_heatmap_same_scale.png"
        pio.write_image(fig_sdg, png_sdg, width=900, height=600, scale=2)
        t0 = step(f"✔ SDG PNG saved to: {os.path.abspath(png_sdg)}", t0)

    except Exception as e:
        print("\n⚠️ PNG export failed.")
        print("   To enable PNG export, install kaleido in this environment:")
        print("       pip install -U kaleido")
        print("   Error was:", repr(e))
else:
    print("\n[STEP 4] PNG export skipped (ENABLE_PNG_EXPORT = False).")

print("\nAll done.")
