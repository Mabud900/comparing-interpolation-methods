"""
Data Interpolation for Missing Values
======================================
A research-grade Streamlit app comparing 5 interpolation methods:
Linear | Polynomial | Cubic Spline | Lagrange | Newton Forward Difference

Author: Chandigarh University — Numerical Methods Project (23SMH-341)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import sys
import os

sys.path.append(os.path.dirname(__file__))
from interpolators import apply_all_methods, evaluate_methods, METHODS

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataFill · Interpolation Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — Dark scientific aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1628 50%, #0a1220 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.15);
}

/* Header hero */
.hero-header {
    background: linear-gradient(135deg, rgba(56,189,248,0.08) 0%, rgba(99,102,241,0.08) 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid rgba(56,189,248,0.2);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Method badge */
.method-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    margin: 0.15rem;
}

/* Best method highlight */
.best-method {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(56,189,248,0.1));
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

/* Info box */
.info-box {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #818cf8;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 0.8rem 0;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stSlider"] label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-family: 'Space Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(14,165,233,0.3) !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 10px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #64748b !important;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOT_COLORS = ["#38bdf8", "#f472b6", "#34d399", "#fb923c", "#a78bfa", "#facc15"]
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(10,14,26,0.9)"
GRID_COLOR = "rgba(56,189,248,0.08)"
AXIS_COLOR = "#334155"

def styled_fig(title=""):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne", size=15, color="#e2e8f0")),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Space Mono, monospace", color="#94a3b8", size=11),
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="rgba(56,189,248,0.15)",
            borderwidth=1,
        ),
        margin=dict(t=50, b=40, l=50, r=20),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=AXIS_COLOR, color="#64748b"),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=AXIS_COLOR, color="#64748b"),
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem 0;">
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.2rem;
                    background:linear-gradient(135deg,#38bdf8,#818cf8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🔬 DataFill Lab
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:0.65rem;
                    color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-top:0.3rem;">
            Numerical Methods · 23SMH-341
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">📂 Data Source</p>', unsafe_allow_html=True)
    
    data_source = st.radio(
        "Choose input:",
        ["Upload CSV", "Use Sample Dataset"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Drop your CSV here",
            type=["csv"],
            help="Needs a numeric index/X column + one or more value columns with NaN gaps."
        )
    
    st.markdown("---")
    st.markdown('<p class="section-title">⚙️ Settings</p>', unsafe_allow_html=True)
    
    selected_methods = st.multiselect(
        "Methods to compare",
        list(METHODS.keys()),
        default=list(METHODS.keys()),
    )

    show_error_analysis = st.checkbox("Show Error Analysis", value=True)
    show_diff_table = st.checkbox("Show Difference Table", value=False)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#334155; line-height:1.8;">
        METHODS<br>
        ├── Linear<br>
        ├── Polynomial (deg 3)<br>
        ├── Cubic Spline<br>
        ├── Lagrange<br>
        └── Newton Forward<br>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Data Interpolation Lab</div>
    <div class="hero-subtitle">Numerical Methods · Missing Value Recovery · Research Comparison Tool</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_sample():
    path = os.path.join(os.path.dirname(__file__), "sample_data", "sensor_data_with_missing.csv")
    return pd.read_csv(path)

df_raw = None

if data_source == "Use Sample Dataset":
    df_raw = load_sample()
    st.markdown("""
    <div class="info-box">
        📡 <b>Sample Dataset</b>: IoT sensor readings (temperature, humidity, pressure) over 50 time steps with ~20% missing values introduced randomly. 
        This simulates real sensor drop-out scenarios.
    </div>
    """, unsafe_allow_html=True)
elif uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
if df_raw is None:
    # Landing / instructions
    col1, col2, col3 = st.columns(3)
    cards = [
        ("📤", "Upload Your Data", "CSV with any numeric columns. Missing values as blank or NaN."),
        ("⚗️", "Compare Methods", "5 interpolation algorithms run simultaneously on your data."),
        ("📊", "Get Insights", "Visual comparison + error metrics to find the best method."),
    ]
    for c, (icon, title, desc) in zip([col1, col2, col3], cards):
        with c:
            st.markdown(f"""
            <div class="metric-card" style="padding:2rem; text-align:center;">
                <div style="font-size:2.5rem;">{icon}</div>
                <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; 
                            color:#e2e8f0; margin:0.8rem 0 0.4rem;">{title}</div>
                <div style="font-size:0.8rem; color:#475569; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        💡 <b>Quick Start</b>: Select <b>"Use Sample Dataset"</b> from the sidebar to explore immediately, 
        or upload your own CSV to analyze your data.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Detect columns ────────────────────────────────────────────────────────
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns (index/X + one value column).")
    st.stop()

st.markdown('<p class="section-title">🗂️ Dataset Overview</p>', unsafe_allow_html=True)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
total_cells = df_raw[numeric_cols].size
missing_cells = df_raw[numeric_cols].isnull().sum().sum()
pct_missing = round(100 * missing_cells / total_cells, 1)

for col_w, val, label in zip(
    [col_m1, col_m2, col_m3, col_m4],
    [len(df_raw), len(numeric_cols), missing_cells, f"{pct_missing}%"],
    ["Rows", "Numeric Columns", "Missing Cells", "Missing %"]
):
    with col_w:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Show raw data preview
with st.expander("👁️ Preview Raw Data", expanded=False):
    st.dataframe(
        df_raw.style.highlight_null(color="rgba(239,68,68,0.2)"),
        use_container_width=True,
        height=250
    )


# ─── Column selector ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-title">🎯 Column Selection</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 2])
with col_left:
    x_col = st.selectbox("X-axis / Index column", numeric_cols, index=0)
with col_right:
    y_cols = st.multiselect(
        "Value columns to interpolate",
        [c for c in numeric_cols if c != x_col],
        default=[c for c in numeric_cols if c != x_col][:1],
    )

if not y_cols:
    st.warning("Select at least one value column.")
    st.stop()


# ─────────────────────────────────────────────
# INTERPOLATION ENGINE
# ─────────────────────────────────────────────
if not selected_methods:
    st.warning("Select at least one method from the sidebar.")
    st.stop()

all_results = {}  # {y_col: {method: filled_series}}

for y_col in y_cols:
    series = df_raw.set_index(x_col)[y_col]
    all_results[y_col] = apply_all_methods(series)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs(["📈 Visualization", "📊 Error Analysis", "🔢 Filled Data", "📚 Method Guide"])


# ═══════════════════════════════
# TAB 1 — VISUALIZATION
# ═══════════════════════════════
with tabs[0]:
    for y_col in y_cols:
        st.markdown(f'<p class="section-title">Column: {y_col}</p>', unsafe_allow_html=True)

        series_orig = df_raw.set_index(x_col)[y_col]
        x_vals = series_orig.index.astype(float)

        fig = styled_fig(f"Interpolation Comparison — {y_col}")

        # Original known points
        known_mask = series_orig.notna()
        fig.add_trace(go.Scatter(
            x=x_vals[known_mask], y=series_orig[known_mask],
            mode="markers",
            marker=dict(color="#e2e8f0", size=6, symbol="circle",
                        line=dict(color="#38bdf8", width=1)),
            name="Known Values",
            zorder=10,
        ))

        # Missing point positions
        missing_mask = series_orig.isna()
        if missing_mask.any():
            fig.add_trace(go.Scatter(
                x=x_vals[missing_mask],
                y=[series_orig.dropna().mean()] * missing_mask.sum(),
                mode="markers",
                marker=dict(color="rgba(239,68,68,0.5)", size=8, symbol="x"),
                name="Missing (position)",
            ))

        # Each interpolation method
        method_results = all_results[y_col]
        for i, method_name in enumerate(selected_methods):
            if method_name not in method_results:
                continue
            filled = method_results[method_name]
            color = PLOT_COLORS[i % len(PLOT_COLORS)]
            
            # Draw the line
            fig.add_trace(go.Scatter(
                x=x_vals, y=filled.values,
                mode="lines",
                line=dict(color=color, width=2, dash="solid"),
                name=method_name,
                opacity=0.8,
            ))
            # Highlight filled-in points
            if missing_mask.any():
                fig.add_trace(go.Scatter(
                    x=x_vals[missing_mask], y=filled[missing_mask].values,
                    mode="markers",
                    marker=dict(color=color, size=9, symbol="diamond",
                                line=dict(color="white", width=1)),
                    name=f"{method_name} (filled)",
                    showlegend=False,
                ))

        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Missing count info
        n_missing = missing_mask.sum()
        st.markdown(f"""
        <div class="info-box">
            ✅ <b>{n_missing} missing value(s)</b> recovered in <i>{y_col}</i> using 
            {len(selected_methods)} method(s). Diamond markers show the filled positions.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════
# TAB 2 — ERROR ANALYSIS
# ═══════════════════════════════
with tabs[1]:
    if not show_error_analysis:
        st.info("Enable Error Analysis in the sidebar to see this tab.")
    else:
        st.markdown("""
        <div class="info-box">
            📐 <b>How it works</b>: We hide 20% of the <i>known</i> values, run each method, then compare 
            predictions vs actual to compute MAE and RMSE. Lower = better.
        </div>
        """, unsafe_allow_html=True)

        for y_col in y_cols:
            st.markdown(f'<p class="section-title">Error Metrics — {y_col}</p>', unsafe_allow_html=True)
            
            series_orig = df_raw.set_index(x_col)[y_col]
            metrics_df = evaluate_methods(series_orig, all_results[y_col])
            
            if metrics_df.empty:
                st.warning("Not enough known values for error analysis (need ≥ 10).")
                continue

            # Bar chart
            fig_err = styled_fig("MAE & RMSE by Method")
            methods_list = metrics_df["Method"].tolist()
            
            fig_err.add_trace(go.Bar(
                name="MAE", x=methods_list, y=metrics_df["MAE"],
                marker_color=PLOT_COLORS[0], opacity=0.85,
                text=metrics_df["MAE"].round(4), textposition="outside",
                textfont=dict(color="#e2e8f0", size=10)
            ))
            fig_err.add_trace(go.Bar(
                name="RMSE", x=methods_list, y=metrics_df["RMSE"],
                marker_color=PLOT_COLORS[1], opacity=0.85,
                text=metrics_df["RMSE"].round(4), textposition="outside",
                textfont=dict(color="#e2e8f0", size=10)
            ))
            fig_err.update_layout(barmode="group", height=380)
            st.plotly_chart(fig_err, use_container_width=True)

            # Table
            col_tbl, col_winner = st.columns([2, 1])
            with col_tbl:
                st.dataframe(
                    metrics_df.set_index("Method").style
                        .background_gradient(cmap="Blues_r", subset=["RMSE"])
                        .format("{:.4f}"),
                    use_container_width=True
                )
            with col_winner:
                best = metrics_df.iloc[0]["Method"]
                best_rmse = metrics_df.iloc[0]["RMSE"]
                st.markdown(f"""
                <div class="best-method">
                    <div style="font-family:'Space Mono',monospace; font-size:0.7rem; 
                                color:#22c55e; text-transform:uppercase; letter-spacing:0.1em;">
                        🏆 Best Method
                    </div>
                    <div style="font-family:'Syne',sans-serif; font-weight:800; 
                                font-size:1.4rem; color:#e2e8f0; margin:0.5rem 0;">
                        {best}
                    </div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#64748b;">
                        RMSE: {best_rmse:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════
# TAB 3 — FILLED DATA
# ═══════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-title">📥 Download Interpolated Datasets</p>', unsafe_allow_html=True)

    method_choice = st.selectbox("Select method for export:", selected_methods)

    # Build the filled DataFrame
    df_filled = df_raw.copy()
    for y_col in y_cols:
        filled_series = all_results[y_col][method_choice]
        df_filled[y_col] = df_filled[x_col].map(
            dict(zip(filled_series.index.astype(float), filled_series.values))
        ).fillna(df_raw[y_col])

    # Show side by side
    col_orig, col_fill = st.columns(2)
    with col_orig:
        st.markdown("**Original (with NaN)**")
        st.dataframe(
            df_raw[y_cols].style.highlight_null(color="rgba(239,68,68,0.15)"),
            height=300, use_container_width=True
        )
    with col_fill:
        st.markdown(f"**After {method_choice} interpolation**")
        st.dataframe(df_filled[y_cols], height=300, use_container_width=True)

    # Download
    csv_buffer = io.StringIO()
    df_filled.to_csv(csv_buffer, index=False)
    st.download_button(
        label=f"⬇️ Download CSV ({method_choice})",
        data=csv_buffer.getvalue(),
        file_name=f"interpolated_{method_choice.replace(' ','_').lower()}.csv",
        mime="text/csv",
    )

    if show_diff_table:
        st.markdown('<p class="section-title">Δ Difference (Filled − Original)</p>', unsafe_allow_html=True)
        diff_df = pd.DataFrame()
        for y_col in y_cols:
            diff_df[y_col] = (df_filled[y_col] - df_raw[y_col]).round(4)
        st.dataframe(
            diff_df.style.background_gradient(cmap="RdYlGn", axis=None),
            use_container_width=True, height=250
        )


# ═══════════════════════════════
# TAB 4 — METHOD GUIDE
# ═══════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-title">📚 Research Guide to Interpolation Methods</p>', unsafe_allow_html=True)

    methods_info = {
        "Linear": {
            "emoji": "📏",
            "formula": "f(x) = y₀ + (x − x₀) · (y₁ − y₀) / (x₁ − x₀)",
            "description": "Connects consecutive known points with straight lines. The simplest method.",
            "best_for": "Uniformly sampled data with slow variation",
            "weakness": "Creates sharp kinks at data points; poor for curved data",
            "complexity": "O(n)",
        },
        "Polynomial (deg 3)": {
            "emoji": "🔵",
            "formula": "P(x) = a₀ + a₁x + a₂x² + a₃x³",
            "description": "Fits a single polynomial through known points. Degree 3 balances fit and stability.",
            "best_for": "Smooth, curve-like data with moderate variation",
            "weakness": "Runge's phenomenon at high degrees; overfits with too many points",
            "complexity": "O(n²)",
        },
        "Cubic Spline": {
            "emoji": "🌊",
            "formula": "Piecewise cubics with C² continuity (zero 2nd deriv at endpoints)",
            "description": "Divides range into segments, fits a cubic on each ensuring smooth joins. Widely used in practice.",
            "best_for": "Any smooth continuous data; best general-purpose choice",
            "weakness": "Slightly complex; can oscillate at boundaries",
            "complexity": "O(n)",
        },
        "Lagrange": {
            "emoji": "🔶",
            "formula": "L(x) = Σ yᵢ · ∏ (x − xⱼ)/(xᵢ − xⱼ), j≠i",
            "description": "Classic polynomial passing through all points. This app uses a local window of 8 nearest points to avoid Runge's phenomenon.",
            "best_for": "Small, well-spaced datasets; educational and research use",
            "weakness": "Very sensitive to outliers; Runge's phenomenon at high order",
            "complexity": "O(n²)",
        },
        "Newton Forward": {
            "emoji": "⚡",
            "formula": "N(x) = Σ Δⁿf(x₀) · C(s,n) where s = (x−x₀)/h",
            "description": "Uses finite differences to construct the interpolating polynomial. Efficient for equally-spaced data.",
            "best_for": "Uniformly spaced data; adding new points incrementally",
            "weakness": "Accumulates round-off error; poor for unequal spacing",
            "complexity": "O(n²) construction, O(n) evaluation",
        },
    }

    for method, info in methods_info.items():
        with st.expander(f"{info['emoji']} {method}", expanded=False):
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown(f"**Description**: {info['description']}")
                st.markdown(f"**Formula**: `{info['formula']}`")
                st.markdown(f"**Best for**: {info['best_for']}")
                st.markdown(f"**Weakness**: {info['weakness']}")
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-family:'Space Mono',monospace; font-size:0.75rem; 
                                color:#38bdf8;">Time Complexity</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin:0.5rem 0;">
                        {info['complexity']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:2rem;">
        🔬 <b>Research Directions</b>: Compare methods on your data's characteristics (smoothness, sampling rate, 
        noise level). Investigate how polynomial degree affects Runge's phenomenon. Study error propagation 
        in Newton's Divided Differences. Explore adaptive splines for non-uniform data.
    </div>
    """, unsafe_allow_html=True)
