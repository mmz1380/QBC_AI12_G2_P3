"""Crash Detection - Part 1 - reporting dashboard.

    streamlit run app_project2.py

A Material-styled report of the whole project: the journey to the final model, a
tab per trained model (with a Keras-style architecture diagram and plain-English
notes on every technique), an error analysis you can filter and page through, a
reinforcement-learning deep-dive, and a live "upload a clip -> prediction" panel.
Light and dark themes with dedicated palettes; all charts Plotly, driven by
reports/metrics.json.
"""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
M = json.loads((ROOT / "reports" / "metrics.json").read_text()) if (ROOT / "reports" / "metrics.json").exists() else {}
PRED_CSV = ROOT / "reports" / "predictions_mvit.csv"
FINAL = M.get("final", {})
THRESHOLD = float(FINAL.get("threshold", 0.71))

st.set_page_config(page_title="QBC12 · Project 2 — Group 6", layout="wide", initial_sidebar_state="collapsed")

# ===========================================================================
#  Theme
# ===========================================================================
THEMES = {
    "Light": {
        "bg": "#eef1f6", "surface": "#ffffff", "surface2": "#f4f7fb", "text": "#1a2233", "muted": "#5b6675",
        "primary": "#1565c0", "divider": "#dde4ee",
        "shadow": "0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.10)",
        "shadow2": "0 6px 18px rgba(16,24,40,.14)",
        "appbar": "linear-gradient(135deg,#1565c0,#1e88e5)", "appbar_text": "#ffffff",
        "blue": "#1565c0", "green": "#2e7d32", "red": "#c62828", "amber": "#b26a00", "purple": "#6a1b9a", "teal": "#00838f",
        "grid": "#e6ebf3", "tmpl": "plotly_white",
        "palette": ["#1565c0", "#2e7d32", "#b26a00", "#6a1b9a", "#00838f", "#c62828", "#4a5568"],
        "hdr": "#334155", "frozen": "#d6dde7", "trainable": "#bcdcfb", "head": "#c6e7c9", "pool": "#fce9b6",
        "ensemble": "#e3c6ef", "input": "#dfe7f2",
    },
    "Dark": {
        "bg": "#0e131b", "surface": "#161c26", "surface2": "#1d2530", "text": "#e8ecf3", "muted": "#aab4c2",
        "primary": "#5b9bd5", "divider": "#2b333f",
        "shadow": "0 1px 3px rgba(0,0,0,.5)", "shadow2": "0 8px 22px rgba(0,0,0,.6)",
        "appbar": "linear-gradient(135deg,#13233b,#1c3a5e)", "appbar_text": "#eaf2fb",
        "blue": "#6db3f2", "green": "#7ed492", "red": "#ff6b6b", "amber": "#ffc061", "purple": "#c99df0", "teal": "#4dd0e1",
        "grid": "#283040", "tmpl": "plotly_dark",
        "palette": ["#6db3f2", "#7ed492", "#ffc061", "#c99df0", "#4dd0e1", "#ff6b6b", "#aab4c2"],
        "hdr": "#0b1220", "frozen": "#39424f", "trainable": "#1f4368", "head": "#22503a", "pool": "#5a4a1e",
        "ensemble": "#42304e", "input": "#232c3a",
    },
}
# Theme is applied client-side: the CSS below defines both palettes as variables
# (Light on :root, Dark on :root[data-theme="dark"]) and a small JS toggle flips
# that attribute + re-themes the Plotly charts, persisted in localStorage. So
# switching light/dark is instant and never reloads the app. The server always
# renders the Light palette for the first paint.
TN = "Light"
T = THEMES[TN]
L, D = THEMES["Light"], THEMES["Dark"]

_pp = st.query_params.get("part")
PART = _pp if _pp in ("1", "2") else st.session_state.get("part_choice", "1")
st.session_state["part_choice"] = PART

st.markdown(f"""
<style>
:root {{
  --bg:{L['bg']}; --surface:{L['surface']}; --surface2:{L['surface2']}; --text:{L['text']};
  --muted:{L['muted']}; --primary:{L['primary']}; --divider:{L['divider']}; --grid:{L['grid']};
  --shadow:{L['shadow']}; --shadow2:{L['shadow2']}; --green:{L['green']}; --red:{L['red']};
  --appbar:{L['appbar']}; --appbar-text:{L['appbar_text']};
}}
:root[data-theme="dark"] {{
  --bg:{D['bg']}; --surface:{D['surface']}; --surface2:{D['surface2']}; --text:{D['text']};
  --muted:{D['muted']}; --primary:{D['primary']}; --divider:{D['divider']}; --grid:{D['grid']};
  --shadow:{D['shadow']}; --shadow2:{D['shadow2']}; --green:{D['green']}; --red:{D['red']};
  --appbar:{D['appbar']}; --appbar-text:{D['appbar_text']};
}}
html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stMain"] {{ background:var(--bg) !important; }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1.2rem; padding-bottom:4rem; max-width:1300px; }}
html, body, [class*="css"], .block-container, p, span, label, div, li, td, th, h1, h2, h3, h4 {{
  font-family:"Roboto","Segoe UI",system-ui,-apple-system,sans-serif; color:var(--text); }}
.appbar {{ display:flex; align-items:center; background:var(--appbar); padding:16px 24px; border-radius:16px;
  box-shadow:var(--shadow2); margin-bottom:8px; }}
.appbar .brand {{ font-size:1.55rem; font-weight:700; color:var(--appbar-text); letter-spacing:-.01em; }}
.appbar .sub {{ color:var(--appbar-text); opacity:.85; font-size:.9rem; margin-top:2px; }}
.sec {{ font-size:1.22rem; font-weight:700; margin:.6rem 0 .9rem; display:flex; align-items:center; gap:8px; }}
.sec::before {{ content:""; width:4px; height:20px; background:var(--primary); border-radius:3px; }}
.note {{ color:var(--muted); font-size:.92rem; line-height:1.65; }}
.card {{ background:var(--surface); border-radius:14px; padding:18px 20px; box-shadow:var(--shadow);
  margin-bottom:14px; border:1px solid var(--divider); }}
.card b {{ color:var(--text); }}
.kpi-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:.2rem 0 1.1rem; }}
.kpi {{ background:var(--surface); border-radius:14px; padding:15px 18px; box-shadow:var(--shadow); border:1px solid var(--divider); }}
.kpi .v {{ font-size:1.8rem; font-weight:700; line-height:1.1; }}
.kpi .l {{ font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-top:5px; }}
.kpi .d {{ font-size:.8rem; margin-top:3px; }}
.chip {{ display:inline-block; font-size:.72rem; font-weight:600; padding:3px 11px; border-radius:999px;
  background:color-mix(in srgb,var(--primary) 16%,transparent); color:var(--primary); margin:0 6px 6px 0;
  border:1px solid color-mix(in srgb,var(--primary) 34%,transparent); }}
.chip.final {{ background:color-mix(in srgb,var(--green) 18%,transparent); color:var(--green); border-color:color-mix(in srgb,var(--green) 40%,transparent); }}
.chip.warn {{ background:color-mix(in srgb,var(--red) 16%,transparent); color:var(--red); border-color:color-mix(in srgb,var(--red) 36%,transparent); }}
.tip {{ display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px; border-radius:50%;
  background:var(--muted); color:var(--surface); font-size:11px; font-weight:700; cursor:help; position:relative; margin-left:6px; }}
.tip:hover::after {{ content:attr(data-tip); position:absolute; bottom:140%; left:50%; transform:translateX(-50%);
  background:var(--text); color:var(--surface); padding:9px 12px; border-radius:8px; font-size:.78rem; font-weight:400;
  width:250px; white-space:normal; z-index:300; box-shadow:var(--shadow2); line-height:1.5; text-align:left; }}
button[data-baseweb="tab"] {{ font-size:.98rem; font-weight:600; padding:10px 6px; color:var(--muted); }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:var(--primary); }}
div[data-baseweb="tab-list"] {{ gap:22px; border-bottom:1px solid var(--divider); overflow-x:auto; }}
div[data-baseweb="tab-highlight"] {{ background:var(--primary); }}
/* inputs / widgets - readable in both themes */
[data-testid="stExpander"] {{ border:1px solid var(--divider); border-radius:12px; background:var(--surface); box-shadow:var(--shadow); }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] p {{ color:var(--text); }}
.stTextInput input, .stNumberInput input, textarea, .stTextArea textarea,
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="input"] > div,
[data-baseweb="textarea"], [data-baseweb="select"] > div {{
  background:var(--surface2) !important; color:var(--text) !important; border-color:var(--divider) !important; }}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {{ color:var(--muted) !important; }}
[data-baseweb="select"] div, [data-baseweb="select"] span, [data-baseweb="select"] svg {{ color:var(--text) !important; fill:var(--text) !important; }}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {{ background:var(--surface) !important; color:var(--text) !important; }}
[data-baseweb="select"] [aria-selected="true"], [role="listbox"] {{ background:var(--surface) !important; }}
/* themed HTML tables (used instead of st.dataframe so they follow the client-side theme) */
.tbl-wrap {{ overflow:auto; border:1px solid var(--divider); border-radius:12px; box-shadow:var(--shadow); background:var(--surface); }}
.tbl {{ border-collapse:collapse; width:100%; font-size:.85rem; }}
.tbl thead th {{ position:sticky; top:0; background:var(--surface2); color:var(--muted); text-align:left;
  padding:9px 13px; font-weight:600; border-bottom:1px solid var(--divider); white-space:nowrap; z-index:1; }}
.tbl tbody td {{ padding:8px 13px; border-bottom:1px solid var(--divider); color:var(--text); white-space:nowrap;
  max-width:560px; overflow:hidden; text-overflow:ellipsis; }}
.tbl tbody tr:hover td {{ background:var(--surface2); }}
[data-baseweb="tag"] {{ background:var(--primary) !important; }}
.stRadio label, .stSelectbox label, .stTextInput label, .stMultiSelect label, .stNumberInput label {{ color:var(--muted) !important; }}
/* file uploader - fix white-on-white in dark */
[data-testid="stFileUploaderDropzone"] {{ background:var(--surface2) !important; border:1px dashed var(--divider) !important; }}
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] * {{ color:var(--text) !important; }}
[data-testid="stFileUploaderDropzone"] button {{ background:var(--primary) !important; color:#fff !important; border:none !important; }}
[data-testid="stFileUploaderDeleteBtn"] svg, [data-testid="stFileUploaderFileData"] {{ color:var(--text) !important; }}
/* metrics readable in both themes; images framed as cards (many figures are white-bg) */
[data-testid="stMetricValue"] {{ color:var(--text) !important; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{ color:var(--muted) !important; }}
[data-testid="stImage"] img {{ border-radius:10px; box-shadow:var(--shadow); }}
/* theme pill toggle (sun / moon) */
.theme-wrap {{ display:flex; justify-content:flex-end; padding-top:26px; }}
.theme-toggle {{ position:relative; display:inline-flex; width:78px; height:32px; border-radius:999px;
  background:#dfe3ea; cursor:pointer; text-decoration:none !important; box-shadow:inset 0 1px 3px rgba(0,0,0,.22);
  border:1px solid rgba(0,0,0,.06); transition:background .25s; }}
:root[data-theme="dark"] .theme-toggle {{ background:#2b2f36; border-color:rgba(255,255,255,.09); }}
.theme-toggle .ic {{ position:absolute; top:0; height:32px; width:39px; display:flex; align-items:center;
  justify-content:center; font-size:15px; z-index:3; transition:color .25s; }}
.theme-toggle .sun {{ left:2px; color:#fff; }}
.theme-toggle .moon {{ right:2px; color:#9aa0a6; }}
.theme-toggle .knob {{ position:absolute; top:3px; left:3px; width:26px; height:26px; border-radius:50%;
  z-index:2; box-shadow:0 1px 3px rgba(0,0,0,.4); transition:left .25s ease, background .25s; background:#f6a821; }}
:root[data-theme="dark"] .theme-toggle .sun {{ color:#9aa0a6; }}
:root[data-theme="dark"] .theme-toggle .knob {{ left:49px; background:#5b6572; }}
:root[data-theme="dark"] .theme-toggle .moon {{ color:#f0f2f5; }}
.part-nav {{ display:flex; gap:7px; align-items:center; height:100%; padding-top:24px; flex-wrap:wrap; }}
.part-pill {{ text-decoration:none !important; font-size:.82rem; font-weight:600; padding:8px 15px; border-radius:999px;
  background:var(--surface2); color:var(--muted) !important; border:1px solid var(--divider); white-space:nowrap; transition:all .18s; }}
.part-pill:hover {{ border-color:var(--primary); }}
.part-pill.active {{ background:var(--primary); color:#fff !important; border-color:var(--primary); box-shadow:var(--shadow); }}
.back-to-top {{ position:fixed; right:22px; bottom:22px; z-index:99; background:var(--primary); color:#fff !important;
  text-decoration:none; padding:10px 15px; border-radius:999px; font-size:.82rem; font-weight:600; box-shadow:var(--shadow2); }}
footer, #MainMenu {{ visibility:hidden; }}
</style>
<div id="top"></div>
<a href="#top" class="back-to-top">Top &uarr;</a>
""", unsafe_allow_html=True)


def tip(text):
    return f'<span class="tip" data-tip="{text}">i</span>'


# ===========================================================================
#  Plotly styling  (per-theme; readable legends; no stray "trace 0")
# ===========================================================================
def styled(fig, height=380, legend=True):
    fig.update_layout(
        template=T["tmpl"], height=height, margin=dict(l=16, r=20, t=74, b=54),
        font=dict(family="Roboto, Segoe UI, system-ui, sans-serif", size=13, color=T["text"]),
        title_font=dict(size=15.5, color=T["text"]), title_x=0.01, title_xanchor="left", title_y=0.98, title_yanchor="top",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=T["palette"], showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.10, x=0, font=dict(size=12, color=T["text"]),
                    bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
        hoverlabel=dict(bgcolor=T["surface"], font_size=12, font_color=T["text"], bordercolor=T["divider"]))
    fig.update_xaxes(showgrid=False, linecolor=T["divider"], zeroline=False,
                     title_font=dict(size=12, color=T["muted"]), title_standoff=14, tickfont=dict(color=T["muted"], size=11))
    fig.update_yaxes(gridcolor=T["grid"], zeroline=False, title_font=dict(size=12, color=T["muted"]),
                     title_standoff=16, tickfont=dict(color=T["muted"], size=11))
    return fig


def show(fig, height=380, legend=True):
    st.plotly_chart(styled(fig, height, legend), use_container_width=True, config={"displayModeBar": False})


def show_raw(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def kpis(items):
    html = '<div class="kpi-row">'
    for v, l, d, col in items:
        dd = f'<div class="d" style="color:{col}">{d}</div>' if d else ""
        html += f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div>{dd}</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


# ===========================================================================
#  Themed HTML table (st.dataframe is canvas-based and ignores our CSS vars, so
#  it stays light in dark mode; this renders a table that follows the theme).
# ===========================================================================
def html_table(df, height=360):
    d = df.copy()
    for c in d.select_dtypes(include=["float", "float64", "float32"]).columns:
        d[c] = d[c].round(4)
    table = d.to_html(index=False, border=0, classes="tbl", escape=True, na_rep="")
    st.markdown(f'<div class="tbl-wrap" style="max-height:{height}px">{table}</div>', unsafe_allow_html=True)


# ===========================================================================
#  Reusable filterable + paginated table
# ===========================================================================
def filter_table(df, key, cat_cols=(), search=True, page_sizes=(10, 25, 50), height=360):
    if search:
        q = st.text_input("Search", key=f"{key}_q", placeholder="type to filter across every column…",
                          label_visibility="collapsed",
                          help="Case-insensitive search over every column at once.")
    else:
        q = ""
    ctrl = st.columns(list(range(1, len(cat_cols) + 1)) and [1] * len(cat_cols) + [1] or [1])
    sel = {}
    for i, col in enumerate(cat_cols):
        opts = sorted([str(v) for v in df[col].dropna().unique()])
        sel[col] = ctrl[i].multiselect(col.replace("_", " "), opts, key=f"{key}_{col}",
                                       placeholder=f"all {col.replace('_',' ')}")
    view = df.copy()
    for col, vals in sel.items():
        if vals:
            view = view[view[col].astype(str).isin(vals)]
    if q:
        m = view.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)
        view = view[m]
    ps = ctrl[-1].selectbox("rows / page", page_sizes, key=f"{key}_ps")
    n = len(view)
    pages = max((n - 1) // ps + 1, 1)
    pg = 1
    if pages > 1:
        pg = st.number_input("page", 1, pages, 1, key=f"{key}_pg")
    st.caption(f"{n} rows · page {int(pg)} of {pages}")
    html_table(view.iloc[(int(pg) - 1) * ps: int(pg) * ps], height=height)


# ===========================================================================
#  Charts
# ===========================================================================
def fig_climb():
    lad = M.get("ladder", [])
    names, vals = [d["name"] for d in lad], [d["accuracy"] for d in lad]
    cols = [T["green"] if i == len(vals) - 1 else T["blue"] for i in range(len(vals))]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=cols, text=[f"{v:.3f}" for v in vals],
                           textposition="outside", textfont=dict(size=11, color=T["text"]),
                           hovertemplate="%{x}<br>accuracy %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.88, line_dash="dash", line_color=T["red"],
                  annotation_text="0.88 target", annotation_position="top right", annotation_font_color=T["red"])
    fig.update_yaxes(range=[0.72, 0.90], title="test accuracy (micro-F1)")
    fig.update_layout(title="The climb to the final model", showlegend=False)
    return fig


def fig_all_models():
    rows = [m for m in M.get("models", []) if m.get("accuracy")]
    names = [m["name"].split(" (")[0].replace("EfficientNetV2", "EffV2").replace("MViT-v2-S", "MViT") for m in rows]
    fig = go.Figure()
    fig.add_bar(name="accuracy", x=names, y=[m["accuracy"] for m in rows], marker_color=T["blue"])
    fig.add_bar(name="crash-F1", x=names, y=[m.get("crash_f1") for m in rows], marker_color=T["green"])
    fig.add_bar(name="ROC AUC", x=names, y=[m.get("auc") for m in rows], marker_color=T["amber"])
    fig.update_layout(title="Every model — accuracy, crash-F1, ROC AUC", barmode="group")
    fig.update_yaxes(range=[0.55, 0.96]); fig.update_xaxes(tickangle=-28)
    return fig


def fig_auc_vs_acc():
    rows = [m for m in M.get("models", []) if m.get("accuracy") and m.get("auc")]
    fam = ["Frame CNN", "Frame CNN + RNN", "Video transformer"]
    col = {"Frame CNN": T["blue"], "Frame CNN + RNN": T["green"], "Video transformer": T["amber"]}
    fig = go.Figure()
    for f in fam:
        mm = [m for m in rows if m["family"] == f]
        if mm:
            fig.add_scatter(x=[m["auc"] for m in mm], y=[m["accuracy"] for m in mm], mode="markers", name=f,
                            marker=dict(size=15, color=col[f], line=dict(width=1.5, color=T["surface"]), opacity=.9),
                            customdata=[m["name"] for m in mm],
                            hovertemplate="%{customdata}<br>AUC %{x:.3f} · acc %{y:.3f}<extra></extra>")
    fig.update_layout(title="Ranking quality vs. accuracy — hover for names (top-right is best)")
    fig.update_xaxes(title="ROC AUC"); fig.update_yaxes(title="test accuracy")
    return fig


def fig_confusion(conf):
    z = [[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["no crash", "crash"], y=["no crash", "crash"],
                               colorscale=[[0, T["surface2"]], [1, T["primary"]]], showscale=False,
                               text=[[str(z[i][j]) for j in range(2)] for i in range(2)], texttemplate="%{text}",
                               textfont={"size": 26, "color": T["text"]},
                               hovertemplate="actual %{y}, predicted %{x}: %{z}<extra></extra>"))
    fig.update_layout(title="Final ensemble — test confusion", xaxis_title="predicted", yaxis_title="actual",
                      yaxis_autorange="reversed", showlegend=False)
    return fig


def fig_training(keys=None):
    fig = go.Figure()
    for i, run in enumerate(M.get("training_runs", [])):
        if keys and run["key"] not in keys:
            continue
        c = run.get("curve", [])
        if c:
            fig.add_scatter(x=[p["epoch"] for p in c], y=[p["val_auc"] for p in c], mode="lines+markers",
                            name=run["label"], line=dict(color=T["palette"][i % len(T["palette"])], width=2.4),
                            marker=dict(size=5))
    fig.add_hline(y=0.90, line_dash="dot", line_color=T["muted"], annotation_text="~0.90 val-AUC ceiling",
                  annotation_position="bottom right", annotation_font_color=T["muted"])
    fig.update_layout(title="MViT training — validation ROC AUC per epoch")
    fig.update_xaxes(title="epoch"); fig.update_yaxes(title="validation ROC AUC", range=[0.6, 0.95])
    return fig


def fig_weakness(field, rows):
    rows = [r for r in rows if r.get("value") not in (None, "nan")]
    rows.sort(key=lambda r: -r["n"])
    names = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_bar(name="recall (crashes caught)", x=names, y=[r["recall"] for r in rows], marker_color=T["green"],
                customdata=[r["n_crash"] for r in rows],
                hovertemplate="%{x}<br>recall %{y:.0%}<br>%{customdata} crash clips<extra></extra>")
    fig.add_bar(name="precision", x=names, y=[r["precision"] for r in rows], marker_color=T["blue"],
                hovertemplate="%{x}<br>precision %{y:.0%}<extra></extra>")
    fig.add_scatter(name="clips in set", x=names, y=[r["n"] for r in rows], yaxis="y2", mode="markers",
                    marker=dict(color=T["muted"], size=11, symbol="diamond"),
                    hovertemplate="%{x}<br>%{y} clips<extra></extra>")
    fig.update_layout(title=f"Performance by {field.replace('_', ' ')}", barmode="group",
                      yaxis=dict(title="rate", range=[0, 1.08], tickformat=".0%"),
                      yaxis2=dict(title="clips", overlaying="y", side="right", showgrid=False,
                                  title_font=dict(color=T["muted"]), tickfont=dict(color=T["muted"])))
    return fig


def fig_rl_bar(rl):
    labels = ["averaging", "logreg stacker", "MLP stacker", "RL (DDQN)"]
    b = rl.get("baselines", {})
    accs = [b.get("averaging", {}).get("accuracy"), b.get("logreg", {}).get("accuracy"),
            b.get("mlp", {}).get("accuracy"), rl.get("rl_ddqn", {}).get("accuracy", {}).get("mean")]
    errs = [0, 0, 0, rl.get("rl_ddqn", {}).get("accuracy", {}).get("std", 0)]
    cols = [T["green"], T["blue"], T["blue"], T["purple"]]
    fig = go.Figure(go.Bar(x=labels, y=accs, marker_color=cols, error_y=dict(type="data", array=errs, visible=True, color=T["muted"]),
                           text=[f"{a:.3f}" if a else "-" for a in accs], textposition="outside", textfont=dict(color=T["text"]),
                           hovertemplate="%{x}<br>test accuracy %{y:.3f}<extra></extra>"))
    fig.update_layout(title="RL stacking vs. averaging vs. conventional stackers", showlegend=False)
    valid = [a for a in accs if a]
    fig.update_yaxes(range=[min(valid + [0.79]) - 0.02, max(valid + [0.86]) + 0.02], title="test accuracy")
    return fig


def fig_rl_curves(rl):
    curves = rl.get("rl_curves_val_acc", [])
    if not curves:
        return None
    fig = go.Figure()
    for i, c in enumerate(curves):
        fig.add_scatter(y=c, mode="lines", name=f"seed {i}", line=dict(width=1.7))
    fig.update_layout(title="DDQN validation accuracy while learning (per seed)")
    fig.update_xaxes(title="episode"); fig.update_yaxes(title="validation accuracy")
    return fig


# --- Keras-style layer diagram --------------------------------------------
KIND_LABEL = {"input": "input", "frozen": "frozen / not trained", "trainable": "trained (fine-tuned)",
              "pool": "op", "head": "output head", "ensemble": "combine"}


def fig_layers(layers, title, height=None):
    """Keras plot_model-style vertical stack: header (layer type) + activation + shapes, arrows between."""
    n = len(layers)
    fig = go.Figure()
    bh, gap = 1.0, 0.52
    total = n * (bh + gap)
    x0, x1 = 0.1, 0.9
    for i, L in enumerate(layers):
        typ, sub, si, so, kind = L
        yt = total - i * (bh + gap)
        yb = yt - bh
        hb = yb + bh * 0.52                                    # header/body split
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=hb, y1=yt, fillcolor=T["hdr"], line=dict(color=T["divider"], width=1.2), layer="below")
        fig.add_annotation(x=0.5, y=(hb + yt) / 2, text=f"<b>{typ}</b>", showarrow=False, font=dict(size=13, color="#ffffff"))
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=yb, y1=hb, fillcolor=T[kind], line=dict(color=T["divider"], width=1.2), layer="below")
        if sub:
            fig.add_annotation(x=0.5, y=hb - bh * 0.14, text=sub, showarrow=False, font=dict(size=11, color=T["text"]))
        shp = f"{si}  →  {so}" if (si and so) else (so or si or "")
        if shp:
            fig.add_annotation(x=0.5, y=yb + bh * 0.12, text=shp, showarrow=False, font=dict(size=10, color=T["muted"]))
        if i < n - 1:
            fig.add_annotation(x=0.5, y=yb - gap + 0.04, ax=0.5, ay=yb - 0.02, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.8, arrowcolor=T["muted"])
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[-0.5, total + 0.3])
    fig.update_layout(title=title, height=height or max(300, n * 88), margin=dict(l=6, r=6, t=48, b=6),
                      template=T["tmpl"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False, font=dict(color=T["text"]))
    return fig


def layer_legend(kinds):
    return '<div style="margin:.2rem 0 .5rem">' + "".join(
        f'<span class="chip" style="background:{T[k]};color:{T["text"]};border-color:{T["divider"]}">{KIND_LABEL[k]}</span>'
        for k in kinds) + "</div>"


# layer specs (real architectures from crash_detection/model.py + mvit.py)
POOL_STACK = [("Input", "pooled clip features", "", "(None, 2560)", "input"),
              ("EfficientNetV2", "backbone → per-frame 1280-d", "16×224×224×3", "T × 1280", "frozen"),
              ("max & mean pool", "over all frames (order-free)", "T × 1280", "(None, 2560)", "pool"),
              ("LayerNormalization", "put features on one scale", "(None, 2560)", "(None, 2560)", "trainable"),
              ("Dropout", "p = 0.5", "(None, 2560)", "(None, 2560)", "trainable"),
              ("Dense", "ReLU · 128 units", "(None, 2560)", "(None, 128)", "trainable"),
              ("Dropout", "p = 0.5", "(None, 128)", "(None, 128)", "trainable"),
              ("Dense", "sigmoid", "(None, 128)", "(None, 1)", "head")]
GRU_STACK = [("Input", "frame feature sequence", "", "(None, 256, 1280)", "input"),
             ("Masking", "skip zero-padded frames", "(None,256,1280)", "(None,256,1280)", "frozen"),
             ("Bidirectional GRU", "96 units × 2 directions", "(None,256,1280)", "(None, 192)", "trainable"),
             ("Dropout", "p = 0.5", "(None, 192)", "(None, 192)", "trainable"),
             ("Dense", "ReLU · 128", "(None, 192)", "(None, 128)", "trainable"),
             ("Dropout", "p = 0.5", "(None, 128)", "(None, 128)", "trainable"),
             ("Dense", "sigmoid", "(None, 128)", "(None, 1)", "head")]
MVIT_STACK = [("Input clip", "16 RGB frames", "", "3 × 16 × 224 × 224", "input"),
              ("Patch embed (Conv3d)", "cut clip into tokens", "3×16×224²", "96 × 8 × 56 × 56", "frozen"),
              ("MViT blocks 1–14", "multiscale attention + MLP (frozen)", "pool space, grow channels", "→ 768-d tokens", "frozen"),
              ("MViT blocks 15–16", "last 2 blocks — fine-tuned", "768-d tokens", "768-d tokens", "trainable"),
              ("LayerNorm + global pool", "one vector per clip", "768-d tokens", "(None, 768)", "trainable"),
              ("Linear head", "768 → 1 logit → sigmoid", "(None, 768)", "(None, 1)", "head")]
MVIT_BLOCK = [("LayerNorm", "", "", "", "frozen"),
              ("Multi-head pooled attention", "1→8 heads, Q/K/V pooled", "", "", "trainable"),
              ("Add residual", "x + attention(x)", "", "", "pool"),
              ("LayerNorm", "", "", "", "frozen"),
              ("MLP: Linear → GELU → Linear", "×4 hidden expansion", "", "", "trainable"),
              ("Add residual", "x + mlp(x)", "", "", "pool")]
QNET_STACK = [("Input state", "3 embeddings + 3 probs", "", "(None, 2307)", "input"),
              ("Dense", "ReLU · 256", "(None, 2307)", "(None, 256)", "trainable"),
              ("Dropout", "p = 0.3", "(None, 256)", "(None, 256)", "trainable"),
              ("Dense", "ReLU · 128", "(None, 256)", "(None, 128)", "trainable"),
              ("Dropout", "p = 0.3", "(None, 128)", "(None, 128)", "trainable"),
              ("Dense", "linear · Q(s,·)", "(None, 128)", "(None, 2)", "head")]
ENSEMBLE_FLOW = [("member 1 — EMA recipe", "MViT-v2-S · acc 0.849", "", "prob₁", "trainable"),
                 ("member 2 — curated", "MViT-v2-S · acc 0.844", "", "prob₂", "trainable"),
                 ("member 3 — revised aug", "MViT-v2-S · acc 0.840", "", "prob₃", "trainable"),
                 ("9-window × mirror TTA (each)", "denoise, then localise the crash", "", "", "pool"),
                 ("average the 3 probabilities", "committee vote", "", "p̄", "ensemble"),
                 ("threshold 0.71", "→ crash / no-crash", "p̄", "decision", "head")]

MODEL_LAYERS = {"effv2_frozen": POOL_STACK, "effv2_pool": POOL_STACK, "bigru": GRU_STACK,
                "mvit_base": MVIT_STACK, "mvit_tta": MVIT_STACK, "mvit_ema": MVIT_STACK,
                "mvit_curated": MVIT_STACK, "mvit_augrevised": MVIT_STACK, "mvit_ensemble": ENSEMBLE_FLOW}

# technique explainers (what · how · why it helps)
TECHNIQUES = {
    "mvit_base": ("Transfer learning + fine-tuning",
        "MViT-v2-S ships pretrained on Kinetics-400 (400 human-action classes). Its early blocks already know generic "
        "motion and texture, so we <b>freeze</b> them and only train the <b>last 2–3 transformer blocks</b> plus a fresh "
        "1-logit head. <b>Why it helps:</b> we get a strong video representation for free and only ~13.6M of 34.2M "
        "parameters have to learn 'crash vs. not' from ~1000 clips — small enough not to over-fit, and it fits a 6&nbsp;GB GPU."),
    "mvit_tta": ("Test-time augmentation (TTA)",
        "A crash occupies one short stretch of a 40-second clip, and a single 16-frame window can miss it. At inference we "
        "slide <b>9 windows</b> across the clip's middle and, for each, also run a <b>left-right mirror</b>. We average the "
        "two mirror views (removes noise), then take the <b>maximum</b> over the 9 windows (the crash lives in one window, so "
        "max finds it). <b>Why it helps:</b> +0.008 accuracy, no retraining — it just looks harder at test time."),
    "mvit_ema": ("Exponential moving average (EMA)",
        "During training we keep a second, shadow copy of the weights, updated after every step as "
        "<code>shadow = 0.995·shadow + 0.005·weights</code>. It is a smoothed average of the whole optimisation path, which "
        "usually sits in a flatter, better-generalising minimum than the last noisy step. Each epoch we score <b>both</b> the "
        "raw and the EMA weights on validation and keep whichever is better. <b>Why it helps:</b> free, stable, +~0.005."),
    "mvit_curated": ("Data curation",
        "Score every training clip with the current model and flag the ones it is <b>confidently wrong</b> about — a crash "
        "scored below 0.15, or a normal clip above 0.85. Those are usually ambiguous or mislabelled near-misses. We drop them "
        "(26 of 1049) and retrain. <b>Why it (partly) helps:</b> on its own it is a wash — it also removes genuinely hard clips "
        "— but it yields a differently-trained model that disagrees with the others, which is exactly what an ensemble wants."),
    "mvit_augrevised": ("Revised data augmentation",
        "The error analysis showed failures cluster in <b>low light and adverse weather</b>. The earlier strong augmentation "
        "was geometric (aggressive crops/rotations) and measurably hurt. The revised version is <b>photometric and "
        "temporally-consistent</b>: brightness ×[0.6,1.2] and gamma [0.7,1.6] skewed <b>darker</b> (manufacture low-light "
        "crashes), HSV saturation jitter, occasional Gaussian sensor noise; gentler crop (0.85–1.0) and rotation (±5°). The "
        "same transform hits every frame so motion survives. <b>Why it helps:</b> highest single-model AUC (0.925); recall "
        "0.84→0.88 and Rain recall →0.94."),
    "mvit_ensemble": ("Committee / ensemble",
        "Average the crash probabilities of the three independently fine-tuned MViT models (each already TTA'd). They were "
        "trained differently (recipe / curated / revised-aug) so they make <b>different</b> mistakes; averaging cancels the "
        "uncorrelated errors. <b>Why it helps:</b> +0.005 over the best single model, and it is the most robust — the final "
        "0.858 / 0.863 F1 / 0.919 AUC."),
    "bigru": ("Bolt-on temporal head (what did NOT work)",
        "Instead of pooling the per-frame CNN features, feed them <b>in order</b> to a bidirectional GRU so the model can use "
        "how the scene changes. <b>Why it failed:</b> on frozen features and ~1000 clips it over-fit and mostly predicted "
        "'no crash'. Motion has to be learned <i>with</i> the pixels (which is exactly what MViT does), not bolted on after."),
    "effv2_frozen": ("Transfer learning — frozen features",
        "Run a pretrained EfficientNetV2 once over every frame, cache the 1280-d feature vectors, and train only a tiny "
        "pooling head. <b>Why:</b> it is nearly free and tells us how far a per-frame image model can go — a real ~0.75 floor."),
    "effv2_pool": ("Fine-tuning the CNN",
        "Unfreeze the CNN's top ~60 layers and teach it what a crash frame looks like, then re-cache features and re-train "
        "the pooling head. <b>Why:</b> the best a per-frame model reached (~0.77) before we accepted a single frame isn't enough."),
}


# ===========================================================================
#  Live prediction
# ===========================================================================
@st.cache_resource(show_spinner="Loading the MViT ensemble…")
def load_models():
    import torch
    from crash_detection.mvit import DEVICE, build_mvit
    ens = ROOT / "artifacts" / "report_ensemble.json"
    members = (json.loads(ens.read_text()).get("members") if ens.exists() else None) \
        or ["mvit_model_recipe.pt", "mvit_model_curated.pt", "mvit_model_augrevised.pt"]
    models = []
    for name in members:
        p = ROOT / "artifacts" / name
        if p.exists():
            m = build_mvit(); m.load_state_dict(torch.load(p, map_location=DEVICE)); m.eval(); models.append(m)
    return models, DEVICE


@st.cache_data(show_spinner=False)
def sample_index():
    if not PRED_CSV.exists():
        return {}
    df = pd.read_csv(PRED_CSV)
    idx = {}
    for r in df.itertuples(index=False):
        idx.setdefault(str(r.clip_key).split("/")[-1], []).append(
            {"clip_key": r.clip_key, "split": r.split, "label": int(r.label),
             "weather": r.weather, "light": r.light_conditions, "scene": r.scene})
    return idx


@st.cache_data(show_spinner=False)
def error_table():
    if not PRED_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRED_CSV)
    df = df[df["split"].isin(["val", "test"])].copy()
    df["result"] = np.where((df.label == 1) & (df.pred == 1), "TP",
                    np.where((df.label == 0) & (df.pred == 0), "TN",
                    np.where((df.label == 0) & (df.pred == 1), "FP", "FN")))
    df["reason"] = [{"FP": f"false alarm — normal {l}/{w}/{s} scored {p:.2f}",
                     "FN": f"missed crash — {l}/{w}/{s}, scored only {p:.2f}",
                     "TP": f"correct catch ({p:.2f})", "TN": f"correct pass ({p:.2f})"}[k]
                    for k, l, w, s, p in zip(df.result, df.light_conditions, df.weather, df.scene, df.prob_ensemble)]
    df["prob"] = df.prob_ensemble.round(3)
    return df.rename(columns={"light_conditions": "light"})[["clip_key", "result", "light", "weather", "scene", "prob", "reason"]]


def frames_from_video(path, size=(224, 224)):
    from crash_detection.data import frame_timestamps
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(frame_timestamps(total, fps).tolist())
    frames, pos = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if pos in wanted:
            f = cv2.resize(f, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        pos += 1
    cap.release()
    return np.stack(frames).astype("float32") / 255.0 if frames else None


def predict(frames, models, device, n_windows=9):
    import torch
    from crash_detection.mvit import clip_indices, to_tensor
    n, per = len(frames), []
    for m in models:
        wins = []
        for c in np.linspace(0.30 * n, 0.70 * n, n_windows).astype(int):
            clip = frames[clip_indices(n, c)]
            views = torch.stack([to_tensor(clip), to_tensor(clip[:, :, ::-1, :].copy())])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = m(views.to(device)).float().squeeze(1)
            wins.append(torch.sigmoid(logits).mean().item())
        per.append(max(wins))
    return float(np.mean(per))


def gauge(prob):
    colour = T["red"] if prob >= THRESHOLD else T["green"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 34, "color": T["text"]}},
        title={"text": "crash probability", "font": {"size": 13, "color": T["muted"]}},
        gauge={"axis": {"range": [0, 100], "tickcolor": T["muted"], "tickfont": {"color": T["muted"]}},
               "bar": {"color": colour}, "bgcolor": T["surface2"], "borderwidth": 0,
               "threshold": {"line": {"color": T["text"], "width": 3}, "value": THRESHOLD * 100},
               "steps": [{"range": [0, THRESHOLD * 100], "color": T["head"]},
                         {"range": [THRESHOLD * 100, 100], "color": T["pool"]}]}))
    fig.update_layout(height=260, margin=dict(l=26, r=26, t=44, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=T["text"]))
    return fig


# ===========================================================================
#  Part 2 — Amazon review sentiment (NLP)   (rendered when ?part=2)
# ===========================================================================
AS = ROOT / "amazon_sentiment_outputs"


def p2_csv(name, sub="dashboard_data"):
    path = AS / sub / name
    return pd.read_csv(path) if path.exists() else None


def p2_json(name, sub="dashboard_data"):
    path = AS / sub / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def p2_image(name, caption=None):
    path = AS / "figures" / name
    if path.exists():
        st.image(str(path), use_container_width=True, caption=caption)
    else:
        st.caption(f"· {name} not generated yet — run `python -m nlp_sentiment.run_all`")


def p2_hbar(df, cat, val, title, color=None, height=360):
    d = df.iloc[::-1]
    fig = go.Figure(go.Bar(y=d[cat].astype(str), x=d[val], orientation="h", marker_color=color or T["primary"]))
    fig.update_layout(title=title)
    show(fig, height=height, legend=False)


@st.cache_resource(show_spinner="Loading the saved sentiment models…")
def p2_load_models():
    from nlp_sentiment import tryit
    return tryit.load_models()


def render_part2():
    kpi = p2_json("kpi_summary.json")
    tabs = st.tabs(["Overview", "Exploratory analysis", "Warranty",
                    "Sentiment model", "Bonus problems", "Try it"])

    # --- Overview ---------------------------------------------------------
    with tabs[0]:
        st.markdown('<div class="sec">Headline result</div>', unsafe_allow_html=True)
        kpis([(f"{kpi.get('best_model_micro_f1', 0.6904):.4f}", "micro-F1 · 5-class", "validation", T["muted"]),
              (f"{kpi.get('best_model_macro_f1', 0.527):.3f}", "macro-F1", "", ""),
              (f"{kpi.get('train_rows', 838944):,}", "training reviews", f"{kpi.get('train_model_rows_after_dedup', 830479):,} after dedup", T["muted"]),
              (f"{kpi.get('test_rows', 20000):,}", "test reviews", "q2_submission.csv", ""),
              (f"{kpi.get('warranty_review_count', 19998):,}", "warranty reviews", f"{kpi.get('warranty_review_share', 0.0238) * 100:.1f}% of all", T["muted"])])
        st.markdown('<div class="note">The graded model is a <b>word + character TF-IDF → LinearSVC</b> '
                    'five-class sentiment classifier. The text-only variant was chosen over the '
                    'text+metadata one (validation micro-F1 gain &lt; 0.001) for simplicity and lower '
                    'distribution-shift risk.</div>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            rd = p2_csv("rating_distribution.csv")
            if rd is not None:
                fig = go.Figure(go.Bar(x=rd["overall"].astype(str), y=rd["count"], marker_color=T["primary"],
                                       text=[f"{p:.0f}%" for p in rd["percentage"]], textposition="outside"))
                fig.update_layout(title="Rating distribution (training reviews)")
                show(fig, legend=False)
        with c2:
            mc = p2_csv("model_comparison.csv")
            if mc is not None:
                fig = go.Figure()
                fig.add_bar(x=mc["model_name"], y=mc["micro_f1"], name="micro-F1", marker_color=T["blue"])
                fig.add_bar(x=mc["model_name"], y=mc["macro_f1"], name="macro-F1", marker_color=T["amber"])
                fig.update_layout(title="Validation model comparison", barmode="group", yaxis=dict(range=[0, 1]))
                show(fig, legend=True)

    # --- Exploratory analysis --------------------------------------------
    with tabs[1]:
        st.markdown('<div class="sec">Brands, reviewers & products</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            tb = p2_csv("top_10_brands_mean_rating.csv")
            if tb is not None:
                p2_hbar(tb.sort_values("mean_rating"), "brand_clean", "mean_rating",
                        "Top-10 brands by mean rating", color=T["green"])
        with c2:
            tr = p2_csv("top_10_helpful_reviewers.csv")
            if tr is not None:
                p2_hbar(tr.sort_values("total_vote"), "reviewerName", "total_vote",
                        "Top-10 reviewers by helpful votes", color=T["purple"])
        tp = p2_csv("top_10_five_star_products.csv")
        if tp is not None:
            with st.expander("Top-10 products by five-star reviews", expanded=False):
                html_table(tp)

        st.markdown('<div class="sec">Word clouds by sentiment' +
                    tip("Stop-words removed except negations; negation phrases merged (not_good) so the cloud "
                        "reflects sentiment, not just topic.") + '</div>', unsafe_allow_html=True)
        wc = st.columns(3)
        with wc[0]:
            st.caption("Positive"); p2_image("02_wordcloud_positive.png")
        with wc[1]:
            st.caption("Neutral"); p2_image("02_wordcloud_neutral.png")
        with wc[2]:
            st.caption("Negative"); p2_image("02_wordcloud_negative.png")

        st.markdown('<div class="sec">Distinctive vocabulary (log-ratio)</div>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            dp = p2_csv("distinctive_positive_words.csv")
            if dp is not None:
                st.caption("Most positive-leaning words")
                html_table(dp[["word", "log2_positive_negative_ratio"]].head(15), height=360)
        with d2:
            dn = p2_csv("distinctive_negative_words.csv")
            if dn is not None:
                st.caption("Most negative-leaning words")
                html_table(dn[["word", "log2_positive_negative_ratio"]].head(15), height=360)

        st.markdown('<div class="sec">Review length</div>', unsafe_allow_html=True)
        lc1, lc2 = st.columns(2)
        with lc1:
            p2_image("04_review_length_full.png")
        with lc2:
            p2_image("04_review_length_filtered.png")

    # --- Warranty ---------------------------------------------------------
    with tabs[2]:
        st.markdown('<div class="sec">Warranty detection precision' +
                    tip("A conservative two-stage pipeline: domain Word2Vec candidate discovery + fuzzy typo "
                        "detection, then boundary-aware phrase/word matching, hand-QC'd by match type.") +
                    '</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 3])
        with c1:
            pc = p2_csv("warranty_manual_precision_by_type.csv", sub="tables")
            if pc is not None:
                fig = go.Figure(go.Bar(x=pc["warranty_match_type"], y=pc["estimated_precision"], marker_color=T["teal"],
                                       text=[f"{p:.2f}" for p in pc["estimated_precision"]], textposition="outside"))
                fig.update_layout(title="Precision by match type", yaxis=dict(range=[0, 1.08]))
                show(fig, legend=False)
        with c2:
            p2_image("07_warranty_satisfaction.png")
        st.markdown('<div class="sec">Per-product warranty satisfaction</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        wcols = ["title_clean", "warranty_review_count", "bayesian_warranty_rating"]
        with c3:
            tw = p2_csv("top_warranty_products.csv")
            if tw is not None:
                st.caption("Highest satisfaction (Bayesian-adjusted)")
                html_table(tw[[c for c in wcols if c in tw.columns]].head(10))
        with c4:
            bw = p2_csv("bottom_warranty_products.csv")
            if bw is not None:
                st.caption("Lowest satisfaction")
                html_table(bw[[c for c in wcols if c in bw.columns]].head(10))

    # --- Sentiment model --------------------------------------------------
    with tabs[3]:
        st.markdown('<div class="sec">Validation performance</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            cm = p2_csv("best_model_confusion_matrix.csv")
            if cm is not None:
                pred_cols = [c for c in cm.columns if c.startswith("pred_")]
                z = cm[pred_cols].values
                fig = go.Figure(go.Heatmap(z=z, x=[c.replace("pred_", "") for c in pred_cols],
                                           y=[str(v).replace("true_", "") for v in cm[cm.columns[0]]],
                                           colorscale="Blues", showscale=False, text=z, texttemplate="%{text}"))
                fig.update_layout(title="Confusion matrix (validation)", xaxis_title="predicted", yaxis_title="true")
                show(fig, legend=False, height=400)
        with c2:
            rep = p2_csv("best_model_classification_report.csv")
            if rep is not None:
                pcls = rep[rep["class_or_average"].astype(str).isin(["1", "2", "3", "4", "5"])]
                fig = go.Figure(go.Bar(x=pcls["class_or_average"].astype(str), y=pcls["f1-score"], marker_color=T["primary"],
                                       text=[f"{v:.2f}" for v in pcls["f1-score"]], textposition="outside"))
                fig.update_layout(title="Per-class F1", yaxis=dict(range=[0, 1]))
                show(fig, legend=False, height=400)
        mc = p2_csv("model_comparison.csv")
        if mc is not None:
            with st.expander("Model comparison — full table", expanded=False):
                html_table(mc)
        nb = p2_csv("conditional_ngram_model_comparison.csv")
        if nb is not None:
            with st.expander("NB-SVM (conditional n-gram) vs the deployed model", expanded=False):
                html_table(nb)

        st.markdown('<div class="sec">Test predictions & distribution shift</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            pdst = p2_csv("test_prediction_distribution.csv")
            if pdst is not None:
                fig = go.Figure(go.Bar(x=pdst["predicted"].astype(str), y=pdst["count"], marker_color=T["blue"]))
                fig.update_layout(title="Test predictions")
                show(fig, legend=False)
        with c4:
            sh = p2_csv("train_test_prediction_distribution_shift.csv")
            if sh is not None:
                fig = go.Figure()
                fig.add_bar(x=sh["rating"].astype(str), y=sh["train_percentage"], name="train %", marker_color=T["muted"])
                fig.add_bar(x=sh["rating"].astype(str), y=sh["test_predicted_percentage"], name="test pred %", marker_color=T["primary"])
                fig.update_layout(title="Train vs test-prediction distribution", barmode="group")
                show(fig, legend=True)

        err = p2_csv("model_error_analysis.csv", sub="tables")
        if err is not None:
            st.markdown('<div class="sec">Largest errors (searchable)</div>', unsafe_allow_html=True)
            keep = [c for c in ["overall", "predicted", "absolute_error", "reviewText_clean"] if c in err.columns]
            filter_table(err[keep] if keep else err, key="p2err",
                         cat_cols=tuple(c for c in ("overall", "predicted") if c in err.columns), height=340)

    # --- Bonus problems ---------------------------------------------------
    with tabs[4]:
        bp = p2_json("business_problem_summary.json", sub="independent_business_problems")
        st.markdown('<div class="sec">Two independent business problems</div>', unsafe_allow_html=True)
        g1, g2 = bp.get("cold_start_business_gate_passed"), bp.get("quality_business_gate_passed")
        st.markdown(
            f'<span class="chip {"final" if g1 else "warn"}">Cold-start gate: {"passed" if g1 else "not passed"}</span>'
            f'<span class="chip {"final" if g2 else "warn"}">Quality-risk gate: {"passed" if g2 else "not passed"}</span>'
            f'<span class="chip">Quality method: {bp.get("quality_recommended_method", "—")}</span>'
            f'<span class="chip">No content filtering</span>', unsafe_allow_html=True)
        st.markdown('<div class="note">4.1 ranks fresh reviews before helpful votes accrue; 4.2 flags the top-10% '
                    'product-quarters at risk of a satisfaction drop — an <b>internal alert list only</b>, nothing is '
                    'auto-filtered or blocked.</div>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            cs = p2_csv("cold_start_model_comparison.csv")
            if cs is not None:
                st.caption("4.1 Cold-start ranking")
                html_table(cs)
        with c2:
            qr = p2_csv("quality_risk_model_comparison.csv")
            if qr is not None:
                st.caption("4.2 Quality-risk models")
                html_table(qr)
        ql = p2_csv("quality_risk_priority_list.csv")
        if ql is not None:
            with st.expander("Quality-risk watch list (top-10% product-quarters)", expanded=False):
                cols = [c for c in ["asin", "title_clean", "brand_clean", "period", "risk_score",
                                    "negative_rate", "future_negative_rate"] if c in ql.columns]
                filter_table(ql[cols] if cols else ql, key="p2ql", height=320)

    # --- Try it -----------------------------------------------------------
    with tabs[5]:
        st.markdown('<div class="sec">Score a review with the saved model' +
                    tip("Runs entirely on the models this project trained — no external API. "
                        "1–2 → negative, 3 → neutral, 4–5 → positive.") + '</div>', unsafe_allow_html=True)
        summary_in = st.text_input("Review title / summary (optional)", key="p2_sum")
        review_in = st.text_area("Review text", key="p2_rev", height=140,
                                 placeholder="Write an English product review…")
        cc = st.columns([1, 1, 3])
        model_choice = cc[0].selectbox("model", ["auto", "svc", "nbsvm"], key="p2_model")
        go_pred = cc[1].button("Analyze", type="primary", use_container_width=True)
        if go_pred:
            if not review_in.strip():
                st.warning("Enter a review first.")
            else:
                try:
                    from nlp_sentiment import tryit
                    out = tryit.predict(review_in, summary=summary_in, model=model_choice, resources=p2_load_models())
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted rating", f"{out['predicted_rating']} / 5")
                    m2.metric("Sentiment", out["sentiment"].title())
                    m3.metric("Confidence", f"{out['normalized_confidence'] * 100:.1f}%")
                    fig = go.Figure(go.Bar(x=[f"{i} ★" for i in range(1, 6)], y=out["class_scores"], marker_color=T["primary"]))
                    fig.update_layout(title="Score per class")
                    show(fig, legend=False, height=300)
                    if out["evidence_ngrams"]:
                        st.markdown("**Top n-grams driving the decision:** " + " · ".join(out["evidence_ngrams"]))
                    st.caption("Confidence is a softmax over decision scores, not a calibrated probability.")
                except Exception as exc:
                    st.error(f"Could not load the model artifacts ({exc}). "
                             f"Run `python -m nlp_sentiment.run_all` first.")


# ===========================================================================
#  App bar + theme toggle
# ===========================================================================
# Client-side theme: flip data-theme on the app root, persist to localStorage,
# and re-theme the Plotly charts in place — all without a Streamlit rerun.
THEME_JS = """
<script>
(function(){
  var pdoc = window.parent.document, proot = pdoc.documentElement;
  var C = {
    dark:  {text:"#e8ecf3", muted:"#aab4c2", grid:"#283040", divider:"#2b333f",
            hmscale:[[0,"#152232"],[0.55,"#2b5c96"],[1,"#6db3f2"]]},
    light: {text:"#1a2233", muted:"#5b6675", grid:"#e6ebf3", divider:"#dde4ee", hmscale:"Blues"}
  };
  function themeCharts(mode, force){
    var P = window.parent.Plotly; if(!P) return;
    var c = C[mode];
    pdoc.querySelectorAll(".js-plotly-plot").forEach(function(gd){
      if(!force && gd.__tmode === mode) return;
      try { P.relayout(gd, {
        "font.color": c.text, "title.font.color": c.text, "legend.font.color": c.text,
        "xaxis.linecolor": c.divider, "xaxis.tickfont.color": c.muted, "xaxis.title.font.color": c.muted,
        "yaxis.gridcolor": c.grid, "yaxis.tickfont.color": c.muted, "yaxis.title.font.color": c.muted,
        "hoverlabel.font.color": c.text
      }); } catch(e){}
      try {
        var hm = [];
        (gd.data||[]).forEach(function(tr,i){ if(tr.type==="heatmap") hm.push(i); });
        if (hm.length) P.restyle(gd, {colorscale:c.hmscale}, hm);
      } catch(e){}
      gd.__tmode = mode;
    });
  }
  function apply(mode){ proot.setAttribute("data-theme", mode); themeCharts(mode, true); }
  function saved(){ try { return localStorage.getItem("qbc_theme")==="dark" ? "dark":"light"; } catch(e){ return "light"; } }
  apply(saved());
  [150, 400, 900].forEach(function(d){ setTimeout(function(){ themeCharts(saved(), false); }, d); });
  try {
    var _deb, MO = window.parent.MutationObserver || window.MutationObserver;
    new MO(function(){ clearTimeout(_deb); _deb = setTimeout(function(){ themeCharts(saved(), false); }, 150); })
      .observe(pdoc.body, {childList:true, subtree:true});
  } catch(e){}
  var btn = pdoc.getElementById("themeToggle");
  if(btn){
    btn.onclick = function(){
      var mode = proot.getAttribute("data-theme")==="dark" ? "light":"dark";
      try { localStorage.setItem("qbc_theme", mode); } catch(e){}
      apply(mode);
    };
  }
})();
</script>
"""

bar, nav, tog = st.columns([4, 2, 1])
with bar:
    if PART == "1":
        st.markdown('<div class="appbar"><div><div class="brand">Crash Detection — Part 1</div>'
                    '<div class="sub">Nexar dashcam collision recognition · MViT-v2-S video transformer · QBC12 Group 6</div>'
                    '</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="appbar"><div><div class="brand">Review Sentiment — Part 2</div>'
                    '<div class="sub">Amazon review NLP · word + char TF-IDF → LinearSVC · 5-class · QBC12 Group 6</div>'
                    '</div></div>', unsafe_allow_html=True)
with nav:
    st.markdown(
        f'<div class="part-nav">'
        f'<a class="part-pill {"active" if PART == "1" else ""}" href="?part=1" target="_self">Part 1 · Crash</a>'
        f'<a class="part-pill {"active" if PART == "2" else ""}" href="?part=2" target="_self">Part 2 · NLP</a>'
        f'</div>', unsafe_allow_html=True)
with tog:
    st.markdown('<div class="theme-wrap"><div class="theme-toggle" id="themeToggle" role="button" tabindex="0" '
                'title="Switch light / dark mode">'
                '<span class="ic sun">&#9728;</span><span class="ic moon">&#9790;</span>'
                '<span class="knob"></span></div></div>', unsafe_allow_html=True)

components.html(THEME_JS, height=0)

if PART == "2":
    render_part2()
    st.stop()

tabs = st.tabs(["Overview", "Models & architecture", "Training", "Error analysis",
                "RL experiment", "Try it", "Data & reproduction"])

# --- Overview -------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="sec">Headline result</div>', unsafe_allow_html=True)
    kpis([(f"{FINAL.get('accuracy', 0.858):.3f}", "Accuracy (micro-F1)", "recognition task", T["muted"]),
          (f"{FINAL.get('auc', 0.919):.3f}", "ROC AUC", "", ""),
          (f"{FINAL.get('crash_f1', 0.863):.3f}", "Crash F1", "", ""),
          (f"{FINAL.get('caught', 101)}/{FINAL.get('crashes', 112)}", "Crashes caught", "recall", T["green"]),
          (f"{FINAL.get('false_alarms', 21)}", "False alarms", f"of {FINAL.get('normal', 113)} normal", T["red"])])
    c1, c2 = st.columns([3, 2])
    with c1:
        show(fig_climb(), height=400, legend=False)
    with c2:
        show(fig_confusion(FINAL.get("confusion", {"tn": 92, "fp": 21, "fn": 11, "tp": 101})), height=400, legend=False)
    st.markdown('<div class="sec">Every model we trained, side by side'
                + tip("Frame-CNN and bi-GRU rows are from an earlier split; the MViT rows share one clean recognition "
                      "split. Full context in the trial-history table below.") + '</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        show(fig_all_models(), height=430)
    with c4:
        show(fig_auc_vs_acc(), height=430)
    with st.expander("Full trial history — every run we kept (searchable, filterable, paged)", expanded=False):
        tr = pd.DataFrame(M.get("all_trials", []))
        if len(tr):
            tr = tr.rename(columns={"acc": "test acc", "auc": "ROC AUC"})
            filter_table(tr, key="trials", cat_cols=("stage", "task"), height=320)
        st.caption("* early runs on a random recognition split (before the clean split-before-training setup). "
                   "'anticipation' = the competition's harder cut-before-impact task.")

# --- Models & architecture ------------------------------------------------
with tabs[1]:
    models = M.get("models", [])
    labels = ["Journey"] + [m["name"].split(" (")[0].replace("EfficientNetV2", "EffV2").replace("MViT-v2-S", "MViT") for m in models]
    pick = st.radio("Model", labels, horizontal=True, label_visibility="collapsed",
                    help="Pick the story, or any single trained model, for its results, Keras-style architecture, and a "
                         "plain-English note on the technique behind it.")
    st.write("")
    if pick == "Journey":
        st.markdown('<div class="sec">How we got to the final model</div>', unsafe_allow_html=True)
        jc, jt = st.columns([1, 1])
        with jc:
            steps = [("Frame CNN + pooling", 0.769, "frozen", "a single frame isn't enough"),
                     ("bi-GRU on features", 0.607, "head", "order bolted on — over-fit"),
                     ("MViT-v2-S video model", 0.822, "trainable", "reads space + time · +0.07"),
                     ("+ TTA + EMA", 0.849, "trainable", "cheap, reliable gains"),
                     ("+ revised-aug member", 0.858, "pool", "targeted low-light retrain"),
                     ("MViT ×3 ensemble", 0.858, "ensemble", "the deliverable")]
            fig = go.Figure()
            n = len(steps); bh, gap = 1.0, 0.5; total = n * (bh + gap)
            for i, (name, acc, kind, note) in enumerate(steps):
                yt = total - i * (bh + gap); yb = yt - bh
                good = i not in (1,)
                fig.add_shape(type="rect", x0=0.06, x1=0.72, y0=yb, y1=yt, fillcolor=T[kind],
                              line=dict(color=T["divider"], width=1.2), layer="below")
                fig.add_annotation(x=0.39, y=yb + bh * 0.62, text=f"<b>{name}</b>", showarrow=False, font=dict(size=12.5, color=T["text"]))
                fig.add_annotation(x=0.39, y=yb + bh * 0.24, text=note, showarrow=False, font=dict(size=10, color=T["muted"]))
                fig.add_annotation(x=0.86, y=(yb + yt) / 2, text=f"<b>{acc:.3f}</b>", showarrow=False,
                                   font=dict(size=14, color=T["green"] if good else T["red"]))
                if i < n - 1:
                    fig.add_annotation(x=0.39, y=yb - gap + 0.04, ax=0.39, ay=yb - 0.02, xref="x", yref="y", axref="x",
                                       ayref="y", showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=1.7, arrowcolor=T["muted"])
            fig.update_xaxes(visible=False, range=[0, 1]); fig.update_yaxes(visible=False, range=[-0.5, total + 0.3])
            fig.update_layout(title="The journey (accuracy at each step)", height=520, margin=dict(l=6, r=6, t=48, b=6),
                              template=T["tmpl"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, font=dict(color=T["text"]))
            show_raw(fig)
        with jt:
            st.markdown(
                f'<div class="card note">We started cheap and only spent compute when a number told us to.<br><br>'
                f'<b>1 · A single frame isn\'t enough.</b> A frozen ImageNet CNN + pooling head reached ~0.75; fine-tuning it '
                f'pushed to ~0.77 and stalled — a still frame rarely contains the collision.<br><br>'
                f'<b>2 · Order, bolted on, didn\'t help.</b> A bi-GRU over the frame features <i>under-performed</i> plain '
                f'pooling — it over-fit. Motion must be learned <i>with</i> the pixels, not after.<br><br>'
                f'<b>3 · A real video model was the jump.</b> MViT-v2-S reads a 16-frame clip at once and attends across space '
                f'and time — <b>+0.07</b> in one switch. We fine-tuned only its last blocks so it fit a 6&nbsp;GB GPU.<br><br>'
                f'<b>4 · Small, honest gains, stacked.</b> TTA (+0.008), an EMA recipe, and a committee. The error analysis '
                f'pointed at low light, so a revised-augmentation retrain added a third member → <b>{FINAL.get("accuracy",0.858):.3f}</b>.<br><br>'
                f'<b>Kept on the record as negatives:</b> strong geometric augmentation and label smoothing hurt; auto-curation '
                f'was a wash; an off-policy RL stacker did not beat plain averaging.</div>', unsafe_allow_html=True)
    else:
        m = models[labels.index(pick) - 1]
        final = '<span class="chip final">FINAL DELIVERABLE</span>' if m.get("final") else ""
        warn = '<span class="chip warn">did not help</span>' if m["key"] == "bigru" else ""
        st.markdown(f'<div class="sec">{m["name"]} {final}{warn}</div>', unsafe_allow_html=True)
        k = [(f"{m['accuracy']:.3f}", "test accuracy", m.get("task", ""), T["muted"])]
        if m.get("auc"):
            k.append((f"{m['auc']:.3f}", "ROC AUC", "", ""))
        if m.get("crash_f1"):
            k.append((f"{m['crash_f1']:.3f}", "crash-F1", "", ""))
        if m.get("val_acc"):
            k.append((f"{m['val_acc']:.3f}", "val accuracy", "", ""))
        kpis(k)
        left, right = st.columns([1, 1])
        with left:
            tech = TECHNIQUES.get(m["key"])
            st.markdown(f'<div class="card"><span class="chip">{m["family"]}</span>'
                        f'<div class="note" style="margin-top:.5rem"><b>Why we tried it.</b> {m["why"]}</div>'
                        f'<div class="note" style="margin-top:.7rem"><b>Size.</b> {m.get("params","")}</div>'
                        f'<div class="note" style="margin-top:.7rem"><b>Verdict.</b> {m["verdict"]}</div></div>',
                        unsafe_allow_html=True)
            if tech:
                st.markdown(f'<div class="card"><b>{tech[0]}</b><div class="note" style="margin-top:.5rem">{tech[1]}</div></div>',
                            unsafe_allow_html=True)
            if m.get("train_run"):
                show(fig_training(keys=[m["train_run"]]), height=320)
                st.caption("Training logs validation ROC-AUC per epoch (not per-epoch loss). Each epoch we evaluate both "
                           "the raw and the EMA weights and keep whichever is higher.")
        with right:
            layers = MODEL_LAYERS.get(m["key"])
            if layers:
                show_raw(fig_layers(layers, "Architecture (input → output)"))
                st.markdown(layer_legend(list(dict.fromkeys(L[4] for L in layers))), unsafe_allow_html=True)
            if m["family"] == "Video transformer" and m["key"] != "mvit_ensemble":
                with st.expander("Inside one MViT block · norm, attention, activation, hyperparameters"):
                    show_raw(fig_layers(MVIT_BLOCK, "One transformer block (repeated ×16)", height=520))
                    st.markdown('<div class="note">MViT-v2-S = a <b>Conv3d patch-embed</b> stem + <b>16 transformer blocks</b> '
                                '+ final norm/pool + head. Each block is <b>LayerNorm → multi-head pooled self-attention → '
                                'residual → LayerNorm → MLP (two Linear layers with a <b>GELU</b> activation, ×4 hidden) → '
                                'residual</b>. "Multiscale" = attention progressively <i>pools</i> the spatial tokens while '
                                '<i>growing</i> the channel width (96 → 768) and head count (1 → 8). <b>Norm</b> is LayerNorm; '
                                'the only nonlinearity inside a block is <b>GELU</b> in the MLP. <b>Hyperparameters:</b> 16 '
                                'frames, 224&nbsp;px, patch stride 2, ~34.2M params; we unfreeze the last 2–3 blocks + final '
                                'norm + head (~13.6M trainable), AdamW lr 1e-4, micro-batch 2 × 8 grad-accumulation.</div>',
                                unsafe_allow_html=True)

# --- Training -------------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="sec">Training dynamics'
                + tip("Every MViT run, validation ROC-AUC after each epoch. They converge fast, then flatten.")
                + '</div>', unsafe_allow_html=True)
    show(fig_training(), height=440)
    a, b = st.columns([2, 1])
    with a:
        st.markdown('<div class="card note">All four MViT runs converge within a few epochs and then sit against a '
                    '<b>~0.90 validation-AUC ceiling</b>. That plateau — not under-training — is the real reason 0.88 '
                    'accuracy is out of reach: 0.88 would need AUC ≈ 0.94. We evaluate raw <i>and</i> EMA weights each '
                    'epoch and keep the better; the threshold that turns probability into a decision is tuned on validation.</div>',
                    unsafe_allow_html=True)
    with b:
        kpis([(f"{FINAL.get('threshold', 0.71):.2f}", "decision threshold", "tuned on val", T["muted"]),
              ("~5–6 min", "per epoch", "6 GB GPU", "")])

# --- Error analysis -------------------------------------------------------
with tabs[3]:
    weak = M.get("weakness")
    st.markdown('<div class="sec">Where the model fails</div>', unsafe_allow_html=True)
    if not weak:
        st.info("Run `python -m crash_detection.analyze` then `python reports/build_assets.py`.")
    else:
        ov = weak["overall"]
        kpis([(f"{weak['n_held_out']}", "held-out clips", "val + test", T["muted"]),
              (f"{ov['recall']:.0%}", "recall", "crashes caught", T["green"]),
              (f"{ov['precision']:.0%}", "precision", "", ""),
              (f"{ov['fp']}", "false positives", "false alarms", T["red"]),
              (f"{ov['fn']}", "false negatives", "missed crashes", T["amber"])])
        fcol, _ = st.columns([1, 2])
        with fcol:
            pick = st.selectbox("Break performance down by", ["light_conditions", "weather", "scene"],
                                format_func=lambda s: s.replace("_", " ").title(),
                                help="Splits the held-out clips by this dashcam metadata field. Bars are recall and "
                                     "precision; diamonds show how many clips fall in each bucket (thin buckets are noisy).")
        show(fig_weakness(pick, weak["by_field"][pick]), height=430)
        with st.expander("Where to collect more data (from the recall + false-alarm gaps)", expanded=True):
            for r in weak["recommendations"]:
                st.markdown(f'<div class="note" style="margin:.35rem 0">• {r}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec">Inspect the mistakes'
                    + tip("Every held-out clip, with a plain-language reason. Filter by outcome, search any field, page through.")
                    + '</div>', unsafe_allow_html=True)
        et = error_table()
        if len(et):
            filter_table(et, key="errors", cat_cols=("result", "light", "weather", "scene"), height=380)
            st.caption("result: TP correct catch · TN correct pass · FP false alarm · FN missed crash.")

# --- RL experiment --------------------------------------------------------
with tabs[4]:
    rl = M.get("rl")
    st.markdown('<div class="sec">Reinforcement-learning stacking — a deep dive</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card note"><b>What is reinforcement learning?</b> An <b>agent</b> in a <b>state</b> s takes an '
        '<b>action</b> a, receives a <b>reward</b> r, and moves to a next state s′. It learns a policy that maximises total '
        'reward. <b>Q-learning</b> learns a value <b>Q(s,a)</b> = the expected future reward of taking action a in state s; '
        'the best action is then <code>argmax<sub>a</sub> Q(s,a)</code>. <b>DQN</b> (Deep Q-Network) approximates Q with a '
        'neural network and trains it <b>off-policy</b> from a <b>replay buffer</b> of past experience, using a slow-moving '
        '<b>target network</b> for stability.</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown('<div class="card"><b>The problem, framed as RL</b><div class="note" style="margin-top:.5rem">'
                    'Our three MViT models each output a crash probability; the deliverable just averages them. Could an agent '
                    'combine them <i>better</i> — trusting whichever model tends to be right in a given situation? We frame it '
                    'as a one-step decision (Lin, Chen &amp; Qi, 2019):<br>'
                    '&nbsp;• <b>state s</b> — the three 768-d embeddings + three probabilities (2307 numbers)<br>'
                    '&nbsp;• <b>action a</b> — predict <i>crash</i> or <i>no-crash</i><br>'
                    '&nbsp;• <b>reward r</b> — +1 if correct, −1 if wrong</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><b>A single training step (worked example)</b>'
                    '<div class="note" style="margin-top:.5rem">Clip #418 (a real crash). The three models say '
                    '<code>p = 0.91, 0.72, 0.88</code>; its state s stacks those with the embeddings. The agent (ε-greedy) '
                    'picks <b>crash</b>. It is right → <b>reward +1</b>. We store (s, a=crash, r=+1, s′) in the replay buffer. '
                    'Later a random batch of 128 such transitions is drawn and the Q-network is nudged so that '
                    '<code>Q(s, crash)</code> moves toward the target y below.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><b>The learning rule (Double-DQN)</b>'
                    '<div style="text-align:center;margin:.7rem 0;font-size:1.05rem;color:var(--text)">'
                    'y = r + γ · Q<sub>target</sub>(s′, argmax<sub>a</sub> Q<sub>online</sub>(s′, a))</div>'
                    '<div class="note">Minimise <code>Huber(Q(s,a) − y)</code>. Picking the action with the <i>online</i> net '
                    'but scoring it with the <i>target</i> net (that\'s the "double") removes the value over-estimation plain '
                    'DQN suffers. γ = 0.1 (near-bandit — clips are independent); ε decays 1.0 → 0.05.</div></div>',
                    unsafe_allow_html=True)
    with r2:
        show_raw(fig_layers(QNET_STACK, "The Q-network (state → Q-value per action)", height=560))
        st.markdown(layer_legend(["input", "trainable", "head"]), unsafe_allow_html=True)
        hp = (rl or {}).get("hyperparams", {})
        st.markdown('<div class="card"><b>Parameters</b>'
                    f'<div class="note" style="margin-top:.4rem">episodes {hp.get("episodes",60)} · γ {hp.get("gamma",0.1)} · '
                    f'lr {hp.get("lr",1e-3)} · batch {hp.get("batch",128)} · replay 50k · target-sync {hp.get("target_sync",250)} · '
                    f'ε {hp.get("eps",[1.0,0.05])[0]}→{hp.get("eps",[1.0,0.05])[1]} · hidden {hp.get("hidden",[256,128])} · '
                    f'seeds {len(hp.get("seeds",[0,1,2,3,4]))}</div></div>', unsafe_allow_html=True)

    if not rl:
        st.info("Run `python -m crash_detection.rl_ensemble` then `python reports/build_assets.py`.")
    else:
        st.markdown('<div class="sec">Results — did it help?</div>', unsafe_allow_html=True)
        rb, rc = st.columns([1, 1])
        with rb:
            show(fig_rl_bar(rl), height=380, legend=False)
        with rc:
            cc = fig_rl_curves(rl)
            if cc:
                show(cc, height=380)
        v = rl.get("rl_ddqn", {})
        st.markdown(
            f'<div class="card"><b>What happened, and why.</b><div class="note" style="margin-top:.5rem">'
            f'The agent reached <b>{v.get("accuracy",{}).get("mean",0):.3f} ± {v.get("accuracy",{}).get("std",0):.3f}</b> '
            f'(best seed {v.get("accuracy",{}).get("best",0):.3f}). It <b>beat</b> conventional logistic-regression and MLP '
            f'stackers on the same features, but <b>not</b> plain averaging '
            f'({rl.get("baselines",{}).get("averaging",{}).get("accuracy",0):.3f}). That is expected: the classes are balanced '
            f'and the three base models are already strong and <i>similar</i>, so there is little structured disagreement for a '
            f'combiner to exploit, and RL\'s exploration only adds variance. RL-for-classification earns its keep on '
            f'<i>imbalanced</i> data through reward shaping — not here. A clean, documented negative — exactly the kind of '
            f'result worth keeping honest.</div></div>', unsafe_allow_html=True)

# --- Try it ---------------------------------------------------------------
with tabs[5]:
    st.markdown('<div class="sec">Upload a dashcam clip and get a prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="note">The ensemble scans the clip in 16-frame windows (with a mirror copy) and reports the '
                'crash probability. If the file is one of our dataset clips, its true label is shown alongside.</div>',
                unsafe_allow_html=True)
    up = st.file_uploader("Dashcam video", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")
    if up:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
            tmp.write(up.read()); tmp_path = tmp.name
        vcol, rcol = st.columns([1, 1])
        vcol.video(tmp_path)
        with rcol:
            with st.spinner("Extracting frames and running the ensemble…"):
                try:
                    frames = frames_from_video(tmp_path)
                    models, device = load_models()
                    if not models:
                        st.error("No trained MViT models in artifacts/. Train them first (see RUN.md).")
                    elif frames is None or len(frames) < 16:
                        st.error("Could not read enough frames from this clip.")
                    else:
                        prob = predict(frames, models, device)
                        show_raw(gauge(prob))
                        got = "CRASH" if prob >= THRESHOLD else "normal"
                        hits = sample_index().get(Path(up.name).stem, [])
                        if hits:
                            h = hits[0]
                            exp = "CRASH" if h["label"] == 1 else "normal"
                            ok = exp == got
                            st.markdown(
                                f'<div class="card"><span class="chip">known sample · {h["split"]}</span><br>'
                                f'Predicted <b>{got}</b> ({prob:.0%}) &nbsp;·&nbsp; ground truth <b>{exp}</b> &nbsp;'
                                f'<span style="color:{T["green"] if ok else T["red"]};font-weight:700">'
                                f'{"correct" if ok else "wrong"}</span>'
                                f'<div class="note" style="margin-top:.4rem">conditions: {h["light"]}, {h["weather"]}, {h["scene"]}</div></div>',
                                unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="card"><span class="chip">out-of-sample clip</span><br>'
                                        f'Predicted <b>{"CRASH / near-miss" if prob>=THRESHOLD else "normal driving"}</b> '
                                        f'&nbsp;·&nbsp; crash probability <b>{prob:.0%}</b></div>', unsafe_allow_html=True)
                        st.caption(f"threshold {THRESHOLD:.2f} · {len(frames)} frames @ 6 fps · {len(models)}-model ensemble + mirror TTA")
                        idx = np.linspace(0, len(frames) - 1, 6).astype(int)
                        st.image([(frames[i] * 255).astype("uint8") for i in idx], caption=[f"{i/6:.1f}s" for i in idx], width=104)
                except Exception as e:
                    st.exception(e)

# --- Data & reproduction --------------------------------------------------
with tabs[6]:
    st.markdown('<div class="sec">The dataset</div>', unsafe_allow_html=True)
    d = M.get("data", {"train": 1049, "val": 226, "test": 225})
    kpis([(f"{sum(d.values()):,}", "recognition videos", "the 1,500 train-source clips", T["muted"]),
          (f"{d.get('train', 0):,}", "train", "70%", ""),
          (f"{d.get('val', 0):,}", "validation", "threshold + early stop", ""),
          (f"{d.get('test', 0):,}", "test", "scored once", "")])
    ca, cb = st.columns([1, 1])
    with ca:
        st.markdown('<div class="card note"><b>What a sample is.</b> A ~40 s dashcam clip, sampled at 6 frames/second and '
                    'resized to 224×224. <b>The label</b> is one column: a crash clip has a <code>time_of_event</code> '
                    'timestamp, a normal one is blank — so <code>time_of_event.notna()</code> is the label. Classes are '
                    'balanced 50/50. Each clip also carries <b>lighting, weather and scene</b> metadata, which the error '
                    'analysis is built on.<br><br><b>Tricks that made it tractable:</b> sample by timestamp (frame i = t·fps, '
                    'so any frame-rate works); cache frames as JPEG once; keep the whole clip and let max-pooling / windowing '
                    'find the brief crash.</div>', unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="card note"><b>Two things that shaped everything.</b><br>'
                    '① <b>Train ≠ test task.</b> The official test clips are cut ~1 s <i>before</i> the crash (anticipation); '
                    'the train clips show it (recognition). We graded on recognition.<br>'
                    '② <b>A leak we caught.</b> An early run scored a fake 0.978 because the feature extractor had seen the '
                    'test videos. The MViT pipeline splits <i>before</i> training, so it can\'t happen.<br><br>'
                    '<b>Reproducibility:</b> one settings file (<code>config.py</code>), deterministic splits (fixed seed), '
                    'crash-safe resumable training, and a fixed <code>requirements.txt</code>. Everything the dashboard needs '
                    'is committed under <code>reports/</code>; the heavy data + weights are regenerated by the steps below.</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="sec">Reproduce it — and what each step costs</div>', unsafe_allow_html=True)
    st.markdown("""
| step | command | time | internet | disk | output |
|---|---|---|---|---|---|
| 1 · Data | `python -m crash_detection.data` | ~1–2 h | yes (HF download) | ~37 GB | videos + 6 fps frames + manifest |
| 2 · Train | `python -m crash_detection.mvit --tag recipe --aug` | ~90 min/model (GPU) | no | ~130 MB/model | a fine-tuned MViT |
| 3 · Ensemble | `python -m crash_detection.mvit --ensemble` | ~10 min (GPU) | no | tiny | the 0.858 headline |
| 4 · Analyze | `python -m crash_detection.analyze` | ~13 min (GPU) | no | ~14 MB | per-sample predictions + embeddings |
| 5 · RL | `python -m crash_detection.rl_ensemble` | ~5 min (CPU) | no | tiny | the RL experiment result |
| 6 · Figures | `python reports/build_assets.py` | seconds | no | ~1 MB | figures + `metrics.json` |
| 7 · Dashboard | `streamlit run app_project2.py` | — | no | — | this page |
""")
    st.caption("Steps 2–5 need the ~6 GB GPU; the dashboard and figures run anywhere. Smoke test: add `--limit 20` to step 1. "
               "Full guide in RUN.md.")
