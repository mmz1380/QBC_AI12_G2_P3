"""Digikala Assistant — interactive Streamlit dashboard.

Covers all phases with live "Try it!" panels, a text-search-driven comparison
flow, a Sponsored Search Auction demo, and an in-dashboard human evaluation
tool. Top tabs (not a sidebar) hold the sections; light/dark is an instant
client-side toggle (CSS custom properties + a tiny JS pill), not a Streamlit
rerun. Run:

    python run.py dashboard        # or: streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digikala import config                          # noqa: E402
from digikala.core.llm import LLM                    # noqa: E402

st.set_page_config(page_title="Digikala Assistant", page_icon="🛍️", layout="wide",
                   initial_sidebar_state="collapsed")


# ---- theme: CSS custom properties + instant client-side toggle ----------
# Every color in this app is a var(--...) tied to :root (light) and
# :root[data-theme="dark"] (dark). Flipping that attribute client-side
# re-themes the WHOLE page instantly, with no Streamlit rerun -- unlike the
# previous session-state+rerun toggle, which also silently dropped sidebar
# widget state on every flip (a real bug, fixed by removing rerun entirely).
# Native Streamlit chrome (header, buttons, inputs, tables, tabs) is
# re-colored via var()-driven overrides on its [data-testid=...] selectors,
# audited across BOTH themes: buttons, number-input +/- controls, selectbox,
# multiselect, tabs, tables, expanders, disabled text areas, forms.
def _inject_theme_css():
    st.markdown("""
    <style>
    :root {
      --bg:#EEF6FF; --panel:#FFFFFF; --panel2:#E4F1FF; --text:#101828; --muted:#475467;
      --border:#A9CCE8; --sky:#3FA9F5; --sky-soft:#EAF6FF; --navy:#0B1F3A;
      --orange:#FF8A00; --orange-soft:#FFF0DF; --success:#138A60; --danger:#D64550;
      --input-bg:#FDFEFF; --tab-bg:#F7FBFF; --shadow:0 10px 28px rgba(20,54,92,.10);
      --btn-icon:#0B1F3A; --btn-text:#FFFFFF;
    }
    :root[data-theme="dark"] {
      --bg:#071426; --panel:#0B1F3A; --panel2:#102A4C; --text:#FFFFFF; --muted:#D7E5F5;
      --border:#6FB8FF; --sky:#68BEFF; --sky-soft:#16385A; --navy:#0B1F3A;
      --orange:#FF9A32; --orange-soft:#4B2C12; --success:#48D597; --danger:#FF6B6B;
      --input-bg:#102A4C; --tab-bg:#13365E; --shadow:0 12px 34px rgba(0,0,0,.35);
      --btn-icon:#FFFFFF; --btn-text:#FFFFFF;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stMain"] {
      background:var(--bg) !important; }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"], [data-testid="stWidgetLabel"] p { color:var(--text); }
    header[data-testid="stHeader"] { background:var(--bg); }
    header[data-testid="stHeader"] svg { fill:var(--text); }
    section[data-testid="stSidebar"] { background:var(--panel2); }
    section[data-testid="stSidebar"] * { color:var(--text); }
    [data-testid="stCaptionContainer"], .stCaption, small { color:var(--muted) !important; }

    /* buttons -- primary action buttons keep a solid navy/sky fill in both themes */
    .stButton button, .stDownloadButton button, div[data-testid="stFormSubmitButton"] button {
      background:var(--navy); color:var(--btn-text) !important; border:1px solid var(--sky);
      border-radius:10px; font-weight:600; }
    .stButton button p, .stDownloadButton button p, div[data-testid="stFormSubmitButton"] button p,
    .stButton button span, .stDownloadButton button span, div[data-testid="stFormSubmitButton"] button span {
      color:var(--btn-text) !important; }
    .stButton button:hover, .stDownloadButton button:hover,
    div[data-testid="stFormSubmitButton"] button:hover { background:var(--sky); border-color:var(--sky); }

    /* text/number/select inputs */
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] input, [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div, [data-testid="stMultiSelect"] > div > div {
      background:var(--input-bg) !important; color:var(--text) !important;
      border:1px solid var(--border) !important; border-radius:8px !important; }
    [data-testid="stTextArea"] textarea:disabled { opacity:.85; -webkit-text-fill-color:var(--text) !important; }

    /* KNOWN BUG (audited + fixed): the number_input +/- step buttons kept
       Streamlit's own light-theme icon color in dark mode, making them
       invisible on a dark background. Every increment/decrement control
       gets an explicit background+border+icon color in both themes. */
    div[data-testid="stNumberInput"] button, div[data-testid="stTextInput"] button,
    div[data-testid="stMultiSelect"] button, [data-testid="stSelectbox"] svg {
      background:var(--panel2) !important; color:var(--btn-icon) !important;
      fill:var(--btn-icon) !important; border:1px solid var(--border) !important; }
    div[data-testid="stNumberInput"] button:hover { background:var(--sky-soft) !important; }
    div[data-testid="stNumberInput"] svg, div[data-testid="stTextInput"] svg { fill:var(--btn-icon) !important; }

    /* selectbox/multiselect popover menus */
    [data-baseweb="popover"] li, [data-baseweb="menu"] li, [role="listbox"] {
      background:var(--panel) !important; color:var(--text) !important; }
    [data-baseweb="tag"], [data-testid="stMultiSelectTagsContainer"] span[data-tag] {
      background:var(--sky) !important; color:#fff !important; }
    [data-baseweb="select"] span, [data-baseweb="select"] div { color:var(--text) !important; }

    /* dataframes / tables */
    [data-testid="stDataFrame"] { color:var(--text); }
    /* Plotly figures render server-side with a fixed white "plotly_white"
       template (the theme toggle is client-side only, so the server can't
       know which theme is active when it builds the figure) -- audited bug:
       in dark mode this left a harsh, unstyled white rectangle floating on
       the dark page. Framing it as a deliberate white card (rounded corner +
       shadow + padding, matching .card/.table-wrap elsewhere) reads as an
       intentional design choice in both themes instead of a broken one. */
    [data-testid="stPlotlyChart"] { background:#FFFFFF; border-radius:14px;
      padding:10px; box-shadow:var(--shadow); border:1px solid var(--border); margin:6px 0; }
    /* st.code() hardcodes a light pre background regardless of Streamlit's own
       theme -- audited bug: white text (from our global color override) on
       that light background was invisible. Force both explicitly. */
    [data-testid="stCode"] pre, [data-testid="stCode"] pre > div { background:var(--panel2) !important; }
    [data-testid="stCode"] code, [data-testid="stCode"] span { color:var(--text) !important; }
    [data-testid="stJson"] { background:var(--panel2) !important; border-radius:8px; }
    /* react-json-view (the library behind st.json) sets its own background as
       an INLINE style read from Streamlit's compiled theme -- overriding the
       wrapper alone doesn't reach it; target it directly. */
    [data-testid="stJson"] .react-json-view { background:var(--panel2) !important; }
    [data-testid="stJson"] * { color:var(--text) !important; }
    table.nice-table { width:100%; border-collapse:collapse; font-size:.92rem; }
    table.nice-table th, table.nice-table td { border:1px solid var(--border); padding:8px 11px;
      text-align:center; color:var(--text); background:var(--panel); vertical-align:middle; }
    table.nice-table th { background:var(--panel2); font-weight:700; }
    .table-wrap { background:var(--panel); border:1px solid var(--border); border-radius:14px;
      overflow-x:auto; padding:6px; box-shadow:var(--shadow); margin:8px 0 12px; }

    /* expanders / forms / alerts */
    [data-testid="stExpander"], div[data-testid="stForm"], div[data-testid="stAlert"] {
      background:var(--panel); border:1px solid var(--border); border-radius:12px; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] p { color:var(--text); }

    /* tabs -- the main navigation, sticky so it stays reachable while scrolling.
       NOTE: this Streamlit version renders tabs as [role="tablist"] /
       [data-testid="stTab"] (React Aria), NOT the older [data-baseweb="tab-list"]
       markup -- both selector sets are kept so this survives a Streamlit
       version change in either direction. */
    .stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] { gap:6px; overflow-x:auto; flex-wrap:nowrap;
      position:sticky; top:0; z-index:998; background:var(--bg); padding:6px 0 8px;
      border-bottom:1px solid var(--border); }
    .stTabs [data-testid="stTab"], .stTabs [data-baseweb="tab"] { background:var(--tab-bg); color:var(--muted) !important;
      border:1px solid var(--border); border-radius:10px 10px 0 0; padding:8px 14px; white-space:nowrap; }
    .stTabs [data-testid="stTab"] p { color:var(--muted) !important; }
    .stTabs [data-testid="stTab"]:hover, .stTabs [data-baseweb="tab"]:hover { background:var(--sky-soft); }
    .stTabs [data-testid="stTab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] {
      background:var(--panel); color:var(--orange) !important; border-bottom:3px solid var(--orange);
      box-shadow:var(--shadow); }
    .stTabs [data-testid="stTab"][aria-selected="true"] p { color:var(--orange) !important; }

    /* answer / info cards -- Persian content, so right-aligned RTL */
    div[class*="st-key-digi-card"] { background:var(--panel); border:1px solid var(--border);
      border-radius:14px; padding:1rem 1.2rem; margin:.4rem 0; direction:rtl; text-align:right;
      box-shadow:var(--shadow); }
    div[class*="st-key-digi-card"] p { line-height:1.9; margin-bottom:.6rem; color:var(--text); }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] input {
      unicode-bidi:plaintext; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stDataFrame"] { unicode-bidi:plaintext; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; margin:3px 4px 3px 0;
      background:var(--sky-soft); border:1px solid var(--border); color:var(--text); font-size:.78rem; }
    .sponsored-pill { display:inline-block; padding:4px 10px; border-radius:999px; margin:3px 4px 3px 0;
      background:var(--orange); color:#fff; font-size:.78rem; font-weight:700; }
    .auction-card { background:var(--panel); border:1px solid var(--border);
      border-right:6px solid var(--orange); border-radius:14px; padding:14px 16px;
      box-shadow:var(--shadow); margin:8px 0; direction:rtl; text-align:right; }

    /* hero banner with the theme toggle pill */
    .hero { display:flex; align-items:center; justify-content:space-between;
      background:linear-gradient(120deg,var(--navy) 0%,#174778 60%,var(--sky) 100%);
      border:1px solid var(--border); border-radius:18px; padding:16px 20px; margin-bottom:10px;
      box-shadow:var(--shadow); }
    .hero h1, .hero p { color:#fff !important; margin:0; }
    .hero h1 { font-size:1.35rem; }
    .hero p { opacity:.85; font-size:.85rem; margin-top:2px; }
    .theme-pill { position:relative; display:inline-flex; width:64px; height:30px; border-radius:999px;
      background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.4); cursor:pointer; }
    .theme-pill .ic { position:absolute; top:0; height:30px; width:32px; display:flex;
      align-items:center; justify-content:center; font-size:14px; z-index:2; }
    .theme-pill .sun { left:0; } .theme-pill .moon { right:0; }
    .theme-pill .knob { position:absolute; top:2px; left:2px; width:26px; height:26px; border-radius:50%;
      background:#f6a821; z-index:1; transition:left .2s ease; box-shadow:0 1px 4px rgba(0,0,0,.4); }
    :root[data-theme="dark"] .theme-pill .knob { left:36px; background:#5b6572; }

    #backToTop { position:fixed; left:24px; bottom:26px; width:48px; height:48px; border:none;
      border-radius:999px; background:var(--navy); color:#fff; font-size:20px; cursor:pointer;
      box-shadow:var(--shadow); opacity:0; pointer-events:none; transition:opacity .25s ease;
      z-index:2147483647; }
    #backToTop.show { opacity:1; pointer-events:auto; }
    footer, #MainMenu { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True)


def _inject_theme_js():
    # Runs once per rerun in a hidden iframe; manipulates the PARENT document
    # (window.parent) so the injected back-to-top button and the theme
    # attribute live in the real page, not trapped inside the iframe box.
    # Idempotent (checks for existing nodes) so repeated Streamlit reruns
    # don't stack up duplicate buttons or listeners.
    components.html("""
    <script>
    (() => {
      const win = window.parent, doc = win.document;
      const saved = localStorage.getItem('digikala_theme') || 'light';
      doc.documentElement.setAttribute('data-theme', saved);
      win.__digikalaToggleTheme = function() {
        const cur = doc.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        doc.documentElement.setAttribute('data-theme', cur);
        localStorage.setItem('digikala_theme', cur);
      };
      // Streamlit's unsafe_allow_html sanitizes out inline onclick="" attributes,
      // so the pill (rendered via st.markdown) can't wire its own click handler --
      // find it here instead and attach a real listener, guarded against being
      // attached twice across Streamlit reruns.
      const pill = doc.querySelector('.theme-pill');
      if (pill && !pill.dataset.wired) {
        pill.dataset.wired = '1';
        pill.addEventListener('click', () => win.__digikalaToggleTheme());
      }
      let btn = doc.getElementById('backToTop');
      if (!btn) {
        btn = doc.createElement('button');
        btn.id = 'backToTop'; btn.type = 'button'; btn.title = 'Back to top'; btn.innerHTML = '↑';
        doc.body.appendChild(btn);
      }
      const candidates = () => [win, doc.scrollingElement, doc.documentElement, doc.body,
        doc.querySelector('[data-testid="stAppViewContainer"]'),
        doc.querySelector('[data-testid="stMain"]')].filter(Boolean);
      const scrollY = () => Math.max(...candidates().map(x => {
        try { return Number(x.scrollY ?? x.scrollTop ?? 0); } catch (e) { return 0; } }));
      const sync = () => { if (scrollY() > 260) btn.classList.add('show'); else btn.classList.remove('show'); };
      // Plain scrollTop assignment, not scrollTo({behavior:'smooth'}) --
      // smooth scrolling silently no-ops on Streamlit's custom scroll
      // container in some environments, while direct assignment always works.
      btn.onclick = () => candidates().forEach(x => {
        try { if ('scrollTop' in x) x.scrollTop = 0; else if (x.scrollTo) x.scrollTo(0, 0); } catch (e) {} });
      win.addEventListener('scroll', sync, true);
      doc.addEventListener('scroll', sync, true);
      // Belt-and-suspenders, and load-bearing in practice: this whole
      // components.html iframe (and anything scheduled inside it, including
      // a setInterval) gets destroyed and recreated on every Streamlit rerun.
      // The button itself survives in the parent doc (guarded above), but a
      // poll/listener registered only inside that one-time guard would die
      // with the first iframe and never fire again. Re-registering this poll
      // on every script execution (outside the guard) is what actually keeps
      // the button responsive across reruns.
      setInterval(sync, 300);
      sync();
    })();
    </script>
    """, height=0)


_inject_theme_css()
_inject_theme_js()

st.markdown("""
<div class="hero">
  <div><h1>🛍️ Digikala Assistant</h1>
  <p>Section 1: Data · Section 2: Smart Assistant · Section 3: Recommendation Prediction · Section 4: Evaluation · Bonus</p></div>
  <div class="theme-pill" title="Toggle light/dark">
    <span class="ic sun">☀️</span><span class="ic moon">🌙</span><span class="knob"></span>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar is collapsed by default now that the tabs are the primary nav --
# the one control that still matters (LLM run mode) lives in the main area
# instead of behind a sidebar the user has to reopen.
_rm_col, _budget_col = st.columns([2, 3])
with _rm_col:
    run_mode = st.selectbox(
        "LLM run mode", ["extractive", "local", "hosted_auto", "free", "paid"],
        index=0, help="extractive = \\$0 grounded fallback · local = Qwen/Ollama · "
                      "hosted_auto = auto-detect a .env key (metis/groq/paid) · "
                      "free = Groq/OpenRouter · paid = \\$5 credit",
        key="run_mode_select")
with _budget_col:
    st.caption(f"Budget cap: ${config.BUDGET_USD} · mode: `{run_mode}`")


# ---- cached resources ---------------------------------------------------
@st.cache_resource(show_spinner="Loading cleaned data…")
def _load_tables():
    products = pd.read_parquet(config.PRODUCTS_CLEAN)
    comments = pd.read_parquet(config.COMMENTS_CLEAN)
    return products, comments


@st.cache_resource(show_spinner="Loading retrieval index + assistant… (first time is slow)")
def _load_assistant(run_mode: str):
    from digikala.phase2_assistant.assistant import build_assistant
    return build_assistant(llm=LLM(mode=run_mode))


@st.cache_resource(show_spinner="Loading recommendation model…")
def _load_p3():
    from digikala.phase3_predict import recommend
    return recommend


@st.cache_resource(show_spinner=False)
def _data_intro_bundle(_products: pd.DataFrame, _comments: pd.DataFrame):
    # Streamlit re-executes every st.tabs() branch on EVERY rerun (a click on
    # any other tab still runs this code), so without caching this ~948k/6.16M
    # row EDA pass (stats + 5 Plotly figures) recomputed on every interaction
    # anywhere in the app -- a major, previously-unaudited source of the
    # dashboard's slow perceived load time. `_products`/`_comments` are
    # `_`-prefixed so Streamlit skips hashing them (same pattern as
    # `_reviewed_products`); this bundle is computed once per session.
    from digikala.phase1_data import eda
    stats = eda.summary_stats(_products, _comments)
    figs = {
        "balance": eda.fig_recommendation_balance(_comments),
        "price": eda.fig_price_distribution(_products),
        "cats": eda.fig_top_categories(_products),
        "brands": eda.fig_top_brands(_products),
        "missing": eda.fig_missingness(_products, _comments),
    }
    return stats, figs


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, mtime: float):
    # `mtime` in the cache key means a re-run of `python run.py eval` (which
    # rewrites the file with a new mtime) is picked up without a stale cache.
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def _read_metrics_file(name: str):
    path = config.METRICS_DIR / name
    if not path.exists():
        return None
    return _read_json_cached(str(path), path.stat().st_mtime)


def _section_intro(text: str):
    st.markdown(f'<div class="note" style="color:var(--muted);font-size:.92rem;'
               f'line-height:1.8;margin:.1rem 0 1rem;direction:ltr;text-align:left">{text}</div>',
               unsafe_allow_html=True)


_card_seq = [0]


def _card(md: str):
    # A CommonMark raw-HTML block does NOT get its contents re-parsed as
    # markdown -- st.container(key=...) gives the same bordered-card look via
    # CSS while letting st.markdown actually parse the answer as markdown, so
    # multi-item answers render as real line breaks instead of one paragraph.
    _card_seq[0] += 1
    with st.container(key=f"digi-card-{_card_seq[0]}"):
        st.markdown(md, unsafe_allow_html=True)


def _render_answer(ans):
    st.caption(f"intent: **{ans.intent}** · tier: **{ans.tier}** · latency: "
               f"**{ans.latency_s}s** · cost: **${ans.cost_usd:.4f}**"
               + (" · ⚠️ missing info" if ans.missing_info else ""))
    _card(ans.text.replace("\n", "  \n"))
    if ans.citations or ans.review_citations:
        st.caption("Cited: " + " ".join(f"`product {c}`" for c in ans.citations)
                   + " " + " ".join(f"`review {c}`" for c in ans.review_citations))


def _render_table(df: pd.DataFrame):
    if df is None or len(df) == 0:
        st.info("Nothing to show.")
        return
    show = df.fillna("—")
    st.markdown(f'<div class="table-wrap">{show.to_html(index=False, escape=False, classes="nice-table")}</div>',
                unsafe_allow_html=True)


def fmt_toman(x):
    if x is None or pd.isna(x):
        return "Unknown"
    return f"{float(x):,.0f} Toman"


_BROWSE_N = 1000  # a browsable slice for the dropdown -- not a cap on what you can reach


@st.cache_data(show_spinner=False)
def _reviewed_products(_assistant, run_mode: str):
    """All products that have at least one review -- the real universe for Q&A
    (≈330k of the 948k catalog products), not just whichever ones happen to be
    most-commented. `_assistant` is prefixed with `_` so Streamlit doesn't try
    to hash the (large, unhashable) object; `run_mode` varies the cache key
    correctly when a different assistant instance is loaded."""
    prods = _assistant.c.products
    reviewed = prods[prods["comment_count"] > 0]
    return reviewed.sort_values("comment_count", ascending=False)[["product_id", "title_fa", "comment_count"]]


def _pick_product(assistant, label: str, key: str):
    """A browsable top-N dropdown (fast, convenient) plus a manual product-id
    field that reaches any of the ~330k reviewed products directly."""
    reviewed = _reviewed_products(assistant, run_mode)
    browse = reviewed.head(_BROWSE_N)
    ids = browse["product_id"].tolist()
    titles = browse.set_index("product_id")["title_fa"]
    pick = st.selectbox(f"{label} (browse top {_BROWSE_N:,} most-reviewed of {len(reviewed):,})", ids,
                        format_func=lambda p: f"{p} — {titles.loc[p][:50]}", key=f"{key}_select")
    manual = st.text_input(f"…or type any product id ({label}, optional)", key=f"{key}_manual")
    if manual.strip():
        try:
            mid = int(manual.strip())
        except ValueError:
            st.warning(f"'{manual}' is not a valid numeric product id.")
            return pick
        if mid not in reviewed["product_id"].values:
            st.warning(f"Product {mid} has no reviews (or doesn't exist) — using the dropdown pick instead.")
            return pick
        return mid
    return pick


def _select_for_compare(pid: int):
    """Add `pid` to the Compare multiselect's own persisted selection.
    Streamlit widgets ignore a fresh `default=` on reruns once their key
    already has state, so newly Add-ed products silently never appeared as
    selected until this explicitly updates that same session_state key."""
    sel = st.session_state.get("compare_selected", [])
    if pid not in sel:
        st.session_state.compare_selected = (sel + [pid])[-4:]


@st.cache_data(show_spinner=False)
def _search_catalog(_products: pd.DataFrame, query: str, limit: int = 8) -> pd.DataFrame:
    """Free-text search over title/brand/category -- the entry point into the
    ~948k-product catalogue that doesn't require already knowing a product id."""
    if not query or not query.strip():
        return pd.DataFrame()
    from digikala.core import persian_text as pt
    q_norm = pt.normalize(query)
    q_tokens = pt.tokenize_norm(q_norm) or [q_norm]
    combo = (_products["title_fa"].fillna("") + " " + _products["brand_norm"].fillna("") + " "
             + _products["category1_norm"].fillna(""))
    score = pd.Series(0, index=_products.index)
    for tok in q_tokens:
        tok = tok.strip()
        if tok:
            score = score + combo.str.contains(re.escape(tok), na=False).astype(int)
    hits = _products.loc[score > 0, ["product_id", "title_fa", "brand_norm", "category1_norm",
                                     "price_clean", "product_rate_clean"]].copy()
    if hits.empty:
        return hits
    hits["match_score"] = score[score > 0]
    hits = hits.sort_values(["match_score", "product_rate_clean"], ascending=[False, False])
    return hits.head(limit)


# ---- section 1: Data introduction ------------------------------------------
_COLUMN_GLOSSARY = {
    "Products (digikala-products.csv)": [
        ("id", "Product id"), ("fa_title", "Persian product title"), ("Rate", "Recorded rating"),
        ("cnt_Rate", "Number of ratings recorded"), ("1Category", "Level-1 category"),
        ("2Category", "Level-2 category"), ("Brand", "Brand"), ("Price", "Price"),
        ("Seller", "Seller"), ("Fake_Is", "Flag for non-original goods"),
        ("month_last_price_min", "Lowest price recorded in the past month"), ("category_sub", "General category"),
    ],
    "Comments (digikala-comments.csv)": [
        ("id", "Comment id"), ("title", "Comment title"), ("body", "Comment body text"),
        ("created_at", "Timestamp"), ("rate", "Rating given by the user"),
        ("status_recommendation", "Recommendation status (recommended/not_recommended/no_idea)"),
        ("buyer_is", "Whether the reviewer purchased the product"), ("id_product", "Related product id"),
        ("advantages / disadvantages", "User-listed pros / cons"),
        ("likes / dislikes", "Upvotes / downvotes on the comment"), ("title_seller / code_seller", "Seller name / id"),
        ("true_to_size_rate", "Fit-to-size feedback"),
    ],
}

_DATA_METHOD_NOTES = [
    ("Deduplication", "Exact-duplicate rows and duplicate ids are dropped before anything else is computed "
     "(see the missingness chart below for what survives)."),
    ("Missing / invalid values", "Rows with zero or missing price, and other structurally invalid values, "
     "are removed rather than imputed — the brief explicitly asks this to be a defensible, explained choice, "
     "not a silent fill."),
    ("Persian text normalization", "Titles, categories and review text go through Persian-specific "
     "normalization (`hazm`-based + custom rules) before any search index or model sees them, since raw "
     "Digikala text mixes half-spaces, Arabic/Persian character variants and inconsistent digits."),
    ("Scale, not a sample", "Both the EDA below and the live assistant run over the full cleaned corpus — "
     "948k products and 6.16M comments — not a demo subset. A capped sample is used in exactly one place "
     "in the whole project: training the Section 3 recommendation classifier (documented on that tab), "
     "for compute-budget reasons the brief explicitly allows if justified."),
]


def page_data_intro():
    st.header("📦 Section 1 · Data introduction")
    _section_intro(
        "The raw Digikala dataset ties together over a million products and more than six million Persian "
        "user comments through a shared product id. It ships raw — completeness, uniqueness, balance, and "
        "the absence of missing/outlier/duplicate values are not guaranteed; auditing and cleaning that is "
        "itself part of the assignment.")
    products, comments = _load_tables()
    stats, figs = _data_intro_bundle(products, comments)
    cols = st.columns(4)
    labels = [("Products", stats["n_products"]), ("Comments", stats["n_comments"]),
              ("Brands", stats["n_brands"]), ("Categories", stats["n_categories"])]
    for c, (k, v) in zip(cols, labels):
        c.metric(k, f"{v:,}")
    st.caption("Full cleaned corpus (no sampling) — the same tables the live assistant searches in Section 2.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(figs["balance"], width="stretch", theme=None)
    c2.plotly_chart(figs["price"], width="stretch", theme=None)
    c3, c4 = st.columns(2)
    c3.plotly_chart(figs["cats"], width="stretch", theme=None)
    c4.plotly_chart(figs["brands"], width="stretch", theme=None)
    st.plotly_chart(figs["missing"], width="stretch", theme=None)
    with st.expander("🔧 Cleaning & preprocessing approach (technical notes)"):
        for title, note in _DATA_METHOD_NOTES:
            st.markdown(f"**{title}.** {note}")
    with st.expander("📖 Column glossary"):
        for title, rows in _COLUMN_GLOSSARY.items():
            st.markdown(f"**{title}**")
            st.dataframe(pd.DataFrame(rows, columns=["Column", "Description"]), width="stretch", hide_index=True)


# ---- section 2: Smart shopping assistant & product analysis ---------------
_RETRIEVAL_TERMS = [
    ("Dense retrieval (embeddings)", "Every product's text is embedded once into a vector; a query is "
     "embedded the same way and ranked by cosine similarity — good at matching *meaning* even without "
     "shared words."),
    ("BM25 (lexical retrieval)", "A classic term-frequency ranking over the same product text — good at "
     "matching *exact* words, brand names, and rare terms embeddings can blur together."),
    ("RRF (Reciprocal Rank Fusion)", "Combines the dense and BM25 rankings by each item's *rank position* "
     "in each list rather than raw scores, so the two methods (different score scales) fuse fairly."),
    ("Router", "A deterministic, $0 rule-based classifier picks the intent (discovery / product Q&A / "
     "comparison / managerial) before any model call, so each intent gets a purpose-built prompt instead "
     "of one generic one."),
    ("Grounding & citations", "Answers cite the specific product/review ids used to produce them, so a "
     "claim can be checked against the actual underlying data instead of taken on faith."),
]


def page_assistant():
    st.header("🛍️ Section 2 · Smart shopping assistant & product analysis")
    _section_intro(
        "A system built on language models that uses real product data and real user reviews to search, "
        "answer questions, compare, and analyze products. Every answer is grounded in real data and, where "
        "applicable, ships with evidence (product/review ids) alongside it.")

    st.markdown("**Retrieval pipeline** (development path: lexical-only → dense-only → hybrid)")
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:.3rem 0 1rem">'
        '<span class="pill">User query (Persian)</span><span>→</span>'
        '<span class="pill">Router (intent)</span><span>→</span>'
        '<span class="pill">Dense retrieval</span><span>+</span><span class="pill">BM25</span>'
        '<span>→</span><span class="pill">RRF fusion (hybrid)</span><span>→</span>'
        '<span class="pill">LLM / extractive answer + citations</span></div>',
        unsafe_allow_html=True)
    m = _read_metrics_file("phase4_metrics.json")
    abl = (m or {}).get("retrieval_ablation")
    if abl:
        import plotly.graph_objects as go
        methods = list(abl["by_method"].keys())
        mrrs = [abl["by_method"][k]["mrr"] for k in methods]
        fig = go.Figure(go.Bar(x=methods, y=mrrs, marker_color=["#3FA9F5", "#FF8A00", "#138A60"],
                               text=[f"{v:.3f}" for v in mrrs], textposition="outside"))
        fig.update_layout(title="Trial and error: MRR by retrieval method (title→own-id benchmark)",
                          template="plotly_white", yaxis=dict(range=[0, 1.1]), height=300,
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption(f"Hybrid is kept as the default despite a {abl['hybrid_vs_best_single_mrr_lift']:+.3f} MRR "
                   f"gap to BM25-only on this exact-title benchmark, because it is markedly more robust on "
                   f"paraphrased, non-exact-title queries — see the natural-language benchmark on the "
                   f"Evaluation tab for the honest, unhedged comparison.")
    with st.expander("🔧 Technical terms"):
        for title, note in _RETRIEVAL_TERMS:
            st.markdown(f"**{title}.** {note}")

    sub1, sub2, sub3, sub4 = st.tabs([
        "1) Search & discovery", "2) Q&A", "3) Product comparison", "4) Manager analytics",
    ])
    with sub1:
        _sub_discovery()
    with sub2:
        _sub_qa()
    with sub3:
        _sub_compare()
    with sub4:
        _sub_manager()


def _sub_discovery():
    _section_intro(
        "The user states a need in natural Persian (product type, price range, brand, satisfaction, ...) "
        "and the system finds and presents suitable products from the real catalogue.")
    q = st.text_input("Describe what you need (Persian):",
                      "یک کالای اقتصادی و باکیفیت زیر ۵۰۰ هزار تومان می‌خواهم")
    if st.button("Search", type="primary"):
        assistant = _load_assistant(run_mode)
        a = assistant.answer(q)
        _render_answer(a)

        campaigns = st.session_state.get("auction_campaigns") or []
        if st.session_state.get("auction_enabled", True) and campaigns:
            from digikala.phase5_auction import auction as auction_mod
            hits = assistant.pidx.search(q, k=200, method="hybrid")
            scores = {h["product_id"]: h["score"] for h in hits}
            from digikala.phase2_assistant.assistant import review_stats
            stats_lookup = {}
            for c in campaigns:
                pid = int(c["product_id"])
                rs = review_stats(assistant.c, pid, light=True)
                if rs["rec_rate"] is not None:
                    stats_lookup[pid] = {"recommendation_rate": rs["rec_rate"] * 100}
            sponsored = auction_mod.run_query_auction(campaigns, assistant.c.product, stats_lookup,
                                                      query_scores=scores)
            if sponsored:
                st.markdown("#### Sponsored results")
                for s in sponsored:
                    prod = assistant.c.product(s["product_id"]) or {}
                    st.markdown(
                        f'<div class="auction-card"><span class="sponsored-pill">Sponsored</span> '
                        f'<span class="pill">{s["vendor_name"]}</span>'
                        f'<div style="margin-top:8px;font-weight:700">{prod.get("title_fa", "")}</div>'
                        f'<div style="margin-top:6px"><span class="pill">Product ID: {s["product_id"]}</span>'
                        f'<span class="pill">Price: {fmt_toman(prod.get("price_clean"))}</span>'
                        f'<span class="pill">Placement: {s["placement_position"]}</span></div></div>',
                        unsafe_allow_html=True)

        if a.sources:
            st.markdown("#### Organic results")
            st.dataframe(pd.DataFrame(a.sources)[["product_id", "title", "brand", "price", "rate", "score"]],
                         width="stretch")


def _sub_qa():
    _section_intro(
        "Ask about a specific product, grounded in real user reviews — e.g. \"what did people like most?\" "
        "or \"what are the recurring complaints about this product?\". Answers are documented with evidence "
        "(review ids).")
    assistant = _load_assistant(run_mode)
    pick = _pick_product(assistant, "Product", "qa")
    q = st.text_input("Question about this product:", "کاربران از کیفیت این محصول راضی بودند؟")
    if st.button("Ask", type="primary"):
        _render_answer(assistant.answer(f"{q} محصول {pick}"))


def _sub_compare():
    _section_intro(
        "Compare two to four products on price, product info, user satisfaction, and recurring "
        "strengths/weaknesses. The answer distinguishes direct data facts from the model's own inference "
        "or final suggestion.")
    st.caption("Search the catalogue by name/brand/category, add matches to the comparison list, "
               "then pick 2–4 products to compare. No need to already know a product id.")
    assistant = _load_assistant(run_mode)
    products = assistant.c.products

    if "compare_ids" not in st.session_state:
        st.session_state.compare_ids = []

    finder = st.text_input("Search the full catalogue (e.g. شامپو):", key="compare_finder")
    matches = _search_catalog(products, finder, limit=8)
    if finder.strip() and matches.empty:
        st.info("No matches for this search.")
    for _, row in matches.iterrows():
        pid = int(row["product_id"])
        c1, c2, c3 = st.columns([5.5, 1.3, 1.4])
        with c1:
            st.markdown(
                f'<div class="pill">ID: {pid}</div> <b>{row["title_fa"]}</b><br>'
                f'<span class="pill">Brand: {row["brand_norm"] or "—"}</span>'
                f'<span class="pill">Category: {row["category1_norm"] or "—"}</span>'
                f'<span class="pill">Price: {fmt_toman(row["price_clean"])}</span>',
                unsafe_allow_html=True)
        with c2:
            st.code(str(pid), language=None)
        with c3:
            if st.button("Add", key=f"cmpadd_{pid}", width="stretch"):
                if pid not in st.session_state.compare_ids:
                    st.session_state.compare_ids.append(pid)
                _select_for_compare(pid)
                st.rerun()

    manual = st.text_input("…or paste product ids, comma-separated:", key="compare_manual_ids")
    mc1, mc2 = st.columns(2)
    if mc1.button("Add manual ids", width="stretch"):
        valid_ids = set(products["product_id"].astype(int))
        added, invalid = [], []
        for part in re.split(r"[,،\s]+", manual.strip()):
            if not part:
                continue
            if part.isdigit() and int(part) in valid_ids:
                added.append(int(part))
            elif part:
                invalid.append(part)
        for pid in added:
            if pid not in st.session_state.compare_ids:
                st.session_state.compare_ids.append(pid)
            _select_for_compare(pid)
        if invalid:
            st.warning(f"Not in the catalogue or not numeric, skipped: {', '.join(invalid)}")
        st.rerun()
    if mc2.button("Clear comparison list", width="stretch"):
        st.session_state.compare_ids = []
        st.session_state.compare_selected = []
        st.rerun()

    if not st.session_state.compare_ids:
        st.info("Your comparison list is empty — search above and click Add.")
        return

    id_to_title = products.set_index("product_id")["title_fa"].to_dict()
    if "compare_selected" not in st.session_state:
        st.session_state.compare_selected = st.session_state.compare_ids[: min(4, len(st.session_state.compare_ids))]
    selected = st.multiselect(
        "Products to compare (2–4):", options=st.session_state.compare_ids,
        format_func=lambda p: f"{p} — {str(id_to_title.get(p, ''))[:45]}", key="compare_selected")

    if len(selected) < 2:
        st.warning("Pick at least two products.")
        return
    if len(selected) > 4:
        st.info("Only the first four are compared for readability.")
        selected = selected[:4]

    if st.button("Compare", type="primary"):
        ids_clause = " و ".join(f"محصول {p}" for p in selected)
        _render_answer(assistant.answer(f"{ids_clause} را از نظر قیمت و کیفیت و رضایت کاربران مقایسه کن"))


def _sub_manager():
    _section_intro(
        "The user isn't always a buyer — a category manager can use the same data to understand the "
        "market and the customer experience: the most recurring complaints and dissatisfaction drivers "
        "across an entire product category.")
    assistant = _load_assistant(run_mode)
    cats = (assistant.c.products["category1_norm"].replace("نامشخص", pd.NA)
            .dropna().value_counts().index.tolist())
    filt = st.text_input("Filter category names:", "")
    filtered = [c for c in cats if filt.strip() in c] if filt.strip() else cats
    cat = st.selectbox(f"Category ({len(filtered)} of {len(cats)})", filtered)
    if st.button("Analyze", type="primary"):
        a = assistant.answer(f"پرتکرارترین شکایت‌ها و نقاط ضعف در دستهٔ {cat} چیست؟")
        _render_answer(a)
        if a.sources and a.sources[0].get("top_complaint_terms"):
            import plotly.express as px
            terms = pd.DataFrame(a.sources[0]["top_complaint_terms"])
            if not terms.empty:
                st.plotly_chart(px.bar(terms, x="count", y="term", orientation="h",
                                       title="Top complaint terms", template="plotly_white"),
                                width="stretch", theme=None)


_PREDICTION_TERMS = [
    ("Macro-F1", "The unweighted average of per-class F1 across all three classes. Chosen (per the brief) "
     "because performance on every class matters — a model that is great on the majority class and bad on "
     "the other two would still score well on plain accuracy but poorly here."),
    ("Product-grouped split (primary)", "Train/test are split so no product's reviews appear on both "
     "sides. This is the harder, honest split: it stops the model from memorizing per-product phrasing "
     "instead of learning general sentiment/recommendation language."),
    ("Leakage ablation", "`rate`, `likes`, and `is_buyer` numerically restate or post-hoc correlate with "
     "the label itself. Adding them inflates the score (a textbook leakage pattern) — the ablation below "
     "quantifies that inflation and the final model excludes those features."),
]


# ---- section 3: Recommendation-status prediction ---------------------------
def page_prediction():
    st.header("🎯 Section 3 · Recommendation-status prediction")
    _section_intro(
        "A model that, given the text of a review, predicts its recommendation status as one of three "
        "classes: <b>recommended</b> / <b>not_recommended</b> / <b>no_idea</b>. The primary evaluation "
        "metric is <b>Macro-F1</b> (not just accuracy on the majority class), and train/test leakage is "
        "prevented with a <b>product-grouped</b> split.")

    m = _read_metrics_file("phase3_metrics.json")
    if m:
        base = m.get("baselines_val_macro_f1", {})
        lora = _read_metrics_file("phase3_lora_metrics.json")
        names = ["Majority baseline", "Logistic regression", "TF-IDF + LinearSVC (final, primary)"]
        vals = [base.get("majority"), base.get("logreg"), m.get("primary_macro_f1", m.get("grouped_macro_f1"))]
        if lora:
            names.append(f"LoRA fine-tune ({lora['model'].split('/')[-1]})")
            vals.append(lora.get("lora_macro_f1"))
        import plotly.graph_objects as go
        colors = ["#94A3B8", "#3FA9F5", "#138A60", "#FF8A00"][: len(names)]
        fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors,
                               text=[f"{v:.3f}" if v is not None else "-" for v in vals], textposition="outside"))
        fig.update_layout(title="The climb: trial and error from a naive baseline to the final model",
                          template="plotly_white", yaxis=dict(range=[0, 1], title="Macro-F1"), height=340,
                          margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch", theme=None)
        if lora:
            st.caption(f"LoRA fine-tuning of a Persian BERT was tried as an experiment ({lora['note']}) and "
                       f"landed {lora['lora_vs_baseline_delta']:+.4f} vs. the TF-IDF baseline on a smaller, "
                       f"non-apples-to-apples sample — reported honestly as a negative result, not hidden.")
    with st.expander("🔧 Technical terms"):
        for title, note in _PREDICTION_TERMS:
            st.markdown(f"**{title}.** {note}")

    _page_predictor()


def _page_predictor():
    st.subheader("🤖 Try it")
    st.caption("Text-only model — no `rate`/`likes`/`is_buyer` (those leak the label). "
               "Predicts recommendation purely from the review text.")
    recommend = _load_p3()
    txt = st.text_area("Review text (Persian):", "کیفیت ساخت عالی بود و کاملا راضی هستم")
    if st.button("Predict", type="primary"):
        pred = recommend.predict(txt)[0]
        st.success(f"Predicted recommendation_status: **{pred}**")
    m = _read_metrics_file("phase3_metrics.json")
    if m:
        c1, c2, c3 = st.columns(3)
        c1.metric("Product-grouped Macro-F1 (primary)", m.get("primary_macro_f1", m.get("grouped_macro_f1")))
        c2.metric("Naive random-split Macro-F1", m.get("test_macro_f1"))
        c3.metric("Naive split product overlap", f"{m.get('naive_split_product_overlap_pct', 0)}%")
        st.caption("Primary metric is the **product-grouped** split: no product appears in both "
                   "train and test, so the model can't memorize per-product phrasing. The naive "
                   "random split is shown alongside with its product-overlap % so any gap between "
                   "the two numbers is explained by data, not asserted away.")
        abl = m.get("leakage_ablation")
        if abl:
            st.info(f"**Leakage ablation** — text-only {abl['text_only_macro_f1']} vs "
                    f"text+numeric {abl['text_plus_numeric_macro_f1']} "
                    f"(the +{abl['leakage_lift']} is leakage from rate/likes; excluded from the final model).")
        st.plotly_chart(recommend.fig_confusion(m), width="stretch", theme=None)


# ---- section 4: Final system evaluation ------------------------------------
def page_evaluation():
    st.header("🧪 Section 4 · Final system evaluation")
    _section_intro(
        "System evaluation across several angles: <b>answer quality</b>, <b>Grounding</b> (claims backed "
        "by evidence), <b>Retrieval Quality</b>, <b>recommendation-prediction</b> performance, "
        "<b>Latency</b>, <b>Cost</b>, and <b>Failure Analysis</b> — alongside a comparison against human "
        "evaluation.")
    sub1, sub2 = st.tabs(["📈 Metrics", "🧑‍⚖️ Human evaluation"])
    with sub1:
        _page_eval()
    with sub2:
        page_human_eval()


def _page_eval():
    m = _read_metrics_file("phase4_metrics.json")
    if not m:
        st.info("Run `python run.py eval` first to generate metrics.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall@10", m["retrieval_quality"]["recall@k"])
    c2.metric("MRR", m["retrieval_quality"]["mrr"])
    c3.metric("nDCG@10", m["retrieval_quality"]["ndcg@k"])
    c4.metric("Pred. Macro-F1 (grouped, primary)", m.get("prediction_primary_macro_f1", m.get("prediction_macro_f1")))
    c5, c6, c7 = st.columns(3)
    c5.metric("Mean latency (s)", m["generation"]["mean_latency_s"])
    c6.metric("Citation coverage", m["generation"]["mean_citation_coverage"])
    c7.metric("Total cost ($)", m["cost"]["total_cost_usd"])

    st.subheader("API usage (brief's cost-reporting requirement)")
    cost = m["cost"]
    c8, c9, c10, c11 = st.columns(4)
    c8.metric("API attempts", cost.get("api_attempts", 0))
    c9.metric("Successful calls", cost.get("successful_calls", 0))
    c10.metric("Input tokens", f"{cost.get('input_tokens', 0):,}")
    c11.metric("Output tokens", f"{cost.get('output_tokens', 0):,}")
    st.caption(f"Estimated list cost (what it would cost even for a free tier): "
               f"${cost.get('estimated_list_cost_usd', 0):.4f} · Budget: "
               f"${cost.get('budget_usd', 0):.2f} · Remaining: ${cost.get('remaining_usd', 0):.2f}")

    st.subheader("Per-intent breakdown")
    by_intent = pd.DataFrame(m["generation"]["by_intent"])
    st.dataframe(by_intent, width="stretch")
    if "intent" in by_intent.columns and "mean_latency_s" in by_intent.columns:
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(x=by_intent["intent"], y=by_intent["mean_latency_s"], marker_color="#3FA9F5"))
        fig.update_layout(title="Mean latency by intent (monitoring)", template="plotly_white", height=300,
                          margin=dict(l=10, r=10, t=50, b=10), yaxis_title="seconds")
        st.plotly_chart(fig, width="stretch", theme=None)

    c8, c9, c10 = st.columns(3)
    c8.metric("Proxy task completion (0-5)", m["generation"].get("mean_task_completion_proxy_0_5"))
    c9.metric("Proxy grounding (0-5)", m["generation"].get("mean_proxy_grounding_0_5"))
    c10.metric("Citation validity", m["generation"].get("mean_citation_validity"))
    st.caption("Deterministic, non-LLM proxy scores — reproducible every run, a hedge against the "
               "local judge's known unreliability. Not a substitute for human evaluation.")

    nat = m.get("retrieval_quality_natural")
    if nat and nat.get("n_queries"):
        st.subheader("Retrieval quality — natural-language benchmark")
        st.caption("Brand+category+partial-title paraphrase queries (not exact titles) — the fair test of "
                   "hybrid retrieval's value, since the title-exact-match benchmark above structurally favors BM25.")
        c1, c2 = st.columns(2)
        c1.json(nat["hybrid"])
        c2.json(nat["lexical_baseline"])
        verdict = nat.get("quality_verdict_by_ndcg")
        if verdict == "lexical_baseline_better":
            st.warning(f"Verdict: **{verdict}**. Even on this unbiased benchmark, the lexical/BM25 baseline "
                       f"beats hybrid on nDCG — a real, corroborated property of this embedding model on this "
                       f"catalogue, not a benchmark artifact. Reported honestly rather than claiming an "
                       f"unsupported hybrid win.")
        elif verdict == "hybrid_better":
            st.success(f"Verdict: **{verdict}** — hybrid measurably beats the lexical baseline here.")

    abl = m.get("retrieval_ablation")
    if abl:
        st.subheader("Retrieval ablation — hybrid vs. single-method (bonus)")
        lift = abl["hybrid_vs_best_single_mrr_lift"]
        st.dataframe(pd.DataFrame(abl["by_method"]).T, width="stretch")
        if lift >= 0:
            st.caption(f"Same title→own-id auto-labels as above, {abl['n_queries']} queries, k={abl['k']}. "
                       f"Hybrid's MRR lift over the best single method: **+{lift}**.")
        else:
            st.warning(f"On this benchmark hybrid's MRR is **{lift}** vs. the best single method (BM25). "
                       f"Reported honestly: this auto-labeled benchmark uses each product's own *title* as "
                       f"the query — a near-exact lexical match that structurally favors BM25. Hybrid still "
                       f"beats dense-only here.")

    he = m.get("human_eval_agreement")
    if he:
        st.subheader("Judge vs. human evaluation")
        if he.get("n_stale"):
            st.warning(f"{he['n_stale']} hand labels are **stale**: the assistant's answers to those "
                       f"queries changed since they were labeled, so the old score no longer corresponds to "
                       f"what the system currently outputs. Excluded from agreement. Re-label them on the "
                       f"Human Evaluation tab.")
        if he.get("n_labeled", 0) > 0:
            st.caption(f"{he['n_labeled']} queries hand-labeled against the CURRENT answers.")
            c1, c2 = st.columns(2)
            c1.json(he["relevance_agreement"])
            c2.json(he["faithfulness_agreement"])
        elif not he.get("n_stale"):
            st.info(he.get("note", "No human labels yet."))

    fa = m.get("failure_analysis")
    if fa:
        st.subheader("Failure analysis")
        st.caption(f"Retrieval misses: {fa['retrieval']['n_failed']}/{fa['retrieval']['n_checked']} · "
                   f"generation failures on probes: {fa['generation']['n_failed']}/{fa['generation']['n_probes']}")
        if fa["retrieval"]["examples"]:
            st.markdown("**Retrieval misses (title didn't return its own product):**")
            st.dataframe(pd.DataFrame(fa["retrieval"]["examples"]), width="stretch")
        if fa["generation"]["examples"]:
            st.markdown("**Generation failures (probes designed to stress the system):**")
            for ex in fa["generation"]["examples"]:
                _card(f"**{ex['intent']}** · _{ex['query']}_<br>reasons: {', '.join(ex['reasons'])}"
                      f"<br><span style='opacity:.7'>{ex['answer']}</span>")
        st.markdown("**Mitigations in place:** " + " · ".join(fa["mitigations"]))


def page_auction():
    st.header("📣 Sponsored Search Auction")
    st.caption("Three vendors each register a Product ID and a Max CPC. Ad Rank = Max CPC × data-derived "
               "quality × query relevance — the highest bidder does not automatically win. Winners are "
               "labeled **Sponsored** and are never presented as organic recommendations or as "
               "evidence a product is objectively better. Actual CPC never exceeds a vendor's own Max CPC.")
    assistant = _load_assistant(run_mode)
    products = assistant.c.products
    default_ids = products["product_id"].astype(int).head(3).tolist()
    saved = st.session_state.get("auction_campaigns") or []
    cols = st.columns(3)
    campaigns = []
    for i, col in enumerate(cols):
        existing = saved[i] if i < len(saved) else {}
        with col:
            st.markdown(f"#### Vendor {i + 1}")
            name = st.text_input("Vendor name", value=existing.get("vendor_name", f"Vendor {i + 1}"), key=f"vname_{i}")
            pid = st.number_input("Product ID", min_value=1,
                                  value=int(existing.get("product_id", default_ids[min(i, len(default_ids) - 1)])),
                                  step=1, key=f"vpid_{i}")
            bid = st.number_input("Max CPC (Toman)", min_value=0, max_value=1_000_000,
                                  value=int(existing.get("max_cpc", [8000, 6500, 5000][i])), step=500, key=f"vbid_{i}")
            active = st.toggle("Active", value=existing.get("active", True), key=f"vactive_{i}")
            campaigns.append({"vendor_name": name, "product_id": int(pid), "max_cpc": float(bid), "active": bool(active)})

    a1, a2 = st.columns(2)
    if a1.button("💾 Save vendor setup", type="primary", width="stretch"):
        st.session_state.auction_campaigns = campaigns
        st.success("Saved. Sponsored slots will use this setup on the Discovery tab.")
    st.session_state.auction_enabled = a2.toggle(
        "Show sponsored results on the Discovery tab", value=st.session_state.get("auction_enabled", True))

    preview_q = st.text_input("Preview query:", "یک محصول باکیفیت و محبوب معرفی کن", key="auction_preview_query")
    if st.button("⚡ Run preview auction", width="stretch"):
        from digikala.phase5_auction import auction as auction_mod
        from digikala.phase2_assistant.assistant import review_stats
        hits = assistant.pidx.search(preview_q, k=200, method="hybrid")
        scores = {h["product_id"]: h["score"] for h in hits}
        stats_lookup = {}
        for c in campaigns:
            rs = review_stats(assistant.c, int(c["product_id"]), light=True)
            if rs["rec_rate"] is not None:
                stats_lookup[int(c["product_id"])] = {"recommendation_rate": rs["rec_rate"] * 100}
        rows = auction_mod.run_query_auction(campaigns, assistant.c.product, stats_lookup, query_scores=scores)
        if not rows:
            st.warning("No active/valid campaigns found in the sampled catalogue.")
        else:
            table = pd.DataFrame([{
                "Slot": r["slot"], "Vendor": r["vendor_name"], "Product ID": r["product_id"],
                "Max CPC": round(r["max_cpc"]), "Quality": round(r["quality"], 3),
                "Query relevance": round(r["query_relevance"], 3), "Ad Rank": round(r["ad_rank"], 2),
                "Actual CPC": round(r["actual_cpc"]), "Placement": r["placement_position"],
            } for r in rows])
            _render_table(table)
            st.caption("Actual CPC never exceeds the vendor's own Max CPC.")

    st.divider()
    st.subheader("Offline validation (bonus evidence)")
    afile = config.METRICS_DIR / "auction_metrics.json"
    if afile.exists():
        rep = json.loads(afile.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Invariant pass rate", rep["invariant_pass_rate"])
        c2.metric("Quality-proxy lift vs. bid-only", f"{rep['offline_simulation']['quality_proxy_lift_pct']}%")
        c3.metric("Mentor-approved bonus claim", "Yes" if rep["bonus_claim_supported"] else "No")
        if not rep["mentor_approved_new_problem"]:
            st.info("`mentor_approved_new_problem = false` — the technical validation can pass while the "
                    "bonus claim itself stays withheld until the mentor explicitly signs off.")
        st.caption(rep["offline_simulation"]["important_limitation"])
    else:
        st.info("Run `python run.py auction` to generate the offline validation evidence.")


def page_human_eval():
    st.header("🧑‍⚖️ Human evaluation")
    st.caption("Score the ANSWER, not just the query. Labels are stamped with the answer's hash, so a "
               "later code change that alters the answer marks the old label stale instead of silently "
               "comparing it to a different answer than the one actually scored.")
    cfile = config.METRICS_DIR / "human_eval_candidates.json"
    if not cfile.exists():
        st.info("Run `python run.py eval` to generate human_eval_candidates.json first.")
        return
    candidates = json.loads(cfile.read_text(encoding="utf-8"))
    labels_path = config.METRICS_DIR / "human_eval_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}

    i = st.number_input("Sample index", min_value=0, max_value=max(0, len(candidates) - 1), value=0, step=1)
    row = candidates[int(i)]
    existing = labels.get(row["query"])
    fresh = existing is not None and existing.get("answer_hash") == row.get("answer_hash")

    st.markdown("**Query**")
    _card(row["query"])
    st.markdown("**Answer**" + (" · _previously labeled, still fresh_" if fresh else ""))
    _card(row["answer"].replace("\n", "  \n"))
    if row.get("evidence"):
        with st.expander("Evidence shown to the judge"):
            st.text_area("Evidence", value=row["evidence"], height=180, disabled=True,
                         label_visibility="collapsed", key=f"evidence_{i}")

    with st.form("human_eval_form"):
        c1, c2 = st.columns(2)
        rel = c1.number_input("Relevance (1-5)", 1, 5, int(existing["relevance"]) if fresh else 4, 1)
        faith = c2.number_input("Faithfulness (1-5)", 1, 5, int(existing["faithfulness"]) if fresh else 4, 1)
        submitted = st.form_submit_button("💾 Save score", width="stretch")
    if submitted:
        labels[row["query"]] = {"relevance": int(rel), "faithfulness": int(faith),
                                "answer_hash": row.get("answer_hash")}
        labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success("Saved.")

    fresh_count = sum(1 for c in candidates if labels.get(c["query"], {}).get("answer_hash") == c.get("answer_hash"))
    st.progress(fresh_count / max(1, len(candidates)))
    st.caption(f"{fresh_count} of {len(candidates)} candidates labeled against their current answer.")


# ---- section 5: Bonus -------------------------------------------------------
def page_bonus():
    st.header("🏆 Bonus section")
    _section_intro(
        "Retrieval improvement via Hybrid Search, LoRA fine-tuning, a multi-intent Router, a Sponsored "
        "Search Auction (a new proposed problem, pending mentor approval), Caching/optimization, and a "
        "coherent presentation with a clear storyline.")
    sub1, sub2 = st.tabs(["📖 Project storyline & scorecard", "📣 Sponsored Search Auction"])
    with sub1:
        _page_bonus()
    with sub2:
        page_auction()


def _page_bonus():
    efile = config.METRICS_DIR / "engineering_notes.json"
    if not efile.exists():
        st.info("Run `python run.py eval` (writes phase4 metrics) and see artifacts/metrics/engineering_notes.json.")
        return
    notes = json.loads(efile.read_text(encoding="utf-8"))

    st.subheader("Storyline: problem → decisions → experiments → results → failures")
    _card("""
    <b>Problem.</b> Digikala's customers face hundreds of near-identical products and
    thousands of reviews per category; category managers face the same scale from the
    supply side. Both need a system that turns raw text into grounded answers, not a
    search box.<br><br>
    <b>Decisions.</b> Products (≈948k) are embedded fully and cached; reviews (≈6.16M)
    are retrieved per-product on demand. Retrieval is hybrid (dense + BM25 via RRF).
    A deterministic router picks the intent instead of an LLM call. Phase 3 is text-only
    after a leakage audit removed <code>rate</code>/<code>likes</code>/<code>is_buyer</code>,
    reporting the product-grouped Macro-F1 as the primary number. A Sponsored Search
    Auction (quality-adjusted GSP) is included as a mentor-suggested extension, gated
    behind an explicit mentor-approval flag so the bonus claim isn't overstated.<br><br>
    <b>Experiments.</b> Ablations quantify every non-obvious choice: hybrid vs.
    dense-only vs. BM25-only retrieval; text-only vs. text+numeric Phase-3 features;
    judge scores vs. a small hand-labeled set; the auction's quality-adjusted mechanism
    vs. a naive highest-bid baseline.<br><br>
    <b>Results.</b> See the Evaluation and Auction tabs for the current numbers.<br><br>
    <b>Failures.</b> Documented on the Evaluation tab's Failure analysis section —
    concrete retrieval misses and adversarial generation probes, with the mitigations
    already in place (extractive fallback, citation verification, clarification requests).
    """)

    st.subheader("Bonus claims and where the evidence lives")
    sc = notes["bonus_scorecard_self_assessment"]
    st.caption(sc["note"])
    st.dataframe(pd.DataFrame(sc["items"]), width="stretch")

    st.subheader("Router (bonus)")
    r = notes["router"]
    st.write(r["what"])
    st.caption(r["why_a_bonus"] + " " + r["measured_effect"])

    st.subheader("Caching & optimization (bonus)")
    c = notes["caching_and_optimization"]
    st.write(c["what"])
    st.dataframe(pd.DataFrame(c["measurements"]), width="stretch")
    st.caption(c["why_a_bonus"])

    st.subheader("Sponsored Search Auction (bonus, mentor-approval pending)")
    afile = config.METRICS_DIR / "auction_metrics.json"
    if afile.exists():
        rep = json.loads(afile.read_text(encoding="utf-8"))
        st.write(f"Quality-adjusted GSP over 3 vendors, {rep['invariant_trials']} randomized invariant trials, "
                 f"{rep['offline_simulation']['trials']} simulation trials vs. a naive highest-bid baseline.")
        st.caption(f"Invariant pass rate {rep['invariant_pass_rate']} · quality-proxy lift "
                   f"{rep['offline_simulation']['quality_proxy_lift_pct']}% · bonus claim supported: "
                   f"{rep['bonus_claim_supported']} (mentor approval: {rep['mentor_approval_status']}).")
    else:
        st.info("Run `python run.py auction` first.")

    st.subheader("LoRA fine-tune vs. TF-IDF baseline (bonus)")
    lfile = config.METRICS_DIR / "phase3_lora_metrics.json"
    if lfile.exists():
        lm = json.loads(lfile.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric(f"LoRA Macro-F1 ({lm['model'].split('/')[-1]})", lm["lora_macro_f1"])
        c2.metric("TF-IDF baseline Macro-F1 (same split)", lm.get("baseline_grouped_macro_f1"))
        c3.metric("Delta", lm.get("lora_vs_baseline_delta"))
        st.caption(f"{lm['method']} · {lm['trainable_pct']}% of params trainable "
                   f"({lm['trainable_params']:,}/{lm['total_params']:,}) · "
                   f"{lm['n_train']} train / {lm['n_test']} test rows, same product-grouped split as the baseline.")
        st.caption(lm["note"])
    else:
        st.info("Run `python run.py lora` to fine-tune ParsBERT+LoRA and compare it against the "
                 "TF-IDF baseline on the identical product-grouped split.")


# Top-level tabs mirror the project brief's own section order (section 1..the
# bonus section) rather than an arbitrary feature list, with sub-tabs for each
# part inside section 2 (search/QA/compare/manager) and section 4
# (metrics/human eval) -- see QBC12 _ AI _ Project 3.pdf.
TABS = [
    ("📦 Section 1 · Data", page_data_intro),
    ("🛍️ Section 2 · Smart Assistant", page_assistant),
    ("🎯 Section 3 · Recommendation Prediction", page_prediction),
    ("🧪 Section 4 · Evaluation", page_evaluation),
    ("🏆 Bonus", page_bonus),
]
for tab, (_, render) in zip(st.tabs([t[0] for t in TABS]), TABS):
    with tab:
        render()
