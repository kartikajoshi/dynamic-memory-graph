"""
Streamlit chat UI (owner: Kartika).

A web interface for the Dynamic Memory Graph pipeline with two modes:
    1. Chat     - talk to the DMG assistant with a LIVE split-screen graph
    2. Compare  - side-by-side baseline vs DMG on the same messages

Run from the project root (Ollama must be running):
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import os
import tempfile
from datetime import datetime

import ollama
from pyvis.network import Network

from config import (
    ANSWER_TEMPERATURE,
    CONTEXT_WINDOW,
    HISTORY_TURNS,
    MODEL_NAME,
    TOP_K,
)
from main import DMGPipeline, format_facts

# ─── Page config ───
st.set_page_config(
    page_title="Dynamic Memory Graph",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme: warm cream on charcoal, serif display headings ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg:          #0d0d0e;
    --panel:       #161617;
    --panel-2:     #1c1c1d;
    --border:      rgba(240,230,210,0.10);
    --border-soft: rgba(240,230,210,0.055);
    --cream:       #f2e8d5;
    --cream-dim:   #d7cdba;
    --text:        #e9e3d6;
    --muted:       #918b7e;
    --faint:       #605c54;
}

.stApp { background: var(--bg) !important; color: var(--text) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] .block-container { padding-top: 1.75rem !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 2.5rem !important; max-width: 1180px !important; }

/* ── Typography (never blanket span/div, or Material icons break) ── */
.stApp, .stApp p, .stApp li, .stApp label,
.stApp input, .stApp textarea,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text);
}
[data-testid="stIconMaterial"],
span[class*="material-symbols"],
.material-symbols-rounded, .material-symbols-outlined, .material-icons {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}
h1, h2, h3, .serif {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--cream) !important;
    letter-spacing: 0.2px !important;
}

/* ── Chat avatars ── */
[data-testid="stChatMessageAvatarUser"] { background: #2a2a2b !important; color: var(--cream) !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--cream) !important; color: #1a1a1a !important; }

/* ── Header bar ── */
.app-header {
    display: flex; align-items: center; gap: 15px;
    padding: 0 0 18px 0; border-bottom: 1px solid var(--border); margin-bottom: 20px;
}
.app-header .brand { font-family: 'Playfair Display', Georgia, serif; font-size: 25px; font-weight: 700; color: var(--cream); line-height: 1; }
.app-header .divider { width: 1px; height: 24px; background: var(--border); }
.app-header .subtitle { font-size: 13.5px; color: var(--muted); letter-spacing: 0.3px; }
.app-header .status { margin-left: auto; display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); }
.app-header .dot { width: 7px; height: 7px; border-radius: 50%; background: #6ee7a8; box-shadow: 0 0 8px rgba(110,231,168,0.6); }

/* ── Mode switcher pills ── */
div[data-testid="stRadio"] > div {
    display: inline-flex !important; gap: 4px !important;
    background: var(--panel-2) !important; border: 1px solid var(--border) !important;
    border-radius: 11px !important; padding: 4px !important;
}
div[data-testid="stRadio"] > div > label {
    text-align: center !important; padding: 8px 26px !important; border-radius: 8px !important;
    font-size: 14px !important; font-weight: 500 !important; color: var(--cream-dim) !important;
    cursor: pointer !important; transition: all 0.15s ease !important; border: none !important; background: transparent !important;
}
div[data-testid="stRadio"] > div > label:hover { color: var(--cream) !important; }
div[data-testid="stRadio"] > div > label:has(input:checked) { background: var(--cream) !important; box-shadow: 0 1px 5px rgba(0,0,0,0.45) !important; }
div[data-testid="stRadio"] > div > label:has(input:checked) * { color: #161616 !important; font-weight: 600 !important; }
div[data-testid="stRadio"] > label { display: none !important; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 9px !important; font-size: 14px !important; font-weight: 500 !important;
    padding: 9px 16px !important; border: 1px solid var(--border) !important;
    background: var(--panel-2) !important; color: var(--cream-dim) !important; transition: all 0.15s ease !important;
}
.stButton > button:hover { border-color: var(--cream-dim) !important; color: var(--cream) !important; background: #232324 !important; }
.stButton > button[kind="primary"] { background: var(--cream) !important; color: #161616 !important; border: 1px solid var(--cream) !important; font-weight: 600 !important; }
.stButton > button[kind="primary"]:hover { background: #fff4e2 !important; color: #000 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] { background: var(--panel-2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 15px 18px !important; }
[data-testid="stMetric"] label, [data-testid="stMetric"] label * { font-size: 11.5px !important; font-weight: 500 !important; letter-spacing: 0.4px !important; color: var(--muted) !important; text-transform: none !important; }
[data-testid="stMetricValue"] { font-family: 'Playfair Display', Georgia, serif !important; font-size: 29px !important; font-weight: 700 !important; color: var(--cream) !important; }

/* ── Labels ── */
.sec-label { font-size: 11.5px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); margin: 4px 0 11px 0; }
.panel-label { font-size: 11.5px; font-weight: 600; letter-spacing: 0.9px; text-transform: uppercase; color: var(--muted); margin: 0 0 12px 2px; display: flex; align-items: center; gap: 8px; }
.panel-label .live { width: 6px; height: 6px; border-radius: 50%; background: #a855f7; box-shadow: 0 0 7px rgba(168,85,247,0.7); }

/* ── Expanders ── */
[data-testid="stExpander"] { background: var(--panel-2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { font-size: 13px !important; font-weight: 500 !important; color: var(--cream-dim) !important; }
[data-testid="stExpander"] summary:hover { color: var(--cream) !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] { background: var(--panel) !important; border: 1px solid var(--border-soft) !important; border-radius: 13px !important; padding: 13px 17px !important; margin-bottom: 9px !important; }
[data-testid="stChatMessage"] p { font-size: 14.5px !important; line-height: 1.65 !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] { border-top: 1px solid var(--border) !important; padding-top: 16px !important; }
[data-testid="stChatInput"] textarea { background: var(--panel-2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; color: var(--text) !important; font-size: 14px !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--faint) !important; }
[data-testid="stChatInput"] button { color: var(--cream) !important; }

/* ── Component iframe (the graph) ── */
iframe { border-radius: 13px !important; border: 1px solid var(--border) !important; background: #141414 !important; }

/* ── Empty state ── */
.empty-wrap { text-align: center; padding: 54px 20px; }
.empty-wrap .glyph { font-size: 28px; color: var(--faint); margin-bottom: 13px; }
.empty-wrap h3 { font-family: 'Playfair Display', Georgia, serif !important; font-size: 24px !important; font-weight: 600 !important; color: var(--cream) !important; margin-bottom: 8px !important; }
.empty-wrap p { font-size: 13.5px !important; color: var(--muted) !important; max-width: 360px; margin: 0 auto; line-height: 1.55; }

/* ── Graph placeholder ── */
.graph-empty { border: 1px solid var(--border); border-radius: 13px; background: #141414; height: 560px; display: flex; align-items: center; justify-content: center; }

/* ── Code blocks ── */
.stCode, code { background: var(--panel-2) !important; border: 1px solid var(--border-soft) !important; color: var(--cream-dim) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 11px !important; border: 1px solid var(--border) !important; background: var(--panel-2) !important; }
[data-testid="stAlert"] * { color: var(--text) !important; }

/* ── Captions ── */
[data-testid="stCaptionContainer"], .stCaption, small { color: var(--muted) !important; font-size: 12px !important; }

/* ── Dividers ── */
hr { border-top: 1px solid var(--border) !important; margin: 15px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(240,230,210,0.15); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(240,230,210,0.3); }
</style>
""", unsafe_allow_html=True)


# ─── Shared pipeline ───
@st.cache_resource
def get_pipeline():
    return DMGPipeline()

pipeline = get_pipeline()


# ─── Obsidian-style graph renderer ───
def build_graph_html(graph, height="560px"):
    """Return an Obsidian-style interactive graph as an HTML string."""
    net = Network(height=height, width="100%", bgcolor="#141414",
                  font_color="#cfc8b8", directed=True)
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "scaling": { "min": 6, "max": 34, "label": { "enabled": false } },
        "font": { "color": "#cfc8b8", "size": 12, "face": "Inter, sans-serif", "vadjust": 2 },
        "borderWidth": 0,
        "shadow": { "enabled": true, "color": "rgba(0,0,0,0.5)", "size": 8, "x": 0, "y": 0 }
      },
      "edges": {
        "color": { "color": "rgba(220,210,190,0.13)", "highlight": "rgba(167,139,250,0.75)", "hover": "rgba(167,139,250,0.55)" },
        "width": 0.7, "selectionWidth": 1.5, "smooth": false,
        "arrows": { "to": { "enabled": false } }
      },
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": { "gravitationalConstant": -55, "centralGravity": 0.012, "springLength": 130, "springConstant": 0.07, "damping": 0.45, "avoidOverlap": 0.6 },
        "stabilization": { "iterations": 90 }
      },
      "interaction": { "hover": true, "tooltipDelay": 120, "hideEdgesOnDrag": false }
    }
    """)

    GREY = {"background": "#c6c2b6", "border": "#c6c2b6",
            "highlight": {"background": "#ffffff", "border": "#ffffff"},
            "hover": {"background": "#e8e4d8", "border": "#e8e4d8"}}
    PURPLE = {"background": "#a855f7", "border": "#c084fc",
              "highlight": {"background": "#c084fc", "border": "#d8b4fe"},
              "hover": {"background": "#c084fc", "border": "#d8b4fe"}}

    for node in graph.nodes():
        nd = graph.nodes[node]
        node_type = nd.get("type")
        label = nd.get("label", node)
        degree = graph.degree(node)
        is_hub = (node == "user")
        net.add_node(
            node, label=label,
            title=f"{label}  ·  {node_type or 'node'}  ·  {degree} link(s)",
            value=degree + 2,
            color=PURPLE if is_hub else GREY,
            font={"size": 15 if is_hub else 12, "face": "Inter, sans-serif",
                  "color": "#e9e3d6" if is_hub else "#b8b2a4", "vadjust": 2},
        )

    for u, v, k, data in graph.edges(keys=True, data=True):
        predicate = data.get("predicate", "").replace("_", " ")
        ended = data.get("ended_at")
        if ended:
            net.add_edge(u, v, title=f"{predicate} (retired)",
                         color="rgba(120,115,105,0.18)", dashes=True, width=0.6)
        else:
            net.add_edge(u, v, title=predicate, width=0.7)

    tmp_file = os.path.join(tempfile.gettempdir(), f"dmg_graph_{id(graph)}.html")
    net.save_graph(tmp_file)
    with open(tmp_file, "r", encoding="utf-8") as f:
        return f.read()


# ─── Header bar ───
st.markdown("""
<div class="app-header">
    <span class="brand">Dynamic Memory Graph</span>
    <span class="divider"></span>
    <span class="subtitle">Conversational memory for small language models</span>
    <span class="status"><span class="dot"></span> Ollama · {model}</span>
</div>
""".format(model=MODEL_NAME), unsafe_allow_html=True)


# ─── Sidebar ───
with st.sidebar:
    st.markdown('<div class="sec-label">Memory</div>', unsafe_allow_html=True)
    stats = pipeline.store.stats()
    col_n, col_e = st.columns(2)
    with col_n:
        st.metric("Entities", stats["nodes"])
    with col_e:
        st.metric("Relations", stats["edges"])

    with st.expander("View stored facts"):
        facts = pipeline.all_facts()
        if facts:
            for fact in facts:
                st.write(f"- {fact['text']}")
        else:
            st.caption("No facts stored yet.")

    st.divider()

    st.markdown('<div class="sec-label">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
1. **Resolve pronouns** — query rewriting
2. **Extract triples** — entity & relation extraction
3. **Update graph** — add / update / delete facts
4. **Retrieve context** — BM25 + dense + graph search
5. **Generate answer** — LLM with retrieved facts
""")

    st.divider()

    if st.button("Reset memory", key="sidebar_reset", use_container_width=True):
        pipeline.reset()
        st.session_state.messages = []
        st.session_state.compare_messages = []
        if "compare_pipeline" in st.session_state:
            st.session_state.compare_pipeline.reset()
        st.rerun()


# ─── Baseline helper ───
BASELINE_SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based on what the user has told you.
If you don't remember or don't know, say so honestly."""


def baseline_respond(message, history):
    messages = [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}]
    recent = (history or [])[-HISTORY_TURNS * 2:]
    messages.extend(recent)
    messages.append({"role": "user", "content": message})
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={"temperature": ANSWER_TEMPERATURE, "num_ctx": CONTEXT_WINDOW},
        )
        return response["message"]["content"].strip()
    except Exception as error:
        return f"[Error: {error}]"


# ─── Mode selector ───
mode = st.radio(
    "mode_selector",
    ["Chat", "Compare"],
    horizontal=True,
    label_visibility="collapsed",
    key="mode",
)

st.write("")


# ═══════════════════════════════════════════════════════════
# MODE: CHAT — split screen (conversation + live graph)
# ═══════════════════════════════════════════════════════════
if mode == "Chat":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_chat, col_graph = st.columns([5, 4], gap="large")

    # ── Left: conversation ──
    with col_chat:
        st.markdown('<div class="panel-label">Conversation</div>', unsafe_allow_html=True)

        if not st.session_state.messages:
            st.markdown("""
            <div class="empty-wrap">
                <div class="glyph">&#9671;</div>
                <h3>Start a conversation</h3>
                <p>Tell me something about yourself, then watch the graph on the
                right grow as the memory captures each new fact.</p>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("details"):
                    details = msg["details"]
                    has_any = (
                        (details.get("rewritten") and details["rewritten"] != details.get("original"))
                        or details.get("actions") or details.get("facts")
                    )
                    if has_any:
                        with st.expander("Pipeline details"):
                            if details.get("rewritten") and details["rewritten"] != details.get("original"):
                                st.markdown(f"**Query rewrite:** {details['rewritten']}")
                            if details.get("actions"):
                                st.markdown("**Memory actions:**")
                                for a in details["actions"]:
                                    st.write(f"- **{a['op']}** — {a['subject']} {a['predicate']} {a['object']}")
                            if details.get("facts"):
                                st.markdown("**Retrieved facts:**")
                                for f in details["facts"]:
                                    st.write(f"- {f['text']}")

    # ── Right: live graph ──
    with col_graph:
        graph = pipeline.store.graph
        st.markdown('<div class="panel-label"><span class="live"></span> Memory graph · live</div>', unsafe_allow_html=True)

        if graph.number_of_nodes() == 0:
            st.markdown("""
            <div class="graph-empty">
                <div style="text-align:center; color:#605c54;">
                    <div style="font-size:26px; margin-bottom:10px;">&#9676;</div>
                    <div style="font-size:13px;">The graph is empty.<br>Send a message to populate it.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Only rebuild the graph HTML when the graph actually changed
            sig = (graph.number_of_nodes(), graph.number_of_edges())
            if st.session_state.get("_graph_sig") != sig:
                st.session_state["_graph_html"] = build_graph_html(graph, height="560px")
                st.session_state["_graph_sig"] = sig
            components.html(st.session_state["_graph_html"], height=574, scrolling=False)
            active_count = sum(1 for _, _, d in graph.edges(data=True) if not d.get("ended_at"))
            st.caption(f"{graph.number_of_nodes()} entities · {active_count} active facts · hover a node for details")

            with st.expander("Stored facts"):
                active_facts = pipeline.store.get_all_facts(active_only=True)
                if active_facts:
                    for fact in active_facts:
                        st.write(f"- {fact['text']}")
                else:
                    st.caption("No facts yet.")

    # ── Input (full width) ──
    if user_input := st.chat_input("Message...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            result = pipeline.chat(user_input)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "details": {
                "original": user_input,
                "rewritten": result["rewritten_query"],
                "actions": result["actions"],
                "facts": result["retrieved_facts"],
            },
        })
        st.rerun()


# ═══════════════════════════════════════════════════════════
# MODE: COMPARE
# ═══════════════════════════════════════════════════════════
elif mode == "Compare":
    if "compare_messages" not in st.session_state:
        st.session_state.compare_messages = []
    if "baseline_history" not in st.session_state:
        st.session_state.baseline_history = []
    if "compare_pipeline" not in st.session_state:
        st.session_state.compare_pipeline = DMGPipeline(
            os.path.join(tempfile.gettempdir(), "dmg_compare_graph.json")
        )
        st.session_state.compare_pipeline.reset()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Baseline — no memory**")
    with col_r:
        st.markdown("**DMG — with memory graph**")

    st.divider()

    if st.session_state.compare_messages:
        for entry in st.session_state.compare_messages:
            st.markdown(f"**You:** {entry['user']}")
            cl, cr = st.columns(2)
            with cl:
                st.info(entry["baseline"])
            with cr:
                st.success(entry["dmg"])
                if entry.get("actions"):
                    with st.expander("Memory actions"):
                        for a in entry["actions"]:
                            st.write(f"- **{a['op']}** — {a['subject']} {a['predicate']} {a['object']}")
            st.divider()

        if st.button("Reset comparison", key="reset_compare"):
            st.session_state.compare_messages = []
            st.session_state.baseline_history = []
            st.session_state.compare_pipeline.reset()
            st.rerun()
    else:
        st.markdown("""
        <div class="empty-wrap">
            <div class="glyph">&#9672;</div>
            <h3>Start a comparison</h3>
            <p>Send a message to test the model with and without memory.
            Try these in order to see the difference.</p>
        </div>
        """, unsafe_allow_html=True)
        for s in [
            "I'm diabetic and I take Metformin every morning.",
            "I live in Bangalore and love filter coffee.",
            "My sister Priya lives in Pune.",
            "What medicine do I take?",
            "Where does my sister live?",
        ]:
            st.code(s, language=None)

    compare_input = st.chat_input("Type a message to compare...", key="compare_input")
    if compare_input:
        with st.spinner("Running both models..."):
            baseline_answer = baseline_respond(compare_input, st.session_state.baseline_history)
            st.session_state.baseline_history.append({"role": "user", "content": compare_input})
            st.session_state.baseline_history.append({"role": "assistant", "content": baseline_answer})
            st.session_state.baseline_history = st.session_state.baseline_history[-HISTORY_TURNS * 2:]

            dmg_result = st.session_state.compare_pipeline.chat(compare_input)
            st.session_state.compare_messages.append({
                "user": compare_input,
                "baseline": baseline_answer,
                "dmg": dmg_result["answer"],
                "actions": dmg_result["actions"],
                "facts": dmg_result["retrieved_facts"],
            })
        st.rerun()
