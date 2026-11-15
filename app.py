"""
Streamlit chat UI (owner: Kartika).

A web interface for the Dynamic Memory Graph pipeline with two modes:
    1. Chat        - talk to the DMG-powered assistant
    2. Compare     - side-by-side baseline vs DMG on the same messages

Plus a knowledge-graph explorer in the sidebar.

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
    page_title="DMG – Dynamic Memory Graph",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
/* ── Reset & Globals ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #fafafa;
    --bg-surface: #ffffff;
    --bg-surface-hover: #f5f5f5;
    --text-primary: #1a1a1a;
    --text-secondary: #6b7280;
    --text-tertiary: #9ca3af;
    --border-color: #e5e7eb;
    --border-light: #f0f0f0;
    --accent: #2563eb;
    --accent-soft: #eff6ff;
    --accent-hover: #1d4ed8;
    --green: #059669;
    --green-soft: #ecfdf5;
    --red: #dc2626;
    --red-soft: #fef2f2;
    --orange: #d97706;
    --purple: #7c3aed;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 2px 8px rgba(0,0,0,0.06);
}

/* ── Typography ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Main container ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 900px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-color) !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header {
    visibility: hidden;
}

/* ── Mode switcher (radio as pills) ── */
div[data-testid="stRadio"] > div {
    display: flex !important;
    gap: 4px !important;
    background: #f3f4f6 !important;
    border-radius: 8px !important;
    padding: 3px !important;
}

div[data-testid="stRadio"] > div > label {
    flex: 1 !important;
    text-align: center !important;
    padding: 8px 16px !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    border: none !important;
    background: transparent !important;
}

div[data-testid="stRadio"] > div > label[data-checked="true"],
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-sm) !important;
}

div[data-testid="stRadio"] > label {
    display: none !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    padding: 16px 20px !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 4px !important;
    border: none !important;
    background: transparent !important;
}

[data-testid="stChatMessage"][data-testid-type="user"] {
    background: transparent !important;
}

[data-testid="stChatMessage"] p {
    font-size: 14.5px !important;
    line-height: 1.65 !important;
    color: var(--text-primary) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-top: 1px solid var(--border-light) !important;
    padding-top: 16px !important;
}

[data-testid="stChatInput"] textarea {
    border: 1.5px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-surface) !important;
    transition: border-color 0.15s ease !important;
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    border: 1px solid var(--border-color) !important;
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    transition: all 0.15s ease !important;
    box-shadow: var(--shadow-sm) !important;
}

.stButton > button:hover {
    background: var(--bg-surface-hover) !important;
    border-color: #d1d5db !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-sm) !important;
    background: #fafbfc !important;
}

[data-testid="stExpander"] summary {
    font-size: 12.5px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    padding: 8px 12px !important;
}

[data-testid="stExpander"] .streamlit-expanderContent {
    font-size: 12.5px !important;
    color: var(--text-secondary) !important;
    padding: 4px 12px 12px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #f9fafb !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 14px !important;
}

[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    color: var(--text-tertiary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* ── Comparison columns ── */
.compare-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 4px 10px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 8px;
}

.compare-baseline {
    color: #b91c1c;
    background: #fef2f2;
}

.compare-dmg {
    color: #047857;
    background: #ecfdf5;
}

.compare-response {
    font-size: 14px;
    line-height: 1.6;
    padding: 14px 16px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-light);
    background: var(--bg-surface);
    margin-bottom: 12px;
}

.compare-user-msg {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    padding: 10px 0 6px;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 12px;
}

/* ── Sidebar graph stats ── */
.sidebar-section-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-tertiary);
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-light);
}

/* ── Suggestion chips ── */
.suggestion-chip {
    display: inline-block;
    padding: 6px 14px;
    margin: 3px 4px;
    border: 1px solid var(--border-color);
    border-radius: 20px;
    font-size: 13px;
    color: var(--text-secondary);
    background: var(--bg-surface);
    cursor: default;
    transition: all 0.15s ease;
}

.suggestion-chip:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-soft);
}

/* ── Page header ── */
.page-header {
    margin-bottom: 24px;
}

.page-header h1 {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
}

.page-header p {
    font-size: 13.5px !important;
    color: var(--text-secondary) !important;
    margin: 0 !important;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-tertiary);
}

.empty-state .icon {
    font-size: 32px;
    margin-bottom: 12px;
    opacity: 0.5;
}

.empty-state h3 {
    font-size: 16px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 6px;
}

.empty-state p {
    font-size: 13.5px;
    max-width: 360px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ── Graph legend ── */
.graph-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 10px 0;
}

.graph-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary);
}

.graph-legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-light) !important;
    margin: 16px 0 !important;
}

/* ── Scrollbar styling ── */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #d1d5db;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #9ca3af;
}

/* ── Tabs override (for graph sub-tabs) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.stTabs [data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    color: var(--text-secondary) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Info/warning boxes ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-size: 13px !important;
    border: none !important;
}

/* ── Pipeline detail styling ── */
.pipeline-detail {
    font-size: 12px;
    color: var(--text-tertiary);
    padding: 4px 0;
}

.pipeline-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    margin-right: 4px;
}

.tag-add { background: #ecfdf5; color: #059669; }
.tag-skip { background: #f3f4f6; color: #6b7280; }
.tag-update { background: #eff6ff; color: #2563eb; }
.tag-delete { background: #fef2f2; color: #dc2626; }
</style>
""", unsafe_allow_html=True)


# ─── Shared pipeline ───
@st.cache_resource
def get_pipeline():
    return DMGPipeline()

pipeline = get_pipeline()


# ─── Sidebar ───
with st.sidebar:
    # Logo / title
    st.markdown("""
    <div style="padding: 4px 0 16px;">
        <div style="font-size: 17px; font-weight: 600; color: #1a1a1a; letter-spacing: -0.3px;">
            Dynamic Memory Graph
        </div>
        <div style="font-size: 12px; color: #9ca3af; margin-top: 2px;">
            Persistent memory for Llama 3.2
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Graph stats
    st.markdown('<div class="sidebar-section-title">Memory</div>', unsafe_allow_html=True)
    stats = pipeline.store.stats()
    col_n, col_e = st.columns(2)
    with col_n:
        st.metric("Entities", stats["nodes"])
    with col_e:
        st.metric("Relations", stats["edges"])

    # Show facts
    with st.expander("View stored facts"):
        facts = pipeline.all_facts()
        if facts:
            for fact in facts:
                st.markdown(f"<div style='font-size:12.5px; color:#4b5563; padding:3px 0;'>&#8226; {fact['text']}</div>", unsafe_allow_html=True)
        else:
            st.caption("No facts stored yet.")

    st.divider()

    # How it works
    st.markdown('<div class="sidebar-section-title">How it works</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Resolve pronouns", "Query rewriting"),
        ("2", "Extract triples", "Entity & relation extraction"),
        ("3", "Update graph", "Add / update / delete facts"),
        ("4", "Retrieve context", "BM25 + dense + graph search"),
        ("5", "Generate answer", "LLM with retrieved facts"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex; gap:10px; align-items:flex-start; padding:5px 0;">
            <div style="min-width:22px; height:22px; border-radius:50%; background:#f3f4f6;
                        font-size:11px; font-weight:600; color:#6b7280; display:flex;
                        align-items:center; justify-content:center;">{num}</div>
            <div>
                <div style="font-size:13px; font-weight:500; color:#1a1a1a;">{title}</div>
                <div style="font-size:11.5px; color:#9ca3af;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Reset
    if st.button("Reset memory", key="sidebar_reset", use_container_width=True):
        pipeline.reset()
        st.session_state.messages = []
        st.session_state.compare_messages = []
        if "compare_pipeline" in st.session_state:
            st.session_state.compare_pipeline.reset()
        st.rerun()

    st.markdown(f"""
    <div style="position:fixed; bottom:16px; font-size:11px; color:#d1d5db;">
        Model: {MODEL_NAME}
    </div>
    """, unsafe_allow_html=True)


# ─── Mode selector ───
mode = st.radio(
    "mode_selector",
    ["Chat", "Compare", "Graph"],
    horizontal=True,
    label_visibility="collapsed",
    key="mode",
)


# ═══════════════════════════════════════════════════════════
# MODE: CHAT
# ═══════════════════════════════════════════════════════════
if mode == "Chat":
    st.markdown("""
    <div class="page-header">
        <h1>Chat</h1>
        <p>Talk naturally — the memory graph remembers your facts across conversations.</p>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Empty state
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">&#9673;</div>
            <h3>Start a conversation</h3>
            <p>Tell me something about yourself and then ask me to recall it later.
            The memory graph will persist your facts.</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("details"):
                details = msg["details"]
                detail_parts = []
                if details.get("rewritten") and details["rewritten"] != details.get("original"):
                    detail_parts.append(f"Rewritten: *{details['rewritten']}*")
                if details.get("actions"):
                    ops = []
                    for a in details["actions"]:
                        tag_class = f"tag-{a['op'].lower()}"
                        ops.append(f"<span class='pipeline-tag {tag_class}'>{a['op']}</span> {a['subject']} {a['predicate']} {a['object']}")
                    detail_parts.append("Memory: " + ", ".join(f"[{a['op']}] {a['subject']} → {a['object']}" for a in details["actions"]))
                if details.get("facts"):
                    detail_parts.append(f"Retrieved {len(details['facts'])} fact(s)")
                if detail_parts:
                    with st.expander("Pipeline details"):
                        if details.get("rewritten") and details["rewritten"] != details.get("original"):
                            st.markdown(f"**Query rewrite:** {details['rewritten']}")
                        if details.get("actions"):
                            for a in details["actions"]:
                                op = a["op"]
                                color = {"ADD": "#059669", "UPDATE": "#2563eb", "DELETE": "#dc2626"}.get(op, "#6b7280")
                                st.markdown(f"<span style='color:{color}; font-weight:500; font-size:12px;'>{op}</span> <span style='font-size:12.5px;'>{a['subject']} {a['predicate']} {a['object']}</span>", unsafe_allow_html=True)
                        if details.get("facts"):
                            st.markdown("**Retrieved facts:**")
                            for f in details["facts"]:
                                st.markdown(f"<span style='font-size:12.5px; color:#4b5563;'>&#8226; {f['text']}</span>", unsafe_allow_html=True)

    # Chat input
    if user_input := st.chat_input("Message...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner(""):
                result = pipeline.chat(user_input)

            st.markdown(result["answer"])

            details = {
                "original": user_input,
                "rewritten": result["rewritten_query"],
                "actions": result["actions"],
                "facts": result["retrieved_facts"],
            }

            has_details = (
                result["rewritten_query"] != user_input
                or result["actions"]
                or result["retrieved_facts"]
            )
            if has_details:
                with st.expander("Pipeline details"):
                    if result["rewritten_query"] != user_input:
                        st.markdown(f"**Query rewrite:** {result['rewritten_query']}")
                    if result["actions"]:
                        for a in result["actions"]:
                            op = a["op"]
                            color = {"ADD": "#059669", "UPDATE": "#2563eb", "DELETE": "#dc2626"}.get(op, "#6b7280")
                            st.markdown(f"<span style='color:{color}; font-weight:500; font-size:12px;'>{op}</span> <span style='font-size:12.5px;'>{a['subject']} {a['predicate']} {a['object']}</span>", unsafe_allow_html=True)
                    if result["retrieved_facts"]:
                        st.markdown("**Retrieved facts:**")
                        for f in result["retrieved_facts"]:
                            st.markdown(f"<span style='font-size:12.5px; color:#4b5563;'>&#8226; {f['text']}</span>", unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "details": details,
            })


# ═══════════════════════════════════════════════════════════
# MODE: COMPARE
# ═══════════════════════════════════════════════════════════

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


elif mode == "Compare":
    st.markdown("""
    <div class="page-header">
        <h1>Compare</h1>
        <p>Send the same message to both models. See which one remembers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Column headers
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<span class="compare-label compare-baseline">Baseline &mdash; No Memory</span>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<span class="compare-label compare-dmg">DMG &mdash; With Memory Graph</span>', unsafe_allow_html=True)

    if "compare_messages" not in st.session_state:
        st.session_state.compare_messages = []
    if "baseline_history" not in st.session_state:
        st.session_state.baseline_history = []
    if "compare_pipeline" not in st.session_state:
        st.session_state.compare_pipeline = DMGPipeline(
            os.path.join(tempfile.gettempdir(), "dmg_compare_graph.json")
        )
        st.session_state.compare_pipeline.reset()

    # Conversation history
    if st.session_state.compare_messages:
        for entry in st.session_state.compare_messages:
            st.markdown(f'<div class="compare-user-msg">{entry["user"]}</div>', unsafe_allow_html=True)
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(f'<div class="compare-response">{entry["baseline"]}</div>', unsafe_allow_html=True)
            with col_right:
                st.markdown(f'<div class="compare-response" style="border-left: 2px solid #059669;">{entry["dmg"]}</div>', unsafe_allow_html=True)
                if entry.get("actions"):
                    with st.expander("Memory actions"):
                        for a in entry["actions"]:
                            op = a["op"]
                            color = {"ADD": "#059669", "UPDATE": "#2563eb", "DELETE": "#dc2626"}.get(op, "#6b7280")
                            st.markdown(f"<span style='color:{color}; font-weight:500; font-size:12px;'>{op}</span> <span style='font-size:12.5px;'>{a['subject']} {a['predicate']} {a['object']}</span>", unsafe_allow_html=True)

        # Reset button at bottom
        if st.button("Reset comparison", key="reset_compare"):
            st.session_state.compare_messages = []
            st.session_state.baseline_history = []
            st.session_state.compare_pipeline.reset()
            st.rerun()

    else:
        # Suggestions for first-time
        st.markdown("""
        <div style="padding: 20px 0 8px;">
            <div style="font-size:13px; font-weight:500; color:#6b7280; margin-bottom:12px;">
                Try these messages in order to see the difference:
            </div>
        </div>
        """, unsafe_allow_html=True)

        suggestions = [
            "I'm diabetic and I take Metformin every morning.",
            "I live in Bangalore and love filter coffee.",
            "My sister Priya lives in Pune.",
            "What medicine do I take?",
            "Where does my sister live?",
        ]
        chips_html = "".join(f'<span class="suggestion-chip">{s}</span>' for s in suggestions)
        st.markdown(f'<div style="margin-bottom:16px;">{chips_html}</div>', unsafe_allow_html=True)

    # Input
    compare_input = st.chat_input("Type a message to compare...", key="compare_input")
    if compare_input:
        with st.spinner("Running both models..."):
            baseline_answer = baseline_respond(
                compare_input, st.session_state.baseline_history
            )
            st.session_state.baseline_history.append(
                {"role": "user", "content": compare_input}
            )
            st.session_state.baseline_history.append(
                {"role": "assistant", "content": baseline_answer}
            )
            st.session_state.baseline_history = st.session_state.baseline_history[
                -HISTORY_TURNS * 2:
            ]

            dmg_result = st.session_state.compare_pipeline.chat(compare_input)

            st.session_state.compare_messages.append({
                "user": compare_input,
                "baseline": baseline_answer,
                "dmg": dmg_result["answer"],
                "actions": dmg_result["actions"],
                "facts": dmg_result["retrieved_facts"],
            })
        st.rerun()


# ═══════════════════════════════════════════════════════════
# MODE: GRAPH
# ═══════════════════════════════════════════════════════════
elif mode == "Graph":
    st.markdown("""
    <div class="page-header">
        <h1>Knowledge Graph</h1>
        <p>Visual explorer for the entities and relations stored in memory.</p>
    </div>
    """, unsafe_allow_html=True)

    graph = pipeline.store.graph

    if graph.number_of_nodes() == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">&#9672;</div>
            <h3>No data yet</h3>
            <p>Chat with the assistant first to build the knowledge graph.
            Entities and relations will appear here.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Stats row
        active_count = sum(1 for _, _, d in graph.edges(data=True) if not d.get("ended_at"))
        retired_count = sum(1 for _, _, d in graph.edges(data=True) if d.get("ended_at"))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entities", graph.number_of_nodes())
        with col2:
            st.metric("Active facts", active_count)
        with col3:
            st.metric("Retired facts", retired_count)

        # Build pyvis graph
        type_colors = {
            "PERSON":     "#3b82f6",
            "CONDITION":  "#ef4444",
            "DRUG":       "#8b5cf6",
            "PLACE":      "#f59e0b",
            "PREFERENCE": "#10b981",
            None:         "#6b7280",
        }

        net = Network(
            height="480px",
            width="100%",
            bgcolor="#fafafa",
            font_color="#374151",
            directed=True,
        )
        net.barnes_hut(
            gravity=-2500,
            central_gravity=0.35,
            spring_length=160,
            spring_strength=0.04,
        )

        for node in graph.nodes():
            nd = graph.nodes[node]
            node_type = nd.get("type")
            label = nd.get("label", node)
            color = type_colors.get(node_type, type_colors[None])
            title = f"{label}\nType: {node_type or 'Unknown'}"
            net.add_node(
                node,
                label=label,
                color=color,
                title=title,
                size=28 if node == "user" else 20,
                font={"size": 13, "face": "Inter, Arial, sans-serif", "color": "#374151"},
                borderWidth=2,
                borderWidthSelected=3,
                shape="dot",
            )

        for u, v, k, data in graph.edges(keys=True, data=True):
            predicate = data.get("predicate", "")
            created = data.get("created_at", "")
            ended = data.get("ended_at")
            label = predicate.replace("_", " ")

            if ended:
                title = f"{label}\nCreated: {created}\nEnded: {ended}"
                net.add_edge(u, v, label=label, title=title,
                             color="#d1d5db", dashes=True, width=1,
                             font={"size": 10, "color": "#9ca3af"})
            else:
                title = f"{label}\nCreated: {created}"
                net.add_edge(u, v, label=label, title=title,
                             color="#6b7280", width=2,
                             font={"size": 10, "color": "#6b7280"},
                             arrows={"to": {"enabled": True, "scaleFactor": 0.7}})

        tmp_file = os.path.join(tempfile.gettempdir(), "dmg_graph.html")
        net.save_graph(tmp_file)

        with open(tmp_file, "r", encoding="utf-8") as f:
            graph_html = f.read()

        # Inject custom background
        graph_html = graph_html.replace(
            "background-color: #fafafa;",
            "background-color: #fafafa; border-radius: 10px; border: 1px solid #e5e7eb;"
        )

        components.html(graph_html, height=500, scrolling=False)

        # Legend
        st.markdown("""
        <div class="graph-legend">
            <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#3b82f6;"></span> Person</div>
            <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#ef4444;"></span> Condition</div>
            <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#8b5cf6;"></span> Drug</div>
            <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#f59e0b;"></span> Place</div>
            <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#10b981;"></span> Preference</div>
            <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#6b7280;"></span> Other</div>
        </div>
        """, unsafe_allow_html=True)

        # Fact tables
        tab_active, tab_retired = st.tabs(["Active facts", "Retired facts"])

        with tab_active:
            active_facts = pipeline.store.get_all_facts(active_only=True)
            if active_facts:
                for fact in active_facts:
                    ts = fact.get("created_at", "")
                    st.markdown(f"<div style='font-size:13px; padding:6px 0; border-bottom:1px solid #f3f4f6;'>"
                                f"<span style='color:#1a1a1a;'>{fact['text']}</span> "
                                f"<span style='color:#d1d5db; font-size:11px; margin-left:8px;'>{ts}</span>"
                                f"</div>", unsafe_allow_html=True)
            else:
                st.caption("No active facts.")

        with tab_retired:
            all_facts = pipeline.store.get_all_facts(active_only=False)
            retired = [f for f in all_facts if f.get("ended_at")]
            if retired:
                for fact in retired:
                    ts = fact.get("ended_at", "")
                    st.markdown(f"<div style='font-size:13px; padding:6px 0; border-bottom:1px solid #f3f4f6;'>"
                                f"<span style='color:#9ca3af; text-decoration:line-through;'>{fact['text']}</span> "
                                f"<span style='color:#d1d5db; font-size:11px; margin-left:8px;'>{ts}</span>"
                                f"</div>", unsafe_allow_html=True)
            else:
                st.caption("No retired facts.")

        # Refresh
        col_r, _ = st.columns([1, 4])
        with col_r:
            if st.button("Refresh", key="refresh_graph"):
                st.rerun()
