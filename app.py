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

# ─── Custom CSS — works on BOTH light and dark Streamlit themes ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

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

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header {
    visibility: hidden;
}

/* ── Mode switcher (radio as pills) ── */
div[data-testid="stRadio"] > div {
    display: flex !important;
    gap: 4px !important;
    background: rgba(128,128,128,0.15) !important;
    border-radius: 8px !important;
    padding: 3px !important;
}

div[data-testid="stRadio"] > div > label {
    flex: 1 !important;
    text-align: center !important;
    padding: 8px 16px !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    border: none !important;
    background: transparent !important;
}

div[data-testid="stRadio"] > div > label[data-checked="true"],
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: rgba(128,128,128,0.2) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.15) !important;
}

div[data-testid="stRadio"] > label {
    display: none !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    padding: 16px 20px !important;
    border-radius: 10px !important;
    margin-bottom: 4px !important;
    border: none !important;
}

[data-testid="stChatMessage"] p {
    font-size: 14.5px !important;
    line-height: 1.65 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-top: 1px solid rgba(128,128,128,0.2) !important;
    padding-top: 16px !important;
}

[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border-radius: 6px !important;
}

[data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    border-radius: 6px !important;
    padding: 12px 14px !important;
}

[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 600 !important;
}

/* ── Tabs override (for graph sub-tabs) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(128,128,128,0.2) !important;
    margin: 16px 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(128,128,128,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(128,128,128,0.5); }
</style>
""", unsafe_allow_html=True)


# ─── Shared pipeline ───
@st.cache_resource
def get_pipeline():
    return DMGPipeline()

pipeline = get_pipeline()


# ─── Sidebar ───
with st.sidebar:
    st.markdown("### Dynamic Memory Graph")
    st.caption(f"Persistent memory for {MODEL_NAME}")

    st.divider()

    # Graph stats
    st.markdown("**MEMORY**")
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
                st.write(f"- {fact['text']}")
        else:
            st.caption("No facts stored yet.")

    st.divider()

    # How it works
    st.markdown("**HOW IT WORKS**")
    st.markdown("""
1. **Resolve pronouns** — Query rewriting
2. **Extract triples** — Entity & relation extraction
3. **Update graph** — Add / update / delete facts
4. **Retrieve context** — BM25 + dense + graph search
5. **Generate answer** — LLM with retrieved facts
""")

    st.divider()

    # Reset
    if st.button("Reset memory", key="sidebar_reset", use_container_width=True):
        pipeline.reset()
        st.session_state.messages = []
        st.session_state.compare_messages = []
        if "compare_pipeline" in st.session_state:
            st.session_state.compare_pipeline.reset()
        st.rerun()


# ─── Baseline helper (must be defined before mode check) ───
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
    ["Chat", "Compare", "Graph"],
    horizontal=True,
    label_visibility="collapsed",
    key="mode",
)


# ═══════════════════════════════════════════════════════════
# MODE: CHAT
# ═══════════════════════════════════════════════════════════
if mode == "Chat":
    st.markdown("## Chat")
    st.caption("Talk naturally — the memory graph remembers your facts across conversations.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Empty state
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown(
            "&nbsp;\n\n"
            "**Start a conversation**\n\n"
            "Tell me something about yourself and then ask me to recall it later. "
            "The memory graph will persist your facts.\n\n"
            "&nbsp;"
        )
        st.markdown("---")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("details"):
                details = msg["details"]
                has_any = (
                    (details.get("rewritten") and details["rewritten"] != details.get("original"))
                    or details.get("actions")
                    or details.get("facts")
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

    # Chat input
    if user_input := st.chat_input("Message...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
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
                        st.markdown("**Memory actions:**")
                        for a in result["actions"]:
                            st.write(f"- **{a['op']}** — {a['subject']} {a['predicate']} {a['object']}")
                    if result["retrieved_facts"]:
                        st.markdown("**Retrieved facts:**")
                        for f in result["retrieved_facts"]:
                            st.write(f"- {f['text']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "details": details,
            })


# ═══════════════════════════════════════════════════════════
# MODE: COMPARE
# ═══════════════════════════════════════════════════════════
elif mode == "Compare":
    st.markdown("## Compare")
    st.caption("Send the same message to both models — see which one remembers.")

    if "compare_messages" not in st.session_state:
        st.session_state.compare_messages = []
    if "baseline_history" not in st.session_state:
        st.session_state.baseline_history = []
    if "compare_pipeline" not in st.session_state:
        st.session_state.compare_pipeline = DMGPipeline(
            os.path.join(tempfile.gettempdir(), "dmg_compare_graph.json")
        )
        st.session_state.compare_pipeline.reset()

    # Column headers
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Baseline — No Memory**")
    with col_r:
        st.markdown("**DMG — With Memory Graph**")

    st.divider()

    # Conversation history
    if st.session_state.compare_messages:
        for entry in st.session_state.compare_messages:
            st.markdown(f"**You:** {entry['user']}")
            col_left, col_right = st.columns(2)
            with col_left:
                st.info(entry["baseline"])
            with col_right:
                st.success(entry["dmg"])
                if entry.get("actions"):
                    with st.expander("Memory actions"):
                        for a in entry["actions"]:
                            st.write(f"- **{a['op']}** — {a['subject']} {a['predicate']} {a['object']}")
            st.divider()

        # Reset button
        if st.button("Reset comparison", key="reset_compare"):
            st.session_state.compare_messages = []
            st.session_state.baseline_history = []
            st.session_state.compare_pipeline.reset()
            st.rerun()

    else:
        # Suggestions
        st.markdown("**Try these messages in order to see the difference:**")
        suggestions = [
            "I'm diabetic and I take Metformin every morning.",
            "I live in Bangalore and love filter coffee.",
            "My sister Priya lives in Pune.",
            "What medicine do I take?",
            "Where does my sister live?",
        ]
        for s in suggestions:
            st.code(s, language=None)

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
    st.markdown("## Knowledge Graph")
    st.caption("Visual explorer for the entities and relations stored in memory.")

    graph = pipeline.store.graph

    if graph.number_of_nodes() == 0:
        st.markdown("---")
        st.markdown(
            "&nbsp;\n\n"
            "**No data yet**\n\n"
            "Chat with the assistant first to build the knowledge graph. "
            "Entities and relations will appear here.\n\n"
            "&nbsp;"
        )
        st.markdown("---")
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

        # Build pyvis graph — dark background for visibility
        type_colors = {
            "PERSON":     "#60a5fa",
            "CONDITION":  "#f87171",
            "DRUG":       "#a78bfa",
            "PLACE":      "#fbbf24",
            "PREFERENCE": "#34d399",
            None:         "#9ca3af",
        }

        net = Network(
            height="480px",
            width="100%",
            bgcolor="#1e1e2e",
            font_color="#e5e5e5",
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
                font={"size": 13, "face": "Inter, Arial, sans-serif", "color": "#e5e5e5"},
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
                             color="#555555", dashes=True, width=1,
                             font={"size": 10, "color": "#777777"})
            else:
                title = f"{label}\nCreated: {created}"
                net.add_edge(u, v, label=label, title=title,
                             color="#a0a0a0", width=2,
                             font={"size": 10, "color": "#cccccc"},
                             arrows={"to": {"enabled": True, "scaleFactor": 0.7}})

        tmp_file = os.path.join(tempfile.gettempdir(), "dmg_graph.html")
        net.save_graph(tmp_file)

        with open(tmp_file, "r", encoding="utf-8") as f:
            graph_html = f.read()

        components.html(graph_html, height=500, scrolling=False)

        # Legend using native Streamlit
        st.markdown(
            "**Legend:** "
            "🔵 Person · "
            "🔴 Condition · "
            "🟣 Drug · "
            "🟡 Place · "
            "🟢 Preference · "
            "⚪ Other"
        )

        # Fact tables
        tab_active, tab_retired = st.tabs(["Active facts", "Retired facts"])

        with tab_active:
            active_facts = pipeline.store.get_all_facts(active_only=True)
            if active_facts:
                for fact in active_facts:
                    ts = fact.get("created_at", "")
                    st.write(f"- {fact['text']}  *({ts})*")
            else:
                st.caption("No active facts.")

        with tab_retired:
            all_facts = pipeline.store.get_all_facts(active_only=False)
            retired = [f for f in all_facts if f.get("ended_at")]
            if retired:
                for fact in retired:
                    ts = fact.get("ended_at", "")
                    st.write(f"- ~~{fact['text']}~~  *({ts})*")
            else:
                st.caption("No retired facts.")

        # Refresh
        if st.button("Refresh graph", key="refresh_graph"):
            st.rerun()
