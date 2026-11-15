"""
Streamlit chat UI (owner: Kartika).

A web interface for the Dynamic Memory Graph pipeline with three modes:
    1. Chat        - talk to the DMG-powered assistant
    2. Compare     - side-by-side baseline vs DMG on the same messages
    3. Graph       - interactive knowledge-graph visualization

Run from the project root (Ollama must be running):
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import os
import tempfile

import ollama
from pyvis.network import Network

from config import (
    ANSWER_TEMPERATURE,
    CONTEXT_WINDOW,
    HISTORY_TURNS,
    MODEL_NAME,
)
from main import DMGPipeline, format_facts

# ─── Page config ───
st.set_page_config(page_title="Dynamic Memory Graph", page_icon="🧠", layout="wide")

# ─── Shared pipeline ───
@st.cache_resource
def get_pipeline():
    """Create or reuse the pipeline across Streamlit reruns."""
    return DMGPipeline()

pipeline = get_pipeline()

# ─── Sidebar ───
with st.sidebar:
    st.header("🧠 DMG")
    stats = pipeline.store.stats()
    st.metric("Nodes", stats["nodes"])
    st.metric("Edges", stats["edges"])

    if st.button("Show all facts"):
        facts = pipeline.all_facts()
        if facts:
            for fact in facts:
                st.write(f"- {fact['text']}")
        else:
            st.info("No facts stored yet.")

    st.divider()
    if st.button("Reset memory", type="secondary"):
        pipeline.reset()
        st.session_state.messages = []
        st.session_state.compare_messages = []
        st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Your message is checked for pronouns\n"
        "2. Facts are extracted as triples\n"
        "3. New facts are added to the graph\n"
        "4. Relevant facts are retrieved\n"
        "5. The LLM answers using those facts"
    )

# ─── Tabs ───
tab_chat, tab_compare, tab_graph = st.tabs(["💬 Chat", "⚖️ Compare", "🕸️ Graph"])


# ═══════════════════════════════════════════════════════════
# TAB 1: CHAT (original functionality)
# ═══════════════════════════════════════════════════════════
with tab_chat:
    st.title("🧠 Dynamic Memory Graph")
    st.caption(f"Persistent conversational memory for {MODEL_NAME}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("details"):
                with st.expander("Pipeline details"):
                    details = msg["details"]
                    if details.get("rewritten") and details["rewritten"] != details.get("original"):
                        st.write(f"**Rewritten query:** {details['rewritten']}")
                    if details.get("actions"):
                        st.write("**Memory actions:**")
                        for action in details["actions"]:
                            st.write(f"- [{action['op']}] {action['subject']} "
                                     f"{action['predicate']} {action['object']}")
                    if details.get("facts"):
                        st.write("**Retrieved facts:**")
                        for fact in details["facts"]:
                            st.write(f"- {fact['text']}")

    if user_input := st.chat_input("Say something...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.chat(user_input)

            st.write(result["answer"])

            details = {
                "original": user_input,
                "rewritten": result["rewritten_query"],
                "actions": result["actions"],
                "facts": result["retrieved_facts"],
            }

            with st.expander("Pipeline details"):
                if result["rewritten_query"] != user_input:
                    st.write(f"**Rewritten query:** {result['rewritten_query']}")
                if result["actions"]:
                    st.write("**Memory actions:**")
                    for action in result["actions"]:
                        st.write(f"- [{action['op']}] {action['subject']} "
                                 f"{action['predicate']} {action['object']}")
                if result["retrieved_facts"]:
                    st.write("**Retrieved facts:**")
                    for fact in result["retrieved_facts"]:
                        st.write(f"- {fact['text']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "details": details,
            })


# ═══════════════════════════════════════════════════════════
# TAB 2: COMPARISON MODE (Baseline vs DMG)
# ═══════════════════════════════════════════════════════════

BASELINE_SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based on what the user has told you.
If you don't remember or don't know, say so honestly."""


def baseline_respond(message, history):
    """Get a response from plain Llama 3.2 3B with only a sliding window."""
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


with tab_compare:
    st.title("⚖️ Baseline vs DMG Comparison")
    st.caption("Same message sent to both — see which one remembers")

    if "compare_messages" not in st.session_state:
        st.session_state.compare_messages = []
    if "baseline_history" not in st.session_state:
        st.session_state.baseline_history = []
    if "compare_pipeline" not in st.session_state:
        st.session_state.compare_pipeline = DMGPipeline(
            os.path.join(tempfile.gettempdir(), "dmg_compare_graph.json")
        )
        st.session_state.compare_pipeline.reset()

    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("Reset comparison", key="reset_compare"):
            st.session_state.compare_messages = []
            st.session_state.baseline_history = []
            st.session_state.compare_pipeline.reset()
            st.rerun()

    # Show conversation history
    for entry in st.session_state.compare_messages:
        st.markdown(f"**You:** {entry['user']}")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("🔴 **Baseline (no memory)**")
            st.info(entry["baseline"])
        with col_right:
            st.markdown("🟢 **DMG (with memory graph)**")
            st.success(entry["dmg"])
            if entry.get("actions"):
                with st.expander("Memory actions"):
                    for action in entry["actions"]:
                        st.write(f"- [{action['op']}] {action['subject']} "
                                 f"{action['predicate']} {action['object']}")
            if entry.get("facts"):
                with st.expander("Retrieved facts"):
                    for fact in entry["facts"]:
                        st.write(f"- {fact['text']}")
        st.divider()

    # Suggested messages
    if not st.session_state.compare_messages:
        st.markdown("**Try these messages in order to see the difference:**")
        suggestions = [
            "I'm diabetic and I take Metformin every morning.",
            "I live in Bangalore and love filter coffee.",
            "My sister Priya lives in Pune.",
            "What is the capital of France?",
            "Can you recommend a good restaurant?",
            "What medicine do I take?",
            "Where does my sister live?",
        ]
        for s in suggestions:
            st.code(s, language=None)

    # Input
    compare_input = st.chat_input("Type a message to compare...", key="compare_input")
    if compare_input:
        with st.spinner("Running both models..."):
            # Baseline
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

            # DMG
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
# TAB 3: INTERACTIVE GRAPH
# ═══════════════════════════════════════════════════════════
with tab_graph:
    st.title("🕸️ Knowledge Graph")
    st.caption("Interactive visualization — drag, zoom, hover over nodes and edges")

    if st.button("Refresh graph", key="refresh_graph"):
        st.rerun()

    graph = pipeline.store.graph

    if graph.number_of_nodes() == 0:
        st.info("The graph is empty. Chat with the assistant first to build up some facts!")
    else:
        # Node stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entities", graph.number_of_nodes())
        with col2:
            active = sum(
                1 for _, _, data in graph.edges(data=True) if not data.get("ended_at")
            )
            st.metric("Active Facts", active)
        with col3:
            ended = sum(
                1 for _, _, data in graph.edges(data=True) if data.get("ended_at")
            )
            st.metric("Retired Facts", ended)

        # Build interactive graph with pyvis
        type_colors = {
            "PERSON": "#4CAF50",
            "CONDITION": "#F44336",
            "DRUG": "#2196F3",
            "PLACE": "#FF9800",
            "PREFERENCE": "#9C27B0",
            None: "#607D8B",
        }

        net = Network(
            height="500px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#333333",
            directed=True,
        )
        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.05,
        )

        # Add nodes
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get("type")
            label = node_data.get("label", node)
            color = type_colors.get(node_type, type_colors[None])
            title = f"{label}\nType: {node_type or 'Unknown'}"
            net.add_node(
                node,
                label=label,
                color=color,
                title=title,
                size=25 if node == "user" else 18,
                font={"size": 14, "face": "Arial"},
                borderWidth=2,
                borderWidthSelected=4,
            )

        # Add edges
        for u, v, k, data in graph.edges(keys=True, data=True):
            predicate = data.get("predicate", "")
            created = data.get("created_at", "")
            ended = data.get("ended_at")
            label = predicate.replace("_", " ")

            if ended:
                title = f"{label}\nCreated: {created}\nEnded: {ended}"
                net.add_edge(
                    u, v,
                    label=label,
                    title=title,
                    color="#BDBDBD",
                    dashes=True,
                    width=1,
                    font={"size": 10, "color": "#999999"},
                )
            else:
                title = f"{label}\nCreated: {created}"
                net.add_edge(
                    u, v,
                    label=label,
                    title=title,
                    color="#333333",
                    width=2,
                    font={"size": 11, "color": "#333333"},
                    arrows={"to": {"enabled": True, "scaleFactor": 0.8}},
                )

        # Save and display
        tmp_file = os.path.join(tempfile.gettempdir(), "dmg_graph.html")
        net.save_graph(tmp_file)

        with open(tmp_file, "r", encoding="utf-8") as f:
            graph_html = f.read()

        components.html(graph_html, height=520, scrolling=False)

        # Legend
        st.markdown("**Legend:**")
        legend_cols = st.columns(6)
        legend_items = [
            ("🟢", "Person"),
            ("🔴", "Condition"),
            ("🔵", "Drug"),
            ("🟠", "Place"),
            ("🟣", "Preference"),
            ("⚫", "Other"),
        ]
        for col, (emoji, label) in zip(legend_cols, legend_items):
            with col:
                st.markdown(f"{emoji} {label}")

        # Fact list
        with st.expander("All active facts"):
            for fact in pipeline.store.get_all_facts(active_only=True):
                st.write(f"- {fact['text']} _{fact.get('created_at', '')}_")

        with st.expander("Retired facts"):
            retired = pipeline.store.get_all_facts(active_only=False)
            retired = [f for f in retired if f.get("ended_at")]
            if retired:
                for fact in retired:
                    st.write(f"- ~~{fact['text']}~~ _{fact.get('ended_at', '')}_")
            else:
                st.write("No retired facts yet.")
