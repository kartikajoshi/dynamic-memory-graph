"""
Streamlit chat UI (owner: Kartika).

A web interface for the Dynamic Memory Graph pipeline with two modes:
    1. Chat     - talk to the DMG assistant with a LIVE split-screen graph
    2. Compare  - side-by-side baselines vs DMG on the same messages

Nov 18 update: one pipeline per browser session; the live graph is coloured by
entity type, highlights what changed / was used each turn, keeps a stable
layout and can replay how memory grew. Compare mode now has a model picker,
two fair baselines (last 4 turns / full history), latency + token evidence,
and a "Run demo script" button that scores every column.

Run from the project root (Ollama must be running):
    streamlit run app.py
"""

import html
import os
import tempfile
import uuid

import streamlit as st

from config import GRAPH_PATH, MODEL_NAME
from modules.graph_viz import build_graph_html, changes_at, stable_layout, timeline_steps
from ui_core import (  # helpers shared with server.py (moved here unchanged)
    COLUMNS,
    EXAMPLES_CHAT,
    EXAMPLES_COMPARE,
    SHORT_NAMES,
    change_caption,
    check_graph,
    column_scores,
    compare_turn,
    diff_changes,
    dmg_respond,
    evidence_line,
    facts_markdown,
    graph_snapshot,
    load_script,
    make_pipeline,
    memory_counts,
)
from ui_core import list_models as _list_models

# ─── Page config ───
st.set_page_config(
    page_title="Dynamic Memory Graph",
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto",
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
.block-container { padding-top: 2rem !important; padding-bottom: 2.5rem !important; max-width: 1240px !important; }

/* ── Typography (never blanket span/div, or Material icons break) ── */
.stApp, .stApp p, .stApp li, .stApp label,
.stApp input, .stApp textarea,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--text);
}
[data-testid="stIconMaterial"],
span[class*="material-symbols"],
.material-symbols-rounded, .material-symbols-outlined, .material-icons {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}
h1, h2, h3, .serif {
    font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important;
    color: var(--cream) !important;
    letter-spacing: 0.2px !important;
}

/* ── Chat avatars ── */
[data-testid="stChatMessageAvatarUser"] { background: #2a2a2b !important; color: var(--cream) !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--cream) !important; color: #1a1a1a !important; }

/* ── Header bar ── */
.app-header {
    display: flex; align-items: center; flex-wrap: wrap; gap: 8px 15px;
    padding: 0 0 18px 0; border-bottom: 1px solid var(--border); margin-bottom: 20px;
}
.app-header .brand { font-family: 'Playfair Display', Georgia, serif; font-size: 25px; font-weight: 700; color: var(--cream); line-height: 1; }
.app-header .divider { width: 1px; height: 24px; background: var(--border); }
.app-header .subtitle { font-size: 13.5px; color: var(--muted); letter-spacing: 0.3px; }
.app-header .status { margin-left: auto; display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); }
.app-header .dot { width: 7px; height: 7px; border-radius: 50%; background: #6ee7a8; box-shadow: 0 0 8px rgba(110,231,168,0.6); }
.app-header .dot.off { background: #f87171; box-shadow: 0 0 8px rgba(248,113,113,0.6); }
@media (max-width: 700px) { .app-header .divider, .app-header .subtitle { display: none; } }

/* ── Mode switcher pills (Streamlit 1.5x+ radio DOM) ── */
[data-testid="stRadioGroup"] {
    display: inline-flex !important; gap: 4px !important;
    background: var(--panel-2) !important; border: 1px solid var(--border) !important;
    border-radius: 11px !important; padding: 4px !important;
}
[data-testid="stRadioOption"] {
    padding: 8px 26px !important; margin: 0 !important; border-radius: 8px !important;
    cursor: pointer !important; transition: all 0.15s ease !important;
}
[data-testid="stRadioOption"] > div > div:first-child { display: none !important; }  /* radio circle */
[data-testid="stRadioOption"] p { font-size: 14px !important; font-weight: 500 !important; color: var(--cream-dim) !important; }
[data-testid="stRadioOption"]:hover p { color: var(--cream) !important; }
[data-testid="stRadioOption"]:has(input:checked) { background: var(--cream) !important; box-shadow: 0 1px 5px rgba(0,0,0,0.45) !important; }
[data-testid="stRadioOption"]:has(input:checked) p { color: #161616 !important; font-weight: 600 !important; }

/* ── Buttons ── */
.stButton button {
    border-radius: 9px !important; font-size: 14px !important; font-weight: 500 !important;
    padding: 9px 16px !important; border: 1px solid var(--border) !important;
    background: var(--panel-2) !important; color: var(--cream-dim) !important; transition: all 0.15s ease !important;
}
.stButton button:hover { border-color: var(--cream-dim) !important; color: var(--cream) !important; background: #232324 !important; }
.stButton button[kind="primary"] { background: var(--cream) !important; color: #161616 !important; border: 1px solid var(--cream) !important; font-weight: 600 !important; }
.stButton button[kind="primary"] * { color: #161616 !important; }
.stButton button[kind="primary"]:hover { background: #fff4e2 !important; color: #000 !important; }
/* Example prompts: compact suggestion chips */
[class*="st-key-ex_"] button { font-size: 13px !important; padding: 8px 13px !important; }
[class*="st-key-ex_"] button p { color: var(--cream-dim) !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] { background: var(--panel-2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 12px 16px !important; }
[data-testid="stMetric"] label, [data-testid="stMetric"] label * { font-size: 11.5px !important; font-weight: 500 !important; letter-spacing: 0.4px !important; color: var(--muted) !important; text-transform: none !important; }
[data-testid="stMetricValue"] { font-family: 'Playfair Display', Georgia, serif !important; font-size: 26px !important; font-weight: 700 !important; color: var(--cream) !important; }

/* ── Sidebar memory counts ── */
.stat-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 12px; }
.stat-row div { background: var(--panel-2); border: 1px solid var(--border); border-radius: 11px; padding: 10px 8px; text-align: center; }
.stat-row b { display: block; font-family: 'Playfair Display', Georgia, serif; font-size: 24px; color: var(--cream); line-height: 1.1; }
.stat-row span { font-size: 10.5px; color: var(--muted); letter-spacing: 0.3px; }

/* ── Labels ── */
.sec-label { font-size: 11.5px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); margin: 4px 0 11px 0; }
.panel-label { font-size: 11.5px; font-weight: 600; letter-spacing: 0.9px; text-transform: uppercase; color: var(--muted); margin: 0 0 12px 2px; display: flex; align-items: center; gap: 8px; }
.panel-label .live { width: 6px; height: 6px; border-radius: 50%; background: #a855f7; box-shadow: 0 0 7px rgba(168,85,247,0.7); }
.note { display: inline-block; font-size: 12.5px; color: var(--cream-dim); background: var(--panel-2); border: 1px solid var(--border); border-radius: 20px; padding: 5px 13px; margin: 0 auto 14px auto; }

/* ── Answer cards (Compare) ── */
[class*="st-key-card_"] { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
.card-head { display: flex; justify-content: space-between; gap: 8px; font-size: 10.5px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; color: var(--muted); }
.card-head .pass { color: #6ee7a8; } .card-head .fail { color: #f87171; }
.you-line { font-size: 14px; color: var(--cream); margin: 16px 0 8px 2px; }
.you-line b { font-size: 10.5px; letter-spacing: 0.9px; text-transform: uppercase; color: var(--muted); margin-right: 8px; }

/* ── Expanders ── */
[data-testid="stExpander"] { background: var(--panel-2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { font-size: 13px !important; font-weight: 500 !important; color: var(--cream-dim) !important; }
[data-testid="stExpander"] summary:hover { color: var(--cream) !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] { background: var(--panel) !important; border: 1px solid var(--border-soft) !important; border-radius: 13px !important; padding: 13px 17px !important; margin-bottom: 9px !important; }
[data-testid="stChatMessage"] p { font-size: 14.5px !important; line-height: 1.65 !important; }

/* ── Chat input ── */
/* One rounded field with the send button inside it (Streamlit nests a bordered textarea
   inside a bordered box by default; flatten the inner one). Structural selectors only, so
   this survives Streamlit's generated class names changing. */
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--bg) !important; }
[data-testid="stBottomBlockContainer"] { padding-top: 10px !important; padding-bottom: 22px !important; }
[data-testid="stChatInput"] > div {
    background: var(--panel-2) !important; border: 1px solid var(--border) !important;
    border-radius: 16px !important; padding: 7px 7px 7px 18px !important;
    box-shadow: 0 8px 28px rgba(0,0,0,0.38) !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(242,232,213,0.32) !important;
    box-shadow: 0 0 0 3px rgba(242,232,213,0.06), 0 8px 28px rgba(0,0,0,0.38) !important;
}
[data-testid="stChatInput"] > div > div { align-items: center !important; gap: 10px !important; }
[data-testid="stChatInputTextArea"] {
    background: transparent !important; border: none !important; border-radius: 0 !important;
    box-shadow: none !important; outline: none !important; padding: 8px 0 !important;
    min-height: 38px !important; color: var(--text) !important;
    font-size: 14.5px !important; line-height: 1.5 !important; caret-color: var(--cream) !important;
}
[data-testid="stChatInputTextArea"]::placeholder { color: var(--faint) !important; }
[data-testid="stChatInputSubmitButton"] {
    width: 38px !important; height: 38px !important; min-width: 38px !important;
    border-radius: 11px !important; background: var(--cream) !important; color: #161616 !important;
    transition: background .15s ease, transform .1s ease !important;
}
[data-testid="stChatInputSubmitButton"] svg { color: #161616 !important; }
[data-testid="stChatInputSubmitButton"]:not(:disabled):hover { background: #fff4e2 !important; }
[data-testid="stChatInputSubmitButton"]:not(:disabled):active { transform: scale(0.94) !important; }
[data-testid="stChatInputSubmitButton"]:disabled { background: rgba(242,232,213,0.08) !important; }
[data-testid="stChatInputSubmitButton"]:disabled svg { color: var(--faint) !important; }

/* ── Iframes (the graph) ── */
iframe { border-radius: 13px !important; border: 1px solid var(--border) !important; background: #141414 !important; }

/* ── Empty state ── */
.empty-wrap { text-align: center; padding: 40px 20px 22px 20px; }
.empty-wrap .glyph { font-size: 28px; color: var(--faint); margin-bottom: 13px; }
.empty-wrap h3 { font-family: 'Playfair Display', Georgia, serif !important; font-size: 24px !important; font-weight: 600 !important; color: var(--cream) !important; margin-bottom: 8px !important; }
.empty-wrap p { font-size: 13.5px !important; color: var(--muted) !important; max-width: 380px; margin: 0 auto; line-height: 1.55; }

/* ── Graph placeholder ── */
.graph-empty { border: 1px solid var(--border); border-radius: 13px; background: #141414; height: 500px; display: flex; align-items: center; justify-content: center; }

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


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=30, show_spinner=False)
def list_models():
    """Chat models installed in Ollama (embedding models left out); [] if Ollama is down."""
    return _list_models()


def render_graph(store, pos_key, changes=None, height=500, show_retired=True, until=None):
    """Draw the memory graph; node positions persist in session_state[pos_key]."""
    positions = stable_layout(store.graph, st.session_state.get(pos_key))
    st.session_state[pos_key] = positions
    page = build_graph_html(store.graph, positions, height, changes, show_retired, until)
    st.iframe(page, alt="Memory graph")  # height follows the content (legend + canvas)


def render_details(res, original=None):
    """Pipeline internals for one DMG answer: rewrite, actions, facts, timings, tokens."""
    rewritten = res.get("rewritten_query")
    if rewritten and rewritten != original:
        st.markdown(f"**Query rewrite:** {rewritten}")
    if res.get("actions"):
        st.markdown("**Memory actions:**\n" + "\n".join(
            f"- **{a.get('op')}** — {a.get('subject')} {a.get('predicate')} {a.get('object')}"
            for a in res["actions"]))
    if res.get("retrieved_facts"):
        st.markdown("**Facts used for the answer:**\n" + "\n".join(
            f"- {fact.get('text')}" for fact in res["retrieved_facts"]))
    timings = {k: v for k, v in (res.get("timings") or {}).items() if isinstance(v, (int, float))}
    if timings:
        steps = " · ".join(f"{k} {v:.2f}s" for k, v in timings.items() if k != "total")
        total = f" — total {timings['total']:.1f}s" if "total" in timings else ""
        st.caption(f"Timings: {steps}{total}")
    usage = res.get("usage") or {}
    if usage:
        st.caption(f"Answer tokens: {usage.get('prompt_tokens', '?')} prompt · "
                   f"{usage.get('completion_tokens', '?')} completion")


def queue_prompt(state_key, text):
    """Button callback: send an example prompt on this rerun."""
    st.session_state[state_key] = text


def example_buttons(prefix, examples, state_key):
    """Clickable example prompts in a two-column grid."""
    cols = st.columns(2)
    for i, text in enumerate(examples):
        cols[i % 2].button(text, key=f"ex_{prefix}_{i}", width="stretch", wrap=True,
                           on_click=queue_prompt, args=(state_key, text))


# ─── Per-session chat pipeline (loads the persisted long-term memory) ───
state = st.session_state
if "pipeline" not in state:
    state.pipeline, _ = make_pipeline(GRAPH_PATH)
    state.loaded_facts = memory_counts(state.pipeline.store)[1]
    state.messages = []
    state.chat_changes = None
pipeline = state.pipeline


def reset_chat_memory():
    """Sidebar button: wipe the long-term memory graph and the conversation."""
    state.pipeline.reset()
    state.messages, state.chat_changes, state.loaded_facts = [], None, 0
    state.pop("chat_pos", None)


# ─── Header bar ───
ollama_up = bool(list_models())
st.markdown(f"""
<div class="app-header">
    <span class="brand">Dynamic Memory Graph</span>
    <span class="divider"></span>
    <span class="subtitle">Conversational memory for small language models</span>
    <span class="status"><span class="dot{'' if ollama_up else ' off'}"></span>
    {'Ollama · ' + html.escape(getattr(pipeline, 'model', MODEL_NAME)) if ollama_up else 'Ollama offline'}</span>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ───
with st.sidebar:
    # Compare mode has its own DMG memory (shown on that page); make clear this is the Chat tab's.
    memory_label = "Chat tab memory" if state.get("mode") == "Compare" else "Long-term memory"
    st.markdown(f'<div class="sec-label">{memory_label}</div>', unsafe_allow_html=True)
    entities, active, retired = memory_counts(pipeline.store)
    st.markdown(f"""<div class="stat-row">
        <div><b>{entities}</b><span>Entities</span></div>
        <div><b>{active}</b><span>Active facts</span></div>
        <div><b>{retired}</b><span>Retired</span></div></div>""", unsafe_allow_html=True)

    with st.expander("View stored facts"):
        st.markdown(facts_markdown(pipeline.store))

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
    st.button("Reset memory", key="sidebar_reset", width="stretch", on_click=reset_chat_memory,
              help="Deletes everything in data/graph.json and clears this chat.")


# ─── Mode selector ───
mode = st.radio(
    "mode_selector",
    ["Chat", "Compare"],
    horizontal=True,
    label_visibility="collapsed",
    key="mode",
)


# ═══════════════════════════════════════════════════════════
# MODE: CHAT — split screen (conversation + live graph)
# ═══════════════════════════════════════════════════════════
if mode == "Chat":
    prompt = st.chat_input("Message...", key="chat_input") or state.pop("chat_queued", None)
    col_chat, col_graph = st.columns([5, 4], gap="large")

    # ── Left: conversation ──
    with col_chat:
        st.markdown('<div class="panel-label">Conversation</div>', unsafe_allow_html=True)

        if not state.messages and not prompt:
            st.markdown("""
            <div class="empty-wrap">
                <div class="glyph">&#9671;</div>
                <h3>Start a conversation</h3>
                <p>Tell me something about yourself, then watch the graph on the
                right grow as the memory captures each new fact.</p>
            </div>
            """, unsafe_allow_html=True)
            if state.loaded_facts:
                st.markdown(f'<div style="text-align:center"><span class="note">Loaded {state.loaded_facts} '
                            f'fact{"s" if state.loaded_facts != 1 else ""} from previous sessions</span></div>',
                            unsafe_allow_html=True)
            example_buttons("chat", EXAMPLES_CHAT, "chat_queued")

        for msg in state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("details"):
                    with st.expander("Pipeline details"):
                        render_details(msg["details"], msg["details"].get("user_message"))

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"), st.spinner("Thinking..."):
                before = graph_snapshot(pipeline.store.graph)
                result = dmg_respond(pipeline, prompt)
            state.chat_changes = diff_changes(before, pipeline.store.graph, result)
            state.messages += [{"role": "user", "content": prompt},
                               {"role": "assistant", "content": result["answer"], "details": result}]
            st.rerun()

    # ── Right: live graph ──
    with col_graph:
        graph = pipeline.store.graph
        st.markdown('<div class="panel-label"><span class="live"></span> Memory graph · live</div>',
                    unsafe_allow_html=True)

        if graph.number_of_edges() == 0:
            st.markdown("""
            <div class="graph-empty">
                <div style="text-align:center; color:#605c54;">
                    <div style="font-size:26px; margin-bottom:10px;">&#9676;</div>
                    <div style="font-size:13px;">The graph is empty.<br>Send a message to populate it.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            steps = timeline_steps(graph)
            c_toggle, c_slider = st.columns([2, 3], vertical_alignment="bottom")
            show_retired = c_toggle.toggle("Retired facts", value=True, key="show_retired",
                                           help="Show facts that were updated or deleted as dashed lines")
            step = len(steps)
            if len(steps) > 1:  # no key: the slider jumps back to "now" when memory grows
                step = c_slider.slider("Replay memory", 1, len(steps), len(steps), format="step %d",
                                       help="Drag back to replay how memory grew, one change at a time")
            if step < len(steps):
                until = steps[step - 1]
                changes = changes_at(graph, until)
                caption = f"Replaying step {step} of {len(steps)} · {until.replace('T', ' ')}"
            else:
                until, changes = None, state.chat_changes
                entities, active, retired = memory_counts(pipeline.store)
                caption = (change_caption(changes) if changes
                           else f"{entities} entities · {active} active facts · {retired} retired")
            render_graph(pipeline.store, "chat_pos", changes, 500, show_retired, until)
            st.caption(f"{caption}  ·  hover for details, scroll to zoom")


# ═══════════════════════════════════════════════════════════
# MODE: COMPARE — two baselines vs DMG on the same messages
# ═══════════════════════════════════════════════════════════
def compare_pipeline(model):
    """The Compare tab's own DMG pipeline (temporary graph per session), rebuilt on model change."""
    if "cmp_graph_path" not in state:
        state.cmp_graph_path = os.path.join(tempfile.gettempdir(), f"dmg_compare_{uuid.uuid4().hex[:8]}.json")
    if state.get("cmp_model_wanted") != model:
        # The graph file survives the swap, so memory is kept when the model changes.
        state.cmp_pipeline, state.cmp_model_used = make_pipeline(state.cmp_graph_path, model)
        state.cmp_model_wanted = model
    return state.cmp_pipeline


def reset_compare():
    """Clear the comparison: transcript, baseline histories, DMG memory, demo results."""
    state.cmp_rows, state.cmp_hist = [], {"recent": [], "full": []}
    state.cmp_graph_checks, state.demo_running = None, False
    state.pop("cmp_pos", None)
    if "cmp_pipeline" in state:
        state.cmp_pipeline.reset()


def start_demo():
    """Button callback: fresh comparison, then the script plays on this rerun."""
    reset_compare()
    state.demo_running = True


def stop_demo():
    """Button callback: stop after the current turn (results so far are kept)."""
    state.demo_running = False


def run_compare_turn(message, models, parallel=True, turn=None):
    """Send one message to all three columns (LLM work in threads) and store the row."""
    row = compare_turn(state.cmp_pipeline, state.cmp_hist, message, models, parallel, turn)
    state.cmp_rows.append(row)
    return row


def render_compare_row(index, row):
    """One user message and the three answers as neutral cards."""
    st.markdown(f'<div class="you-line"><b>You</b>{html.escape(row["user"])}</div>', unsafe_allow_html=True)
    for col, (cid, title) in zip(st.columns(3), COLUMNS.items()):
        res = row["cols"][cid]
        score = (row.get("scores") or {}).get(cid)
        badge = "" if score is None else ('<span class="pass">✅ pass</span>' if score[0]
                                          else '<span class="fail">❌ fail</span>')
        with col, st.container(border=True, key=f"card_{index}_{cid}"):
            st.markdown(f'<div class="card-head"><span>{title}</span>{badge}</div>', unsafe_allow_html=True)
            st.markdown(res.get("answer", ""))
            if score and not score[0]:
                st.caption("; ".join(str(r) for r in score[1]))
            st.caption(evidence_line(res))
            if cid == "dmg" and (res.get("retrieved_facts") or res.get("actions")):
                with st.expander(f"{len(res.get('retrieved_facts') or [])} facts used · "
                                 f"{len(res.get('actions') or [])} memory actions"):
                    render_details(res, row["user"])


if mode == "Compare":
    for key, default in [("cmp_rows", []), ("cmp_hist", {"recent": [], "full": []}),
                         ("cmp_graph_checks", None), ("demo_running", False)]:
        state.setdefault(key, default)
    prompt = st.chat_input("Type a message to compare...", key="compare_input") or state.pop("cmp_queued", None)

    # ── Controls ──
    options = list_models() or [MODEL_NAME]
    c_model, c_per, c_par, c_graph = st.columns([3, 2, 2, 2], vertical_alignment="bottom")
    base_model = c_model.selectbox("Model", options, key="cmp_model",
                                   index=options.index(MODEL_NAME) if MODEL_NAME in options else 0)
    per_column = c_per.toggle("Model per column", key="cmp_per_col")
    parallel = c_par.toggle("Run in parallel", value=True, key="cmp_parallel",
                            help="Columns run at the same time. If Ollama serves one request at a time, "
                                 "latency then includes waiting for the other columns.")
    show_graph = c_graph.toggle("DMG graph", value=True, key="cmp_show_graph")

    c_run, c_reset, _ = st.columns([2, 2, 5])
    # One button (same key) that flips between run and stop, so no stale copy lingers mid-run.
    c_run.button("Stop demo" if state.demo_running else "Run demo script", key="demo_run", type="primary",
                 width="stretch", on_click=stop_demo if state.demo_running else start_demo,
                 help="Plays evaluation/demo_script.json (15 turns) through all three columns "
                      "and scores each answer. Takes a few minutes.")
    c_reset.button("Reset comparison", key="reset_compare", width="stretch", on_click=reset_compare)
    progress_slot = st.empty()

    # ── Column headers: title, model, running score ──
    scores = column_scores(state.cmp_rows)
    models = {}
    for col, (cid, title) in zip(st.columns(3), COLUMNS.items()):
        with col:
            st.markdown(f"**{title}**")
            if per_column:
                models[cid] = st.selectbox(f"Model for {title}", options, key=f"cmp_model_{cid}",
                                           index=options.index(base_model), label_visibility="collapsed")
            else:
                models[cid] = base_model
            passed, scored = scores[cid]
            if scored:
                st.metric("Demo score", f"{passed}/{scored}")
    cmp_pipeline = compare_pipeline(models["dmg"])
    for cid, col in zip(COLUMNS, st.columns(3)):
        used = state.cmp_model_used if cid == "dmg" else models[cid]
        warn = " (pipeline ignores the model choice)" if cid == "dmg" and used != models["dmg"] else ""
        col.caption(f"{used}{warn}")

    if show_graph and cmp_pipeline.store.graph.number_of_edges():
        last = state.cmp_rows[-1]["cols"]["dmg"].get("changes") if state.cmp_rows else None
        g_col, facts_col = st.columns([3, 2], gap="medium")
        with g_col:
            render_graph(cmp_pipeline.store, "cmp_pos", last, height=400)
        with facts_col:
            entities, active, retired = memory_counts(cmp_pipeline.store)
            st.markdown('<div class="panel-label"><span class="live"></span> What DMG remembers</div>',
                        unsafe_allow_html=True)
            st.caption(f"{entities} entities · {active} active facts · {retired} retired"
                       + (f"  ·  last turn: {change_caption(last)}" if last else ""))
            with st.container(height=360, border=False):
                st.markdown(facts_markdown(cmp_pipeline.store))

    st.divider()

    # ── Transcript ──
    for i, row in enumerate(state.cmp_rows):
        render_compare_row(i, row)
    live_area = st.container()

    if prompt:  # a typed message stops a running demo
        state.demo_running = False
        live_area.markdown(f'<div class="you-line"><b>You</b>{html.escape(prompt)}</div>', unsafe_allow_html=True)
        with live_area, st.spinner("Running all three columns..."):
            run_compare_turn(prompt, models, parallel)
        st.rerun()

    if state.demo_running:
        script = load_script()
        turns = script["turns"]
        for i in range(len(state.cmp_rows), len(turns)):
            running = column_scores(state.cmp_rows)
            score_text = " · ".join(f"{SHORT_NAMES[c]} {p}/{n}" for c, (p, n) in running.items() if n)
            progress_slot.progress(i / len(turns), text=f"Demo turn {i + 1}/{len(turns)}: “{turns[i]['message']}”"
                                   + (f"  —  score so far: {score_text}" if score_text else ""))
            row = run_compare_turn(turns[i]["message"], models, parallel, turn=turns[i])
            with live_area:
                render_compare_row(i, row)
        state.cmp_graph_checks = check_graph(state.cmp_pipeline.store, script)
        state.demo_running = False
        st.rerun()

    if state.cmp_graph_checks:
        checks = state.cmp_graph_checks
        passed = sum(1 for _, ok in checks if ok)
        with st.expander(f"DMG graph checks after the demo · {passed}/{len(checks)} passed", expanded=True):
            st.markdown("  \n".join(f"{'✅' if ok else '❌'} `{check}`" for check, ok in checks))

    if not state.cmp_rows and not state.demo_running:
        st.markdown("""
        <div class="empty-wrap">
            <div class="glyph">&#9672;</div>
            <h3>Start a comparison</h3>
            <p>Each message goes to two memory-less baselines and to DMG.
            Try these in order, or press <b>Run demo script</b> for the scored 15-turn script.</p>
        </div>
        """, unsafe_allow_html=True)
        example_buttons("cmp", EXAMPLES_COMPARE, "cmp_queued")
