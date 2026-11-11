"""
Streamlit chat UI (owner: Kartika).

A web interface for the Dynamic Memory Graph pipeline. Shows the chat,
extracted triples, memory actions, and retrieved facts in real time.

Run from the project root (Ollama must be running):
    streamlit run app.py
"""

import streamlit as st

from config import MODEL_NAME
from main import DMGPipeline, format_facts

st.set_page_config(page_title="Dynamic Memory Graph", page_icon="🧠", layout="wide")

st.title("🧠 Dynamic Memory Graph")
st.caption(f"Persistent conversational memory for {MODEL_NAME}")


@st.cache_resource
def get_pipeline():
    """Create or reuse the pipeline across Streamlit reruns."""
    return DMGPipeline()


pipeline = get_pipeline()

# Sidebar: memory stats and controls.
with st.sidebar:
    st.header("Memory")
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

# Chat history in session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history.
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

# Chat input.
if user_input := st.chat_input("Say something..."):
    # Show user message.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Run the pipeline.
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
