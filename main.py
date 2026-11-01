"""
Integration pipeline (owner: Kartika).

Wires every module into one chat turn:

    user message
      -> query_rewriter.rewrite()       resolve "it", "that", ...
      -> extractor.extract()            pull triples from the message
      -> memory_manager.update_graph()  ADD / SKIP into the graph
      -> graph_store.save()             persist to data/graph.json
      -> retriever.retrieve()           pick the relevant facts
      -> Llama 3.2 3B                   answer using only those facts

Run a terminal chat (from the project root):
    python main.py

Commands inside the chat: /facts, /reset, /quit
"""

from datetime import datetime

import ollama

from config import (
    ANSWER_TEMPERATURE,
    CONTEXT_WINDOW,
    GRAPH_PATH,
    HISTORY_TURNS,
    MODEL_NAME,
    TOP_K,
    ensure_dirs,
)
from modules.extractor import extract
from modules.graph_store import GraphStore
from modules.memory_manager import update_graph
from modules.query_rewriter import rewrite
from modules.retriever import retrieve

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant with long-term memory.
Below are facts you remember about the user from earlier conversations.
Use them when they are relevant to the question. Do not invent facts that are not listed.
If the facts do not help, answer normally.

Known facts:
{facts}"""


def format_facts(facts):
    """Turn a list of fact dicts into bullet lines for the prompt."""
    if not facts:
        return "(no relevant facts stored yet)"
    return "\n".join(f"- {fact['text']}" for fact in facts)


class DMGPipeline:
    """One chat session backed by the persistent memory graph."""

    def __init__(self, graph_path=GRAPH_PATH):
        """Load the graph from disk and start with an empty short-term history."""
        ensure_dirs()
        self.store = GraphStore(graph_path).load()
        self.history = []

    def build_messages(self, query, facts):
        """Build the LLM messages: system prompt with retrieved facts, then the question."""
        return [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT.format(facts=format_facts(facts))},
            {"role": "user", "content": query},
        ]

    def generate_answer(self, query, facts):
        """Ask Llama 3.2 3B to answer the query using the retrieved facts."""
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=self.build_messages(query, facts),
                options={"temperature": ANSWER_TEMPERATURE, "num_ctx": CONTEXT_WINDOW},
            )
            return response["message"]["content"].strip()
        except Exception as error:
            return f"[Could not reach Ollama: {error}. Is `ollama serve` running?]"

    def chat(self, user_message):
        """Run one full turn of the pipeline and return everything it produced."""
        timestamp = datetime.now().isoformat(timespec="seconds")

        rewritten = rewrite(user_message, self.history)
        triples = extract(user_message)
        actions = update_graph(self.store, triples, timestamp)
        if any(action["op"] != "SKIP" for action in actions):
            self.store.save()

        facts = retrieve(rewritten, self.store, top_k=TOP_K)
        answer = self.generate_answer(rewritten, facts)

        # Keep only a short window of raw history - long-term memory lives in the graph.
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-HISTORY_TURNS * 2:]

        return {
            "user_message": user_message,
            "rewritten_query": rewritten,
            "triples": triples,
            "actions": actions,
            "retrieved_facts": facts,
            "answer": answer,
        }

    def all_facts(self):
        """Return every active fact in memory."""
        return self.store.get_all_facts(active_only=True)

    def reset(self):
        """Wipe the memory graph and the short-term history."""
        self.store.clear()
        self.store.save()
        self.history = []


def run_cli():
    """Simple terminal chat loop for testing the pipeline."""
    pipeline = DMGPipeline()
    print(f"DMG chat ({MODEL_NAME}). Memory: {pipeline.store.stats()}")
    print("Commands: /facts  /reset  /quit\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/facts":
            print(format_facts(pipeline.all_facts()), "\n")
            continue
        if message == "/reset":
            pipeline.reset()
            print("Memory cleared.\n")
            continue

        result = pipeline.chat(message)
        if result["rewritten_query"] != message:
            print(f"  (rewritten: {result['rewritten_query']})")
        for action in result["actions"]:
            print(f"  [{action['op']}] {action['subject']} {action['predicate']} {action['object']}")
        if result["retrieved_facts"]:
            print("  facts used:", "; ".join(f["text"] for f in result["retrieved_facts"]))
        print(f"Bot: {result['answer']}\n")


if __name__ == "__main__":
    run_cli()
