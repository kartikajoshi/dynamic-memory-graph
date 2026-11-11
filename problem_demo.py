"""
Problem demonstration (owner: Kartika).

Shows the core problem DMG solves: a small LLM with a short context window
forgets what the user said a few turns ago because it has no persistent memory.

This script runs TWO conversations with the same messages:
    1. Baseline  - plain Llama 3.2 3B with only a sliding window (forgets)
    2. DMG       - the full pipeline with the memory graph (remembers)

Run from the project root (Ollama must be running with llama3.2:3b pulled):
    python problem_demo.py
"""

from datetime import datetime

import ollama

from config import ANSWER_TEMPERATURE, CONTEXT_WINDOW, HISTORY_TURNS, MODEL_NAME
from main import DMGPipeline

DEMO_MESSAGES = [
    "Hi! I'm diabetic and I take Metformin every morning.",
    "I also live in Bangalore and love filter coffee.",
    "My sister Priya lives in Pune.",
    "What is the capital of France?",
    "Can you recommend a good restaurant?",
    "What's the weather like today?",
    "By the way, I'm allergic to penicillin.",
    "What medicine do I take?",
    "Where does my sister live?",
    "Do I have any medical conditions?",
]

SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based on what the user has told you.
If you don't remember or don't know, say so honestly."""


def run_baseline():
    """Run the demo messages through plain Llama 3.2 3B with a sliding history window."""
    print("=" * 70)
    print("BASELINE: Plain Llama 3.2 3B (sliding window, no persistent memory)")
    print("=" * 70)

    history = []
    for message in DEMO_MESSAGES:
        print(f"\nYou: {message}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-HISTORY_TURNS * 2:])
        messages.append({"role": "user", "content": message})

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                options={"temperature": ANSWER_TEMPERATURE, "num_ctx": CONTEXT_WINDOW},
            )
            answer = response["message"]["content"].strip()
        except Exception as error:
            answer = f"[Error: {error}]"

        print(f"Bot: {answer}")
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})


def run_dmg():
    """Run the demo messages through the full DMG pipeline."""
    print("\n" + "=" * 70)
    print("DMG: Dynamic Memory Graph (persistent graph-backed memory)")
    print("=" * 70)

    pipeline = DMGPipeline()
    pipeline.reset()

    for message in DEMO_MESSAGES:
        print(f"\nYou: {message}")
        result = pipeline.chat(message)

        for action in result["actions"]:
            print(f"  [{action['op']}] {action['subject']} {action['predicate']} {action['object']}")

        if result["retrieved_facts"]:
            print("  Facts used:", "; ".join(f["text"] for f in result["retrieved_facts"]))

        print(f"Bot: {result['answer']}")

    print(f"\n  Memory stats: {pipeline.store.stats()}")
    print("  All stored facts:")
    for fact in pipeline.all_facts():
        print(f"    - {fact['text']}")


if __name__ == "__main__":
    print("Dynamic Memory Graph - Problem Demonstration")
    print(f"Model: {MODEL_NAME}, Context window: {CONTEXT_WINDOW} tokens\n")
    print("This demo sends the same 10 messages to two systems:")
    print("  1. Baseline (plain LLM, sliding window)")
    print("  2. DMG (with persistent memory graph)")
    print("Watch how the baseline forgets early facts while DMG remembers.\n")

    run_baseline()
    run_dmg()

    print("\n" + "=" * 70)
    print("Notice: the baseline cannot answer recall questions about early turns,")
    print("while DMG retrieves the stored facts and answers correctly.")
    print("=" * 70)
