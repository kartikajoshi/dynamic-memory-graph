"""
Query rewriter (owner: Abhishek).

Resolves pronouns and vague references in the user's latest message using
the last few chat turns, so retrieval gets a self-contained question.

    "What are its side effects?"  ->  "What are the side effects of Metformin?"

If the message has no vague references it is returned unchanged, without
calling the LLM.

Usage (from the project root):
    from modules.query_rewriter import rewrite
    rewrite("What are its side effects?", history)

`history` is a list of {"role": "user" | "assistant", "content": "..."} dicts.

Quick test:
    python -m modules.query_rewriter
"""

import re

import ollama

from config import CONTEXT_WINDOW, HISTORY_TURNS, LLM_TEMPERATURE, MODEL_NAME

# Words that usually point back to something said earlier.
REFERENCE_WORDS = {
    "it", "its", "it's", "itself", "that", "this", "those", "these",
    "they", "them", "their", "theirs", "there",
    "he", "him", "his", "she", "her", "hers", "same",
}

SYSTEM_PROMPT = """You rewrite the user's latest question so it can be understood on its own.
Replace pronouns and vague references (it, its, that, there, he, she, they, ...) with the short
name of the thing they refer to in the conversation (e.g. "Metformin", not "Metformin every morning").
Change nothing else: do not add details, do not answer the question.
If nothing needs replacing, repeat the question unchanged.
Reply with the rewritten question only - no explanation, no quotes.

Examples (conversation mentions Metformin and Bangalore):
Can I take it at night?  ->  Can I take Metformin at night?
Any good doctors there?  ->  Any good doctors in Bangalore?"""

# A rewrite only swaps a pronoun for a short name; a much longer reply means the model added details.
MAX_EXTRA_WORDS = 5


def needs_rewrite(query):
    """Return True if the query contains a pronoun or vague reference."""
    words = re.findall(r"[a-z']+", query.lower())
    return any(word in REFERENCE_WORDS for word in words)


def _format_history(history):
    """Turn the last few turns into plain text for the prompt."""
    recent = (history or [])[-HISTORY_TURNS * 2:]
    lines = []
    for message in recent:
        speaker = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{speaker}: {message.get('content', '').strip()}")
    return "\n".join(lines)


def _clean_output(text, original):
    """Take the first line of the model's reply and fall back to the original if it looks wrong."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return original
    cleaned = re.sub(r"^(rewritten( question)?|question)\s*:\s*", "", lines[0], flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \"'")
    if not cleaned or len(cleaned.split()) > len(original.split()) + MAX_EXTRA_WORDS:
        return original
    return cleaned


def rewrite(query, history):
    """Return a self-contained version of `query`, resolving references from `history`."""
    query = (query or "").strip()
    if not query or not history or not needs_rewrite(query):
        return query

    prompt = f"Conversation:\n{_format_history(history)}\n\nLatest question: {query}\n\nRewritten question:"
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": LLM_TEMPERATURE, "num_ctx": CONTEXT_WINDOW},
        )
        return _clean_output(response["message"]["content"], query)
    except Exception as error:  # Ollama not running, model missing, etc.
        print(f"[query_rewriter] Ollama call failed: {error}")
        return query


if __name__ == "__main__":
    history = [
        {"role": "user", "content": "I'm diabetic and I take Metformin."},
        {"role": "assistant", "content": "Thanks for telling me. Metformin is a common diabetes medicine."},
    ]
    for q in ["What are its side effects?", "What is the weather today?"]:
        print(f"{q!r}  ->  {rewrite(q, history)!r}")
