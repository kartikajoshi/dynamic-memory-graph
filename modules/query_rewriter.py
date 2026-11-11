"""
Query rewriter (owner: Abhishek).

Resolves pronouns and vague references in the user's latest message using
the last few chat turns, so retrieval gets a self-contained question.

    "What are its side effects?"  ->  "What are the side effects of Metformin?"

If the message has no vague references it is returned unchanged, without
calling the LLM.

Nov 2 update: handles queries without coreferences gracefully - improved
needs_rewrite() to avoid false positives on possessive names and common
phrases, added logging, better fallback behaviour.

Usage (from the project root):
    from modules.query_rewriter import rewrite
    rewrite("What are its side effects?", history)

`history` is a list of {"role": "user" | "assistant", "content": "..."} dicts.

Quick test:
    python -m modules.query_rewriter
"""

import re
import logging

import ollama

from config import CONTEXT_WINDOW, HISTORY_TURNS, LLM_TEMPERATURE, MODEL_NAME

logger = logging.getLogger(__name__)

# Words that usually point back to something said earlier.
REFERENCE_WORDS = {
    "it", "its", "it's", "itself", "that", "this", "those", "these",
    "they", "them", "their", "theirs", "there",
    "he", "him", "his", "she", "her", "hers", "same",
}

# Phrases where a reference word does NOT indicate coreference (false positives).
SAFE_PATTERNS = [
    r"\bthat's? (good|great|nice|fine|okay|ok|cool|awesome|right|correct|true|bad|wrong)\b",
    r"\bthis is\b",         # "this is interesting" - about the current thing
    r"\bthere (is|are|was|were)\b",  # existential "there is"
    r"\bhis (name|age)\b",  # asking about someone explicitly
    r"\bher (name|age)\b",
]

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
    """Return True if the query contains a pronoun or vague reference that needs resolving."""
    query_lower = query.lower()

    # Check safe patterns first - these are false positives.
    for pattern in SAFE_PATTERNS:
        if re.search(pattern, query_lower):
            # Remove the matched safe phrase and check if other references remain.
            query_lower = re.sub(pattern, "", query_lower)

    words = re.findall(r"[a-z']+", query_lower)
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
        logger.debug("Empty LLM response, returning original query")
        return original
    cleaned = re.sub(r"^(rewritten( question)?|question)\s*:\s*", "", lines[0], flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \"'")
    if not cleaned or len(cleaned.split()) > len(original.split()) + MAX_EXTRA_WORDS:
        logger.debug("LLM response too long or empty, returning original query")
        return original
    return cleaned


def rewrite(query, history):
    """Return a self-contained version of `query`, resolving references from `history`."""
    query = (query or "").strip()
    if not query:
        return query

    # No history -> nothing to resolve references against.
    if not history:
        return query

    # No references detected -> return as-is without calling the LLM.
    if not needs_rewrite(query):
        logger.debug("No coreferences detected, skipping rewrite for: %s", query[:80])
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
        rewritten = _clean_output(response["message"]["content"], query)
        if rewritten != query:
            logger.info("Rewrote: %r -> %r", query, rewritten)
        return rewritten
    except Exception as error:
        logger.warning("Ollama call failed: %s — returning original query", error)
        return query


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    history = [
        {"role": "user", "content": "I'm diabetic and I take Metformin."},
        {"role": "assistant", "content": "Thanks for telling me. Metformin is a common diabetes medicine."},
    ]
    test_queries = [
        "What are its side effects?",       # should rewrite
        "What is the weather today?",        # no references
        "Is it good for my condition?",      # should rewrite
        "That's great, thanks!",             # safe pattern - no rewrite
        "Tell me more about that drug",      # should rewrite
    ]
    for q in test_queries:
        result = rewrite(q, history)
        changed = " (rewritten)" if result != q else " (unchanged)"
        print(f"  {q!r}  ->  {result!r}{changed}")
