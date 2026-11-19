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

Nov 18 update: also rewrites statements ("She is allergic to peanuts." ->
"Priya is allergic to peanuts.") so the extractor never stores them under
User; rewrite() takes model=. Rewrites that drop the original words or add
names not in the conversation are rejected, and a leftover he/she falls back
to the last "my <relation> <Name>" the user mentioned.

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
from modules.extractor import relation_triples

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

SYSTEM_PROMPT = """You rewrite the user's latest message (a question or a statement) so it can be understood on its own.
Replace pronouns and vague references (it, its, that, there, he, she, they, ...) with the short
name of the thing or person they refer to in the conversation (e.g. "Metformin", not "Metformin every morning").
He, she, they, him, her always mean another person named in the conversation - never the user.
Keep "I", "me" and "my" exactly as they are.
Change nothing else: do not add details, do not answer the question.
If nothing needs replacing, repeat the message unchanged.
Reply with the rewritten message only - no explanation, no quotes.

Examples (conversation mentions Metformin, Bangalore and the user's brother Arjun):
Can I take it at night?  ->  Can I take Metformin at night?
Any good doctors there?  ->  Any good doctors in Bangalore?
He is allergic to dust.  ->  Arjun is allergic to dust."""

# A rewrite only swaps a pronoun for a short name; a much longer reply means the model added details.
MAX_EXTRA_WORDS = 5

# Small words a rewrite may add ("its side effects" -> "the side effects of Metformin").
GLUE_WORDS = {"the", "a", "an", "of", "in", "at", "to", "for", "with", "on", "and"}

# Relations that rule out "she" / "he" when falling back to a named person.
MALE_RELATIONS = {"brother", "father", "husband", "son", "boyfriend", "uncle", "grandfather"}
FEMALE_RELATIONS = {"sister", "mother", "wife", "daughter", "girlfriend", "aunt", "grandmother"}


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


def _words(text):
    """Lowercase words of a text."""
    return re.findall(r"[a-z']+", text.lower())


def _clean_output(text, original, history_text=None):
    """Take the first line of the model's reply and fall back to the original if it looks wrong."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        logger.debug("Empty LLM response, returning original query")
        return original
    cleaned = re.sub(r"^(rewritten( question| message)?|question|message)\s*:\s*", "", lines[0], flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \"'")
    if not cleaned or len(cleaned.split()) > len(original.split()) + MAX_EXTRA_WORDS:
        logger.debug("LLM response too long or empty, returning original query")
        return original
    # A rewrite must keep the original's other words (small models sometimes echo an example)...
    kept = [w for w in _words(original) if w not in REFERENCE_WORDS]
    new_words = set(_words(cleaned))
    if kept and sum(w in new_words for w in kept) < 0.75 * len(kept):
        logger.debug("LLM rewrite dropped original words, returning original query")
        return original
    # A pronoun must be replaced by a NEW name, not one already in the sentence
    # ("She is a fan of Taylor Swift" -> "Taylor Swift is a fan of Taylor Swift").
    removed_refs = [w for w in _words(original) if w in REFERENCE_WORDS and w not in new_words]
    if removed_refs and not (new_words - set(_words(original)) - GLUE_WORDS):
        logger.debug("LLM rewrite replaced a pronoun with a name already in the sentence")
        return original
    # ...and every name it adds must come from the conversation, not from the prompt's examples.
    if history_text is not None:
        added = new_words - set(_words(original)) - GLUE_WORDS
        if any(word not in history_text.lower() for word in added):
            logger.debug("LLM rewrite added words not in the conversation, returning original query")
            return original
    return cleaned


def _resolve_he_she(text, history):
    """Fallback: replace a leftover he/she with the last matching "my <relation> <Name>"."""
    match = re.search(r"\b(she|he)\b", text, flags=re.IGNORECASE)
    if not match:
        return text
    she = match.group(1).lower() == "she"
    right, wrong = (FEMALE_RELATIONS, MALE_RELATIONS) if she else (MALE_RELATIONS, FEMALE_RELATIONS)
    # Newest first; skip relations of the other gender ("friend" fits both).
    candidates = [(triple["predicate"][len("has_"):], triple["object"])
                  for message in reversed(history or []) if message.get("role") == "user"
                  for triple in reversed(relation_triples(message.get("content", "")))]
    candidates = [(relation, name) for relation, name in candidates if relation not in wrong]
    if not candidates:
        return text
    name = next((name for relation, name in candidates if relation in right), candidates[0][1])
    return re.sub(rf"\b{match.group(1)}\b", name, text, flags=re.IGNORECASE)


def rewrite(query, history, model=None):
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

    # Deterministic first: he/she -> the last "my <relation> <Name>" of that gender.
    # If nothing else needs resolving, the LLM is not needed at all.
    resolved = _resolve_he_she(query, history)
    if resolved != query and not needs_rewrite(resolved):
        logger.info("Rewrote (rule): %r -> %r", query, resolved)
        return resolved
    query_for_llm = resolved

    history_text = _format_history(history)
    prompt = f"Conversation:\n{history_text}\n\nLatest message: {query_for_llm}\n\nRewritten message:"
    rewritten = query_for_llm
    try:
        response = ollama.chat(
            model=model or MODEL_NAME,
            think=False,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": LLM_TEMPERATURE, "num_ctx": CONTEXT_WINDOW},
        )
        rewritten = _clean_output(response["message"]["content"], query_for_llm, history_text)
    except Exception as error:
        logger.warning("Ollama call failed: %s — returning original query", error)
    rewritten = _resolve_he_she(rewritten, history)
    if rewritten != query:
        logger.info("Rewrote: %r -> %r", query, rewritten)
    return rewritten


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
    assert _clean_output("Arjun is allergic to dust.", "Can I take it at night?") == "Can I take it at night?"
    assert _clean_output("Priya is allergic to peanuts.", "She is allergic to peanuts.") == "Priya is allergic to peanuts."
    assert _clean_output("Apollo Hospital.", "He works at Apollo Hospital.") == "He works at Apollo Hospital."
    assert _clean_output("Can Raj help me with Metformin?", "Can he help me with it?",
                         "User: My friend Raj is a cardiologist.") == "Can he help me with it?"
    raj_history = [{"role": "user", "content": "My sister Priya and my friend Raj are visiting."}]
    assert _resolve_he_she("He works at Apollo Hospital.", raj_history) == "Raj works at Apollo Hospital."
    assert _resolve_he_she("She is allergic to peanuts.", raj_history) == "Priya is allergic to peanuts."
    history_sister = [
        {"role": "user", "content": "My sister Priya lives in Pune."},
        {"role": "assistant", "content": "Pune is a lovely city."},
    ]
    print(f"  'She is allergic to peanuts.'  ->  {rewrite('She is allergic to peanuts.', history_sister)!r}")
    for q in test_queries:
        result = rewrite(q, history)
        changed = " (rewritten)" if result != q else " (unchanged)"
        print(f"  {q!r}  ->  {result!r}{changed}")
