"""
Entity and relation extractor (owner: Arnav).

Turns one user message into knowledge-graph triples using Llama 3.2 3B
through Ollama. Each triple is a dict:

    {"subject": "User", "predicate": "takes",
     "object": "Metformin", "object_type": "DRUG"}

Oct 30 update: improved extraction prompt with clearer rules, added a
third-person example, better error handling with retries, and logging.

Usage (from the project root):
    from modules.extractor import extract
    extract("I'm diabetic and take Metformin")

Quick test:
    python -m modules.extractor
"""

import json
import re
import logging

import ollama

from config import CONTEXT_WINDOW, ENTITY_TYPES, LLM_TEMPERATURE, MODEL_NAME, USER_NODE

logger = logging.getLogger(__name__)

# Words that mean "the person chatting" -> always mapped to USER_NODE.
SELF_WORDS = {"i", "me", "my", "myself", "mine", "user", "the user", "speaker"}

SYSTEM_PROMPT = f"""You extract personal facts from a user's chat message as knowledge-graph triples.

Return ONLY a JSON object in this exact shape:
{{"triples": [{{"subject": "...", "predicate": "...", "object": "...", "object_type": "..."}}]}}

Rules:
- The person writing the message is always the subject "{USER_NODE}".
- Third-person entities (sister, friend, doctor) can also be subjects.
- predicate is a short lowercase verb phrase joined with underscores, e.g. has, takes, lives_in, likes, dislikes, allergic_to, has_sister.
- object is a short noun phrase in Title Case, e.g. "Diabetes", "Metformin", "Bangalore".
- object_type must be exactly one of: {", ".join(ENTITY_TYPES)}.
- Only extract facts the user states about themselves or people in their life.
- Do NOT extract facts from questions, greetings, or general knowledge statements.
- Do NOT extract opinions about things that are not personal preferences.
- If there are no extractable facts, return {{"triples": []}}.
- Each fact should be a single, atomic statement (one predicate per triple)."""

# Few-shot examples shown to the model before the real message.
FEW_SHOT = [
    (
        "I'm diabetic and take Metformin every morning.",
        [
            {"subject": "User", "predicate": "has", "object": "Diabetes", "object_type": "CONDITION"},
            {"subject": "User", "predicate": "takes", "object": "Metformin", "object_type": "DRUG"},
        ],
    ),
    (
        "My sister Priya lives in Pune and I love South Indian food.",
        [
            {"subject": "User", "predicate": "has_sister", "object": "Priya", "object_type": "PERSON"},
            {"subject": "Priya", "predicate": "lives_in", "object": "Pune", "object_type": "PLACE"},
            {"subject": "User", "predicate": "likes", "object": "South Indian Food", "object_type": "PREFERENCE"},
        ],
    ),
    (
        "My friend Raj is a doctor and he recommended I try yoga.",
        [
            {"subject": "User", "predicate": "has_friend", "object": "Raj", "object_type": "PERSON"},
            {"subject": "Raj", "predicate": "is", "object": "Doctor", "object_type": "PERSON"},
            {"subject": "User", "predicate": "does", "object": "Yoga", "object_type": "PREFERENCE"},
        ],
    ),
    ("What are the side effects of ibuprofen?", []),
]

MAX_RETRIES = 2


def _build_messages(text):
    """Build the chat messages: system rules, few-shot examples, then the real message."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_text, example_triples in FEW_SHOT:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": json.dumps({"triples": example_triples})})
    messages.append({"role": "user", "content": text})
    return messages


def _parse_response(raw):
    """Parse the model's reply into a list of raw triple dicts; return [] if it is not valid JSON."""
    if not raw or not raw.strip():
        logger.warning("Empty response from LLM")
        return []

    # Try parsing the full response first.
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Fall back to the first {...} block in case the model added extra text.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("No JSON object found in LLM response: %s", raw[:200])
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Failed to parse extracted JSON: %s", match.group(0)[:200])
            return []

    if isinstance(data, dict):
        data = data.get("triples", [])
    if not isinstance(data, list):
        logger.warning("Parsed data is not a list: %s", type(data))
        return []
    return data


def _clean_triple(triple):
    """Validate and normalise one triple; return None if it is unusable."""
    if not isinstance(triple, dict):
        return None

    subject = str(triple.get("subject", "")).strip()
    predicate = str(triple.get("predicate", "")).strip()
    obj = str(triple.get("object", "")).strip()
    object_type = str(triple.get("object_type", "")).strip().upper()

    if not subject or not predicate or not obj:
        return None

    # Validate object_type - try to fix common variations before rejecting.
    if object_type not in ENTITY_TYPES:
        # Try singular form and common aliases.
        type_fixes = {
            "PERSONS": "PERSON", "PEOPLE": "PERSON", "NAME": "PERSON",
            "CONDITIONS": "CONDITION", "DISEASE": "CONDITION", "ILLNESS": "CONDITION",
            "DRUGS": "DRUG", "MEDICINE": "DRUG", "MEDICATION": "DRUG",
            "PLACES": "PLACE", "LOCATION": "PLACE", "CITY": "PLACE",
            "PREFERENCES": "PREFERENCE", "HOBBY": "PREFERENCE", "INTEREST": "PREFERENCE",
        }
        object_type = type_fixes.get(object_type, object_type)
        if object_type not in ENTITY_TYPES:
            logger.debug("Invalid object_type '%s' for triple: %s", object_type, triple)
            return None

    # Map self-references to USER_NODE.
    if subject.lower() in SELF_WORDS:
        subject = USER_NODE
    if obj.lower() in SELF_WORDS:
        obj = USER_NODE

    # Normalise predicate to snake_case.
    predicate = re.sub(r"[^a-z0-9_]", "", predicate.lower().replace(" ", "_")).strip("_")
    if not predicate:
        return None

    return {"subject": subject, "predicate": predicate, "object": obj, "object_type": object_type}


def extract(text):
    """Extract (subject, predicate, object, object_type) triples from a user message."""
    if not text or not text.strip():
        return []

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=_build_messages(text.strip()),
                format="json",
                options={"temperature": LLM_TEMPERATURE, "num_ctx": CONTEXT_WINDOW},
            )
            raw = response["message"]["content"]
            break
        except Exception as error:
            last_error = error
            logger.warning("[extractor] Attempt %d/%d failed: %s", attempt, MAX_RETRIES, error)
            if attempt == MAX_RETRIES:
                logger.error("[extractor] All %d attempts failed. Last error: %s",
                             MAX_RETRIES, last_error)
                return []

    triples, seen = [], set()
    for item in _parse_response(raw):
        cleaned = _clean_triple(item)
        if cleaned is None:
            continue
        key = (cleaned["subject"].lower(), cleaned["predicate"], cleaned["object"].lower())
        if key not in seen:
            seen.add(key)
            triples.append(cleaned)

    logger.info("Extracted %d triples from: %s", len(triples), text[:80])
    return triples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    samples = [
        "I'm diabetic and take Metformin.",
        "I moved to Bangalore last month and I hate spicy food.",
        "My friend Raj is a doctor and he recommended I try yoga.",
        "What is the capital of France?",
        "",
    ]
    for sample in samples:
        print(f"Input: {sample!r}")
        for t in extract(sample):
            print("   ", t)
        print()
