"""
Entity and relation extractor (owner: Arnav).

Turns one user message into knowledge-graph triples using Llama 3.2 3B
through Ollama. Each triple is a dict:

    {"subject": "User", "predicate": "takes",
     "object": "Metformin", "object_type": "DRUG"}

Usage (from the project root):
    from modules.extractor import extract
    extract("I'm diabetic and take Metformin")

Quick test:
    python -m modules.extractor
"""

import json
import re

import ollama

from config import CONTEXT_WINDOW, ENTITY_TYPES, LLM_TEMPERATURE, MODEL_NAME, USER_NODE

# Words that mean "the person chatting" -> always mapped to USER_NODE.
SELF_WORDS = {"i", "me", "my", "myself", "mine", "user", "the user", "speaker"}

SYSTEM_PROMPT = f"""You extract personal facts from a user's chat message as knowledge-graph triples.

Return ONLY a JSON object in this exact shape:
{{"triples": [{{"subject": "...", "predicate": "...", "object": "...", "object_type": "..."}}]}}

Rules:
- The person writing the message is always the subject "{USER_NODE}".
- predicate is a short lowercase verb phrase joined with underscores, e.g. has, takes, lives_in, likes, dislikes, allergic_to, has_sister.
- object is a short noun phrase in Title Case, e.g. "Diabetes", "Metformin", "Bangalore".
- object_type must be exactly one of: {", ".join(ENTITY_TYPES)}.
- Only extract facts the user states about themselves or people in their life.
- Ignore questions, greetings and general knowledge.
- If there are no facts, return {{"triples": []}}."""

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
    ("What are the side effects of ibuprofen?", []),
]


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
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Fall back to the first {...} block in case the model added extra text.
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        data = data.get("triples", [])
    return data if isinstance(data, list) else []


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
    if object_type not in ENTITY_TYPES:
        return None

    if subject.lower() in SELF_WORDS:
        subject = USER_NODE
    predicate = re.sub(r"[^a-z0-9_]", "", predicate.lower().replace(" ", "_")).strip("_")
    if not predicate:
        return None

    return {"subject": subject, "predicate": predicate, "object": obj, "object_type": object_type}


def extract(text):
    """Extract (subject, predicate, object, object_type) triples from a user message."""
    if not text or not text.strip():
        return []

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=_build_messages(text.strip()),
            format="json",
            options={"temperature": LLM_TEMPERATURE, "num_ctx": CONTEXT_WINDOW},
        )
        raw = response["message"]["content"]
    except Exception as error:  # Ollama not running, model missing, etc.
        print(f"[extractor] Ollama call failed: {error}")
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
    return triples


if __name__ == "__main__":
    samples = [
        "I'm diabetic and take Metformin.",
        "I moved to Bangalore last month and I hate spicy food.",
        "What is the capital of France?",
    ]
    for sample in samples:
        print(sample)
        for t in extract(sample):
            print("   ", t)
