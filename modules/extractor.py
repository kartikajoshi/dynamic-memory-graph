"""
Entity and relation extractor (owner: Arnav).

Turns one user message into knowledge-graph triples using Llama 3.2 3B
through Ollama. Each triple is a dict:

    {"subject": "User", "predicate": "takes",
     "object": "Metformin", "object_type": "DRUG"}

Oct 30 update: improved extraction prompt with clearer rules, added a
third-person example, better error handling with retries, and logging.

Nov 18 update: questions/requests are skipped (no memory writes), new entity
types (ORGANIZATION, FOOD, OTHER fallback), few-shots use entities that are
not in the demo, "my sister Priya" always yields User has_sister Priya, and
unresolved he/she/they are never mapped to User. extract() takes model=.

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
- The person writing the message (I, me, my) is the subject "{USER_NODE}".
- Named people in the user's life (e.g. a sister called Asha) can also be subjects.
  He, she, they, him, her are NEVER "{USER_NODE}".
- predicate is a short lowercase verb phrase joined with underscores, e.g. has, takes, lives_in,
  works_at, likes, dislikes, allergic_to, has_sister, stopped_taking.
- A medicine the user takes (even if a doctor prescribed it) is "{USER_NODE} takes <Medicine>".
- object is a short noun phrase in Title Case, e.g. "Asthma", "Salbutamol", "Jaipur".
- Never use a role without a name (doctor, friend, someone) as subject or object.
- object_type must be exactly one of: {", ".join(ENTITY_TYPES)}.
- Only extract facts the user states about themselves or people in their life.
- Do NOT extract facts from questions, greetings, or general knowledge statements.
- If there are no extractable facts, return {{"triples": []}}.
- Each fact should be a single, atomic statement (one predicate per triple)."""

# Few-shot examples shown to the model before the real message.
# Nov 18: example entities deliberately differ from the demo/eval data to avoid copy-paste leakage.
FEW_SHOT = [
    (
        "I'm asthmatic and use a Salbutamol inhaler daily.",
        [
            {"subject": "User", "predicate": "has", "object": "Asthma", "object_type": "CONDITION"},
            {"subject": "User", "predicate": "takes", "object": "Salbutamol", "object_type": "DRUG"},
        ],
    ),
    (
        "My brother Arjun lives in Jaipur and works at Wipro.",
        [
            {"subject": "User", "predicate": "has_brother", "object": "Arjun", "object_type": "PERSON"},
            {"subject": "Arjun", "predicate": "lives_in", "object": "Jaipur", "object_type": "PLACE"},
            {"subject": "Arjun", "predicate": "works_at", "object": "Wipro", "object_type": "ORGANIZATION"},
        ],
    ),
    (
        "I enjoy badminton but I can't stand loud music, and I'm allergic to shellfish.",
        [
            {"subject": "User", "predicate": "likes", "object": "Badminton", "object_type": "PREFERENCE"},
            {"subject": "User", "predicate": "dislikes", "object": "Loud Music", "object_type": "PREFERENCE"},
            {"subject": "User", "predicate": "allergic_to", "object": "Shellfish", "object_type": "FOOD"},
        ],
    ),
    (
        "My cardiologist started me on Atorvastatin.",
        [
            {"subject": "User", "predicate": "takes", "object": "Atorvastatin", "object_type": "DRUG"},
        ],
    ),
    (
        "I no longer take Ibuprofen.",
        [
            {"subject": "User", "predicate": "stopped_taking", "object": "Ibuprofen", "object_type": "DRUG"},
        ],
    ),
    # No question example: questions never reach the model (see statement_part), and a
    # trailing empty example made small models (llama3.2:1b) return [] for everything.
]

# A sentence starting with one of these words (or ending in "?") is a question or request.
QUESTION_STARTERS = {
    "what", "where", "who", "whom", "whose", "when", "why", "how", "which",
    "do", "does", "did", "can", "could", "should", "shall", "is", "are", "am",
    "was", "were", "will", "would", "have", "has", "may",
    "tell", "recommend", "give", "suggest", "explain", "show", "list", "help", "any",
}

# "my sister Priya" -> User has_sister Priya (added if the model missed it).
RELATIONS = (
    "sister", "brother", "mother", "father", "mom", "dad", "wife", "husband", "friend",
    "son", "daughter", "boss", "colleague", "cousin", "uncle", "aunt", "grandmother",
    "grandfather", "partner", "girlfriend", "boyfriend", "roommate", "manager", "doctor",
)
# An optional adjective is allowed: "my best friend Aditi", "my older brother Karan".
RELATION_PATTERN = re.compile(
    r"\b[Mm]y\s+(?:(?:best|close|old|older|elder|younger|little|big|baby|twin|childhood)\s+)?("
    + "|".join(RELATIONS) + r")\s+([A-Z][a-zA-Z]+)")
RELATION_ALIASES = {"mom": "mother", "dad": "father"}
NOT_NAMES = {"I", "Is", "Was", "And", "But", "Has", "Lives", "Works", "Also"}

# Third-person pronouns; if one is left unresolved, the fact is not about the user.
THIRD_PERSON = {"he", "she", "they", "him", "her", "them"}
FIRST_PERSON = {"i", "i'm", "im", "me", "my", "mine", "myself", "i've", "i'd", "i'll"}

MAX_RETRIES = 2


def is_question(sentence):
    """Return True if a sentence is a question or request (not a statement of fact)."""
    sentence = sentence.strip().lower()
    if sentence.endswith("?"):
        return True
    words = re.findall(r"[a-z']+", sentence)
    if words and words[0] == "please":
        words = words[1:]
    return bool(words) and words[0] in QUESTION_STARTERS


def statement_part(text):
    """Drop question/request sentences and return the rest ("" if nothing is left)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(s for s in sentences if s and not is_question(s))


def relation_triples(text):
    """Deterministic rule: "my sister Priya" -> User has_sister Priya (PERSON)."""
    triples = []
    for relation, name in RELATION_PATTERN.findall(text):
        if name in NOT_NAMES:
            continue
        relation = RELATION_ALIASES.get(relation, relation)
        triples.append({"subject": USER_NODE, "predicate": f"has_{relation}",
                        "object": name, "object_type": "PERSON"})
    return triples


def is_grounded(name, text):
    """True if `name` is User or one of its words appears in the message.

    Catches names copied from the few-shot examples (e.g. "Arjun" for "He works at ...").
    Matches on the first 4 letters, so "Diabetes" is grounded by "diabetic".
    ponytail: misses paraphrases like "sci-fi" -> "Science Fiction".
    """
    if name == USER_NODE:
        return True
    text = text.lower()
    return any(word[:4] in text for word in re.findall(r"[a-z0-9]{3,}", name.lower()))


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
            "ACTIVITY": "PREFERENCE",
            "ORGANISATION": "ORGANIZATION", "ORGANIZATIONS": "ORGANIZATION", "ORG": "ORGANIZATION",
            "COMPANY": "ORGANIZATION", "EMPLOYER": "ORGANIZATION", "SCHOOL": "ORGANIZATION",
            "COLLEGE": "ORGANIZATION", "UNIVERSITY": "ORGANIZATION", "HOSPITAL": "ORGANIZATION",
            "FOODS": "FOOD", "DISH": "FOOD", "CUISINE": "FOOD", "DRINK": "FOOD",
            "BEVERAGE": "FOOD", "INGREDIENT": "FOOD", "ALLERGEN": "FOOD",
        }
        object_type = type_fixes.get(object_type, object_type)
        if object_type not in ENTITY_TYPES:
            logger.debug("Unknown object_type '%s' -> OTHER for triple: %s", object_type, triple)
            object_type = "OTHER"

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


def extract(text, model=None):
    """Extract (subject, predicate, object, object_type) triples from a user message.

    Questions and requests are dropped first, so they never write to memory.
    """
    text = statement_part(text or "")
    if not text:
        return []

    raw = ""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=model or MODEL_NAME,
                messages=_build_messages(text),
                format="json",
                think=False,
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

    # An unresolved "she/he/they" sentence with no "I/my" is about someone else, never the user.
    words = set(re.findall(r"[a-z']+", text.lower()))
    about_someone_else = bool(words & THIRD_PERSON) and not (words & FIRST_PERSON)

    triples, seen = [], set()
    for item in _parse_response(raw) + relation_triples(text):
        cleaned = _clean_triple(item)
        if cleaned is None:
            continue
        if about_someone_else and cleaned["subject"] == USER_NODE:
            continue
        if not (is_grounded(cleaned["subject"], text) and is_grounded(cleaned["object"], text)):
            logger.info("Dropped ungrounded triple: %s", cleaned)
            continue
        key = (cleaned["subject"].lower(), cleaned["predicate"], cleaned["object"].lower())
        if key not in seen:
            seen.add(key)
            triples.append(cleaned)

    logger.info("Extracted %d triples from: %s", len(triples), text[:80])
    return triples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    # Deterministic checks (no LLM needed).
    assert is_question("What medicines do I take?")
    assert is_question("Recommend a history book in one line.")
    assert is_question("Can I take it at night?")
    assert not is_question("I can't stand spicy food.")
    assert not is_question("I stopped taking Metformin.")
    assert statement_part("I moved to Pune. Any good doctors there?") == "I moved to Pune."
    assert relation_triples("My sister Priya lives in Pune.")[0]["predicate"] == "has_sister"
    assert relation_triples("My doctor also put me on Insulin.") == []
    assert is_grounded("Diabetes", "I'm diabetic") and not is_grounded("Arjun", "He works at Apollo")
    print("deterministic checks passed\n")

    samples = [
        "I'm diabetic and take Metformin.",
        "I moved to Bangalore last month and I hate spicy food.",
        "My friend Raj is a doctor and he recommended I try yoga.",
        "I work at Infosys and I'm allergic to peanuts.",
        "She is allergic to peanuts.",
        "What is the capital of France?",
        "",
    ]
    for sample in samples:
        print(f"Input: {sample!r}")
        for t in extract(sample):
            print("   ", t)
        print()
