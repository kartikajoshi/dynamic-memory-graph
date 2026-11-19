"""
Memory manager (owner: Aditya).

Decides what happens to each extracted triple before it goes into the graph.
Supported operations:
    ADD    - a new fact is stored as an edge with created_at = timestamp
    SKIP   - the same fact is already stored (or the triple is junk), nothing is added
    UPDATE - a single-valued fact (lives_in, works_at, ...) gets a new object, or a
             like flips to a dislike (or back): the old edge gets ended_at = timestamp,
             and the new one is added
    DELETE - the user negates something (stopped taking, no longer, quit, left);
             the matching old edge gets ended_at = timestamp, nothing new is added

Entity alias resolution maps common nicknames and short forms to a canonical
entity name so "Bangalore" and "Bengaluru" are treated as the same node.

Nov 9 update: added edge deduplication - before adding a new edge, checks for
near-duplicate facts (same subject, similar predicate, similar object) and
merges them instead of creating parallel edges.

Nov 18 update: only single-valued predicates supersede (takes/has/likes now
accumulate); predicates are stored in canonical form (moved_to -> lives_in);
negations are matched by prefix/stem (stops_taking, no_longer_*, quit, ...)
and are never stored as facts; dislikes are real facts that flip likes;
pronoun and unnamed-role triples (she, doctor, someone) are skipped.

Usage (from the project root):
    from modules.memory_manager import update_graph
    actions = update_graph(store, triples, "2026-10-29T10:00:00")

Quick test:
    python -m modules.memory_manager
"""

import re

from config import USER_NODE

# Words that refer to the person chatting.
SELF_WORDS = {"i", "me", "my", "myself", "mine", "user", "the user"}

# A predicate starting with one of these negates the rest: "stops_taking" -> takes.
NEGATION_PREFIXES = (
    "no_longer_", "not_", "never_", "doesnt_", "dont_", "does_not_", "do_not_",
    "didnt_", "did_not_", "isnt_", "is_not_", "is_no_longer_",
    "stopped_", "stops_", "stop_", "quit_", "quits_", "quitted_",
    "gave_up_", "gives_up_", "give_up_", "discontinued_", "ceased_",
)

# Bare negations: they end whatever the subject had with that object ("quit Smoking").
NEGATION_WORDS = {
    "stopped", "stops", "stop", "quit", "quits", "gave_up", "gives_up", "give_up",
    "no_longer", "not", "is_not", "isnt", "discontinued", "ceased", "dropped",
    "left", "leaves", "moved_from", "moved_out_of", "moved_away_from", "relocated_from",
}

# Only these predicates hold one value at a time; a new object replaces the old one.
# Everything else (takes, has, likes, allergic_to, ...) accumulates.
SINGLE_VALUED = {"lives_in", "works_at", "studies_at", "married_to", "age"}

# Like and dislike of the same thing cannot both be true.
OPPOSITES = {"likes": "dislikes", "dislikes": "likes"}

# Pronouns are never stored as entities.
PRONOUNS = {
    "he", "she", "they", "him", "her", "them", "it", "his", "hers", "their",
    "someone", "somebody", "anyone", "something", "this", "that",
}

# Roles without a name ("my doctor", "a friend") are junk as objects; service roles also as subjects.
GENERIC_ROLES = {
    "doctor", "physician", "nurse", "dentist", "pharmacist", "therapist", "specialist",
    "cardiologist", "friend", "colleague", "boss", "people", "person", "family",
    "sister", "brother", "mother", "father", "wife", "husband", "son", "daughter",
    "parents", "partner", "cousin", "uncle", "aunt",
}
SERVICE_ROLES = {"doctor", "physician", "nurse", "dentist", "pharmacist", "therapist",
                 "specialist", "cardiologist", "people", "person"}

# Maps alternate names -> canonical name so the graph stays consistent.
ENTITY_ALIASES = {
    "bengaluru": "Bangalore",
    "bombay": "Mumbai",
    "madras": "Chennai",
    "calcutta": "Kolkata",
    "metformin hcl": "Metformin",
    "metformin hydrochloride": "Metformin",
    "mom": "Mother",
    "dad": "Father",
    "mum": "Mother",
    "papa": "Father",
    "sis": "Sister",
    "bro": "Brother",
}

# Predicate synonyms for deduplication.
PREDICATE_SYNONYMS = {
    "has": {"has", "have", "having", "suffering_from", "diagnosed_with", "is_diagnosed_with",
            "suffers_from"},
    "takes": {"takes", "take", "taking", "took", "uses", "using", "on", "is_on", "is_taking",
              "prescribed", "was_prescribed", "is_prescribed", "started_taking", "put_on",
              "is_put_on", "was_put_on", "started_on"},
    "lives_in": {"lives_in", "live_in", "living_in", "resides_in", "based_in", "located_in",
                 "moved_to", "relocated_to", "shifted_to", "moved_into", "lives_at"},
    "studies_at": {"studies_at", "study_at", "studying_at", "studies_in", "is_student",
                   "is_student_at", "is_a_student_at", "student_at", "enrolled_at", "attends"},
    "likes": {"likes", "like", "loves", "love", "enjoys", "enjoy", "prefers", "prefer",
              "adores", "is_fond_of", "fond_of", "is_fan_of", "is_a_fan_of", "fan_of",
              "is_a_huge_fan_of", "is_huge_fan_of", "is_big_fan_of", "is_a_big_fan_of"},
    "dislikes": {"dislikes", "dislike", "hates", "hate", "detests", "cant_stand",
                 "cannot_stand", "does_not_like", "doesnt_like", "dont_like", "do_not_like",
                 "not_like", "does_not_enjoy", "doesnt_enjoy", "dont_enjoy"},
    "works_at": {"works_at", "work_at", "working_at", "employed_at", "employed_by",
                 "works_for", "work_for", "works_in"},
    "allergic_to": {"allergic_to", "allergy_to", "is_allergic_to", "has_allergy_to",
                    "has_allergy"},
}

# Build reverse lookup: predicate -> canonical form.
_PREDICATE_CANONICAL = {}
for canonical, synonyms in PREDICATE_SYNONYMS.items():
    for syn in synonyms:
        _PREDICATE_CANONICAL[syn] = canonical


def normalize_entity(name):
    """Clean an entity name: trim, resolve aliases, map self-references to USER_NODE."""
    cleaned = " ".join(str(name).split()).strip(" .,!?;:'\"")
    if cleaned.lower() in SELF_WORDS:
        return USER_NODE
    canonical = ENTITY_ALIASES.get(cleaned.lower())
    if canonical:
        return canonical
    return cleaned


def normalize_predicate(predicate):
    """Clean a predicate into lowercase_with_underscores."""
    cleaned = str(predicate).strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", cleaned).strip("_")


def _canonical_predicate(predicate):
    """Return the canonical form of a predicate for dedup matching."""
    return _PREDICATE_CANONICAL.get(predicate, predicate)


def is_duplicate(graph_store, subject, predicate, obj):
    """Return True if this exact fact is already active in the graph."""
    return graph_store.has_edge(subject, predicate, obj, active_only=True)


def _is_near_duplicate(graph_store, subject, predicate, obj):
    """Check for near-duplicate: same subject, synonym predicate, same object."""
    canonical = _canonical_predicate(predicate)
    synonyms = PREDICATE_SYNONYMS.get(canonical, {predicate})

    for syn in synonyms:
        if graph_store.has_edge(subject, syn, obj, active_only=True):
            return True
    return False


def _find_existing_canonical(graph_store, subject, predicate):
    """Find active edges with the same subject and any synonym predicate."""
    canonical = _canonical_predicate(predicate)
    synonyms = PREDICATE_SYNONYMS.get(canonical, {predicate})

    results = []
    for syn in synonyms:
        results.extend(graph_store.find_edges(subject=subject, predicate=syn, active_only=True))
    return results


def _end_edge(graph_store, edge_fact, timestamp):
    """Set ended_at on an existing edge to retire it."""
    source = edge_fact["source"]
    target = edge_fact["target"]
    key = edge_fact["key"]
    graph_store.graph[source][target][key]["ended_at"] = timestamp


def negated_predicate(predicate):
    """If `predicate` negates a fact, return the positive predicate it removes, else None.

    "stops_taking" -> "takes", "no_longer_lives_in" -> "lives_in",
    bare words like "quit" / "left" -> "" (any predicate with that object).
    """
    if predicate in NEGATION_WORDS:
        return ""
    for prefix in NEGATION_PREFIXES:
        if predicate.startswith(prefix):
            return _canonical_predicate(predicate[len(prefix):])
    return None


def _verb_stem(predicate):
    """Rough stem of a predicate's first word: "drinking" / "drinks" -> "drink"."""
    word = predicate.split("_")[0]
    for suffix in ("ing", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _bare(name):
    """Lowercase a name and drop a leading my/the/a: "My Doctor" -> "doctor"."""
    return re.sub(r"^(my|the|a|an|your)\s+", "", name.lower())


def is_junk(subject, predicate, obj):
    """Return True for triples that are not real facts: pronouns, unnamed roles, self-loops."""
    subject_bare, obj_bare = _bare(subject), _bare(obj)
    if subject_bare in PRONOUNS or obj_bare in PRONOUNS:
        return True
    if subject_bare in SERVICE_ROLES:
        return True  # "Doctor prescribed Insulin" - the fact is "User takes Insulin"
    if obj_bare in GENERIC_ROLES and predicate not in ("is", "is_a", "works_as"):
        return True  # "User is_treated_by Doctor", "User has Sister" (no name)
    return subject_bare == obj_bare


def _retire_negated(graph_store, subject, predicate, obj, timestamp):
    """End the active facts that a negation like "stopped_taking Metformin" cancels.

    Returns the list of retired fact dicts (empty if nothing matched).
    """
    positive = negated_predicate(predicate)
    rest = predicate
    for prefix in NEGATION_PREFIXES:
        if predicate.startswith(prefix):
            rest = predicate[len(prefix):]
            break
    retired = []
    # "quit Coffee" also ends "drinks Black Coffee": the negated object's words must all
    # appear in the stored object (one direction only, so "quit Black Coffee" keeps "Coffee").
    negated_words = set(_bare(obj).split())
    for edge_fact in graph_store.find_edges(subject=subject, active_only=True):
        if not negated_words <= set(_bare(edge_fact["object"]).split()):
            continue
        edge_predicate = edge_fact["predicate"]
        if (positive == ""
                or _canonical_predicate(edge_predicate) == positive
                or _verb_stem(edge_predicate) == _verb_stem(rest)):
            _end_edge(graph_store, edge_fact, timestamp)
            retired.append(edge_fact)
    return retired


def update_graph(graph_store, triples, timestamp):
    """Apply ADD / SKIP / UPDATE / DELETE for each triple and return the actions taken."""
    actions = []
    for triple in triples or []:
        subject = normalize_entity(triple.get("subject", ""))
        predicate = _canonical_predicate(normalize_predicate(triple.get("predicate", "")))
        obj = normalize_entity(triple.get("object", ""))
        object_type = triple.get("object_type")

        if not subject or not predicate or not obj:
            continue

        # --- SKIP: pronouns / unnamed roles are not facts ---
        if is_junk(subject, predicate, obj):
            actions.append({"op": "SKIP", "subject": subject,
                            "predicate": predicate, "object": obj})
            continue

        # --- DELETE: the predicate is a negation; it is never stored itself ---
        if negated_predicate(predicate) is not None:
            retired = _retire_negated(graph_store, subject, predicate, obj, timestamp)
            for edge_fact in retired:
                actions.append({"op": "DELETE", "subject": subject,
                                "predicate": edge_fact["predicate"], "object": obj})
            if not retired:
                actions.append({"op": "SKIP", "subject": subject,
                                "predicate": predicate, "object": obj})
            continue

        # --- SKIP: exact duplicate already active ---
        if is_duplicate(graph_store, subject, predicate, obj):
            actions.append({"op": "SKIP", "subject": subject,
                            "predicate": predicate, "object": obj})
            continue

        # --- SKIP: near-duplicate (synonym predicate, same object) ---
        if _is_near_duplicate(graph_store, subject, predicate, obj):
            actions.append({"op": "SKIP", "subject": subject,
                            "predicate": predicate, "object": obj})
            continue

        # --- UPDATE: single-valued predicate with a new object, or like <-> dislike flip ---
        existing = []
        if predicate in SINGLE_VALUED:
            existing = _find_existing_canonical(graph_store, subject, predicate)
        if predicate in OPPOSITES:
            existing = [edge_fact for edge_fact in
                        _find_existing_canonical(graph_store, subject, OPPOSITES[predicate])
                        if edge_fact["object"].lower() == obj.lower()]
        if existing:
            for edge_fact in existing:
                _end_edge(graph_store, edge_fact, timestamp)
            graph_store.add_edge(subject, predicate, obj,
                                obj_type=object_type, created_at=timestamp)
            actions.append({"op": "UPDATE", "subject": subject,
                            "predicate": predicate, "object": obj})
            continue

        # --- ADD: brand-new fact ---
        graph_store.add_edge(subject, predicate, obj,
                             obj_type=object_type, created_at=timestamp)
        actions.append({"op": "ADD", "subject": subject,
                        "predicate": predicate, "object": obj})

    return actions


if __name__ == "__main__":
    import os
    import tempfile

    from modules.graph_store import GraphStore

    store = GraphStore(os.path.join(tempfile.gettempdir(), "dmg_test_memory.json"))

    # Round 1: basic adds
    round1 = [
        {"subject": "I", "predicate": "has", "object": "Diabetes", "object_type": "CONDITION"},
        {"subject": "User", "predicate": "takes", "object": "Metformin", "object_type": "DRUG"},
        {"subject": "User", "predicate": "lives_in", "object": "Bengaluru", "object_type": "PLACE"},
    ]
    print("--- Round 1 (ADD) ---")
    for action in update_graph(store, round1, "2026-10-29T10:00:00"):
        print(action)

    # Round 2: duplicate, near-duplicate, update, delete
    round2 = [
        {"subject": "User", "predicate": "has", "object": "Diabetes", "object_type": "CONDITION"},
        {"subject": "User", "predicate": "taking", "object": "Metformin", "object_type": "DRUG"},
        {"subject": "User", "predicate": "moved_to", "object": "Mumbai", "object_type": "PLACE"},
        {"subject": "User", "predicate": "stops_taking", "object": "Metformin", "object_type": "DRUG"},
    ]
    print("\n--- Round 2 (SKIP / SKIP-near-dup / UPDATE / DELETE) ---")
    for action in update_graph(store, round2, "2026-10-29T11:00:00"):
        print(action)

    print("\nStats:", store.stats())
    print("Active facts:")
    for fact in store.get_all_facts(active_only=True):
        print("   ", fact["text"])
