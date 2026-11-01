"""
Memory manager (owner: Aditya).

Decides what happens to each extracted triple before it goes into the graph.
Supported operations:
    ADD    - a new fact is stored as an edge with created_at = timestamp
    SKIP   - the exact same fact is already stored, so nothing is added
    UPDATE - an existing fact about the same (subject, predicate) is superseded:
             the old edge gets ended_at = timestamp, and the new one is added
    DELETE - the user explicitly negates something (dislikes, stopped, no longer);
             the matching old edge gets ended_at = timestamp, nothing new is added

Entity alias resolution maps common nicknames and short forms to a canonical
entity name so "Bangalore" and "Bengaluru" are treated as the same node.

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

# Predicates that signal the user is negating / removing a fact.
NEGATION_PREDICATES = {
    "does_not_have", "doesnt_have", "dont_have", "do_not_have",
    "stopped", "stopped_taking", "quit", "no_longer_has",
    "no_longer_takes", "no_longer_lives_in", "left", "moved_from",
    "dislikes", "hates", "allergic_to_not", "not",
    "does_not_like", "doesnt_like", "dont_like", "do_not_like",
    "is_not", "isnt", "stopped_using", "gave_up",
}

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


def normalize_entity(name):
    """Clean an entity name: trim, resolve aliases, map self-references to USER_NODE."""
    cleaned = " ".join(str(name).split()).strip(" .,!?;:'\"")
    if cleaned.lower() in SELF_WORDS:
        return USER_NODE
    # Check the alias dictionary for a canonical form.
    canonical = ENTITY_ALIASES.get(cleaned.lower())
    if canonical:
        return canonical
    return cleaned


def normalize_predicate(predicate):
    """Clean a predicate into lowercase_with_underscores."""
    cleaned = str(predicate).strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", cleaned).strip("_")


def is_duplicate(graph_store, subject, predicate, obj):
    """Return True if this exact fact is already active in the graph."""
    return graph_store.has_edge(subject, predicate, obj, active_only=True)


def _find_existing(graph_store, subject, predicate):
    """Find active edges with the same subject and predicate (to detect updates)."""
    return graph_store.find_edges(subject=subject, predicate=predicate, active_only=True)


def _end_edge(graph_store, edge_fact, timestamp):
    """Set ended_at on an existing edge to retire it."""
    source = edge_fact["source"]
    target = edge_fact["target"]
    key = edge_fact["key"]
    graph_store.graph[source][target][key]["ended_at"] = timestamp


def _is_negation(predicate):
    """Return True if the predicate signals removal of a fact."""
    return predicate in NEGATION_PREDICATES


def _opposite_predicate(predicate):
    """Map a negation predicate back to the positive form for lookup."""
    mapping = {
        "does_not_have": "has", "doesnt_have": "has",
        "dont_have": "has", "do_not_have": "has",
        "stopped_taking": "takes", "no_longer_takes": "takes",
        "no_longer_has": "has",
        "no_longer_lives_in": "lives_in", "moved_from": "lives_in", "left": "lives_in",
        "dislikes": "likes", "hates": "likes",
        "does_not_like": "likes", "doesnt_like": "likes",
        "dont_like": "likes", "do_not_like": "likes",
        "is_not": "is", "isnt": "is",
        "stopped": "takes", "quit": "takes",
        "stopped_using": "uses", "gave_up": "has",
    }
    return mapping.get(predicate)


def update_graph(graph_store, triples, timestamp):
    """Apply ADD / SKIP / UPDATE / DELETE for each triple and return the actions taken."""
    actions = []
    for triple in triples or []:
        subject = normalize_entity(triple.get("subject", ""))
        predicate = normalize_predicate(triple.get("predicate", ""))
        obj = normalize_entity(triple.get("object", ""))
        object_type = triple.get("object_type")

        if not subject or not predicate or not obj:
            continue

        # --- DELETE: the predicate itself is a negation ---
        if _is_negation(predicate):
            positive = _opposite_predicate(predicate)
            if positive:
                existing = _find_existing(graph_store, subject, positive)
                ended_count = 0
                for edge_fact in existing:
                    if edge_fact["object"].lower() == obj.lower():
                        _end_edge(graph_store, edge_fact, timestamp)
                        ended_count += 1
                if ended_count:
                    actions.append({"op": "DELETE", "subject": subject,
                                    "predicate": positive, "object": obj})
                    continue
            # Nothing to delete - just skip
            actions.append({"op": "SKIP", "subject": subject,
                            "predicate": predicate, "object": obj})
            continue

        # --- SKIP: exact duplicate already active ---
        if is_duplicate(graph_store, subject, predicate, obj):
            actions.append({"op": "SKIP", "subject": subject,
                            "predicate": predicate, "object": obj})
            continue

        # --- UPDATE: same (subject, predicate) but different object ---
        existing = _find_existing(graph_store, subject, predicate)
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

    # First round: basic adds
    round1 = [
        {"subject": "I", "predicate": "has", "object": "Diabetes", "object_type": "CONDITION"},
        {"subject": "User", "predicate": "takes", "object": "Metformin", "object_type": "DRUG"},
        {"subject": "User", "predicate": "lives_in", "object": "Bengaluru", "object_type": "PLACE"},
    ]
    print("--- Round 1 (ADD) ---")
    for action in update_graph(store, round1, "2026-10-29T10:00:00"):
        print(action)

    # Second round: duplicate, update, delete
    round2 = [
        {"subject": "User", "predicate": "has", "object": "Diabetes", "object_type": "CONDITION"},
        {"subject": "User", "predicate": "lives_in", "object": "Mumbai", "object_type": "PLACE"},
        {"subject": "User", "predicate": "stopped_taking", "object": "Metformin", "object_type": "DRUG"},
    ]
    print("\n--- Round 2 (SKIP / UPDATE / DELETE) ---")
    for action in update_graph(store, round2, "2026-10-29T11:00:00"):
        print(action)

    print("\nStats:", store.stats())
    print("Active facts:")
    for fact in store.get_all_facts(active_only=True):
        print("   ", fact["text"])
