"""
Deterministic tests for the memory manager (no LLM, no Ollama needed).

Run from the project root:
    python -m tests.test_memory
    python tests/test_memory.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.extractor import is_grounded, is_question, relation_triples, statement_part
from modules.graph_store import GraphStore
from modules.memory_manager import update_graph
from modules.query_rewriter import _clean_output, _resolve_he_she


def triple(subject, predicate, obj, object_type="OTHER"):
    return {"subject": subject, "predicate": predicate, "object": obj, "object_type": object_type}


def new_store():
    return GraphStore(os.path.join(tempfile.gettempdir(), "dmg_test_memory_unit.json"))


def active(store):
    return {fact["text"] for fact in store.get_all_facts(active_only=True)}


def retired(store):
    return {fact["text"] for fact in store.get_all_facts(active_only=False) if fact["ended_at"]}


def test_multi_valued_accumulates():
    store = new_store()
    update_graph(store, [triple("User", "takes", "Metformin")], "t1")
    update_graph(store, [triple("User", "takes", "Insulin")], "t2")
    update_graph(store, [triple("User", "likes", "Coffee")], "t3")
    update_graph(store, [triple("User", "loves", "Cricket")], "t4")
    assert {"User takes Metformin", "User takes Insulin",
            "User likes Coffee", "User likes Cricket"} <= active(store), active(store)


def test_single_valued_updates():
    store = new_store()
    update_graph(store, [triple("User", "lives_in", "Bengaluru")], "t1")
    actions = update_graph(store, [triple("User", "lives_in", "Mumbai")], "t2")
    assert actions[0]["op"] == "UPDATE"
    assert "User lives in Mumbai" in active(store)
    assert "User lives in Bangalore" in retired(store)  # alias Bengaluru -> Bangalore


def test_moved_to_is_lives_in():
    store = new_store()
    update_graph(store, [triple("User", "lives_in", "Bangalore")], "t1")
    update_graph(store, [triple("User", "lives_in", "Mumbai"),
                         triple("User", "moved_to", "Mumbai")], "t2")
    assert active(store) == {"User lives in Mumbai"}, active(store)


def test_negation_stems():
    for negation in ("stops_taking", "stopped_taking", "quit", "no_longer_takes", "stopped"):
        store = new_store()
        update_graph(store, [triple("User", "takes", "Metformin"),
                             triple("User", "takes", "Insulin")], "t1")
        actions = update_graph(store, [triple("User", negation, "Metformin")], "t2")
        assert actions[0]["op"] == "DELETE", (negation, actions)
        assert active(store) == {"User takes Insulin"}, (negation, active(store))
        assert "User takes Metformin" in retired(store)


def test_negation_without_match_is_not_stored():
    store = new_store()
    actions = update_graph(store, [triple("User", "stops_taking", "Aspirin")], "t1")
    assert actions[0]["op"] == "SKIP"
    assert active(store) == set()


def test_dislikes_stored():
    store = new_store()
    update_graph(store, [triple("User", "hates", "Spicy Food")], "t1")
    assert active(store) == {"User dislikes Spicy Food"}, active(store)


def test_like_dislike_flip():
    store = new_store()
    update_graph(store, [triple("User", "likes", "Tea")], "t1")
    update_graph(store, [triple("User", "dislikes", "Tea")], "t2")
    assert active(store) == {"User dislikes Tea"}, active(store)
    update_graph(store, [triple("User", "loves", "Tea")], "t3")
    assert active(store) == {"User likes Tea"}, active(store)


def test_no_junk_generic_roles():
    store = new_store()
    update_graph(store, [
        triple("User", "is_treated_by", "Doctor"),
        triple("Doctor", "prescribed", "Insulin"),
        triple("She", "allergic_to", "Peanuts"),
        triple("User", "has", "My Doctor"),
        triple("User", "was_put_on", "Insulin"),
        triple("Raj", "is", "Doctor"),
    ], "t1")
    assert active(store) == {"User takes Insulin", "Raj is Doctor"}, active(store)


def test_extractor_rules():
    assert is_question("What medicines do I take?")
    assert is_question("Give me a pasta tip in one line.")
    assert not is_question("I stopped taking Metformin.")
    assert not is_question("I can't stand spicy food.")
    assert statement_part("I moved to Pune. Any good doctors there?") == "I moved to Pune."
    assert relation_triples("My sister Priya lives in Pune.") == [
        triple("User", "has_sister", "Priya", "PERSON")]
    assert is_grounded("Diabetes", "I'm diabetic and take Metformin.")
    assert not is_grounded("Arjun", "He works at Apollo Hospital.")  # few-shot leak


def test_rewriter_guards():
    history = [{"role": "user", "content": "My sister Priya and my friend Raj are visiting."}]
    assert _resolve_he_she("He works at Apollo Hospital.", history) == "Raj works at Apollo Hospital."
    assert _resolve_he_she("She is allergic to peanuts.", history) == "Priya is allergic to peanuts."
    # Rewrites that drop the message or add names from outside the conversation are rejected.
    assert _clean_output("Apollo Hospital.", "He works at Apollo Hospital.") == "He works at Apollo Hospital."
    assert _clean_output("Can I take Metformin?", "Can I take it?", "User: I love tea.") == "Can I take it?"


def test_holdout_regressions():
    """Bugs found by evaluation/holdout_script.json (Nov 19)."""
    # "my best friend Aditi" is a relation, and "she" resolves to her without the LLM.
    assert relation_triples("My best friend Aditi lives in Chennai.") == [
        triple("User", "has_friend", "Aditi", "PERSON")]
    history = [{"role": "user", "content": "My best friend Aditi lives in Chennai and works at Zomato."}]
    assert _resolve_he_she("She is a huge fan of Taylor Swift.", history) == \
        "Aditi is a huge fan of Taylor Swift."
    # A pronoun replaced by a name already in the sentence is rejected.
    assert _clean_output("Taylor Swift is a huge fan of Taylor Swift.",
                         "She is a huge fan of Taylor Swift.") == "She is a huge fan of Taylor Swift."
    # "quit Coffee" retires "drinks Black Coffee" (partial object match).
    store = new_store()
    update_graph(store, [triple("User", "drinks", "Black Coffee")], "t1")
    update_graph(store, [triple("User", "stopped_drinking", "Coffee")], "t2")
    assert "User drinks Black Coffee" in retired(store) and not active(store)
    # is_student is studies_at.
    update_graph(store, [triple("User", "is_student", "BITS Pilani")], "t3")
    assert "User studies at BITS Pilani" in active(store)


def test_medical_advice_guard():
    from main import remove_medical_advice
    drugs = ["Thyroxine"]
    out = remove_medical_advice("You can take Thyroxine with your morning coffee.",
                                "Should I take Thyroxine with my morning coffee?", drugs)
    assert "you can take" not in out.lower() and "Thyroxine" in out and "doctor" in out
    assert remove_medical_advice("You can still take Thyroxine with your green tea.",
                                 "I quit coffee, I only drink green tea now.", drugs) == "Got it."
    safe = "You take Thyroxine. Please ask your doctor or pharmacist what time is best for you to take it."
    assert remove_medical_advice(safe, "Can I take it at night?", drugs) == safe
    recipe = "Boil the pasta al dente. You can have it with olive oil."
    assert remove_medical_advice(recipe, "Give me a pasta tip.", drugs) == recipe
    # Leaks seen in the live Compare run (Nov 19).
    out = remove_medical_advice(
        "You may want to consider taking it before your morning coffee, as it can help with "
        "absorption. You take Thyroxine.", "Should I take Thyroxine with my morning coffee?", drugs)
    assert "consider taking" not in out and "doctor" in out
    assert remove_medical_advice("There's no known interaction between the two.",
                                 "I quit coffee last week, I only drink green tea now.", drugs) == "Got it."
    out = remove_medical_advice(
        "It's generally recommended to take Levothyroxine with food, as it can help with absorption.",
        "Should I take Thyroxine with my morning coffee?", drugs)
    assert "recommended" not in out and "food" not in out and "Thyroxine" in out and "doctor" in out
    # Listing question: not advice, answer kept as is.
    listing = "You take Thyroxine."
    assert remove_medical_advice(listing, "What medicines do I take?", drugs) == listing
    # No medicine on record -> "interaction" in a normal answer is left alone.
    social = "Good interaction with colleagues helps at an internship."
    assert remove_medical_advice(social, "Any internship tips?", []) == social


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"\nall {len(tests)} tests passed")
