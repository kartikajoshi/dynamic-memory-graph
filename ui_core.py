"""
UI helpers shared by the Streamlit app (app.py) and the API server (server.py).

Nothing in here imports Streamlit. These functions were moved out of app.py unchanged;
app.py imports them back, so the Streamlit app behaves exactly as before. New code for
the API (compare_turn, stale_mentions, edge ids) lives at the bottom.

Owner of the move: UI redesign (Nov 19). Pipeline modules are not touched.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import ollama

from config import ANSWER_TEMPERATURE, CONTEXT_WINDOW, HISTORY_TURNS, MODEL_NAME
from main import DMGPipeline

try:  # shared checks used by evaluation/run_demo.py
    from evaluation.run_demo import check_graph, check_turn, load_script
except Exception:  # run_demo.py missing or mid-edit: minimal equivalents
    def load_script(path=None):
        """Load evaluation/demo_script.json (or `path`)."""
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation", "demo_script.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def check_turn(answer, turn):
        """All `expect` keywords present and no `forbid` phrase present -> (ok, reasons)."""
        text = (answer or "").lower()
        reasons = [f"missing '{w}'" for w in turn.get("expect", []) if w.lower() not in text]
        reasons += [f"contains '{p}'" for p in turn.get("forbid", []) if p.lower() in text]
        return not reasons, reasons

    def check_graph(store, script):
        """Final graph checks (see the script's description) -> [(check, ok), ...]."""
        facts = store.get_all_facts(active_only=False)
        active = [f["text"].lower() for f in facts if not f.get("ended_at")]
        retired = [f["text"].lower() for f in facts if f.get("ended_at")]
        has = lambda texts, words: any(all(w.lower() in t for w in words) for t in texts)  # noqa: E731
        return ([(f"active: {' + '.join(w)}", has(active, w)) for w in script.get("final_active", [])]
                + [(f"retired: {' + '.join(w)}", has(retired, w) and not has(active, w))
                   for w in script.get("final_retired", [])]
                + [(f"absent: {' + '.join(w)}", not has(active, w)) for w in script.get("final_absent", [])])


EXAMPLES_CHAT = [
    "I'm diabetic and I take Metformin every morning.",
    "My sister Priya lives in Pune.",
    "I live in Bengaluru and work at Infosys.",
    "What medicine do I take?",
]
EXAMPLES_COMPARE = EXAMPLES_CHAT[:3] + ["What medicine do I take?", "Where does my sister live?"]

COLUMNS = {
    "recent": f"Baseline · last {HISTORY_TURNS} turns",
    "full": "Baseline · full history",
    "dmg": "DMG · memory graph",
}
SHORT_NAMES = {"recent": "Last-4", "full": "Full", "dmg": "DMG"}
ANSWER_RESERVE = 512  # tokens left free for the baseline's answer when trimming history

BASELINE_SYSTEM_PROMPT = """You are a helpful assistant.
Use the conversation so far to answer questions about the user.
Answer general questions normally."""


def make_pipeline(graph_path, model=None):
    """Create a DMGPipeline and return (pipeline, model actually used)."""
    if model and model != MODEL_NAME:
        try:
            return DMGPipeline(graph_path=graph_path, model=model), model
        except TypeError:  # older pipeline without the `model` argument
            pass
    return DMGPipeline(graph_path=graph_path), MODEL_NAME


def list_models():
    """Chat models installed in Ollama (embedding models left out); [] if Ollama is down."""
    try:
        names = [m.get("model") or m.get("name") for m in ollama.list()["models"]]
    except Exception:
        return []
    return sorted(name for name in names if name and "embed" not in name)


def memory_counts(store):
    """(entities, active facts, retired facts) for a GraphStore."""
    facts = store.get_all_facts(active_only=False)
    retired = sum(1 for fact in facts if fact.get("ended_at"))
    return store.graph.number_of_nodes(), len(facts) - retired, retired


def facts_markdown(store):
    """Bullet list of every stored fact; retired ones are struck through."""
    facts = store.get_all_facts(active_only=False)
    return "\n".join(f"- ~~{f['text']}~~" if f.get("ended_at") else f"- {f['text']}"
                     for f in facts) or "_No facts stored yet._"


def graph_snapshot(graph):
    """Node ids plus active / retired edge keys, used to diff a turn."""
    active, retired = set(), set()
    for u, v, k, data in graph.edges(keys=True, data=True):
        (retired if data.get("ended_at") else active).add((u, v, k))
    return set(graph.nodes), active, retired


def diff_changes(before, graph, result):
    """What one turn changed in the graph, plus which facts the answer used."""
    nodes0, active0, retired0 = before
    nodes1, active1, retired1 = graph_snapshot(graph)
    new_edges = (active1 | retired1) - active0 - retired0
    norm = lambda name: " ".join(str(name).lower().split())  # noqa: E731  same ids as the graph store
    update_pairs = {(norm(a.get("subject")), norm(a.get("object")))
                    for a in result.get("actions") or [] if a.get("op") == "UPDATE"}
    updated = {e for e in new_edges if (e[0], e[1]) in update_pairs}
    return {
        "new_nodes": nodes1 - nodes0,
        "new_edges": new_edges - updated,
        "updated_edges": updated,
        "retired_edges": retired1 & active0,
        "used_edges": {(f.get("source"), f.get("target"), f.get("key"))
                       for f in result.get("retrieved_facts") or []},
    }


def change_caption(changes):
    """'+2 facts · 1 updated · 3 used for this answer' style summary."""
    parts = []
    for key, text in [("new_edges", "+{} fact{}"), ("updated_edges", "{} updated"),
                      ("retired_edges", "{} retired"), ("used_edges", "{} used for this answer")]:
        n = len(changes.get(key, ()))
        if n:
            parts.append(text.format(n, "s" if n > 1 else ""))
    return " · ".join(parts) or "No memory changes this turn"


def dmg_respond(pipeline, message):
    """One timed DMG turn (safe to run in a worker thread: no Streamlit calls)."""
    start = time.perf_counter()
    try:
        result = pipeline.chat(message)
    except Exception as error:
        return {"answer": f"[Error: {error}]", "latency": time.perf_counter() - start}
    usage = result.get("usage") or {}
    return {**result, "latency": time.perf_counter() - start,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens")}


def baseline_respond(message, history, model):
    """One timed baseline turn: plain chat with the given history, no memory graph."""
    messages = [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}, *history,
                {"role": "user", "content": message}]
    start = time.perf_counter()
    try:
        response = ollama.chat(model=model, messages=messages, think=False,
                               options={"temperature": ANSWER_TEMPERATURE, "num_ctx": CONTEXT_WINDOW})
        answer = response["message"]["content"].strip()
        tokens = (response.get("prompt_eval_count"), response.get("eval_count"))
    except Exception as error:
        answer, tokens = f"[Error: {error}]", (None, None)
    return {"answer": answer, "latency": time.perf_counter() - start,
            "prompt_tokens": tokens[0], "completion_tokens": tokens[1]}


def fit_history(history, message):
    """Drop the oldest turns until the prompt fits the context window (~4 chars per token).

    Returns (kept_history, dropped_turns).
    """
    budget = CONTEXT_WINDOW - ANSWER_RESERVE - (len(BASELINE_SYSTEM_PROMPT) + len(message)) // 4
    kept = list(history)
    while kept and sum(len(m["content"]) // 4 + 4 for m in kept) > budget:
        kept = kept[2:]  # one user + assistant pair
    return kept, (len(history) - len(kept)) // 2


def evidence_line(res):
    """'2.3 s · 412 prompt tok · 38 out' for an answer card."""
    bits = []
    if res.get("latency") is not None:
        bits.append(f"{res['latency']:.1f} s")
    if res.get("prompt_tokens") is not None:
        bits.append(f"{res['prompt_tokens']} prompt tok")
    if res.get("completion_tokens") is not None:
        bits.append(f"{res['completion_tokens']} out")
    if res.get("note"):
        bits.append(res["note"])
    return " · ".join(bits)


def column_scores(rows):
    """{column: (passed, scored)} over the scored (demo) rows."""
    totals = {cid: [0, 0] for cid in COLUMNS}
    for row in rows:
        for cid, (ok, _) in (row.get("scores") or {}).items():
            totals[cid][0] += bool(ok)
            totals[cid][1] += 1
    return totals


def compare_turn(cmp_pipeline, hist, message, models, parallel=True, turn=None, trace=False):
    """Send one message to all three columns (LLM work in threads); return the row.

    Moved out of app.run_compare_turn so the API can use it. `hist` is the baselines'
    {"recent": [...], "full": [...]} history and is updated in place. trace=True (API
    only) also records the retriever's ranked output in row["retrieved_raw"].
    """
    full_history, dropped = fit_history(hist["full"], message)
    before = graph_snapshot(cmp_pipeline.store.graph)
    with ThreadPoolExecutor(max_workers=3 if parallel else 1) as pool:
        jobs = {
            "recent": pool.submit(baseline_respond, message, hist["recent"][-HISTORY_TURNS * 2:], models["recent"]),
            "full": pool.submit(baseline_respond, message, full_history, models["full"]),
            "dmg": (pool.submit(traced, dmg_respond, _ignore, cmp_pipeline, message) if trace
                    else pool.submit(dmg_respond, cmp_pipeline, message)),
        }
        results = {cid: job.result() for cid, job in jobs.items()}
    raw = None
    if trace:
        results["dmg"], raw = results["dmg"]
    if dropped:
        results["full"]["note"] = f"oldest {dropped} turn{'s' if dropped > 1 else ''} trimmed to fit context"
        results["full"]["dropped_turns"] = dropped
    for cid in ("recent", "full"):
        hist[cid] += [{"role": "user", "content": message},
                      {"role": "assistant", "content": results[cid]["answer"]}]
    results["dmg"]["changes"] = diff_changes(before, cmp_pipeline.store.graph, results["dmg"])
    row = {"user": message, "cols": results}
    if trace:
        row["retrieved_raw"] = raw
    if turn and (turn.get("expect") or turn.get("forbid")):
        row["scores"] = {cid: check_turn(res["answer"], turn) for cid, res in results.items()}
    return row


# ───────── API helpers (new) ─────────

def edge_id(edge):
    """(u, v, k) -> 'u|v|k', the edge id used by the web UI."""
    u, v, k = edge
    return f"{u}|{v}|{k}"


def changes_json(changes):
    """diff_changes() output with sets of tuples turned into JSON lists of edge ids."""
    if not changes:
        return None
    out = {}
    for key, value in changes.items():
        if key == "new_nodes":
            out[key] = sorted(value)
        else:
            out[key] = sorted(edge_id(e) for e in value)
    return out


PAST_CUES = ("stopped", "no longer", "not taking", "don't take", "do not take", "quit", "used to",
             "previously", "formerly", "former ", "moved from", "moved out", "moved away",
             "discontinued", "anymore", "any more", "switched from", "instead of", "were ", "was ")


def stale_mentions(answer, store):
    """Sentences of `answer` that state a retired fact as if it were still true.

    Evidence for the Compare scoreboard ("Full history: you live in Bengaluru, retired 3
    turns ago"); scores never use it. A hit is a sentence that names the object of a
    retired fact (its label or a memory_manager.ENTITY_ALIASES alias, whole word,
    case-insensitive), where that object is not also the object of an active fact with the
    same subject and predicate, and the sentence has no past/negation cue ("stopped",
    "moved from", "no longer", ...). Heuristic: the UI quotes the sentence so a reader can
    judge it.
    """
    import re
    try:
        from modules.memory_manager import ENTITY_ALIASES
    except Exception:
        ENTITY_ALIASES = {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer or "") if s.strip()]
    if not sentences:
        return []
    facts = store.get_all_facts(active_only=False)
    active = {(f["source"], f["predicate"], f["target"]) for f in facts if not f.get("ended_at")}
    hits = []
    for fact in facts:
        if not fact.get("ended_at") or (fact["source"], fact["predicate"], fact["target"]) in active:
            continue
        obj = fact["object"].lower()
        names = {obj} | {alias for alias, canon in ENTITY_ALIASES.items() if str(canon).lower() == obj}
        patterns = [re.compile(rf"\b{re.escape(n)}\b", re.I) for n in names if n]
        for sentence in sentences:
            low = sentence.lower()
            if any(p.search(sentence) for p in patterns) and not any(c in low for c in PAST_CUES):
                hits.append({"fact": fact["text"], "ended_at": fact["ended_at"], "sentence": sentence,
                             "edge": edge_id((fact["source"], fact["target"], fact["key"]))})
                break
    return hits


# ───────── Live trace (API only; app.py never installs it) ─────────
# The web UI shows each pipeline stage as it happens. DMGPipeline.chat() only reports
# timings at the end, so install_trace() wraps the five stage functions it calls with
# pass-through wrappers: same arguments, same return value, no behaviour change. They do
# something only on a thread where a listener was set with traced(); everywhere else they
# just call the original. Disable with DMG_LIVE_TRACE=0.
import threading as _threading

_trace_local = _threading.local()
TRACE_STAGES = [("rewrite", "rewrite"), ("extract", "extract"),
                ("update", "update_graph"), ("retrieve", "retrieve")]


def _stage_summary(stage, value):
    """Small JSON-ready summary of what a stage returned (for the live trace)."""
    try:
        if stage == "rewrite":
            return {"rewritten": value}
        if stage == "extract":
            return {"triples": len(value or [])}
        if stage == "update":
            ops = {}
            for action in value or []:
                ops[action.get("op")] = ops.get(action.get("op"), 0) + 1
            return {"ops": ops}
        if stage == "retrieve":
            return {"facts": len(value or [])}
        if stage == "answer":
            return {"completion_tokens": (value[1] or {}).get("completion_tokens")}
    except Exception:
        pass
    return {}


def _wrap_stage(stage, fn):
    def wrapper(*args, **kwargs):
        listener = getattr(_trace_local, "listener", None)
        if listener is None:
            return fn(*args, **kwargs)
        start = time.perf_counter()
        listener({"type": "stage", "stage": stage, "status": "start"})
        value = fn(*args, **kwargs)
        if stage == "retrieve":
            _trace_local.retrieved_raw = list(value or [])
        listener({"type": "stage", "stage": stage, "status": "end",
                  "seconds": round(time.perf_counter() - start, 3), **_stage_summary(stage, value)})
        return value
    wrapper.__wrapped__ = fn
    wrapper.__name__ = getattr(fn, "__name__", stage)
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def install_trace():
    """Wrap main's stage functions once (idempotent). Returns True when the trace is on."""
    if os.environ.get("DMG_LIVE_TRACE", "1") == "0":
        return False
    import main
    if getattr(main, "_dmg_ui_trace", False):
        return True
    for stage, name in TRACE_STAGES:
        setattr(main, name, _wrap_stage(stage, getattr(main, name)))
    DMGPipeline.generate_answer = _wrap_stage("answer", DMGPipeline.generate_answer)
    main._dmg_ui_trace = True
    return True


def _ignore(event):
    """Listener that drops events (used to capture only the retriever output)."""


def traced(fn, listener, *args, **kwargs):
    """Run fn(*args) on this thread with `listener` receiving stage events.

    Returns (fn's return value, retriever output or None). The retriever output is the
    ranked list retrieve() returned, before add_link_facts() added linking facts.
    """
    _trace_local.listener, _trace_local.retrieved_raw = listener, None
    try:
        return fn(*args, **kwargs), _trace_local.retrieved_raw
    finally:
        _trace_local.listener, _trace_local.retrieved_raw = None, None


def retrieval_ranks(result, retrieved_raw=None):
    """Rank of each retrieved fact, in the order the answer prompt lists them.

    With the live trace, rank = position in the retriever's ranked output and facts that
    add_link_facts() added for context are marked linked (rank None). Without it, rank is
    the prompt order (basis "prompt order").
    """
    facts = result.get("retrieved_facts") or []
    ids = [(f.get("source"), f.get("target"), f.get("key")) for f in facts]
    if retrieved_raw is not None:
        raw = {(f.get("source"), f.get("target"), f.get("key")): i + 1 for i, f in enumerate(retrieved_raw)}
        rows = [{"id": edge_id(e), "text": f.get("text"), "rank": raw.get(e), "linked": e not in raw}
                for e, f in zip(ids, facts)]
        return rows, "retriever"
    return ([{"id": edge_id(e), "text": f.get("text"), "rank": i + 1, "linked": False}
             for i, (e, f) in enumerate(zip(ids, facts))], "prompt order")
