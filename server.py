"""
HTTP API for the web UI (owner: UI redesign, Nov 19).

Run from the project root (Ollama must be running):
    uvicorn server:app --port 8000
then open http://localhost:8000 (it serves web/dist once `npm run build` has run in web/).

The pipeline is untouched: this file calls DMGPipeline.chat() and the helpers in
ui_core.py, the same ones app.py uses. One local user, so state lives in memory:
the Chat pipeline uses data/graph.json (override with DMG_GRAPH_PATH), Compare uses
its own temporary graph, exactly like the Streamlit app.

Endpoints (all JSON):
    GET  /api/status                       Ollama up?, models, counts, settings
    GET  /api/graph?scope=&until=&retired= graph for the 3D view (scope chat|compare)
    GET  /api/facts?scope=                 every stored fact, retired ones included
    GET  /api/chat/history                 this server session's chat turns
    POST /api/chat          {message}      one DMG turn
    POST /api/chat/stream   {message}      same turn as server-sent events (live trace)
    POST /api/chat/clear                   clear the conversation, keep the memory graph
    POST /api/reset                        wipe the Chat memory graph and history
    GET  /api/scripts                      demo scripts for Compare
    GET  /api/demo-script?name=demo|long   one script
    GET  /api/compare/state                rows, totals, graph checks
    POST /api/compare/turn  {message | script + turn_index, model, models, parallel}
    POST /api/compare/checks {script}      final graph checks (check_graph) for a script
    POST /api/compare/reset
"""

import json
import os
import queue
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import CONTEXT_WINDOW, EVAL_DIR, GRAPH_PATH, HISTORY_TURNS, MODEL_NAME, TOP_K
from modules.graph_viz import EDGE_COLORS, HUB_COLOR, OTHER_COLOR, TYPE_COLORS, changes_at, graph_payload, timeline_steps
from ui_core import (
    ANSWER_RESERVE,
    COLUMNS,
    EXAMPLES_CHAT,
    EXAMPLES_COMPARE,
    changes_json,
    check_graph,
    column_scores,
    compare_turn,
    diff_changes,
    dmg_respond,
    edge_id,
    graph_snapshot,
    install_trace,
    list_models,
    load_script,
    make_pipeline,
    memory_counts,
    retrieval_ranks,
    stale_mentions,
    traced,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, "web", "dist")
SCRIPTS = {"demo": os.path.join(EVAL_DIR, "demo_script.json"),
           "long": os.path.join(EVAL_DIR, "long_script.json")}
SCRIPT_LABELS = {"demo": "Demo", "long": "Long conversation"}

TRACE_ON = install_trace()
app = FastAPI(title="Dynamic Memory Graph", docs_url="/api/docs", openapi_url="/api/openapi.json")


# ───────── state ─────────

class ChatState:
    """The Chat tab: one pipeline on the persistent graph, plus this session's turns."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pipeline = None
        self.turns = []        # [{user, answer, result, changes, retrieval, stamp, at}]
        self.changes = None    # diff_changes() of the latest turn (sets of tuples)
        self.ranks = {}        # {(u, v, k): rank} for the latest answer
        self.loaded_facts = 0

    def get(self):
        if self.pipeline is None:
            path = os.environ.get("DMG_GRAPH_PATH") or GRAPH_PATH
            self.pipeline, _ = make_pipeline(path)
            self.loaded_facts = memory_counts(self.pipeline.store)[1]
        return self.pipeline


class CompareState:
    """The Compare tab: its own DMG pipeline on a temporary graph and two baseline histories."""

    def __init__(self):
        self.lock = threading.Lock()
        self.graph_path = os.path.join(tempfile.gettempdir(), f"dmg_compare_{uuid.uuid4().hex[:8]}.json")
        self.pipeline, self.model_used, self.model_wanted = None, None, None
        self.reset_rows()

    def reset_rows(self):
        self.hist = {"recent": [], "full": []}
        self.rows = []          # internal rows (scores as tuples, changes as sets)
        self.retired_at = {}    # edge id -> row index where DMG retired it
        self.checks = None
        self.script = None

    def get(self, model=None):
        model = model or self.model_wanted or MODEL_NAME
        if self.pipeline is None or self.model_wanted != model:
            # The graph file survives a model swap, so memory is kept (same as app.py).
            self.pipeline, self.model_used = make_pipeline(self.graph_path, model)
            self.model_wanted = model
        return self.pipeline


CHAT, CMP = ChatState(), CompareState()
_models_cache = {"at": 0.0, "models": []}


def cached_models():
    """list_models() cached for 30 s, like app.py's st.cache_data(ttl=30)."""
    if time.monotonic() - _models_cache["at"] > 30:
        _models_cache.update(at=time.monotonic(), models=list_models())
    return _models_cache["models"]


def counts_json(store):
    entities, active, retired = memory_counts(store)
    return {"entities": entities, "active": active, "retired": retired}


def _ranks(retrieval, result):
    """{(u, v, k): rank} for the facts the retriever ranked (linked context facts have none)."""
    return {(f.get("source"), f.get("target"), f.get("key")): row["rank"]
            for row, f in zip(retrieval, result.get("retrieved_facts") or []) if row["rank"] is not None}


def stamp_of(graph, changes):
    """Find the timestamp the pipeline used for this turn's writes, from the changed edges."""
    for key in ("new_edges", "updated_edges"):
        for u, v, k in changes.get(key, ()):
            if graph.has_edge(u, v, k):
                return graph.edges[u, v, k].get("created_at")
    for u, v, k in changes.get("retired_edges", ()):
        if graph.has_edge(u, v, k):
            return graph.edges[u, v, k].get("ended_at")
    return None


def graph_json(store, changes=None, ranks=None, until=None, show_retired=True, turns=None):
    """Graph payload + counts + timeline steps (with the chat turn that wrote each step)."""
    graph = store.graph
    steps = timeline_steps(graph)
    if until is not None:
        if until not in steps:
            raise HTTPException(400, f"unknown timeline step: {until}")
        changes, ranks = changes_at(graph, until), None
    payload = graph_payload(graph, changes, show_retired, until, ranks)
    by_stamp = {t["stamp"]: i for i, t in enumerate(turns or []) if t.get("stamp")}
    step_rows = [{"at": s, "turn": by_stamp.get(s),
                  "message": (turns or [])[by_stamp[s]]["user"] if s in by_stamp else None}
                 for s in steps]
    return {**payload, "counts": counts_json(store), "steps": step_rows, "until": until,
            "changes": changes_json(changes) if changes and until is None else None}


def result_json(result):
    """A pipeline result (or baseline answer) as JSON; sets and tuples become lists."""
    out = {k: v for k, v in result.items() if k != "changes"}
    if "changes" in result:
        out["changes"] = changes_json(result["changes"])
    return jsonable_encoder(out)


# ───────── status & graph ─────────

@app.get("/api/status")
def status():
    models = cached_models()
    with_store = CHAT.get().store
    return {
        "ollama": bool(models), "models": models or [MODEL_NAME], "default_model": MODEL_NAME,
        "chat_model": getattr(CHAT.pipeline, "model", MODEL_NAME),
        "counts": counts_json(with_store), "loaded_facts": CHAT.loaded_facts,
        "settings": {"context_window": CONTEXT_WINDOW, "history_turns": HISTORY_TURNS,
                     "top_k": TOP_K, "answer_reserve": ANSWER_RESERVE},
        "live_trace": TRACE_ON, "columns": COLUMNS,
        "examples": {"chat": EXAMPLES_CHAT, "compare": EXAMPLES_COMPARE},
        "palette": {"types": TYPE_COLORS, "other": OTHER_COLOR, "hub": HUB_COLOR, "edges": EDGE_COLORS},
    }


@app.get("/api/graph")
def graph(scope: str = "chat", until: str | None = None, retired: bool = True):
    if scope == "compare":
        store = CMP.get().store
        last = CMP.rows[-1]["cols"]["dmg"] if CMP.rows else None
        return graph_json(store, last.get("changes") if last else None,
                          CMP.rows[-1].get("_ranks") if CMP.rows else None, until, retired)
    store = CHAT.get().store
    return graph_json(store, CHAT.changes, CHAT.ranks, until, retired, CHAT.turns)


@app.get("/api/facts")
def facts(scope: str = "chat"):
    store = (CMP.get() if scope == "compare" else CHAT.get()).store
    rows = [{"id": edge_id((f["source"], f["target"], f["key"])), **f}
            for f in store.get_all_facts(active_only=False)]
    return jsonable_encoder({"facts": rows, "counts": counts_json(store)})


# ───────── chat ─────────

class ChatIn(BaseModel):
    message: str


def _chat_turn(message, listener=None):
    """One DMG turn on the Chat pipeline (caller holds CHAT.lock). Returns the JSON reply."""
    pipeline = CHAT.get()
    before = graph_snapshot(pipeline.store.graph)
    result, raw = traced(dmg_respond, listener or (lambda event: None), pipeline, message)
    changes = diff_changes(before, pipeline.store.graph, result)
    retrieval, basis = retrieval_ranks(result, raw)
    ranks = _ranks(retrieval, result)
    stamp = stamp_of(pipeline.store.graph, changes)
    CHAT.changes, CHAT.ranks = changes, ranks
    turn = {"user": message, "answer": result.get("answer"), "result": result_json(result),
            "changes": changes_json(changes), "retrieval": retrieval, "rank_basis": basis,
            "stamp": stamp, "at": time.time()}
    CHAT.turns.append(turn)
    return {**turn, "index": len(CHAT.turns) - 1,
            "graph": graph_json(pipeline.store, changes, ranks, turns=CHAT.turns)}


def _message(text):
    text = (text or "").strip()
    if not text:
        raise HTTPException(422, "message is empty")
    return text


@app.post("/api/chat")
def chat(body: ChatIn):
    message = _message(body.message)
    if not CHAT.lock.acquire(blocking=False):
        raise HTTPException(409, "a chat turn is already running")
    try:
        return jsonable_encoder(_chat_turn(message))
    finally:
        CHAT.lock.release()


@app.post("/api/chat/stream")
def chat_stream(body: ChatIn):
    """Server-sent events: `stage` events as each pipeline stage starts / ends, then `done`."""
    message = _message(body.message)
    if not CHAT.lock.acquire(blocking=False):
        raise HTTPException(409, "a chat turn is already running")
    events = queue.Queue()
    t0 = time.perf_counter()

    def listener(event):
        events.put({**event, "t": round(time.perf_counter() - t0, 3)})

    def work():
        try:
            events.put({"type": "done", "reply": jsonable_encoder(_chat_turn(message, listener))})
        except Exception as error:  # report, don't hang the stream
            events.put({"type": "error", "error": str(error)})
        finally:
            CHAT.lock.release()
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        yield f"data: {json.dumps({'type': 'start', 'live_trace': TRACE_ON})}\n\n"
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if event is None:
                return
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/chat/history")
def chat_history():
    CHAT.get()
    return jsonable_encoder({"turns": CHAT.turns, "loaded_facts": CHAT.loaded_facts})


@app.post("/api/chat/clear")
def chat_clear():
    """Clear this session's conversation. Memory (the graph) is kept."""
    with CHAT.lock:
        CHAT.get()
        CHAT.turns, CHAT.changes, CHAT.ranks = [], None, {}
    return {"ok": True}


@app.post("/api/reset")
def reset():
    with CHAT.lock:
        CHAT.get().reset()
        CHAT.turns, CHAT.changes, CHAT.ranks, CHAT.loaded_facts = [], None, {}, 0
    return {"ok": True, "counts": counts_json(CHAT.get().store)}


# ───────── compare ─────────

def _script(name):
    if name not in SCRIPTS:
        raise HTTPException(404, f"unknown script: {name}")
    if not os.path.exists(SCRIPTS[name]):
        raise HTTPException(404, f"missing file: {os.path.relpath(SCRIPTS[name], BASE_DIR)}")
    return load_script(SCRIPTS[name])


@app.get("/api/scripts")
def scripts():
    out = []
    for name, path in SCRIPTS.items():
        if os.path.exists(path):
            script = load_script(path)
            n = len(script["turns"])
            out.append({"name": name, "turns": n, "scored": sum(1 for t in script["turns"]
                                                                if t.get("expect") or t.get("forbid")),
                        "label": f"{SCRIPT_LABELS[name]} ({n} turns)" if name == "demo" else SCRIPT_LABELS[name]})
    return {"scripts": out}


@app.get("/api/demo-script")
def demo_script(name: str = "demo"):
    return {"name": name, **_script(name)}


class CompareIn(BaseModel):
    message: str | None = None
    model: str | None = None
    models: dict[str, str] | None = None   # per column: recent / full / dmg
    parallel: bool = True
    script: str | None = None               # with turn_index: play that scripted turn (scored)
    turn_index: int | None = None


def _row_json(index, row):
    cols = {cid: {**result_json(res), "model": row["models"][cid]} for cid, res in row["cols"].items()}
    scores = None
    if row.get("scores"):
        scores = {cid: {"ok": bool(ok), "reasons": list(reasons)} for cid, (ok, reasons) in row["scores"].items()}
    stale = {}
    for cid, hits in row.get("stale", {}).items():
        stale[cid] = [{**h, "turns_ago": (index - CMP.retired_at[h["edge"]]) if h["edge"] in CMP.retired_at else None}
                      for h in hits]
    return {"index": index, "user": row["user"], "cols": cols, "scores": scores, "stale": stale,
            "script": row.get("script"), "turn_index": row.get("turn_index"),
            "retrieval": row.get("retrieval"), "rank_basis": row.get("rank_basis")}


def _totals():
    return {cid: {"passed": p, "scored": n} for cid, (p, n) in column_scores(CMP.rows).items()}


def _compare_state():
    return {"rows": [_row_json(i, r) for i, r in enumerate(CMP.rows)], "totals": _totals(),
            "checks": CMP.checks, "script": CMP.script,
            "model": CMP.model_wanted or MODEL_NAME, "dmg_model_used": CMP.model_used or MODEL_NAME}


@app.get("/api/compare/state")
def compare_state():
    return jsonable_encoder(_compare_state())


@app.post("/api/compare/turn")
def compare_turn_api(body: CompareIn):
    turn = None
    if body.script is not None and body.turn_index is not None:
        turns = _script(body.script)["turns"]
        if not 0 <= body.turn_index < len(turns):
            raise HTTPException(400, f"turn_index out of range (0..{len(turns) - 1})")
        turn = turns[body.turn_index]
        message = turn["message"]
    else:
        message = _message(body.message)
    base = body.model or CMP.model_wanted or MODEL_NAME
    models = {cid: (body.models or {}).get(cid) or base for cid in COLUMNS}
    if not CMP.lock.acquire(blocking=False):
        raise HTTPException(409, "a comparison turn is already running")
    try:
        if body.script and body.turn_index == 0:
            CMP.script = body.script
        pipeline = CMP.get(models["dmg"])
        row = _run_compare(pipeline, message, models, body.parallel, turn)
        if turn is not None:
            row.update(script=body.script, turn_index=body.turn_index)
        CMP.rows.append(row)
        index = len(CMP.rows) - 1
        for e in row["cols"]["dmg"]["changes"]["retired_edges"]:
            CMP.retired_at.setdefault(edge_id(e), index)
        out = _row_json(index, row)
        return jsonable_encoder({"row": out, "totals": _totals(),
                                 "graph": graph_json(pipeline.store, row["cols"]["dmg"]["changes"], row["_ranks"]),
                                 "dmg_model_used": CMP.model_used})
    finally:
        CMP.lock.release()


def _run_compare(pipeline, message, models, parallel, turn):
    """ui_core.compare_turn() (same as app.py) plus ranks and stale-fact evidence for the UI."""
    row = compare_turn(pipeline, CMP.hist, message, models, parallel, turn, trace=True)
    dmg = row["cols"]["dmg"]
    retrieval, basis = retrieval_ranks(dmg, row.pop("retrieved_raw") if TRACE_ON else None)
    row.update(models=models, retrieval=retrieval, rank_basis=basis, _ranks=_ranks(retrieval, dmg),
               stale={cid: stale_mentions(res.get("answer"), pipeline.store) for cid, res in row["cols"].items()})
    return row


class ChecksIn(BaseModel):
    script: str = "demo"


@app.post("/api/compare/checks")
def compare_checks(body: ChecksIn):
    with CMP.lock:
        checks = check_graph(CMP.get().store, _script(body.script))
        CMP.checks = {"script": body.script, "results": [{"check": c, "ok": bool(ok)} for c, ok in checks]}
    return CMP.checks


@app.post("/api/compare/reset")
def compare_reset():
    with CMP.lock:
        CMP.reset_rows()
        if CMP.pipeline is not None:
            CMP.pipeline.reset()
    return {"ok": True}


# ───────── web UI (built files) ─────────

if os.path.isdir(os.path.join(DIST_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "unknown API route")
    target = os.path.normpath(os.path.join(DIST_DIR, path))
    if path and target.startswith(DIST_DIR + os.sep) and os.path.isfile(target):
        return FileResponse(target)
    index = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index):  # never cache the page itself: its hashed asset names change on every build
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    return JSONResponse({"error": "web UI not built", "hint": "cd web && npm install && npm run build"}, 404)
