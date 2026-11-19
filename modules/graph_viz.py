"""
Graph visualization helper (owner: Arpit).

Generates a visual representation of the memory graph using NetworkX and
matplotlib. Saves the output as a PNG image for use in the README and
presentations.

Nov 18 update: added build_graph_html(), the interactive pyvis graph used by
the Streamlit app (colour by entity type + legend, predicate labels on edges,
highlights for what changed / was used this turn, retired facts dashed,
timeline replay). stable_layout() keeps node positions fixed between turns.
Both renderers share the TYPE_COLORS palette.

Nov 19 update (UI redesign): the per-node / per-edge data that build_graph_html()
draws now comes from graph_payload(), which the web UI's API also serves. Output of
build_graph_html() is unchanged (checked byte for byte against the Nov 18 version).

Usage (from the project root):
    from modules.graph_viz import visualize_graph
    visualize_graph(store, "screenshots/graph.png")

    from modules.graph_viz import stable_layout, build_graph_html
    positions = stable_layout(store.graph)
    html = build_graph_html(store.graph, positions)

Quick test:
    python -m modules.graph_viz
"""

import html
import os
import re
from collections import Counter

import networkx as nx
from pyvis.network import Network

from config import GRAPH_PATH, USER_NODE

USER_ID = USER_NODE.lower()  # node id of the hub ("user")

# Categorical colours, checked for colour-blind separation on a dark background.
# Keep this order. Types not listed here (or None) use OTHER_COLOR.
TYPE_COLORS = {
    "PERSON": "#3987e5",
    "CONDITION": "#d95926",
    "DRUG": "#199e70",
    "PLACE": "#c98500",
    "PREFERENCE": "#d55181",
    "ORGANIZATION": "#0ca30c",
    "FOOD": "#e66767",
}
OTHER_COLOR = "#8a8578"
HUB_COLOR = "#a855f7"

# Edge colours by what happened to the fact.
EDGE_COLORS = {
    "active": "rgba(220,210,190,0.30)",
    "new": "#6ee7a8",
    "updated": "#fcd34d",
    "retired": "rgba(150,140,125,0.35)",
    "retired_now": "#f87171",
}
USED_COLOR = "#f2e8d5"  # facts retrieved for the latest answer
BG = "#141414"

VIS_OPTIONS = """
{
  "nodes": {
    "shape": "dot",
    "scaling": { "min": 13, "max": 28, "label": { "enabled": false } },
    "font": { "color": "#cfc8b8", "size": 13, "face": "Inter, sans-serif",
              "strokeWidth": 3, "strokeColor": "#141414", "vadjust": 2 }
  },
  "edges": {
    "width": 1, "selectionWidth": 0, "hoverWidth": 0,
    "arrows": { "to": { "enabled": false, "scaleFactor": 0.45 } },
    "font": { "size": 11, "color": "#a39d8f", "face": "Inter, sans-serif",
              "strokeWidth": 4, "strokeColor": "#141414", "align": "middle" }
  },
  "physics": {
    "enabled": false,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.005,
                          "springLength": 120, "springConstant": 0.05,
                          "damping": 0.38, "avoidOverlap": 0.3 },
    "maxVelocity": 30, "minVelocity": 0.45, "timestep": 0.45,
    "stabilization": { "enabled": false }
  },
  "layout": { "randomSeed": 42 },
  "interaction": { "hover": true, "tooltipDelay": 250, "dragNodes": true,
                   "hoverConnectedEdges": false, "selectConnectedEdges": false,
                   "zoomSpeed": 0.6 }
}
"""

EXTRA_CSS = """
body { margin: 0; background: #141414; font-family: Inter, -apple-system, sans-serif; }
center, h1 { display: none; }
.card, #mynetwork { border: none !important; float: none !important; }
div.vis-tooltip { background: #1c1c1d; color: #e9e3d6; border: 1px solid rgba(240,230,210,0.15);
  border-radius: 8px; padding: 6px 9px; font: 12px Inter, sans-serif; white-space: pre-line; }
#dmg-legend { display: flex; flex-wrap: wrap; gap: 4px 13px; padding: 9px 12px 5px;
  font-size: 11px; color: #b8b2a4; border-bottom: 1px solid rgba(240,230,210,0.06); }
#dmg-legend span { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
#dmg-legend i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
#dmg-legend b { width: 14px; height: 0; display: inline-block; border-top: 2px solid; }
"""


# Obsidian-style interaction, injected after pyvis draws the graph (`network`, `nodes`,
# `edges` are pyvis globals). Modelled on a screen recording of Obsidian's graph view:
#  - hover (or drag) a node: it gets an accent ring, its edges turn accent purple and show
#    their relation + arrow, and everything else fades to ~15% over ~0.3 s (eased tween).
#    Moving off fades back. Click pins the focus; click empty space to release it.
#  - dragging turns physics on: neighbours swing along on springs, the rest drifts, and on
#    release the graph glides to rest (low damping), then physics switches off again.
# ponytail: dragged positions live only in the browser; the next Streamlit rerun redraws
# from the Python layout.
INTERACTION_JS = """
<script>
(function () {
  if (typeof network === "undefined") return;
  var ACCENT = "#8b5cf6", DIM = 0.15, FADE_MS = 300;
  var origNodes = {}, origEdges = {};
  nodes.get().forEach(function (n) {
    origNodes[n.id] = {color: n.color, font: n.font, shadow: n.shadow, borderWidth: n.borderWidth};
  });
  edges.get().forEach(function (e) {
    origEdges[e.id] = {color: e.color, font: e.font, width: e.width, label: e.label,
                       arrows: e.arrows, rel: e.rel};
  });

  // "#rrggbb" / "rgb(..)" / "rgba(..)" -> same colour with its alpha multiplied by a.
  function withAlpha(c, a) {
    if (typeof c !== "string") return c;
    var m = c.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    if (m) return "rgba(" + parseInt(m[1], 16) + "," + parseInt(m[2], 16) + "," +
                  parseInt(m[3], 16) + "," + a.toFixed(3) + ")";
    m = c.match(/^rgba?\\(([^,]+),([^,]+),([^,)]+)(?:,([^)]+))?\\)$/);
    if (m) return "rgba(" + m[1] + "," + m[2] + "," + m[3] + "," +
                  ((m[4] === undefined ? 1 : parseFloat(m[4])) * a).toFixed(3) + ")";
    return c;
  }
  function fadeColor(c, a) {
    if (typeof c === "string") return withAlpha(c, a);
    var out = {};
    for (var key in c) out[key] = fadeColor(c[key], a);
    return out;
  }

  var focusId = null, nodeSet = null, edgeSet = null, k = 0, raf = null;
  var pinned = null, dragging = false;

  function paint() {
    var a = 1 - (1 - DIM) * k;  // alpha for everything outside the focus
    nodes.update(Object.keys(origNodes).map(function (id) {
      var o = origNodes[id], font = o.font || {};
      if (id === focusId && k > 0)
        return {id: id, font: o.font, borderWidth: 3,
                color: Object.assign({}, o.color, {border: "#c4b5fd"}),
                shadow: {enabled: true, color: "rgba(139,92,246," + (0.8 * k).toFixed(3) + ")",
                         size: 22, x: 0, y: 0}};
      if (nodeSet.has(id)) return Object.assign({id: id}, o);
      return {id: id, color: fadeColor(o.color, a), borderWidth: o.borderWidth,
              shadow: k > 0.05 ? false : o.shadow,
              font: Object.assign({}, font, {color: withAlpha(font.color || "#cfc8b8", a),
                                             strokeWidth: k > 0.5 ? 0 : (font.strokeWidth || 3)})};
    }));
    edges.update(Object.keys(origEdges).map(function (id) {
      var o = origEdges[id];
      if (edgeSet.has(id) && k > 0.35)
        return {id: id, width: 1.8, label: o.rel || "",
                color: {color: ACCENT, highlight: ACCENT, hover: ACCENT},
                arrows: {to: {enabled: true, scaleFactor: 0.45}},
                font: {color: "#e9e3d6", size: 11, strokeWidth: 4, strokeColor: "#141414"}};
      if (edgeSet.has(id)) return Object.assign({id: id}, o);
      var col = typeof o.color === "string" ? o.color : (o.color && o.color.color) || "#888888";
      return {id: id, width: o.width, label: k > 0.35 ? "" : o.label, arrows: o.arrows,
              color: {color: withAlpha(col, a), highlight: withAlpha(col, a), hover: withAlpha(col, a)},
              font: Object.assign({}, o.font || {}, {color: "rgba(0,0,0,0)", strokeWidth: 0})};
    }));
  }

  function animateTo(target) {
    if (raf) cancelAnimationFrame(raf);
    var start = k, t0 = performance.now();
    function step(now) {
      var p = Math.min(1, (now - t0) / FADE_MS);
      k = start + (target - start) * (1 - Math.pow(1 - p, 3));  // ease-out cubic
      paint();
      if (p < 1) { raf = requestAnimationFrame(step); return; }
      raf = null;
      if (target === 0) {  // fully faded back: restore the exact original styling
        focusId = null;
        nodes.update(Object.keys(origNodes).map(function (id) { return Object.assign({id: id}, origNodes[id]); }));
        edges.update(Object.keys(origEdges).map(function (id) { return Object.assign({id: id}, origEdges[id]); }));
      }
    }
    raf = requestAnimationFrame(step);
  }

  function focus(id) {
    if (id === undefined || id === null) return;
    if (focusId === id && k > 0.99) return;
    focusId = id;
    nodeSet = new Set(network.getConnectedNodes(id).concat([id]));
    edgeSet = new Set(network.getConnectedEdges(id));
    animateTo(1);
  }
  function unfocus() { if (focusId !== null) animateTo(0); }

  network.on("hoverNode", function (p) { if (!dragging && pinned === null) focus(p.node); });
  network.on("blurNode", function () { if (!dragging && pinned === null) unfocus(); });
  network.on("click", function (p) {
    if (p.nodes.length) { pinned = p.nodes[0]; focus(pinned); }
    else { pinned = null; network.unselectAll(); unfocus(); }
  });
  network.on("dragStart", function (p) {
    if (!p.nodes.length) return;  // dragging empty space pans, like Obsidian
    dragging = true;
    focus(p.nodes[0]);
    network.setOptions({physics: {enabled: true}});
  });
  network.on("dragEnd", function (p) {
    if (!dragging) return;
    dragging = false;
    var over = network.getNodeAt(p.pointer.DOM);
    if (pinned !== null) focus(pinned);
    else if (over !== undefined) focus(over);
    else unfocus();
  });
  // Keep physics on for the WHOLE drag. vis fires "stabilized" whenever the graph comes to
  // rest - including while you pause holding a node - and switching physics off then froze
  // the neighbours for the rest of that drag. So: never stop mid-drag, and re-arm if needed.
  network.on("dragging", function (p) {
    if (dragging && !network.physics.options.enabled) network.setOptions({physics: {enabled: true}});
  });
  // After release the graph glides to rest; then the simulation stops (saves CPU, keeps layout).
  network.on("stabilized", function () {
    if (!dragging) network.setOptions({physics: {enabled: false}});
  });
  // Releasing the mouse outside the canvas can skip "dragEnd"; don't stay stuck in drag mode.
  window.addEventListener("pointerup", function () {
    setTimeout(function () {
      if (!dragging) return;
      dragging = false;
      if (pinned === null) unfocus();
    }, 80);
  });
})();
</script>
"""


def type_color(node_type):
    """Colour for an entity type, with a fallback for unknown / missing types."""
    return TYPE_COLORS.get(str(node_type).upper(), OTHER_COLOR) if node_type else OTHER_COLOR


def stable_layout(graph, previous=None, spacing=130):
    """Return {node: (x, y)} in pixels, keeping every node in `previous` where it was.

    Only new nodes are placed (spring layout with the old ones pinned), so the
    picture does not reshuffle every turn. The User hub is pinned at (0, 0).
    """
    pos = {n: p for n, p in (previous or {}).items() if n in graph}
    if USER_ID in graph:
        pos.setdefault(USER_ID, (0.0, 0.0))
    if all(n in pos for n in graph):
        return pos
    init = {n: (x / spacing, y / spacing) for n, (x, y) in pos.items()}
    placed = nx.spring_layout(nx.Graph(graph), pos=init or None, fixed=list(init) or None,
                              k=1.0, seed=42, iterations=200)
    return {n: (float(x) * spacing, float(y) * spacing) for n, (x, y) in placed.items()}


def timeline_steps(graph):
    """Sorted timestamps at which memory changed (facts created or retired)."""
    stamps = set()
    for _, _, data in graph.edges(data=True):
        stamps.update(t for t in (data.get("created_at"), data.get("ended_at")) if t)
    return sorted(stamps)


def changes_at(graph, stamp):
    """What changed at one timeline step, in the same shape build_graph_html() expects."""
    new_edges, retired, first_seen = set(), set(), {}
    for u, v, k, data in graph.edges(keys=True, data=True):
        created = data.get("created_at") or ""
        if created == stamp:
            new_edges.add((u, v, k))
        if data.get("ended_at") == stamp:
            retired.add((u, v, k))
        for node in (u, v):
            first_seen[node] = min(first_seen.get(node, created), created)
    return {"new_nodes": {n for n, t in first_seen.items() if t == stamp},
            "new_edges": new_edges, "retired_edges": retired}


def _visible_edges(graph, show_retired, until):
    """Edges to draw as (u, v, k, data, ended_at), applying the retired toggle and replay time."""
    edges = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        ended = data.get("ended_at")
        if until is not None:
            if (data.get("created_at") or "") > until:
                continue  # not created yet at this step
            if ended and ended > until:
                ended = None  # still active at this step
        if ended and not show_retired:
            continue
        edges.append((u, v, k, data, ended))
    return edges


def _legend_html(type_names, edge_states):
    """Legend strip above the canvas: entity types present plus the edge highlights in use."""
    items = [f'<span><i style="background:{HUB_COLOR}"></i>User</span>']
    for name in type_names:
        label = name.title() if name in TYPE_COLORS else "Other"
        items.append(f'<span><i style="background:{type_color(name)}"></i>{html.escape(label)}</span>')
    edge_labels = {"new": ("New", EDGE_COLORS["new"], "solid"),
                   "updated": ("Updated", EDGE_COLORS["updated"], "solid"),
                   "retired_now": ("Retired now", EDGE_COLORS["retired_now"], "dashed"),
                   "retired": ("Retired", "#8a8578", "dashed"),
                   "used": ("Used in answer", USED_COLOR, "solid; border-top-width:3px")}
    for state, (label, color, style) in edge_labels.items():
        if state in edge_states:
            items.append(f'<span><b style="border-color:{color}; border-top-style:{style}"></b>{label}</span>')
    return f'<div id="dmg-legend">{"".join(items)}</div>'


def _edge_state(key, ended, changes):
    """What happened to one fact: retired_now / retired / updated / new / active."""
    if key in changes.get("retired_edges", set()):
        return "retired_now"
    if ended:
        return "retired"
    if key in changes.get("updated_edges", set()):
        return "updated"
    if key in changes.get("new_edges", set()):
        return "new"
    return "active"


def graph_payload(graph, changes=None, show_retired=True, until=None, ranks=None):
    """The visible graph as plain JSON-ready data, shared by build_graph_html() and the web UI.

    Same arguments as build_graph_html() (minus layout). `ranks` maps (u, v, k) to the
    1-based retrieval rank of facts used for the latest answer.

    Returns {"nodes": [...], "links": [...]} in drawing order:
      node  id, label, type, color, degree, isHub, isNew, createdAt (first visible fact)
      link  id ("u|v|k"), source, target, key, predicate, label, text, createdAt, endedAt,
            state (active / new / updated / retired / retired_now), used, rank
    """
    changes = changes or {}
    ranks = ranks or {}
    new_nodes = changes.get("new_nodes", set())
    edges = _visible_edges(graph, show_retired, until)
    degree = Counter(n for u, v, *_ in edges for n in (u, v))
    first_seen = {}
    for u, v, k, data, ended in edges:
        created = data.get("created_at") or ""
        for n in (u, v):
            first_seen[n] = min(first_seen.get(n, created), created)

    nodes = []
    for node in graph.nodes():
        if node not in degree:
            continue
        data = graph.nodes[node]
        node_type = data.get("type")
        is_hub = node == USER_ID
        nodes.append({
            "id": node, "label": data.get("label", node), "type": node_type,
            "color": HUB_COLOR if is_hub else type_color(node_type),
            "degree": degree[node], "isHub": is_hub, "isNew": node in new_nodes,
            "createdAt": first_seen.get(node) or None,
        })

    links = []
    for u, v, k, data, ended in edges:
        key = (u, v, k)
        predicate = str(data.get("predicate", ""))
        label = predicate.replace("_", " ")
        links.append({
            "id": f"{u}|{v}|{k}", "source": u, "target": v, "key": k,
            "predicate": predicate, "label": label,
            "text": f"{graph.nodes[u].get('label', u)} {label} {graph.nodes[v].get('label', v)}",
            "createdAt": data.get("created_at"), "endedAt": ended,
            "state": _edge_state(key, ended, changes),
            "used": key in changes.get("used_edges", set()),
            "rank": ranks.get(key),
        })
    return {"nodes": nodes, "links": links}


def build_graph_html(graph, positions, height=560, changes=None, show_retired=True, until=None):
    """Return the memory graph as a self-contained (offline) interactive HTML page.

    positions     {node: (x, y)} from stable_layout()
    changes       sets for the latest turn: new_nodes, new_edges, updated_edges,
                  retired_edges and used_edges (edges are (u, v, k) tuples)
    show_retired  draw retired facts as dashed lines
    until         ISO timestamp for timeline replay: hide facts created later and
                  show facts retired later as still active
    """
    payload = graph_payload(graph, changes, show_retired, until)

    net = Network(height=f"{height}px", width="100%", bgcolor=BG, font_color="#cfc8b8",
                  directed=True, cdn_resources="in_line")
    net.set_options(VIS_OPTIONS)

    type_names = []
    for item in payload["nodes"]:
        node, label, node_type = item["id"], item["label"], item["type"]
        is_hub, is_new, color = item["isHub"], item["isNew"], item["color"]
        if not is_hub:
            name = str(node_type).upper() if node_type and str(node_type).upper() in TYPE_COLORS else "OTHER"
            if name not in type_names:
                type_names.append(name)
        x, y = positions.get(node, (0.0, 0.0))
        net.add_node(
            node, label=label, x=x, y=y, value=item["degree"] + 2,
            title=f"{label}\n{node_type or 'no type'} · {item['degree']} link(s)" + ("\nnew this turn" if is_new else ""),
            color={"background": color, "border": "#fff4e2" if is_new else color,
                   "highlight": {"background": color, "border": "#ffffff"},
                   "hover": {"background": color, "border": "#ffffff"}},
            borderWidth=3 if is_new else 1,
            shadow={"enabled": True, "size": 24 if is_new else 8, "x": 0, "y": 0,
                    "color": "rgba(255,244,226,0.85)" if is_new else "rgba(0,0,0,0.5)"},
            font={"size": 15 if is_hub else 13, "color": "#e9e3d6" if is_hub else "#cfc8b8"},
        )

    edge_states = set()
    parallel = Counter()  # curve parallel edges so their labels don't overlap
    for item in payload["links"]:
        u, v, ended = item["source"], item["target"], item["endedAt"]
        state, used, predicate = item["state"], item["used"], item["label"]
        edge_states.update({state, "used"} if used else {state})

        pair = frozenset((u, v))
        bend = parallel[pair]
        parallel[pair] += 1
        title = item["text"]
        title += f"\nadded {item['createdAt'] or '?'}" + (f"\nretired {ended}" if ended else "")
        # Obsidian look: plain facts are thin unlabelled lines (hover a node to see its
        # relations); only what changed this turn keeps its label and arrow.
        highlighted = used or state in ("new", "updated", "retired_now")
        net.add_edge(
            u, v, label=predicate if highlighted else "", rel=predicate, title=title,
            dashes=bool(ended), **({"arrows": "to"} if highlighted else {}),
            color=USED_COLOR if used and state == "active" else EDGE_COLORS[state],
            width=3.2 if used else 2.2 if highlighted else 1,
            smooth={"enabled": True, "type": "curvedCW", "roundness": 0.2 * bend} if bend else False,
            font={"color": "#f2e8d5" if used else "#6f6a60" if ended else "#a39d8f",
                  "size": 12 if used else 11},
        )

    page = net.generate_html()
    # Drop pyvis' Bootstrap CDN tags (unused) so the page has no network dependencies.
    page = re.sub(r"<(link|script)[^>]*bootstrap[^>]*>(\s*</script>)?", "", page)
    legend = _legend_html(type_names, edge_states)
    page = page.replace("</head>", f"<style>{EXTRA_CSS}</style></head>", 1)
    page = page.replace("<body>", f"<body>{legend}", 1)
    return page.replace("</body>", f"{INTERACTION_JS}</body>", 1)


def visualize_graph(graph_store, output_path="screenshots/graph.png",
                    active_only=True, figsize=(14, 10)):
    """Draw the memory graph and save it as a PNG image.

    Nodes are colour-coded by type, edges are labelled with predicates,
    and ended (inactive) edges are drawn as dashed grey lines.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving to file.
    import matplotlib.pyplot as plt

    graph = graph_store.graph
    if graph.number_of_nodes() == 0:
        print("Graph is empty, nothing to visualize.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Dynamic Memory Graph", fontsize=16, fontweight="bold", pad=20)

    # Layout.
    pos = nx.spring_layout(graph, seed=42, k=2.0)

    # Colour nodes by type (same palette as the app).
    node_colors = []
    node_labels = {}
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_colors.append(HUB_COLOR if node == USER_ID else type_color(node_data.get("type")))
        label = node_data.get("label", node)
        node_labels[node] = label

    # Draw nodes.
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors,
                           node_size=1500, alpha=0.9, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax,
                            font_size=9, font_weight="bold")

    # Separate active and inactive edges.
    active_edges = []
    inactive_edges = []
    edge_labels_active = {}
    edge_labels_inactive = {}

    for u, v, k, data in graph.edges(keys=True, data=True):
        predicate = data.get("predicate", "")
        if data.get("ended_at"):
            inactive_edges.append((u, v))
            edge_labels_inactive[(u, v)] = predicate.replace("_", " ")
        else:
            active_edges.append((u, v))
            edge_labels_active[(u, v)] = predicate.replace("_", " ")

    # Draw active edges (solid).
    if active_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=active_edges, ax=ax,
                               edge_color="#333333", width=1.5, alpha=0.8,
                               arrows=True, arrowsize=15,
                               connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels_active, ax=ax,
                                     font_size=7, font_color="#333333")

    # Draw inactive edges (dashed grey).
    if inactive_edges and not active_only:
        nx.draw_networkx_edges(graph, pos, edgelist=inactive_edges, ax=ax,
                               edge_color="#BDBDBD", width=1.0, alpha=0.5,
                               style="dashed", arrows=True, arrowsize=10,
                               connectionstyle="arc3,rad=0.1")

    # Legend.
    legend_items = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                               markersize=10, label=name.title())
                    for name, color in [("User", HUB_COLOR), *TYPE_COLORS.items(), ("Other", OTHER_COLOR)]]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9,
              title="Entity types", title_fontsize=10)

    ax.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Graph saved to {output_path}")


if __name__ == "__main__":
    from modules.graph_store import GraphStore
    import tempfile

    # Build a sample graph for testing.
    tmp_path = os.path.join(tempfile.gettempdir(), "dmg_viz_test.json")
    store = GraphStore(tmp_path)
    store.add_edge("User", "has", "Diabetes", obj_type="CONDITION", created_at="2026-10-23T10:00:00")
    store.add_edge("User", "takes", "Metformin", obj_type="DRUG", created_at="2026-10-23T10:00:00")
    store.add_edge("User", "lives_in", "Bangalore", obj_type="PLACE", created_at="2026-10-24T09:00:00")
    store.add_edge("User", "likes", "Filter Coffee", obj_type="PREFERENCE", created_at="2026-10-24T09:00:00")
    store.add_edge("User", "has_sister", "Priya", obj_type="PERSON", created_at="2026-10-25T08:00:00")
    store.add_edge("Priya", "lives_in", "Pune", obj_type="PLACE", created_at="2026-10-25T08:05:00")

    # Interactive HTML: stable layout + offline page + legend.
    first = stable_layout(store.graph)
    store.add_edge("User", "works_at", "Infosys", obj_type="ORGANIZATION", created_at="2026-10-26T08:00:00")
    second = stable_layout(store.graph, first)
    assert all(second[n] == first[n] for n in first), "existing nodes must not move"
    page = build_graph_html(store.graph, second, changes=changes_at(store.graph, "2026-10-26T08:00:00"))
    assert "cdnjs" not in page and "bootstrap" not in page, "graph must render offline"
    assert "dmg-legend" in page and "works at" in page and TYPE_COLORS["ORGANIZATION"] in page
    print("build_graph_html OK:", len(page), "chars")

    output = "screenshots/graph.png"
    visualize_graph(store, output)
    print(f"Test graph visualization saved to {output}")
