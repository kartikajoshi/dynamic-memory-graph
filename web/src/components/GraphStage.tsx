// The canvas side: the 3D memory graph plus everything that floats over it (legend, node card,
// glass toolbar with the timeline, empty-state prompts). One instance serves Chat and Compare:
// switching modes swaps the engine's "world" instead of remounting the canvas.
import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import type { Graph3D } from '../graph/scene';
import { api, hhmm, hhmmss, plural, type GLink, type GraphPayload } from '../api';
import { EDGE_COLORS, HUB_UI, typeColor, typeName } from '../palette';
import { Icon } from './bits';
import Fallback2D from './Fallback2D';

export interface GraphHandle {
  highlight(ids: string[] | null): void;
  flyToLink(id: string): void;
  select(id: string | null): void;
  recenter(): void;
  clear(): void;
  toggleRetired(): void;
  setMode3d(on: boolean): void;
  togglePlay(): void;
  view(): { showRetired: boolean; mode3d: boolean };
}

interface Props {
  scope: 'chat' | 'compare';
  payload: GraphPayload | null;
  eventKey: number; // bumps when a new turn lands for this scope: animate it and play the retrieved flow
  reduced: boolean;
  examples: string[];
  canSend: boolean;
  onExample: (text: string) => void;
  onEntities: (ents: { id: string; label: string; type: string | null }[]) => void;
  children?: ReactNode;
}

type Preview = { left: number; at: string; i: number; rows: [string, string][] };

const retrievedIds = (p: GraphPayload) =>
  p.links.filter((l) => l.rank != null && !l.endedAt).sort((a, b) => (a.rank ?? 9) - (b.rank ?? 9)).map((l) => l.id);

export default forwardRef<GraphHandle, Props>(function GraphStage(props, ref) {
  const { scope, payload, eventKey, reduced, examples, canSend, onExample, onEntities, children } = props;
  const host = useRef<HTMLDivElement>(null);
  const legendRef = useRef<HTMLDivElement>(null);
  const toolbarRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLElement>(null);
  const g = useRef<Graph3D | null>(null);
  const [ready, setReady] = useState(false);
  const [noGL, setNoGL] = useState(false);
  const [shown, setShown] = useState<GraphPayload | null>(null); // what is on screen (live or a replay step)
  const [showRetired, setShowRetired] = useState(true);
  const [mode3d, setMode3dState] = useState(!reduced);
  const [replay, setReplayIdx] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const lastEvent = useRef<Record<string, number>>({});
  const cache = useRef(new Map<string, GraphPayload>());
  const cacheSig = useRef('');
  const replayToken = useRef(0);
  const playTimer = useRef<number | null>(null);
  const live = useRef<GraphPayload | null>(null);
  live.current = payload;

  // ───────── engine lifecycle (code-split: the shell paints before three.js loads) ─────────
  useEffect(() => {
    let dead = false;
    let view: Graph3D | null = null;
    import('../graph/scene.js')
      .then(({ Graph3D }) => {
        if (dead || !host.current) return;
        try {
          view = new Graph3D(host.current, { reducedMotion: reduced, onSelect: (id) => setSelected(id) });
        } catch {
          setNoGL(true); // no WebGL: the 2D SVG view takes over, never a blank box
          return;
        }
        if (reduced) view.setMode3d(false);
        g.current = view;
        setReady(true);
      })
      .catch(() => setNoGL(true));
    return () => {
      dead = true;
      view?.dispose();
      g.current = null;
      setReady(false);
    };
    // reduced motion is pushed separately below; the engine is built once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => { g.current?.setReducedMotion(reduced); }, [reduced, ready]);

  const applyPayload = useCallback((p: GraphPayload, animate: boolean) => {
    const view = g.current;
    setShown(p);
    if (!view) return;
    view.apply(
      p.nodes.map((n) => ({ id: n.id, label: n.label, type: n.type })),
      p.links.map((l) => ({ id: l.id, source: l.source, target: l.target, predicate: l.predicate, state: l.state, rank: l.rank != null ? l.rank - 1 : null })),
      { animate },
    );
    const hot = new Set<string>();
    if (!p.until && p.changes) {
      const byId = new Map(p.links.map((l) => [l.id, l]));
      for (const id of [...p.changes.new_edges, ...p.changes.updated_edges, ...p.changes.retired_edges]) {
        const l = byId.get(id);
        if (l) { hot.add(l.source); hot.add(l.target); }
      }
    }
    for (const n of view.nodes.values()) n.hot = hot.has(n.id);
    view.setEmptyField(!p.links.length && !p.until);
  }, []);

  // ───────── live payload in ─────────
  useEffect(() => {
    const view = g.current;
    if (!payload || (!view && !noGL)) return;
    const switched = !!view && view.worldKey !== scope;
    if (switched) {
      view!.useWorld(scope);
      setSelected(null);
      setPreview(null);
    }
    const isNew = eventKey > 0 && eventKey !== lastEvent.current[scope];
    if (replay != null && !isNew && !switched) return; // a replay step is on screen; keep it
    if (replay != null) { stopPlay(); setReplayIdx(null); }
    applyPayload(payload, isNew && !switched);
    onEntities(payload.nodes.map((n) => ({ id: n.id, label: n.label, type: n.isHub ? null : n.type })));
    if (isNew) {
      lastEvent.current[scope] = eventKey;
      const ids = retrievedIds(payload);
      const drew = payload.changes && (payload.changes.new_edges.length || payload.changes.updated_edges.length);
      if (ids.length && view) window.setTimeout(() => g.current?.playRetrieved(ids), drew ? 750 : 120);
    }
    if (switched || !view?.metrics.framed) {
      if (view) view.metrics.framed = 1;
      window.setTimeout(() => g.current?.recenter(), 40);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, noGL, scope, payload, eventKey]);

  // ───────── time travel ─────────
  const steps = payload?.steps ?? [];
  const sig = `${scope}:${steps.map((s) => s.at).join(',')}`;
  if (sig !== cacheSig.current) { cacheSig.current = sig; cache.current.clear(); }

  const stepPayload = useCallback(async (at: string) => {
    const hit = cache.current.get(at);
    if (hit) return hit;
    const p = await api.graph(scope, at);
    cache.current.set(at, p);
    return p;
  }, [scope]);

  const setReplay = useCallback(async (i: number | null) => {
    const n = live.current?.steps.length ?? 0;
    const token = ++replayToken.current;
    g.current?.select(null);
    setSelected(null);
    if (i == null || i >= n - 1) {
      setReplayIdx(null);
      if (live.current) applyPayload(live.current, true);
      return;
    }
    setReplayIdx(i);
    try {
      const p = await stepPayload(live.current!.steps[i].at);
      if (token === replayToken.current) applyPayload(p, true);
    } catch { /* step vanished (memory reset): stay live */ }
  }, [applyPayload, stepPayload]);

  function stopPlay() {
    if (playTimer.current) window.clearTimeout(playTimer.current);
    playTimer.current = null;
    setPlaying(false);
  }
  const togglePlay = useCallback(async () => {
    if (playTimer.current || playing) { stopPlay(); return; }
    const st = live.current?.steps ?? [];
    if (st.length < 2) return;
    setPlaying(true);
    await Promise.all(st.slice(0, -1).map((s) => stepPayload(s.at).catch(() => null))); // scrubbing never waits on the network
    // gaps between memory changes clamped to 350-1200 ms: a session with pauses plays as a short film
    const gap = (a: string, b: string) => Math.max(350, Math.min(1200, (Date.parse(b) - Date.parse(a)) * 0.06));
    let i = 0;
    const tick = () => {
      if (i >= st.length) { stopPlay(); return; }
      setReplay(i === st.length - 1 ? null : i);
      playTimer.current = window.setTimeout(tick, i + 1 < st.length ? gap(st[i].at, st[i + 1].at) : 900);
      i++;
    };
    tick();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, setReplay, stepPayload]);
  useEffect(() => () => stopPlay(), [scope]);

  const showPreview = async (el: HTMLElement, i: number) => {
    const at = steps[i].at;
    const tr = el.getBoundingClientRect(), sr = host.current!.getBoundingClientRect();
    const left = tr.left + tr.width / 2 - sr.left;
    let p: GraphPayload;
    try { p = await stepPayload(at); } catch { return; }
    const rows: [string, string][] = [];
    const ids: string[] = [];
    for (const l of p.links) {
      if (l.createdAt === at && l.state !== 'retired_now') { rows.push(['+', l.text]); ids.push(l.id); }
      if (l.endedAt === at) { rows.push(['−', l.text]); ids.push(l.id); }
    }
    setPreview({ left, at, i, rows });
    g.current?.highlightLinks(ids);
  };
  const hidePreview = () => { setPreview(null); g.current?.highlightLinks(null); };

  // ───────── view controls ─────────
  const toggleRetired = () => {
    const on = !showRetired;
    setShowRetired(on);
    g.current?.setRetiredVisible(on);
  };
  const setMode3d = (on: boolean) => { setMode3dState(on); g.current?.setMode3d(on); };

  useImperativeHandle(ref, () => ({
    highlight: (ids) => g.current?.highlightLinks(ids),
    flyToLink: (id) => g.current?.flyToLink(id),
    select: (id) => g.current?.select(id),
    recenter: () => g.current?.recenter(),
    clear: () => { g.current?.select(null); setSelected(null); g.current?.clear(); },
    toggleRetired,
    setMode3d,
    togglePlay,
    view: () => ({ showRetired, mode3d }),
  }));

  // free area for framing: the canvas minus legend, toolbar and node card
  useLayoutEffect(() => {
    const update = () => {
      const st = host.current, view = g.current;
      if (!st || !view) return;
      const sr = st.getBoundingClientRect();
      let l = 14, t = 14, r = sr.width - 14, b = sr.height - 14;
      const lg = legendRef.current, tb = toolbarRef.current, card = cardRef.current;
      if (lg && !selected) t = lg.getBoundingClientRect().bottom - sr.top + 10;
      if (tb) b = tb.getBoundingClientRect().top - sr.top - 10;
      if (card && window.innerWidth > 900) r = card.getBoundingClientRect().left - sr.left - 10;
      if (r - l < 120) { l = 14; r = sr.width - 14; }
      view.setFreeRect({ l, t, r, b });
    };
    update();
    const ro = new ResizeObserver(update);
    if (host.current) ro.observe(host.current);
    return () => ro.disconnect();
  });

  // the card only exists after the click re-renders: frame again once the free rect knows about it
  useEffect(() => { if (selected) g.current?.select(selected); }, [selected]);

  // pointer parallax: overlays float at different depths (CSS reads --px / --py)
  const onPointerMove = (e: React.PointerEvent) => {
    if (reduced || !host.current) return;
    const r = host.current.getBoundingClientRect();
    host.current.style.setProperty('--px', (((e.clientX - r.left) / r.width) * 2 - 1).toFixed(3));
    host.current.style.setProperty('--py', (((e.clientY - r.top) / r.height) * 2 - 1).toFixed(3));
  };

  const cur = replay ?? steps.length - 1;
  const x = (i: number) => (steps.length <= 1 ? 50 : 3 + (i / (steps.length - 1)) * 94);
  const scrubTo = (clientX: number, track: HTMLElement) => {
    if (!steps.length) return;
    const r = track.getBoundingClientRect();
    const p = ((clientX - r.left) / r.width) * 100;
    let best = 0, bd = 1e9;
    steps.forEach((_, i) => { const d = Math.abs(x(i) - p); if (d < bd) { bd = d; best = i; } });
    if (best !== cur) setReplay(best);
  };
  const stepKind = (i: number) => {
    const at = steps[i].at;
    const links = (live.current?.links ?? []);
    const created = links.some((l) => l.createdAt === at), ended = links.some((l) => l.endedAt === at);
    return created && ended ? 'update' : ended ? 'retire' : 'add';
  };

  const empty = !!payload && !payload.links.length && replay == null;
  const p = shown;
  const card = selected && p ? nodeCard(selected, p, steps.length ? [steps[0].at, steps[steps.length - 1].at] : null) : null;

  return (
    <section className={`stage3d scope-${scope}`} aria-label="Memory graph" ref={host} onPointerMove={onPointerMove}>
      <div className="vignette" />
      {noGL && p && <Fallback2D payload={p} showRetired={showRetired} onSelect={setSelected} />}
      {p && p.links.length > 0 && !selected && (
        <div className="legend glass depth-1" ref={legendRef} aria-hidden="true">
          <span><i style={{ background: HUB_UI }} />User</span>
          {[...new Set(p.nodes.filter((n) => !n.isHub).map((n) => typeName(n.type)))].map((t) => (
            <span key={t}><i style={{ background: typeColor(t) }} />{t}</span>
          ))}
          {legendStates(p.links, showRetired)}
        </div>
      )}

      {empty && scope === 'chat' && (
        <div className="prompts">
          <div className="eyebrow"><b>Nothing in memory yet</b>Tell it something. Watch the fact leave the core.</div>
          {examples.map((t, i) => (
            <button key={t} className="pcard glass" data-tilt disabled={!canSend} onClick={() => onExample(t)}>
              <span className="q">0{i + 1}</span>{t}
            </button>
          ))}
        </div>
      )}

      {card && (
        <aside className="nodecard glass depth-3" ref={cardRef} aria-label={`${card.label} details`} data-tilt>
          <header>
            <span className="sw" style={{ background: card.color }} />
            <h3>{card.label}</h3>
            <button className="x" aria-label="Close" onClick={() => { g.current?.select(null); setSelected(null); }}><Icon name="x" size={13} /></button>
          </header>
          <div className="meta">{card.meta}</div>
          {card.facts.map((f) => (
            <div key={f.id} className={`fact ${f.endedAt ? 'retired' : ''}`} onMouseEnter={() => g.current?.highlightLinks([f.id])} onMouseLeave={() => g.current?.highlightLinks(null)}>
              <div className="txt">{f.text}</div>
              <div className="life" style={{ ['--c' as string]: f.color }}>
                <b className={f.endedAt ? 'ret' : ''} style={{ left: `${f.a}%`, width: `${Math.max(1.5, f.b - f.a)}%` }} />
                {f.endedAt && <span className="end" style={{ left: `calc(${f.b}% - 1px)` }} />}
              </div>
              <div className="when"><span>added {hhmm(f.createdAt)}</span><span>{f.endedAt ? `retired ${hhmm(f.endedAt)}` : 'active'}</span></div>
            </div>
          ))}
        </aside>
      )}

      {replay != null && steps[replay] && (
        <div className="replaying glass depth-2">Replaying · step {replay + 1} of {steps.length} · {hhmmss(steps[replay].at)}</div>
      )}
      {preview && (
        <div className="preview glass" style={{ left: preview.left }} onPointerLeave={hidePreview}>
          <div className="t"><span>step {preview.i + 1}/{steps.length}</span><span>{hhmmss(preview.at)}</span></div>
          {steps[preview.i]?.message && <div className="said">“{steps[preview.i].message}”</div>}
          {preview.rows.map(([s, t], k) => (
            <div className="row" key={k}><span className="s" style={{ color: s === '+' ? 'var(--text)' : 'var(--muted)' }}>{s}</span><span>{t}</span></div>
          ))}
        </div>
      )}

      <div className="toolbar glass depth-2" ref={toolbarRef}>
        <button className="tb" aria-pressed={showRetired} title="Show retired facts as dashed lines" onClick={toggleRetired}>
          <span className="switch" /><span className="txt">Retired</span>
        </button>
        <div className="dim2" role="group" aria-label="View">
          <button aria-pressed={!mode3d} onClick={() => setMode3d(false)}>2D</button>
          <button aria-pressed={mode3d} onClick={() => setMode3d(true)} disabled={noGL}>3D</button>
        </div>
        <button className="tb icon" title="Recentre (F)" aria-label="Recentre graph" onClick={() => g.current?.recenter()}><Icon name="target" /></button>
        <span className="vsep" />
        <div className="timeline">
          <button className="tb icon" aria-label={playing ? 'Pause replay' : 'Replay history'} title="Replay history" disabled={steps.length < 2} onClick={togglePlay}>
            <Icon name={playing ? 'pause' : 'play'} size={13} />
          </button>
          <div
            className="track" role="slider" tabIndex={0} aria-label="Memory timeline"
            aria-valuemin={1} aria-valuemax={Math.max(1, steps.length)} aria-valuenow={cur + 1}
            aria-valuetext={steps[cur] ? `step ${cur + 1} of ${steps.length}, ${hhmmss(steps[cur].at)}` : 'empty'}
            onPointerDown={(e) => { const t = e.currentTarget; t.setPointerCapture(e.pointerId); hidePreview(); scrubTo(e.clientX, t); }}
            onPointerMove={(e) => { if (e.currentTarget.hasPointerCapture(e.pointerId)) scrubTo(e.clientX, e.currentTarget); }}
            onKeyDown={(e) => {
              if (!steps.length) return;
              if (e.key === 'ArrowLeft') { setReplay(Math.max(0, cur - 1)); e.preventDefault(); }
              if (e.key === 'ArrowRight') { setReplay(cur + 1); e.preventDefault(); }
            }}
          >
            <div className="base" />
            <div className="fill" style={{ width: `${steps.length ? x(Math.max(0, cur)) : 0}%` }} />
            {steps.map((s, i) => (
              <div
                key={s.at} className={`tick k-${stepKind(i)} ${i > cur ? 'future' : ''} ${i === cur && replay != null ? 'on' : ''}`}
                style={{ left: `${x(i)}%` }}
                onPointerEnter={(e) => showPreview(e.currentTarget, i)}
                onPointerLeave={(e) => { if (!(e.relatedTarget as HTMLElement)?.closest?.('.preview')) hidePreview(); }}
              ><i /></div>
            ))}
            {steps.length > 0 && <div className="thumb2" style={{ left: `${x(Math.max(0, cur))}%` }} />}
          </div>
          <button className={`now ${replay == null ? 'live' : ''}`} onClick={() => { stopPlay(); setReplay(null); }} title="Back to live">
            {replay == null ? '● Now' : 'Now'}
          </button>
        </div>
      </div>
      {children}
    </section>
  );
});

function legendStates(links: GLink[], showRetired: boolean) {
  const st = new Set(links.map((l) => l.state));
  const chip = (c: string, text: string, line: string) => <span key={text}><b style={{ borderColor: c, borderTopStyle: line as 'solid' }} />{text}</span>;
  return [
    st.has('new') && chip(EDGE_COLORS.new, 'New', 'solid'),
    st.has('updated') && chip(EDGE_COLORS.updated, 'Updated', 'solid'),
    st.has('retired_now') && chip(EDGE_COLORS.retired_now, 'Retired now', 'dashed'),
    st.has('retired') && showRetired && chip('#8a8578', 'Retired', 'dashed'),
    links.some((l) => l.rank != null && !l.endedAt) && chip(EDGE_COLORS.used, 'Retrieved', 'solid'),
  ];
}

function nodeCard(id: string, p: GraphPayload, span: [string, string] | null) {
  const node = p.nodes.find((n) => n.id === id);
  if (!node) return null;
  const isHub = node.isHub;
  const [t0, t1] = span ?? ['', ''];
  const frac = (s: string | null) => (s && t1 > t0 ? (Date.parse(s) - Date.parse(t0)) / (Date.parse(t1) - Date.parse(t0)) : 1);
  const byId = new Map(p.nodes.map((n) => [n.id, n]));
  const facts = p.links
    .filter((l) => l.source === id || l.target === id)
    .sort((a, b) => (a.endedAt ? 1 : 0) - (b.endedAt ? 1 : 0))
    .map((l) => {
      const other = byId.get(l.source === id ? l.target : l.source);
      return { ...l, color: other?.isHub ? HUB_UI : typeColor(other?.type), a: frac(l.createdAt) * 100, b: l.endedAt ? frac(l.endedAt) * 100 : 100 };
    });
  return {
    label: node.label,
    color: isHub ? HUB_UI : typeColor(node.type),
    meta: `${isHub ? 'USER · memory core' : String(node.type || 'OTHER').toUpperCase()} · ${plural(facts.length, 'fact')}`,
    facts,
  };
}
