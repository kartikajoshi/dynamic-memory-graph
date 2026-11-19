// App shell: top bar, Chat | Compare, one shared graph canvas, facts sheet, ⌘K.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { api, plural, type GraphPayload, type Script, type Status, type Turn } from './api';
import { HUB_UI, typeColor } from './palette';
import { Mark, Num } from './components/bits';
import GraphStage, { type GraphHandle } from './components/GraphStage';
import ChatView, { type Pending } from './components/ChatView';
import CompareView from './components/CompareView';
import FactsPanel from './components/FactsPanel';
import CommandMenu, { type Command } from './components/CommandMenu';

type Mode = 'chat' | 'compare';

function useReducedMotion() {
  const q = useMemo(() => window.matchMedia('(prefers-reduced-motion: reduce)'), []);
  const [on, setOn] = useState(q.matches);
  useEffect(() => { const f = () => setOn(q.matches); q.addEventListener('change', f); return () => q.removeEventListener('change', f); }, [q]);
  return on;
}

// tilt toward the cursor (≤ 4°) on anything marked data-tilt; off under reduced motion
function useTilt(off: boolean) {
  useEffect(() => {
    if (off) return;
    let last: HTMLElement | null = null;
    const move = (e: PointerEvent) => {
      const el = (e.target as HTMLElement)?.closest?.('[data-tilt]') as HTMLElement | null;
      if (last && last !== el) { last.style.removeProperty('--rx'); last.style.removeProperty('--ry'); }
      last = el;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const k = Math.min(4, 1400 / Math.max(r.width, r.height)); // big surfaces tilt less
      el.style.setProperty('--rx', `${(-((e.clientY - r.top) / r.height - 0.5) * k).toFixed(2)}deg`);
      el.style.setProperty('--ry', `${(((e.clientX - r.left) / r.width - 0.5) * k).toFixed(2)}deg`);
    };
    document.addEventListener('pointermove', move);
    return () => document.removeEventListener('pointermove', move);
  }, [off]);
}

export default function App() {
  const reduced = useReducedMotion();
  useTilt(reduced);
  const [status, setStatus] = useState<Status | null>(null);
  const [offline, setOffline] = useState(false); // the API itself is unreachable
  const [mode, setMode] = useState<Mode>('chat');
  const [chatGraph, setChatGraph] = useState<GraphPayload | null>(null);
  const [chatEvent, setChatEvent] = useState(0);
  const [cmpGraph, setCmpGraph] = useState<GraphPayload | null>(null);
  const [cmpEvent, setCmpEvent] = useState(0);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [pending, setPending] = useState<Pending | null>(null);
  const [loadedFacts, setLoadedFacts] = useState(0);
  const [facts, setFacts] = useState<{ confirm?: boolean } | null>(null);
  const [cmdk, setCmdk] = useState(false);
  const [showCmpGraph, setShowCmpGraph] = useState(true);
  const [entities, setEntities] = useState<{ id: string; label: string; type: string | null }[]>([]);
  const [announce, setAnnounce] = useState('');
  const [toast, setToast] = useState<string | null>(null);
  const [scripts, setScripts] = useState<Script[]>([]);
  const [chatDemo, setChatDemo] = useState<{ i: number; n: number; stopping: boolean } | null>(null);
  const stopChatDemo = useRef(false);
  const graph = useRef<GraphHandle>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const demo = useRef<{ run: () => void; stop: () => void; running: boolean }>({ run: () => {}, stop: () => {}, running: false });
  const segRef = useRef<HTMLDivElement>(null);
  const [thumb, setThumb] = useState({ x: 0, w: 0 });

  // ───────── boot + status polling ─────────
  const refreshStatus = useCallback(() => api.status().then((s) => { setStatus(s); setOffline(false); return s; }).catch(() => { setOffline(true); return null; }), []);
  useEffect(() => {
    refreshStatus().then((s) => s && setLoadedFacts(s.loaded_facts));
    api.history().then((h) => { setTurns(h.turns); setLoadedFacts(h.loaded_facts); }).catch(() => {});
    api.graph('chat').then(setChatGraph).catch(() => {});
    api.scripts().then((r) => setScripts(r.scripts)).catch(() => {});
    const id = window.setInterval(refreshStatus, 15000);
    return () => window.clearInterval(id);
  }, [refreshStatus]);

  const flash = (t: string) => { setToast(t); window.setTimeout(() => setToast((x) => (x === t ? null : x)), 2400); };

  // ───────── chat ─────────
  const send = useCallback(async (message: string): Promise<boolean> => {
    setPending({ message, live: {} });
    try {
      const turn = await api.chatStream(message, (e) =>
        setPending((p) => (p ? { ...p, live: { ...p.live, [e.stage]: { status: e.status, seconds: e.seconds } } } : p)));
      setTurns((t) => [...t, turn]);
      setPending(null);
      if (turn.graph) { setChatGraph(turn.graph); setChatEvent((k) => k + 1); }
      const c = turn.changes;
      const bits = [c?.new_edges.length && `${plural(c.new_edges.length, 'fact')} added`, c?.updated_edges.length && `${c.updated_edges.length} updated`, c?.retired_edges.length && `${c.retired_edges.length} retired`].filter(Boolean);
      setAnnounce(bits.length ? bits.join(', ') : `${turn.retrieval.length} facts retrieved`);
      return true;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setPending((p) => (p ? { ...p, error: `That turn failed: ${msg}` } : p));
      refreshStatus();
      return false;
    }
  }, [refreshStatus]);

  const resetMemory = async (quiet = false) => {
    await api.reset();
    graph.current?.clear();
    setTurns([]); setLoadedFacts(0); setPending(null);
    const g = await api.graph('chat');
    setChatGraph(g);
    setChatEvent((k) => k + 1);
    setFacts(null);
    refreshStatus();
    if (!quiet) flash('Memory reset');
  };

  const clearChat = async (withMemory: boolean) => {
    if (withMemory) { await resetMemory(); return; }
    await api.clearChat();
    setTurns([]); setPending(null);
    const g = await api.graph('chat');
    setChatGraph(g);
    setLoadedFacts(g.counts.active);
    flash('Conversation cleared · memory kept');
  };

  // Chat demo: the scripted messages go through the real pipeline one by one, with a pause
  // after each answer so the graph animations (draw-in, retire, retrieved flow) can play
  const runChatDemo = async (script: string, fresh: boolean) => {
    if (chatDemo || pending) return;
    stopChatDemo.current = false;
    try {
      const sc = await api.script(script);
      if (fresh) await resetMemory(true);
      for (let i = 0; i < sc.turns.length; i++) {
        if (stopChatDemo.current) break;
        setChatDemo({ i, n: sc.turns.length, stopping: false });
        if (!(await send(sc.turns[i].message))) break;
        if (i < sc.turns.length - 1 && !stopChatDemo.current) await new Promise((r) => window.setTimeout(r, reduced ? 600 : 2600));
      }
    } catch (e) {
      flash(`Demo stopped: ${e instanceof Error ? e.message : e}`);
    } finally {
      setChatDemo(null);
    }
  };
  const stopDemo = () => { stopChatDemo.current = true; setChatDemo((d) => (d ? { ...d, stopping: true } : d)); };

  // ───────── derived ─────────
  const scope: Mode = mode;
  const payload = scope === 'chat' ? chatGraph : cmpGraph;
  const counts = payload?.counts ?? status?.counts ?? { entities: 0, active: 0, retired: 0 };
  const typeOf = useMemo(() => {
    const m = new Map((chatGraph?.nodes ?? []).map((n) => [n.label.toLowerCase(), n.type]));
    return (label: string) => m.get(label.toLowerCase());
  }, [chatGraph]);
  const ollama = !!status?.ollama && !offline;
  const model = status?.chat_model ?? 'llama3.2:3b';

  useLayoutEffect(() => {
    const b = segRef.current?.querySelector<HTMLButtonElement>(`[data-mode="${mode}"]`);
    if (b) setThumb({ x: b.offsetLeft - 3, w: b.offsetWidth });
  }, [mode]);

  // ───────── keys ─────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes((document.activeElement as HTMLElement)?.tagName);
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setCmdk((v) => !v); return; }
      if (e.key === 'Escape') {
        if (cmdk) setCmdk(false);
        else if (facts) setFacts(null);
        else graph.current?.select(null);
        return;
      }
      if (typing) return;
      if (e.key === 'f' || e.key === 'F') graph.current?.recenter();
      if (e.key === '/') { e.preventDefault(); input.current?.focus(); }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [cmdk, facts]);

  const commands: Command[] = [
    { group: 'Actions', label: mode === 'chat' ? 'Switch to Compare' : 'Switch to Chat', run: () => setMode(mode === 'chat' ? 'compare' : 'chat') },
    ...(mode === 'chat' ? [chatDemo ? { group: 'Actions', label: 'Stop demo', run: stopDemo } : { group: 'Actions', label: 'Run demo (keeps memory)', run: () => runChatDemo('demo', false) }] : []),
    ...(mode === 'compare' ? [demo.current.running ? { group: 'Actions', label: 'Stop demo', run: () => demo.current.stop() } : { group: 'Actions', label: 'Run demo', run: () => demo.current.run() }] : []),
    { group: 'Actions', label: graph.current?.view().showRetired ? 'Hide retired facts' : 'Show retired facts', run: () => graph.current?.toggleRetired() },
    { group: 'Actions', label: graph.current?.view().mode3d ? 'Switch to 2D' : 'Switch to 3D', run: () => graph.current?.setMode3d(!graph.current?.view().mode3d) },
    { group: 'Actions', label: 'Open facts', run: () => setFacts({}) },
    { group: 'Actions', label: 'Recentre graph', run: () => graph.current?.recenter(), keys: ['F'] },
    { group: 'Actions', label: 'Replay memory history', run: () => graph.current?.togglePlay() },
    { group: 'Actions', label: 'Clear conversation (keep memory)', run: () => { setMode('chat'); clearChat(false); } },
    { group: 'Actions', label: 'Reset memory…', run: () => { setMode('chat'); setFacts({ confirm: true }); } },
    ...entities.map((n) => ({ group: 'Go to entity', label: n.label, run: () => graph.current?.select(n.id), color: n.type == null ? HUB_UI : typeColor(n.type) })),
  ];

  return (
    <div className={`app mode-${mode} ${mode === 'compare' && !showCmpGraph ? 'nograph' : ''}`}>
      <header className="topbar">
        <div className="brand"><Mark /><span className="name">DMG</span><span className="sub">Dynamic Memory Graph</span></div>
        <div className="seg" role="group" aria-label="Mode" ref={segRef}>
          <span className="thumb" aria-hidden="true" style={{ width: thumb.w, transform: `translateX(${thumb.x}px)` }} />
          <button data-mode="chat" aria-pressed={mode === 'chat'} onClick={() => setMode('chat')}>Chat</button>
          <button data-mode="compare" aria-pressed={mode === 'compare'} onClick={() => setMode('compare')}>Compare</button>
        </div>
        <div className="right">
          <button className="pill counts" title="Open facts" onClick={() => setFacts({})}>
            <span className="seg-ent"><b><Num value={counts.entities} /></b> entities</span>
            <span className="sep">·</span>
            <span><b><Num value={counts.active} /></b> facts</span>
            <span className="sep">·</span>
            <span className="seg-ret"><b><Num value={counts.retired} /></b> retired</span>
          </button>
          <span className={`pill status ${ollama ? '' : 'down'}`} title={offline ? 'DMG server unreachable' : ollama ? 'Ollama on localhost:11434' : 'Ollama is offline: run `ollama serve`'}>
            <span className="dot" /><span className="mono">{offline ? 'server offline' : ollama ? model : 'Ollama offline'}</span>
          </span>
          <button className="pill kbtn" title="Command menu (⌘K)" aria-label="Open command menu" onClick={() => setCmdk(true)}><span className="mono">⌘K</span></button>
        </div>
      </header>

      <main className="main">
        {mode === 'chat' ? (
          <ChatView
            turns={turns} pending={pending} links={chatGraph?.links ?? []} ollama={ollama} loadedFacts={loadedFacts}
            model={model} reduced={reduced} highlight={(ids) => graph.current?.highlight(ids)} typeOf={typeOf}
            onSend={send} onRetry={() => pending && send(pending.message)} onDismissError={() => setPending(null)} inputRef={input}
            facts={(chatGraph?.counts.active ?? 0) + (chatGraph?.counts.retired ?? 0)} scripts={scripts} demo={chatDemo}
            onRunDemo={runChatDemo} onStopDemo={stopDemo} onClear={clearChat}
          />
        ) : (
          <CompareView
            models={status?.models ?? [model]} defaultModel={status?.default_model ?? model} ollama={ollama}
            contextWindow={status?.settings.context_window ?? 2048} showGraph={showCmpGraph} setShowGraph={setShowCmpGraph}
            onGraph={(g, fresh) => { setCmpGraph(g); if (fresh) setCmpEvent((k) => k + 1); }}
            highlight={(ids) => graph.current?.highlight(ids)} registerRun={(r) => { demo.current = r; }}
          />
        )}
        <GraphStage
          ref={graph} scope={scope} payload={payload} eventKey={scope === 'chat' ? chatEvent : cmpEvent} reduced={reduced}
          examples={status?.examples.chat ?? []} canSend={ollama && !pending} onExample={send} onEntities={setEntities}
        >
          {mode === 'compare' && cmpGraph && cmpGraph.links.length > 0 && (
            <div className="remembers glass depth-1">
              <h4>What DMG remembers</h4>
              <ul>
                {[...cmpGraph.links].sort((a, b) => (a.endedAt ? 1 : 0) - (b.endedAt ? 1 : 0)).map((l) => (
                  <li key={l.id} className={l.endedAt ? 'ret' : ''} onMouseEnter={() => graph.current?.highlight([l.id])} onMouseLeave={() => graph.current?.highlight(null)}>
                    <i style={{ background: typeColor(cmpGraph.nodes.find((n) => n.id === l.target)?.type) }} />{l.text}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </GraphStage>
      </main>

      {facts && (
        <FactsPanel
          scope={scope} version={(payload?.counts.active ?? 0) * 1000 + (payload?.counts.retired ?? 0) + chatEvent} startConfirm={facts.confirm}
          onClose={() => setFacts(null)} onFly={(id) => graph.current?.flyToLink(id)} onReset={() => resetMemory()}
        />
      )}
      {cmdk && <CommandMenu commands={commands} onClose={() => setCmdk(false)} />}
      {toast && <div className="toast glass">{toast}</div>}
      <div className="sr" aria-live="polite">{announce}</div>
    </div>
  );
}
