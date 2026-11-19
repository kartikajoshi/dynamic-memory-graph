// Compare: the same messages go to two memory-less baselines and to DMG, side by side.
// Scores come only from the server (check_turn / check_graph); the client loops through the
// scripted turns so a run can stop between them.
import { useEffect, useRef, useState } from 'react';
import { api, plural, type Checks, type Col, type CompareRow, type GraphPayload, type Script, type Totals } from '../api';
import { Icon, Num } from './bits';

const COLS: [Col, string][] = [['recent', 'Last 4 turns'], ['full', 'Full history'], ['dmg', 'DMG']];

interface Props {
  models: string[];
  defaultModel: string;
  ollama: boolean;
  contextWindow: number;
  showGraph: boolean;
  setShowGraph: (on: boolean) => void;
  onGraph: (p: GraphPayload, fresh: boolean) => void;
  highlight: (ids: string[] | null) => void;
  registerRun: (api: { run: () => void; stop: () => void; running: boolean }) => void;
}

export default function CompareView(p: Props) {
  const [rows, setRows] = useState<CompareRow[]>([]);
  const [totals, setTotals] = useState<Totals>({});
  const [checks, setChecks] = useState<Checks | null>(null);
  const [scripts, setScripts] = useState<Script[]>([]);
  const [script, setScript] = useState('demo');
  const [model, setModel] = useState(p.defaultModel);
  const [perCol, setPerCol] = useState<Partial<Record<Col, string>>>({});
  const [perColOn, setPerColOn] = useState(false);
  const [parallel, setParallel] = useState(true);
  const [settings, setSettings] = useState(false);
  const [run, setRun] = useState<{ i: number; n: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [text, setText] = useState('');
  const stop = useRef(false);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.scripts().then((r) => setScripts(r.scripts)).catch(() => {});
    api.compareState().then((s) => { setRows(s.rows); setTotals(s.totals); setChecks(s.checks); if (s.script) setScript(s.script); }).catch(() => {});
    api.graph('compare').then((g) => p.onGraph(g, false)).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => { const el = list.current; if (el) el.scrollTop = el.scrollHeight; }, [rows.length, busy]);

  const models = () => (perColOn ? { recent: perCol.recent || model, full: perCol.full || model, dmg: perCol.dmg || model } : undefined);

  async function turn(body: Parameters<typeof api.compareTurn>[0]) {
    setBusy(true);
    setError(null);
    try {
      const r = await api.compareTurn({ ...body, model, models: models(), parallel });
      setRows((rs) => [...rs, r.row]);
      setTotals(r.totals);
      p.onGraph(r.graph, true);
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function runDemo() {
    const meta = scripts.find((s) => s.name === script);
    if (!meta || run) return;
    stop.current = false;
    await api.compareReset();
    setRows([]); setTotals({}); setChecks(null);
    api.graph('compare').then((g) => p.onGraph(g, false)).catch(() => {});
    for (let i = 0; i < meta.turns; i++) {
      if (stop.current) break;
      setRun({ i, n: meta.turns });
      if (!(await turn({ script, turn_index: i }))) break;
      if (i === meta.turns - 1) setChecks(await api.compareChecks(script).catch(() => null));
    }
    setRun(null);
  }
  const stopDemo = () => { stop.current = true; };
  const reset = async () => { await api.compareReset(); setRows([]); setTotals({}); setChecks(null); setSettings(false); api.graph('compare').then((g) => p.onGraph(g, false)); };

  useEffect(() => { p.registerRun({ run: runDemo, stop: stopDemo, running: !!run }); });

  const send = () => {
    const t = text.trim();
    if (!t || busy || run) return;
    setText('');
    turn({ message: t });
  };

  const chatModels = p.models.filter((m) => !/embed/i.test(m));
  const latestStale = (cid: Col) => [...rows].reverse().find((r) => r.stale[cid]?.length);

  return (
    <section className="compare" aria-label="Compare">
      <div className="cbar">
        <label className="field"><span>Model</span>
          <select value={model} onChange={(e) => setModel(e.target.value)} disabled={!!run}>
            {chatModels.map((m) => <option key={m}>{m}</option>)}
          </select>
        </label>
        <label className="field"><span>Script</span>
          <select value={script} onChange={(e) => setScript(e.target.value)} disabled={!!run}>
            {scripts.map((s) => <option key={s.name} value={s.name}>{s.label}</option>)}
          </select>
        </label>
        {run ? (
          <button className="primary" onClick={stopDemo}><Icon name="stop" size={12} />Stop · <span className="mono">{run.i + 1}/{run.n}</span></button>
        ) : (
          <button className="primary" onClick={runDemo} disabled={!p.ollama || busy || !scripts.length}><Icon name="play" size={12} />Run demo</button>
        )}
        <button className="ghostbtn" onClick={reset} disabled={!!run || busy || !rows.length} title="Clear all three conversations and DMG's compare memory">
          <Icon name="retry" size={13} />Reset
        </button>
        <div className="pop-wrap">
          <button className="tb icon" aria-label="Compare settings" aria-expanded={settings} onClick={() => setSettings(!settings)}><Icon name="gear" /></button>
          {settings && (
            <div className="popover glass">
              <label className="chk"><input type="checkbox" checked={perColOn} onChange={(e) => setPerColOn(e.target.checked)} />Model per column</label>
              {perColOn && COLS.map(([cid, name]) => (
                <label className="field sub" key={cid}><span>{name}</span>
                  <select value={perCol[cid] || model} onChange={(e) => setPerCol({ ...perCol, [cid]: e.target.value })}>
                    {chatModels.map((m) => <option key={m}>{m}</option>)}
                  </select>
                </label>
              ))}
              <label className="chk"><input type="checkbox" checked={parallel} onChange={(e) => setParallel(e.target.checked)} />Run columns in parallel</label>
              <label className="chk"><input type="checkbox" checked={p.showGraph} onChange={(e) => p.setShowGraph(e.target.checked)} />Show DMG graph</label>
            </div>
          )}
        </div>
      </div>

      <div className="board">
        {COLS.map(([cid, name]) => {
          const t = totals[cid] ?? { passed: 0, scored: 0 };
          const lead = cid === 'dmg';
          const stale = latestStale(cid);
          return (
            <div className={`col-score ${lead ? 'lead' : ''}`} key={cid} data-tilt>
              <div className="cname">{name}</div>
              <div className="big mono"><Num value={t.passed} /><span className="of">/{t.scored}</span></div>
              <Ribbon rows={rows} cid={cid} window={p.contextWindow} />
              <div className="strip" aria-label={`${name}: per-turn results`}>
                {rows.filter((r) => r.turn_index != null).map((r) => {
                  const s = r.scores?.[cid];
                  return <span key={r.index} className={s ? (s.ok ? 'ok' : 'bad') : 'na'} title={`turn ${(r.turn_index ?? 0) + 1}${s ? (s.ok ? ' · pass' : ` · fail: ${s.reasons.join(', ')}`) : ' · not scored'}`}>{s ? (s.ok ? '●' : '✕') : '○'}</span>;
                })}
              </div>
              {stale && cid !== 'dmg' && (
                <div className="stale">said “{stale.stale[cid]![0].sentence.slice(0, 70)}” · {stale.stale[cid]![0].turns_ago != null ? `retired ${plural(stale.stale[cid]![0].turns_ago!, 'turn')} earlier` : 'a retired fact'}</div>
              )}
            </div>
          );
        })}
      </div>

      <div className="ctranscript" ref={list}>
        {!rows.length && !busy && (
          <div className="cempty">
            <h2>Same messages, three memories</h2>
            <p>Run the scored demo, or send your own messages. Each one goes to a model with the last 4 turns, a model with the full history, and DMG with its memory graph.</p>
          </div>
        )}
        {rows.map((r) => (
          <div className="crow" key={r.index}>
            <div className="cuser"><span className="mono n">{(r.turn_index ?? r.index) + 1}</span>{r.user}</div>
            <div className="ccards">
              {COLS.map(([cid, name]) => {
                const c = r.cols[cid];
                const s = r.scores?.[cid];
                const hits = r.stale[cid] ?? [];
                const ids = cid === 'dmg' ? (r.retrieval ?? []).map((f) => f.id) : [];
                return (
                  <article className={`ccard ${cid}`} key={cid} data-tilt>
                    <header><span className="cn">{name}</span>{s && <span className={`pf ${s.ok ? 'ok' : 'bad'}`}><i />{s.ok ? 'pass' : 'fail'}</span>}</header>
                    <div className="ans">{c.answer}</div>
                    {s && !s.ok && <div className="reasons">{s.reasons.join(' · ')}</div>}
                    {cid !== 'dmg' && hits.map((h) => <div className="stalehit" key={h.edge}>states a retired fact: {h.fact}{h.turns_ago != null ? ` · retired ${plural(h.turns_ago, 'turn')} earlier` : ''}</div>)}
                    <footer className="mono">
                      {[c.latency != null && `${c.latency.toFixed(1)} s`, c.prompt_tokens != null && `${c.prompt_tokens} in`, c.completion_tokens != null && `${c.completion_tokens} out`, c.note].filter(Boolean).join(' · ')}
                      {ids.length > 0 && <button className="used" onMouseEnter={() => p.highlight(ids)} onMouseLeave={() => p.highlight(null)} onFocus={() => p.highlight(ids)} onBlur={() => p.highlight(null)}>{ids.length} retrieved</button>}
                    </footer>
                  </article>
                );
              })}
            </div>
          </div>
        ))}
        {busy && <div className="crow"><div className="ccards">{COLS.map(([cid]) => <article className="ccard skel" key={cid}><div className="shim" /></article>)}</div></div>}
        {error && <div className="cerr">{error}</div>}
        {checks && (
          <div className="results glass">
            <h3>Results · {checks.script === 'long' ? 'Long conversation' : 'Demo'}</h3>
            <div className="rscores">{COLS.map(([cid, name]) => <div key={cid}><span>{name}</span><b className="mono">{totals[cid]?.passed ?? 0}/{totals[cid]?.scored ?? 0}</b></div>)}</div>
            <h4>DMG graph checks · {checks.results.filter((c) => c.ok).length}/{checks.results.length}</h4>
            <ul className="checks">{checks.results.map((c) => <li key={c.check} className={c.ok ? 'ok' : 'bad'}><span className="mono">{c.ok ? '✓' : '✕'}</span>{c.check}</li>)}</ul>
          </div>
        )}
      </div>

      <form className="composer ccomp" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <div className="inputwrap">
          <input aria-label="Message to all three" placeholder={run ? 'Demo running…' : 'Send one message to all three'} value={text} onChange={(e) => setText(e.target.value)} disabled={!p.ollama || !!run} />
          <button className={`send ${text.trim() && !busy ? 'ready' : ''}`} type="submit" aria-label="Send" disabled={busy || !!run}><Icon name="send" size={14} /></button>
        </div>
      </form>
    </section>
  );
}

// prompt tokens per turn as an extruded ribbon (SVG with a depth face), context window dashed,
// turns where this column stated a retired fact marked with a tick
function Ribbon({ rows, cid, window: win }: { rows: CompareRow[]; cid: Col; window: number }) {
  const W = 220, H = 46, D = 4;
  const vals = rows.map((r) => r.cols[cid]?.prompt_tokens ?? 0);
  if (!vals.length) return <div className="ribbon empty mono">prompt tokens per turn</div>;
  const max = Math.max(win, ...vals);
  const x = (i: number) => (vals.length === 1 ? W / 2 : (i / (vals.length - 1)) * (W - D));
  const y = (v: number) => H - 4 - (v / max) * (H - 10);
  const top = vals.map((v, i) => `${x(i)},${y(v)}`).join(' ');
  const area = `0,${H} ${top} ${x(vals.length - 1)},${H}`;
  const back = vals.map((v, i) => `${x(i) + D},${y(v) - D}`).join(' ');
  const last = vals[vals.length - 1];
  return (
    <div className="ribbon">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
        <line x1="0" x2={W} y1={y(win)} y2={y(win)} className="win" />
        <polyline points={back} className="back" />
        {vals.map((v, i) => i < vals.length - 1 && <polygon key={i} className="side" points={`${x(i)},${y(v)} ${x(i + 1)},${y(vals[i + 1])} ${x(i + 1) + D},${y(vals[i + 1]) - D} ${x(i) + D},${y(v) - D}`} />)}
        <polygon points={area} className="area" />
        <polyline points={top} className="top" />
        {rows.map((r, i) => (r.stale[cid]?.length ? <line key={i} x1={x(i)} x2={x(i)} y1={y(vals[i]) - 8} y2={y(vals[i]) - 2} className="stalemark" /> : null))}
      </svg>
      <span className="mono">{last} prompt tok</span>
    </div>
  );
}
