// Facts sheet: every stored fact grouped by type, lifetimes on the session timeline, search,
// click to fly the graph to that edge. "Reset memory" lives here behind a confirm step.
import { useEffect, useRef, useState } from 'react';
import { api, hhmm, type Fact } from '../api';
import { typeColor, typeName } from '../palette';
import { Icon } from './bits';

interface Props {
  scope: 'chat' | 'compare';
  version: number; // refetch when the graph changes
  onClose: () => void;
  onFly: (id: string) => void;
  onReset: () => Promise<void>;
  startConfirm?: boolean;
}

export default function FactsPanel({ scope, version, onClose, onFly, onReset, startConfirm }: Props) {
  const [facts, setFacts] = useState<Fact[] | null>(null);
  const [q, setQ] = useState('');
  const [confirm, setConfirm] = useState(!!startConfirm);
  const [busy, setBusy] = useState(false);
  const search = useRef<HTMLInputElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);

  useEffect(() => { api.facts(scope).then((r) => setFacts(r.facts)).catch(() => setFacts([])); }, [scope, version]);
  useEffect(() => { (confirm ? cancel : search).current?.focus(); }, [confirm]);

  const all = facts ?? [];
  const stamps = all.flatMap((f) => [f.created_at, f.ended_at]).filter(Boolean).sort() as string[];
  const t0 = stamps[0], t1 = stamps[stamps.length - 1];
  const frac = (s: string | null) => (s && t0 && t1 > t0 ? (Date.parse(s) - Date.parse(t0)) / (Date.parse(t1) - Date.parse(t0)) : 1);
  const groups = new Map<string, Fact[]>();
  for (const f of all) {
    if (q && !f.text.toLowerCase().includes(q.toLowerCase())) continue;
    const k = typeName(f.object_type);
    groups.set(k, [...(groups.get(k) ?? []), f]);
  }
  const active = all.filter((f) => !f.ended_at).length;

  return (
    <aside className="sheet glass" role="dialog" aria-label="Facts">
      <header>
        <h3>{scope === 'compare' ? 'What DMG remembers' : 'Facts'}</h3>
        <span className="mono">{active} active · {all.length - active} retired</span>
        <button className="x tb icon" style={{ marginLeft: 'auto' }} aria-label="Close facts" onClick={onClose}><Icon name="x" size={13} /></button>
      </header>
      <label className="search"><Icon name="search" size={14} /><input ref={search} placeholder="Search facts" aria-label="Search facts" value={q} onChange={(e) => setQ(e.target.value)} /></label>
      <div className="list">
        {facts && !groups.size && <div className="empty-note">{all.length ? 'No facts match.' : 'Nothing in memory yet.'}</div>}
        {[...groups].map(([t, rows]) => (
          <div key={t}>
            <div className="grp"><i style={{ background: typeColor(t) }} />{t} · {rows.length}</div>
            {rows.sort((a, b) => (a.ended_at ? 1 : 0) - (b.ended_at ? 1 : 0)).map((f) => {
              const a = frac(f.created_at) * 100, b = f.ended_at ? frac(f.ended_at) * 100 : 100;
              return (
                <div key={f.id} className={`fact ${f.ended_at ? 'retired' : ''}`} tabIndex={0} role="button" data-tilt
                  onClick={() => onFly(f.id)} onKeyDown={(e) => e.key === 'Enter' && onFly(f.id)}>
                  <div className="txt">{f.text}</div>
                  <div className="life" style={{ ['--c' as string]: typeColor(t) }}>
                    <b className={f.ended_at ? 'ret' : ''} style={{ left: `${a}%`, width: `${Math.max(1.5, b - a)}%` }} />
                    {f.ended_at && <span className="end" style={{ left: `calc(${b}% - 1px)` }} />}
                  </div>
                  <div className="when"><span>added {hhmm(f.created_at)}</span><span>{f.ended_at ? `retired ${hhmm(f.ended_at)}` : 'active'}</span></div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      {scope === 'chat' && (
        <footer>
          {!confirm ? (
            <button className="danger" onClick={() => setConfirm(true)} disabled={!all.length}>Reset memory</button>
          ) : (
            <div className="confirm">
              <p>Delete all {all.length} facts? This can't be undone.</p>
              <div className="row">
                <button className="ghostbtn" ref={cancel} onClick={() => setConfirm(false)}>Cancel</button>
                <button className="danger solid" disabled={busy} onClick={async () => { setBusy(true); try { await onReset(); } finally { setBusy(false); setConfirm(false); } }}>Delete</button>
              </div>
            </div>
          )}
        </footer>
      )}
    </aside>
  );
}
