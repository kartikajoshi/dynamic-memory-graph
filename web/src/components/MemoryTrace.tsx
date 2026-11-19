// Memory trace: Rewrite → Extract → Update → Retrieve → Answer under each answer.
// While a turn runs, chips light up from the server's live stage events (real progress, not faked);
// once it lands, every chip shows its real duration, or "—" when the stage was skipped.
import { useState, type ReactNode } from 'react';
import { fmtS, plural, type GLink, type StageName, type Turn } from '../api';
import { HUB_UI, typeColor } from '../palette';

export type LiveStages = Partial<Record<StageName, { status: 'start' | 'end'; seconds?: number }>>;
const STAGES: [StageName, string][] = [['rewrite', 'Rewrite'], ['extract', 'Extract'], ['update', 'Update'], ['retrieve', 'Retrieve'], ['answer', 'Answer']];

interface Ctx {
  highlight: (ids: string[] | null) => void;
  linkText: (id: string) => string;
  typeOf: (label: string) => string | null | undefined;
  model: string;
  reduced: boolean;
}

function opCounts(turn: Turn) {
  const ops = { ADD: 0, UPDATE: 0, DELETE: 0, SKIP: 0 };
  for (const a of turn.result.actions || []) ops[a.op] = (ops[a.op] || 0) + 1;
  return ops;
}
const opsText = (o: ReturnType<typeof opCounts>) => [o.ADD && `+${o.ADD}`, o.UPDATE && `~${o.UPDATE}`, o.DELETE && `−${o.DELETE}`].filter(Boolean).join(' ');
const total = (turn: Turn) => {
  const t = turn.result.timings;
  return t.total ?? t.rewrite + t.extract + t.update + t.retrieve + t.answer;
};

// a stage was skipped when it took < 5 ms and produced nothing
function stageValue(key: StageName, turn: Turn): ReactNode | null {
  const r = turn.result, t = r.timings;
  const dur = (s: number, output: boolean) => (s < 0.005 && !output ? null : fmtS(s) ?? '<0.01s');
  if (key === 'rewrite') return dur(t.rewrite, r.rewritten_query !== r.user_message);
  if (key === 'extract') return dur(t.extract, r.triples.length > 0);
  if (key === 'update') {
    const o = opCounts(turn);
    if (!r.actions.length) return null;
    return <span className="ops">{o.ADD ? <span>+{o.ADD}</span> : null}{o.UPDATE ? <span>~{o.UPDATE}</span> : null}{o.DELETE ? <span>−{o.DELETE}</span> : null}{o.SKIP ? <span className="skipn">{o.SKIP} skip</span> : null}</span>;
  }
  return fmtS(key === 'retrieve' ? t.retrieve : t.answer) ?? '<0.01s';
}

export function RunningRail({ live }: { live: LiveStages }) {
  return (
    <div className="trace running">
      <div className="rail live" aria-label="Memory trace, running">
        {STAGES.map(([key, k]) => {
          const s = live[key];
          const state = !s ? 'wait' : s.status === 'start' ? 'active' : 'done';
          return (
            <span key={key} className={`stage s-${state}`}>
              <span className="k">{k}</span>
              <span className="v">{state === 'done' ? fmtS(s!.seconds) ?? '—' : state === 'active' ? '…' : ''}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

export function TraceRail({ turn, latest, ctx }: { turn: Turn; latest: boolean; ctx: Ctx }) {
  const [open, setOpen] = useState<StageName | null>(null);
  const [expanded, setExpanded] = useState(false);
  const ops = opCounts(turn);
  const nRetrieved = turn.retrieval.length;
  const toggle = (key: StageName) => { ctx.highlight(null); setOpen(open === key ? null : key); };
  return (
    <div className={`trace ${expanded ? 'open' : ''}`}>
      <button className="tsum" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
        <span className="f">{[`${total(turn).toFixed(1)} s`, opsText(ops), `${nRetrieved} retrieved`].filter(Boolean).join(' · ')}</span>
        <span className="c">5 steps · {total(turn).toFixed(1)} s</span>
        <span className="chev" aria-hidden="true">▾</span>
      </button>
      <div className={`rail ${latest ? 'lift' : ''}`} role="tablist" aria-label="Memory trace">
        {STAGES.map(([key, k], i) => {
          const v = stageValue(key, turn);
          return (
            <button
              key={key} className="stage" role="tab" aria-expanded={open === key} aria-label={`${k}${v ? '' : ', skipped'}`}
              style={{ ['--i' as string]: i, ['--hold' as string]: `${Math.min(900, 120 + (turn.result.timings[key] || 0) * 220)}ms` }}
              onClick={() => toggle(key)}
              onKeyDown={(e) => {
                if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
                const btns = [...(e.currentTarget.parentElement?.children ?? [])] as HTMLElement[];
                btns[(i + (e.key === 'ArrowRight' ? 1 : btns.length - 1)) % btns.length]?.focus();
                e.preventDefault();
              }}
            >
              <span className="k">{k}</span>
              <span className={`v ${v ? '' : 'skip'}`}>{v ?? '—'}</span>
            </button>
          );
        })}
      </div>
      {open && <div className="tpanel"><StagePanel stage={open} turn={turn} ctx={ctx} /></div>}
    </div>
  );
}

function wordDiff(a: string, b: string) {
  const A = a.split(' '), B = b.split(' ');
  const setA = new Set(A), setB = new Set(B);
  return (
    <div className="diff">
      <div className="row"><span className="lab">before</span><span>{A.map((w, i) => <span key={i}>{setB.has(w) ? w : <del>{w}</del>}{i < A.length - 1 ? ' ' : ''}</span>)}</span></div>
      <div className="row"><span className="lab">after</span><span>{B.map((w, i) => <span key={i}>{setA.has(w) ? w : <ins>{w}</ins>}{i < B.length - 1 ? ' ' : ''}</span>)}</span></div>
    </div>
  );
}

const Ent = ({ name, type }: { name: string; type?: string | null }) => (
  <span className="ent"><i style={{ background: name.toLowerCase() === 'user' ? HUB_UI : typeColor(type) }} />{name}</span>
);

function StagePanel({ stage, turn, ctx }: { stage: StageName; turn: Turn; ctx: Ctx }) {
  const r = turn.result, t = r.timings, ch = turn.changes;
  if (stage === 'rewrite') return (
    <div className="panel" role="tabpanel">
      <h4>Rewrite<span className="mono">{fmtS(t.rewrite) || 'skipped'}</span></h4>
      {r.rewritten_query === r.user_message ? <div>Nothing to resolve: no pronouns or vague references.</div> : wordDiff(r.user_message, r.rewritten_query)}
    </div>
  );
  if (stage === 'extract') return (
    <div className="panel" role="tabpanel">
      <h4>Extract · {plural(r.triples.length, 'triple')}<span className="mono">{fmtS(t.extract) || 'skipped'}</span></h4>
      {!r.triples.length ? <div>No facts in this message.</div> : (
        <div className="triples">
          {r.triples.map((x, i) => (
            <div className="triple" key={i}>
              <Ent name={x.subject} type={ctx.typeOf(x.subject)} /><span className="pred">—{x.predicate}→</span><Ent name={x.object} type={x.object_type} />
              <span className="faint">{x.object_type}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
  if (stage === 'update') {
    const ids = ch ? [...ch.new_edges, ...ch.updated_edges, ...ch.retired_edges] : [];
    return (
      <div className="panel" role="tabpanel" onMouseEnter={() => ids.length && ctx.highlight(ids)} onMouseLeave={() => ctx.highlight(null)}>
        <h4>Update graph<span className="mono">{fmtS(t.update) || '<0.01s'}</span></h4>
        {!r.actions.length && <div>No memory changes this turn.</div>}
        <div className="triples">
          {r.actions.map((a, i) => (
            <div className="triple" key={i}><span className={`tag ${a.op.toLowerCase()}`}>{a.op}</span><span>{a.subject} {a.predicate.replace(/_/g, ' ')} {a.object}</span></div>
          ))}
          {ch?.retired_edges.map((id) => (
            <div className="triple muted" key={id}><span className="tag ended">ENDED</span><s>{ctx.linkText(id)}</s><span className="faint">kept as history</span></div>
          ))}
        </div>
      </div>
    );
  }
  if (stage === 'retrieve') {
    const rows = [...turn.retrieval].sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99));
    return (
      <div className="panel" role="tabpanel">
        <h4>Retrieved into the prompt · {rows.length}<span className="mono">{fmtS(t.retrieve)} · BM25 + dense + graph → RRF</span></h4>
        <ol className="ranked">
          {rows.map((f) => (
            <li key={f.id} onMouseEnter={() => ctx.highlight([f.id])} onMouseLeave={() => ctx.highlight(null)}>
              <span className="r">{f.rank ?? '·'}</span><span>{f.text}</span>
              <span className="mono faint small">{f.linked ? 'linked' : ''}</span>
            </li>
          ))}
        </ol>
        <div className="foot">{turn.rank_basis === 'retriever' ? 'rank order from fusion (no scores)' : 'prompt order'} · {r.usage.prompt_tokens} prompt tokens</div>
      </div>
    );
  }
  return (
    <div className="panel" role="tabpanel">
      <h4>Answer<span className="mono">{fmtS(t.answer)}</span></h4>
      <div className="mono" style={{ color: 'var(--text)' }}>{r.usage.prompt_tokens} tok in · {r.usage.completion_tokens} out · {ctx.model}</div>
      <div className="foot">turn total {fmtS(total(turn))}</div>
    </div>
  );
}

// "+2 facts · 1 updated · 5 retrieved" under the latest answer; hover lights the edges
export function ChangeLine({ turn, links, highlight }: { turn: Turn; links: GLink[]; highlight: (ids: string[] | null) => void }) {
  const c = turn.changes;
  const liveIds = new Set(links.filter((l) => !l.endedAt).map((l) => l.id));
  const retrieved = turn.retrieval.map((f) => f.id).filter((id) => liveIds.has(id));
  const parts: [string, string, string[]][] = [];
  if (c?.new_edges.length) parts.push(['c-new', `+${plural(c.new_edges.length, 'fact')}`, c.new_edges]);
  if (c?.updated_edges.length) parts.push(['c-upd', `${c.updated_edges.length} updated`, c.updated_edges]);
  if (c?.retired_edges.length) parts.push(['c-ret', `${c.retired_edges.length} retired`, c.retired_edges]);
  if (retrieved.length) parts.push(['c-used', `${retrieved.length} retrieved`, retrieved]);
  return (
    <div className="changes">
      {parts.length ? parts.map(([cls, text, ids]) => (
        <button key={cls} className={cls} onMouseEnter={() => highlight(ids)} onMouseLeave={() => highlight(null)} onFocus={() => highlight(ids)} onBlur={() => highlight(null)}>{text}</button>
      )) : 'No memory changes'}
    </div>
  );
}
