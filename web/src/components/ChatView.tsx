// Conversation column: thread with the memory trace under each answer, and the composer.
import { useEffect, useRef, useState } from 'react';
import type { GLink, Script, Turn } from '../api';
import { ChangeLine, RunningRail, TraceRail, type LiveStages } from './MemoryTrace';
import { Icon } from './bits';

export interface Pending { message: string; live: LiveStages; error?: string }

interface Props {
  turns: Turn[];
  pending: Pending | null;
  links: GLink[];
  ollama: boolean;
  loadedFacts: number;
  model: string;
  reduced: boolean;
  highlight: (ids: string[] | null) => void;
  typeOf: (label: string) => string | null | undefined;
  onSend: (text: string) => void;
  onRetry: () => void;
  onDismissError: () => void;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  facts: number;
  scripts: Script[];
  demo: { i: number; n: number; stopping: boolean } | null;
  onRunDemo: (script: string, fresh: boolean) => void;
  onStopDemo: () => void;
  onClear: (withMemory: boolean) => void;
}

// top of the conversation: play a scripted demo, or reset
function ChatBar(p: Props) {
  const [open, setOpen] = useState<'demo' | 'reset' | null>(null);
  const [script, setScript] = useState('demo');
  const [fresh, setFresh] = useState(true);
  const bar = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => { if (!bar.current?.contains(e.target as Node)) setOpen(null); };
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(null); };
    document.addEventListener('mousedown', away);
    document.addEventListener('keydown', esc);
    return () => { document.removeEventListener('mousedown', away); document.removeEventListener('keydown', esc); };
  }, [open]);
  const busy = !!p.pending && !p.pending.error;
  const meta = p.scripts.find((s) => s.name === script);
  return (
    <div className="chatbar" ref={bar}>
      <span className="ctitle">Conversation{p.turns.length > 0 && <span className="mono faint"> · {p.turns.length} turn{p.turns.length === 1 ? '' : 's'}</span>}</span>
      {p.demo ? (
        <button className="primary" onClick={p.onStopDemo} disabled={p.demo.stopping}>
          <Icon name="stop" size={12} />{p.demo.stopping ? 'Stopping…' : 'Stop'} · <span className="mono">{p.demo.i + 1}/{p.demo.n}</span>
        </button>
      ) : (
        <div className="pop-wrap">
          <button className="primary" aria-expanded={open === 'demo'} disabled={!p.ollama || busy || !p.scripts.length} onClick={() => setOpen(open === 'demo' ? null : 'demo')}>
            <Icon name="play" size={12} />Run demo
          </button>
          {open === 'demo' && (
            <div className="popover glass">
              <label className="field"><span>Script</span>
                <select value={script} onChange={(e) => setScript(e.target.value)}>
                  {p.scripts.map((s) => <option key={s.name} value={s.name}>{s.label}</option>)}
                </select>
              </label>
              <label className="chk"><input type="checkbox" checked={fresh} onChange={(e) => setFresh(e.target.checked)} />
                Start from empty memory{fresh && p.facts > 0 ? ` (deletes ${p.facts} facts)` : ''}</label>
              <div className="pnote">Plays {meta ? `${meta.turns} messages` : 'the script'} one by one through the real pipeline, with a pause after each answer so the graph can animate.</div>
              <button className="primary" onClick={() => { setOpen(null); p.onRunDemo(script, fresh); }}><Icon name="play" size={12} />Start</button>
            </div>
          )}
        </div>
      )}
      <div className="pop-wrap">
        <button className="ghostbtn" aria-expanded={open === 'reset'} disabled={!!p.demo || busy} onClick={() => setOpen(open === 'reset' ? null : 'reset')}>
          <Icon name="retry" size={13} />Reset
        </button>
        {open === 'reset' && (
          <div className="popover glass">
            <button className="pitem" onClick={() => { setOpen(null); p.onClear(false); }} disabled={!p.turns.length && !p.pending}>
              <b>Clear conversation</b><span>Keeps memory: {p.facts} fact{p.facts === 1 ? '' : 's'} stay in the graph.</span>
            </button>
            <button className="pitem danger-item" onClick={() => { setOpen(null); p.onClear(true); }}>
              <b>Clear conversation and memory</b><span>Deletes all {p.facts} facts. This can't be undone.</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

const EXPLAIN: [string, React.ReactNode][] = [
  ['Rewrite', <>resolve pronouns: <span className="mono">its → Metformin’s</span></>],
  ['Extract', <>pull facts: <span className="mono">(User, takes, Metformin)</span></>],
  ['Update', 'add, update, or retire. Never erase.'],
  ['Retrieve', 'pick the top 5 facts for this message'],
  ['Answer', 'from those facts, not the whole chat'],
];

export default function ChatView(p: Props) {
  const thread = useRef<HTMLDivElement>(null);
  const [text, setText] = useState('');
  const linkText = (id: string) => p.links.find((l) => l.id === id)?.text ?? id.split('|').slice(0, 2).join(' → ');
  const ctx = { highlight: p.highlight, linkText, typeOf: p.typeOf, model: p.model, reduced: p.reduced };
  const busy = (!!p.pending && !p.pending.error) || !!p.demo;

  useEffect(() => {
    const el = thread.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [p.turns.length, p.pending?.message, p.pending?.error]);

  useEffect(() => {
    const ta = p.inputRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(108, ta.scrollHeight)}px`;
  }, [text, p.inputRef]);

  const send = () => {
    const t = text.trim();
    if (!t || busy || !p.ollama) return;
    setText('');
    p.onSend(t);
  };

  return (
    <section className="chat" aria-label="Conversation">
      <ChatBar {...p} />
      <div className="thread" ref={thread}>
        {!p.turns.length && !p.pending && (
          <div className="thread-empty">
            <h2>Long-term memory for a 3B model</h2>
            <p>Llama 3.2 3B forgets after a few turns. DMG keeps what you tell it in a graph. Every message goes through five steps:</p>
            <div className="explain">
              {EXPLAIN.map(([k, d]) => (
                <div className="row" key={k}><span className="stage"><span className="k" style={{ color: 'var(--text)' }}>{k}</span></span><span className="d">{d}</span></div>
              ))}
            </div>
          </div>
        )}
        {p.turns.map((turn, i) => {
          const latest = i === p.turns.length - 1 && !p.pending;
          return (
            <div key={turn.at}>
              <div className="msg user"><div className="bubble">{turn.user}</div></div>
              <div className={`msg bot ${latest ? 'latest' : ''}`}>
                <div className="answer">{turn.answer}</div>
                <TraceRail turn={turn} latest={latest} ctx={ctx} />
                {latest && <ChangeLine turn={turn} links={p.links} highlight={p.highlight} />}
              </div>
            </div>
          );
        })}
        {p.pending && (
          <div>
            <div className="msg user"><div className="bubble">{p.pending.message}</div></div>
            <div className={`msg bot latest ${p.pending.error ? 'error' : ''}`}>
              {p.pending.error ? (
                <>
                  <div className="answer">{p.pending.error}</div>
                  <div className="erow">
                    <button className="ghostbtn" onClick={p.onRetry}><Icon name="retry" size={13} /> Retry</button>
                    <button className="ghostbtn" onClick={p.onDismissError}>Dismiss</button>
                  </div>
                </>
              ) : (
                <>
                  <div className="answer pending">Thinking with memory…</div>
                  <RunningRail live={p.pending.live} />
                </>
              )}
            </div>
          </div>
        )}
      </div>
      <div className="composer">
        {!p.ollama ? (
          <div className="hint offline"><span className="dotv bad" />Ollama is offline. Start it with <code className="mono">ollama serve</code>, then try again.</div>
        ) : p.loadedFacts > 0 && !p.turns.length && !p.pending ? (
          <div className="hint"><span className="dotv" />Remembers {p.loadedFacts} facts from before</div>
        ) : null}
        <form className="inputwrap" onSubmit={(e) => { e.preventDefault(); send(); }}>
          <label className="sr" htmlFor="input">Message</label>
          <textarea
            id="input" ref={p.inputRef} rows={1} value={text} autoComplete="off" disabled={!p.ollama || !!p.demo}
            placeholder={!p.ollama ? 'Waiting for Ollama…' : p.demo ? `Demo running · ${p.demo.i + 1}/${p.demo.n}` : 'Tell DMG something, or ask about what it remembers'}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
              if (e.key === 'ArrowUp' && !text && p.turns.length) { setText(p.turns[p.turns.length - 1].user); e.preventDefault(); }
            }}
          />
          <button className={`send ${text.trim() && !busy ? 'ready' : ''}`} type="submit" aria-label="Send" disabled={busy || !p.ollama}><Icon name="send" size={14} /></button>
        </form>
        <div className="under">
          <span><kbd>⏎</kbd> send  <kbd>⇧⏎</kbd> new line  <kbd>↑</kbd> edit last</span>
          <span className="mono">{p.model}</span>
        </div>
      </div>
    </section>
  );
}
