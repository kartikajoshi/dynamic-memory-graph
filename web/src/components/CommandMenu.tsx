// ⌘K: opens instantly (keyboard-initiated actions are never animated).
import { useEffect, useRef, useState } from 'react';

export interface Command { group: string; label: string; run: () => void; keys?: string[]; color?: string }

export default function CommandMenu({ commands, onClose }: { commands: Command[]; onClose: () => void }) {
  const [q, setQ] = useState('');
  const [sel, setSel] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => input.current?.focus(), []);
  const shown = commands.filter((c) => c.label.toLowerCase().includes(q.toLowerCase()));
  const run = (i: number) => { const c = shown[i]; onClose(); c?.run(); };
  let lastGroup = '';
  return (
    <>
      <div className="cmdk-backdrop" onClick={onClose} />
      <div className="cmdk glass" role="dialog" aria-label="Command menu">
        <input
          ref={input} placeholder="Type a command or an entity" aria-label="Command" value={q}
          role="combobox" aria-expanded="true" aria-controls="cmdk-list" aria-activedescendant={shown[sel] ? `cmd-${sel}` : undefined}
          onChange={(e) => { setQ(e.target.value); setSel(0); }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') { setSel(Math.min(shown.length - 1, sel + 1)); e.preventDefault(); }
            if (e.key === 'ArrowUp') { setSel(Math.max(0, sel - 1)); e.preventDefault(); }
            if (e.key === 'Enter') { run(sel); e.preventDefault(); }
            if (e.key === 'Escape') { onClose(); e.preventDefault(); }
          }}
        />
        <ul role="listbox" id="cmdk-list">
          {shown.map((c, i) => {
            const head = c.group !== lastGroup ? <li className="h" key={`h-${c.group}`} role="presentation">{c.group}</li> : null;
            lastGroup = c.group;
            return [head, (
              <li key={`${c.group}-${c.label}`} id={`cmd-${i}`} className="i" role="option" aria-selected={i === sel} onClick={() => run(i)} onMouseMove={() => sel !== i && setSel(i)}>
                {c.color && <i style={{ background: c.color }} />}{c.label}
                <span className="k">{c.keys?.map((k) => <kbd key={k}>{k}</kbd>)}</span>
              </li>
            )];
          })}
          {!shown.length && <li className="h">No matches</li>}
        </ul>
      </div>
    </>
  );
}
