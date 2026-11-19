// Small shared pieces: icons, the digit roll, the logo mark.
import type { ReactNode } from 'react';

const P: Record<string, ReactNode> = {
  target: <><circle cx="8" cy="8" r="5.2" stroke="currentColor" strokeWidth="1.4" /><circle cx="8" cy="8" r="1.6" fill="currentColor" /></>,
  play: <path d="M5 3.5v9l7.5-4.5z" fill="currentColor" />,
  pause: <><rect x="4" y="3.5" width="2.6" height="9" rx="0.8" fill="currentColor" /><rect x="9.4" y="3.5" width="2.6" height="9" rx="0.8" fill="currentColor" /></>,
  stop: <rect x="4" y="4" width="8" height="8" rx="1.5" fill="currentColor" />,
  x: <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />,
  search: <><circle cx="7" cy="7" r="4.5" stroke="currentColor" strokeWidth="1.4" /><path d="M10.5 10.5L14 14" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></>,
  send: <path d="M8 13V3M8 3L3.5 7.5M8 3l4.5 4.5" stroke="currentColor" strokeWidth="1.7" fill="none" strokeLinecap="round" strokeLinejoin="round" />,
  gear: <><circle cx="8" cy="8" r="2.2" stroke="currentColor" strokeWidth="1.4" /><path d="M8 1.8v1.8M8 12.4v1.8M1.8 8h1.8M12.4 8h1.8M3.6 3.6l1.3 1.3M11.1 11.1l1.3 1.3M3.6 12.4l1.3-1.3M11.1 4.9l1.3-1.3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></>,
  retry: <path d="M12.5 6.5A4.8 4.8 0 1 0 13 9.5M12.8 2.8v3.8H9" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />,
  chev: <path d="M4.5 6.5L8 10l3.5-3.5" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />,
};

export function Icon({ name, size = 15 }: { name: keyof typeof P | string; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 16 16" fill="none" aria-hidden="true">{P[name]}</svg>;
}

// digit roll: only changed digits move (spring easing from --spring)
const DIGITS = Array.from({ length: 10 }, (_, k) => <i key={k}>{k}</i>);
export function Num({ value }: { value: number | string }) {
  const s = String(value);
  return (
    <span className="num" aria-label={s}>
      {[...s].map((ch, i) => (
        <span className="digit" key={s.length - i} aria-hidden="true">
          {/^\d$/.test(ch) ? <span style={{ transform: `translateY(-${+ch * 1.2}em)` }}>{DIGITS}</span> : <span>{ch}</span>}
        </span>
      ))}
    </span>
  );
}

// node-and-edge glyph, stacked a few px deep with CSS 3D
const markSvg = (cls?: string) => (
  <svg className={cls} viewBox="0 0 22 22" fill="none" key={cls || 'top'}>
    <line x1="6" y1="15" x2="16" y2="7" stroke="#55555f" strokeWidth="1.5" strokeLinecap="round" />
    <circle cx="6" cy="15" r="3.6" fill="#ededf0" />
    <circle cx="16" cy="7" r="2.4" fill="#8b8b96" />
  </svg>
);
export const Mark = () => (
  <div className="mark" aria-hidden="true"><div className="mark-inner">{markSvg('l3')}{markSvg('l2')}{markSvg('l1')}{markSvg()}</div></div>
);
