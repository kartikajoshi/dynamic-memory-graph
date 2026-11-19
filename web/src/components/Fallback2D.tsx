// No WebGL: the same graph as a 2D SVG (hub in the centre, facts on rings by hop distance).
// Same colours, same node card and Facts panel; never a blank box.
import type { GraphPayload } from '../api';
import { HUB_UI, typeColor } from '../palette';

export default function Fallback2D({ payload, showRetired, onSelect }: { payload: GraphPayload; showRetired: boolean; onSelect: (id: string) => void }) {
  const links = payload.links.filter((l) => showRetired || !l.endedAt);
  const hop = new Map<string, number>([['user', 0]]);
  const q = ['user'];
  while (q.length) {
    const c = q.shift()!;
    for (const l of links) {
      const nb = l.source === c ? l.target : l.target === c ? l.source : null;
      if (nb && !hop.has(nb)) { hop.set(nb, hop.get(c)! + 1); q.push(nb); }
    }
  }
  const pos = new Map<string, [number, number]>([['user', [0, 0]]]);
  const rings = new Map<number, string[]>();
  for (const n of payload.nodes) if (!n.isHub) { const h = Math.min(3, hop.get(n.id) ?? 1); rings.set(h, [...(rings.get(h) ?? []), n.id]); }
  for (const [h, ids] of rings) ids.forEach((id, i) => {
    const a = (i / ids.length) * Math.PI * 2 + h * 0.4;
    pos.set(id, [Math.cos(a) * 120 * h, Math.sin(a) * 120 * h]);
  });
  const R = 130 * Math.max(1, ...rings.keys()) + 40;
  return (
    <svg className="fallback2d" viewBox={`${-R} ${-R} ${2 * R} ${2 * R}`} role="img" aria-label="Memory graph (2D view: WebGL is unavailable)">
      {links.map((l) => {
        const a = pos.get(l.source), b = pos.get(l.target);
        if (!a || !b) return null;
        return <line key={l.id} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={l.rank != null ? '#f2e8d5' : '#c9c5bd'} strokeOpacity={l.endedAt ? 0.25 : 0.4} strokeDasharray={l.endedAt ? '4 4' : undefined} />;
      })}
      {payload.nodes.map((n) => {
        const p = pos.get(n.id);
        if (!p) return null;
        return (
          <g key={n.id} transform={`translate(${p[0]} ${p[1]})`} onClick={() => onSelect(n.id)} style={{ cursor: 'pointer' }}>
            <circle r={n.isHub ? 12 : 5 + 2 * Math.sqrt(n.degree)} fill={n.isHub ? HUB_UI : typeColor(n.type)} />
            <text y={n.isHub ? 26 : 20} textAnchor="middle" fill="#8b8b96" fontSize="11">{n.label}</text>
          </g>
        );
      })}
    </svg>
  );
}
