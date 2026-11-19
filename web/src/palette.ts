// Data colours for the new UI. Python's graph_viz.TYPE_COLORS stays the source for the Streamlit
// app; here a client-side override keeps every pair ΔE76 ≥ 13.6 under normal, deutan, protan and
// tritan vision (Machado 2009 simulation, severity 1.0).
export const TYPE_COLORS_UI: Record<string, string> = {
  PERSON: '#3987e5',
  CONDITION: '#d0461b',
  DRUG: '#04965d',
  PLACE: '#c98500',
  PREFERENCE: '#be4880',
  ORGANIZATION: '#3bc33c',
  FOOD: '#e66767',
};
export const OTHER_COLOR_UI = '#9c9d92';
// The core is neutral in the new UI (Python's payload still says #a855f7 for the hub).
export const HUB_UI = '#d4d4da';
export const EDGE_COLORS = { new: '#6ee7a8', updated: '#fcd34d', retired_now: '#f87171', used: '#f2e8d5' };

export const typeKey = (t?: string | null) => {
  const k = String(t || '').toUpperCase();
  return k in TYPE_COLORS_UI ? k : 'OTHER';
};
export const typeColor = (t?: string | null) => TYPE_COLORS_UI[typeKey(t)] || OTHER_COLOR_UI;
export const typeName = (t?: string | null) => {
  const k = typeKey(t);
  return k[0] + k.slice(1).toLowerCase();
};
