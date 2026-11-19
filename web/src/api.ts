// Typed client for server.py. Shapes mirror the FastAPI responses; nothing is derived here.

export interface GNode { id: string; label: string; type: string | null; degree: number; isHub: boolean; isNew: boolean; createdAt: string | null }
export interface GLink {
  id: string; source: string; target: string; predicate: string; label: string; text: string;
  createdAt: string | null; endedAt: string | null; state: 'active' | 'new' | 'updated' | 'retired' | 'retired_now'; used: boolean; rank: number | null;
}
export interface Counts { entities: number; active: number; retired: number }
export interface Step { at: string; turn: number | null; message: string | null }
export interface Changes { new_nodes: string[]; new_edges: string[]; updated_edges: string[]; retired_edges: string[]; used_edges: string[] }
export interface GraphPayload { nodes: GNode[]; links: GLink[]; counts: Counts; steps: Step[]; until: string | null; changes: Changes | null }

export interface Triple { subject: string; predicate: string; object: string; object_type?: string }
export interface Action { op: 'ADD' | 'SKIP' | 'UPDATE' | 'DELETE'; subject: string; predicate: string; object: string }
export interface Timings { rewrite: number; extract: number; update: number; retrieve: number; answer: number; total?: number }
export interface Result {
  user_message: string; rewritten_query: string; triples: Triple[]; actions: Action[];
  retrieved_facts: { text: string; source: string; target: string; key: number; object_type?: string }[];
  answer: string; timings: Timings; usage: { prompt_tokens: number; completion_tokens: number };
}
export interface Retrieved { id: string; text: string; rank: number | null; linked: boolean }
export interface Turn {
  user: string; answer: string; result: Result; changes: Changes | null; retrieval: Retrieved[];
  rank_basis: string; stamp: string | null; at: number; index?: number; graph?: GraphPayload;
}
export interface Status {
  ollama: boolean; models: string[]; default_model: string; chat_model: string; counts: Counts; loaded_facts: number;
  settings: { context_window: number; history_turns: number; top_k: number; answer_reserve: number };
  live_trace: boolean; columns: Record<string, string>; examples: { chat: string[]; compare: string[] };
}
export interface Fact {
  id: string; subject: string; predicate: string; object: string; object_type?: string; text: string;
  created_at: string | null; ended_at: string | null; source: string; target: string; key: number;
}
export type StageName = 'rewrite' | 'extract' | 'update' | 'retrieve' | 'answer';
export interface StageEvent { type: 'stage'; stage: StageName; status: 'start' | 'end'; seconds?: number; t: number; [k: string]: unknown }

// compare
export type Col = 'recent' | 'full' | 'dmg';
export interface ColResult extends Partial<Result> {
  answer: string; model: string; latency?: number; prompt_tokens?: number | null; completion_tokens?: number | null;
  note?: string; dropped_turns?: number; changes?: Changes;
}
export interface Stale { fact: string; ended_at: string; sentence: string; edge: string; turns_ago: number | null }
export interface CompareRow {
  index: number; user: string; cols: Record<Col, ColResult>; scores: Record<Col, { ok: boolean; reasons: string[] }> | null;
  stale: Partial<Record<Col, Stale[]>>; script: string | null; turn_index: number | null; retrieval: Retrieved[] | null;
}
export interface Totals { [c: string]: { passed: number; scored: number } }
export interface CompareState { rows: CompareRow[]; totals: Totals; checks: Checks | null; script: string | null; model: string; dmg_model_used: string }
export interface Checks { script: string; results: { check: string; ok: boolean }[] }
export interface Script { name: string; turns: number; scored: number; label: string }

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: { 'content-type': 'application/json' }, ...init });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail ?? msg; } catch { /* not JSON */ }
    throw new ApiError(res.status, String(msg));
  }
  return res.json();
}
const post = <T,>(path: string, body: unknown = {}) => call<T>(path, { method: 'POST', body: JSON.stringify(body) });

export const api = {
  status: () => call<Status>('/api/status'),
  graph: (scope: 'chat' | 'compare' = 'chat', until?: string | null) =>
    call<GraphPayload>(`/api/graph?scope=${scope}${until ? `&until=${encodeURIComponent(until)}` : ''}`),
  facts: (scope: 'chat' | 'compare' = 'chat') => call<{ facts: Fact[]; counts: Counts }>(`/api/facts?scope=${scope}`),
  history: () => call<{ turns: Turn[]; loaded_facts: number }>('/api/chat/history'),
  reset: () => post<{ ok: boolean; counts: Counts }>('/api/reset'),
  clearChat: () => post<{ ok: boolean }>('/api/chat/clear'),
  scripts: () => call<{ scripts: Script[] }>('/api/scripts'),
  script: (name: string) => call<{ name: string; turns: { message: string; expect?: string[]; forbid?: string[] }[] }>(`/api/demo-script?name=${name}`),
  compareState: () => call<CompareState>('/api/compare/state'),
  compareTurn: (body: { message?: string; script?: string; turn_index?: number; model?: string; models?: Partial<Record<Col, string>>; parallel?: boolean }) =>
    post<{ row: CompareRow; totals: Totals; graph: GraphPayload; dmg_model_used: string }>('/api/compare/turn', body),
  compareChecks: (script: string) => post<Checks>('/api/compare/checks', { script }),
  compareReset: () => post<{ ok: boolean }>('/api/compare/reset'),

  // One chat turn as server-sent events: each pipeline stage as it starts / ends, then the reply.
  async chatStream(message: string, onStage: (e: StageEvent) => void): Promise<Turn> {
    const res = await fetch('/api/chat/stream', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ message }) });
    if (!res.ok || !res.body) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail ?? msg; } catch { /* not JSON */ }
      throw new ApiError(res.status, String(msg));
    }
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      let cut;
      while ((cut = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        const line = chunk.split('\n').find((l) => l.startsWith('data: '));
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === 'stage') onStage(ev);
        else if (ev.type === 'done') return ev.reply as Turn;
        else if (ev.type === 'error') throw new ApiError(500, ev.error);
      }
    }
    throw new ApiError(500, 'stream ended without a reply');
  },
};

// small formatting helpers shared by views
export const fmtS = (s?: number | null) => (s == null || s < 0.005 ? null : s < 10 ? `${s.toFixed(2)}s` : `${s.toFixed(1)}s`);
export const hhmm = (iso?: string | null) => (iso ? iso.slice(11, 16) : '');
export const hhmmss = (iso?: string | null) => (iso ? iso.slice(11, 19) : '');
export const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;
