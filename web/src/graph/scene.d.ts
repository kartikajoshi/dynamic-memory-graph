// Types for the imperative 3D view (scene.js). Only what React calls.
export interface SceneNodeIn { id: string; label: string; type?: string | null }
export interface SceneLinkIn { id: string; source: string; target: string; predicate: string; state: string; rank: number | null }
export interface SceneNode { id: string; label: string; type?: string; present: boolean; hot: boolean }
export interface FreeRect { l: number; t: number; r: number; b: number }
export class Graph3D {
  constructor(host: HTMLElement, opts?: { onSelect?: (id: string | null) => void; onHover?: (id: string | null) => void; reducedMotion?: boolean });
  nodes: Map<string, SceneNode>;
  worldKey: string;
  selected: string | null;
  metrics: Record<string, number>;
  apply(nodes: SceneNodeIn[], links: SceneLinkIn[], opts?: { animate?: boolean }): void;
  playRetrieved(linkIds: string[]): void;
  highlightLinks(ids: string[] | null): void;
  select(id: string | null): void;
  flyToLink(id: string): void;
  recenter(): void;
  clear(): void;
  dispose(): void;
  useWorld(key: string): void;
  setFreeRect(rect: FreeRect): void;
  setEmptyField(on: boolean): void;
  setRetiredVisible(on: boolean): void;
  setMode3d(on: boolean): void;
  setReducedMotion(on: boolean): void;
  occlusion(linkIds: string[]): unknown[];
}
