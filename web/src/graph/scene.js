// Memory graph in 3D (three.js + d3-force-3d). React drives it through apply() / playRetrieved() /
// select(); edge states always come from Python (graph_payload).
// v2 layout: a tabletop. Facts lie on a shallow slab (thin in y), on rings by hop distance from the
// core, with an edge-node repulsion force so edges don't run through unrelated nodes. The camera
// orbits around the vertical axis (auto-rotate keeps the same view quality at every azimuth).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { forceSimulation, forceLink, forceManyBody, forceRadial, forceCollide } from 'd3-force-3d';
import { HUB_UI, EDGE_COLORS, typeColor } from '../palette';

const BG = 0x08080a;
const ACTIVE_EDGE = new THREE.Color('#c9c5bd');
const RANK_PARTICLES = [8, 6, 5, 4, 3];
const RANK_BRIGHT = [1.0, 0.8, 0.65, 0.55, 0.45];
const SHELL = [0, 96, 172, 236, 290]; // base ring radius by hop distance from the core (grows with load)
const HUB_R = 9;
const FOV = 28;
const POLAR = (52 * Math.PI) / 180; // camera 38° above the table
const MAX_PERSPECTIVE = 1.15; // nearest vs farthest node of equal degree, on screen
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const clamp01 = (t) => Math.max(0, Math.min(1, t));
export const nodeRadius = (deg) => 3 + 2.2 * Math.sqrt(Math.max(1, deg));

let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;

// label text width without touching layout (Geist 500 11.5px / Geist Mono 11px)
const measureCtx = document.createElement('canvas').getContext('2d');
const textW = (text, mono) => { measureCtx.font = mono ? '400 11px "Geist Mono"' : '500 11.5px Geist'; return measureCtx.measureText(text).width; };

// d3 force: push nodes off edges they aren't part of (and edges off nodes), the core included
function forceEdgeRepel(getLinks) {
  let nodes = [];
  function force(alpha) {
    const links = getLinks();
    for (const n of nodes) {
      const rn = n.id === 'user' ? HUB_R : nodeRadius(n.degree || 1);
      const min = (rn + 7) * 2.2;
      for (const l of links) {
        const a = l.source, b = l.target;
        if (a === n || b === n || typeof a !== 'object') continue;
        const abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
        const len2 = abx * abx + aby * aby + abz * abz || 1e-6;
        const t = Math.max(0, Math.min(1, ((n.x - a.x) * abx + (n.y - a.y) * aby + (n.z - a.z) * abz) / len2));
        const dx = n.x - (a.x + abx * t), dy = n.y - (a.y + aby * t), dz = n.z - (a.z + abz * t);
        const d = Math.hypot(dx, dy, dz) || 0.01;
        if (d >= min) continue;
        const k = ((min - d) / d) * alpha * 0.9;
        n.vx += dx * k; n.vy += dy * k * 0.3; n.vz += dz * k;
        a.vx -= dx * k * (1 - t) * 0.6; a.vz -= dz * k * (1 - t) * 0.6;
        b.vx -= dx * k * t * 0.6; b.vz -= dz * k * t * 0.6;
      }
    }
  }
  force.initialize = (_) => { nodes = _; };
  return force;
}

export class Graph3D {
  constructor(host, { onSelect, onHover, reducedMotion = false } = {}) {
    this.host = host;
    this.onSelect = onSelect || (() => {});
    this.onHover = onHover || (() => {});
    this.reduced = reducedMotion;
    this.nodes = new Map();
    this.links = new Map();
    this.showRetired = true;
    this.mode3d = true;
    this.focus = { id: null, k: 0, target: 0, set: null };
    this.highlight = null;
    this.particles = [];
    this.userInteracted = false;
    this.lastInteraction = -1e9; // "never touched"
    this.clock = new THREE.Timer();
    this.fly = null;
    this.orbit = null;
    this.hubPulse = { x: 1, v: 0 };
    this.emptyField = null;
    this.free = null; // free canvas rect {l,t,r,b} in px (minus legend / toolbar / card)
    this.viewOff = { x: 0, y: 0 };
    this.metrics = {};
    this.worlds = new Map();
    this.worldKey = 'chat';

    const w = host.clientWidth, h = host.clientHeight;
    const renderer = (this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' }));
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    // composer renders to a linear target but the clear colour gets sRGB-encoded twice: pre-compensate once
    const bgLin = new THREE.Color(BG);
    const dec = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
    renderer.setClearColor(new THREE.Color().setRGB(dec(bgLin.r), dec(bgLin.g), dec(bgLin.b), THREE.LinearSRGBColorSpace), 1);
    renderer.toneMapping = THREE.NoToneMapping; // Neutral/ACES crush near-black greys; data colours render exact
    renderer.domElement.className = 'gl';
    host.prepend(renderer.domElement);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(w, h);
    this.labelRenderer.domElement.className = 'labels';
    host.appendChild(this.labelRenderer.domElement);

    const scene = (this.scene = new THREE.Scene());
    scene.fog = new THREE.FogExp2(BG, 0.0007);
    this.camera = new THREE.PerspectiveCamera(FOV, w / h, 1, 8000);
    this.setOrbit(0.35, POLAR, 900);

    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(120, 260, 160);
    scene.add(key);

    const controls = (this.controls = new OrbitControls(this.camera, renderer.domElement));
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.55;
    controls.zoomSpeed = 0.8;
    controls.minDistance = 120;
    controls.maxDistance = 3000;
    controls.minPolarAngle = (15 * Math.PI) / 180; // never edge-on: the table would collapse into a line
    controls.maxPolarAngle = (78 * Math.PI) / 180;
    controls.autoRotateSpeed = 0.33; // ~2°/s around the vertical axis
    controls.addEventListener('start', () => this.interacted());

    this.composer = new EffectComposer(renderer);
    this.composer.addPass(new RenderPass(scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.55, 0.3, 0.78);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.sphereGeo = new THREE.SphereGeometry(1, 32, 20);
    this.buildHub();
    this.buildParticles();

    this.simLinks = [];
    this.sim = forceSimulation([], 3)
      .force('link', forceLink([]).id((d) => d.id).distance((l) => (l.source.id === 'user' || l.target.id === 'user' ? SHELL[1] : 74)).strength(0.55))
      .force('charge', forceManyBody().strength(-120).distanceMax(260))
      .force('radial', forceRadial((d) => SHELL[Math.min(4, d.hop || 1)], 0, 0, 0).strength((d) => (d.id === 'user' ? 0 : 0.45)))
      .force('edgeRepel', forceEdgeRepel(() => this.simLinks))
      .force('collide', forceCollide((d) => (d.id === 'user' ? HUB_R + 18 : nodeRadius(d.degree || 1) + 12)).strength(0.9))
      .alphaDecay(0.035)
      .velocityDecay(0.34)
      .stop();
    this.simActive = false;

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2(9, 9);
    this.pointerDirty = false;
    const el = renderer.domElement;
    const aim = (e) => {
      const r = el.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.pointerDirty = true;
    };
    el.addEventListener('pointermove', aim);
    el.addEventListener('pointerleave', () => { this.pointer.set(9, 9); this.pointerDirty = true; });
    el.addEventListener('pointerdown', (e) => { aim(e); this.downAt = [e.clientX, e.clientY]; this.interacted(); });
    el.addEventListener('wheel', () => this.interacted(), { passive: true });
    el.addEventListener('pointerup', (e) => {
      if (!this.downAt) return;
      const moved = Math.hypot(e.clientX - this.downAt[0], e.clientY - this.downAt[1]);
      this.downAt = null;
      if (moved > 4) return;
      aim(e); // taps and touch have no pointermove before the click
      const hit = this.pick();
      this.select(hit ? hit.id : null);
    });
    host.addEventListener('pointerdown', (e) => {
      if (e.target !== el || e.button !== 0) return;
      aim(e);
      const hit = this.pick();
      if (!hit) return;
      e.stopPropagation(); // the camera stays still while a node is held
      this.startDrag(hit.id === 'user' ? this.hubNode() : hit, e);
    }, { capture: true });
    window.addEventListener('pointermove', (e) => this.dragMove(e));
    window.addEventListener('pointerup', (e) => this.endDrag(e));
    window.addEventListener('pointercancel', (e) => this.endDrag(e));
    el.addEventListener('dblclick', () => { if (!this.pick()) this.recenter(); });

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.loop = this.loop.bind(this);
    this.raf = requestAnimationFrame(this.loop);
  }

  dispose() {
    this.dead = true;
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.controls.dispose();
    this.composer.dispose?.();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labelRenderer.domElement.remove();
  }

  // ───────── drag: the graph stays live while a node is held (Obsidian feel) ─────────
  // Every other node is released into the simulation, so neighbours trail on their springs;
  // on release the held node is not pinned either: it floats back until the forces balance.
  startDrag(node, e) {
    this.interacted();
    this.drag = { node, x0: e.clientX, y0: e.clientY, moved: false, plane: new THREE.Plane() };
    this.hubHome = false;
    this.controls.enabled = false;
    for (const n of this.nodes.values()) if (n !== node) { n.fx = n.fz = undefined; n.fy = this.mode3d ? n.y : 0; }
    this.loosen(true);
    this.sim.alphaTarget(0.3).alpha(Math.max(this.sim.alpha(), 0.4));
    this.simActive = true; this.simTicks = 0;
    if (!this.selected) this.setFocus(node.id);
    this.renderer.domElement.style.cursor = 'grabbing';
  }
  dragMove(e) {
    const d = this.drag;
    if (!d) return;
    if (!d.moved && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < 4) return;
    d.moved = true;
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const n = d.node;
    d.plane.set(new THREE.Vector3(0, 1, 0), -n.y); // the table plane at the node's height (camera is never edge-on: polar ≤ 78°)
    const p = this.raycaster.ray.intersectPlane(d.plane, new THREE.Vector3());
    if (!p) return;
    n.fx = n.x = p.x; n.fz = n.z = p.z; n.fy = n.y;
    this.lastInteraction = performance.now();
  }
  endDrag(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    this.controls.enabled = true;
    this.renderer.domElement.style.cursor = 'pointer';
    if (d.node.id === 'user') this.hubHome = true; // the core glides back to the centre
    else { d.node.fx = d.node.fz = undefined; d.node.fy = d.node.y; } // let it float back, at its height
    this.sim.force('radial').strength((n) => (n.id === 'user' ? 0 : 0.12)); // rings ease back while it floats
    this.sim.alphaTarget(0).alpha(Math.max(this.sim.alpha(), 0.5));
    this.simActive = true; this.simTicks = 0;
    if (!d.moved) this.select(d.node.id); // a tap is still a click
    else if (!this.selected) this.setFocus(this.hoverId || null);
    void e;
  }

  // a stable height per node (from its id), for nodes that lost theirs in 2D
  homeY(n) {
    let h = 2166136261;
    for (const ch of n.id) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
    return (((h >>> 0) % 1000) / 1000 - 0.5) * 36;
  }

  loosen(on) {
    this.floating = on;
    this.sim.force('radial').strength((n) => (n.id === 'user' ? 0 : on ? 0.03 : 0.45));
    this.sim.force('link').strength(on ? 0.9 : 0.55);
    this.sim.alphaDecay(on ? 0.012 : 0.035).velocityDecay(on ? 0.18 : 0.34);
  }

  interacted() { this.userInteracted = true; this.lastInteraction = performance.now(); this.controls.autoRotate = false; this.orbit = null; }
  setReducedMotion(on) { this.reduced = on; if (on) this.controls.autoRotate = false; }

  setOrbit(azimuth, polar, dist, target = new THREE.Vector3()) {
    const s = new THREE.Spherical(dist, polar, azimuth);
    this.camera.position.setFromSpherical(s).add(target);
    this.camera.lookAt(target);
  }

  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
    this.composer.setSize(w, h);
    for (const l of this.links.values()) l.mat.resolution.set(w, h);
  }

  // free area of the canvas (px, relative to the canvas) where the graph should sit
  setFreeRect(rect) { this.free = rect; }

  // ───────── the memory core ─────────
  buildHub() {
    const g = (this.hubGroup = new THREE.Group());
    // glass core: bright fresnel rim, see-through centre, so the facts inside it stay visible
    const coreMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: { deep: { value: new THREE.Color('#9a9aa2') }, rim: { value: new THREE.Color(HUB_UI) }, glow: { value: 0.0 }, opacity: { value: 1 } },
      vertexShader: `varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix*vec4(position,1.0); vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
      fragmentShader: `uniform vec3 deep; uniform vec3 rim; uniform float glow; uniform float opacity; varying vec3 vN; varying vec3 vV;
        void main(){ float f = pow(1.0 - abs(dot(normalize(vN), vV)), 2.0);
          float light = 0.6 + 0.4 * clamp(dot(normalize(vN), normalize(vec3(0.3, 0.8, 0.5))), 0.0, 1.0);
          vec3 c = mix(deep * 0.42, rim, f) * light * (1.0 + glow);
          gl_FragColor = vec4(c, (0.2 + 0.8 * f) * opacity); }`,
    });
    const core = new THREE.Mesh(this.sphereGeo, coreMat);
    core.scale.setScalar(HUB_R);
    g.add(core);
    this.hubCore = core;
    // memory inside the core: one point per active fact (bright), dim points for retired ones
    const MAXF = 600;
    const fg = new THREE.BufferGeometry();
    fg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAXF * 3), 3));
    fg.setAttribute('alpha', new THREE.BufferAttribute(new Float32Array(MAXF), 1));
    fg.setDrawRange(0, 0);
    this.coreFacts = new THREE.Points(fg, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { flare: { value: 0 }, scale: { value: 1 } },
      vertexShader: `attribute float alpha; varying float vA; uniform float scale;
        void main(){ vA = alpha; vec4 mv = modelViewMatrix*vec4(position,1.0); gl_PointSize = scale*(2.2 + 2.0*alpha)*(520.0/-mv.z); gl_Position = projectionMatrix*mv; }`,
      fragmentShader: `uniform float flare; varying float vA;
        void main(){ float d = length(gl_PointCoord-0.5); float a = smoothstep(0.5, 0.0, d); gl_FragColor = vec4(vec3(0.93, 0.92, 0.9) * (0.75 + flare), a * vA); }`,
    }));
    this.coreFacts.frustumCulled = false;
    this.coreFacts.renderOrder = 2;
    this.coreFactsMax = MAXF;
    core.add(this.coreFacts);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(16, 0.09, 6, 180), new THREE.MeshBasicMaterial({ color: new THREE.Color('#8b8b96'), transparent: true, opacity: 0.45 }));
    const tilt = new THREE.Group();
    tilt.rotation.set(Math.PI / 2 + 0.35, 0.15, 0);
    tilt.add(ring);
    g.add(tilt);
    this.hubRing = ring;
    this.scene.add(g);
    this.hubLabel = this.makeLabel('User', 'hub', HUB_UI);
    this.hubLabel.userData.w = textW('User') + 10;
    g.add(this.hubLabel);
  }

  buildParticles() {
    const MAX = 900;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX * 3), 3));
    geo.setAttribute('alpha', new THREE.BufferAttribute(new Float32Array(MAX), 1));
    geo.setAttribute('size', new THREE.BufferAttribute(new Float32Array(MAX), 1));
    geo.setDrawRange(0, 0);
    const mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { color: { value: new THREE.Color(EDGE_COLORS.used).multiplyScalar(3.0) }, scale: { value: 1 } },
      vertexShader: `attribute float alpha; attribute float size; varying float vA; uniform float scale;
        void main(){ vA = alpha; vec4 mv = modelViewMatrix*vec4(position,1.0); gl_PointSize = size*scale*(1400.0/-mv.z); gl_Position = projectionMatrix*mv; }`,
      fragmentShader: `uniform vec3 color; varying float vA;
        void main(){ vec2 c = gl_PointCoord-0.5; float d = length(c); float a = smoothstep(0.5,0.0,d); a = a*a*(3.0-2.0*a); gl_FragColor = vec4(color*a*vA, a*vA); }`,
    });
    this.pPoints = new THREE.Points(geo, mat);
    this.pPoints.frustumCulled = false;
    this.scene.add(this.pPoints);
    this.pMax = MAX;
  }

  setEmptyField(on) {
    if (on && !this.emptyField) {
      const N = 160, pos = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        const r = 26 + Math.pow(rand(), 1.6) * 120, th = rand() * Math.PI * 2;
        pos.set([r * Math.cos(th), (rand() - 0.5) * 30, r * Math.sin(th)], i * 3);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const mat = new THREE.PointsMaterial({ color: 0x8b8b96, size: 1.6, sizeAttenuation: true, transparent: true, opacity: 0.0, depthWrite: false });
      this.emptyField = new THREE.Points(geo, mat);
      this.emptyField.userData.target = 0.4;
      this.scene.add(this.emptyField);
    } else if (!on && this.emptyField) this.emptyField.userData.target = 0;
  }

  makeLabel(text, cls, color) {
    const outer = document.createElement('div');
    outer.className = `lbl ${cls || ''}`;
    const inner = document.createElement('div');
    inner.textContent = text;
    inner.style.setProperty('--c', color || '#888');
    outer.appendChild(inner);
    const o = new CSS2DObject(outer);
    o.userData = { shown: true, streak: 0 };
    return o;
  }
  makeEdgeLabel(text) {
    const outer = document.createElement('div');
    outer.className = 'elbl';
    const inner = document.createElement('div');
    inner.textContent = text;
    outer.appendChild(inner);
    const o = new CSS2DObject(outer);
    o.visible = false;
    o.userData = { shown: false, streak: 0, w: textW(text, true) + 12 };
    return o;
  }

  // ───────── data in ─────────
  apply(nodesIn, linksIn, { animate = true } = {}) {
    const fresh = [];
    const want = new Set(nodesIn.map((n) => n.id));
    for (const n of nodesIn) {
      if (n.id === 'user') continue;
      let node = this.nodes.get(n.id);
      if (!node) { node = this.createNode(n); fresh.push(node); }
      node.present = true;
    }
    for (const node of this.nodes.values()) if (!want.has(node.id)) node.present = false;
    const wantL = new Set(linksIn.map((l) => l.id));
    for (const l of linksIn) {
      const link = this.links.get(l.id) || this.createLink(l);
      link.present = true;
      this.setLinkState(link, l.state, animate, l.rank);
    }
    for (const link of this.links.values()) if (!wantL.has(link.id)) link.present = false;
    this.updateDegrees();
    this.computeHops();
    this.placeNew(fresh, linksIn, animate);
    this.restartSim(fresh, animate);
    this.refreshCore();
  }

  // deterministic spot inside the unit sphere per fact id, so points never jump between turns
  refreshCore() {
    const pos = this.coreFacts.geometry.attributes.position.array, al = this.coreFacts.geometry.attributes.alpha.array;
    let i = 0;
    for (const l of this.links.values()) {
      if (!l.present || i >= this.coreFactsMax) continue;
      let h = 2166136261;
      for (const ch of l.id) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
      const r1 = ((h >>> 0) % 1000) / 1000, r2 = ((h >>> 10) % 1000) / 1000, r3 = ((h >>> 20) % 1000) / 1000;
      const r = 0.72 * Math.cbrt(0.08 + 0.92 * r1), th = r2 * Math.PI * 2, ph = Math.acos(2 * r3 - 1);
      pos.set([r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th)], i * 3);
      al[i] = this.isRetired(l) ? 0.22 : 0.9;
      i++;
    }
    this.coreFacts.geometry.setDrawRange(0, i);
    this.coreFacts.geometry.attributes.position.needsUpdate = true;
    this.coreFacts.geometry.attributes.alpha.needsUpdate = true;
  }

  // Chat and Compare share this canvas; each world keeps its own nodes, positions and camera
  useWorld(key) {
    if (key === this.worldKey) return;
    const objs = () => [...[...this.nodes.values()].flatMap((n) => [n.mesh, n.labelObj]), ...[...this.links.values()].flatMap((l) => [l.line, l.label])];
    for (const o of objs()) this.scene.remove(o);
    this.worlds.set(this.worldKey, { nodes: this.nodes, links: this.links, cam: this.camera.position.clone(), target: this.controls.target.clone(), interacted: this.userInteracted, last: this.lastInteraction });
    const w = this.worlds.get(key);
    this.worldKey = key;
    this.particles = []; this.highlight = null; this.selected = null; this.fly = null; this.orbit = null; this.simActive = false;
    this.focus = { id: null, k: 0, target: 0, set: null };
    this.nodes = w ? w.nodes : new Map();
    this.links = w ? w.links : new Map();
    for (const o of objs()) this.scene.add(o);
    if (w) { this.camera.position.copy(w.cam); this.controls.target.copy(w.target); this.userInteracted = w.interacted; this.lastInteraction = w.last; } else { this.lastInteraction = -1e9; this.setOrbit(0.35, POLAR, 900); this.controls.target.set(0, 0, 0); }
    this.sim.nodes([this.hubNode(), ...this.nodes.values()]);
    this.sim.force('link').links([...this.links.values()].map((l) => ({ source: l.source, target: l.target })));
    this.simLinks = this.sim.force('link').links();
    this.refreshCore();
  }

  shellR(hop) {
    const h = Math.min(4, hop || 1);
    const count = [...this.nodes.values()].filter((n) => (n.hop || 1) === h).length;
    const grow = Math.max(0, (count * 40) / (2 * Math.PI) - SHELL[h]);
    return SHELL[h] + grow + (h > 1 ? Math.max(0, this.shellR(h - 1) - SHELL[h - 1]) : 0);
  }

  computeHops() {
    const hop = new Map([['user', 0]]); const q = ['user'];
    while (q.length) {
      const c = q.shift();
      for (const x of this.links.values()) {
        const nb = x.source === c ? x.target : x.target === c ? x.source : null;
        if (nb && !hop.has(nb)) { hop.set(nb, hop.get(c) + 1); q.push(nb); }
      }
    }
    for (const n of this.nodes.values()) n.hop = hop.get(n.id) || 1;
  }

  // deterministic placement: 1-hop facts into the widest gap on the inner ring,
  // n-hop facts just beyond their parent (so Priya's facts sit outside Priya)
  placeNew(fresh, linksIn, animate) {
    const ang = (n) => Math.atan2(n.z, n.x);
    for (const node of fresh) {
      let x, z;
      const R = this.shellR(node.hop);
      if (node.hop <= 1) {
        const angles = [...this.nodes.values()].filter((n) => n !== node && n.hop === 1 && n.x !== undefined).map(ang).sort((a, b) => a - b);
        let a = -0.5;
        if (angles.length === 1) a = angles[0] + Math.PI;
        else if (angles.length > 1) {
          let best = -1;
          for (let i = 0; i < angles.length; i++) {
            const a0 = angles[i], a1 = i + 1 < angles.length ? angles[i + 1] : angles[0] + Math.PI * 2;
            if (a1 - a0 > best) { best = a1 - a0; a = (a0 + a1) / 2; }
          }
        }
        x = Math.cos(a) * R; z = Math.sin(a) * R;
      } else {
        const l = linksIn.find((e) => (e.source === node.id || e.target === node.id) && (this.nodes.get(e.source === node.id ? e.target : e.source)?.hop === node.hop - 1 || (e.source === node.id ? e.target : e.source) === 'user'));
        const pid = l ? (l.source === node.id ? l.target : l.source) : 'user';
        const p = this.pos(pid);
        const siblings = [...this.nodes.values()].filter((n) => n !== node && n.parent === pid).length;
        node.parent = pid;
        const a = (p && p.x !== undefined ? ang(p) : 0) + (siblings % 2 ? 1 : -1) * (0.16 + 0.12 * Math.floor(siblings / 2)) + (siblings === 0 ? 0.08 : 0);
        x = Math.cos(a) * R; z = Math.sin(a) * R;
      }
      node.x = x; node.z = z; node.y = (rand() - 0.5) * 24;
      if (!this.mode3d) node.y = 0;
      node.spring = { x: animate && !this.reduced ? 0 : 1, v: 0, target: 1 };
      node.bornAt = performance.now();
      // the first facts fly out of the core: start at the hub and let the spring carry them out
      node.from = animate && !this.reduced ? { x: 0, y: 0, z: 0, t: 0 } : null;
    }
  }

  hubNode() {
    if (!this._hub) this._hub = { id: 'user', x: 0, y: 0, z: 0, fx: 0, fy: 0, fz: 0, hop: 0, degree: 1 };
    return this._hub;
  }

  createNode(n) {
    const color = new THREE.Color(typeColor(n.type));
    const mat = new THREE.MeshLambertMaterial({ color, emissive: color.clone().multiplyScalar(0.3), transparent: true, opacity: 1 });
    const mesh = new THREE.Mesh(this.sphereGeo, mat);
    mesh.userData.id = n.id;
    this.scene.add(mesh);
    const label = this.makeLabel(n.label, '', typeColor(n.type));
    label.userData.w = textW(n.label) + 16;
    this.scene.add(label);
    const node = { id: n.id, label: n.label, type: n.type, mesh, mat, labelObj: label, degree: 1, alpha: 1, present: true, hot: false, spring: { x: 1, v: 0, target: 1 } };
    this.nodes.set(n.id, node);
    return node;
  }

  createLink(l) {
    const geo = new LineGeometry();
    geo.setPositions([0, 0, 0, 0, 0, 0.001]);
    const mat = new LineMaterial({ color: 0xffffff, linewidth: 1.2, transparent: true, opacity: 0.3, depthWrite: false, dashed: false, dashSize: 3, gapSize: 3 });
    mat.resolution.set(this.host.clientWidth, this.host.clientHeight);
    const line = new Line2(geo, mat);
    line.frustumCulled = false;
    this.scene.add(line);
    const label = this.makeEdgeLabel(l.predicate.replace(/_/g, ' '));
    this.scene.add(label);
    const link = { id: l.id, source: l.source, target: l.target, predicate: l.predicate, line, geo, mat, label, state: 'active', rank: null, draw: 1, flash: null, alpha: 0, present: true, color: ACTIVE_EDGE.clone() };
    this.links.set(l.id, link);
    return link;
  }

  setLinkState(link, state, animate, rank = null) {
    const prev = link.state;
    link.state = state; link.rank = rank;
    const now = performance.now();
    const anim = animate && !this.reduced;
    if (state === 'new' && prev !== 'new') { link.draw = anim ? 0 : 1; link.flash = anim ? { c: new THREE.Color(EDGE_COLORS.new), t0: now, dur: 1600 } : null; }
    else if (state === 'updated' && prev !== 'updated') { link.draw = anim ? 0 : 1; link.flash = anim ? { c: new THREE.Color(EDGE_COLORS.updated), t0: now, dur: 1400 } : null; }
    else if (state === 'retired_now' && prev !== 'retired_now') link.flash = anim ? { c: new THREE.Color(EDGE_COLORS.retired_now), t0: now, dur: 1500 } : null;
  }

  restartSim(fresh, animate) {
    const hub = this.hubNode();
    const all = [hub, ...this.nodes.values()];
    for (const n of all) {
      if (n === hub) continue;
      if (fresh.includes(n)) { n.fx = n.fz = undefined; n.fy = this.mode3d ? undefined : 0; n.vx = n.vy = n.vz = 0; }
      else { n.fx = n.x; n.fy = this.mode3d ? n.y : 0; n.fz = n.z; }
    }
    this.sim.nodes(all);
    this.sim.force('link').links([...this.links.values()].map((l) => ({ source: l.source, target: l.target })));
    this.simLinks = this.sim.force('link').links();
    const radii = [0, 1, 2, 3, 4].map((hh) => (hh ? this.shellR(hh) : 0));
    this.sim.force('radial').radius((d) => radii[Math.min(4, d.hop || 1)]);
    if (!fresh.length) return;
    this.sim.alpha(0.9);
    if (!animate || this.reduced) { for (let i = 0; i < 180; i++) this.tickSim(); this.settle(false); }
    else { this.simActive = true; this.simTicks = 0; }
  }

  // one physics step + the slab: |y| stays within ±30% of the table's radius
  tickSim() {
    this.sim.tick();
    let R = SHELL[1];
    for (const n of this.nodes.values()) R = Math.max(R, Math.hypot(n.x, n.z));
    const lim = this.mode3d ? 0.3 * R : 0;
    for (const n of this.nodes.values()) {
      if (n.fy !== undefined && n.fy !== null) continue;
      n.vy -= n.y * 0.02;
      if (Math.abs(n.y) > lim) { n.y = Math.sign(n.y) * lim; n.vy = 0; }
    }
  }

  settle(wasLive = true) {
    this.simActive = false;
    if (this.floating) this.loosen(false);
    for (const n of this.nodes.values()) { n.fx = n.x; n.fy = this.mode3d ? n.y : 0; n.fz = n.z; }
    if (wasLive && !this.fly && !this.orbit && performance.now() - this.lastInteraction > 6000 && !this.allInView()) this.recenter();
  }

  updateDegrees() {
    const deg = new Map();
    for (const l of this.links.values()) {
      if (!l.present || (!this.showRetired && this.isRetired(l))) continue;
      deg.set(l.source, (deg.get(l.source) || 0) + 1);
      deg.set(l.target, (deg.get(l.target) || 0) + 1);
    }
    for (const n of this.nodes.values()) n.degree = deg.get(n.id) || 0;
  }

  isRetired(l) { return l.state === 'retired' || l.state === 'retired_now'; }
  pos(id) { return id === 'user' ? this.hubNode() : this.nodes.get(id); }
  worldR(id) { if (id === 'user') return HUB_R; const n = this.nodes.get(id); return n ? nodeRadius(n.degree) * (n.scaleK || 1) : 4; }

  // ───────── screen-space helpers (also used by the occlusion check) ─────────
  project(p, out = new THREE.Vector3()) {
    out.set(p.x, p.y, p.z).project(this.camera);
    const w = this.host.clientWidth, h = this.host.clientHeight;
    return { x: (out.x * 0.5 + 0.5) * w, y: (-out.y * 0.5 + 0.5) * h, z: out.z };
  }
  projR(id) {
    const p = this.pos(id); const d = this.camera.position.distanceTo(new THREE.Vector3(p.x, p.y, p.z));
    return (this.worldR(id) * this.host.clientHeight / 2) / (Math.tan((FOV * Math.PI) / 360) * d);
  }
  // edges that pass within 1.5× the on-screen radius of a node that isn't one of their endpoints
  occlusion(linkIds) {
    const bad = [];
    const ids = ['user', ...[...this.nodes.values()].filter((n) => n.present && n.mesh.visible).map((n) => n.id)];
    const P = new Map(ids.map((id) => [id, { ...this.project(this.pos(id)), r: this.projR(id) }]));
    for (const lid of linkIds) {
      const l = this.links.get(lid); if (!l || !P.has(l.source) || !P.has(l.target)) continue;
      const a = P.get(l.source), b = P.get(l.target);
      for (const id of ids) {
        if (id === l.source || id === l.target) continue;
        const c = P.get(id);
        const abx = b.x - a.x, aby = b.y - a.y, len2 = abx * abx + aby * aby || 1e-6;
        const t = Math.max(0, Math.min(1, ((c.x - a.x) * abx + (c.y - a.y) * aby) / len2));
        const d = Math.hypot(c.x - (a.x + abx * t), c.y - (a.y + aby * t));
        if (d < 1.5 * c.r) bad.push({ link: lid, node: id, d: +d.toFixed(1), limit: +(1.5 * c.r).toFixed(1) });
      }
    }
    return bad;
  }

  sweep(linkIds, n = 36) {
    const target = this.controls.target.clone();
    const cur = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(target));
    const saved = this.camera.position.clone(); let clean = 0; const worst = [];
    for (let i = 0; i < n; i++) {
      this.setOrbit(cur.theta + (i * Math.PI * 2) / n, cur.phi, cur.radius, target); this.camera.updateMatrixWorld();
      const v = this.occlusion(linkIds); if (!v.length) clean++; else worst.push(v.length);
    }
    this.camera.position.copy(saved); this.camera.lookAt(target); this.camera.updateMatrixWorld();
    return { clean, of: n };
  }

  // when retrieval happens and the user isn't steering: ease to the azimuth where the given edges
  // overlap other nodes least on screen (12 samples, current angle wins ties)
  bestAzimuth(linkIds) {
    // evaluate at the camera's destination if it's mid-flight
    const target = (this.fly ? this.fly.target : this.controls.target).clone();
    const cur = new THREE.Spherical().setFromVector3((this.fly ? this.fly.to : this.camera.position).clone().sub(target));
    const saved = this.camera.position.clone(), savedT = this.controls.target.clone();
    let best = { az: cur.theta, score: Infinity };
    const N = 24; // < 2 ms at 40 facts; finds clean angles twice as often as 12
    for (let i = 0; i < N; i++) {
      const az = cur.theta + (i * Math.PI * 2) / N;
      this.setOrbit(az, cur.phi, cur.radius, target);
      this.camera.updateMatrixWorld();
      const score = this.occlusion(linkIds).reduce((s, v) => s + 1 + (v.limit - v.d) / v.limit, 0) + Math.min(i, N - i) * 0.005;
      if (score < best.score) best = { az, score };
    }
    this.camera.position.copy(saved); this.camera.lookAt(savedT); this.camera.updateMatrixWorld();
    return best;
  }
  orbitTo(az) {
    if (this.fly) { // compose with a flight in progress: rotate its destination instead of fighting it
      const s = new THREE.Spherical().setFromVector3(this.fly.to.clone().sub(this.fly.target));
      s.theta = az; this.fly.to.setFromSpherical(s).add(this.fly.target); return;
    }
    if (this.reduced) { const s = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(this.controls.target)); this.setOrbit(az, s.phi, s.radius, this.controls.target); return; }
    this.orbit = { az };
  }

  // ───────── retrieved flow ─────────
  playRetrieved(linkIds) {
    this.retrievedAt = performance.now();
    this.pauseRotateUntil = performance.now() + 8000;
    if (performance.now() - this.lastInteraction > 6000 && this.mode3d) {
      const b = this.bestAzimuth(linkIds);
      this.orbitTo(b.az);
    }
    if (this.reduced) return;
    const t0 = performance.now() + (this.orbit ? 450 : 0);
    linkIds.forEach((id, rank) => {
      const l = this.links.get(id);
      if (!l) return;
      const path = this.pathToHub(l);
      const count = RANK_PARTICLES[rank] || 3;
      for (let i = 0; i < count; i++) this.particles.push({ path, start: t0 + rank * 110 + i * (1150 / count) + rand() * 90, speed: 125 + rand() * 30, rank, size: 5.4 - rank * 0.55 + rand() * 0.6 });
    });
    this.firstArrival = null;
  }

  pathToHub(l) {
    const dist = this.hubDistances();
    const [a, b] = (dist.get(l.source) ?? 99) <= (dist.get(l.target) ?? 99) ? [l.source, l.target] : [l.target, l.source];
    const path = [b, a];
    let cur = a, guard = 0;
    while (cur !== 'user' && guard++ < 8) {
      let best = null;
      for (const x of this.links.values()) {
        if (!x.present || this.isRetired(x)) continue;
        const nb = x.source === cur ? x.target : x.target === cur ? x.source : null;
        if (nb && (dist.get(nb) ?? 99) < (dist.get(cur) ?? 99)) best = nb;
      }
      if (!best) break;
      path.push(best); cur = best;
    }
    return path;
  }
  hubDistances() {
    const dist = new Map([['user', 0]]); const q = ['user'];
    while (q.length) {
      const c = q.shift();
      for (const x of this.links.values()) {
        if (!x.present || this.isRetired(x)) continue;
        const nb = x.source === c ? x.target : x.target === c ? x.source : null;
        if (nb && !dist.has(nb)) { dist.set(nb, dist.get(c) + 1); q.push(nb); }
      }
    }
    return dist;
  }

  // ───────── interaction ─────────
  pick() {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const meshes = [...this.nodes.values()].filter((n) => n.mesh.visible).map((n) => n.mesh);
    meshes.push(this.hubCore);
    const hit = this.raycaster.intersectObjects(meshes, false)[0];
    if (!hit) {
      const w = this.host.clientWidth, h = this.host.clientHeight;
      const px = (this.pointer.x * 0.5 + 0.5) * w, py = (-this.pointer.y * 0.5 + 0.5) * h;
      let best = null, bd = Infinity;
      for (const n of this.nodes.values()) {
        if (!n.mesh.visible) continue;
        const p = this.project(n.mesh.position), d = Math.hypot(p.x - px, p.y - py);
        if (d < Math.max(16, this.projR(n.id) + 10) && d < bd) { bd = d; best = n; }
      }
      return best;
    }
    return hit.object === this.hubCore ? { id: 'user' } : this.nodes.get(hit.object.userData.id);
  }

  focusSet(id) {
    const set = new Set([id]);
    for (const l of this.links.values()) {
      if (!l.present || (!this.showRetired && this.isRetired(l))) continue;
      if (l.source === id) set.add(l.target);
      if (l.target === id) set.add(l.source);
    }
    return set;
  }
  focusLinks(id) { return [...this.links.values()].filter((l) => l.present && (l.source === id || l.target === id) && (this.showRetired || !this.isRetired(l))).map((l) => l.id); }
  setFocus(id) {
    if (id) this.focus = { id, k: this.focus.k, target: 1, set: this.focusSet(id) };
    else this.focus.target = 0;
  }
  highlightLinks(ids) { this.highlight = ids && ids.length ? new Set(ids) : null; }

  select(id) {
    this.selected = id;
    this.onSelect(id);
    if (!id) { this.setFocus(null); return; }
    this.setFocus(id);
    // frame the node with its neighbours, from the azimuth where its edges overlap least
    this.interacted();
    const ids = [...this.focus.set];
    const best = this.bestAzimuth(this.focusLinks(id));
    this.fitTo(ids, best.az);
  }

  flyToLink(id) {
    const l = this.links.get(id);
    if (!l) return;
    this.highlightLinks([id]);
    this.fitTo([l.source, l.target], null, 1.8);
  }

  // zoom-to-fit inside the free area: the principal point moves to the free rect's centre
  // (setViewOffset), then the distance is solved per node so every node lands inside it
  fitTo(ids, azimuth = null, pad = 1, polar = null) {
    const pts = ids.map((id) => this.pos(id)).filter((p) => p && p.x !== undefined);
    if (!pts.length) return;
    const box = new THREE.Box3();
    for (const p of pts) box.expandByPoint(new THREE.Vector3(p.x, p.y, p.z));
    const target = box.getCenter(new THREE.Vector3());
    const cur = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(this.controls.target));
    const az = azimuth ?? cur.theta, phi = this.mode3d ? (polar ?? cur.phi) : 0.02;
    const dir = new THREE.Vector3().setFromSpherical(new THREE.Spherical(1, phi, az));
    const fwd = dir.clone().negate();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(right, fwd).normalize();
    const w = this.host.clientWidth, h = this.host.clientHeight;
    const f = this.free || { l: 0, t: 0, r: w, b: h };
    const fw = (f.r - f.l) / 2 - 18, fh = (f.b - f.t) / 2 - 18;
    const k = h / 2 / Math.tan((FOV * Math.PI) / 360);
    let D = 380; // never closer than this: an empty memory shouldn't fill the frame with the core
    for (const id of ids) {
      const p = this.pos(id); if (!p || p.x === undefined) continue;
      const v = new THREE.Vector3(p.x, p.y, p.z).sub(target);
      const r = this.worldR(id) * pad + 14; // radius + room for its label
      const x = Math.abs(v.dot(right)) + r, y = Math.abs(v.dot(up)) + r + 10, z = v.dot(dir);
      D = Math.max(D, z + (x * k) / fw, z + (y * k) / fh);
    }
    const to = target.clone().add(dir.multiplyScalar(D));
    if (this.reduced) { this.camera.position.copy(to); this.controls.target.copy(target); return; }
    this.fly = { to, target };
  }
  recenter(polar = null) {
    this.fitTo(['user', ...[...this.nodes.values()].filter((n) => n.present && n.spring.target > 0).map((n) => n.id)], null, 1, polar);
  }
  allInView() {
    const f = this.free || { l: 0, t: 0, r: this.host.clientWidth, b: this.host.clientHeight };
    for (const n of this.nodes.values()) {
      if (!n.present) continue;
      const p = this.project(n);
      if (p.x < f.l + 10 || p.x > f.r - 10 || p.y < f.t + 10 || p.y > f.b - 24) return false;
    }
    return true;
  }

  setRetiredVisible(on) { this.showRetired = on; this.updateDegrees(); }

  setMode3d(on) {
    if (on === this.mode3d) return;
    this.mode3d = on;
    this.interacted();
    if (!on) {
      for (const n of this.nodes.values()) { n.y3 = n.y; n.yT = 0; }
      this.nudgeOverlaps();
      this.controls.enableRotate = false;
    } else {
      for (const n of this.nodes.values()) n.yT = n.y3 ?? this.homeY(n);
      this.controls.enableRotate = true;
    }
    setTimeout(() => this.recenter(on ? POLAR : null), 30);
  }
  nudgeOverlaps() {
    const arr = [...this.nodes.values()];
    const P = arr.map((n) => ({ x: n.x, z: n.z, r: nodeRadius(n.degree) }));
    for (let it = 0; it < 40; it++) {
      for (const a of P) { const d = Math.hypot(a.x, a.z) || 0.01, min = HUB_R + a.r + 14; if (d < min) { a.x *= min / d; a.z *= min / d; } }
      for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
        const a = P[i], b = P[j]; const dx = b.x - a.x, dz = b.z - a.z, d = Math.hypot(dx, dz) || 0.01, min = a.r + b.r + 16;
        if (d < min) { const push = (min - d) / 2; a.x -= (dx / d) * push; a.z -= (dz / d) * push; b.x += (dx / d) * push; b.z += (dz / d) * push; }
      }
    }
    arr.forEach((n, i) => { if (Math.hypot(P[i].x - n.x, P[i].z - n.z) > 0.01) { n.xT = P[i].x; n.zT = P[i].z; } });
  }

  clear() {
    for (const n of this.nodes.values()) { this.scene.remove(n.mesh); this.scene.remove(n.labelObj); n.labelObj.element.remove(); n.mat.dispose(); }
    for (const l of this.links.values()) { this.scene.remove(l.line); this.scene.remove(l.label); l.label.element.remove(); l.geo.dispose(); l.mat.dispose(); }
    this.nodes.clear(); this.links.clear(); this.particles = [];
    this.refreshCore();
    this.sim.nodes([this.hubNode()]); this.sim.force('link').links([]); this.simLinks = [];
    seed = 7;
  }

  // ───────── greedy screen-space label placement ─────────
  // priority: hub > focused/hovered > changed this turn > the rest (by degree). A label tries
  // below / above / right / left of its node and hides if all four collide with a higher-priority
  // label or with any node disc. Edge labels slide along their edge, never inside the core.
  placeLabels(f, dimOut) {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    const discs = [];
    const P = new Map();
    for (const id of ['user', ...this.nodes.keys()]) {
      const n = id === 'user' ? this.hubNode() : this.nodes.get(id);
      if (id !== 'user' && !(n.present && n.mesh.visible)) continue;
      const p = this.project(n); const r = this.projR(id) * (id === 'user' ? 1 : n.spring.x);
      P.set(id, { ...p, r }); discs.push({ id, x: p.x, y: p.y, r: r + 3 });
    }
    const placed = [];
    const H = 18;
    const hitsRect = (R) => placed.some((q) => R.x < q.x + q.w && R.x + R.w > q.x && R.y < q.y + q.h && R.y + R.h > q.y);
    const hitsDisc = (R, own) => discs.some((d) => d.id !== own && d.x + d.r > R.x && d.x - d.r < R.x + R.w && d.y + d.r > R.y && d.y - d.r < R.y + R.h &&
      Math.hypot(Math.max(R.x - d.x, 0, d.x - R.x - R.w), Math.max(R.y - d.y, 0, d.y - R.y - R.h)) < d.r);
    const inside = (R) => R.x > 2 && R.y > 2 && R.x + R.w < w - 2 && R.y + R.h < h - 2;
    const cand = [];
    const many = this.nodes.size > 100;
    cand.push({ kind: 'n', id: 'user', obj: this.hubLabel, pri: 0 });
    for (const n of this.nodes.values()) {
      if (!P.has(n.id) || many) continue;
      const foc = f.set && f.k > 0.3 && f.set.has(n.id);
      const pri = foc ? 1 : n.hot ? 2 : this.nodes.size <= 100 ? 4 : 9;
      cand.push({ kind: 'n', id: n.id, obj: n.labelObj, pri, deg: n.degree });
    }
    for (const l of this.links.values()) {
      if (!l.showLabel) continue;
      const foc = f.set && f.k > 0.3 && (l.source === f.id || l.target === f.id);
      cand.push({ kind: 'e', id: l.id, obj: l.label, link: l, pri: foc ? 1.5 : 3 });
    }
    cand.sort((a, b) => a.pri - b.pri || (b.deg || 0) - (a.deg || 0));
    for (const c of cand) {
      let ok = null;
      if (c.kind === 'n') {
        const p = P.get(c.id); if (!p) continue;
        const Wd = c.obj.userData.w, gap = p.r + 4;
        const opts = [[0, gap + H / 2], [0, -gap - H / 2], [gap + Wd / 2, 0], [-gap - Wd / 2, 0]];
        for (const [dx, dy] of opts) {
          const R = { x: p.x + dx - Wd / 2, y: p.y + dy - H / 2, w: Wd, h: H };
          if (inside(R) && !hitsRect(R) && !hitsDisc(R, c.id)) { ok = { R, dx, dy }; break; }
        }
      } else {
        const l = c.link; const a = P.get(l.source), b = P.get(l.target);
        if (!a || !b) continue;
        const Wd = c.obj.userData.w; const hub = P.get('user');
        for (const t of [0.5, 0.62, 0.38, 0.74, 0.26]) {
          const x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t;
          const R = { x: x - Wd / 2, y: y - H / 2, w: Wd, h: H };
          if (Math.hypot(x - hub.x, y - hub.y) < hub.r * 1.6 + Wd / 2) continue; // never on the core
          if (inside(R) && !hitsRect(R) && !hitsDisc(R, null)) { ok = { R, t }; break; }
        }
      }
      const U = c.obj.userData;
      const want = !!ok;
      if (want === U.shown) U.streak = 0;
      else if (++U.streak > (want ? 6 : 2)) { U.shown = want; U.streak = 0; }
      if (ok && U.shown) {
        placed.push(ok.R);
        if (c.kind === 'n') { U.dx = ok.dx; U.dy = ok.dy; }
        else U.t = ok.t;
      } else if (U.shown && U.R) placed.push(U.R); // keep its last spot reserved while it fades
      if (ok) U.R = ok.R;
      c.obj.userData.final = U.shown && (ok || U.R);
    }
    return P;
  }

  // ───────── frame ─────────
  loop() {
    if (this.dead) return;
    this.raf = requestAnimationFrame(this.loop);
    this.clock.update();
    const dt = Math.min(0.05, this.clock.getDelta());
    const now = performance.now();
    const t = now / 1000;

    if (this.simActive) { this.tickSim(); if (!this.drag && (++this.simTicks > (this.floating ? 600 : 120) || this.sim.alpha() < 0.01)) this.settle(true); }

    const kz = this.reduced ? 1 : 1 - Math.exp(-dt * 7);
    for (const n of this.nodes.values()) {
      for (const ax of ['x', 'y', 'z']) {
        const T = n[ax + 'T'];
        if (T === undefined) continue;
        n[ax] += (T - n[ax]) * kz; n['f' + ax] = n[ax];
        if (Math.abs(T - n[ax]) < 0.05) { n[ax] = T; n['f' + ax] = T; n[ax + 'T'] = undefined; }
      }
    }

    // view offset: the principal point eases to the centre of the free area
    const W = this.host.clientWidth, Hh = this.host.clientHeight;
    if (this.free) {
      const tx = W / 2 - (this.free.l + this.free.r) / 2, ty = Hh / 2 - (this.free.t + this.free.b) / 2;
      this.viewOff.x += (tx - this.viewOff.x) * (this.reduced ? 1 : 1 - Math.exp(-dt * 6));
      this.viewOff.y += (ty - this.viewOff.y) * (this.reduced ? 1 : 1 - Math.exp(-dt * 6));
      this.camera.setViewOffset(W, Hh, this.viewOff.x, this.viewOff.y, W, Hh);
    }

    const rotateOk = !this.userInteracted && !this.reduced && this.mode3d && !this.orbit && now > (this.pauseRotateUntil || 0);
    this.controls.autoRotate = rotateOk;

    if (this.fly) {
      const k = 1 - Math.exp(-dt * 5.2);
      this.camera.position.lerp(this.fly.to, k);
      this.controls.target.lerp(this.fly.target, k);
      if (this.camera.position.distanceTo(this.fly.to) < 0.5) this.fly = null;
    }
    if (this.orbit) {
      const s = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(this.controls.target));
      let d = this.orbit.az - s.theta; d = Math.atan2(Math.sin(d), Math.cos(d));
      s.theta += d * (1 - Math.exp(-dt * 4));
      this.camera.position.setFromSpherical(s).add(this.controls.target);
      if (Math.abs(d) < 0.002) this.orbit = null;
    }
    this.controls.update();

    if (this.pointerDirty) {
      this.pointerDirty = false;
      const hit = this.pick();
      const id = hit ? hit.id : null;
      if (!this.drag) this.renderer.domElement.style.cursor = hit ? 'pointer' : 'grab';
      if (id !== this.hoverId) { this.hoverId = id; this.onHover(id); if (!this.selected && !this.drag) this.setFocus(id); }
    }
    const f = this.focus;
    f.k += (f.target - f.k) * (this.reduced ? 1 : 1 - Math.exp(-dt * 10));
    if (f.target === 0 && f.k < 0.01) { f.k = 0; f.id = null; f.set = null; }
    const dimOut = 1 - 0.85 * f.k;

    // core (follows its sim node, which is pinned at the origin except while dragged)
    const hub = this.hubNode();
    if (this.hubHome) {
      const k = this.reduced ? 1 : 1 - Math.exp(-dt * 3.5);
      for (const ax of ['x', 'y', 'z']) { hub[ax] += (0 - hub[ax]) * k; hub['f' + ax] = hub[ax]; }
      if (Math.hypot(hub.x, hub.y, hub.z) < 0.3) { hub.x = hub.y = hub.z = hub.fx = hub.fy = hub.fz = 0; this.hubHome = false; }
    }
    this.hubGroup.position.set(hub.x, hub.y, hub.z);
    const breath = this.nodes.size === 0 && !this.reduced ? Math.sin((t * Math.PI * 2) / 4) : 0;
    this.hubPulse.v += (1 - this.hubPulse.x) * 170 * dt - this.hubPulse.v * 14 * dt;
    this.hubPulse.x += this.hubPulse.v * dt;
    const hubK = this.nodes.size ? Math.pow(this.camera.position.length() / Math.max(1, this.camera.position.distanceTo(this.controls.target)), 1 - (this._eps ?? 1)) : 1;
    this.hubCore.scale.setScalar(HUB_R * hubK * this.hubPulse.x * (1 + 0.035 * breath));
    this.hubCore.material.uniforms.glow.value = Math.max(0, this.hubPulse.x - 1) * 5;
    this.coreFacts.material.uniforms.flare.value = Math.max(0, this.hubPulse.x - 1) * 9;
    this.coreFacts.material.uniforms.scale.value = this.renderer.getPixelRatio();
    this.coreFacts.rotation.y += this.reduced ? 0 : dt * 0.05;
    if (!this.reduced) this.hubRing.rotation.z += dt * 0.15;
    const hubDim = f.set && !f.set.has('user') ? dimOut : 1;
    this.hubCore.material.uniforms.opacity.value = hubDim;
    this.hubRing.material.opacity = 0.45 * hubDim;

    if (this.emptyField) {
      const m = this.emptyField.material;
      m.opacity += (this.emptyField.userData.target - m.opacity) * (1 - Math.exp(-dt * 1.6));
      if (!this.reduced) this.emptyField.rotation.y += dt * 0.012;
      if (this.emptyField.userData.target === 0 && m.opacity < 0.01) { this.scene.remove(this.emptyField); this.emptyField = null; }
    }

    // perspective cap: per-node world scale so on-screen size varies with depth by at most 15%
    const camPos = this.camera.position;
    const dRef = camPos.distanceTo(this.controls.target);
    let dmin = Infinity, dmax = 0;
    for (const n of this.nodes.values()) {
      if (!n.present) continue;
      n.camD = camPos.distanceTo(new THREE.Vector3(n.x, n.y, n.z));
      dmin = Math.min(dmin, n.camD); dmax = Math.max(dmax, n.camD);
    }
    const ratio = dmax > 0 ? dmax / dmin : 1;
    const eps = ratio > MAX_PERSPECTIVE ? Math.log(MAX_PERSPECTIVE) / Math.log(ratio) : 1;
    this.metrics.perspective = +Math.pow(ratio, eps).toFixed(3);
    this._eps = eps;

    for (const n of this.nodes.values()) {
      const visible = n.present && !(!this.showRetired && n.degree === 0);
      n.spring.target = visible ? 1 : 0;
      if (this.reduced) n.spring.x = n.spring.target;
      else { n.spring.v += (n.spring.target - n.spring.x) * 170 * dt - n.spring.v * 18 * dt; n.spring.x += n.spring.v * dt; }
      const s = Math.max(0, n.spring.x);
      n.scaleK = n.camD ? Math.pow(n.camD / dRef, 1 - eps) : 1;
      let px = n.x, py = n.y, pz = n.z;
      if (n.from) { // fly out of the core on birth
        n.from.t = Math.min(1, n.from.t + dt / 0.7); const e = easeOut(n.from.t);
        px = n.from.x + (n.x - n.from.x) * e; py = n.from.y + (n.y - n.from.y) * e; pz = n.from.z + (n.z - n.from.z) * e;
        if (n.from.t >= 1) n.from = null;
      }
      n.mesh.position.set(px, py, pz);
      n.labelObj.position.set(px, py, pz);
      const lift = this.drag?.node === n ? 1.28 : n.id === this.hoverId ? 1.14 : 1;
      n.hk = (n.hk ?? 1) + (lift - (n.hk ?? 1)) * (this.reduced ? 1 : 1 - Math.exp(-dt * 12));
      n.mesh.scale.setScalar(nodeRadius(n.degree) * n.scaleK * s * n.hk);
      n.mesh.visible = s > 0.02;
      const inFocus = !f.set || f.set.has(n.id);
      n.alpha = inFocus ? 1 : dimOut;
      n.mat.opacity = n.alpha;
    }

    // links
    for (const l of this.links.values()) {
      const A = this.pos(l.source), B = this.pos(l.target);
      if (!A || !B || A.x === undefined || B.x === undefined) continue;
      const a = l.source === 'user' ? A : A.mesh.position, b = l.target === 'user' ? B : B.mesh.position;
      const retired = this.isRetired(l);
      const visible = l.present && (!retired || this.showRetired) && (this.nodes.get(l.target)?.spring.x ?? 1) > 0.05 && (l.source === 'user' || (this.nodes.get(l.source)?.spring.x ?? 1) > 0.05);
      l.alpha += ((visible ? 1 : 0) - l.alpha) * (this.reduced ? 1 : 1 - Math.exp(-dt * 8));
      if (l.draw < 1) l.draw = Math.min(1, l.draw + dt / 0.6);
      const d = easeOut(l.draw);
      const ex = a.x + (b.x - a.x) * d, ey = a.y + (b.y - a.y) * d, ez = a.z + (b.z - a.z) * d;
      const arr = l.geo.attributes.instanceStart.data.array;
      arr[0] = a.x; arr[1] = a.y; arr[2] = a.z; arr[3] = ex; arr[4] = ey; arr[5] = ez;
      l.geo.attributes.instanceStart.data.needsUpdate = true;
      l.end = { x: ex, y: ey, z: ez };

      let col = ACTIVE_EDGE, op = 0.3, w = 1.15, dashed = false;
      const used = l.rank !== null && l.rank !== undefined && !retired;
      if (retired) { op = 0.22; w = 1; dashed = true; }
      if (used) {
        const since = this.retrievedAt ? (now - this.retrievedAt) / 1000 : 99;
        const live = this.reduced ? 0.35 : since < 2.8 ? 1 : Math.max(0.35, 1 - ((since - 2.8) / 1.2) * 0.65);
        const rb = RANK_BRIGHT[l.rank];
        col = new THREE.Color(EDGE_COLORS.used).multiplyScalar(0.55 + 0.45 * rb * live);
        op = 0.25 + 0.55 * rb * (0.4 + 0.6 * live);
        w = 1.1 + 0.9 * rb * (0.5 + 0.5 * live);
      }
      if (l.state === 'new' && !used) { col = new THREE.Color(EDGE_COLORS.new).multiplyScalar(0.8); op = 0.7; w = 1.8; }
      if (l.state === 'updated' && !used) { col = new THREE.Color(EDGE_COLORS.updated).multiplyScalar(0.8); op = 0.75; w = 1.8; }
      if (l.state === 'retired_now') { col = new THREE.Color(EDGE_COLORS.retired_now).multiplyScalar(0.55); op = 0.45; w = 1.3; }
      l.color.copy(col);
      if (l.flash) {
        const p = (now - l.flash.t0) / l.flash.dur;
        if (p >= 1) l.flash = null;
        else {
          const k = Math.sin(Math.min(1, p * 1.4) * Math.PI) * (1 - p * 0.3);
          l.color.lerp(l.flash.c.clone().multiplyScalar(2.0), clamp01(k));
          op = Math.max(op, 0.95 * k); w = Math.max(w, 1.4 + 1.6 * k);
          if (l.state === 'retired_now' && p > 0.55) dashed = true;
        }
      }
      if (this.highlight) {
        if (this.highlight.has(l.id)) { op = Math.max(op, 0.95); w = Math.max(w, 2.6); l.color.lerp(new THREE.Color(1.6, 1.55, 1.45), 0.6); }
        else op *= 0.35;
      }
      const inFocus = !f.set || (f.set.has(l.source) && f.set.has(l.target) && (l.source === f.id || l.target === f.id));
      if (!inFocus) op *= dimOut;
      l.mat.color.copy(l.color);
      l.mat.opacity = op * l.alpha;
      l.mat.linewidth = w;
      if (dashed !== l.mat.dashed) { l.mat.dashed = dashed; l.mat.needsUpdate = true; }
      if (dashed) l.line.computeLineDistances();
      l.line.visible = l.alpha > 0.01;
      l.used = used;
      l.showLabel = l.alpha > 0.5 && ((f.set && f.k > 0.3 && inFocus) || l.state === 'new' || l.state === 'updated' || l.state === 'retired_now' || (this.highlight && this.highlight.has(l.id)));
    }

    // labels: tiers + greedy collision (needs the camera matrices of this frame)
    this.camera.updateMatrixWorld();
    const P = this.placeLabels(f, dimOut);
    const hubU = this.hubLabel.userData;
    this.hubLabel.element.firstChild.style.transform = `translate(${hubU.dx || 0}px, ${hubU.dy ?? 22}px)`;
    this.hubLabel.element.firstChild.style.opacity = hubU.final ? (f.set && !f.set.has('user') ? 0.2 : 1) : 0;
    for (const n of this.nodes.values()) {
      const U = n.labelObj.userData, el = n.labelObj.element.firstChild;
      const hot = n.hot || (f.set && f.k > 0.3 && f.set.has(n.id));
      el.parentElement.classList.toggle('hot', !!hot);
      const fade = clamp01(1.4 - ((n.camD || 0) - (dRef + 120)) / 600);
      let o = U.final ? (hot ? 1 : this.nodes.size <= 20 ? fade : 0.62 * fade) : 0;
      if (f.set && !f.set.has(n.id)) o *= 0.15;
      o *= Math.max(0, n.spring.x);
      el.style.opacity = o.toFixed(3);
      if (U.dx !== undefined) el.style.transform = `translate(${U.dx}px, ${U.dy}px)`;
      n.labelObj.visible = o > 0.02 || !!U.final;
    }
    for (const l of this.links.values()) {
      const U = l.label.userData;
      const show = !!(l.showLabel && U.final && l.end);
      l.label.visible = show;
      if (show) {
        const A = l.source === 'user' ? this.hubNode() : this.nodes.get(l.source).mesh.position;
        const tt = U.t ?? 0.5;
        l.label.position.set(A.x + (l.end.x - A.x) * tt, A.y + (l.end.y - A.y) * tt, A.z + (l.end.z - A.z) * tt);
        l.label.element.className = `elbl ${l.used ? 'used' : l.state}`;
      }
    }

    // particles
    const pos = this.pPoints.geometry.attributes.position.array;
    const al = this.pPoints.geometry.attributes.alpha.array;
    const sz = this.pPoints.geometry.attributes.size.array;
    let np = 0;
    this.particles = this.particles.filter((p) => {
      const age = (now - p.start) / 1000;
      if (age < 0) return true;
      let travel = age * p.speed;
      for (let i = 0; i < p.path.length - 1; i++) {
        const A = this.pos(p.path[i]), B = this.pos(p.path[i + 1]);
        if (!A || !B) return false;
        const seg = Math.hypot(B.x - A.x, B.y - A.y, B.z - A.z);
        if (travel <= seg) {
          const head = Math.min(1, age * 4);
          for (const [back, fa, fs] of [[0, 1, 1], [5, 0.45, 0.75], [10, 0.2, 0.55]]) {
            const kk = Math.max(0, (travel - back) / seg);
            if (np >= this.pMax) break;
            pos[np * 3] = A.x + (B.x - A.x) * kk; pos[np * 3 + 1] = A.y + (B.y - A.y) * kk; pos[np * 3 + 2] = A.z + (B.z - A.z) * kk;
            al[np] = head * fa * (0.55 + 0.45 * RANK_BRIGHT[p.rank]); sz[np] = p.size * fs; np++;
          }
          return true;
        }
        travel -= seg;
      }
      if (!this.firstArrival) { this.firstArrival = now; this.hubPulse.v += 2.2; } else this.hubPulse.v += 0.12;
      return false;
    });
    this.pPoints.geometry.setDrawRange(0, np);
    this.pPoints.geometry.attributes.position.needsUpdate = true;
    this.pPoints.geometry.attributes.alpha.needsUpdate = true;
    this.pPoints.geometry.attributes.size.needsUpdate = true;
    this.pPoints.material.uniforms.scale.value = this.renderer.getPixelRatio();

    this.composer.render();
    this.labelRenderer.render(this.scene, this.camera);
    this._frames = (this._frames || 0) + 1;
    void P;
  }
}
