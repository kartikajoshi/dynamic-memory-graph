import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './fonts.css';
import './styles.css';
import App from './App';

// spring easing as CSS linear(): stiffness 170, damping 20 (~2.5% overshoot), for digit rolls
(function springVar() {
  const pts: number[] = [];
  let x = 0, v = 0;
  const dt = 1 / 120;
  for (let t = 0; t <= 0.7 + 1e-9; t += dt) { pts.push(x); const a = 170 * (1 - x) - 20 * v; v += a * dt; x += v * dt; }
  const step = Math.max(1, Math.floor(pts.length / 40));
  const out: string[] = [];
  for (let i = 0; i < pts.length; i += step) out.push(pts[i].toFixed(3));
  out.push('1');
  document.documentElement.style.setProperty('--spring', `linear(${out.join(', ')})`);
})();

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
