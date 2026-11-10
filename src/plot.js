// Canvas-компоненты: график аппроксимации, кривая ошибки, схема сети.

export function colors() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n) => cs.getPropertyValue(n).trim();
  return {
    fg: g('--fg'),
    muted: g('--muted'),
    line: g('--line'),
    lineStrong: g('--line-strong'),
    accent: g('--accent'),
    accent2: g('--accent-2'),
    glow: g('--glow'),
    ok: g('--ok'),
    warn: g('--warn'),
    truth: g('--truth'),
    surface: g('--plot-bg'),
    dark: document.documentElement.dataset.theme === 'dark',
  };
}

const MONO = '"JetBrains Mono", ui-monospace, monospace';

export function dprOf(canvas) {
  const w = canvas.getBoundingClientRect().width;
  return w > 0 ? canvas.width / w : 1;
}

export function fitCanvas(canvas, onResize) {
  const ro = new ResizeObserver(() => {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    onResize();
  });
  ro.observe(canvas);
  return ro;
}

function niceStep(span, count) {
  const raw = span / count;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / p;
  const n = f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10;
  return n * p;
}
function ticks(min, max, count) {
  const step = niceStep(max - min, count);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { out, step };
}
function fmt(v, step) {
  const d = Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(d);
}

export class FitPlot {
  constructor(canvas, opts = {}) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.minimal = !!opts.minimal;
    this.interactive = opts.interactive !== false;
    this.onAdd = opts.onAdd;
    this.onRemove = opts.onRemove;
    this.p = null;
    this.pred = null;
    this.hoverX = null;
    this.paint = false;
    this._lastAdd = null;
    fitCanvas(canvas, () => this.render());
    if (this.interactive) this._bind();
  }

  // p = { range:[a,b], ext:[lo,hi], gridX, trueY|null, train:{x,y}, draw:boolean, showExtrap:boolean }
  setProblem(p) {
    this.p = p;
    this.pred = null;
    this._yRange();
    this.render();
  }
  setTrain(train) {
    this.p.train = train;
    this._yRange();
    this.render();
  }
  setPred(pred) {
    this.pred = pred;
    this.render();
  }
  setExtrap(v) {
    if (!this.p) return;
    this.p.showExtrap = v;
    this.render();
  }

  _yRange() {
    const p = this.p;
    let lo = Infinity;
    let hi = -Infinity;
    const eat = (arr) => {
      if (!arr) return;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    };
    eat(p.trueY);
    eat(p.train && p.train.y);
    if (p.draw) {
      lo = Math.min(lo, -1.2);
      hi = Math.max(hi, 1.2);
    }
    if (!Number.isFinite(lo)) {
      lo = -1;
      hi = 1;
    }
    let span = hi - lo;
    if (span < 0.2) {
      const mid = (hi + lo) / 2;
      lo = mid - 0.1;
      hi = mid + 0.1;
      span = 0.2;
    }
    this.y0 = lo - span * 0.12;
    this.y1 = hi + span * 0.12;
  }

  _geom() {
    const { width: W, height: H } = this.c;
    const dpr = dprOf(this.c);
    const pad = this.minimal
      ? { l: 6 * dpr, r: 6 * dpr, t: 8 * dpr, b: 8 * dpr }
      : { l: 46 * dpr, r: 24 * dpr, t: 14 * dpr, b: 30 * dpr };
    const p = this.p;
    const [x0, x1] = p.showExtrap ? p.ext : p.range;
    return { W, H, dpr, pad, x0, x1, pw: W - pad.l - pad.r, ph: H - pad.t - pad.b };
  }
  _sx(g, x) {
    return g.pad.l + ((x - g.x0) / (g.x1 - g.x0)) * g.pw;
  }
  _sy(g, y) {
    return g.pad.t + (1 - (y - this.y0) / (this.y1 - this.y0)) * g.ph;
  }

  render() {
    if (!this.p) return;
    const ctx = this.ctx;
    const g = this._geom();
    const C = colors();
    const { W, H, dpr, pad } = g;
    ctx.clearRect(0, 0, W, H);
    const p = this.p;

    ctx.save();
    ctx.font = `${10.5 * dpr}px ${MONO}`;

    if (!this.minimal) {
      const ty = ticks(this.y0, this.y1, 5);
      const tx = ticks(g.x0, g.x1, 7);
      ctx.lineWidth = dpr;
      ctx.strokeStyle = C.line;
      ctx.fillStyle = C.muted;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (const v of ty.out) {
        const y = this._sy(g, v);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(W - pad.r, y);
        ctx.stroke();
        ctx.fillText(fmt(v, ty.step), pad.l - 8 * dpr, y);
      }
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (const v of tx.out) {
        const x = this._sx(g, v);
        ctx.beginPath();
        ctx.moveTo(x, pad.t);
        ctx.lineTo(x, H - pad.b);
        ctx.stroke();
        ctx.fillText(fmt(v, tx.step), x, H - pad.b + 8 * dpr);
      }
      if (this.y0 < 0 && this.y1 > 0) {
        ctx.strokeStyle = C.lineStrong;
        const y = this._sy(g, 0);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(W - pad.r, y);
        ctx.stroke();
      }
    }

    ctx.beginPath();
    ctx.rect(pad.l, pad.t, g.pw, g.ph);
    ctx.clip();

    // области вне обучающего диапазона (экстраполяция)
    if (p.showExtrap) {
      const xa = this._sx(g, p.range[0]);
      const xb = this._sx(g, p.range[1]);
      ctx.fillStyle = C.dark ? 'rgba(255,255,255,0.035)' : 'rgba(0,0,0,0.05)';
      ctx.fillRect(pad.l, pad.t, xa - pad.l, g.ph);
      ctx.fillRect(xb, pad.t, W - pad.r - xb, g.ph);
      ctx.strokeStyle = C.lineStrong;
      ctx.setLineDash([4 * dpr, 5 * dpr]);
      ctx.beginPath();
      ctx.moveTo(xa, pad.t);
      ctx.lineTo(xa, H - pad.b);
      ctx.moveTo(xb, pad.t);
      ctx.lineTo(xb, H - pad.b);
      ctx.stroke();
      ctx.setLineDash([]);
      if (!this.minimal) {
        ctx.fillStyle = C.muted;
        ctx.textBaseline = 'top';
        ctx.font = `${9.5 * dpr}px ${MONO}`;
        ctx.textAlign = 'left';
        if (xa - pad.l > 70 * dpr) ctx.fillText('ВНЕ ОБУЧЕНИЯ', pad.l + 8 * dpr, pad.t + 8 * dpr);
        ctx.textAlign = 'right';
        if (W - pad.r - xb > 70 * dpr) ctx.fillText('ВНЕ ОБУЧЕНИЯ', W - pad.r - 8 * dpr, pad.t + 8 * dpr);
        ctx.font = `${10.5 * dpr}px ${MONO}`;
      }
    }

    if (p.trueY) {
      ctx.beginPath();
      ctx.lineWidth = (this.minimal ? 1.2 : 1.6) * dpr;
      ctx.strokeStyle = C.truth;
      ctx.lineJoin = 'round';
      for (let i = 0; i < p.gridX.length; i++) {
        const x = this._sx(g, p.gridX[i]);
        const y = this._sy(g, p.trueY[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    let head = null;
    if (this.pred) {
      const grad = ctx.createLinearGradient(pad.l, 0, W - pad.r, 0);
      grad.addColorStop(0, C.accent);
      grad.addColorStop(0.5, C.accent2);
      grad.addColorStop(1, C.accent);
      ctx.save();
      ctx.shadowColor = C.glow;
      ctx.shadowBlur = (C.dark ? 20 : 10) * dpr;
      ctx.lineWidth = (this.minimal ? 2.6 : 3.2) * dpr;
      ctx.strokeStyle = grad;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let best = -Infinity;
      for (let i = 0; i < p.gridX.length; i++) {
        const v = this.pred[i];
        const x = this._sx(g, p.gridX[i]);
        const y = this._sy(g, v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        if (this.hoverX == null && p.gridX[i] >= p.range[0] && p.gridX[i] <= p.range[1] && v > best) {
          best = v;
          head = { x, y };
        }
      }
      ctx.stroke();
      ctx.restore();
    }

    const tr = p.train;
    if (tr) {
      const r = (this.minimal ? 2.6 : 3.8) * dpr;
      for (let i = 0; i < tr.x.length; i++) {
        const x = this._sx(g, tr.x[i]);
        const y = this._sy(g, tr.y[i]);
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = C.ok;
        ctx.fill();
        ctx.lineWidth = 1.2 * dpr;
        ctx.strokeStyle = C.dark ? 'rgba(5,8,14,0.9)' : 'rgba(255,255,255,0.9)';
        ctx.stroke();
      }
    }

    if (this.hoverX != null && this.pred) this._probe(g, C, ctx);
    else if (head) this._dot(ctx, head.x, head.y, C, dpr);

    ctx.restore();
  }

  _dot(ctx, x, y, C, dpr) {
    ctx.save();
    ctx.shadowColor = C.glow;
    ctx.shadowBlur = 26 * dpr;
    ctx.fillStyle = C.glow;
    ctx.beginPath();
    ctx.arc(x, y, 11 * dpr, 0, Math.PI * 2);
    ctx.globalAlpha = 0.45;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = C.dark ? '#cfe5ff' : '#ffffff';
    ctx.shadowBlur = 14 * dpr;
    ctx.beginPath();
    ctx.arc(x, y, 5.6 * dpr, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2 * dpr;
    ctx.strokeStyle = C.accent;
    ctx.stroke();
    ctx.restore();
  }

  _probe(g, C, ctx) {
    const p = this.p;
    const { dpr, pad, W } = g;
    const X = this.hoverX;
    const gx = p.gridX;
    let k = 0;
    let bd = Infinity;
    for (let i = 0; i < gx.length; i++) {
      const d = Math.abs(gx[i] - X);
      if (d < bd) {
        bd = d;
        k = i;
      }
    }
    const x = this._sx(g, gx[k]);
    ctx.strokeStyle = C.lineStrong;
    ctx.lineWidth = dpr;
    ctx.setLineDash([3 * dpr, 4 * dpr]);
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + g.ph);
    ctx.stroke();
    ctx.setLineDash([]);
    const py = this._sy(g, this.pred[k]);
    this._dot(ctx, x, py, C, dpr);
    if (p.trueY) {
      const ty = this._sy(g, p.trueY[k]);
      ctx.beginPath();
      ctx.arc(x, ty, 4 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = C.truth;
      ctx.fill();
    }
    const lines = [`x = ${gx[k].toFixed(3)}`, `ŷ = ${this.pred[k].toFixed(3)}`];
    if (p.trueY) lines.push(`f = ${p.trueY[k].toFixed(3)}`, `|Δ| = ${Math.abs(this.pred[k] - p.trueY[k]).toFixed(3)}`);
    ctx.font = `${11 * dpr}px ${MONO}`;
    const lh = 15 * dpr;
    const bw = Math.max(...lines.map((s) => ctx.measureText(s).width)) + 20 * dpr;
    const bh = lines.length * lh + 14 * dpr;
    let bx = x + 14 * dpr;
    if (bx + bw > W - pad.r) bx = x - bw - 14 * dpr;
    let by = Math.max(pad.t + 6 * dpr, Math.min(py - bh / 2, pad.t + g.ph - bh - 6 * dpr));
    ctx.fillStyle = C.dark ? 'rgba(10,16,30,0.92)' : 'rgba(255,255,255,0.94)';
    ctx.strokeStyle = C.lineStrong;
    roundRect(ctx, bx, by, bw, bh, 10 * dpr);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = C.fg;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    lines.forEach((s, i) => ctx.fillText(s, bx + 10 * dpr, by + 7 * dpr + lh * (i + 0.5)));
  }

  _data(ev) {
    const r = this.c.getBoundingClientRect();
    const g = this._geom();
    const px = ((ev.clientX - r.left) / r.width) * g.W;
    const py = ((ev.clientY - r.top) / r.height) * g.H;
    const x = g.x0 + ((px - g.pad.l) / g.pw) * (g.x1 - g.x0);
    const y = this.y0 + (1 - (py - g.pad.t) / g.ph) * (this.y1 - this.y0);
    return { x, y, px, py, g };
  }

  _bind() {
    const c = this.c;
    c.addEventListener('pointermove', (ev) => {
      if (!this.p) return;
      const d = this._data(ev);
      if (this.p.draw) {
        if (this.paint && ev.buttons) this._maybeAdd(d, true);
        return;
      }
      if (d.x >= d.g.x0 && d.x <= d.g.x1) this.hoverX = d.x;
      this.render();
    });
    c.addEventListener('pointerleave', () => {
      this.hoverX = null;
      this.paint = false;
      this.render();
    });
    c.addEventListener('pointerdown', (ev) => {
      if (!this.p) return;
      if (!this.p.draw) {
        const d = this._data(ev);
        this.hoverX = d.x;
        this.render();
        return;
      }
      c.setPointerCapture(ev.pointerId);
      const d = this._data(ev);
      const tr = this.p.train;
      if (tr) {
        for (let i = 0; i < tr.x.length; i++) {
          const dx = this._sx(d.g, tr.x[i]) - d.px;
          const dy = this._sy(d.g, tr.y[i]) - d.py;
          if (Math.hypot(dx, dy) < 14 * d.g.dpr) {
            this.onRemove && this.onRemove(i);
            this.paint = false;
            return;
          }
        }
      }
      this.paint = true;
      this._lastAdd = null;
      this._maybeAdd(d, false);
    });
    const up = (ev) => {
      this.paint = false;
      if (!this.p || this.p.draw) return;
      if (ev.pointerType === 'touch') {
        this.hoverX = null;
        this.render();
      }
    };
    c.addEventListener('pointerup', up);
    c.addEventListener('pointercancel', up);
  }

  _maybeAdd(d, dragging) {
    const p = this.p;
    if (d.x < p.range[0] || d.x > p.range[1]) return;
    if (this._lastAdd && dragging) {
      if (Math.hypot(d.px - this._lastAdd.px, d.py - this._lastAdd.py) < 12 * d.g.dpr) return;
    }
    this._lastAdd = { px: d.px, py: d.py };
    this.onAdd && this.onAdd(d.x, Math.max(this.y0, Math.min(this.y1, d.y)));
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const sup = (n) => '10' + String(n).split('').map((ch) => SUP[ch]).join('');

export class LossChart {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.train = [];
    this.test = [];
    this.total = 1;
    fitCanvas(canvas, () => this.render());
  }
  set(train, test, total) {
    this.train = train;
    this.test = test;
    this.total = Math.max(total, train.length, 1);
    this.render();
  }

  render() {
    const ctx = this.ctx;
    const { width: W, height: H } = this.c;
    const dpr = dprOf(this.c);
    const C = colors();
    ctx.clearRect(0, 0, W, H);
    const pad = { l: 46 * dpr, r: 12 * dpr, t: 10 * dpr, b: 26 * dpr };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;
    ctx.font = `${10 * dpr}px ${MONO}`;

    const all = [...this.train, ...this.test].filter((v) => v > 0 && Number.isFinite(v));
    let lo = -4;
    let hi = 0;
    if (all.length) {
      lo = Math.floor(Math.log10(Math.min(...all)) - 0.05);
      hi = Math.ceil(Math.log10(Math.max(...all)) + 0.05);
      if (hi - lo < 2) lo = hi - 2;
    }
    const sy = (v) => pad.t + (1 - (Math.log10(Math.max(v, 1e-12)) - lo) / (hi - lo)) * ph;
    const sx = (i) => pad.l + (i / Math.max(1, this.total - 1)) * pw;

    ctx.lineWidth = dpr;
    ctx.fillStyle = C.muted;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const maxLabels = Math.max(2, Math.floor(ph / (22 * dpr)));
    const stepD = Math.max(1, Math.ceil((hi - lo) / maxLabels));
    for (let d = lo; d <= hi; d += stepD) {
      const y = sy(Math.pow(10, d));
      ctx.strokeStyle = C.line;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(W - pad.r, y);
      ctx.stroke();
      ctx.fillText(sup(d), pad.l - 8 * dpr, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const tx = ticks(0, this.total, 4);
    for (const v of tx.out) ctx.fillText(String(Math.round(v)), sx(v), H - pad.b + 8 * dpr);
    ctx.textAlign = 'left';
    if (H / dpr > 150) ctx.fillText('эпоха', pad.l, H - 10 * dpr);

    const series = (arr, color, width, dash) => {
      if (!arr.length) return;
      const stride = Math.max(1, Math.floor(arr.length / (pw / (1.4 * dpr))));
      ctx.beginPath();
      ctx.lineWidth = width * dpr;
      ctx.strokeStyle = color;
      ctx.setLineDash(dash ? dash.map((v) => v * dpr) : []);
      ctx.lineJoin = 'round';
      let first = true;
      for (let s = 0; s < arr.length; s += stride) {
        let acc = 0;
        let cnt = 0;
        for (let j = s; j < Math.min(arr.length, s + stride); j++) {
          acc += Math.log(Math.max(arr[j], 1e-12));
          cnt++;
        }
        const v = Math.exp(acc / cnt);
        const x = sx(s + (cnt - 1) / 2);
        const y = sy(v);
        if (first) {
          ctx.moveTo(x, y);
          first = false;
        } else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, pw, ph);
    ctx.clip();
    series(this.test, C.warn, 1.6, [5, 4]); // тест – пунктиром, чтобы не полагаться только на цвет
    series(this.train, C.accent, 1.8);
    ctx.restore();
  }
}

export class NetViz {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.sizes = [1, 1];
    this.viz = null;
    this.act = 'relu';
    fitCanvas(canvas, () => this.render());
  }
  set(sizes, act, viz) {
    this.sizes = sizes;
    this.act = act;
    this.viz = viz;
    this.render();
  }
  render() {
    const ctx = this.ctx;
    const { width: W, height: H } = this.c;
    const dpr = dprOf(this.c);
    const C = colors();
    ctx.clearRect(0, 0, W, H);
    const sizes = this.sizes;
    const nL = sizes.length;
    const padX = 34 * dpr;
    const top = 26 * dpr;
    const bot = 14 * dpr;
    const colX = (i) => padX + (i / (nL - 1)) * (W - 2 * padX);
    const shown = sizes.map((s) => Math.min(s, 10));
    const nodeY = (i, j) => {
      const n = shown[i];
      const h = H - top - bot;
      return n === 1 ? top + h / 2 : top + (j / (n - 1)) * h;
    };
    if (this.viz) {
      let maxW = 1e-9;
      for (const L of this.viz) for (const v of L.w) maxW = Math.max(maxW, Math.abs(v));
      this.viz.forEach((L, l) => {
        if (L.nin !== shown[l] || L.nout !== shown[l + 1]) return;
        for (let j = 0; j < L.nout; j++) {
          for (let k = 0; k < L.nin; k++) {
            const w = L.w[j * L.nin + k];
            const a = Math.min(1, Math.abs(w) / maxW);
            ctx.strokeStyle = w >= 0 ? C.accent : C.warn;
            ctx.globalAlpha = 0.06 + 0.7 * a;
            ctx.lineWidth = (0.6 + 1.4 * a) * dpr;
            ctx.beginPath();
            ctx.moveTo(colX(l), nodeY(l, k));
            ctx.lineTo(colX(l + 1), nodeY(l + 1, j));
            ctx.stroke();
          }
        }
      });
      ctx.globalAlpha = 1;
    }
    ctx.font = `${10 * dpr}px ${MONO}`;
    ctx.textAlign = 'center';
    for (let i = 0; i < nL; i++) {
      const x = colX(i);
      for (let j = 0; j < shown[i]; j++) {
        const y = nodeY(i, j);
        ctx.beginPath();
        ctx.arc(x, y, (shown[i] > 8 ? 3.6 : 4.6) * dpr, 0, Math.PI * 2);
        ctx.fillStyle = C.surface;
        ctx.fill();
        ctx.lineWidth = 1.5 * dpr;
        ctx.strokeStyle = i === 0 || i === nL - 1 ? C.ok : C.fg;
        ctx.stroke();
      }
      ctx.fillStyle = C.muted;
      ctx.textBaseline = 'alphabetic';
      const label = i === 0 ? 'x' : i === nL - 1 ? 'ŷ' : String(sizes[i]);
      ctx.fillText(label, x, 14 * dpr);
    }
  }
}
