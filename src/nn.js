// Нейросеть: многослойный перцептрон 1 → … → 1, обратное распространение ошибки и оптимизатор Adam.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng) {
  // преобразование Бокса — Мюллера
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// x = linspace(a, b, n), y = f(x) + N(0, noise); dist = 'random' – точки случайно на отрезке
export function generateData(fn, range, n, noise, rng, dist = 'linspace') {
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const [a, b] = range;
  if (dist === 'random') {
    for (let i = 0; i < n; i++) x[i] = a + (b - a) * rng();
    x.sort();
  } else {
    for (let i = 0; i < n; i++) x[i] = n === 1 ? (a + b) / 2 : a + ((b - a) * i) / (n - 1);
  }
  for (let i = 0; i < n; i++) {
    y[i] = fn(x[i]);
    if (noise > 0) y[i] += gaussian(rng) * noise;
  }
  return { x, y };
}

export function linspace(a, b, n) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = a + ((b - a) * i) / (n - 1);
  return out;
}

export class MLP {
  constructor(hidden, activation = 'relu', rng = Math.random) {
    this.sizes = [1, ...hidden, 1];
    this.activation = activation;
    this.L = this.sizes.length - 1; // число линейных слоёв
    this.W = [];
    this.b = [];
    this.gW = [];
    this.gb = [];
    // состояние Adam
    this.mW = [];
    this.vW = [];
    this.mb = [];
    this.vb = [];
    this.t = 0;

    for (let l = 0; l < this.L; l++) {
      const nin = this.sizes[l];
      const nout = this.sizes[l + 1];
      const bound = 1 / Math.sqrt(nin); // инициализация U(−1/√nin, 1/√nin)
      const W = new Float64Array(nin * nout);
      const b = new Float64Array(nout);
      for (let i = 0; i < W.length; i++) W[i] = (rng() * 2 - 1) * bound;
      for (let i = 0; i < nout; i++) b[i] = (rng() * 2 - 1) * bound;
      this.W.push(W);
      this.b.push(b);
      this.gW.push(new Float64Array(W.length));
      this.gb.push(new Float64Array(nout));
      this.mW.push(new Float64Array(W.length));
      this.vW.push(new Float64Array(W.length));
      this.mb.push(new Float64Array(nout));
      this.vb.push(new Float64Array(nout));
    }
    this._cap = 0;
    this._acts = null;
    this._deltas = null;
  }

  get paramCount() {
    let n = 0;
    for (let l = 0; l < this.L; l++) n += this.W[l].length + this.b[l].length;
    return n;
  }

  _ensure(B) {
    if (this._cap >= B) return;
    this._cap = B;
    this._acts = this.sizes.map((s) => new Float64Array(B * s));
    this._deltas = this.sizes.map((s) => new Float64Array(B * s));
  }

  /** Прямой проход для батча из B примеров. */
  forward(X, B) {
    this._ensure(B);
    const acts = this._acts;
    const a0 = acts[0];
    for (let i = 0; i < B; i++) a0[i] = X[i];
    const act = this.activation;
    for (let l = 0; l < this.L; l++) {
      const nin = this.sizes[l];
      const nout = this.sizes[l + 1];
      const A = acts[l];
      const Z = acts[l + 1];
      const W = this.W[l];
      const bias = this.b[l];
      const hidden = l < this.L - 1;
      for (let i = 0; i < B; i++) {
        const ai = i * nin;
        const zi = i * nout;
        for (let j = 0; j < nout; j++) {
          let s = bias[j];
          const wj = j * nin;
          for (let k = 0; k < nin; k++) s += A[ai + k] * W[wj + k];
          if (hidden) {
            if (act === 'relu') s = s > 0 ? s : 0;
            else if (act === 'tanh') s = Math.tanh(s);
            else s = 1 / (1 + Math.exp(-s));
          }
          Z[zi + j] = s;
        }
      }
    }
    return acts[this.L];
  }

  predict(x, out = new Float64Array(x.length)) {
    const CH = 256;
    for (let s = 0; s < x.length; s += CH) {
      const B = Math.min(CH, x.length - s);
      const o = this.forward(x.subarray(s, s + B), B);
      for (let i = 0; i < B; i++) out[s + i] = o[i];
    }
    return out;
  }

  mse(x, y) {
    const p = this.predict(x);
    let s = 0;
    for (let i = 0; i < x.length; i++) {
      const d = p[i] - y[i];
      s += d * d;
    }
    return s / x.length;
  }

  /** Один шаг обучения: прямой проход, MSE, обратное распространение, Adam. Возвращает ошибку батча. */
  step(X, Y, B, lr) {
    const out = this.forward(X, B);
    const L = this.L;
    const deltas = this._deltas;
    const acts = this._acts;

    // MSE = Σ(ŷ−y)²/B и её производная 2(ŷ−y)/B
    let loss = 0;
    const dOut = deltas[L];
    for (let i = 0; i < B; i++) {
      const d = out[i] - Y[i];
      loss += d * d;
      dOut[i] = (2 * d) / B;
    }
    loss /= B;

    // обратное распространение ошибки
    const act = this.activation;
    for (let l = L - 1; l >= 0; l--) {
      const nin = this.sizes[l];
      const nout = this.sizes[l + 1];
      const A = acts[l];
      const D = deltas[l + 1];
      const Dprev = deltas[l];
      const W = this.W[l];
      const gW = this.gW[l];
      const gb = this.gb[l];
      gW.fill(0);
      gb.fill(0);
      const needPrev = l > 0;
      if (needPrev) Dprev.fill(0, 0, B * nin);
      for (let i = 0; i < B; i++) {
        const ai = i * nin;
        const di = i * nout;
        for (let j = 0; j < nout; j++) {
          const d = D[di + j];
          if (d === 0) continue;
          gb[j] += d;
          const wj = j * nin;
          if (needPrev) {
            for (let k = 0; k < nin; k++) {
              gW[wj + k] += d * A[ai + k];
              Dprev[ai + k] += d * W[wj + k];
            }
          } else {
            for (let k = 0; k < nin; k++) gW[wj + k] += d * A[ai + k];
          }
        }
        if (needPrev) {
          // производная функции активации предыдущего слоя
          if (act === 'relu') {
            for (let k = 0; k < nin; k++) if (A[ai + k] <= 0) Dprev[ai + k] = 0;
          } else if (act === 'tanh') {
            for (let k = 0; k < nin; k++) Dprev[ai + k] *= 1 - A[ai + k] * A[ai + k];
          } else {
            for (let k = 0; k < nin; k++) Dprev[ai + k] *= A[ai + k] * (1 - A[ai + k]);
          }
        }
      }
    }

    this._adam(lr);
    return loss;
  }

  _adam(lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.t++;
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    const stepSize = lr / bc1;
    const sbc2 = Math.sqrt(bc2);
    for (let l = 0; l < this.L; l++) {
      this._adamArr(this.W[l], this.gW[l], this.mW[l], this.vW[l], stepSize, sbc2, beta1, beta2, eps);
      this._adamArr(this.b[l], this.gb[l], this.mb[l], this.vb[l], stepSize, sbc2, beta1, beta2, eps);
    }
  }

  _adamArr(p, g, m, v, stepSize, sbc2, beta1, beta2, eps) {
    for (let i = 0; i < p.length; i++) {
      const gi = g[i];
      const mi = beta1 * m[i] + (1 - beta1) * gi;
      const vi = beta2 * v[i] + (1 - beta2) * gi * gi;
      m[i] = mi;
      v[i] = vi;
      p[i] -= (stepSize * mi) / (Math.sqrt(vi) / sbc2 + eps);
    }
  }
}

export class Trainer {
  constructor(model, train, { lr = 0.001, batchSize = 32, rng = Math.random } = {}) {
    this.model = model;
    this.x = train.x;
    this.y = train.y;
    this.lr = lr;
    this.batchSize = Math.max(1, batchSize);
    this.rng = rng;
    this.n = train.x.length;
    this.idx = new Uint32Array(this.n);
    for (let i = 0; i < this.n; i++) this.idx[i] = i;
    const cap = Math.min(this.batchSize, this.n);
    this.bx = new Float64Array(cap);
    this.by = new Float64Array(cap);
    this.epoch = 0;
  }

  /** Одна эпоха: перемешивание и мини-батчи. Возвращает среднюю ошибку по батчам. */
  trainEpoch() {
    const { idx, n, batchSize, rng } = this;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = idx[i];
      idx[i] = idx[j];
      idx[j] = t;
    }
    let total = 0;
    let batches = 0;
    for (let s = 0; s < n; s += batchSize) {
      const B = Math.min(batchSize, n - s);
      for (let i = 0; i < B; i++) {
        const k = idx[s + i];
        this.bx[i] = this.x[k];
        this.by[i] = this.y[k];
      }
      total += this.model.step(this.bx, this.by, B, this.lr);
      batches++;
    }
    this.epoch++;
    return total / batches;
  }
}
