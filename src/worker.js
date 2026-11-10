// Web Worker: обучение идёт вне основного потока, интерфейс не зависает.
import { MLP, Trainer, mulberry32 } from './nn.js';

let session = 0;
let model = null;
let trainer = null;
let job = null;
let running = false;
let epoch = 0;
let total = 0;
let speed = Infinity; // эпох в секунду
let pace = { t: 0, e: 0 };
let trainLossBuf = [];
let testLossBuf = [];
let lastPost = 0;
let evalCost = 5;
let timer = null;
let t0 = 0;
let elapsedBase = 0;
// тестовая ошибка считается не каждую эпоху, чтобы не замедлять обучение
let emaTrain = 1;
let emaTest = 0;
let sinceTest = 1e9;
let lastTest = NaN;

self.onmessage = (ev) => {
  const m = ev.data;
  switch (m.type) {
    case 'init': return init(m);
    case 'play': return play(m);
    case 'pause': return pause();
    case 'speed': speed = m.speed; resetPace(); return;
  }
};

function init(m) {
  running = false;
  clearTimeout(timer);
  session = m.session;
  job = m;
  const rng = mulberry32(m.cfg.seed);
  model = new MLP(m.cfg.hidden, m.cfg.activation, rng);
  trainer = new Trainer(model, m.train, { lr: m.cfg.lr, batchSize: m.cfg.batchSize, rng });
  epoch = 0;
  total = 0;
  emaTrain = 1;
  emaTest = 0;
  sinceTest = 1e9;
  lastTest = NaN;
  trainLossBuf = [];
  testLossBuf = [];
  elapsedBase = 0;
  post(true);
}

function play(m) {
  if (!model) return;
  total = m.total;
  if (m.speed !== undefined) speed = m.speed;
  if (epoch >= total) { post(true); return; }
  running = true;
  t0 = performance.now();
  resetPace();
  clearTimeout(timer);
  timer = setTimeout(loop, 0);
}

function pause() {
  if (running) elapsedBase += performance.now() - t0;
  running = false;
  clearTimeout(timer);
  if (model) post(true);
}

function resetPace() {
  pace = { t: performance.now(), e: epoch };
}

function loop() {
  if (!running) return;
  const start = performance.now();
  const budget = 24;
  while (epoch < total) {
    const now = performance.now();
    if (now - start > budget) break;
    if (speed !== Infinity && epoch >= pace.e + ((now - pace.t) * speed) / 1000 + 1) break;
    const ta = performance.now();
    const l = trainer.trainEpoch();
    emaTrain = 0.9 * emaTrain + 0.1 * (performance.now() - ta);
    epoch++;
    trainLossBuf.push(l);
    if (job.curve) {
      sinceTest++;
      if (epoch <= 3 || sinceTest * emaTrain >= 2 * emaTest) {
        const tb = performance.now();
        lastTest = model.mse(job.curve.x, job.curve.y);
        emaTest = 0.8 * emaTest + 0.2 * (performance.now() - tb);
        sinceTest = 0;
      }
      testLossBuf.push(lastTest);
    }
  }
  const now = performance.now();
  const done = epoch >= total;
  if (done) {
    running = false;
    elapsedBase += now - t0;
  }
  // предсказание для графика – не чаще, чем позволяет его стоимость
  if (done || now - lastPost >= Math.max(33, evalCost * 4)) post(done);
  if (running) timer = setTimeout(loop, speed === Infinity ? 0 : 8);
}

function post(final) {
  const s = performance.now();
  const g = job.grid;
  const pred = model.predict(g.x, new Float64Array(g.x.length));
  let testMse = null;
  if (g.trueY) {
    let acc = 0;
    let cnt = 0;
    for (let i = 0; i < g.x.length; i++) {
      if (g.x[i] >= g.lo && g.x[i] <= g.hi) {
        const d = pred[i] - g.trueY[i];
        acc += d * d;
        cnt++;
      }
    }
    testMse = cnt ? acc / cnt : null;
  }
  const trainMse = model.mse(job.train.x, job.train.y);
  const viz = vizSnapshot();
  evalCost = performance.now() - s;
  lastPost = performance.now();
  const msg = {
    type: 'progress',
    session,
    epoch,
    total,
    running,
    done: epoch > 0 && epoch >= total && !running,
    trainLoss: trainLossBuf,
    testLoss: testLossBuf,
    pred,
    trainMse,
    testMse,
    viz,
    ms: elapsedBase + (running ? performance.now() - t0 : 0),
  };
  trainLossBuf = [];
  testLossBuf = [];
  self.postMessage(msg, [pred.buffer]);
}

// срез весов для схемы сети (не больше MAXN узлов на слой)
const MAXN = 10;
function pick(n) {
  if (n <= MAXN) return Array.from({ length: n }, (_, i) => i);
  return Array.from({ length: MAXN }, (_, i) => Math.round((i * (n - 1)) / (MAXN - 1)));
}
function vizSnapshot() {
  const layers = [];
  for (let l = 0; l < model.L; l++) {
    const nin = model.sizes[l];
    const nout = model.sizes[l + 1];
    const pi = pick(nin);
    const po = pick(nout);
    const w = new Float32Array(pi.length * po.length);
    for (let j = 0; j < po.length; j++) {
      for (let k = 0; k < pi.length; k++) w[j * pi.length + k] = model.W[l][po[j] * nin + pi[k]];
    }
    layers.push({ nin: pi.length, nout: po.length, w });
  }
  return layers;
}
