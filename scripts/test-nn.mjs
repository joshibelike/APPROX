// Проверка градиентов и контрольные эксперименты.
import { MLP, Trainer, mulberry32, generateData, linspace } from '../src/nn.js';

const f1 = (x) => Math.sin(2 * Math.PI * x) * Math.exp(-x);
const f2 = (x) => Math.abs(x - 0.5);
const f3 = (x) => Math.sin(5 * Math.PI * x) + 0.3 * Math.cos(15 * Math.PI * x);

// численная проверка градиента
for (const act of ['relu', 'tanh', 'sigmoid']) {
  const rng = mulberry32(7);
  const net = new MLP([6, 5], act, rng);
  const X = Float64Array.from([0.1, 0.4, 0.8, 1.3]);
  const Y = Float64Array.from([0.3, -0.2, 0.5, 0.1]);
  const lossAt = () => {
    const o = net.forward(X, 4);
    let s = 0;
    for (let i = 0; i < 4; i++) s += (o[i] - Y[i]) ** 2;
    return s / 4;
  };
  // считаем градиенты шагом с lr=0 (Adam при lr=0 не двигает параметры)
  net.step(X, Y, 4, 0);
  let maxErr = 0;
  for (let l = 0; l < net.L; l++) {
    for (let i = 0; i < net.W[l].length; i += 3) {
      const old = net.W[l][i];
      const h = 1e-6;
      net.W[l][i] = old + h;
      const lp = lossAt();
      net.W[l][i] = old - h;
      const lm = lossAt();
      net.W[l][i] = old;
      const num = (lp - lm) / (2 * h);
      maxErr = Math.max(maxErr, Math.abs(num - net.gW[l][i]));
    }
  }
  console.log(`grad-check ${act}: max |Δ| = ${maxErr.toExponential(2)}`, maxErr < 1e-6 ? 'OK' : 'FAIL');
  if (maxErr >= 1e-6) process.exitCode = 1;
}

// контрольные эксперименты
function run(name, fn, range, n, noise, hidden, epochs, bs, seed = 1) {
  const rng = mulberry32(seed);
  const data = generateData(fn, range, n, noise, rng);
  const net = new MLP(hidden, 'relu', rng);
  const tr = new Trainer(net, data, { lr: 0.001, batchSize: bs, rng });
  const t0 = performance.now();
  let last = 0;
  for (let e = 0; e < epochs; e++) last = tr.trainEpoch();
  const tx = linspace(range[0], range[1], 500);
  const ty = tx.map(fn);
  const mse = net.mse(tx, ty);
  console.log(`${name}: train(loss) = ${last.toExponential(2)}  test MSE = ${mse.toFixed(6)}  (${((performance.now() - t0) / 1000).toFixed(1)} c)`);
  return mse;
}

run('Пример 1  f₁ 100 точек [128,128,64] 1000 эпох', f1, [0, 2], 100, 0.02, [128, 128, 64], 1000, 32);
run('Пример 3  f₂  80 точек [64,64,64]   800 эпох', f2, [0, 1], 80, 0.01, [64, 64, 64], 800, 32);
run('Переобуч.  f₃  10 точек [16,16]      1000 эпох', f3, [0, 1], 10, 0.05, [16, 16], 1000, 16);
run('Переобуч.  f₃  10 точек [64,64,64]   1000 эпох', f3, [0, 1], 10, 0.05, [64, 64, 64], 1000, 16);
run('Переобуч.  f₃  10 точек [256]x4       500 эпох', f3, [0, 1], 10, 0.05, [256, 256, 256, 256], 500, 16);
