import '@fontsource-variable/manrope';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';
import '@fontsource/jetbrains-mono/600.css';
import '@fontsource/roboto-condensed/300.css';
import '@fontsource/roboto-condensed/400.css';
import '@fontsource/tiny5/400.css';
import './style.css';

import { mulberry32, generateData, linspace } from './nn.js';
import { FUNCTIONS, MODE_CUSTOM, MODE_DRAW, compileExpression } from './functions.js';
import { RemoteTrainer } from './trainer.js';
import { FitPlot, LossChart, NetViz } from './plot.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const renderables = []; // всё, что надо перерисовать при смене темы

function fmtMse(v) {
  if (v == null || !Number.isFinite(v)) return '—';
  if (v >= 0.01) return v.toFixed(3);
  if (v >= 0.001) return v.toFixed(4);
  if (v >= 0.00001) return v.toFixed(5);
  return v.toExponential(1);
}
const debounce = (fn, ms) => {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
};
const fillRange = (el) => {
  const p = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.setProperty('--p', p + '%');
};
$$('input[type=range]').forEach((r) => {
  fillRange(r);
  r.addEventListener('input', () => fillRange(r));
});

// Тема и меню
const root = document.documentElement;
function setTheme(t) {
  root.dataset.theme = t;
  try { localStorage.setItem('approx-theme', t); } catch (e) { /* приватный режим */ }
  const btn = $('#themeBtn');
  btn.setAttribute('aria-label', t === 'dark' ? 'Включить светлую тему' : 'Включить тёмную тему');
  requestAnimationFrame(() => renderables.forEach((r) => r.render()));
}
$('#themeBtn').addEventListener('click', () => setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark'));
setTheme(root.dataset.theme);

const menuBtn = $('#menuBtn');
const nav = $('#nav');
const closeMenu = () => {
  nav.classList.remove('open');
  menuBtn.setAttribute('aria-expanded', 'false');
};
menuBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const open = nav.classList.toggle('open');
  menuBtn.setAttribute('aria-expanded', String(open));
});
nav.addEventListener('click', (e) => e.target.closest('a') && closeMenu());
document.addEventListener('click', (e) => !nav.contains(e.target) && closeMenu());
document.addEventListener('keydown', (e) => e.key === 'Escape' && closeMenu());

// Бегущие теги
$$('.track[data-chips]').forEach((tr) => {
  const chips = tr.dataset.chips.split('|');
  const set = chips.map((c) => `<span class="tag">${c}</span>`).join('');
  tr.innerHTML = set.repeat(4);
});

// Общие построители
function makeProblem({ fn, range, n, noise, dist, seed, drawPts, showExtrap }) {
  const [a, b] = range;
  const L = b - a;
  const ext = [a - 0.25 * L, b + 0.25 * L];
  const gridX = linspace(ext[0], ext[1], 600);
  let train;
  if (drawPts) {
    train = { x: Float64Array.from(drawPts.map((p) => p.x)), y: Float64Array.from(drawPts.map((p) => p.y)) };
  } else {
    train = generateData(fn, range, n, noise, mulberry32(seed), dist);
  }
  const trueY = fn ? gridX.map((x) => fn(x)) : null;
  let curve = null;
  if (fn) {
    const cx = linspace(a, b, Math.max(30, Math.min(100, train.x.length * 2)));
    curve = { x: cx, y: cx.map((x) => fn(x)) };
  }
  return {
    train,
    curve,
    grid: { x: gridX, trueY, lo: a, hi: b },
    view: { range: [a, b], ext, gridX, trueY, train, draw: !!drawPts, showExtrap },
  };
}

const paramCount = (hidden) => {
  const s = [1, ...hidden, 1];
  let n = 0;
  for (let i = 0; i < s.length - 1; i++) n += s[i] * s[i + 1] + s[i + 1];
  return n;
};
const archText = (hidden) => ['1', ...hidden, '1'].join(' → ');

// HERO: живая тренировка
(function hero() {
  const canvas = $('#heroCanvas');
  const plot = new FitPlot(canvas, { minimal: true, interactive: false });
  renderables.push(plot);
  const mseEl = $('#heroMse');
  const epochEl = $('#heroEpoch');
  const toggle = $('#heroToggle');
  let paused = false;
  let visible = true;
  let token = 0;
  let seed = 11;
  const fnDef = FUNCTIONS.f1;

  const trainer = new RemoteTrainer((s) => {
    if (s.pred) plot.setPred(s.pred);
    mseEl.textContent = fmtMse(s.testMse);
    epochEl.textContent = `эпоха ${s.epoch}`;
    if (s.done && !reduceMotion) {
      const t = ++token;
      setTimeout(() => t === token && !paused && restart(), 3200);
    }
  });

  function restart() {
    token++;
    seed++;
    const p = makeProblem({ fn: fnDef.fn, range: fnDef.range, n: 40, noise: 0.02, dist: 'linspace', seed, showExtrap: false });
    plot.setProblem(p.view);
    trainer.init({ cfg: { hidden: [32, 32], activation: 'relu', lr: 0.004, batchSize: 32, seed }, train: p.train, grid: p.grid, curve: null });
    if (!paused && visible) trainer.play(1000, reduceMotion ? 400 : 140);
  }
  toggle.addEventListener('click', () => {
    paused = !paused;
    toggle.setAttribute('aria-pressed', String(paused));
    toggle.setAttribute('aria-label', paused ? 'Продолжить живую демонстрацию' : 'Пауза живой демонстрации');
    if (paused) trainer.pause();
    else if (trainer.state.done) restart();
    else trainer.play(1000, 140);
  });
  new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (!visible) trainer.pause();
    else if (!paused && !trainer.state.done) trainer.play(1000, reduceMotion ? 400 : 140);
  }, { threshold: 0.05 }).observe($('.hero'));
  restart();
})();

// ЛАБОРАТОРИЯ
const PRESETS = {
  smooth: { fn: 'f1', n: 100, noise: 0.02, hidden: [128, 128, 64], act: 'relu', lr: 0.001, batch: 32, epochs: 1000 },
  kink: { fn: 'f2', n: 80, noise: 0.01, hidden: [64, 64, 64], act: 'relu', lr: 0.001, batch: 32, epochs: 800 },
  few: { fn: 'f3', n: 10, noise: 0.05, hidden: [64, 64, 64], act: 'relu', lr: 0.001, batch: 16, epochs: 1000 },
};
const SIZES = [2, 4, 8, 16, 32, 64, 128, 256];
const MAX_LAYERS = 6;

const lab = {
  fn: 'f1',
  range: [0, 2],
  n: 100,
  noise: 0.02,
  dist: 'linspace',
  hidden: [128, 128, 64],
  act: 'relu',
  lr: 0.001,
  batch: 32,
  epochs: 1000,
  speed: 300,
  extrap: true,
  dataSeed: 1,
  netSeed: 1,
  draw: [],
  customFn: compileExpression('sin(3*x) * x').fn,
  total: 0,
  lastMse: null,
};

const fitPlot = new FitPlot($('#fitCanvas'), {
  onAdd(x, y) {
    lab.draw.push({ x, y });
    refreshDrawView();
  },
  onRemove(i) {
    lab.draw.splice(i, 1);
    refreshDrawView();
  },
});
const lossChart = new LossChart($('#lossCanvas'));
const netViz = new NetViz($('#netCanvas'));
renderables.push(fitPlot, lossChart, netViz);

const trainer = new RemoteTrainer(onLabUpdate);
const runBtn = $('#runBtn');
const runLabel = $('#runLabel');

function labFn() {
  if (lab.fn === MODE_DRAW) return null;
  if (lab.fn === MODE_CUSTOM) return lab.customFn;
  return FUNCTIONS[lab.fn].fn;
}

function setRunState(state, label) {
  runBtn.dataset.state = state;
  runLabel.textContent = label;
}

let ready = true;
function apply() {
  clearTimeout(applyTimer);
  const isDraw = lab.fn === MODE_DRAW;
  const p = makeProblem({
    fn: labFn(),
    range: lab.range,
    n: lab.n,
    noise: lab.noise,
    dist: lab.dist,
    seed: lab.dataSeed,
    drawPts: isDraw ? lab.draw : null,
    showExtrap: lab.extrap,
  });
  fitPlot.setProblem(p.view);
  lossChart.set([], [], lab.epochs);
  lab.lastMse = null;
  $('#mseTrend').textContent = '';
  $('#labCard').classList.toggle('is-draw', isDraw);
  $('#mseLabel').textContent = isDraw ? 'Среднеквадратичная ошибка на ваших точках' : 'Среднеквадратичная ошибка на тесте';
  ready = !isDraw || lab.draw.length >= 3;
  $('#drawHint').hidden = !(isDraw && lab.draw.length < 3);
  if (!ready) {
    trainer.cancel();
    runBtn.disabled = true;
    setRunState('idle', 'Нужно 3+ точки');
    $('#labStatus').textContent = `Точек: ${lab.draw.length}`;
    netViz.set([1, ...lab.hidden, 1], lab.act, null);
    $('#mseBig').textContent = '—';
    $('#mseSub').textContent = 'поставьте точки на графике';
    return;
  }
  runBtn.disabled = false;
  lab.total = 0;
  trainer.init({
    cfg: { hidden: lab.hidden, activation: lab.act, lr: lab.lr, batchSize: lab.batch, seed: lab.netSeed },
    train: p.train,
    grid: p.grid,
    curve: p.curve,
  });
  setRunState('idle', 'Обучить');
  $('#labStatus').textContent = `Эпоха 0 / ${lab.epochs}`;
  $('#mseBig').textContent = '—';
  $('#mseSub').textContent = 'готово к обучению';
}
let applyTimer;
const applyLater = (ms = 140) => {
  clearTimeout(applyTimer);
  applyTimer = setTimeout(apply, ms);
};

function refreshDrawView() {
  const t = { x: Float64Array.from(lab.draw.map((p) => p.x)), y: Float64Array.from(lab.draw.map((p) => p.y)) };
  fitPlot.setTrain(t);
  $('#drawHint').hidden = lab.draw.length >= 3;
  $('#labStatus').textContent = `Точек: ${lab.draw.length}`;
  applyLater(350);
}

function onLabUpdate(s) {
  if (!ready) return;
  if (s.pred) fitPlot.setPred(s.pred);
  const total = s.total || lab.epochs;
  $('#labStatus').textContent = `Эпоха ${s.epoch} / ${total}`;
  lossChart.set(s.trainLoss, s.testLoss, total);
  const shown = s.testMse != null ? s.testMse : s.trainMse;
  if (s.epoch > 0) {
    $('#mseBig').textContent = fmtMse(shown);
    const tr = $('#mseTrend');
    if (lab.lastMse != null && Math.abs(shown - lab.lastMse) > 1e-12) {
      const better = shown < lab.lastMse;
      tr.textContent = better ? '↘' : '↗';
      tr.style.color = better ? 'var(--ok)' : 'var(--warn)';
    }
    lab.lastMse = shown;
    $('#mseSub').textContent = `train ${fmtMse(s.trainMse)} · ${(s.ms / 1000).toFixed(1)} с`;
  }
  netViz.set([1, ...lab.hidden, 1], lab.act, s.viz);
  if (s.running) setRunState('running', 'Пауза');
  else if (s.done) setRunState('done', 'Дообучить');
  else if (s.epoch > 0) setRunState('idle', 'Продолжить');
}

runBtn.addEventListener('click', () => {
  if (!ready) return;
  const s = trainer.state;
  if (s.running) {
    trainer.pause();
    return;
  }
  if (s.epoch === 0) lab.total = lab.epochs;
  else if (s.done || s.epoch >= lab.total) lab.total = s.epoch + lab.epochs;
  setRunState('running', 'Пауза');
  trainer.play(lab.total, lab.speed);
});
$('#resetBtn').addEventListener('click', () => {
  lab.netSeed++;
  apply();
});
$('#reshuffle').addEventListener('click', () => {
  lab.dataSeed++;
  apply();
});

// Панель настроек
const el = {
  fnSelect: $('#fnSelect'), nRange: $('#nRange'), nOut: $('#nOut'), noiseRange: $('#noiseRange'), noiseOut: $('#noiseOut'),
  x0: $('#x0'), x1: $('#x1'), epochRange: $('#epochRange'), epochOut: $('#epochOut'),
  batchSel: $('#batchSel'), lrSel: $('#lrSel'), speedSel: $('#speedSel'), extrapChk: $('#extrapChk'),
};

function syncFnInfo() {
  const isCustom = lab.fn === MODE_CUSTOM;
  const isDraw = lab.fn === MODE_DRAW;
  const def = FUNCTIONS[lab.fn];
  $('#fnFormula').textContent = isDraw ? 'Свои точки' : isCustom ? `f(x) = ${$('#exprInput').value}` : def.formula;
  $('#fnHint').textContent = isDraw ? 'Рисуйте на графике: сеть восстановит кривую по вашим точкам.' : isCustom ? 'Любая формула от x' : def.hint;
  $('#labCard .legend li:first-child').hidden = isDraw; // в режиме рисования истинной функции нет
  $('#customField').hidden = !isCustom;
  $('#drawField').hidden = !isDraw;
  $('#drawHint').hidden = !(isDraw && lab.draw.length < 3);
  $('#fitCanvas').parentElement.classList.toggle('drawing', isDraw);
  for (const id of ['nField', 'noiseField', 'distField', 'reshuffle']) $('#' + id).hidden = isDraw;
}

function renderLayers() {
  const ul = $('#layers');
  ul.innerHTML = '';
  lab.hidden.forEach((size, i) => {
    const li = document.createElement('li');
    li.className = 'layer';
    li.innerHTML = `<span class="mono">Слой ${i + 1}</span>
      <div class="stepper">
        <button class="mini" type="button" data-d="-1" aria-label="Меньше нейронов в слое ${i + 1}"><svg class="ic" viewBox="0 0 24 24"><path d="M5 12h14"/></svg></button>
        <output aria-live="polite">${size}</output>
        <button class="mini" type="button" data-d="1" aria-label="Больше нейронов в слое ${i + 1}"><svg class="ic" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg></button>
      </div>
      <button class="mini" type="button" data-rm="1" aria-label="Удалить слой ${i + 1}" ${lab.hidden.length === 1 ? 'disabled' : ''}><svg class="ic" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg></button>`;
    const idx = SIZES.indexOf(size);
    $$('[data-d]', li).forEach((b) => {
      const d = +b.dataset.d;
      if ((d < 0 && idx <= 0) || (d > 0 && idx >= SIZES.length - 1)) b.disabled = true;
      b.addEventListener('click', () => {
        lab.hidden[i] = SIZES[Math.max(0, Math.min(SIZES.length - 1, idx + d))];
        archChanged();
      });
    });
    $('[data-rm]', li).addEventListener('click', () => {
      lab.hidden.splice(i, 1);
      archChanged();
    });
    ul.appendChild(li);
  });
  $('#addLayer').disabled = lab.hidden.length >= MAX_LAYERS;
  const pc = paramCount(lab.hidden);
  $('#paramInfo').textContent = `${archText(lab.hidden)} · ${pc.toLocaleString('ru-RU')} парам.${pc > 150000 ? ' · тяжёлая сеть, обучение дольше' : ''}`;
  $('#netInfo').textContent = `${archText(lab.hidden)} · ${lab.act === 'relu' ? 'ReLU' : lab.act} · ${pc.toLocaleString('ru-RU')} парам.`;
  $$('[data-arch]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.arch === lab.hidden.join(','))));
}
function archChanged() {
  renderLayers();
  applyLater(200);
}

function syncUI() {
  el.fnSelect.value = lab.fn;
  el.nRange.value = lab.n; el.nOut.textContent = lab.n;
  el.noiseRange.value = lab.noise; el.noiseOut.textContent = lab.noise.toFixed(3).replace(/0$/, '');
  el.x0.value = lab.range[0]; el.x1.value = lab.range[1];
  el.epochRange.value = lab.epochs; el.epochOut.textContent = lab.epochs;
  el.batchSel.value = String(lab.batch);
  el.lrSel.value = String(lab.lr);
  el.speedSel.value = String(lab.speed);
  el.extrapChk.checked = lab.extrap;
  $$('[data-dist]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.dist === lab.dist)));
  $$('[data-act]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.act === lab.act)));
  $$('input[type=range]').forEach(fillRange);
  syncFnInfo();
  renderLayers();
}

el.fnSelect.addEventListener('change', () => {
  lab.fn = el.fnSelect.value;
  if (FUNCTIONS[lab.fn]) lab.range = [...FUNCTIONS[lab.fn].range];
  else lab.range = [0, 1];
  syncUI();
  apply();
});
el.nRange.addEventListener('input', () => { lab.n = +el.nRange.value; el.nOut.textContent = lab.n; applyLater(); });
el.noiseRange.addEventListener('input', () => { lab.noise = +el.noiseRange.value; el.noiseOut.textContent = lab.noise.toFixed(3).replace(/0$/, ''); applyLater(); });
const rangeInput = () => {
  let a = parseFloat(el.x0.value);
  let b = parseFloat(el.x1.value);
  if (!Number.isFinite(a) || !Number.isFinite(b) || b - a < 0.1) return;
  lab.range = [a, b];
  if (lab.fn === MODE_DRAW) lab.draw = lab.draw.filter((p) => p.x >= a && p.x <= b);
  applyLater(250);
};
el.x0.addEventListener('input', rangeInput);
el.x1.addEventListener('input', rangeInput);
el.epochRange.addEventListener('input', () => {
  lab.epochs = +el.epochRange.value;
  el.epochOut.textContent = lab.epochs;
  if (trainer.state.epoch === 0) $('#labStatus').textContent = `Эпоха 0 / ${lab.epochs}`;
});
el.batchSel.addEventListener('change', () => { lab.batch = +el.batchSel.value; applyLater(60); });
el.lrSel.addEventListener('change', () => { lab.lr = +el.lrSel.value; applyLater(60); });
el.speedSel.addEventListener('change', () => {
  lab.speed = +el.speedSel.value;
  if (trainer.state.running) trainer.setSpeed(lab.speed);
});
el.extrapChk.addEventListener('change', () => { lab.extrap = el.extrapChk.checked; fitPlot.setExtrap(lab.extrap); });
$$('[data-dist]').forEach((b) => b.addEventListener('click', () => {
  lab.dist = b.dataset.dist;
  $$('[data-dist]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  apply();
}));
$$('[data-act]').forEach((b) => b.addEventListener('click', () => {
  lab.act = b.dataset.act;
  $$('[data-act]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  renderLayers();
  applyLater(60);
}));
$$('[data-arch]').forEach((b) => b.addEventListener('click', () => {
  lab.hidden = b.dataset.arch.split(',').map(Number);
  archChanged();
}));
$('#addLayer').addEventListener('click', () => {
  lab.hidden.push(16);
  archChanged();
});

const exprInput = $('#exprInput');
const exprErr = $('#exprErr');
exprInput.addEventListener('input', debounce(() => {
  const r = compileExpression(exprInput.value);
  exprErr.hidden = !r.error;
  exprErr.textContent = r.error || '';
  exprInput.setAttribute('aria-invalid', String(!!r.error));
  if (r.fn) {
    lab.customFn = r.fn;
    syncFnInfo();
    apply();
  }
}, 350));

$('#drawClear').addEventListener('click', () => { lab.draw = []; refreshDrawView(); });
$('#drawSample').addEventListener('click', () => {
  const [a, b] = lab.range;
  const rng = mulberry32(5);
  lab.draw = Array.from({ length: 14 }, (_, i) => {
    const t = i / 13;
    return { x: a + (b - a) * t, y: 0.75 * Math.sin(2 * Math.PI * t * 1.3) * (1 - 0.35 * t) + (rng() - 0.5) * 0.12 };
  });
  refreshDrawView();
});

$$('[data-preset]').forEach((b) => b.addEventListener('click', () => {
  const p = PRESETS[b.dataset.preset];
  Object.assign(lab, p, { hidden: [...p.hidden], range: [...FUNCTIONS[p.fn].range], netSeed: lab.netSeed + 1 });
  syncUI();
  apply();
  trainer.play((lab.total = lab.epochs), lab.speed);
  setRunState('running', 'Пауза');
  $('#labCard').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
}));

syncUI();
apply();

// СРАВНЕНИЕ ТРЁХ СЕТЕЙ
const CMP_MODELS = [
  { name: 'Малая', hidden: [16, 16], epochs: 1000 },
  { name: 'Средняя', hidden: [64, 64, 64], epochs: 1000 },
  { name: 'Большая', hidden: [256, 256, 256, 256], epochs: 500 },
];
const cmp = { fn: 'f3', n: 10, noise: 0.05, seed: 1, cards: [], varY: 1 };
const VERDICT = {
  good: 'Хорошо',
  over: 'Переобучение',
  under: 'Недообучение',
  run: 'Обучается',
  idle: 'Готово к старту',
};

function buildCmpCards() {
  const grid = $('#cmpGrid');
  CMP_MODELS.forEach((m, i) => {
    const art = document.createElement('article');
    art.className = 'glass-card cmp-card';
    art.innerHTML = `
      <div class="cmp-head">
        <div><h3>${m.name} сеть</h3><span class="mono">${m.hidden.join('·')} · ${paramCount(m.hidden).toLocaleString('ru-RU')} парам. · ${m.epochs} эпох</span></div>
        <span class="badge" data-v="idle">${VERDICT.idle}</span>
      </div>
      <div class="plot"><canvas role="img" aria-label="Аппроксимация сетью «${m.name}»"></canvas></div>
      <div class="cmp-stats">
        <div class="stat"><span class="label mono">Train MSE</span><b data-s="train">—</b></div>
        <div class="stat"><span class="label mono">Test MSE</span><b data-s="test">—</b></div>
        <div class="stat"><span class="label mono">Эпоха</span><b data-s="epoch">0</b></div>
      </div>
      <div class="cmp-prog" aria-hidden="true"><i></i></div>
      <div class="mini-loss"><canvas role="img" aria-label="Кривые ошибки сети «${m.name}»"></canvas></div>`;
    grid.appendChild(art);
    const plot = new FitPlot($('.plot canvas', art), {});
    const loss = new LossChart($('.mini-loss canvas', art));
    renderables.push(plot, loss);
    const card = { m, art, plot, loss, badge: $('.badge', art), state: null };
    card.trainer = new RemoteTrainer((s) => cmpUpdate(i, s));
    cmp.cards.push(card);
  });
}

function verdict(train, test) {
  const nt = test / cmp.varY;
  const ntr = train / cmp.varY;
  if (ntr > 0.15) return 'under';
  if (nt < 0.03) return 'good';
  return nt > 4 * ntr ? 'over' : 'good';
}

function cmpUpdate(i, s) {
  const c = cmp.cards[i];
  if (s.pred) c.plot.setPred(s.pred);
  c.loss.set(s.trainLoss, s.testLoss, c.m.epochs);
  $('[data-s=train]', c.art).textContent = fmtMse(s.trainMse);
  $('[data-s=test]', c.art).textContent = s.testMse != null ? fmtMse(s.testMse) : '—';
  $('[data-s=epoch]', c.art).textContent = s.epoch;
  $('.cmp-prog i', c.art).style.width = (s.epoch / c.m.epochs) * 100 + '%';
  let v = s.epoch === 0 ? 'idle' : 'run';
  if (s.done) v = verdict(s.trainMse, s.testMse);
  c.badge.dataset.v = v;
  c.badge.textContent = VERDICT[v];
  c.running = s.running;
  c.done = s.done;
  updateCmpBtn();
}

function updateCmpBtn() {
  const any = cmp.cards.some((c) => c.running);
  const all = cmp.cards.every((c) => c.done);
  const started = cmp.cards.some((c) => c.trainer.state.epoch > 0);
  $('#cmpRun').dataset.state = any ? 'running' : 'idle';
  $('#cmpRunLabel').textContent = any ? 'Пауза' : all ? 'Заново' : started ? 'Продолжить' : 'Обучить все три';
}

function cmpApply() {
  const def = FUNCTIONS[cmp.fn];
  const p = makeProblem({ fn: def.fn, range: def.range, n: cmp.n, noise: cmp.noise, dist: 'linspace', seed: cmp.seed, showExtrap: false });
  // тест на 500 точках внутри отрезка
  const gx = linspace(def.range[0], def.range[1], 500);
  const gy = gx.map((x) => def.fn(x));
  const mean = gy.reduce((a, b) => a + b, 0) / gy.length;
  cmp.varY = Math.max(1e-6, gy.reduce((a, b) => a + (b - mean) ** 2, 0) / gy.length);
  const view = { range: def.range, ext: def.range, gridX: gx, trueY: gy, train: p.train, draw: false, showExtrap: false };
  const cx = linspace(def.range[0], def.range[1], Math.max(30, Math.min(100, cmp.n * 2)));
  const curveSet = { x: cx, y: cx.map((x) => def.fn(x)) };
  cmp.cards.forEach((c, i) => {
    c.running = false;
    c.done = false;
    c.plot.setProblem(view);
    c.loss.set([], [], c.m.epochs);
    c.trainer.init({
      cfg: { hidden: c.m.hidden, activation: 'relu', lr: 0.001, batchSize: 16, seed: 100 + i * 7 + cmp.seed },
      train: p.train,
      grid: { x: gx, trueY: gy, lo: def.range[0], hi: def.range[1] },
      curve: curveSet,
    });
    c.badge.dataset.v = 'idle';
    c.badge.textContent = VERDICT.idle;
    $('[data-s=train]', c.art).textContent = '—';
    $('[data-s=test]', c.art).textContent = '—';
    $('[data-s=epoch]', c.art).textContent = '0';
    $('.cmp-prog i', c.art).style.width = '0';
  });
  updateCmpBtn();
}
const cmpApplyLater = debounce(cmpApply, 160);

buildCmpCards();
cmpApply();

$('#cmpRun').addEventListener('click', () => {
  const any = cmp.cards.some((c) => c.running);
  if (any) {
    cmp.cards.forEach((c) => c.trainer.pause());
    return;
  }
  if (cmp.cards.every((c) => c.done)) cmpApply();
  cmp.cards.forEach((c) => {
    c.trainer.play(c.m.epochs, Infinity);
    c.running = true;
  });
  updateCmpBtn();
});
$('#cmpReset').addEventListener('click', () => {
  cmp.seed++;
  cmpApply();
});
$('#cmpFn').addEventListener('change', (e) => { cmp.fn = e.target.value; cmpApply(); });
$('#cmpN').addEventListener('input', (e) => { cmp.n = +e.target.value; $('#cmpNOut').textContent = cmp.n; cmpApplyLater(); });
$('#cmpNoise').addEventListener('input', (e) => { cmp.noise = +e.target.value; $('#cmpNoiseOut').textContent = cmp.noise.toFixed(3).replace(/0$/, ''); cmpApplyLater(); });
