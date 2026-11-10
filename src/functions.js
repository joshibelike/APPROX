// Тестовые функции: f1–f3 из работы, остальные добавлены для примера.

export const FUNCTIONS = {
  f1: {
    id: 'f1',
    name: 'Гладкая',
    tag: 'f₁',
    formula: 'sin(2πx)·e⁻ˣ',
    hint: 'Гладкая нелинейная функция с непрерывной производной',
    fn: (x) => Math.sin(2 * Math.PI * x) * Math.exp(-x),
    range: [0, 2],
  },
  f2: {
    id: 'f2',
    name: 'Излом',
    tag: 'f₂',
    formula: '|x − 0.5|',
    hint: 'Недифференцируема в точке x = 0.5',
    fn: (x) => Math.abs(x - 0.5),
    range: [0, 1],
  },
  f3: {
    id: 'f3',
    name: 'Высокая частота',
    tag: 'f₃',
    formula: 'sin(5πx) + 0.3·cos(15πx)',
    hint: 'Частые колебания — проверка мелких деталей',
    fn: (x) => Math.sin(5 * Math.PI * x) + 0.3 * Math.cos(15 * Math.PI * x),
    range: [0, 1],
  },
  gauss: {
    id: 'gauss',
    name: 'Гауссиана',
    tag: 'e',
    formula: 'e^(−((x−0.5)/0.12)²)',
    hint: 'Узкий гладкий «колокол»',
    fn: (x) => Math.exp(-(((x - 0.5) / 0.12) ** 2)),
    range: [0, 1],
  },
  step: {
    id: 'step',
    name: 'Ступенька',
    tag: 'H',
    formula: 'x < 0.5 ? 0 : 1',
    hint: 'Разрыв первого рода — сеть вынуждена «сгладить» скачок',
    fn: (x) => (x < 0.5 ? 0 : 1),
    range: [0, 1],
  },
  poly: {
    id: 'poly',
    name: 'Кубика',
    tag: 'x³',
    formula: 'x³ − x',
    hint: 'Полином — удобно смотреть экстраполяцию',
    fn: (x) => x * x * x - x,
    range: [-1.5, 1.5],
  },
};

export const MODE_CUSTOM = 'custom';
export const MODE_DRAW = 'draw';

const ALLOWED = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan, exp: Math.exp, abs: Math.abs, sqrt: Math.sqrt,
  log: Math.log, ln: Math.log, tanh: Math.tanh, sign: Math.sign, floor: Math.floor, min: Math.min,
  max: Math.max, pow: Math.pow, atan: Math.atan, asin: Math.asin, acos: Math.acos, cosh: Math.cosh,
  sinh: Math.sinh, round: Math.round, ceil: Math.ceil,
};
const CONSTS = { pi: Math.PI, e: Math.E };

/** Возвращает { fn } или { error }. Допустимы: x, числа, + − * / ^ ( ) , и функции из ALLOWED. */
export function compileExpression(src) {
  const raw = String(src || '').trim();
  if (!raw) return { error: 'Введите формулу' };
  if (raw.length > 200) return { error: 'Слишком длинная формула' };
  let s = raw.replace(/π/g, 'pi').replace(/,/g, '.').replace(/\^/g, '**').replace(/×/g, '*').replace(/−/g, '-');
  // «;» – разделитель аргументов (запятая здесь десятичная)
  s = s.replace(/;/g, ',');
  if (!/^[0-9a-zA-Z_+\-*/().,\s*]*$/.test(s)) return { error: 'Недопустимые символы' };
  const idents = s.match(/[a-zA-Z_][a-zA-Z_0-9]*/g) || [];
  for (const id of idents) {
    const k = id.toLowerCase();
    if (k !== 'x' && !(k in ALLOWED) && !(k in CONSTS)) return { error: `Неизвестное имя: ${id}` };
  }
  s = s.replace(/\bln\b/gi, 'log');
  try {
    const names = [...Object.keys(ALLOWED), ...Object.keys(CONSTS)];
    const vals = [...Object.values(ALLOWED), ...Object.values(CONSTS)];
    const f = new Function(...names, 'x', `"use strict"; return (${s});`);
    const fn = (x) => f(...vals, x);
    const probe = fn(0.5);
    if (typeof probe !== 'number') return { error: 'Формула должна возвращать число' };
    return { fn: (x) => {
      const v = f(...vals, x);
      return Number.isFinite(v) ? v : 0;
    } };
  } catch (e) {
    return { error: 'Ошибка в формуле (для минус-квадрата пишите 0 - x^2)' };
  }
}
