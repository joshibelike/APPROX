// Обёртка над Web Worker: хранит историю ошибок и последнее предсказание.

export class RemoteTrainer {
  constructor(onUpdate) {
    this.onUpdate = onUpdate;
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.session = 0;
    this.state = this._blank();
    this.worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type !== 'progress' || m.session !== this.session) return;
      const s = this.state;
      s.epoch = m.epoch;
      s.total = m.total;
      s.running = m.running;
      s.done = m.done;
      for (const v of m.trainLoss) s.trainLoss.push(v);
      for (const v of m.testLoss) s.testLoss.push(v);
      s.pred = m.pred;
      s.trainMse = m.trainMse;
      s.testMse = m.testMse;
      s.viz = m.viz;
      s.ms = m.ms;
      this.onUpdate(s);
    };
  }

  _blank() {
    return { epoch: 0, total: 0, running: false, done: false, trainLoss: [], testLoss: [], pred: null, trainMse: null, testMse: null, viz: null, ms: 0 };
  }

  /** job: { cfg:{hidden,activation,lr,batchSize,seed}, train:{x,y}, grid:{x,trueY,lo,hi}, curve:{x,y}|null } */
  init(job) {
    this.session++;
    this.state = this._blank();
    this.worker.postMessage({ type: 'init', session: this.session, ...job });
  }
  /** Остановить и забыть текущую сессию. */
  cancel() {
    this.session++;
    this.state = this._blank();
    this.worker.postMessage({ type: 'pause' });
  }
  play(total, speed = Infinity) {
    this.worker.postMessage({ type: 'play', total, speed });
  }
  pause() {
    this.worker.postMessage({ type: 'pause' });
  }
  setSpeed(speed) {
    this.worker.postMessage({ type: 'speed', speed });
  }
  dispose() {
    this.worker.terminate();
  }
}
