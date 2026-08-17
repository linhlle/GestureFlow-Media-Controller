// Measure the browser demo's real frame behaviour, with and without a hand.
//
// This drives the actual demo page in a real Chrome against a fake camera
// device, rather than re-implementing the loop in a test harness. That matters:
// the bug being measured here is about *when* work happens on the main thread,
// and a re-implementation would have quietly reproduced whichever scheduling the
// author of the harness had in mind.
//
// Instrumentation is injected from outside, so the page's own source is
// untouched and the before/after runs measure the same thing:
//
//   requestAnimationFrame  wrapped -> per-callback main-thread duration
//   detectForVideo         wrapped -> MediaPipe solve time
//   Recognizer.process     wrapped -> forest + FSM time
//   drawImage(video, ...)  wrapped -> when a camera frame actually reaches the
//                                     screen, which is what "smooth" means
//   PerformanceObserver    longtask -> main-thread blocking
//
// ES modules are singletons per URL, so importing the same MediaPipe and
// recognizer URLs the page imports hands back the very objects it will use.
// Patching their prototypes before the camera starts instruments the real
// instances.
//
// Usage:
//   node scripts/webbench.mjs --y4m <dir> --label before --out <dir> [--keep-open]

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';

const REPO = normalize(join(dirname(fileURLToPath(import.meta.url)), '..'));
const WEB = join(REPO, 'web');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const MEASURE_MS = 9000;   // per condition, after warmup
const WARMUP_MS = 3500;    // model load, camera spin-up, tracker lock-on

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml',
};

// ---------------------------------------------------------------------------
// Static server
// ---------------------------------------------------------------------------

function serve(root) {
  return new Promise((resolve) => {
    const server = createServer(async (req, res) => {
      let p = decodeURIComponent(req.url.split('?')[0]);
      if (p === '/') p = '/index.html';
      // cleanUrls, as vercel.json configures in production.
      let file = join(root, normalize(p).replace(/^(\.\.[/\\])+/, ''));
      if (!existsSync(file) && existsSync(`${file}.html`)) file = `${file}.html`;
      try {
        const body = await readFile(file);
        res.writeHead(200, {
          'Content-Type': MIME[extname(file)] || 'application/octet-stream',
          'Cache-Control': 'no-store',
        });
        res.end(body);
      } catch {
        res.writeHead(404).end('not found');
      }
    });
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

// ---------------------------------------------------------------------------
// Minimal CDP client
// ---------------------------------------------------------------------------

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.logs = [];
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id !== undefined) {
        const p = this.pending.get(msg.id);
        if (!p) return;
        this.pending.delete(msg.id);
        if (msg.error) p.reject(new Error(JSON.stringify(msg.error)));
        else p.resolve(msg.result);
      } else if (msg.method === 'Runtime.consoleAPICalled') {
        this.logs.push({
          level: msg.params.type,
          text: msg.params.args
            .map((a) => a.value ?? a.description ?? a.type).join(' '),
        });
      } else if (msg.method === 'Runtime.exceptionThrown') {
        const d = msg.params.exceptionDetails;
        this.logs.push({
          level: 'exception',
          text: d.exception?.description || d.text,
        });
      }
    });
  }

  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async eval(expression, awaitPromise = true) {
    const r = await this.send('Runtime.evaluate', {
      expression, awaitPromise, returnByValue: true,
    });
    if (r.exceptionDetails) {
      throw new Error(r.exceptionDetails.exception?.description
                      || r.exceptionDetails.text);
    }
    return r.result.value;
  }
}

// Chrome writes the port it actually bound to here. Asking for a fixed port and
// assuming you got it is how a leftover browser from a previous run ends up
// being measured instead of the one just launched.
async function devtoolsPort(profile) {
  for (let i = 0; i < 120; i++) {
    try {
      const txt = await readFile(join(profile, 'DevToolsActivePort'), 'utf8');
      const port = parseInt(txt.split('\n')[0], 10);
      if (port > 0) return port;
    } catch { /* not written yet */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error('Chrome never reported a DevTools port');
}

async function connect(port) {
  for (let i = 0; i < 120; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/list`);
      const targets = await res.json();
      const page = targets.find((t) => t.type === 'page');
      if (page) {
        const ws = new WebSocket(page.webSocketDebuggerUrl);
        await new Promise((ok, no) => {
          ws.addEventListener('open', ok, { once: true });
          ws.addEventListener('error', no, { once: true });
        });
        return new CDP(ws);
      }
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error('could not attach to Chrome');
}

// ---------------------------------------------------------------------------
// Injected instrumentation. Runs before any page script.
// ---------------------------------------------------------------------------

const PROBE = `
(() => {
  const B = {
    raf: [],        // [startMs, durationMs]
    infer: [],      // [startMs, durationMs]
    recog: [],      // [startMs, durationMs]
    present: [],    // ms at which a camera frame was blitted to a canvas
    longtasks: [],  // [startMs, durationMs]
    patched: { infer: false, recog: false },
  };
  window.__bench = B;

  const rafOrig = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => rafOrig((t) => {
    const t0 = performance.now();
    try { return cb(t); } finally { B.raf.push([t0, performance.now() - t0]); }
  });

  const di = CanvasRenderingContext2D.prototype.drawImage;
  CanvasRenderingContext2D.prototype.drawImage = function (src, ...rest) {
    // Only a video blit counts as "a camera frame reached the screen".
    if (typeof HTMLVideoElement !== 'undefined' && src instanceof HTMLVideoElement) {
      B.present.push(performance.now());
    }
    return di.call(this, src, ...rest);
  };

  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) B.longtasks.push([e.startTime, e.duration]);
    }).observe({ type: 'longtask', buffered: true });
  } catch {}

  // Truncate in place. The detectForVideo/process wrappers close over the array
  // they were handed, so reassigning here would leave them pushing into an
  // orphan and report an inference rate of zero on a page that is plainly
  // inferring.
  B.reset = () => {
    for (const k of ['raf', 'infer', 'recog', 'present', 'longtasks']) B[k].length = 0;
  };
})();
`;

const PATCH = (mpUrl, breakRecognizer) => `
(async () => {
  const B = window.__bench;
  const wrap = (obj, name, bucket) => {
    if (!obj || typeof obj[name] !== 'function') return false;
    const orig = obj[name];
    obj[name] = function (...args) {
      const t0 = performance.now();
      try { return orig.apply(this, args); }
      finally { bucket.push([t0, performance.now() - t0]); }
    };
    return true;
  };
  const mp = await import(${JSON.stringify(mpUrl)});
  B.patched.infer = wrap(mp.HandLandmarker.prototype, 'detectForVideo', B.infer);
  const rec = await import('/js/recognizer.js');
  B.patched.recog = wrap(rec.Recognizer.prototype, 'process', B.recog);
  ${breakRecognizer ? `
  // Fault injection: make the recognizer throw the same way the undeclared
  // swipeArmed did. A loop that survives this keeps presenting camera frames;
  // a loop that reschedules on its last line stops dead. This is the assertion
  // the fix exists to satisfy, run against the real page.
  //
  // Armed by frame count, not by wall clock: the harness will not start
  // measuring until the page has painted, so a fault that fires too early just
  // looks like a page that never started, and one that fires too late is not
  // in the measurement window at all.
  let framesLeft = 100;
  B.injected = 0;
  const inner = rec.Recognizer.prototype.process;
  rec.Recognizer.prototype.process = function (...args) {
    if (framesLeft-- <= 0) {
      B.injected += 1;
      throw new ReferenceError('swipeArmed is not defined');
    }
    return inner.apply(this, args);
  };
  ` : ''}
  return B.patched;
})()
`;

const ENV_PROBE = `
(() => {
  const out = {};
  const c = document.createElement('canvas');
  const gl = c.getContext('webgl2') || c.getContext('webgl');
  if (gl) {
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    out.webglVersion = gl.getParameter(gl.VERSION);
    out.renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '(masked)';
    out.vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : '(masked)';
  } else {
    out.renderer = '(no webgl)';
  }
  // WASM SIMD: v128 const + local. WASM threads: shared memory.
  out.wasmSimd = WebAssembly.validate(new Uint8Array([
    0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11
  ]));
  try { new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true }); out.wasmThreads = true; }
  catch { out.wasmThreads = false; }
  out.crossOriginIsolated = self.crossOriginIsolated;
  out.hardwareConcurrency = navigator.hardwareConcurrency;
  out.deviceMemory = navigator.deviceMemory ?? null;
  out.wasmAssets = performance.getEntriesByType('resource')
    .map((e) => e.name).filter((n) => /wasm|\\.task$/.test(n));
  return out;
})()
`;

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

const pct = (sorted, p) => (sorted.length
  ? sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))]
  : null);

function summarize(pairs) {
  const d = pairs.map((x) => x[1]).sort((a, b) => a - b);
  if (!d.length) return { n: 0 };
  return {
    n: d.length,
    p50: +pct(d, 50).toFixed(2),
    p95: +pct(d, 95).toFixed(2),
    max: +d[d.length - 1].toFixed(2),
    mean: +(d.reduce((a, b) => a + b, 0) / d.length).toFixed(2),
  };
}

function rate(times, windowMs) {
  return times.length ? +(times.length / (windowMs / 1000)).toFixed(1) : 0;
}

function gaps(times) {
  const g = [];
  for (let i = 1; i < times.length; i++) g.push(times[i] - times[i - 1]);
  g.sort((a, b) => a - b);
  if (!g.length) return { n: 0 };
  return {
    n: g.length,
    p50: +pct(g, 50).toFixed(1),
    p95: +pct(g, 95).toFixed(1),
    max: +g[g.length - 1].toFixed(1),
  };
}

// ---------------------------------------------------------------------------
// One condition = one Chrome, because the fake camera file is a launch flag.
// ---------------------------------------------------------------------------

async function runCondition({ y4m, page, breakRecognizer }) {
  const profile = join(tmpdir(), `gfbench-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const chrome = spawn(CHROME, [
    '--remote-debugging-port=0',
    `--user-data-dir=${profile}`,
    '--no-first-run', '--no-default-browser-check', '--no-sandbox',
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    `--use-file-for-fake-video-capture=${y4m}`,
    '--autoplay-policy=no-user-gesture-required',
    // Without these Chrome treats the window as occluded and stops servicing
    // requestAnimationFrame entirely, so the page under test never renders a
    // single frame and every number comes back zero.
    '--disable-backgrounding-occluded-windows',
    '--disable-renderer-backgrounding',
    '--disable-background-timer-throttling',
    '--disable-features=CalculateNativeWinOcclusion',
    '--window-size=1280,900',
    'about:blank',
  ], { stdio: 'ignore' });

  let cdp;
  try {
    cdp = await connect(await devtoolsPort(profile));
    await cdp.send('Runtime.enable');
    await cdp.send('Page.enable');
    await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: PROBE });
    await cdp.send('Page.navigate', { url: page });
    await cdp.send('Page.bringToFront').catch(() => {});
    await new Promise((r) => setTimeout(r, 1500));

    const mpUrl = await cdp.eval(`
      (async () => {
        const src = await (await fetch('/js/demo.js')).text();
        return src.match(/from\\s+'([^']*tasks-vision[^']*)'/)[1];
      })()
    `);
    const patched = await cdp.eval(PATCH(mpUrl, breakRecognizer));
    if (!patched.infer || !patched.recog) {
      throw new Error(`instrumentation failed to attach: ${JSON.stringify(patched)}`);
    }

    await cdp.eval(`document.getElementById('start-btn').click(); true`, false);

    // Wait for the loop to actually be running rather than trusting a fixed
    // sleep: the first Start downloads the WASM runtime and the .task model
    // over the network and compiles them, which is far slower and far more
    // variable than anything being measured here.
    let ready = false;
    for (let i = 0; i < 240; i++) {
      await new Promise((r) => setTimeout(r, 500));
      const n = await cdp.eval('window.__bench.present.length');
      if (n > 30) { ready = true; break; }
    }
    if (!ready) {
      const state = await cdp.eval(`({
        overlay: document.getElementById('overlay-text').textContent,
        hint: document.getElementById('overlay-hint').textContent,
        hidden: document.getElementById('overlay').classList.contains('hidden'),
        readyState: document.getElementById('video').readyState,
        vw: document.getElementById('video').videoWidth,
        present: window.__bench.present.length,
        raf: window.__bench.raf.length,
        infer: window.__bench.infer.length,
      })`);
      throw new Error(`demo never started: ${JSON.stringify(state)}\n`
                      + `console: ${JSON.stringify(cdp.logs.slice(-15))}`);
    }
    await new Promise((r) => setTimeout(r, WARMUP_MS));

    // Discard warmup: model compile and tracker lock-on are one-time costs and
    // would otherwise dominate the percentiles.
    await cdp.eval('window.__bench.reset(); true', false);
    const t0 = await cdp.eval('performance.now()');
    await new Promise((r) => setTimeout(r, MEASURE_MS));
    const t1 = await cdp.eval('performance.now()');

    const b = await cdp.eval(`(() => {
      const B = window.__bench;
      return { raf: B.raf, infer: B.infer, recog: B.recog,
               present: B.present, longtasks: B.longtasks };
    })()`);

    const env = await cdp.eval(ENV_PROBE);
    const injected = await cdp.eval('window.__bench.injected ?? 0');
    const handSeen = await cdp.eval(`
      document.getElementById('r-pose').textContent
    `);
    const window_ms = t1 - t0;

    const blocking = b.longtasks.reduce((a, [, d]) => a + Math.max(0, d - 50), 0);

    return {
      env,
      injected_faults: injected,
      poseReadout: handSeen,
      window_ms: +window_ms.toFixed(0),
      presented_fps: rate(b.present, window_ms),
      inference_fps: rate(b.infer, window_ms),
      raf_callbacks_per_s: rate(b.raf, window_ms),
      present_gap_ms: gaps(b.present),
      mediapipe_solve_ms: summarize(b.infer),
      recognizer_ms: summarize(b.recog),
      raf_callback_ms: summarize(b.raf),
      longtasks: {
        count: b.longtasks.length,
        total_blocking_ms: +blocking.toFixed(0),
        worst_ms: b.longtasks.length
          ? +Math.max(...b.longtasks.map((x) => x[1])).toFixed(1) : 0,
      },
      console: cdp.logs.slice(-25),
    };
  } finally {
    try { cdp?.ws.close(); } catch {}
    chrome.kill('SIGTERM');
    await new Promise((r) => setTimeout(r, 400));
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
}

// ---------------------------------------------------------------------------

async function main() {
  const args = Object.fromEntries(
    process.argv.slice(2).reduce((acc, a, i, arr) => {
      if (a.startsWith('--')) {
        const next = arr[i + 1];
        // A trailing flag has no next argument at all. Treating that as its
        // value silently turns `--break-recognizer` into undefined, which is
        // falsy -- the flag then does nothing and the run looks like a pass.
        acc.push([a.slice(2), next === undefined || next.startsWith('--') ? true : next]);
      }
      return acc;
    }, []),
  );
  const y4mDir = args.y4m;
  const label = args.label || 'run';
  const outDir = args.out || join(REPO, 'bench');
  const pagePath = args.page || '/demo.html';

  if (!y4mDir) throw new Error('--y4m <dir> is required');
  await mkdir(outDir, { recursive: true });

  const server = await serve(WEB);
  const { port: httpPort } = server.address();
  const pageUrl = `http://127.0.0.1:${httpPort}${pagePath}`;

  const results = {};
  for (const cond of ['hand', 'nohand']) {
    process.stderr.write(`[webbench] ${label}/${cond} ...\n`);
    results[cond] = await runCondition({
      y4m: join(y4mDir, `${cond}.y4m`),
      page: pageUrl,
      breakRecognizer: Boolean(args['break-recognizer']),
    });
    process.stderr.write(`[webbench] ${label}/${cond} done\n`);
  }

  server.close();

  const out = {
    label,
    page: pagePath,
    measured_at: new Date().toISOString(),
    warmup_ms: WARMUP_MS,
    ...results,
  };
  const path = join(outDir, `webbench-${label}.json`);
  await writeFile(path, JSON.stringify(out, null, 2));

  for (const cond of ['hand', 'nohand']) {
    const r = results[cond];
    console.log(`\n=== ${label} / ${cond} ===`);
    console.log(`  presented fps      ${r.presented_fps}   (gap p50 ${r.present_gap_ms.p50}ms  p95 ${r.present_gap_ms.p95}ms  max ${r.present_gap_ms.max}ms)`);
    console.log(`  inference fps      ${r.inference_fps}   (rAF callbacks/s ${r.raf_callbacks_per_s})`);
    console.log(`  mediapipe solve    p50 ${r.mediapipe_solve_ms.p50}ms  p95 ${r.mediapipe_solve_ms.p95}ms  max ${r.mediapipe_solve_ms.max}ms  (n=${r.mediapipe_solve_ms.n})`);
    console.log(`  recognizer         p50 ${r.recognizer_ms.p50}ms  p95 ${r.recognizer_ms.p95}ms  (n=${r.recognizer_ms.n})`);
    console.log(`  rAF callback cost  p50 ${r.raf_callback_ms.p50}ms  p95 ${r.raf_callback_ms.p95}ms  max ${r.raf_callback_ms.max}ms`);
    console.log(`  long tasks         ${r.longtasks.count}  blocking ${r.longtasks.total_blocking_ms}ms  worst ${r.longtasks.worst_ms}ms`);
    console.log(`  pose readout       ${r.poseReadout}`);
    if (r.injected_faults) console.log(`  injected faults    ${r.injected_faults}`);
  }
  console.log(`\n  renderer: ${results.hand.env.renderer}`);
  console.log(`  wasm simd: ${results.hand.env.wasmSimd}  threads: ${results.hand.env.wasmThreads}  crossOriginIsolated: ${results.hand.env.crossOriginIsolated}`);
  console.log(`  wasm assets: ${results.hand.env.wasmAssets.map((s) => s.split('/').pop()).join(', ')}`);
  console.log(`\nwrote ${path}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
