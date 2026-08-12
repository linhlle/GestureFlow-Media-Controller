// Live recognition preview.
//
// Pulls hand landmarks from MediaPipe Tasks Vision (WASM, running locally),
// feeds them to the ported recognizer, and draws the result. Nothing here
// touches the network beyond loading the model files, and no video frame ever
// leaves the device.

import { FilesetResolver, HandLandmarker }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { Forest } from './forest.js';
import { Recognizer, LM } from './recognizer.js';

const WASM_ROOT =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// Pose names for the shipped 4-class model. Index == class label.
const POSE_NAMES = ['Neutral', 'L-Shape', 'High-Five', '2-Finger'];

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const els = {
  video: document.getElementById('video'),
  canvas: document.getElementById('canvas'),
  overlay: document.getElementById('overlay'),
  overlayText: document.getElementById('overlay-text'),
  overlayHint: document.getElementById('overlay-hint'),
  startBtn: document.getElementById('start-btn'),
  stopBtn: document.getElementById('stop-btn'),
  fps: document.getElementById('fps-readout'),
  pose: document.getElementById('r-pose'),
  conf: document.getElementById('r-conf'),
  vote: document.getElementById('r-vote'),
  voteBar: document.getElementById('r-vote-bar'),
  index: document.getElementById('r-index'),
  thumb: document.getElementById('r-thumb'),
  pinch: document.getElementById('r-pinch'),
  pinchBar: document.getElementById('r-pinch-bar'),
  rpinch: document.getElementById('r-rpinch'),
  rpinchBar: document.getElementById('r-rpinch-bar'),
  scroll: document.getElementById('r-scroll'),
  log: document.getElementById('event-log'),
  pills: document.querySelectorAll('.mode-pill'),
};

const ctx = els.canvas.getContext('2d');

let handLandmarker = null;
let recognizer = null;
let stream = null;
let running = false;
let lastVideoTime = -1;
let frameTimes = [];

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

async function initModels() {
  setOverlay('Loading the recognizer…', '');
  const [vision, forestPayload] = await Promise.all([
    FilesetResolver.forVisionTasks(WASM_ROOT),
    Forest.load('/models/forest.json'),
  ]);

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numHands: 1,
    // Matches MediaPipeConfig in gestureflow/config.py.
    minHandDetectionConfidence: 0.8,
    minTrackingConfidence: 0.5,
  });

  recognizer = new Recognizer(forestPayload);
}

async function start() {
  els.startBtn.disabled = true;
  try {
    if (!handLandmarker) await initModels();

    setOverlay('Waiting for camera permission…', '');
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });

    els.video.srcObject = stream;
    await els.video.play();

    els.canvas.width = els.video.videoWidth || 640;
    els.canvas.height = els.video.videoHeight || 480;

    els.overlay.classList.add('hidden');
    els.stopBtn.disabled = false;
    running = true;
    requestAnimationFrame(loop);
  } catch (err) {
    els.startBtn.disabled = false;
    handleStartError(err);
  }
}

function handleStartError(err) {
  const name = err && err.name;
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    setOverlay(
      'Camera access was blocked.',
      'Allow camera access for this site in your browser settings, then press '
      + 'Start again. Nothing is uploaded — the video is processed here.',
    );
  } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    setOverlay('No camera found.',
               'Connect a webcam and press Start again.');
  } else if (name === 'NotReadableError') {
    setOverlay('The camera is in use by another app.',
               'Close anything else using it (including the GestureFlow '
               + 'desktop app) and press Start again.');
  } else {
    setOverlay('Could not start the demo.', String(err && err.message ? err.message : err));
  }
  els.overlay.classList.remove('hidden');
}

function stop() {
  running = false;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  els.stopBtn.disabled = true;
  els.startBtn.disabled = false;
  setOverlay('Camera stopped.', '');
  els.overlay.classList.remove('hidden');
  els.fps.textContent = '';
}

function setOverlay(text, hint) {
  els.overlayText.textContent = text;
  els.overlayHint.textContent = hint || '';
  els.startBtn.style.display = text.includes('Loading') || text.includes('Waiting')
    ? 'none' : '';
}

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

function loop() {
  if (!running) return;

  const nowMs = performance.now();
  if (els.video.currentTime !== lastVideoTime && els.video.readyState >= 2) {
    lastVideoTime = els.video.currentTime;

    const detection = handLandmarker.detectForVideo(els.video, nowMs);
    const landmarks = detection.landmarks && detection.landmarks.length
      ? detection.landmarks[0]
      : null;

    // The recognizer thinks in seconds, like the Python original.
    const result = recognizer.process(landmarks, nowMs / 1000);

    draw(result);
    updatePanels(result);
    logEvents(result);
    trackFps(nowMs);
  }

  requestAnimationFrame(loop);
}

function trackFps(nowMs) {
  frameTimes.push(nowMs);
  while (frameTimes.length > 1 && nowMs - frameTimes[0] > 2000) frameTimes.shift();
  if (frameTimes.length > 5) {
    const span = (nowMs - frameTimes[0]) / 1000;
    const fps = (frameTimes.length - 1) / span;
    // Measured here, in this browser, right now -- not a claim about the
    // desktop app, which is a different pipeline on different hardware.
    els.fps.textContent = `${fps.toFixed(0)} fps in this tab`;
  }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

function draw(result) {
  const { width: w, height: h } = els.canvas;

  ctx.save();
  // Mirror, so moving right on screen matches moving right in the room.
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(els.video, 0, 0, w, h);
  ctx.restore();

  if (!result.landmarks) return;

  const pt = (lm) => [(1 - lm.x) * w, lm.y * h];
  const accent = getComputedStyle(document.documentElement)
    .getPropertyValue('--accent').trim() || '#4dd4c4';

  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(230, 237, 243, 0.55)';
  for (const [a, b] of HAND_CONNECTIONS) {
    const [x1, y1] = pt(result.landmarks[a]);
    const [x2, y2] = pt(result.landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  ctx.fillStyle = accent;
  for (const lm of result.landmarks) {
    const [x, y] = pt(lm);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  drawChargeArc(pt, result.landmarks[LM.INDEX_TIP], result.holdProgress, accent);
  drawChargeArc(pt, result.landmarks[LM.MIDDLE_TIP], result.rightHoldProgress,
                '#4dc4d4');

  if (result.scrollActive) {
    const [wx, wy] = pt(result.landmarks[LM.WRIST]);
    const dir = result.scrollDelta >= 0 ? -1 : 1;
    ctx.strokeStyle = '#6fcf7f';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(wx, wy);
    ctx.lineTo(wx, wy + dir * 34);
    ctx.stroke();
  }

  drawModeLabel(result);
}

function drawChargeArc(pt, landmark, progress, color) {
  if (!progress || progress <= 0) return;
  const [x, y] = pt(landmark);
  ctx.strokeStyle = 'rgba(120,120,120,0.6)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 20, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, 20, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
  ctx.stroke();
}

function drawModeLabel(result) {
  const label = result.mode === 'tracking' ? 'tracking' : result.mode;
  ctx.font = '600 16px ui-monospace, Menlo, monospace';
  const text = label.toUpperCase();
  const padding = 8;
  const metrics = ctx.measureText(text);
  ctx.fillStyle = 'rgba(11, 15, 20, 0.75)';
  ctx.fillRect(12, 12, metrics.width + padding * 2, 28);
  ctx.fillStyle = result.mode === 'tracking' ? '#9aa7b6' : '#4dd4c4';
  ctx.fillText(text, 12 + padding, 31);
}

// ---------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------

function setFlag(el, on, onText, offText) {
  el.textContent = on ? onText : offText;
  el.classList.toggle('on', !!on);
  el.classList.toggle('off', !on);
}

function updatePanels(result) {
  els.pills.forEach((pill) => {
    pill.classList.toggle('active', pill.dataset.mode === result.mode);
  });

  const poseName = POSE_NAMES[result.rawPrediction] || `class ${result.rawPrediction}`;
  els.pose.textContent = result.landmarks ? poseName : '—';
  els.conf.textContent = result.landmarks
    ? `${(result.confidence * 100).toFixed(0)}%` : '—';

  els.vote.textContent = `${result.voteScore}/10`;
  els.voteBar.style.width = `${Math.min(100, (result.voteScore / 10) * 100)}%`;

  setFlag(els.index, result.indexExtended, 'yes', 'no');
  setFlag(els.thumb, result.thumbRaised, 'yes', 'no');
  setFlag(els.pinch, result.fsmActive, result.fsmState.toLowerCase(), 'idle');
  setFlag(els.rpinch, result.rightFsmActive,
          result.rightFsmState.toLowerCase(), 'idle');
  setFlag(els.scroll, result.scrollActive,
          result.scrollDelta !== 0 ? `${result.scrollDelta > 0 ? 'up' : 'down'} ${Math.abs(result.scrollDelta)}` : 'armed',
          'idle');

  els.pinchBar.style.width = `${result.holdProgress * 100}%`;
  els.rpinchBar.style.width = `${result.rightHoldProgress * 100}%`;
}

let lastScrollLogged = 0;

function logEvents(result) {
  if (result.clickFired) addEvent('left click');
  if (result.rightClickFired) addEvent('right click');
  if (result.action !== null) {
    addEvent(`command: ${POSE_NAMES[result.action] || result.action}`);
  }
  // Scroll fires many times a second; log at most every 400ms so the panel
  // stays readable.
  if (result.scrollDelta !== 0 && performance.now() - lastScrollLogged > 400) {
    lastScrollLogged = performance.now();
    addEvent(`scroll ${result.scrollDelta > 0 ? 'up' : 'down'} ${Math.abs(result.scrollDelta)}`);
  }
}

function addEvent(text) {
  const row = document.createElement('div');
  const time = new Date().toLocaleTimeString([], { hour12: false });
  row.innerHTML = `<span class="t">${time}</span> ${text}`;
  els.log.prepend(row);
  while (els.log.childElementCount > 40) els.log.lastElementChild.remove();
}

// ---------------------------------------------------------------------------

els.startBtn.addEventListener('click', start);
els.stopBtn.addEventListener('click', stop);

if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  setOverlay('This browser cannot access the camera.',
             'Try a recent version of Chrome, Edge, Firefox, or Safari.');
  els.startBtn.disabled = true;
}

window.addEventListener('pagehide', () => {
  if (stream) stream.getTracks().forEach((t) => t.stop());
});
