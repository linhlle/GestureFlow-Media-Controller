// Numeric parity harness: runs the JS recognizer over fixtures produced by the
// Python side and prints results as JSON for comparison.
//
// The textual checks in tests/test_web_parity.py catch a constant drifting.
// This catches the harder case: the constants agreeing while the logic does
// not. tests/test_js_parity.py drives it and diffs the output.
//
// Usage: node scripts/parity_check.mjs <fixtures.json>

import { readFileSync } from 'node:fs';
import {
  normalizeLandmarks,
  velocityToClicks,
  indexExtended,
  thumbRaised,
  strictFist,
  isTrueScrollFist,
  ClickFSM,
  ScrollFSM,
  GestureDebouncer,
  DEFAULTS,
} from '../web/js/recognizer.js';
import { Forest } from '../web/js/forest.js';

const fixturePath = process.argv[2];
const fixtures = JSON.parse(readFileSync(fixturePath, 'utf8'));

const toLm = (rows) => rows.map(([x, y, z]) => ({ x, y, z }));

const out = {};

// --- feature vector --------------------------------------------------------
out.normalize = fixtures.normalize.map((rows) => normalizeLandmarks(toLm(rows)));

// --- scroll click maths ----------------------------------------------------
out.velocityToClicks = fixtures.velocity.map(([v, cfg]) =>
  velocityToClicks(v, {
    sensitivity: cfg.sensitivity,
    velocityExponent: cfg.velocity_exponent,
    step: cfg.step,
  }));

// --- geometric predicates --------------------------------------------------
out.predicates = fixtures.predicates.map((rows) => {
  const lm = toLm(rows);
  return {
    indexExtended: indexExtended(lm),
    thumbRaised: thumbRaised(lm),
    strictFist: strictFist(lm),
    isTrueScrollFist: isTrueScrollFist(lm),
  };
});

// --- click FSM over a distance sequence ------------------------------------
out.clickSequences = fixtures.clickSequences.map((seq) => {
  const fsm = new ClickFSM(DEFAULTS.click);
  const events = [];
  seq.forEach((dist, i) => {
    const lm = Array.from({ length: 21 }, (_, k) => ({ x: k, y: k, z: 0 }));
    lm[4] = { x: 0.5, y: 0.5, z: 0 };
    lm[8] = { x: 0.5 + dist, y: 0.5, z: 0 };
    fsm.update(lm, i / 30);
    events.push({ state: fsm.state, fired: fsm.clickFired,
                  progress: Number(fsm.holdProgress.toFixed(6)) });
  });
  return events;
});

// --- scroll FSM over a wrist track -----------------------------------------
out.scrollSequences = fixtures.scrollSequences.map((seq) => {
  const fsm = new ScrollFSM(DEFAULTS.scroll);
  return seq.map(([rows, t]) => {
    fsm.update(toLm(rows), t);
    return { state: fsm.state, delta: fsm.scrollDelta };
  });
});

// --- debouncer -------------------------------------------------------------
out.debounce = fixtures.debounce.map((seq) => {
  const db = new GestureDebouncer(DEFAULTS);
  return seq.map(([label, conf, t]) => ({
    action: db.update(label, conf, t),
    stable: db.stableGesture,
    score: db.voteScore,
  }));
});

// --- forest ----------------------------------------------------------------
if (fixtures.forestPath && fixtures.forestSamples) {
  const forest = new Forest(JSON.parse(readFileSync(fixtures.forestPath, 'utf8')));
  out.forest = fixtures.forestSamples.map((s) => forest.predictProba(s));
}

process.stdout.write(JSON.stringify(out));
