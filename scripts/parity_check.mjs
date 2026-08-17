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
  SwipeFSM,
  ZoomFSM,
  PauseFSM,
  GestureDebouncer,
  Recognizer,
  rockHorns,
  zoomPose,
  thumbIndexAngle,
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

// --- click FSM over a landmark sequence -------------------------------------
// Landmarks come from the fixture rather than being rebuilt here. Constructing
// them on both sides meant the two could disagree about the hand itself, and
// once thresholds became ratios of hand scale that disagreement silently made
// the comparison meaningless.
out.clickSequences = fixtures.clickSequences.map((seq) => {
  const fsm = new ClickFSM(DEFAULTS.click);
  return seq.map(([rows, t]) => {
    fsm.update(toLm(rows), t);
    return { state: fsm.state, fired: fsm.clickFired,
             progress: Number(fsm.holdProgress.toFixed(6)) };
  });
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

// --- new detectors ----------------------------------------------------------
out.swipeSequences = (fixtures.swipeSequences || []).map((seq) => {
  const fsm = new SwipeFSM(DEFAULTS.swipe);
  return seq.map(([rows, t]) => {
    fsm.update(toLm(rows), t);
    return { state: fsm.state, direction: fsm.direction };
  });
});

out.zoomSequences = (fixtures.zoomSequences || []).map((seq) => {
  const fsm = new ZoomFSM(DEFAULTS.zoom);
  return seq.map(([rows, t]) => {
    fsm.update(toLm(rows), t);
    return { state: fsm.state, direction: fsm.direction };
  });
});

out.pauseSequences = (fixtures.pauseSequences || []).map((seq) => {
  const fsm = new PauseFSM(DEFAULTS.pause);
  return seq.map(([rows, t]) => {
    fsm.update(toLm(rows), t);
    return { paused: fsm.paused, toggled: fsm.toggled,
             progress: Number(fsm.progress.toFixed(6)) };
  });
});

out.newPredicates = (fixtures.newPredicates || []).map((rows) => {
  const lm = toLm(rows);
  return {
    rockHorns: rockHorns(lm),
    zoomPose: zoomPose(lm, DEFAULTS.zoom),
    thumbIndexAngle: Number(thumbIndexAngle(lm).toFixed(6)),
  };
});

// --- forest ----------------------------------------------------------------
if (fixtures.forestPath && fixtures.forestSamples) {
  const forest = new Forest(JSON.parse(readFileSync(fixtures.forestPath, 'utf8')));
  out.forest = fixtures.forestSamples.map((s) => forest.predictProba(s));
}

// --- the whole recognizer ---------------------------------------------------
// Everything above exercises a part. This exercises Recognizer.process, which
// is the single function the demo page actually calls -- and which had no
// coverage at all until a ReferenceError in it shipped and killed the demo's
// render loop on the first frame containing a hand.
//
// `undefinedKeys` is the cheap general guard: any field the page reads that
// process() forgets to populate shows up here rather than as a blank readout
// nobody notices.
if (fixtures.forestPath && fixtures.recognizerSequences) {
  const forest = new Forest(JSON.parse(readFileSync(fixtures.forestPath, 'utf8')));
  out.recognizer = fixtures.recognizerSequences.map((seq) => {
    const rec = new Recognizer(forest);
    return seq.map(([rows, t]) => {
      const r = rec.process(rows === null ? null : toLm(rows), t);
      return {
        mode: r.mode,
        stableGesture: r.stableGesture,
        rawPrediction: r.rawPrediction,
        action: r.action,
        scrollActive: r.scrollActive,
        scrollDelta: r.scrollDelta,
        swipeArmed: r.swipeArmed,
        swipeDirection: r.swipeDirection,
        zoomActive: r.zoomActive,
        zoomDirection: r.zoomDirection,
        cursorActive: r.cursorActive,
        volumeActive: r.volumeActive,
        dragging: r.dragging,
        fsmActive: r.fsmActive,
        rightFsmActive: r.rightFsmActive,
        indexExtended: r.indexExtended,
        thumbRaised: r.thumbRaised,
        paused: r.paused,
        hasLandmarks: r.landmarks !== null,
        undefinedKeys: Object.keys(r).filter((k) => r[k] === undefined).sort(),
      };
    });
  });
}

process.stdout.write(JSON.stringify(out));
