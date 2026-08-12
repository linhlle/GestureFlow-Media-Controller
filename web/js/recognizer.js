// Gesture recognition, ported from the Python desktop app.
//
// Every function here mirrors a specific piece of gestureflow/ and must stay
// behaviourally identical to it, so the browser preview and the desktop app
// agree about what a gesture is:
//
//   normalizeLandmarks   <- gestureflow/utils.py:normalize_landmarks
//   ClickFSM             <- gestureflow/click_fsm.py
//   ScrollFSM            <- gestureflow/scroll_fsm.py
//   GestureDebouncer     <- gestureflow/debouncer.py
//   DEFAULTS             <- gestureflow/config.py
//
// tests/test_web_parity.py checks the ports against the Python originals on
// shared fixtures, so a change to one side that is not mirrored fails CI.

export const DEFAULTS = {
  confidenceThreshold: 0.80,
  voteWindowSize: 10,
  voteThreshold: 7,
  cmdCooldown: 1.3,
  click: { close: 0.045, open: 0.065, minHoldFrames: 4, cooldown: 0.4 },
  rightClick: { close: 0.045, open: 0.065, minHoldFrames: 5, cooldown: 0.6 },
  scroll: {
    sensitivity: 0.008, minHoldFrames: 5, cooldown: 0.05,
    step: 2, velocityExponent: 1.6,
  },
};

// Landmark indices (MediaPipe hand model).
export const LM = {
  WRIST: 0, THUMB_MCP: 2, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_TIP: 20,
};

// ---------------------------------------------------------------------------
// Feature vector
// ---------------------------------------------------------------------------

/**
 * 21 landmarks -> 63 floats: wrist-relative, then scaled by the largest
 * absolute component. Translation- and scale-invariant, range [-1, 1].
 */
export function normalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length === 0) return new Array(63).fill(0);

  const base = landmarks[0];
  const relative = [];
  for (const lm of landmarks) {
    relative.push(lm.x - base.x, lm.y - base.y, lm.z - base.z);
  }

  let maxVal = 0;
  for (const v of relative) {
    const a = Math.abs(v);
    if (a > maxVal) maxVal = a;
  }
  if (maxVal === 0) return new Array(63).fill(0);

  return relative.map((v) => v / maxVal);
}

// ---------------------------------------------------------------------------
// Geometric predicates
// ---------------------------------------------------------------------------

export function pinchDistance(landmarks, a, b) {
  const p = landmarks[a];
  const q = landmarks[b];
  return Math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2);
}

export function indexExtended(landmarks, margin = 0.04) {
  return landmarks[LM.INDEX_TIP].y < landmarks[LM.INDEX_PIP].y - margin;
}

export function thumbRaised(landmarks, margin = 0.04) {
  return landmarks[LM.THUMB_TIP].y < landmarks[LM.THUMB_MCP].y - margin;
}

const CURL_PAIRS = [
  [LM.INDEX_TIP, LM.INDEX_MCP],
  [LM.MIDDLE_TIP, LM.MIDDLE_MCP],
  [LM.RING_TIP, LM.RING_MCP],
  [LM.PINKY_TIP, LM.PINKY_MCP],
];

export function strictFist(landmarks, threshold = 0.03) {
  let curled = 0;
  for (const [tip, knuckle] of CURL_PAIRS) {
    if (landmarks[tip].y > landmarks[knuckle].y + threshold) curled += 1;
  }
  return curled === 4;
}

/** Three exclusion gates, in the same priority order as the Python version. */
export function isTrueScrollFist(landmarks) {
  if (indexExtended(landmarks)) return false;
  if (thumbRaised(landmarks)) return false;
  return strictFist(landmarks);
}

// ---------------------------------------------------------------------------
// Click FSM
// ---------------------------------------------------------------------------

export const ClickState = { IDLE: 'IDLE', PRESSING: 'PRESSING', HELD: 'HELD' };

export class ClickFSM {
  constructor(config = DEFAULTS.click, lmA = LM.THUMB_TIP, lmB = LM.INDEX_TIP) {
    this.cfg = config;
    this.lmA = lmA;
    this.lmB = lmB;
    this.state = ClickState.IDLE;
    this.holdFrames = 0;
    this.clickFired = false;
    // -Infinity, matching the Python fix: a 0 seed would suppress every click
    // in the first cooldown seconds after load.
    this.lastClickTime = -Infinity;
  }

  update(landmarks, now) {
    this.clickFired = false;
    if (!landmarks) {
      this.reset();
      return;
    }
    this.transition(pinchDistance(landmarks, this.lmA, this.lmB), now);
  }

  transition(dist, now) {
    const cfg = this.cfg;
    if (this.state === ClickState.IDLE) {
      if (dist < cfg.close) {
        this.holdFrames = 1;
        this.state = ClickState.PRESSING;
      }
    } else if (this.state === ClickState.PRESSING) {
      if (dist < cfg.close) {
        this.holdFrames += 1;
        if (this.holdFrames >= cfg.minHoldFrames) this.state = ClickState.HELD;
      } else {
        this.reset();
      }
    } else if (this.state === ClickState.HELD) {
      // Fires on the release edge only, so holding does not auto-repeat.
      if (dist > cfg.open) {
        if (now - this.lastClickTime >= cfg.cooldown) {
          this.clickFired = true;
          this.lastClickTime = now;
        }
        this.reset();
      }
    }
  }

  reset() {
    this.state = ClickState.IDLE;
    this.holdFrames = 0;
  }

  get isActive() {
    return this.state === ClickState.PRESSING || this.state === ClickState.HELD;
  }

  get holdProgress() {
    if (this.cfg.minHoldFrames === 0) return 1;
    return Math.min(1, this.holdFrames / this.cfg.minHoldFrames);
  }
}

// ---------------------------------------------------------------------------
// Scroll FSM
// ---------------------------------------------------------------------------

export const ScrollState = {
  IDLE: 'IDLE', FIST_DETECTED: 'FIST_DETECTED', SCROLLING: 'SCROLLING',
};

// Matches _CLICK_PRECISION in scroll_fsm.py: quantize before ceil() so float
// residue does not buy an extra scroll click.
const CLICK_PRECISION = 9;

function roundTo(value, dp) {
  const f = 10 ** dp;
  return Math.round(value * f) / f;
}

export function velocityToClicks(velocity, cfg) {
  const ratio = roundTo(velocity / cfg.sensitivity, CLICK_PRECISION);
  const magnitude = roundTo(Math.abs(ratio) ** cfg.velocityExponent, CLICK_PRECISION);
  return Math.sign(ratio) * Math.ceil(magnitude) * cfg.step;
}

export class ScrollFSM {
  constructor(config = DEFAULTS.scroll) {
    this.cfg = config;
    this.state = ScrollState.IDLE;
    this.holdFrames = 0;
    this.prevWristY = 0;
    this.lastScrollTime = -Infinity;
    this.scrollDelta = 0;
  }

  update(landmarks, now) {
    this.scrollDelta = 0;
    if (!landmarks) {
      this.reset();
      return;
    }
    this.transition(isTrueScrollFist(landmarks), landmarks[LM.WRIST].y, now);
  }

  transition(fist, wristY, now) {
    const cfg = this.cfg;
    if (this.state === ScrollState.IDLE) {
      if (fist) {
        this.holdFrames = 1;
        this.prevWristY = wristY;
        this.state = ScrollState.FIST_DETECTED;
      }
    } else if (this.state === ScrollState.FIST_DETECTED) {
      if (fist) {
        this.holdFrames += 1;
        if (this.holdFrames >= cfg.minHoldFrames) {
          this.prevWristY = wristY;
          this.state = ScrollState.SCROLLING;
        }
      } else {
        this.reset();
      }
    } else if (this.state === ScrollState.SCROLLING) {
      if (!fist) {
        this.reset();
        return;
      }
      if (now - this.lastScrollTime < cfg.cooldown) {
        this.prevWristY = wristY;
        return;
      }
      const velocity = this.prevWristY - wristY;
      this.prevWristY = wristY;
      if (Math.abs(velocity) > cfg.sensitivity) {
        const clicks = velocityToClicks(velocity, cfg);
        if (clicks !== 0) {
          this.scrollDelta = clicks;
          this.lastScrollTime = now;
        }
      }
    }
  }

  reset() {
    this.state = ScrollState.IDLE;
    this.holdFrames = 0;
    this.prevWristY = 0;
  }

  get isActive() {
    return this.state === ScrollState.FIST_DETECTED
        || this.state === ScrollState.SCROLLING;
  }
}

// ---------------------------------------------------------------------------
// Debouncer
// ---------------------------------------------------------------------------

export class GestureDebouncer {
  constructor(config = DEFAULTS) {
    this.windowSize = config.voteWindowSize;
    this.threshold = config.voteThreshold;
    this.cooldown = config.cmdCooldown;
    this.confidenceThreshold = config.confidenceThreshold;
    this.history = [];
    this.lastCmdTime = -Infinity;
  }

  update(prediction, confidence, now) {
    // Low-confidence frames are clamped to Neutral rather than discarded, so
    // noise actively drains the window instead of leaving it stale.
    const effective = confidence >= this.confidenceThreshold ? prediction : 0;
    this.history.push(effective);
    if (this.history.length > this.windowSize) this.history.shift();

    const [stable, score] = this.majority();
    if (stable === 0 || score < this.threshold) return null;
    if (now - this.lastCmdTime < this.cooldown) return null;

    this.lastCmdTime = now;
    this.history = [];
    return stable;
  }

  majority() {
    if (this.history.length === 0) return [0, 0];
    const counts = new Map();
    for (const v of this.history) counts.set(v, (counts.get(v) || 0) + 1);
    let best = 0;
    let bestCount = 0;
    for (const [label, count] of counts) {
      if (count > bestCount) {
        best = label;
        bestCount = count;
      }
    }
    return [best, bestCount];
  }

  get stableGesture() { return this.majority()[0]; }
  get voteScore() { return this.majority()[1]; }
}

// ---------------------------------------------------------------------------
// Combined recognizer
// ---------------------------------------------------------------------------

export class Recognizer {
  constructor(forest, config = DEFAULTS) {
    this.forest = forest;
    this.config = config;
    this.debouncer = new GestureDebouncer(config);
    this.leftFSM = new ClickFSM(config.click, LM.THUMB_TIP, LM.INDEX_TIP);
    this.rightFSM = new ClickFSM(config.rightClick, LM.MIDDLE_TIP, LM.INDEX_TIP);
    this.scrollFSM = new ScrollFSM(config.scroll);
  }

  /** Mirrors InferenceThread._process plus GestureRouter's mode predicates. */
  process(landmarks, now) {
    if (!landmarks) {
      this.debouncer.update(0, 1.0, now);
      this.leftFSM.update(null, now);
      this.rightFSM.update(null, now);
      this.scrollFSM.update(null, now);
      return this.emptyResult();
    }

    const features = normalizeLandmarks(landmarks);
    const probs = this.forest ? this.forest.predictProba(features) : [1, 0, 0, 0];

    let rawPrediction = 0;
    let confidence = 0;
    for (let i = 0; i < probs.length; i += 1) {
      if (probs[i] > confidence) {
        confidence = probs[i];
        rawPrediction = i;
      }
    }

    const action = this.debouncer.update(rawPrediction, confidence, now);
    const stable = this.debouncer.stableGesture;

    if (stable !== 0) {
      this.leftFSM.update(null, now);
      this.rightFSM.update(null, now);
      this.scrollFSM.update(null, now);
    } else {
      this.leftFSM.update(landmarks, now);
      this.rightFSM.update(landmarks, now);
      this.scrollFSM.update(landmarks, now);
    }

    const idxExt = indexExtended(landmarks);
    const thumbUp = thumbRaised(landmarks);
    // Matches GestureRouter._geometric_modes_suppressed: an emitted action
    // suppresses the geometric modes even though the debouncer has already
    // cleared its window and stable reads 0 again.
    const suppressed = stable !== 0 || action !== null;

    const scrollActive = !suppressed && this.scrollFSM.isActive;
    const cursorActive = !suppressed && !this.leftFSM.isActive
      && !this.rightFSM.isActive && !scrollActive && !thumbUp && idxExt;
    const volumeActive = !suppressed && !scrollActive && !idxExt && thumbUp
      && landmarks[LM.THUMB_TIP].y < landmarks[LM.INDEX_MCP].y;

    return {
      landmarks,
      stableGesture: stable,
      voteScore: this.debouncer.voteScore,
      confidence,
      rawPrediction,
      action,
      clickFired: this.leftFSM.clickFired,
      fsmActive: this.leftFSM.isActive,
      fsmState: this.leftFSM.state,
      holdProgress: this.leftFSM.holdProgress,
      rightClickFired: this.rightFSM.clickFired,
      rightFsmActive: this.rightFSM.isActive,
      rightFsmState: this.rightFSM.state,
      rightHoldProgress: this.rightFSM.holdProgress,
      scrollDelta: this.scrollFSM.scrollDelta,
      scrollActive,
      indexExtended: idxExt,
      thumbRaised: thumbUp,
      cursorActive,
      volumeActive,
      mode: modeOf({ suppressed, stable, scrollActive, cursorActive, volumeActive,
                     leftActive: this.leftFSM.isActive,
                     rightActive: this.rightFSM.isActive }),
    };
  }

  emptyResult() {
    return {
      landmarks: null, stableGesture: 0, voteScore: this.debouncer.voteScore,
      confidence: 0, rawPrediction: 0, action: null,
      clickFired: false, fsmActive: false, fsmState: ClickState.IDLE,
      holdProgress: 0, rightClickFired: false, rightFsmActive: false,
      rightFsmState: ClickState.IDLE, rightHoldProgress: 0,
      scrollDelta: 0, scrollActive: false, indexExtended: false,
      thumbRaised: false, cursorActive: false, volumeActive: false,
      mode: 'none',
    };
  }
}

function modeOf(s) {
  if (s.stable !== 0 || s.suppressed) return 'command';
  if (s.leftActive) return 'left-click';
  if (s.rightActive) return 'right-click';
  if (s.scrollActive) return 'scroll';
  if (s.volumeActive) return 'volume';
  if (s.cursorActive) return 'cursor';
  return 'tracking';
}
