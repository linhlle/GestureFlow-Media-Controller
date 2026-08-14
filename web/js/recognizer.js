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
  // Thresholds are fractions of hand scale, not absolute image distances --
  // see handScale() below and DIAGNOSIS.md.
  click: { close: 0.28, open: 0.41, minHoldFrames: 4, cooldown: 0.4 },
  rightClick: { close: 0.22, open: 0.38, minHoldFrames: 5, cooldown: 0.6 },
  scroll: {
    sensitivity: 0.05, minHoldFrames: 5, cooldown: 0.05,
    step: 2, velocityExponent: 1.6, axisRatio: 1.0,
  },
  swipe: {
    enabled: true, sensitivity: 0.16, minHoldFrames: 3,
    cooldown: 0.6, axisRatio: 1.5, releaseRatio: 0.5,
  },
  zoom: {
    enabled: true, minSeparation: 0.55, sensitivity: 0.06,
    minHoldFrames: 4, cooldown: 0.12, curlMargin: 0.12,
    minAngleDegrees: 65.0,
  },
  pause: { enabled: true, holdSeconds: 1.5, margin: 0.25 },
  drag: { enabled: true, holdSeconds: 0.55 },
};

// Landmark indices (MediaPipe hand model).
export const LM = {
  WRIST: 0, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_TIP: 20,
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

const MIN_HAND_SCALE = 1e-6;
// Below this the hand has effectively no size; treat it as no hand at all.
const DEGENERATE_HAND_SCALE = 1e-3;

/**
 * True when the landmarks carry no measurable hand.
 *
 * Every threshold is a fraction of hand scale, so a scale of zero makes every
 * distance read as "touching" -- a collapsed hand looks like a permanent pinch.
 */
export function isDegenerate(landmarks) {
  const a = landmarks[LM.WRIST];
  const b = landmarks[LM.MIDDLE_MCP];
  const raw = Math.sqrt(
    (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2,
  );
  return raw <= DEGENERATE_HAND_SCALE;
}

/**
 * Wrist to middle-finger MCP: the reference length every threshold is measured
 * against. It spans the rigid palm, so unlike a fingertip span it does not
 * change as the hand opens, closes, or points.
 */
export function handScale(landmarks) {
  const a = landmarks[LM.WRIST];
  const b = landmarks[LM.MIDDLE_MCP];
  const d = Math.sqrt(
    (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2,
  );
  return d > MIN_HAND_SCALE ? d : MIN_HAND_SCALE;
}

export function indexExtended(landmarks, margin = 0.25) {
  return landmarks[LM.INDEX_TIP].y
       < landmarks[LM.INDEX_PIP].y - margin * handScale(landmarks);
}

/**
 * A straight thumb pointing up and clear of the hand.
 *
 * Comparing the tip against its own MCP alone is true for almost any posture,
 * including a fist, where the thumb folds across the fingers. Requiring the
 * thumb to be straight and clear of the index knuckle distinguishes a real
 * thumbs-up from a thumb tucked into a fist.
 */
export function thumbRaised(landmarks, margin = 0.25) {
  const scale = handScale(landmarks);
  const tip = landmarks[LM.THUMB_TIP];
  const ip = landmarks[LM.THUMB_IP];
  const mcp = landmarks[LM.THUMB_MCP];

  const straight = tip.y < ip.y - margin * 0.5 * scale
                && ip.y < mcp.y - margin * 0.3 * scale;
  const clearOfHand = tip.y < landmarks[LM.INDEX_MCP].y - margin * scale;
  return straight && clearOfHand;
}

const CURL_PAIRS = [
  [LM.INDEX_TIP, LM.INDEX_MCP],
  [LM.MIDDLE_TIP, LM.MIDDLE_MCP],
  [LM.RING_TIP, LM.RING_MCP],
  [LM.PINKY_TIP, LM.PINKY_MCP],
];

export function strictFist(landmarks, threshold = 0.19) {
  const limit = threshold * handScale(landmarks);
  let curled = 0;
  for (const [tip, knuckle] of CURL_PAIRS) {
    if (landmarks[tip].y > landmarks[knuckle].y + limit) curled += 1;
  }
  return curled === 4;
}

/**
 * Index and pinky up, middle and ring down -- "rock horns", the pause pose.
 * Mirrors rock_horns() in gestureflow/pause_fsm.py.
 */
export function rockHorns(landmarks, margin = 0.25) {
  const scale = handScale(landmarks);
  const up = margin * scale;
  const down = margin * 0.6 * scale;

  return landmarks[LM.INDEX_TIP].y < landmarks[LM.INDEX_PIP].y - up
      && landmarks[LM.PINKY_TIP].y < landmarks[LM.PINKY_PIP].y - up
      && landmarks[LM.MIDDLE_TIP].y > landmarks[LM.MIDDLE_MCP].y + down
      && landmarks[LM.RING_TIP].y > landmarks[LM.RING_MCP].y + down;
}

/** Scroll's side of the axis arbitration: vertical at least ties. */
export function verticalDominates(vx, vy, ratio = 1.0) {
  return Math.abs(vy) >= ratio * Math.abs(vx);
}

/** Swipe's side: horizontal has to win clearly, not just edge ahead. */
export function horizontalDominates(vx, vy, ratio = 1.5) {
  return Math.abs(vx) > ratio * Math.abs(vy);
}

/** Thumb-to-index distance in hand-widths. Mirrors _spread in zoom_fsm.py. */
export function thumbIndexSpread(landmarks) {
  const a = landmarks[LM.THUMB_TIP];
  const b = landmarks[LM.INDEX_TIP];
  const raw = Math.sqrt(
    (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2,
  );
  return raw / handScale(landmarks);
}

/**
 * Angle in degrees between thumb and index.
 *
 * This is what separates a zoom pose from ordinary pointing: a pointing hand
 * holds the thumb away from the index too, so distance alone reads a cursor
 * gesture as a zoom.
 */
export function thumbIndexAngle(landmarks) {
  const tx = landmarks[LM.THUMB_TIP].x - landmarks[LM.THUMB_MCP].x;
  const ty = landmarks[LM.THUMB_TIP].y - landmarks[LM.THUMB_MCP].y;
  const ix = landmarks[LM.INDEX_TIP].x - landmarks[LM.INDEX_MCP].x;
  const iy = landmarks[LM.INDEX_TIP].y - landmarks[LM.INDEX_MCP].y;

  const tn = Math.hypot(tx, ty);
  const inn = Math.hypot(ix, iy);
  if (tn === 0 || inn === 0) return 0;

  let cosine = (tx * ix + ty * iy) / (tn * inn);
  cosine = Math.max(-1, Math.min(1, cosine));
  return (Math.acos(cosine) * 180) / Math.PI;
}

/** Mirrors zoom_pose() in gestureflow/zoom_fsm.py. */
export function zoomPose(landmarks, cfg = DEFAULTS.zoom) {
  if (isDegenerate(landmarks)) return false;
  const scale = handScale(landmarks);
  const curl = cfg.curlMargin * scale;

  const indexOut = landmarks[LM.INDEX_TIP].y < landmarks[LM.INDEX_PIP].y;
  const othersCurled =
    landmarks[LM.MIDDLE_TIP].y > landmarks[LM.MIDDLE_MCP].y + curl
    && landmarks[LM.RING_TIP].y > landmarks[LM.RING_MCP].y + curl
    && landmarks[LM.PINKY_TIP].y > landmarks[LM.PINKY_MCP].y + curl;

  return indexOut && othersCurled
      && thumbIndexSpread(landmarks) > cfg.minSeparation
      && thumbIndexAngle(landmarks) >= cfg.minAngleDegrees;
}

/** Three exclusion gates, in the same priority order as the Python version. */
export function isTrueScrollFist(landmarks) {
  if (isDegenerate(landmarks)) return false;
  if (indexExtended(landmarks)) return false;
  if (thumbRaised(landmarks)) return false;
  return strictFist(landmarks);
}

// ---------------------------------------------------------------------------
// Click FSM
// ---------------------------------------------------------------------------

export const ClickState = {
  IDLE: 'IDLE', PRESSING: 'PRESSING', HELD: 'HELD', DRAGGING: 'DRAGGING',
};

export class ClickFSM {
  constructor(config = DEFAULTS.click, lmA = LM.THUMB_TIP, lmB = LM.INDEX_TIP,
              drag = null) {
    this.cfg = config;
    // Drag is opt-in per FSM: only the left pinch drags, matching Python.
    this.drag = drag;
    this.lmA = lmA;
    this.lmB = lmB;
    this.state = ClickState.IDLE;
    this.holdFrames = 0;
    this.clickFired = false;
    this.dragStarted = false;
    this.dragEnded = false;
    this.heldSince = Infinity;
    // -Infinity, matching the Python fix: a 0 seed would suppress every click
    // in the first cooldown seconds after load.
    this.lastClickTime = -Infinity;
  }

  update(landmarks, now) {
    this.clickFired = false;
    this.dragStarted = false;
    this.dragEnded = false;
    if (!landmarks || isDegenerate(landmarks)) {
      // A hand leaving frame mid-drag must still release the button.
      if (this.state === ClickState.DRAGGING) this.dragEnded = true;
      this.reset();
      return;
    }
    // In hand-widths, not image units: a hand further from the camera
    // produces smaller raw distances, so an absolute threshold silently gets
    // easier to cross as the user leans back.
    this.transition(
      pinchDistance(landmarks, this.lmA, this.lmB) / handScale(landmarks),
      now,
    );
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
        if (this.holdFrames >= cfg.minHoldFrames) {
          this.state = ClickState.HELD;
          this.heldSince = now;
        }
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
      } else if (this.dragDue(now)) {
        this.state = ClickState.DRAGGING;
        this.dragStarted = true;
      }
    } else if (this.state === ClickState.DRAGGING) {
      if (dist > cfg.open) {
        this.dragEnded = true;
        this.reset();
      }
    }
  }

  dragDue(now) {
    if (!this.drag || !this.drag.enabled) return false;
    if (this.heldSince === Infinity) return false;
    return now - this.heldSince >= this.drag.holdSeconds;
  }

  get dragging() {
    return this.state === ClickState.DRAGGING;
  }

  reset() {
    this.state = ClickState.IDLE;
    this.holdFrames = 0;
    this.heldSince = Infinity;
  }

  get isActive() {
    return this.state === ClickState.PRESSING
        || this.state === ClickState.HELD
        || this.state === ClickState.DRAGGING;
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
    this.prevWristX = 0;
    this.lastScrollTime = -Infinity;
    this.scrollDelta = 0;
  }

  update(landmarks, now) {
    this.scrollDelta = 0;
    if (!landmarks) {
      this.reset();
      return;
    }
    // Wrist position stays raw; the velocity derived from it is what gets
    // scaled, in transition(). Dividing the position by hand scale would
    // couple the two, since hand scale is itself measured from the wrist.
    this.transition(
      isTrueScrollFist(landmarks),
      landmarks[LM.WRIST].y,
      now,
      handScale(landmarks),
      landmarks[LM.WRIST].x,
    );
  }

  transition(fist, wristY, now, scale = 1.0, wristX = 0.0) {
    const cfg = this.cfg;
    if (this.state === ScrollState.IDLE) {
      if (fist) {
        this.holdFrames = 1;
        this.prevWristY = wristY;
        this.prevWristX = wristX;
        this.state = ScrollState.FIST_DETECTED;
      }
    } else if (this.state === ScrollState.FIST_DETECTED) {
      if (fist) {
        this.holdFrames += 1;
        if (this.holdFrames >= cfg.minHoldFrames) {
          this.prevWristY = wristY;
          this.prevWristX = wristX;
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
        this.prevWristX = wristX;
        return;
      }
      // Hand-widths per frame, so the same physical movement scrolls the
      // same amount at any distance from the camera.
      const velocity = (this.prevWristY - wristY) / scale;
      const horizontal = (wristX - this.prevWristX) / scale;
      this.prevWristY = wristY;
      this.prevWristX = wristX;

      // A mostly-sideways movement is a swipe, not a scroll.
      if (!verticalDominates(horizontal, velocity, cfg.axisRatio)) return;
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
    this.prevWristX = 0;
  }

  get isActive() {
    return this.state === ScrollState.FIST_DETECTED
        || this.state === ScrollState.SCROLLING;
  }
}

// ---------------------------------------------------------------------------
// Swipe FSM  <- gestureflow/swipe_fsm.py
// ---------------------------------------------------------------------------

export const SwipeState = {
  IDLE: 'IDLE', ARMED: 'ARMED', COOLING: 'COOLING',
};

export class SwipeFSM {
  constructor(config = DEFAULTS.swipe) {
    this.cfg = config;
    this.state = SwipeState.IDLE;
    this.holdFrames = 0;
    this.prevX = 0;
    this.prevY = 0;
    this.havePrev = false;
    this.lastFire = -Infinity;
    this.direction = null;
  }

  get fired() { return this.direction !== null; }

  get isArmed() {
    return this.state === SwipeState.ARMED || this.state === SwipeState.COOLING;
  }

  update(landmarks, now) {
    this.direction = null;

    if (!this.cfg.enabled || !landmarks || !isTrueScrollFist(landmarks)) {
      this.reset();
      return;
    }

    const scale = handScale(landmarks);
    const x = landmarks[LM.WRIST].x;
    const y = landmarks[LM.WRIST].y;

    if (this.state === SwipeState.IDLE) {
      this.state = SwipeState.ARMED;
      this.holdFrames = 1;
      this.remember(x, y);
      return;
    }

    const vx = this.havePrev ? (x - this.prevX) / scale : 0;
    const vy = this.havePrev ? (y - this.prevY) / scale : 0;
    this.remember(x, y);
    const speed = Math.abs(vx);

    if (this.state === SwipeState.COOLING) {
      // One flick is one fire, however many frames it spans.
      if (speed < this.cfg.sensitivity * this.cfg.releaseRatio) {
        this.state = SwipeState.ARMED;
      }
      return;
    }

    this.holdFrames += 1;
    if (this.holdFrames < this.cfg.minHoldFrames) return;
    if (now - this.lastFire < this.cfg.cooldown) return;
    if (speed <= this.cfg.sensitivity) return;
    if (!horizontalDominates(vx, vy, this.cfg.axisRatio)) return;

    this.direction = vx > 0 ? 'right' : 'left';
    this.lastFire = now;
    this.state = SwipeState.COOLING;
  }

  remember(x, y) {
    this.prevX = x;
    this.prevY = y;
    this.havePrev = true;
  }

  reset() {
    this.state = SwipeState.IDLE;
    this.holdFrames = 0;
    this.havePrev = false;
  }
}

// ---------------------------------------------------------------------------
// Zoom FSM  <- gestureflow/zoom_fsm.py
// ---------------------------------------------------------------------------

export const ZoomState = {
  IDLE: 'IDLE', ARMING: 'ARMING', ZOOMING: 'ZOOMING',
};

export class ZoomFSM {
  constructor(config = DEFAULTS.zoom) {
    this.cfg = config;
    this.state = ZoomState.IDLE;
    this.holdFrames = 0;
    this.prevSpread = 0;
    this.lastFire = -Infinity;
    this.direction = null;
  }

  get fired() { return this.direction !== null; }

  get isActive() {
    return this.state === ZoomState.ARMING || this.state === ZoomState.ZOOMING;
  }

  update(landmarks, now) {
    this.direction = null;

    if (!this.cfg.enabled || !landmarks || !zoomPose(landmarks, this.cfg)) {
      this.reset();
      return;
    }

    const spread = thumbIndexSpread(landmarks);

    if (this.state === ZoomState.IDLE) {
      this.state = ZoomState.ARMING;
      this.holdFrames = 1;
      this.prevSpread = spread;
      return;
    }

    if (this.state === ZoomState.ARMING) {
      this.holdFrames += 1;
      this.prevSpread = spread;
      if (this.holdFrames >= this.cfg.minHoldFrames) {
        this.state = ZoomState.ZOOMING;
      }
      return;
    }

    if (now - this.lastFire < this.cfg.cooldown) {
      this.prevSpread = spread;
      return;
    }

    const delta = spread - this.prevSpread;
    this.prevSpread = spread;
    if (Math.abs(delta) <= this.cfg.sensitivity) return;

    this.direction = delta > 0 ? 'in' : 'out';
    this.lastFire = now;
  }

  reset() {
    this.state = ZoomState.IDLE;
    this.holdFrames = 0;
    this.prevSpread = 0;
  }
}

// ---------------------------------------------------------------------------
// Pause FSM  <- gestureflow/pause_fsm.py
// ---------------------------------------------------------------------------

export const PauseState = {
  IDLE: 'IDLE', HOLDING: 'HOLDING', LATCHED: 'LATCHED',
};

export class PauseFSM {
  constructor(config = DEFAULTS.pause) {
    this.cfg = config;
    this.state = PauseState.IDLE;
    this.holdStarted = Infinity;
    this.paused = false;
    this.toggled = false;
    this.progress = 0;
  }

  update(landmarks, now) {
    this.toggled = false;
    if (!this.cfg.enabled) {
      this.progress = 0;
      return;
    }

    const present = !!landmarks && rockHorns(landmarks, this.cfg.margin);

    if (!present) {
      // A broken pose resets the timer rather than pausing it.
      this.state = PauseState.IDLE;
      this.holdStarted = Infinity;
      this.progress = 0;
      return;
    }

    if (this.state === PauseState.LATCHED) {
      this.progress = 1;
      return;
    }

    if (this.state !== PauseState.HOLDING) {
      this.state = PauseState.HOLDING;
      this.holdStarted = now;
    }

    const held = now - this.holdStarted;
    const holdFor = this.cfg.holdSeconds;
    this.progress = holdFor <= 0 ? 1 : Math.min(1, held / holdFor);

    if (held >= holdFor) {
      this.paused = !this.paused;
      this.toggled = true;
      this.state = PauseState.LATCHED;
      this.progress = 1;
    }
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
    this.leftFSM = new ClickFSM(config.click, LM.THUMB_TIP, LM.INDEX_TIP,
                                config.drag);
    this.rightFSM = new ClickFSM(config.rightClick, LM.MIDDLE_TIP, LM.INDEX_TIP);
    this.scrollFSM = new ScrollFSM(config.scroll);
    this.swipeFSM = new SwipeFSM(config.swipe);
    this.zoomFSM = new ZoomFSM(config.zoom);
    this.pauseFSM = new PauseFSM(config.pause);
  }

  /** Mirrors InferenceThread._process plus GestureRouter's mode predicates. */
  process(landmarks, now) {
    // Pause runs above the classifier, matching Python: the kill switch has to
    // work even if the model reads the pose as something else.
    this.pauseFSM.update(landmarks, now);

    if (!landmarks) {
      this.debouncer.update(0, 1.0, now);
      this.leftFSM.update(null, now);
      this.rightFSM.update(null, now);
      this.scrollFSM.update(null, now);
      this.swipeFSM.update(null, now);
      this.zoomFSM.update(null, now);
      return this.emptyResult();
    }

    if (this.pauseFSM.paused) {
      this.leftFSM.update(null, now);
      this.rightFSM.update(null, now);
      this.scrollFSM.update(null, now);
      this.swipeFSM.update(null, now);
      this.zoomFSM.update(null, now);
      return {
        ...this.emptyResult(),
        landmarks,
        paused: true,
        pauseProgress: this.pauseFSM.progress,
        mode: 'paused',
      };
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
      this.swipeFSM.update(null, now);
      this.zoomFSM.update(null, now);
    } else {
      // A fist resolves to scroll or swipe and nothing else, and a zoom pose
      // owns thumb and index outright. Both decisions are made here, once, so
      // the click FSMs never have to be consistent with a second threshold.
      const fist = isTrueScrollFist(landmarks);
      this.zoomFSM.update(landmarks, now);
      const busy = fist || this.zoomFSM.isActive;
      this.leftFSM.update(busy ? null : landmarks, now);
      this.rightFSM.update(busy ? null : landmarks, now);
      this.scrollFSM.update(landmarks, now);
      this.swipeFSM.update(landmarks, now);
    }

    const idxExt = indexExtended(landmarks);
    const thumbUp = thumbRaised(landmarks);
    // Matches GestureRouter._geometric_modes_suppressed: an emitted action
    // suppresses the geometric modes even though the debouncer has already
    // cleared its window and stable reads 0 again.
    const suppressed = stable !== 0 || action !== null;

    const scrollActive = !suppressed && this.scrollFSM.isActive;
    // No thumb gate here, matching GestureRouter.cursor_enabled: exclusivity
    // against volume comes from the index-up / index-down split instead.
    const cursorActive = !suppressed && !this.leftFSM.isActive
      && !this.rightFSM.isActive && !scrollActive && idxExt;
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
      swipeDirection: this.swipeFSM.direction,
      swipeArmed,
      zoomDirection: this.zoomFSM.direction,
      zoomActive,
      dragging: this.leftFSM.dragging,
      dragStarted: this.leftFSM.dragStarted,
      dragEnded: this.leftFSM.dragEnded,
      paused: false,
      pauseProgress: this.pauseFSM.progress,
      indexExtended: idxExt,
      thumbRaised: thumbUp,
      cursorActive,
      volumeActive,
      mode: modeOf({
        suppressed, stable, scrollActive, cursorActive, volumeActive,
        swipeArmed, zoomActive,
        dragging: this.leftFSM.dragging,
        leftActive: this.leftFSM.isActive,
        rightActive: this.rightFSM.isActive,
      }),
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
      swipeDirection: null, swipeArmed: false,
      zoomDirection: null, zoomActive: false,
      dragging: false, dragStarted: false, dragEnded: false,
      paused: false, pauseProgress: 0,
      mode: 'none',
    };
  }
}

/** Same precedence ladder as GestureRouter.active_mode in Python. */
function modeOf(s) {
  if (s.stable !== 0 || s.suppressed) return 'command';
  if (s.scrollActive) return 'scroll';
  if (s.swipeArmed) return 'swipe';
  if (s.zoomActive) return 'zoom';
  if (s.dragging) return 'drag';
  if (s.leftActive) return 'left-click';
  if (s.rightActive) return 'right-click';
  if (s.volumeActive) return 'volume';
  if (s.cursorActive) return 'cursor';
  return 'tracking';
}
