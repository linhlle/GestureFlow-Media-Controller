// Client-side mirror of gestureflow/commands.py.
//
// The builder validates before it lets you download, so a config that reaches
// the desktop app has already been through the same rules the app will apply.
// The Python side still validates on load -- this is a better error message,
// not a substitute for server-side (or in this case app-side) validation.
//
// tests/test_web_parity.py asserts these constants match the Python ones, so
// the two cannot silently diverge.

export const SCHEMA_VERSION = 2;

// Gestures detected geometrically rather than classified by the model, so they
// are bound by name instead of by model label. Mirrors NAMED_GESTURES in
// gestureflow/commands.py.
export const NAMED_GESTURES = [
  { id: 'swipe_left', label: 'Swipe left', note: 'closed fist, flick left' },
  { id: 'swipe_right', label: 'Swipe right', note: 'closed fist, flick right' },
  { id: 'zoom_in', label: 'Zoom in', note: 'thumb and index spread apart' },
  { id: 'zoom_out', label: 'Zoom out', note: 'thumb and index drawn together' },
];

export const VALID_KEYS = new Set([
  'command', 'cmd', 'ctrl', 'control', 'alt', 'option', 'shift', 'fn',
  'enter', 'return', 'tab', 'space', 'esc', 'escape', 'backspace', 'delete',
  'up', 'down', 'left', 'right', 'home', 'end', 'pageup', 'pagedown',
  'capslock', 'insert',
  ...Array.from({ length: 20 }, (_, i) => `f${i + 1}`),
  ...'abcdefghijklmnopqrstuvwxyz'.split(''),
  ...'0123456789'.split(''),
  '-', '=', '[', ']', '\\', ';', "'", ',', '.', '/', '`',
]);

export const MEDIA_ACTIONS = [
  'playpause', 'next', 'previous', 'volumeup', 'volumedown', 'mute',
];

export const ACTION_TYPES = [
  { id: 'hotkey', label: 'Keyboard shortcut',
    help: 'Sends a key combination, e.g. command + space.' },
  { id: 'keypress', label: 'Single keypress',
    help: 'Presses one key.' },
  { id: 'media', label: 'Media key',
    help: 'Play/pause, track skip, volume, or mute.' },
  { id: 'launch', label: 'Launch app',
    help: 'Opens an application by name, e.g. Notes.' },
  { id: 'applescript', label: 'AppleScript',
    help: 'Runs an AppleScript snippet through osascript.' },
  { id: 'url', label: 'Open a link',
    help: 'Opens a URL in your default browser. http and https only.' },
  { id: 'text', label: 'Type text',
    help: 'Types a snippet. Typed rather than pasted, so your clipboard is '
        + 'left alone.' },
  { id: 'chord', label: 'Key sequence',
    help: 'An ordered list of shortcuts, with a pause between each.' },
  { id: 'shell', label: 'Shell command',
    help: 'Runs a command as an argument list. It is executed without a '
        + 'shell, so pipes and redirects do not apply.' },
];

// Mirrors URL_SCHEMES / the length caps in commands.py.
export const URL_SCHEMES = ['http', 'https'];
export const MAX_URL_LENGTH = 2000;
export const MAX_TEXT_LENGTH = 500;
export const MAX_CHORD_STEPS = 12;
export const MAX_CHORD_DELAY = 2.0;

// Anchored, matching _APP_NAME_RE in commands.py.
const APP_NAME_RE = /^[A-Za-z0-9 ._+-]{1,64}$/;

/** Poses the shipped model can predict. Index == label. */
export const KNOWN_POSES = [
  { label: 0, name: 'Neutral', note: 'no gesture — enables cursor, click, scroll, volume' },
  { label: 1, name: 'L-Shape', note: 'thumb and index at a right angle' },
  { label: 2, name: 'High-Five', note: 'open palm facing the camera' },
  { label: 3, name: '2-Finger', note: 'index and middle extended' },
];

export const MODEL_CLASSES = [0, 1, 2, 3];
export const NEUTRAL_LABEL = 0;

export function validateAction(action, where) {
  const errors = [];
  const fail = (msg) => errors.push(`${where}: ${msg}`);

  if (!action || !action.type) {
    fail('an action type is required');
    return errors;
  }

  switch (action.type) {
    case 'hotkey': {
      const keys = (action.keys || []).filter((k) => k && k.trim());
      if (keys.length === 0) fail('add at least one key');
      if (keys.length > 5) fail('at most 5 keys');
      keys.forEach((k) => {
        if (!VALID_KEYS.has(k.trim().toLowerCase())) {
          fail(`"${k}" is not a recognized key name`);
        }
      });
      break;
    }
    case 'keypress': {
      const key = (action.key || '').trim().toLowerCase();
      if (!key) fail('choose a key');
      else if (!VALID_KEYS.has(key)) fail(`"${action.key}" is not a recognized key`);
      break;
    }
    case 'media':
      if (!MEDIA_ACTIONS.includes(action.media)) fail('choose a media action');
      break;
    case 'launch': {
      const app = (action.app || '').trim();
      if (!app) fail('enter an application name');
      else if (!APP_NAME_RE.test(app)) {
        fail('application names may only contain letters, digits, spaces, '
             + 'and . _ + -');
      }
      break;
    }
    case 'applescript':
      if (!(action.script || '').trim()) fail('enter a script');
      else if (action.script.length > 4000) fail('script exceeds 4000 characters');
      break;
    case 'shell': {
      const argv = (action.argv || []).filter((a) => a !== '');
      if (argv.length === 0) fail('enter at least the command name');
      if (argv.length > 32) fail('at most 32 arguments');
      break;
    }
    case 'url': {
      const url = (action.url || '').trim();
      if (!url) fail('enter a URL');
      else if (url.length > MAX_URL_LENGTH) {
        fail(`URL is longer than the ${MAX_URL_LENGTH}-character limit`);
      } else {
        const scheme = url.includes(':') ? url.split(':', 1)[0].toLowerCase() : '';
        if (!URL_SCHEMES.includes(scheme)) {
          fail('URLs must start with http:// or https://');
        } else if (/[\r\n]/.test(url)) {
          fail('a URL cannot contain line breaks');
        }
      }
      break;
    }
    case 'text': {
      const text = action.text || '';
      if (!text) fail('enter some text');
      else if (text.length > MAX_TEXT_LENGTH) {
        fail(`text is longer than the ${MAX_TEXT_LENGTH}-character limit`);
      } else if (/[\u0000-\u0008\u000b-\u001f]/.test(text)) {
        fail('text contains control characters that cannot be typed');
      }
      break;
    }
    case 'chord': {
      const steps = (action.steps || []).filter(
        (st) => (st.keys || []).some((k) => k && k.trim()));
      if (steps.length === 0) fail('add at least one step');
      if (steps.length > MAX_CHORD_STEPS) {
        fail(`at most ${MAX_CHORD_STEPS} steps`);
      }
      steps.forEach((step, i) => {
        const keys = (step.keys || []).filter((k) => k && k.trim());
        if (keys.length > 5) fail(`step ${i + 1}: at most 5 keys`);
        keys.forEach((k) => {
          if (!VALID_KEYS.has(k.trim().toLowerCase())) {
            fail(`step ${i + 1}: "${k}" is not a recognized key name`);
          }
        });
        const delay = step.delay === undefined ? 0.05 : Number(step.delay);
        if (!Number.isFinite(delay) || delay < 0 || delay > MAX_CHORD_DELAY) {
          fail(`step ${i + 1}: delay must be between 0 and ${MAX_CHORD_DELAY}s`);
        }
      });
      break;
    }
    default:
      fail(`unknown action type "${action.type}"`);
  }
  return errors;
}

const NAMED_IDS = NAMED_GESTURES.map((g) => g.id);

/** True when a binding is keyed by geometric gesture rather than model label. */
export function isNamed(binding) {
  return typeof binding.gesture === 'string' && binding.gesture !== '';
}

export function validateConfig(bindings) {
  const errors = [];
  if (bindings.length === 0) errors.push('Add at least one gesture binding.');

  const seenLabels = new Map();
  const seenGestures = new Map();

  bindings.forEach((b, i) => {
    const where = b.name ? `"${b.name}"` : `gesture ${i + 1}`;

    if (isNamed(b)) {
      if (!NAMED_IDS.includes(b.gesture)) {
        errors.push(`${where}: unknown gesture "${b.gesture}"`);
      } else if (seenGestures.has(b.gesture)) {
        errors.push(`${where}: ${b.gesture} is already bound to `
                    + `"${seenGestures.get(b.gesture)}"`);
      } else {
        seenGestures.set(b.gesture, b.name || where);
      }
    } else if (!Number.isInteger(b.label) || b.label < 0) {
      errors.push(`${where}: choose a gesture`);
    } else if (b.label === NEUTRAL_LABEL) {
      errors.push(`${where}: Neutral is the "no gesture" state and cannot have `
                  + 'an action bound to it');
    } else if (seenLabels.has(b.label)) {
      errors.push(`${where}: gesture ${b.label} is already bound to `
                  + `"${seenLabels.get(b.label)}"`);
    } else {
      seenLabels.set(b.label, b.name || where);
    }

    if (!(b.name || '').trim()) errors.push(`${where}: give it a name`);
    else if (b.name.length > 64) errors.push(`${where}: name is over 64 characters`);

    errors.push(...validateAction(b.action, where));
  });

  return errors;
}

/** Warnings that do not block export but are worth surfacing. */
export function configWarnings(bindings) {
  const warnings = [];
  const bound = new Set(bindings.filter((b) => !isNamed(b)).map((b) => b.label));

  MODEL_CLASSES.filter((c) => c !== NEUTRAL_LABEL).forEach((c) => {
    if (!bound.has(c)) {
      const pose = KNOWN_POSES.find((p) => p.label === c);
      warnings.push(
        `Gesture ${c} (${pose ? pose.name : '?'}) has no action bound. The app `
        + 'refuses to start when a pose it can recognize does nothing — bind it '
        + 'or retrain without it.',
      );
    }
  });

  bindings.forEach((b) => {
    if (isNamed(b)) return;
    if (Number.isInteger(b.label) && !MODEL_CLASSES.includes(b.label)) {
      warnings.push(
        `Gesture ${b.label} is bound, but the shipped model cannot predict it. `
        + 'It will never fire until you collect data for it and retrain.',
      );
    }
  });

  return warnings;
}

export function toConfigObject(bindings) {
  return {
    version: SCHEMA_VERSION,
    neutral_label: NEUTRAL_LABEL,
    gestures: bindings
      .slice()
      .sort((a, b) => {
        // Pose bindings first, then named gestures, each in a stable order --
        // matching how CommandSet.to_dict emits them on the Python side.
        if (isNamed(a) !== isNamed(b)) return isNamed(a) ? 1 : -1;
        if (isNamed(a)) return a.gesture.localeCompare(b.gesture);
        return a.label - b.label;
      })
      .map((b) => {
        const entry = isNamed(b) ? { gesture: b.gesture } : { label: b.label };
        entry.name = (b.name || '').trim();
        if ((b.description || '').trim()) entry.description = b.description.trim();
        entry.action = actionToObject(b.action);
        return entry;
      }),
  };
}

function actionToObject(action) {
  switch (action.type) {
    case 'hotkey':
      return { type: 'hotkey',
               keys: action.keys.filter((k) => k && k.trim())
                 .map((k) => k.trim().toLowerCase()) };
    case 'keypress':
      return { type: 'keypress', key: action.key.trim().toLowerCase() };
    case 'media':
      return { type: 'media', action: action.media };
    case 'launch':
      return { type: 'launch', app: action.app.trim() };
    case 'applescript':
      return { type: 'applescript', script: action.script };
    case 'shell':
      return { type: 'shell', argv: action.argv.filter((a) => a !== '') };
    case 'url':
      return { type: 'url', url: action.url.trim() };
    case 'text':
      return { type: 'text', text: action.text };
    case 'chord':
      return {
        type: 'chord',
        steps: (action.steps || [])
          .filter((st) => (st.keys || []).some((k) => k && k.trim()))
          .map((st) => ({
            keys: st.keys.filter((k) => k && k.trim())
              .map((k) => k.trim().toLowerCase()),
            delay: st.delay === undefined ? 0.05 : Number(st.delay),
          })),
      };
    default:
      throw new Error(`unknown action type ${action.type}`);
  }
}

/** Minimal YAML emitter for the shapes this schema produces. */
export function toYaml(config) {
  const lines = [
    '# GestureFlow command bindings',
    '# Built at the GestureFlow command builder.',
    '# Save as ~/.gestureflow/commands.yaml',
    '',
    `version: ${config.version}`,
    `neutral_label: ${config.neutral_label}`,
    '',
    'gestures:',
  ];

  for (const g of config.gestures) {
    lines.push(g.gesture !== undefined
      ? `  - gesture: ${g.gesture}`
      : `  - label: ${g.label}`);
    lines.push(`    name: ${yamlScalar(g.name)}`);
    if (g.description) lines.push(`    description: ${yamlScalar(g.description)}`);
    lines.push('    action:');
    lines.push(`      type: ${g.action.type}`);

    if (g.action.type === 'chord') {
      lines.push('      steps:');
      for (const step of g.action.steps) {
        lines.push(`        - keys: [${step.keys.map(yamlScalar).join(', ')}]`);
        lines.push(`          delay: ${step.delay}`);
      }
    } else if (g.action.type === 'url') {
      lines.push(`      url: ${yamlScalar(g.action.url)}`);
    } else if (g.action.type === 'text') {
      lines.push(`      text: ${yamlScalar(g.action.text)}`);
    } else if (g.action.type === 'hotkey') {
      lines.push(`      keys: [${g.action.keys.map(yamlScalar).join(', ')}]`);
    } else if (g.action.type === 'keypress') {
      lines.push(`      key: ${yamlScalar(g.action.key)}`);
    } else if (g.action.type === 'media') {
      lines.push(`      action: ${g.action.action}`);
    } else if (g.action.type === 'launch') {
      lines.push(`      app: ${yamlScalar(g.action.app)}`);
    } else if (g.action.type === 'applescript') {
      lines.push('      script: |-');
      for (const line of g.action.script.split('\n')) {
        lines.push(`        ${line}`);
      }
    } else if (g.action.type === 'shell') {
      lines.push(`      argv: [${g.action.argv.map(yamlScalar).join(', ')}]`);
    }
    lines.push('');
  }

  return `${lines.join('\n').trimEnd()}\n`;
}

// Quote anything that YAML would otherwise reinterpret: digits, booleans,
// nulls, or strings carrying structural characters.
function yamlScalar(value) {
  const s = String(value);
  const needsQuotes = s === ''
    || /^[\d.+-]/.test(s)
    || /[:#\[\]{}&*!|>'"%@`,]/.test(s)
    || /^(y|n|yes|no|true|false|on|off|null|~)$/i.test(s)
    || s !== s.trim();
  if (!needsQuotes) return s;
  return `"${s.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
}

/** Parse a config the user pastes or uploads, so edits round-trip. */
export function fromConfigObject(raw) {
  if (!raw || typeof raw !== 'object') throw new Error('Config must be an object.');
  if (raw.version !== SCHEMA_VERSION) {
    throw new Error(`Unsupported config version ${raw.version} `
                    + `(this builder writes version ${SCHEMA_VERSION}).`);
  }
  if (!Array.isArray(raw.gestures)) throw new Error('Config has no gestures list.');

  return raw.gestures.map((g) => ({
    label: g.label,
    gesture: g.gesture,
    name: g.name || '',
    description: g.description || '',
    action: actionFromObject(g.action || {}),
  }));
}

function actionFromObject(a) {
  const base = {
    type: a.type || 'hotkey',
    keys: ['command', ''],
    key: '',
    media: 'playpause',
    app: '',
    script: '',
    argv: [''],
    url: '',
    text: '',
    steps: [{ keys: ['command', ''], delay: 0.05 }],
  };
  if (a.type === 'hotkey') base.keys = (a.keys || []).slice();
  if (a.type === 'keypress') base.key = a.key || '';
  if (a.type === 'media') base.media = a.action || 'playpause';
  if (a.type === 'launch') base.app = a.app || '';
  if (a.type === 'applescript') base.script = a.script || '';
  if (a.type === 'shell') base.argv = (a.argv || []).slice();
  if (a.type === 'url') base.url = a.url || '';
  if (a.type === 'text') base.text = a.text || '';
  if (a.type === 'chord') base.steps = (a.steps || []).map((st) => ({
    keys: (st.keys || []).slice(),
    delay: st.delay === undefined ? 0.05 : st.delay,
  }));
  return base;
}
