// Command builder UI.
//
// Renders one card per binding, validates continuously against the same rules
// gestureflow/commands.py applies, and emits a config the desktop app accepts
// unchanged. State lives in localStorage so a half-finished config survives a
// reload.

import {
  ACTION_TYPES,
  KNOWN_POSES,
  MEDIA_ACTIONS,
  configWarnings,
  fromConfigObject,
  toConfigObject,
  toYaml,
  validateConfig,
} from './schema.js';
import { BridgeClient } from './bridge.js';

const STORAGE_KEY = 'gestureflow.builder.v1';

const DEFAULT_BINDINGS = [
  { label: 1, name: 'Spotlight', description: 'Open Spotlight search',
    action: newAction('hotkey', { keys: ['command', 'space'] }) },
  { label: 2, name: 'Mission Control', description: 'Show all open windows',
    action: newAction('hotkey', { keys: ['ctrl', 'up'] }) },
  { label: 3, name: 'App Switcher', description: 'Cycle between applications',
    action: newAction('hotkey', { keys: ['command', 'tab'] }) },
];

function newAction(type = 'hotkey', overrides = {}) {
  return {
    type,
    keys: ['command', ''],
    key: 'escape',
    media: 'playpause',
    app: '',
    script: '',
    argv: [''],
    ...overrides,
  };
}

let bindings = load();

const els = {
  list: document.getElementById('bindings'),
  preview: document.getElementById('yaml-preview'),
  messages: document.getElementById('messages'),
  addBtn: document.getElementById('add-btn'),
  resetBtn: document.getElementById('reset-btn'),
  importBtn: document.getElementById('import-btn'),
  importFile: document.getElementById('import-file'),
  downloadBtn: document.getElementById('download-btn'),
  copyBtn: document.getElementById('copy-btn'),
  pushBtn: document.getElementById('push-btn'),
  bridgeStatus: document.getElementById('bridge-status'),
};

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length) return parsed;
    }
  } catch {
    // Corrupt or unavailable storage is not worth failing over; fall through
    // to defaults.
  }
  return structuredClone(DEFAULT_BINDINGS);
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(bindings));
  } catch {
    // Private browsing: the builder still works, it just will not persist.
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render() {
  els.list.innerHTML = '';
  bindings.forEach((binding, index) => {
    els.list.appendChild(renderBinding(binding, index));
  });
  updatePreview();
  save();
}

function renderBinding(binding, index) {
  const card = el('div', 'binding');

  const head = el('div', 'binding-head');
  head.appendChild(gestureSelect(binding, index));
  const spacer = el('span', 'spacer');
  head.appendChild(spacer);
  const remove = el('button', 'btn secondary small');
  remove.textContent = 'Remove';
  remove.addEventListener('click', () => {
    bindings.splice(index, 1);
    render();
  });
  head.appendChild(remove);
  card.appendChild(head);

  card.appendChild(field('Name', textInput(binding.name, (v) => {
    binding.name = v;
    updatePreview();
    save();
  }), 'Shown in the app HUD when the gesture fires.'));

  card.appendChild(field('Description (optional)',
    textInput(binding.description, (v) => {
      binding.description = v;
      updatePreview();
      save();
    })));

  card.appendChild(actionEditor(binding, index));
  return card;
}

function gestureSelect(binding, index) {
  const wrap = el('div');
  wrap.style.display = 'flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '0.5rem';

  const select = document.createElement('select');
  select.style.width = 'auto';
  KNOWN_POSES.forEach((pose) => {
    const opt = document.createElement('option');
    opt.value = String(pose.label);
    opt.textContent = `${pose.label} — ${pose.name}`;
    opt.disabled = pose.label === 0;
    select.appendChild(opt);
  });

  // Allow labels beyond the shipped model, for users who have retrained.
  for (let extra = 4; extra <= 9; extra += 1) {
    const opt = document.createElement('option');
    opt.value = String(extra);
    opt.textContent = `${extra} — custom (needs retraining)`;
    select.appendChild(opt);
  }

  select.value = String(binding.label);
  select.addEventListener('change', () => {
    binding.label = Number(select.value);
    render();
  });

  wrap.appendChild(select);

  const pose = KNOWN_POSES.find((p) => p.label === binding.label);
  if (pose) {
    const note = el('span', 'gesture-label');
    note.textContent = pose.note;
    wrap.appendChild(note);
  }
  return wrap;
}

function actionEditor(binding) {
  const wrap = el('div');

  const select = document.createElement('select');
  ACTION_TYPES.forEach((type) => {
    const opt = document.createElement('option');
    opt.value = type.id;
    opt.textContent = type.label;
    select.appendChild(opt);
  });
  select.value = binding.action.type;
  select.addEventListener('change', () => {
    binding.action.type = select.value;
    render();
  });

  wrap.appendChild(field('Action', select,
    (ACTION_TYPES.find((t) => t.id === binding.action.type) || {}).help));

  const a = binding.action;

  if (a.type === 'hotkey') {
    const row = el('div', 'field-row');
    [0, 1, 2].forEach((i) => {
      row.appendChild(textInput(a.keys[i] || '', (v) => {
        a.keys[i] = v;
        updatePreview();
        save();
      }, i === 0 ? 'command' : 'key (optional)'));
    });
    wrap.appendChild(field('Keys', row,
      'Modifiers plus one key. Leave unused boxes empty.'));
  } else if (a.type === 'keypress') {
    wrap.appendChild(field('Key', textInput(a.key, (v) => {
      a.key = v;
      updatePreview();
      save();
    }, 'escape')));
  } else if (a.type === 'media') {
    const select2 = document.createElement('select');
    MEDIA_ACTIONS.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      select2.appendChild(opt);
    });
    select2.value = a.media;
    select2.addEventListener('change', () => {
      a.media = select2.value;
      updatePreview();
      save();
    });
    wrap.appendChild(field('Media action', select2));
  } else if (a.type === 'launch') {
    wrap.appendChild(field('Application name', textInput(a.app, (v) => {
      a.app = v;
      updatePreview();
      save();
    }, 'Notes'), 'Exactly as it appears in your Applications folder.'));
  } else if (a.type === 'applescript') {
    const ta = document.createElement('textarea');
    ta.value = a.script;
    ta.placeholder = 'tell application "Music" to playpause';
    ta.addEventListener('input', () => {
      a.script = ta.value;
      updatePreview();
      save();
    });
    wrap.appendChild(field('Script', ta,
      'Runs via osascript. macOS will ask permission the first time.'));
  } else if (a.type === 'shell') {
    const row = el('div', 'field-row');
    [0, 1, 2].forEach((i) => {
      row.appendChild(textInput(a.argv[i] || '', (v) => {
        a.argv[i] = v;
        updatePreview();
        save();
      }, i === 0 ? 'command' : 'argument (optional)'));
    });
    wrap.appendChild(field('Command and arguments', row,
      'One box per argument. Run without a shell, so no pipes or redirects.'));
  }

  return wrap;
}

// ---------------------------------------------------------------------------
// Preview and validation
// ---------------------------------------------------------------------------

function updatePreview() {
  const errors = validateConfig(bindings);
  els.messages.innerHTML = '';

  if (errors.length) {
    const box = el('div', 'error-box');
    box.textContent = errors.join('\n');
    box.style.whiteSpace = 'pre-wrap';
    els.messages.appendChild(box);
    els.preview.textContent = '# fix the problems above to see the config';
    els.downloadBtn.disabled = true;
    els.copyBtn.disabled = true;
    els.pushBtn.disabled = true;
    return;
  }

  const warnings = configWarnings(bindings);
  warnings.forEach((w) => {
    const box = el('div', 'notice');
    box.style.margin = '0 0 0.6rem';
    box.style.fontSize = '0.83rem';
    box.textContent = w;
    els.messages.appendChild(box);
  });

  els.preview.textContent = toYaml(toConfigObject(bindings));
  els.downloadBtn.disabled = false;
  els.copyBtn.disabled = false;
  els.pushBtn.disabled = !bridge.connected;
}

// ---------------------------------------------------------------------------
// Small DOM helpers
// ---------------------------------------------------------------------------

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function textInput(value, onChange, placeholder = '') {
  const input = document.createElement('input');
  input.type = 'text';
  input.value = value || '';
  input.placeholder = placeholder;
  input.addEventListener('input', () => onChange(input.value));
  return input;
}

function field(labelText, control, hint) {
  const wrap = el('div', 'field');
  const label = document.createElement('label');
  label.textContent = labelText;
  wrap.appendChild(label);
  wrap.appendChild(control);
  if (hint) {
    const p = el('p', 'hint');
    p.textContent = hint;
    wrap.appendChild(p);
  }
  return wrap;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

els.addBtn.addEventListener('click', () => {
  const used = new Set(bindings.map((b) => b.label));
  let next = 1;
  while (used.has(next) && next < 10) next += 1;
  bindings.push({ label: next, name: '', description: '', action: newAction() });
  render();
});

els.resetBtn.addEventListener('click', () => {
  bindings = structuredClone(DEFAULT_BINDINGS);
  render();
});

els.importBtn.addEventListener('click', () => els.importFile.click());

els.importFile.addEventListener('change', async () => {
  const file = els.importFile.files && els.importFile.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const raw = file.name.endsWith('.json')
      ? JSON.parse(text)
      : parseSimpleYaml(text);
    bindings = fromConfigObject(raw);
    render();
    flash('Imported ' + file.name, 'ok');
  } catch (err) {
    flash(`Could not import: ${err.message}`, 'error');
  }
  els.importFile.value = '';
});

els.downloadBtn.addEventListener('click', () => {
  const yaml = toYaml(toConfigObject(bindings));
  const blob = new Blob([yaml], { type: 'text/yaml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'commands.yaml';
  a.click();
  URL.revokeObjectURL(url);
});

els.copyBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(toYaml(toConfigObject(bindings)));
    flash('Copied to clipboard', 'ok');
  } catch {
    flash('Clipboard blocked — use Download instead', 'error');
  }
});

els.pushBtn.addEventListener('click', async () => {
  try {
    await bridge.pushConfig(toConfigObject(bindings));
    flash('Sent to the desktop app', 'ok');
  } catch (err) {
    flash(`Push failed: ${err.message}`, 'error');
  }
});

function flash(message, kind) {
  const box = el('div', kind === 'ok' ? 'ok-box' : 'error-box');
  box.textContent = message;
  els.messages.prepend(box);
  setTimeout(() => box.remove(), 4000);
}

// ---------------------------------------------------------------------------
// Optional local bridge
// ---------------------------------------------------------------------------

const bridge = new BridgeClient();

bridge.onStatus = (connected) => {
  const dot = els.bridgeStatus.querySelector('.status-dot');
  dot.classList.toggle('live', connected);
  els.bridgeStatus.lastChild.textContent = connected
    ? 'Local bridge connected — you can push directly'
    : 'Local bridge not detected';
  els.pushBtn.disabled = !connected || els.downloadBtn.disabled;
};

bridge.tryConnect();

// ---------------------------------------------------------------------------
// A YAML subset parser, just wide enough to re-import what we emit.
// Full YAML is not worth a dependency here; anything this cannot handle is
// reported rather than guessed at.
// ---------------------------------------------------------------------------

function parseSimpleYaml(text) {
  const out = { gestures: [] };
  let current = null;
  let inAction = false;
  let scriptLines = null;

  for (const rawLine of text.split('\n')) {
    const line = rawLine.replace(/\s+$/, '');
    if (scriptLines !== null) {
      if (/^ {8}/.test(line)) {
        scriptLines.push(line.slice(8));
        continue;
      }
      current.action.script = scriptLines.join('\n');
      scriptLines = null;
    }

    if (!line.trim() || line.trim().startsWith('#')) continue;

    let m;
    if ((m = line.match(/^version:\s*(\d+)/))) {
      out.version = Number(m[1]);
    } else if ((m = line.match(/^neutral_label:\s*(\d+)/))) {
      out.neutral_label = Number(m[1]);
    } else if (line.match(/^gestures:/)) {
      // container line
    } else if ((m = line.match(/^ {2}- label:\s*(\d+)/))) {
      current = { label: Number(m[1]), action: {} };
      out.gestures.push(current);
      inAction = false;
    } else if ((m = line.match(/^ {4}name:\s*(.+)/))) {
      current.name = unquote(m[1]);
    } else if ((m = line.match(/^ {4}description:\s*(.+)/))) {
      current.description = unquote(m[1]);
    } else if (line.match(/^ {4}action:/)) {
      inAction = true;
    } else if (inAction && (m = line.match(/^ {6}(\w+):\s*(.*)$/))) {
      const [, key, value] = m;
      if (key === 'script' && value.trim().startsWith('|')) {
        scriptLines = [];
      } else if (value.trim().startsWith('[')) {
        current.action[key] = value.trim()
          .slice(1, -1)
          .split(',')
          .map((s) => unquote(s.trim()))
          .filter((s) => s !== '');
      } else {
        current.action[key] = unquote(value);
      }
    }
  }

  if (scriptLines !== null) current.action.script = scriptLines.join('\n');
  if (out.version === undefined) {
    throw new Error('no "version:" line found — is this a GestureFlow config?');
  }
  return out;
}

function unquote(s) {
  const t = s.trim();
  if ((t.startsWith('"') && t.endsWith('"'))
      || (t.startsWith("'") && t.endsWith("'"))) {
    return t.slice(1, -1).replace(/\\"/g, '"').replace(/\\\\/g, '\\');
  }
  return t;
}

render();
