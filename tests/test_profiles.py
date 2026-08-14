"""App-context profiles.

A profile changes what a gesture means depending on which app is in front, so
a swipe can be "next slide" in Keynote and "next desktop" everywhere else.

The frontmost-app lookup is the part that touches the OS, so it sits behind a
provider interface with a fake implementation. Everything below tests the
resolution logic with no operating system in the loop; the one thing that
genuinely needs macOS is called out and skipped elsewhere.
"""

from __future__ import annotations

import platform
import threading
import time

import pytest

from gestureflow.commands import CommandConfigError, parse_commands
from gestureflow.frontmost import (
    FrontmostApp,
    PollingFrontmostApp,
    StaticFrontmostApp,
    make_provider,
    query_frontmost_macos,
)


def cfg(profiles=None):
    doc = {
        "version": 2,
        "neutral_label": 0,
        "gestures": [
            {"label": 1, "name": "Spotlight",
             "action": {"type": "hotkey", "keys": ["command", "space"]}},
            {"gesture": "swipe_left", "name": "Previous desktop",
             "action": {"type": "hotkey", "keys": ["ctrl", "left"]}},
            {"gesture": "swipe_right", "name": "Next desktop",
             "action": {"type": "hotkey", "keys": ["ctrl", "right"]}},
        ],
    }
    if profiles is not None:
        doc["profiles"] = profiles
    return doc


PRESENTATION = {
    "name": "presentation",
    "match": {"apps": ["Keynote", "Microsoft PowerPoint"]},
    "gestures": [
        {"gesture": "swipe_right", "name": "Next slide",
         "action": {"type": "keypress", "key": "right"}},
        {"gesture": "swipe_left", "name": "Previous slide",
         "action": {"type": "keypress", "key": "left"}},
    ],
}

EDITOR = {
    "name": "editor",
    "match": {"apps": ["Code"]},
    "gestures": [
        {"gesture": "swipe_right", "name": "Next tab",
         "action": {"type": "hotkey", "keys": ["command", "shift", "]"]}},
    ],
}


class TestResolution:
    def test_no_profiles_means_the_default_always_applies(self):
        commands = parse_commands(cfg())
        assert commands.resolve("Keynote") is commands
        assert commands.resolve(None) is commands

    def test_a_matching_app_gets_its_profile(self):
        commands = parse_commands(cfg([PRESENTATION]))
        resolved = commands.resolve("Keynote")
        assert resolved.get_named("swipe_right").name == "Next slide"

    def test_matching_is_case_insensitive(self):
        commands = parse_commands(cfg([PRESENTATION]))
        for name in ("keynote", "KEYNOTE", "Keynote", "  Keynote  "):
            assert commands.resolve(name).get_named("swipe_right").name \
                == "Next slide"

    def test_any_app_in_the_match_list_works(self):
        commands = parse_commands(cfg([PRESENTATION]))
        assert commands.resolve("Microsoft PowerPoint") \
            .get_named("swipe_right").name == "Next slide"

    def test_an_unmatched_app_falls_back_to_the_default(self):
        commands = parse_commands(cfg([PRESENTATION]))
        assert commands.resolve("Safari").get_named("swipe_right").name \
            == "Next desktop"

    def test_an_unknown_app_falls_back_to_the_default(self):
        """Provider failure looks exactly like no profile matching."""
        commands = parse_commands(cfg([PRESENTATION]))
        assert commands.resolve(None).get_named("swipe_right").name \
            == "Next desktop"

    def test_a_profile_inherits_what_it_does_not_override(self):
        """A profile is a small edit, not a second whole config."""
        commands = parse_commands(cfg([EDITOR]))
        resolved = commands.resolve("Code")
        assert resolved.get_named("swipe_right").name == "Next tab"
        assert resolved.get_named("swipe_left").name == "Previous desktop"
        assert resolved.get(1).name == "Spotlight"

    def test_first_matching_profile_wins(self):
        """Config order is the tie-break, so precedence reads top to bottom."""
        both = dict(EDITOR)
        both = {**EDITOR, "match": {"apps": ["Keynote"]}, "name": "other"}
        commands = parse_commands(cfg([PRESENTATION, both]))
        assert commands.resolve("Keynote").get_named("swipe_right").name \
            == "Next slide"

    def test_profile_for_reports_the_name(self):
        commands = parse_commands(cfg([PRESENTATION]))
        assert commands.profile_for("Keynote") == "presentation"
        assert commands.profile_for("Safari") is None
        assert commands.profile_for(None) is None

    def test_resolving_twice_does_not_compound(self):
        commands = parse_commands(cfg([PRESENTATION, EDITOR]))
        once = commands.resolve("Keynote")
        # Resolving the *original* again must give a clean answer, not one
        # layered on the previous resolution.
        assert commands.resolve("Code").get_named("swipe_right").name \
            == "Next tab"
        assert once.get_named("swipe_right").name == "Next slide"


class TestProfileValidation:
    def test_a_profile_needs_a_name(self):
        with pytest.raises(CommandConfigError, match="name"):
            parse_commands(cfg([{"match": {"apps": ["X"]},
                                 "gestures": PRESENTATION["gestures"]}]))

    def test_a_profile_needs_match_apps(self):
        with pytest.raises(CommandConfigError, match="match"):
            parse_commands(cfg([{"name": "p",
                                 "gestures": PRESENTATION["gestures"]}]))

    def test_an_empty_app_list_is_rejected(self):
        with pytest.raises(CommandConfigError, match="non-empty"):
            parse_commands(cfg([{"name": "p", "match": {"apps": []},
                                 "gestures": PRESENTATION["gestures"]}]))

    def test_a_profile_that_overrides_nothing_is_rejected(self):
        """It would silently do nothing, which is worse than a load error."""
        with pytest.raises(CommandConfigError, match="does nothing"):
            parse_commands(cfg([{"name": "p", "match": {"apps": ["X"]},
                                 "gestures": []}]))

    def test_duplicate_profile_names_are_rejected(self):
        with pytest.raises(CommandConfigError, match="already defined"):
            parse_commands(cfg([PRESENTATION, PRESENTATION]))

    def test_bindings_inside_a_profile_are_validated_too(self):
        with pytest.raises(CommandConfigError, match="not a recognized key"):
            parse_commands(cfg([{
                "name": "p", "match": {"apps": ["X"]},
                "gestures": [{"gesture": "swipe_right", "name": "X",
                              "action": {"type": "hotkey",
                                         "keys": ["comand", "k"]}}],
            }]))

    def test_profiles_need_version_2(self):
        doc = cfg([PRESENTATION])
        doc["version"] = 1
        # Strip the named gestures so version 2 is required only by profiles.
        doc["gestures"] = [doc["gestures"][0]]
        with pytest.raises(CommandConfigError, match="version 2"):
            parse_commands(doc)

    def test_the_error_names_the_offending_profile(self):
        with pytest.raises(CommandConfigError) as exc:
            parse_commands(cfg([{
                "name": "presentation", "match": {"apps": ["X"]},
                "gestures": [{"gesture": "swipe_right", "name": "Slide",
                              "action": {"type": "url", "url": "javascript:x"}}],
            }]))
        assert "presentation" in str(exc.value)


class TestProfileRoundTrip:
    def test_profiles_survive_a_yaml_round_trip(self):
        import yaml
        original = parse_commands(cfg([PRESENTATION, EDITOR]))
        reparsed = parse_commands(yaml.safe_load(original.to_yaml()))
        assert [p.name for p in reparsed.profiles] == ["presentation", "editor"]
        assert reparsed.resolve("Keynote").get_named("swipe_right").name \
            == "Next slide"

    def test_a_config_without_profiles_round_trips_unchanged(self):
        original = parse_commands(cfg())
        assert "profiles" not in original.to_dict()


class TestProviders:
    def test_the_base_provider_reports_unknown(self):
        assert FrontmostApp().current() is None

    def test_the_static_provider_reports_what_it_was_given(self):
        assert StaticFrontmostApp("Keynote").current() == "Keynote"

    def test_a_disabled_provider_never_polls(self):
        provider = make_provider(enabled=False)
        assert isinstance(provider, FrontmostApp)
        assert not isinstance(provider, PollingFrontmostApp)
        assert provider.current() is None

    def test_polling_caches_the_answer(self):
        calls = []

        def fake():
            calls.append(1)
            return "Keynote"

        provider = PollingFrontmostApp(interval=0.02, query=fake)
        provider.start()
        try:
            deadline = time.monotonic() + 1.0
            while provider.current() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert provider.current() == "Keynote"
        finally:
            provider.stop()
        assert calls, "the provider never queried"

    def test_a_query_that_raises_degrades_to_unknown(self):
        """Automation permission denied must not break anything."""
        def explode():
            raise RuntimeError("Automation permission denied")

        provider = PollingFrontmostApp(interval=0.02, query=explode)
        provider.start()
        try:
            time.sleep(0.1)
            assert provider.current() is None
        finally:
            provider.stop()

    def test_the_reader_never_blocks_on_the_subprocess(self):
        """current() must be a cached read, never a spawn."""
        def slow():
            time.sleep(0.5)
            return "Slow"

        provider = PollingFrontmostApp(interval=0.01, query=slow)
        provider.start()
        try:
            start = time.monotonic()
            for _ in range(200):
                provider.current()
            elapsed = time.monotonic() - start
            assert elapsed < 0.1, (
                f"200 reads took {elapsed:.3f}s -- current() is doing work"
            )
        finally:
            provider.stop()

    def test_stopping_is_idempotent_and_joins(self):
        provider = PollingFrontmostApp(interval=0.02,
                                       query=lambda: "X")
        provider.start()
        provider.stop()
        provider.stop()
        assert not any(t.name == "frontmost-app" and t.is_alive()
                       for t in threading.enumerate())

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_the_real_query_returns_a_string_or_none(self):
        """Never raises, whatever the permission state."""
        result = query_frontmost_macos()
        assert result is None or isinstance(result, str)

    @pytest.mark.skipif(platform.system() == "Darwin",
                        reason="tests the non-macOS path")
    def test_the_real_query_is_none_off_macos(self):
        assert query_frontmost_macos() is None


class TestControllerProfileSwitching:
    """The controller layers profiles onto the unresolved base set."""

    def _controller(self, commands):
        from gestureflow.controller import SystemController
        ctrl = SystemController.__new__(SystemController)
        ctrl._commands = commands
        ctrl._base_commands = commands
        ctrl._current_app = None
        ctrl._active_profile = None
        return ctrl

    def test_switching_apps_swaps_the_bindings(self):
        commands = parse_commands(cfg([PRESENTATION, EDITOR]))
        ctrl = self._controller(commands)

        ctrl.set_frontmost_app("Keynote")
        assert ctrl.commands.get_named("swipe_right").name == "Next slide"

        ctrl.set_frontmost_app("Code")
        assert ctrl.commands.get_named("swipe_right").name == "Next tab"

        ctrl.set_frontmost_app("Safari")
        assert ctrl.commands.get_named("swipe_right").name == "Next desktop"

    def test_switching_back_and_forth_does_not_compound(self):
        commands = parse_commands(cfg([PRESENTATION, EDITOR]))
        ctrl = self._controller(commands)
        for app in ("Keynote", "Code", "Keynote", "Safari", "Keynote"):
            ctrl.set_frontmost_app(app)
        assert ctrl.commands.get_named("swipe_right").name == "Next slide"
        assert ctrl.commands.get_named("swipe_left").name == "Previous slide"

    def test_the_same_app_twice_is_a_no_op(self):
        commands = parse_commands(cfg([PRESENTATION]))
        ctrl = self._controller(commands)
        ctrl.set_frontmost_app("Keynote")
        first = ctrl.commands
        ctrl.set_frontmost_app("Keynote")
        assert ctrl.commands is first

    def test_active_profile_is_reported(self):
        commands = parse_commands(cfg([PRESENTATION]))
        ctrl = self._controller(commands)
        ctrl.set_frontmost_app("Keynote")
        assert ctrl.active_profile == "presentation"
        ctrl.set_frontmost_app("Safari")
        assert ctrl.active_profile is None

    def test_a_config_without_profiles_ignores_app_changes(self):
        commands = parse_commands(cfg())
        ctrl = self._controller(commands)
        ctrl.set_frontmost_app("Keynote")
        assert ctrl.commands is commands

    def test_hot_reload_reapplies_the_current_profile(self):
        """A config edit while Keynote is front must land in Keynote's profile."""
        ctrl = self._controller(parse_commands(cfg([PRESENTATION])))
        ctrl.set_frontmost_app("Keynote")

        updated = dict(PRESENTATION)
        updated["gestures"] = [
            {"gesture": "swipe_right", "name": "Advance",
             "action": {"type": "keypress", "key": "space"}},
        ]
        ctrl.set_commands(parse_commands(cfg([updated])))

        assert ctrl.commands.get_named("swipe_right").name == "Advance"
        assert ctrl.active_profile == "presentation"
