# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugins, hooks, called agents and the declared model, through the Python loader."""

import json
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric import Fabric, FabricConfig, load_agent_package

from _utils.agent_packages import (
    LOCK_NAME,
    agent_document,
    declared,
    definition_document,
    read_lock,
    write_json,
    write_package,
)

HOOK_FORMAT = "https://agent-plugins.org/schemas/1.1.0/hooks.schema.json"
PLUGINS_KEY = "nemo_fabric.plugins.v1"
HOOKS_KEY = "nemo_fabric.hooks.v1"
AGENTS_KEY = "nemo_fabric.agents.v1"
REVIEWER = {
    "description": "Reviews a rendered video against the checklist.",
    "instructions": "Check each item of the checklist and report what fails.",
}
SMALL = "ollama::llama3.2:3b"
LARGE = "ollama::mistral-small3.2:24b"


def hooks_json(hook_format: str) -> str:
    handlers = {"PreToolUse": [{"command": "./check-tool-call"}]}
    return json.dumps({"$schema": hook_format, "hooks": handlers})


@pytest.fixture(name="video_producer")
def video_producer_fixture() -> dict[str, Any]:
    """A video producer with one plugin, one hook and one called agent."""
    return agent_document(
        "video-producer",
        description="Makes branded explainer videos timed to recorded narration.",
        limits={"max_turns": 60},
        plugins={"remotion": {"source": "./plugins/remotion"}},
        hooks={"tool-guard": {"source": "./hooks/tool-guard"}},
        agents={"reviewer": {"source": "./agents/reviewer"}},
    )


@pytest.fixture(name="video_producer_files")
def video_producer_files_fixture() -> dict[str, str]:
    """The bytes of the video producer's parts, by path inside the package."""
    reviewer = agent_document("reviewer", limits={"max_turns": 80}, **REVIEWER)
    return {
        "plugins/remotion/plugin.json": json.dumps({"name": "remotion"}),
        "hooks/tool-guard/hooks.json": hooks_json(HOOK_FORMAT),
        "hooks/tool-guard/check-tool-call": "#!/bin/sh\n",
        "agents/reviewer/agent.json": json.dumps(reviewer, indent=2) + "\n",
    }


@pytest.fixture(name="video_producer_dir")
def video_producer_dir_fixture(
    tmp_path: Path, video_producer: dict[str, Any], video_producer_files: dict[str, str]
) -> Path:
    root = tmp_path / "video-producer"
    write_package(root, video_producer, video_producer_files)
    return root


def test_plugin_hook_and_called_agent_reach_an_adapter_that_declares_the_keys(
    video_producer_dir, package_parts_config, package_parts_dir
):
    report = load_agent_package(
        video_producer_dir, package_parts_config, base_dir=package_parts_dir
    )

    assert report.outcome == "ready"
    assert report.adapter["extensions"] == (AGENTS_KEY, HOOKS_KEY, PLUGINS_KEY)
    for item_id, path in [
        ("agent:/plugins/remotion", f"{PLUGINS_KEY}.paths.0"),
        ("agent:/hooks/tool-guard", f"{HOOKS_KEY}.entries.0"),
        ("agent:/agents/reviewer", f"{AGENTS_KEY}.reviewer"),
        ("agent:/agents/reviewer/limits/max_turns", f"{AGENTS_KEY}.reviewer.max_turns"),
    ]:
        item = declared(report, item_id)
        assert item["status"] == "mapped"
        assert item["mapped_to"] == (path,)
    limit = declared(report, "agent:/agents/reviewer/limits/max_turns")
    assert limit["detail"] == "60 applies, lowered from 80: the caller allows 60"
    assert limit["stated_by"] == "called_agent"

    # The adapter gets the three keys as root extension keys of its config.
    plan = Fabric().plan(report.to_config(), base_dir=report.base_dir)
    hook_dir = video_producer_dir / "hooks" / "tool-guard"
    assert plan["agent_config"]["extensions"] == {
        PLUGINS_KEY: {"paths": [str(video_producer_dir / "plugins" / "remotion")]},
        HOOKS_KEY: {"entries": [{"path": str(hook_dir), "format": HOOK_FORMAT}]},
        AGENTS_KEY: {"reviewer": {**REVIEWER, "max_turns": 60}},
    }


def test_the_callers_own_values_under_a_key_stay_first(
    video_producer_dir, package_parts_config, package_parts_dir
):
    translator = {"description": "Translates captions.", "instructions": "Translate."}
    own = {
        PLUGINS_KEY: {"paths": ["/studio/plugins/brand-kit"]},
        AGENTS_KEY: {"translator": translator},
    }
    config = FabricConfig.from_mapping({**package_parts_config.to_mapping(), **own})

    report = load_agent_package(video_producer_dir, config, base_dir=package_parts_dir)

    mapped = report.to_config().to_mapping()
    assert mapped[PLUGINS_KEY]["paths"] == [
        "/studio/plugins/brand-kit",
        str(video_producer_dir / "plugins" / "remotion"),
    ]
    plugin = declared(report, "agent:/plugins/remotion")
    assert plugin["mapped_to"] == (f"{PLUGINS_KEY}.paths.1",)
    assert mapped[AGENTS_KEY]["translator"] == translator
    assert sorted(mapped[AGENTS_KEY]) == ["reviewer", "translator"]


def test_a_hook_format_the_adapter_does_not_list_is_not_written(
    tmp_path,
    video_producer,
    video_producer_files,
    package_parts_config,
    package_parts_dir,
):
    other = "https://hooks.example.com/schemas/2.0/hooks.schema.json"
    video_producer_files["hooks/tool-guard/hooks.json"] = hooks_json(other)
    package = tmp_path / "video-producer"
    write_package(package, video_producer, video_producer_files)
    load = {"config": package_parts_config, "base_dir": package_parts_dir}

    stopped = load_agent_package(package, **load)
    warned = load_agent_package(package, **load, if_unsupported="warn")

    assert stopped.outcome == "stopped"
    assert stopped.stopped_by == ("agent:/hooks/tool-guard",)
    hook = declared(warned, "agent:/hooks/tool-guard")
    assert hook["status"] == "not_mapped"
    assert hook["gap"] == "adapter_does_not_accept"
    assert hook["effect"] == "warned"
    assert warned.outcome == "ready"
    assert HOOKS_KEY not in warned.to_config().to_mapping()


def test_parts_follow_the_settings_on_an_adapter_that_declares_no_key(
    video_producer_dir, shim_config, hermes_shim_agent_dir
):
    config = FabricConfig.from_mapping(
        {**shim_config.to_mapping(), "metadata": {"name": "video-producer"}}
    )

    load = {"config": config, "base_dir": hermes_shim_agent_dir}

    stopped = load_agent_package(video_producer_dir, **load)
    warned = load_agent_package(video_producer_dir, **load, if_unsupported="warn")

    assert stopped.outcome == "stopped"
    assert stopped.stopped_by == (
        "agent:/limits/max_turns",
        "agent:/plugins/remotion",
        "agent:/hooks/tool-guard",
        "agent:/agents/reviewer",
    )
    for item_id in stopped.stopped_by[1:]:
        item = declared(warned, item_id)
        assert item["status"] == "not_mapped"
        assert item["gap"] == "adapter_does_not_accept"
    assert warned.outcome == "ready"
    written = set(warned.to_config().to_mapping())
    assert not {PLUGINS_KEY, HOOKS_KEY, AGENTS_KEY} & written


def test_a_bound_called_agent_is_not_handed_to_the_adapter(
    video_producer_dir, package_parts_config, package_parts_dir
):
    report = load_agent_package(
        video_producer_dir,
        package_parts_config,
        base_dir=package_parts_dir,
        if_unsupported="warn",
        bindings={"agents.reviewer": "studio.agents:reviewer"},
    )

    reviewer = declared(report, "agent:/agents/reviewer")
    assert reviewer["status"] == "bound"
    assert reviewer["bound_to"] == "studio.agents:reviewer"
    assert AGENTS_KEY not in report.to_config().to_mapping()


CHECKLIST = "agent:/agents/reviewer/skills/checklist"
MAX_TURNS = "agent:/agents/reviewer/limits/max_turns"


@pytest.fixture(name="load_with_reviewer")
def load_with_reviewer_fixture(tmp_path, package_parts_config, package_parts_dir):
    """Load a video producer whose called agent, the reviewer, has these members."""

    def load(reviewer, *, entry=None, limits=None, files=None, config=None, **options):
        producer = agent_document(
            "video-producer",
            description="Makes branded explainer videos timed to recorded narration.",
            agents={"reviewer": {"source": "./agents/reviewer", **(entry or {})}},
            **({"limits": limits} if limits else {}),
        )
        document = agent_document("reviewer", **REVIEWER, **reviewer)
        package = tmp_path / "video-producer"
        write_package(
            package,
            producer,
            {
                **(files or {}),
                "agents/reviewer/agent.json": json.dumps(document, indent=2) + "\n",
            },
        )
        return load_agent_package(
            package,
            config or package_parts_config,
            base_dir=package_parts_dir,
            **options,
        )

    return load


def test_values_on_a_called_agents_skill_stop_a_load(load_with_reviewer):
    skill = {"source": "./skills/checklist", "variables": {"strictness": "high"}}

    report = load_with_reviewer(
        {"skills": {"checklist": skill}},
        files={"agents/reviewer/skills/checklist/SKILL.md": "# Checklist\n"},
    )

    # The folder reaches the adapter. Nothing hands the values to the model.
    assert declared(report, CHECKLIST)["status"] == "mapped"
    values = declared(report, f"{CHECKLIST}/variables")
    assert values["kind"] == "part_values"
    assert values["status"] == "not_mapped"
    assert values["gap"] == "no_fabric_field"
    assert report.outcome == "stopped"
    assert report.stopped_by == (f"{CHECKLIST}/variables",)


@pytest.mark.parametrize(
    "bindings", [{}, {"agents.reviewer": "studio.agents:reviewer"}]
)
def test_nothing_below_an_optional_called_agent_stops_a_load(
    load_with_reviewer, bindings
):
    declares = {
        "tools": [{"builtin": "read_file"}],
        "limits": {"max_total_tokens": 100000},
    }

    required = load_with_reviewer(declares, bindings=bindings)
    optional = load_with_reviewer(
        declares, entry={"optional": True}, bindings=bindings
    )

    below = (
        "agent:/agents/reviewer/limits/max_total_tokens",
        "agent:/agents/reviewer/tools/0",
    )
    assert required.outcome == "stopped"
    assert required.stopped_by == below
    assert optional.outcome == "ready"
    for item_id in below:
        item = declared(optional, item_id)
        assert item["status"] == "not_mapped"
        assert item["required"] is False
        assert item["effect"] == "none"


@pytest.mark.parametrize(
    "reviewer_limits, runtime, turns, detail",
    [
        (
            {"max_turns": 80},
            {"max_turns": 30},
            30,
            "30 applies, lowered from 80: the config's runtime.max_turns allows 30",
        ),
        (
            None,
            {},
            60,
            "60 applies: the agent states no turn limit, and the caller allows 60",
        ),
    ],
    ids=["the_config_runs_fewer_turns", "no_limit_of_its_own"],
)
def test_a_called_agent_gets_the_turn_limit_its_caller_runs_with(
    load_with_reviewer, package_parts_config, reviewer_limits, runtime, turns, detail
):
    config = FabricConfig.from_mapping(
        {**package_parts_config.to_mapping(), "runtime": runtime}
    )

    report = load_with_reviewer(
        {"limits": reviewer_limits} if reviewer_limits else {},
        limits={"max_turns": 60},
        config=config,
    )

    assert report.outcome == "ready"
    mapped = report.to_config().to_mapping()
    assert mapped[AGENTS_KEY]["reviewer"]["max_turns"] == turns
    limit = declared(report, MAX_TURNS)
    assert limit["status"] == "mapped"
    assert limit["detail"] == detail


def test_two_called_agents_share_one_skill_folder(
    tmp_path, package_parts_config, package_parts_dir
):
    house_style = {"house-style": {"source": "./../../skills/house-style"}}
    names = ("captioner", "reviewer")
    producer = agent_document(
        "video-producer",
        description="Makes branded explainer videos timed to recorded narration.",
        agents={name: {"source": f"./agents/{name}"} for name in names},
    )
    files = {"skills/house-style/SKILL.md": "# House style\n"}
    for name in names:
        member = agent_document(name, skills=house_style, **REVIEWER)
        files[f"agents/{name}/agent.json"] = json.dumps(member, indent=2) + "\n"
    package = tmp_path / "video-producer"
    write_package(package, producer, files)
    # One folder that two documents reach has one lock entry, at the first path.
    shared = "$/agents/captioner/skills/house-style"
    lock = read_lock(package)
    entries = {entry["address"]: entry for entry in lock["entries"]}
    del entries["$/agents/reviewer/skills/house-style"]
    entries["$/agents/reviewer"]["dependencies"] = [shared]
    lock["entries"] = list(entries.values())
    write_json(package / LOCK_NAME, lock)

    report = load_agent_package(
        package, package_parts_config, base_dir=package_parts_dir
    )

    assert report.outcome == "ready"
    folder = {"paths": [str(package / "skills" / "house-style")]}
    mapped = report.to_config().to_mapping()
    for name in names:
        skill = declared(report, f"agent:/agents/{name}/skills/house-style")
        assert skill["status"] == "mapped"
        assert skill["address"] == shared
        assert mapped[AGENTS_KEY][name]["skills"] == folder


@pytest.mark.parametrize("held", [["studio-translator"], "studio-translator"])
def test_a_callers_value_under_the_agents_key_in_another_shape_is_kept(
    tmp_path, video_producer_dir, package_parts_config, package_parts_dir, held
):
    # An adapter that names the key and takes any value under it.
    descriptor_dir = package_parts_dir / "adapters" / "package-parts"
    descriptor = json.loads(
        (descriptor_dir / "package-parts.fabric-adapter.json").read_text("utf-8")
    )
    descriptor["extension_schemas"]["agent_config"]["properties"][AGENTS_KEY] = {}
    write_json(tmp_path / "adapters" / "open.fabric-adapter.json", descriptor)
    config = FabricConfig.from_mapping(
        {**package_parts_config.to_mapping(), AGENTS_KEY: held}
    )

    report = load_agent_package(
        video_producer_dir, config, base_dir=tmp_path, if_unsupported="warn"
    )

    reviewer = declared(report, "agent:/agents/reviewer")
    assert reviewer["status"] == "not_mapped"
    assert reviewer["gap"] == "unsupported_shape"
    assert report.to_config().to_mapping()[AGENTS_KEY] == held


def test_a_schema_refusal_keeps_the_callers_values_out_of_the_report(
    tmp_path, video_producer_dir, package_parts_config, package_parts_dir
):
    # An adapter that takes one called agent, and a caller that brings its own.
    descriptor_dir = package_parts_dir / "adapters" / "package-parts"
    descriptor = json.loads(
        (descriptor_dir / "package-parts.fabric-adapter.json").read_text("utf-8")
    )
    keys = descriptor["extension_schemas"]["agent_config"]["properties"]
    keys[AGENTS_KEY]["maxProperties"] = 1
    write_json(tmp_path / "adapters" / "one.fabric-adapter.json", descriptor)
    translator = {
        "description": "Translates captions.",
        "instructions": "Use the key STUDIO-KEY-12345 for the glossary service.",
    }
    config = FabricConfig.from_mapping(
        {**package_parts_config.to_mapping(), AGENTS_KEY: {"translator": translator}}
    )

    report = load_agent_package(
        video_producer_dir, config, base_dir=tmp_path, if_unsupported="warn"
    )

    reviewer = declared(report, "agent:/agents/reviewer")
    assert reviewer["status"] == "not_mapped"
    assert reviewer["gap"] == "adapter_does_not_accept"
    assert f"`/{AGENTS_KEY}`" in reviewer["detail"]
    assert "STUDIO-KEY-12345" not in json.dumps(dict(reviewer))


def test_a_package_keeps_its_status_when_no_config_is_given(tmp_path):
    producer = agent_document(
        "video-producer",
        description="Makes branded explainer videos timed to recorded narration.",
        packages={"whisper": {"source": "pypi::faster-whisper", "version": "1.2.1"}},
    )
    package = tmp_path / "video-producer"
    write_package(package, producer)

    listed = load_agent_package(package)
    missing = load_agent_package(package, facts={"packages": {"pypi": {}}})
    installed = load_agent_package(
        package, facts={"packages": {"pypi": {"faster-whisper": "1.2.1"}}}
    )

    whisper = "agent:/packages/whisper"
    assert declared(listed, whisper)["status"] == "not_evaluated"
    assert declared(missing, whisper)["status"] == "not_met"
    assert declared(installed, whisper)["status"] == "met_here"
    assert installed.outcome == "verified_only"


@pytest.fixture(name="weather_dir")
def weather_dir_fixture(tmp_path: Path) -> Path:
    """A weather agent whose model is a variable, with one definition for each model."""
    root = tmp_path / "weather"
    weather = agent_document(
        "weather",
        description="Answers questions about the weather for a place and a day.",
        model={"source": {"var": "model"}, "parameters": {"temperature": 0.2}},
        variables={
            "model": {"type": "string", "enum": [SMALL, LARGE], "default": SMALL}
        },
    )
    write_package(root, weather)
    for name, model in [("weather-small", SMALL), ("weather-large", LARGE)]:
        write_package(
            root,
            definition_document(name, "./../../agent.json", variables={"model": model}),
            entry=f"variants/{name}/agent-definition.json",
        )
    return root


def served(model: str) -> dict[str, str]:
    return {"provider": "test", "model": model.split("::")[1]}


@pytest.mark.parametrize(
    "name, selected, kept",
    [("weather-small", SMALL, LARGE), ("weather-large", LARGE, SMALL)],
)
def test_a_model_the_config_offers_under_the_declared_name_becomes_the_default(
    weather_dir, shim_config, hermes_shim_agent_dir, name, selected, kept
):
    config = FabricConfig.from_mapping(
        {
            **shim_config.to_mapping(),
            "metadata": {"name": name},
            "models": {SMALL: served(SMALL), LARGE: served(LARGE)},
        }
    )

    report = load_agent_package(
        weather_dir,
        config,
        base_dir=hermes_shim_agent_dir,
        entry=f"variants/{name}/agent-definition.json",
    )

    assert report.outcome == "ready"
    assert report.model == {
        "declared_source": selected,
        "provider": "test",
        "model": served(selected)["model"],
        "origin": "package",
        "parameters_applied": ("temperature",),
        "parameters_not_applied": (),
    }
    source = declared(report, "agent:/model/source")
    assert source["status"] == "mapped"
    assert source["mapped_to"] == ("models.default",)
    # The offered entry moved to the default. The other entry stays where it was.
    models = report.to_config().models
    assert sorted(models) == sorted(["default", kept])
    assert models["default"].model == served(selected)["model"]
    assert models["default"].temperature == 0.2
    assert models[kept].model == served(kept)["model"]
    assert models[kept].temperature is None
    # The config with the selected default plans on the adapter.
    plan = Fabric().plan(report.to_config(), base_dir=report.base_dir)
    planned = plan["agent_config"]["models"]["default"]
    assert planned["model"] == served(selected)["model"]


@pytest.mark.parametrize(
    "models, origin, model, status, gap",
    [
        (
            {"default": served("test::served-model"), SMALL: served(SMALL)},
            "caller",
            "served-model",
            "skipped",
            None,
        ),
        # Planning and the adapters run the one model of a config with no default.
        ({"summary": served(LARGE)}, "caller", "mistral-small3.2:24b", "skipped", None),
        ({}, "harness_default", None, "not_mapped", "model_not_configured"),
    ],
    ids=["caller_default_kept", "callers_only_model", "nothing_configured"],
)
def test_a_declared_model_never_moves_what_the_caller_chose_and_never_stops(
    weather_dir, shim_config, hermes_shim_agent_dir, models, origin, model, status, gap
):
    config = FabricConfig.from_mapping(
        {**shim_config.to_mapping(), "metadata": {"name": "weather"}, "models": models}
    )

    report = load_agent_package(weather_dir, config, base_dir=hermes_shim_agent_dir)

    assert report.outcome == "ready"
    assert report.model["origin"] == origin
    assert report.model["model"] == model
    source = declared(report, "agent:/model/source")
    assert source["status"] == status
    assert source["gap"] == gap
    assert source["effect"] == "none"
    # Every entry is where the caller put it, with the provider and model it named.
    assert {
        name: {"provider": model.provider, "model": model.model}
        for name, model in report.to_config().models.items()
    } == models
