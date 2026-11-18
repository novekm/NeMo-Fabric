# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent package loads with bindings for custom tools and called agents."""

import json
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric import Fabric, FabricConfig, FabricConfigError, load_agent_package

from _utils.agent_packages import (
    LOCK_NAME,
    agent_document,
    declared,
    file_digest,
    read_lock,
    write_json,
    write_package,
)

LEAD = "team-lead"
LEAD_TOOLS = ("route_to_member", "deliver_report")
MEMBERS = {
    "planner": ("plan_steps", "estimate_effort"),
    "researcher": ("search.notes",),
    "writer": ("draft_section", "check_style"),
    "reviewer": ("score_draft",),
}
INSTRUCTIONS = "Route each task to one member. Deliver one report."
WRITER = "$/agents/writer"


def tool(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "input_schema": {"type": "object", "properties": {}},
    }


def binding_key(item_id: str) -> str:
    """The binding key of a line: its pointer with a dot for each slash.

    A tool name may hold dots and no name holds a slash, so a key is read from
    an id and never the other way round.
    """
    return item_id.split(":", 1)[1].lstrip("/").replace("/", ".")


def bound_line(report, key):
    """The line a binding key names."""
    return next(item for item in report.declared if binding_key(item["id"]) == key)


@pytest.fixture(name="team_dir")
def team_dir_fixture(tmp_path: Path) -> Path:
    """A lead with instructions and two custom tools, and the four agents it calls.

    Each called agent has its own instructions and its own custom tools. The
    application that hosts the team supplies the code of every tool and runs
    every member.
    """
    lead = agent_document(
        LEAD,
        description="Routes each task to a member and delivers the result.",
        instructions=INSTRUCTIONS,
        custom_tools={name: tool(f"The lead's {name}.") for name in LEAD_TOOLS},
        agents={name: {"source": f"./agents/{name}"} for name in MEMBERS},
    )
    files = {}
    for name, tools in MEMBERS.items():
        member = agent_document(
            name,
            description=f"The {name} of the team.",
            instructions=f"You are the {name}.",
            custom_tools={
                tool_name: tool(f"The {name}'s {tool_name}.") for tool_name in tools
            },
        )
        files[f"agents/{name}/agent.json"] = json.dumps(member, indent=2) + "\n"
    root = tmp_path / "team"
    write_package(root, lead, files)
    return root


@pytest.fixture(name="team_bindings")
def team_bindings_fixture() -> dict[str, str]:
    """One binding for each custom tool and each called agent of the team."""
    bindings = {
        f"custom_tools.{name}": f"app.tools.lead:{name}" for name in LEAD_TOOLS
    }
    for member, tools in MEMBERS.items():
        bindings[f"agents.{member}"] = f"app.agents:{member}"
        for name in tools:
            key = f"agents.{member}.custom_tools.{name}"
            bindings[key] = f"app.tools.{member}:{name}"
    return bindings


@pytest.fixture(name="team_base_dir")
def team_base_dir_fixture(hermes_shim_agent_dir: Path) -> Path:
    """A base directory whose test adapter also accepts system instructions."""
    descriptor_path = (
        hermes_shim_agent_dir
        / "adapters"
        / "hermes-shim"
        / "hermes-shim.fabric-adapter.json"
    )
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["config"]["accepts"].append("instructions.system")
    write_json(descriptor_path, descriptor)
    return hermes_shim_agent_dir


@pytest.fixture(name="team_config")
def team_config_fixture(shim_config: FabricConfig) -> FabricConfig:
    return FabricConfig.from_mapping(
        {**shim_config.to_mapping(), "metadata": {"name": LEAD}}
    )


@pytest.fixture(name="strict_load")
def strict_load_fixture(team_dir, team_config, team_base_dir):
    """The load an application does: both settings stop."""

    def load(bindings, **options):
        return load_agent_package(
            team_dir,
            team_config,
            base_dir=team_base_dir,
            if_unsupported="stop",
            if_unverified="stop",
            bindings=bindings,
            **options,
        )

    return load


async def test_bound_team_is_accounted_for_and_starts_from_the_report(
    team_dir, team_bindings, strict_load
):
    held_lock_sha256 = load_agent_package(team_dir).lock["sha256"]

    report = strict_load(team_bindings, expected_lock_sha256=held_lock_sha256)

    assert report.outcome == "ready"
    assert report.stopped_by == ()
    assert {check["result"] for check in report.digests} == {"verified"}
    assert len(team_bindings) == 12
    for key, target in team_bindings.items():
        line = bound_line(report, key)
        assert line["status"] == "bound", key
        assert line["bound_to"] == target, key
        assert line["required"] is True, key
        assert line["effect"] == "none", key
        assert line["mapped_to"] == (), key
    # A line below a called agent names that agent's document.
    member_tool = declared(report, "agent:/agents/writer/custom_tools/check_style")
    assert member_tool["stated_by"] == "called_agent"
    assert member_tool["document"] == "writer"
    assert declared(report, "agent:/agents/writer")["address"] == WRITER
    # Every line that could stop a load is accounted for.
    required = {item["status"] for item in report.declared if item["required"]}
    assert required == {"mapped", "bound"}
    # The lines with a target give back the bindings that were passed.
    applied = {
        binding_key(item["id"]): item["bound_to"]
        for item in report.declared
        if item["bound_to"]
    }
    assert applied == team_bindings
    assert declared(report, "agent:/agents/researcher/custom_tools/search.notes")

    # The report's config and base directory are all a start needs.
    config = report.to_config()
    assert config.instructions.system.content == INSTRUCTIONS
    # The bindings are in the report only. The config holds none of them.
    encoded = json.dumps(config.to_mapping())
    assert all(target not in encoded for target in team_bindings.values())
    result = await Fabric().run(config, base_dir=report.base_dir, input="hi")
    assert result.status == "succeeded"


def test_team_with_no_binding_stops_under_default_settings(
    team_dir, team_config, team_base_dir
):
    report = load_agent_package(team_dir, team_config, base_dir=team_base_dir)

    assert report.outcome == "stopped"
    assert set(report.stopped_by) == {
        *(f"agent:/custom_tools/{name}" for name in LEAD_TOOLS),
        *(f"agent:/agents/{name}" for name in MEMBERS),
    }
    for item_id in report.stopped_by:
        line = declared(report, item_id)
        assert line["status"] == "not_mapped"
        assert line["bound_to"] is None
        # A caller that gives no binding reads no word about bindings.
        assert "binding" not in line["detail"]
    # The test adapter declares no key for called agents and takes no tool
    # definitions, so nothing supplies the tools.
    for name in MEMBERS:
        member = declared(report, f"agent:/agents/{name}")
        assert member["gap"] == "adapter_does_not_accept"
    for name in LEAD_TOOLS:
        tool = declared(report, f"agent:/custom_tools/{name}")
        assert tool["gap"] == "not_bound"
    # What a called agent declares is listed only when the caller runs it.
    assert all(item["stated_by"] != "called_agent" for item in report.declared)


@pytest.mark.parametrize(
    "removed",
    [
        "custom_tools.route_to_member",
        "agents.researcher",
        "agents.writer.custom_tools.check_style",
    ],
)
def test_removing_a_required_binding_refuses_the_start(
    team_bindings, strict_load, removed
):
    # A binding below a called agent goes when that agent's binding goes.
    bindings = {
        key: target
        for key, target in team_bindings.items()
        if key != removed and not key.startswith(f"{removed}.")
    }

    report = strict_load(bindings)

    assert report.outcome == "stopped"
    assert [binding_key(item_id) for item_id in report.stopped_by] == [removed]
    assert report.config is None
    line = bound_line(report, removed)
    assert line["status"] == "not_mapped"
    # The line names the key to add.
    assert line["detail"].endswith(f"gives a binding for `{removed}`")
    with pytest.raises(FabricConfigError) as raised:
        report.to_config()
    assert raised.value.code == "package_requirements_unmet"
    assert raised.value.details["stopped_by"] == list(report.stopped_by)


def lock_entry(lock, address):
    return next(entry for entry in lock["entries"] if entry["address"] == address)


def change_member(team_dir):
    path = team_dir / "agents" / "writer" / "agent.json"
    path.write_bytes(path.read_bytes().replace(b"You are the", b"Do not be the"))


def change_member_and_lock(team_dir):
    change_member(team_dir)
    lock = read_lock(team_dir)
    lock_entry(lock, WRITER)["digest"] = file_digest(
        team_dir / "agents" / "writer" / "agent.json"
    )
    write_json(team_dir / LOCK_NAME, lock)


def point_lock_at_another_member(team_dir):
    lock = read_lock(team_dir)
    reviewer = lock_entry(lock, "$/agents/reviewer")
    lock_entry(lock, WRITER).update(
        source=reviewer["source"], digest=reviewer["digest"]
    )
    write_json(team_dir / LOCK_NAME, lock)


@pytest.mark.parametrize(
    "change,anchored,code,subject",
    [
        pytest.param(
            change_member, False, "package_digest_mismatch", WRITER, id="member-changed"
        ),
        pytest.param(
            change_member_and_lock,
            True,
            "package_lock_digest_mismatch",
            LOCK_NAME,
            id="member-and-lock-changed",
        ),
        pytest.param(
            point_lock_at_another_member,
            False,
            "package_lock_entry_mismatch",
            WRITER,
            id="lock-names-another-source",
        ),
    ],
)
def test_bindings_do_not_rescue_a_changed_package(
    team_dir, team_bindings, strict_load, change, anchored, code, subject
):
    # A caller that holds the lock's digest passes it. It catches a lock that
    # was written again to match a changed member.
    held = load_agent_package(team_dir).lock["sha256"] if anchored else None
    change(team_dir)

    with pytest.raises(FabricConfigError) as raised:
        strict_load(team_bindings, expected_lock_sha256=held)

    assert raised.value.code == code
    assert raised.value.details["subject"] == subject


@pytest.mark.parametrize(
    "extra,code,subject",
    [
        pytest.param(
            {"custom_tools.no_such_tool": "app.tools:nothing"},
            "package_binding_undeclared",
            "bindings.custom_tools.no_such_tool",
            id="tool-not-declared",
        ),
        pytest.param(
            {"agents.nobody": "app.agents:nobody"},
            "package_binding_undeclared",
            "bindings.agents.nobody",
            id="agent-not-declared",
        ),
        pytest.param(
            {"agents.writer.custom_tools.score_draft": "app.tools.reviewer:score"},
            "package_binding_undeclared",
            "bindings.agents.writer.custom_tools.score_draft",
            id="tool-of-another-member",
        ),
        pytest.param(
            {"policies.licensed": "app.policies:licensed"},
            "package_options_invalid",
            "bindings.policies.licensed",
            id="not-a-tool-or-an-agent",
        ),
        pytest.param(
            {"custom_tools.deliver_report": ""},
            "package_options_invalid",
            "bindings.custom_tools.deliver_report",
            id="empty-target",
        ),
        pytest.param(
            {"custom_tools.deliver_report": {"ref": "app.tools.lead:deliver_report"}},
            "package_options_invalid",
            "options",
            id="target-not-text",
        ),
        pytest.param(
            {"custom_tools.deliver_report": "app.tools.lead:deliver\u2028import os"},
            "package_options_invalid",
            "bindings.custom_tools.deliver_report",
            id="target-with-a-line-separator",
        ),
        pytest.param(
            {"custom_tools.deliver_report": "\u200b"},
            "package_options_invalid",
            "bindings.custom_tools.deliver_report",
            id="target-that-shows-nothing",
        ),
        pytest.param(
            {"custom_tools.deliver_report": "app\u202esloot:"},
            "package_options_invalid",
            "bindings.custom_tools.deliver_report",
            id="target-that-reverses-its-text",
        ),
        pytest.param(
            {"custom_tools.deliver\nreport": "app.tools.lead:deliver_report"},
            "package_options_invalid",
            "bindings",
            id="key-with-a-line-break",
        ),
        pytest.param(
            {"custom_tools." + "a" * 2000: "app.tools.lead:deliver_report"},
            "package_options_invalid",
            "bindings",
            id="key-too-long",
        ),
    ],
)
def test_wrong_binding_is_refused_by_code_and_subject(
    team_bindings, strict_load, extra, code, subject
):
    with pytest.raises(FabricConfigError) as raised:
        strict_load({**team_bindings, **extra})

    assert raised.value.code == code
    assert raised.value.details["subject"] == subject


def test_binding_below_an_agent_needs_that_agents_binding(team_bindings, strict_load):
    del team_bindings["agents.writer"]

    with pytest.raises(FabricConfigError) as raised:
        strict_load(team_bindings)

    assert raised.value.code == "package_options_invalid"
    subject = raised.value.details["subject"]
    assert subject == "bindings.agents.writer.custom_tools.check_style"


def test_bindings_must_be_a_mapping(strict_load):
    with pytest.raises(FabricConfigError) as raised:
        strict_load(["custom_tools.route_to_member"])

    assert raised.value.code == "package_options_invalid"
    assert raised.value.details["subject"] == "bindings"


def test_refusal_of_a_key_does_not_repeat_the_key(team_bindings, strict_load):
    key = "custom_tools.\x1b[31mdeliver_report"

    with pytest.raises(FabricConfigError) as raised:
        strict_load({**team_bindings, key: "app.tools.lead:deliver_report"})

    assert raised.value.code == "package_options_invalid"
    assert key not in str(raised.value)
    assert "\x1b" not in json.dumps(raised.value.details, ensure_ascii=False)


@pytest.fixture(name="member_with_more")
def member_with_more_fixture(team_dir):
    """The writer also declares a token budget, a tool, a credential and a need."""
    path = team_dir / "agents" / "writer" / "agent.json"
    member = json.loads(path.read_text(encoding="utf-8"))
    member.update(
        limits={"max_total_tokens": 100000},
        tools=[{"builtin": "shell"}],
        credentials={"style-guide": {"description": "Reads the style guide."}},
        runtime={"software": {"python3": {"version": ">= 3.11"}}},
        extensions={"my-app": {"lane": "drafts"}},
    )
    write_json(path, member)
    lock = read_lock(team_dir)
    lock_entry(lock, WRITER)["digest"] = file_digest(path)
    write_json(team_dir / LOCK_NAME, lock)
    return {
        "agent:/agents/writer/limits/max_total_tokens": "not_mapped",
        "agent:/agents/writer/tools/0": "not_mapped",
        "agent:/agents/writer/credentials/style-guide": "could_not_be_checked",
        "agent:/agents/writer/runtime/software/python3": "could_not_be_checked",
    }


def test_bound_member_keeps_a_line_for_each_other_thing_it_declares(
    team_bindings, strict_load, member_with_more
):
    report = strict_load(team_bindings)

    # The writer and its two tools are bound, and that accounts for nothing
    # else the writer declares: each other required thing stops the load.
    assert declared(report, "agent:/agents/writer")["status"] == "bound"
    assert report.outcome == "stopped"
    assert list(report.stopped_by) == list(member_with_more)
    assert report.config is None
    for item_id, status in member_with_more.items():
        line = declared(report, item_id)
        assert line["status"] == status, item_id
        assert line["stated_by"] == "called_agent", item_id
        assert line["document"] == "writer", item_id
        assert line["required"] is True, item_id
        assert line["bound_to"] is None, item_id
    extension = declared(report, "agent:/agents/writer/extensions/my-app")
    assert extension["status"] == "skipped"
    assert extension["required"] is False
    with pytest.raises(FabricConfigError) as raised:
        report.to_config()
    assert raised.value.code == "package_requirements_unmet"


def test_things_of_a_bound_member_follow_the_two_settings(
    team_dir, team_config, team_base_dir, team_bindings, member_with_more
):
    def load(if_unsupported, if_unverified):
        return load_agent_package(
            team_dir,
            team_config,
            base_dir=team_base_dir,
            bindings=team_bindings,
            if_unsupported=if_unsupported,
            if_unverified=if_unverified,
        )

    def warned_about(report):
        return {item["id"] for item in report.declared if item["effect"] == "warned"}

    unchecked = {
        item_id for item_id, status in member_with_more.items() if status != "not_mapped"
    }

    # The default settings stop on what is not mapped and warn about the rest.
    default = load("stop", "warn")
    assert default.outcome == "stopped"
    assert set(default.stopped_by) == set(member_with_more) - unchecked
    assert warned_about(default) == unchecked

    # Told to continue, the load is ready and each of the things is a warning.
    warned = load("warn", "warn")
    assert warned.outcome == "ready"
    assert warned_about(warned) == set(member_with_more)
    # Nothing of the writer reaches the config.
    encoded = json.dumps(warned.to_config().to_mapping())
    assert "style-guide" not in encoded
    assert "max_total_tokens" not in encoded


def test_a_bound_agent_changes_what_the_timeout_line_says(
    team_dir, team_config, team_base_dir, team_bindings
):
    path = team_dir / "agent.json"
    lead = json.loads(path.read_text(encoding="utf-8"))
    lead["limits"] = {"timeout_seconds": 60}
    write_json(path, lead)
    lock = read_lock(team_dir)
    lock_entry(lock, "$")["digest"] = file_digest(path)
    write_json(team_dir / LOCK_NAME, lock)

    def timeout_line(bindings):
        report = load_agent_package(
            team_dir, team_config, base_dir=team_base_dir, bindings=bindings, if_unsupported="warn"
        )
        line = declared(report, "agent:/limits/timeout_seconds")
        assert line["status"] == "mapped"
        assert report.to_config().runtime.timeout_seconds == 60
        return line["detail"]

    assert timeout_line(None).endswith("so one invocation is the whole tree")
    bound = timeout_line(team_bindings)
    assert "whole tree" not in bound
    assert "the caller's to bound" in bound


def test_load_with_bindings_keeps_the_callers_own_config_fields(
    team_bindings, team_config, team_dir, team_base_dir
):
    # An application that wants the mapping in the config puts it there
    # itself. The loader keeps what the package does not set.
    own = {**team_config.to_mapping()}
    own["metadata"] = {**own["metadata"], "my_app_bindings": dict(team_bindings)}

    report = load_agent_package(
        team_dir,
        FabricConfig.from_mapping(own),
        base_dir=team_base_dir,
        if_unsupported="stop",
        if_unverified="stop",
        bindings=team_bindings,
    )

    config = report.to_config()
    assert config.to_mapping()["metadata"]["my_app_bindings"] == team_bindings
    plan = Fabric().plan(config, base_dir=report.base_dir)
    assert plan.agent_name == LEAD
