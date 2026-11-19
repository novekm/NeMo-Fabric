# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agent package example under ``examples/agent_package``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.agent_package.__main__ import PACKAGE_DIR, VARIANT_ENTRY, main
from nemo_fabric import Fabric, FabricConfig, load_agent_package

AGENT_ADDRESSES = ["$", "$/connectors/changelog", "$/skills/release-notes"]
VARIANT_ADDRESSES = [
    "$",
    "$/agent",
    "$/agent/connectors/changelog",
    "$/agent/skills/release-notes",
    "$/skills/short-form",
]
ON_WINDOWS = sys.platform == "win32"


@pytest.fixture(name="example_config")
def example_config_fixture() -> FabricConfig:
    """The config a caller of the example writes: the test shim adapter, the example agent's name and a model."""
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "release-notes-agent"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
                "settings": {},
            },
            "discovery": {"local_paths": ["adapters"]},
            "models": {"default": {"provider": "test", "model": "test-model"}},
            "environment": {"provider": "local", "workspace": "./repos/my-service"},
        }
    )


@pytest.mark.parametrize(
    "entry,kind,name,addresses",
    [
        (None, "agent", "release-notes-agent", AGENT_ADDRESSES),
        (VARIANT_ENTRY, "agent-definition", "release-notes-short", VARIANT_ADDRESSES),
    ],
    ids=["agent", "variant"],
)
def test_example_package_is_verified_against_its_lock(entry, kind, name, addresses):
    report = load_agent_package(PACKAGE_DIR, entry=entry)

    assert report.outcome == "verified_only"
    assert report.format_version == "0.0.1"
    assert report.entry["kind"] == kind
    assert report.entry["name"] == name
    assert report.agent["name"] == "release-notes-agent"
    assert report.lock["entries"] == len(addresses)
    assert [check["address"] for check in report.digests] == addresses
    assert {check["result"] for check in report.digests} == {"verified"}
    assert report.config is None


@pytest.mark.parametrize(
    "variables,value,origin",
    [(None, 12, "default"), ({"max_turns": 8}, 8, "start")],
    ids=["default", "given-at-start"],
)
def test_example_variable_takes_the_first_value_given(variables, value, origin):
    report = load_agent_package(PACKAGE_DIR, variables=variables)

    assert [dict(variable) for variable in report.variables] == [
        {"name": "max_turns", "type": "number", "value": value, "origin": origin}
    ]


def test_example_agent_maps_onto_the_example_config(example_config, hermes_shim_agent_dir):
    report = load_agent_package(
        PACKAGE_DIR,
        example_config,
        base_dir=hermes_shim_agent_dir,
        if_unsupported="warn",
    )

    assert report.outcome == "ready"
    assert list(report.stopped_by) == []
    assert {check["result"] for check in report.digests} == {"verified"}
    declared = {item["id"]: item for item in report.declared}
    skill = declared["agent:/skills/release-notes"]
    assert skill["status"] == "mapped"
    assert skill["stated_by"] == "agent"
    # The shim accepts skills and MCP servers. It does not accept system instructions.
    instructions = declared["agent:/instructions"]
    assert instructions["status"] == "not_mapped"
    assert instructions["gap"] == "adapter_does_not_accept"
    assert instructions["effect"] == "warned"
    connector = declared["agent:/connectors/changelog"]
    if ON_WINDOWS:
        assert connector["status"] == "not_mapped"
        assert connector["gap"] == "unsupported_platform"
    else:
        assert connector["status"] == "mapped"
        assert list(connector["mapped_to"]) == ["mcp.servers.changelog"]
    config = report.to_config()
    assert config.metadata.name == "release-notes-agent"
    assert [Path(path) for path in config.skills.paths] == [
        (PACKAGE_DIR / "skills" / "release-notes").resolve()
    ]
    assert config.runtime.timeout_seconds == 600


def test_example_agent_stops_by_default_on_the_shim(example_config, hermes_shim_agent_dir):
    report = load_agent_package(
        PACKAGE_DIR,
        example_config,
        base_dir=hermes_shim_agent_dir,
    )

    assert report.outcome == "stopped"
    assert "agent:/instructions" in report.stopped_by
    assert report.config is None


def test_example_variant_reads_the_definition_with_its_agent(example_config, hermes_shim_agent_dir):
    # The config names the entry document, which is the definition here.
    example_config.metadata.name = "release-notes-short"

    report = load_agent_package(
        PACKAGE_DIR,
        example_config,
        base_dir=hermes_shim_agent_dir,
        entry=VARIANT_ENTRY,
        if_unsupported="warn",
    )

    assert report.outcome == "ready"
    assert [check["address"] for check in report.digests] == VARIANT_ADDRESSES
    assert {check["result"] for check in report.digests} == {"verified"}
    declared = {item["id"]: item for item in report.declared}
    added = declared["definition:/skills/short-form"]
    assert added["status"] == "mapped"
    assert added["stated_by"] == "definition"
    assert declared["agent:/skills/release-notes"]["status"] == "mapped"
    assert declared["definition:/limits/max_turns"]["stated_by"] == "definition"
    config = report.to_config()
    assert [Path(path) for path in config.skills.paths] == [
        (PACKAGE_DIR / "skills" / "release-notes").resolve(),
        (PACKAGE_DIR / "variants" / "short" / "skills" / "short-form").resolve(),
    ]


async def test_example_agent_runs_on_the_shim(example_config, hermes_shim_agent_dir):
    report = load_agent_package(
        PACKAGE_DIR,
        example_config,
        base_dir=hermes_shim_agent_dir,
        if_unsupported="warn",
    )

    result = await Fabric().run(report.to_config(), base_dir=report.base_dir, input="hi")

    assert result.status == "succeeded"
    assert [Path(path) for path in result.output["native_skill_paths"]] == [
        (PACKAGE_DIR / "skills" / "release-notes").resolve()
    ]
    assert result.output["native_mcp_servers"] == ([] if ON_WINDOWS else ["changelog"])


def test_example_command_prints_the_report(capsys):
    assert main([]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["outcome"] == "verified_only"
    assert [check["address"] for check in report["digests"]] == AGENT_ADDRESSES


def test_example_command_names_a_refusal(capsys):
    assert main(["--var", "nope=1"]) == 1

    assert "package_variable_undeclared" in capsys.readouterr().err


def test_example_connector_serves_its_tool():
    connector_dir = PACKAGE_DIR / "connectors" / "changelog"
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_changes", "arguments": {}}},
    ]

    # -B keeps Python from writing a bytecode folder into the pinned connector folder.
    completed = subprocess.run(
        [sys.executable, "-B", "server.py"],
        cwd=connector_dir,
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    replies = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2, 3]
    assert replies[0]["result"]["serverInfo"]["name"] == "changelog"
    assert [tool["name"] for tool in replies[1]["result"]["tools"]] == ["list_changes"]
    changes = json.loads(replies[2]["result"]["content"][0]["text"])
    assert [change["kind"] for change in changes] == ["added", "changed", "fixed", "internal"]
