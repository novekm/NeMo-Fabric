# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load an agent package and run it through a real adapter."""

import sys
from pathlib import Path

from nemo_fabric import Fabric, FabricConfig, load_agent_package

from _utils.agent_packages import declared, definition_document, write_package


async def test_loaded_package_runs_with_its_skill_connector_and_model(
    tmp_path, agent, package_files, shim_config, hermes_shim_agent_dir
):
    agent["connectors"] = {"changelog": {"source": "./connectors/changelog"}}
    agent["tools"] = [{"connector": "changelog", "tool": "list_changes"}]
    agent["runtime"] = {
        "software": {"python3": {"version": ">= 3.11", "reason": "Runs the connector."}}
    }
    package = tmp_path / "package"
    write_package(package, agent, package_files)
    skill_dir = (package / "skills" / "release-notes").resolve()
    connector_dir = (package / "connectors" / "changelog").resolve()
    # The caller's config chooses the model. The document's default is reported.
    served_config = FabricConfig.from_mapping(
        {
            **shim_config.to_mapping(),
            "models": {"default": {"provider": "test", "model": "served-model"}},
        }
    )

    report = load_agent_package(
        package,
        served_config,
        base_dir=hermes_shim_agent_dir,
        variables={"timeout": 300},
        if_unsupported="stop" if sys.platform != "win32" else "warn",
    )

    assert report.outcome == "ready"
    assert report.adapter["adapter_id"] == "test.fabric.hermes_shim"
    assert report.base_dir == hermes_shim_agent_dir
    assert {check["result"] for check in report.digests} == {"verified"}
    assert report.model == {
        "declared_source": "test::test-model",
        "provider": "test",
        "model": "served-model",
        "origin": "caller",
        "parameters_applied": (),
        "parameters_not_applied": (),
    }
    assert declared(report, "agent:/model/source")["status"] == "skipped"
    skill = declared(report, "agent:/skills/release-notes")
    assert skill["status"] == "mapped"
    assert skill["mapped_to"] == ("skills.paths.0",)
    assert declared(report, "agent:/limits/timeout_seconds")["status"] == "mapped"
    # Nothing here can confirm the need, so it is reported and the load continues.
    software = declared(report, "agent:/runtime/software/python3")
    assert software["status"] == "could_not_be_checked"
    assert software["effect"] == "warned"
    assert software["reason"] == "Runs the connector."

    config = report.to_config()
    assert isinstance(config, FabricConfig)
    assert config.metadata.name == "release-notes-agent"
    assert config.metadata.description == "Drafts release notes from a list of changes."
    assert config.runtime.timeout_seconds == 300
    assert config.models["default"].model == "served-model"
    assert [Path(path) for path in config.skills.paths] == [skill_dir]

    connector = declared(report, "agent:/connectors/changelog")
    tool = declared(report, "agent:/tools/0")
    if sys.platform == "win32":
        # A carried connector is started through /bin/sh, which Windows lacks.
        assert connector["status"] == "not_mapped"
        assert connector["gap"] == "unsupported_platform"
        assert tool["status"] == "not_mapped"
        assert tool["gap"] == "unsupported_platform"
        expected_servers = []
    else:
        assert connector["status"] == "mapped"
        assert connector["mapped_to"] == ("mcp.servers.changelog",)
        # The server's tools are known only when it runs.
        assert tool["status"] == "could_not_be_checked"
        assert tool["effect"] == "warned"
        server = config.mcp.servers["changelog"]
        assert server.transport == "stdio"
        assert server.url == "/bin/sh"
        assert server.args[2] == "sh"
        assert Path(server.args[3]) == connector_dir
        assert server.args[4:] == ["python3", "server.py"]
        expected_servers = ["changelog"]

    result = await Fabric().run(config, base_dir=report.base_dir, input="hi")

    assert result.status == "succeeded"
    assert [Path(path) for path in result.output["native_skill_paths"]] == [skill_dir]
    assert result.output["native_mcp_servers"] == expected_servers


async def test_instructions_the_adapter_does_not_accept_stop_or_warn(
    tmp_path, agent, package_files, shim_config, hermes_shim_agent_dir
):
    agent["instructions"] = "Write one short line for each change."
    package = tmp_path / "package"
    write_package(package, agent, package_files)

    stopped = load_agent_package(package, shim_config, base_dir=hermes_shim_agent_dir)
    warned = load_agent_package(
        package, shim_config, base_dir=hermes_shim_agent_dir, if_unsupported="warn"
    )

    assert stopped.outcome == "stopped"
    assert stopped.stopped_by == ("agent:/instructions",)
    instructions = declared(stopped, "agent:/instructions")
    assert instructions["status"] == "not_mapped"
    assert instructions["gap"] == "adapter_does_not_accept"
    assert instructions["effect"] == "stopped"

    assert warned.outcome == "ready"
    assert declared(warned, "agent:/instructions")["effect"] == "warned"
    assert warned.adapter["instruction_mode"] is None
    config = warned.to_config()
    assert config.instructions is None
    result = await Fabric().run(config, base_dir=warned.base_dir, input="hi")
    assert result.status == "succeeded"


async def test_definition_adds_a_skill_and_runs_with_its_agent(
    package_dir, shim_config, hermes_shim_agent_dir
):
    entry = "variants/short/agent-definition.json"
    write_package(
        package_dir,
        definition_document(
            "release-notes-short",
            "./../../agent.json",
            description="The release notes agent, held to a short summary.",
            variables={"timeout": 300},
            skills={"short-form": {"source": "./skills/short-form"}},
            limits={"timeout_seconds": 600},
        ),
        {"variants/short/skills/short-form/SKILL.md": "# Short form\n"},
        entry=entry,
    )
    config = FabricConfig.from_mapping(
        {**shim_config.to_mapping(), "metadata": {"name": "release-notes-short"}}
    )
    agent_skill = (package_dir / "skills" / "release-notes").resolve()
    added_skill = (package_dir / "variants" / "short" / "skills" / "short-form").resolve()

    report = load_agent_package(
        package_dir, config, base_dir=hermes_shim_agent_dir, entry=entry
    )

    assert report.outcome == "ready"
    assert report.entry["kind"] == "agent-definition"
    assert report.entry["name"] == "release-notes-short"
    assert report.agent["kind"] == "agent"
    assert report.agent["name"] == "release-notes-agent"
    assert report.variables == (
        {"name": "timeout", "type": "number", "value": 300, "origin": "definition"},
    )
    added = declared(report, "definition:/skills/short-form")
    assert added["stated_by"] == "definition"
    assert added["document"] == "release-notes-short"
    assert added["status"] == "mapped"
    assert declared(report, "agent:/skills/release-notes")["stated_by"] == "agent"
    # The definition asks for 600 seconds and the agent allows 300: the smaller applies.
    assert any(note["code"] == "limit_lowered" for note in report.notes)
    config = report.to_config()
    assert config.metadata.name == "release-notes-short"
    assert config.runtime.timeout_seconds == 300
    assert [Path(path) for path in config.skills.paths] == [agent_skill, added_skill]

    result = await Fabric().run(config, base_dir=report.base_dir, input="hi")

    assert result.status == "succeeded"
    assert [Path(path) for path in result.output["native_skill_paths"]] == [
        agent_skill,
        added_skill,
    ]
