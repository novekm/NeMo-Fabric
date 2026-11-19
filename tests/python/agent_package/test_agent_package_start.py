# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a runtime from an agent package in one call."""

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import nemo_fabric.client as client_module
import pytest
from nemo_fabric import (
    AgentPackageDeclaredItem,
    AgentPackageDigestCheck,
    AgentPackageReport,
    Fabric,
    FabricConfig,
    FabricConfigError,
    RuntimeStatus,
)

from _utils.agent_packages import definition_document, write_package


@pytest.fixture(name="client")
def client_fixture() -> Fabric:
    return Fabric()


@pytest.fixture(name="mock_start_runtime")
def mock_start_runtime_fixture(client: Fabric) -> AsyncMock:
    """Stands in for the start, so a test can show that nothing was started."""
    client.start_runtime = AsyncMock(spec=client.start_runtime)
    return client.start_runtime


async def test_one_call_starts_a_runtime_from_a_package(
    client, package_dir, shim_config, hermes_shim_agent_dir
):
    skill_dir = (package_dir / "skills" / "release-notes").resolve()

    runtime = await client.start_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        options={"variables": {"timeout": 300}},
    )
    try:
        assert runtime.status is RuntimeStatus.ACTIVE
        result = await runtime.invoke(input="hi")
    finally:
        await runtime.stop()

    assert result.status == "succeeded"
    assert [Path(path) for path in result.output["native_skill_paths"]] == [skill_dir]
    assert runtime.status is RuntimeStatus.STOPPED


async def test_the_started_runtime_holds_the_load_report(
    client, package_dir, shim_config, hermes_shim_agent_dir
):
    skill_dir = (package_dir / "skills" / "release-notes").resolve()

    runtime = await client.start_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        options={"variables": {"timeout": 300}},
    )
    try:
        report = runtime.agent_package_report
    finally:
        await runtime.stop()

    assert isinstance(report, AgentPackageReport)
    assert report.outcome == "ready"
    skill = next(
        line for line in report.declared if line.id == "agent:/skills/release-notes"
    )
    assert isinstance(skill, AgentPackageDeclaredItem)
    assert (skill.status, skill.effect, skill.required) == ("mapped", "none", True)
    assert skill["status"] == "mapped"
    pinned = next(
        check for check in report.digests if check.address == "$/skills/release-notes"
    )
    assert isinstance(pinned, AgentPackageDigestCheck)
    assert (pinned.result, pinned.path) == ("verified", str(skill_dir))


async def test_a_runtime_started_without_a_package_holds_no_load_report(
    client, shim_config, hermes_shim_agent_dir
):
    runtime = await client.start_runtime(shim_config, base_dir=hermes_shim_agent_dir)
    try:
        assert runtime.agent_package_report is None
    finally:
        await runtime.stop()


async def test_start_passes_the_mapped_config_and_the_runtime_arguments(
    client, mock_start_runtime, package_dir, shim_config, hermes_shim_agent_dir
):
    runtime = await client.start_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        options={"variables": {"timeout": 300}},
        overrides={"temperature": 0},
        streaming=True,
    )

    assert runtime is mock_start_runtime.return_value
    mock_start_runtime.assert_awaited_once()
    (config,) = mock_start_runtime.await_args.args
    assert config.metadata.name == "release-notes-agent"
    assert config.runtime.timeout_seconds == 300
    assert mock_start_runtime.await_args.kwargs == {
        "base_dir": hermes_shim_agent_dir,
        "session_ref": None,
        "overrides": {"temperature": 0},
        "streaming": True,
    }


async def test_start_past_a_required_thing_is_logged(
    caplog, client, mock_start_runtime, package_dir, shim_config, hermes_shim_agent_dir
):
    # Nothing is known about the machine, so the need the definition adds could
    # not be checked, and the default setting lets the load continue.
    entry = "variants/yearly/agent-definition.json"
    write_package(
        package_dir,
        definition_document(
            "release-notes-yearly",
            "./../../agent.json",
            description="The release notes agent, for a year of changes.",
            runtime={"compute": {"memory": "16gb"}},
        ),
        entry=entry,
    )
    config = FabricConfig.from_mapping(
        {**shim_config.to_mapping(), "metadata": {"name": "release-notes-yearly"}}
    )

    with caplog.at_level(logging.WARNING, logger=client_module.logger.name):
        await client.start_agent_package(
            package_dir,
            config,
            base_dir=hermes_shim_agent_dir,
            options={"entry": entry},
        )

    mock_start_runtime.assert_awaited_once()
    assert caplog.messages == [
        "Agent package release-notes-yearly starts although the load continued "
        "past required things: definition:/runtime/compute/memory. Each id is a "
        "line of the load report's declared list."
    ]


async def test_start_with_nothing_passed_over_logs_nothing(
    caplog, client, mock_start_runtime, package_dir, shim_config, hermes_shim_agent_dir
):
    with caplog.at_level(logging.WARNING, logger=client_module.logger.name):
        await client.start_agent_package(
            package_dir, shim_config, base_dir=hermes_shim_agent_dir
        )

    mock_start_runtime.assert_awaited_once()
    assert caplog.messages == []


async def test_stopped_load_starts_nothing(
    tmp_path,
    client,
    mock_start_runtime,
    agent,
    package_files,
    shim_config,
    hermes_shim_agent_dir,
):
    # The test adapter does not accept system instructions, so the load stops.
    agent["instructions"] = "Write one short line for each change."
    package = tmp_path / "package"
    write_package(package, agent, package_files)

    with pytest.raises(FabricConfigError) as caught:
        await client.start_agent_package(
            package, shim_config, base_dir=hermes_shim_agent_dir
        )

    assert caught.value.code == "package_requirements_unmet"
    assert caught.value.details == {"stopped_by": ["agent:/instructions"]}
    mock_start_runtime.assert_not_awaited()


async def test_refused_package_starts_nothing(
    client, mock_start_runtime, package_dir, shim_config, hermes_shim_agent_dir
):
    skill = package_dir / "skills" / "release-notes" / "SKILL.md"
    skill.write_text("# Changed after the lock was written\n", encoding="utf-8")

    with pytest.raises(FabricConfigError) as caught:
        await client.start_agent_package(
            package_dir, shim_config, base_dir=hermes_shim_agent_dir
        )

    assert caught.value.stage == "agent_package"
    assert caught.value.code == "package_digest_mismatch"
    mock_start_runtime.assert_not_awaited()


@pytest.mark.parametrize(
    ("options", "subject"),
    [
        ({"variables": {"timeout": 300}, "policy": "strict"}, "policy"),
        (["entry"], "options"),
    ],
)
async def test_unknown_option_is_refused_before_the_load(
    client, mock_start_runtime, shim_config, tmp_path, options, subject
):
    with pytest.raises(FabricConfigError) as caught:
        await client.start_agent_package(
            tmp_path / "no-such-package", shim_config, options=options
        )

    assert caught.value.stage == "agent_package"
    assert caught.value.code == "package_options_invalid"
    assert caught.value.details == {"subject": subject}
    mock_start_runtime.assert_not_awaited()
