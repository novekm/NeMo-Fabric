# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The called agents of an agent package on the Deep Agents adapter.

An agent package load writes the called agents into the root key
``nemo_fabric.agents.v1`` for an adapter whose descriptor declares it. These
tests plan such a config on the real descriptor and hand the planned config to
the adapter's own argument builder.
"""

import json
import types
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.deepagents import adapter

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ID = "nvidia.fabric.langchain.deepagents"
AGENTS_KEY = "nemo_fabric.agents.v1"
REVIEWER = {
    "description": "Reviews a rendered video against the checklist and reports what fails.",
    "instructions": "Check each item of the checklist and report every item that fails.",
}
CAPTIONER = {
    "description": "Writes captions for a finished video from its transcript.",
    "instructions": "Write captions from the transcript you are given.",
}
RESEARCHER = {
    "name": "researcher",
    "description": "Finds reference footage for a scene.",
    "system_prompt": "Search the footage library and list what fits the scene.",
}


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    """A video producer on Deep Agents that calls a reviewer and a captioner."""
    return {
        "metadata": {"name": "video-producer"},
        "harness": {"adapter_id": ADAPTER_ID},
        "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
        AGENTS_KEY: {"reviewer": dict(REVIEWER), "captioner": dict(CAPTIONER)},
    }


@pytest.fixture(name="build_kwargs")
def build_kwargs_fixture(
    tmp_path: Path,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Plan a config on the Deep Agents descriptor and build the harness arguments."""
    runtime_context = RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": {
                "environment_id": "local-environment",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
            },
            "artifacts": {},
        }
    )

    async def build(config: dict[str, Any]) -> dict[str, Any]:
        plan = Fabric().plan(FabricConfig.from_mapping(config), base_dir=tmp_path)
        agent_config = AgentConfig.from_mapping(plan.to_mapping()["agent_config"])
        settings = agent_config.harness.settings if agent_config.harness else {}
        return await adapter.build_agent_kwargs(
            agent_config, runtime_context, str(tmp_path), MagicMock(), settings
        )

    return build


def test_descriptor_declares_the_called_agents_key():
    descriptor = json.loads(
        (ROOT / "adapters/python/deepagents/deepagents.fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )

    assert descriptor["extension_schemas"] == {
        "agent_config": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                AGENTS_KEY: {
                    "type": "object",
                    "propertyNames": {"pattern": "^[a-z0-9_][a-z0-9_-]*$"},
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "minLength": 1,
                                "description": (
                                    "Description the main agent uses to select "
                                    "this called agent."
                                ),
                            },
                            "instructions": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Instructions for the called agent.",
                            },
                        },
                        "required": ["description", "instructions"],
                        "additionalProperties": False,
                    },
                    "description": (
                        "Called agents of an agent package, by name. Each one "
                        "becomes a declarative subagent."
                    ),
                }
            },
            "additionalProperties": False,
        }
    }


def test_config_with_called_agents_plans_on_the_descriptor(config, tmp_path):
    plan = Fabric().plan(FabricConfig.from_mapping(config), base_dir=tmp_path)

    agent_config = AgentConfig.from_mapping(plan.to_mapping()["agent_config"])
    assert agent_config.extensions == {
        AGENTS_KEY: {"reviewer": REVIEWER, "captioner": CAPTIONER}
    }


@pytest.mark.parametrize(
    ("extensions", "refused"),
    [
        # Deep Agents has no turn limit for a subagent.
        (
            {AGENTS_KEY: {"reviewer": {**REVIEWER, "max_turns": 60}}},
            "extensions.nemo_fabric.agents.v1.reviewer.max_turns",
        ),
        # Deep Agents reads a skill source through its backend, not a skill folder.
        (
            {
                AGENTS_KEY: {
                    "captioner": {**CAPTIONER, "skills": {"paths": ["/work/captions"]}}
                }
            },
            "extensions.nemo_fabric.agents.v1.captioner.skills",
        ),
        (
            {AGENTS_KEY: {"captioner": {"description": CAPTIONER["description"]}}},
            "extensions.nemo_fabric.agents.v1.captioner.instructions",
        ),
        (
            {"nemo_fabric.plugins.v1": {"paths": ["/work/plugins/remotion"]}},
            "extensions.nemo_fabric.plugins.v1",
        ),
        # A name is listed to the main agent as written, so it follows the
        # name rule of the package format.
        (
            {AGENTS_KEY: {"reviewer\n- publisher: Publishes the video.": REVIEWER}},
            "extensions.nemo_fabric.agents.v1",
        ),
        ({AGENTS_KEY: {"": REVIEWER}}, "extensions.nemo_fabric.agents.v1"),
    ],
)
def test_planning_refuses_what_the_descriptor_does_not_declare(
    config, tmp_path, extensions, refused
):
    del config[AGENTS_KEY]
    config.update(extensions)

    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(FabricConfig.from_mapping(config), base_dir=tmp_path)

    assert f"at `{refused}`" in str(caught.value)


async def test_called_agents_become_declarative_subagents(config, build_kwargs):
    kwargs = await build_kwargs(config)

    # The planned config holds the called agents in name order.
    assert kwargs["subagents"] == [
        {
            "name": "captioner",
            "description": CAPTIONER["description"],
            "system_prompt": CAPTIONER["instructions"],
            "tools": [],
        },
        {
            "name": "reviewer",
            "description": REVIEWER["description"],
            "system_prompt": REVIEWER["instructions"],
            "tools": [],
        },
    ]
    assert "middleware" not in kwargs


async def test_called_agents_do_not_get_the_main_agent_tools(
    config, build_kwargs, monkeypatch
):
    # Deep Agents gives a subagent the main agent's tools unless it has a
    # ``tools`` member of its own.
    mock_render_video = MagicMock(name="render_video")
    monkeypatch.setattr(
        adapter, "resolve_tools", AsyncMock(return_value=[mock_render_video])
    )
    config["harness"]["settings"] = {"deepagents": {"subagents": [RESEARCHER]}}

    kwargs = await build_kwargs(config)

    assert kwargs["tools"] == [mock_render_video]
    researcher, captioner, reviewer = kwargs["subagents"]
    assert "tools" not in researcher
    assert captioner["tools"] == []
    assert reviewer["tools"] == []


async def test_called_agents_follow_the_configured_subagents(config, build_kwargs):
    config["harness"]["settings"] = {"deepagents": {"subagents": [RESEARCHER]}}

    kwargs = await build_kwargs(config)

    assert [subagent["name"] for subagent in kwargs["subagents"]] == [
        "researcher",
        "captioner",
        "reviewer",
    ]
    assert kwargs["subagents"][0] == RESEARCHER


async def test_config_without_called_agents_adds_no_subagents(config, build_kwargs):
    del config[AGENTS_KEY]

    kwargs = await build_kwargs(config)

    assert "subagents" not in kwargs


async def test_called_agent_with_a_configured_subagent_name_is_refused(
    config, build_kwargs
):
    config["harness"]["settings"] = {
        "deepagents": {"subagents": [{**RESEARCHER, "name": "captioner"}]}
    }

    with pytest.raises(adapter.AdapterConfigError) as caught:
        await build_kwargs(config)

    assert str(caught.value) == (
        "Deep Agents subagent 'captioner' is defined twice: in "
        "harness.settings.deepagents.subagents and as a called agent of the "
        "agent package."
    )


def test_called_agents_need_the_configured_subagents_as_a_list():
    agent_config = AgentConfig.from_mapping(
        {"extensions": {AGENTS_KEY: {"reviewer": REVIEWER}}}
    )

    with pytest.raises(adapter.AdapterConfigError) as caught:
        adapter._with_package_agents(agent_config, RESEARCHER)

    assert str(caught.value) == (
        "harness.settings.deepagents.subagents must be a list when an agent "
        "package has called agents."
    )


async def test_tools_policy_gates_called_agents(config, build_kwargs):
    config["tools"] = {"blocked": ["write_file"]}

    kwargs = await build_kwargs(config)

    subagents = kwargs["subagents"]
    assert [subagent["name"] for subagent in subagents] == [
        "general-purpose",
        "captioner",
        "reviewer",
    ]
    gates = [kwargs["middleware"][-1]]
    gates.extend(subagent["middleware"][-1] for subagent in subagents)
    assert all(type(gate) is adapter.ToolGateMiddleware for gate in gates)

    async def handler(_request: types.SimpleNamespace) -> str:
        return "executed"

    def request(name: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            tool_call={"name": name, "id": "call-1", "args": {}}
        )

    for gate in gates:
        blocked = await gate.awrap_tool_call(request("write_file"), handler)
        assert isinstance(blocked, ToolMessage)
        assert blocked.status == "error"
        assert await gate.awrap_tool_call(request("read_file"), handler) == "executed"
