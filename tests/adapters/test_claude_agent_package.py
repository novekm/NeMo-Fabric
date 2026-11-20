# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Claude adapter takes what an agent package declares beyond the typed config.

The loader writes declared plugins, hooks and called agents under named root
keys of the config. The Claude descriptor declares the keys, so planning admits
them and the adapter finds them in ``AgentConfig.extensions``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import AgentDefinition
from claude_agent_sdk import ClaudeAgentOptions
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import inspect_adapter
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.claude import adapter
from nemo_fabric_adapters.claude import package_hooks

PLUGINS_KEY = "nemo_fabric.plugins.v1"
HOOKS_KEY = "nemo_fabric.hooks.v1"
AGENTS_KEY = "nemo_fabric.agents.v1"
HOOKS_FORMAT = "https://agent-plugins.org/schemas/1.1.0/hooks.schema.json"
SHELL_DENIAL = "This agent renders through the render-queue connector."

# Handler programs. Each reads the event the adapter writes to standard input.
DENY_SHELL = f"""
import json, sys
event = json.load(sys.stdin)
if (event.get("tool_profile") or {{}}).get("name") == "shell":
    json.dump({{"decision": "deny", "reason": "{SHELL_DENIAL}"}}, sys.stdout)
"""
RECORD_RUN = """
import json, os, sys
seen = {
    "event": json.load(sys.stdin),
    "args": sys.argv[1:],
    "cwd": os.getcwd(),
    "environment": dict(os.environ),
}
with open(os.path.join(os.environ["PLUGIN_DATA"], "seen.json"), "w") as stream:
    json.dump(seen, stream)
"""
LEAVE_A_CHILD = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
with open(os.path.join(os.environ["PLUGIN_DATA"], "child.pid"), "w") as stream:
    stream.write(str(child.pid))
time.sleep(30)
"""

runs_a_handler = pytest.mark.skipif(
    os.name == "nt", reason="uses an executable POSIX hook handler"
)


@pytest.fixture(name="descriptor")
def descriptor_fixture(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "adapters" / "python" / "claude" / "claude.fabric-adapter.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(name="package")
def package_fixture(tmp_path: Path) -> Path:
    """The folder of a video producer package, with its `remotion` plugin."""
    root = tmp_path / "video-producer"
    manifest = root / "plugins" / "remotion" / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"name": "remotion", "version": "1.2.0"}), encoding="utf-8"
    )
    return root


@pytest.fixture(name="write_hook")
def write_hook_fixture(package: Path) -> Callable[..., Path]:
    """Write a hook folder into the package: its `hooks.json` and its handlers."""

    def write(
        name: str, hooks: dict[str, Any], programs: dict[str, str] | None = None
    ) -> Path:
        folder = package / "hooks" / name
        folder.mkdir(parents=True)
        (folder / "hooks.json").write_text(
            json.dumps({"$schema": HOOKS_FORMAT, "hooks": hooks}), encoding="utf-8"
        )
        for file_name, program in (programs or {}).items():
            handler = folder / file_name
            handler.write_text(f"#!{sys.executable}{program}", encoding="utf-8")
            handler.chmod(0o755)
        return folder

    return write


@pytest.fixture(name="tool_guard")
def tool_guard_fixture(write_hook: Callable[..., Path]) -> Path:
    """The package's hook: it denies a shell command before it runs."""
    return write_hook(
        "tool-guard",
        {"PreToolUse": [{"command": "./check-tool-call"}]},
        {"check-tool-call": DENY_SHELL},
    )


@pytest.fixture(name="extensions")
def extensions_fixture(package: Path, tool_guard: Path) -> dict[str, Any]:
    """The root keys the loader writes for the package."""
    return {
        PLUGINS_KEY: {"paths": [str(package / "plugins" / "remotion")]},
        HOOKS_KEY: {"entries": [{"path": str(tool_guard), "format": HOOKS_FORMAT}]},
        AGENTS_KEY: {
            "reviewer": {
                "description": "Reviews a rendered video against the checklist.",
                "instructions": "Check each item of the checklist. Do not fix anything.",
                "max_turns": 60,
            },
            "captioner": {
                "description": "Writes the captions file for a finished video.",
                "instructions": "Write captions in the language you are given.",
            },
        },
    }


@pytest.fixture(name="fabric_config")
def fabric_config_fixture(extensions: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": {"name": "video-producer"},
        "harness": {"adapter_id": "nvidia.fabric.claude"},
        "models": {"default": {"provider": "anthropic", "model": "claude-test-model"}},
        **extensions,
    }


@pytest.fixture(name="agent_config")
def agent_config_fixture(extensions: dict[str, Any]) -> dict[str, Any]:
    """The config the adapter is handed: the root keys arrive under `extensions`."""
    return {
        "models": {"default": {"provider": "anthropic", "model": "claude-test-model"}},
        "extensions": extensions,
    }


@pytest.fixture(name="runtime_context")
def runtime_context_fixture(tmp_path: Path, deadline_millis: int) -> RuntimeContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return RuntimeContext.from_mapping(
        {
            "deadline_millis": deadline_millis,
            "runtime_id": "runtime-video-producer",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": {
                "environment_id": "environment-video-producer",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
                "workspace": str(workspace),
                "env": {},
            },
            "artifacts": {"root": str(tmp_path / "artifacts"), "artifacts": []},
        }
    )


@pytest.fixture(name="build_options")
def build_options_fixture(
    agent_config: dict[str, Any], runtime_context: RuntimeContext, tmp_path: Path
) -> Callable[..., ClaudeAgentOptions]:
    """Build the Claude options from `agent_config` as it is when called."""

    def build(
        *, relay: adapter.ClaudeRelaySettings | None = None
    ) -> ClaudeAgentOptions:
        return adapter.build_options(
            AgentConfig.from_mapping(agent_config),
            runtime_context,
            str(tmp_path),
            relay=relay,
        )

    return build


@pytest.fixture(name="claude_input")
def claude_input_fixture(tmp_path: Path) -> dict[str, Any]:
    """The members Claude Code sends with every hook event."""
    return {
        "session_id": "session-video-producer",
        "transcript_path": str(tmp_path / "transcript.jsonl"),
        "cwd": str(tmp_path / "workspace"),
        "permission_mode": "default",
    }


def test_planning_admits_the_package_keys(
    fabric_config: dict[str, Any], descriptor: dict[str, Any]
):
    inspection = inspect_adapter(FabricConfig.from_mapping(fabric_config), descriptor)

    assert not inspection.issues
    assert inspection.capability_profile.adapter_id == "nvidia.fabric.claude"


def test_plan_hands_the_package_keys_to_the_adapter(
    fabric_config: dict[str, Any],
    extensions: dict[str, Any],
    runtime_context: RuntimeContext,
    tmp_path: Path,
):
    plan = Fabric().plan(FabricConfig.from_mapping(fabric_config), base_dir=tmp_path)

    assert plan["agent_config"]["extensions"] == extensions
    options = adapter.build_options(
        AgentConfig.from_mapping(plan["agent_config"]), runtime_context, str(tmp_path)
    )
    assert options.plugins == [
        {"type": "local", "path": extensions[PLUGINS_KEY]["paths"][0]}
    ]
    assert list(options.hooks) == ["PreToolUse"]
    assert set(options.agents) == {"reviewer", "captioner"}


@pytest.mark.parametrize(
    ("key", "value", "field"),
    [
        pytest.param(
            PLUGINS_KEY,
            {"paths": ["plugins/remotion"], "marketplace": "media-team"},
            "nemo_fabric.plugins.v1.marketplace",
            id="plugin-member-not-declared",
        ),
        pytest.param(
            HOOKS_KEY,
            {
                "entries": [
                    {
                        "path": "hooks/tool-guard",
                        "format": "https://example.com/schemas/hooks.schema.json",
                    }
                ]
            },
            "nemo_fabric.hooks.v1.entries.0.format",
            id="hook-in-another-format",
        ),
        pytest.param(
            AGENTS_KEY,
            {
                "captioner": {
                    "description": "Writes the captions file for a finished video.",
                    "instructions": "Write captions in the language you are given.",
                    "connectors": {"paths": ["agents/captioner/connectors/queue"]},
                }
            },
            "nemo_fabric.agents.v1.captioner.connectors",
            id="called-agent-member-not-declared",
        ),
        pytest.param(
            AGENTS_KEY,
            {
                "Video Captioner": {
                    "description": "Writes the captions file for a finished video.",
                    "instructions": "Write captions in the language you are given.",
                }
            },
            "nemo_fabric.agents.v1",
            id="called-agent-name-outside-the-format",
        ),
        pytest.param(
            "nemo_fabric.themes.v1",
            {"paths": ["themes/brand"]},
            "nemo_fabric.themes.v1",
            id="key-not-declared",
        ),
    ],
)
def test_planning_refuses_what_the_descriptor_does_not_declare(
    fabric_config: dict[str, Any],
    descriptor: dict[str, Any],
    key: str,
    value: dict[str, Any],
    field: str,
):
    fabric_config[key] = value

    with pytest.raises(FabricConfigError) as caught:
        inspect_adapter(
            FabricConfig.from_mapping(fabric_config), descriptor, diagnostics=False
        )

    assert f"`extensions.{field}`" in str(caught.value)


def test_build_options_passes_package_plugins_after_the_staged_plugins(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    package: Path,
    tmp_path: Path,
):
    skill = tmp_path / "skills" / "captions"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Captions\n", encoding="utf-8")
    agent_config["skills"] = {"paths": [str(skill)]}
    relay = adapter.ClaudeRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=tmp_path / "nemo-relay",
            config_path=tmp_path / "relay-config" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay-config" / "gateway.log",
        ),
        plugin_config={"version": 1, "components": []},
        plugin_path=tmp_path / "relay-plugin",
    )

    options = build_options(relay=relay)

    staged, relay_plugin, declared = options.plugins
    assert (Path(staged["path"]) / "skills" / "captions" / "SKILL.md").is_file()
    assert relay_plugin == {"type": "local", "path": str(relay.plugin_path)}
    assert declared == {"type": "local", "path": str(package / "plugins" / "remotion")}


def test_build_options_enables_skills_for_a_package_plugin(
    build_options: Callable[..., ClaudeAgentOptions],
):
    options = build_options()

    assert len(options.plugins) == 1
    assert options.skills == "all"


@pytest.mark.parametrize(
    "broken",
    ["missing", "file", "no-manifest"],
)
def test_build_options_rejects_a_package_plugin_claude_cannot_read(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    package: Path,
    tmp_path: Path,
    broken: str,
):
    plugin = package / "plugins" / broken
    if broken == "file":
        plugin.write_text("not a folder", encoding="utf-8")
    elif broken != "missing":
        plugin.mkdir()
    agent_config["extensions"][PLUGINS_KEY]["paths"].append(str(plugin))

    with pytest.raises(adapter.AdapterConfigError) as caught:
        build_options()

    assert caught.value.code == "claude_invalid_configuration"
    assert str(plugin) in caught.value.message
    assert not (tmp_path / "artifacts").exists()


def test_build_options_passes_called_agents(
    build_options: Callable[..., ClaudeAgentOptions],
):
    options = build_options()

    assert options.agents == {
        "reviewer": AgentDefinition(
            description="Reviews a rendered video against the checklist.",
            prompt="Check each item of the checklist. Do not fix anything.",
            maxTurns=60,
        ),
        "captioner": AgentDefinition(
            description="Writes the captions file for a finished video.",
            prompt="Write captions in the language you are given.",
        ),
    }


def test_build_options_loads_a_package_plugin_that_keeps_its_manifest_at_the_root(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    package: Path,
):
    # The layout an agent package gives a plugin: plugin.json, no .claude-plugin.
    plugin = package / "plugins" / "review-kit"
    (plugin / "skills" / "checklist").mkdir(parents=True)
    (plugin / "plugin.json").write_text('{"name": "review-kit"}', encoding="utf-8")
    (plugin / "skills" / "checklist" / "SKILL.md").write_text(
        "---\nname: checklist\ndescription: Checks.\n---\nCheck.\n", encoding="utf-8"
    )
    agent_config["extensions"][PLUGINS_KEY]["paths"] = [str(plugin)]
    agent_config["extensions"][AGENTS_KEY]["reviewer"]["plugins"] = {
        "paths": [str(plugin)]
    }

    options = build_options()

    staged = [Path(entry["path"]) for entry in options.plugins]
    assert len(staged) == 2
    for folder in staged:
        assert folder != plugin
        assert (folder / ".claude-plugin" / "plugin.json").read_text(
            encoding="utf-8"
        ) == '{"name": "review-kit"}'
        assert (folder / "skills" / "checklist" / "SKILL.md").is_file()
    assert options.agents["reviewer"].skills == ["review-kit:checklist"]
    # The package's own folder is not written to.
    assert not (plugin / ".claude-plugin").exists()


def test_build_options_gives_a_called_agent_its_tools_skills_and_plugins(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    package: Path,
    tmp_path: Path,
):
    skill = package / "agents" / "reviewer" / "skills" / "release-checklist"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: release-checklist\ndescription: Checks a release.\n---\nCheck it.\n",
        encoding="utf-8",
    )
    plugin = package / "plugins" / "remotion"
    (plugin / "skills" / "frames").mkdir(parents=True)
    (plugin / "skills" / "frames" / "SKILL.md").write_text(
        "---\nname: frames\ndescription: Counts frames.\n---\nCount.\n",
        encoding="utf-8",
    )
    entry = agent_config["extensions"][AGENTS_KEY]["reviewer"]
    entry["tools"] = ["read_file", "grep"]
    entry["skills"] = {"paths": [str(skill)]}
    entry["plugins"] = {"paths": [str(plugin)]}

    options = build_options()

    reviewer = options.agents["reviewer"]
    assert reviewer.tools == ["Read", "Grep"]
    assert reviewer.skills == [
        "nemo-fabric-agent-reviewer:release-checklist",
        "remotion:frames",
    ]
    # The other called agent gets none of them.
    assert options.agents["captioner"].tools is None
    assert options.agents["captioner"].skills is None
    # The session loads the staged skills of the agent and the agent's plugin.
    paths = [plugin_entry["path"] for plugin_entry in options.plugins]
    staged = [path for path in paths if "agent-skills" in path]
    assert len(staged) == 1
    assert (Path(staged[0]) / "skills" / "release-checklist" / "SKILL.md").is_file()
    assert str(plugin) in paths
    assert options.skills == "all"


def test_build_options_rejects_a_called_agents_tool_claude_has_no_name_for(
    agent_config: dict[str, Any], build_options: Callable[..., ClaudeAgentOptions]
):
    agent_config["extensions"][AGENTS_KEY]["reviewer"]["tools"] = ["teleport"]

    with pytest.raises(adapter.AdapterConfigError) as caught:
        build_options()

    assert caught.value.code == "claude_invalid_configuration"
    assert "teleport" in caught.value.message


def test_build_options_rejects_a_called_agents_skill_folder_without_a_skill(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    package: Path,
):
    empty = package / "agents" / "reviewer" / "skills" / "empty"
    empty.mkdir(parents=True)
    agent_config["extensions"][AGENTS_KEY]["reviewer"]["skills"] = {
        "paths": [str(empty)]
    }

    with pytest.raises(adapter.AdapterConfigError) as caught:
        build_options()

    assert caught.value.code == "claude_invalid_configuration"
    assert str(empty) in caught.value.message


def test_build_options_rejects_a_called_agents_plugin_without_a_manifest(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    package: Path,
):
    plugin = package / "agents" / "reviewer" / "plugins" / "bare"
    plugin.mkdir(parents=True)
    agent_config["extensions"][AGENTS_KEY]["reviewer"]["plugins"] = {
        "paths": [str(plugin)]
    }

    with pytest.raises(adapter.AdapterConfigError) as caught:
        build_options()

    assert caught.value.code == "claude_invalid_configuration"
    assert str(plugin) in caught.value.message


def test_build_options_passes_no_agents_without_called_agents(
    agent_config: dict[str, Any], build_options: Callable[..., ClaudeAgentOptions]
):
    del agent_config["extensions"][AGENTS_KEY]

    assert build_options().agents is None


def test_build_options_rejects_a_called_agent_without_instructions(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    tmp_path: Path,
):
    del agent_config["extensions"][AGENTS_KEY]["reviewer"]["instructions"]

    with pytest.raises(adapter.AdapterConfigError) as caught:
        build_options()

    assert caught.value.code == "claude_invalid_configuration"
    assert "nemo_fabric.agents.v1.reviewer" in caught.value.message
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("name", "max_turns", "message"),
    [
        pytest.param(
            "reviewer",
            0,
            "nemo_fabric.agents.v1.reviewer.max_turns must be a whole number of at least 1",
            id="turn-limit-zero",
        ),
        pytest.param(
            "reviewer",
            "many",
            "nemo_fabric.agents.v1.reviewer.max_turns must be a whole number of at least 1",
            id="turn-limit-text",
        ),
        pytest.param(
            "reviewer",
            True,
            "nemo_fabric.agents.v1.reviewer.max_turns must be a whole number of at least 1",
            id="turn-limit-true",
        ),
        pytest.param(
            "reviewer",
            2.5,
            "nemo_fabric.agents.v1.reviewer.max_turns must be a whole number of at least 1",
            id="turn-limit-not-whole",
        ),
        pytest.param(
            "",
            60,
            "nemo_fabric.agents.v1 has a called agent without a name",
            id="empty-name",
        ),
    ],
)
def test_build_options_rejects_a_called_agent_claude_cannot_take(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    tmp_path: Path,
    name: str,
    max_turns: Any,
    message: str,
):
    agent_config["extensions"][AGENTS_KEY][name] = {
        "description": "Reviews a rendered video against the checklist.",
        "instructions": "Check each item of the checklist. Do not fix anything.",
        "max_turns": max_turns,
    }

    with pytest.raises(adapter.AdapterConfigError) as caught:
        build_options()

    assert caught.value.code == "claude_invalid_configuration"
    assert message in caught.value.message
    assert not (tmp_path / "artifacts").exists()


def test_build_options_passes_a_turn_limit_that_planning_admits_as_whole(
    agent_config: dict[str, Any], build_options: Callable[..., ClaudeAgentOptions]
):
    # Planning admits 60.0 for an integer, so the adapter does too.
    agent_config["extensions"][AGENTS_KEY]["reviewer"]["max_turns"] = 60.0

    turn_limit = build_options().agents["reviewer"].maxTurns

    assert (turn_limit, type(turn_limit)) == (60, int)


@runs_a_handler
@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected"),
    [
        pytest.param(
            "Bash",
            {"command": "ffmpeg -i narration.wav out.mp4"},
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": SHELL_DENIAL,
                }
            },
            id="shell-call-denied",
        ),
        pytest.param(
            "Read", {"file_path": "/work/beat-sheet.md"}, {}, id="file-read-no-opinion"
        ),
    ],
)
async def test_package_hook_answers_a_claude_tool_call(
    build_options: Callable[..., ClaudeAgentOptions],
    claude_input: dict[str, Any],
    tool_name: str,
    tool_input: dict[str, Any],
    expected: dict[str, Any],
):
    (matcher,) = build_options().hooks["PreToolUse"]
    (callback,) = matcher.hooks

    output = await callback(
        {
            **claude_input,
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": "toolu_01",
        },
        "toolu_01",
        {"signal": None},
    )

    assert output == expected


def test_package_hook_joins_the_tools_policy_hook(
    agent_config: dict[str, Any], build_options: Callable[..., ClaudeAgentOptions]
):
    agent_config["tools"] = {"blocked": ["WebFetch"]}

    options = build_options()

    assert [len(matcher.hooks) for matcher in options.hooks["PreToolUse"]] == [1, 1]


@runs_a_handler
async def test_handler_runs_the_way_the_hooks_format_defines(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    claude_input: dict[str, Any],
    write_hook: Callable[..., Path],
    tmp_path: Path,
):
    hook = write_hook(
        "audit-log",
        {
            "PreToolUse": [
                {
                    "command": "./record",
                    "args": [
                        "--policy",
                        "${PLUGIN_ROOT}/policy.json",
                        "--state",
                        "${PLUGIN_DATA}",
                        "${HOME}",
                    ],
                }
            ]
        },
        {"record": RECORD_RUN},
    )
    agent_config["extensions"][HOOKS_KEY]["entries"] = [
        {"path": str(hook), "format": HOOKS_FORMAT}
    ]
    os.environ["ANTHROPIC_API_KEY"] = "model-credential"
    (callback,) = build_options().hooks["PreToolUse"][0].hooks

    output = await callback(
        {
            **claude_input,
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "npx remotion render", "timeout": 600000},
            "tool_use_id": "toolu_02",
            "agent_id": "reviewer-1",
        },
        "toolu_02",
        {"signal": None},
    )

    assert output == {}
    (seen_path,) = (tmp_path / "artifacts" / ".fabric" / "claude" / "hooks").glob(
        "*/seen.json"
    )
    seen = json.loads(seen_path.read_text(encoding="utf-8"))
    root, data = str(hook.resolve()), str(seen_path.parent.resolve())
    # Claude's own members, such as transcript_path, are not part of the event.
    assert seen["event"] == {
        "hook_event_name": "PreToolUse",
        "session_id": "session-video-producer",
        "cwd": claude_input["cwd"],
        "tool_use_id": "toolu_02",
        "tool_name": "Bash",
        "tool_input": {"command": "npx remotion render", "timeout": 600000},
        "tool_profile": {"name": "shell", "input": {"command": "npx remotion render"}},
    }
    assert seen["args"] == [
        "--policy",
        f"{root}/policy.json",
        "--state",
        data,
        "${HOME}",
    ]
    assert Path(seen["cwd"]) == hook.resolve()
    assert seen["environment"]["PLUGIN_ROOT"] == root
    assert seen["environment"]["PLUGIN_DATA"] == data
    assert "ANTHROPIC_API_KEY" not in seen["environment"]


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "profile"),
    [
        pytest.param(
            "Bash",
            {"command": "npm test", "timeout": 30000},
            {"name": "shell", "input": {"command": "npm test"}},
            id="shell",
        ),
        pytest.param(
            "Read",
            {"file_path": "/work/beat-sheet.md", "limit": 40},
            {"name": "file.read", "input": {"path": "/work/beat-sheet.md"}},
            id="file-read",
        ),
        pytest.param(
            "Write",
            {"file_path": "/work/captions.srt", "content": "1\n"},
            {
                "name": "file.modify",
                "input": {
                    "operations": [{"kind": "write", "path": "/work/captions.srt"}]
                },
            },
            id="file-write",
        ),
        pytest.param(
            "Edit",
            {"file_path": "/work/beat-sheet.md", "old_string": "a", "new_string": "b"},
            {
                "name": "file.modify",
                "input": {
                    "operations": [{"kind": "edit", "path": "/work/beat-sheet.md"}]
                },
            },
            id="file-edit",
        ),
        pytest.param(
            "NotebookEdit",
            {"notebook_path": "/work/review.ipynb", "new_source": "x = 1"},
            {
                "name": "file.modify",
                "input": {
                    "operations": [{"kind": "edit", "path": "/work/review.ipynb"}]
                },
            },
            id="notebook-edit",
        ),
        pytest.param(
            "WebFetch",
            {"url": "https://example.com/docs", "prompt": "Summarize the page."},
            {"name": "web.fetch", "input": {"url": "https://example.com/docs"}},
            id="web-fetch",
        ),
        pytest.param(
            "WebSearch",
            {"query": "remotion render flags"},
            {"name": "web.search", "input": {"queries": ["remotion render flags"]}},
            id="web-search",
        ),
        pytest.param(
            "mcp__render-queue__render_video",
            {"beats": 12},
            {
                "name": "mcp",
                "input": {"name": "render_video", "arguments": {"beats": 12}},
            },
            id="mcp",
        ),
        pytest.param("Read", {"file_path": "beat-sheet.md"}, None, id="relative-path"),
        pytest.param(
            "WebFetch", {"url": "example.com/docs"}, None, id="url-without-scheme"
        ),
        pytest.param("Grep", {"pattern": "TODO"}, None, id="tool-outside-the-six"),
        pytest.param("Bash", "npm test", None, id="input-not-an-object"),
    ],
)
def test_tool_profile_gives_the_portable_view_of_a_claude_tool_call(
    tool_name: str, tool_input: Any, profile: dict[str, Any] | None
):
    assert package_hooks.tool_profile(tool_name, tool_input) == profile


@runs_a_handler
@pytest.mark.parametrize(
    ("program", "failure"),
    [
        pytest.param(
            "\nimport sys\nsys.exit(1)\n", "exited with status 1", id="exit-1"
        ),
        pytest.param(
            "\nprint('{}')\n",
            "wrote an answer PreToolUse does not define",
            id="empty-object",
        ),
        pytest.param(
            "\nprint('checked')\n", "wrote output that is not JSON", id="not-json"
        ),
        pytest.param(
            "\nprint('null')\n", "wrote an answer PreToolUse does not define", id="null"
        ),
        pytest.param(
            "\nprint('[' * 100000)\n",
            "wrote output that is not JSON",
            id="json-nested-too-deep",
        ),
        pytest.param(
            '\nprint(\'{"decision": "allow", "reason": "ok"}\')\n',
            "wrote an answer PreToolUse does not define",
            id="allow-with-a-reason",
        ),
        pytest.param("\nimport time\ntime.sleep(60)\n", "timed out", id="hangs"),
    ],
)
async def test_failed_handler_denies_the_tool_call(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    claude_input: dict[str, Any],
    write_hook: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    program: str,
    failure: str,
):
    hook = write_hook(
        "licensed-assets",
        {"PreToolUse": [{"command": "./check"}]},
        {"check": program},
    )
    agent_config["extensions"][HOOKS_KEY]["entries"] = [
        {"path": str(hook), "format": HOOKS_FORMAT}
    ]
    monkeypatch.setattr(package_hooks, "HANDLER_TIMEOUT_SECONDS", 2.0)
    (callback,) = build_options().hooks["PreToolUse"][0].hooks

    output = await callback(
        {
            **claude_input,
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "/work/logo.png"},
            "tool_use_id": "toolu_03",
        },
        "toolu_03",
        {"signal": None},
    )

    decision = output["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert decision["permissionDecisionReason"] == (
        f"The hook 'licensed-assets' could not check this tool call: the handler {failure}."
    )


@runs_a_handler
async def test_timed_out_handler_takes_its_child_processes_with_it(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    claude_input: dict[str, Any],
    write_hook: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    hook = write_hook(
        "licensed-assets",
        {"PreToolUse": [{"command": "./check"}]},
        {"check": LEAVE_A_CHILD},
    )
    agent_config["extensions"][HOOKS_KEY]["entries"] = [
        {"path": str(hook), "format": HOOKS_FORMAT}
    ]
    monkeypatch.setattr(package_hooks, "HANDLER_TIMEOUT_SECONDS", 2.0)
    (callback,) = build_options().hooks["PreToolUse"][0].hooks

    # The child holds the handler's standard output open for 30 seconds.
    async with asyncio.timeout(10):
        output = await callback(
            {
                **claude_input,
                "hook_event_name": "PreToolUse",
                "tool_name": "Read",
                "tool_input": {"file_path": "/work/logo.png"},
                "tool_use_id": "toolu_07",
            },
            "toolu_07",
            {"signal": None},
        )

    assert "timed out" in output["hookSpecificOutput"]["permissionDecisionReason"]
    (pid_path,) = (tmp_path / "artifacts" / ".fabric" / "claude" / "hooks").glob(
        "*/child.pid"
    )
    child = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        for _ in range(100):
            os.kill(child, 0)
            await asyncio.sleep(0.05)


@runs_a_handler
@pytest.mark.parametrize(
    ("event", "claude_event", "tool_use_id", "answer", "expected"),
    [
        pytest.param(
            "PreToolUse",
            {"tool_name": "Bash", "tool_input": {}, "tool_use_id": "toolu_04"},
            "toolu_04",
            {"decision": "allow"},
            # Claude takes a hook's allow as approval of the call, past its
            # permission mode. A package cannot approve its own tool calls.
            {},
            id="allow-approves-nothing",
        ),
        pytest.param(
            "PostToolUse",
            {
                "tool_name": "Read",
                "tool_input": {},
                "tool_response": "ok",
                "tool_use_id": "toolu_04",
            },
            "toolu_04",
            {"additional_context": "Check the render against the checklist."},
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": "Check the render against the checklist.",
                }
            },
            id="context-after-a-tool-call",
        ),
        pytest.param(
            "UserPromptSubmit",
            {"prompt": "Render the video."},
            None,
            {"decision": "deny", "reason": "A render needs an approved beat sheet."},
            {"decision": "block", "reason": "A render needs an approved beat sheet."},
            id="prompt-denied",
        ),
        pytest.param(
            "Stop",
            {"stop_hook_active": False},
            None,
            {"decision": "continue", "reason": "Write the captions file first."},
            {"decision": "block", "reason": "Write the captions file first."},
            id="continuation-requested",
        ),
        pytest.param(
            "Stop",
            {"stop_hook_active": True},
            None,
            {"decision": "deny", "reason": "Not an answer Stop defines."},
            {},
            id="failure-outside-a-tool-call-has-no-opinion",
        ),
    ],
)
async def test_package_hook_maps_each_event_to_claude(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    claude_input: dict[str, Any],
    write_hook: Callable[..., Path],
    event: str,
    claude_event: dict[str, Any],
    tool_use_id: str | None,
    answer: dict[str, Any],
    expected: dict[str, Any],
):
    hook = write_hook(
        "checklist",
        {event: [{"command": "./answer"}]},
        {"answer": f"\nprint({json.dumps(answer)!r})\n"},
    )
    agent_config["extensions"][HOOKS_KEY]["entries"] = [
        {"path": str(hook), "format": HOOKS_FORMAT}
    ]
    (callback,) = build_options().hooks[event][0].hooks

    output = await callback(
        {**claude_input, "hook_event_name": event, **claude_event},
        tool_use_id,
        {"signal": None},
    )

    assert output == expected


@runs_a_handler
async def test_post_tool_use_handler_receives_a_failed_tool_call(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    claude_input: dict[str, Any],
    write_hook: Callable[..., Path],
    tmp_path: Path,
):
    advice = "Check the render queue before you retry."
    hook = write_hook(
        "render-log",
        {"PostToolUse": [{"command": "./record"}]},
        {
            "record": RECORD_RUN
            + f'json.dump({{"additional_context": "{advice}"}}, sys.stdout)\n'
        },
    )
    agent_config["extensions"][HOOKS_KEY]["entries"] = [
        {"path": str(hook), "format": HOOKS_FORMAT}
    ]
    # Claude reports a failed tool call through an event of its own.
    (callback,) = build_options().hooks["PostToolUseFailure"][0].hooks
    failed_call = {
        **claude_input,
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "mcp__render-queue__render_video",
        "tool_input": {"beats": 12},
        "tool_use_id": "toolu_06",
        "error": "Render failed: no free worker.",
    }

    output = await callback(failed_call, "toolu_06", {"signal": None})

    assert output == {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUseFailure",
            "additionalContext": advice,
        }
    }
    (seen_path,) = (tmp_path / "artifacts" / ".fabric" / "claude" / "hooks").glob(
        "*/seen.json"
    )
    assert json.loads(seen_path.read_text(encoding="utf-8"))["event"] == {
        "hook_event_name": "PostToolUse",
        "session_id": "session-video-producer",
        "cwd": claude_input["cwd"],
        "tool_use_id": "toolu_06",
        "tool_name": "mcp__render-queue__render_video",
        "tool_input": {"beats": 12},
        "tool_result": {"error": "Render failed: no free worker."},
    }
    # An interrupted call has no completed result, so no handler runs for it.
    seen_path.unlink()
    interrupted_call = {**failed_call, "is_interrupt": True}
    assert await callback(interrupted_call, "toolu_06", {"signal": None}) == {}
    assert not seen_path.exists()


@runs_a_handler
@pytest.mark.parametrize("path_entry", ["folder", "working-directory"])
async def test_bare_command_is_found_on_path(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    claude_input: dict[str, Any],
    write_hook: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path_entry: str,
):
    programs = tmp_path / "bin"
    programs.mkdir()
    program = programs / "check-render-queue"
    program.write_text(f"#!{sys.executable}{DENY_SHELL}", encoding="utf-8")
    program.chmod(0o755)
    if path_entry == "folder":
        os.environ["PATH"] = f"{programs}{os.pathsep}{os.environ['PATH']}"
    else:
        # An empty PATH entry is the working directory of the adapter process,
        # not the hook folder the handler later starts in.
        monkeypatch.chdir(programs)
        os.environ["PATH"] = f"{os.pathsep}{os.environ['PATH']}"
    hook = write_hook(
        "queue-guard", {"PreToolUse": [{"command": "check-render-queue"}]}
    )
    agent_config["extensions"][HOOKS_KEY]["entries"] = [
        {"path": str(hook), "format": HOOKS_FORMAT}
    ]
    (callback,) = build_options().hooks["PreToolUse"][0].hooks

    output = await callback(
        {
            **claude_input,
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf renders"},
            "tool_use_id": "toolu_05",
        },
        "toolu_05",
        {"signal": None},
    )

    assert output["hookSpecificOutput"]["permissionDecisionReason"] == SHELL_DENIAL


@runs_a_handler
async def test_hook_behind_a_link_keeps_its_declared_name(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    claude_input: dict[str, Any],
    write_hook: Callable[..., Path],
    package: Path,
):
    stored = write_hook(
        "guards-2026-11",
        {"PreToolUse": [{"command": "./check"}]},
        {"check": "\nimport sys\nsys.exit(1)\n"},
    )
    declared = package / "hooks" / "licensed-assets"
    declared.symlink_to(stored, target_is_directory=True)
    agent_config["extensions"][HOOKS_KEY]["entries"] = [
        {"path": str(declared), "format": HOOKS_FORMAT}
    ]
    (callback,) = build_options().hooks["PreToolUse"][0].hooks

    output = await callback(
        {
            **claude_input,
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "/work/logo.png"},
            "tool_use_id": "toolu_08",
        },
        "toolu_08",
        {"signal": None},
    )

    assert output["hookSpecificOutput"]["permissionDecisionReason"] == (
        "The hook 'licensed-assets' could not check this tool call: "
        "the handler exited with status 1."
    )
    (stored / "check").chmod(0o644)
    with pytest.raises(adapter.AdapterConfigError) as caught:
        build_options()
    assert "hook 'licensed-assets': a PreToolUse handler" in caught.value.message


@pytest.mark.parametrize(
    ("hooks", "message"),
    [
        pytest.param(
            {"PreToolUse": [{"command": "./../check-tool-call"}]},
            "'./../check-tool-call', which is outside the hook",
            id="command-outside-the-hook",
        ),
        pytest.param(
            {"PreToolUse": [{"command": "/usr/bin/env"}]},
            "a command is a bare name or starts with ./",
            id="absolute-command",
        ),
        pytest.param(
            {"PreToolUse": [{"command": "./hooks.json"}]},
            "'./hooks.json', which is not an executable file",
            id="command-not-executable",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="Windows has no executable mark"
            ),
        ),
        pytest.param(
            {"PreToolUse": [{"command": "no-such-render-guard"}]},
            "'no-such-render-guard', which is not on PATH",
            id="bare-command-not-found",
        ),
        pytest.param(
            {"PreToolUse": [{"command": "./check\0-tool-call"}]},
            "has a NUL character in its command or args",
            id="nul-in-the-command",
        ),
        pytest.param(
            {"PreToolUse": [{"command": "render-guard", "args": ["--policy", "a\0b"]}]},
            "has a NUL character in its command or args",
            id="nul-in-an-argument",
        ),
        pytest.param(
            {"SessionStart": [{"command": "./check-tool-call"}]},
            "declares SessionStart, which the Claude adapter cannot run",
            id="event-without-a-callback",
        ),
        pytest.param(
            {"PreToolUse": []},
            "PreToolUse must list at least one handler",
            id="no-handler",
        ),
        pytest.param(
            {"PreToolUse": [{"command": "./check-tool-call", "timeout": 5}]},
            "must hold command, args and extensions only",
            id="handler-member-not-in-the-format",
        ),
    ],
)
def test_build_options_rejects_a_hook_it_cannot_run(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    write_hook: Callable[..., Path],
    package: Path,
    tmp_path: Path,
    hooks: dict[str, Any],
    message: str,
):
    (package / "hooks" / "check-tool-call").write_text("outside", encoding="utf-8")
    hook = write_hook("render-guard", hooks)
    agent_config["extensions"][HOOKS_KEY]["entries"].append(
        {"path": str(hook), "format": HOOKS_FORMAT}
    )

    with pytest.raises(adapter.AdapterConfigError) as caught:
        build_options()

    assert caught.value.code == "claude_invalid_configuration"
    assert message in caught.value.message
    assert "hook 'render-guard'" in caught.value.message
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("document", "message"),
    [
        pytest.param(None, "has no readable hooks.json", id="no-hooks-json"),
        pytest.param(
            "[" * 100000, "has no readable hooks.json", id="json-nested-too-deep"
        ),
        pytest.param(
            json.dumps(
                {
                    "$schema": "https://example.com/schemas/hooks.schema.json",
                    "hooks": {},
                }
            ),
            "is not in the Agent Plugins 1.1.0 hooks format",
            id="another-format",
        ),
        pytest.param(
            json.dumps({"$schema": HOOKS_FORMAT, "hooks": {}, "version": 2}),
            "hooks.json must hold $schema and hooks only",
            id="member-not-in-the-format",
        ),
    ],
)
def test_build_options_rejects_a_hooks_json_it_cannot_read(
    agent_config: dict[str, Any],
    build_options: Callable[..., ClaudeAgentOptions],
    tool_guard: Path,
    document: str | None,
    message: str,
):
    if document is None:
        (tool_guard / "hooks.json").unlink()
    else:
        (tool_guard / "hooks.json").write_text(document, encoding="utf-8")

    with pytest.raises(adapter.AdapterConfigError) as caught:
        build_options()

    assert caught.value.code == "claude_invalid_configuration"
    assert message in caught.value.message
    assert "hook 'tool-guard'" in caught.value.message
    assert str(tool_guard.resolve()) in caught.value.message
