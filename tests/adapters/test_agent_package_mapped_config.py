# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A config mapped from an agent package is one the first-party adapters take.

The loader reports a thing as mapped when it wrote it into the config. These
tests hand that config to the adapters' own config builders, so a mapping the
adapters refuse at start is caught here.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric import FabricConfig
from nemo_fabric import _native
from nemo_fabric import load_agent_package
from nemo_fabric_adapter_contract.models import AgentConfig

from _utils.agent_packages import agent_document
from _utils.agent_packages import write_package
from _utils.openclaw_skills import runtime_context

AGENT = "release-notes-agent"
SKILL_TEXT = "---\nname: release-notes\ndescription: Draft release notes.\n---\n"


@pytest.fixture(name="package")
def package_fixture(tmp_path: Path) -> Path:
    """One folder skill, one skill that is one file, two folders with one name, one connector."""
    root = tmp_path / "package"
    write_package(
        root,
        agent_document(
            AGENT,
            description="Drafts release notes from a list of changes.",
            instructions="Write one short line for each change.",
            model={"source": "ollama::llama3.2:3b"},
            limits={"max_turns": 8},
            skills={
                "release-notes": {"source": "./skills/release-notes"},
                "one-file": {"source": "./skills/one-file.md"},
                "left": {"source": "./left/review"},
                "right": {"source": "./right/review"},
            },
            connectors={"changelog": {"source": "./connectors/changelog"}},
        ),
        {
            "skills/release-notes/SKILL.md": SKILL_TEXT,
            "skills/one-file.md": "# One file\n",
            "left/review/SKILL.md": "# Left\n",
            "right/review/SKILL.md": "# Right\n",
            "connectors/changelog/mcp.json": json.dumps(
                {
                    "mcpServers": {
                        "main": {
                            "type": "stdio",
                            "command": "python3",
                            "args": ["server.py"],
                        }
                    }
                }
            ),
            "connectors/changelog/server.py": "print('ready')\n",
        },
    )
    return root


@pytest.fixture(name="caller_skill")
def caller_skill_fixture(tmp_path: Path) -> Path:
    """One skill folder of the caller's own, outside the package."""
    folder = tmp_path / "caller" / "team-notes"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: team-notes\ndescription: Team notes.\n---\n", encoding="utf-8"
    )
    return folder.resolve()


def config_for(
    adapter_id: str,
    *,
    with_model: bool = True,
    skills: dict[str, Any] | None = None,
) -> FabricConfig:
    mapping: dict[str, Any] = {
        "metadata": {"name": AGENT},
        "harness": {"adapter_id": adapter_id},
    }
    if with_model:
        mapping["models"] = {
            "default": {"provider": "nvidia", "model": "nvidia/test-model"}
        }
    if skills is not None:
        mapping["skills"] = skills
    return FabricConfig.from_mapping(mapping)


def adapter_config(package: Path, base_dir: Path, config: FabricConfig) -> AgentConfig:
    """Load the package, plan the mapped config, and return what the adapter is handed."""
    report = load_agent_package(package, config, base_dir=base_dir, if_unsupported="warn")
    assert report.outcome == "ready"
    by_id = {item["id"]: item for item in report.declared}
    # One file is not a shape the adapters take, and a folder name maps once.
    assert by_id["agent:/skills/one-file"]["gap"] == "unsupported_shape"
    assert by_id["agent:/skills/right"]["gap"] == "key_clash"
    plan = json.loads(
        _native.plan_config(
            json.dumps(report.to_config().to_mapping()), str(report.base_dir)
        )
    )
    return AgentConfig.from_mapping(plan["agent_config"])


def expected_skills(package: Path) -> list[Path]:
    return [
        (package / "left" / "review").resolve(),
        (package / "skills" / "release-notes").resolve(),
    ]


@pytest.mark.skipif(
    sys.version_info >= (3, 15), reason="Hermes adapter requires Python 3.14 or earlier"
)
def test_hermes_config_builder_takes_the_mapped_config(package: Path, tmp_path: Path):
    from nemo_fabric_adapters.hermes import configuration

    agent_config = adapter_config(
        package, tmp_path, config_for("nvidia.fabric.hermes")
    )

    built = configuration.build_hermes_config(agent_config, workspace=".")

    assert built["model"]["default"] == "nvidia/test-model"
    assert sorted(Path(path) for path in built["skills"]["external_dirs"]) == (
        expected_skills(package)
    )
    assert built["agent"]["max_turns"] == 8
    if sys.platform != "win32":
        server = built["mcp_servers"]["changelog"]
        assert server["command"] == "/bin/sh"
        assert server["args"][-2:] == ["python3", "server.py"]


@pytest.mark.skipif(
    sys.version_info >= (3, 15), reason="Hermes adapter requires Python 3.14 or earlier"
)
def test_hermes_needs_a_model_the_caller_configures(package: Path, tmp_path: Path):
    from nemo_fabric_adapters.hermes import configuration

    config = config_for("nvidia.fabric.hermes", with_model=False)
    report = load_agent_package(package, config, base_dir=tmp_path, if_unsupported="warn")

    # The document's default model is reported and not written into the config.
    assert report.outcome == "ready"
    assert report.model["origin"] == "harness_default"
    assert report.model["declared_source"] == "ollama::llama3.2:3b"
    assert "models" not in report.to_config().to_mapping()
    # Hermes has no default of its own, so its config builder asks for a model.
    agent_config = adapter_config(package, tmp_path, config)
    with pytest.raises(ValueError, match="requires a default model"):
        configuration.build_hermes_config(agent_config, workspace=".")


def test_openclaw_takes_the_mapped_skill_paths(package: Path, tmp_path: Path):
    from nemo_fabric_adapters.openclaw import adapter

    agent_config = adapter_config(
        package, tmp_path, config_for("nvidia.fabric.openclaw")
    )

    paths = adapter._resolve_skill_paths(agent_config, tmp_path)

    assert sorted(Path(path) for path in paths) == expected_skills(package)


def test_openhands_takes_the_mapped_skill_paths(package: Path, tmp_path: Path):
    from nemo_fabric_adapters.openhands import adapter

    agent_config = adapter_config(
        package, tmp_path, config_for("nvidia.fabric.openhands")
    )

    paths = adapter._skill_paths(agent_config, str(tmp_path))

    assert sorted(paths) == expected_skills(package)
    # The adapters name a skill by its folder: each name is used once.
    assert len({path.name for path in paths}) == len(paths)


@pytest.mark.skipif(
    sys.version_info >= (3, 15), reason="Hermes adapter requires Python 3.14 or earlier"
)
def test_hermes_takes_the_mapped_skill_entries(
    package: Path, caller_skill: Path, tmp_path: Path
):
    from nemo_fabric_adapters.hermes import configuration
    from nemo_fabric_adapters.hermes import skills as hermes_skills

    # The caller's config uses skills.directories, so the package's skill
    # folders are added there as `skill` entries and never to skills.paths.
    config = config_for(
        "nvidia.fabric.hermes", skills={"directories": [{"skill": str(caller_skill)}]}
    )
    agent_config = adapter_config(package, tmp_path, config)

    assert list(agent_config.skills.paths) == []
    assert [Path(entry.skill) for entry in agent_config.skills.directories] == [
        caller_skill,
        *expected_skills(package),
    ]
    # Hermes checks every entry at start: each one holds a readable SKILL.md.
    plan = hermes_skills.plan_skills(agent_config, base_dir=tmp_path)
    built = configuration.build_hermes_config(
        agent_config, workspace=".", skill_plan=plan
    )
    assert sorted(Path(path) for path in built["skills"]["external_dirs"]) == sorted(
        [caller_skill, *expected_skills(package)]
    )


def test_openclaw_takes_the_mapped_skill_entries(
    package: Path, caller_skill: Path, tmp_path: Path
):
    from nemo_fabric_adapters.openclaw import adapter

    config = config_for(
        "nvidia.fabric.openclaw",
        skills={"directories": [{"skill": str(caller_skill)}]},
    )
    agent_config = adapter_config(package, tmp_path, config)

    # OpenClaw checks every entry at start: each one holds a readable SKILL.md.
    plan = adapter._skill_plan(
        agent_config, runtime_context(tmp_path), base_dir=tmp_path, env={}
    )
    section = adapter._skills_section(agent_config, tmp_path, plan)

    assert sorted(Path(path) for path in section["load"]["extraDirs"]) == sorted(
        [caller_skill, *expected_skills(package)]
    )


def test_adapter_without_directory_loading_gets_no_skill_entry(
    package: Path, tmp_path: Path
):
    # OpenHands accepts skills and lists no `skill` entries in its descriptor.
    # Planning refuses an entry kind an adapter does not list, so none is written.
    config = config_for("nvidia.fabric.openhands", skills={"directories": []})

    report = load_agent_package(package, config, base_dir=tmp_path, if_unsupported="warn")

    assert report.outcome == "ready"
    skills = [item for item in report.declared if item["field"] == "skills"]
    assert [item["status"] for item in skills] == ["not_mapped"] * 4
    assert {item["gap"] for item in skills} == {"adapter_does_not_accept"}
    mapped = report.to_config().to_mapping()
    assert mapped["skills"] == {"paths": [], "directories": []}
    plan = json.loads(_native.plan_config(json.dumps(mapped), str(report.base_dir)))
    assert plan["capability_plan"]["skill_directories"] == []
