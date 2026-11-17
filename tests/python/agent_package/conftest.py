# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric import FabricConfig

from _utils.agent_packages import REPO_ROOT, agent_document, write_package

AGENT_NAME = "release-notes-agent"
SKILL_TEXT = """---
name: release-notes
description: Draft release notes from a list of changes.
---

# Release notes

Group the changes under Added, Changed and Fixed.
"""
SERVER_TEXT = """import sys

sys.exit(0)
"""


@pytest.fixture(name="agent")
def agent_fixture() -> dict[str, Any]:
    """An agent every part of which the test adapter accepts."""
    return agent_document(
        AGENT_NAME,
        description="Drafts release notes from a list of changes.",
        model={"source": "test::test-model"},
        limits={"timeout_seconds": {"var": "timeout"}},
        variables={
            "timeout": {
                "type": "number",
                "default": 600,
                "enum": [300, 600],
                "description": "Longest one draft may take, in seconds.",
            }
        },
        skills={"release-notes": {"source": "./skills/release-notes"}},
    )


@pytest.fixture(name="package_files")
def package_files_fixture() -> dict[str, str]:
    """The bytes of every part file, by path inside the package."""
    return {
        "skills/release-notes/SKILL.md": SKILL_TEXT,
        "skills/short-form/SKILL.md": SKILL_TEXT.replace("release-notes", "short-form"),
        "hooks/tool-guard/hooks.json": json.dumps({"hooks": {}}, indent=2) + "\n",
        "connectors/changelog/mcp.json": json.dumps(
            {
                "mcpServers": {
                    "main": {
                        "type": "stdio",
                        "command": "python3",
                        "args": ["server.py"],
                    }
                }
            },
            indent=2,
        )
        + "\n",
        "connectors/changelog/server.py": SERVER_TEXT,
    }


@pytest.fixture(name="package_dir")
def package_dir_fixture(
    tmp_path: Path, agent: dict[str, Any], package_files: dict[str, str]
) -> Path:
    """A package folder written from explicit bytes, with its lock."""
    root = tmp_path / "package"
    write_package(root, agent, package_files)
    return root


@pytest.fixture(name="shim_config")
def shim_config_fixture() -> FabricConfig:
    """A complete config that selects the test adapter and names no model."""
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": AGENT_NAME},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {
                "input_schema": "chat",
                "output_schema": "message",
                "artifacts": "./artifacts",
            },
            "environment": {
                "provider": "local",
                "workspace": "./repos/my-service",
                "artifacts": "./artifacts/local",
            },
        }
    )


@pytest.fixture(name="package_parts_dir")
def package_parts_dir_fixture() -> Path:
    """The base directory of the test adapter that declares the three package keys.

    It takes plugins, hooks and called agents. The adapter is a descriptor
    only: a config plans on it and nothing starts.
    """
    return REPO_ROOT / "tests" / "fixtures" / "agent_packages"


@pytest.fixture(name="package_parts_config")
def package_parts_config_fixture() -> FabricConfig:
    """A complete config for the video producer that selects that adapter."""
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "video-producer"},
            "harness": {"adapter_id": "test.fabric.package_parts"},
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {},
        }
    )
