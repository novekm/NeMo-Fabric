<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Package Example

This example loads an agent package with the NVIDIA NeMo Fabric Python SDK and prints the load report. It needs no model endpoint and no network. The [agent packages guide](../../docs/sdk/agent-packages.mdx) explains what a package is and what the loader does with each field.

## What Is in the Folder

The package is the folder `release-notes-agent/`. Every file it pins is in the folder, so nothing is fetched.

| Path | What it is |
| --- | --- |
| `release-notes-agent/agent.json` | The agent: its instructions, one variable, its limits, one skill, one connector and one software need. |
| `release-notes-agent/oadl.lock.json` | The agent's lock. It pins `agent.json` and each part by digest. |
| `release-notes-agent/skills/release-notes/` | A skill folder with its `SKILL.md`. |
| `release-notes-agent/connectors/changelog/` | A connector carried in the package: `mcp.json` says how to start it, `server.py` is the MCP server, and `changes.json` is the data it serves. The server uses the Python standard library only. |
| `release-notes-agent/variants/short/agent-definition.json` | A definition. It points at the agent with `./../../agent.json`, adds one skill, adds a line of instructions and lowers `max_turns` from 12 to 6. |
| `release-notes-agent/variants/short/oadl.lock.json` | The definition's lock. It pins the definition, the agent it points at, the agent's parts and the skill the definition adds. |
| `release-notes-agent/variants/short/skills/short-form/` | The skill the definition adds. |

## Verify the Package

Build NeMo Fabric first, as the [install guide](../../docs/getting-started/install.mdx) describes. Then run the example from the repository root:

```bash
.venv/bin/python -m examples.agent_package
```

With no config, the loader reads the agent, checks it against its schema, and checks every part against the lock. The report has `outcome: verified_only`. Its `digests` list has one line for each lock entry, with the digest, the result `verified` and the folder or file that was checked.

Load the definition and the agent it points at:

```bash
.venv/bin/python -m examples.agent_package --variant
```

The report's `entry` is the definition and its `agent` is the agent. Each line of `declared` says which of the two documents stated it.

## Map the Package Onto a Config

Pass a complete NeMo Fabric config to map the package onto it. The config holds what the package has no field for: the harness adapter, the model endpoint and the environment. Its `metadata.name` must equal the name of the entry document, which is `release-notes-agent` for the agent and `release-notes-short` for the variant.

Save this as `config.json`, and replace the model fields with your own endpoint:

```json
{
  "metadata": {"name": "release-notes-agent"},
  "harness": {"adapter_id": "nvidia.fabric.hermes"},
  "models": {
    "default": {
      "provider": "openai",
      "model": "served-model",
      "base_url": "https://inference.example.com/v1",
      "api_key_env": "MODEL_API_KEY"
    }
  }
}
```

Then run:

```bash
.venv/bin/python -m examples.agent_package --config config.json --var max_turns=8
```

The report has `outcome: ready` and a `config` member. That config is a normal `FabricConfig`: pass `report.to_config()` to `Fabric.plan`, `Fabric.run` or `Fabric.start_runtime`. The command exits with status 2 when the outcome is `stopped`, and the report's `stopped_by` list names what stopped it. Add `--if-unsupported warn` to continue and read the report instead.

## Change the Package

A lock digest is computed from bytes, with one rule for a file and one for a folder:

- **A file:** `sha256:` and the SHA-256 of the file's bytes.
- **A folder:** `sha256:` and the SHA-256 of a compact JSON array. The array holds one `[path, file digest]` pair for every file under the folder, sorted by path. The path is relative to the folder and uses `/`.

The digests in the two lock files were computed with that rule and checked with the loader. After you change a file, run the example again. The loader refuses the package with `package_digest_mismatch`, and the message gives the pinned digest, the computed digest and the path. Write the computed digest into each lock entry that pins the file. `agent.json` and both parts are pinned in both lock files.

Every file in a pinned folder counts. A stray file, such as a `__pycache__` folder inside a skill or connector folder, changes the folder's digest. The one exception is a file named `.DS_Store`: the loader leaves it out of a folder's digest. The `.gitattributes` file in this directory turns off line-ending conversion for the package, so a checkout on Windows has the same bytes. For the same reason `.pre-commit-config.yaml` leaves `release-notes-agent/` out of the copyright header hook: the hook rewrites the year in a header, which would change a pinned file.
