<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Kiro Adapter

This package provides the `nvidia.fabric.kiro` process adapter. It starts a
caller-installed Kiro CLI V3 ACP agent, creates one persistent ACP session per
NeMo Fabric runtime, and reuses that session across ordered invocations.

## Install the Adapter

Install Node.js 22.19.0 or newer, install and authenticate the Kiro CLI, then
install the adapter:

```bash
curl -fsSL https://cli.kiro.dev/install | bash
kiro-cli login
npm install nemo-fabric-adapters-kiro
```

For non-interactive authentication, set `KIRO_API_KEY` instead of using the
browser login. The adapter never installs or updates Kiro at runtime.

For source development, run these commands from the repository root:

```bash
just install-typescript-kiro
npm run build --prefix adapter-contract/typescript
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-common
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-kiro
```

Register the packaged descriptor at
`node_modules/nemo-fabric-adapters-kiro/kiro.fabric-adapter.json`. Source
checkouts can register `adapters/typescript/kiro/kiro.fabric-adapter.json`.

## Tested Kiro Version

Minimum supported: Kiro CLI 2.37.1. Tested on Kiro CLI 2.37.1 and 3.5.0 (`kiro-cli acp --agent-engine=v3`, accepted by both). On 3.5.0 every mapped part runs; effort does not, because no model on the test account offers an effort option. Hooks still need the ACP client opt-in, which the adapter sends. ACP shows no workspace trust prompt on either version, so the adapter does not touch trust settings.

## Supported Configuration

The adapter maps configuration Kiro V3 exposes through ACP or loads from the session's workspace:

- The optional `default` NeMo Fabric model role selects an advertised Kiro model.
  Its provider is `kiro`; `api_key_env`, when present, must be
  `KIRO_API_KEY`.
- `harness.settings.mode_id` selects an advertised ACP session mode.
- `harness.settings.effort` selects an advertised Kiro effort level.
- `harness.settings.permission_mode` is `deny` by default. Set it to
  `allow_once` to choose only one-time allow options explicitly offered by
  Kiro. The adapter never grants persistent permission.
- NeMo Fabric MCP stdio, streamable HTTP, and SSE definitions are sent with
  `session/new`. Stdio executables must be absolute. Network MCP endpoints
  require HTTPS except for loopback development endpoints. Startup waits for
  Kiro to report every configured MCP server as connected before the first
  invocation.

- The package's own agent and every called agent it names are written as Kiro custom
  agents, `<workspace>/.kiro/agents/<name>.json`, before the session starts, and removed
  when it stops. The session selects the package's own agent as its mode, so
  - `instructions.system` (append mode only) is its `prompt`;
  - `tools.enabled` and `tools.blocked` choose the Kiro tools it can see (`read_file`,
    `write_file`, `edit_file`, `grep`, `glob`, `shell`, `web_fetch`, `web_search`, or Kiro's
    own names). With none listed it sees no tool but the sub-agent tool, its MCP servers'
    tools and its skills;
  - `skills.paths` folders are `skill://` resources;
  - each MCP server is a tool it sees (`@server`).
- Called agents (`nemo_fabric.agents.v1`) get their instructions as `prompt`, their `tools`
  (same names), and their `skills` and `plugins` folders as `skill://` resources. Kiro's own
  sub-agent tool hands a task to one by name. `max_turns` and a per-agent model are not
  taken.
- The agents' permission rules pre-allow exactly what the package names, so
  `permission_mode` can stay `deny`: the tools an agent lists, the skills by name, the
  agents the main agent may call and its MCP servers. Kiro runs a sub-agent under the
  session agent's rules, so the main agent's file carries those of the agents it calls.
  Anything else Kiro asks about is answered by `permission_mode`.
  `harness.settings.mode_id` cannot be set when the package gives the session an agent.

The adapter never overwrites or removes an agent file it did not write: such a file stops
startup. Kiro's advertised modes, models, effort values, and
MCP transport capabilities are authoritative. Startup fails instead of silently
ignoring an unsupported selection. Maximum turns, depth and parallelism limits, hooks,
per-agent models, arbitrary model endpoints, telemetry, and streaming are not advertised
by this adapter; Kiro has no setting for the limits, and it did not run agent hooks in the
ACP sessions the adapter starts.

See the [Kiro integration guide](../../../docs/integrations/harness/kiro.mdx)
for a complete configuration example.

## Runtime health

`Fabric.check()` supports `live`. The adapter observes the selected ACP child
process identifier, exit status, and signal status through the shared process
runner. A retained session identifier is insufficient. A busy process remains
live, and a stopped or exited process fails the liveness observation.

`active` and `ready` are unsupported. Checks do not send prompts, change session settings, or cancel invocations.

Refer to the [SDK health-check overview](../../../docs/sdk/health-checks.mdx) for shared behavior and a comparison of native evidence across adapters.


## Invocation deadlines

Kiro invocation deadlines use canonical error code `timeout`. Error metadata retains `owner: "adapter"`, the native cause, and the enforced timeout in seconds or absolute `deadline_millis`. Setup and transport failures retain their existing classifications.

When `runtime.timeout_seconds` is omitted or null, NeMo Fabric does not impose a total prompt deadline. An explicit invocation deadline is forwarded to the ACP session; startup and shutdown remain separately bounded.
