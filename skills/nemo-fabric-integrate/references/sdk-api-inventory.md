<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SDK API Inventory

`Fabric()` is the primary entrypoint. It is a plain, reusable object — not a
lifecycle context manager — and can plan, diagnose, or start multiple
independent runtimes. The generated
[client reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.client.md),
[runtime reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.runtime.md),
[native OpenAI streaming reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.openai_streaming.md),
and [Relay streaming reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.streaming.md)
document the public methods with their `async` and keyword-only markers. The
installed `nemo_fabric` package also ships type information (`py.typed`) for
static analysis.

## `Fabric` Methods

The following table lists the `Fabric` methods and when to use each:

| Method | Async | Use When | Returns |
| --- | --- | --- | --- |
| `plan(config, *, base_dir=...)` | No | You need the selected adapter, capability routing, and runtime capabilities before running. | `RunPlan` |
| `doctor(config, *, base_dir=...)` | Yes | You need preflight diagnostics for adapter resolution, capability routing, declared requirements, and environment assumptions. | `DoctorReport` |
| `run(config, *, base_dir=..., input=... \| request=...)` | Yes | You need one complete start, invoke, result, and stop cycle. | `RunResult` |
| `start_runtime(config, *, base_dir=..., overrides=..., streaming=False, launch_collector=None, completion_wait_timeout=1.0)` | Yes | You need state across multiple ordered invocations. Pass `streaming=True` with NVIDIA NeMo Relay enabled only to provision `invoke_stream(...)`. | `Runtime` |
| `start_agent_package(package_dir, config, *, base_dir=..., options=..., check=..., session_ref=..., overrides=..., streaming=False)` | Yes | You need a runtime started from an agent package in one call. `options` holds the `load_agent_package` options by name. A `check` callable gets the load report, and the start goes ahead only when it returns a mapping whose `decision` is `"allow"`. A `checker` program named in `options` is asked first. | `Runtime`, with the load report as `agent_package_report` |
| `prepare_environment(config, *, base_dir=...)` | Yes | Prepare an environment independently of a runtime; preparation depends on provider support. | `EnvironmentHandle` |
| `attach_environment(config, reference, *, base_dir=...)` | Yes | Verify a caller-owned resource with a typed `EnvironmentReference` when the provider supports attachment. | `EnvironmentHandle` |
| `start_runtime_in(config, environment, *, base_dir=..., overrides=..., streaming=False)` | Yes | Start a runtime in a typed environment when the selected runtime provider supports it. | `Runtime` |
| `release_environment(environment)` | Yes | Release the environment separately after runtime stop; local release preserves files. | `None` |

`input` and `request` on `run(...)` are mutually exclusive. Use `input=...` for
the common case; use `request=RunRequest(...)` when the invocation needs a
caller-owned `request_id`, `context`, or overrides.

The public `inspect_adapter(config, descriptor)` helper returns `AdapterInspection`,
with an empty `issues` list and an immutable `capability_profile` on success.
Compatibility failures return structured `AdapterCompatibilityIssue` values with
`adapter_id`, `field`, and `reason`; incompatible reports have no profile.
Check issues before deployment. `diagnostics=True` is the default; pass `diagnostics=False` for exception-based rejection. Malformed inputs raise in both modes. Inspection and planning both require declared Relay ATIF output support and exactly one observability component when ATIF is requested.
The nested `AdapterCapabilityProfile` supports host admission without harness SDKs.
Both `run()` and `start_runtime()` accept optional
`expected_descriptor_sha256` to verify the task plan's descriptor before startup.
The profile is configuration-dependent for telemetry. Registered workflows
require target metadata and bind both descriptors in the fingerprint; attached
services require separate admission. The profile is not observed execution provenance.

The public `load_agent_package(package_dir, config=None, *, base_dir=..., entry=..., variables=..., instruction_mode=..., if_unsupported="stop", if_unverified="warn", check_machine=False, facts=..., expected_lock_sha256=..., bindings=..., checker=..., checker_timeout_seconds=...)`
function is synchronous. It reads an agent package, checks every part against
the package's lock, and returns an immutable `AgentPackageReport`. With a
config, `report.to_config()` returns the mapped `FabricConfig` when
`report.outcome` is `ready`. With no config, the package is verified only and
`report.digests` lists each lock entry. It fetches nothing. The config's
`models["default"]` is the model choice. With none, the `models` entry named
with the exact text of the package's default model becomes the default, and
`report.model["origin"]` is `package`. Plugins, hooks and called agents are
written under the root keys `nemo_fabric.plugins.v1`, `nemo_fabric.hooks.v1`
and `nemo_fabric.agents.v1` when the selected adapter's descriptor declares
the key, and `report.adapter["extensions"]` lists the declared keys. A custom
tool keeps the config's own `tools.definitions.<name>` entry. The `packages`
fact lists installed software packages by registry for the loader to compare
with the lock; a required package that no fact tells about stops the load.
Pass `expected_lock_sha256` to refuse a
package whose lock file is not the one you recorded. Pass `bindings` to account
for the custom tools your application supplies and the called agents it runs:
a key is `custom_tools.<name>` or `agents.<name>`, a value is one line of text
that names your target, and the loader never imports or runs it. A bound thing
is a `report.declared` line with status `bound` and the target in `bound_to`;
the mapped config does not hold the bindings. Pass `checker` to have a program
decide whether the loaded package may start: the program, then its arguments.
It is started with no shell, gets the package folder, the path of the load
report file and the agent name in `NEMO_FABRIC_PACKAGE_DIR`,
`NEMO_FABRIC_PACKAGE_REPORT` and `NEMO_FABRIC_AGENT_NAME`, and answers with
one JSON object and exit code 0 for `"decision": "allow"` or 2 for
`"decision": "deny"`. Anything else is a deny. `report.checker_result` holds
the answer, and a load that is not allowed has `report.outcome == "denied"` and
no config. An empty `checker` list is refused. See the
[agent packages guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/agent-packages.mdx).

The public `fetch_agent_package(package_dir, out_dir, *, entry=..., expected_lock_sha256=..., oci_layout=...)`
function is synchronous. It and `fetch_agent_package_from_git` are the only
package functions that use the network. It completes a package from its own
lock: content the lock pins from git is fetched at the pinned commit with the
`git` program, and content pinned as an OCI artifact is read from the image
layout folder given as `oci_layout`. Everything placed is checked against the
digest the lock pins before `out_dir` is written, and `out_dir` holds what the
lock pins and nothing else. `fetch_agent_package_from_git(source, ref, out_dir)`
takes the same options and first fetches a package that is itself a folder in
a git repository: `source` is the text `git::<repository>//<folder>`. Both
return an immutable `AgentPackageFetchReport`: load its `package_dir`, and
pass its `lock["sha256"]` to `load_agent_package` as `expected_lock_sha256`.

OpenShell environments use `start_runtime_in` with a prepared or attached
handle. They support buffered invocation and multiple independent bridge
runtime IDs per sandbox. Invocations for different runtime IDs can proceed
independently. The task SDK owns each runtime ID and normalized result. Declared
files become host paths under the configured artifact destination; a failed SDK
result still carries its output, error, and artifacts. Lost responses are not
replayed. See the [OpenShell provider guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/environment-providers/openshell/README.md).

## Runtime Methods

The following table lists the `Runtime` members for driving a stateful runtime.

| Member | Async | Notes |
| --- | --- | --- |
| `invoke(*, input=... \| request=...)` | Yes | One turn on an active runtime. One active invocation at a time; overlap raises `FabricStateError`. |
| `invoke_openai_stream(*, input=... \| request=...)` | No | Start exactly one descriptor-gated native invocation and return an async `OpenAIInvokeStream` of `chat.completion.chunk` mappings. Await `stream.result()` for the separate terminal `RunResult`. |
| `invoke_stream(*, input=... \| request=...)` | No | Start one NeMo Relay turn and return an async `InvokeStream` of raw ATOF records. Await `stream.result()` for the terminal `RunResult`. |
| `resource_snapshot()` | Yes | Timestamped memory, process count, and CPU time of the local adapter host and its descendants. Does not wait for an active invocation. |
| `stop()` | Yes | Stop the runtime. Called automatically by `async with`. |
| `status` | No | `RuntimeStatus`: `ACTIVE`, `STOPPED`, or `FAILED`. |
| `supports_openai_streaming` | No | `True` when the selected descriptor declares `capabilities.streaming` for native OpenAI Chat Completions chunks. |
| `supports_streaming` | No | `True` when NeMo Relay ATOF streaming was enabled at runtime startup. |
| `runtime_id` | No | Opaque identifier for this runtime lifecycle. |
| `messages` / `invocations` | No | Copied harness history and per-turn IDs. |

Always use a runtime as an async context manager so cleanup runs on exit.
Shutdown is attempted, not guaranteed — `stop()`, including the automatic call at
`async with` exit, can raise `FabricRuntimeError`:

```python
async with await fabric.start_runtime(config, base_dir=base) as runtime:
    result = await runtime.invoke(input="…")
```

## Stream Handles

The two stream handles represent independent contracts:

| Class | Async Iteration | Terminal Result | Early Close |
| --- | --- | --- | --- |
| `OpenAIInvokeStream` | Adapter-native `chat.completion.chunk` mappings; an empty stream is valid. | `await result()` drains and discards unread chunks, then returns the separate authoritative `RunResult`. | `await aclose()` drains and discards unread chunks without cancelling the invocation. |
| `InvokeStream` | Raw invocation-correlated ATOF records generated by NeMo Relay. | `await result()` returns the separate authoritative `RunResult`. | `await aclose()` drains and discards unread records without cancelling the invocation. |

Fully consume either stream or call `await stream.aclose()` before starting
another turn. For native OpenAI streaming, `result()` also finalizes and
discards unread chunks. Relay `InvokeStream.result()` does not consume unread
ATOF records.

## Execution Model

NVIDIA NeMo Fabric separates configuration, planning, runtime lifecycle, and
individual invocations:

```text
FabricConfig -> plan() -> RunPlan -> start_runtime() -> Runtime -> invoke() -> RunResult
                                                            \-> invoke_openai_stream() -> OpenAIInvokeStream
                                                            \-> invoke_stream() -> InvokeStream
```

- `Fabric` is a lightweight facade; it holds no started state and needs no
  cleanup.
- A `Runtime` owns stateful execution and shutdown, so it is the object used
  with `async with`.
- A runtime is a logical execution boundary, not necessarily an operating-system
  process. Harness-native threads, sessions, and conversations remain
  adapter-owned state associated with the runtime.
- Runtime hosting is adapter-declared, not consumer-configured. The bundled
  Claude, Codex, Deep Agents, and Hermes Agent adapters retain their native runtime
  resources in one local host. Local `process` and `python` adapters use the
  same host lifecycle. A host crash or protocol timeout terminates the runtime;
  the application can explicitly start a new runtime according to its policy.
- The application owns scheduling, queues, retries, and how many runtimes to
  run. NeMo Fabric provides only the runtime contract.
