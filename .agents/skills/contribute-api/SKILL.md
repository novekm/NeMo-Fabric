---
name: contribute-api
description: Contribute a new NVIDIA NeMo Fabric public API surface safely, with Rust, CLI, Python, TypeScript, schema, adapter, and documentation parity in mind
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Contribute A New API Surface

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when contributing a public API addition or behavior change to the
runtime or bindings.

## Default Guidance

- Start from the shared Rust core behavior first
- Decide whether the CLI, PyO3 binding, Python SDK, type stubs, schemas, or the
  Python and TypeScript adapter-contract bindings must expose the new surface
- Keep every affected public surface in parity
- `RunRequest.relay_session_root` is forwarded as the typed `AgentRunRequest.relay_session_root` field. Keep Rust, Python, TypeScript, and schema representations in parity. Context keys do not control Relay propagation; preserve the adapter fallback for unusable UUIDs.
- OpenShell task commands serialize public SDK config/request/handle/result mappings. Keep forwarding buffered and preserve task-generated IDs, normalized failed results, telemetry references, and declared artifacts. Do not normalize an SDK result as an adapter result. Task paths remain POSIX strings; host artifact destinations stay separate. Unknown outcomes bind the immutable sandbox identity through reattachment and must not be replayed.
- Runtime discovery uses the typed public `Fabric.runtime_status()` operation and `sdk/runtime-status-report` schema. Keep `fabric-agent-ctl status` delegated to that SDK operation and OpenShell status routed through the existing provider command transport. Preserve recorded lifecycle state separately from health observations, use null handles for stopped records, and exclude environment variables and arbitrary provider metadata from discovery snapshots. Each client caps all runtime records, including pending starts, at 128; evict only stopped records and reject full-capacity starts with `runtime_limit_reached` before runtime effects.
- Hermes and OpenClaw cold resume take an optional `session_ref` directly at runtime start. Omission starts a fresh conversation. `runtime.keep_state` controls retention. Keep SDK, managed commands, and adapter contexts aligned. Native-only Hermes configurations retain state, advertise `session_resume: false`, and reject explicit SDK session selection.
- Metadata inspection takes canonical descriptors in memory. Python
  `inspect_adapter` returns `AdapterInspection`; consume the nested
  `capability_profile` only after checking `issues`. Compatibility failures
  return issues by default (`diagnostics=True`); support `diagnostics=False`
  for exception-based rejection. Malformed inputs raise in both modes.
  Keep startup validation rules shared and preserve malformed-input errors.
  Inspection and planning share Relay ATIF admission: requested output requires
  descriptor support and exactly one observability component.
- Agent packages load through `load_agent_package` in `nemo_fabric_core::agent_package`, the `nemo_fabric.agent_package` module, and `fabric-agent-ctl start --package`. `Fabric.start_agent_package` is `load_agent_package`, then the caller's optional `check`, then `start_runtime`. The `checker` option names a program the load asks last, in `crates/fabric-core/src/agent_package/checker.rs`, so every surface that loads a package has it. Both fail closed: the start goes ahead only on `allow`, and NeMo Fabric knows no checker by name. Fetching is a step of its own: `fetch_agent_package` and `fetch_agent_package_from_git`. Keep the Rust options and report types, the four `sdk/agent-package-*` schemas, the Python functions and report classes, and `docs/sdk/agent-packages.mdx` in parity. The load fetches nothing and adds no `FabricConfig` field, and a field it cannot map stays a line in the report. Refusal codes are stable: add a code instead of changing what one means. The guide has the rules for models, skills, bindings and the root extension keys. After a change, run the case runner in `crates/fabric-core/src/agent_package/conformance.rs` and `tests/examples/test_agent_package_example.py`.
- Update docs and examples in the same branch

## Minimum Acceptance

- Public behavior is clearly described
- Every affected public surface is covered
- The validation matrix matches the changed surfaces
- PR notes explain the user-facing change

## References

- `validate-change`
- `review-doc-style`
- `docs/python-sdk-contract.md`
- `schemas/SCHEMA.md`
- `justfile`

- Software provenance uses `FabricConfig.provenance` and optional `RunResult.execution`. Keep Rust, shared Python evidence dataclasses in `nemo-fabric-evidence`, SDK result mappings, generated TypeScript types and the canonical `schemas/evidence/execution-evidence.schema.json` aligned. The shared schema is generated from Rust and consumed by both NB and SB bindings; keep the TypeScript adapter-contract schema copy byte-identical. Preserve strict adapter result shapes: lifecycle evidence is carried beside the outcome. Optional report failure must not replace invocation outcomes.
