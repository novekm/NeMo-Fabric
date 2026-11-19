<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent package schemas, OADL 0.0.1

These four files are the JSON Schema files of the Open Agentic Declarative Language
(OADL), version 0.0.1. NVIDIA NeMo Fabric compiles them into `nemo-fabric-core` and
checks an agent package against them. It never fetches a schema.

| File | What it checks |
|---|---|
| `agent.schema.json` | `agent.json` |
| `agent-definition.schema.json` | `agent-definition.json` |
| `lock.schema.json` | `oadl.lock.json` |
| `reference.schema.json` | The reference shape that the other three use |

## Where they come from

They are byte-for-byte copies of the files under `schemas/0.0.1/` in the
`NVIDIA/oadl` repository. Do not edit them here.

The files refer to each other by the address in their `$id`. The loader reads that
address from the files and never spells it in code. A document names the same address in
its `$schema`, so a package written for another version does not load.

## How to add or replace a version

1. Copy the four files from the format's `schemas/<version>/` folder into
   `assets/oadl/<version>/`.
2. Run `cargo test -p nemo-fabric-core --locked --lib agent_package::`.
3. Rewrite the lock of every example and fixture package: the digest of a document
   changes when its `$schema` changes.

This folder is named for the format so that its `agent.schema.json` is not confused with
NeMo Fabric's own `schemas/sdk/agent.schema.json`.
