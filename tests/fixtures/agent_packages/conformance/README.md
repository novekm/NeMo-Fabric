<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Package Conformance Cases

This folder holds a subset of the offline conformance cases of OADL version 0.0.1. The NVIDIA NeMo Fabric agent package loader in `crates/fabric-core/src/agent_package/` is tested against them. Nothing here is fetched or run as an agent.

The files under `cases/` are byte-for-byte copies of case files from the `conformance/` folder of the `NVIDIA/oadl` repository. Do not edit them. `manifest.json` lists them in the order of the format's own manifest and keeps the rule id (`requirement`) each case is evidence for.

## What a Case Is

Each line of `manifest.json` names one case file and what must happen:

- `"error": null` means the loader must accept the case.
- Any other value is the start of a finding of the format's own checker. The loader must refuse the case, and its refusal code must be one of the codes the runner maps that finding to.

A case file names its `mode`. The mode says what the file holds: one document, one lock, a definition with its agent, an agent with facts about one run, or a small package with its lock.

The format's checker reports every defect of a case, and the loader stops at the first one it finds. A refusal agrees when it answers the finding the manifest names or another finding the checker gives for the same case. For a refusal that names a variable or an entry, the runner also checks that the loader names the same one.

## How the Subset Was Chosen

The subset is chosen by rule from the format's manifest of 811 cases, not by hand:

1. For each mode the runner runs, the first case in manifest order that must be accepted.
2. For each mode the runner runs and each class of finding that mode's cases expect, the first case in manifest order that expects it. The class of a schema finding is `schema`. The class of any other finding is its text before the first colon.
3. Ten cases by name:
   - Six `compatibility` cases, which the mapping tests read: `called-agents-unsupported-is-named`, `required-part-unsupported-is-named`, `optional-part-unsupported-never-stops`, `one-required-entry-among-optional-ones-is-named`, `hook-that-cannot-be-run-is-named` and `linked-policy-unreadable-is-named`.
   - Four cases about the lock check: `optional-part-unresolved-left-out-of-the-lock`, `missing-selected-skill`, `agents-entry-folder-without-agent-json` and `local-source-version-recorded`.

The rule gives 80 cases. `manifest.json` records the SHA-256 of the format's manifest that the rule was applied to, in `source_manifest_sha256`.

## Which Tests Read It

| Test | What it does |
| --- | --- |
| `conformance_cases_agree` in `crates/fabric-core/src/agent_package/conformance.rs` | Runs every case except the `compatibility` ones through the reader, and compares the result with the manifest line. |
| The mapping tests in `crates/fabric-core/src/agent_package/mapping.rs` | Read the six `compatibility` cases. |

Run the case runner with its summary:

```bash
cargo test -p nemo-fabric-core --locked --lib agent_package::conformance -- --nocapture
```

To run a full conformance root instead of this subset, set `NEMO_FABRIC_AGENT_PACKAGE_CONFORMANCE` to a folder that holds a `manifest.json` and its `cases/`:

```bash
NEMO_FABRIC_AGENT_PACKAGE_CONFORMANCE=/path/to/oadl/conformance \
  cargo test -p nemo-fabric-core --locked --lib agent_package::conformance -- --nocapture
```

## What the Runner Does Not Run

The runner lists every case it does not run, with the reason. On a full root these are:

- `graph` cases that need a resolver's capture: pins, selections or more than one package. The loader takes a package that is already resolved and fetches nothing.
- Two `graph` cases whose verdict comes from capture data that a folder of files does not hold.
- `canonical` cases. The loader hashes lists of strings and needs no general canonical encoder.
- `compatibility` cases. The mapping tests read six of them. The other four are the comparison cases for a receiver that supports called agents, plugins, hooks or a readable policy. The mapping tests cover an adapter that takes called agents, plugins and hooks with packages of their own. The loader reads no policy.

## Replace the Subset

After the format's cases change, run `select_cases.py` with the format's conformance folder. It applies the rule, copies the chosen files, and rewrites `manifest.json` with the new SHA-256:

```bash
python tests/fixtures/agent_packages/conformance/select_cases.py /path/to/oadl/conformance
```

The cases repeat the format's schema address in `$schema`, so replace them together with the schema files in `crates/fabric-core/assets/oadl/<version>/`.
