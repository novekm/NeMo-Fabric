# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy the rule-chosen subset of the package format's conformance cases into this folder.

Usage::

    python tests/fixtures/agent_packages/conformance/select_cases.py /path/to/oadl/conformance

The rule is stated in the README beside this file. Case files are copied byte for byte, and
``manifest.json`` is rewritten. Only the standard library is used, and nothing is fetched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Modes the case runner in crates/fabric-core/src/agent_package/conformance.rs runs.
RUN_MODES = frozenset({"document", "text", "lock", "start", "definition", "run", "called"})
# Graph cases whose verdict comes from capture data that a folder of files does not hold.
CAPTURE_ONLY_GRAPH_CASES = frozenset({"stored-part-not-stated", "stored-part-states-dependencies"})
# Cases chosen by name: six for the mapping tests, four about the lock check.
NAMED_CASES = (
    "called-agents-unsupported-is-named",
    "required-part-unsupported-is-named",
    "optional-part-unsupported-never-stops",
    "one-required-entry-among-optional-ones-is-named",
    "hook-that-cannot-be-run-is-named",
    "linked-policy-unreadable-is-named",
    "optional-part-unresolved-left-out-of-the-lock",
    "missing-selected-skill",
    "agents-entry-folder-without-agent-json",
    "local-source-version-recorded",
)


def case_name(record: dict[str, Any]) -> str:
    return Path(record["path"]).stem


def is_run(record: dict[str, Any], case: dict[str, Any]) -> bool:
    """Return whether the case runner runs this case."""
    if case["mode"] in RUN_MODES:
        return True
    if case["mode"] != "graph":
        return False
    inventory = case["inventory"]
    one_local_package = (
        len(inventory["packages"]) == 1
        and not inventory.get("pins")
        and not inventory.get("selections")
    )
    return one_local_package and case_name(record) not in CAPTURE_ONLY_GRAPH_CASES


def finding_class(error: str | None) -> str | None:
    """Return ``schema`` for a schema finding, else the finding's text before the first colon."""
    if error is None:
        return None
    return "schema" if error.startswith("schema:") else error.split(":", 1)[0]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    source = Path(argv[1])
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    records = manifest["cases"]
    by_name = {case_name(record): record for record in records}

    chosen: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str | None]] = set()
    for record in records:
        case = json.loads((source / record["path"]).read_text(encoding="utf-8"))
        key = (case["mode"], finding_class(record["error"]))
        if is_run(record, case) and key not in seen:
            seen.add(key)
            chosen[case_name(record)] = record
    for name in NAMED_CASES:
        chosen[name] = by_name[name]
    ordered = [record for record in records if case_name(record) in chosen]

    cases_dir = HERE / "cases"
    if cases_dir.exists():
        shutil.rmtree(cases_dir)
    cases_dir.mkdir()
    for record in ordered:
        shutil.copyfile(source / record["path"], HERE / record["path"])
    subset = {
        "format": manifest["format"],
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_cases": len(records),
        "cases": [{
                "path": record["path"],
                "error": record["error"],
                "requirement": record["requirement"],
            }
            for record in ordered],
    }
    (HERE / "manifest.json").write_bytes((json.dumps(subset, indent=2) + "\n").encode("utf-8"))
    print(f"copied {len(ordered)} of {len(records)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
