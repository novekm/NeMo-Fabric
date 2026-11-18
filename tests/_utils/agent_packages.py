# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write agent packages for tests, with a lock computed from the bytes written.

This is a second implementation of the digest rule, with the standard library
only, so a test does not check the loader against itself. Every file is written
from explicit bytes: no test depends on checkout bytes or line endings.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "crates" / "fabric-core" / "assets" / "oadl" / "0.4.0"
LOCK_NAME = "oadl.lock.json"
PART_TYPES = {
    "skills": "skill",
    "plugins": "plugin",
    "connectors": "connector",
    "hooks": "hook",
    "packages": "package",
    "policies": "policy",
    "agents": "agent",
}
TOOLS_HASH = "sha256:" + "a" * 64
CARD_DIGEST = "sha256:" + "b" * 64
ARTIFACT_DIGEST = "sha256:" + "c" * 64


def schema_id(kind: str) -> str:
    """The `$id` of a vendored schema: `agent`, `agent-definition` or `lock`."""
    schema = json.loads((SCHEMA_DIR / f"{kind}.schema.json").read_text(encoding="utf-8"))
    return schema["$id"]


def agent_document(name: str, **members: Any) -> dict[str, Any]:
    """An agent document with the three required fields and the given members."""
    return {"$schema": schema_id("agent"), "type": "agent", "name": name, **members}


def definition_document(name: str, agent_source: str, **members: Any) -> dict[str, Any]:
    """A definition that points at one agent, with the given members."""
    return {
        "$schema": schema_id("agent-definition"),
        "type": "agent-definition",
        "name": name,
        "agent": {"source": agent_source},
        **members,
    }


def write_bytes(path: Path, content: bytes | str) -> Path:
    """Write exact bytes, creating the folders above the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    return path


def write_json(path: Path, value: Any) -> Path:
    """Write a JSON value as two-space indented text with one final newline."""
    return write_bytes(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def file_digest(path: Path) -> str:
    """`sha256:` and the SHA-256 of the file's bytes."""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def folder_digest(folder: Path) -> str:
    """`sha256:` and the SHA-256 of the folder's sorted `[path, file digest]` pairs.

    A file named `.DS_Store` is never one of a folder's files.
    """
    pairs = [
        [path.relative_to(folder).as_posix(), file_digest(path)]
        for path in folder.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    ]
    pairs.sort(key=lambda pair: pair[0].encode("utf-8"))
    encoded = json.dumps(pairs, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _segment(name: str) -> str:
    return name.replace("~", "~0").replace("/", "~1")


def _relative_url(target: Path, lock_dir: Path) -> str:
    return "./" + Path(os.path.relpath(target, lock_dir)).as_posix()


def _direct(entries: list[dict[str, Any]], parent: str) -> list[str]:
    """The addresses of the entries the document at `parent` references itself."""
    return sorted(
        entry["address"]
        for entry in entries
        if entry["address"].rsplit("/", 2)[0] == parent
    )


def _part_entries(
    document: Mapping[str, Any],
    document_dir: Path,
    parent: str,
    lock_dir: Path,
) -> list[dict[str, Any]]:
    """Lock entries for the parts one document references.

    A local source is hashed. A running connector or agent is a pointer. A
    package from a registry is pinned by a made-up artifact digest. Any other
    source, and a local source with no bytes, is left out of the lock: the
    caller marks such a reference optional. The parts of a called agent that
    is kept in the package are locked below it.
    """
    entries = []
    for field, entry_type in PART_TYPES.items():
        for name, reference in (document.get(field) or {}).items():
            address = f"{parent}/{field}/{_segment(name)}"
            source = reference["source"]
            if source.startswith("./"):
                target = (document_dir / source).resolve()
                if not target.exists():
                    continue
                # A called agent kept in a folder is pinned by its document.
                if entry_type == "agent" and target.is_dir():
                    target = target / "agent.json"
                below = []
                if entry_type == "agent":
                    called = json.loads(target.read_text(encoding="utf-8"))
                    below = _part_entries(called, target.parent, address, lock_dir)
                entries.append(
                    {
                        "address": address,
                        "type": entry_type,
                        "source": {
                            "type": "local",
                            "url": _relative_url(target, lock_dir),
                        },
                        "version": "local",
                        "digest": folder_digest(target)
                        if target.is_dir()
                        else file_digest(target),
                        "dependencies": _direct(below, address),
                    }
                )
                entries.extend(below)
            elif source.startswith(("mcp::", "a2a::")) and not reference.get("optional"):
                interface = {"declared_version": reference["version"]}
                if source.startswith("mcp::"):
                    interface["tools_hash"] = TOOLS_HASH
                else:
                    interface["card_digest"] = CARD_DIGEST
                entries.append(
                    {
                        "address": address,
                        "type": entry_type,
                        "source": {"type": "pointer"},
                        "version": reference["version"],
                        "constraint": reference["version"],
                        "interface": interface,
                        "dependencies": [],
                    }
                )
            elif field == "packages" and "::" in source and not reference.get("optional"):
                registry, _, name_in_registry = source.partition("::")
                entries.append(
                    {
                        "address": address,
                        "type": entry_type,
                        "source": {
                            "type": "registry",
                            "url": name_in_registry,
                            "registry": registry,
                        },
                        "version": reference["version"],
                        "constraint": reference["version"],
                        "digest": ARTIFACT_DIGEST,
                        "dependencies": [],
                    }
                )
    return entries


def write_lock(entry_path: Path) -> Path:
    """Compute and write the lock beside an entry document that is already on disk.

    The entry document is an agent, or a definition whose agent is a local file.
    """
    entry_path = entry_path.resolve()
    lock_dir = entry_path.parent
    document = json.loads(entry_path.read_text(encoding="utf-8"))
    parts = _part_entries(document, lock_dir, "$", lock_dir)
    entries = []
    dependencies = _direct(parts, "$")
    if document["type"] == "agent-definition":
        agent_path = (lock_dir / document["agent"]["source"]).resolve()
        if agent_path.is_dir():
            agent_path = agent_path / "agent.json"
        agent = json.loads(agent_path.read_text(encoding="utf-8"))
        agent_parts = _part_entries(agent, agent_path.parent, "$/agent", lock_dir)
        entries.append(
            {
                "address": "$/agent",
                "type": "agent",
                "source": {
                    "type": "local",
                    "url": _relative_url(agent_path, lock_dir),
                },
                "version": "local",
                "digest": file_digest(agent_path),
                "dependencies": _direct(agent_parts, "$/agent"),
            }
        )
        entries.extend(agent_parts)
        dependencies.append("$/agent")
    entries.append(
        {
            "address": "$",
            "type": document["type"],
            "source": {"type": "local", "url": f"./{entry_path.name}"},
            "version": "local",
            "digest": file_digest(entry_path),
            "dependencies": sorted(dependencies),
        }
    )
    entries.extend(parts)
    entries.sort(key=lambda entry: entry["address"])
    return write_json(
        lock_dir / LOCK_NAME,
        {
            "$schema": schema_id("lock"),
            "type": "lock",
            "generator": "nemo-fabric-tests",
            "entries": entries,
        },
    )


def write_package(
    root: Path,
    document: Mapping[str, Any],
    files: Mapping[str, bytes | str] | None = None,
    *,
    entry: str | None = None,
) -> Path:
    """Write the files, the entry document and its lock. Returns the entry document's path.

    `files` maps a path inside the package to its content. `entry` is the entry
    document's path inside the package; the default is `agent.json` or
    `agent-definition.json` at the root, by the document's type.
    """
    for relative, content in (files or {}).items():
        write_bytes(root / relative, content)
    name = entry or f"{document['type']}.json"
    entry_path = write_json(root / name, document)
    write_lock(entry_path)
    return entry_path


def read_lock(folder: Path) -> dict[str, Any]:
    """The lock in a folder, as a mapping."""
    return json.loads((folder / LOCK_NAME).read_text(encoding="utf-8"))


def declared(report: Any, item_id: str) -> Mapping[str, Any]:
    """The line of a load report's `declared` list that has this id."""
    return next(item for item in report.declared if item["id"] == item_id)
