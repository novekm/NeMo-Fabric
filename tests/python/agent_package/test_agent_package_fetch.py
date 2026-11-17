# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent package fetch: content a lock pins is placed, checked and loaded."""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric import (
    AgentPackageFetchReport,
    FabricConfigError,
    fetch_agent_package,
    fetch_agent_package_from_git,
    load_agent_package,
)

from _utils.agent_packages import (
    LOCK_NAME,
    agent_document,
    file_digest,
    folder_digest,
    read_lock,
    schema_id,
    write_bytes,
    write_json,
)

STORYBOARD = """---
name: storyboard
description: Plan the shots of a video.
---

List every shot before rendering.
"""
CHECKLIST = """---
name: checklist
description: Review a change.
---

Report every item that fails.
"""
MISSING_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def git(folder: Path, *arguments: str) -> str:
    done = subprocess.run(
        [
            "git",
            "-C",
            str(folder),
            "-c",
            "user.name=Tests",
            "-c",
            "user.email=tests@example.com",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def write_lock(package: Path, entries: list[dict[str, Any]]) -> None:
    write_json(
        package / LOCK_NAME,
        {
            "$schema": schema_id("lock"),
            "type": "lock",
            "generator": "tests/1",
            "entries": entries,
        },
    )


def local_entry(package: Path, name: str, entry_type: str, dependencies: list[str]):
    return {
        "address": "$",
        "type": entry_type,
        "source": {"type": "local", "url": f"./{name}"},
        "version": "local",
        "digest": file_digest(package / name),
        "dependencies": dependencies,
    }


@pytest.fixture(name="repository")
def repository_fixture(tmp_path: Path) -> dict[str, Any]:
    """A bare git repository of a media team, reached as ``file://``.

    It holds a ``storyboard`` skill, a ``video-producer`` agent that keeps a
    skill of its own in its folder, and a ``checklist`` skill.
    """
    work = tmp_path / "media-team-parts"
    write_bytes(work / "skills" / "storyboard" / "SKILL.md", STORYBOARD)
    write_bytes(work / "skills" / "checklist" / "SKILL.md", CHECKLIST)
    write_json(
        work / "video-producer" / "agent.json",
        agent_document(
            "video-producer",
            description="Produces a short video from a brief.",
            skills={"storyboard": {"source": "./skills/storyboard"}},
        ),
    )
    write_bytes(
        work / "video-producer" / "skills" / "storyboard" / "SKILL.md", STORYBOARD
    )
    git(work, "init", "--quiet")
    git(work, "add", "--all")
    git(work, "commit", "--quiet", "--message", "Add the shared parts")
    bare = tmp_path / "media-team-parts.git"
    git(work, "clone", "--quiet", "--bare", ".", str(bare))
    return {
        "url": f"file://{bare}",
        "commit": git(work, "rev-parse", "HEAD"),
        "work": work,
    }


def git_entry(
    repository: dict[str, Any],
    address: str,
    entry_type: str,
    subpath: str,
    dependencies: tuple[str, ...] = (),
) -> dict[str, Any]:
    """The lock entry for a folder or an agent of the repository, written by hand."""
    target = repository["work"] / subpath
    if entry_type == "agent":
        target = target / "agent.json"
    return {
        "address": address,
        "type": entry_type,
        "source": {"type": "git", "url": repository["url"], "subpath": subpath},
        "version": repository["commit"],
        "commit": repository["commit"],
        "digest": folder_digest(target) if target.is_dir() else file_digest(target),
        "dependencies": list(dependencies),
    }


def git_reference(repository: dict[str, Any], subpath: str) -> dict[str, str]:
    return {
        "source": f"git::{repository['url']}//{subpath}",
        "version": repository["commit"],
    }


@pytest.fixture(name="video_producer")
def video_producer_fixture(tmp_path: Path, repository: dict[str, Any]) -> Path:
    """A video producer whose ``storyboard`` skill is pinned from git.

    The folder holds the agent and its lock, and no skill.
    """
    package = tmp_path / "video-producer"
    write_json(
        package / "agent.json",
        agent_document(
            "video-producer",
            description="Produces a short video from a brief.",
            skills={"storyboard": git_reference(repository, "skills/storyboard")},
        ),
    )
    write_lock(
        package,
        [
            local_entry(package, "agent.json", "agent", ["$/skills/storyboard"]),
            git_entry(repository, "$/skills/storyboard", "skill", "skills/storyboard"),
        ],
    )
    return package


@pytest.fixture(name="short_definition")
def short_definition_fixture(tmp_path: Path, repository: dict[str, Any]) -> Path:
    """A definition whose agent is shared from git.

    The folder holds the definition and its lock, and no agent.
    """
    package = tmp_path / "video-producer-short"
    write_json(
        package / "agent-definition.json",
        {
            "$schema": schema_id("agent-definition"),
            "type": "agent-definition",
            "name": "video-producer-short",
            "agent": git_reference(repository, "video-producer"),
            "limits": {"max_turns": 20},
        },
    )
    write_lock(
        package,
        [
            local_entry(
                package, "agent-definition.json", "agent-definition", ["$/agent"]
            ),
            git_entry(
                repository,
                "$/agent",
                "agent",
                "video-producer",
                ("$/agent/skills/storyboard",),
            ),
            git_entry(
                repository,
                "$/agent/skills/storyboard",
                "skill",
                "video-producer/skills/storyboard",
            ),
        ],
    )
    return package


def results(package: Path) -> dict[str, str]:
    digests = load_agent_package(package).digests
    return {row["address"]: row["result"] for row in digests}


def test_a_skill_pinned_from_git_is_fetched_and_then_loads(
    tmp_path: Path, repository: dict[str, Any], video_producer: Path
):
    assert results(video_producer) == {
        "$": "verified",
        "$/skills/storyboard": "absent",
    }
    out_dir = tmp_path / "build" / "video-producer"

    report = fetch_agent_package(video_producer, out_dir)

    assert isinstance(report, AgentPackageFetchReport)
    assert report.package_dir == out_dir
    assert report.not_fetched == ()
    assert [dict(row) for row in report.fetched] == [
        {
            "address": "$/skills/storyboard",
            "entry_type": "skill",
            "source_url": repository["url"],
            "source_subpath": "skills/storyboard",
            "commit": repository["commit"],
            "digest": folder_digest(repository["work"] / "skills" / "storyboard"),
            "path": str(out_dir / "skills" / "storyboard"),
        }
    ]
    loaded = load_agent_package(
        report.package_dir, expected_lock_sha256=report.lock["sha256"]
    )
    assert all(row["result"] == "verified" for row in loaded.digests)
    # The package that was given is not written to, and nothing is left beside
    # the result.
    assert sorted(path.name for path in video_producer.iterdir()) == [
        "agent.json",
        LOCK_NAME,
    ]
    assert [path.name for path in out_dir.parent.iterdir()] == ["video-producer"]


def test_a_definition_whose_agent_is_in_git_loads_after_the_fetch(
    tmp_path: Path, short_definition: Path
):
    with pytest.raises(FabricConfigError) as refused:
        load_agent_package(short_definition)
    assert refused.value.code == "package_agent_unavailable"

    report = fetch_agent_package(short_definition, tmp_path / "build")

    assert [row["address"] for row in report.fetched] == [
        "$/agent",
        "$/agent/skills/storyboard",
    ]
    loaded = load_agent_package(report.package_dir)
    assert loaded.entry["name"] == "video-producer-short"
    assert loaded.agent["name"] == "video-producer"
    assert loaded.agent["path"] == str(tmp_path / "build" / "agent" / "agent.json")
    assert all(row["result"] == "verified" for row in loaded.digests)


@pytest.fixture(name="code_review_lead")
def code_review_lead_fixture(tmp_path: Path, repository: dict[str, Any]) -> Path:
    """A code review lead and its reviewer, who use one ``checklist`` skill.

    The reviewer is in a second repository. The folder holds the lead and its
    lock, a render that no entry pins, and a link to that render.
    """
    work = tmp_path / "code-review-team"
    write_json(
        work / "reviewer" / "agent.json",
        agent_document(
            "reviewer",
            description="Reviews a change against the checklist.",
            skills={"checklist": git_reference(repository, "skills/checklist")},
        ),
    )
    git(work, "init", "--quiet")
    git(work, "add", "--all")
    git(work, "commit", "--quiet", "--message", "Add the reviewer")
    bare = tmp_path / "code-review-team.git"
    git(work, "clone", "--quiet", "--bare", ".", str(bare))
    team = {
        "url": f"file://{bare}",
        "commit": git(work, "rev-parse", "HEAD"),
        "work": work,
    }
    # One entry pins the checklist, at the first path in sort order.
    shared = "$/agents/reviewer/skills/checklist"
    package = tmp_path / "code-review-lead"
    write_json(
        package / "agent.json",
        agent_document(
            "code-review-lead",
            description="Leads a code review.",
            agents={"reviewer": git_reference(team, "reviewer")},
            skills={"checklist": git_reference(repository, "skills/checklist")},
        ),
    )
    write_lock(
        package,
        [
            local_entry(package, "agent.json", "agent", ["$/agents/reviewer", shared]),
            git_entry(team, "$/agents/reviewer", "agent", "reviewer", (shared,)),
            git_entry(repository, shared, "skill", "skills/checklist"),
        ],
    )
    write_bytes(package / "renders" / "draft.mp4", b"a render, not a part\n")
    (package / "latest.mp4").symlink_to("renders/draft.mp4")
    return package


def test_a_skill_a_called_agent_from_git_also_uses_is_fetched_once(
    tmp_path: Path, code_review_lead: Path
):
    assert results(code_review_lead) == {
        "$": "verified",
        "$/agents/reviewer": "absent",
        "$/agents/reviewer/skills/checklist": "absent",
    }
    out_dir = tmp_path / "build"

    report = fetch_agent_package(code_review_lead, out_dir)

    assert [row["address"] for row in report.fetched] == [
        "$/agents/reviewer",
        "$/agents/reviewer/skills/checklist",
    ]
    assert set(results(out_dir).values()) == {"verified"}
    # Only what the lock pins is copied: not the render, and not the link to it.
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "agent.json",
        "agents",
        LOCK_NAME,
    ]


def test_a_load_report_leaves_the_password_out_of_an_address(video_producer: Path):
    with_password = "https://build:hunter2@git.example.com/media-team/parts.git"
    document = video_producer / "agent.json"
    lock = read_lock(video_producer)
    document.write_text(
        document.read_text().replace(lock["entries"][1]["source"]["url"], with_password)
    )
    lock["entries"][0]["digest"] = file_digest(document)
    lock["entries"][1]["source"]["url"] = with_password
    write_json(video_producer / LOCK_NAME, lock)

    report = load_agent_package(video_producer)

    assert report.digests[1]["source_url"] == (
        "https://build:***@git.example.com/media-team/parts.git"
    )
    assert "hunter2" not in json.dumps(report.to_dict(), default=str)


def pin_other_bytes(package: Path, repository: dict[str, Any]) -> Path:
    """Pin the digest the skill has with one more byte than the repository holds."""
    changed = write_bytes(package.parent / "changed" / "SKILL.md", STORYBOARD + "\n")
    lock = read_lock(package)
    lock["entries"][1]["digest"] = folder_digest(changed.parent)
    write_json(package / LOCK_NAME, lock)
    return package.parent / "build"


def pin_a_missing_commit(package: Path, repository: dict[str, Any]) -> Path:
    """Select a commit the repository does not have, in the document and in the lock."""
    document = package / "agent.json"
    document.write_text(
        document.read_text().replace(repository["commit"], MISSING_COMMIT)
    )
    lock = read_lock(package)
    lock["entries"][0]["digest"] = file_digest(document)
    lock["entries"][1].update(version=MISSING_COMMIT, commit=MISSING_COMMIT)
    write_json(package / LOCK_NAME, lock)
    return package.parent / "build"


def use_an_out_dir_that_holds_a_file(package: Path, repository: dict[str, Any]) -> Path:
    out_dir = package.parent / "build"
    write_bytes(out_dir / "notes.txt", "kept\n")
    return out_dir


def use_an_out_dir_inside_the_package(
    package: Path, repository: dict[str, Any]
) -> Path:
    return package / "build"


def name_the_package_folder_another_way(
    package: Path, repository: dict[str, Any]
) -> Path:
    return package.parent / "staging" / ".." / package.name / "build"


@pytest.mark.parametrize(
    "change,code,subject",
    [
        pytest.param(
            pin_other_bytes,
            "package_digest_mismatch",
            "$/skills/storyboard",
            id="changed_byte",
        ),
        pytest.param(
            pin_a_missing_commit,
            "package_fetch_failed",
            "$/skills/storyboard",
            id="missing_commit",
        ),
        pytest.param(
            use_an_out_dir_that_holds_a_file,
            "package_out_dir_in_use",
            None,
            id="out_dir_holds_a_file",
        ),
        pytest.param(
            use_an_out_dir_inside_the_package,
            "package_out_dir_in_use",
            None,
            id="out_dir_inside_the_package",
        ),
        pytest.param(
            name_the_package_folder_another_way,
            "package_out_dir_in_use",
            None,
            id="out_dir_inside_the_package_through_dots",
        ),
    ],
)
def test_a_fetch_that_is_refused_leaves_nothing(
    repository: dict[str, Any], video_producer: Path, change, code, subject
):
    out_dir = change(video_producer, repository)
    # The folder that would hold ``out_dir``, and the folder beside the package.
    folders = (out_dir.parent.resolve(), video_producer.parent)
    before = [sorted(path.name for path in folder.iterdir()) for folder in folders]

    with pytest.raises(FabricConfigError) as refused:
        fetch_agent_package(video_producer, out_dir)

    assert refused.value.stage == "agent_package"
    assert refused.value.code == code
    if subject is not None:
        assert refused.value.details["subject"] == subject
    after = [sorted(path.name for path in folder.iterdir()) for folder in folders]
    assert after == before
    assert not (out_dir / "agent.json").exists()


def test_the_lock_digest_the_caller_holds_is_checked_before_the_fetch(
    tmp_path: Path, video_producer: Path
):
    with pytest.raises(FabricConfigError) as refused:
        fetch_agent_package(
            video_producer, tmp_path / "build", expected_lock_sha256="0" * 64
        )

    assert refused.value.code == "package_lock_digest_mismatch"
    assert not (tmp_path / "build").exists()


def test_a_package_with_nothing_to_fetch_is_returned_as_it_is(
    tmp_path: Path, package_dir: Path
):
    report = fetch_agent_package(package_dir, tmp_path / "build")

    assert report.package_dir == package_dir
    assert report.lock["path"] == str(package_dir / LOCK_NAME)
    assert report.fetched == ()
    assert report.not_fetched == ()
    assert not (tmp_path / "build").exists()


def test_an_entry_that_is_not_a_path_is_refused_by_name(
    tmp_path: Path, package_dir: Path
):
    with pytest.raises(FabricConfigError) as refused:
        fetch_agent_package(package_dir, tmp_path / "build", entry=7)

    assert refused.value.code == "package_options_invalid"
    assert refused.value.details["subject"] == "entry"


@pytest.fixture(name="agents_source")
def agents_source_fixture(tmp_path: Path, repository: dict[str, Any]) -> str:
    """The git source of a package that is a folder in a second repository.

    The folder holds the agent and its lock. The lock pins the agent's skill
    from the first repository. The second repository is tagged ``v2.1.0``.
    """
    work = tmp_path / "media-team-agents"
    package = work / "video-producer"
    write_json(
        package / "agent.json",
        agent_document(
            "video-producer",
            description="Produces a short video from a brief.",
            skills={"storyboard": git_reference(repository, "skills/storyboard")},
        ),
    )
    write_lock(
        package,
        [
            local_entry(package, "agent.json", "agent", ["$/skills/storyboard"]),
            git_entry(repository, "$/skills/storyboard", "skill", "skills/storyboard"),
        ],
    )
    git(work, "init", "--quiet")
    git(work, "add", "--all")
    git(work, "commit", "--quiet", "--message", "Add the video producer")
    git(work, "tag", "v2.1.0")
    bare = tmp_path / "media-team-agents.git"
    git(work, "clone", "--quiet", "--bare", ".", str(bare))
    return f"git::file://{bare}//video-producer"


def test_a_package_that_is_itself_in_git_is_fetched_at_a_tag(
    tmp_path: Path, repository: dict[str, Any], agents_source: str
):
    out_dir = tmp_path / "build"

    report = fetch_agent_package_from_git(agents_source, "v2.1.0", out_dir)

    assert report.package_dir == out_dir
    package, skill = report.fetched
    assert package["address"] == "$"
    assert package["entry_type"] == "agent"
    assert package["source_subpath"] == "video-producer"
    assert len(package["commit"]) == 40
    assert package["path"] == str(out_dir)
    assert skill["address"] == "$/skills/storyboard"
    assert skill["commit"] == repository["commit"]
    loaded = load_agent_package(
        report.package_dir, expected_lock_sha256=report.lock["sha256"]
    )
    assert all(row["result"] == "verified" for row in loaded.digests)

    # The lock digest the caller holds is checked. A refused fetch leaves nothing.
    with pytest.raises(FabricConfigError) as refused:
        fetch_agent_package_from_git(
            agents_source,
            "v2.1.0",
            tmp_path / "other",
            expected_lock_sha256="0" * 64,
        )
    assert refused.value.code == "package_lock_digest_mismatch"
    assert not (tmp_path / "other").exists()


def fetch_a_git_source_as_a_folder(source: str, out_dir: Path):
    return fetch_agent_package(source, out_dir)


def fetch_a_git_source_given_as_a_path(source: str, out_dir: Path):
    # A path object folds the ``//`` before the folder into ``/``.
    return fetch_agent_package_from_git(Path(source), "v2.1.0", out_dir)


def fetch_at_a_ref_that_is_not_text(source: str, out_dir: Path):
    return fetch_agent_package_from_git(source, 210, out_dir)


@pytest.mark.parametrize(
    "fetch,subject",
    [
        pytest.param(fetch_a_git_source_as_a_folder, "package_dir", id="as_a_folder"),
        pytest.param(fetch_a_git_source_given_as_a_path, "source", id="as_a_path"),
        pytest.param(fetch_at_a_ref_that_is_not_text, "ref", id="ref_not_text"),
    ],
)
def test_a_git_source_in_the_wrong_place_is_refused_by_name(
    tmp_path: Path, agents_source: str, fetch, subject
):
    with pytest.raises(FabricConfigError) as refused:
        fetch(agents_source, tmp_path / "build")

    assert refused.value.code == "package_options_invalid"
    assert refused.value.details["subject"] == subject
    assert not (tmp_path / "build").exists()


PLUGIN_FILES = {
    ".claude-plugin/plugin.json": b'{"name": "remotion"}\n',
    "skills/remotion/SKILL.md": b"---\nname: remotion\n---\nRender with Remotion.\n",
}
TAR_LAYER = "application/vnd.oci.image.layer.v1.tar"


def write_blob(layout: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    write_bytes(layout / "blobs" / "sha256" / digest, content)
    return f"sha256:{digest}"


def write_artifact(layout: Path, layer: bytes, layer_type: str) -> str:
    """Write one artifact into a layout folder by hand.

    Returns the digest of its manifest, which is the digest a lock pins.
    """
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.empty.v1+json",
            "digest": write_blob(layout, b"{}"),
            "size": 2,
        },
        "layers": [
            {
                "mediaType": layer_type,
                "digest": write_blob(layout, layer),
                "size": len(layer),
            }
        ],
    }
    return write_blob(layout, json.dumps(manifest).encode("utf-8"))


@pytest.fixture(name="plugin_layer")
def plugin_layer_fixture() -> bytes:
    """The files of a plugin as a tar archive that the standard library writes."""
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, content in PLUGIN_FILES.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    return archive.getvalue()


def plugin_package(root: Path, digest: str) -> Path:
    """A video producer whose ``remotion`` plugin is pinned as an OCI artifact."""
    write_json(
        root / "agent.json",
        agent_document(
            "video-producer",
            description="Produces a short video from a brief.",
            plugins={
                "remotion": {
                    "source": "oci::registry.example.com/plugins/remotion",
                    "version": "1.2.0",
                }
            },
        ),
    )
    write_lock(
        root,
        [
            local_entry(root, "agent.json", "agent", ["$/plugins/remotion"]),
            {
                "address": "$/plugins/remotion",
                "type": "plugin",
                "source": {
                    "type": "oci",
                    "url": "registry.example.com/plugins/remotion",
                },
                "version": "1.2.0",
                "constraint": "1.2.0",
                "digest": digest,
                "dependencies": [],
            },
        ],
    )
    return root


def test_a_plugin_pinned_as_an_oci_artifact_is_unpacked_and_then_loads(
    tmp_path: Path, plugin_layer: bytes
):
    layout = tmp_path / "layout"
    digest = write_artifact(layout, plugin_layer, TAR_LAYER)
    package = plugin_package(tmp_path / "video-producer", digest)
    assert results(package)["$/plugins/remotion"] == "not_verifiable"
    left = fetch_agent_package(package, tmp_path / "build")
    assert left.package_dir == package
    assert [row["address"] for row in left.not_fetched] == ["$/plugins/remotion"]

    report = fetch_agent_package(package, tmp_path / "build", oci_layout=layout)

    plugin = tmp_path / "build" / "plugins" / "remotion"
    assert [dict(row) for row in report.fetched] == [
        {
            "address": "$/plugins/remotion",
            "entry_type": "plugin",
            "source_url": "registry.example.com/plugins/remotion",
            "source_subpath": None,
            "commit": None,
            "digest": digest,
            "path": str(plugin),
        }
    ]
    for name, content in PLUGIN_FILES.items():
        assert (plugin / name).read_bytes() == content
    loaded = load_agent_package(report.package_dir)
    row = next(row for row in loaded.digests if row["address"] == "$/plugins/remotion")
    assert row["result"] == "verified"
    assert row["path"] == str(plugin)

    # A load checks the files against the lock's digest again.
    (plugin / "skills" / "remotion" / "SKILL.md").write_bytes(b"Render anything.\n")
    with pytest.raises(FabricConfigError) as refused:
        load_agent_package(report.package_dir)
    assert refused.value.code == "package_digest_mismatch"
    assert refused.value.details["subject"] == "$/plugins/remotion"


def test_a_compressed_layer_is_refused_by_name(tmp_path: Path):
    layout = tmp_path / "layout"
    digest = write_artifact(layout, b"\x1f\x8b compressed", TAR_LAYER + "+gzip")
    package = plugin_package(tmp_path / "video-producer", digest)

    with pytest.raises(FabricConfigError) as refused:
        fetch_agent_package(package, tmp_path / "build", oci_layout=layout)

    assert refused.value.code == "package_file_unreadable"
    assert refused.value.details["subject"] == "$/plugins/remotion"
    assert "a compressed layer is not read" in str(refused.value)
    assert not (tmp_path / "build").exists()
