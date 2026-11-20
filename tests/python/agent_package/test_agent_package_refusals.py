# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent package loads: what is refused, what stops a load, and what is only reported."""

import json
import shutil
import sys
from pathlib import Path

import pytest
from nemo_fabric import (
    AgentPackageReport,
    Fabric,
    FabricConfig,
    FabricConfigError,
    load_agent_package,
)

from _utils.agent_packages import (
    LOCK_NAME,
    agent_document,
    definition_document,
    read_lock,
    write_bytes,
    write_json,
    write_lock,
    write_package,
)

SKILL = "$/skills/release-notes"


def append_byte(package, relative):
    path = package / relative
    path.write_bytes(path.read_bytes() + b"\n")


def change_description(package):
    path = package / "agent.json"
    path.write_bytes(path.read_bytes().replace(b"Drafts release", b"Crafts release"))


def add_file(package):
    (package / "skills" / "release-notes" / "extra.md").write_bytes(b"extra\n")


def delete_skill(package):
    shutil.rmtree(package / "skills" / "release-notes")


def delete_lock(package):
    (package / LOCK_NAME).unlink()


def remove_lock_entry(package):
    lock = read_lock(package)
    lock["entries"] = [entry for entry in lock["entries"] if entry["address"] != SKILL]
    for entry in lock["entries"]:
        entry["dependencies"] = [
            address for address in entry["dependencies"] if address != SKILL
        ]
    write_json(package / LOCK_NAME, lock)


def add_lock_entry(package):
    lock = read_lock(package)
    skill = next(entry for entry in lock["entries"] if entry["address"] == SKILL)
    extra = {
        **skill,
        "address": "$/skills/short-form",
        "source": {"type": "local", "url": "./skills/short-form"},
    }
    lock["entries"].append(extra)
    root = next(entry for entry in lock["entries"] if entry["address"] == "$")
    root["dependencies"] = sorted([*root["dependencies"], extra["address"]])
    write_json(package / LOCK_NAME, lock)


def write_duplicate_key(package):
    path = package / "agent.json"
    path.write_bytes(path.read_bytes().replace(b"{\n", b'{\n  "type": "agent",\n', 1))


@pytest.mark.parametrize(
    "change,code,subject",
    [
        pytest.param(
            lambda package: append_byte(package, "skills/release-notes/SKILL.md"),
            "package_digest_mismatch",
            SKILL,
            id="tampered-part",
        ),
        pytest.param(
            change_description, "package_digest_mismatch", "$", id="tampered-entry"
        ),
        pytest.param(add_file, "package_digest_mismatch", SKILL, id="extra-file"),
        pytest.param(delete_skill, "package_part_missing", SKILL, id="part-deleted"),
        pytest.param(
            remove_lock_entry,
            "package_lock_entry_missing",
            SKILL,
            id="lock-entry-removed",
        ),
        pytest.param(
            add_lock_entry,
            "package_lock_entry_extra",
            "$/skills/short-form",
            id="lock-entry-added",
        ),
    ],
)
def test_changed_package_is_refused_by_code_and_subject(
    package_dir, shim_config, hermes_shim_agent_dir, change, code, subject
):
    change(package_dir)

    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir, shim_config, base_dir=hermes_shim_agent_dir)

    assert raised.value.code == code
    assert raised.value.details["subject"] == subject
    assert raised.value.stage == "agent_package"
    # The same refusal with no config: the check does not depend on the mapping.
    with pytest.raises(FabricConfigError) as verified:
        load_agent_package(package_dir)
    assert verified.value.code == code


def test_digest_mismatch_message_gives_both_digests_and_the_path(package_dir):
    lock = read_lock(package_dir)
    pinned = next(entry for entry in lock["entries"] if entry["address"] == SKILL)
    append_byte(package_dir, "skills/release-notes/SKILL.md")

    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir)

    message = str(raised.value)
    assert pinned["digest"] in message
    assert message.count("sha256:") == 2
    assert "release-notes" in message


def test_ds_store_file_in_a_pinned_folder_does_not_change_its_digest(
    package_dir, agent, package_files
):
    before = load_agent_package(package_dir)
    lock = read_lock(package_dir)
    skill = package_dir / "skills" / "release-notes"
    # macOS Finder leaves this file in a folder it has shown.
    write_bytes(skill / ".DS_Store", b"\x00\x00\x00\x01Bud1")
    write_bytes(skill / "references" / ".DS_Store", b"Bud1")

    after = load_agent_package(package_dir)

    assert after.to_mapping() == before.to_mapping()
    # A lock written with the file there pins the same digests.
    write_package(package_dir, agent, package_files)
    assert read_lock(package_dir) == lock


@pytest.mark.parametrize("name", [".ds_store", ".DS_Store.bak", "Thumbs.db"])
def test_file_with_any_other_name_in_a_pinned_folder_changes_its_digest(
    package_dir, name
):
    write_bytes(package_dir / "skills" / "release-notes" / name, b"Bud1")

    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir)

    assert raised.value.code == "package_digest_mismatch"
    assert raised.value.details["subject"] == SKILL


def test_part_that_is_one_file_named_ds_store_is_refused_as_missing(
    package_dir, agent, package_files
):
    # A file of that name is never part of a package, whatever the lock pins.
    agent["policies"] = {"finder": {"source": "./policies/.DS_Store"}}
    package_files["policies/.DS_Store"] = "permit(principal, action, resource);\n"
    write_package(package_dir, agent, package_files)
    assert "$/policies/finder" in {
        entry["address"] for entry in read_lock(package_dir)["entries"]
    }

    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir)

    assert raised.value.code == "package_part_missing"
    assert raised.value.details["subject"] == "$/policies/finder"


@pytest.mark.parametrize(
    "change,code",
    [
        pytest.param(delete_lock, "package_lock_missing", id="no-lock"),
        pytest.param(
            write_duplicate_key, "package_json_duplicate_key", id="duplicate-key"
        ),
        pytest.param(
            lambda package: (package / "agent.json").unlink(),
            "package_entry_missing",
            id="no-entry-document",
        ),
    ],
)
def test_unreadable_package_is_refused(package_dir, change, code):
    change(package_dir)

    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir)

    assert raised.value.code == code


def test_missing_package_folder_is_refused(tmp_path):
    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(tmp_path / "nowhere")

    assert raised.value.code == "package_not_found"


@pytest.mark.skipif(
    sys.platform == "win32", reason="Creating a symbolic link needs a privilege"
)
def test_link_in_a_part_is_refused(package_dir):
    link = package_dir / "skills" / "release-notes" / "linked.md"
    link.symlink_to(package_dir / "agent.json")

    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir)

    assert raised.value.code == "package_symlink_refused"


@pytest.mark.parametrize(
    "declared,given,code",
    [
        pytest.param({"type": "number"}, {}, "package_variable_missing", id="missing"),
        pytest.param(
            {"type": "number", "default": 600},
            {"timeout": "long"},
            "package_variable_type",
            id="wrong-type",
        ),
        pytest.param(
            {"type": "number", "default": 600, "enum": [300, 600]},
            {"timeout": 100},
            "package_variable_not_allowed",
            id="outside-the-allowed-values",
        ),
    ],
)
def test_variable_value_is_refused_by_name(
    tmp_path, agent, package_files, declared, given, code
):
    agent["variables"]["timeout"] = declared
    write_package(tmp_path / "package", agent, package_files)

    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(tmp_path / "package", variables=given)

    assert raised.value.code == code
    assert raised.value.details["subject"] == "timeout"


def test_value_for_an_undeclared_variable_is_refused_by_name(package_dir):
    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir, variables={"nope": 1})

    assert raised.value.code == "package_variable_undeclared"
    assert raised.value.details["subject"] == "nope"


def test_variable_values_and_their_origin_are_reported(package_dir):
    default = load_agent_package(package_dir)
    given = load_agent_package(package_dir, variables={"timeout": 300})

    assert default.variables == (
        {"name": "timeout", "type": "number", "value": 600, "origin": "default"},
    )
    assert given.variables == (
        {"name": "timeout", "type": "number", "value": 300, "origin": "start"},
    )


@pytest.mark.parametrize(
    "members,code,subject",
    [
        pytest.param(
            {"skills": {"release-notes": {"source": "./skills/short-form"}}},
            "package_definition_name_in_use",
            "skills.release-notes",
            id="name-in-use",
        ),
        pytest.param(
            {"model": {"source": "test::other-model"}},
            "package_schema_invalid",
            "variants/short/agent-definition.json#",
            id="names-a-model",
        ),
    ],
)
def test_definition_that_replaces_is_refused(package_dir, members, code, subject):
    entry = "variants/short/agent-definition.json"
    write_package(
        package_dir,
        definition_document("release-notes-short", "./../../agent.json", **members),
        {"variants/short/skills/short-form/SKILL.md": "# Short form\n"},
        entry=entry,
    )

    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir, entry=entry)

    assert raised.value.code == code
    assert raised.value.details["subject"] == subject


def test_config_for_another_agent_is_refused(
    package_dir, shim_config, hermes_shim_agent_dir
):
    other = FabricConfig.from_mapping(
        {**shim_config.to_mapping(), "metadata": {"name": "another-agent"}}
    )

    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir, other, base_dir=hermes_shim_agent_dir)

    assert raised.value.code == "package_config_conflict"
    assert raised.value.details["subject"] == "metadata.name"


def test_config_mistake_is_raised_as_plan_raises_it(
    package_dir, shim_config, hermes_shim_agent_dir
):
    unknown = FabricConfig.from_mapping(
        {
            **shim_config.to_mapping(),
            "harness": {"adapter_id": "test.fabric.no_such_adapter"},
        }
    )

    with pytest.raises(FabricConfigError) as planned:
        Fabric().plan(unknown, base_dir=hermes_shim_agent_dir)
    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir, unknown, base_dir=hermes_shim_agent_dir)

    # A mistake in the config is not a package refusal: no stage and no code.
    assert raised.value.stage is None
    assert raised.value.code is None
    assert str(raised.value) == str(planned.value)


@pytest.mark.parametrize(
    "options,subject",
    [
        pytest.param({"if_unsupported": "ignore"}, "options", id="bad-setting"),
        pytest.param({"facts": {"ram": "64gb"}}, "facts.ram", id="unknown-fact"),
        pytest.param(
            {"facts": {"network": "Internal"}}, "facts.network", id="bad-network-fact"
        ),
        pytest.param(
            {"facts": {"memory": "16GB"}}, "facts.memory", id="bad-memory-fact"
        ),
        pytest.param(
            {"facts": {"model": {"context": 128000}}},
            "facts.model.context",
            id="unknown-model-fact-member",
        ),
        pytest.param(
            {"facts": {"gpus": [{"interface": {"cuda": "12.4"}}]}},
            "facts.gpus.0.interface.cuda",
            id="bad-gpu-interface-fact",
        ),
        pytest.param({"facts": ["cpus"]}, "facts", id="facts-not-a-mapping"),
        pytest.param({"variables": 5}, "variables", id="variables-not-a-mapping"),
        pytest.param({"entry": 5}, "entry", id="entry-not-a-path"),
        pytest.param(
            {"expected_lock_sha256": "abc"},
            "expected_lock_sha256",
            id="lock-digest-not-a-digest",
        ),
    ],
)
def test_invalid_options_are_refused(package_dir, options, subject):
    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir, **options)

    assert raised.value.code == "package_options_invalid"
    assert raised.value.details["subject"] == subject


def test_config_mapping_is_not_accepted(package_dir, shim_config):
    with pytest.raises(FabricConfigError, match="FabricConfig.from_mapping"):
        load_agent_package(package_dir, shim_config.to_mapping())


def test_report_members_are_the_members_the_schema_lists(repo_root):
    schema = json.loads(
        (repo_root / "schemas/sdk/agent-package-report.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert AgentPackageReport._fields == set(schema["properties"])


def test_required_thing_that_is_not_mapped_stops_or_warns(
    tmp_path, agent, package_files, shim_config, hermes_shim_agent_dir
):
    agent["hooks"] = {"tool-guard": {"source": "./hooks/tool-guard"}}
    package = tmp_path / "package"
    write_package(package, agent, package_files)

    stopped = load_agent_package(package, shim_config, base_dir=hermes_shim_agent_dir)
    warned = load_agent_package(
        package, shim_config, base_dir=hermes_shim_agent_dir, if_unsupported="warn"
    )

    assert stopped.outcome == "stopped"
    assert stopped.stopped_by == ("agent:/hooks/tool-guard",)
    assert stopped.config is None
    assert stopped.config_sha256 is None
    with pytest.raises(FabricConfigError) as raised:
        stopped.to_config()
    assert raised.value.code == "package_requirements_unmet"
    assert raised.value.details["stopped_by"] == ["agent:/hooks/tool-guard"]
    assert 'if_unsupported="warn"' in str(raised.value)

    assert warned.outcome == "ready"
    assert warned.stopped_by == ()
    hook = next(
        item for item in warned.declared if item["id"] == "agent:/hooks/tool-guard"
    )
    assert hook["status"] == "not_mapped"
    # The test adapter declares no key for hooks.
    assert hook["gap"] == "adapter_does_not_accept"
    assert hook["effect"] == "warned"
    assert hook["required"] is True
    assert isinstance(warned.to_config(), FabricConfig)


def test_optional_reference_left_out_of_the_lock_is_reported_and_never_stops(
    tmp_path, agent, package_files, shim_config, hermes_shim_agent_dir
):
    agent["agents"] = {
        "translator": {
            "source": "a2a::https://agents.example.com/translator",
            "version": "2.1.0",
            "optional": True,
        }
    }
    package = tmp_path / "package"
    write_package(package, agent, package_files)
    assert all(
        entry["address"] != "$/agents/translator"
        for entry in read_lock(package)["entries"]
    )

    report = load_agent_package(package, shim_config, base_dir=hermes_shim_agent_dir)

    assert report.outcome == "ready"
    translator = next(
        item for item in report.declared if item["id"] == "agent:/agents/translator"
    )
    assert translator["status"] == "not_mapped"
    assert translator["gap"] == "not_resolved"
    assert translator["effect"] == "none"
    assert translator["required"] is False


@pytest.mark.parametrize(
    "source",
    [
        "test::test-model",
        "ollama::llama3.2:3b",
        "hf::meta-llama/Llama-3.3-70B-Instruct",
        "Ollama::llama3",
        "registry.example.com/models/m",
    ],
)
def test_document_model_is_reported_and_never_written_into_the_config(
    tmp_path, agent, package_files, shim_config, hermes_shim_agent_dir, source
):
    agent["model"] = {"source": source, "parameters": {"temperature": 0.2}}
    package = tmp_path / "package"
    write_package(package, agent, package_files)

    report = load_agent_package(package, shim_config, base_dir=hermes_shim_agent_dir)

    # A catalog name is not a provider. The caller's config chooses the model,
    # and this config names none, so no model is configured at all.
    assert report.outcome == "ready"
    assert report.model == {
        "declared_source": source,
        "provider": None,
        "model": None,
        "origin": "harness_default",
        "parameters_applied": (),
        "parameters_not_applied": ("temperature",),
        "called": (),
    }
    for item_id in ("agent:/model/source", "agent:/model/parameters/temperature"):
        item = next(item for item in report.declared if item["id"] == item_id)
        assert item["status"] == "not_mapped"
        assert item["gap"] == "model_not_configured"
        assert item["effect"] == "none"
        assert item["mapped_to"] == ()
    assert "models" not in report.to_config().to_mapping()


def test_config_model_wins_over_the_document_model(
    package_dir, shim_config, hermes_shim_agent_dir
):
    config = FabricConfig.from_mapping(
        {
            **shim_config.to_mapping(),
            "models": {"default": {"provider": "test", "model": "served-model"}},
        }
    )

    report = load_agent_package(package_dir, config, base_dir=hermes_shim_agent_dir)

    assert report.model["origin"] == "caller"
    assert report.model["model"] == "served-model"
    assert report.to_config().models["default"].model == "served-model"


def test_load_without_a_config_verifies_and_reports_only(package_dir):
    report = load_agent_package(package_dir)

    assert isinstance(report, AgentPackageReport)
    assert report.outcome == "verified_only"
    assert report.adapter is None
    assert report.config is None
    assert report.model["origin"] == "not_evaluated"
    assert {item["status"] for item in report.declared} == {"not_evaluated"}
    assert report.entry["name"] == "release-notes-agent"
    assert report.entry["kind"] == "agent"
    assert report.agent == report.entry
    with pytest.raises(FabricConfigError) as raised:
        report.to_config()
    assert raised.value.code == "package_config_required"


def test_digests_report_every_lock_entry_for_drift_detection_and_mounting(package_dir):
    lock = read_lock(package_dir)
    pinned = {entry["address"]: entry["digest"] for entry in lock["entries"]}

    first = load_agent_package(package_dir)
    second = load_agent_package(package_dir)

    assert {check["address"]: check["digest"] for check in first.digests} == pinned
    assert {check["result"] for check in first.digests} == {"verified"}
    skill = next(check for check in first.digests if check["address"] == SKILL)
    assert skill["digest_kind"] == "folder"
    assert skill["entry_type"] == "skill"
    assert skill["source_type"] == "local"
    assert Path(skill["path"]) == (package_dir / "skills" / "release-notes").resolve()
    entry = next(check for check in first.digests if check["address"] == "$")
    assert entry["digest_kind"] == "file"
    assert entry["digest"] == first.entry["digest"]
    assert first.lock["entries"] == len(pinned)
    assert len(first.lock["sha256"]) == 64
    # The same package gives the same values.
    assert first.to_mapping() == second.to_mapping()


def test_expected_lock_digest_anchors_the_lock(package_dir):
    recorded = load_agent_package(package_dir).lock["sha256"]

    # The digest the caller recorded is accepted, with or without its prefix.
    load_agent_package(package_dir, expected_lock_sha256=recorded)
    load_agent_package(package_dir, expected_lock_sha256=f"sha256:{recorded}")

    # Someone who can write a part can write the lock beside it.
    append_byte(package_dir, "skills/release-notes/SKILL.md")
    write_lock(package_dir / "agent.json")

    # The folder agrees with its own lock again, so the plain check passes.
    rewritten = load_agent_package(package_dir)
    assert rewritten.lock["sha256"] != recorded
    assert {check["result"] for check in rewritten.digests} == {"verified"}
    # The recorded digest catches the swap, before anything else is read.
    with pytest.raises(FabricConfigError) as raised:
        load_agent_package(package_dir, expected_lock_sha256=recorded)
    assert raised.value.code == "package_lock_digest_mismatch"
    assert raised.value.details["subject"] == LOCK_NAME
    assert recorded in str(raised.value)


def test_skill_that_is_one_file_is_not_mapped(
    tmp_path, agent, package_files, shim_config, hermes_shim_agent_dir
):
    agent["skills"]["review"] = {"source": "./skills/review.md"}
    package_files["skills/review.md"] = "# Review\n"
    package = tmp_path / "package"
    write_package(package, agent, package_files)

    stopped = load_agent_package(package, shim_config, base_dir=hermes_shim_agent_dir)
    warned = load_agent_package(
        package, shim_config, base_dir=hermes_shim_agent_dir, if_unsupported="warn"
    )

    # The format accepts one file as a skill, so the package is valid. The
    # adapters take a folder that holds SKILL.md, so the file is not handed on.
    assert stopped.outcome == "stopped"
    assert stopped.stopped_by == ("agent:/skills/review",)
    review = next(
        item for item in warned.declared if item["id"] == "agent:/skills/review"
    )
    assert review["status"] == "not_mapped"
    assert review["gap"] == "unsupported_shape"
    assert review["effect"] == "warned"
    assert [Path(path) for path in warned.to_config().skills.paths] == [
        (package / "skills" / "release-notes").resolve()
    ]


def test_skill_folder_name_is_used_once(
    tmp_path, agent, package_files, shim_config, hermes_shim_agent_dir
):
    agent["skills"] = {
        "left": {"source": "./left/review"},
        "right": {"source": "./right/review", "optional": True},
    }
    package_files["left/review/SKILL.md"] = "# Left\n"
    package_files["right/review/SKILL.md"] = "# Right\n"
    package = tmp_path / "package"
    write_package(package, agent, package_files)

    report = load_agent_package(package, shim_config, base_dir=hermes_shim_agent_dir)

    # A harness names a skill by its folder, so one folder name maps once.
    assert report.outcome == "ready"
    by_id = {item["id"]: item for item in report.declared}
    assert by_id["agent:/skills/left"]["status"] == "mapped"
    assert "`review`" in by_id["agent:/skills/left"]["detail"]
    right = by_id["agent:/skills/right"]
    assert right["status"] == "not_mapped"
    assert right["gap"] == "key_clash"
    assert right["effect"] == "none"
    assert [Path(path) for path in report.to_config().skills.paths] == [
        (package / "left" / "review").resolve()
    ]


def test_two_entries_for_one_skill_folder_map_once(
    tmp_path, agent, package_files, shim_config, hermes_shim_agent_dir
):
    agent["skills"] = {
        "first": {"source": "./skills/release-notes"},
        "second": {"source": "./skills/release-notes", "optional": True},
    }
    package = tmp_path / "package"
    write_package(package, agent, package_files)
    # One part reached by two paths has one lock entry, at the first path.
    lock = read_lock(package)
    lock["entries"] = [
        entry for entry in lock["entries"] if entry["address"] != "$/skills/second"
    ]
    root = next(entry for entry in lock["entries"] if entry["address"] == "$")
    root["dependencies"].remove("$/skills/second")
    write_json(package / LOCK_NAME, lock)

    report = load_agent_package(package, shim_config, base_dir=hermes_shim_agent_dir)

    assert report.outcome == "ready"
    by_id = {item["id"]: item for item in report.declared}
    assert by_id["agent:/skills/first"]["status"] == "mapped"
    assert by_id["agent:/skills/second"]["status"] == "not_mapped"
    assert by_id["agent:/skills/second"]["gap"] == "key_clash"
    assert len(report.to_config().skills.paths) == 1


@pytest.mark.parametrize(
    "facts, outcome, status, effect",
    [
        # NeMo Fabric installs no package and looks for none, so a required
        # one does not start by default until a fact tells about its registry.
        (None, "stopped", "not_mapped", "stopped"),
        ({"packages": {"npm": {}}}, "stopped", "not_mapped", "stopped"),
        ({"packages": {"pypi": {}}}, "stopped", "not_met", "stopped"),
        (
            {"packages": {"pypi": {"faster-whisper": "1.2.0"}}},
            "stopped",
            "not_met",
            "stopped",
        ),
        (
            {"packages": {"pypi": {"faster-whisper": None}}},
            "ready",
            "could_not_be_checked",
            "warned",
        ),
        # A name on PyPI compares after its normalising.
        (
            {"packages": {"pypi": {"Faster_Whisper": "1.2.1"}}},
            "ready",
            "met_here",
            "none",
        ),
    ],
    ids=[
        "no_fact",
        "registry_not_given",
        "not_installed",
        "another_version",
        "version_not_known",
        "installed",
    ],
)
def test_required_package_is_compared_with_the_packages_fact(
    tmp_path,
    agent,
    package_files,
    shim_config,
    hermes_shim_agent_dir,
    facts,
    outcome,
    status,
    effect,
):
    agent["packages"] = {
        "whisper": {"source": "pypi::faster-whisper", "version": "1.2.1"}
    }
    package = tmp_path / "package"
    write_package(package, agent, package_files)

    report = load_agent_package(
        package, shim_config, base_dir=hermes_shim_agent_dir, facts=facts
    )

    assert report.outcome == outcome
    whisper = next(
        item for item in report.declared if item["id"] == "agent:/packages/whisper"
    )
    assert whisper["status"] == status
    assert whisper["gap"] == ("no_fabric_field" if status == "not_mapped" else None)
    assert whisper["effect"] == effect
    assert whisper["mapped_to"] == ()
    # The pin is in the report for a caller that installs the package itself.
    pin = next(
        check for check in report.digests if check["address"] == "$/packages/whisper"
    )
    assert pin["version"] == "1.2.1"
    assert pin["result"] == "not_verifiable"


def test_called_agent_that_cannot_start_is_noted_and_does_not_refuse_its_caller(
    tmp_path, agent, package_files, shim_config, hermes_shim_agent_dir
):
    agent["agents"] = {
        "helper": {"source": "./agents/helper/agent.json", "optional": True}
    }
    package_files["agents/helper/agent.json"] = (
        json.dumps(
            agent_document(
                "helper",
                description="Helps with one task.",
                instructions={"var": "tone"},
                variables={"tone": {"type": "string"}},
            ),
            indent=2,
        )
        + "\n"
    )
    package = tmp_path / "package"
    write_package(package, agent, package_files)

    report = load_agent_package(package, shim_config, base_dir=hermes_shim_agent_dir)

    # The helper's variable has no value, so the helper could not be started.
    # It is optional, and its caller loads either way.
    assert report.outcome == "ready"
    note = next(
        note for note in report.notes if note["code"] == "called_agent_cannot_start"
    )
    assert note["subject"] == "$/agents/helper"
    assert "tone" in note["text"]
    helper = next(
        item for item in report.declared if item["id"] == "agent:/agents/helper"
    )
    assert helper["status"] == "not_mapped"
    assert helper["effect"] == "none"


def test_gpu_fact_with_no_interface_is_not_counted_as_not_met(
    tmp_path, agent, package_files
):
    agent["runtime"] = {
        "compute": {
            "gpu": {"interface": {"cuda": {"version": ">= 12.0"}}, "memory": "16gb"}
        }
    }
    package = tmp_path / "package"
    write_package(package, agent, package_files)

    # The shape the machine probe gives: memory and nothing else.
    probed = load_agent_package(package, facts={"gpus": [{"memory": "24gb"}]})
    stated = load_agent_package(
        package,
        facts={
            "gpus": [
                {"memory": "24gb", "interface": {"cuda": {"version": "12.4"}}}
            ]
        },
    )

    def gpu(report):
        return next(
            item
            for item in report.declared
            if item["id"] == "agent:/runtime/compute/gpu"
        )

    assert gpu(probed)["status"] == "could_not_be_checked"
    assert gpu(stated)["status"] == "met_here"


@pytest.mark.skipif(
    sys.platform == "win32", reason="A carried connector needs /bin/sh"
)
def test_connector_keys_that_are_not_read_are_named(
    tmp_path, agent, package_files, shim_config, hermes_shim_agent_dir
):
    agent["connectors"] = {"changelog": {"source": "./connectors/changelog"}}
    package_files["connectors/changelog/mcp.json"] = (
        json.dumps(
            {
                "mcpServers": {
                    "main": {
                        "type": "stdio",
                        "command": "python3",
                        "args": ["server.py"],
                        "cwd": ".",
                        "disabled": True,
                        "alwaysAllow": ["list_changes"],
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )
    package = tmp_path / "package"
    write_package(package, agent, package_files)

    report = load_agent_package(package, shim_config, base_dir=hermes_shim_agent_dir)

    connector = next(
        item for item in report.declared if item["id"] == "agent:/connectors/changelog"
    )
    assert connector["status"] == "mapped"
    assert "alwaysAllow, disabled" in connector["detail"]
    note = next(
        note for note in report.notes if note["code"] == "connector_key_not_read"
    )
    assert note["subject"] == "mcp.servers.changelog"
    # `cwd` of `.` is the connector's own folder.
    server = report.to_config().mcp.servers["changelog"]
    assert Path(server.args[3]) == (package / "connectors" / "changelog").resolve()
