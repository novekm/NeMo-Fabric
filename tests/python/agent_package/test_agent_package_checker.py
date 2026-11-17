# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A checker program decides whether a loaded agent package may start."""

import json
import os
import sys
from pathlib import Path

import pytest
from nemo_fabric import (
    AgentPackageCheckerResult,
    FabricConfig,
    FabricConfigError,
    load_agent_package,
)

from _utils.agent_packages import write_package

if sys.platform == "win32":
    pytest.skip("A checker has not been run on Windows yet", allow_module_level=True)

ALLOW = {
    "decision": "allow",
    "rules": ["approved-hooks"],
    "checker": {"name": "test-checker", "version": "1"},
}
DENY = {
    "decision": "deny",
    "reasons": ["The skill release-notes is not on the approved list."],
    "rules": ["approved-skills"],
}
# The test checker sleeps this long when a limit must end it, in seconds.
A_DAY = 24 * 60 * 60


def test_checker_that_allows_leaves_the_load_ready_with_its_answer(
    package_dir, shim_config, hermes_shim_agent_dir, package_checker
):
    report = load_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        checker=package_checker("allow"),
    )

    assert report.outcome == "ready"
    assert report.checker_result == {
        "decision": "allow",
        "failure": None,
        "exit_code": 0,
        "detail": None,
        "answer": ALLOW,
    }
    assert report.to_config().metadata.name == "release-notes-agent"
    # The checker changes nothing else of the report.
    unchecked = load_agent_package(
        package_dir, shim_config, base_dir=hermes_shim_agent_dir
    )
    assert unchecked.checker_result is None
    assert {**report.to_mapping(), "checker_result": None} == unchecked.to_mapping()


def test_checker_result_is_typed_and_reads_by_attribute(
    package_dir, shim_config, hermes_shim_agent_dir, package_checker
):
    report = load_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        checker=package_checker("deny"),
    )

    result = report.checker_result
    assert isinstance(result, AgentPackageCheckerResult)
    assert (result.decision, result.failure, result.exit_code) == ("deny", None, 2)
    assert result.answer == DENY
    assert result["decision"] == "deny"


def test_checker_that_denies_gives_no_config_and_its_reasons(
    package_dir, shim_config, hermes_shim_agent_dir, package_checker
):
    report = load_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        checker=package_checker("deny"),
    )

    assert report.outcome == "denied"
    assert report.config is None
    assert len(report.config_sha256) == 64
    assert report.checker_result == {
        "decision": "deny",
        "failure": None,
        "exit_code": 2,
        "detail": None,
        "answer": DENY,
    }
    with pytest.raises(FabricConfigError) as caught:
        report.to_config()
    assert caught.value.stage == "agent_package"
    assert caught.value.code == "package_check_denied"
    assert caught.value.details == {"answer": DENY}


@pytest.mark.parametrize(
    ("answer", "failure", "exit_code", "kept"),
    [
        ("exit-3", "unexpected_exit", 3, ALLOW),
        ("no-output", "invalid_answer", 0, None),
        ("no-json", "invalid_answer", 0, None),
        ("two-objects", "invalid_answer", 0, None),
        ("deny-with-exit-0", "invalid_answer", 0, DENY),
        ("allow-with-exit-2", "invalid_answer", 2, ALLOW),
    ],
)
def test_checker_that_does_not_answer_as_it_must_is_a_deny(
    package_dir,
    shim_config,
    hermes_shim_agent_dir,
    package_checker,
    answer,
    failure,
    exit_code,
    kept,
):
    report = load_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        checker=package_checker(answer),
    )

    result = {
        "decision": "deny",
        "failure": failure,
        "exit_code": exit_code,
        "detail": None,
        "answer": kept,
    }
    assert report.outcome == "denied"
    assert report.config is None
    assert report.checker_result == result
    with pytest.raises(FabricConfigError, match=failure) as caught:
        report.to_config()
    assert caught.value.code == "package_check_failed"
    del result["decision"]
    assert caught.value.details == result


def test_checker_that_crashes_is_a_deny_and_its_error_text_is_in_the_report(
    package_dir, shim_config, hermes_shim_agent_dir, package_checker
):
    report = load_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        checker=package_checker("crash"),
    )

    assert report.outcome == "denied"
    result = report.checker_result
    assert result["failure"] == "unexpected_exit"
    assert result["exit_code"] == 1
    assert result["answer"] is None
    # The end of what the checker wrote to standard error.
    assert result["detail"].endswith("RuntimeError: the policy folder is not readable")
    with pytest.raises(FabricConfigError) as caught:
        report.to_config()
    assert caught.value.code == "package_check_failed"
    assert caught.value.details["detail"] == result["detail"]


@pytest.mark.parametrize(
    "answer",
    [
        # The checker sleeps for a day after it wrote, and the load ends it.
        "too-much-output",
        "too-much-error-text",
        # The checker answers `allow` after it wrote, and ends by itself.
        "too-much-output-then-allow",
        "too-much-error-text-then-allow",
    ],
)
def test_checker_that_writes_more_than_it_may_has_one_result_however_it_ended(
    package_dir, shim_config, hermes_shim_agent_dir, package_checker, answer
):
    # The time limit is a day too. So a load whose checker sleeps returns only
    # because the write limit ended that checker, and no clock is read.
    report = load_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        checker=package_checker(answer),
        checker_timeout_seconds=A_DAY,
    )

    assert report.outcome == "denied"
    assert report.config is None
    assert report.checker_result == {
        "decision": "deny",
        "failure": "output_too_large",
        "exit_code": None,
        "detail": "the checker wrote more than 1048576 bytes",
        "answer": None,
    }
    with pytest.raises(FabricConfigError, match="output_too_large") as caught:
        report.to_config()
    assert caught.value.code == "package_check_failed"


def test_checker_that_cannot_be_started_is_a_deny(
    tmp_path, package_dir, shim_config, hermes_shim_agent_dir
):
    report = load_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        checker=[str(tmp_path / "no-such-checker")],
    )

    assert report.outcome == "denied"
    result = report.checker_result.to_mapping()
    # The operating system says why, in its own words.
    assert result.pop("detail").startswith("the checker cannot be started: ")
    assert result == {
        "decision": "deny",
        "failure": "not_started",
        "exit_code": None,
        "answer": None,
    }


def test_checker_that_runs_past_the_time_limit_is_ended_and_is_a_deny(
    package_dir, shim_config, hermes_shim_agent_dir, package_checker
):
    # This checker sleeps for a day. The load returns only because the time
    # limit ended it, and no clock is read.
    report = load_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        checker=package_checker("hang"),
        checker_timeout_seconds=0.5,
    )

    assert report.outcome == "denied"
    assert report.checker_result == {
        "decision": "deny",
        "failure": "timed_out",
        "exit_code": None,
        "detail": None,
        "answer": None,
    }


def test_checker_gets_three_variables_and_nothing_of_the_callers(
    tmp_path,
    package_dir,
    shim_config,
    hermes_shim_agent_dir,
    package_checker,
):
    secret = "a-token-only-the-harness-may-see"
    os.environ["MODEL_API_KEY"] = secret
    config = shim_config.to_mapping()
    config["environment"]["env"] = {"RELEASE_FEED_TOKEN": secret}
    record = tmp_path / "given.json"
    checker = package_checker("allow", record)

    report = load_agent_package(
        package_dir,
        FabricConfig.from_mapping(config),
        base_dir=hermes_shim_agent_dir,
        checker=checker,
    )

    assert report.outcome == "ready"
    assert secret in json.dumps(report.to_mapping()["config"])
    assert secret not in record.read_text(encoding="utf-8")
    given = json.loads(record.read_text(encoding="utf-8"))
    # Nothing follows the checker's own arguments.
    assert given["arguments"] == checker[2:]
    environment = given["environment"]
    assert environment["NEMO_FABRIC_PACKAGE_DIR"] == str(report.package_dir)
    assert environment["NEMO_FABRIC_AGENT_NAME"] == "release-notes-agent"
    assert "MODEL_API_KEY" not in environment
    assert "VIRTUAL_ENV" not in environment
    # The report it read is this load's report without the config, in a
    # private file that is gone once the checker answered.
    expected = report.to_mapping()
    del expected["config"]
    assert given["report"] == {**expected, "checker_result": None}
    assert not Path(environment["NEMO_FABRIC_PACKAGE_REPORT"]).parent.exists()


def test_checker_is_not_asked_about_a_package_that_cannot_start(
    tmp_path,
    agent,
    package_files,
    package_dir,
    shim_config,
    hermes_shim_agent_dir,
    package_checker,
):
    record = tmp_path / "given.json"
    checker = package_checker("allow", record)

    # With no config nothing can start.
    verified = load_agent_package(package_dir, checker=checker)
    assert verified.outcome == "verified_only"
    assert verified.checker_result is None

    # The test adapter does not accept system instructions, so the load stops.
    agent["instructions"] = "Write one short line for each change."
    write_package(package_dir, agent, package_files)
    stopped = load_agent_package(
        package_dir, shim_config, base_dir=hermes_shim_agent_dir, checker=checker
    )
    assert stopped.outcome == "stopped"
    assert stopped.checker_result is None

    # A package that fails its own checks is refused.
    skill = package_dir / "skills" / "release-notes" / "SKILL.md"
    skill.write_text("# Changed after the lock was written\n", encoding="utf-8")
    with pytest.raises(FabricConfigError) as caught:
        load_agent_package(
            package_dir, shim_config, base_dir=hermes_shim_agent_dir, checker=checker
        )
    assert caught.value.code == "package_digest_mismatch"

    assert not record.exists()


@pytest.mark.parametrize(
    ("options", "subject"),
    [
        ({"checker": "/opt/policy/bin/check-package --strict"}, "checker"),
        ({"checker": {"program": "/opt/policy/bin/check-package"}}, "checker"),
        ({"checker": ["/opt/policy/bin/check-package", 3]}, "options"),
        # An empty list never means that the package starts unchecked.
        ({"checker": []}, "checker"),
        ({"checker": ()}, "checker"),
        ({"checker_timeout_seconds": 0}, "checker_timeout_seconds"),
        ({"checker_timeout_seconds": -5}, "checker_timeout_seconds"),
        ({"checker_timeout_seconds": "30"}, "options"),
    ],
)
def test_checker_setting_that_cannot_be_used_is_refused(
    package_dir, shim_config, hermes_shim_agent_dir, options, subject
):
    with pytest.raises(FabricConfigError) as caught:
        load_agent_package(
            package_dir, shim_config, base_dir=hermes_shim_agent_dir, **options
        )

    assert caught.value.stage == "agent_package"
    assert caught.value.code == "package_options_invalid"
    assert caught.value.details["subject"] == subject
