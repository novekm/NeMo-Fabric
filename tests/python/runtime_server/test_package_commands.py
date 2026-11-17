# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host spec: `start --package` loads an agent package before the start request is sent."""

import asyncio
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator

if sys.platform == "win32":
    pytest.skip(
        "The Unix command bridge requires a Unix platform", allow_module_level=True
    )

from nemo_fabric import FabricConfig
from nemo_fabric.agent_package import _PACKAGE_OPTIONS
from nemo_fabric.runtime_server import cli

from _utils.agent_packages import agent_document, write_json, write_package

AGENT = "bridge-agent"
SKILL = "$/skills/notes"


@pytest.fixture(name="bridge_agent")
def bridge_agent_fixture():
    return agent_document(
        AGENT,
        description="An agent started through the command bridge.",
        limits={"timeout_seconds": {"var": "timeout"}},
        variables={"timeout": {"type": "number", "default": 600}},
        skills={"notes": {"source": "./skills/notes"}},
    )


@pytest.fixture(name="bridge_files")
def bridge_files_fixture():
    return {
        "skills/notes/SKILL.md": "# Notes\n",
        "hooks/guard/hooks.json": '{"hooks": {}}\n',
    }


@pytest.fixture(name="bridge_package")
def bridge_package_fixture(tmp_path, bridge_agent, bridge_files):
    package = tmp_path / "package"
    write_package(package, bridge_agent, bridge_files)
    return package


@pytest.fixture(name="bridge_config_path")
def bridge_config_path_fixture(tmp_path, hermes_shim_agent_dir):
    # The base directory of a bridge start is the socket's folder.
    os.environ["FABRIC_HOST_SOCKET"] = str(hermes_shim_agent_dir / "fabric.sock")
    return write_json(
        tmp_path / "config.json",
        {
            "metadata": {"name": AGENT},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
        },
    )


@pytest.fixture(name="mock_client")
def mock_client_fixture(monkeypatch):
    mock_client = AsyncMock(
        return_value={
            "operation": "start",
            "status": "succeeded",
            "changed": True,
            "result": {"runtime_id": "runtime-1"},
            "error": None,
        }
    )
    monkeypatch.setattr(cli.transport, "client", mock_client)
    return mock_client


@pytest.fixture(name="report_path")
def report_path_fixture(tmp_path):
    return tmp_path / "report.json"


@pytest.fixture(name="report_schema")
def report_schema_fixture(repo_root):
    return Draft202012Validator(
        json.loads(
            (repo_root / "schemas/sdk/agent-package-report.schema.json").read_text(
                encoding="utf-8"
            )
        )
    )


@pytest.fixture(name="start")
def start_fixture(bridge_config_path, bridge_package, report_path):
    # A package start always names a report file: stdout carries only a fixed cause.
    return [
        "start",
        "--agent",
        AGENT,
        "--config",
        str(bridge_config_path),
        "--package",
        str(bridge_package),
        "--report",
        str(report_path),
    ]


def test_package_start_builds_the_existing_request_with_the_mapped_config(
    start, bridge_package, bridge_config_path
):
    plain = cli.parse_command(start[:5])
    assert start[5] == "--package"

    request = cli.parse_command(start)

    assert request.keys() == plain.keys()
    assert request["interface_version"] == plain["interface_version"]
    assert request["operation"] == "start"
    assert request["agent"] == AGENT
    config = FabricConfig.model_validate(request["config"])
    assert config.metadata.name == AGENT
    assert (
        config.metadata.description == "An agent started through the command bridge."
    )
    assert config.harness.adapter_id == "test.fabric.hermes_shim"
    assert config.runtime.timeout_seconds == 600
    assert [Path(path) for path in config.skills.paths] == [
        (bridge_package / "skills" / "notes").resolve()
    ]


def test_package_start_sends_the_mapped_config_and_writes_a_private_report(
    start, report_path, report_schema, mock_client, capsys
):
    report_path.write_text("stale")
    report_path.chmod(0o644)

    assert cli.main(start) == 0

    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "succeeded"
    assert response["result"] == {"runtime_id": "runtime-1"}
    mock_client.assert_awaited_once()
    sent = mock_client.await_args.args[0]
    assert sent["operation"] == "start"
    assert sent["config"]["skills"]["paths"]
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "ready"
    assert "config" not in report
    assert len(report["config_sha256"]) == 64
    assert report["schema_version"] == "fabric.agent_package_report/v1alpha1"
    assert list(report_schema.iter_errors(report)) == []
    assert {check["address"]: check["result"] for check in report["digests"]} == {
        "$": "verified",
        SKILL: "verified",
    }


def test_package_start_without_a_report_file_is_refused_before_the_package_is_read(
    start, bridge_package, mock_client, capsys
):
    # A hook the loader cannot run would be warned about and dropped with
    # nothing said, because stdout carries no names. So the report is required.
    (bridge_package / "agent.json").write_text("not read", encoding="utf-8")

    assert cli.main(start[:-2]) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "invalid_request"
    assert response["error"]["effects"] == "none"
    mock_client.assert_not_awaited()


def test_warned_things_are_named_in_the_report_and_the_start_is_sent(
    start, tmp_path, report_path, bridge_agent, bridge_files, bridge_package, mock_client
):
    bridge_agent["hooks"] = {"guard": {"source": "./hooks/guard"}}
    bridge_agent["limits"]["max_depth"] = 2
    write_package(bridge_package, bridge_agent, bridge_files)
    options_path = write_json(tmp_path / "options.json", {"if_unsupported": "warn"})

    assert cli.main([*start, "--package-options", str(options_path)]) == 0

    mock_client.assert_awaited_once()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "ready"
    warned = {item["id"] for item in report["declared"] if item["effect"] == "warned"}
    assert warned == {"agent:/hooks/guard", "agent:/limits/max_depth"}


def test_package_options_file_sets_values_and_settings(
    start, tmp_path, report_path, bridge_agent, bridge_files, bridge_package, mock_client
):
    bridge_agent["hooks"] = {"guard": {"source": "./hooks/guard"}}
    write_package(bridge_package, bridge_agent, bridge_files)
    options_path = write_json(
        tmp_path / "options.json",
        {"variables": {"timeout": 120}, "if_unsupported": "warn"},
    )

    assert cli.main([*start, "--package-options", str(options_path)]) == 0

    sent = mock_client.await_args.args[0]
    assert sent["config"]["runtime"]["timeout_seconds"] == 120
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["if_unsupported"] == "warn"
    hook = next(
        item for item in report["declared"] if item["id"] == "agent:/hooks/guard"
    )
    assert hook["effect"] == "warned"


def test_binding_from_the_options_file_accounts_for_a_custom_tool(
    start, tmp_path, report_path, bridge_agent, bridge_files, bridge_package, mock_client
):
    bridge_agent["custom_tools"] = {
        "request_approval": {
            "description": "Ask a person to approve.",
            "input_schema": {"type": "object"},
        }
    }
    write_package(bridge_package, bridge_agent, bridge_files)
    tool = "agent:/custom_tools/request_approval"
    options_path = write_json(
        tmp_path / "options.json",
        {"bindings": {"custom_tools.request_approval": "app.tools:request_approval"}},
    )

    # With no binding the required tool stops the load, and nothing is sent.
    assert cli.main(start) == 1
    mock_client.assert_not_awaited()
    stopped = json.loads(report_path.read_text(encoding="utf-8"))
    assert stopped["stopped_by"] == [tool]

    assert cli.main([*start, "--package-options", str(options_path)]) == 0

    mock_client.assert_awaited_once()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "ready"
    line = next(item for item in report["declared"] if item["id"] == tool)
    assert line["status"] == "bound"
    assert line["bound_to"] == "app.tools:request_approval"
    # The binding is in the report. The config that was sent does not hold it.
    assert "request_approval" not in json.dumps(mock_client.await_args.args[0])


def test_tampered_package_fails_validation_without_sending_or_printing_values(
    start, report_path, bridge_package, mock_client, capsys
):
    skill = bridge_package / "skills" / "notes" / "SKILL.md"
    skill.write_bytes(skill.read_bytes() + b"x")

    assert cli.main(start) == 1

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    response = json.loads(captured.out)
    assert response["operation"] == "start"
    assert response["status"] == "failed"
    assert response["changed"] is False
    assert response["result"] is None
    assert response["error"] == {
        "code": "validation_failed",
        "stage": "validation",
        "message": "Fabric rejected the configuration. Cause: package_digest_mismatch.",
        "effects": "none",
    }
    assert str(bridge_package) not in captured.out
    mock_client.assert_not_awaited()
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {"schema_version", "outcome", "error"}
    assert report["schema_version"] == "fabric.agent_package_refusal/v1alpha1"
    assert report["outcome"] == "refused"
    assert set(report["error"]) == {"code", "subject", "message"}
    assert report["error"]["code"] == "package_digest_mismatch"
    assert report["error"]["subject"] == SKILL
    assert "sha256:" in report["error"]["message"]


def test_stopped_load_fails_validation_and_reports_what_stopped_it(
    start,
    report_path,
    report_schema,
    bridge_agent,
    bridge_files,
    bridge_package,
    mock_client,
    capsys,
):
    bridge_agent["hooks"] = {"guard": {"source": "./hooks/guard"}}
    write_package(bridge_package, bridge_agent, bridge_files)

    assert cli.main(start) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "validation_failed"
    assert response["error"]["message"].endswith(
        "Cause: package_requirements_unmet."
    )
    assert response["error"]["effects"] == "none"
    mock_client.assert_not_awaited()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "stopped"
    assert report["stopped_by"] == ["agent:/hooks/guard"]
    assert "config" not in report
    assert report["config_sha256"] is None
    assert list(report_schema.iter_errors(report)) == []


@pytest.mark.parametrize(
    "options,code,cause,subject",
    [
        pytest.param(
            {"caller_verified": True},
            "validation_failed",
            "package_options_invalid",
            "caller_verified",
            id="unknown-option",
        ),
        pytest.param(
            {"if_unsupported": "ignore"},
            "validation_failed",
            "package_options_invalid",
            "options",
            id="bad-setting",
        ),
        pytest.param(
            {"variables": ["timeout"]},
            "validation_failed",
            "package_options_invalid",
            "variables",
            id="variables-not-an-object",
        ),
        pytest.param(
            {"entry": 5},
            "validation_failed",
            "package_options_invalid",
            "entry",
            id="entry-not-a-path",
        ),
        pytest.param(
            {"bindings": ["custom_tools.request_approval"]},
            "validation_failed",
            "package_options_invalid",
            "bindings",
            id="bindings-not-an-object",
        ),
        pytest.param(
            {"checker": []},
            "validation_failed",
            "package_options_invalid",
            "checker",
            id="checker-with-no-program",
        ),
        pytest.param(
            {"bindings": {"custom_tools.request_approval": "app.tools:ask"}},
            "validation_failed",
            "package_binding_undeclared",
            "bindings.custom_tools.request_approval",
            id="binding-for-a-tool-the-package-does-not-declare",
        ),
        pytest.param(["if_unsupported"], "invalid_request", None, None, id="not-an-object"),
    ],
)
def test_invalid_package_options_send_nothing(
    start, tmp_path, report_path, mock_client, capsys, options, code, cause, subject
):
    options_path = write_json(tmp_path / "options.json", options)

    assert cli.main([*start, "--package-options", str(options_path)]) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == code
    assert response["error"]["effects"] == "none"
    if cause is not None:
        assert response["error"]["message"].endswith(f"Cause: {cause}.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["outcome"] == "refused"
        assert report["error"]["code"] == cause
        assert report["error"]["subject"] == subject
    mock_client.assert_not_awaited()


def test_expected_lock_digest_from_the_options_file_refuses_a_swapped_lock(
    start, tmp_path, report_path, mock_client, capsys
):
    options_path = write_json(
        tmp_path / "options.json", {"expected_lock_sha256": "0" * 64}
    )

    assert cli.main([*start, "--package-options", str(options_path)]) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["message"].endswith(
        "Cause: package_lock_digest_mismatch."
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error"]["code"] == "package_lock_digest_mismatch"
    assert report["error"]["subject"] == "oadl.lock.json"
    mock_client.assert_not_awaited()


def test_checker_that_allows_sends_the_start_and_its_answer_is_in_the_report(
    start, tmp_path, report_path, report_schema, mock_client, package_checker
):
    options_path = write_json(
        tmp_path / "options.json",
        {"checker": package_checker("allow"), "checker_timeout_seconds": 20},
    )

    assert cli.main([*start, "--package-options", str(options_path)]) == 0

    mock_client.assert_awaited_once()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "ready"
    assert report["checker_result"] == {
        "decision": "allow",
        "failure": None,
        "exit_code": 0,
        "detail": None,
        "answer": {
            "decision": "allow",
            "rules": ["approved-hooks"],
            "checker": {"name": "test-checker", "version": "1"},
        },
    }
    assert list(report_schema.iter_errors(report)) == []


def test_checker_that_denies_sends_nothing_and_its_reasons_are_in_the_report(
    start, tmp_path, report_path, report_schema, mock_client, capsys, package_checker
):
    options_path = write_json(
        tmp_path / "options.json", {"checker": package_checker("deny")}
    )

    assert cli.main([*start, "--package-options", str(options_path)]) == 1

    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"] == {
        "code": "validation_failed",
        "stage": "validation",
        "message": "Fabric rejected the configuration. Cause: package_check_denied.",
        "effects": "none",
    }
    assert "approved" not in captured.out
    mock_client.assert_not_awaited()
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "denied"
    assert "config" not in report
    assert report["checker_result"]["decision"] == "deny"
    assert report["checker_result"]["failure"] is None
    assert report["checker_result"]["answer"] == {
        "decision": "deny",
        "reasons": ["The skill release-notes is not on the approved list."],
        "rules": ["approved-skills"],
    }
    assert list(report_schema.iter_errors(report)) == []


@pytest.mark.parametrize(
    "answer,limit,failure,exit_code,detail",
    [
        pytest.param(
            "crash",
            {},
            "unexpected_exit",
            1,
            "RuntimeError: the policy folder is not readable",
            id="crashes",
        ),
        pytest.param("no-json", {}, "invalid_answer", 0, None, id="prints-no-json"),
        pytest.param(
            "two-objects", {}, "invalid_answer", 0, None, id="prints-two-objects"
        ),
        pytest.param(
            "deny-with-exit-0", {}, "invalid_answer", 0, None, id="denies-with-exit-0"
        ),
        # This checker sleeps for a day after it wrote. With a time limit of a
        # day, the command returns only because the write limit ended it.
        pytest.param(
            "too-much-output",
            {"checker_timeout_seconds": 86400},
            "output_too_large",
            None,
            "the checker wrote more than 1048576 bytes",
            id="writes-more-than-it-may",
        ),
        pytest.param(
            "too-much-error-text-then-allow",
            {},
            "output_too_large",
            None,
            "the checker wrote more than 1048576 bytes",
            id="writes-more-than-it-may-and-then-allows",
        ),
        # Only this checker is waited for, so only it gets a short limit. It
        # sleeps for a day, so the command returns only because it was ended.
        pytest.param(
            "hang",
            {"checker_timeout_seconds": 0.5},
            "timed_out",
            None,
            None,
            id="runs-past-the-time-limit",
        ),
    ],
)
def test_checker_that_fails_sends_nothing_and_the_report_says_how(
    start,
    tmp_path,
    report_path,
    report_schema,
    mock_client,
    capsys,
    package_checker,
    answer,
    limit,
    failure,
    exit_code,
    detail,
):
    options_path = write_json(
        tmp_path / "options.json", {"checker": package_checker(answer), **limit}
    )

    assert cli.main([*start, "--package-options", str(options_path)]) == 1

    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert response["error"]["code"] == "validation_failed"
    assert response["error"]["message"].endswith("Cause: package_check_failed.")
    assert response["error"]["effects"] == "none"
    # What the checker wrote is in the report file and nowhere else.
    assert "policy folder" not in captured.out + captured.err
    mock_client.assert_not_awaited()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "denied"
    assert "config" not in report
    result = report["checker_result"]
    assert result["decision"] == "deny"
    assert result["failure"] == failure
    assert result["exit_code"] == exit_code
    if detail is None:
        assert result["detail"] is None
    else:
        assert result["detail"].endswith(detail)
    assert list(report_schema.iter_errors(report)) == []


def test_package_checker_flag_names_the_checker(
    start, report_path, mock_client, capsys, package_checker
):
    denied = [*start, "--package-checker", json.dumps(package_checker("deny"))]
    assert cli.main(denied) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["message"].endswith("Cause: package_check_denied.")
    mock_client.assert_not_awaited()

    allowed = [*start, "--package-checker", json.dumps(package_checker("allow"))]
    assert cli.main(allowed) == 0

    mock_client.assert_awaited_once()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["checker_result"]["decision"] == "allow"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("/opt/policy/bin/check-package", id="not-json"),
        pytest.param('"/opt/policy/bin/check-package"', id="one-text"),
        pytest.param("[]", id="empty-list"),
        pytest.param('["/opt/policy/bin/check-package", 3]', id="a-number"),
        pytest.param('{"program": "/opt/policy/bin/check-package"}', id="an-object"),
    ],
)
def test_package_checker_flag_that_is_not_a_list_of_text_is_refused(
    start, report_path, mock_client, capsys, value
):
    assert cli.main([*start, "--package-checker", value]) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "invalid_request"
    assert response["error"]["effects"] == "none"
    # The package was not read: no report was written.
    assert not report_path.exists()
    mock_client.assert_not_awaited()


def test_checker_named_by_the_flag_and_by_the_options_file_is_refused(
    start, tmp_path, report_path, mock_client, capsys, package_checker
):
    options_path = write_json(
        tmp_path / "options.json", {"checker": package_checker("allow")}
    )
    arguments = [
        *start,
        "--package-options",
        str(options_path),
        "--package-checker",
        json.dumps(package_checker("allow")),
    ]

    assert cli.main(arguments) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "invalid_request"
    assert not report_path.exists()
    mock_client.assert_not_awaited()


def test_checker_is_not_asked_about_a_package_that_cannot_start(
    start,
    tmp_path,
    report_path,
    bridge_agent,
    bridge_files,
    bridge_package,
    mock_client,
    capsys,
    package_checker,
):
    record = tmp_path / "given.json"
    arguments = [
        *start,
        "--package-checker",
        json.dumps(package_checker("allow", record)),
    ]

    # A required hook the test adapter does not take stops the load.
    bridge_agent["hooks"] = {"guard": {"source": "./hooks/guard"}}
    write_package(bridge_package, bridge_agent, bridge_files)
    assert cli.main(arguments) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["error"]["message"].endswith("Cause: package_requirements_unmet.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["checker_result"] is None

    # A package that fails its own checks is refused.
    skill = bridge_package / "skills" / "notes" / "SKILL.md"
    skill.write_bytes(skill.read_bytes() + b"x")
    assert cli.main(arguments) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["error"]["message"].endswith("Cause: package_digest_mismatch.")

    assert not record.exists()
    mock_client.assert_not_awaited()


def test_checker_gets_no_argument_and_nothing_of_the_config(
    start,
    tmp_path,
    bridge_config_path,
    report_path,
    mock_client,
    package_checker,
):
    secret = "a-token-only-the-harness-may-see"
    os.environ["MODEL_API_KEY"] = secret
    config = json.loads(bridge_config_path.read_text(encoding="utf-8"))
    config["environment"] = {"provider": "local", "env": {"NOTES_TOKEN": secret}}
    write_json(bridge_config_path, config)
    record = tmp_path / "given.json"
    checker = package_checker("allow", record)

    assert cli.main([*start, "--package-checker", json.dumps(checker)]) == 0

    # The config that was sent holds the value. The checker saw none of it.
    assert secret in json.dumps(mock_client.await_args.args[0])
    assert secret not in record.read_text(encoding="utf-8")
    given = json.loads(record.read_text(encoding="utf-8"))
    assert given["arguments"] == checker[2:]
    environment = given["environment"]
    assert environment["NEMO_FABRIC_AGENT_NAME"] == AGENT
    assert Path(environment["NEMO_FABRIC_PACKAGE_DIR"]).name == "package"
    assert "MODEL_API_KEY" not in environment
    assert "FABRIC_HOST_SOCKET" not in environment
    # It read the report of this load, before its own answer was added.
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert {**given["report"], "checker_result": report["checker_result"]} == report
    assert given["report"]["checker_result"] is None
    assert not Path(environment["NEMO_FABRIC_PACKAGE_REPORT"]).exists()


def test_no_checker_named_asks_nothing_and_the_report_says_so(
    start, report_path, mock_client
):
    assert cli.main(start) == 0

    mock_client.assert_awaited_once()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "ready"
    assert report["checker_result"] is None


def test_every_package_option_is_an_option_the_schema_lists(repo_root):
    schema = json.loads(
        (repo_root / "schemas/sdk/agent-package-options.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert _PACKAGE_OPTIONS == set(schema["properties"])


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(["--package-options", "options.json"], id="options"),
        pytest.param(["--report", "report.json"], id="report"),
        pytest.param(
            ["--package-checker", '["/opt/policy/bin/check-package"]'], id="checker"
        ),
        pytest.param(
            ["--package-options", "options.json", "--report", "report.json"],
            id="options-and-report",
        ),
    ],
)
def test_package_flags_require_a_package(
    bridge_config_path, mock_client, capsys, flags
):
    arguments = ["start", "--agent", AGENT, "--config", str(bridge_config_path)]

    assert cli.main([*arguments, *flags]) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "invalid_request"
    assert response["error"]["effects"] == "none"
    mock_client.assert_not_awaited()


def test_package_is_not_allowed_with_a_request_file(
    tmp_path, bridge_config_path, bridge_package, mock_client, capsys
):
    request_path = write_json(
        tmp_path / "request.json",
        {"config": json.loads(bridge_config_path.read_text(encoding="utf-8"))},
    )

    assert (
        cli.main(
            [
                "start",
                "--agent",
                AGENT,
                "--request",
                str(request_path),
                "--package",
                str(bridge_package),
            ]
        )
        == 1
    )

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "invalid_request"
    mock_client.assert_not_awaited()


def test_wrong_agent_is_rejected_before_the_package_is_read(
    start, report_path, mock_client, capsys
):
    arguments = list(start)
    arguments[2] = "another-agent"

    assert cli.main(arguments) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "wrong_agent"
    assert not report_path.exists()
    mock_client.assert_not_awaited()


def test_report_that_cannot_be_written_fails_the_command_and_sends_nothing(
    start, tmp_path, mock_client, capsys
):
    arguments = [*start[:-1], str(tmp_path / "missing-folder" / "report.json")]

    assert cli.main(arguments) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "invalid_request"
    assert response["error"]["effects"] == "none"
    mock_client.assert_not_awaited()


def test_package_reports_platform_unsupported_without_loading(
    start, report_path, monkeypatch, mock_client, capsys
):
    monkeypatch.setattr(cli.transport, "fcntl", None)

    assert cli.main(start) == 1

    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "unsupported"
    assert response["error"]["code"] == "platform_unsupported"
    assert not report_path.exists()
    mock_client.assert_not_awaited()


async def test_package_start_runs_through_real_command_processes(
    blocking_adapter_config, tmp_path, package_checker
):
    # A short folder under /tmp keeps the socket path inside the platform limit.
    with tempfile.TemporaryDirectory(prefix="fh-package-", dir="/tmp") as directory:
        root = Path(directory)
        for name in ("adapter.py", "blocking.fabric-adapter.json"):
            shutil.copyfile(tmp_path / name, root / name)
        (root / "release").write_text("release")
        socket = root / "fabric.sock"
        environment = {**os.environ, "FABRIC_HOST_SOCKET": str(socket)}
        command = [str(Path(sys.prefix) / "bin" / "fabric-agent-ctl")]
        config = blocking_adapter_config.to_mapping()
        name = config["metadata"]["name"]
        config_path = write_json(root / "config.json", config)
        package = root / "package"
        write_package(
            package,
            agent_document(
                name,
                description="An agent started from a package by real processes.",
                limits={"timeout_seconds": 120},
            ),
        )
        report_path = root / "report.json"

        async def call(operation, *flags, stdin=None):
            process = await asyncio.create_subprocess_exec(
                *command,
                operation,
                *flags,
                env=environment,
                stdin=asyncio.subprocess.PIPE
                if stdin is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # A command ends by its own deadlines, so the test sets none. How long a
            # command process takes to start says nothing about a package.
            stdout, stderr = await process.communicate(stdin)
            assert len(stdout.splitlines()) == 1, stderr
            response = json.loads(stdout)
            assert process.returncode == (
                0 if response["status"] == "succeeded" else 1
            ), stderr
            return response

        host = await asyncio.create_subprocess_exec(
            *command,
            "serve",
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            # The host makes its socket or exits. The test does not ask how fast.
            while not socket.exists():
                assert host.returncode is None
                await asyncio.sleep(0.02)
            package_flags = (
                "--agent",
                name,
                "--config",
                str(config_path),
                "--package",
                str(package),
            )

            # Without a report file the package is not read and nothing starts.
            refused = await call("start", *package_flags)
            assert refused["error"]["code"] == "invalid_request"
            assert not report_path.exists()

            # A checker that denies stops the real command before any start.
            denied = await call(
                "start",
                *package_flags,
                "--report",
                str(report_path),
                "--package-checker",
                json.dumps(package_checker("deny")),
            )
            assert denied["error"]["code"] == "validation_failed"
            assert denied["error"]["message"].endswith("Cause: package_check_denied.")
            assert denied["error"]["effects"] == "none"
            denial = json.loads(report_path.read_text(encoding="utf-8"))
            assert denial["outcome"] == "denied"
            assert denial["checker_result"]["answer"]["rules"] == ["approved-skills"]

            started = await call("start", *package_flags, "--report", str(report_path))
            assert started["status"] == "succeeded", started
            runtime_id = started["result"]["runtime_id"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            assert report["outcome"] == "ready"
            assert "config" not in report
            assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
            assert {check["result"] for check in report["digests"]} == {"verified"}
            timeout = next(
                item
                for item in report["declared"]
                if item["id"] == "agent:/limits/timeout_seconds"
            )
            assert timeout["status"] == "mapped"

            invoked = await call(
                "invoke",
                "--agent",
                name,
                "--runtime-id",
                runtime_id,
                "--input",
                "-",
                stdin=b'{"message":"hello"}',
            )
            assert invoked["status"] == "succeeded", invoked
            assert invoked["result"]["output"] == {"response": "released"}
            stopped = await call(
                "stop", "--agent", name, "--runtime-id", runtime_id
            )
            assert stopped["status"] == "succeeded", stopped

            # A changed package is refused by the real command before any start.
            document = package / "agent.json"
            document.write_bytes(document.read_bytes() + b" ")
            tampered = await call("start", *package_flags, "--report", str(report_path))
            assert tampered["error"]["code"] == "validation_failed"
            assert tampered["error"]["message"].endswith(
                "Cause: package_digest_mismatch."
            )
            assert str(root) not in json.dumps(tampered)
            refusal = json.loads(report_path.read_text(encoding="utf-8"))
            assert refusal["outcome"] == "refused"
            assert refusal["error"]["subject"] == "$"
        finally:
            if host.returncode is None:
                host.terminate()
                # The host bounds its own exit after a stop request.
                await host.communicate()
        assert host.returncode == 0
