# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The facts a checker URL is sent, printed without starting anything, and the agent named for the tool-call hook."""

import hashlib
import http.server
import json
import threading
from types import SimpleNamespace

import pytest
from nemo_fabric import (
    FabricConfig,
    FabricConfigError,
    agent_package_start_facts,
    load_agent_package,
)
from nemo_fabric.client import _with_guardrails_identity


class _Checker:
    """A local checker URL that allows, and keeps each body it was sent."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        checker = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - the name http.server calls
                length = int(self.headers["Content-Length"])
                checker.bodies.append(json.loads(self.rfile.read(length)))
                body = b'{"decision": "allow"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = (
            f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/hooks/agent-start"
        )
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def test_the_facts_are_exactly_what_the_checker_url_is_sent(
    package_dir, shim_config, hermes_shim_agent_dir
):
    checker = _Checker()
    try:
        load_agent_package(
            package_dir,
            shim_config,
            base_dir=hermes_shim_agent_dir,
            checker_url=checker.url,
            checker_enforcers=["openshell"],
        )
    finally:
        checker.close()

    facts = agent_package_start_facts(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        enforcers=["openshell"],
    )

    assert checker.bodies == [facts]
    lock = (package_dir / "oadl.lock.json").read_bytes()
    assert facts["agent_digest"] == "sha256:" + hashlib.sha256(lock).hexdigest()
    assert facts["hook_event_name"] == "AgentStart"


def test_the_facts_need_no_config_and_start_nothing(package_dir):
    facts = agent_package_start_facts(package_dir)
    assert facts["agent_name"] == "release-notes-agent"
    assert facts["enforcers_present"] == []


@pytest.mark.parametrize("option", ["checker_url", "checker", "checker_enforcers"])
def test_a_checker_option_is_refused(package_dir, option):
    value = {
        "checker_url": "http://127.0.0.1:1/",
        "checker": ["true"],
        "checker_enforcers": [],
    }[option]
    with pytest.raises(FabricConfigError) as raised:
        agent_package_start_facts(package_dir, **{option: value})
    assert raised.value.code == "package_options_invalid"


def _claude(settings: dict) -> FabricConfig:
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "release-reviewer"},
            "harness": {"adapter_id": "nvidia.fabric.claude", "settings": settings},
        }
    )


REPORT = SimpleNamespace(entry={"name": "release-reviewer"}, lock={"sha256": "ab" * 32})


def test_the_guardrails_setting_is_given_the_package_name_and_digest():
    config = _with_guardrails_identity(
        _claude({"nemo_guardrails": {"url": "http://127.0.0.1:8100"}}), REPORT
    )
    assert config.harness.settings["nemo_guardrails"] == {
        "url": "http://127.0.0.1:8100",
        "agent_name": "release-reviewer",
        "agent_digest": "sha256:" + "ab" * 32,
    }


def test_a_value_the_config_sets_is_kept_and_no_setting_adds_none():
    kept = _with_guardrails_identity(
        _claude(
            {"nemo_guardrails": {"url": "http://h:1", "agent_digest": "sha256:mine"}}
        ),
        REPORT,
    )
    assert kept.harness.settings["nemo_guardrails"]["agent_digest"] == "sha256:mine"
    plain = _claude({})
    assert _with_guardrails_identity(plain, REPORT) is plain
