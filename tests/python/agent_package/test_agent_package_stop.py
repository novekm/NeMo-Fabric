# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop a running agent for a reason, with the agents it calls, and watch a URL that says a rule stopped it."""

import asyncio
import http.server
import json
import threading

import pytest
from nemo_fabric import Fabric, FabricError, RuntimeStatus


class _StopServer:
    """A local stand-in for a server that says whether a rule stopped a run: `null`, then the answers given."""

    def __init__(self, answers: list[dict | None]) -> None:
        self.paths: list[str] = []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - the name http.server calls
                server.paths.append(self.path)
                stop = answers.pop(0) if len(answers) > 1 else answers[0]
                body = json.dumps(
                    {"run_id": self.path.rsplit("/", 1)[-1], "stop": stop}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = (
            f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/runs/{{runtime_id}}"
        )
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


STOP = {
    "outcome": "stop",
    "reason": "tool call 'delete_index' is a stop tool",
    "tool_name": "delete_index",
    "by": "nemoguardrails hooks-server",
}


async def _start_two(package_dir, shim_config, hermes_shim_agent_dir):
    """A lead started from the package, and a called agent's runtime started apart from it."""
    client = Fabric()
    lead = await client.start_agent_package(
        package_dir,
        shim_config,
        base_dir=hermes_shim_agent_dir,
        options={"variables": {"timeout": 300}},
    )
    called = await client.start_runtime(shim_config, base_dir=hermes_shim_agent_dir)
    return lead, called


async def test_stop_for_stops_the_lead_and_its_called_agents_and_keeps_the_session(
    package_dir, shim_config, hermes_shim_agent_dir
):
    lead, called = await _start_two(package_dir, shim_config, hermes_shim_agent_dir)
    result = await lead.invoke(input="hi")
    assert result.status == "succeeded"
    # The host's own process records (`.fabric/hosts/`) go with the host when it stops, as
    # they do for `Runtime.stop()`. Everything else the run left is session history.
    kept = sorted(
        path
        for path in hermes_shim_agent_dir.rglob("*")
        if path.is_file() and ".fabric/hosts/" not in path.as_posix()
    )

    record = await lead.stop_for(
        reason=STOP["reason"],
        requested_by="nemoguardrails hooks-server",
        violation=STOP,
        called=[called],
    )

    assert lead.status is RuntimeStatus.STOPPED
    assert called.status is RuntimeStatus.STOPPED
    assert [
        (item["runtime_id"], item["role"], item["stopped"])
        for item in record["runtimes"]
    ] == [
        (lead.runtime_id, "lead", True),
        (called.runtime_id, "called", True),
    ]
    assert (record["reason"], record["requested_by"], record["violation"]) == (
        STOP["reason"],
        "nemoguardrails hooks-server",
        STOP,
    )
    assert record["events"][-1]["kind"] == "runtime.stop_requested"
    # Session history stays: every file the run left is still there.
    assert all(path.exists() for path in kept)
    # A later plain stop is a no-op.
    await lead.stop()


async def test_a_kill_switch_needs_a_reason_and_who_asked(
    package_dir, shim_config, hermes_shim_agent_dir
):
    lead, called = await _start_two(package_dir, shim_config, hermes_shim_agent_dir)
    try:
        with pytest.raises(FabricError):
            await lead.stop_for(reason=" ", requested_by="operator")
        assert lead.status is RuntimeStatus.ACTIVE
        record = await lead.stop_for(
            reason="the operator ended the run", requested_by="operator"
        )
        assert record.get("violation") is None and lead.status is RuntimeStatus.STOPPED
    finally:
        await called.stop()


@pytest.mark.parametrize("outcome", ["stop", "quarantine"])
async def test_a_watched_stop_url_ends_the_run_with_its_called_agents(
    package_dir, shim_config, hermes_shim_agent_dir, outcome
):
    lead, called = await _start_two(package_dir, shim_config, hermes_shim_agent_dir)
    stop = {**STOP, "outcome": outcome}
    server = _StopServer([None, None, stop])
    try:
        record = await asyncio.wait_for(
            lead.watch_stop_url(server.url, called=[called], interval_seconds=0.05),
            timeout=30,
        )
    finally:
        server.close()

    assert (
        lead.status is RuntimeStatus.STOPPED and called.status is RuntimeStatus.STOPPED
    )
    assert server.paths[0] == f"/v1/runs/{lead.runtime_id}"
    assert len(server.paths) == 3
    assert record.get("quarantine", False) is (outcome == "quarantine")
    assert (record["reason"], record["requested_by"], record["violation"]) == (
        stop["reason"],
        stop["by"],
        stop,
    )


async def test_a_watch_ends_when_the_run_stops_otherwise(
    package_dir, shim_config, hermes_shim_agent_dir
):
    lead, called = await _start_two(package_dir, shim_config, hermes_shim_agent_dir)
    await called.stop()
    server = _StopServer([None])
    try:
        watch = lead.watch_stop_url(server.url, interval_seconds=0.05)
        await asyncio.sleep(0.2)
        await lead.stop()
        assert await asyncio.wait_for(watch, timeout=10) is None
    finally:
        server.close()


async def test_a_stop_url_that_cannot_be_reached_leaves_the_run_running(
    package_dir, shim_config, hermes_shim_agent_dir
):
    lead, called = await _start_two(package_dir, shim_config, hermes_shim_agent_dir)
    server = _StopServer([None])
    url = server.url
    server.close()
    try:
        watch = lead.watch_stop_url(url, interval_seconds=0.05, timeout_seconds=0.5)
        await asyncio.sleep(0.5)
        assert not watch.done() and lead.status is RuntimeStatus.ACTIVE
        watch.cancel()
    finally:
        await lead.stop()
        await called.stop()


class _CountingNative:
    """The native module, counting the native stops asked for each runtime."""

    def __init__(self, native) -> None:
        self._native = native
        self.stops: list[str] = []

    def __getattr__(self, name):
        return getattr(self._native, name)

    def stop_runtime(self, plan_json, runtime_json):
        self.stops.append(json.loads(runtime_json)["runtime_id"])
        return self._native.stop_runtime(plan_json, runtime_json)

    def stop_runtime_for(self, plan_json, runtime_json, called_json, request_json):
        self.stops.append(json.loads(runtime_json)["runtime_id"])
        self.stops.extend(item["runtime"]["runtime_id"] for item in json.loads(called_json))
        return self._native.stop_runtime_for(plan_json, runtime_json, called_json, request_json)


async def test_the_watch_a_manual_stop_and_cleanup_join_one_stop(
    package_dir, shim_config, hermes_shim_agent_dir, monkeypatch
):
    lead, called = await _start_two(package_dir, shim_config, hermes_shim_agent_dir)
    counting = _CountingNative(lead._client._native_module())
    monkeypatch.setattr(lead._client, "_native_module", lambda: counting)
    server = _StopServer([STOP])
    try:
        watch = lead.watch_stop_url(server.url, called=[called], interval_seconds=0.01)
        results = await asyncio.wait_for(
            asyncio.gather(
                watch,
                lead.stop_for(reason="the operator ended the run", requested_by="operator", called=[called]),
                lead.stop(),
                lead.__aexit__(None, None, None),
                return_exceptions=True,
            ),
            timeout=30,
        )
    finally:
        server.close()

    assert not [result for result in results if isinstance(result, BaseException)]
    assert lead.status is RuntimeStatus.STOPPED and called.status is RuntimeStatus.STOPPED
    # One native stop for each runtime, however many paths asked.
    assert sorted(counting.stops) == sorted([lead.runtime_id, called.runtime_id])
    await lead.stop()
    assert len(counting.stops) == 2


async def test_an_operator_can_choose_to_stop_a_run_whose_stop_url_cannot_be_reached(
    package_dir, shim_config, hermes_shim_agent_dir
):
    lead, called = await _start_two(package_dir, shim_config, hermes_shim_agent_dir)
    server = _StopServer([None])
    url = server.url
    server.close()
    record = await asyncio.wait_for(
        lead.watch_stop_url(
            url,
            called=[called],
            interval_seconds=0.05,
            timeout_seconds=0.5,
            if_unreachable="stop",
            unreachable_after_seconds=0.3,
        ),
        timeout=30,
    )

    assert lead.status is RuntimeStatus.STOPPED and called.status is RuntimeStatus.STOPPED
    assert record["requested_by"] == "nemo-fabric"
    assert "did not answer for 0.3 seconds" in record["reason"]
    assert record.get("violation") is None


async def test_by_default_a_run_keeps_running_when_its_stop_url_cannot_be_reached(
    package_dir, shim_config, hermes_shim_agent_dir
):
    lead, called = await _start_two(package_dir, shim_config, hermes_shim_agent_dir)
    server = _StopServer([None])
    url = server.url
    server.close()
    try:
        watch = lead.watch_stop_url(
            url,
            interval_seconds=0.05,
            timeout_seconds=0.5,
            if_unreachable="keep_running",
            unreachable_after_seconds=0.1,
        )
        await asyncio.sleep(0.6)
        assert not watch.done() and lead.status is RuntimeStatus.ACTIVE
        watch.cancel()
    finally:
        await lead.stop()
        await called.stop()


async def test_an_unknown_unreachable_choice_is_refused(
    package_dir, shim_config, hermes_shim_agent_dir
):
    lead, called = await _start_two(package_dir, shim_config, hermes_shim_agent_dir)
    try:
        with pytest.raises(FabricError):
            lead.watch_stop_url("http://127.0.0.1:1/", if_unreachable="pause")
    finally:
        await lead.stop()
        await called.stop()
