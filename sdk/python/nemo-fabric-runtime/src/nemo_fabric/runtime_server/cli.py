# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fabric-agent-ctl: strict named commands and one JSON response per command."""

from __future__ import annotations

import asyncio
import io
import json
import os
import select
import stat
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nemo_fabric import FabricConfig, FabricConfigError, load_agent_package
from nemo_fabric.agent_package import _PACKAGE_OPTIONS
from nemo_fabric.runtime_server import transport
from nemo_fabric.runtime_server.protocol import (
    CONFIG_OPERATIONS,
    INTERFACE_VERSION,
    OPERATIONS,
    READ_SECONDS,
    REQUEST_LIMIT,
    SOCKET_PATH,
    ProtocolError,
    response_transport_limit,
    decode_json,
    decode_object,
    encode_request,
    encode_response,
    envelope,
)
from nemo_fabric.runtime_server.state import RuntimeServer


def read_input(path: str) -> Any:
    try:
        if path == "-":
            if sys.stdin is None or sys.stdin.isatty():
                raise ValueError("stdin is unavailable or a terminal")
            try:
                descriptor = sys.stdin.fileno()
            except io.UnsupportedOperation:
                encoded = sys.stdin.buffer.read(REQUEST_LIMIT + 1)
            else:
                encoded = bytearray()
                deadline = time.monotonic() + READ_SECONDS
                while len(encoded) <= REQUEST_LIMIT:
                    remaining = deadline - time.monotonic()
                    if (
                        remaining <= 0
                        or not select.select([descriptor], [], [], remaining)[0]
                    ):
                        raise ValueError("stdin deadline expired")
                    chunk = os.read(
                        descriptor, min(65536, REQUEST_LIMIT + 1 - len(encoded))
                    )
                    if not chunk:
                        break
                    encoded.extend(chunk)
                encoded = bytes(encoded)
        else:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("expected regular input file")
                encoded = stream.read(REQUEST_LIMIT + 1)
        if len(encoded) > REQUEST_LIMIT:
            raise ValueError("input exceeds limit")
        return decode_json(encoded)
    except (OSError, ValueError, RecursionError) as error:
        raise ProtocolError("invalid_input") from error


def write_report(path: str, report: dict[str, Any]) -> None:
    """Create or replace the load report file with mode 0600."""
    try:
        encoded = (
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(encoded)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        raise ProtocolError() from error


# The report file of a refusal is not a load report, so it has its own version.
REFUSAL_SCHEMA_VERSION = "fabric.agent_package_refusal/v1alpha1"


def load_package_config(
    package_dir: str,
    config: Any,
    *,
    options: Any,
    report_path: str,
) -> dict[str, Any]:
    """Load an agent package with the caller's config and return the mapped config.

    The package is read and checked in this short-lived client process, before
    the start request is sent. A checker the options name is asked there too.
    A refusal, a stop or a start the checker does not allow becomes
    ``validation_failed`` with the package code as its cause. No path, value
    or message is printed: the report file carries them. The report file is
    always written, so nothing a package declares is dropped without a word.
    """
    if transport.fcntl is None:
        raise ProtocolError("platform_unsupported", "transport", unsupported=True)
    if not isinstance(options, dict):
        raise ProtocolError()
    try:
        typed_config = FabricConfig.model_validate(config)
    except ValidationError as error:
        raise ProtocolError("validation_failed", "validation") from error
    socket_path = Path(os.environ.get("FABRIC_HOST_SOCKET", SOCKET_PATH))
    try:
        unknown = sorted(options.keys() - _PACKAGE_OPTIONS)
        if unknown:
            raise FabricConfigError(
                f"{unknown[0]} is not an agent package option",
                stage="agent_package",
                code="package_options_invalid",
                details={"subject": unknown[0]},
            )
        report = load_agent_package(
            package_dir, typed_config, base_dir=socket_path.parent, **options
        ).to_mapping()
    except FabricConfigError as error:
        write_report(
            report_path,
            {
                "schema_version": REFUSAL_SCHEMA_VERSION,
                "outcome": "refused",
                "error": {
                    "code": error.code,
                    "subject": error.details.get("subject"),
                    "message": str(error),
                },
            },
        )
        raise ProtocolError(
            "validation_failed", "validation", cause=error.code
        ) from error
    mapped = report.pop("config", None)
    write_report(report_path, report)
    if report.get("outcome") == "denied":
        raise ProtocolError(
            "validation_failed",
            "validation",
            cause="package_check_denied"
            if report["checker_result"]["failure"] is None
            else "package_check_failed",
        )
    if report.get("outcome") != "ready" or not isinstance(mapped, dict):
        raise ProtocolError(
            "validation_failed", "validation", cause="package_requirements_unmet"
        )
    return mapped


def read_checker(text: str) -> list[str]:
    """Read ``--package-checker``: one JSON array, the program and then its arguments."""
    try:
        command = decode_json(text.encode("utf-8"))
    except (ValueError, RecursionError) as error:
        raise ProtocolError() from error
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) for part in command)
    ):
        raise ProtocolError()
    return command


def parse_command(arguments: list[str]) -> dict[str, Any]:
    if not arguments or arguments[0] not in OPERATIONS:
        raise ProtocolError()
    operation = arguments[0]
    flags: dict[str, Any] = {}
    index = 1
    while index < len(arguments):
        flag = arguments[index]
        if flag in flags:
            raise ProtocolError()
        if flag in ("--live", "--active", "--ready"):
            flags[flag] = True
            index += 1
            continue
        if flag in (
            "--agent",
            "--runtime-id",
            "--request-id",
            "--config",
            "--session-ref",
            "--input",
            "--request",
            "--timeout-seconds",
            "--package",
            "--package-options",
            "--package-checker",
            "--report",
        ):
            if index + 1 == len(arguments) or (
                flag != "--request-id" and arguments[index + 1].startswith("--")
            ):
                raise ProtocolError()
            flags[flag] = arguments[index + 1]
            index += 2
        else:
            raise ProtocolError()
    required = set()
    if operation in ("start", "invoke", "stop", "status", "result"):
        required.add("--agent")
    if operation in ("invoke", "stop", "status", "result"):
        required.add("--runtime-id")
    if operation in CONFIG_OPERATIONS:
        config_flags = flags.keys() & {"--config", "--request"}
        if len(config_flags) != 1:
            raise ProtocolError()
        required.update(config_flags)
    if operation == "result":
        required.add("--request-id")
    allowed = required.copy()
    if operation == "start" and "--config" in flags:
        allowed.update({"--session-ref", "--package"})
        if "--package" in flags:
            # The report names what was warned about, left out or refused.
            # Stdout carries only a fixed cause, so the report is required.
            required.add("--report")
            allowed.update({"--package-options", "--package-checker", "--report"})
    if operation == "check":
        allowed.add("--timeout-seconds")
        if flags.keys() & {"--agent", "--runtime-id"}:
            required.update({"--agent", "--runtime-id"})
            allowed.update(required)
            levels = flags.keys() & {"--live", "--active", "--ready"}
            if len(levels) > 1:
                raise ProtocolError()
            allowed.update(levels)
    if operation == "invoke":
        payload_flags = flags.keys() & {"--input", "--request"}
        if len(payload_flags) != 1:
            raise ProtocolError()
        allowed.update(payload_flags)
    if not required <= flags.keys() or not flags.keys() <= allowed:
        raise ProtocolError()
    request = {
        "interface_version": INTERFACE_VERSION,
        "operation": operation,
    }
    if operation == "check" and "--runtime-id" in flags:
        request["level"] = next(iter(levels))[2:] if levels else "live"
    if "--timeout-seconds" in flags:
        try:
            request["timeout_seconds"] = float(flags["--timeout-seconds"])
        except ValueError as error:
            raise ProtocolError() from error
    for flag in (
        "--agent",
        "--runtime-id",
        "--request-id",
    ):
        if flag in flags:
            request[flag[2:].replace("-", "_")] = flags[flag]
    if operation in CONFIG_OPERATIONS:
        if "--config" in flags:
            request["config"] = read_input(flags["--config"])
        else:
            payload = read_input(flags["--request"])
            if (
                not isinstance(payload, dict)
                or "config" not in payload
                or not payload.keys() <= {"config", "session_ref"}
            ):
                raise ProtocolError()
            request.update(payload)
    if "--session-ref" in flags:
        request["session_ref"] = read_input(flags["--session-ref"])
    if "--package" in flags:
        # Reject a request the server would reject before reading the package.
        encode_request(request)
        options = (
            read_input(flags["--package-options"])
            if "--package-options" in flags
            else {}
        )
        if "--package-checker" in flags:
            # One place names the checker: the flag or the options file.
            if not isinstance(options, dict) or "checker" in options:
                raise ProtocolError()
            options["checker"] = read_checker(flags["--package-checker"])
        request["config"] = load_package_config(
            flags["--package"],
            request["config"],
            options=options,
            report_path=flags["--report"],
        )
    if operation == "invoke":
        flag = next(iter(payload_flags))
        request[flag[2:]] = read_input(flags[flag])
    encode_request(request)
    return request


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    operation = arguments[0] if arguments and arguments[0] in OPERATIONS else "unknown"
    response_limit = response_transport_limit()
    try:
        response_limit = int(
            os.environ.get("FABRIC_RESPONSE_LIMIT_BYTES", response_limit)
        )
        if response_limit <= 0:
            raise ProtocolError()
        request = parse_command(arguments)
        if transport.fcntl is None:
            raise ProtocolError("platform_unsupported", "transport", unsupported=True)
        socket_path = Path(os.environ.get("FABRIC_HOST_SOCKET", SOCKET_PATH))
        if operation == "serve":
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            host = RuntimeServer(directory=socket_path.parent)
            asyncio.run(transport.serve(host, socket_path=socket_path))
            return 0
        response = asyncio.run(
            transport.client(
                request, socket_path=socket_path, response_limit=response_limit
            )
        )
    except Exception as error:
        if operation == "serve":
            print(
                str(error)
                if isinstance(error, ProtocolError)
                and error.code
                in (
                    "platform_unsupported",
                    "cleanup_unconfirmed",
                    "runtime_busy",
                )
                else "Fabric host could not start or shut down cleanly.",
                file=sys.stderr,
            )
            return 1
        if not isinstance(error, ProtocolError):
            error = ProtocolError()
        response = envelope(operation, error=error)
    encoded = encode_response(response, limit=response_limit)
    sys.stdout.write(encoded.decode("utf-8"))
    return 0 if decode_object(encoded)["status"] == "succeeded" else 1
