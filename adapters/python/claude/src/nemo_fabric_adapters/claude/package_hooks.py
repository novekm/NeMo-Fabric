# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the hooks of an agent package as Claude Agent SDK hook callbacks.

A hook is a folder with a ``hooks.json`` in the Agent Plugins 1.1.0 hooks
format. Claude Code reads hooks in a format of its own, so the adapter runs
each handler itself, the way the Agent Plugins format says a client does: a new
process for each event, the hook folder as its working directory, one JSON
object on standard input and one answer on standard output.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import signal
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from claude_agent_sdk import HookCallback
from claude_agent_sdk import HookMatcher

LOGGER = logging.getLogger(__name__)

HOOKS_FORMAT = "https://agent-plugins.org/schemas/1.1.0/hooks.schema.json"
# The events of the format that the adapter has a Claude hook callback for.
# The format also defines SessionStart.
SUPPORTED_EVENTS = ("UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop")
# Under the SDK's default hook timeout of 60 seconds, so a handler that hangs
# fails here, where a failed PreToolUse handler denies the call.
HANDLER_TIMEOUT_SECONDS = 45.0
# A handler gets operating-system variables only: no variable that holds a
# model credential and no value from the agent's environment.
HANDLER_ENV_NAMES = (
    "COMSPEC",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
    "USERPROFILE",
)
_PLACEHOLDER = re.compile(r"\$\{(PLUGIN_ROOT|PLUGIN_DATA)\}")
# The one answer of each event that carries a reason.
_REASONED_DECISIONS = {
    "UserPromptSubmit": "deny",
    "PreToolUse": "deny",
    "Stop": "continue",
}
_CONTEXT_EVENTS = {"UserPromptSubmit", "PostToolUse"}
# Claude reports a tool call that failed through an event of its own. The
# format sends every completed tool result to PostToolUse.
_CLAUDE_EVENTS = {"PostToolUse": ("PostToolUse", "PostToolUseFailure")}
# Claude's file tools, as the kind of change and the input member with the path.
_FILE_MODIFY_TOOLS = {
    "Write": ("write", "file_path"),
    "Edit": ("edit", "file_path"),
    "NotebookEdit": ("edit", "notebook_path"),
}


class _HandlerFailure(Exception):
    """One handler run that gave no answer the format defines."""


@dataclass(frozen=True)
class _Handler:
    """One handler of a hook, ready to start."""

    hook: str
    root: Path
    executable: str
    args: tuple[str, ...]
    environment: dict[str, str]


def package_hook_matchers(
    entries: Any, *, base_dir: str, data_root: Path
) -> dict[str, list[HookMatcher]]:
    """Return the SDK hook matchers that run the handlers of each hook entry.

    Every ``hooks.json`` is checked first, and a hook the adapter cannot run
    as written raises ``ValueError`` with the hook and the reason. Each hook
    gets a data folder of its own below ``data_root``.
    """

    if not isinstance(entries, list):
        raise ValueError("hook entries must be a list")
    handlers: dict[str, list[_Handler]] = {}
    data_folders: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("a hook entry must be an object with a path")
        path = Path(base_dir) / entry["path"]
        # Messages name the folder as declared. Paths use where it really is.
        hook, root = Path(os.path.abspath(path)).name, path.resolve()
        hooks = _read_hooks(hook, root, entry.get("format"))
        data = data_root.resolve() / sha256(str(root).encode()).hexdigest()
        environment = {
            name: os.environ[name] for name in HANDLER_ENV_NAMES if name in os.environ
        }
        environment.update(PLUGIN_ROOT=str(root), PLUGIN_DATA=str(data))
        for event, configured in hooks.items():
            for handler in configured:
                handlers.setdefault(event, []).append(
                    _handler(hook, root, event, handler, environment)
                )
        data_folders.append(data)
    for data in data_folders:
        data.mkdir(parents=True, exist_ok=True)
    return {
        claude_event: [
            HookMatcher(hooks=[_callback(event, claude_event, item) for item in items])
        ]
        for event, items in handlers.items()
        for claude_event in _CLAUDE_EVENTS.get(event, (event,))
    }


def _read_hooks(name: str, root: Path, hooks_format: Any) -> dict[str, list[Any]]:
    try:
        document = json.loads((root / "hooks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as error:
        raise ValueError(f"hook {name!r} has no readable hooks.json: {root}") from error
    if (
        hooks_format != HOOKS_FORMAT
        or not isinstance(document, dict)
        or document.get("$schema") != HOOKS_FORMAT
    ):
        raise ValueError(
            f"hook {name!r} is not in the Agent Plugins 1.1.0 hooks format: {root}"
        )
    hooks = document.get("hooks")
    if set(document) != {"$schema", "hooks"} or not isinstance(hooks, dict):
        raise ValueError(
            f"hook {name!r}: hooks.json must hold $schema and hooks only: {root}"
        )
    for event, configured in hooks.items():
        if event not in SUPPORTED_EVENTS:
            raise ValueError(
                f"hook {name!r} declares {event}, which the Claude adapter "
                f"cannot run; it runs {', '.join(SUPPORTED_EVENTS)}: {root}"
            )
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                f"hook {name!r}: {event} must list at least one handler: {root}"
            )
    return hooks


def _handler(
    name: str, root: Path, event: str, handler: Any, environment: dict[str, str]
) -> _Handler:
    def invalid(reason: str) -> ValueError:
        return ValueError(f"hook {name!r}: a {event} handler {reason}: {root}")

    if not isinstance(handler, dict) or not set(handler) <= {
        "command",
        "args",
        "extensions",
    }:
        raise invalid("must hold command, args and extensions only")
    command = handler.get("command")
    args = handler.get("args", [])
    extensions = handler.get("extensions", {})
    if not isinstance(command, str) or not command:
        raise invalid("needs a command")
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        raise invalid("needs args as a list of strings")
    if not isinstance(extensions, dict) or any(
        not isinstance(value, dict) for value in extensions.values()
    ):
        raise invalid("needs extensions as an object of objects")
    if "\0" in command or any("\0" in arg for arg in args):
        # No process can start with one, so it fails here and not at each event.
        raise invalid("has a NUL character in its command or args")
    if command.startswith("./"):
        path = (root / command).resolve()
        if not path.is_relative_to(root):
            raise invalid(f"has the command {command!r}, which is outside the hook")
        if not path.is_file() or not os.access(path, os.X_OK):
            raise invalid(
                f"has the command {command!r}, which is not an executable file"
            )
        executable = str(path)
    elif "/" in command or "\\" in command:
        raise invalid(
            f"has the command {command!r}; a command is a bare name or starts with ./"
        )
    else:
        found = shutil.which(command, path=environment.get("PATH"))
        if found is None:
            raise invalid(f"has the command {command!r}, which is not on PATH")
        # An empty PATH entry gives a relative path, which would be looked up
        # again from the hook folder the handler starts in.
        executable = os.path.abspath(found)
    return _Handler(
        hook=name,
        root=root,
        executable=executable,
        # One pass, so text that a replacement brings in is not expanded again.
        args=tuple(
            _PLACEHOLDER.sub(lambda match: environment[match.group(1)], arg)
            for arg in args
        ),
        environment=environment,
    )


def tool_profile(tool_name: str, tool_input: Any) -> dict[str, Any] | None:
    """Return the format's portable view of one Claude tool call, if it has one.

    The format names six kinds of call. A call that is none of them, or whose
    input does not have the shape the kind needs, has no portable view.
    """

    if not isinstance(tool_input, dict):
        return None
    if tool_name.startswith("mcp__"):
        # Claude names an MCP tool mcp__<server>__<tool> and writes every
        # character outside letters, digits, `_` and `-` as `_`, so for a tool
        # with such a character this is Claude's spelling of its name.
        _, _, tool = tool_name.removeprefix("mcp__").partition("__")
        if tool:
            return _profile("mcp", {"name": tool, "arguments": tool_input})
    elif tool_name == "Bash":
        if isinstance(command := tool_input.get("command"), str):
            return _profile("shell", {"command": command})
    elif tool_name == "Read":
        if path := _absolute_path(tool_input.get("file_path")):
            return _profile("file.read", {"path": path})
    elif tool_name in _FILE_MODIFY_TOOLS:
        kind, member = _FILE_MODIFY_TOOLS[tool_name]
        if path := _absolute_path(tool_input.get(member)):
            return _profile(
                "file.modify", {"operations": [{"kind": kind, "path": path}]}
            )
    elif tool_name == "WebFetch":
        if url := _http_url(tool_input.get("url")):
            return _profile("web.fetch", {"url": url})
    elif tool_name == "WebSearch":
        if isinstance(query := tool_input.get("query"), str):
            return _profile("web.search", {"queries": [query]})
    return None


def _profile(name: str, profile_input: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "input": profile_input}


def _absolute_path(value: Any) -> str | None:
    return value if isinstance(value, str) and os.path.isabs(value) else None


def _http_url(value: Any) -> str | None:
    try:
        parts = urlsplit(value) if isinstance(value, str) else None
    except ValueError:
        return None
    if parts is None or parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return value


def _event_input(
    event: str, hook_input: dict[str, Any], tool_use_id: str | None
) -> dict[str, Any]:
    """Build the format's event from Claude's hook input, with no other member."""

    payload: dict[str, Any] = {
        "hook_event_name": event,
        "session_id": hook_input["session_id"],
        "cwd": hook_input["cwd"],
    }
    if event == "UserPromptSubmit":
        payload["prompt"] = hook_input["prompt"]
    elif event == "Stop":
        payload["continuation_active"] = hook_input["stop_hook_active"]
    else:
        payload["tool_use_id"] = tool_use_id or hook_input["tool_use_id"]
        payload["tool_name"] = hook_input["tool_name"]
        payload["tool_input"] = hook_input["tool_input"]
        if event == "PreToolUse":
            if profile := tool_profile(payload["tool_name"], payload["tool_input"]):
                payload["tool_profile"] = profile
        elif "tool_response" in hook_input:
            payload["tool_result"] = hook_input["tool_response"]
        else:
            # A failed call: Claude gives its error text and no tool_response.
            payload["tool_result"] = {"error": hook_input["error"]}
    return payload


def _is_result(event: str, value: Any) -> bool:
    """Say whether ``value`` is one of the closed answers the event defines."""

    if not isinstance(value, dict):
        return False

    def text(name: str) -> bool:
        return isinstance(value.get(name), str) and value[name] != ""

    members = set(value)
    if members == {"additional_context"}:
        return event in _CONTEXT_EVENTS and text("additional_context")
    if members == {"decision"}:
        return event == "PreToolUse" and value["decision"] == "allow"
    return (
        members == {"decision", "reason"}
        and value["decision"] == _REASONED_DECISIONS.get(event)
        and text("reason")
    )


async def _run(
    handler: _Handler,
    event: str,
    hook_input: dict[str, Any],
    tool_use_id: str | None,
) -> dict[str, Any] | None:
    """Run one handler once and return its answer, or ``None`` for no opinion."""

    try:
        payload = json.dumps(_event_input(event, hook_input, tool_use_id)).encode()
    except (KeyError, TypeError, ValueError) as error:
        raise _HandlerFailure("the Claude event could not be read") from error
    try:
        process = await asyncio.create_subprocess_exec(
            handler.executable,
            *handler.args,
            cwd=handler.root,
            env=handler.environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,  # POSIX only: lets the time limit stop handler children.
        )
    except OSError as error:
        raise _HandlerFailure("the handler could not be started") from error
    try:
        async with asyncio.timeout(HANDLER_TIMEOUT_SECONDS):
            output, _ = await process.communicate(payload)
    except TimeoutError as error:
        raise _HandlerFailure("the handler timed out") from error
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            await process.wait()
    if process.returncode != 0:
        raise _HandlerFailure(f"the handler exited with status {process.returncode}")
    if not output.strip(b" \t\n\r"):
        return None
    try:
        result = json.loads(output.decode("utf-8"))
    except (ValueError, RecursionError) as error:
        raise _HandlerFailure("the handler wrote output that is not JSON") from error
    if not _is_result(event, result):
        raise _HandlerFailure(f"the handler wrote an answer {event} does not define")
    return result


def _callback(event: str, claude_event: str, handler: _Handler) -> HookCallback:
    async def run_handler(
        hook_input: dict[str, Any],
        tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        if hook_input.get("is_interrupt"):
            # An interrupted call has no completed result for PostToolUse.
            return {}
        try:
            result = await _run(handler, event, hook_input, tool_use_id)
        except _HandlerFailure as failure:
            LOGGER.warning("Hook %s failed on %s: %s", handler.hook, event, failure)
            if event != "PreToolUse":
                return {}
            # The format lets a client deny a tool call whose handler failed.
            return _deny_tool_call(
                f"The hook '{handler.hook}' could not check this tool call: {failure}."
            )
        if result is None:
            return {}
        if event == "PreToolUse":
            if result["decision"] == "allow":
                # Claude takes a hook's allow as approval of the call, past
                # permission_mode. The format lets the client's own policy
                # stand over an allow, so here it approves nothing.
                return {}
            return _deny_tool_call(result["reason"])
        if "additional_context" in result:
            return {
                "hookSpecificOutput": {
                    "hookEventName": claude_event,
                    "additionalContext": result["additional_context"],
                }
            }
        # A denied prompt and a requested continuation are both `block` to Claude.
        return {"decision": "block", "reason": result["reason"]}

    return run_handler


def _deny_tool_call(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
