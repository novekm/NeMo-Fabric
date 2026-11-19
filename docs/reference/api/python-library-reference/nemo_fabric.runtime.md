---
title: "Runtime"
slug: "/reference/api/python-library-reference/runtime"
description: "Drive stateful multi-turn execution through the Runtime API."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.runtime`

Runtime lifecycle support for the NVIDIA NeMo Fabric Python SDK.



---


## <kbd>class</kbd> `RuntimeStatus`

Lifecycle state of a runtime.

``ACTIVE`` accepts invocations, ``STOPPED`` has released its runtime, and ``FAILED`` records a lifecycle failure that prevents further invocations but still permits cleanup.






### Inheritance

Direct bases: `str`, `Enum`.

### Values

The enum defines the following values:

| Name | Value |
| --- | --- |
| `ACTIVE` | `active` |
| `STOPPED` | `stopped` |
| `FAILED` | `failed` |

---


## <kbd>class</kbd> `Runtime`

One logical, stateful harness execution.

Create runtimes with ``Fabric.start_runtime()`` rather than calling the constructor. A runtime serializes invocations and preserves adapter-owned harness state across turns. Use it as an asynchronous context manager to stop the runtime on exit.

Runtime-scoped overrides are recursively merged with invocation overrides; invocation values win.


---

### <kbd>property</kbd> agent_package_report

Return the load report of ``Fabric.start_agent_package()``, or None.

---

### <kbd>property</kbd> handle

Return a detached snapshot of the runtime handle.

---

### <kbd>property</kbd> invocations

Return copied request, runtime, and invocation IDs for completed turns.

---

### <kbd>property</kbd> messages

Return a deep copy of the latest harness-provided message history.

---

### <kbd>property</kbd> recovery_warnings

Return recovery warnings. Acknowledge them through a new RunRequest.

---

### <kbd>property</kbd> runtime_id

Return the unique identifier for this started runtime lifecycle.

---

### <kbd>property</kbd> session_ref

Return the confirmed durable native session, or None for legacy sessions.

---

### <kbd>property</kbd> status

Return the current ``ACTIVE``, ``STOPPED``, or ``FAILED`` state.

---

### <kbd>property</kbd> supports_openai_streaming

Return whether the selected adapter implements native OpenAI streaming.

---

### <kbd>property</kbd> supports_streaming

Return whether NVIDIA NeMo Relay ATOF streaming is enabled.



---


### <kbd>method</kbd> `invoke`

```python
async def invoke(*, input: Any = None, request: RunRequest | None = None) -> RunResult
```

Run one turn on this runtime.

``input`` and ``request`` are mutually exclusive. Runtime overrides are merged below invocation overrides from ``RunRequest``. Concurrent turns on the same runtime are rejected.



**Args:**

 - <b>`input`</b>:  JSON-compatible turn input.
 - <b>`request`</b>:  Complete validated ``RunRequest``.



**Returns:**
 The normalized ``RunResult`` for this turn.



**Raises:**

 - <b>`FabricConfigError`</b>:  If request fields conflict or are not  JSON-compatible.
 - <b>`FabricStateError`</b>:  If the runtime is not active, is stopping, or is  already running a turn.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is missing.
 - <b>`FabricRuntimeError`</b>:  If native invocation fails before returning a  normalized result.

---


### <kbd>method</kbd> `invoke_openai_stream`

```python
def invoke_openai_stream(
    *,
    input: Any = None,
    request: RunRequest | None = None,
) -> OpenAIInvokeStream
```

Start one turn and stream native OpenAI chat-completion chunks.

The returned stream yields ``chat.completion.chunk`` mappings. Await ``stream.result()`` for the separate normalized terminal result.



**Raises:**

 - <b>`FabricCapabilityError`</b>:  If the selected adapter does not advertise  native OpenAI streaming.
 - <b>`FabricConfigError`</b>:  If request fields conflict or are not  JSON-compatible.
 - <b>`FabricStateError`</b>:  If another turn or stream is active.

---


### <kbd>method</kbd> `invoke_stream`

```python
def invoke_stream(
    *,
    input: Any = None,
    request: RunRequest | None = None,
) -> InvokeStream
```

Start one turn and stream raw NeMo Relay ATOF records as they arrive.

``input`` and ``request`` are mutually exclusive. The returned ``InvokeStream`` yields raw ATOF dictionaries. Await ``stream.result()`` for the terminal normalized ``RunResult``.



**Raises:**

 - <b>`FabricCapabilityError`</b>:  If the runtime was not started with NeMo Relay  enabled and ``streaming=True``.
 - <b>`FabricConfigError`</b>:  If request fields conflict or are not  JSON-compatible.
 - <b>`FabricStateError`</b>:  If another turn or stream is active.

---


### <kbd>method</kbd> `resource_snapshot`

```python
async def resource_snapshot() -> RuntimeResourceSnapshot
```

Return a timestamped resource snapshot of this runtime.

The snapshot covers the local adapter host process and its live descendants. It does not wait for a running invocation, so it can be called while ``invoke()`` is in flight. It observes resources only and enforces no limits.



**Raises:**

 - <b>`ResourceObservationUnavailableError`</b>:  If the runtime stopped, is  stopping, its adapter host process exited, a live process in  its tree or the process table cannot be read, or the process  tree kept changing during the read.
 - <b>`ResourceObservationUnsupportedError`</b>:  If the runtime's adapter  kind, environment provider, or operating system has no  resource reader.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is missing.
 - <b>`FabricRuntimeError`</b>:  If the native snapshot call fails otherwise.

---


### <kbd>method</kbd> `stop`

```python
async def stop() -> None
```

Stop the runtime, interrupting an active buffered invocation if needed.

Repeated calls after a successful stop are no-ops. A failed runtime may still be stopped so its resources are released. A stop already under way, by this method, :meth:`stop_for` or a stop watch, is joined, not asked for again.



**Raises:**

 - <b>`FabricStateError`</b>:  If the runtime is already stopping or has an  streaming invocation in flight.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is missing.
 - <b>`FabricRuntimeError`</b>:  If native runtime shutdown fails.

---


### <kbd>method</kbd> `stop_for`

```python
async def stop_for(
    *,
    reason: str,
    requested_by: str,
    violation: Mapping[str, Any] | None = None,
    quarantine: bool = False,
    called: Sequence[Runtime] = (),
) -> dict[str, Any]
```

Stop this runtime for a reason, with the runtimes of the agents it calls.

A rule that ends a run, or a person with no violation (a kill switch), calls this. The lead stops first, so it starts no new call, then each runtime in ``called``. A runtime that fails to stop does not keep the others running. Nothing else is deleted: the workspace, the artifacts and the harness session stay.



**Args:**

 - <b>`reason`</b>:  Why the run ends. Not blank.
 - <b>`requested_by`</b>:  Who asked. Not blank.
 - <b>`violation`</b>:  The decision that asked for the stop, as its rule wrote  it. ``None`` for a stop no rule asked for.
 - <b>`quarantine`</b>:  Whether the stop is a quarantine. The quarantine itself  is kept by the checker that refuses later starts.
 - <b>`called`</b>:  Runtimes of agents this one calls, started apart from it.



**Returns:**

 - <b>`The stop record`</b>:  ``stop_id``, ``reason``, ``requested_by``, ``violation``, ``quarantine``, the outcome of each runtime, and the events, ending with ``runtime.stop_requested``.



**Raises:**

 - <b>`FabricError`</b>:  If the reason or the requester is blank.

---


### <kbd>method</kbd> `watch_stop_url`

```python
def watch_stop_url(
    url: str,
    *,
    called: Sequence[Runtime] = (),
    interval_seconds: float = 0.5,
    timeout_seconds: float = 5.0,
    if_unreachable: Literal['keep_running', 'stop'] = 'keep_running',
    unreachable_after_seconds: float = 30.0,
) -> asyncio.Task[dict[str, Any] | None]
```

Watch a URL that says whether a rule stopped this run, and stop it when one did.

``url`` may hold ``{runtime_id}``, such as ``http://127.0.0.1:8100/v1/runs/{runtime_id}``, the NeMo Guardrails hooks server's answer to which run a stop belongs to. The harness names the run in each check with this runtime's id. The answer is ``{"stop": null}``, or ``{"stop": {"outcome": "stop" | "quarantine", "reason": ..., "by": ...}}``. On a stop the run ends through :meth:`stop_for`, with ``called``, the reason and who asked, the answer as the violation, and ``quarantine`` for a quarantine.

The watch ends when this runtime stops by any means. Cancel the task to stop watching.

A URL that cannot be reached, or answers with another status or body, is logged and asked again. What the run does meanwhile is the operator's choice, ``if_unreachable``: ``"keep_running"`` (the default) leaves it running; ``"stop"`` ends it, with ``called``, once the URL has not answered for ``unreachable_after_seconds`` in a row, requested by ``nemo-fabric`` with the reason that the server could not be reached. Either way a harness hook that cannot reach the server denies the call it asked about.



**Returns:**
  A task whose result is the stop record, or ``None`` when the  runtime stopped otherwise.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
