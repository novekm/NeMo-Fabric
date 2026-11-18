# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load an agent package: a folder with an agent document, its lock and its parts."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from nemo_fabric.errors import FabricConfigError, FabricError
from nemo_fabric.models import FabricConfig
from nemo_fabric.types import FabricMapping, _FabricConfigSnapshot

_STAGE = "agent_package"


class AgentPackageDeclaredItem(FabricMapping):
    """One thing a document declares, and what the load did with it.

    Attributes:
        id: The stating document and a JSON Pointer into it, for example
            ``agent:/skills/release-notes``.
        kind: The kind of declared thing, such as ``part``, ``limit`` or
            ``need``.
        stated_by: The document that states it: ``agent`` or ``definition``.
        document: That document's name.
        field: The top-level field, for example ``skills`` or ``runtime``.
        name: The entry key or member name, as written.
        address: The lock address, for a part.
        required: ``False`` for a thing marked optional and for a thing that
            is only reported.
        status: What the load established, such as ``mapped``,
            ``not_mapped``, ``bound``, ``not_met`` or
            ``could_not_be_checked``.
        gap: Why, when the status is ``not_mapped``.
        connector: The connector a need is met through, or the connector of a
            connector tool.
        effect: What the load did about it: ``none``, ``warned`` or
            ``stopped``.
        mapped_to: The config paths it was written to.
        bound_to: The caller's target, for a custom tool or a called agent
            with a binding.
        reason: The author's reason, unchanged.
        detail: The loader's explanation.
    """

    id: str
    kind: str
    stated_by: str
    document: str
    field: str
    name: str | None
    address: str | None
    required: bool
    status: str
    gap: str | None
    connector: str | None
    effect: str
    mapped_to: Sequence[str]
    bound_to: str | None
    reason: str | None
    detail: str | None
    _fields = frozenset(
        {
            "id",
            "kind",
            "stated_by",
            "document",
            "field",
            "name",
            "address",
            "required",
            "status",
            "gap",
            "connector",
            "effect",
            "mapped_to",
            "bound_to",
            "reason",
            "detail",
        }
    )


class AgentPackageDigestCheck(FabricMapping):
    """One lock entry and what the load checked about it.

    Attributes:
        address: The lock address.
        entry_type: The entry's type, such as ``skill`` or ``connector``.
        source_type: Where the lock says the content comes from.
        source_url: The recorded source address, with any password left out.
        source_subpath: The recorded path below the source address.
        version: The exact version the lock selected.
        commit: The git commit, for a git source.
        digest: The lock's digest.
        digest_kind: What the digest is a digest of.
        interface: The interface pin of something already running.
        result: What the load established about the content, such as
            ``verified``, ``pointer`` or ``absent``.
        path: Where the bytes were verified.
        retracted: The lock's retraction text, when the entry is retracted.
    """

    address: str
    entry_type: str
    source_type: str
    source_url: str | None
    source_subpath: str | None
    version: str
    commit: str | None
    digest: str | None
    digest_kind: str | None
    interface: Mapping[str, Any] | None
    result: str
    path: str | None
    retracted: str | None
    _fields = frozenset(
        {
            "address",
            "entry_type",
            "source_type",
            "source_url",
            "source_subpath",
            "version",
            "commit",
            "digest",
            "digest_kind",
            "interface",
            "result",
            "path",
            "retracted",
        }
    )
    _json_fields = frozenset({"interface"})


class AgentPackageReport(FabricMapping):
    """What one agent package load read, checked and mapped.

    The statuses are valid for this load only. A runtime that is kept for many
    invocations loads the package again to get fresh statuses.

    Attributes:
        schema_version: Report schema version.
        format_version: Version of the package format that was read.
        package_dir: The package folder, absolute.
        base_dir: The base directory that relative config paths resolve against.
        outcome: ``ready`` when the report holds a config, ``stopped`` when a
            required thing stopped the load, ``verified_only`` when no config
            was given.
        if_unsupported: The setting used for a required thing whose status is
            ``not_mapped`` or ``not_met``.
        if_unverified: The setting used for a required thing whose status is
            ``could_not_be_checked``.
        stopped_by: Ids of the declared things that stopped the load.
        entry: The entry document, with its kind, name, lock address, path
            and digest.
        agent: The agent that runs. Equal to ``entry`` unless the entry
            document is a definition.
        lock: The lock file, with its path, sha256, generator and entry count.
        adapter: The selected adapter and what its descriptor accepts, when a
            config was given. ``adapter["extensions"]`` lists the root
            extension keys the descriptor declares. The loader hands plugins,
            hooks and called agents to an adapter under such a key, for
            example ``nemo_fabric.plugins.v1``.
        variables: Every variable of the agent with its value and where the
            value came from.
        model: What was configured for the model, and who chose it. The
            caller's ``models["default"]`` is the model choice. When the
            config sets none and has a ``models`` entry named with the text
            of the document's default model, the loader moves that entry to
            ``models["default"]`` and ``model["origin"]`` is ``package``.
        declared: Every declared thing with its status, as
            ``AgentPackageDeclaredItem`` lines. ``mapped`` means the
            loader wrote it into the config and the config planned on the
            selected adapter. It does not show that the harness loaded it.
            ``bound`` means the caller gave a binding for a custom tool or a
            called agent. The line's ``bound_to`` is the caller's target.
        digests: Every lock entry with its digest, what was checked and the
            path the bytes were verified at, as ``AgentPackageDigestCheck``
            lines.
        notes: Things the loader reports that do not stop a load.
        config: The mapped configuration snapshot. Present only when
            ``outcome`` is ``ready``.
        config_sha256: The sha256 of the mapped config, for drift detection.
    """

    schema_version: str
    format_version: str
    package_dir: Path
    base_dir: Path
    outcome: str
    if_unsupported: str
    if_unverified: str
    stopped_by: Sequence[str]
    entry: Mapping[str, Any]
    agent: Mapping[str, Any]
    lock: Mapping[str, Any]
    adapter: Mapping[str, Any] | None
    variables: Sequence[Mapping[str, Any]]
    model: Mapping[str, Any]
    declared: Sequence[AgentPackageDeclaredItem]
    digests: Sequence[AgentPackageDigestCheck]
    notes: Sequence[Mapping[str, Any]]
    config: _FabricConfigSnapshot | None
    config_sha256: str | None
    _fields = frozenset(
        {
            "schema_version",
            "format_version",
            "package_dir",
            "base_dir",
            "outcome",
            "if_unsupported",
            "if_unverified",
            "stopped_by",
            "entry",
            "agent",
            "lock",
            "adapter",
            "variables",
            "model",
            "declared",
            "digests",
            "notes",
            "config",
            "config_sha256",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in ("package_dir", "base_dir", "outcome"):
            if field not in data:
                raise FabricConfigError(f"AgentPackageReport {field} is required")
        data["package_dir"] = Path(data["package_dir"])
        data["base_dir"] = Path(data["base_dir"])
        data.setdefault("adapter", None)
        data.setdefault("config_sha256", None)
        data["declared"] = tuple(
            AgentPackageDeclaredItem.from_mapping(line)
            for line in data.get("declared", [])
        )
        data["digests"] = tuple(
            AgentPackageDigestCheck.from_mapping(line)
            for line in data.get("digests", [])
        )
        config = data.get("config")
        data["config"] = (
            None if config is None else _FabricConfigSnapshot.from_mapping(config)
        )
        return data

    def to_config(self) -> FabricConfig:
        """Return the mapped configuration, ready for ``Fabric.plan`` or ``start_runtime``.

        Use ``base_dir`` of this report as the base directory of that call.

        Returns:
            The complete typed ``FabricConfig`` the package was mapped to.

        Raises:
            FabricConfigError: With code ``package_requirements_unmet`` when a
                required thing stopped the load, and with code
                ``package_config_required`` when the package was loaded
                without a config.
        """

        outcome = self._data["outcome"]
        config = self._data.get("config")
        if outcome == "stopped":
            raise FabricConfigError(
                "The agent package load stopped: "
                + ", ".join(self._data.get("stopped_by", ()))
                + ". Each id is a line of the report's declared list, with the"
                ' reason. Load with if_unsupported="warn" to continue and decide from'
                " the report.",
                stage=_STAGE,
                code="package_requirements_unmet",
                details={"stopped_by": list(self._data.get("stopped_by", ()))},
            )
        if config is None:
            raise FabricConfigError(
                "The agent package was loaded without a config, so nothing was mapped",
                stage=_STAGE,
                code="package_config_required",
            )
        return FabricConfig.from_mapping(config.to_mapping())


def load_agent_package(
    package_dir: str | os.PathLike[str],
    config: FabricConfig | None = None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    entry: str | os.PathLike[str] | None = None,
    variables: Mapping[str, str | int | float | bool] | None = None,
    instruction_mode: Literal["replace", "append"] | None = None,
    if_unsupported: Literal["stop", "warn"] = "stop",
    if_unverified: Literal["warn", "stop"] = "warn",
    check_machine: bool = False,
    facts: Mapping[str, Any] | None = None,
    expected_lock_sha256: str | None = None,
    bindings: Mapping[str, str] | None = None,
) -> AgentPackageReport:
    """Read, check and map one agent package. Fetches nothing.

    The package folder holds ``agent.json`` or ``agent-definition.json``, the
    lock file ``oadl.lock.json`` beside it, and the parts. The package is
    already resolved and locked. This function resolves nothing, installs
    nothing and never uses the network.

    Every part is checked against the digest the lock pins. The check happens
    once, at load time. It does not protect the files for the life of a
    runtime.

    The lock sits in the folder it describes, so the check shows that the
    parts match that lock and no more. Pass ``expected_lock_sha256`` to anchor
    the lock to a digest held somewhere else, for example in a registry.

    The call is synchronous. With ``check_machine=True`` it can take a few
    seconds; an async caller uses ``asyncio.to_thread``.

    Args:
        package_dir: The package folder.
        config: The caller's complete typed ``FabricConfig``. It holds what a
            package has no field for, such as the adapter, the model endpoint
            and the environment. ``metadata.name`` must equal the entry
            document's name. To let the package choose among models you
            serve, leave ``models["default"]`` out and name each ``models``
            entry with the text a document gives as its default model, for
            example ``"ollama::mistral-small3.2:24b"``. A config that holds
            one model under another name keeps it as the model. With ``None``
            the package is verified and reported, and nothing is mapped.
        base_dir: Base directory for resolving relative config paths.
        entry: The entry document, as a path inside the package, when it is
            not at the root. For example a definition kept at
            ``variants/short/agent-definition.json``.
        variables: Values for the agent's variables, by name. They win over a
            definition's values and over defaults.
        instruction_mode: How the instruction text is applied. ``None`` uses
            ``replace`` when the adapter offers it, else ``append``.
        if_unsupported: What to do about a required thing whose status is
            ``not_mapped`` or ``not_met``. Either ``stop`` or ``warn``.
        if_unverified: What to do about a required thing whose status is
            ``could_not_be_checked``. Either ``warn`` or ``stop``.
        check_machine: Probe this machine for runtime facts such as
            processors, memory, free storage, GPUs and program versions. It
            does not look for installed software packages.
        facts: Facts the caller knows about the place the agent runs. They
            win over probed facts, key by key. The ``packages`` fact lists
            the installed software packages by registry, for example
            ``{"pypi": {"faster-whisper": "1.2.1"}}``. The loader compares
            it with the version the lock pins for each package the agent
            declares. It looks for no package itself, so a required package
            of a registry the fact does not name stops a load under
            ``if_unsupported="stop"``.
        expected_lock_sha256: The sha256 the caller holds for the lock file,
            as 64 lowercase hex characters. A ``sha256`` prefix with its
            colon is accepted before them. The load is refused with
            ``package_lock_digest_mismatch`` when the lock file in the package
            has another digest. A report's ``lock["sha256"]`` is a value to
            keep for this.
        bindings: Targets of your own for the custom tools and the called
            agents the package declares, keyed by the place of the declared
            thing. A key is ``custom_tools.<name>`` or ``agents.<name>`` for
            the agent that runs. For a thing a called agent declares, the
            same follows that agent's key, as in
            ``agents.<name>.custom_tools.<tool>``. A value is one line of
            text that names your target, for example a Python reference or a
            registered name. The loader never imports, resolves or runs it.
            A bound thing has the status ``bound`` and does not stop the
            load. With no binding it stops a load under ``if_unsupported="stop"``.
            A key that names nothing the package declares is refused with
            ``package_binding_undeclared``. A binding for a called agent
            accounts for nothing else that agent declares. Each such thing
            has a line of its own in the report. The mapped config does not
            hold the bindings. They are in the report.

    Returns:
        An ``AgentPackageReport``. Its ``outcome`` is ``ready``, ``stopped``
        or ``verified_only``.

    Raises:
        FabricConfigError: If the package is refused. ``stage`` is
            ``agent_package``, ``code`` is a stable refusal code such as
            ``package_digest_mismatch`` and ``details["subject"]`` names the
            thing it is about. A mistake in the config itself, such as an
            unknown adapter, is raised as ``Fabric.plan`` raises it, with no
            stage and no code.
        FabricNativeUnavailableError: If the native extension is not
            installed.
    """

    options: dict[str, Any] = {
        "if_unsupported": if_unsupported,
        "if_unverified": if_unverified,
        "check_machine": check_machine,
    }
    if entry is not None:
        options["entry"] = _option_path("entry", entry)
    if variables is not None:
        options["variables"] = _option_mapping("variables", variables)
    if instruction_mode is not None:
        options["instruction_mode"] = instruction_mode
    if facts is not None:
        options["facts"] = _option_mapping("facts", facts)
    if expected_lock_sha256 is not None:
        options["expected_lock_sha256"] = expected_lock_sha256
    if bindings is not None:
        options["bindings"] = _option_mapping("bindings", bindings)
    return AgentPackageReport.from_mapping(
        _load_report(package_dir, config, base_dir=base_dir, options=options)
    )


# The keyword options of ``load_agent_package`` by name. An options file of
# ``fabric-agent-ctl`` and the ``options`` of ``Fabric.start_agent_package``
# may hold these keys and no other.
_PACKAGE_OPTIONS = frozenset(inspect.signature(load_agent_package).parameters) - {
    "package_dir",
    "config",
    "base_dir",
}


def _option_mapping(name: str, value: Any) -> dict[str, Any]:
    """Copy an option that is given as a mapping, or refuse it by name."""

    if not isinstance(value, Mapping):
        raise FabricConfigError(
            f"The agent package option {name} must be a mapping",
            stage=_STAGE,
            code="package_options_invalid",
            details={"subject": name},
        )
    return dict(value)


def _load_report(
    package_dir: str | os.PathLike[str],
    config: FabricConfig | None,
    *,
    base_dir: str | os.PathLike[str] | None,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Call the native loader and return the report as the core wrote it."""

    from nemo_fabric.client import Fabric, _config_json

    config_json = None if config is None else _config_json(config)
    try:
        options_json = json.dumps(dict(options), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise FabricConfigError(
            f"The agent package options are not JSON-compatible: {error}",
            stage=_STAGE,
            code="package_options_invalid",
            details={"subject": "options"},
        ) from error

    native = Fabric()._require_native_module("load_agent_package")
    try:
        raw = native.load_agent_package(
            os.fspath(package_dir),
            config_json,
            options_json,
            None if base_dir is None else os.fspath(base_dir),
        )
        return json.loads(raw)
    except FabricError:
        raise
    except Exception as error:
        code = getattr(error, "code", None)
        if not isinstance(code, str):
            # Not a package refusal: a mistake in the config itself, raised as
            # ``Fabric.plan`` raises it.
            raise FabricConfigError(str(error)) from error
        details = getattr(error, "details", None)
        raise FabricConfigError(
            str(error),
            stage=_STAGE,
            code=code,
            details=details if isinstance(details, Mapping) else None,
        ) from error


def _option_path(name: str, value: Any) -> str:
    """Return an option that is given as a path as text, or refuse it by name."""

    try:
        return os.fspath(value)
    except TypeError as error:
        raise FabricConfigError(
            f"The agent package option {name} must be a path",
            stage=_STAGE,
            code="package_options_invalid",
            details={"subject": name},
        ) from error
