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


class AgentPackageCheckerResult(FabricMapping):
    """What the checker of one agent package load answered.

    Attributes:
        decision: ``allow`` or ``deny``. A checker that failed is a ``deny``.
        failure: Why the checker counts as failed: ``not_started``,
            ``timed_out``, ``output_too_large``, ``unexpected_exit`` or
            ``invalid_answer``. ``None`` when it answered as it must.
        exit_code: The exit code of the checker. ``None`` when it was not
            started, ran past the time limit, wrote more than it may, or was
            ended by a signal.
        detail: More about a failure: why the checker could not be started,
            that it wrote more than it may, or else the end of what it wrote
            to standard error.
        answer: The JSON object the checker wrote. The order of its keys is
            not kept, and a number is kept as a JSON number. Each read gives
            a plain copy. ``None`` when the checker wrote no single JSON
            object, was not started, ran past the time limit or wrote more
            than it may.
    """

    decision: str
    failure: str | None
    exit_code: int | None
    detail: str | None
    answer: Mapping[str, Any] | None
    _fields = frozenset({"decision", "failure", "exit_code", "detail", "answer"})
    _json_fields = frozenset({"answer"})


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
            required thing stopped the load, ``denied`` when the checker did
            not allow the start, ``verified_only`` when no config was given.
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
            Present when ``outcome`` is ``ready`` or ``denied``.
        checker_result: What the checker named by the ``checker`` option
            answered, as an ``AgentPackageCheckerResult``, or ``None`` when no
            checker was asked.
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
    checker_result: AgentPackageCheckerResult | None
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
            "checker_result",
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
        result = data.get("checker_result")
        data["checker_result"] = (
            None if result is None else AgentPackageCheckerResult.from_mapping(result)
        )
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
                required thing stopped the load. With code
                ``package_check_denied`` when the checker did not allow the
                start, and ``details["answer"]`` holds its answer. With code
                ``package_check_failed`` when the checker failed, and
                ``details["failure"]`` and ``details["detail"]`` say how.
                With code
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
        if outcome == "denied":
            result = self["checker_result"]
            if result["failure"] is None:
                raise FabricConfigError(
                    "The checker did not allow the agent package to start. The"
                    " report's checker_result holds its answer.",
                    stage=_STAGE,
                    code="package_check_denied",
                    details={"answer": result["answer"]},
                )
            raise FabricConfigError(
                f"The checker of the agent package failed ({result['failure']}), so"
                " the package may not start.",
                stage=_STAGE,
                code="package_check_failed",
                details={
                    key: result[key]
                    for key in ("failure", "exit_code", "detail", "answer")
                },
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
    checker: Sequence[str] | None = None,
    checker_timeout_seconds: float | None = None,
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
    seconds, and with ``checker`` as long as the checker runs; an async caller
    uses ``asyncio.to_thread``.

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
        checker: A program that decides whether the loaded package may
            start: the program, then its arguments. ``None`` asks no
            checker, and an empty list is refused. It is asked once, after
            the package passed its own checks and was mapped, and it is
            started with no shell. It gets the package folder in
            ``NEMO_FABRIC_PACKAGE_DIR``, the path of a copy of the load
            report without its config in ``NEMO_FABRIC_PACKAGE_REPORT`` and
            the name of the entry document in ``NEMO_FABRIC_AGENT_NAME``.
            Nothing is added to its arguments. Of your environment it keeps
            operating-system variables such as ``PATH`` and ``HOME``, and no
            other. It writes one JSON object to standard output. Exit code 0
            with ``"decision": "allow"`` lets the package start, and exit
            code 2 with ``"decision": "deny"`` does not. Anything else is a
            deny, reported as a checker that failed. The report holds the
            answer in ``checker_result``. A load the checker does not allow
            has the outcome ``denied`` and no config.
        checker_timeout_seconds: The time limit of the checker, in seconds.
            ``None`` uses 30. A checker that runs longer is ended, and the
            load is denied.

    Returns:
        An ``AgentPackageReport``. Its ``outcome`` is ``ready``, ``stopped``,
        ``denied`` or ``verified_only``.

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
    if checker is not None:
        # One text is a sequence of characters, and is not a program with arguments.
        if isinstance(checker, str) or not isinstance(checker, Sequence):
            raise FabricConfigError(
                "The agent package option checker must be a list: the program,"
                " then its arguments",
                stage=_STAGE,
                code="package_options_invalid",
                details={"subject": "checker"},
            )
        options["checker"] = list(checker)
    if checker_timeout_seconds is not None:
        options["checker_timeout_seconds"] = checker_timeout_seconds
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


class AgentPackageFetchReport(FabricMapping):
    """What one agent package fetch placed and what it left.

    Attributes:
        package_dir: The folder to load, absolute. It is ``out_dir`` when
            something was placed, else the package folder as it was given.
        lock: The lock file of that folder, with its path, sha256, generator
            and entry count. Keep ``lock["sha256"]`` and pass it to
            ``load_agent_package`` as ``expected_lock_sha256``.
        fetched: One row for each lock entry whose content was fetched,
            placed and checked against the digest the lock pins. A row has
            ``address``, ``entry_type``, ``source_url``, ``source_subpath``,
            ``commit``, ``digest`` and ``path``.
        not_fetched: One row for each lock entry whose content is still not
            in the folder. A row has ``address``, ``entry_type``,
            ``source_type``, ``source_url`` and ``reason``.
    """

    package_dir: Path
    lock: Mapping[str, Any]
    fetched: Sequence[Mapping[str, Any]]
    not_fetched: Sequence[Mapping[str, Any]]
    _fields = frozenset({"package_dir", "lock", "fetched", "not_fetched"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "package_dir" not in data:
            raise FabricConfigError("AgentPackageFetchReport package_dir is required")
        data["package_dir"] = Path(data["package_dir"])
        return data


def fetch_agent_package(
    package_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    entry: str | os.PathLike[str] | None = None,
    expected_lock_sha256: str | None = None,
    oci_layout: str | os.PathLike[str] | None = None,
) -> AgentPackageFetchReport:
    """Fetch the content the lock of an agent package pins.

    A lock can pin a part, a called agent or the agent of a definition from a
    git repository or an OCI registry without the folder holding its bytes.
    This function gets those bytes and nothing else: it resolves no version,
    writes no lock and installs nothing. ``load_agent_package`` then loads
    the result, and still never uses the network.

    The package is read first, so a package that a load would refuse is
    refused before any network use. Each pinned commit is fetched with the
    ``git`` program and the sign-in it already has. Git's own prompt is
    turned off, and only the ``https``, ``ssh`` and ``file`` transports are
    used. A program git starts, such as ``ssh``, can still ask on a terminal.
    Each folder is read as the bytes git stores and placed at the path of its
    lock address in a copy of the package. The copy is then read like any
    package, so every entry in it has the digest its lock entry pins. Only
    then does the copy become ``out_dir``. A fetch that is refused leaves
    nothing.

    The copy holds what the lock pins: the lock file, the entry document and
    the content of every entry the folder already holds. A file of the
    package folder that no entry pins is not copied.

    No registry is asked for OCI content. It is read from ``oci_layout``, a
    folder in the OCI image layout form that you fill with the tool you sign
    in to the registry with. The artifact is the manifest the lock's digest
    names, with one layer that is a tar archive and is not compressed. Its
    files are unpacked at the path of the lock address. Its blobs are kept
    in ``.oci`` at the root of the package, and a load uses them to check
    the files against the lock's digest again. Keeping them there is how
    NeMo Fabric does it: the package format does not say where they go.

    The call is synchronous and can take as long as the fetches do. An async
    caller uses ``asyncio.to_thread``.

    Args:
        package_dir: The package folder. It is never written to.
        out_dir: Where the complete package is written. It must be missing or
            empty, and outside ``package_dir``.
        entry: The entry document, as a path inside the package, when it is
            not at the root.
        expected_lock_sha256: The sha256 the caller holds for the lock file,
            as for ``load_agent_package``. A lock file with another digest is
            refused before anything is fetched.
        oci_layout: A folder in the OCI image layout form that holds the
            artifacts the lock pins from an OCI registry. With ``None``, OCI
            content is not fetched and is listed in ``not_fetched``.

    Returns:
        An ``AgentPackageFetchReport``. Load its ``package_dir``. When
        nothing was placed, that is ``package_dir`` as it was given, and
        ``out_dir`` is not created.

    Raises:
        FabricConfigError: If the package or the fetch is refused. ``stage``
            is ``agent_package`` and ``details["subject"]`` names the thing
            the refusal is about. ``code`` is one of the codes a load uses,
            such as ``package_digest_mismatch`` for fetched bytes that do not
            have the pinned digest, or ``package_file_unreadable`` for an
            artifact in a form that is not read. It is
            ``package_fetch_failed`` when a required entry cannot be
            fetched, and ``package_out_dir_in_use`` when ``out_dir`` holds
            something.
            An optional entry that cannot be fetched is not an error. It is a
            row of ``not_fetched``. Fetched bytes that do not have the
            pinned digest are an error for an optional entry too.
        FabricNativeUnavailableError: If the native extension is not
            installed.
    """

    from nemo_fabric.client import Fabric

    folder = os.fspath(package_dir)
    if folder.startswith("git::"):
        raise FabricConfigError(
            "A package in a git repository is fetched with"
            " fetch_agent_package_from_git",
            stage=_STAGE,
            code="package_options_invalid",
            details={"subject": "package_dir"},
        )
    options = _fetch_options(entry, expected_lock_sha256, oci_layout)
    native = Fabric()._require_native_module("fetch_agent_package")
    return _fetch_report(
        lambda: native.fetch_agent_package(folder, os.fspath(out_dir), options)
    )


def fetch_agent_package_from_git(
    source: str,
    ref: str,
    out_dir: str | os.PathLike[str],
    *,
    entry: str | os.PathLike[str] | None = None,
    expected_lock_sha256: str | None = None,
    oci_layout: str | os.PathLike[str] | None = None,
) -> AgentPackageFetchReport:
    """Fetch an agent package that is a folder in a git repository.

    The folder is read at ``ref`` as the bytes git stores and written whole
    to ``out_dir``. It is then read as a package and completed from its own
    lock, as ``fetch_agent_package`` completes a folder. A folder that is not
    a package is refused, and nothing is left in ``out_dir``.

    Nothing you hold pins a package that is fetched this way, and a tag or a
    branch is a name the repository can move. The first row of ``fetched``
    has the address ``$`` and the commit the folder was read at. To pin more,
    pass ``expected_lock_sha256``.

    The call is synchronous and can take as long as the fetches do. An async
    caller uses ``asyncio.to_thread``.

    Args:
        source: The git source of the package, written the way a document
            writes one: ``git::<repository>//<folder>``. It is text, never a
            path object.
        ref: The tag, the branch or the full commit to read the folder at.
        out_dir: Where the package is written. It must be missing or empty.
        entry: The entry document, as a path inside the package, when it is
            not at the root.
        expected_lock_sha256: The sha256 the caller holds for the lock file
            of the package. A lock file with another digest is refused.
        oci_layout: A folder in the OCI image layout form that holds the
            artifacts the lock pins from an OCI registry.

    Returns:
        An ``AgentPackageFetchReport``. Its ``package_dir`` is ``out_dir``.

    Raises:
        FabricConfigError: If the source, the package or the fetch is
            refused, with the codes ``fetch_agent_package`` uses. ``code`` is
            ``package_options_invalid`` when ``source`` is not a git source
            or is not text.
        FabricNativeUnavailableError: If the native extension is not
            installed.
    """

    from nemo_fabric.client import Fabric

    for name, value in (("source", source), ("ref", ref)):
        # A path object folds the ``//`` of a source into ``/``.
        if not isinstance(value, str):
            raise FabricConfigError(
                f"The agent package option {name} must be text",
                stage=_STAGE,
                code="package_options_invalid",
                details={"subject": name},
            )
    options = _fetch_options(entry, expected_lock_sha256, oci_layout)
    native = Fabric()._require_native_module("fetch_agent_package_from_git")
    return _fetch_report(
        lambda: native.fetch_agent_package_from_git(
            source, ref, os.fspath(out_dir), options
        )
    )


def _fetch_options(
    entry: str | os.PathLike[str] | None,
    expected_lock_sha256: str | None,
    oci_layout: str | os.PathLike[str] | None,
) -> str:
    """Write the options of a fetch as the JSON object the core reads."""

    options: dict[str, Any] = {}
    if entry is not None:
        options["entry"] = _option_path("entry", entry)
    if expected_lock_sha256 is not None:
        options["expected_lock_sha256"] = expected_lock_sha256
    if oci_layout is not None:
        options["oci_layout"] = _option_path("oci_layout", oci_layout)
    return json.dumps(options)


def _fetch_report(fetch: Callable[[], str]) -> AgentPackageFetchReport:
    """Run a native fetch and return its report, or raise its refusal."""

    try:
        raw = fetch()
    except FabricError:
        raise
    except Exception as error:
        details = getattr(error, "details", None)
        raise FabricConfigError(
            str(error),
            stage=_STAGE,
            code=getattr(error, "code", None),
            details=details if isinstance(details, Mapping) else None,
        ) from error
    return AgentPackageFetchReport.from_mapping(json.loads(raw))


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
