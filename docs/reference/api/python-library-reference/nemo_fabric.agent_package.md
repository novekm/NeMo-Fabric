---
title: "Agent Packages"
slug: "/reference/api/python-library-reference/agent-package"
description: "Load an agent package: check its parts against the lock and map it to a configuration."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.agent_package`

Load an agent package: a folder with an agent document, its lock and its parts.


---


## <kbd>function</kbd> `load_agent_package`

```python
load_agent_package(
    package_dir: 'str | PathLike[str]',
    config: 'FabricConfig | None' = None,
    base_dir: 'str | PathLike[str] | None' = None,
    entry: 'str | PathLike[str] | None' = None,
    variables: 'Mapping[str, str | int | float | bool] | None' = None,
    instruction_mode: "Literal['replace', 'append'] | None" = None,
    if_unsupported: "Literal['stop', 'warn']" = 'stop',
    if_unverified: "Literal['warn', 'stop']" = 'warn',
    check_machine: 'bool' = False,
    facts: 'Mapping[str, Any] | None' = None,
    expected_lock_sha256: 'str | None' = None,
    bindings: 'Mapping[str, str] | None' = None
) → AgentPackageReport
```

Read, check and map one agent package. Fetches nothing.

The package folder holds ``agent.json`` or ``agent-definition.json``, the lock file ``oadl.lock.json`` beside it, and the parts. The package is already resolved and locked. This function resolves nothing, installs nothing and never uses the network.

Every part is checked against the digest the lock pins. The check happens once, at load time. It does not protect the files for the life of a runtime.

The lock sits in the folder it describes, so the check shows that the parts match that lock and no more. Pass ``expected_lock_sha256`` to anchor the lock to a digest held somewhere else, for example in a registry.

The call is synchronous. With ``check_machine=True`` it can take a few seconds; an async caller uses ``asyncio.to_thread``.



**Args:**

 - <b>`package_dir`</b>:  The package folder.
 - <b>`config`</b>:  The caller's complete typed ``FabricConfig``. It holds what a  package has no field for, such as the adapter, the model endpoint  and the environment. ``metadata.name`` must equal the entry  document's name. To let the package choose among models you  serve, leave ``models["default"]`` out and name each ``models``  entry with the text a document gives as its default model, for
 - <b>`example ``"ollama`</b>: :mistral-small3.2:24b"``. A config that holds one model under another name keeps it as the model. With ``None`` the package is verified and reported, and nothing is mapped.
 - <b>`base_dir`</b>:  Base directory for resolving relative config paths.
 - <b>`entry`</b>:  The entry document, as a path inside the package, when it is  not at the root. For example a definition kept at  ``variants/short/agent-definition.json``.
 - <b>`variables`</b>:  Values for the agent's variables, by name. They win over a  definition's values and over defaults.
 - <b>`instruction_mode`</b>:  How the instruction text is applied. ``None`` uses  ``replace`` when the adapter offers it, else ``append``.
 - <b>`if_unsupported`</b>:  What to do about a required thing whose status is  ``not_mapped`` or ``not_met``. Either ``stop`` or ``warn``.
 - <b>`if_unverified`</b>:  What to do about a required thing whose status is  ``could_not_be_checked``. Either ``warn`` or ``stop``.
 - <b>`check_machine`</b>:  Probe this machine for runtime facts such as  processors, memory, free storage, GPUs and program versions. It  does not look for installed software packages.
 - <b>`facts`</b>:  Facts the caller knows about the place the agent runs. They  win over probed facts, key by key. The ``packages`` fact lists  the installed software packages by registry, for example
 - <b>```{"pypi"`</b>:  {"faster-whisper": "1.2.1"}}``. The loader compares it with the version the lock pins for each package the agent declares. It looks for no package itself, so a required package of a registry the fact does not name stops a load under ``if_unsupported="stop"``.
 - <b>`expected_lock_sha256`</b>:  The sha256 the caller holds for the lock file,  as 64 lowercase hex characters. A ``sha256`` prefix with its  colon is accepted before them. The load is refused with  ``package_lock_digest_mismatch`` when the lock file in the package  has another digest. A report's ``lock["sha256"]`` is a value to  keep for this.
 - <b>`bindings`</b>:  Targets of your own for the custom tools and the called  agents the package declares, keyed by the place of the declared  thing. A key is ``custom_tools.<name>`` or ``agents.<name>`` for  the agent that runs. For a thing a called agent declares, the  same follows that agent's key, as in  ``agents.<name>.custom_tools.<tool>``. A value is one line of  text that names your target, for example a Python reference or a  registered name. The loader never imports, resolves or runs it.  A bound thing has the status ``bound`` and does not stop the  load. With no binding it stops a load under ``if_unsupported="stop"``.  A key that names nothing the package declares is refused with  ``package_binding_undeclared``. A binding for a called agent  accounts for nothing else that agent declares. Each such thing  has a line of its own in the report. The mapped config does not  hold the bindings. They are in the report.



**Returns:**
 An ``AgentPackageReport``. Its ``outcome`` is ``ready``, ``stopped`` or ``verified_only``.



**Raises:**

 - <b>`FabricConfigError`</b>:  If the package is refused. ``stage`` is  ``agent_package``, ``code`` is a stable refusal code such as  ``package_digest_mismatch`` and ``details["subject"]`` names the  thing it is about. A mistake in the config itself, such as an  unknown adapter, is raised as ``Fabric.plan`` raises it, with no  stage and no code.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.


---


## <kbd>class</kbd> `AgentPackageDeclaredItem`

One thing a document declares, and what the load did with it.



**Attributes:**

 - <b>`id`</b>:  The stating document and a JSON Pointer into it, for example
 - <b>```agent`</b>: /skills/release-notes``.
 - <b>`kind`</b>:  The kind of declared thing, such as ``part``, ``limit`` or  ``need``.
 - <b>`stated_by`</b>:  The document that states it: ``agent`` or ``definition``.
 - <b>`document`</b>:  That document's name.
 - <b>`field`</b>:  The top-level field, for example ``skills`` or ``runtime``.
 - <b>`name`</b>:  The entry key or member name, as written.
 - <b>`address`</b>:  The lock address, for a part.
 - <b>`required`</b>:  ``False`` for a thing marked optional and for a thing that  is only reported.
 - <b>`status`</b>:  What the load established, such as ``mapped``,  ``not_mapped``, ``bound``, ``not_met`` or  ``could_not_be_checked``.
 - <b>`gap`</b>:  Why, when the status is ``not_mapped``.
 - <b>`connector`</b>:  The connector a need is met through, or the connector of a  connector tool.
 - <b>`effect`</b>:  What the load did about it: ``none``, ``warned`` or  ``stopped``.
 - <b>`mapped_to`</b>:  The config paths it was written to.
 - <b>`bound_to`</b>:  The caller's target, for a custom tool or a called agent  with a binding.
 - <b>`reason`</b>:  The author's reason, unchanged.
 - <b>`detail`</b>:  The loader's explanation.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `id` | `str` |
| `kind` | `str` |
| `stated_by` | `str` |
| `document` | `str` |
| `field` | `str` |
| `name` | `str \| None` |
| `address` | `str \| None` |
| `required` | `bool` |
| `status` | `str` |
| `gap` | `str \| None` |
| `connector` | `str \| None` |
| `effect` | `str` |
| `mapped_to` | `Sequence[str]` |
| `bound_to` | `str \| None` |
| `reason` | `str \| None` |
| `detail` | `str \| None` |

---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.




---


## <kbd>class</kbd> `AgentPackageDigestCheck`

One lock entry and what the load checked about it.



**Attributes:**

 - <b>`address`</b>:  The lock address.
 - <b>`entry_type`</b>:  The entry's type, such as ``skill`` or ``connector``.
 - <b>`source_type`</b>:  Where the lock says the content comes from.
 - <b>`source_url`</b>:  The recorded source address, with any password left out.
 - <b>`source_subpath`</b>:  The recorded path below the source address.
 - <b>`version`</b>:  The exact version the lock selected.
 - <b>`commit`</b>:  The git commit, for a git source.
 - <b>`digest`</b>:  The lock's digest.
 - <b>`digest_kind`</b>:  What the digest is a digest of.
 - <b>`interface`</b>:  The interface pin of something already running.
 - <b>`result`</b>:  What the load established about the content, such as  ``verified``, ``pointer`` or ``absent``.
 - <b>`path`</b>:  Where the bytes were verified.
 - <b>`retracted`</b>:  The lock's retraction text, when the entry is retracted.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `address` | `str` |
| `entry_type` | `str` |
| `source_type` | `str` |
| `source_url` | `str \| None` |
| `source_subpath` | `str \| None` |
| `version` | `str` |
| `commit` | `str \| None` |
| `digest` | `str \| None` |
| `digest_kind` | `str \| None` |
| `interface` | `Mapping[str, Any] \| None` |
| `result` | `str` |
| `path` | `str \| None` |
| `retracted` | `str \| None` |

---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.




---


## <kbd>class</kbd> `AgentPackageReport`

What one agent package load read, checked and mapped.

The statuses are valid for this load only. A runtime that is kept for many invocations loads the package again to get fresh statuses.



**Attributes:**

 - <b>`schema_version`</b>:  Report schema version.
 - <b>`format_version`</b>:  Version of the package format that was read.
 - <b>`package_dir`</b>:  The package folder, absolute.
 - <b>`base_dir`</b>:  The base directory that relative config paths resolve against.
 - <b>`outcome`</b>:  ``ready`` when the report holds a config, ``stopped`` when a  required thing stopped the load, ``verified_only`` when no config  was given.
 - <b>`if_unsupported`</b>:  The setting used for a required thing whose status is  ``not_mapped`` or ``not_met``.
 - <b>`if_unverified`</b>:  The setting used for a required thing whose status is  ``could_not_be_checked``.
 - <b>`stopped_by`</b>:  Ids of the declared things that stopped the load.
 - <b>`entry`</b>:  The entry document, with its kind, name, lock address, path  and digest.
 - <b>`agent`</b>:  The agent that runs. Equal to ``entry`` unless the entry  document is a definition.
 - <b>`lock`</b>:  The lock file, with its path, sha256, generator and entry count.
 - <b>`adapter`</b>:  The selected adapter and what its descriptor accepts, when a  config was given. ``adapter["extensions"]`` lists the root  extension keys the descriptor declares. The loader hands plugins,  hooks and called agents to an adapter under such a key, for  example ``nemo_fabric.plugins.v1``.
 - <b>`variables`</b>:  Every variable of the agent with its value and where the  value came from.
 - <b>`model`</b>:  What was configured for the model, and who chose it. The  caller's ``models["default"]`` is the model choice. When the  config sets none and has a ``models`` entry named with the text  of the document's default model, the loader moves that entry to  ``models["default"]`` and ``model["origin"]`` is ``package``.
 - <b>`declared`</b>:  Every declared thing with its status, as  ``AgentPackageDeclaredItem`` lines. ``mapped`` means the  loader wrote it into the config and the config planned on the  selected adapter. It does not show that the harness loaded it.  ``bound`` means the caller gave a binding for a custom tool or a  called agent. The line's ``bound_to`` is the caller's target.
 - <b>`digests`</b>:  Every lock entry with its digest, what was checked and the  path the bytes were verified at, as ``AgentPackageDigestCheck``  lines.
 - <b>`notes`</b>:  Things the loader reports that do not stop a load.
 - <b>`config`</b>:  The mapped configuration snapshot. Present only when  ``outcome`` is ``ready``.
 - <b>`config_sha256`</b>:  The sha256 of the mapped config, for drift detection.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `schema_version` | `str` |
| `format_version` | `str` |
| `package_dir` | `Path` |
| `base_dir` | `Path` |
| `outcome` | `str` |
| `if_unsupported` | `str` |
| `if_unverified` | `str` |
| `stopped_by` | `Sequence[str]` |
| `entry` | `Mapping[str, Any]` |
| `agent` | `Mapping[str, Any]` |
| `lock` | `Mapping[str, Any]` |
| `adapter` | `Mapping[str, Any] \| None` |
| `variables` | `Sequence[Mapping[str, Any]]` |
| `model` | `Mapping[str, Any]` |
| `declared` | `Sequence[AgentPackageDeclaredItem]` |
| `digests` | `Sequence[AgentPackageDigestCheck]` |
| `notes` | `Sequence[Mapping[str, Any]]` |
| `config` | `_FabricConfigSnapshot \| None` |
| `config_sha256` | `str \| None` |

---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>method</kbd> `to_config`

```python
def to_config() -> FabricConfig
```

Return the mapped configuration, ready for ``Fabric.plan`` or ``start_runtime``.

Use ``base_dir`` of this report as the base directory of that call.



**Returns:**
  The complete typed ``FabricConfig`` the package was mapped to.



**Raises:**

 - <b>`FabricConfigError`</b>:  With code ``package_requirements_unmet`` when a  required thing stopped the load, and with code  ``package_config_required`` when the package was loaded  without a config.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
