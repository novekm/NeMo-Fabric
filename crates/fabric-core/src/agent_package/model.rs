// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The data model shared by the package reader, the mapping and the tests.
//!
//! The `pub` types appear in the load report. The `pub(crate)` types are the
//! reader's own model and never leave the crate.

use std::collections::BTreeMap;
use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ---- pub: also in the report ----

/// The kind of document a package starts from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum DocumentKind {
    /// An agent (`agent.json`).
    Agent,
    /// A definition that points at one agent (`agent-definition.json`).
    AgentDefinition,
}

/// The type of one lock entry.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "kebab-case")]
pub enum EntryType {
    /// A skill folder.
    Skill,
    /// A plugin.
    Plugin,
    /// A connector (an MCP server).
    Connector,
    /// A hook folder.
    Hook,
    /// A software package fetched from its own registry.
    Package,
    /// A policy package.
    Policy,
    /// An agent document.
    Agent,
    /// A definition document.
    AgentDefinition,
    /// A pinned model.
    Model,
}

/// Where a lock entry says its content comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// A path inside the package.
    Local,
    /// A git repository.
    Git,
    /// An OCI registry.
    Oci,
    /// Another registry.
    Registry,
    /// Something already running, or a model: nothing is carried.
    Pointer,
}

/// The document that states a declared thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StatedBy {
    /// The agent.
    Agent,
    /// The definition that points at the agent.
    Definition,
    /// An agent the agent that runs calls, at any depth. Only a thing below
    /// a called agent the caller bound, or one that went to the adapter, has
    /// this value.
    CalledAgent,
}

/// What the loader established about one lock entry's content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Integrity {
    /// The bytes are here and have the pinned digest.
    Verified,
    /// The pin is an artifact digest that cannot be worked out from files.
    NotVerifiable,
    /// The entry points at something that is not carried.
    Pointer,
    /// Fetched content is not in the package.
    Absent,
    /// The entry sits below a document or part the loader did not read.
    NotChecked,
}

/// What a lock digest is a digest of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DigestKind {
    /// One file.
    File,
    /// A folder listing.
    Folder,
    /// A registry artifact.
    Artifact,
}

/// Where a variable's value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValueOrigin {
    /// Whoever started the agent.
    Start,
    /// The definition.
    Definition,
    /// The entry that points at the agent.
    Entry,
    /// The variable's default.
    Default,
}

/// What a load does about a required thing that is not met or not checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PackageAction {
    /// Stop the load.
    Stop,
    /// Report it and continue.
    Warn,
}

/// The kind of a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NoteCode {
    /// A limit was asked higher than allowed and the smaller number applies.
    LimitLowered,
    /// A `version` written on a local source has no effect.
    LocalVersionIgnored,
    /// The lock marks the entry as retracted.
    Retracted,
    /// Two identical entries count once.
    EntryKeptOnce,
    /// A part resolved through a registry: its recorded source was accepted as is.
    SourceNotCompared,
    /// A server in a carried connector's `mcp.json` has a key the loader does
    /// not read. The key has no effect on how the server is started.
    ConnectorKeyNotRead,
    /// A called agent has a variable with no value, or a value that does not
    /// fit its field. That agent could not be started. Its caller still loads.
    CalledAgentCannotStart,
}

/// One variable with the value it has in this load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VariableValue {
    /// Variable name.
    pub name: String,
    /// Declared type: `string`, `number` or `boolean`.
    #[serde(rename = "type")]
    pub value_type: String,
    /// The value used.
    pub value: Value,
    /// Where the value came from.
    pub origin: ValueOrigin,
}

/// The interface pin of something already running.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct LockInterface {
    /// The version the running thing declares.
    pub declared_version: String,
    /// A hash of a connector's tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_hash: Option<String>,
    /// A digest of a running agent's card.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card_digest: Option<String>,
}

/// Something the loader reports that does not stop a load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Note {
    /// The kind of note.
    pub code: NoteCode,
    /// The thing the note is about.
    pub subject: String,
    /// The note in plain words.
    pub text: String,
}

/// One document the loader read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DocumentRecord {
    /// Agent or definition.
    pub kind: DocumentKind,
    /// The document's `name`.
    pub name: String,
    /// The document's lock address.
    pub address: String,
    /// The file that was read.
    pub path: PathBuf,
    /// The digest of the file's bytes.
    pub digest: String,
}

/// The lock file the loader checked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct LockRecord {
    /// The lock file.
    pub path: PathBuf,
    /// The sha256 of the lock file's bytes, as lowercase hex.
    pub sha256: String,
    /// The tool that wrote the lock, when the lock names it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,
    /// How many entries the lock has.
    pub entries: usize,
}

// ---- pub(crate): the reader's model ----

/// A field of a document that holds references to parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PartField {
    Skills,
    Plugins,
    Connectors,
    Hooks,
    Packages,
    Policies,
    Agents,
}

impl PartField {
    /// Every part field, in the order the format lists them.
    pub(crate) const ALL: [Self; 7] = [
        Self::Skills,
        Self::Plugins,
        Self::Connectors,
        Self::Hooks,
        Self::Packages,
        Self::Policies,
        Self::Agents,
    ];

    /// The field's key in a document.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Skills => "skills",
            Self::Plugins => "plugins",
            Self::Connectors => "connectors",
            Self::Hooks => "hooks",
            Self::Packages => "packages",
            Self::Policies => "policies",
            Self::Agents => "agents",
        }
    }

    /// The type of the lock entry a reference under this field resolves to.
    pub(crate) const fn entry_type(self) -> EntryType {
        match self {
            Self::Skills => EntryType::Skill,
            Self::Plugins => EntryType::Plugin,
            Self::Connectors => EntryType::Connector,
            Self::Hooks => EntryType::Hook,
            Self::Packages => EntryType::Package,
            Self::Policies => EntryType::Policy,
            Self::Agents => EntryType::Agent,
        }
    }
}

/// The status of one need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NeedStatus {
    MetHere,
    MetRemotely,
    NotMet,
    CouldNotBeChecked,
}

impl NeedStatus {
    /// The format's text for the status. The tests compare it with the format's cases.
    #[cfg(test)]
    pub(crate) fn format_text(self, connector: Option<&str>) -> String {
        match self {
            Self::MetHere => "met here".to_string(),
            Self::MetRemotely => match connector {
                Some(name) => format!("met remotely through connector {name}"),
                None => "met remotely".to_string(),
            },
            Self::NotMet => "not met".to_string(),
            Self::CouldNotBeChecked => "could not be checked".to_string(),
        }
    }
}

/// The files of one verified folder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FolderListing {
    /// The folder digest.
    pub digest: String,
    /// Relative path to `sha256:...` file digest.
    pub files: BTreeMap<String, String>,
}

/// One lock entry, joined with the reference that reaches it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct LockedPart {
    pub address: String,
    pub entry_type: EntryType,
    /// None for `$`, `$/agent` and `$/model`.
    pub field: Option<PartField>,
    pub name: Option<String>,
    /// Address of the document that declares it.
    pub owner: String,
    /// Definition only for a part the definition adds.
    pub stated_by: StatedBy,
    pub optional: bool,
    /// JSON Pointer of the entry in its document.
    pub pointer: Option<String>,
    /// The reference as written.
    pub source_text: Option<String>,
    pub version_text: Option<String>,
    pub source_type: SourceType,
    pub source_url: Option<String>,
    pub source_subpath: Option<String>,
    pub registry: Option<String>,
    /// From the lock.
    pub version: String,
    pub commit: Option<String>,
    pub digest: Option<String>,
    pub digest_kind: Option<DigestKind>,
    pub interface: Option<LockInterface>,
    pub retracted: Option<String>,
    pub dependencies: Vec<String>,
    /// Absolute: a file for a document, a folder or a file for a part.
    pub path: Option<PathBuf>,
    pub integrity: Integrity,
    /// The entry's variables, references filled.
    pub values: Option<Map<String, Value>>,
}

/// An optional reference that the lock leaves out. It is reported and never stops a start.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct LeftOutPart {
    pub field: PartField,
    pub name: String,
    pub owner: String,
    pub stated_by: StatedBy,
    pub pointer: String,
    pub source_text: String,
}

/// One piece of the instruction text and the document it came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct InstructionPart {
    pub from: StatedBy,
    pub name: String,
    pub text: String,
}

/// What reading a definition together with its agent did.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) struct TogetherSummary {
    pub added: Vec<String>,
    pub kept_once: Vec<String>,
    pub instructions: Vec<InstructionPart>,
}

/// One agent another agent lists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct CalledAgent {
    pub name: String,
    pub address: String,
    pub under: String,
    pub limits: Map<String, Value>,
    pub model_source: Option<String>,
    /// `called agent` or `caller`.
    pub model_from: String,
    pub values: Vec<VariableValue>,
    pub notes: Vec<Note>,
    /// The names that lead to it from the agent that runs, its own name last.
    pub place: Vec<String>,
    /// The `name` its document gives. `name` is the key its caller lists it under.
    pub document: String,
    /// Its document with a value for each variable. The mapping lists what it declares.
    pub filled: Value,
    /// The custom tools its document declares, by name.
    pub custom_tools: Vec<String>,
    /// The agents its document lists.
    pub agents: Vec<ListedAgent>,
}

/// One entry of a called agent's `agents`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ListedAgent {
    pub name: String,
    pub optional: bool,
    /// The lock entry the reference leads to. None: the lock leaves it out.
    pub address: Option<String>,
}

/// What the caller asks the reader for.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub(crate) struct ReadRequest {
    pub entry: Option<PathBuf>,
    pub variables: Map<String, Value>,
    /// The sha256 the caller holds for the lock file. The lock must have it.
    pub expected_lock_sha256: Option<String>,
}

/// Everything the reader established about one package.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PackageReading {
    /// Absolute. The boundary for every local path.
    pub package_dir: PathBuf,
    /// The folder of the entry document and its lock.
    pub entry_dir: PathBuf,
    pub entry: DocumentRecord,
    /// Equals `entry` when the entry is an agent.
    pub agent_record: DocumentRecord,
    pub lock: LockRecord,
    /// As written.
    pub definition: Option<Value>,
    pub agent_as_written: Value,
    /// The agent that runs: values filled, read together with the definition.
    pub agent: Value,
    pub variables: Vec<VariableValue>,
    /// Every lock entry, sorted by address.
    pub parts: Vec<LockedPart>,
    pub left_out: Vec<LeftOutPart>,
    pub called: Vec<CalledAgent>,
    /// Address to the files of each verified folder.
    pub listings: BTreeMap<String, FolderListing>,
    pub together: Option<TogetherSummary>,
    pub notes: Vec<Note>,
}

impl PackageReading {
    /// The lock entry for a root-level part of the agent that runs.
    ///
    /// The entry sits at the address of the field and the name, below the document that
    /// states the part. A part that an earlier path also reaches has its one entry at that
    /// earlier address: it is then found by what the reference points at.
    pub(crate) fn part(&self, field: PartField, name: &str) -> Option<&LockedPart> {
        let segments = [field.as_str(), name];
        let mut owners = vec![self.agent_record.address.as_str()];
        if self.entry.address != self.agent_record.address {
            owners.push(self.entry.address.as_str());
        }
        for owner in &owners {
            let wanted = super::paths::address(owner, &segments);
            if let Some(part) = self.parts.iter().find(|part| part.address == wanted) {
                return Some(part);
            }
        }
        self.shared_part(field, name)
    }

    /// The entry of a root-level part that sits at another path's address.
    fn shared_part(&self, field: PartField, name: &str) -> Option<&LockedPart> {
        let written = |document: &Value| document.get(field.as_str())?.get(name).cloned();
        // The agent's own entry wins: a definition entry of the same name is identical to it.
        let (reference, stated_in) = match written(&self.agent_as_written) {
            Some(reference) => (reference, &self.agent_record),
            None => (written(self.definition.as_ref()?)?, &self.entry),
        };
        // One folder used with two sets of values has two entries. The one with this
        // reference's values is the one meant.
        let values = self
            .agent
            .get(field.as_str())
            .and_then(|entries| entries.get(name))
            .and_then(|entry| entry.get("variables"))
            .and_then(Value::as_object);
        self.referenced_part(field, &reference, stated_in.path.parent()?, values)
    }

    /// The lock entry for a part a called agent states.
    ///
    /// As for a root-level part, the entry sits at the address of the field and the name
    /// below the called agent, or at the address of an earlier path that reaches the same
    /// part: two called agents that use one skill folder share one entry.
    pub(crate) fn called_part(
        &self,
        called: &CalledAgent,
        field: PartField,
        name: &str,
    ) -> Option<&LockedPart> {
        let wanted = super::paths::address(&called.address, &[field.as_str(), name]);
        if let Some(part) = self.parts.iter().find(|part| part.address == wanted) {
            return Some(part);
        }
        let document = self
            .parts
            .iter()
            .find(|part| part.address == called.address)?;
        let reference = called.filled.get(field.as_str())?.get(name)?;
        let values = reference.get("variables").and_then(Value::as_object);
        self.referenced_part(
            field,
            reference,
            document.path.as_deref()?.parent()?,
            values,
        )
    }

    /// The entry a reference leads to, found by what the reference points at. `folder`
    /// holds the document that states the reference. `values` are the values the
    /// reference gives, with every variable reference filled.
    fn referenced_part(
        &self,
        field: PartField,
        reference: &Value,
        folder: &std::path::Path,
        values: Option<&Map<String, Value>>,
    ) -> Option<&LockedPart> {
        let source = reference.get("source")?.as_str()?;
        let wanted_type = field.entry_type();
        let mut candidates: Vec<&LockedPart> = match source.strip_prefix("./") {
            Some(relative) => {
                let target = super::paths::normalize_inside(&self.package_dir, folder, relative)?;
                let document = target.join("agent.json");
                self.parts
                    .iter()
                    .filter(|part| part.source_type == SourceType::Local)
                    .filter(|part| {
                        part.path.as_deref() == Some(target.as_path())
                            || (wanted_type == EntryType::Agent
                                && part.path.as_deref() == Some(document.as_path()))
                    })
                    .collect()
            }
            None => {
                let version = reference.get("version").and_then(Value::as_str);
                self.parts
                    .iter()
                    .filter(|part| part.source_type != SourceType::Local)
                    .filter(|part| {
                        part.source_text.as_deref() == Some(source)
                            && part.version_text.as_deref() == version
                    })
                    .collect()
            }
        };
        candidates.retain(|part| part.entry_type == wanted_type);
        // With no entry that has these values, any has the same bytes.
        candidates
            .iter()
            .find(|part| part.values.as_ref() == values)
            .or(candidates.first())
            .copied()
    }
}

/// Facts about one load. Member names match the format's offline cases, so a
/// case's `facts` deserializes as is.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Facts {
    pub cpus: Option<u64>,
    /// Size text: `64gb`.
    pub memory: Option<String>,
    pub storage: Option<String>,
    /// `none`, `internal` or `external`.
    pub network: Option<String>,
    /// The name this harness goes by.
    pub harness: Option<String>,
    /// Other names of this harness (filled by the loader).
    pub harness_aliases: Vec<String>,
    /// `{interface, memory, family, model}`.
    pub gpus: Option<Vec<Value>>,
    /// The same, plus `connector`.
    pub remote_gpus: Option<Vec<Value>>,
    /// Absent key: not on the machine. Null: there, version unknown.
    pub software: Option<BTreeMap<String, Option<String>>>,
    /// Installed software packages, by registry name and then by package
    /// name. Absent registry: not looked at. Absent package: not installed.
    /// Null: installed, version unknown.
    pub packages: Option<BTreeMap<String, BTreeMap<String, Option<String>>>>,
    /// `{name, capabilities, context_tokens, parameters_applied}`.
    pub model: Option<Value>,
    pub known: KnownNames,
}

/// Names the receiver knows, used to tell "not met" from "could not be checked".
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct KnownNames {
    pub interfaces: Option<Vec<String>>,
    pub families: Vec<String>,
    pub capabilities: Vec<String>,
    pub harnesses: Vec<String>,
}

/// One need with its status in this load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct NeedLine {
    /// The format's text: `runtime.compute.memory`, `definition.runtime.software.python3`.
    pub need: String,
    /// JSON Pointer in the stating document: `/runtime/compute/memory`.
    pub pointer: String,
    pub status: NeedStatus,
    /// For `MetRemotely`.
    pub connector: Option<String>,
    pub optional: bool,
    pub stated_by: StatedBy,
    pub document: String,
    /// The author's text, unchanged.
    pub reason: Option<String>,
    /// What was compared.
    pub detail: Option<String>,
}

/// Whether the needs let the agent start. `needs::decide` returns it.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Decision {
    pub starts: bool,
    pub required_not_met: Vec<String>,
    pub required_not_checked: Vec<String>,
}
