// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The load report: what was read, what was checked and what was mapped.

use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::digest::hex_text;
use super::model::{
    DigestKind, DocumentRecord, EntryType, Integrity, LockInterface, LockRecord, Note,
    PackageAction, PackageReading, SourceType, StatedBy, VariableValue,
};
use crate::config::{AdapterConfigField, FabricConfig, InstructionMode};
use crate::error::{FabricError, Result};

/// Schema version of the load report.
pub(super) const REPORT_SCHEMA_VERSION: &str = "fabric.agent_package_report/v1alpha1";

/// What one agent package load established.
///
/// The statuses are valid for this load only. A runtime that is kept for many
/// invocations loads again to get fresh statuses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AgentPackageReport {
    /// Report schema version.
    pub schema_version: String,
    /// Version of the package format that was read.
    pub format_version: String,
    /// The package folder, absolute.
    pub package_dir: PathBuf,
    /// The base directory that relative config paths resolve against.
    pub base_dir: PathBuf,
    /// Whether the load gave a config, stopped, or only verified the package.
    pub outcome: PackageOutcome,
    /// The setting used for a required thing that is not mapped or not met.
    pub if_unsupported: PackageAction,
    /// The setting used for a required thing that could not be checked.
    pub if_unverified: PackageAction,
    /// Ids of the declared things that stopped the load.
    pub stopped_by: Vec<String>,
    /// The entry document.
    pub entry: DocumentRecord,
    /// The agent that runs. Equal to `entry` unless the entry is a definition.
    pub agent: DocumentRecord,
    /// The lock file that was checked.
    pub lock: LockRecord,
    /// The selected adapter, when a config was given.
    pub adapter: Option<AdapterReport>,
    /// Every variable of the agent with the value it has in this load.
    pub variables: Vec<VariableValue>,
    /// What was configured for the model.
    pub model: ModelReport,
    /// Every declared thing, with its status.
    pub declared: Vec<DeclaredItem>,
    /// Every lock entry, with what was checked.
    pub digests: Vec<DigestCheck>,
    /// Things the loader reports that do not stop a load.
    pub notes: Vec<Note>,
    /// The mapped config. Present only when `outcome` is `ready`.
    pub config: Option<FabricConfig>,
    /// The sha256 of the mapped config, as lowercase hex, for drift detection.
    /// Present when `outcome` is `ready` or `denied`.
    pub config_sha256: Option<String>,
    /// What the checker named in the options answered. None: no checker was
    /// named, or the load ended before a checker is asked.
    pub checker_result: Option<CheckerResult>,
}

/// How a load ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PackageOutcome {
    /// The package was checked and mapped. The report holds a config.
    Ready,
    /// A required thing stopped the load. The report holds no config.
    Stopped,
    /// The package was checked and mapped, and the checker did not allow the
    /// start. The report holds no config.
    Denied,
    /// No config was given. The package was checked and nothing was mapped.
    VerifiedOnly,
}

/// What the checker answered about one loaded package, and what the load made of it.
///
/// The checker is a program the caller names in the options. It is asked once,
/// after the package passed its own checks and was mapped, and before anything
/// starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CheckerResult {
    /// Whether the package may start. A checker that failed is a `deny`.
    pub decision: CheckerDecision,
    /// Why the checker counts as failed. None: it answered as it must.
    pub failure: Option<CheckerFailure>,
    /// The exit code of the checker. None: it was not started, it ran past
    /// the time limit, it wrote more than it may, or a signal ended it.
    pub exit_code: Option<i32>,
    /// More about a failure, as text: why the checker could not be started,
    /// that it wrote more than it may, or else the end of what it wrote to
    /// standard error. None: the checker did not fail, or there is nothing
    /// to add.
    pub detail: Option<String>,
    /// The JSON object the checker wrote. The load reads its `decision` and
    /// nothing else. The order of its keys is not kept, and a number is kept
    /// as a JSON number, so a long whole number can lose digits. None: the
    /// checker wrote no single JSON object, or it was not started, ran past
    /// the time limit or wrote more than it may. What such a checker wrote
    /// is not read.
    pub answer: Option<serde_json::Map<String, Value>>,
}

/// Whether a loaded package may start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckerDecision {
    /// The checker exited with 0 and its answer has `decision` set to `allow`.
    Allow,
    /// The checker exited with 2 and its answer has `decision` set to `deny`,
    /// or the checker failed.
    Deny,
}

/// Why a checker counts as failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckerFailure {
    /// The program could not be started, or the report could not be written for it.
    NotStarted,
    /// The program ran longer than the time limit. The loader ended it.
    TimedOut,
    /// The program wrote more than 1 MiB to standard output or to standard
    /// error. The loader ends a program that is still running. The result
    /// is the same when the program had already ended by itself: it has no
    /// exit code and no answer.
    OutputTooLarge,
    /// The program ended with an exit code that is neither 0 nor 2, or a signal ended it.
    UnexpectedExit,
    /// The program wrote something that is not one JSON object, or a
    /// `decision` that does not fit its exit code.
    InvalidAnswer,
}

/// The adapter a package was mapped for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterReport {
    /// Selected adapter id.
    pub adapter_id: String,
    /// Config fields the adapter's descriptor says it accepts.
    pub accepts: Vec<AdapterConfigField>,
    /// Root extension keys the adapter's descriptor declares, for example
    /// `nemo_fabric.plugins.v1`. The loader hands plugins, hooks and called
    /// agents to an adapter under such a key.
    pub extensions: Vec<String>,
    /// The instruction mode the loader wrote, when it wrote instructions.
    pub instruction_mode: Option<InstructionMode>,
}

/// One thing a document declares, and what the loader did with it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DeclaredItem {
    /// The stating document and a JSON Pointer into it, for example
    /// `agent:/skills/release-notes`. For a thing a called agent declares, the
    /// pointer goes on below the entry that lists the called agent, for
    /// example `agent:/agents/captioner/custom_tools/request_approval`.
    pub id: String,
    /// The kind of declared thing.
    pub kind: DeclaredKind,
    /// The document that states it.
    pub stated_by: StatedBy,
    /// That document's name.
    pub document: String,
    /// The top-level field, for example `skills`, `limits`, `runtime` or `model`.
    pub field: String,
    /// The entry key or member name, as written.
    pub name: Option<String>,
    /// The lock address, for a part.
    pub address: Option<String>,
    /// False when the thing is marked optional, when it sits below a called
    /// agent that is marked optional, and for things that are only reported.
    pub required: bool,
    /// What the loader established.
    pub status: DeclaredStatus,
    /// Why, when the status is `not_mapped`.
    pub gap: Option<MappingGap>,
    /// The connector a need is met through, or the connector of a connector tool.
    pub connector: Option<String>,
    /// What the load did about it.
    pub effect: ItemEffect,
    /// The config paths it was written to.
    pub mapped_to: Vec<String>,
    /// The target the caller bound it to, for a custom tool or a called agent
    /// with a binding. It is the caller's text, unchanged: the loader did not
    /// import, resolve or run it.
    pub bound_to: Option<String>,
    /// The author's reason, unchanged.
    pub reason: Option<String>,
    /// The loader's explanation.
    pub detail: Option<String>,
}

/// The kind of a declared thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeclaredKind {
    /// A plain field such as `name`, `instructions`, an output or an extension.
    Field,
    /// A member of `limits`.
    Limit,
    /// A referenced part, a custom tool or a credential.
    Part,
    /// The values written on a part entry.
    PartValues,
    /// An entry of `tools`.
    Tool,
    /// A runtime need or a need from any model.
    Need,
    /// The default model or one of its settings.
    ModelSetting,
}

/// The status of a declared thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeclaredStatus {
    /// Written into the config, and the config planned on the selected adapter.
    /// It does not show that the harness loaded it.
    Mapped,
    /// Not written into the config. `gap` says why.
    NotMapped,
    /// A custom tool or a called agent the caller supplies. `bound_to` names
    /// the caller's target. Nothing was written into the config, and it does
    /// not show that the target exists.
    Bound,
    /// Nothing here could confirm it.
    CouldNotBeChecked,
    /// Read and left alone. It asks nothing of a receiver.
    Skipped,
    /// No config was given, so nothing was mapped.
    NotEvaluated,
    /// A need that is met on this machine, or a software package that is
    /// installed at the version the lock pins.
    MetHere,
    /// A need that is met through a named connector.
    MetRemotely,
    /// A need that is not met, or a software package that is not installed
    /// at the version the lock pins.
    NotMet,
}

/// Why a declared thing was not mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MappingGap {
    /// NeMo Fabric has no config field for it.
    NoFabricField,
    /// The selected adapter's descriptor does not list it.
    AdapterDoesNotAccept,
    /// Pinned, fetched content is not in the package.
    ContentAbsent,
    /// The content is pinned by an artifact digest.
    ContentNotVerifiable,
    /// An optional reference the lock leaves out.
    NotResolved,
    /// A source type the loader does not load.
    SourceTypeUnsupported,
    /// A running connector with no server entry in the caller's config, or a
    /// custom tool with no binding and no `tools.definitions` entry there.
    NotBound,
    /// A shape the loader or the adapters cannot take: an `mcp.json` the loader
    /// cannot read, a server that is not stdio, a skill, plugin or hook that
    /// is one file and not a folder, or a hook whose `hooks.json` names no
    /// format in `$schema`.
    UnsupportedShape,
    /// A name that is already taken: two carried servers with one key, or two
    /// skill folders with one folder name.
    KeyClash,
    /// A carried connector on a platform without `/bin/sh`.
    UnsupportedPlatform,
    /// The caller's config selects no model: it sets no `models.default`,
    /// names no `models` entry after the document's default model, and does
    /// not hold exactly one model.
    ModelNotConfigured,
    /// A number NeMo Fabric cannot hold, for example a turn limit above 32 bits.
    ValueOutOfRange,
}

/// What the load did about one declared thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ItemEffect {
    /// Nothing: it is mapped, met, optional, or only reported.
    None,
    /// It was reported and the load continued.
    Warned,
    /// It stopped the load.
    Stopped,
}

/// One lock entry and what the loader checked about it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DigestCheck {
    /// The lock address.
    pub address: String,
    /// The entry's type.
    pub entry_type: EntryType,
    /// Where the lock says the content comes from.
    pub source_type: SourceType,
    /// The recorded source address, with any password left out.
    pub source_url: Option<String>,
    /// The recorded path below the source address.
    pub source_subpath: Option<String>,
    /// The exact version the lock selected.
    pub version: String,
    /// The git commit, for a git source.
    pub commit: Option<String>,
    /// The lock's digest.
    pub digest: Option<String>,
    /// What the digest is a digest of.
    pub digest_kind: Option<DigestKind>,
    /// The interface pin of something already running.
    pub interface: Option<LockInterface>,
    /// What the loader established about the content.
    pub result: Integrity,
    /// Where the bytes were verified.
    pub path: Option<PathBuf>,
    /// The lock's retraction text, when the entry is retracted.
    pub retracted: Option<String>,
}

/// What was configured for the model. The model each run used is not reported here.
///
/// The caller's `models.default` is the model choice. A config that sets none
/// offers a model by naming a `models` entry with the text of the document's
/// default model, and the loader moves that entry to `models.default`. A
/// config that holds one model under another name has chosen that model. The
/// loader writes no model of its own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelReport {
    /// The document's default model, after variables.
    pub declared_source: Option<String>,
    /// The configured provider.
    pub provider: Option<String>,
    /// The configured model. None: the config names no model, and what the
    /// harness uses is not known here.
    pub model: Option<String>,
    /// Who chose the configured model.
    pub origin: ModelOrigin,
    /// Document model settings that were written into the config.
    pub parameters_applied: Vec<String>,
    /// Document model settings that were not written into the config.
    pub parameters_not_applied: Vec<String>,
}

/// Who chose the configured model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelOrigin {
    /// The caller's config selects the model: it sets `models.default`, or it
    /// holds exactly one model, which planning and the adapters run.
    Caller,
    /// The config sets no `models.default` and offers a model under the name
    /// of the document's default model. That entry is now `models.default`.
    Package,
    /// The config holds no model. The harness uses its own default when it
    /// has one.
    HarnessDefault,
    /// No config was given, so no model was configured.
    NotEvaluated,
}

/// One report line per lock entry.
pub(super) fn digest_checks(reading: &PackageReading) -> Vec<DigestCheck> {
    reading
        .parts
        .iter()
        .map(|part| DigestCheck {
            address: part.address.clone(),
            entry_type: part.entry_type,
            source_type: part.source_type,
            source_url: part
                .source_url
                .as_deref()
                .map(super::document::without_password),
            source_subpath: part.source_subpath.clone(),
            version: part.version.clone(),
            commit: part.commit.clone(),
            digest: part.digest.clone(),
            digest_kind: part.digest_kind,
            interface: part.interface.clone(),
            result: part.integrity,
            path: part.path.clone(),
            retracted: part.retracted.clone(),
        })
        .collect()
}

/// Apply the two settings to every item. Returns the outcome and the ids that stopped the load.
pub(super) fn apply_actions(
    items: &mut [DeclaredItem],
    if_unsupported: PackageAction,
    if_unverified: PackageAction,
) -> (PackageOutcome, Vec<String>) {
    let mut stopped_by = Vec::new();
    for item in items.iter_mut() {
        let action = match item.status {
            _ if !item.required => None,
            DeclaredStatus::NotMapped | DeclaredStatus::NotMet => Some(if_unsupported),
            DeclaredStatus::CouldNotBeChecked => Some(if_unverified),
            DeclaredStatus::Mapped
            | DeclaredStatus::Bound
            | DeclaredStatus::Skipped
            | DeclaredStatus::NotEvaluated
            | DeclaredStatus::MetHere
            | DeclaredStatus::MetRemotely => None,
        };
        item.effect = match action {
            Some(PackageAction::Stop) => {
                stopped_by.push(item.id.clone());
                ItemEffect::Stopped
            }
            Some(PackageAction::Warn) => ItemEffect::Warned,
            None => ItemEffect::None,
        };
    }
    let outcome = if stopped_by.is_empty() {
        PackageOutcome::Ready
    } else {
        PackageOutcome::Stopped
    };
    (outcome, stopped_by)
}

/// The sha256 of the config as compact JSON with keys sorted at every level, as lowercase hex.
pub(super) fn config_sha256(config: &FabricConfig) -> Result<String> {
    let value = serde_json::to_value(config).map_err(FabricError::SerializeJson)?;
    let encoded = serde_json::to_vec(&sorted(value)).map_err(FabricError::SerializeJson)?;
    Ok(hex_text(Sha256::digest(&encoded)))
}

/// Rebuild every object with its keys in sorted order, whatever map type backs `Value`.
fn sorted(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<(String, Value)> = object.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, sorted(value)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sorted).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, required: bool, status: DeclaredStatus) -> DeclaredItem {
        DeclaredItem {
            id: id.to_string(),
            kind: DeclaredKind::Part,
            stated_by: StatedBy::Agent,
            document: "agent".to_string(),
            field: "hooks".to_string(),
            name: None,
            address: None,
            required,
            status,
            gap: None,
            connector: None,
            effect: ItemEffect::None,
            mapped_to: Vec::new(),
            bound_to: None,
            reason: None,
            detail: None,
        }
    }

    #[test]
    fn actions_follow_status_required_and_setting() {
        use DeclaredStatus::*;
        use ItemEffect::{None as NoEffect, Stopped, Warned};
        use PackageAction::{Stop, Warn};
        // (status, required, if_unsupported, if_unverified, expected effect)
        let table = [
            (NotMapped, true, Stop, Warn, Stopped),
            (NotMapped, true, Warn, Stop, Warned),
            (NotMapped, false, Stop, Stop, NoEffect),
            (NotMet, true, Stop, Warn, Stopped),
            (NotMet, true, Warn, Stop, Warned),
            (NotMet, false, Stop, Stop, NoEffect),
            (CouldNotBeChecked, true, Stop, Warn, Warned),
            (CouldNotBeChecked, true, Warn, Stop, Stopped),
            (CouldNotBeChecked, false, Stop, Stop, NoEffect),
            (Mapped, true, Stop, Stop, NoEffect),
            (Bound, true, Stop, Stop, NoEffect),
            (Skipped, false, Stop, Stop, NoEffect),
            (NotEvaluated, true, Stop, Stop, NoEffect),
            (MetHere, true, Stop, Stop, NoEffect),
            (MetRemotely, true, Stop, Stop, NoEffect),
        ];
        for (status, required, if_unsupported, if_unverified, expected) in table {
            let mut items = vec![item("agent:/hooks/guard", required, status)];
            let (outcome, stopped_by) = apply_actions(&mut items, if_unsupported, if_unverified);

            assert_eq!(items[0].effect, expected, "{status:?} required={required}");
            if expected == Stopped {
                assert_eq!(outcome, PackageOutcome::Stopped);
                assert_eq!(stopped_by, vec!["agent:/hooks/guard".to_string()]);
            } else {
                assert_eq!(outcome, PackageOutcome::Ready);
                assert!(stopped_by.is_empty());
            }
        }
    }

    #[test]
    fn stopped_by_lists_every_stopping_item_in_order() {
        let mut items = vec![
            item("agent:/hooks/a", true, DeclaredStatus::NotMapped),
            item("agent:/skills/b", true, DeclaredStatus::Mapped),
            item("agent:/policies/c", true, DeclaredStatus::NotMapped),
        ];

        let (outcome, stopped_by) =
            apply_actions(&mut items, PackageAction::Stop, PackageAction::Warn);

        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(stopped_by, vec!["agent:/hooks/a", "agent:/policies/c"]);
    }

    /// Every error of `value` against the committed schema snapshot `name`.
    fn schema_errors(name: crate::schema::SchemaName, value: &Value) -> Vec<String> {
        let schema = crate::schema::generate_schema(name).expect("schema generation");
        let validator = jsonschema::validator_for(&schema).expect("a valid schema");
        validator
            .iter_errors(value)
            .map(|error| format!("{}: {error}", error.instance_path()))
            .collect()
    }

    fn sample_report() -> AgentPackageReport {
        let entry = DocumentRecord {
            kind: crate::agent_package::model::DocumentKind::Agent,
            name: "release-notes-agent".to_string(),
            address: "$".to_string(),
            path: PathBuf::from("/sandbox/agent/agent.json"),
            digest: format!("sha256:{}", "0".repeat(64)),
        };
        let config: FabricConfig = serde_json::from_value(serde_json::json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "release-notes-agent"},
            "harness": {"adapter_id": "nvidia.fabric.hermes"},
            "runtime": {"max_turns": 12},
            "skills": {"paths": ["/sandbox/agent/skills/release-notes"]}
        }))
        .expect("config");
        let mut declared = item("agent:/skills/release-notes", true, DeclaredStatus::Mapped);
        declared.name = Some("release-notes".to_string());
        declared.address = Some("$/skills/release-notes".to_string());
        declared.mapped_to = vec!["skills.paths.0".to_string()];
        let mut unmapped = item("agent:/hooks/guard", false, DeclaredStatus::NotMapped);
        unmapped.gap = Some(MappingGap::NoFabricField);
        unmapped.detail = Some("NeMo Fabric runs no hooks".to_string());
        AgentPackageReport {
            schema_version: REPORT_SCHEMA_VERSION.to_string(),
            format_version: "0.4.0".to_string(),
            package_dir: PathBuf::from("/sandbox/agent"),
            base_dir: PathBuf::from("/sandbox"),
            outcome: PackageOutcome::Ready,
            if_unsupported: PackageAction::Stop,
            if_unverified: PackageAction::Warn,
            stopped_by: Vec::new(),
            entry: entry.clone(),
            agent: entry,
            lock: LockRecord {
                path: PathBuf::from("/sandbox/agent/oadl.lock.json"),
                sha256: "1".repeat(64),
                generator: Some("example".to_string()),
                entries: 2,
            },
            adapter: Some(AdapterReport {
                adapter_id: "nvidia.fabric.hermes".to_string(),
                accepts: vec![AdapterConfigField::Skills, AdapterConfigField::MaxTurns],
                extensions: vec!["nemo_fabric.plugins.v1".to_string()],
                instruction_mode: Some(InstructionMode::Replace),
            }),
            variables: vec![VariableValue {
                name: "max_turns".to_string(),
                value_type: "number".to_string(),
                value: serde_json::json!(12),
                origin: crate::agent_package::model::ValueOrigin::Default,
            }],
            model: ModelReport {
                declared_source: Some("ollama::llama3.2:3b".to_string()),
                provider: None,
                model: None,
                origin: ModelOrigin::HarnessDefault,
                parameters_applied: Vec::new(),
                parameters_not_applied: vec!["temperature".to_string()],
            },
            declared: vec![declared, unmapped],
            digests: vec![DigestCheck {
                address: "$/skills/release-notes".to_string(),
                entry_type: EntryType::Skill,
                source_type: SourceType::Local,
                source_url: Some("./skills/release-notes".to_string()),
                source_subpath: None,
                version: "local".to_string(),
                commit: None,
                digest: Some(format!("sha256:{}", "2".repeat(64))),
                digest_kind: Some(DigestKind::Folder),
                interface: None,
                result: Integrity::Verified,
                path: Some(PathBuf::from("/sandbox/agent/skills/release-notes")),
                retracted: None,
            }],
            notes: vec![Note {
                code: crate::agent_package::model::NoteCode::LimitLowered,
                subject: "runtime.max_turns".to_string(),
                text: "the smaller number applies".to_string(),
            }],
            config_sha256: Some(config_sha256(&config).expect("digest")),
            config: Some(config),
            checker_result: None,
        }
    }

    #[test]
    fn a_report_serializes_to_its_committed_schema_and_round_trips() {
        let report = sample_report();
        let encoded = serde_json::to_value(&report).expect("encode");

        assert_eq!(
            schema_errors(crate::schema::SchemaName::AgentPackageReport, &encoded),
            Vec::<String>::new()
        );
        assert_eq!(
            serde_json::from_value::<AgentPackageReport>(encoded.clone()).expect("decode"),
            report
        );
        assert_eq!(encoded["outcome"], "ready");
        assert_eq!(encoded["declared"][1]["gap"], "no_fabric_field");
        assert_eq!(encoded["digests"][0]["result"], "verified");
        assert_eq!(encoded["model"]["origin"], "harness_default");
        // A report with no config keeps the keys, as null.
        let mut stopped = report;
        stopped.outcome = PackageOutcome::Stopped;
        stopped.config = None;
        stopped.config_sha256 = None;
        stopped.adapter = None;
        let encoded = serde_json::to_value(&stopped).expect("encode");
        assert_eq!(
            schema_errors(crate::schema::SchemaName::AgentPackageReport, &encoded),
            Vec::<String>::new()
        );
        assert!(encoded["config"].is_null());
        assert!(encoded["config_sha256"].is_null());
        assert_eq!(encoded["outcome"], "stopped");
    }

    #[test]
    fn a_denied_report_holds_the_checker_result_and_serializes_to_its_committed_schema() {
        let mut denied = sample_report();
        denied.outcome = PackageOutcome::Denied;
        denied.config = None;
        let answer = serde_json::json!({
            "decision": "deny",
            "reasons": ["the hook guard is not on the approved list"],
            "checker": {"name": "platform-policy", "version": "3"}
        });
        let results = [
            CheckerResult {
                decision: CheckerDecision::Deny,
                failure: None,
                exit_code: Some(2),
                detail: None,
                answer: answer.as_object().cloned(),
            },
            CheckerResult {
                decision: CheckerDecision::Deny,
                failure: Some(CheckerFailure::OutputTooLarge),
                exit_code: None,
                detail: Some("the checker wrote more than 1048576 bytes".to_string()),
                answer: None,
            },
            CheckerResult {
                decision: CheckerDecision::Deny,
                failure: Some(CheckerFailure::TimedOut),
                exit_code: None,
                detail: Some("waiting for the policy folder".to_string()),
                answer: None,
            },
        ];
        for result in results {
            denied.checker_result = Some(result);
            let encoded = serde_json::to_value(&denied).expect("encode");

            assert_eq!(
                schema_errors(crate::schema::SchemaName::AgentPackageReport, &encoded),
                Vec::<String>::new()
            );
            assert_eq!(
                serde_json::from_value::<AgentPackageReport>(encoded.clone()).expect("decode"),
                denied
            );
            assert_eq!(encoded["outcome"], "denied");
            assert!(encoded["config"].is_null());
        }
        let encoded = serde_json::to_value(&denied).expect("encode");
        assert_eq!(
            encoded["checker_result"],
            serde_json::json!({
                "decision": "deny",
                "failure": "timed_out",
                "exit_code": null,
                "detail": "waiting for the policy folder",
                "answer": null
            })
        );
        // A report written before the checker existed has no `checker_result` member.
        let mut earlier = serde_json::to_value(sample_report()).expect("encode");
        earlier
            .as_object_mut()
            .expect("object")
            .remove("checker_result");
        assert_eq!(
            serde_json::from_value::<AgentPackageReport>(earlier).expect("decode"),
            sample_report()
        );
    }

    #[test]
    fn options_serialize_to_their_committed_schema() {
        use crate::agent_package::options::AgentPackageOptions;

        let options: AgentPackageOptions = serde_json::from_value(serde_json::json!({
            "entry": "variants/short/agent-definition.json",
            "variables": {"max_items": 20, "tone": "plain", "draft": true},
            "instruction_mode": "append",
            "if_unsupported": "warn",
            "if_unverified": "stop",
            "check_machine": true,
            "facts": {"network": "internal", "software": {"uv": "0.4.18"}},
            "checker": ["/opt/policy/bin/check-package", "--policy", "/sandbox/policy"],
            "checker_timeout_seconds": 20
        }))
        .expect("options");

        for value in [
            serde_json::to_value(&options).expect("encode"),
            serde_json::to_value(AgentPackageOptions::default()).expect("encode"),
            serde_json::json!({}),
        ] {
            assert_eq!(
                schema_errors(crate::schema::SchemaName::AgentPackageOptions, &value),
                Vec::<String>::new()
            );
        }
        assert!(
            !schema_errors(
                crate::schema::SchemaName::AgentPackageOptions,
                &serde_json::json!({"caller_verified": true})
            )
            .is_empty()
        );
    }

    #[test]
    fn config_digest_is_stable_and_does_not_depend_on_key_order() {
        let first: FabricConfig = serde_json::from_value(serde_json::json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "a", "description": "d"},
            "harness": {"adapter_id": "x", "settings": {"b": 1, "a": 2}},
            "runtime": {}
        }))
        .expect("config");
        let second: FabricConfig = serde_json::from_value(serde_json::json!({
            "runtime": {},
            "harness": {"settings": {"a": 2, "b": 1}, "adapter_id": "x"},
            "metadata": {"description": "d", "name": "a"},
            "schema_version": "fabric.agent/v1alpha1"
        }))
        .expect("config");

        let digest = config_sha256(&first).expect("digest");

        assert_eq!(digest, config_sha256(&second).expect("digest"));
        assert_eq!(digest.len(), 64);
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
        let mut changed = first.clone();
        changed.metadata.description = Some("e".to_string());
        assert_ne!(digest, config_sha256(&changed).expect("digest"));
    }
}
