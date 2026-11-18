// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Map the agent that runs onto a `FabricConfig`, and list every declared thing.
//!
//! "Mapped" has one meaning here: the loader wrote the thing into the config.
//! The caller of this module then plans that config on the selected adapter.
//! Nothing in this module shows that a harness loaded anything.
//!
//! The config has no field for plugins, hooks or called agents. Each kind
//! goes to the adapter under a root extension key, and only to an adapter
//! whose descriptor declares that key.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::adapter::AdapterSupport;
use super::bindings::{AGENTS, CUSTOM_TOOLS, binding_key, binds_called_agent};
use super::connector::{self, CarriedConnector};
use super::digest::bytes_digest;
use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::json::{MAX_DOCUMENT_BYTES, parse_document, read_regular_file};
use super::model::{
    CalledAgent, DigestKind, DocumentKind, Facts, FolderListing, Integrity, LockedPart, NeedLine,
    NeedStatus, Note, NoteCode, PackageReading, PartField, SourceType, StatedBy,
};
use super::needs::need_lines;
use super::options::AgentPackageOptions;
use super::paths::json_pointer;
use super::report::{
    DeclaredItem, DeclaredKind, DeclaredStatus, ItemEffect, MappingGap, ModelOrigin, ModelReport,
};
use crate::config::{
    AdapterConfigField, FabricConfig, InstructionConfig, InstructionMode, McpServerConfig,
    SkillConfig, SkillDirectory,
};
use crate::error::Result;

/// The result of mapping one package onto the caller's config.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Mapped {
    pub config: FabricConfig,
    pub items: Vec<DeclaredItem>,
    pub model: ModelReport,
    pub notes: Vec<Note>,
    /// The instruction mode the loader wrote, when it wrote instructions.
    pub instruction_mode: Option<InstructionMode>,
}

/// Map the package onto the caller's complete config.
pub(super) fn map_package(
    reading: &PackageReading,
    base: FabricConfig,
    support: &AdapterSupport,
    options: &AgentPackageOptions,
    needs: &[NeedLine],
    facts: &Facts,
) -> Result<Mapped> {
    let mut mapper = Mapper::new(reading, base, support, options, facts, true);
    mapper.run(needs)?;
    Ok(mapper.finish())
}

/// The same list of declared things with status `not_evaluated` (no config given).
/// Needs keep their own status, and so does a software package the
/// `packages` fact tells about: neither is compared with a config.
pub(super) fn unmapped_items(
    reading: &PackageReading,
    options: &AgentPackageOptions,
    needs: &[NeedLine],
    facts: &Facts,
) -> Vec<DeclaredItem> {
    let support = AdapterSupport {
        adapter_id: String::new(),
        accepts: BTreeSet::new(),
        instruction_modes: Vec::new(),
        skill_entries: Vec::new(),
        extension_schema: None,
    };
    let mut mapper = Mapper::new(
        reading,
        placeholder_config(&reading.entry.name),
        &support,
        options,
        facts,
        false,
    );
    // With `evaluate` off nothing is compared with a config, so nothing refuses.
    let _ = mapper.run(needs);
    mapper.items
}

/// What is known about the model when no config was given.
pub(super) fn unmapped_model(reading: &PackageReading) -> ModelReport {
    ModelReport {
        declared_source: declared_model_source(reading),
        provider: None,
        model: None,
        origin: ModelOrigin::NotEvaluated,
        parameters_applied: Vec::new(),
        parameters_not_applied: Vec::new(),
    }
}

/// The requested mode when the adapter offers it. With no request: replace when
/// the adapter offers it, as NeMo Fabric's own default does, else append.
pub(super) fn pick_instruction_mode(
    requested: Option<InstructionMode>,
    support: &AdapterSupport,
) -> Option<InstructionMode> {
    let offered = |mode| support.instruction_modes.contains(&mode);
    match requested {
        Some(mode) => offered(mode).then_some(mode),
        None if offered(InstructionMode::Replace) => Some(InstructionMode::Replace),
        None if offered(InstructionMode::Append) => Some(InstructionMode::Append),
        None => None,
    }
}

/// The one place that turns skill folders into `SkillConfig`.
///
/// The caller's entries are kept and the package's skill folders are added
/// after them, in the form the caller's config uses. A config that sets
/// `skills.directories` gets one `skill` entry per folder there, because
/// `directories` cannot be combined with `skills.paths`. Any other config gets
/// one `skills.paths` entry per folder.
pub(super) fn skill_config(base: Option<SkillConfig>, package_paths: Vec<PathBuf>) -> SkillConfig {
    let mut skills = base.unwrap_or_default();
    match &mut skills.directories {
        Some(directories) => {
            directories.extend(package_paths.into_iter().map(SkillDirectory::Skill));
        }
        None => skills.paths.extend(package_paths),
    }
    skills
}

/// The root extension key that carries plugin folders: `{"paths": [...]}`.
const PLUGINS_KEY: &str = "nemo_fabric.plugins.v1";
/// The root extension key that carries hook folders, each with the format of
/// its handlers: `{"entries": [{"path", "format"}]}`.
const HOOKS_KEY: &str = "nemo_fabric.hooks.v1";
/// The root extension key that carries called agents by name:
/// `{"<name>": {"description", "instructions", "max_turns", "skills"}}`.
const AGENTS_KEY: &str = "nemo_fabric.agents.v1";
/// The file in a hook's folder that lists its handlers.
const HOOKS_FILE: &str = "hooks.json";

const CREDENTIAL_NOT_SEEN: &str = "the platform binds the credential; the loader sees no value";
const OUTPUTS_NOT_USED: &str = "NeMo Fabric does not use outputs";
const EXTENSIONS_SKIPPED: &str = "data for another tool; NeMo Fabric skips it";
const SKILL_VALUES_NOT_HANDED: &str =
    "NeMo Fabric has no way to hand values to the model with a skill";

/// One of the two documents a load can read.
#[derive(Clone)]
struct Doc<'a> {
    stated_by: StatedBy,
    name: &'a str,
    value: &'a Value,
}

impl Doc<'_> {
    fn item(
        &self,
        kind: DeclaredKind,
        field: &str,
        name: Option<&str>,
        segments: &[&str],
    ) -> DeclaredItem {
        DeclaredItem {
            id: format!(
                "{}:{}",
                stated_by_text(self.stated_by),
                json_pointer(segments)
            ),
            kind,
            stated_by: self.stated_by,
            document: self.name.to_string(),
            field: field.to_string(),
            name: name.map(str::to_string),
            address: None,
            required: true,
            status: DeclaredStatus::NotEvaluated,
            gap: None,
            connector: None,
            effect: ItemEffect::None,
            mapped_to: Vec::new(),
            bound_to: None,
            reason: None,
            detail: None,
        }
    }

    fn object(&self, key: &str) -> Option<&Map<String, Value>> {
        self.value.get(key).and_then(Value::as_object)
    }
}

impl DeclaredItem {
    fn mapped(mut self, paths: Vec<String>) -> Self {
        self.status = DeclaredStatus::Mapped;
        self.mapped_to = paths;
        self
    }

    fn not_mapped(mut self, gap: MappingGap, detail: impl Into<String>) -> Self {
        self.status = DeclaredStatus::NotMapped;
        self.gap = Some(gap);
        self.detail = Some(detail.into());
        self
    }

    /// The caller supplies it. The target is kept as the caller wrote it.
    fn bound(mut self, target: &str, detail: impl Into<String>) -> Self {
        self.status = DeclaredStatus::Bound;
        self.bound_to = Some(target.to_string());
        self.detail = Some(detail.into());
        self
    }

    fn unchecked(mut self, detail: impl Into<String>) -> Self {
        self.status = DeclaredStatus::CouldNotBeChecked;
        self.detail = Some(detail.into());
        self
    }

    fn skipped(mut self, detail: impl Into<String>) -> Self {
        self.status = DeclaredStatus::Skipped;
        self.detail = Some(detail.into());
        self
    }

    fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// Whether a connector can carry the tools that name it.
#[derive(Clone, Copy)]
enum ConnectorState {
    Usable,
    Gap(MappingGap),
}

/// What happened to one of the two limits NeMo Fabric has a field for.
enum LimitOutcome {
    Mapped { path: &'static str, detail: String },
    NotMapped { gap: MappingGap, detail: String },
}

struct Mapper<'a> {
    reading: &'a PackageReading,
    support: &'a AdapterSupport,
    options: &'a AgentPackageOptions,
    /// Facts about the place the agent runs. The mapping reads the installed
    /// software packages from them. Needs are compared before the mapping.
    facts: &'a Facts,
    /// False: list every declared thing and compare nothing with a config.
    evaluate: bool,
    base: FabricConfig,
    config: FabricConfig,
    items: Vec<DeclaredItem>,
    notes: Vec<Note>,
    model: ModelReport,
    /// The `models` key of the model the config selects, when it selects one.
    model_key: Option<String>,
    instruction_mode: Option<InstructionMode>,
    skill_paths: Vec<PathBuf>,
    /// The last path segment of every skill folder in the config: each
    /// `skills.paths` entry, or each `skill` entry of `skills.directories`.
    /// A harness can name a skill by its folder, so each one may be used once.
    skill_names: BTreeSet<std::ffi::OsString>,
    servers: BTreeMap<String, McpServerConfig>,
    connectors: BTreeMap<String, ConnectorState>,
}

impl<'a> Mapper<'a> {
    fn new(
        reading: &'a PackageReading,
        base: FabricConfig,
        support: &'a AdapterSupport,
        options: &'a AgentPackageOptions,
        facts: &'a Facts,
        evaluate: bool,
    ) -> Self {
        // The model the config selects, as planning and the adapters read it:
        // `models.default`, or the only entry of a config that sets no default.
        let caller_model = base
            .models
            .get_key_value("default")
            .or_else(|| base.models.iter().next().filter(|_| base.models.len() == 1));
        let model = ModelReport {
            declared_source: declared_model_source(reading),
            provider: caller_model.map(|(_, model)| model.provider.clone()),
            model: caller_model.map(|(_, model)| model.model.clone()),
            origin: match (evaluate, caller_model) {
                (false, _) => ModelOrigin::NotEvaluated,
                (true, Some(_)) => ModelOrigin::Caller,
                (true, None) => ModelOrigin::HarnessDefault,
            },
            parameters_applied: Vec::new(),
            parameters_not_applied: Vec::new(),
        };
        let model_key = caller_model.map(|(key, _)| key.clone());
        Self {
            reading,
            support,
            options,
            facts,
            evaluate,
            config: base.clone(),
            base,
            items: Vec::new(),
            notes: Vec::new(),
            model,
            model_key,
            instruction_mode: None,
            skill_paths: Vec::new(),
            skill_names: BTreeSet::new(),
            servers: BTreeMap::new(),
            connectors: BTreeMap::new(),
        }
    }

    fn run(&mut self, needs: &[NeedLine]) -> Result<()> {
        self.identity()?;
        self.instructions()?;
        self.model_settings();
        self.limits();
        self.skills();
        self.plugins();
        self.connectors()?;
        self.hooks()?;
        self.tools();
        self.custom_tools();
        self.packages();
        self.called_agents()?;
        self.policies();
        self.credentials();
        self.needs(needs);
        self.reported_only("outputs", OUTPUTS_NOT_USED);
        self.reported_only("extensions", EXTENSIONS_SKIPPED);
        Ok(())
    }

    fn finish(mut self) -> Mapped {
        if !self.skill_paths.is_empty() {
            self.config.skills = Some(skill_config(
                self.config.skills.take(),
                std::mem::take(&mut self.skill_paths),
            ));
        }
        if !self.servers.is_empty() {
            self.config
                .mcp
                .get_or_insert_with(Default::default)
                .servers
                .extend(std::mem::take(&mut self.servers));
        }
        Mapped {
            config: self.config,
            items: self.items,
            model: self.model,
            notes: self.notes,
            instruction_mode: self.instruction_mode,
        }
    }

    /// The agent first, then the definition when the entry document is one.
    fn docs(&self) -> Vec<Doc<'a>> {
        let reading = self.reading;
        let mut docs = vec![Doc {
            stated_by: StatedBy::Agent,
            name: &reading.agent_record.name,
            value: &reading.agent_as_written,
        }];
        if reading.entry.kind == DocumentKind::AgentDefinition
            && let Some(definition) = &reading.definition
        {
            docs.push(Doc {
                stated_by: StatedBy::Definition,
                name: &reading.entry.name,
                value: definition,
            });
        }
        docs
    }

    /// Entries of one map field, per document. An entry the definition repeats
    /// from the agent counts once, as the agent's.
    fn entries(&self, field: &str) -> Vec<(Doc<'a>, &'a String, &'a Value)> {
        let docs = self.docs();
        let agent_names: BTreeSet<&String> = docs[0]
            .value
            .get(field)
            .and_then(Value::as_object)
            .map(|entries| entries.keys().collect())
            .unwrap_or_default();
        let mut found = Vec::new();
        for doc in docs {
            let Some(entries) = doc.value.get(field).and_then(Value::as_object) else {
                continue;
            };
            for (name, entry) in entries {
                if doc.stated_by == StatedBy::Definition && agent_names.contains(name) {
                    continue;
                }
                found.push((doc.clone(), name, entry));
            }
        }
        found
    }

    /// A value with a variable reference replaced by the variable's value.
    fn filled(&self, value: &Value) -> Value {
        if let Some(object) = value.as_object()
            && object.len() == 1
            && let Some(name) = object.get("var").and_then(Value::as_str)
            && let Some(variable) = self
                .reading
                .variables
                .iter()
                .find(|variable| variable.name == name)
        {
            return variable.value.clone();
        }
        value.clone()
    }

    fn note(&mut self, code: NoteCode, subject: &str, text: String) {
        self.notes.push(Note {
            code,
            subject: subject.to_string(),
            text,
        });
    }

    // ---- name and description ----

    fn identity(&mut self) -> Result<()> {
        let docs = self.docs();
        let entry = docs.last().expect("a load always has an agent").clone();
        for doc in &docs {
            let mut item = doc.item(DeclaredKind::Field, "name", None, &["name"]);
            let is_entry = doc.stated_by == entry.stated_by;
            item.required = is_entry;
            if self.evaluate {
                if is_entry {
                    if self.base.metadata.name != entry.name {
                        return Err(conflict(
                            "metadata.name",
                            format!(
                                "the config names `{}` and the package's entry document names `{}`",
                                self.base.metadata.name, entry.name
                            ),
                        ));
                    }
                    item = item.mapped(vec!["metadata.name".to_string()]);
                } else {
                    item = item.skipped("the definition's name is the agent's name in this load");
                }
            }
            self.items.push(item);
        }

        let description = |doc: &Doc<'a>| {
            doc.value
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        let used = docs
            .iter()
            .rev()
            .find_map(|doc| description(doc).map(|text| (doc.stated_by, text)));
        for doc in &docs {
            if description(doc).is_none() {
                continue;
            }
            let mut item = doc.item(DeclaredKind::Field, "description", None, &["description"]);
            let is_used = used
                .as_ref()
                .is_some_and(|(stated_by, _)| *stated_by == doc.stated_by);
            item.required = is_used;
            if self.evaluate {
                item = if is_used {
                    item.mapped(vec!["metadata.description".to_string()])
                } else {
                    item.skipped("the definition's description is used")
                };
            }
            self.items.push(item);
        }
        if self.evaluate
            && let Some((_, text)) = used
        {
            if let Some(existing) = &self.base.metadata.description
                && *existing != text
            {
                return Err(conflict(
                    "metadata.description",
                    "the config and the package give different descriptions",
                ));
            }
            self.config.metadata.description = Some(text);
        }
        Ok(())
    }

    // ---- instructions ----

    fn instructions(&mut self) -> Result<()> {
        let stating: Vec<Doc<'a>> = self
            .docs()
            .into_iter()
            .filter(|doc| doc.value.get("instructions").is_some())
            .collect();
        if stating.is_empty() {
            return Ok(());
        }
        let mut outcome: Option<std::result::Result<InstructionMode, (MappingGap, String)>> = None;
        if self.evaluate {
            if self
                .base
                .instructions
                .as_ref()
                .is_some_and(|instructions| instructions.system.is_some())
            {
                return Err(conflict(
                    "instructions.system",
                    "the package states the agent's instructions, so the config must not set instructions.system",
                ));
            }
            let text = self
                .reading
                .agent
                .get("instructions")
                .and_then(Value::as_str);
            let accepted = self.support.accepts(AdapterConfigField::SystemInstructions);
            let mode = pick_instruction_mode(self.options.instruction_mode, self.support);
            outcome = Some(match (text, accepted, mode) {
                (_, false, _) => Err((
                    MappingGap::AdapterDoesNotAccept,
                    "the adapter does not accept instructions.system".to_string(),
                )),
                (_, true, None) => Err((
                    MappingGap::AdapterDoesNotAccept,
                    "the adapter does not offer the requested instruction mode".to_string(),
                )),
                (Some(text), true, Some(mode)) if !text.trim().is_empty() => {
                    self.config
                        .instructions
                        .get_or_insert_with(Default::default)
                        .system = Some(InstructionConfig {
                        content: text.to_string(),
                        mode,
                        extensions: BTreeMap::new(),
                    });
                    self.instruction_mode = Some(mode);
                    Ok(mode)
                }
                _ => Err((
                    MappingGap::NoFabricField,
                    "the text is empty or only white space, which NeMo Fabric cannot hold"
                        .to_string(),
                )),
            });
        }
        let both = stating.len() > 1;
        for doc in &stating {
            let mut item = doc.item(DeclaredKind::Field, "instructions", None, &["instructions"]);
            match &outcome {
                None => {}
                Some(Ok(mode)) => {
                    let mode = match mode {
                        InstructionMode::Replace => "replace",
                        InstructionMode::Append => "append",
                    };
                    let order = if both {
                        " The agent's text comes first, then one empty line, then the definition's text."
                    } else {
                        ""
                    };
                    item = item
                        .mapped(vec!["instructions.system.content".to_string()])
                        .with_detail(format!("written with mode `{mode}`.{order}"));
                }
                Some(Err((gap, detail))) => item = item.not_mapped(*gap, detail.clone()),
            }
            self.items.push(item);
        }
        Ok(())
    }

    // ---- model ----

    fn model_settings(&mut self) {
        let doc = self.docs().remove(0);
        let Some(written) = doc.object("model") else {
            return;
        };
        let merged = self.reading.agent.get("model");
        if written.contains_key("source") {
            let mut item = doc.item(
                DeclaredKind::ModelSetting,
                "model",
                Some("source"),
                &["model", "source"],
            );
            item.required = false;
            if self.evaluate {
                item = self.default_model(item);
            }
            self.items.push(item);
        }
        if written.contains_key("revision") {
            let mut item = doc.item(
                DeclaredKind::ModelSetting,
                "model",
                Some("revision"),
                &["model", "revision"],
            );
            item.required = false;
            if self.evaluate {
                item = item.not_mapped(
                    MappingGap::NoFabricField,
                    "NeMo Fabric has no field for a model revision",
                );
            }
            self.items.push(item);
        }
        let Some(parameters) = written.get("parameters").and_then(Value::as_object) else {
            return;
        };
        for key in parameters.keys() {
            let mut item = doc.item(
                DeclaredKind::ModelSetting,
                "model",
                Some(key.as_str()),
                &["model", "parameters", key.as_str()],
            );
            item.required = false;
            if self.evaluate {
                let value = merged
                    .and_then(|model| model.get("parameters"))
                    .and_then(|parameters| parameters.get(key))
                    .cloned()
                    .unwrap_or_else(|| self.filled(&parameters[key]));
                item = self.model_parameter(item, key, &value);
            }
            self.items.push(item);
        }
    }

    /// The config's `models.default` is the caller's choice and is kept. A
    /// config that sets none offers a model by naming a `models` entry with
    /// the text of the document's default model, and that entry becomes
    /// `models.default`. A config that holds one model under another name
    /// has chosen that model: planning and the adapters run it.
    ///
    /// The loader writes no entry of its own. The text before `::` names a
    /// catalog, and a catalog is not a NeMo Fabric provider: a provider also
    /// needs an endpoint and a key, which only the caller has.
    fn default_model(&mut self, item: DeclaredItem) -> DeclaredItem {
        if self.base.models.contains_key("default") {
            return item
                .skipped("the config sets models.default, and that is the model configured");
        }
        let source = self.model.declared_source.clone().unwrap_or_default();
        if let Some(offered) = self.config.models.remove(&source) {
            self.model.provider = Some(offered.provider.clone());
            self.model.model = Some(offered.model.clone());
            self.model.origin = ModelOrigin::Package;
            self.model_key = Some("default".to_string());
            self.config.models.insert("default".to_string(), offered);
            return item
                .mapped(vec!["models.default".to_string()])
                .with_detail(format!(
                    "the config offers this model as models.{source}, and that entry is now models.default"
                ));
        }
        match &self.model_key {
            Some(only) => item.skipped(format!(
                "the config holds one model, models.{only}, and that is the model configured"
            )),
            None => item.not_mapped(
                MappingGap::ModelNotConfigured,
                if self.base.models.is_empty() {
                    "the config holds no model. The harness uses its own default when it has one, and an adapter that needs a model in the config fails at start".to_string()
                } else {
                    format!(
                        "the config holds more than one model, sets no models.default, and names none `{source}`. Planning refuses a config that selects no model"
                    )
                },
            ),
        }
    }

    fn model_parameter(&mut self, item: DeclaredItem, key: &str, value: &Value) -> DeclaredItem {
        let (field, member) = match key {
            "temperature" => (AdapterConfigField::ModelTemperature, "temperature"),
            "max_output_tokens" => (AdapterConfigField::ModelMaxTokens, "max_tokens"),
            _ => {
                self.model.parameters_not_applied.push(key.to_string());
                return item.not_mapped(
                    MappingGap::NoFabricField,
                    "NeMo Fabric has no model setting for it",
                );
            }
        };
        let accepted = self.support.accepts(field);
        let selected = self
            .model_key
            .as_ref()
            .and_then(|role| Some((role, self.config.models.get_mut(role)?)));
        let Some((role, entry)) = selected else {
            self.model.parameters_not_applied.push(key.to_string());
            return item.not_mapped(
                MappingGap::ModelNotConfigured,
                "the config selects no model to carry it",
            );
        };
        let already_set = match key {
            "temperature" => entry.temperature.is_some(),
            _ => entry.max_tokens.is_some(),
        };
        if already_set {
            self.model.parameters_not_applied.push(key.to_string());
            return item.skipped("the config already sets it, and the config's value is kept");
        }
        if !accepted {
            self.model.parameters_not_applied.push(key.to_string());
            return item.not_mapped(
                MappingGap::AdapterDoesNotAccept,
                format!("the adapter does not accept {}", field_name(field)),
            );
        }
        let written = match key {
            "temperature" => value
                .as_f64()
                .map(|number| entry.temperature = Some(number)),
            _ => whole_number(value)
                .filter(|number| *number > 0)
                .map(|number| entry.max_tokens = Some(number)),
        };
        if written.is_none() {
            self.model.parameters_not_applied.push(key.to_string());
            return item.not_mapped(
                MappingGap::NoFabricField,
                "the value is not a number NeMo Fabric can hold",
            );
        }
        self.model.parameters_applied.push(key.to_string());
        item.mapped(vec![format!("models.{role}.{member}")])
    }

    // ---- limits ----

    fn limits(&mut self) {
        let merged = self.reading.agent.get("limits").and_then(Value::as_object);
        let mut turns = None;
        let mut timeout = None;
        if self.evaluate {
            if let Some(value) = merged.and_then(|limits| limits.get("max_turns")) {
                turns = Some(self.max_turns(value));
            }
            if let Some(value) = merged.and_then(|limits| limits.get("timeout_seconds")) {
                timeout = Some(self.timeout_seconds(value));
            }
        }
        for doc in self.docs() {
            let Some(limits) = doc.object("limits") else {
                continue;
            };
            for key in limits.keys() {
                let mut item = doc.item(
                    DeclaredKind::Limit,
                    "limits",
                    Some(key.as_str()),
                    &["limits", key.as_str()],
                );
                if self.evaluate {
                    let outcome = match key.as_str() {
                        "max_turns" => turns.as_ref(),
                        "timeout_seconds" => timeout.as_ref(),
                        _ => None,
                    };
                    item = match outcome {
                        Some(LimitOutcome::Mapped { path, detail }) => item
                            .mapped(vec![(*path).to_string()])
                            .with_detail(detail.clone()),
                        Some(LimitOutcome::NotMapped { gap, detail }) => {
                            item.not_mapped(*gap, detail.clone())
                        }
                        None => item.not_mapped(MappingGap::NoFabricField, enforced_elsewhere(key)),
                    };
                }
                self.items.push(item);
            }
        }
    }

    /// A number the format accepts and NeMo Fabric cannot hold is not a
    /// refusal: the limit is not mapped, and whoever runs the agent decides.
    fn max_turns(&mut self, value: &Value) -> LimitOutcome {
        let Some(package) = whole_number(value)
            .and_then(|number| u32::try_from(number).ok())
            .filter(|number| *number > 0)
        else {
            return LimitOutcome::NotMapped {
                gap: MappingGap::ValueOutOfRange,
                detail: "NeMo Fabric holds a turn limit as a whole number from 1 to 4294967295"
                    .to_string(),
            };
        };
        if !self.support.accepts(AdapterConfigField::MaxTurns) {
            return LimitOutcome::NotMapped {
                gap: MappingGap::AdapterDoesNotAccept,
                detail: "the adapter does not accept runtime.max_turns".to_string(),
            };
        }
        let applies = match self.base.runtime.max_turns {
            Some(caller) if caller != package => {
                let smaller = caller.min(package);
                self.note(
                    NoteCode::LimitLowered,
                    "runtime.max_turns",
                    format!(
                        "the config sets {caller} and the package sets {package}; the smaller number, {smaller}, applies"
                    ),
                );
                smaller
            }
            _ => package,
        };
        self.config.runtime.max_turns = Some(applies);
        LimitOutcome::Mapped {
            path: "runtime.max_turns",
            detail: format!("{applies} applies"),
        }
    }

    fn timeout_seconds(&mut self, value: &Value) -> LimitOutcome {
        let Some(package) = value.as_f64().filter(|seconds| {
            seconds.is_finite()
                && *seconds > 0.0
                && std::time::Duration::try_from_secs_f64(*seconds).is_ok()
        }) else {
            return LimitOutcome::NotMapped {
                gap: MappingGap::ValueOutOfRange,
                detail: "NeMo Fabric holds a timeout as a finite number of seconds above zero"
                    .to_string(),
            };
        };
        let applies = match self.base.runtime.timeout_seconds {
            Some(caller) if caller != package => {
                let smaller = caller.min(package);
                self.note(
                    NoteCode::LimitLowered,
                    "runtime.timeout_seconds",
                    format!(
                        "the config sets {caller} and the package sets {package}; the smaller number, {smaller}, applies"
                    ),
                );
                smaller
            }
            _ => package,
        };
        self.config.runtime.timeout_seconds = Some(applies);
        // The format's timeout covers the runs of the called agents too. NeMo
        // Fabric bounds its own invocation, and only a caller that runs no
        // called agent has the whole tree in it.
        let covers = if binds_called_agent(&self.options.bindings) {
            "It bounds each NeMo Fabric invocation. A binding says the caller runs a called agent: a run of that agent outside the invocation is the caller's to bound"
        } else {
            "It bounds each invocation; NeMo Fabric runs no called agents, so one invocation is the whole tree"
        };
        LimitOutcome::Mapped {
            path: "runtime.timeout_seconds",
            detail: format!("{applies} seconds applies. {covers}"),
        }
    }

    // ---- parts ----

    /// The lock entry of a root-level part, and whether the lock leaves the reference out.
    fn lookup(&self, field: PartField, name: &str) -> (Option<&'a LockedPart>, bool) {
        let reading = self.reading;
        let left_out = reading.left_out.iter().any(|left| {
            left.field == field
                && left.name == name
                && (left.owner == reading.entry.address
                    || left.owner == reading.agent_record.address)
        });
        (reading.part(field, name), left_out)
    }

    /// The lock entry of a part a called agent states, and whether the lock
    /// leaves the reference out.
    fn called_lookup(
        &self,
        called: &CalledAgent,
        field: PartField,
        name: &str,
    ) -> (Option<&'a LockedPart>, bool) {
        let reading = self.reading;
        let left_out = reading
            .left_out
            .iter()
            .any(|left| left.field == field && left.name == name && left.owner == called.address);
        (reading.called_part(called, field, name), left_out)
    }

    /// The verified folder of a carried part, or why its content cannot be used.
    fn content(
        &self,
        field: PartField,
        name: &str,
        entry: &Value,
    ) -> std::result::Result<(&'a LockedPart, PathBuf), (MappingGap, String)> {
        let (part, left_out) = self.lookup(field, name);
        Self::carried(part, left_out, entry)
    }

    /// The same for the lock entry a reference leads to, whichever document
    /// states the reference.
    fn carried(
        part: Option<&'a LockedPart>,
        left_out: bool,
        entry: &Value,
    ) -> std::result::Result<(&'a LockedPart, PathBuf), (MappingGap, String)> {
        if left_out {
            return Err((
                MappingGap::NotResolved,
                "the lock leaves this optional reference out".to_string(),
            ));
        }
        let source = entry.get("source").and_then(Value::as_str).unwrap_or("");
        if !is_loadable_source(source) {
            return Err((
                MappingGap::SourceTypeUnsupported,
                "the loader does not load this source type".to_string(),
            ));
        }
        let Some(part) = part else {
            return Err((
                MappingGap::NotResolved,
                "the lock has no entry for this reference".to_string(),
            ));
        };
        match (part.integrity, &part.path) {
            (Integrity::Verified, Some(path)) => Ok((part, path.clone())),
            (Integrity::Absent, _) => Err((
                MappingGap::ContentAbsent,
                "the content is pinned and is not in the package; the loader fetches nothing"
                    .to_string(),
            )),
            (Integrity::NotVerifiable, _) => Err((
                MappingGap::ContentNotVerifiable,
                "the content is pinned by an artifact digest that cannot be worked out from files"
                    .to_string(),
            )),
            _ if part.source_type == SourceType::Pointer => Err((
                MappingGap::SourceTypeUnsupported,
                "the source points at something that is not carried".to_string(),
            )),
            _ => Err((
                MappingGap::ContentAbsent,
                "the content was not verified in the package".to_string(),
            )),
        }
    }

    fn part_item(
        &self,
        doc: &Doc<'a>,
        field: PartField,
        name: &str,
        entry: &Value,
    ) -> DeclaredItem {
        let mut item = doc.item(
            DeclaredKind::Part,
            field.as_str(),
            Some(name),
            &[field.as_str(), name],
        );
        item.required = !is_optional(entry);
        item.address = self.lookup(field, name).0.map(|part| part.address.clone());
        item
    }

    fn skills(&mut self) {
        // The caller's config chooses the form. `skills.directories` cannot be
        // combined with `skills.paths`, so a config that sets it gets `skill`
        // entries, and any other config gets paths.
        let caller_entries = self
            .base
            .skills
            .as_ref()
            .and_then(|skills| skills.directories.as_deref());
        let structured = caller_entries.is_some();
        let caller_folders: Vec<&std::path::Path> = match caller_entries {
            // A `folder` entry holds skills the loader does not look for.
            Some(entries) => entries
                .iter()
                .filter_map(|entry| match entry {
                    SkillDirectory::Skill(path) => Some(path.as_path()),
                    SkillDirectory::Folder(_) => None,
                })
                .collect(),
            None => self
                .base
                .skills
                .iter()
                .flat_map(|skills| &skills.paths)
                .map(PathBuf::as_path)
                .collect(),
        };
        let base_count = match caller_entries {
            Some(entries) => entries.len(),
            None => caller_folders.len(),
        };
        // The caller's own skill folders take their names first.
        self.skill_names = caller_folders
            .into_iter()
            .filter_map(|path| path.file_name().map(std::ffi::OsStr::to_os_string))
            .collect();
        for (doc, name, entry) in self.entries("skills") {
            let mut item = self.part_item(&doc, PartField::Skills, name, entry);
            let mut values = entry.get("variables").map(|_| {
                let mut values = doc.item(
                    DeclaredKind::PartValues,
                    "skills",
                    Some(name.as_str()),
                    &["skills", name.as_str(), "variables"],
                );
                values.required = item.required;
                values.address = item.address.clone();
                values
            });
            if self.evaluate {
                item = match self.content(PartField::Skills, name, entry) {
                    Err((gap, detail)) => item.not_mapped(gap, detail),
                    Ok(_) if !self.support.accepts(AdapterConfigField::Skills) => item.not_mapped(
                        MappingGap::AdapterDoesNotAccept,
                        "the adapter does not accept skills",
                    ),
                    // Planning refuses an entry kind the descriptor does not list.
                    Ok(_) if structured && !self.support.takes_skill_entries() => item.not_mapped(
                        MappingGap::AdapterDoesNotAccept,
                        "the config sets skills.directories, and the adapter does not list `skill` entries under skills.directory_loading",
                    ),
                    Ok((part, path)) => match self.skill_folder(part, &path) {
                        Err((gap, detail)) => item.not_mapped(gap, detail),
                        Ok(folder_name) => {
                            let index = base_count + self.skill_paths.len();
                            self.skill_paths.push(path);
                            let item = item.mapped(vec![if structured {
                                format!("skills.directories.{index}.skill")
                            } else {
                                format!("skills.paths.{index}")
                            }]);
                            if folder_name == *name {
                                item
                            } else if structured {
                                item.with_detail(format!(
                                    "the adapter reads the skill's name from its SKILL.md, and a skill with no name there is named `{folder_name}` after its folder"
                                ))
                            } else {
                                item.with_detail(format!(
                                    "a harness names a skill by its folder, so this skill is named `{folder_name}`"
                                ))
                            }
                        }
                    },
                };
                values = values.map(|values| {
                    values.not_mapped(MappingGap::NoFabricField, SKILL_VALUES_NOT_HANDED)
                });
            }
            self.items.push(item);
            self.items.extend(values);
        }
    }

    /// Whether the adapters can take a verified skill, and the name its folder
    /// gives it. The adapters take a skill as a folder that holds `SKILL.md` and
    /// can name it by the folder, so one file is not a shape they take and a
    /// folder name can be used once. The rule is the same for `skills.paths`
    /// and for `skill` entries of `skills.directories`.
    fn skill_folder(
        &mut self,
        part: &LockedPart,
        path: &std::path::Path,
    ) -> std::result::Result<String, (MappingGap, String)> {
        if part.digest_kind == Some(DigestKind::File) {
            return Err((
                MappingGap::UnsupportedShape,
                "the skill is one file. NeMo Fabric adapters take a skill as a folder that holds SKILL.md"
                    .to_string(),
            ));
        }
        let Some(folder_name) = path.file_name() else {
            return Err((
                MappingGap::UnsupportedShape,
                "the skill's folder has no name a harness can use".to_string(),
            ));
        };
        let shown = folder_name.to_string_lossy().into_owned();
        if !self.skill_names.insert(folder_name.to_os_string()) {
            return Err((
                MappingGap::KeyClash,
                format!(
                    "a harness can name a skill by its folder, and another skill folder is already named `{shown}`"
                ),
            ));
        }
        Ok(shown)
    }

    /// NeMo Fabric evaluates no policy, and no adapter takes one.
    fn policies(&mut self) {
        for (doc, name, entry) in self.entries(PartField::Policies.as_str()) {
            let mut item = self.part_item(&doc, PartField::Policies, name, entry);
            if self.evaluate {
                item = if self.lookup(PartField::Policies, name).1 {
                    item.not_mapped(
                        MappingGap::NotResolved,
                        "the lock leaves this optional reference out",
                    )
                } else {
                    item.not_mapped(
                        MappingGap::NoFabricField,
                        "NeMo Fabric evaluates no policy and hands the agent to no evaluator",
                    )
                };
            }
            self.items.push(item);
        }
    }

    // ---- plugins, hooks ----

    /// Write a root extension key when the adapter takes it: its descriptor
    /// declares the key, and the config's root extension keys still fit its
    /// schema with the new value. Else nothing is written, and the reason is
    /// returned.
    fn offer_extension(&mut self, key: &str, value: Value) -> std::result::Result<(), String> {
        let mut extensions = self.config.extensions.clone();
        extensions.insert(key.to_string(), value);
        match self.support.extension_gap(key, &extensions) {
            Some(reason) => Err(reason),
            None => {
                self.config.extensions = extensions;
                Ok(())
            }
        }
    }

    /// The value of the root extension key `key` with `entry` added at the end
    /// of its list `member`, and the index the entry has there. What the
    /// config already holds under the key stays first. None: the config holds
    /// the key in another shape.
    fn appended(&self, key: &str, member: &str, entry: Value) -> Option<(Value, usize)> {
        let mut held = match self.config.extensions.get(key) {
            None => Map::new(),
            Some(Value::Object(held)) => held.clone(),
            Some(_) => return None,
        };
        let mut list = match held.remove(member) {
            None => Vec::new(),
            Some(Value::Array(list)) => list,
            Some(_) => return None,
        };
        let index = list.len();
        list.push(entry);
        held.insert(member.to_string(), Value::Array(list));
        Some((Value::Object(held), index))
    }

    /// Add one entry to a list under a root extension key, for a part whose
    /// verified folder goes to the adapter.
    fn offer_entry(
        &mut self,
        item: DeclaredItem,
        key: &str,
        member: &str,
        entry: Value,
    ) -> DeclaredItem {
        let Some((value, index)) = self.appended(key, member, entry) else {
            return item.not_mapped(MappingGap::UnsupportedShape, held_in_another_shape(key));
        };
        match self.offer_extension(key, value) {
            Ok(()) => item.mapped(vec![format!("{key}.{member}.{index}")]),
            Err(reason) => item.not_mapped(MappingGap::AdapterDoesNotAccept, reason),
        }
    }

    /// A verified plugin folder goes to the adapter under
    /// `nemo_fabric.plugins.v1`, after the paths the config holds there.
    fn plugins(&mut self) {
        for (doc, name, entry) in self.entries(PartField::Plugins.as_str()) {
            let mut item = self.part_item(&doc, PartField::Plugins, name, entry);
            if self.evaluate {
                item = match self
                    .content(PartField::Plugins, name, entry)
                    .and_then(|(part, path)| folder_text(part, &path, "plugin"))
                {
                    Err((gap, detail)) => item.not_mapped(gap, detail),
                    Ok(folder) => {
                        self.offer_entry(item, PLUGINS_KEY, "paths", Value::String(folder))
                    }
                };
                if item.status == DeclaredStatus::Mapped {
                    item = item.with_detail(
                        "handed to the adapter. The report does not say that the harness loaded it",
                    );
                }
            }
            self.items.push(item);
        }
    }

    /// A verified hook folder goes to the adapter under `nemo_fabric.hooks.v1`
    /// with the format its `hooks.json` names, after the entries the config
    /// holds there. An adapter's schema lists the formats it runs, so a hook
    /// in another format is not mapped.
    fn hooks(&mut self) -> Result<()> {
        for (doc, name, entry) in self.entries(PartField::Hooks.as_str()) {
            let mut item = self.part_item(&doc, PartField::Hooks, name, entry);
            if self.evaluate {
                item = match self
                    .content(PartField::Hooks, name, entry)
                    .and_then(|(part, path)| Ok((part, folder_text(part, &path, "hook")?)))
                {
                    Err((gap, detail)) => item.not_mapped(gap, detail),
                    Ok((part, folder)) => self.hook(item, part, folder)?,
                };
            }
            self.items.push(item);
        }
        Ok(())
    }

    fn hook(
        &mut self,
        item: DeclaredItem,
        part: &LockedPart,
        folder: String,
    ) -> Result<DeclaredItem> {
        // An adapter that takes no hooks gets none, whatever the hook's format.
        if let Some(reason) = self
            .support
            .extension_gap(HOOKS_KEY, &self.config.extensions)
        {
            return Ok(item.not_mapped(MappingGap::AdapterDoesNotAccept, reason));
        }
        let Some(listing) = self.reading.listings.get(&part.address) else {
            return Ok(item.not_mapped(
                MappingGap::ContentAbsent,
                "the hook's files were not listed when the package was checked",
            ));
        };
        let Some(format) = hook_format(part, Path::new(&folder), listing)? else {
            return Ok(item.not_mapped(
                MappingGap::UnsupportedShape,
                format!(
                    "the hook's {HOOKS_FILE} is not a JSON object that names its format in `$schema`"
                ),
            ));
        };
        let entry = serde_json::json!({"path": folder, "format": format});
        let item = self.offer_entry(item, HOOKS_KEY, "entries", entry);
        Ok(if item.status == DeclaredStatus::Mapped {
            item.with_detail(format!(
                "handed to the adapter with the format `{format}`. The report does not say that a handler ran"
            ))
        } else {
            item
        })
    }

    /// A software package is installed by its registry's own tool, never by
    /// the loader. The loader compares the version the lock pins with the
    /// `packages` fact, and writes nothing into the config.
    ///
    /// The comparison reads the lock and the fact, so a package the fact
    /// tells about has its status with or without a config, as a need has.
    fn packages(&mut self) {
        for (doc, name, entry) in self.entries("packages") {
            let item = self.part_item(&doc, PartField::Packages, name, entry);
            let item = match self.lookup(PartField::Packages, name) {
                (Some(part), false) => self.package(item, part),
                _ if !self.evaluate => item,
                (_, true) => item.not_mapped(
                    MappingGap::NotResolved,
                    "the lock leaves this optional reference out",
                ),
                (None, false) => item.not_mapped(
                    MappingGap::NotResolved,
                    "the lock has no entry for this reference",
                ),
            };
            self.items.push(item);
        }
    }

    /// Whether the package a lock entry pins is installed at the pinned
    /// version, as the `packages` fact tells it. With no fact about the
    /// package's registry nothing looked for it, and it is not mapped.
    fn package(&self, mut item: DeclaredItem, part: &LockedPart) -> DeclaredItem {
        // The lock's digest is of the registry's artifact, and an installed
        // package is not that artifact any more.
        const DIGEST: &str = "The lock's digest is of the registry's artifact and was not compared with the installed files";
        let pinned = part.version.as_str();
        let told = match (
            part.source_type,
            part.registry.as_deref(),
            part.source_url.as_deref(),
        ) {
            (SourceType::Registry, Some(registry), Some(name)) => self
                .facts
                .packages
                .as_ref()
                .and_then(|packages| packages.get(registry))
                .map(|installed| (registry, name, installed))
                .ok_or_else(|| {
                    (
                        MappingGap::NoFabricField,
                        format!(
                            "the lock pins {name} {pinned} from {registry}. NeMo Fabric installs no package and looks for none, and the `packages` fact says nothing about the registry `{registry}`"
                        ),
                    )
                }),
            _ => Err((
                MappingGap::SourceTypeUnsupported,
                "the loader looks for a package a lock pins from a registry, and this one has another source".to_string(),
            )),
        };
        let (registry, name, installed) = match told {
            Ok(told) => told,
            Err(_) if !self.evaluate => return item,
            Err((gap, detail)) => return item.not_mapped(gap, detail),
        };
        // Every name of the fact that names this package. PyPI has more than
        // one spelling for one package.
        let mut versions: Vec<&Option<String>> = installed
            .iter()
            .filter(|(installed, _)| same_package(registry, installed, name))
            .map(|(_, version)| version)
            .collect();
        versions.dedup();
        let (status, detail) = match versions.as_slice() {
            [] => (
                DeclaredStatus::NotMet,
                format!("{name} is not installed from {registry}. The lock pins {pinned}"),
            ),
            [None] => (
                DeclaredStatus::CouldNotBeChecked,
                format!("{name} is installed and its version is not known. The lock pins {pinned}"),
            ),
            [Some(version)] if version == pinned => (
                DeclaredStatus::MetHere,
                format!("{name} {pinned} is installed. {DIGEST}"),
            ),
            [Some(version)] => (
                DeclaredStatus::NotMet,
                format!("{name} {version} is installed, and the lock pins {pinned}"),
            ),
            _ => (
                DeclaredStatus::CouldNotBeChecked,
                format!(
                    "the `packages` fact names {name} more than once under {registry}, with different versions. The lock pins {pinned}"
                ),
            ),
        };
        item.status = status;
        item.with_detail(detail)
    }

    // ---- connectors ----

    fn connectors(&mut self) -> Result<()> {
        // Sorted by name across both documents, so a key clash always falls on
        // the later connector.
        let mut entries = self.entries("connectors");
        entries.sort_by(|left, right| left.1.cmp(right.1));
        for (doc, name, entry) in entries {
            let mut item = self.part_item(&doc, PartField::Connectors, name, entry);
            if !self.evaluate {
                self.items.push(item);
                continue;
            }
            let source = entry.get("source").and_then(Value::as_str).unwrap_or("");
            let accepted = self.support.accepts(AdapterConfigField::Mcp);
            item = if source.starts_with("mcp::") {
                self.running_connector(item, name, accepted)
            } else {
                match self.content(PartField::Connectors, name, entry) {
                    Err((gap, detail)) => item.not_mapped(gap, detail),
                    Ok(_) if !accepted => item.not_mapped(
                        MappingGap::AdapterDoesNotAccept,
                        "the adapter does not accept mcp",
                    ),
                    Ok((part, _)) => self.carried_connector(item, part)?,
                }
            };
            let state = match (item.status, item.gap) {
                (DeclaredStatus::NotMapped, Some(gap)) => ConnectorState::Gap(gap),
                _ => ConnectorState::Usable,
            };
            self.connectors.insert(name.clone(), state);
            self.items.push(item);
        }
        Ok(())
    }

    /// A connector that is already running. The platform binds it: the caller's
    /// config holds `mcp.servers.<name>` and the loader keeps that entry unchanged.
    fn running_connector(&self, item: DeclaredItem, name: &str, accepted: bool) -> DeclaredItem {
        if self.lookup(PartField::Connectors, name).1 {
            return item.not_mapped(
                MappingGap::NotResolved,
                "the lock leaves this optional reference out",
            );
        }
        if !accepted {
            return item.not_mapped(
                MappingGap::AdapterDoesNotAccept,
                "the adapter does not accept mcp",
            );
        }
        let bound = self
            .base
            .mcp
            .as_ref()
            .is_some_and(|mcp| mcp.servers.contains_key(name));
        if !bound {
            return item.not_mapped(
                MappingGap::NotBound,
                format!("the config has no mcp.servers.{name} entry that binds the running server"),
            );
        }
        let mut item = item.unchecked(
            "the pinned version and tools hash were not compared with the running server",
        );
        item.mapped_to = vec![format!("mcp.servers.{name}")];
        item
    }

    /// A connector carried in the package: every server in its `mcp.json`
    /// becomes one stdio entry, or the connector is not mapped at all.
    fn carried_connector(&mut self, item: DeclaredItem, part: &LockedPart) -> Result<DeclaredItem> {
        let Some(listing) = self.reading.listings.get(&part.address) else {
            return Ok(item.not_mapped(
                MappingGap::ContentAbsent,
                "the connector's files were not listed when the package was checked",
            ));
        };
        let servers = match connector::read_carried_connector(part, listing)? {
            CarriedConnector::NotMapped { gap, detail } => return Ok(item.not_mapped(gap, detail)),
            CarriedConnector::Servers(servers) => servers,
        };
        let mut entries = Vec::new();
        let mut started_from = Vec::new();
        let mut not_read = Vec::new();
        for server in &servers {
            if self
                .base
                .mcp
                .as_ref()
                .is_some_and(|mcp| mcp.servers.contains_key(&server.key))
            {
                return Err(conflict(
                    &format!("mcp.servers.{}", server.key),
                    "the config sets a server under the key of a connector the package carries",
                ));
            }
            if self.servers.contains_key(&server.key)
                || entries.iter().any(|(key, _)| key == &server.key)
            {
                return Ok(item.not_mapped(
                    MappingGap::KeyClash,
                    format!(
                        "server key `{}` is already used by another carried server",
                        server.key
                    ),
                ));
            }
            let Some(config) = connector::stdio_server(server) else {
                return Ok(item.not_mapped(
                    MappingGap::UnsupportedPlatform,
                    "a carried connector is started through /bin/sh, which is not on this platform",
                ));
            };
            entries.push((server.key.clone(), config));
            started_from.push(if server.working_dir_shown.is_empty() {
                "the connector's folder".to_string()
            } else {
                format!("`{}` in the connector's folder", server.working_dir_shown)
            });
            if !server.keys_not_read.is_empty() {
                not_read.push((server.key.clone(), server.keys_not_read.join(", ")));
            }
        }
        let paths = entries
            .iter()
            .map(|(key, _)| format!("mcp.servers.{key}"))
            .collect();
        self.servers.extend(entries);
        started_from.dedup();
        let mut detail = format!(
            "started from {} with the command in its mcp.json; what the command needs is a need under runtime.software",
            started_from.join(" and ")
        );
        // A key the loader does not read changes nothing about how the server
        // starts. It is named, never dropped without a word.
        for (key, keys) in not_read {
            detail.push_str(&format!(
                ". Keys of server `{key}` that were not read: {keys}"
            ));
            self.note(
                NoteCode::ConnectorKeyNotRead,
                &format!("mcp.servers.{key}"),
                format!(
                    "the server entry in mcp.json has keys the loader does not read, and they have no effect: {keys}"
                ),
            );
        }
        Ok(item.mapped(paths).with_detail(detail))
    }

    // ---- tools, custom tools, credentials ----

    fn tools(&mut self) {
        let mut seen: Vec<(Option<String>, String)> = Vec::new();
        for doc in self.docs() {
            let Some(tools) = doc.value.get("tools").and_then(Value::as_array) else {
                continue;
            };
            for (index, tool) in tools.iter().enumerate() {
                let index = index.to_string();
                let text = |key: &str| tool.get(key).and_then(Value::as_str).map(str::to_string);
                let (connector, name) = match (text("builtin"), text("connector"), text("tool")) {
                    (Some(builtin), _, _) => (None, builtin),
                    (None, Some(connector), Some(name)) => (Some(connector), name),
                    _ => continue,
                };
                // An item both documents list is kept once, as the first one.
                if seen.contains(&(connector.clone(), name.clone())) {
                    continue;
                }
                seen.push((connector.clone(), name.clone()));
                let mut item = doc.item(
                    DeclaredKind::Tool,
                    "tools",
                    Some(name.as_str()),
                    &["tools", index.as_str()],
                );
                item.reason = text("description");
                item.connector = connector.clone();
                if self.evaluate {
                    item = match connector {
                        None => item.unchecked(
                            "the format defines no built-in tool names and NeMo Fabric's tool names are adapter-native; the harness keeps its default tools",
                        ),
                        Some(connector) => match self.connectors.get(&connector) {
                            Some(ConnectorState::Usable) => item.unchecked(
                                "the tool comes with its connector's server, and the server's tools are known only when it runs",
                            ),
                            Some(ConnectorState::Gap(gap)) => item.not_mapped(
                                *gap,
                                format!("its connector `{connector}` is not mapped"),
                            ),
                            None => item.not_mapped(
                                MappingGap::NotResolved,
                                format!("its connector `{connector}` is not declared"),
                            ),
                        },
                    };
                }
                self.items.push(item);
            }
        }
    }

    fn custom_tools(&mut self) {
        for (doc, name, _) in self.entries(CUSTOM_TOOLS) {
            let mut item = doc.item(
                DeclaredKind::Part,
                CUSTOM_TOOLS,
                Some(name.as_str()),
                &[CUSTOM_TOOLS, name.as_str()],
            );
            if self.evaluate {
                item = self.custom_tool(item, &[], name);
            }
            self.items.push(item);
        }
    }

    /// A custom tool is code the application that hosts the agent supplies.
    /// With a binding the caller says which code that is. Else the caller's
    /// config can name the code for the adapter: a `tools.definitions` entry
    /// under the tool's name is kept as it is, the way a running connector's
    /// server entry is. With neither, nothing supplies the tool. `place`
    /// holds the called agents that lead to it.
    ///
    /// The document gives a description and an input schema. It cannot fill a
    /// definition itself: the `kind` and the `ref` of one are the adapter's
    /// words for code that is installed where the harness runs.
    fn custom_tool(&self, item: DeclaredItem, place: &[String], name: &str) -> DeclaredItem {
        let key = binding_key(place, CUSTOM_TOOLS, name);
        if let Some(target) = self.options.bindings.get(&key) {
            return item.bound(
                target,
                "the caller supplies this tool. The loader did not import, resolve or run the target",
            );
        }
        // The config's tool definitions are the tools of the agent that runs.
        let accepted = self.support.accepts(AdapterConfigField::ToolDefinitions);
        let defined = place.is_empty()
            && self
                .base
                .tools
                .as_ref()
                .is_some_and(|tools| tools.definitions.contains_key(name));
        if accepted && defined {
            let mut item = item.unchecked(
                "the config's own tool definition is kept. The code it names was not compared with the description and the input schema the document gives",
            );
            item.mapped_to = vec![format!("tools.definitions.{name}")];
            return item;
        }
        let missing = if !place.is_empty() {
            "nothing supplies this tool of a called agent".to_string()
        } else if accepted {
            format!("the config has no tools.definitions.{name} entry that names the tool's code")
        } else {
            "the adapter does not accept tools.definitions, so the config cannot name the tool's code".to_string()
        };
        item.not_mapped(
            MappingGap::NotBound,
            format!("{missing}{}", self.key_to_bind("supplies the tool", &key)),
        )
    }

    /// The key a caller that gives bindings would add for a line that has
    /// none. A load with no binding keeps the text it always had.
    fn key_to_bind(&self, does: &str, key: &str) -> String {
        if self.options.bindings.is_empty() {
            return String::new();
        }
        format!(". A caller that {does} gives a binding for `{key}`")
    }

    // ---- called agents ----

    /// NeMo Fabric runs no called agents itself. A caller that runs one says
    /// so with a binding, and then answers for what that agent declares. With
    /// no binding, an agent whose document is in the package goes to the
    /// adapter under `nemo_fabric.agents.v1`, for the harness to run.
    fn called_agents(&mut self) -> Result<()> {
        for (doc, name, entry) in self.entries(AGENTS) {
            let item = self.part_item(&doc, PartField::Agents, name, entry);
            let (part, left_out) = self.lookup(PartField::Agents, name);
            self.called_agent(
                &doc,
                item,
                part,
                left_out,
                vec![name.clone()],
                &[AGENTS, name.as_str()],
            )?;
        }
        Ok(())
    }

    /// One called agent. `place` holds the names that lead to it, its own
    /// last, and `segments` the same steps as a pointer.
    ///
    /// A binding accounts for running the agent, never for its content: the
    /// agent's document is in the package and was checked against the lock,
    /// or the agent is not bound. The same holds for an agent that goes to
    /// the adapter.
    fn called_agent(
        &mut self,
        root: &Doc<'a>,
        mut item: DeclaredItem,
        part: Option<&'a LockedPart>,
        left_out: bool,
        place: Vec<String>,
        segments: &[&str],
    ) -> Result<()> {
        let reading = self.reading;
        let (name, above) = place.split_last().expect("a called agent has a name");
        let key = binding_key(above, AGENTS, name);
        let target = self.options.bindings.get(&key);
        // The agent as the reader started it: its document was read and each
        // of its variables has a value.
        let started = reading.called.iter().find(|called| called.place == place);
        let caller_runs_parent = above.split_last().is_some_and(|(parent, further)| {
            let parent = binding_key(further, AGENTS, parent);
            self.options.bindings.contains_key(&parent)
        });
        // Who runs the agent, when its document is here: the caller that
        // bound it, or the harness that gets an entry for it.
        let mut run_by = (target.is_some() && !left_out).then_some(RunBy::Caller);
        if self.evaluate {
            let not_applied = if target.is_some() {
                "the binding was not applied: "
            } else {
                ""
            };
            item = match (target, started, part.map(|part| part.integrity)) {
                _ if left_out => item.not_mapped(
                    MappingGap::NotResolved,
                    format!("{not_applied}the lock leaves this optional reference out"),
                ),
                (Some(target), Some(_), _) => item.bound(
                    target,
                    "the caller runs this agent, with the name, description and instructions its document gives. The document was checked against the lock. Everything else it declares follows, each thing on a line of its own: a binding accounts for a custom tool and for an agent it lists, and for nothing else",
                ),
                // An agent that a called agent lists. A caller that runs the
                // called agent runs this one too, and says so with a binding.
                // An entry the harness gets has no place for it.
                (None, Some(_), _) if !above.is_empty() => item.not_mapped(
                    MappingGap::NoFabricField,
                    if caller_runs_parent {
                        format!(
                            "NeMo Fabric runs no called agents. The entry was checked against the lock{}",
                            self.key_to_bind("runs the agent", &key)
                        )
                    } else {
                        "the entry an adapter gets for a called agent has no place for the agents it lists".to_string()
                    },
                ),
                (None, Some(called), _) => {
                    // A name the config already holds under the key is the
                    // caller's own agent.
                    if self
                        .config
                        .extensions
                        .get(AGENTS_KEY)
                        .is_some_and(|agents| agents.get(name).is_some())
                    {
                        return Err(conflict(
                            &format!("{AGENTS_KEY}.{name}"),
                            "the config sets a called agent under the name of an agent the package lists",
                        ));
                    }
                    match self.called_entry(name, called) {
                        Ok(entry) => {
                            let item = item
                                .mapped(vec![format!("{AGENTS_KEY}.{name}")])
                                .with_detail(entry.summary());
                            run_by = Some(RunBy::Harness(entry));
                            item
                        }
                        Err((gap, reason)) => item.not_mapped(
                            gap,
                            format!("{reason}{}", self.key_to_bind("runs the agent", &key)),
                        ),
                    }
                }
                (Some(target), None, Some(Integrity::Pointer)) => {
                    let mut item = item.unchecked(
                        "the caller calls this running agent. The pinned version and card digest were not compared with it",
                    );
                    item.bound_to = Some(target.clone());
                    item
                }
                (None, None, Some(Integrity::Pointer)) => item.not_mapped(
                    MappingGap::NoFabricField,
                    format!(
                        "NeMo Fabric hands no running agent to a harness. The entry was checked against the lock{}",
                        self.key_to_bind("runs the agent", &key)
                    ),
                ),
                (_, None, Some(Integrity::Absent)) => item.not_mapped(
                    MappingGap::ContentAbsent,
                    format!(
                        "{not_applied}the agent's document is pinned and is not in the package; the loader fetches nothing"
                    ),
                ),
                (_, None, Some(Integrity::NotVerifiable)) => item.not_mapped(
                    MappingGap::ContentNotVerifiable,
                    format!(
                        "{not_applied}the agent's document is pinned by an artifact digest that cannot be worked out from files"
                    ),
                ),
                (_, None, Some(Integrity::Verified)) => item.not_mapped(
                    MappingGap::NoFabricField,
                    format!(
                        "{not_applied}the called agent could not be started. A note names the variable"
                    ),
                ),
                (_, None, _) => item.not_mapped(
                    MappingGap::NoFabricField,
                    format!("{not_applied}the agent's document was not read"),
                ),
            };
        }
        // An entry marked optional never stops a load, and neither does
        // anything the agent it lists declares.
        let optional = !item.required;
        self.items.push(item);
        if let (Some(run_by), Some(called)) = (run_by, started) {
            self.declared_by_called_agent(&Below {
                root,
                called,
                segments,
                run_by: &run_by,
                optional,
            })?;
        }
        Ok(())
    }

    /// The entry an adapter gets for a called agent with no binding, or why
    /// the adapter gets none.
    ///
    /// The description and the instructions are offered first. The turn limit
    /// and each skill folder are then offered one at a time, and the entry
    /// keeps each one the adapter's schema takes. The agents the config holds
    /// under the key stay as they are.
    fn called_entry(
        &mut self,
        name: &str,
        called: &'a CalledAgent,
    ) -> std::result::Result<CalledEntry, (MappingGap, String)> {
        let held = match self.config.extensions.get(AGENTS_KEY) {
            None => Map::new(),
            Some(Value::Object(held)) => held.clone(),
            Some(_) => {
                return Err((
                    MappingGap::UnsupportedShape,
                    held_in_another_shape(AGENTS_KEY),
                ));
            }
        };
        let offer = |mapper: &mut Self, members: &Map<String, Value>| {
            let mut agents = held.clone();
            agents.insert(name.to_string(), Value::Object(members.clone()));
            mapper
                .offer_extension(AGENTS_KEY, Value::Object(agents))
                .map_err(|reason| (MappingGap::AdapterDoesNotAccept, reason))
        };
        let mut members = Map::new();
        for member in ["description", "instructions"] {
            if let Some(text) = called.filled.get(member) {
                members.insert(member.to_string(), text.clone());
            }
        }
        offer(self, &members)?;
        let mut entry = CalledEntry::default();
        if let Some((applies, allowed_by)) = self.called_turn_limit(called) {
            let mut with = members.clone();
            with.insert("max_turns".to_string(), Value::from(applies));
            let kept = offer(self, &with).map(|()| format!("{AGENTS_KEY}.{name}.max_turns"));
            if kept.is_ok() {
                members = with;
            }
            entry.max_turns = Some(TurnLimit {
                applies,
                allowed_by,
                asked: called
                    .filled
                    .pointer("/limits/max_turns")
                    .and_then(whole_number),
                kept,
            });
        }
        let mut folders = Vec::new();
        let skills = called.filled.get("skills").and_then(Value::as_object);
        for (skill, reference) in skills.into_iter().flatten() {
            let (part, left_out) = self.called_lookup(called, PartField::Skills, skill);
            let folder = Self::carried(part, left_out, reference)
                .and_then(|(part, path)| folder_text(part, &path, "skill"));
            let kept = folder.and_then(|folder| {
                let mut with = members.clone();
                folders.push(Value::String(folder));
                with.insert(
                    "skills".to_string(),
                    serde_json::json!({"paths": folders.clone()}),
                );
                match offer(self, &with) {
                    Ok(()) => {
                        members = with;
                        Ok(format!(
                            "{AGENTS_KEY}.{name}.skills.paths.{}",
                            folders.len() - 1
                        ))
                    }
                    Err(refused) => {
                        folders.pop();
                        Err(refused)
                    }
                }
            });
            entry.skills.insert(skill.clone(), kept);
        }
        Ok(entry)
    }

    /// The turn limit a called agent runs under, and who allows that many
    /// turns: the smallest of its own limit, its caller's, and the limit the
    /// config now holds for the run. None when nothing limits it.
    ///
    /// The limits the reader gave the agent are the smaller of its own and
    /// its caller's, as the documents state them. The config's
    /// `runtime.max_turns` is what the caller really runs with.
    fn called_turn_limit(&self, called: &CalledAgent) -> Option<(u64, &'static str)> {
        const CALLER: &str = "the caller";
        const CONFIG: &str = "the config's runtime.max_turns";
        let stated = called.limits.get("max_turns").and_then(whole_number);
        let config = self.config.runtime.max_turns.map(u64::from);
        match (stated, config) {
            (Some(stated), Some(config)) if config < stated => Some((config, CONFIG)),
            (Some(stated), _) => Some((stated, CALLER)),
            (None, Some(config)) => Some((config, CONFIG)),
            (None, None) => None,
        }
    }

    /// What a called agent declares, in the order the things of the agent
    /// that runs are listed in. Its document is in the package, and the
    /// caller or the harness runs it.
    ///
    /// For an agent the caller runs, its custom tools and the agents it lists
    /// each need a binding of their own. For an agent the harness runs, the
    /// turn limit and the skill folders its entry carries are mapped. NeMo
    /// Fabric configures one agent, so it applies nothing else a called agent
    /// declares and compares none of it with a fact. Each such thing keeps a
    /// line, and neither a binding nor an entry accounts for it.
    fn declared_by_called_agent(&mut self, below: &Below<'_, 'a>) -> Result<()> {
        let reading = self.reading;
        let called = below.called;
        self.called_model_settings(below);
        self.called_entries(below, DeclaredKind::Limit, "limits");
        self.inherited_turn_limit(below);
        for field in ["skills", "plugins", "connectors", "hooks"] {
            self.called_entries(below, DeclaredKind::Part, field);
        }
        self.called_tools(below);
        for name in &called.custom_tools {
            let item = below.item(
                DeclaredKind::Part,
                CUSTOM_TOOLS,
                name,
                &[CUSTOM_TOOLS, name.as_str()],
            );
            // A caller that runs the agent supplies its tools. An entry the
            // harness gets has no place for one.
            let item = match below.run_by {
                RunBy::Caller if self.evaluate => self.custom_tool(item, &called.place, name),
                RunBy::Caller => item,
                RunBy::Harness(_) => self.not_applied(item, below.run_by),
            };
            self.items.push(item);
        }
        self.called_entries(below, DeclaredKind::Part, "packages");
        for listed in &called.agents {
            let mut item = below.item(
                DeclaredKind::Part,
                AGENTS,
                &listed.name,
                &[AGENTS, listed.name.as_str()],
            );
            item.required &= !listed.optional;
            item.address = listed.address.clone();
            let part = listed
                .address
                .as_ref()
                .and_then(|address| reading.parts.iter().find(|part| &part.address == address));
            let mut place = called.place.clone();
            place.push(listed.name.clone());
            let mut further = below.segments.to_vec();
            further.extend([AGENTS, listed.name.as_str()]);
            self.called_agent(
                below.root,
                item,
                part,
                listed.address.is_none(),
                place,
                &further,
            )?;
        }
        self.called_entries(below, DeclaredKind::Part, "policies");
        self.called_credentials(below);
        self.called_needs(below);
        self.called_reported_only(below, "outputs", OUTPUTS_NOT_USED);
        self.called_reported_only(below, "extensions", EXTENSIONS_SKIPPED);
        Ok(())
    }

    /// A called agent that states no turn limit runs under its caller's. The
    /// limit its entry was offered has a line at the place the agent would
    /// state one, so an adapter that takes no turn limit does not leave the
    /// caller's limit unsaid.
    fn inherited_turn_limit(&mut self, below: &Below<'_, 'a>) {
        let RunBy::Harness(entry) = below.run_by else {
            return;
        };
        let Some(limit) = entry
            .max_turns
            .as_ref()
            .filter(|limit| limit.asked.is_none())
        else {
            return;
        };
        let item = below.item(
            DeclaredKind::Limit,
            "limits",
            "max_turns",
            &["limits", "max_turns"],
        );
        self.items.push(match &limit.kept {
            Ok(path) => item.mapped(vec![path.clone()]).with_detail(limit.detail()),
            Err((gap, reason)) => item.not_mapped(
                *gap,
                format!(
                    "{reason}. The agent states no turn limit, and {} allows {}",
                    limit.allowed_by, limit.applies
                ),
            ),
        });
    }

    /// A thing of a called agent that NeMo Fabric has no place for.
    fn not_applied(&self, item: DeclaredItem, run_by: &RunBy) -> DeclaredItem {
        if !self.evaluate {
            return item;
        }
        item.not_mapped(
            MappingGap::NoFabricField,
            match run_by {
                RunBy::Caller => {
                    "NeMo Fabric configures one agent and applies nothing a called agent declares. The caller that runs the agent has to apply this, and no binding accounts for it"
                }
                RunBy::Harness(_) => {
                    "the entry an adapter gets for a called agent has no place for this, so the harness does not get it"
                }
            },
        )
    }

    /// The default model of a called agent and its settings. As for the agent
    /// that runs, none of them is required.
    fn called_model_settings(&mut self, below: &Below<'_, 'a>) {
        let Some(model) = below.object("model") else {
            return;
        };
        let parameters = model.get("parameters").and_then(Value::as_object);
        let settings = ["source", "revision"]
            .into_iter()
            .filter(|key| model.contains_key(*key))
            .map(|key| (key, vec!["model", key]))
            .chain(
                parameters
                    .into_iter()
                    .flatten()
                    .map(|(key, _)| (key.as_str(), vec!["model", "parameters", key.as_str()])),
            );
        for (name, tail) in settings {
            let mut item = below.item(DeclaredKind::ModelSetting, "model", name, &tail);
            item.required = false;
            let item = self.not_applied(item, below.run_by);
            self.items.push(item);
        }
    }

    /// The entries of one map field of a called agent: its limits, or the
    /// parts of one kind. An entry marked optional is not required. The turn
    /// limit and the skill folders that were offered for the agent's entry
    /// have the status the offer gave them. The values written on a skill
    /// have a line of their own, as they have for the agent that runs.
    fn called_entries(&mut self, below: &Below<'_, 'a>, kind: DeclaredKind, field: &str) {
        let reading = self.reading;
        let part_field = PartField::ALL
            .into_iter()
            .find(|part| part.as_str() == field);
        for (name, entry) in below.object(field).into_iter().flatten() {
            let mut item = below.item(kind, field, name, &[field, name.as_str()]);
            item.required &= !is_optional(entry);
            if let Some(part_field) = part_field {
                item.address = reading
                    .called_part(below.called, part_field, name)
                    .map(|part| part.address.clone());
            }
            let values = (part_field == Some(PartField::Skills)
                && entry.get("variables").is_some())
            .then(|| {
                let mut values = below.item(
                    DeclaredKind::PartValues,
                    field,
                    name,
                    &[field, name.as_str(), "variables"],
                );
                values.required = item.required;
                values.address = item.address.clone();
                values
            });
            let offered = match (below.run_by, field, name.as_str()) {
                (RunBy::Harness(entry), "limits", "max_turns") => entry
                    .max_turns
                    .as_ref()
                    .map(|limit| (&limit.kept, Some(limit.detail()))),
                (RunBy::Harness(entry), "skills", skill) => {
                    entry.skills.get(skill).map(|kept| (kept, None))
                }
                _ => None,
            };
            let item = match offered {
                Some((Ok(path), detail)) => {
                    let item = item.mapped(vec![path.clone()]);
                    match detail {
                        Some(detail) => item.with_detail(detail),
                        None => item,
                    }
                }
                Some((Err((gap, reason)), _)) => item.not_mapped(*gap, reason.clone()),
                None => self.not_applied(item, below.run_by),
            };
            self.items.push(item);
            // No entry an adapter gets carries the values, whoever runs the agent.
            self.items.extend(values.map(|values| match below.run_by {
                RunBy::Harness(_) => {
                    values.not_mapped(MappingGap::NoFabricField, SKILL_VALUES_NOT_HANDED)
                }
                RunBy::Caller => self.not_applied(values, below.run_by),
            }));
        }
    }

    fn called_tools(&mut self, below: &Below<'_, 'a>) {
        let tools = below.called.filled.get("tools").and_then(Value::as_array);
        for (index, tool) in tools.into_iter().flatten().enumerate() {
            let index = index.to_string();
            let text = |key: &str| tool.get(key).and_then(Value::as_str).map(str::to_string);
            let Some(name) = text("builtin").or_else(|| text("tool")) else {
                continue;
            };
            let mut item = below.item(
                DeclaredKind::Tool,
                "tools",
                &name,
                &["tools", index.as_str()],
            );
            item.reason = text("description");
            item.connector = text("connector");
            let item = self.not_applied(item, below.run_by);
            self.items.push(item);
        }
    }

    /// The platform of the caller binds a credential, as for the agent that runs.
    fn called_credentials(&mut self, below: &Below<'_, 'a>) {
        for (name, entry) in below.object("credentials").into_iter().flatten() {
            let mut item = below.item(
                DeclaredKind::Part,
                "credentials",
                name,
                &["credentials", name.as_str()],
            );
            item.reason = entry
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string);
            if self.evaluate {
                item = item.unchecked(CREDENTIAL_NOT_SEEN);
            }
            self.items.push(item);
        }
    }

    /// The needs of a called agent. The facts of a load describe the place the
    /// agent that runs is started in and the model it runs on. The loader has
    /// no such fact for a called agent, so none of its needs could be checked.
    fn called_needs(&mut self, below: &Below<'_, 'a>) {
        for line in need_lines(&below.called.filled, None, &Facts::default()) {
            let tail: Vec<String> = line
                .pointer
                .split('/')
                .skip(1)
                .map(|segment| segment.replace("~1", "/").replace("~0", "~"))
                .collect();
            let tail: Vec<&str> = tail.iter().map(String::as_str).collect();
            let field = tail.first().copied().unwrap_or("runtime");
            let name = tail.last().copied().unwrap_or(field);
            let mut item = below.item(DeclaredKind::Need, field, name, &tail);
            item.required &= !line.optional;
            item.connector = line.connector;
            item.reason = line.reason;
            if self.evaluate {
                item = item.unchecked(match below.run_by {
                    RunBy::Caller => {
                        "the loader has no fact about where the caller runs this agent or which model runs it"
                    }
                    RunBy::Harness(_) => {
                        "the facts of a load are about the agent that runs. The loader has no fact about where the harness runs this agent or which model runs it"
                    }
                });
            }
            self.items.push(item);
        }
    }

    fn called_reported_only(&mut self, below: &Below<'_, 'a>, field: &str, detail: &str) {
        for (name, _) in below.object(field).into_iter().flatten() {
            let mut item = below.item(DeclaredKind::Field, field, name, &[field, name.as_str()]);
            item.required = false;
            if self.evaluate {
                item = item.skipped(detail);
            }
            self.items.push(item);
        }
    }

    fn credentials(&mut self) {
        for (doc, name, entry) in self.entries("credentials") {
            let mut item = doc.item(
                DeclaredKind::Part,
                "credentials",
                Some(name.as_str()),
                &["credentials", name.as_str()],
            );
            item.reason = entry
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string);
            if self.evaluate {
                item = item.unchecked(CREDENTIAL_NOT_SEEN);
            }
            self.items.push(item);
        }
    }

    // ---- needs ----

    /// Needs keep the status the need check gave them, with or without a config.
    fn needs(&mut self, needs: &[NeedLine]) {
        for line in needs {
            let segments: Vec<String> = line
                .pointer
                .split('/')
                .skip(1)
                .map(|segment| segment.replace("~1", "/").replace("~0", "~"))
                .collect();
            self.items.push(DeclaredItem {
                id: format!("{}:{}", stated_by_text(line.stated_by), line.pointer),
                kind: DeclaredKind::Need,
                stated_by: line.stated_by,
                document: line.document.clone(),
                field: segments
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "runtime".to_string()),
                name: segments.last().cloned(),
                address: None,
                required: !line.optional,
                status: match line.status {
                    NeedStatus::MetHere => DeclaredStatus::MetHere,
                    NeedStatus::MetRemotely => DeclaredStatus::MetRemotely,
                    NeedStatus::NotMet => DeclaredStatus::NotMet,
                    NeedStatus::CouldNotBeChecked => DeclaredStatus::CouldNotBeChecked,
                },
                gap: None,
                connector: line.connector.clone(),
                effect: ItemEffect::None,
                mapped_to: Vec::new(),
                bound_to: None,
                reason: line.reason.clone(),
                detail: line.detail.clone(),
            });
        }
    }

    // ---- outputs and extensions ----

    fn reported_only(&mut self, field: &str, detail: &str) {
        for (doc, name, _) in self.entries(field) {
            let mut item = doc.item(
                DeclaredKind::Field,
                field,
                Some(name.as_str()),
                &[field, name.as_str()],
            );
            item.required = false;
            if self.evaluate {
                item = item.skipped(detail);
            }
            self.items.push(item);
        }
    }
}

/// Why NeMo Fabric cannot hold a limit, and the component that enforces it.
///
/// NeMo Fabric bounds one invocation of one agent. It does not see the model
/// calls of the agents that agent calls, and it starts none of them.
fn enforced_elsewhere(limit: &str) -> &'static str {
    match limit {
        "max_total_tokens" => {
            "NeMo Fabric does not count tokens across model calls. A token budget for the agent and the agents it calls is enforced where every one of their model calls passes: a model gateway"
        }
        "max_depth" => {
            "NeMo Fabric starts no called agent. How deep calls go is enforced by whatever starts a called agent: the harness, or a gateway in front of it"
        }
        "max_parallel" => {
            "NeMo Fabric starts no called agent. How many run at one time is enforced by whatever starts a called agent: the harness, or a gateway in front of it"
        }
        _ => "NeMo Fabric cannot enforce this limit",
    }
}

/// The config's own value under a root extension key is the caller's. The
/// loader adds to a value in the shape it writes, and replaces none.
fn held_in_another_shape(key: &str) -> String {
    format!("the config holds `{key}` in a shape the loader cannot add to")
}

/// A whole number, also when it is written with a zero fraction such as `10.0`.
fn whole_number(value: &Value) -> Option<u64> {
    if let Some(number) = value.as_u64() {
        return Some(number);
    }
    let number = value.as_f64()?;
    (number.fract() == 0.0 && (0.0..=u64::MAX as f64).contains(&number)).then_some(number as u64)
}

fn stated_by_text(stated_by: StatedBy) -> &'static str {
    match stated_by {
        StatedBy::Agent => "agent",
        StatedBy::Definition => "definition",
        StatedBy::CalledAgent => "called_agent",
    }
}

/// Who runs a called agent whose document is in the package.
enum RunBy {
    /// The caller, which gave a binding for it.
    Caller,
    /// The harness, which gets an entry for it from the adapter.
    Harness(CalledEntry),
}

/// What the adapter's entry for a called agent carries beyond the agent's
/// description and instructions.
#[derive(Default)]
struct CalledEntry {
    max_turns: Option<TurnLimit>,
    /// Each skill of the agent, by the skill's key: the config path its
    /// folder has, or why the entry does not carry it.
    skills: BTreeMap<String, Offered>,
}

/// The config path of one member offered for a called agent's entry, or why
/// the entry does not carry it.
type Offered = std::result::Result<String, (MappingGap, String)>;

impl CalledEntry {
    /// What the harness gets for the agent, for the agent's own line.
    fn summary(&self) -> String {
        let mut carried = "its description and instructions".to_string();
        if let Some(limit) = self.max_turns.as_ref().filter(|limit| limit.kept.is_ok()) {
            carried.push_str(&format!(", max_turns {}", limit.applies));
        }
        match self.skills.values().filter(|kept| kept.is_ok()).count() {
            0 => {}
            1 => carried.push_str(", 1 skill folder"),
            folders => carried.push_str(&format!(", {folders} skill folders")),
        }
        format!(
            "handed to the adapter with {carried}. The document was checked against the lock. Everything else it declares follows, each thing on a line of its own"
        )
    }
}

/// The turn limit offered for a called agent's entry.
struct TurnLimit {
    /// The smallest of the agent's own limit, its caller's and the config's.
    applies: u64,
    /// Who allows that many turns, when it is not the agent itself: the
    /// caller, or the config.
    allowed_by: &'static str,
    /// The limit the agent's own document asks for, when it states one.
    asked: Option<u64>,
    kept: Offered,
}

impl TurnLimit {
    /// The number that applies, and where it comes from when it is not the
    /// agent's own.
    fn detail(&self) -> String {
        let (applies, allowed_by) = (self.applies, self.allowed_by);
        match self.asked {
            Some(asked) if asked > applies => {
                format!("{applies} applies, lowered from {asked}: {allowed_by} allows {applies}")
            }
            Some(_) => format!("{applies} applies"),
            None => format!(
                "{applies} applies: the agent states no turn limit, and {allowed_by} allows {applies}"
            ),
        }
    }
}

/// A called agent whose document is in the package, and where its entry sits.
struct Below<'b, 'a> {
    /// The document of the agent that runs, or of the definition, that leads here.
    root: &'b Doc<'a>,
    called: &'a CalledAgent,
    /// The pointer of the entry that lists the called agent.
    segments: &'b [&'b str],
    run_by: &'b RunBy,
    /// The entry that lists the called agent, or one on the way to it, is
    /// marked optional. Nothing below such an entry is required.
    optional: bool,
}

impl<'a> Below<'_, 'a> {
    /// A line for a thing the called agent declares. `tail` is its pointer in
    /// the called agent's document. The id goes on below the entry that lists
    /// the called agent, so it is one id for one place.
    fn item(&self, kind: DeclaredKind, field: &str, name: &str, tail: &[&str]) -> DeclaredItem {
        let mut pointer = self.segments.to_vec();
        pointer.extend(tail);
        DeclaredItem {
            id: format!(
                "{}:{}",
                stated_by_text(self.root.stated_by),
                json_pointer(&pointer)
            ),
            kind,
            stated_by: StatedBy::CalledAgent,
            document: self.called.document.clone(),
            field: field.to_string(),
            name: Some(name.to_string()),
            address: None,
            required: !self.optional,
            status: DeclaredStatus::NotEvaluated,
            gap: None,
            connector: None,
            effect: ItemEffect::None,
            mapped_to: Vec::new(),
            bound_to: None,
            reason: None,
            detail: None,
        }
    }

    fn object(&self, key: &str) -> Option<&'a Map<String, Value>> {
        self.called.filled.get(key).and_then(Value::as_object)
    }
}

fn is_optional(entry: &Value) -> bool {
    entry.get("optional").and_then(Value::as_bool) == Some(true)
}

/// Whether two names name one package of a registry. Names on PyPI compare
/// after its normalising: lower case, and each run of `-`, `_` and `.` as one
/// `-`. Any other registry compares the names as written.
fn same_package(registry: &str, left: &str, right: &str) -> bool {
    if registry != "pypi" {
        return left == right;
    }
    let normal = |name: &str| {
        let mut normal = String::with_capacity(name.len());
        for character in name.chars() {
            if !matches!(character, '-' | '_' | '.') {
                normal.extend(character.to_lowercase());
            } else if !normal.ends_with('-') {
                normal.push('-');
            }
        }
        normal
    };
    normal(left) == normal(right)
}

/// The verified folder of a part as the text an adapter gets. An adapter takes
/// such a part as a folder, so one file is not a shape it takes.
fn folder_text(
    part: &LockedPart,
    path: &Path,
    kind: &str,
) -> std::result::Result<String, (MappingGap, String)> {
    if part.digest_kind == Some(DigestKind::File) {
        return Err((
            MappingGap::UnsupportedShape,
            format!("the {kind} is one file. An adapter takes a {kind} as a folder"),
        ));
    }
    path.to_str().map(str::to_string).ok_or_else(|| {
        (
            MappingGap::UnsupportedShape,
            format!("the path of the {kind}'s folder is not text an adapter can take"),
        )
    })
}

/// The format of the handlers in a verified hook folder: the `$schema` text
/// of its `hooks.json`. None when the file is not a JSON object that names one.
///
/// The bytes that are parsed are compared with the digest in the verified
/// folder listing, so the file that is read is the file that was hashed.
fn hook_format(
    part: &LockedPart,
    folder: &Path,
    listing: &FolderListing,
) -> PackageResult<Option<String>> {
    let subject = part.address.as_str();
    let Some(listed) = listing.files.get(HOOKS_FILE) else {
        return Ok(None);
    };
    let bytes = read_regular_file(&folder.join(HOOKS_FILE), subject, MAX_DOCUMENT_BYTES)?;
    let read = bytes_digest(&bytes);
    if read != *listed {
        return Err(PackageError::new(
            PackageErrorCode::DigestMismatch,
            subject,
            format!(
                "{HOOKS_FILE} changed after the hook was checked: checked {listed}, read {read}"
            ),
        ));
    }
    Ok(parse_document(&bytes, subject).ok().and_then(|document| {
        document
            .get("$schema")
            .and_then(Value::as_str)
            .map(str::to_string)
    }))
}

/// The source types whose content the loader can use or bind.
fn is_loadable_source(source: &str) -> bool {
    ["./", "git::", "oci::", "mcp::", "a2a::"]
        .iter()
        .any(|prefix| source.starts_with(prefix))
}

/// The document's default model, after variables.
fn declared_model_source(reading: &PackageReading) -> Option<String> {
    reading
        .agent
        .get("model")
        .and_then(|model| model.get("source"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// The wire name of an adapter config field, for example `models.temperature`.
fn field_name(field: AdapterConfigField) -> String {
    serde_json::to_value(field)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_default()
}

fn conflict(field: &str, message: impl Into<String>) -> crate::error::FabricError {
    PackageError::new(PackageErrorCode::ConfigConflict, field, message).into()
}

/// A config that only carries the entry document's name, for listing without a config.
fn placeholder_config(name: &str) -> FabricConfig {
    serde_json::from_value(serde_json::json!({
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": name},
        "runtime": {},
    }))
    .expect("the placeholder config is a valid FabricConfig")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::agent_package::adapter::ExtensionSchema;
    use crate::agent_package::model::{
        DocumentRecord, EntryType, FolderListing, LeftOutPart, LockRecord, PackageAction,
        ValueOrigin, VariableValue,
    };
    use crate::agent_package::reader::testing;
    use crate::agent_package::report::{PackageOutcome, apply_actions};
    use crate::config::{McpTransport, SkillDirectoryKind, SkillDuplicatePolicy};
    use crate::error::FabricError;

    const NAME: &str = "release-notes-agent";
    const SCHEMA: &str = "https://example.com/schemas/0.4.0/agent.schema.json";

    fn agent(members: Value) -> Value {
        let mut document = json!({"$schema": SCHEMA, "type": "agent", "name": NAME});
        document
            .as_object_mut()
            .expect("an object")
            .extend(members.as_object().expect("an object").clone());
        document
    }

    fn record(kind: DocumentKind, name: &str, address: &str) -> DocumentRecord {
        DocumentRecord {
            kind,
            name: name.to_string(),
            address: address.to_string(),
            path: PathBuf::from("/pkg/document.json"),
            digest: "sha256:00".to_string(),
        }
    }

    /// A reading whose entry document is the given agent, with no lock entries.
    fn reading(agent: Value) -> PackageReading {
        let name = agent["name"].as_str().expect("a name").to_string();
        let entry = record(DocumentKind::Agent, &name, "$");
        PackageReading {
            package_dir: PathBuf::from("/pkg"),
            entry_dir: PathBuf::from("/pkg"),
            entry: entry.clone(),
            agent_record: entry,
            lock: LockRecord {
                path: PathBuf::from("/pkg/oadl.lock.json"),
                sha256: "00".to_string(),
                generator: None,
                entries: 0,
            },
            definition: None,
            agent_as_written: agent.clone(),
            agent,
            variables: Vec::new(),
            parts: Vec::new(),
            left_out: Vec::new(),
            called: Vec::new(),
            listings: BTreeMap::new(),
            together: None,
            notes: Vec::new(),
        }
    }

    /// A reading whose entry document is a definition of the given agent.
    /// `runs` is the agent that runs: the two documents read together.
    fn reading_with_definition(agent: Value, definition: Value, runs: Value) -> PackageReading {
        let mut reading = reading(agent);
        reading.agent_record.address = "$/agent".to_string();
        reading.entry = record(
            DocumentKind::AgentDefinition,
            definition["name"].as_str().expect("a name"),
            "$",
        );
        reading.definition = Some(definition);
        reading.agent = runs;
        reading
    }

    fn part(
        field: PartField,
        name: &str,
        source: &str,
        source_type: SourceType,
        integrity: Integrity,
    ) -> LockedPart {
        let entry_type = match field {
            PartField::Skills => EntryType::Skill,
            PartField::Plugins => EntryType::Plugin,
            PartField::Connectors => EntryType::Connector,
            PartField::Hooks => EntryType::Hook,
            PartField::Packages => EntryType::Package,
            PartField::Policies => EntryType::Policy,
            PartField::Agents => EntryType::Agent,
        };
        LockedPart {
            address: format!("$/{}/{name}", field.as_str()),
            entry_type,
            field: Some(field),
            name: Some(name.to_string()),
            owner: "$".to_string(),
            stated_by: StatedBy::Agent,
            optional: false,
            pointer: Some(format!("/{}/{name}", field.as_str())),
            source_text: Some(source.to_string()),
            version_text: None,
            source_type,
            source_url: Some(source.to_string()),
            source_subpath: None,
            registry: None,
            version: "local".to_string(),
            commit: None,
            digest: Some("sha256:11".to_string()),
            digest_kind: None,
            interface: None,
            retracted: None,
            dependencies: Vec::new(),
            path: (integrity == Integrity::Verified)
                .then(|| PathBuf::from(format!("/pkg/{}/{name}", field.as_str()))),
            integrity,
            values: None,
        }
    }

    fn local(field: PartField, name: &str) -> LockedPart {
        part(
            field,
            name,
            &format!("./{}/{name}", field.as_str()),
            SourceType::Local,
            Integrity::Verified,
        )
    }

    fn left_out(field: PartField, name: &str, source: &str) -> LeftOutPart {
        LeftOutPart {
            field,
            name: name.to_string(),
            owner: "$".to_string(),
            stated_by: StatedBy::Agent,
            pointer: format!("/{}/{name}", field.as_str()),
            source_text: source.to_string(),
        }
    }

    fn support(accepts: &[AdapterConfigField], modes: &[InstructionMode]) -> AdapterSupport {
        AdapterSupport {
            adapter_id: "test.adapter".to_string(),
            accepts: accepts.iter().copied().collect(),
            instruction_modes: modes.to_vec(),
            skill_entries: Vec::new(),
            extension_schema: None,
        }
    }

    /// An adapter that accepts everything and lists these `skills.directories` entry kinds.
    fn with_skill_entries(kinds: &[SkillDirectoryKind]) -> AdapterSupport {
        AdapterSupport {
            skill_entries: kinds.to_vec(),
            ..everything()
        }
    }

    fn everything() -> AdapterSupport {
        support(
            &[
                AdapterConfigField::Models,
                AdapterConfigField::ModelTemperature,
                AdapterConfigField::ModelMaxTokens,
                AdapterConfigField::SystemInstructions,
                AdapterConfigField::MaxTurns,
                AdapterConfigField::Mcp,
                AdapterConfigField::Skills,
            ],
            &[InstructionMode::Replace, InstructionMode::Append],
        )
    }

    fn without(field: AdapterConfigField) -> AdapterSupport {
        let mut support = everything();
        support.accepts.remove(&field);
        if field == AdapterConfigField::SystemInstructions {
            support.instruction_modes.clear();
        }
        support
    }

    fn base(members: Value) -> FabricConfig {
        let mut config = json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": NAME},
            "harness": {"adapter_id": "test.adapter"},
            "runtime": {},
        });
        for (key, value) in members.as_object().expect("an object") {
            config[key] = value.clone();
        }
        serde_json::from_value(config).expect("a config")
    }

    fn map_with(
        reading: &PackageReading,
        base: FabricConfig,
        support: &AdapterSupport,
        options: &AgentPackageOptions,
    ) -> Mapped {
        map_with_facts(reading, base, support, options, &Facts::default())
    }

    fn map_with_facts(
        reading: &PackageReading,
        base: FabricConfig,
        support: &AdapterSupport,
        options: &AgentPackageOptions,
        facts: &Facts,
    ) -> Mapped {
        map_package(reading, base, support, options, &[], facts).expect("the package maps")
    }

    fn map(reading: &PackageReading, support: &AdapterSupport) -> Mapped {
        map_with(
            reading,
            base(json!({})),
            support,
            &AgentPackageOptions::default(),
        )
    }

    fn refusal(reading: &PackageReading, base: FabricConfig) -> (String, String) {
        match map_package(
            reading,
            base,
            &everything(),
            &AgentPackageOptions::default(),
            &[],
            &Facts::default(),
        ) {
            Err(FabricError::AgentPackage { code, subject, .. }) => (code, subject),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    fn item<'a>(mapped: &'a Mapped, id: &str) -> &'a DeclaredItem {
        mapped
            .items
            .iter()
            .find(|item| item.id == id)
            .unwrap_or_else(|| panic!("no item {id} in {:?}", ids(&mapped.items)))
    }

    fn ids(items: &[DeclaredItem]) -> Vec<&str> {
        items.iter().map(|item| item.id.as_str()).collect()
    }

    fn stopped_by(mapped: &mut Mapped) -> (PackageOutcome, Vec<String>) {
        apply_actions(&mut mapped.items, PackageAction::Stop, PackageAction::Warn)
    }

    // ---- name and description ----

    #[test]
    fn name_and_description_fill_metadata() {
        let reading = reading(agent(json!({"description": "Writes release notes."})));

        let mapped = map(&reading, &everything());

        assert_eq!(mapped.config.metadata.name, NAME);
        assert_eq!(
            mapped.config.metadata.description.as_deref(),
            Some("Writes release notes.")
        );
        let name = item(&mapped, "agent:/name");
        assert_eq!(name.status, DeclaredStatus::Mapped);
        assert_eq!(name.mapped_to, vec!["metadata.name"]);
        assert_eq!(name.kind, DeclaredKind::Field);
        assert_eq!(name.document, NAME);
        let description = item(&mapped, "agent:/description");
        assert_eq!(description.status, DeclaredStatus::Mapped);
        assert_eq!(description.mapped_to, vec!["metadata.description"]);
    }

    #[test]
    fn a_config_for_another_agent_is_a_conflict() {
        let reading = reading(agent(json!({})));
        let other = base(json!({"metadata": {"name": "another-agent"}}));

        assert_eq!(
            refusal(&reading, other),
            (
                "package_config_conflict".to_string(),
                "metadata.name".to_string()
            )
        );
    }

    #[test]
    fn a_different_description_is_a_conflict_and_the_same_one_is_not() {
        let reading = reading(agent(json!({"description": "Writes release notes."})));
        let different = base(json!({"metadata": {"name": NAME, "description": "Other."}}));
        let same = base(json!({
            "metadata": {"name": NAME, "description": "Writes release notes."}
        }));

        assert_eq!(
            refusal(&reading, different),
            (
                "package_config_conflict".to_string(),
                "metadata.description".to_string()
            )
        );
        let mapped = map_with(
            &reading,
            same,
            &everything(),
            &AgentPackageOptions::default(),
        );
        assert_eq!(
            mapped.config.metadata.description.as_deref(),
            Some("Writes release notes.")
        );
    }

    #[test]
    fn a_config_description_is_kept_when_the_package_has_none() {
        let reading = reading(agent(json!({})));
        let config = base(json!({"metadata": {"name": NAME, "description": "Caller text."}}));

        let mapped = map_with(
            &reading,
            config,
            &everything(),
            &AgentPackageOptions::default(),
        );

        assert_eq!(
            mapped.config.metadata.description.as_deref(),
            Some("Caller text.")
        );
    }

    // ---- instructions ----

    #[test]
    fn instructions_are_written_with_replace_when_the_adapter_offers_it() {
        let reading = reading(agent(json!({"instructions": "Write short notes."})));

        let mapped = map(&reading, &everything());

        let system = mapped
            .config
            .instructions
            .as_ref()
            .and_then(|instructions| instructions.system.as_ref())
            .expect("system instructions");
        assert_eq!(system.content, "Write short notes.");
        assert_eq!(system.mode, InstructionMode::Replace);
        assert_eq!(mapped.instruction_mode, Some(InstructionMode::Replace));
        let instructions = item(&mapped, "agent:/instructions");
        assert_eq!(instructions.status, DeclaredStatus::Mapped);
        assert_eq!(instructions.mapped_to, vec!["instructions.system.content"]);
        assert!(instructions.required);
    }

    #[test]
    fn instruction_mode_follows_the_request_and_what_the_adapter_offers() {
        use InstructionMode::{Append, Replace};
        // (requested, offered, expected)
        let table = [
            (None, vec![Replace, Append], Some(Replace)),
            (None, vec![Replace], Some(Replace)),
            (None, vec![Append], Some(Append)),
            (None, vec![], None),
            (Some(Append), vec![Replace, Append], Some(Append)),
            (Some(Append), vec![Replace], None),
            (Some(Replace), vec![Append], None),
        ];
        for (requested, offered, expected) in table {
            let support = support(&[AdapterConfigField::SystemInstructions], &offered);

            assert_eq!(
                pick_instruction_mode(requested, &support),
                expected,
                "{requested:?} {offered:?}"
            );
        }
    }

    #[test]
    fn a_requested_mode_the_adapter_does_not_offer_leaves_instructions_unmapped() {
        let reading = reading(agent(json!({"instructions": "Write short notes."})));
        let support = support(
            &[AdapterConfigField::SystemInstructions],
            &[InstructionMode::Replace],
        );
        let options = AgentPackageOptions {
            instruction_mode: Some(InstructionMode::Append),
            ..AgentPackageOptions::default()
        };

        let mapped = map_with(&reading, base(json!({})), &support, &options);

        assert!(mapped.config.instructions.is_none());
        let instructions = item(&mapped, "agent:/instructions");
        assert_eq!(instructions.status, DeclaredStatus::NotMapped);
        assert_eq!(instructions.gap, Some(MappingGap::AdapterDoesNotAccept));
    }

    #[test]
    fn instructions_follow_if_unsupported_when_the_adapter_does_not_accept_them() {
        let reading = reading(agent(json!({"instructions": "Write short notes."})));

        let mut mapped = map(&reading, &without(AdapterConfigField::SystemInstructions));

        assert!(mapped.config.instructions.is_none());
        assert_eq!(mapped.instruction_mode, None);
        let instructions = item(&mapped, "agent:/instructions");
        assert_eq!(instructions.status, DeclaredStatus::NotMapped);
        assert_eq!(instructions.gap, Some(MappingGap::AdapterDoesNotAccept));
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(stopped, vec!["agent:/instructions"]);
    }

    #[test]
    fn config_instructions_conflict_with_the_agents_and_are_kept_without_them() {
        let config = || {
            base(json!({
                "instructions": {"system": {"content": "Caller text.", "mode": "append"}}
            }))
        };
        let with_instructions = reading(agent(json!({"instructions": "Write short notes."})));
        let without_instructions = reading(agent(json!({})));

        assert_eq!(
            refusal(&with_instructions, config()),
            (
                "package_config_conflict".to_string(),
                "instructions.system".to_string()
            )
        );
        let mapped = map_with(
            &without_instructions,
            config(),
            &everything(),
            &AgentPackageOptions::default(),
        );
        assert_eq!(
            mapped
                .config
                .instructions
                .expect("instructions")
                .system
                .expect("system")
                .content,
            "Caller text."
        );
    }

    // ---- model ----

    #[test]
    fn the_config_model_runs_and_takes_unset_document_parameters() {
        let reading = reading(agent(json!({
            "model": {
                "source": "ollama::llama3.2:3b",
                "revision": "abc",
                "parameters": {"temperature": 0.2, "max_output_tokens": 800, "effort": "low"}
            }
        })));
        let config = base(json!({
            "models": {"default": {"provider": "openai", "model": "served", "temperature": 0.7}}
        }));

        let mapped = map_with(
            &reading,
            config,
            &everything(),
            &AgentPackageOptions::default(),
        );

        let model = &mapped.config.models["default"];
        assert_eq!(
            (model.provider.as_str(), model.model.as_str()),
            ("openai", "served")
        );
        assert_eq!(model.temperature, Some(0.7));
        assert_eq!(model.max_tokens, Some(800));
        assert_eq!(mapped.model.origin, ModelOrigin::Caller);
        assert_eq!(
            mapped.model.declared_source.as_deref(),
            Some("ollama::llama3.2:3b")
        );
        assert_eq!(mapped.model.provider.as_deref(), Some("openai"));
        assert_eq!(mapped.model.model.as_deref(), Some("served"));
        assert_eq!(mapped.model.parameters_applied, vec!["max_output_tokens"]);
        assert_eq!(
            mapped.model.parameters_not_applied,
            vec!["effort", "temperature"]
        );
        let source = item(&mapped, "agent:/model/source");
        assert_eq!(source.status, DeclaredStatus::Skipped);
        assert!(!source.required);
        assert_eq!(source.kind, DeclaredKind::ModelSetting);
        let revision = item(&mapped, "agent:/model/revision");
        assert_eq!(revision.status, DeclaredStatus::NotMapped);
        assert_eq!(revision.gap, Some(MappingGap::NoFabricField));
        assert_eq!(
            item(&mapped, "agent:/model/parameters/temperature").status,
            DeclaredStatus::Skipped
        );
        let tokens = item(&mapped, "agent:/model/parameters/max_output_tokens");
        assert_eq!(tokens.status, DeclaredStatus::Mapped);
        assert_eq!(tokens.mapped_to, vec!["models.default.max_tokens"]);
        let effort = item(&mapped, "agent:/model/parameters/effort");
        assert_eq!(effort.status, DeclaredStatus::NotMapped);
        assert_eq!(effort.gap, Some(MappingGap::NoFabricField));
    }

    #[test]
    fn a_model_offered_under_the_document_name_becomes_the_default() {
        let reading = reading(agent(json!({
            "model": {
                "source": "ollama::mistral-small3.2:24b",
                "parameters": {"temperature": 0.2, "max_output_tokens": 4096}
            }
        })));
        let config = base(json!({
            "models": {
                "ollama::mistral-small3.2:24b": {
                    "provider": "openai",
                    "model": "mistral-small3.2:24b",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "max_tokens": 2048
                },
                "summary": {"provider": "openai", "model": "llama3.2:3b"}
            }
        }));

        let mapped = map_with(
            &reading,
            config.clone(),
            &everything(),
            &AgentPackageOptions::default(),
        );

        // The offered entry moved as the caller wrote it, and took the one
        // setting the caller left unset. The other entry stays.
        let mut expected = config.models.clone();
        let mut selected = expected
            .remove("ollama::mistral-small3.2:24b")
            .expect("the offered entry");
        selected.temperature = Some(0.2);
        expected.insert("default".to_string(), selected);
        assert_eq!(mapped.config.models, expected);
        assert_eq!(mapped.config.models["default"].max_tokens, Some(2048));
        assert_eq!(mapped.model.origin, ModelOrigin::Package);
        assert_eq!(mapped.model.provider.as_deref(), Some("openai"));
        assert_eq!(mapped.model.model.as_deref(), Some("mistral-small3.2:24b"));
        assert_eq!(mapped.model.parameters_applied, vec!["temperature"]);
        let source = item(&mapped, "agent:/model/source");
        assert_eq!(source.status, DeclaredStatus::Mapped);
        assert_eq!(source.mapped_to, vec!["models.default"]);
        assert!(!source.required);
        assert_eq!(
            item(&mapped, "agent:/model/parameters/temperature").mapped_to,
            vec!["models.default.temperature"]
        );
    }

    #[test]
    fn the_offered_name_is_the_document_model_after_variables() {
        let mut reading = reading(agent(json!({
            "model": {"source": {"var": "model"}},
            "variables": {"model": {"type": "string", "default": "ollama::llama3.2:3b"}}
        })));
        reading.agent["model"]["source"] = json!("ollama::mistral-small3.2:24b");
        let config = base(json!({
            "models": {
                "ollama::llama3.2:3b": {"provider": "openai", "model": "llama3.2:3b"},
                "ollama::mistral-small3.2:24b": {"provider": "openai", "model": "mistral-small3.2:24b"}
            }
        }));

        let mapped = map_with(
            &reading,
            config,
            &everything(),
            &AgentPackageOptions::default(),
        );

        assert_eq!(
            mapped.config.models.keys().collect::<Vec<_>>(),
            ["default", "ollama::llama3.2:3b"]
        );
        assert_eq!(
            mapped.config.models["default"].model,
            "mistral-small3.2:24b"
        );
        assert_eq!(mapped.model.origin, ModelOrigin::Package);
    }

    #[test]
    fn a_config_default_model_is_kept_when_the_document_model_is_offered_too() {
        let reading = reading(agent(json!({"model": {"source": "ollama::llama3.2:3b"}})));
        let config = base(json!({
            "models": {
                "default": {"provider": "openai", "model": "served"},
                "ollama::llama3.2:3b": {"provider": "openai", "model": "llama3.2:3b"}
            }
        }));

        let mapped = map_with(
            &reading,
            config.clone(),
            &everything(),
            &AgentPackageOptions::default(),
        );

        assert_eq!(mapped.config.models, config.models);
        assert_eq!(mapped.model.origin, ModelOrigin::Caller);
        assert_eq!(mapped.model.model.as_deref(), Some("served"));
        assert_eq!(
            item(&mapped, "agent:/model/source").status,
            DeclaredStatus::Skipped
        );
    }

    #[test]
    fn the_only_model_of_a_config_is_the_callers_choice() {
        // Planning and the adapters run the one model of a config that sets
        // no default, whatever its name.
        let reading = reading(agent(json!({
            "model": {"source": "ollama::mistral-small3.2:24b", "parameters": {"temperature": 0.2}}
        })));
        let config = base(json!({
            "models": {"summary": {"provider": "openai", "model": "llama3.2:3b"}}
        }));

        let mut mapped = map_with(
            &reading,
            config.clone(),
            &everything(),
            &AgentPackageOptions::default(),
        );

        let mut expected = config.models.clone();
        expected.get_mut("summary").expect("the entry").temperature = Some(0.2);
        assert_eq!(mapped.config.models, expected);
        assert_eq!(mapped.model.origin, ModelOrigin::Caller);
        assert_eq!(mapped.model.provider.as_deref(), Some("openai"));
        assert_eq!(mapped.model.model.as_deref(), Some("llama3.2:3b"));
        assert_eq!(mapped.model.parameters_applied, vec!["temperature"]);
        let source = item(&mapped, "agent:/model/source");
        assert_eq!(source.status, DeclaredStatus::Skipped);
        assert!(
            source
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("models.summary"))
        );
        assert_eq!(
            item(&mapped, "agent:/model/parameters/temperature").mapped_to,
            vec!["models.summary.temperature"]
        );
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Ready);
        assert!(stopped.is_empty());
    }

    #[test]
    fn several_models_with_no_default_and_none_offered_select_no_model() {
        // Planning refuses this config. The line says why before it does.
        let reading = reading(agent(json!({
            "model": {"source": "ollama::mistral-small3.2:24b", "parameters": {"temperature": 0.2}}
        })));
        let config = base(json!({
            "models": {
                "summary": {"provider": "openai", "model": "llama3.2:3b"},
                "vision": {"provider": "openai", "model": "llava:13b"}
            }
        }));

        let mapped = map_with(
            &reading,
            config.clone(),
            &everything(),
            &AgentPackageOptions::default(),
        );

        assert_eq!(mapped.config.models, config.models);
        assert_eq!(mapped.model.origin, ModelOrigin::HarnessDefault);
        assert_eq!(mapped.model.model, None);
        let source = item(&mapped, "agent:/model/source");
        assert_eq!(source.status, DeclaredStatus::NotMapped);
        assert_eq!(source.gap, Some(MappingGap::ModelNotConfigured));
        assert!(
            source
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("`ollama::mistral-small3.2:24b`"))
        );
        assert!(!source.required);
        assert_eq!(
            item(&mapped, "agent:/model/parameters/temperature").gap,
            Some(MappingGap::ModelNotConfigured)
        );
    }

    #[test]
    fn no_model_is_written_when_the_config_offers_none() {
        // A catalog name is not a NeMo Fabric provider, so the loader writes no
        // model of its own, whatever the adapter accepts.
        let table = [
            ("ollama::llama3.2:3b", everything()),
            ("hf::meta-llama/Llama-3.3-70B-Instruct", everything()),
            ("test::test-model", everything()),
            ("registry.example.com/models/m", everything()),
            ("ollama::llama3", without(AdapterConfigField::Models)),
        ];
        for (source, support) in table {
            let reading = reading(agent(json!({
                "model": {"source": source, "parameters": {"temperature": 0.2}}
            })));

            let mut mapped = map(&reading, &support);

            assert!(mapped.config.models.is_empty(), "{source}");
            assert_eq!(mapped.model.declared_source.as_deref(), Some(source));
            assert_eq!(mapped.model.origin, ModelOrigin::HarnessDefault);
            assert_eq!(mapped.model.provider, None);
            assert_eq!(mapped.model.model, None);
            let item_source = item(&mapped, "agent:/model/source");
            assert_eq!(item_source.status, DeclaredStatus::NotMapped);
            assert_eq!(item_source.gap, Some(MappingGap::ModelNotConfigured));
            assert!(item_source.mapped_to.is_empty());
            assert!(!item_source.required);
            let temperature = item(&mapped, "agent:/model/parameters/temperature");
            assert_eq!(temperature.status, DeclaredStatus::NotMapped);
            assert_eq!(temperature.gap, Some(MappingGap::ModelNotConfigured));
            assert_eq!(mapped.model.parameters_not_applied, vec!["temperature"]);
            let (outcome, stopped) = stopped_by(&mut mapped);
            assert_eq!(outcome, PackageOutcome::Ready);
            assert!(stopped.is_empty());
        }
    }

    #[test]
    fn model_parameters_need_the_adapter_to_accept_them() {
        let reading = reading(agent(json!({
            "model": {
                "source": "test::m",
                "parameters": {"temperature": 0.2, "max_output_tokens": 800}
            }
        })));
        let support = support(&[AdapterConfigField::Models], &[]);
        let config = base(json!({
            "models": {"default": {"provider": "openai", "model": "served"}}
        }));

        let mapped = map_with(&reading, config, &support, &AgentPackageOptions::default());

        let model = &mapped.config.models["default"];
        assert_eq!(model.temperature, None);
        assert_eq!(model.max_tokens, None);
        for id in [
            "agent:/model/parameters/temperature",
            "agent:/model/parameters/max_output_tokens",
        ] {
            assert_eq!(item(&mapped, id).status, DeclaredStatus::NotMapped);
            assert_eq!(
                item(&mapped, id).gap,
                Some(MappingGap::AdapterDoesNotAccept)
            );
            assert!(!item(&mapped, id).required);
        }
    }

    // ---- limits ----

    #[test]
    fn limits_map_where_fabric_has_a_field_and_the_adapter_accepts_it() {
        let reading = reading(agent(json!({
            "limits": {
                "max_turns": 12,
                "timeout_seconds": 600,
                "max_parallel": 2,
                "max_depth": 1,
                "max_total_tokens": 100000
            }
        })));

        let mut mapped = map(&reading, &everything());

        assert_eq!(mapped.config.runtime.max_turns, Some(12));
        assert_eq!(mapped.config.runtime.timeout_seconds, Some(600.0));
        let turns = item(&mapped, "agent:/limits/max_turns");
        assert_eq!(turns.status, DeclaredStatus::Mapped);
        assert_eq!(turns.mapped_to, vec!["runtime.max_turns"]);
        assert_eq!(turns.kind, DeclaredKind::Limit);
        let timeout = item(&mapped, "agent:/limits/timeout_seconds");
        assert_eq!(timeout.status, DeclaredStatus::Mapped);
        assert_eq!(timeout.mapped_to, vec!["runtime.timeout_seconds"]);
        // (a limit NeMo Fabric cannot hold, the component its line names)
        for (name, enforcer) in [
            ("max_parallel", "whatever starts a called agent"),
            ("max_depth", "whatever starts a called agent"),
            ("max_total_tokens", "a model gateway"),
        ] {
            let limit = item(&mapped, &format!("agent:/limits/{name}"));
            assert_eq!(limit.status, DeclaredStatus::NotMapped, "{name}");
            assert_eq!(limit.gap, Some(MappingGap::NoFabricField));
            assert!(limit.required);
            let detail = limit.detail.as_deref().expect("a detail");
            assert!(detail.contains(enforcer), "{name}: {detail}");
        }
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(
            stopped,
            vec![
                "agent:/limits/max_depth",
                "agent:/limits/max_parallel",
                "agent:/limits/max_total_tokens"
            ]
        );
    }

    #[test]
    fn max_turns_needs_the_adapter_and_timeout_does_not() {
        let reading = reading(agent(json!({
            "limits": {"max_turns": 12, "timeout_seconds": 600}
        })));

        let mapped = map(&reading, &without(AdapterConfigField::MaxTurns));

        assert_eq!(mapped.config.runtime.max_turns, None);
        assert_eq!(mapped.config.runtime.timeout_seconds, Some(600.0));
        let turns = item(&mapped, "agent:/limits/max_turns");
        assert_eq!(turns.status, DeclaredStatus::NotMapped);
        assert_eq!(turns.gap, Some(MappingGap::AdapterDoesNotAccept));
        assert_eq!(
            item(&mapped, "agent:/limits/timeout_seconds").status,
            DeclaredStatus::Mapped
        );
    }

    #[test]
    fn the_smaller_of_the_config_and_package_limits_applies_and_is_noted() {
        let reading = reading(agent(json!({
            "limits": {"max_turns": 12, "timeout_seconds": 600}
        })));
        // (config max_turns, config timeout, expected max_turns, expected timeout, notes)
        let table = [
            (Some(5), Some(900.0), 5, 600.0, 2),
            (Some(40), Some(60.0), 12, 60.0, 2),
            (Some(12), Some(600.0), 12, 600.0, 0),
            (None, None, 12, 600.0, 0),
        ];
        for (turns, timeout, expected_turns, expected_timeout, notes) in table {
            let mut runtime = json!({});
            if let Some(turns) = turns {
                runtime["max_turns"] = json!(turns);
            }
            if let Some(timeout) = timeout {
                runtime["timeout_seconds"] = json!(timeout);
            }

            let mapped = map_with(
                &reading,
                base(json!({"runtime": runtime})),
                &everything(),
                &AgentPackageOptions::default(),
            );

            assert_eq!(mapped.config.runtime.max_turns, Some(expected_turns));
            assert_eq!(
                mapped.config.runtime.timeout_seconds,
                Some(expected_timeout)
            );
            assert_eq!(mapped.notes.len(), notes, "{turns:?} {timeout:?}");
            assert!(
                mapped
                    .notes
                    .iter()
                    .all(|note| note.code == NoteCode::LimitLowered)
            );
        }
    }

    #[test]
    fn a_limit_fabric_cannot_hold_is_reported_and_follows_if_unsupported() {
        // (limits, the id that is not mapped)
        let table = [
            (
                json!({"max_turns": 4_294_967_296u64}),
                "agent:/limits/max_turns",
            ),
            (
                json!({"timeout_seconds": 1e300}),
                "agent:/limits/timeout_seconds",
            ),
        ];
        for (limits, id) in table {
            let reading = reading(agent(json!({"limits": limits})));

            let mut mapped = map(&reading, &everything());

            assert_eq!(mapped.config.runtime.max_turns, None, "{id}");
            assert_eq!(mapped.config.runtime.timeout_seconds, None, "{id}");
            let limit = item(&mapped, id);
            assert_eq!(limit.status, DeclaredStatus::NotMapped);
            assert_eq!(limit.gap, Some(MappingGap::ValueOutOfRange));
            assert!(limit.required);
            let (outcome, stopped) = stopped_by(&mut mapped);
            assert_eq!(outcome, PackageOutcome::Stopped);
            assert_eq!(stopped, vec![id]);
            let (outcome, _) =
                apply_actions(&mut mapped.items, PackageAction::Warn, PackageAction::Warn);
            assert_eq!(outcome, PackageOutcome::Ready);
        }
    }

    #[test]
    fn a_whole_number_written_with_a_zero_fraction_is_a_turn_limit() {
        let reading = reading(agent(json!({"limits": {"max_turns": 10.0}})));

        let mapped = map(&reading, &everything());

        assert_eq!(mapped.config.runtime.max_turns, Some(10));
        assert_eq!(
            item(&mapped, "agent:/limits/max_turns").status,
            DeclaredStatus::Mapped
        );
        assert_eq!(whole_number(&json!(10.5)), None);
        assert_eq!(whole_number(&json!(-1.0)), None);
        assert_eq!(whole_number(&json!(7)), Some(7));
    }

    // ---- skills ----

    #[test]
    fn verified_skills_are_added_after_the_config_paths() {
        let mut reading = reading(agent(json!({
            "skills": {
                "release-notes": {"source": "./skills/release-notes"},
                "tone": {"source": "./skills/tone", "variables": {"style": "plain"}}
            }
        })));
        reading.parts = vec![
            local(PartField::Skills, "release-notes"),
            local(PartField::Skills, "tone"),
        ];
        let config = base(json!({"skills": {"paths": ["./skills/caller"]}}));

        let mut mapped = map_with(
            &reading,
            config,
            &everything(),
            &AgentPackageOptions::default(),
        );

        assert_eq!(
            mapped.config.skills.as_ref().expect("skills").paths,
            vec![
                PathBuf::from("./skills/caller"),
                PathBuf::from("/pkg/skills/release-notes"),
                PathBuf::from("/pkg/skills/tone"),
            ]
        );
        let skill = item(&mapped, "agent:/skills/release-notes");
        assert_eq!(skill.status, DeclaredStatus::Mapped);
        assert_eq!(skill.mapped_to, vec!["skills.paths.1"]);
        assert_eq!(skill.address.as_deref(), Some("$/skills/release-notes"));
        assert_eq!(skill.kind, DeclaredKind::Part);
        assert_eq!(skill.field, "skills");
        assert_eq!(skill.name.as_deref(), Some("release-notes"));
        assert_eq!(
            item(&mapped, "agent:/skills/tone").mapped_to,
            vec!["skills.paths.2"]
        );
        let values = item(&mapped, "agent:/skills/tone/variables");
        assert_eq!(values.kind, DeclaredKind::PartValues);
        assert_eq!(values.status, DeclaredStatus::NotMapped);
        assert_eq!(values.gap, Some(MappingGap::NoFabricField));
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(stopped, vec!["agent:/skills/tone/variables"]);
    }

    #[test]
    fn a_skill_that_is_one_file_is_not_mapped() {
        let mut reading = reading(agent(json!({
            "skills": {"review": {"source": "./skills/review.md"}}
        })));
        let mut skill = part(
            PartField::Skills,
            "review",
            "./skills/review.md",
            SourceType::Local,
            Integrity::Verified,
        );
        skill.digest_kind = Some(DigestKind::File);
        skill.path = Some(PathBuf::from("/pkg/skills/review.md"));
        reading.parts = vec![skill];

        let mut mapped = map(&reading, &everything());

        // The adapters take a skill as a folder that holds SKILL.md.
        assert!(mapped.config.skills.is_none());
        let review = item(&mapped, "agent:/skills/review");
        assert_eq!(review.status, DeclaredStatus::NotMapped);
        assert_eq!(review.gap, Some(MappingGap::UnsupportedShape));
        assert!(review.mapped_to.is_empty());
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(stopped, vec!["agent:/skills/review"]);
    }

    #[test]
    fn a_skill_folder_name_is_used_once() {
        // Three entries: two folders that share the name `review`, and one
        // folder named like a skill the config already lists.
        let mut reading = reading(agent(json!({
            "skills": {
                "left": {"source": "./left/review"},
                "right": {"source": "./right/review", "optional": true},
                "tone": {"source": "./skills/tone"}
            }
        })));
        let folder = |name: &str, source: &str| {
            let mut skill = part(
                PartField::Skills,
                name,
                source,
                SourceType::Local,
                Integrity::Verified,
            );
            skill.digest_kind = Some(DigestKind::Folder);
            skill.path = Some(PathBuf::from("/pkg").join(source.trim_start_matches("./")));
            skill
        };
        reading.parts = vec![
            folder("left", "./left/review"),
            folder("right", "./right/review"),
            folder("tone", "./skills/tone"),
        ];
        let config = base(json!({"skills": {"paths": ["/caller/skills/tone"]}}));

        let mut mapped = map_with(
            &reading,
            config,
            &everything(),
            &AgentPackageOptions::default(),
        );

        assert_eq!(
            mapped.config.skills.as_ref().expect("skills").paths,
            vec![
                PathBuf::from("/caller/skills/tone"),
                PathBuf::from("/pkg/left/review"),
            ]
        );
        let left = item(&mapped, "agent:/skills/left");
        assert_eq!(left.status, DeclaredStatus::Mapped);
        assert_eq!(left.mapped_to, vec!["skills.paths.1"]);
        assert!(
            left.detail
                .as_deref()
                .is_some_and(|detail| detail.contains("`review`")),
            "{:?}",
            left.detail
        );
        for id in ["agent:/skills/right", "agent:/skills/tone"] {
            let clash = item(&mapped, id);
            assert_eq!(clash.status, DeclaredStatus::NotMapped, "{id}");
            assert_eq!(clash.gap, Some(MappingGap::KeyClash), "{id}");
        }
        // The optional one never stops. The required one follows if_unsupported.
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(stopped, vec!["agent:/skills/tone"]);
    }

    #[test]
    fn a_config_that_sets_directories_gets_skill_entries() {
        let mut reading = reading(agent(json!({
            "skills": {
                "release-notes": {"source": "./skills/release-notes"},
                "tone": {"source": "./skills/tone"}
            }
        })));
        reading.parts = vec![
            local(PartField::Skills, "release-notes"),
            local(PartField::Skills, "tone"),
        ];
        let config = base(json!({"skills": {
            "directories": [{"folder": "./team"}, {"skill": "/caller/skills/caller"}],
            "bundled": {"exclude": ["github-auth"]},
            "duplicates": "last_wins"
        }}));

        let mut mapped = map_with(
            &reading,
            config.clone(),
            &with_skill_entries(&[SkillDirectoryKind::Skill, SkillDirectoryKind::Folder]),
            &AgentPackageOptions::default(),
        );

        // `directories` cannot be combined with `paths`, so nothing is added to `paths`.
        let skills = mapped.config.skills.as_ref().expect("skills");
        assert!(skills.paths.is_empty());
        assert_eq!(
            skills.directories,
            Some(vec![
                SkillDirectory::Folder(PathBuf::from("./team")),
                SkillDirectory::Skill(PathBuf::from("/caller/skills/caller")),
                SkillDirectory::Skill(PathBuf::from("/pkg/skills/release-notes")),
                SkillDirectory::Skill(PathBuf::from("/pkg/skills/tone")),
            ])
        );
        // The caller's other skill settings are kept as written.
        let caller = config.skills.as_ref().expect("skills");
        assert_eq!(skills.bundled, caller.bundled);
        assert_eq!(skills.duplicates, Some(SkillDuplicatePolicy::LastWins));
        let skill = item(&mapped, "agent:/skills/release-notes");
        assert_eq!(skill.status, DeclaredStatus::Mapped);
        assert_eq!(skill.mapped_to, vec!["skills.directories.2.skill"]);
        assert_eq!(
            item(&mapped, "agent:/skills/tone").mapped_to,
            vec!["skills.directories.3.skill"]
        );
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Ready);
        assert!(stopped.is_empty());
    }

    #[test]
    fn the_config_chooses_the_skill_form() {
        // (the caller's skills, whether the package's skill becomes a `skill` entry)
        let table = [
            (json!({"directories": []}), true),
            (json!({"directories": null}), false),
            (json!({"paths": []}), false),
            (json!({"bundled": {"exclude": ["github-auth"]}}), false),
        ];
        for (skills, structured) in table {
            let mut reading = reading(agent(json!({
                "skills": {"notes": {"source": "./skills/notes"}}
            })));
            reading.parts = vec![local(PartField::Skills, "notes")];

            let mapped = map_with(
                &reading,
                base(json!({"skills": skills.clone()})),
                &with_skill_entries(&[SkillDirectoryKind::Skill]),
                &AgentPackageOptions::default(),
            );

            let folder = PathBuf::from("/pkg/skills/notes");
            let written = mapped.config.skills.as_ref().expect("skills");
            let notes = item(&mapped, "agent:/skills/notes");
            assert_eq!(notes.status, DeclaredStatus::Mapped, "{skills}");
            if structured {
                assert!(written.paths.is_empty(), "{skills}");
                assert_eq!(
                    written.directories,
                    Some(vec![SkillDirectory::Skill(folder)]),
                    "{skills}"
                );
                assert_eq!(notes.mapped_to, vec!["skills.directories.0.skill"]);
            } else {
                assert_eq!(written.paths, vec![folder], "{skills}");
                assert_eq!(written.directories, None, "{skills}");
                assert_eq!(notes.mapped_to, vec!["skills.paths.0"]);
            }
        }
    }

    #[test]
    fn a_skill_entry_the_adapter_does_not_list_is_not_mapped() {
        // Planning refuses a `skill` entry for both adapters: one declares no
        // directory loading, the other lists `folder` entries only.
        for support in [
            everything(),
            with_skill_entries(&[SkillDirectoryKind::Folder]),
        ] {
            let mut reading = reading(agent(json!({
                "skills": {"notes": {"source": "./skills/notes"}}
            })));
            reading.parts = vec![local(PartField::Skills, "notes")];
            let config = base(json!({"skills": {"directories": []}}));

            let mut mapped = map_with(
                &reading,
                config.clone(),
                &support,
                &AgentPackageOptions::default(),
            );

            // The caller's skills are left as written: no entry and no path is added.
            assert_eq!(mapped.config.skills, config.skills);
            let notes = item(&mapped, "agent:/skills/notes");
            assert_eq!(notes.status, DeclaredStatus::NotMapped);
            assert_eq!(notes.gap, Some(MappingGap::AdapterDoesNotAccept));
            assert!(notes.mapped_to.is_empty());
            let (outcome, stopped) = stopped_by(&mut mapped);
            assert_eq!(outcome, PackageOutcome::Stopped);
            assert_eq!(stopped, vec!["agent:/skills/notes"]);
        }
    }

    #[test]
    fn a_skill_entry_of_the_config_takes_its_folder_name_first() {
        let mut reading = reading(agent(json!({
            "skills": {
                "tone": {"source": "./skills/tone"},
                "team": {"source": "./skills/team"}
            }
        })));
        reading.parts = vec![
            local(PartField::Skills, "tone"),
            local(PartField::Skills, "team"),
        ];
        // The `skill` entry names one skill folder. The `folder` entry holds
        // skills the loader does not look for, so its own name is not taken.
        let config = base(json!({"skills": {"directories": [
            {"skill": "/caller/skills/tone"},
            {"folder": "/caller/team"}
        ]}}));

        let mapped = map_with(
            &reading,
            config,
            &with_skill_entries(&[SkillDirectoryKind::Skill, SkillDirectoryKind::Folder]),
            &AgentPackageOptions::default(),
        );

        assert_eq!(
            mapped.config.skills.as_ref().expect("skills").directories,
            Some(vec![
                SkillDirectory::Skill(PathBuf::from("/caller/skills/tone")),
                SkillDirectory::Folder(PathBuf::from("/caller/team")),
                SkillDirectory::Skill(PathBuf::from("/pkg/skills/team")),
            ])
        );
        let tone = item(&mapped, "agent:/skills/tone");
        assert_eq!(tone.status, DeclaredStatus::NotMapped);
        assert_eq!(tone.gap, Some(MappingGap::KeyClash));
        let team = item(&mapped, "agent:/skills/team");
        assert_eq!(team.status, DeclaredStatus::Mapped);
        assert_eq!(team.mapped_to, vec!["skills.directories.2.skill"]);
    }

    #[test]
    fn skills_that_cannot_be_used_name_their_gap() {
        // (entry, lock entry, left out, support, expected gap, required)
        let git = "git::https://git.example.com/org/skills.git//notes";
        let oci = "oci::registry.example.com/org/skills/notes";
        let table = [
            (
                json!({"source": "./skills/notes"}),
                Some(local(PartField::Skills, "notes")),
                false,
                without(AdapterConfigField::Skills),
                MappingGap::AdapterDoesNotAccept,
                true,
            ),
            (
                json!({"source": git, "version": "1.2.0"}),
                Some(part(
                    PartField::Skills,
                    "notes",
                    git,
                    SourceType::Git,
                    Integrity::Absent,
                )),
                false,
                everything(),
                MappingGap::ContentAbsent,
                true,
            ),
            (
                json!({"source": oci, "version": "1.2.0", "optional": true}),
                Some(part(
                    PartField::Skills,
                    "notes",
                    oci,
                    SourceType::Oci,
                    Integrity::NotVerifiable,
                )),
                false,
                everything(),
                MappingGap::ContentNotVerifiable,
                false,
            ),
            (
                json!({"source": oci, "version": "1.2.0", "optional": true}),
                None,
                true,
                everything(),
                MappingGap::NotResolved,
                false,
            ),
            (
                json!({"source": "npm::@org/notes", "version": "1.2.0"}),
                Some(part(
                    PartField::Skills,
                    "notes",
                    "npm::@org/notes",
                    SourceType::Registry,
                    Integrity::NotVerifiable,
                )),
                false,
                everything(),
                MappingGap::SourceTypeUnsupported,
                true,
            ),
        ];
        for (entry, locked, is_left_out, support, gap, required) in table {
            let mut reading = reading(agent(json!({"skills": {"notes": entry}})));
            reading.parts.extend(locked);
            if is_left_out {
                reading
                    .left_out
                    .push(left_out(PartField::Skills, "notes", oci));
            }

            let mut mapped = map(&reading, &support);

            assert!(mapped.config.skills.is_none(), "{gap:?}");
            let skill = item(&mapped, "agent:/skills/notes");
            assert_eq!(skill.status, DeclaredStatus::NotMapped);
            assert_eq!(skill.gap, Some(gap));
            assert_eq!(skill.required, required);
            let (outcome, _) = stopped_by(&mut mapped);
            assert_eq!(
                outcome == PackageOutcome::Stopped,
                required,
                "{gap:?} required={required}"
            );
            assert_eq!(
                item(&mapped, "agent:/skills/notes").effect,
                if required {
                    ItemEffect::Stopped
                } else {
                    ItemEffect::None
                }
            );
        }
    }

    // ---- parts NeMo Fabric has no field for ----

    #[test]
    fn plugins_hooks_agents_and_policies_are_named_on_an_adapter_that_declares_no_key() {
        let mut reading = reading(agent(json!({
            "plugins": {"charts": {"source": "./plugins/charts"}},
            "hooks": {"tool-guard": {"source": "./hooks/tool-guard", "optional": true}},
            "agents": {
                "coder": {"source": "./agents/coder", "variables": {"depth": 2}},
                "translator": {
                    "source": "a2a::https://agents.example.com/translator",
                    "version": "2.1.0",
                    "optional": true
                }
            },
            "policies": {"licensed": {"source": "./policies/licensed"}}
        })));
        reading.parts = vec![
            local(PartField::Plugins, "charts"),
            local(PartField::Hooks, "tool-guard"),
            local(PartField::Agents, "coder"),
            local(PartField::Policies, "licensed"),
        ];
        reading.left_out.push(left_out(
            PartField::Agents,
            "translator",
            "a2a::https://agents.example.com/translator",
        ));

        let mut mapped = map(&reading, &everything());

        // (id, gap, required). The reader did not start the called agent.
        let table = [
            (
                "agent:/plugins/charts",
                MappingGap::AdapterDoesNotAccept,
                true,
            ),
            (
                "agent:/hooks/tool-guard",
                MappingGap::AdapterDoesNotAccept,
                false,
            ),
            ("agent:/agents/coder", MappingGap::NoFabricField, true),
            ("agent:/agents/translator", MappingGap::NotResolved, false),
            ("agent:/policies/licensed", MappingGap::NoFabricField, true),
        ];
        for (id, gap, required) in table {
            let declared = item(&mapped, id);
            assert_eq!(declared.status, DeclaredStatus::NotMapped, "{id}");
            assert_eq!(declared.gap, Some(gap), "{id}");
            assert_eq!(declared.required, required, "{id}");
            assert!(declared.mapped_to.is_empty());
        }
        assert_eq!(
            item(&mapped, "agent:/agents/coder").address.as_deref(),
            Some("$/agents/coder")
        );
        assert_eq!(item(&mapped, "agent:/agents/translator").address, None);
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(
            stopped,
            vec![
                "agent:/plugins/charts",
                "agent:/agents/coder",
                "agent:/policies/licensed"
            ]
        );
        assert_eq!(
            item(&mapped, "agent:/agents/translator").effect,
            ItemEffect::None
        );
    }

    // ---- plugins, hooks and called agents, to an adapter that declares the key ----

    const HOOK_FORMAT: &str = "https://agent-plugins.org/schemas/1.1.0/hooks.schema.json";

    /// The adapter of `everything()` with a descriptor schema that names these
    /// root extension keys and is closed, as an adapter writes it.
    fn declaring(properties: Value) -> AdapterSupport {
        AdapterSupport {
            extension_schema: json!({
                "type": "object",
                "properties": properties,
                "additionalProperties": false
            })
            .as_object()
            .map(ExtensionSchema::new),
            ..everything()
        }
    }

    fn paths_schema() -> Value {
        json!({
            "type": "object",
            "properties": {"paths": {"type": "array", "items": {"type": "string", "minLength": 1}}},
            "required": ["paths"],
            "additionalProperties": false
        })
    }

    fn plugins_schema() -> Value {
        json!({PLUGINS_KEY: paths_schema()})
    }

    /// The schema of an adapter that runs hooks in these formats.
    fn hooks_schema(formats: &[&str]) -> Value {
        json!({HOOKS_KEY: {
            "type": "object",
            "properties": {"entries": {"type": "array", "items": {
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1}, "format": {"enum": formats}},
                "required": ["path", "format"],
                "additionalProperties": false
            }}},
            "required": ["entries"],
            "additionalProperties": false
        }})
    }

    /// The schema of an adapter whose entry for a called agent takes the
    /// description, the instructions and these optional members.
    fn agents_schema(optional: &[&str]) -> Value {
        let mut members = json!({
            "description": {"type": "string", "minLength": 1},
            "instructions": {"type": "string", "minLength": 1}
        });
        for member in optional {
            members[*member] = match *member {
                "max_turns" => json!({"type": "integer", "minimum": 1}),
                _ => paths_schema(),
            };
        }
        json!({AGENTS_KEY: {"type": "object", "additionalProperties": {
            "type": "object",
            "properties": members,
            "required": ["description", "instructions"],
            "additionalProperties": false
        }}})
    }

    /// A video producer with a plugin, a hook and two called agents, written
    /// to disk and read back. The folders, the listings and the called agents
    /// are the reader's own.
    fn video_producer(label: &str, hooks_json: &str) -> (testing::Package, PackageReading) {
        let package = testing::Package::new(label);
        package.write_json(
            "agent.json",
            &testing::agent(
                "video-producer",
                json!({
                    "limits": {"max_turns": 60},
                    "plugins": {"remotion": {"source": "./plugins/remotion"}},
                    "hooks": {"tool-guard": {"source": "./hooks/tool-guard"}},
                    "agents": {
                        "reviewer": {"source": "./agents/reviewer"},
                        "captioner": {"source": "./agents/captioner", "optional": true}
                    }
                }),
            ),
        );
        package.write("plugins/remotion/plugin.json", r#"{"name": "remotion"}"#);
        package.write("hooks/tool-guard/hooks.json", hooks_json);
        package.write("hooks/tool-guard/check-tool-call", "#!/bin/sh\n");
        package.write_json(
            "agents/reviewer/agent.json",
            &testing::agent(
                "reviewer",
                json!({
                    "description": "Reviews a rendered video against the checklist.",
                    "instructions": "Check each item of the checklist and report what fails.",
                    "limits": {"max_turns": 80}
                }),
            ),
        );
        package.write_json(
            "agents/captioner/agent.json",
            &testing::agent(
                "captioner",
                json!({
                    "description": "Writes captions for a finished video.",
                    "instructions": "Write captions from the transcript you are given.",
                    "model": {"source": "ollama::llama3.2:3b"},
                    "limits": {"max_turns": 10, "max_total_tokens": 100000},
                    "skills": {"captions": {"source": "./skills/captions"}}
                }),
            ),
        );
        package.write(
            "agents/captioner/skills/captions/SKILL.md",
            "---\nname: captions\n---\nKeep each caption to two lines.\n",
        );
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &[
                        "$/agents/captioner",
                        "$/agents/reviewer",
                        "$/hooks/tool-guard",
                        "$/plugins/remotion",
                    ],
                ),
                package.file_entry(
                    "$/agents/captioner",
                    "agent",
                    "agents/captioner/agent.json",
                    "./agents/captioner/agent.json",
                    &["$/agents/captioner/skills/captions"],
                ),
                package.folder_entry(
                    "$/agents/captioner/skills/captions",
                    "skill",
                    "agents/captioner/skills/captions",
                    "./agents/captioner/skills/captions",
                ),
                package.file_entry(
                    "$/agents/reviewer",
                    "agent",
                    "agents/reviewer/agent.json",
                    "./agents/reviewer/agent.json",
                    &[],
                ),
                package.folder_entry(
                    "$/hooks/tool-guard",
                    "hook",
                    "hooks/tool-guard",
                    "./hooks/tool-guard",
                ),
                package.folder_entry(
                    "$/plugins/remotion",
                    "plugin",
                    "plugins/remotion",
                    "./plugins/remotion",
                ),
            ],
        );
        let reading = package.read().expect("the package reads");
        (package, reading)
    }

    fn hooks_json(format: &str) -> String {
        json!({"$schema": format, "hooks": {"PreToolUse": [{"command": "./check-tool-call"}]}})
            .to_string()
    }

    /// The verified folder of a part, as the text the config holds.
    fn folder(reading: &PackageReading, address: &str) -> String {
        reading
            .parts
            .iter()
            .find(|part| part.address == address)
            .and_then(|part| part.path.as_ref())
            .unwrap_or_else(|| panic!("no folder for {address}"))
            .to_string_lossy()
            .into_owned()
    }

    fn video_producer_config(members: Value) -> FabricConfig {
        let mut config = base(members);
        config.metadata.name = "video-producer".to_string();
        config
    }

    #[test]
    fn a_plugin_folder_goes_to_an_adapter_that_declares_the_key() {
        let mut reading = reading(agent(json!({
            "plugins": {
                "charts": {"source": "./plugins/charts"},
                "slides": {"source": "./plugins/slides", "optional": true}
            }
        })));
        reading.parts = vec![
            local(PartField::Plugins, "charts"),
            local(PartField::Plugins, "slides"),
        ];
        // The caller's own plugin folder stays first.
        let config = base(json!({PLUGINS_KEY: {"paths": ["/caller/plugins/brand"]}}));

        let mut mapped = map_with(
            &reading,
            config,
            &declaring(plugins_schema()),
            &AgentPackageOptions::default(),
        );

        assert_eq!(
            mapped.config.extensions,
            BTreeMap::from([(
                PLUGINS_KEY.to_string(),
                json!({"paths": ["/caller/plugins/brand", "/pkg/plugins/charts", "/pkg/plugins/slides"]})
            )])
        );
        for (id, path) in [
            ("agent:/plugins/charts", "nemo_fabric.plugins.v1.paths.1"),
            ("agent:/plugins/slides", "nemo_fabric.plugins.v1.paths.2"),
        ] {
            let plugin = item(&mapped, id);
            assert_eq!(plugin.status, DeclaredStatus::Mapped, "{id}");
            assert_eq!(plugin.mapped_to, vec![path], "{id}");
        }
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Ready);
        assert!(stopped.is_empty());
    }

    #[test]
    fn a_plugin_the_loader_cannot_hand_over_names_its_gap() {
        let entry = json!({"plugins": {"charts": {"source": "./plugins/charts"}}});
        let mut one_file = local(PartField::Plugins, "charts");
        one_file.digest_kind = Some(DigestKind::File);
        let fetched = part(
            PartField::Plugins,
            "charts",
            "git::https://git.example.com/plugins.git//charts",
            SourceType::Git,
            Integrity::Absent,
        );
        // (the lock entry, the config's own value under the key, the gap)
        let table = [
            (one_file, json!({}), MappingGap::UnsupportedShape),
            (fetched, json!({}), MappingGap::ContentAbsent),
            (
                local(PartField::Plugins, "charts"),
                json!({PLUGINS_KEY: {"paths": "/caller/plugins/brand"}}),
                MappingGap::UnsupportedShape,
            ),
        ];
        for (locked, held, gap) in table {
            let mut reading = reading(agent(entry.clone()));
            if locked.source_type == SourceType::Git {
                reading.agent_as_written["plugins"]["charts"] = json!({
                    "source": "git::https://git.example.com/plugins.git//charts",
                    "version": "1.0.0"
                });
            }
            reading.parts = vec![locked];
            let config = base(held);

            let mut mapped = map_with(
                &reading,
                config.clone(),
                &declaring(plugins_schema()),
                &AgentPackageOptions::default(),
            );

            let plugin = item(&mapped, "agent:/plugins/charts");
            assert_eq!(plugin.status, DeclaredStatus::NotMapped, "{gap:?}");
            assert_eq!(plugin.gap, Some(gap));
            // Nothing was written: the config's root keys are the caller's.
            assert_eq!(mapped.config.extensions, config.extensions, "{gap:?}");
            let (_, stopped) = stopped_by(&mut mapped);
            assert_eq!(stopped, vec!["agent:/plugins/charts"]);
        }
    }

    #[test]
    fn a_hook_folder_goes_to_the_adapter_with_the_format_its_hooks_json_names() {
        let (_package, reading) = video_producer("hook", &hooks_json(HOOK_FORMAT));
        // The caller's own hook stays first.
        let own = json!({"path": "/caller/hooks/audit", "format": HOOK_FORMAT});
        let config = video_producer_config(json!({HOOKS_KEY: {"entries": [own]}}));

        let mapped = map_with(
            &reading,
            config,
            &declaring(hooks_schema(&[HOOK_FORMAT])),
            &AgentPackageOptions::default(),
        );

        assert_eq!(
            mapped.config.extensions[HOOKS_KEY],
            json!({"entries": [
                own,
                {"path": folder(&reading, "$/hooks/tool-guard"), "format": HOOK_FORMAT}
            ]})
        );
        let hook = item(&mapped, "agent:/hooks/tool-guard");
        assert_eq!(hook.status, DeclaredStatus::Mapped);
        assert_eq!(hook.mapped_to, vec!["nemo_fabric.hooks.v1.entries.1"]);
        assert!(
            hook.detail
                .as_deref()
                .is_some_and(|detail| detail.contains(HOOK_FORMAT))
        );
    }

    #[test]
    fn a_hook_in_a_format_the_adapter_does_not_run_is_not_mapped() {
        let other = "https://hooks.example.com/schemas/2.0/hooks.schema.json";
        // (the text of hooks.json, the gap, a piece of the reason)
        let table = [
            (
                hooks_json(other),
                MappingGap::AdapterDoesNotAccept,
                "/nemo_fabric.hooks.v1/entries/0/format",
            ),
            (
                r#"{"hooks": {}}"#.to_string(),
                MappingGap::UnsupportedShape,
                "`$schema`",
            ),
            (
                "not JSON".to_string(),
                MappingGap::UnsupportedShape,
                "`$schema`",
            ),
        ];
        for (text, gap, reason) in table {
            let (_package, reading) = video_producer("hook-format", &text);

            let mut mapped = map_with(
                &reading,
                video_producer_config(json!({})),
                &declaring(hooks_schema(&[HOOK_FORMAT])),
                &AgentPackageOptions::default(),
            );

            let hook = item(&mapped, "agent:/hooks/tool-guard");
            assert_eq!(hook.status, DeclaredStatus::NotMapped, "{text}");
            assert_eq!(hook.gap, Some(gap), "{text}");
            let detail = hook.detail.as_deref().expect("a detail");
            assert!(detail.contains(reason), "{detail}");
            assert!(!mapped.config.extensions.contains_key(HOOKS_KEY), "{text}");
            let (_, stopped) = stopped_by(&mut mapped);
            assert!(stopped.contains(&"agent:/hooks/tool-guard".to_string()));
        }
    }

    #[test]
    fn a_hooks_json_that_changed_after_the_check_is_refused() {
        let (_package, mut reading) = video_producer("hook-changed", &hooks_json(HOOK_FORMAT));
        reading
            .listings
            .get_mut("$/hooks/tool-guard")
            .expect("a listing")
            .files
            .insert("hooks.json".to_string(), "sha256:33".to_string());

        let refused = map_package(
            &reading,
            video_producer_config(json!({})),
            &declaring(hooks_schema(&[HOOK_FORMAT])),
            &AgentPackageOptions::default(),
            &[],
            &Facts::default(),
        );

        match refused {
            Err(FabricError::AgentPackage { code, subject, .. }) => {
                assert_eq!(
                    (code.as_str(), subject.as_str()),
                    ("package_digest_mismatch", "$/hooks/tool-guard")
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_called_agent_goes_to_the_adapter_with_what_its_schema_takes() {
        let (_package, reading) = video_producer("called", &hooks_json(HOOK_FORMAT));
        let captions = folder(&reading, "$/agents/captioner/skills/captions");
        let reviewer = json!({
            "description": "Reviews a rendered video against the checklist.",
            "instructions": "Check each item of the checklist and report what fails."
        });
        let captioner = json!({
            "description": "Writes captions for a finished video.",
            "instructions": "Write captions from the transcript you are given."
        });
        let with = |entry: &Value, members: Value| {
            let mut entry = entry.clone();
            for (key, value) in members.as_object().expect("an object") {
                entry[key] = value.clone();
            }
            entry
        };
        // (the optional members the adapter's schema lists, the key it gets,
        // the status of the reviewer's turn limit and of the captioner's skill)
        let table = [
            (
                vec!["max_turns", "skills"],
                json!({
                    "reviewer": with(&reviewer, json!({"max_turns": 60})),
                    "captioner": with(&captioner, json!({"max_turns": 10, "skills": {"paths": [captions]}}))
                }),
                DeclaredStatus::Mapped,
                DeclaredStatus::Mapped,
            ),
            (
                vec!["max_turns"],
                json!({
                    "reviewer": with(&reviewer, json!({"max_turns": 60})),
                    "captioner": with(&captioner, json!({"max_turns": 10}))
                }),
                DeclaredStatus::Mapped,
                DeclaredStatus::NotMapped,
            ),
            (
                vec!["skills"],
                json!({
                    "reviewer": reviewer,
                    "captioner": with(&captioner, json!({"skills": {"paths": [captions]}}))
                }),
                DeclaredStatus::NotMapped,
                DeclaredStatus::Mapped,
            ),
            (
                vec![],
                json!({"reviewer": reviewer, "captioner": captioner}),
                DeclaredStatus::NotMapped,
                DeclaredStatus::NotMapped,
            ),
        ];
        for (optional, expected, turns, skill) in table {
            let mapped = map_with(
                &reading,
                video_producer_config(json!({})),
                &declaring(agents_schema(&optional)),
                &AgentPackageOptions::default(),
            );

            assert_eq!(
                mapped.config.extensions[AGENTS_KEY], expected,
                "{optional:?}"
            );
            for name in ["reviewer", "captioner"] {
                let agent = item(&mapped, &format!("agent:/agents/{name}"));
                assert_eq!(agent.status, DeclaredStatus::Mapped, "{optional:?}");
                assert_eq!(
                    agent.mapped_to,
                    vec![format!("nemo_fabric.agents.v1.{name}")]
                );
                assert_eq!(agent.bound_to, None);
            }
            // A member the entry carries is mapped. One the schema refuses
            // keeps its line, and that line stops a strict load.
            let limit = item(&mapped, "agent:/agents/reviewer/limits/max_turns");
            assert_eq!(limit.status, turns, "{optional:?}");
            assert_eq!(limit.stated_by, StatedBy::CalledAgent);
            let folder = item(&mapped, "agent:/agents/captioner/skills/captions");
            assert_eq!(folder.status, skill, "{optional:?}");
            if turns == DeclaredStatus::Mapped {
                assert_eq!(
                    limit.mapped_to,
                    vec!["nemo_fabric.agents.v1.reviewer.max_turns"]
                );
                assert_eq!(
                    limit.detail.as_deref(),
                    Some("60 applies, lowered from 80: the caller allows 60")
                );
            } else {
                assert_eq!(limit.gap, Some(MappingGap::AdapterDoesNotAccept));
            }
            if skill == DeclaredStatus::Mapped {
                assert_eq!(
                    folder.mapped_to,
                    vec!["nemo_fabric.agents.v1.captioner.skills.paths.0"]
                );
            } else {
                assert_eq!(folder.gap, Some(MappingGap::AdapterDoesNotAccept));
            }
            // What an entry has no place for keeps a line of its own. The
            // captioner's entry is optional, so nothing below it is required.
            let tokens = item(&mapped, "agent:/agents/captioner/limits/max_total_tokens");
            assert_eq!(tokens.status, DeclaredStatus::NotMapped);
            assert_eq!(tokens.gap, Some(MappingGap::NoFabricField));
            assert!(!tokens.required);
            let model = item(&mapped, "agent:/agents/captioner/model/source");
            assert_eq!(model.status, DeclaredStatus::NotMapped);
            assert!(!model.required);
        }
    }

    #[test]
    fn a_called_agent_follows_the_settings_on_an_adapter_that_declares_no_key() {
        let (_package, reading) = video_producer("called-none", &hooks_json(HOOK_FORMAT));

        let mut mapped = map_with(
            &reading,
            video_producer_config(json!({})),
            &declaring(plugins_schema()),
            &AgentPackageOptions::default(),
        );

        assert!(!mapped.config.extensions.contains_key(AGENTS_KEY));
        for name in ["reviewer", "captioner"] {
            let agent = item(&mapped, &format!("agent:/agents/{name}"));
            assert_eq!(agent.status, DeclaredStatus::NotMapped, "{name}");
            assert_eq!(agent.gap, Some(MappingGap::AdapterDoesNotAccept), "{name}");
        }
        // Nothing below an agent nobody runs is listed.
        assert!(
            mapped
                .items
                .iter()
                .all(|item| item.stated_by != StatedBy::CalledAgent)
        );
        // The required agent stops the load. The optional one never does.
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert!(stopped.contains(&"agent:/agents/reviewer".to_string()));
        assert!(!stopped.contains(&"agent:/agents/captioner".to_string()));
    }

    #[test]
    fn the_config_keeps_its_own_called_agents_and_a_shared_name_is_a_conflict() {
        let (_package, reading) = video_producer("called-own", &hooks_json(HOOK_FORMAT));
        let own = json!({"description": "Translates captions.", "instructions": "Translate."});
        let support = declaring(agents_schema(&["max_turns", "skills"]));

        let mapped = map_with(
            &reading,
            video_producer_config(json!({AGENTS_KEY: {"translator": own}})),
            &support,
            &AgentPackageOptions::default(),
        );

        let agents = mapped.config.extensions[AGENTS_KEY]
            .as_object()
            .expect("an object");
        assert_eq!(agents["translator"], own);
        assert_eq!(
            agents.keys().collect::<Vec<_>>(),
            ["captioner", "reviewer", "translator"]
        );

        let refused = map_package(
            &reading,
            video_producer_config(json!({AGENTS_KEY: {"reviewer": own}})),
            &support,
            &AgentPackageOptions::default(),
            &[],
            &Facts::default(),
        );
        match refused {
            Err(FabricError::AgentPackage { code, subject, .. }) => {
                assert_eq!(
                    (code.as_str(), subject.as_str()),
                    ("package_config_conflict", "nemo_fabric.agents.v1.reviewer")
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_bound_called_agent_is_the_callers_and_is_not_handed_to_the_adapter() {
        let (_package, reading) = video_producer("called-bound", &hooks_json(HOOK_FORMAT));
        let options = AgentPackageOptions {
            bindings: BTreeMap::from([(
                "agents.reviewer".to_string(),
                "studio.agents:reviewer".to_string(),
            )]),
            ..AgentPackageOptions::default()
        };

        let mapped = map_with(
            &reading,
            video_producer_config(json!({})),
            &declaring(agents_schema(&["max_turns", "skills"])),
            &options,
        );

        let reviewer = item(&mapped, "agent:/agents/reviewer");
        assert_eq!(reviewer.status, DeclaredStatus::Bound);
        assert_eq!(reviewer.bound_to.as_deref(), Some("studio.agents:reviewer"));
        assert!(reviewer.mapped_to.is_empty());
        // The caller applies what its agent declares: nothing of it is mapped.
        assert_eq!(
            item(&mapped, "agent:/agents/reviewer/limits/max_turns").status,
            DeclaredStatus::NotMapped
        );
        let agents = mapped.config.extensions[AGENTS_KEY]
            .as_object()
            .expect("an object");
        assert_eq!(agents.keys().collect::<Vec<_>>(), ["captioner"]);
    }

    /// A video producer and the agents it calls, written to disk and read
    /// back. Each called agent is its name, what the producer writes on its
    /// entry beside the source, what its document adds, and the lock entries
    /// its document depends on. `skills` holds the lock address and the
    /// folder of each skill.
    fn team(
        label: &str,
        producer: Value,
        called: &[(&str, Value, Value, &[&str])],
        skills: &[(&str, &str)],
    ) -> (testing::Package, PackageReading) {
        let package = testing::Package::new(label);
        let mut entries = Vec::new();
        for (address, folder) in skills {
            package.write(&format!("{folder}/SKILL.md"), testing::SKILL);
            entries.push(package.folder_entry(address, "skill", folder, &format!("./{folder}")));
        }
        let mut agents = Map::new();
        let mut listed = Vec::new();
        for (name, entry, document, dependencies) in called {
            let path = format!("agents/{name}/agent.json");
            let mut members = json!({"instructions": format!("You are the {name}.")});
            for (key, value) in document.as_object().expect("an object") {
                members[key] = value.clone();
            }
            package.write_json(&path, &testing::agent(name, members));
            let mut reference = json!({"source": format!("./agents/{name}")});
            for (key, value) in entry.as_object().expect("an object") {
                reference[key] = value.clone();
            }
            agents.insert(name.to_string(), reference);
            let address = format!("$/agents/{name}");
            entries.push(package.file_entry(
                &address,
                "agent",
                &path,
                &format!("./{path}"),
                dependencies,
            ));
            listed.push(address);
        }
        let mut members = producer;
        members["agents"] = Value::Object(agents);
        package.write_json("agent.json", &testing::agent("video-producer", members));
        let listed: Vec<&str> = listed.iter().map(String::as_str).collect();
        entries.push(package.file_entry("$", "agent", "agent.json", "./agent.json", &listed));
        entries.sort_by_key(|entry| entry["address"].as_str().map(str::to_string));
        package.write_lock("", entries);
        let reading = package.read().expect("the package reads");
        (package, reading)
    }

    const CHECKLIST: &str = "$/agents/reviewer/skills/checklist";

    /// A video producer whose reviewer has a checklist skill of its own.
    fn reviewer_with_checklist(label: &str, skill: Value) -> (testing::Package, PackageReading) {
        team(
            label,
            json!({}),
            &[(
                "reviewer",
                json!({}),
                json!({"skills": {"checklist": skill}}),
                &[CHECKLIST],
            )],
            &[(CHECKLIST, "agents/reviewer/skills/checklist")],
        )
    }

    fn bind_reviewer() -> AgentPackageOptions {
        AgentPackageOptions {
            bindings: BTreeMap::from([(
                "agents.reviewer".to_string(),
                "studio.agents:reviewer".to_string(),
            )]),
            ..AgentPackageOptions::default()
        }
    }

    #[test]
    fn the_values_on_a_called_agents_skill_have_a_line_that_stops_a_load() {
        let (_package, reading) = reviewer_with_checklist(
            "called-values",
            json!({"source": "./skills/checklist", "variables": {"strictness": "high"}}),
        );
        let values = "agent:/agents/reviewer/skills/checklist/variables";
        // The harness runs the reviewer, then the caller does.
        for options in [AgentPackageOptions::default(), bind_reviewer()] {
            let handed = options.bindings.is_empty();

            let mut mapped = map_with(
                &reading,
                video_producer_config(json!({})),
                &declaring(agents_schema(&["skills"])),
                &options,
            );

            let line = item(&mapped, values);
            assert_eq!(line.kind, DeclaredKind::PartValues, "{handed}");
            assert_eq!(line.stated_by, StatedBy::CalledAgent);
            assert_eq!(line.status, DeclaredStatus::NotMapped, "{handed}");
            assert_eq!(line.gap, Some(MappingGap::NoFabricField));
            assert_eq!(line.address.as_deref(), Some(CHECKLIST));
            assert!(line.required);
            // The folder reaches the adapter, and the values do not.
            assert_eq!(
                item(&mapped, "agent:/agents/reviewer/skills/checklist").status,
                if handed {
                    DeclaredStatus::Mapped
                } else {
                    DeclaredStatus::NotMapped
                }
            );
            let (outcome, stopped) = stopped_by(&mut mapped);
            assert_eq!(outcome, PackageOutcome::Stopped);
            assert!(stopped.contains(&values.to_string()), "{stopped:?}");
        }
    }

    #[test]
    fn nothing_below_an_optional_called_agent_stops_a_load() {
        let declares = json!({
            "tools": [{"builtin": "read_file"}],
            "limits": {"max_total_tokens": 100000},
            "skills": {"checklist": {"source": "./skills/checklist", "variables": {"strictness": "high"}}},
            "custom_tools": {"request_approval": {"description": "Ask a person.", "input_schema": {}}}
        });
        // (what the producer writes on the reviewer's entry, whether a line
        // below the reviewer may stop the load)
        for (entry, stops) in [(json!({"optional": true}), false), (json!({}), true)] {
            let (_package, reading) = team(
                "called-optional",
                json!({}),
                &[("reviewer", entry, declares.clone(), &[CHECKLIST])],
                &[(CHECKLIST, "agents/reviewer/skills/checklist")],
            );
            for options in [AgentPackageOptions::default(), bind_reviewer()] {
                let mut mapped = map_with(
                    &reading,
                    video_producer_config(json!({})),
                    &declaring(agents_schema(&["skills"])),
                    &options,
                );

                let below: Vec<&DeclaredItem> = mapped
                    .items
                    .iter()
                    .filter(|item| item.stated_by == StatedBy::CalledAgent)
                    .collect();
                assert!(below.len() >= 5, "{}", below.len());
                assert!(
                    below.iter().all(|item| item.required == stops),
                    "{stops}: {:?}",
                    ids(&mapped.items)
                );
                let (outcome, stopped) = stopped_by(&mut mapped);
                assert_eq!(outcome == PackageOutcome::Stopped, stops, "{stopped:?}");
            }
        }
    }

    #[test]
    fn a_called_agent_runs_under_the_turn_limit_its_caller_runs_with() {
        let id = "agent:/agents/reviewer/limits/max_turns";
        let path = "nemo_fabric.agents.v1.reviewer.max_turns";
        // (the producer's limits, the config's runtime, the reviewer's limits,
        // the limit in the reviewer's entry, the detail of its line)
        let table = [
            (
                json!({"max_turns": 60}),
                json!({}),
                json!({"max_turns": 20}),
                Some(20),
                Some("20 applies"),
            ),
            (
                json!({"max_turns": 60}),
                json!({}),
                json!({"max_turns": 80}),
                Some(60),
                Some("60 applies, lowered from 80: the caller allows 60"),
            ),
            // The config lets the caller run fewer turns than its document asks for.
            (
                json!({"max_turns": 60}),
                json!({"max_turns": 30}),
                json!({"max_turns": 80}),
                Some(30),
                Some("30 applies, lowered from 80: the config's runtime.max_turns allows 30"),
            ),
            // An agent that states no limit runs under its caller's, and says so.
            (
                json!({"max_turns": 60}),
                json!({}),
                json!({"max_total_tokens": 100000}),
                Some(60),
                Some("60 applies: the agent states no turn limit, and the caller allows 60"),
            ),
            (
                json!({"max_total_tokens": 800000}),
                json!({"max_turns": 30}),
                json!({"max_total_tokens": 100000}),
                Some(30),
                Some(
                    "30 applies: the agent states no turn limit, and the config's runtime.max_turns allows 30",
                ),
            ),
            // Nothing limits the turns of either agent.
            (
                json!({"max_total_tokens": 800000}),
                json!({}),
                json!({"max_total_tokens": 100000}),
                None,
                None,
            ),
        ];
        for (producer, runtime, reviewer, turns, detail) in table {
            let (_package, reading) = team(
                "called-turns",
                json!({"limits": producer}),
                &[("reviewer", json!({}), json!({"limits": reviewer}), &[])],
                &[],
            );

            let mapped = map_with(
                &reading,
                video_producer_config(json!({"runtime": runtime})),
                &declaring(agents_schema(&["max_turns"])),
                &AgentPackageOptions::default(),
            );

            let entry = &mapped.config.extensions[AGENTS_KEY]["reviewer"];
            assert_eq!(entry.get("max_turns"), turns.map(Value::from).as_ref());
            let line = mapped.items.iter().find(|item| item.id == id);
            assert_eq!(line.is_some(), turns.is_some(), "{detail:?}");
            if let Some(line) = line {
                assert_eq!(line.status, DeclaredStatus::Mapped);
                assert_eq!(line.stated_by, StatedBy::CalledAgent);
                assert_eq!(line.mapped_to, vec![path]);
                assert_eq!(line.detail.as_deref(), detail);
            }
        }
    }

    #[test]
    fn a_turn_limit_the_adapter_does_not_take_stops_a_load_stated_or_not() {
        let id = "agent:/agents/reviewer/limits/max_turns".to_string();
        // The reviewer states a limit, then takes its caller's.
        for reviewer in [json!({"limits": {"max_turns": 80}}), json!({})] {
            let (_package, reading) = team(
                "called-turns-refused",
                json!({"limits": {"max_turns": 60}}),
                &[("reviewer", json!({}), reviewer.clone(), &[])],
                &[],
            );

            let mut mapped = map_with(
                &reading,
                video_producer_config(json!({})),
                &declaring(agents_schema(&[])),
                &AgentPackageOptions::default(),
            );

            let entry = &mapped.config.extensions[AGENTS_KEY]["reviewer"];
            assert_eq!(entry.get("max_turns"), None, "{reviewer}");
            let line = item(&mapped, &id);
            assert_eq!(line.status, DeclaredStatus::NotMapped, "{reviewer}");
            assert_eq!(line.gap, Some(MappingGap::AdapterDoesNotAccept));
            assert!(line.required);
            let (_, stopped) = stopped_by(&mut mapped);
            assert_eq!(stopped, vec![id.clone()], "{reviewer}");
        }
    }

    #[test]
    fn two_called_agents_that_share_a_skill_folder_both_get_it() {
        // One folder that two documents reach has one lock entry, at the
        // address of the first path, and both documents depend on it.
        let shared = "$/agents/captioner/skills/house-style";
        let member = json!({"skills": {"house-style": {"source": "./../../skills/house-style"}}});
        let (_package, reading) = team(
            "called-shared",
            json!({}),
            &[
                ("captioner", json!({}), member.clone(), &[shared]),
                ("reviewer", json!({}), member, &[shared]),
            ],
            &[(shared, "skills/house-style")],
        );

        let mut mapped = map_with(
            &reading,
            video_producer_config(json!({})),
            &declaring(agents_schema(&["skills"])),
            &AgentPackageOptions::default(),
        );

        let house_style = folder(&reading, shared);
        for name in ["captioner", "reviewer"] {
            let skill = item(&mapped, &format!("agent:/agents/{name}/skills/house-style"));
            assert_eq!(skill.status, DeclaredStatus::Mapped, "{name}");
            assert_eq!(skill.address.as_deref(), Some(shared), "{name}");
            assert_eq!(
                mapped.config.extensions[AGENTS_KEY][name]["skills"],
                json!({"paths": [house_style]}),
                "{name}"
            );
        }
        let (outcome, _) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Ready);
    }

    #[test]
    fn a_called_agents_skill_that_cannot_be_used_names_its_gap() {
        // (what the lock entry of the skill pins and what the check found,
        // the gap of the skill's line)
        let table = [
            (
                DigestKind::File,
                Integrity::Verified,
                MappingGap::UnsupportedShape,
            ),
            (
                DigestKind::Folder,
                Integrity::Absent,
                MappingGap::ContentAbsent,
            ),
            (
                DigestKind::Artifact,
                Integrity::NotVerifiable,
                MappingGap::ContentNotVerifiable,
            ),
        ];
        for (digest_kind, integrity, gap) in table {
            let (_package, mut reading) =
                reviewer_with_checklist("called-gap", json!({"source": "./skills/checklist"}));
            let skill = reading
                .parts
                .iter_mut()
                .find(|part| part.address == CHECKLIST)
                .expect("the skill");
            skill.digest_kind = Some(digest_kind);
            skill.integrity = integrity;

            let mut mapped = map_with(
                &reading,
                video_producer_config(json!({})),
                &declaring(agents_schema(&["skills"])),
                &AgentPackageOptions::default(),
            );

            // The agent reaches the adapter without the skill, and the
            // skill's own line says why.
            let reviewer = &mapped.config.extensions[AGENTS_KEY]["reviewer"];
            assert_eq!(reviewer.get("skills"), None, "{gap:?}");
            let line = item(&mapped, "agent:/agents/reviewer/skills/checklist");
            assert_eq!(line.status, DeclaredStatus::NotMapped, "{gap:?}");
            assert_eq!(line.gap, Some(gap));
            let (_, stopped) = stopped_by(&mut mapped);
            assert_eq!(stopped, vec!["agent:/agents/reviewer/skills/checklist"]);
        }
    }

    #[test]
    fn a_config_value_under_the_agents_key_in_another_shape_is_kept() {
        let (_package, reading) = team(
            "called-shape",
            json!({}),
            &[("reviewer", json!({}), json!({}), &[])],
            &[],
        );
        // An adapter that names the key and takes any value under it.
        let support = declaring(json!({AGENTS_KEY: {}}));
        for held in [json!(["studio-translator"]), json!("studio-translator")] {
            let config = video_producer_config(json!({AGENTS_KEY: held}));

            let mut mapped = map_with(
                &reading,
                config.clone(),
                &support,
                &AgentPackageOptions::default(),
            );

            assert_eq!(mapped.config.extensions, config.extensions, "{held}");
            let reviewer = item(&mapped, "agent:/agents/reviewer");
            assert_eq!(reviewer.status, DeclaredStatus::NotMapped, "{held}");
            assert_eq!(reviewer.gap, Some(MappingGap::UnsupportedShape));
            let (_, stopped) = stopped_by(&mut mapped);
            assert_eq!(stopped, vec!["agent:/agents/reviewer"]);
        }
    }

    #[test]
    fn a_binding_is_applied_only_to_an_agent_whose_document_is_here() {
        let source = "git::https://git.example.com/agents.git//coder";
        let bound = AgentPackageOptions {
            bindings: BTreeMap::from([(
                "agents.coder".to_string(),
                "app.agents:coder".to_string(),
            )]),
            ..AgentPackageOptions::default()
        };
        // A caller that gives bindings, and none for this agent.
        let others = AgentPackageOptions {
            bindings: BTreeMap::from([(
                "custom_tools.ask".to_string(),
                "app.tools:ask".to_string(),
            )]),
            ..AgentPackageOptions::default()
        };
        // (what the lock check found, status, gap, whether the target is kept)
        let table = [
            (
                Integrity::Absent,
                DeclaredStatus::NotMapped,
                Some(MappingGap::ContentAbsent),
                false,
            ),
            (
                Integrity::NotVerifiable,
                DeclaredStatus::NotMapped,
                Some(MappingGap::ContentNotVerifiable),
                false,
            ),
            // A running agent: the caller's target is kept and nothing compared the pin.
            (
                Integrity::Pointer,
                DeclaredStatus::CouldNotBeChecked,
                None,
                true,
            ),
            // The document is here and the reader could not start the agent.
            (
                Integrity::Verified,
                DeclaredStatus::NotMapped,
                Some(MappingGap::NoFabricField),
                false,
            ),
        ];
        for (integrity, status, gap, kept) in table {
            let mut reading = reading(agent(json!({
                "agents": {"coder": {"source": source, "version": "1.0.0"}}
            })));
            reading.parts = vec![part(
                PartField::Agents,
                "coder",
                source,
                SourceType::Git,
                integrity,
            )];

            let mapped = map_with(&reading, base(json!({})), &everything(), &bound);

            let coder = item(&mapped, "agent:/agents/coder");
            assert_eq!(coder.status, status, "{integrity:?}");
            assert_eq!(coder.gap, gap, "{integrity:?}");
            assert_eq!(coder.bound_to.is_some(), kept, "{integrity:?}");
            if status == DeclaredStatus::NotMapped {
                let detail = coder.detail.as_deref().expect("a detail");
                assert!(
                    detail.starts_with("the binding was not applied"),
                    "{detail}"
                );
            }

            // With no binding of its own the line has the same gap, and it
            // names the key to add only where a binding would be applied:
            // the caller's target is kept.
            let mapped = map_with(&reading, base(json!({})), &everything(), &others);

            let coder = item(&mapped, "agent:/agents/coder");
            let detail = coder.detail.as_deref().expect("a detail");
            assert_eq!(
                coder.gap,
                Some(gap.unwrap_or(MappingGap::NoFabricField)),
                "{integrity:?}"
            );
            assert_eq!(
                detail.ends_with("gives a binding for `agents.coder`"),
                kept,
                "{integrity:?}: {detail}"
            );
        }

        // An optional agent the lock leaves out stays left out.
        let mut reading = reading(agent(json!({
            "agents": {"coder": {"source": source, "version": "1.0.0", "optional": true}}
        })));
        reading
            .left_out
            .push(left_out(PartField::Agents, "coder", source));

        let mapped = map_with(&reading, base(json!({})), &everything(), &bound);

        let coder = item(&mapped, "agent:/agents/coder");
        assert_eq!(coder.status, DeclaredStatus::NotMapped);
        assert_eq!(coder.gap, Some(MappingGap::NotResolved));
        assert_eq!(coder.bound_to, None);
        assert!(!coder.required);
    }

    #[test]
    fn packages_credentials_custom_tools_outputs_and_extensions_are_reported() {
        let mut reading = reading(agent(json!({
            "packages": {
                "whisper": {"source": "pypi::faster-whisper", "version": "1.0.3"},
                "extra": {"source": "pypi::extra", "version": "1.0.0", "optional": true}
            },
            "credentials": {"asset-library": {"description": "Read assets."}},
            "custom_tools": {
                "request_approval": {"description": "Ask a person.", "input_schema": {}}
            },
            "outputs": {"summary": {"type": "string"}},
            "extensions": {"claude-code": {"rules": []}}
        })));
        reading.parts = vec![pinned_package("whisper", "pypi", "faster-whisper", "1.0.3")];
        reading
            .left_out
            .push(left_out(PartField::Packages, "extra", "pypi::extra"));

        let mut mapped = map(&reading, &everything());

        // NeMo Fabric installs no package and looks for none. With no
        // `packages` fact a required one is not accounted for.
        let package = item(&mapped, "agent:/packages/whisper");
        assert_eq!(package.status, DeclaredStatus::NotMapped);
        assert_eq!(package.gap, Some(MappingGap::NoFabricField));
        assert!(package.required);
        let extra = item(&mapped, "agent:/packages/extra");
        assert_eq!(extra.status, DeclaredStatus::NotMapped);
        assert_eq!(extra.gap, Some(MappingGap::NotResolved));
        let credential = item(&mapped, "agent:/credentials/asset-library");
        assert_eq!(credential.status, DeclaredStatus::CouldNotBeChecked);
        assert_eq!(credential.reason.as_deref(), Some("Read assets."));
        let tool = item(&mapped, "agent:/custom_tools/request_approval");
        assert_eq!(tool.status, DeclaredStatus::NotMapped);
        assert_eq!(tool.gap, Some(MappingGap::NotBound));
        for id in ["agent:/outputs/summary", "agent:/extensions/claude-code"] {
            assert_eq!(item(&mapped, id).status, DeclaredStatus::Skipped);
            assert!(!item(&mapped, id).required);
        }
        // Default settings: the custom tool and the required package stop, what
        // could not be checked warns, and the optional package does nothing.
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(
            stopped,
            vec![
                "agent:/custom_tools/request_approval",
                "agent:/packages/whisper"
            ]
        );
        assert_eq!(
            item(&mapped, "agent:/packages/extra").effect,
            ItemEffect::None
        );
        assert_eq!(
            item(&mapped, "agent:/credentials/asset-library").effect,
            ItemEffect::Warned
        );
    }

    // ---- software packages ----

    /// A lock entry for a software package pinned from a registry.
    fn pinned_package(key: &str, registry: &str, name: &str, version: &str) -> LockedPart {
        let mut locked = part(
            PartField::Packages,
            key,
            &format!("{registry}::{name}"),
            SourceType::Registry,
            Integrity::NotVerifiable,
        );
        locked.registry = Some(registry.to_string());
        locked.source_url = Some(name.to_string());
        locked.version = version.to_string();
        locked
    }

    /// A video producer that needs one package from PyPI and one from npm.
    fn with_packages() -> PackageReading {
        let mut reading = reading(agent(json!({
            "packages": {
                "whisper": {"source": "pypi::faster-whisper", "version": "1.2.1"},
                "remotion": {"source": "npm::@remotion/cli", "version": "4.0.532", "optional": true}
            }
        })));
        reading.parts = vec![
            pinned_package("whisper", "pypi", "faster-whisper", "1.2.1"),
            pinned_package("remotion", "npm", "@remotion/cli", "4.0.532"),
        ];
        reading
    }

    fn facts(value: Value) -> Facts {
        serde_json::from_value(value).expect("facts")
    }

    #[test]
    fn a_pinned_package_is_compared_with_the_packages_fact() {
        use DeclaredStatus::{CouldNotBeChecked, MetHere, NotMapped, NotMet};
        // (the fact, the status of the PyPI package, the status of the npm package)
        let table = [
            // Nothing looked for a package of a registry that is not given.
            (json!({}), NotMapped, NotMapped),
            (
                json!({"packages": {"npm": {"@remotion/cli": "4.0.532"}}}),
                NotMapped,
                MetHere,
            ),
            // A registry is named as the lock writes it.
            (
                json!({"packages": {"PyPI": {"faster-whisper": "1.2.1"}}}),
                NotMapped,
                NotMapped,
            ),
            // A package left out of a registry that is given is not installed.
            (json!({"packages": {"pypi": {}, "npm": {}}}), NotMet, NotMet),
            // Null: installed, and the version is not known.
            (
                json!({"packages": {"pypi": {"faster-whisper": null}}}),
                CouldNotBeChecked,
                NotMapped,
            ),
            (
                json!({"packages": {"pypi": {"faster-whisper": "1.2.1"}}}),
                MetHere,
                NotMapped,
            ),
            // The installed version must be the one the lock pins.
            (
                json!({"packages": {
                    "pypi": {"faster-whisper": "1.2.0"},
                    "npm": {"@remotion/cli": "4.0.531"}
                }}),
                NotMet,
                NotMet,
            ),
            // A PyPI name compares after normalising. An npm name does not.
            (
                json!({"packages": {
                    "pypi": {"Faster_Whisper": "1.2.1"},
                    "npm": {"@Remotion/CLI": "4.0.532"}
                }}),
                MetHere,
                NotMet,
            ),
            (
                json!({"packages": {"pypi": {"faster.whisper": "1.2.1"}}}),
                MetHere,
                NotMapped,
            ),
            // Two spellings of one PyPI name: one version is that version,
            // and two versions say nothing about which is installed.
            (
                json!({"packages": {"pypi": {"faster-whisper": "1.2.1", "Faster.Whisper": "1.2.1"}}}),
                MetHere,
                NotMapped,
            ),
            (
                json!({"packages": {"pypi": {"faster-whisper": "9.9.9", "Faster.Whisper": "1.2.1"}}}),
                CouldNotBeChecked,
                NotMapped,
            ),
        ];
        for (fact, whisper, remotion) in table {
            let mut mapped = map_with_facts(
                &with_packages(),
                base(json!({})),
                &everything(),
                &AgentPackageOptions::default(),
                &facts(fact.clone()),
            );

            let pypi = item(&mapped, "agent:/packages/whisper");
            assert_eq!(pypi.status, whisper, "{fact}");
            assert_eq!(
                pypi.gap,
                (whisper == NotMapped).then_some(MappingGap::NoFabricField),
                "{fact}"
            );
            assert!(pypi.mapped_to.is_empty());
            assert_eq!(pypi.address.as_deref(), Some("$/packages/whisper"));
            assert_eq!(
                item(&mapped, "agent:/packages/remotion").status,
                remotion,
                "{fact}"
            );
            // A required package that is not installed, or that nothing
            // looked for, stops the load. One whose version could not be
            // checked warns, and an optional one does neither.
            let (_, stopped) = stopped_by(&mut mapped);
            assert_eq!(
                stopped,
                if matches!(whisper, NotMet | NotMapped) {
                    vec!["agent:/packages/whisper"]
                } else {
                    Vec::<&str>::new()
                },
                "{fact}"
            );
            assert_eq!(
                item(&mapped, "agent:/packages/whisper").effect,
                match whisper {
                    NotMet | NotMapped => ItemEffect::Stopped,
                    CouldNotBeChecked => ItemEffect::Warned,
                    _ => ItemEffect::None,
                },
                "{fact}"
            );
            assert_eq!(
                item(&mapped, "agent:/packages/remotion").effect,
                ItemEffect::None
            );
        }
    }

    #[test]
    fn a_package_the_fact_tells_about_keeps_its_status_without_a_config() {
        let options = AgentPackageOptions::default();
        let status = |fact: Value| {
            let items = unmapped_items(&with_packages(), &options, &[], &facts(fact));
            items
                .iter()
                .find(|item| item.id == "agent:/packages/whisper")
                .expect("the package")
                .status
        };

        assert_eq!(
            status(json!({"packages": {"pypi": {"faster-whisper": "1.2.1"}}})),
            DeclaredStatus::MetHere
        );
        assert_eq!(
            status(json!({"packages": {"pypi": {}}})),
            DeclaredStatus::NotMet
        );
        // With no fact about its registry it is listed like everything else.
        assert_eq!(status(json!({})), DeclaredStatus::NotEvaluated);
    }

    #[test]
    fn an_installed_package_says_its_digest_was_not_compared() {
        let mapped = map_with_facts(
            &with_packages(),
            base(json!({})),
            &everything(),
            &AgentPackageOptions::default(),
            &facts(json!({"packages": {"pypi": {"faster-whisper": "1.2.1"}}})),
        );

        assert_eq!(
            item(&mapped, "agent:/packages/whisper").detail.as_deref(),
            Some(
                "faster-whisper 1.2.1 is installed. The lock's digest is of the registry's artifact and was not compared with the installed files"
            )
        );
        // The check writes nothing into the config.
        assert_eq!(mapped.config, base(json!({})));
    }

    #[test]
    fn a_package_with_another_source_than_a_registry_is_not_looked_for() {
        let mut reading = with_packages();
        reading.parts[0].source_type = SourceType::Git;
        reading.parts[0].registry = None;

        let mapped = map_with_facts(
            &reading,
            base(json!({})),
            &everything(),
            &AgentPackageOptions::default(),
            &facts(json!({"packages": {"pypi": {"faster-whisper": "1.2.1"}}})),
        );

        let package = item(&mapped, "agent:/packages/whisper");
        assert_eq!(package.status, DeclaredStatus::NotMapped);
        assert_eq!(package.gap, Some(MappingGap::SourceTypeUnsupported));
    }

    // ---- custom tools ----

    fn approval_agent() -> PackageReading {
        reading(agent(json!({
            "custom_tools": {"request_approval": {
                "description": "Show the beat sheet to the person and wait for approval.",
                "input_schema": {"type": "object"}
            }}
        })))
    }

    /// An adapter that accepts everything, and `tools.definitions` too.
    fn taking_definitions() -> AdapterSupport {
        let mut support = everything();
        support.accepts.insert(AdapterConfigField::ToolDefinitions);
        support
    }

    #[test]
    fn a_custom_tool_keeps_the_tool_definition_of_the_config() {
        let reading = approval_agent();
        let definition = json!({"kind": "module", "ref": "tools/request_approval.ts#create"});
        let config = base(json!({"tools": {"definitions": {
            "request_approval": definition,
            "word_count": {"kind": "module", "ref": "tools/word_count.ts"}
        }}}));

        let mut mapped = map_with(
            &reading,
            config.clone(),
            &taking_definitions(),
            &AgentPackageOptions::default(),
        );

        // The loader keeps the caller's entries and writes none of its own.
        assert_eq!(mapped.config.tools, config.tools);
        let tool = item(&mapped, "agent:/custom_tools/request_approval");
        assert_eq!(tool.status, DeclaredStatus::CouldNotBeChecked);
        assert_eq!(tool.gap, None);
        assert_eq!(tool.mapped_to, vec!["tools.definitions.request_approval"]);
        assert_eq!(tool.bound_to, None);
        assert!(tool.required);
        // Nothing compared the code with the document, so it follows `if_unverified`.
        let (outcome, _) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Ready);
        assert_eq!(
            item(&mapped, "agent:/custom_tools/request_approval").effect,
            ItemEffect::Warned
        );
        let (outcome, stopped) =
            apply_actions(&mut mapped.items, PackageAction::Stop, PackageAction::Stop);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(stopped, vec!["agent:/custom_tools/request_approval"]);
    }

    #[test]
    fn a_binding_for_a_custom_tool_comes_before_a_tool_definition() {
        let reading = approval_agent();
        let config = base(json!({"tools": {"definitions": {
            "request_approval": {"kind": "module", "ref": "tools/request_approval.ts#create"}
        }}}));
        let options = AgentPackageOptions {
            bindings: BTreeMap::from([(
                "custom_tools.request_approval".to_string(),
                "studio.tools:request_approval".to_string(),
            )]),
            ..AgentPackageOptions::default()
        };

        let mapped = map_with(&reading, config.clone(), &taking_definitions(), &options);

        let tool = item(&mapped, "agent:/custom_tools/request_approval");
        assert_eq!(tool.status, DeclaredStatus::Bound);
        assert_eq!(
            tool.bound_to.as_deref(),
            Some("studio.tools:request_approval")
        );
        assert!(tool.mapped_to.is_empty());
        assert_eq!(mapped.config.tools, config.tools);
    }

    #[test]
    fn a_custom_tool_that_nothing_supplies_is_not_bound() {
        let reading = approval_agent();
        // (the adapter, the config's tools, a piece of the detail)
        let table = [
            (
                taking_definitions(),
                json!({"definitions": {"word_count": {"kind": "module", "ref": "tools/word_count.ts"}}}),
                "the config has no tools.definitions.request_approval entry",
            ),
            (
                everything(),
                json!({}),
                "the adapter does not accept tools.definitions",
            ),
        ];
        for (support, tools, reason) in table {
            let mut mapped = map_with(
                &reading,
                base(json!({"tools": tools})),
                &support,
                &AgentPackageOptions::default(),
            );

            let tool = item(&mapped, "agent:/custom_tools/request_approval");
            assert_eq!(tool.status, DeclaredStatus::NotMapped, "{reason}");
            assert_eq!(tool.gap, Some(MappingGap::NotBound), "{reason}");
            let detail = tool.detail.as_deref().expect("a detail");
            assert!(detail.contains(reason), "{detail}");
            assert!(tool.mapped_to.is_empty());
            let (outcome, stopped) = stopped_by(&mut mapped);
            assert_eq!(outcome, PackageOutcome::Stopped);
            assert_eq!(stopped, vec!["agent:/custom_tools/request_approval"]);
        }
    }

    // ---- connectors and tools ----

    fn running_agent() -> PackageReading {
        let source = "mcp::https://mcp.example.com/issues";
        let mut reading = reading(agent(json!({
            "connectors": {"issues": {"source": source, "version": "1.4.0"}},
            "tools": [
                {"builtin": "read_file", "description": "Read the changelog."},
                {"connector": "issues", "tool": "search"}
            ]
        })));
        reading.parts = vec![part(
            PartField::Connectors,
            "issues",
            source,
            SourceType::Pointer,
            Integrity::Pointer,
        )];
        reading
    }

    fn bound_config() -> FabricConfig {
        base(json!({
            "mcp": {"servers": {
                "issues": {
                    "transport": "streamable-http",
                    "url": "https://mcp.example.com/issues",
                    "exposure": "harness_native"
                },
                "caller-own": {
                    "transport": "streamable-http",
                    "url": "https://mcp.example.com/other",
                    "exposure": "harness_native"
                }
            }}
        }))
    }

    #[test]
    fn a_bound_running_connector_could_not_be_checked_and_keeps_the_config_entry() {
        let reading = running_agent();
        let config = bound_config();
        let servers = config.mcp.clone().expect("mcp").servers;

        let mut mapped = map_with(
            &reading,
            config,
            &everything(),
            &AgentPackageOptions::default(),
        );

        assert_eq!(mapped.config.mcp.as_ref().expect("mcp").servers, servers);
        let connector = item(&mapped, "agent:/connectors/issues");
        assert_eq!(connector.status, DeclaredStatus::CouldNotBeChecked);
        assert_eq!(connector.mapped_to, vec!["mcp.servers.issues"]);
        assert!(
            connector
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("were not compared"))
        );
        let builtin = item(&mapped, "agent:/tools/0");
        assert_eq!(builtin.kind, DeclaredKind::Tool);
        assert_eq!(builtin.name.as_deref(), Some("read_file"));
        assert_eq!(builtin.status, DeclaredStatus::CouldNotBeChecked);
        assert_eq!(builtin.reason.as_deref(), Some("Read the changelog."));
        let tool = item(&mapped, "agent:/tools/1");
        assert_eq!(tool.name.as_deref(), Some("search"));
        assert_eq!(tool.connector.as_deref(), Some("issues"));
        assert_eq!(tool.status, DeclaredStatus::CouldNotBeChecked);
        // Default: warn. With if_unverified stop, each of the three stops.
        let (outcome, _) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Ready);
        let (outcome, stopped) =
            apply_actions(&mut mapped.items, PackageAction::Stop, PackageAction::Stop);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(
            stopped,
            vec![
                "agent:/connectors/issues",
                "agent:/tools/0",
                "agent:/tools/1"
            ]
        );
    }

    #[test]
    fn a_running_connector_that_is_not_bound_or_not_accepted_is_not_mapped() {
        let reading = running_agent();
        // (config, support, expected gap)
        let table = [
            (base(json!({})), everything(), MappingGap::NotBound),
            (
                bound_config(),
                without(AdapterConfigField::Mcp),
                MappingGap::AdapterDoesNotAccept,
            ),
        ];
        for (config, support, gap) in table {
            let mut mapped = map_with(&reading, config, &support, &AgentPackageOptions::default());

            let connector = item(&mapped, "agent:/connectors/issues");
            assert_eq!(connector.status, DeclaredStatus::NotMapped);
            assert_eq!(connector.gap, Some(gap));
            // The tool of a connector that is not mapped is not mapped either.
            let tool = item(&mapped, "agent:/tools/1");
            assert_eq!(tool.status, DeclaredStatus::NotMapped);
            assert_eq!(tool.gap, Some(gap));
            let (outcome, stopped) = stopped_by(&mut mapped);
            assert_eq!(outcome, PackageOutcome::Stopped);
            assert_eq!(stopped, vec!["agent:/connectors/issues", "agent:/tools/1"]);
        }
    }

    /// A reading with carried connectors whose `mcp.json` files are on disk.
    fn carried(label: &str, connectors: &[(&str, Value)]) -> (PackageReading, PathBuf) {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "nemo-fabric-agent-package-mapping-{label}-{}-{unique}",
            std::process::id()
        ));
        let mut entries = Map::new();
        let mut parts = Vec::new();
        let mut listings = BTreeMap::new();
        for (name, mcp) in connectors {
            let folder = root.join("connectors").join(name);
            std::fs::create_dir_all(&folder).expect("connector folder should be created");
            let bytes = serde_json::to_vec_pretty(mcp).expect("encode mcp.json");
            std::fs::write(folder.join("mcp.json"), &bytes).expect("mcp.json should be written");
            let folder = folder
                .canonicalize()
                .expect("connector folder should resolve");
            let mut locked = local(PartField::Connectors, name);
            locked.path = Some(folder);
            listings.insert(
                locked.address.clone(),
                FolderListing {
                    digest: "sha256:22".to_string(),
                    files: BTreeMap::from([(
                        "mcp.json".to_string(),
                        crate::agent_package::digest::bytes_digest(&bytes),
                    )]),
                },
            );
            parts.push(locked);
            entries.insert(
                (*name).to_string(),
                json!({"source": format!("./connectors/{name}")}),
            );
        }
        let mut reading = reading(agent(json!({"connectors": entries})));
        reading.parts = parts;
        reading.listings = listings;
        (reading, root)
    }

    fn stdio(command: &str) -> Value {
        json!({"type": "stdio", "command": command, "args": ["server.js"]})
    }

    #[cfg(unix)]
    #[test]
    fn a_carried_connector_becomes_a_stdio_server_that_starts_in_its_folder() {
        let (mut reading, root) = carried(
            "one",
            &[(
                "changelog",
                json!({"mcpServers": {"main": {
                    "type": "stdio",
                    "command": "node",
                    "args": ["server.js"],
                    "env": {"LOG": "1"}
                }}}),
            )],
        );
        reading.agent_as_written["tools"] = json!([{"connector": "changelog", "tool": "list"}]);
        reading.agent = reading.agent_as_written.clone();
        let folder = reading.parts[0].path.clone().expect("a folder");

        let mapped = map(&reading, &everything());

        let servers = &mapped.config.mcp.as_ref().expect("mcp").servers;
        let server = &servers["changelog"];
        assert_eq!(server.transport, McpTransport::Stdio);
        assert_eq!(server.url, "/bin/sh");
        assert_eq!(
            server.args[3..],
            [
                folder.to_str().expect("text").to_string(),
                "node".to_string(),
                "server.js".to_string()
            ]
        );
        assert_eq!(server.env["LOG"], "1");
        let connector = item(&mapped, "agent:/connectors/changelog");
        assert_eq!(connector.status, DeclaredStatus::Mapped);
        assert_eq!(connector.mapped_to, vec!["mcp.servers.changelog"]);
        let tool = item(&mapped, "agent:/tools/0");
        assert_eq!(tool.status, DeclaredStatus::CouldNotBeChecked);
        assert_eq!(tool.connector.as_deref(), Some("changelog"));
        std::fs::remove_dir_all(root).expect("temporary folder should be removed");
    }

    #[cfg(unix)]
    #[test]
    fn mcp_json_keys_that_are_not_read_are_named_in_the_item_and_in_a_note() {
        let (reading, root) = carried(
            "keys",
            &[(
                "changelog",
                json!({"mcpServers": {"main": {
                    "type": "stdio",
                    "command": "node",
                    "cwd": ".",
                    "disabled": true,
                    "alwaysAllow": ["list"]
                }}}),
            )],
        );

        let mapped = map(&reading, &everything());

        // The server is mapped: the keys change nothing about how it starts.
        let connector = item(&mapped, "agent:/connectors/changelog");
        assert_eq!(connector.status, DeclaredStatus::Mapped);
        let detail = connector.detail.as_deref().expect("a detail");
        assert!(
            detail.contains("started from the connector's folder"),
            "{detail}"
        );
        assert!(detail.contains("alwaysAllow, disabled"), "{detail}");
        let note = mapped
            .notes
            .iter()
            .find(|note| note.code == NoteCode::ConnectorKeyNotRead)
            .expect("a note about the keys");
        assert_eq!(note.subject, "mcp.servers.changelog");
        assert!(note.text.contains("alwaysAllow, disabled"), "{}", note.text);
        std::fs::remove_dir_all(root).expect("temporary folder should be removed");
    }

    #[cfg(unix)]
    #[test]
    fn two_carried_servers_with_one_key_leave_the_later_connector_unmapped() {
        let (reading, root) = carried(
            "clash",
            &[
                (
                    "files",
                    json!({"mcpServers": {"read": stdio("node"), "write": stdio("node")}}),
                ),
                ("files-read", json!({"mcpServers": {"main": stdio("node")}})),
            ],
        );

        let mapped = map(&reading, &everything());

        let servers = &mapped.config.mcp.as_ref().expect("mcp").servers;
        assert_eq!(
            servers.keys().collect::<Vec<_>>(),
            vec!["files-read", "files-write"]
        );
        assert_eq!(
            item(&mapped, "agent:/connectors/files").mapped_to,
            vec!["mcp.servers.files-read", "mcp.servers.files-write"]
        );
        let later = item(&mapped, "agent:/connectors/files-read");
        assert_eq!(later.status, DeclaredStatus::NotMapped);
        assert_eq!(later.gap, Some(MappingGap::KeyClash));
        std::fs::remove_dir_all(root).expect("temporary folder should be removed");
    }

    #[test]
    fn a_carried_connector_follows_if_unsupported_when_it_cannot_be_started() {
        // (mcp.json, support, expected gap)
        let table = [
            (
                json!({"mcpServers": {"main": {"type": "http", "url": "https://x.example"}}}),
                everything(),
                MappingGap::UnsupportedShape,
            ),
            (
                json!({"servers": {}}),
                everything(),
                MappingGap::UnsupportedShape,
            ),
            (
                json!({"mcpServers": {"main": stdio("node")}}),
                without(AdapterConfigField::Mcp),
                MappingGap::AdapterDoesNotAccept,
            ),
        ];
        for (mcp, support, gap) in table {
            let (reading, root) = carried("gap", &[("changelog", mcp)]);

            let mut mapped = map(&reading, &support);

            assert!(mapped.config.mcp.is_none(), "{gap:?}");
            let connector = item(&mapped, "agent:/connectors/changelog");
            assert_eq!(connector.status, DeclaredStatus::NotMapped);
            assert_eq!(connector.gap, Some(gap));
            let (outcome, stopped) = stopped_by(&mut mapped);
            assert_eq!(outcome, PackageOutcome::Stopped);
            assert_eq!(stopped, vec!["agent:/connectors/changelog"]);
            std::fs::remove_dir_all(root).expect("temporary folder should be removed");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_config_server_under_a_carried_key_is_a_conflict() {
        let (reading, root) = carried(
            "conflict",
            &[("changelog", json!({"mcpServers": {"main": stdio("node")}}))],
        );
        let config = base(json!({
            "mcp": {"servers": {"changelog": {
                "transport": "stdio", "url": "/usr/bin/other", "exposure": "harness_native"
            }}}
        }));

        assert_eq!(
            refusal(&reading, config),
            (
                "package_config_conflict".to_string(),
                "mcp.servers.changelog".to_string()
            )
        );
        std::fs::remove_dir_all(root).expect("temporary folder should be removed");
    }

    #[test]
    fn an_mcp_json_that_changed_after_the_check_is_refused() {
        let (mut reading, root) = carried(
            "changed",
            &[("changelog", json!({"mcpServers": {"main": stdio("node")}}))],
        );
        reading
            .listings
            .get_mut("$/connectors/changelog")
            .expect("a listing")
            .files
            .insert("mcp.json".to_string(), "sha256:33".to_string());

        assert_eq!(
            refusal(&reading, base(json!({}))),
            (
                "package_digest_mismatch".to_string(),
                "$/connectors/changelog".to_string()
            )
        );
        std::fs::remove_dir_all(root).expect("temporary folder should be removed");
    }

    #[test]
    fn an_item_listed_twice_is_reported_once() {
        let reading = reading(agent(json!({
            "tools": [{"builtin": "read_file"}, {"builtin": "read_file"}, {"builtin": "grep"}]
        })));

        let mapped = map(&reading, &everything());

        assert_eq!(
            ids(&mapped.items)
                .into_iter()
                .filter(|id| id.starts_with("agent:/tools/"))
                .collect::<Vec<_>>(),
            vec!["agent:/tools/0", "agent:/tools/2"]
        );
    }

    // ---- needs ----

    fn need(pointer: &str, status: NeedStatus, optional: bool) -> NeedLine {
        NeedLine {
            need: pointer.trim_start_matches('/').replace('/', "."),
            pointer: pointer.to_string(),
            status,
            connector: (status == NeedStatus::MetRemotely).then(|| "render".to_string()),
            optional,
            stated_by: StatedBy::Agent,
            document: NAME.to_string(),
            reason: Some("Generated scenes are large.".to_string()),
            detail: Some("needs 32gb, has 16gb".to_string()),
        }
    }

    #[test]
    fn needs_keep_their_status_reason_and_detail() {
        let reading = reading(agent(json!({})));
        let needs = [
            need("/runtime/compute/memory", NeedStatus::NotMet, false),
            need("/runtime/compute/gpu", NeedStatus::MetRemotely, true),
            need(
                "/runtime/software/ffmpeg",
                NeedStatus::CouldNotBeChecked,
                false,
            ),
            need("/runtime/network", NeedStatus::MetHere, false),
            need("/model/capabilities/0", NeedStatus::NotMet, true),
        ];

        let mut mapped = map_package(
            &reading,
            base(json!({})),
            &everything(),
            &AgentPackageOptions::default(),
            &needs,
            &Facts::default(),
        )
        .expect("the package maps");

        let memory = item(&mapped, "agent:/runtime/compute/memory");
        assert_eq!(memory.kind, DeclaredKind::Need);
        assert_eq!(memory.field, "runtime");
        assert_eq!(memory.name.as_deref(), Some("memory"));
        assert_eq!(memory.status, DeclaredStatus::NotMet);
        assert!(memory.required);
        assert_eq!(
            memory.reason.as_deref(),
            Some("Generated scenes are large.")
        );
        assert_eq!(memory.detail.as_deref(), Some("needs 32gb, has 16gb"));
        let gpu = item(&mapped, "agent:/runtime/compute/gpu");
        assert_eq!(gpu.status, DeclaredStatus::MetRemotely);
        assert_eq!(gpu.connector.as_deref(), Some("render"));
        assert!(!gpu.required);
        assert_eq!(
            item(&mapped, "agent:/runtime/network").status,
            DeclaredStatus::MetHere
        );
        assert_eq!(item(&mapped, "agent:/model/capabilities/0").field, "model");
        let (outcome, stopped) = stopped_by(&mut mapped);
        assert_eq!(outcome, PackageOutcome::Stopped);
        assert_eq!(stopped, vec!["agent:/runtime/compute/memory"]);
        assert_eq!(
            item(&mapped, "agent:/runtime/software/ffmpeg").effect,
            ItemEffect::Warned
        );
        assert_eq!(
            item(&mapped, "agent:/model/capabilities/0").effect,
            ItemEffect::None
        );
    }

    // ---- no config ----

    #[test]
    fn without_a_config_every_declared_thing_is_listed_and_nothing_is_evaluated() {
        let mut reading = reading(agent(json!({
            "description": "Writes release notes.",
            "instructions": "Write short notes.",
            "model": {"source": "ollama::llama3.2:3b", "parameters": {"temperature": 0.2}},
            // A number NeMo Fabric cannot hold must not refuse a load without a config.
            "limits": {"max_turns": 4_294_967_296u64, "max_depth": 1},
            "skills": {"notes": {"source": "./skills/notes", "variables": {"style": "plain"}}},
            "hooks": {"guard": {"source": "./hooks/guard"}},
            "connectors": {"issues": {"source": "mcp::https://mcp.example.com", "version": "1.0.0"}},
            "tools": [{"builtin": "read_file"}, {"connector": "issues", "tool": "search"}],
            "extensions": {"claude-code": {}}
        })));
        reading.parts = vec![
            local(PartField::Skills, "notes"),
            local(PartField::Hooks, "guard"),
        ];
        let needs = [need("/runtime/compute/memory", NeedStatus::NotMet, false)];

        let items = unmapped_items(
            &reading,
            &AgentPackageOptions::default(),
            &needs,
            &Facts::default(),
        );

        assert_eq!(
            ids(&items),
            vec![
                "agent:/name",
                "agent:/description",
                "agent:/instructions",
                "agent:/model/source",
                "agent:/model/parameters/temperature",
                "agent:/limits/max_depth",
                "agent:/limits/max_turns",
                "agent:/skills/notes",
                "agent:/skills/notes/variables",
                "agent:/connectors/issues",
                "agent:/hooks/guard",
                "agent:/tools/0",
                "agent:/tools/1",
                "agent:/runtime/compute/memory",
                "agent:/extensions/claude-code",
            ]
        );
        for item in &items {
            if item.kind == DeclaredKind::Need {
                assert_eq!(item.status, DeclaredStatus::NotMet);
            } else {
                assert_eq!(item.status, DeclaredStatus::NotEvaluated, "{}", item.id);
                assert_eq!(item.gap, None);
                assert!(item.mapped_to.is_empty());
            }
            assert_eq!(item.effect, ItemEffect::None);
        }
        let model = unmapped_model(&reading);
        assert_eq!(model.origin, ModelOrigin::NotEvaluated);
        assert_eq!(
            model.declared_source.as_deref(),
            Some("ollama::llama3.2:3b")
        );
        assert_eq!(model.model, None);
    }

    // ---- a definition read together with its agent ----

    #[test]
    fn each_item_names_the_document_that_states_it() {
        let written = agent(json!({
            "description": "Writes release notes.",
            "instructions": {"var": "job"},
            "variables": {"job": {"type": "string"}, "turns": {"type": "number", "default": 12}},
            "limits": {"max_turns": {"var": "turns"}, "timeout_seconds": 600},
            "skills": {"notes": {"source": "./skills/notes"}},
            "tools": [{"builtin": "read_file"}]
        }));
        let definition = json!({
            "$schema": "https://example.com/schemas/0.4.0/agent-definition.schema.json",
            "type": "agent-definition",
            "name": "short-notes",
            "description": "Short release notes.",
            "agent": {"source": "./../../agent.json"},
            "variables": {"job": "Draft notes."},
            "instructions": "Keep it under ten lines.",
            "limits": {"max_turns": 5},
            "skills": {
                "notes": {"source": "./skills/notes"},
                "brevity": {"source": "./skills/brevity"}
            },
            "tools": [{"builtin": "read_file"}, {"builtin": "grep"}]
        });
        let mut runs = written.clone();
        runs["instructions"] = json!("Draft notes.\n\nKeep it under ten lines.");
        runs["limits"] = json!({"max_turns": 5, "timeout_seconds": 600});
        runs["skills"]["brevity"] = json!({"source": "./skills/brevity"});
        runs["tools"] = json!([{"builtin": "read_file"}, {"builtin": "grep"}]);
        let mut reading = reading_with_definition(written, definition, runs);
        reading.variables = vec![
            VariableValue {
                name: "job".to_string(),
                value_type: "string".to_string(),
                value: json!("Draft notes."),
                origin: ValueOrigin::Definition,
            },
            VariableValue {
                name: "turns".to_string(),
                value_type: "number".to_string(),
                value: json!(12),
                origin: ValueOrigin::Default,
            },
        ];
        let mut agent_skill = local(PartField::Skills, "notes");
        agent_skill.address = "$/agent/skills/notes".to_string();
        agent_skill.owner = "$/agent".to_string();
        let mut added_skill = local(PartField::Skills, "brevity");
        added_skill.stated_by = StatedBy::Definition;
        reading.parts = vec![agent_skill, added_skill];
        let config = base(json!({"metadata": {"name": "short-notes"}}));

        let mapped = map_with(
            &reading,
            config,
            &everything(),
            &AgentPackageOptions::default(),
        );

        assert_eq!(mapped.config.metadata.name, "short-notes");
        assert_eq!(
            mapped.config.metadata.description.as_deref(),
            Some("Short release notes.")
        );
        assert_eq!(
            mapped
                .config
                .instructions
                .as_ref()
                .expect("instructions")
                .system
                .as_ref()
                .expect("system")
                .content,
            "Draft notes.\n\nKeep it under ten lines."
        );
        assert_eq!(mapped.config.runtime.max_turns, Some(5));
        assert_eq!(
            mapped.config.skills.as_ref().expect("skills").paths,
            vec![
                PathBuf::from("/pkg/skills/notes"),
                PathBuf::from("/pkg/skills/brevity")
            ]
        );
        assert_eq!(
            ids(&mapped.items),
            vec![
                "agent:/name",
                "definition:/name",
                "agent:/description",
                "definition:/description",
                "agent:/instructions",
                "definition:/instructions",
                "agent:/limits/max_turns",
                "agent:/limits/timeout_seconds",
                "definition:/limits/max_turns",
                "agent:/skills/notes",
                "definition:/skills/brevity",
                "agent:/tools/0",
                "definition:/tools/1",
            ]
        );
        assert_eq!(item(&mapped, "agent:/name").status, DeclaredStatus::Skipped);
        assert_eq!(
            item(&mapped, "definition:/name").status,
            DeclaredStatus::Mapped
        );
        assert_eq!(
            item(&mapped, "agent:/description").status,
            DeclaredStatus::Skipped
        );
        for id in ["agent:/instructions", "definition:/instructions"] {
            assert_eq!(item(&mapped, id).status, DeclaredStatus::Mapped);
        }
        // Both documents' limits map to one config path, and 5 applies.
        for id in ["agent:/limits/max_turns", "definition:/limits/max_turns"] {
            let limit = item(&mapped, id);
            assert_eq!(limit.mapped_to, vec!["runtime.max_turns"]);
            assert_eq!(limit.detail.as_deref(), Some("5 applies"));
        }
        let added = item(&mapped, "definition:/skills/brevity");
        assert_eq!(added.stated_by, StatedBy::Definition);
        assert_eq!(added.document, "short-notes");
        assert_eq!(added.address.as_deref(), Some("$/skills/brevity"));
        let kept = item(&mapped, "agent:/skills/notes");
        assert_eq!(kept.stated_by, StatedBy::Agent);
        assert_eq!(kept.document, NAME);
        assert_eq!(kept.address.as_deref(), Some("$/agent/skills/notes"));
    }

    // ---- the format's compatibility cases ----

    /// A reading for one case document: local parts are verified, registry
    /// parts are pinned by an artifact digest, and an optional running agent
    /// is left out of the lock.
    fn case_reading(document: &Value) -> PackageReading {
        let mut reading = reading(document.clone());
        for field in PartField::ALL {
            let Some(entries) = document.get(field.as_str()).and_then(Value::as_object) else {
                continue;
            };
            for (name, entry) in entries {
                let source = entry["source"].as_str().expect("a source");
                if source.starts_with("./") {
                    reading.parts.push(local(field, name));
                } else if source.starts_with("a2a::") && is_optional(entry) {
                    reading.left_out.push(left_out(field, name, source));
                } else {
                    reading.parts.push(part(
                        field,
                        name,
                        source,
                        SourceType::Oci,
                        Integrity::NotVerifiable,
                    ));
                }
            }
        }
        reading
    }

    #[test]
    fn compatibility_cases_name_each_required_part_that_cannot_be_supported() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/agent_packages/conformance");
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(root.join("manifest.json")).expect("the vendored manifest"),
        )
        .expect("the manifest is JSON");
        // (case, ids that stop a default load)
        let table: [(&str, &[&str]); 6] = [
            (
                "called-agents-unsupported-is-named",
                &["agent:/agents/helper"],
            ),
            (
                "required-part-unsupported-is-named",
                &["agent:/plugins/remotion"],
            ),
            ("optional-part-unsupported-never-stops", &[]),
            (
                "one-required-entry-among-optional-ones-is-named",
                &["agent:/plugins/charts"],
            ),
            (
                "hook-that-cannot-be-run-is-named",
                &["agent:/hooks/tool-guard"],
            ),
            (
                "linked-policy-unreadable-is-named",
                &["agent:/policies/licensed-assets"],
            ),
        ];
        for (name, expected) in table {
            let path = format!("cases/{name}.json");
            let record = manifest["cases"]
                .as_array()
                .expect("cases")
                .iter()
                .find(|record| record["path"] == path.as_str())
                .unwrap_or_else(|| panic!("{path} is not in the vendored manifest"));
            let case: Value =
                serde_json::from_slice(&std::fs::read(root.join(&path)).expect("the case file"))
                    .expect("the case is JSON");
            assert_eq!(case["mode"], "compatibility", "{name}");
            let document = &case["documents"][0];
            let reading = case_reading(document);
            let config = base(json!({
                "metadata": {"name": document["name"].as_str().expect("a name")}
            }));

            let mut mapped = map_with(
                &reading,
                config,
                &everything(),
                &AgentPackageOptions::default(),
            );
            let (outcome, stopped) = stopped_by(&mut mapped);

            assert_eq!(stopped, expected, "{name}");
            // The case's own verdict: null accepts, anything else names what is missing.
            match record["error"].as_str() {
                None => assert_eq!(outcome, PackageOutcome::Ready, "{name}"),
                Some(error) => {
                    assert_eq!(outcome, PackageOutcome::Stopped, "{name}");
                    let named = error.rsplit(':').next().expect("a named thing");
                    let stopping = item(&mapped, &stopped[0]);
                    assert!(
                        stopping.field == named || stopping.name.as_deref() == Some(named),
                        "{name}: {error} is not {}",
                        stopping.id
                    );
                }
            }
            // Nothing declared is dropped: every part entry has an item.
            for field in PartField::ALL {
                for entry in document
                    .get(field.as_str())
                    .and_then(Value::as_object)
                    .into_iter()
                    .flat_map(|entries| entries.keys())
                {
                    let id = format!("agent:/{}/{entry}", field.as_str());
                    assert_eq!(item(&mapped, &id).status, DeclaredStatus::NotMapped);
                }
            }
        }
    }
}
