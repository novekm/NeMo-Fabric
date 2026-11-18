// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the selected adapter accepts, read from its descriptor at load time.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde_json::{Map, Value};

use crate::adapter_contract::AdapterExtensionPoint;
use crate::config::{
    AdapterConfigField, FabricConfig, InstructionMode, ResolveContext, SkillDirectoryKind,
    discovered_adapter_ids, resolve_diagnostic_plan_from_config_with_adapter_directories,
};
use crate::error::{FabricError, Result};

/// The config fields, instruction modes, skill entries and root extension
/// keys the selected adapter accepts.
#[derive(Debug, Clone)]
pub(super) struct AdapterSupport {
    pub adapter_id: String,
    pub accepts: BTreeSet<AdapterConfigField>,
    pub instruction_modes: Vec<InstructionMode>,
    /// The `skills.directories` entry kinds the descriptor lists under
    /// `skills.directory_loading`. Empty when it declares no directory loading.
    pub skill_entries: Vec<SkillDirectoryKind>,
    /// The descriptor's schema for the root extension keys of the config:
    /// its `extension_schemas.agent_config` entry.
    pub extension_schema: Option<ExtensionSchema>,
}

/// An adapter's schema for the root extension keys of the config, compiled
/// once for a load.
#[derive(Debug, Clone)]
pub(super) struct ExtensionSchema {
    /// The keys the schema names under `properties`.
    keys: Vec<String>,
    /// The compiled schema, or why it does not compile.
    validator: std::result::Result<jsonschema::Validator, String>,
}

impl ExtensionSchema {
    pub(super) fn new(schema: &Map<String, Value>) -> Self {
        Self {
            keys: schema
                .get("properties")
                .and_then(Value::as_object)
                .map(|properties| properties.keys().cloned().collect())
                .unwrap_or_default(),
            validator: jsonschema::validator_for(&Value::Object(schema.clone()))
                .map_err(|error| error.to_string()),
        }
    }
}

impl AdapterSupport {
    pub(super) fn accepts(&self, field: AdapterConfigField) -> bool {
        self.accepts.contains(&field)
    }

    /// Whether planning takes a `skill` entry of `skills.directories` for this adapter.
    pub(super) fn takes_skill_entries(&self) -> bool {
        self.skill_entries.contains(&SkillDirectoryKind::Skill)
    }

    /// The root extension keys the descriptor names under `properties`.
    pub(super) fn extension_keys(&self) -> &[String] {
        self.extension_schema
            .as_ref()
            .map_or(&[], |schema| schema.keys.as_slice())
    }

    /// Why the adapter does not take the root extension key `key` as it
    /// stands in `extensions`, or `None` when it does.
    ///
    /// The adapter takes the key when its descriptor names it under
    /// `properties` and the whole of `extensions` fits the schema, as
    /// planning checks it. An open schema would let any key through, and
    /// that says nothing about the adapter reading it, so a key that is not
    /// named does not count.
    ///
    /// The reason goes into the load report, and `extensions` holds the
    /// caller's own values, so it names the place and never the value.
    pub(super) fn extension_gap(
        &self,
        key: &str,
        extensions: &BTreeMap<String, Value>,
    ) -> Option<String> {
        let Some(schema) = self
            .extension_schema
            .as_ref()
            .filter(|schema| schema.keys.iter().any(|named| named == key))
        else {
            return Some(format!(
                "the adapter's descriptor does not declare `{key}` under extension_schemas.{}",
                AdapterExtensionPoint::AgentConfig.as_str()
            ));
        };
        let validator = match &schema.validator {
            Ok(validator) => validator,
            Err(error) => {
                return Some(format!(
                    "the adapter's extension schema is not valid: {error}"
                ));
            }
        };
        let value = Value::Object(extensions.clone().into_iter().collect());
        let error = validator.iter_errors(&value).next()?;
        let place = error.instance_path().as_str();
        let reason = error.masked_with("the value");
        Some(if place.is_empty() {
            format!("the adapter's extension schema refuses it: {reason}")
        } else {
            format!("the adapter's extension schema refuses it at `{place}`: {reason}")
        })
    }
}

/// Plans the caller's config with the existing diagnostic planner and reads the
/// selected descriptor's `config` block. A config that does not plan is the
/// caller's error and is returned unchanged.
pub(super) fn adapter_support(
    config: &FabricConfig,
    context: &ResolveContext,
    adapter_directories: &[PathBuf],
) -> Result<AdapterSupport> {
    let plan = resolve_diagnostic_plan_from_config_with_adapter_directories(
        config.clone(),
        context.clone(),
        adapter_directories,
    )?;
    let Some(resolved) = plan.adapter_descriptor else {
        return Err(FabricError::InvalidConfig {
            field: "harness".to_string(),
            reason: "no adapter descriptor was resolved for the agent package".to_string(),
        });
    };
    let descriptor = resolved.descriptor;
    let accepts: BTreeSet<AdapterConfigField> = descriptor.config.accepts.iter().copied().collect();
    // A descriptor that predates mode discovery means `replace`.
    let instruction_modes = if accepts.contains(&AdapterConfigField::SystemInstructions) {
        descriptor
            .config
            .system_instruction_modes
            .clone()
            .unwrap_or_else(|| vec![InstructionMode::Replace])
    } else {
        Vec::new()
    };
    let skill_entries = descriptor
        .skills
        .as_ref()
        .and_then(|skills| skills.directory_loading.as_ref())
        .map(|loading| loading.supported_entries.clone())
        .unwrap_or_default();
    let extension_schema = descriptor
        .extension_schemas
        .get(&AdapterExtensionPoint::AgentConfig)
        .map(ExtensionSchema::new);
    Ok(AdapterSupport {
        adapter_id: descriptor.adapter_id,
        accepts,
        instruction_modes,
        skill_entries,
        extension_schema,
    })
}

/// (names of this harness, names NeMo Fabric knows).
///
/// The names of this harness are the selected adapter id and its last dot
/// segment. The second list holds the same two names for every adapter that
/// descriptor discovery finds for this config. It is empty when discovery
/// fails, and then a harness need for another name could not be checked.
pub(super) fn harness_names(
    config: &FabricConfig,
    support: &AdapterSupport,
    context: &ResolveContext,
    adapter_directories: &[PathBuf],
) -> (Vec<String>, Vec<String>) {
    let own = names_of(&support.adapter_id);
    let mut known = Vec::new();
    for adapter_id in
        discovered_adapter_ids(config, context, adapter_directories).unwrap_or_default()
    {
        for name in names_of(&adapter_id) {
            if !known.contains(&name) {
                known.push(name);
            }
        }
    }
    (own, known)
}

/// An adapter id and its last dot segment, when the two differ.
fn names_of(adapter_id: &str) -> Vec<String> {
    let mut names = vec![adapter_id.to_string()];
    if let Some(last) = adapter_id.rsplit('.').next()
        && !last.is_empty()
        && last != adapter_id
    {
        names.push(last.to_string());
    }
    names
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const AGENTS_KEY: &str = "nemo_fabric.agents.v1";

    /// An adapter whose descriptor has this schema for the root extension keys.
    fn support(schema: Value) -> AdapterSupport {
        AdapterSupport {
            adapter_id: "test.fabric.package_parts".to_string(),
            accepts: BTreeSet::new(),
            instruction_modes: Vec::new(),
            skill_entries: Vec::new(),
            extension_schema: schema.as_object().map(ExtensionSchema::new),
        }
    }

    #[test]
    fn a_key_counts_only_when_the_schema_names_it() {
        let extensions = BTreeMap::from([(AGENTS_KEY.to_string(), json!({}))]);
        let named = support(json!({"type": "object", "properties": {AGENTS_KEY: {}}}));
        // An open schema lets the key through and says nothing about reading it.
        let open = support(json!({"type": "object"}));

        assert_eq!(named.extension_keys(), [AGENTS_KEY]);
        assert_eq!(named.extension_gap(AGENTS_KEY, &extensions), None);
        assert!(open.extension_keys().is_empty());
        assert_eq!(
            open.extension_gap(AGENTS_KEY, &extensions).as_deref(),
            Some(
                "the adapter's descriptor does not declare `nemo_fabric.agents.v1` under extension_schemas.agent_config"
            )
        );
    }

    #[test]
    fn a_refusal_names_the_place_and_never_the_value() {
        // The caller's own entry holds text that must stay out of a report.
        let own = json!({"instructions": "Use the key STUDIO-KEY-12345 for the glossary."});
        let extensions = BTreeMap::from([(
            AGENTS_KEY.to_string(),
            json!({"translator": own, "reviewer": {"instructions": "Review."}}),
        )]);
        // (the schema of the key, a piece of the reason)
        let table = [
            (
                json!({"type": "object", "maxProperties": 1}),
                "more than 1 property",
            ),
            (json!({"type": "array"}), r#"is not of type "array""#),
            (json!({"enum": [{}]}), "is not one of"),
            (
                json!({"type": "object", "additionalProperties": {"required": ["description"]}}),
                r#""description" is a required property"#,
            ),
        ];
        for (schema, piece) in table {
            let support = support(json!({"type": "object", "properties": {AGENTS_KEY: schema}}));

            let reason = support
                .extension_gap(AGENTS_KEY, &extensions)
                .expect("a refusal");

            assert!(reason.contains("`/nemo_fabric.agents.v1"), "{reason}");
            assert!(reason.contains(piece), "{reason}");
            assert!(!reason.contains("STUDIO-KEY-12345"), "{reason}");
        }
    }

    #[test]
    fn names_are_the_id_and_its_last_segment() {
        assert_eq!(
            names_of("nvidia.fabric.claude"),
            vec!["nvidia.fabric.claude".to_string(), "claude".to_string()]
        );
        assert_eq!(names_of("claude"), vec!["claude".to_string()]);
        assert_eq!(names_of("odd."), vec!["odd.".to_string()]);
    }
}
