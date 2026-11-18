// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The vendored schemas of the package format.
//!
//! The four files under `assets/oadl/0.4.0/` are copies of the format's own schema files. This
//! module never spells a schema address: it reads `$id` from the files.

use std::sync::OnceLock;

use serde_json::Value;

use super::error::{PackageError, PackageErrorCode, PackageResult};

const AGENT: &str = include_str!("../../assets/oadl/0.4.0/agent.schema.json");
const AGENT_DEFINITION: &str = include_str!("../../assets/oadl/0.4.0/agent-definition.schema.json");
const LOCK: &str = include_str!("../../assets/oadl/0.4.0/lock.schema.json");
const REFERENCE: &str = include_str!("../../assets/oadl/0.4.0/reference.schema.json");

/// A document type that has a schema of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SchemaKind {
    Agent,
    AgentDefinition,
    Lock,
}

/// One schema failure: where in the document, which keyword, and the validator's words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchemaFailure {
    /// RFC 6901 JSON Pointer of the failing place. Empty for the top of the document.
    pub pointer: String,
    /// The JSON Schema keyword that refused it.
    pub keyword: String,
    pub message: String,
}

struct Loaded {
    agent_id: String,
    agent_definition_id: String,
    lock_id: String,
    agent: jsonschema::Validator,
    agent_definition: jsonschema::Validator,
    lock: jsonschema::Validator,
}

fn loaded() -> &'static Loaded {
    static LOADED: OnceLock<Loaded> = OnceLock::new();
    LOADED.get_or_init(|| {
        // The files are compiled in, so a failure here is a defect of this build.
        let parse = |text: &str| -> (String, Value) {
            let schema: Value = serde_json::from_str(text).expect("vendored schema is JSON");
            let id = schema["$id"]
                .as_str()
                .expect("vendored schema has an `$id`")
                .to_string();
            (id, schema)
        };
        let (agent_id, agent) = parse(AGENT);
        let (agent_definition_id, agent_definition) = parse(AGENT_DEFINITION);
        let (lock_id, lock) = parse(LOCK);
        let (reference_id, reference) = parse(REFERENCE);
        // The four files refer to each other by `$id`. Nothing else is ever retrieved: the
        // crate is built without a retriever.
        let registry = jsonschema::Registry::new()
            .add(agent_id.clone(), agent.clone())
            .and_then(|registry| {
                registry.add(agent_definition_id.clone(), agent_definition.clone())
            })
            .and_then(|registry| registry.add(lock_id.clone(), lock.clone()))
            .and_then(|registry| registry.add(reference_id, reference))
            .and_then(|registry| registry.prepare())
            .expect("vendored schemas form one registry");
        let build = |schema: &Value| {
            jsonschema::options()
                .with_registry(&registry)
                .build(schema)
                .expect("vendored schema compiles")
        };
        Loaded {
            agent: build(&agent),
            agent_definition: build(&agent_definition),
            lock: build(&lock),
            agent_id,
            agent_definition_id,
            lock_id,
        }
    })
}

/// The format version this loader reads.
pub(crate) fn format_version() -> &'static str {
    "0.4.0"
}

/// The `$id` read from the vendored file.
pub(crate) fn schema_id(kind: SchemaKind) -> &'static str {
    let loaded = loaded();
    match kind {
        SchemaKind::Agent => &loaded.agent_id,
        SchemaKind::AgentDefinition => &loaded.agent_definition_id,
        SchemaKind::Lock => &loaded.lock_id,
    }
}

/// The first schema failure of a document, in a fixed order: by place, then by keyword.
pub(crate) fn first_failure(kind: SchemaKind, document: &Value) -> Option<SchemaFailure> {
    let loaded = loaded();
    let validator = match kind {
        SchemaKind::Agent => &loaded.agent,
        SchemaKind::AgentDefinition => &loaded.agent_definition,
        SchemaKind::Lock => &loaded.lock,
    };
    validator
        .iter_errors(document)
        .map(|error| SchemaFailure {
            pointer: error.instance_path().as_str().to_string(),
            keyword: error.kind().keyword().to_string(),
            message: error.to_string(),
        })
        .min_by(|left, right| {
            (&left.pointer, &left.keyword, &left.message).cmp(&(
                &right.pointer,
                &right.keyword,
                &right.message,
            ))
        })
}

/// Checks one document against its schema.
///
/// The error subject is `subject`, then `#` and the JSON Pointer of the failing place. The
/// message names the keyword.
pub(crate) fn validate_schema(
    kind: SchemaKind,
    document: &Value,
    subject: &str,
) -> PackageResult<()> {
    match first_failure(kind, document) {
        None => Ok(()),
        Some(failure) => Err(PackageError::new(
            PackageErrorCode::SchemaInvalid,
            format!("{subject}#{}", failure.pointer),
            format!("{}: {}", failure.keyword, failure.message),
        )),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn the_four_files_load_and_each_id_is_read_from_its_file() {
        for (kind, text) in [
            (SchemaKind::Agent, AGENT),
            (SchemaKind::AgentDefinition, AGENT_DEFINITION),
            (SchemaKind::Lock, LOCK),
        ] {
            let schema: Value = serde_json::from_str(text).expect("schema is JSON");
            assert_eq!(schema["$id"], schema_id(kind));
            assert!(schema_id(kind).ends_with(".schema.json"));
        }
        let reference: Value = serde_json::from_str(REFERENCE).expect("schema is JSON");
        assert!(reference["$id"].is_string());
        assert_eq!(format_version(), "0.4.0");
        assert!(schema_id(SchemaKind::Agent).contains(format_version()));
    }

    #[test]
    fn the_smallest_agent_validates() {
        let agent =
            json!({"$schema": schema_id(SchemaKind::Agent), "type": "agent", "name": "example"});
        validate_schema(SchemaKind::Agent, &agent, "agent.json").expect("valid agent");
    }

    #[test]
    fn a_reference_is_checked_through_the_shared_schema() {
        // A source that is not local needs a version: the rule lives in the reference schema.
        let agent = json!({
            "$schema": schema_id(SchemaKind::Agent),
            "type": "agent",
            "name": "example",
            "plugins": {"remotion": {"source": "oci::registry.example.com/plugins/remotion"}}
        });
        let error =
            validate_schema(SchemaKind::Agent, &agent, "agent.json").expect_err("no version");
        assert_eq!(error.code, PackageErrorCode::SchemaInvalid);
        assert_eq!(error.subject, "agent.json#/plugins/remotion");
        assert!(error.message.starts_with("required:"), "{}", error.message);
    }

    #[test]
    fn an_unknown_field_is_named_by_place_and_keyword() {
        let agent = json!({
            "$schema": schema_id(SchemaKind::Agent),
            "type": "agent",
            "name": "example",
            "sandbox": true
        });
        let error =
            validate_schema(SchemaKind::Agent, &agent, "agent.json").expect_err("unknown field");
        assert_eq!(error.subject, "agent.json#");
        assert!(
            error.message.starts_with("additionalProperties:"),
            "{}",
            error.message
        );
    }

    #[test]
    fn a_definition_that_names_a_model_is_refused() {
        let definition = json!({
            "$schema": schema_id(SchemaKind::AgentDefinition),
            "type": "agent-definition",
            "name": "example",
            "agent": {"source": "./agent.json"},
            "model": {"source": "ollama::llama3.2:3b"}
        });
        let error = validate_schema(
            SchemaKind::AgentDefinition,
            &definition,
            "agent-definition.json",
        )
        .expect_err("a definition has no model field");
        assert_eq!(error.code, PackageErrorCode::SchemaInvalid);
    }

    #[test]
    fn a_lock_validates_against_its_schema() {
        let lock = json!({
            "$schema": schema_id(SchemaKind::Lock),
            "type": "lock",
            "entries": [{
                "address": "$",
                "type": "agent",
                "source": {"type": "local", "url": "./agent.json"},
                "version": "local",
                "digest": format!("sha256:{}", "0".repeat(64)),
                "dependencies": []
            }]
        });
        validate_schema(SchemaKind::Lock, &lock, "oadl.lock.json").expect("valid lock");
        let mut without_digest = lock.clone();
        without_digest["entries"][0]
            .as_object_mut()
            .expect("entry")
            .remove("digest");
        assert!(first_failure(SchemaKind::Lock, &without_digest).is_some());
    }
}
