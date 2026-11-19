// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One typed document: its shape, the checks no schema makes, and its references.

use serde_json::{Map, Value};

use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::model::{DocumentKind, PartField};
use super::paths::json_pointer;
use super::schemas::{SchemaKind, schema_id, unsupported_schema_message, validate_schema};

/// The JSON type word of a variable value: `string`, `number` or `boolean`.
pub(crate) fn type_of(value: &Value) -> Option<&'static str> {
    match value {
        Value::Bool(_) => Some("boolean"),
        Value::Number(_) => Some("number"),
        Value::String(_) => Some("string"),
        _ => None,
    }
}

/// Two JSON values that are the same value. `1` and `1.0` are the same number.
pub(crate) fn same_json(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => a == b || a.as_f64() == b.as_f64(),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_json(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, x)| b.get(key).is_some_and(|y| same_json(x, y)))
        }
        _ => left == right,
    }
}

/// The name a variable reference stands for: `{"var": "<name>"}` written in place of a whole value.
pub(crate) fn var_name(value: &Value) -> Option<&str> {
    let object = value.as_object()?;
    if object.len() != 1 {
        return None;
    }
    object.get("var")?.as_str()
}

/// Why a value is not allowed by its variable, or None when it is.
///
/// A variable narrows its type with a list of allowed values and, for a number, with a
/// `minimum` and a `maximum`. Both ends of the range are allowed.
pub(crate) fn refusal_reason(value: &Value, declaration: &Value) -> Option<String> {
    if let Some(listed) = declaration.get("enum").and_then(Value::as_array)
        && !listed
            .iter()
            .any(|item| type_of(item) == type_of(value) && same_json(item, value))
    {
        return Some("is not one of the allowed values".to_string());
    }
    let number = value.as_f64()?;
    if let Some(minimum) = declaration.get("minimum").and_then(Value::as_f64)
        && number < minimum
    {
        return Some(format!("is below the minimum, {minimum}"));
    }
    if let Some(maximum) = declaration.get("maximum").and_then(Value::as_f64)
        && number > maximum
    {
        return Some(format!("is above the maximum, {maximum}"));
    }
    None
}

/// The name of one item of `tools`: `builtin.<name>` or `<connector>.<tool>`.
pub(crate) fn tool_name(item: &Value) -> String {
    match item.get("builtin").and_then(Value::as_str) {
        Some(name) => format!("builtin.{name}"),
        None => format!(
            "{}.{}",
            item.get("connector").and_then(Value::as_str).unwrap_or(""),
            item.get("tool").and_then(Value::as_str).unwrap_or("")
        ),
    }
}

/// The connector tools, by name, that name a connector which is in none of `connectors`.
pub(crate) fn missing_connectors(
    tools: Option<&Value>,
    connectors: &[Option<&Value>],
) -> Vec<String> {
    let has = |name: &str| {
        connectors
            .iter()
            .any(|map| map.is_some_and(|map| map.get(name).is_some()))
    };
    tools
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| {
            item.get("connector")
                .and_then(Value::as_str)
                .is_some_and(|name| !has(name))
        })
        .map(tool_name)
        .collect()
}

/// One place of an agent that may hold a variable reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VarPlace {
    /// The keys from the top of the document down to the place.
    pub path: Vec<String>,
    /// The place in the format's dotted text: `limits.max_turns`.
    pub place: String,
    /// The type the place takes. None under the `variables` object of an entry, where the
    /// type is whatever the part declares.
    pub kind: Option<&'static str>,
}

fn place(path: &[&str], kind: Option<&'static str>) -> VarPlace {
    VarPlace {
        path: path.iter().map(|key| key.to_string()).collect(),
        place: path.join("."),
        kind,
    }
}

/// Every place of an agent that is written and may hold a variable reference.
pub(crate) fn var_places(agent: &Value) -> Vec<VarPlace> {
    let mut places = Vec::new();
    if agent.get("instructions").is_some() {
        places.push(place(&["instructions"], Some("string")));
    }
    if let Some(model) = agent.get("model") {
        if model.get("source").is_some() {
            places.push(place(&["model", "source"], Some("string")));
        }
        for key in keys(model.get("parameters")) {
            let kind = if key == "effort" { "string" } else { "number" };
            places.push(place(&["model", "parameters", key], Some(kind)));
        }
    }
    for key in keys(agent.get("limits")) {
        places.push(place(&["limits", key], Some("number")));
    }
    if agent.pointer("/runtime/harness/name").is_some() {
        places.push(place(&["runtime", "harness", "name"], Some("string")));
    }
    for field in ["skills", "agents"] {
        for name in keys(agent.get(field)) {
            for key in keys(agent[field][name].get("variables")) {
                places.push(place(&[field, name, "variables", key], None));
            }
        }
    }
    places
}

fn keys(value: Option<&Value>) -> impl Iterator<Item = &str> {
    value
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|object| object.keys().map(String::as_str))
}

/// The value at a place.
pub(crate) fn value_at<'a>(document: &'a Value, path: &[String]) -> Option<&'a Value> {
    path.iter().try_fold(document, |value, key| value.get(key))
}

/// The value at a place, to be replaced.
pub(crate) fn value_at_mut<'a>(document: &'a mut Value, path: &[String]) -> Option<&'a mut Value> {
    path.iter()
        .try_fold(document, |value, key| value.get_mut(key))
}

/// The variables a document declares, by name. Empty for a document that declares none.
pub(crate) fn declared_variables(document: &Value) -> Map<String, Value> {
    document
        .get("variables")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

/// Object, known `type`, `$schema` equal to that type's schema id, schema, then the agent rules.
pub(crate) fn check_document(document: &Value, subject: &str) -> PackageResult<DocumentKind> {
    let Some(object) = document.as_object() else {
        return Err(PackageError::new(
            PackageErrorCode::DocumentType,
            subject,
            "the document is not a JSON object",
        ));
    };
    let (kind, schema) = match object.get("type").and_then(Value::as_str) {
        Some("agent") => (DocumentKind::Agent, SchemaKind::Agent),
        Some("agent-definition") => (DocumentKind::AgentDefinition, SchemaKind::AgentDefinition),
        Some(other) => {
            return Err(PackageError::new(
                PackageErrorCode::DocumentType,
                subject,
                format!(
                    "`type` is `{other}`: an entry document is an `agent` or an `agent-definition`"
                ),
            ));
        }
        None => {
            return Err(PackageError::new(
                PackageErrorCode::DocumentType,
                subject,
                "`type` is missing or is not text",
            ));
        }
    };
    if object.get("$schema").and_then(Value::as_str) != Some(schema_id(schema)) {
        return Err(PackageError::new(
            PackageErrorCode::SchemaVersion,
            subject,
            unsupported_schema_message(object.get("$schema"), schema_id(schema)),
        ));
    }
    validate_schema(schema, document, subject)?;
    if kind == DocumentKind::Agent {
        check_agent_rules(document, subject)?;
    }
    Ok(kind)
}

/// The four checks no schema makes: a connector tool names a connector the agent has; a reference names
/// a declared variable; the variable has the type its place takes; a default is one of the allowed values.
pub(crate) fn check_agent_rules(agent: &Value, subject: &str) -> PackageResult<()> {
    if let Some(tool) = missing_connectors(agent.get("tools"), &[agent.get("connectors")])
        .into_iter()
        .next()
    {
        return Err(PackageError::new(
            PackageErrorCode::ToolConnectorUnknown,
            tool.clone(),
            format!("{subject}: tools: {tool} names a connector the agent does not have"),
        ));
    }
    let declared = declared_variables(agent);
    for (name, declaration) in &declared {
        if let (Some(minimum), Some(maximum)) = (
            declaration.get("minimum").and_then(Value::as_f64),
            declaration.get("maximum").and_then(Value::as_f64),
        ) && minimum > maximum
        {
            return Err(PackageError::new(
                PackageErrorCode::VariableBoundsInvalid,
                name.clone(),
                format!(
                    "{subject}: variables.{name}: the minimum, {minimum}, is greater than the maximum, {maximum}"
                ),
            ));
        }
        if let Some(default) = declaration.get("default")
            && let Some(reason) = refusal_reason(default, declaration)
        {
            return Err(PackageError::new(
                PackageErrorCode::VariableDefaultNotAllowed,
                name.clone(),
                format!("{subject}: variables.{name}: the default {reason}"),
            ));
        }
    }
    for place in var_places(agent) {
        let Some(name) = value_at(agent, &place.path).and_then(var_name) else {
            continue;
        };
        let Some(declaration) = declared.get(name) else {
            return Err(PackageError::new(
                PackageErrorCode::VariableReferenceUndeclared,
                name,
                format!(
                    "{subject}: {}: the document declares no variable named `{name}`",
                    place.place
                ),
            ));
        };
        let declared_type = declaration.get("type").and_then(Value::as_str);
        if let Some(kind) = place.kind
            && declared_type != Some(kind)
        {
            return Err(PackageError::new(
                PackageErrorCode::VariableReferenceType,
                name,
                format!(
                    "{subject}: {}: the place takes a {kind}, and the variable `{name}` is a {}",
                    place.place,
                    declared_type.unwrap_or("value of no type")
                ),
            ));
        }
    }
    Ok(())
}

/// One reference to a part, as a document writes it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Reference {
    pub field: PartField,
    pub name: String,
    /// JSON Pointer of the entry in its document.
    pub pointer: String,
    pub source: String,
    pub version: Option<String>,
    pub optional: bool,
    /// The entry's `variables`, as written.
    pub variables: Option<Map<String, Value>>,
}

/// Every reference to a part that an agent or a definition holds, by field and then by name.
pub(crate) fn references(document: &Value) -> Vec<Reference> {
    let mut found = Vec::new();
    for field in PartField::ALL {
        let Some(entries) = document.get(field.as_str()).and_then(Value::as_object) else {
            continue;
        };
        for (name, entry) in entries {
            let Some(source) = entry.get("source").and_then(Value::as_str) else {
                continue;
            };
            found.push(Reference {
                field,
                name: name.clone(),
                pointer: json_pointer(&[field.as_str(), name]),
                source: source.to_string(),
                version: entry
                    .get("version")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                optional: entry.get("optional").and_then(Value::as_bool) == Some(true),
                variables: entry.get("variables").and_then(Value::as_object).cloned(),
            });
        }
    }
    found
}

/// A source, read as local or as `type::address`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Source {
    /// `./path`: the text after `./`.
    Local { path: String },
    /// `type::address`. For a git source the address is the repository and the subpath is the
    /// folder after `//`.
    Typed {
        kind: String,
        address: String,
        subpath: Option<String>,
    },
}

/// Reads a source as local or as `type::address`. The loader never reads a version out of it.
pub(crate) fn parse_source(source: &str) -> Source {
    if let Some(path) = source.strip_prefix("./") {
        return Source::Local {
            path: path.to_string(),
        };
    }
    let Some((kind, address)) = source.split_once("::") else {
        return Source::Typed {
            kind: String::new(),
            address: source.to_string(),
            subpath: None,
        };
    };
    if kind != "git" {
        return Source::Typed {
            kind: kind.to_string(),
            address: address.to_string(),
            subpath: None,
        };
    }
    // git::<repository>//<folder>. A repository written as git@host:group/name has no scheme.
    let (prefix, rest) = match address.split_once("://") {
        Some((scheme, rest)) => (format!("{scheme}://"), rest),
        None => (String::new(), address),
    };
    let (repository, folder) = match rest.split_once("//") {
        Some((repository, folder)) => (repository, folder),
        None => (rest, ""),
    };
    Source::Typed {
        kind: kind.to_string(),
        address: format!("{prefix}{repository}"),
        subpath: (!folder.is_empty()).then(|| folder.to_string()),
    }
}

/// The part of an address that a message or a report leaves out: the password between
/// `://` and `@`, or all of what stands before `@` when the address is not `ssh`, where a
/// token is often written in place of a user name.
pub(crate) fn secret(url: &str) -> Option<(usize, usize)> {
    let start = url.find("://")? + 3;
    let authority = url[start..].split('/').next()?;
    let user_info = &authority[..authority.rfind('@')?];
    match user_info.find(':') {
        Some(colon) => Some((start + colon + 1, start + user_info.len())),
        None if url.starts_with("ssh://") => None,
        None => Some((start, start + user_info.len())),
    }
}

/// An address as a message or a report shows it, with no password in it.
pub(crate) fn without_password(url: &str) -> String {
    match secret(url) {
        Some((start, end)) => format!("{}***{}", &url[..start], &url[end..]),
        None => url.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn agent(more: Value) -> Value {
        let mut document = json!({
            "$schema": schema_id(SchemaKind::Agent),
            "type": "agent",
            "name": "example"
        });
        document
            .as_object_mut()
            .expect("object")
            .extend(more.as_object().expect("object").clone());
        document
    }

    fn refusal(document: &Value) -> PackageError {
        check_document(document, "agent.json").expect_err("refused")
    }

    #[test]
    fn an_agent_and_a_definition_are_recognized() {
        assert_eq!(
            check_document(&agent(json!({})), "agent.json").expect("agent"),
            DocumentKind::Agent
        );
        let definition = json!({
            "$schema": schema_id(SchemaKind::AgentDefinition),
            "type": "agent-definition",
            "name": "example",
            "agent": {"source": "./agent.json"}
        });
        assert_eq!(
            check_document(&definition, "agent-definition.json").expect("definition"),
            DocumentKind::AgentDefinition
        );
    }

    #[test]
    fn the_type_and_the_schema_address_are_checked_before_the_shape() {
        assert_eq!(refusal(&json!([])).code, PackageErrorCode::DocumentType);
        assert_eq!(
            refusal(&json!({"name": "example"})).code,
            PackageErrorCode::DocumentType
        );
        assert_eq!(
            refusal(&json!({"type": "lock", "entries": []})).code,
            PackageErrorCode::DocumentType
        );
        let mut other = agent(json!({}));
        other["$schema"] = json!("https://example.com/other.schema.json");
        assert_eq!(refusal(&other).code, PackageErrorCode::SchemaVersion);
        let mut swapped = agent(json!({}));
        swapped["$schema"] = json!(schema_id(SchemaKind::AgentDefinition));
        assert_eq!(refusal(&swapped).code, PackageErrorCode::SchemaVersion);
        let error = refusal(&agent(json!({"inputs": {}})));
        assert_eq!(error.code, PackageErrorCode::SchemaInvalid);
        assert_eq!(error.subject, "agent.json#");
    }

    #[test]
    fn a_connector_tool_names_a_connector_the_agent_has() {
        let good = agent(json!({
            "connectors": {"queue": {"source": "./connectors/queue"}},
            "tools": [{"connector": "queue", "tool": "render"}, {"builtin": "read_file"}]
        }));
        check_document(&good, "agent.json").expect("the connector is there");
        let error = refusal(&agent(
            json!({"tools": [{"connector": "queue", "tool": "render"}]}),
        ));
        assert_eq!(error.code, PackageErrorCode::ToolConnectorUnknown);
        assert_eq!(error.subject, "queue.render");
    }

    #[test]
    fn a_range_keeps_a_default_inside_it_and_its_minimum_below_its_maximum() {
        let range = |more: Value| {
            let mut declaration = json!({"type": "number", "minimum": 5, "maximum": 60});
            declaration
                .as_object_mut()
                .expect("object")
                .extend(more.as_object().expect("object").clone());
            agent(json!({"variables": {"turns": declaration}}))
        };
        for edge in [5, 60] {
            check_document(&range(json!({"default": edge})), "agent.json")
                .expect("both ends of the range are allowed");
        }
        let outside = refusal(&range(json!({"default": 61})));
        assert_eq!(outside.code, PackageErrorCode::VariableDefaultNotAllowed);
        assert_eq!(outside.subject, "turns");
        let inverted = refusal(&range(json!({"minimum": 70})));
        assert_eq!(inverted.code, PackageErrorCode::VariableBoundsInvalid);
        assert_eq!(inverted.subject, "turns");
    }

    #[test]
    fn a_default_is_one_of_the_allowed_values() {
        let good = agent(
            json!({"variables": {"size": {"type": "number", "enum": [1, 2], "default": 2.0}}}),
        );
        check_document(&good, "agent.json").expect("the default is listed");
        let error = refusal(&agent(json!({
            "variables": {"size": {"type": "number", "enum": [1, 2], "default": 3}}
        })));
        assert_eq!(error.code, PackageErrorCode::VariableDefaultNotAllowed);
        assert_eq!(error.subject, "size");
    }

    #[test]
    fn a_reference_names_a_declared_variable_of_the_type_its_place_takes() {
        let good = agent(json!({
            "instructions": {"var": "job"},
            "model": {"source": {"var": "model"}, "parameters": {"effort": {"var": "effort"}, "temperature": {"var": "heat"}}},
            "limits": {"max_turns": {"var": "turns"}},
            "runtime": {"harness": {"name": {"var": "harness"}}},
            "skills": {"review": {"source": "./skills/review", "variables": {"depth": {"var": "turns"}, "tone": {"var": "job"}}}},
            "variables": {
                "job": {"type": "string"}, "model": {"type": "string"}, "effort": {"type": "string"},
                "heat": {"type": "number"}, "turns": {"type": "number"}, "harness": {"type": "string"}
            }
        }));
        check_document(&good, "agent.json").expect("every reference fits");

        let error = refusal(&agent(json!({"instructions": {"var": "job"}})));
        assert_eq!(error.code, PackageErrorCode::VariableReferenceUndeclared);
        assert_eq!(error.subject, "job");

        let error = refusal(&agent(json!({
            "limits": {"max_turns": {"var": "job"}},
            "variables": {"job": {"type": "string"}}
        })));
        assert_eq!(error.code, PackageErrorCode::VariableReferenceType);
        assert_eq!(error.subject, "job");

        let error = refusal(&agent(json!({
            "model": {"parameters": {"effort": {"var": "heat"}}},
            "variables": {"heat": {"type": "number"}}
        })));
        assert_eq!(error.code, PackageErrorCode::VariableReferenceType);
    }

    #[test]
    fn references_are_listed_with_their_pointers() {
        let document = agent(json!({
            "skills": {"review": {"source": "./skills/review", "variables": {"depth": 2}}},
            "agents": {"helper": {"source": "a2a::https://agents.example.com/helper", "version": "2.1.0", "optional": true}},
            "connectors": {"a/b": {"source": "mcp::https://mcp.example.com/x", "version": "1.0"}}
        }));
        let found = references(&document);
        let summary: Vec<(&str, &str, &str, bool)> = found
            .iter()
            .map(|r| {
                (
                    r.field.as_str(),
                    r.name.as_str(),
                    r.pointer.as_str(),
                    r.optional,
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                ("skills", "review", "/skills/review", false),
                ("connectors", "a/b", "/connectors/a~1b", false),
                ("agents", "helper", "/agents/helper", true),
            ]
        );
        assert_eq!(found[0].variables.as_ref().expect("variables")["depth"], 2);
        assert_eq!(found[2].version.as_deref(), Some("2.1.0"));
    }

    #[test]
    fn a_source_is_read_as_local_or_as_type_and_address() {
        assert_eq!(
            parse_source("./skills/review"),
            Source::Local {
                path: "skills/review".to_string()
            }
        );
        let typed = |kind: &str, address: &str, subpath: Option<&str>| Source::Typed {
            kind: kind.to_string(),
            address: address.to_string(),
            subpath: subpath.map(str::to_string),
        };
        assert_eq!(
            parse_source("git::https://git.example.com/media/parts.git//skills/review"),
            typed(
                "git",
                "https://git.example.com/media/parts.git",
                Some("skills/review")
            )
        );
        assert_eq!(
            parse_source("git::https://git.example.com/media/parts.git"),
            typed("git", "https://git.example.com/media/parts.git", None)
        );
        assert_eq!(
            parse_source("git::git@git.example.com:media/parts.git//reviewer"),
            typed(
                "git",
                "git@git.example.com:media/parts.git",
                Some("reviewer")
            )
        );
        assert_eq!(
            parse_source("oci::registry.example.com/media/skills/review"),
            typed("oci", "registry.example.com/media/skills/review", None)
        );
        assert_eq!(
            parse_source("npm::@remotion/cli"),
            typed("npm", "@remotion/cli", None)
        );
        assert_eq!(
            parse_source("mcp::https://mcp.example.com/assets"),
            typed("mcp", "https://mcp.example.com/assets", None)
        );
    }

    #[test]
    fn numbers_compare_by_value() {
        assert!(same_json(&json!(1), &json!(1.0)));
        assert!(same_json(
            &json!({"a": [1, "x"]}),
            &json!({"a": [1.0, "x"]})
        ));
        assert!(!same_json(&json!(1), &json!(true)));
        assert!(!same_json(&json!({"a": 1}), &json!({"a": 1, "b": 2})));
        assert!(refusal_reason(&json!(2), &json!({"type": "number", "enum": [1, 2.0]})).is_none());
        assert!(refusal_reason(&json!("2"), &json!({"type": "number", "enum": [1, 2]})).is_some());
        assert!(refusal_reason(&json!("anything"), &json!({"type": "string"})).is_none());
    }
}
