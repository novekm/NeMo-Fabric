// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Variables: where each value comes from, and the values an entry gives to a part.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::document::{
    allowed, declared_variables, type_of, value_at, value_at_mut, var_name, var_places,
};
use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::model::{ValueOrigin, VariableValue};
use super::schemas::{SchemaKind, first_failure};

/// An agent with every variable reference replaced by its value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedAgent {
    /// The agent with each `{"var": "<name>"}` replaced. It still declares its variables.
    pub agent: Value,
    /// The value of each variable, by name.
    pub values: Vec<VariableValue>,
}

fn declared_type(declaration: &Value) -> &str {
    declaration
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// First match wins: `given` (the start), `entry` (an agents entry or a definition), the default.
/// Refuses: an undeclared name, a wrong type, a value outside the allowed values, a variable with no value
/// and no default, and a value that does not fit the field it lands in (the filled agent is validated again).
pub(crate) fn resolve_variables(
    agent: &Value,
    given: &Map<String, Value>,
    entry: Option<&Map<String, Value>>,
    entry_origin: ValueOrigin,
) -> PackageResult<ResolvedAgent> {
    let agent_name = agent
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("the agent");
    let declared = declared_variables(agent);
    let mut values: BTreeMap<&str, (&Value, ValueOrigin)> = BTreeMap::new();
    let empty = Map::new();
    for (origin, supplied) in [
        (ValueOrigin::Start, given),
        (entry_origin, entry.unwrap_or(&empty)),
    ] {
        for (name, value) in supplied {
            let Some(declaration) = declared.get(name) else {
                return Err(PackageError::new(
                    PackageErrorCode::VariableUndeclared,
                    name.clone(),
                    format!("variables.{name}: {agent_name} declares no variable of that name"),
                ));
            };
            if type_of(value) != Some(declared_type(declaration)) {
                return Err(PackageError::new(
                    PackageErrorCode::VariableType,
                    name.clone(),
                    format!(
                        "variables.{name}: the value is not a {}",
                        declared_type(declaration)
                    ),
                ));
            }
            if !allowed(value, declaration) {
                return Err(PackageError::new(
                    PackageErrorCode::VariableNotAllowed,
                    name.clone(),
                    format!("variables.{name}: the value is not one of the allowed values"),
                ));
            }
            values.entry(name.as_str()).or_insert((value, origin));
        }
    }
    for (name, declaration) in &declared {
        if values.contains_key(name.as_str()) {
            continue;
        }
        match declaration.get("default") {
            Some(default) => {
                values.insert(name.as_str(), (default, ValueOrigin::Default));
            }
            None => {
                return Err(PackageError::new(
                    PackageErrorCode::VariableMissing,
                    name.clone(),
                    format!(
                        "variables.{name}: no value was given and it has no default; {agent_name} does not start"
                    ),
                ));
            }
        }
    }

    let mut resolved = agent.clone();
    let mut filled = false;
    for place in var_places(agent) {
        let Some(name) = value_at(agent, &place.path).and_then(var_name) else {
            continue;
        };
        let Some((value, _)) = values.get(name) else {
            return Err(PackageError::new(
                PackageErrorCode::VariableReferenceUndeclared,
                name,
                format!(
                    "{}: {agent_name} declares no variable named `{name}`",
                    place.place
                ),
            ));
        };
        if let Some(slot) = value_at_mut(&mut resolved, &place.path) {
            *slot = (*value).clone();
            filled = true;
        }
    }
    // A value is checked where it lands: a turn limit of 0 is refused like one written in the file.
    if filled && let Some(failure) = first_failure(SchemaKind::Agent, &resolved) {
        return Err(PackageError::new(
            PackageErrorCode::VariableValueInvalid,
            failure.pointer.clone(),
            format!(
                "a value does not fit the field it is written into: {}: {}",
                failure.keyword, failure.message
            ),
        ));
    }
    Ok(ResolvedAgent {
        agent: resolved,
        values: values
            .into_iter()
            .map(|(name, (value, origin))| VariableValue {
                name: name.to_string(),
                value_type: declared
                    .get(name)
                    .map(declared_type)
                    .unwrap_or_default()
                    .to_string(),
                value: value.clone(),
                origin,
            })
            .collect(),
    })
}

/// Values written on an entry against the variables its target declares. `own`: the variables of the document that holds the entry.
///
/// A value written in place is of the declared type and one of the allowed values. A value
/// written as a reference takes its type from `own`: its value is known when the agent is
/// started, and is checked then.
pub(crate) fn check_entry_values(
    given: &Map<String, Value>,
    declared: &Map<String, Value>,
    own: Option<&Map<String, Value>>,
    subject: &str,
) -> PackageResult<()> {
    for (name, value) in given {
        let Some(declaration) = declared.get(name) else {
            return Err(PackageError::new(
                PackageErrorCode::VariableUndeclared,
                name.clone(),
                format!(
                    "{subject}: variables.{name}: the target declares no variable of that name"
                ),
            ));
        };
        let reference = var_name(value);
        let kind = match reference {
            Some(variable) => own
                .and_then(|own| own.get(variable))
                .and_then(|declaration| declaration.get("type"))
                .and_then(Value::as_str),
            None => type_of(value),
        };
        if kind.is_some_and(|kind| kind != declared_type(declaration)) {
            return Err(PackageError::new(
                PackageErrorCode::VariableType,
                name.clone(),
                format!(
                    "{subject}: variables.{name}: the value is not a {}",
                    declared_type(declaration)
                ),
            ));
        }
        if reference.is_none() && !allowed(value, declaration) {
            return Err(PackageError::new(
                PackageErrorCode::VariableNotAllowed,
                name.clone(),
                format!("{subject}: variables.{name}: the value is not one of the allowed values"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::schemas::schema_id;
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

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().expect("object").clone()
    }

    fn summary(resolved: &ResolvedAgent) -> Vec<(String, Value, ValueOrigin)> {
        resolved
            .values
            .iter()
            .map(|value| (value.name.clone(), value.value.clone(), value.origin))
            .collect()
    }

    fn with_three_variables() -> Value {
        agent(json!({
            "instructions": {"var": "job"},
            "limits": {"max_turns": {"var": "turns"}},
            "model": {"source": {"var": "model"}},
            "skills": {"review": {"source": "./skills/review", "variables": {"depth": {"var": "turns"}, "mode": "full"}}},
            "variables": {
                "job": {"type": "string"},
                "turns": {"type": "number", "default": 20},
                "model": {"type": "string", "enum": ["ollama::small", "ollama::large"], "default": "ollama::small"}
            }
        }))
    }

    #[test]
    fn the_start_wins_over_the_entry_and_the_entry_over_the_default() {
        let resolved = resolve_variables(
            &with_three_variables(),
            &object(json!({"job": "from the start"})),
            Some(&object(json!({"job": "from the entry", "turns": 5}))),
            ValueOrigin::Definition,
        )
        .expect("every variable has a value");
        assert_eq!(
            summary(&resolved),
            [
                (
                    "job".to_string(),
                    json!("from the start"),
                    ValueOrigin::Start
                ),
                (
                    "model".to_string(),
                    json!("ollama::small"),
                    ValueOrigin::Default
                ),
                ("turns".to_string(), json!(5), ValueOrigin::Definition),
            ]
        );
        assert_eq!(resolved.values[2].value_type, "number");
        assert_eq!(resolved.agent["instructions"], "from the start");
        assert_eq!(resolved.agent["limits"]["max_turns"], 5);
        assert_eq!(resolved.agent["model"]["source"], "ollama::small");
        assert_eq!(
            resolved.agent["skills"]["review"]["variables"],
            json!({"depth": 5, "mode": "full"})
        );
        // The declarations stay.
        assert_eq!(
            resolved.agent["variables"],
            with_three_variables()["variables"]
        );
    }

    #[test]
    fn an_agent_with_no_variables_is_returned_as_written() {
        let document = agent(json!({"instructions": "Do the work."}));
        let resolved = resolve_variables(&document, &Map::new(), None, ValueOrigin::Entry)
            .expect("nothing to fill");
        assert_eq!(resolved.agent, document);
        assert!(resolved.values.is_empty());
    }

    #[test]
    fn each_refusal_names_the_variable() {
        let document = with_three_variables();
        let refuse = |given: Value, entry: Option<Value>| {
            resolve_variables(
                &document,
                &object(given),
                entry.map(object).as_ref(),
                ValueOrigin::Entry,
            )
            .expect_err("refused")
        };
        let error = refuse(json!({}), None);
        assert_eq!(
            (error.code, error.subject.as_str()),
            (PackageErrorCode::VariableMissing, "job")
        );
        let error = refuse(json!({"job": "x", "nope": 1}), None);
        assert_eq!(
            (error.code, error.subject.as_str()),
            (PackageErrorCode::VariableUndeclared, "nope")
        );
        let error = refuse(json!({"job": "x"}), Some(json!({"tone": "kind"})));
        assert_eq!(
            (error.code, error.subject.as_str()),
            (PackageErrorCode::VariableUndeclared, "tone")
        );
        let error = refuse(json!({"job": 5}), None);
        assert_eq!(
            (error.code, error.subject.as_str()),
            (PackageErrorCode::VariableType, "job")
        );
        let error = refuse(json!({"job": "x", "turns": true}), None);
        assert_eq!(
            (error.code, error.subject.as_str()),
            (PackageErrorCode::VariableType, "turns")
        );
        let error = refuse(json!({"job": "x", "model": "ollama::other"}), None);
        assert_eq!(
            (error.code, error.subject.as_str()),
            (PackageErrorCode::VariableNotAllowed, "model")
        );
    }

    #[test]
    fn a_value_is_checked_where_it_lands() {
        let error = resolve_variables(
            &with_three_variables(),
            &object(json!({"job": "x", "turns": 0})),
            None,
            ValueOrigin::Entry,
        )
        .expect_err("a turn limit of 0");
        assert_eq!(error.code, PackageErrorCode::VariableValueInvalid);
        assert_eq!(error.subject, "/limits/max_turns");
    }

    #[test]
    fn entry_values_are_checked_against_what_the_target_declares() {
        let declared = object(json!({
            "language": {"type": "string", "enum": ["en", "de"]},
            "max_line_length": {"type": "number"}
        }));
        let own = object(json!({"lines": {"type": "number"}, "tone": {"type": "string"}}));
        let check = |given: Value| {
            check_entry_values(&object(given), &declared, Some(&own), "$/agents/captioner")
        };
        check(json!({"language": "de", "max_line_length": 37})).expect("values in place");
        check(json!({"max_line_length": {"var": "lines"}})).expect("a reference of the same type");
        // A reference to a variable of an unknown type is checked when the agent starts.
        check(json!({"max_line_length": {"var": "unknown"}})).expect("checked at the start");
        let error = check(json!({"tone": "kind"})).expect_err("undeclared");
        assert_eq!(
            (error.code, error.subject.as_str()),
            (PackageErrorCode::VariableUndeclared, "tone")
        );
        let error = check(json!({"max_line_length": "long"})).expect_err("another type");
        assert_eq!(error.code, PackageErrorCode::VariableType);
        let error = check(json!({"max_line_length": {"var": "tone"}}))
            .expect_err("a reference of another type");
        assert_eq!(error.code, PackageErrorCode::VariableType);
        let error = check(json!({"language": "fr"})).expect_err("outside the allowed values");
        assert_eq!(
            (error.code, error.subject.as_str()),
            (PackageErrorCode::VariableNotAllowed, "language")
        );
    }
}
