// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checks for an agent that another agent lists.

use serde_json::{Map, Value};

use super::definition::smallest_limits;
use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::model::{CalledAgent, ValueOrigin};
use super::paths::address;
use super::variables::resolve_variables;

fn name_of(document: &Value) -> &str {
    document.get("name").and_then(Value::as_str).unwrap_or("")
}

/// What holds for the agent `name` that `caller` lists: it resolves to an agent, has a description,
/// takes the entry's values, runs under the caller's limits, and takes the caller's model when it names none.
///
/// `caller` is the calling agent as it is written. Its variables take their defaults here.
/// The package reader has a caller whose variables already have their values and calls
/// `called_under`, so only the tests call this form.
#[cfg(test)]
pub(crate) fn check_called_agent(
    caller: &Value,
    name: &str,
    document: &Value,
    depth: u64,
) -> PackageResult<CalledAgent> {
    check_shape(name, document)?;
    let caller = resolve_variables(caller, &Map::new(), None, ValueOrigin::Entry)?.agent;
    called_under(&caller, name, document, depth)
}

/// The called agent is an agent, and it has a description for the model to read.
fn check_shape(name: &str, document: &Value) -> PackageResult<()> {
    let kind = document.get("type").and_then(Value::as_str);
    if kind != Some("agent") {
        return Err(PackageError::new(
            PackageErrorCode::ReferenceType,
            address("$", &["agents", name]),
            format!(
                "agents.{name}: it resolves to type {}, not to an agent",
                kind.unwrap_or("none")
            ),
        ));
    }
    let described = document
        .get("description")
        .and_then(Value::as_str)
        .is_some_and(|text| !text.is_empty());
    if !described {
        return Err(PackageError::new(
            PackageErrorCode::CalledAgentDescription,
            address("$", &["agents", name]),
            format!("agents.{name}: the agent it resolves to has no description"),
        ));
    }
    Ok(())
}

/// The same checks for a caller whose variables already have their values.
///
/// `depth` is how many calls below the first agent the called agent runs.
pub(crate) fn called_under(
    caller: &Value,
    name: &str,
    document: &Value,
    depth: u64,
) -> PackageResult<CalledAgent> {
    check_shape(name, document)?;
    let entry_values = caller
        .get("agents")
        .and_then(|agents| agents.get(name))
        .and_then(|entry| entry.get("variables"))
        .and_then(Value::as_object);
    let own = resolve_variables(document, &Map::new(), entry_values, ValueOrigin::Entry)?;
    let caller_name = name_of(caller);
    let mut notes = Vec::new();
    let limits = smallest_limits(
        caller.get("limits").and_then(Value::as_object),
        own.agent.get("limits").and_then(Value::as_object),
        name,
        &format!("the caller {caller_name}"),
        depth,
        &mut notes,
    );
    let own_model = own
        .agent
        .pointer("/model/source")
        .and_then(Value::as_str)
        .map(str::to_string);
    let (model_source, model_from) = match own_model {
        Some(source) => (Some(source), "called agent"),
        None => (
            caller
                .pointer("/model/source")
                .and_then(Value::as_str)
                .map(str::to_string),
            "caller",
        ),
    };
    let custom_tools = document
        .get("custom_tools")
        .and_then(Value::as_object)
        .map(|tools| tools.keys().cloned().collect())
        .unwrap_or_default();
    Ok(CalledAgent {
        name: name.to_string(),
        address: address("$", &["agents", name]),
        under: caller_name.to_string(),
        limits,
        model_source,
        model_from: model_from.to_string(),
        values: own.values,
        notes,
        // The package reader knows the place and where each listed agent leads.
        place: vec![name.to_string()],
        document: name_of(document).to_string(),
        filled: own.agent,
        custom_tools,
        agents: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::model::NoteCode;
    use super::super::schemas::{SchemaKind, schema_id};
    use super::*;

    fn agent(name: &str, more: Value) -> Value {
        let mut document = json!({
            "$schema": schema_id(SchemaKind::Agent),
            "type": "agent",
            "name": name,
            "description": "Does one task."
        });
        document
            .as_object_mut()
            .expect("object")
            .extend(more.as_object().expect("object").clone());
        document
    }

    #[test]
    fn a_called_agent_takes_the_entry_values_the_smaller_limits_and_its_own_model() {
        let caller = agent(
            "video-producer",
            json!({
                "model": {"source": "ollama::large"},
                "limits": {"max_turns": 60, "max_depth": 1, "max_total_tokens": 800000},
                "variables": {"width": {"type": "number", "default": 37}},
                "agents": {"captioner": {"source": "./agents/captioner", "variables": {"max_line_length": {"var": "width"}}}}
            }),
        );
        let captioner = agent(
            "captioner",
            json!({
                "model": {"source": "ollama::small"},
                "limits": {"max_turns": 10, "max_total_tokens": 100000},
                "variables": {
                    "language": {"type": "string", "default": "en"},
                    "max_line_length": {"type": "number", "default": 42}
                }
            }),
        );
        let called = check_called_agent(&caller, "captioner", &captioner, 1).expect("called agent");
        assert_eq!(called.name, "captioner");
        assert_eq!(called.address, "$/agents/captioner");
        assert_eq!(called.under, "video-producer");
        assert_eq!(
            Value::Object(called.limits),
            json!({"max_turns": 10, "max_depth": 0, "max_total_tokens": 100000})
        );
        assert_eq!(called.model_source.as_deref(), Some("ollama::small"));
        assert_eq!(called.model_from, "called agent");
        let values: Vec<(&str, &Value, ValueOrigin)> = called
            .values
            .iter()
            .map(|value| (value.name.as_str(), &value.value, value.origin))
            .collect();
        assert_eq!(
            values,
            [
                ("language", &json!("en"), ValueOrigin::Default),
                ("max_line_length", &json!(37), ValueOrigin::Entry),
            ]
        );
        assert!(called.notes.is_empty());
    }

    #[test]
    fn a_higher_limit_is_lowered_and_noted_and_no_model_means_the_callers() {
        let caller = agent(
            "video-producer",
            json!({
                "model": {"source": "ollama::large"},
                "limits": {"max_turns": 60},
                "agents": {"reviewer": {"source": "./agents/reviewer"}}
            }),
        );
        let reviewer = agent("reviewer", json!({"limits": {"max_turns": 80}}));
        let called = check_called_agent(&caller, "reviewer", &reviewer, 1).expect("called agent");
        assert_eq!(Value::Object(called.limits), json!({"max_turns": 60}));
        assert_eq!(called.model_source.as_deref(), Some("ollama::large"));
        assert_eq!(called.model_from, "caller");
        assert_eq!(called.notes.len(), 1);
        assert_eq!(called.notes[0].code, NoteCode::LimitLowered);
        assert_eq!(
            called.notes[0].text,
            "limits.max_turns: reviewer asked for 80, the caller video-producer allows 60, 60 applies"
        );

        // Neither names a model: the default is the caller's, and the caller has none to give.
        let plain = agent(
            "lead",
            json!({"agents": {"reviewer": {"source": "./agents/reviewer"}}}),
        );
        let called = check_called_agent(&plain, "reviewer", &reviewer, 1).expect("called agent");
        assert_eq!(called.model_source, None);
        assert_eq!(called.model_from, "caller");
    }

    #[test]
    fn what_is_not_an_agent_or_has_no_description_is_refused() {
        let caller = agent(
            "lead",
            json!({"agents": {"helper": {"source": "./helper"}}}),
        );
        let definition = json!({"type": "agent-definition", "name": "helper", "agent": {"source": "./agent.json"}});
        let error =
            check_called_agent(&caller, "helper", &definition, 1).expect_err("a definition");
        assert_eq!(error.code, PackageErrorCode::ReferenceType);
        assert_eq!(error.subject, "$/agents/helper");

        let mut silent = agent("helper", json!({}));
        silent
            .as_object_mut()
            .expect("object")
            .remove("description");
        let error = check_called_agent(&caller, "helper", &silent, 1).expect_err("no description");
        assert_eq!(error.code, PackageErrorCode::CalledAgentDescription);
    }

    #[test]
    fn a_value_the_called_agent_does_not_declare_is_refused() {
        let caller = agent(
            "lead",
            json!({"agents": {"helper": {"source": "./helper", "variables": {"tone": "kind"}}}}),
        );
        let error = check_called_agent(&caller, "helper", &agent("helper", json!({})), 1)
            .expect_err("undeclared");
        assert_eq!(error.code, PackageErrorCode::VariableUndeclared);
        assert_eq!(error.subject, "tone");

        // A variable of the called agent with no value and no default stops it.
        let needy = agent("helper", json!({"variables": {"job": {"type": "string"}}}));
        let plain = agent(
            "lead",
            json!({"agents": {"helper": {"source": "./helper"}}}),
        );
        let error = check_called_agent(&plain, "helper", &needy, 1).expect_err("no value");
        assert_eq!(error.code, PackageErrorCode::VariableMissing);
    }

    #[test]
    fn a_caller_with_its_values_is_not_started_again() {
        // The caller declares a variable with no default. Its value is already written in.
        let caller = agent(
            "lead",
            json!({
                "variables": {"job": {"type": "string"}},
                "instructions": "Draft release notes.",
                "agents": {"helper": {"source": "./helper"}}
            }),
        );
        let helper = agent("helper", json!({}));
        called_under(&caller, "helper", &helper, 1).expect("the caller has its values");
        let error = check_called_agent(&caller, "helper", &helper, 1).expect_err("as written");
        assert_eq!(error.code, PackageErrorCode::VariableMissing);
    }
}
