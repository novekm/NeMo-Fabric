// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A definition read together with the agent it points at.
//!
//! A definition adds and tightens. It never replaces: the agent always wins.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use super::document::{missing_connectors, same_json, tool_name};
use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::model::{
    InstructionPart, Note, NoteCode, StatedBy, TogetherSummary, ValueOrigin, VariableValue,
};
use super::variables::resolve_variables;

/// The five limits, in the order the format lists them.
pub(crate) const LIMITS: [&str; 5] = [
    "max_turns",
    "timeout_seconds",
    "max_parallel",
    "max_depth",
    "max_total_tokens",
];

/// The named maps a definition adds entries to, in the order they are read.
pub(crate) const ADDED_MAPS: [&str; 9] = [
    "skills",
    "plugins",
    "connectors",
    "hooks",
    "packages",
    "agents",
    "policies",
    "custom_tools",
    "credentials",
];

/// A definition and its agent, read together.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Together {
    /// The agent as it runs.
    pub agent: Value,
    pub values: Vec<VariableValue>,
    pub summary: TogetherSummary,
    /// The numbers that apply.
    pub limits: Map<String, Value>,
    pub notes: Vec<Note>,
}

fn name_of(document: &Value) -> &str {
    document.get("name").and_then(Value::as_str).unwrap_or("")
}

fn map_of<'a>(document: &'a Value, field: &str) -> Option<&'a Map<String, Value>> {
    document.get(field).and_then(Value::as_object)
}

/// Whether a definition and the agent it points at are stored in one folder, read from the pointer alone.
///
/// That is so when the pointer is local and names a file beside the definition. A pointer that
/// names a folder leads to the `agent.json` at that folder's root, which is another folder.
/// The package reader knows where both documents are stored, so only the tests need this.
#[cfg(test)]
pub(crate) fn stored_beside(definition: &Value) -> bool {
    let Some(relative) = definition
        .pointer("/agent/source")
        .and_then(Value::as_str)
        .and_then(|source| source.strip_prefix("./"))
    else {
        return false;
    };
    let mut segments: Vec<&str> = Vec::new();
    for segment in relative.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                if segments.last().is_some_and(|last| *last != "..") {
                    segments.pop();
                } else {
                    segments.push("..");
                }
            }
            name => segments.push(name),
        }
    }
    match segments.as_slice() {
        [] => true,
        [file] => file.ends_with(".json"),
        _ => false,
    }
}

/// Why a definition cannot be read together with its agent.
///
/// An entry whose name the agent already uses under the same field is refused, naming it,
/// unless the two entries are identical: the same JSON value and, for a local source, the same
/// stored folder. A connector tool the definition adds names a connector either document has.
///
/// `one_folder`: whether both documents are stored in one folder.
pub(crate) fn check_together(
    definition: &Value,
    agent: &Value,
    one_folder: bool,
) -> PackageResult<()> {
    let agent_name = name_of(agent);
    for field in ADDED_MAPS {
        let (Some(added), Some(existing)) = (map_of(definition, field), map_of(agent, field))
        else {
            continue;
        };
        for (name, entry) in added {
            let Some(theirs) = existing.get(name) else {
                continue;
            };
            let local = entry
                .get("source")
                .and_then(Value::as_str)
                .is_some_and(|source| source.starts_with("./"));
            let why = if !same_json(theirs, entry) {
                "the two entries are not identical"
            } else if !one_folder && local {
                "the same local path leads to another folder there"
            } else {
                continue;
            };
            return Err(PackageError::new(
                PackageErrorCode::DefinitionNameInUse,
                format!("{field}.{name}"),
                format!(
                    "{field}.{name}: {agent_name} already has an entry of that name, and {why}"
                ),
            ));
        }
    }
    if let Some(tool) = missing_connectors(
        definition.get("tools"),
        &[agent.get("connectors"), definition.get("connectors")],
    )
    .into_iter()
    .next()
    {
        return Err(PackageError::new(
            PackageErrorCode::ToolConnectorUnknown,
            tool.clone(),
            format!(
                "tools: {tool} names a connector that neither {agent_name} nor the definition has"
            ),
        ));
    }
    Ok(())
}

/// Read a definition together with its agent. `one_folder`: whether both documents are stored in one folder.
pub(crate) fn read_together(
    definition: &Value,
    agent: &Value,
    given: &Map<String, Value>,
    one_folder: bool,
) -> PackageResult<Together> {
    if agent.get("type").and_then(Value::as_str) != Some("agent") {
        return Err(PackageError::new(
            PackageErrorCode::ReferenceType,
            "$/agent",
            format!(
                "agent: it resolves to type {}, not to an agent",
                agent.get("type").and_then(Value::as_str).unwrap_or("none")
            ),
        ));
    }
    let resolved = resolve_variables(
        agent,
        given,
        map_of(definition, "variables"),
        ValueOrigin::Definition,
    )?;
    check_together(definition, agent, one_folder)?;

    let definition_name = name_of(definition);
    let agent_name = name_of(agent);
    let mut together = resolved.agent;
    let mut summary = TogetherSummary::default();
    let mut notes = Vec::new();
    for field in ADDED_MAPS {
        let Some(added) = map_of(definition, field) else {
            continue;
        };
        for (name, entry) in added {
            let listed = format!("{field}.{name}");
            if together
                .get(field)
                .is_some_and(|map| map.get(name).is_some())
            {
                notes.push(Note {
                    code: NoteCode::EntryKeptOnce,
                    subject: listed.clone(),
                    text: format!(
                        "{listed}: the definition {definition_name} and the agent {agent_name} hold the same entry; it counts once"
                    ),
                });
                summary.kept_once.push(listed);
                continue;
            }
            if let Some(object) = together.as_object_mut() {
                let map = object
                    .entry(field)
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Some(map) = map.as_object_mut() {
                    map.insert(name.clone(), entry.clone());
                }
            }
            summary.added.push(listed);
        }
    }
    if let Some(tools) = definition.get("tools").and_then(Value::as_array) {
        let mut listed: BTreeSet<String> = together
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(tool_name)
            .collect();
        for item in tools {
            let name = tool_name(item);
            if !listed.insert(name.clone()) {
                notes.push(Note {
                    code: NoteCode::EntryKeptOnce,
                    subject: format!("tools.{name}"),
                    text: format!("tools.{name}: the tool is already in the list; it is kept once"),
                });
                summary.kept_once.push(format!("tools.{name}"));
                continue;
            }
            if let Some(object) = together.as_object_mut() {
                let list = object
                    .entry("tools")
                    .or_insert_with(|| Value::Array(Vec::new()));
                if let Some(list) = list.as_array_mut() {
                    list.push(item.clone());
                }
            }
            summary.added.push(format!("tools.{name}"));
        }
    }
    if let Some(asked) = map_of(definition, "limits") {
        let limits = smallest_limits(
            map_of(&together, "limits"),
            Some(asked),
            &format!("the definition {definition_name}"),
            &format!("the agent {agent_name}"),
            0,
            &mut notes,
        );
        together["limits"] = Value::Object(limits);
    }
    // The agent's text stays and comes first. The format does not say what sits between the
    // two texts: this loader puts one empty line.
    for (from, document) in [
        (StatedBy::Agent, &together),
        (StatedBy::Definition, definition),
    ] {
        if let Some(text) = document.get("instructions").and_then(Value::as_str) {
            summary.instructions.push(InstructionPart {
                from,
                name: name_of(document).to_string(),
                text: text.to_string(),
            });
        }
    }
    if definition.get("instructions").is_some() {
        let text: Vec<&str> = summary
            .instructions
            .iter()
            .map(|part| part.text.as_str())
            .collect();
        together["instructions"] = Value::String(text.join("\n\n"));
    }
    let limits = map_of(&together, "limits").cloned().unwrap_or_default();
    Ok(Together {
        agent: together,
        values: resolved.values,
        summary,
        limits,
        notes,
    })
}

fn number(value: Option<&Value>) -> Option<f64> {
    value.and_then(Value::as_f64)
}

/// For each limit the smaller number. A higher asked number is lowered and noted, never refused.
///
/// `depth` is how many calls below the first agent the asker runs. `max_depth` counts levels
/// of called agents from that first agent, so a called agent has what is left.
pub(crate) fn smallest_limits(
    cap: Option<&Map<String, Value>>,
    asked: Option<&Map<String, Value>>,
    asker: &str,
    owner: &str,
    depth: u64,
    notes: &mut Vec<Note>,
) -> Map<String, Value> {
    let mut out = Map::new();
    for key in LIMITS {
        let mut allowed = cap.and_then(|cap| cap.get(key)).cloned();
        let wanted = asked.and_then(|asked| asked.get(key));
        if key == "max_depth"
            && depth > 0
            && let Some(levels) = number(allowed.as_ref())
        {
            let left = (levels - depth as f64).max(0.0);
            allowed = Some(match allowed.as_ref().and_then(Value::as_u64) {
                Some(levels) => Value::from(levels.saturating_sub(depth)),
                None => Value::from(left),
            });
        }
        match (number(allowed.as_ref()), number(wanted)) {
            (None, None) => {}
            (None, Some(_)) => {
                out.insert(key.to_string(), wanted.cloned().unwrap_or(Value::Null));
            }
            (Some(allowed_number), Some(wanted_number)) if wanted_number <= allowed_number => {
                out.insert(key.to_string(), wanted.cloned().unwrap_or(Value::Null));
            }
            (Some(_), wanted_number) => {
                let applies = allowed.unwrap_or(Value::Null);
                if wanted_number.is_some() {
                    let wanted = wanted.cloned().unwrap_or(Value::Null);
                    notes.push(Note {
                        code: NoteCode::LimitLowered,
                        subject: format!("limits.{key}"),
                        text: format!(
                            "limits.{key}: {asker} asked for {wanted}, {owner} allows {applies}, {applies} applies"
                        ),
                    });
                }
                out.insert(key.to_string(), applies);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::schemas::{SchemaKind, schema_id};
    use super::*;

    fn agent(more: Value) -> Value {
        let mut document = json!({
            "$schema": schema_id(SchemaKind::Agent),
            "type": "agent",
            "name": "team-assistant"
        });
        document
            .as_object_mut()
            .expect("object")
            .extend(more.as_object().expect("object").clone());
        document
    }

    fn definition(more: Value) -> Value {
        let mut document = json!({
            "$schema": schema_id(SchemaKind::AgentDefinition),
            "type": "agent-definition",
            "name": "ada-release-notes",
            "agent": {"source": "./agent.json"}
        });
        document
            .as_object_mut()
            .expect("object")
            .extend(more.as_object().expect("object").clone());
        document
    }

    fn together(definition: &Value, agent: &Value) -> PackageResult<Together> {
        read_together(definition, agent, &Map::new(), stored_beside(definition))
    }

    #[test]
    fn a_definition_gives_values_adds_parts_and_tools() {
        let base = agent(json!({
            "instructions": {"var": "job"},
            "variables": {"job": {"type": "string"}},
            "skills": {"review": {"source": "./skills/review"}},
            "connectors": {"queue": {"source": "./connectors/queue"}},
            "tools": [{"connector": "queue", "tool": "render"}, {"builtin": "read_file"}]
        }));
        let result = together(
            &definition(json!({
                "variables": {"job": "Draft release notes."},
                "skills": {"glossary": {"source": "./skills/glossary"}},
                "connectors": {"units": {"source": "./connectors/units"}},
                "custom_tools": {"ask": {"description": "Ask.", "input_schema": {"type": "object"}}},
                "tools": [
                    {"connector": "units", "tool": "convert"},
                    {"builtin": "read_file"},
                    {"connector": "queue", "tool": "render"}
                ]
            })),
            &base,
        )
        .expect("read together");
        assert_eq!(
            result.summary.added,
            [
                "skills.glossary",
                "connectors.units",
                "custom_tools.ask",
                "tools.units.convert"
            ]
        );
        assert_eq!(
            result.summary.kept_once,
            ["tools.builtin.read_file", "tools.queue.render"]
        );
        assert_eq!(result.agent["instructions"], "Draft release notes.");
        assert_eq!(
            result.agent["skills"]["glossary"]["source"],
            "./skills/glossary"
        );
        assert_eq!(
            result.agent["skills"]["review"]["source"],
            "./skills/review"
        );
        assert_eq!(
            result.agent["tools"],
            json!([
                {"connector": "queue", "tool": "render"},
                {"builtin": "read_file"},
                {"connector": "units", "tool": "convert"}
            ])
        );
        assert_eq!(result.values[0].origin, ValueOrigin::Definition);
        assert_eq!(result.notes.len(), 2);
        assert!(
            result
                .notes
                .iter()
                .all(|note| note.code == NoteCode::EntryKeptOnce)
        );
    }

    #[test]
    fn the_start_wins_over_the_definition() {
        let base = agent(json!({
            "model": {"source": {"var": "model"}},
            "variables": {"model": {"type": "string", "default": "ollama::small"}}
        }));
        let result = read_together(
            &definition(json!({"variables": {"model": "ollama::large"}})),
            &base,
            json!({"model": "ollama::other"})
                .as_object()
                .expect("object"),
            false,
        )
        .expect("read together");
        assert_eq!(result.agent["model"]["source"], "ollama::other");
        assert_eq!(result.values[0].origin, ValueOrigin::Start);
    }

    #[test]
    fn an_identical_entry_counts_once_and_another_under_the_same_name_is_refused() {
        let base = agent(json!({
            "skills": {"review": {"source": "git::https://git.example.com/parts.git//review", "version": "1.0"}}
        }));
        let same = definition(json!({
            "skills": {"review": {"source": "git::https://git.example.com/parts.git//review", "version": "1.0"}}
        }));
        let result = together(&same, &base).expect("identical entries");
        assert_eq!(result.summary.kept_once, ["skills.review"]);
        assert!(result.summary.added.is_empty());

        let other = definition(json!({
            "skills": {"review": {"source": "git::https://git.example.com/parts.git//review", "version": "2.0"}}
        }));
        let error = together(&other, &base).expect_err("another version under the same name");
        assert_eq!(error.code, PackageErrorCode::DefinitionNameInUse);
        assert_eq!(error.subject, "skills.review");
    }

    #[test]
    fn the_same_local_path_is_identical_only_in_one_folder() {
        let base = agent(json!({"skills": {"review": {"source": "./skills/review"}}}));
        let beside = definition(json!({"skills": {"review": {"source": "./skills/review"}}}));
        together(&beside, &base).expect("both documents are in one folder");
        // The pointer names a folder, so the agent is stored in another folder.
        let mut apart = beside.clone();
        apart["agent"]["source"] = json!("./team-assistant");
        let error = together(&apart, &base).expect_err("two folders");
        assert_eq!(error.code, PackageErrorCode::DefinitionNameInUse);
        // The caller knows where the two documents are stored.
        read_together(&apart, &base, &Map::new(), true).expect("told it is one folder");
        let error = read_together(&beside, &base, &Map::new(), false).expect_err("told it is two");
        assert_eq!(error.code, PackageErrorCode::DefinitionNameInUse);
    }

    #[test]
    fn where_the_two_documents_are_stored_is_read_from_the_pointer() {
        let pointer = |source: &str| definition(json!({"agent": {"source": source}}));
        assert!(stored_beside(&pointer("./agent.json")));
        assert!(stored_beside(&pointer("./x/../agent.json")));
        assert!(!stored_beside(&pointer("./team-assistant")));
        assert!(!stored_beside(&pointer("./../../agent.json")));
        assert!(!stored_beside(&pointer("./agents/agent.json")));
        assert!(!stored_beside(&json!({
            "agent": {"source": "oci::registry.example.com/agent", "version": "1.0"}
        })));
    }

    #[test]
    fn a_tool_of_a_connector_neither_document_has_is_refused() {
        let error = together(
            &definition(json!({"tools": [{"connector": "queue", "tool": "render"}]})),
            &agent(json!({})),
        )
        .expect_err("no such connector");
        assert_eq!(error.code, PackageErrorCode::ToolConnectorUnknown);
        assert_eq!(error.subject, "queue.render");
    }

    #[test]
    fn for_each_limit_the_smaller_number_applies() {
        let base = agent(
            json!({"limits": {"max_turns": 60, "timeout_seconds": 3600, "max_total_tokens": 600000}}),
        );
        let result = together(
            &definition(
                json!({"limits": {"max_turns": 25, "timeout_seconds": 7200.5, "max_parallel": 2}}),
            ),
            &base,
        )
        .expect("read together");
        assert_eq!(
            Value::Object(result.limits.clone()),
            json!({"max_turns": 25, "timeout_seconds": 3600, "max_parallel": 2, "max_total_tokens": 600000})
        );
        assert_eq!(result.agent["limits"], Value::Object(result.limits));
        // The larger number is not an error: the agent's number applies and it is reported.
        assert_eq!(result.notes.len(), 1);
        assert_eq!(result.notes[0].code, NoteCode::LimitLowered);
        assert_eq!(result.notes[0].subject, "limits.timeout_seconds");
        assert_eq!(
            result.notes[0].text,
            "limits.timeout_seconds: the definition ada-release-notes asked for 7200.5, the agent team-assistant allows 3600, 3600 applies"
        );
    }

    #[test]
    fn a_called_agent_has_the_depth_that_is_left() {
        let mut notes = Vec::new();
        let cap = json!({"max_depth": 1, "max_turns": 60});
        let asked = json!({"max_depth": 3, "max_turns": 80});
        let limits = smallest_limits(
            cap.as_object(),
            asked.as_object(),
            "helper",
            "the caller lead",
            1,
            &mut notes,
        );
        assert_eq!(
            Value::Object(limits),
            json!({"max_depth": 0, "max_turns": 60})
        );
        assert_eq!(notes.len(), 2);
        let limits = smallest_limits(
            cap.as_object(),
            None,
            "helper",
            "the caller lead",
            5,
            &mut Vec::new(),
        );
        assert_eq!(
            Value::Object(limits),
            json!({"max_depth": 0, "max_turns": 60})
        );
        let limits = smallest_limits(
            None,
            asked.as_object(),
            "helper",
            "the caller lead",
            1,
            &mut Vec::new(),
        );
        assert_eq!(
            Value::Object(limits),
            json!({"max_depth": 3, "max_turns": 80})
        );
    }

    #[test]
    fn the_agent_text_comes_first() {
        let base = agent(json!({"instructions": "Report every item that fails."}));
        let result = together(
            &definition(json!({"instructions": "Write in German."})),
            &base,
        )
        .expect("read together");
        assert_eq!(
            result.agent["instructions"],
            "Report every item that fails.\n\nWrite in German."
        );
        assert_eq!(
            result.summary.instructions,
            [
                InstructionPart {
                    from: StatedBy::Agent,
                    name: "team-assistant".to_string(),
                    text: "Report every item that fails.".to_string()
                },
                InstructionPart {
                    from: StatedBy::Definition,
                    name: "ada-release-notes".to_string(),
                    text: "Write in German.".to_string()
                },
            ]
        );
        // With no text in the agent, the definition's text is the whole text.
        let result = together(
            &definition(json!({"instructions": "Write in German."})),
            &agent(json!({})),
        )
        .expect("read together");
        assert_eq!(result.agent["instructions"], "Write in German.");
        assert_eq!(result.summary.instructions.len(), 1);
    }

    #[test]
    fn a_pointer_that_is_not_an_agent_is_refused() {
        let error =
            together(&definition(json!({})), &definition(json!({}))).expect_err("a definition");
        assert_eq!(error.code, PackageErrorCode::ReferenceType);
    }

    #[test]
    fn the_limits_are_the_ones_the_schema_lists() {
        let schema: Value =
            serde_json::from_str(include_str!("../../assets/oadl/0.0.1/agent.schema.json"))
                .expect("schema is JSON");
        let listed: BTreeSet<&str> = schema["$defs"]["limits"]["properties"]
            .as_object()
            .expect("limits")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(listed, LIMITS.into_iter().collect());
    }
}
