// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bindings: the caller's own targets for custom tools and called agents.
//!
//! A binding is data. Nothing here imports, resolves or runs a target. This
//! module reads the keys and checks that each one names a thing the package
//! declares. The mapping gives the bound thing its line in the report.

use std::collections::BTreeMap;

use serde_json::Value;

use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::model::{CalledAgent, PackageReading};

/// The field of custom tools, in a document and in a binding key.
pub(super) const CUSTOM_TOOLS: &str = "custom_tools";
/// The field of called agents, in a document and in a binding key.
pub(super) const AGENTS: &str = "agents";
pub(super) const LIMITS: &str = "limits";
/// The limits a binding can name: the ones NeMo Fabric cannot hold itself, which bound the agent and every agent
/// it calls. The binding names who enforces each, so a load does not stop for them and they are not dropped.
pub(super) const BINDABLE_LIMITS: [&str; 3] = ["max_depth", "max_parallel", "max_total_tokens"];

/// The longest key and the longest target, in bytes.
const MAX_BYTES: usize = 1024;

/// Whether a binding names a called agent or a thing below one. A thing below
/// a called agent needs that agent's binding, so one such key says the caller
/// runs a called agent.
pub(super) fn binds_called_agent(bindings: &BTreeMap<String, String>) -> bool {
    bindings.keys().any(|key| key.starts_with("agents."))
}

/// Whether text stays on one line and shows all of itself: no control
/// character, no line or paragraph separator, and no format character that
/// hides text or changes its direction.
fn is_one_plain_line(text: &str) -> bool {
    !text.chars().any(|character| {
        character.is_control()
            || matches!(
                character,
                '\u{00AD}'
                    | '\u{061C}'
                    | '\u{180E}'
                    | '\u{200B}'..='\u{200F}'
                    | '\u{2028}'..='\u{202E}'
                    | '\u{2060}'..='\u{206F}'
                    | '\u{FEFF}'
                    | '\u{FFF9}'..='\u{FFFB}'
                    | '\u{E0000}'..='\u{E007F}'
            )
    })
}

/// The key of one thing: `<field>.<name>`, after `agents.<name>.` for each
/// called agent in `place` that leads to it.
pub(super) fn binding_key(place: &[String], field: &str, name: &str) -> String {
    let mut key = String::new();
    for agent in place {
        key.push_str(AGENTS);
        key.push('.');
        key.push_str(agent);
        key.push('.');
    }
    key.push_str(field);
    key.push('.');
    key.push_str(name);
    key
}

/// What a key names: a custom tool or a called agent, and the called agents
/// that lead to it, outermost first.
struct Named<'a> {
    place: Vec<&'a str>,
    field: &'static str,
    name: &'a str,
}

/// Reads a key. An agent's name holds no dot. A tool's name may, so
/// `custom_tools.` takes the rest of the key.
fn parse_key(key: &str) -> Option<Named<'_>> {
    let mut place = Vec::new();
    let mut rest = key;
    if let Some(name) = key.strip_prefix("limits.") {
        return BINDABLE_LIMITS.contains(&name).then_some(Named {
            place,
            field: LIMITS,
            name,
        });
    }
    loop {
        if let Some(name) = rest.strip_prefix("custom_tools.") {
            return (!name.is_empty()).then_some(Named {
                place,
                field: CUSTOM_TOOLS,
                name,
            });
        }
        let after = rest.strip_prefix("agents.")?;
        let Some((agent, below)) = after.split_once('.') else {
            return (!after.is_empty()).then_some(Named {
                place,
                field: AGENTS,
                name: after,
            });
        };
        if agent.is_empty() {
            return None;
        }
        place.push(agent);
        rest = below;
    }
}

/// Check every binding before anything is mapped.
///
/// A key and a target are each one plain line of text, the key is read, and
/// it names a custom tool or a called agent the package declares. What a
/// called agent declares is evaluated only when the caller runs that agent,
/// so a key below `agents.<name>` needs a binding for `agents.<name>` too.
pub(super) fn check_bindings(
    reading: &PackageReading,
    bindings: &BTreeMap<String, String>,
) -> PackageResult<()> {
    for (key, target) in bindings {
        // A key is held to the rule of a target before any message repeats it.
        if key.len() > MAX_BYTES || !is_one_plain_line(key) {
            return Err(PackageError::new(
                PackageErrorCode::OptionsInvalid,
                "bindings",
                format!(
                    "a binding key is one line of text, at most {MAX_BYTES} bytes, with no control, separator or hidden format character"
                ),
            ));
        }
        let subject = format!("bindings.{key}");
        let invalid = |message: String| {
            PackageError::new(PackageErrorCode::OptionsInvalid, &subject, message)
        };
        let undeclared = |message: String| {
            PackageError::new(PackageErrorCode::BindingUndeclared, &subject, message)
        };
        if target.trim().is_empty() || target.len() > MAX_BYTES || !is_one_plain_line(target) {
            return Err(invalid(format!(
                "a binding target is one line of text, from 1 to {MAX_BYTES} bytes, with no control, separator or hidden format character"
            )));
        }
        let Some(named) = parse_key(key) else {
            return Err(invalid(
                "a binding key is `custom_tools.<name>` or `agents.<name>`, alone or after the key of a called agent, or `limits.max_depth`, `limits.max_parallel` or `limits.max_total_tokens`. A binding applies to custom tools, called agents and those limits only"
                    .to_string(),
            ));
        };

        // Walk down the called agents that lead to the thing.
        let mut place: Vec<String> = Vec::new();
        let mut called: Option<&CalledAgent> = None;
        for agent in &named.place {
            let agent_key = binding_key(&place, AGENTS, agent);
            if !declares(reading, called, AGENTS, agent) {
                return Err(undeclared(format!(
                    "the package declares no called agent at `{agent_key}`"
                )));
            }
            if !bindings.contains_key(&agent_key) {
                return Err(invalid(format!(
                    "a binding below `{agent_key}` needs a binding for `{agent_key}`: what a called agent declares is evaluated only when the caller runs that agent"
                )));
            }
            place.push((*agent).to_string());
            called = reading.called.iter().find(|called| called.place == place);
            if called.is_none() {
                return Err(undeclared(format!(
                    "the package holds no document for `{agent_key}` that the loader could read and start, so it declares nothing below it"
                )));
            }
        }
        if !declares(reading, called, named.field, named.name) {
            let kind = match named.field {
                AGENTS => "called agent",
                LIMITS => "limit",
                _ => "custom tool",
            };
            return Err(undeclared(format!(
                "the package declares no {kind} at `{key}`"
            )));
        }
    }
    Ok(())
}

/// Whether `field` holds `name`: in the agent that runs, or in one called agent.
fn declares(
    reading: &PackageReading,
    called: Option<&CalledAgent>,
    field: &str,
    name: &str,
) -> bool {
    match called {
        None => reading
            .agent
            .get(field)
            .and_then(Value::as_object)
            .is_some_and(|entries| entries.contains_key(name)),
        Some(called) if field == AGENTS => called.agents.iter().any(|listed| listed.name == name),
        Some(called) => called.custom_tools.iter().any(|tool| tool == name),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::super::load::load_agent_package;
    use super::super::model::{NoteCode, PackageAction, StatedBy};
    use super::super::options::AgentPackageOptions;
    use super::super::reader::testing::{MADE_UP, Package, SKILL, agent, definition, git_entry};
    use super::super::report::{
        AgentPackageReport, DeclaredItem, DeclaredKind, DeclaredStatus, ItemEffect, MappingGap,
        PackageOutcome,
    };
    use super::*;
    use crate::config::{FabricConfig, ResolveContext};
    use crate::error::FabricError;

    fn named(key: &str) -> Option<(Vec<&str>, &str, &str)> {
        parse_key(key).map(|named| (named.place, named.field, named.name))
    }

    #[test]
    fn a_key_names_a_custom_tool_or_a_called_agent_at_any_depth() {
        // (key, the called agents that lead to the thing, field, name)
        let table = [
            (
                "custom_tools.request_approval",
                vec![],
                CUSTOM_TOOLS,
                "request_approval",
            ),
            ("agents.captioner", vec![], AGENTS, "captioner"),
            (
                "agents.captioner.custom_tools.spell_check",
                vec!["captioner"],
                CUSTOM_TOOLS,
                "spell_check",
            ),
            (
                "agents.captioner.agents.translator",
                vec!["captioner"],
                AGENTS,
                "translator",
            ),
            (
                "agents.a.agents.b.custom_tools.t",
                vec!["a", "b"],
                CUSTOM_TOOLS,
                "t",
            ),
            // A tool name may hold dots, and may look like a key.
            (
                "custom_tools.search.web",
                vec![],
                CUSTOM_TOOLS,
                "search.web",
            ),
            (
                "agents.a.custom_tools.agents.b",
                vec!["a"],
                CUSTOM_TOOLS,
                "agents.b",
            ),
        ];
        for (key, place, field, name) in table {
            assert_eq!(named(key), Some((place, field, name)), "{key}");
        }
    }

    #[test]
    fn a_key_for_anything_else_is_not_read() {
        for key in [
            "",
            "custom_tools",
            "custom_tools.",
            "agents",
            "agents.",
            "agents..custom_tools.t",
            "agents.a.",
            "agents.a.b",
            "skills.review",
            "hooks.guard",
            "policies.licensed",
            "runtime.compute.memory",
            "agents.a.policies.licensed",
            "request_approval",
        ] {
            assert!(named(key).is_none(), "{key}");
        }
    }

    #[test]
    fn a_key_is_built_the_way_it_is_read() {
        let place = ["a".to_string(), "b".to_string()];

        assert_eq!(binding_key(&[], AGENTS, "a"), "agents.a");
        assert_eq!(
            binding_key(&place, CUSTOM_TOOLS, "search.web"),
            "agents.a.agents.b.custom_tools.search.web"
        );
        assert_eq!(
            named(&binding_key(&place, CUSTOM_TOOLS, "search.web")),
            Some((vec!["a", "b"], CUSTOM_TOOLS, "search.web"))
        );
    }

    // ---- a whole load, on packages written to disk ----

    const ROUTE: &str = "app.tools.lead:route_to_agent";
    const WRITER: &str = "app.agents:writer";

    /// The test adapter. It accepts models, mcp and skills.
    fn context() -> ResolveContext {
        ResolveContext::new(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/hermes-shim-agent"),
        )
    }

    fn config(name: &str) -> FabricConfig {
        serde_json::from_value(json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": name},
            "harness": {"adapter_id": "test.fabric.hermes_shim"},
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {},
        }))
        .expect("a config")
    }

    fn tool(description: &str) -> Value {
        json!({"description": description, "input_schema": {"type": "object"}})
    }

    fn document_entry(package: &Package, at: &str, path: &str, dependencies: &[&str]) -> Value {
        package.file_entry(at, "agent", path, &format!("./{path}"), dependencies)
    }

    /// A lead with one custom tool. It calls a writer and, optionally, a
    /// checker. The writer has two custom tools and calls a translator, and
    /// the translator has one custom tool.
    fn team(name: &str) -> Package {
        let package = Package::new(name);
        package.write_json(
            "agent.json",
            &agent(
                "lead",
                json!({
                    "custom_tools": {"route_to_agent": tool("Hand a task to a member.")},
                    "agents": {
                        "writer": {"source": "./agents/writer"},
                        "checker": {"source": "./agents/checker", "optional": true}
                    }
                }),
            ),
        );
        package.write_json(
            "agents/writer/agent.json",
            &agent(
                "writer",
                json!({
                    "custom_tools": {
                        "spell_check": tool("Check the spelling of a draft."),
                        "search.web": tool("Search the web.")
                    },
                    "agents": {"translator": {"source": "./agents/translator"}}
                }),
            ),
        );
        package.write_json(
            "agents/writer/agents/translator/agent.json",
            &agent(
                "translator",
                json!({"custom_tools": {"glossary": tool("Look a term up.")}}),
            ),
        );
        package.write_json(
            "agents/checker/agent.json",
            &agent(
                "checker",
                json!({"custom_tools": {"lint": tool("Lint a draft.")}}),
            ),
        );
        lock_team(&package);
        package
    }

    fn lock_team(package: &Package) {
        package.write_lock(
            "",
            vec![
                document_entry(
                    package,
                    "$",
                    "agent.json",
                    &["$/agents/checker", "$/agents/writer"],
                ),
                document_entry(
                    package,
                    "$/agents/checker",
                    "agents/checker/agent.json",
                    &[],
                ),
                document_entry(
                    package,
                    "$/agents/writer",
                    "agents/writer/agent.json",
                    &["$/agents/writer/agents/translator"],
                ),
                document_entry(
                    package,
                    "$/agents/writer/agents/translator",
                    "agents/writer/agents/translator/agent.json",
                    &[],
                ),
            ],
        );
    }

    /// A binding for every required thing of the team. The checker is optional
    /// and has none.
    fn team_bindings() -> BTreeMap<String, String> {
        [
            ("custom_tools.route_to_agent", ROUTE),
            ("agents.writer", WRITER),
            (
                "agents.writer.custom_tools.spell_check",
                "app.tools.writer:spell_check",
            ),
            (
                "agents.writer.custom_tools.search.web",
                "app.tools.writer:search_web",
            ),
            ("agents.writer.agents.translator", "app.agents:translator"),
            (
                "agents.writer.agents.translator.custom_tools.glossary",
                "app.tools.translator:glossary",
            ),
        ]
        .into_iter()
        .map(|(key, target)| (key.to_string(), target.to_string()))
        .collect()
    }

    fn with(bindings: BTreeMap<String, String>) -> AgentPackageOptions {
        AgentPackageOptions {
            bindings,
            ..AgentPackageOptions::default()
        }
    }

    fn one(key: &str, target: &str) -> BTreeMap<String, String> {
        BTreeMap::from([(key.to_string(), target.to_string())])
    }

    fn load(package: &Package, options: AgentPackageOptions) -> AgentPackageReport {
        load_agent_package(&package.root, Some(config("lead")), options, context())
            .expect("the package loads")
    }

    fn refusal(package: &Package, options: AgentPackageOptions) -> (String, String) {
        match load_agent_package(&package.root, Some(config("lead")), options, context()) {
            Err(FabricError::AgentPackage { code, subject, .. }) => (code, subject),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    fn item<'a>(report: &'a AgentPackageReport, id: &str) -> &'a DeclaredItem {
        report
            .declared
            .iter()
            .find(|item| item.id == id)
            .unwrap_or_else(|| panic!("no line {id}"))
    }

    /// The id of the line a binding key names, in a package whose entry is an agent.
    fn line_id(key: &str) -> String {
        let named = parse_key(key).expect("a key");
        let mut id = "agent:".to_string();
        for agent in named.place {
            id.push_str(&format!("/agents/{agent}"));
        }
        id.push_str(&format!("/{}/{}", named.field, named.name));
        id
    }

    #[test]
    fn default_settings_stop_on_a_custom_tool_and_a_called_agent_with_no_binding() {
        let package = team("unbound");

        let report = load(&package, AgentPackageOptions::default());

        assert_eq!(report.outcome, PackageOutcome::Stopped);
        assert!(report.config.is_none());
        assert_eq!(
            report.stopped_by,
            ["agent:/custom_tools/route_to_agent", "agent:/agents/writer"]
        );
        // A caller that gives no binding reads no word about bindings. The
        // test adapter declares no key for called agents.
        for (id, gap, detail) in [
            (
                "agent:/custom_tools/route_to_agent",
                MappingGap::NotBound,
                "the adapter does not accept tools.definitions, so the config cannot name the tool's code",
            ),
            (
                "agent:/agents/writer",
                MappingGap::AdapterDoesNotAccept,
                "the adapter's descriptor does not declare `nemo_fabric.agents.v1` under extension_schemas.agent_config",
            ),
        ] {
            let line = item(&report, id);
            assert_eq!(line.status, DeclaredStatus::NotMapped, "{id}");
            assert_eq!(line.gap, Some(gap), "{id}");
            assert_eq!(line.effect, ItemEffect::Stopped, "{id}");
            assert_eq!(line.bound_to, None, "{id}");
            assert_eq!(line.detail.as_deref(), Some(detail), "{id}");
        }
        // An optional called agent with no binding is reported and stops nothing.
        let checker = item(&report, "agent:/agents/checker");
        assert_eq!(checker.status, DeclaredStatus::NotMapped);
        assert!(!checker.required);
        assert_eq!(checker.effect, ItemEffect::None);
        // What a called agent declares is listed only when the caller runs it.
        assert!(
            report
                .declared
                .iter()
                .all(|item| item.stated_by != StatedBy::CalledAgent)
        );
    }

    #[test]
    fn bindings_account_for_custom_tools_and_called_agents_at_every_depth() {
        let package = team("bound");
        let bindings = team_bindings();
        let strict = AgentPackageOptions {
            if_unverified: PackageAction::Stop,
            ..with(bindings.clone())
        };

        let report = load(&package, strict);

        assert_eq!(report.outcome, PackageOutcome::Ready);
        assert!(report.stopped_by.is_empty());
        for (key, target) in &bindings {
            let line = item(&report, &line_id(key));
            assert_eq!(line.status, DeclaredStatus::Bound, "{key}");
            assert_eq!(line.bound_to.as_deref(), Some(target.as_str()), "{key}");
            assert_eq!(line.gap, None, "{key}");
            assert_eq!(line.effect, ItemEffect::None, "{key}");
            assert!(line.required, "{key}");
            assert!(line.mapped_to.is_empty(), "{key}");
        }
        // A line below a called agent names that agent's document.
        let glossary = item(
            &report,
            "agent:/agents/writer/agents/translator/custom_tools/glossary",
        );
        assert_eq!(glossary.stated_by, StatedBy::CalledAgent);
        assert_eq!(glossary.document, "translator");
        assert_eq!(glossary.field, "custom_tools");
        assert_eq!(glossary.name.as_deref(), Some("glossary"));
        let translator = item(&report, "agent:/agents/writer/agents/translator");
        assert_eq!(translator.document, "writer");
        assert_eq!(
            translator.address.as_deref(),
            Some("$/agents/writer/agents/translator")
        );
        // The lead's own lines keep the lead as their document.
        let writer = item(&report, "agent:/agents/writer");
        assert_eq!(writer.stated_by, StatedBy::Agent);
        assert_eq!(writer.document, "lead");
        // The optional checker has no binding: nothing below it is listed.
        assert_eq!(
            item(&report, "agent:/agents/checker").status,
            DeclaredStatus::NotMapped
        );
        assert!(
            report
                .declared
                .iter()
                .all(|item| !item.id.starts_with("agent:/agents/checker/"))
        );
        // Every called agent was still read and checked against the lock.
        assert!(
            report
                .digests
                .iter()
                .all(|check| check.result == super::super::model::Integrity::Verified)
        );

        // A binding writes nothing into the config: the config is the one a
        // load with no binding gives when it is told to continue.
        let warned = load(
            &package,
            AgentPackageOptions {
                if_unsupported: PackageAction::Warn,
                ..AgentPackageOptions::default()
            },
        );
        assert_eq!(warned.outcome, PackageOutcome::Ready);
        assert_eq!(report.config, warned.config);
        assert_eq!(report.config_sha256, warned.config_sha256);
    }

    #[test]
    fn removing_a_required_binding_stops_the_load() {
        let package = team("one-removed");
        let all = team_bindings();
        for removed in all.keys() {
            // A binding below a called agent goes when that agent's binding goes.
            let below = format!("{removed}.");
            let bindings: BTreeMap<String, String> = all
                .iter()
                .filter(|(key, _)| *key != removed && !key.starts_with(&below))
                .map(|(key, target)| (key.clone(), target.clone()))
                .collect();

            let report = load(&package, with(bindings));

            assert_eq!(report.outcome, PackageOutcome::Stopped, "{removed}");
            assert_eq!(report.stopped_by, [line_id(removed)], "{removed}");
            assert!(report.config.is_none(), "{removed}");
            let line = item(&report, &line_id(removed));
            assert_eq!(line.status, DeclaredStatus::NotMapped, "{removed}");
            assert_eq!(line.bound_to, None, "{removed}");
            // To a caller that gives bindings the line names the key to add.
            let detail = line.detail.as_deref().expect("a detail");
            assert!(
                detail.ends_with(&format!("gives a binding for `{removed}`")),
                "{detail}"
            );
        }
    }

    #[test]
    fn a_binding_that_names_nothing_the_package_declares_is_refused() {
        let package = team("undeclared");
        let undeclared = "package_binding_undeclared";
        let invalid = "package_options_invalid";
        // (key, the other bindings given with it, code)
        let table = [
            ("custom_tools.no_such_tool", vec![], undeclared),
            ("agents.nobody", vec![], undeclared),
            // A tool of the writer is not a tool of the lead.
            ("custom_tools.spell_check", vec![], undeclared),
            (
                "agents.writer.custom_tools.no_such_tool",
                vec!["agents.writer"],
                undeclared,
            ),
            (
                "agents.writer.agents.nobody",
                vec!["agents.writer"],
                undeclared,
            ),
            ("agents.nobody.custom_tools.lint", vec![], undeclared),
            // What a called agent declares is evaluated only when the caller runs it.
            ("agents.writer.custom_tools.spell_check", vec![], invalid),
            (
                "agents.writer.agents.translator.custom_tools.glossary",
                vec!["agents.writer"],
                invalid,
            ),
            // A binding applies to custom tools and called agents only.
            ("skills.review", vec![], invalid),
            ("policies.licensed", vec![], invalid),
            ("runtime.software.python3", vec![], invalid),
            ("agents.writer.hooks.guard", vec!["agents.writer"], invalid),
            ("route_to_agent", vec![], invalid),
        ];
        for (key, others, code) in table {
            let mut bindings = one(key, "app:target");
            for other in others {
                bindings.insert(other.to_string(), "app:other".to_string());
            }

            let refused = refusal(&package, with(bindings.clone()));

            assert_eq!(
                refused,
                (code.to_string(), format!("bindings.{key}")),
                "{key}"
            );
            // The same holds when the package is only verified.
            match load_agent_package(&package.root, None, with(bindings), context()) {
                Err(FabricError::AgentPackage { code: found, .. }) => assert_eq!(found, code),
                other => panic!("expected a refusal for {key}, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_target_is_one_line_of_text() {
        let package = team("targets");
        let key = "custom_tools.route_to_agent";
        for target in [
            "".to_string(),
            "   ".to_string(),
            "app.tools:route\nimport os".to_string(),
            "app.tools:route\u{0}".to_string(),
            "app.tools:route\timport".to_string(),
            // A second line by a separator that is not a control character.
            "app.tools:route\u{2028}import os".to_string(),
            "app.tools:route\u{2029}import os".to_string(),
            "app.tools:route\u{85}import os".to_string(),
            // Text that hides itself or reverses what is shown.
            "\u{200B}".to_string(),
            "app.tools:\u{200B}route".to_string(),
            "app\u{202E}etuor:".to_string(),
            "app\u{2066}tools:route".to_string(),
            "\u{FEFF}app.tools:route".to_string(),
            "a".repeat(MAX_BYTES + 1),
        ] {
            assert_eq!(
                refusal(&package, with(one(key, &target))),
                (
                    "package_options_invalid".to_string(),
                    format!("bindings.{key}")
                ),
                "{target:?}"
            );
        }
        // Any other line of text is kept as the caller wrote it.
        for target in [
            "route",
            "app.tools.lead:route_to_agent",
            "registry://tools/route to agent",
            &"a".repeat(MAX_BYTES),
        ] {
            let report = load(&package, with(one(key, target)));

            assert_eq!(
                item(&report, "agent:/custom_tools/route_to_agent")
                    .bound_to
                    .as_deref(),
                Some(target)
            );
        }
    }

    #[test]
    fn a_key_is_checked_before_a_message_repeats_it() {
        let package = team("keys");
        for key in [
            "custom_tools.route\nto_agent".to_string(),
            "custom_tools.\u{1b}[31mroute_to_agent".to_string(),
            "custom_tools.route\u{2028}to_agent".to_string(),
            "agents.wri\u{202E}ter".to_string(),
            format!("custom_tools.{}", "a".repeat(MAX_BYTES)),
        ] {
            let refused = load_agent_package(
                &package.root,
                Some(config("lead")),
                with(one(&key, "app:target")),
                context(),
            );

            let Err(FabricError::AgentPackage {
                code,
                subject,
                message,
                ..
            }) = refused
            else {
                panic!("expected a refusal for {key:?}");
            };
            assert_eq!(code, "package_options_invalid", "{key:?}");
            // Neither the subject nor the message holds the key.
            assert_eq!(subject, "bindings", "{key:?}");
            assert!(!message.contains(&key), "{key:?}");
        }
        // The longest key that is read is refused for what it names, by name.
        let longest = format!("custom_tools.{}", "a".repeat(MAX_BYTES - 13));
        assert_eq!(longest.len(), MAX_BYTES);
        assert_eq!(
            refusal(&package, with(one(&longest, "app:target"))),
            (
                "package_binding_undeclared".to_string(),
                format!("bindings.{longest}")
            )
        );
    }

    /// A lead with one custom tool that calls a writer. The writer has one
    /// custom tool and declares one thing of every other kind.
    fn team_with_a_full_member(name: &str) -> Package {
        let package = Package::new(name);
        package.write_json(
            "agent.json",
            &agent(
                "lead",
                json!({
                    "custom_tools": {"route_to_agent": tool("Hand a task to a member.")},
                    "agents": {"writer": {"source": "./agents/writer"}}
                }),
            ),
        );
        package.write_json(
            "agents/writer/agent.json",
            &agent(
                "writer",
                json!({
                    "instructions": "Write the draft.",
                    "model": {
                        "source": "ollama::llama3.2:3b",
                        "parameters": {"temperature": 0.2},
                        "min_context_tokens": 32000
                    },
                    "limits": {"max_total_tokens": 100000, "max_turns": 8},
                    "skills": {
                        "review": {"source": "./skills/review"},
                        "style": {
                            "source": "git::https://git.example.com/parts.git//style",
                            "version": "~> 1.0"
                        },
                        "tone": {
                            "source": "git::https://git.example.com/parts.git//tone",
                            "version": "~> 1.0",
                            "optional": true
                        }
                    },
                    "connectors": {
                        "tickets": {"source": "mcp::https://tickets.example.com/mcp", "version": "2.3.0"}
                    },
                    "hooks": {"guard": {"source": "./hooks/guard"}},
                    "tools": [{"connector": "tickets", "tool": "delete_ticket"}],
                    "custom_tools": {"spell_check": tool("Check the spelling of a draft.")},
                    "policies": {"licensed": {"source": "./policies/licensed"}},
                    "credentials": {"tickets-token": {"description": "Reads tickets."}},
                    "runtime": {"compute": {"memory": "4096gb"}},
                    "outputs": {"draft": {"type": "string", "description": "The draft."}},
                    "extensions": {"dispatch": {"role": "member"}}
                }),
            ),
        );
        package.write("agents/writer/skills/review/SKILL.md", SKILL);
        package.write("agents/writer/hooks/guard/hooks.json", "{\"hooks\": {}}\n");
        package.write(
            "agents/writer/policies/licensed/policy.rego",
            "package licensed\n",
        );
        let writer = "$/agents/writer";
        let parts = [
            "$/agents/writer/connectors/tickets",
            "$/agents/writer/hooks/guard",
            "$/agents/writer/policies/licensed",
            "$/agents/writer/skills/review",
            "$/agents/writer/skills/style",
        ];
        package.write_lock(
            "",
            vec![
                document_entry(&package, "$", "agent.json", &[writer]),
                document_entry(&package, writer, "agents/writer/agent.json", &parts),
                json!({
                    "address": "$/agents/writer/connectors/tickets", "type": "connector",
                    "source": {"type": "pointer"}, "version": "2.3.0", "constraint": "2.3.0",
                    "interface": {"declared_version": "2.3.0", "tools_hash": MADE_UP},
                    "dependencies": []
                }),
                package.folder_entry(
                    "$/agents/writer/hooks/guard",
                    "hook",
                    "agents/writer/hooks/guard",
                    "./agents/writer/hooks/guard",
                ),
                package.folder_entry(
                    "$/agents/writer/policies/licensed",
                    "policy",
                    "agents/writer/policies/licensed",
                    "./agents/writer/policies/licensed",
                ),
                package.folder_entry(
                    "$/agents/writer/skills/review",
                    "skill",
                    "agents/writer/skills/review",
                    "./agents/writer/skills/review",
                ),
                git_entry(
                    "$/agents/writer/skills/style",
                    "skill",
                    "style",
                    "1.0.3",
                    "~> 1.0",
                    MADE_UP,
                ),
            ],
        );
        package
    }

    fn full_member_bindings() -> BTreeMap<String, String> {
        let mut bindings = one("custom_tools.route_to_agent", ROUTE);
        bindings.insert("agents.writer".to_string(), WRITER.to_string());
        bindings.insert(
            "agents.writer.custom_tools.spell_check".to_string(),
            "app.tools.writer:spell_check".to_string(),
        );
        bindings
    }

    #[test]
    fn a_bound_called_agent_keeps_a_line_for_everything_else_it_declares() {
        use DeclaredKind::{Field, Limit, ModelSetting, Need, Part, Tool};
        use DeclaredStatus::{Bound, CouldNotBeChecked, NotMapped, Skipped};

        let package = team_with_a_full_member("full-member");
        // The machine of the load has less memory than the writer needs. It is
        // a fact about where the lead starts, not about where the writer runs.
        let facts = json!({"memory": "8gb"}).as_object().cloned();
        let options = |if_unsupported, if_unverified| AgentPackageOptions {
            if_unsupported,
            if_unverified,
            facts: facts.clone(),
            ..with(full_member_bindings())
        };

        let report = load(&package, options(PackageAction::Stop, PackageAction::Stop));

        // (id below `agent:/agents/writer`, kind, required, status)
        let lines = [
            ("/model/source", ModelSetting, false, NotMapped),
            (
                "/model/parameters/temperature",
                ModelSetting,
                false,
                NotMapped,
            ),
            ("/limits/max_total_tokens", Limit, true, NotMapped),
            ("/limits/max_turns", Limit, true, NotMapped),
            ("/skills/review", Part, true, NotMapped),
            ("/skills/style", Part, true, NotMapped),
            ("/skills/tone", Part, false, NotMapped),
            ("/connectors/tickets", Part, true, NotMapped),
            ("/hooks/guard", Part, true, NotMapped),
            ("/tools/0", Tool, true, NotMapped),
            ("/custom_tools/spell_check", Part, true, Bound),
            ("/policies/licensed", Part, true, NotMapped),
            ("/credentials/tickets-token", Part, true, CouldNotBeChecked),
            ("/runtime/compute/memory", Need, true, CouldNotBeChecked),
            ("/model/min_context_tokens", Need, true, CouldNotBeChecked),
            ("/outputs/draft", Field, false, Skipped),
            ("/extensions/dispatch", Field, false, Skipped),
        ];
        let below: Vec<&DeclaredItem> = report
            .declared
            .iter()
            .filter(|item| item.id.starts_with("agent:/agents/writer/"))
            .collect();
        assert_eq!(
            below
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            lines
                .iter()
                .map(|(id, ..)| format!("agent:/agents/writer{id}"))
                .collect::<Vec<_>>()
        );
        for (line, (id, kind, required, status)) in below.iter().zip(lines) {
            assert_eq!(line.kind, kind, "{id}");
            assert_eq!(line.required, required, "{id}");
            assert_eq!(line.status, status, "{id}");
            assert_eq!(line.stated_by, StatedBy::CalledAgent, "{id}");
            assert_eq!(line.document, "writer", "{id}");
            assert!(line.mapped_to.is_empty(), "{id}");
            if status == NotMapped {
                assert_eq!(line.gap, Some(MappingGap::NoFabricField), "{id}");
            }
        }
        // The writer itself is bound, and that accounts for none of the rest:
        // every required line that is not bound stops a strict load.
        assert_eq!(item(&report, "agent:/agents/writer").status, Bound);
        assert_eq!(report.outcome, PackageOutcome::Stopped);
        assert!(report.config.is_none());
        let stops: Vec<String> = lines
            .iter()
            .filter(|(_, _, required, status)| *required && *status != Bound)
            .map(|(id, ..)| format!("agent:/agents/writer{id}"))
            .collect();
        assert_eq!(report.stopped_by, stops);
        // A part keeps the address of its lock entry, also one whose content is absent.
        for (id, address) in [
            (
                "agent:/agents/writer/skills/review",
                Some("$/agents/writer/skills/review"),
            ),
            (
                "agent:/agents/writer/skills/style",
                Some("$/agents/writer/skills/style"),
            ),
            ("agent:/agents/writer/skills/tone", None),
        ] {
            assert_eq!(item(&report, id).address.as_deref(), address, "{id}");
        }
        let tool = item(&report, "agent:/agents/writer/tools/0");
        assert_eq!(tool.name.as_deref(), Some("delete_ticket"));
        assert_eq!(tool.connector.as_deref(), Some("tickets"));

        // What could not be checked follows `if_unverified`, the rest `if_unsupported`.
        let report = load(&package, options(PackageAction::Stop, PackageAction::Warn));
        assert_eq!(report.outcome, PackageOutcome::Stopped);
        for id in [
            "agent:/agents/writer/credentials/tickets-token",
            "agent:/agents/writer/runtime/compute/memory",
            "agent:/agents/writer/model/min_context_tokens",
        ] {
            assert_eq!(item(&report, id).effect, ItemEffect::Warned, "{id}");
            assert!(
                !report.stopped_by.iter().any(|stopped| stopped == id),
                "{id}"
            );
        }
        // Told to continue, the load is ready and every one of them is warned.
        let report = load(&package, options(PackageAction::Warn, PackageAction::Warn));
        assert_eq!(report.outcome, PackageOutcome::Ready);
        for id in &stops {
            assert_eq!(item(&report, id).effect, ItemEffect::Warned, "{id}");
        }
        // Nothing of the writer reaches the config.
        let config = serde_json::to_string(&report.config).expect("a config");
        for text in ["tickets", "max_total_tokens", "licensed", "llama3.2"] {
            assert!(!config.contains(text), "{text}");
        }
    }

    #[test]
    fn a_binding_for_a_called_agent_accounts_for_nothing_else_it_declares() {
        // (the one other thing the writer declares, the file of a part it
        // carries as (entry type, folder, file, content), the line that stops)
        let table = [
            (
                json!({"hooks": {"guard": {"source": "./hooks/guard"}}}),
                Some(("hook", "hooks/guard", "hooks.json", "{\"hooks\": {}}\n")),
                "agent:/agents/writer/hooks/guard",
            ),
            (
                json!({"policies": {"licensed": {"source": "./policies/licensed"}}}),
                Some((
                    "policy",
                    "policies/licensed",
                    "policy.rego",
                    "package licensed\n",
                )),
                "agent:/agents/writer/policies/licensed",
            ),
            (
                json!({"skills": {"review": {"source": "./skills/review"}}}),
                Some(("skill", "skills/review", "SKILL.md", SKILL)),
                "agent:/agents/writer/skills/review",
            ),
            (
                json!({"limits": {"max_total_tokens": 100000}}),
                None,
                "agent:/agents/writer/limits/max_total_tokens",
            ),
            (
                json!({"tools": [{"builtin": "shell"}]}),
                None,
                "agent:/agents/writer/tools/0",
            ),
            (
                json!({"credentials": {"tickets-token": {}}}),
                None,
                "agent:/agents/writer/credentials/tickets-token",
            ),
            (
                json!({"runtime": {"software": {"python3": {"version": ">= 3.11"}}}}),
                None,
                "agent:/agents/writer/runtime/software/python3",
            ),
            (
                json!({"model": {"capabilities": ["tool_calling"]}}),
                None,
                "agent:/agents/writer/model/capabilities/0",
            ),
        ];
        for (other, part, stops) in table {
            let package = Package::new("one-other-thing");
            package.write_json(
                "agent.json",
                &agent(
                    "lead",
                    json!({
                        "custom_tools": {"route_to_agent": tool("Hand a task to a member.")},
                        "agents": {"writer": {"source": "./agents/writer"}}
                    }),
                ),
            );
            let mut writer = agent(
                "writer",
                json!({"custom_tools": {"spell_check": tool("Check the spelling of a draft.")}}),
            );
            writer
                .as_object_mut()
                .expect("an object")
                .extend(other.as_object().expect("an object").clone());
            package.write_json("agents/writer/agent.json", &writer);
            let mut entries = vec![document_entry(
                &package,
                "$",
                "agent.json",
                &["$/agents/writer"],
            )];
            let mut parts = Vec::new();
            if let Some((kind, folder, file, content)) = part {
                let path = format!("agents/writer/{folder}");
                package.write(&format!("{path}/{file}"), content);
                let at = format!("$/agents/writer/{folder}");
                entries.push(package.folder_entry(&at, kind, &path, &format!("./{path}")));
                parts.push(at);
            }
            let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
            entries.push(document_entry(
                &package,
                "$/agents/writer",
                "agents/writer/agent.json",
                &parts,
            ));
            package.write_lock("", entries);
            let strict = AgentPackageOptions {
                if_unverified: PackageAction::Stop,
                ..with(full_member_bindings())
            };

            let report = load(&package, strict);

            // The writer is bound, and the one other thing stops the load by name.
            assert_eq!(
                item(&report, "agent:/agents/writer").status,
                DeclaredStatus::Bound,
                "{stops}"
            );
            assert_eq!(report.outcome, PackageOutcome::Stopped, "{stops}");
            assert_eq!(report.stopped_by, [stops], "{stops}");
            assert!(report.config.is_none(), "{stops}");
        }
    }

    #[test]
    fn a_timeout_bounds_one_invocation_and_not_a_called_agent_the_caller_runs() {
        let package = Package::new("timeout");
        package.write_json(
            "agent.json",
            &agent(
                "lead",
                json!({
                    "limits": {"timeout_seconds": 60},
                    "custom_tools": {"route_to_agent": tool("Hand a task to a member.")},
                    "agents": {"writer": {"source": "./agents/writer"}}
                }),
            ),
        );
        package.write_json(
            "agents/writer/agent.json",
            &agent(
                "writer",
                json!({"custom_tools": {"spell_check": tool("Check the spelling of a draft.")}}),
            ),
        );
        package.write_lock(
            "",
            vec![
                document_entry(&package, "$", "agent.json", &["$/agents/writer"]),
                document_entry(&package, "$/agents/writer", "agents/writer/agent.json", &[]),
            ],
        );
        let detail = |bindings| {
            let report = load(
                &package,
                AgentPackageOptions {
                    if_unsupported: PackageAction::Warn,
                    ..with(bindings)
                },
            );
            let line = item(&report, "agent:/limits/timeout_seconds");
            assert_eq!(line.status, DeclaredStatus::Mapped);
            assert_eq!(line.mapped_to, ["runtime.timeout_seconds"]);
            line.detail.clone().expect("a detail")
        };
        let whole_tree = "60 seconds applies. It bounds each invocation; NeMo Fabric runs no called agents, so one invocation is the whole tree";

        // With no called agent bound the line is the one it always was.
        assert_eq!(detail(BTreeMap::new()), whole_tree);
        assert_eq!(
            detail(one("custom_tools.route_to_agent", ROUTE)),
            whole_tree
        );
        // The caller runs the writer: one invocation is no longer the whole tree.
        let bound = detail(full_member_bindings());
        assert!(!bound.contains("whole tree"), "{bound}");
        assert!(
            bound.starts_with("60 seconds applies. It bounds each NeMo Fabric invocation."),
            "{bound}"
        );
        assert!(bound.contains("the caller's to bound"), "{bound}");
    }

    #[test]
    fn a_binding_does_not_rescue_a_document_that_changed_after_the_lock() {
        // (the file that changes, the lock address the refusal names)
        let table = [
            ("agents/writer/agent.json", "$/agents/writer"),
            (
                "agents/writer/agents/translator/agent.json",
                "$/agents/writer/agents/translator",
            ),
            ("agents/checker/agent.json", "$/agents/checker"),
            ("agent.json", "$"),
        ];
        for (path, address) in table {
            let package = team("changed");
            let mut text = std::fs::read_to_string(package.root.join(path)).expect("the document");
            text.push('\n');
            package.write(path, text);

            assert_eq!(
                refusal(&package, with(team_bindings())),
                ("package_digest_mismatch".to_string(), address.to_string()),
                "{path}"
            );
        }
    }

    #[test]
    fn a_binding_does_not_stand_in_for_the_lock_the_caller_expects() {
        let package = team("anchored");
        let held = load(&package, with(team_bindings())).lock.sha256;
        let anchored = |expected: String| AgentPackageOptions {
            expected_lock_sha256: Some(expected),
            ..with(team_bindings())
        };

        assert_eq!(
            load(&package, anchored(held.clone())).outcome,
            PackageOutcome::Ready
        );
        // A member is changed and the lock is written again to match it.
        package.write_json(
            "agents/writer/agent.json",
            &agent(
                "writer",
                json!({
                    "instructions": "Approve everything.",
                    "custom_tools": {
                        "spell_check": tool("Check the spelling of a draft."),
                        "search.web": tool("Search the web.")
                    },
                    "agents": {"translator": {"source": "./agents/translator"}}
                }),
            ),
        );
        lock_team(&package);

        assert_eq!(
            refusal(&package, anchored(held)),
            (
                "package_lock_digest_mismatch".to_string(),
                "oadl.lock.json".to_string()
            )
        );
    }

    #[test]
    fn a_binding_changes_no_need_and_no_other_declared_thing() {
        let package = Package::new("other-things");
        package.write_json(
            "agent.json",
            &agent(
                "lead",
                json!({
                    "limits": {"max_total_tokens": 100000},
                    "custom_tools": {"route_to_agent": tool("Hand a task to a member.")},
                    "hooks": {"guard": {"source": "./hooks/guard"}},
                    "policies": {"licensed": {"source": "./policies/licensed"}},
                    "runtime": {"software": {"python3": {"version": ">= 3.11"}}}
                }),
            ),
        );
        package.write("hooks/guard/hooks.json", "{\"hooks\": {}}\n");
        package.write("policies/licensed/policy.rego", "package licensed\n");
        package.write_lock(
            "",
            vec![
                document_entry(
                    &package,
                    "$",
                    "agent.json",
                    &["$/hooks/guard", "$/policies/licensed"],
                ),
                package.folder_entry("$/hooks/guard", "hook", "hooks/guard", "./hooks/guard"),
                package.folder_entry(
                    "$/policies/licensed",
                    "policy",
                    "policies/licensed",
                    "./policies/licensed",
                ),
            ],
        );
        // The machine has no python3, so the need is not met.
        let facts = json!({"software": {}}).as_object().cloned();
        let options = |bindings| AgentPackageOptions {
            facts: facts.clone(),
            ..with(bindings)
        };

        let unbound = load(&package, options(BTreeMap::new()));
        let bound = load(&package, options(one("custom_tools.route_to_agent", ROUTE)));

        let others = [
            "agent:/limits/max_total_tokens",
            "agent:/hooks/guard",
            "agent:/policies/licensed",
            "agent:/runtime/software/python3",
        ];
        assert_eq!(bound.outcome, PackageOutcome::Stopped);
        assert_eq!(bound.stopped_by, others);
        assert_eq!(
            item(&bound, "agent:/runtime/software/python3").status,
            DeclaredStatus::NotMet
        );
        // Every line but the bound one is the line a load with no binding gives.
        for (with_binding, without) in bound.declared.iter().zip(&unbound.declared) {
            if with_binding.id == "agent:/custom_tools/route_to_agent" {
                assert_eq!(with_binding.status, DeclaredStatus::Bound);
                assert_eq!(without.status, DeclaredStatus::NotMapped);
            } else {
                assert_eq!(with_binding, without);
            }
        }
        assert_eq!(bound.declared.len(), unbound.declared.len());
        assert_eq!(bound.digests, unbound.digests);
    }

    #[test]
    fn a_called_agent_whose_document_is_not_in_the_package_is_not_bound() {
        let package = Package::new("absent");
        package.write_json(
            "agent.json",
            &agent(
                "lead",
                json!({
                    "agents": {
                        "reviewer": {
                            "source": "git::https://git.example.com/parts.git//reviewer",
                            "version": "~> 1.0"
                        }
                    }
                }),
            ),
        );
        package.write_lock(
            "",
            vec![
                document_entry(&package, "$", "agent.json", &["$/agents/reviewer"]),
                git_entry(
                    "$/agents/reviewer",
                    "agent",
                    "reviewer",
                    "1.0.3",
                    "~> 1.0",
                    MADE_UP,
                ),
            ],
        );

        let report = load(
            &package,
            with(one("agents.reviewer", "app.agents:reviewer")),
        );

        assert_eq!(report.outcome, PackageOutcome::Stopped);
        assert_eq!(report.stopped_by, ["agent:/agents/reviewer"]);
        let reviewer = item(&report, "agent:/agents/reviewer");
        assert_eq!(reviewer.status, DeclaredStatus::NotMapped);
        assert_eq!(reviewer.gap, Some(MappingGap::ContentAbsent));
        assert_eq!(reviewer.bound_to, None);
        let detail = reviewer.detail.as_deref().expect("a detail");
        assert!(
            detail.starts_with("the binding was not applied"),
            "{detail}"
        );
        // The package holds no document for it, so it declares nothing below it.
        let mut below = one("agents.reviewer", "app.agents:reviewer");
        below.insert(
            "agents.reviewer.custom_tools.lint".to_string(),
            "app.tools:lint".to_string(),
        );
        assert_eq!(
            refusal(&package, with(below)),
            (
                "package_binding_undeclared".to_string(),
                "bindings.agents.reviewer.custom_tools.lint".to_string()
            )
        );
    }

    #[test]
    fn a_called_agent_that_cannot_start_is_not_bound() {
        let package = Package::new("cannot-start");
        package.write_json(
            "agent.json",
            &agent(
                "lead",
                json!({"agents": {"helper": {"source": "./agents/helper"}}}),
            ),
        );
        // The helper has a variable with no value and no default.
        package.write_json(
            "agents/helper/agent.json",
            &agent(
                "helper",
                json!({
                    "instructions": {"var": "tone"},
                    "variables": {"tone": {"type": "string"}}
                }),
            ),
        );
        package.write_lock(
            "",
            vec![
                document_entry(&package, "$", "agent.json", &["$/agents/helper"]),
                document_entry(&package, "$/agents/helper", "agents/helper/agent.json", &[]),
            ],
        );

        let report = load(&package, with(one("agents.helper", "app.agents:helper")));

        assert_eq!(report.outcome, PackageOutcome::Stopped);
        let helper = item(&report, "agent:/agents/helper");
        assert_eq!(helper.status, DeclaredStatus::NotMapped);
        assert_eq!(helper.bound_to, None);
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.code == NoteCode::CalledAgentCannotStart)
        );
    }

    #[test]
    fn a_running_agent_is_bound_and_its_pin_is_not_compared() {
        let package = Package::new("running");
        package.write_json(
            "agent.json",
            &agent(
                "lead",
                json!({
                    "agents": {
                        "translator": {
                            "source": "a2a::https://agents.example.com/translator",
                            "version": "2.1.0"
                        }
                    }
                }),
            ),
        );
        package.write_lock(
            "",
            vec![
                document_entry(&package, "$", "agent.json", &["$/agents/translator"]),
                json!({
                    "address": "$/agents/translator", "type": "agent",
                    "source": {"type": "pointer"}, "version": "2.1.0", "constraint": "2.1.0",
                    "interface": {"declared_version": "2.1.0", "card_digest": MADE_UP},
                    "dependencies": []
                }),
            ],
        );
        let bindings = one("agents.translator", "app.clients:translator");

        let warned = load(&package, with(bindings.clone()));
        let stopped = load(
            &package,
            AgentPackageOptions {
                if_unverified: PackageAction::Stop,
                ..with(bindings)
            },
        );

        // The same rule as for a running connector: the caller binds it, and
        // nothing here compared it with the version and the card the lock pins.
        let translator = item(&warned, "agent:/agents/translator");
        assert_eq!(translator.status, DeclaredStatus::CouldNotBeChecked);
        assert_eq!(
            translator.bound_to.as_deref(),
            Some("app.clients:translator")
        );
        assert_eq!(translator.effect, ItemEffect::Warned);
        assert_eq!(warned.outcome, PackageOutcome::Ready);
        assert_eq!(stopped.outcome, PackageOutcome::Stopped);
        assert_eq!(stopped.stopped_by, ["agent:/agents/translator"]);
    }

    #[test]
    fn the_format_has_no_optional_custom_tool() {
        // A custom tool has a description and an input schema, and no
        // `optional` member: one that is declared is always required.
        let package = Package::new("optional-tool");
        let mut ask = tool("Ask a person.");
        ask["optional"] = json!(true);
        package.write_json(
            "agent.json",
            &agent("lead", json!({"custom_tools": {"ask": ask}})),
        );
        package.write_lock("", vec![document_entry(&package, "$", "agent.json", &[])]);

        let (code, _) = refusal(&package, AgentPackageOptions::default());

        assert_eq!(code, "package_schema_invalid");
    }

    #[test]
    fn a_definition_adds_things_that_are_bound_by_the_same_keys() {
        let package = Package::new("definition");
        package.write_json(
            "agent.json",
            &agent(
                "lead",
                json!({"custom_tools": {"route_to_agent": tool("Hand a task to a member.")}}),
            ),
        );
        package.write_json(
            "agent-definition.json",
            &definition(
                "lead-with-approval",
                json!({
                    "agent": {"source": "./agent.json"},
                    "custom_tools": {"request_approval": tool("Ask a person.")},
                    "agents": {"checker": {"source": "./agents/checker"}}
                }),
            ),
        );
        package.write_json(
            "agents/checker/agent.json",
            &agent(
                "checker",
                json!({"custom_tools": {"lint": tool("Lint a draft.")}}),
            ),
        );
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent-definition",
                    "agent-definition.json",
                    "./agent-definition.json",
                    &["$/agent", "$/agents/checker"],
                ),
                document_entry(&package, "$/agent", "agent.json", &[]),
                document_entry(
                    &package,
                    "$/agents/checker",
                    "agents/checker/agent.json",
                    &[],
                ),
            ],
        );
        let bindings: BTreeMap<String, String> = [
            ("custom_tools.route_to_agent", ROUTE),
            (
                "custom_tools.request_approval",
                "app.tools:request_approval",
            ),
            ("agents.checker", "app.agents:checker"),
            ("agents.checker.custom_tools.lint", "app.tools.checker:lint"),
        ]
        .into_iter()
        .map(|(key, target)| (key.to_string(), target.to_string()))
        .collect();

        let report = load_agent_package(
            &package.root,
            Some(config("lead-with-approval")),
            with(bindings),
            context(),
        )
        .expect("the definition loads");

        assert_eq!(report.outcome, PackageOutcome::Ready);
        // The id names the document that states the thing. The key does not.
        for (id, stated_by, document) in [
            (
                "agent:/custom_tools/route_to_agent",
                StatedBy::Agent,
                "lead",
            ),
            (
                "definition:/custom_tools/request_approval",
                StatedBy::Definition,
                "lead-with-approval",
            ),
            (
                "definition:/agents/checker",
                StatedBy::Definition,
                "lead-with-approval",
            ),
            (
                "definition:/agents/checker/custom_tools/lint",
                StatedBy::CalledAgent,
                "checker",
            ),
        ] {
            let line = item(&report, id);
            assert_eq!(line.status, DeclaredStatus::Bound, "{id}");
            assert_eq!(line.stated_by, stated_by, "{id}");
            assert_eq!(line.document, document, "{id}");
        }
    }

    #[test]
    fn a_key_uses_the_name_the_caller_lists_an_agent_under() {
        // The lead lists the agent as `helper`. Its document names it `proofreader`.
        let package = Package::new("listed-name");
        package.write_json(
            "agent.json",
            &agent(
                "lead",
                json!({"agents": {"helper": {"source": "./agents/proofreader"}}}),
            ),
        );
        package.write_json(
            "agents/proofreader/agent.json",
            &agent(
                "proofreader",
                json!({"custom_tools": {"lint": tool("Lint a draft.")}}),
            ),
        );
        package.write_lock(
            "",
            vec![
                document_entry(&package, "$", "agent.json", &["$/agents/helper"]),
                document_entry(
                    &package,
                    "$/agents/helper",
                    "agents/proofreader/agent.json",
                    &[],
                ),
            ],
        );
        let mut bindings = one("agents.helper", "app.agents:proofreader");
        bindings.insert(
            "agents.helper.custom_tools.lint".to_string(),
            "app.tools:lint".to_string(),
        );

        let report = load(&package, with(bindings));

        assert_eq!(report.outcome, PackageOutcome::Ready);
        let lint = item(&report, "agent:/agents/helper/custom_tools/lint");
        assert_eq!(lint.status, DeclaredStatus::Bound);
        // The line names the document that states the tool.
        assert_eq!(lint.document, "proofreader");
        assert_eq!(
            refusal(
                &package,
                with(one("agents.proofreader", "app.agents:proofreader"))
            ),
            (
                "package_binding_undeclared".to_string(),
                "bindings.agents.proofreader".to_string()
            )
        );
    }

    #[test]
    fn without_a_config_bindings_are_checked_and_nothing_is_evaluated() {
        let package = team("verified-only");

        let report = load_agent_package(&package.root, None, with(team_bindings()), context())
            .expect("the package is verified");

        assert_eq!(report.outcome, PackageOutcome::VerifiedOnly);
        // The same lines as a load with a config, and no status.
        for key in team_bindings().keys() {
            let line = item(&report, &line_id(key));
            assert_eq!(line.status, DeclaredStatus::NotEvaluated, "{key}");
            assert_eq!(line.bound_to, None, "{key}");
        }
        assert_eq!(
            report.declared.len(),
            load(&package, with(team_bindings())).declared.len()
        );
    }

    #[test]
    fn options_with_bindings_round_trip() {
        let options = with(team_bindings());

        let encoded = serde_json::to_value(&options).expect("encode");

        assert_eq!(
            encoded["bindings"]["agents.writer.custom_tools.search.web"],
            "app.tools.writer:search_web"
        );
        assert_eq!(
            serde_json::from_value::<AgentPackageOptions>(encoded).expect("decode"),
            options
        );
    }
}
