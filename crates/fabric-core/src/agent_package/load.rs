// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The two entry points: read, check and map one agent package.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::adapter::{AdapterSupport, adapter_support, harness_names};
use super::bindings::check_bindings;
use super::error::PackageError;
use super::facts::{overlay_facts, probe_facts};
use super::mapping::{map_package, unmapped_items, unmapped_model};
use super::model::{Facts, PackageReading, ReadRequest};
use super::needs::need_lines;
use super::options::AgentPackageOptions;
use super::reader::read_package;
use super::report::{
    AdapterReport, AgentPackageReport, PackageOutcome, REPORT_SCHEMA_VERSION, apply_actions,
    config_sha256, digest_checks,
};
use super::schemas::format_version;
use crate::config::{
    FabricConfig, ResolveContext, resolve_run_plan_from_config_with_adapter_directories,
};
use crate::error::{FabricError, Result};

impl From<PackageError> for FabricError {
    fn from(error: PackageError) -> Self {
        Self::AgentPackage {
            code: error.code.as_str().to_string(),
            subject: error.subject,
            message: error.message,
        }
    }
}

/// Read, check and map one agent package. Fetches nothing.
///
/// The package folder holds `agent.json` or `agent-definition.json`, the lock
/// file `oadl.lock.json` beside it, and the parts. The package is already
/// resolved and locked: this function resolves nothing, installs nothing and
/// never uses the network.
///
/// `config` is the caller's complete config. It holds what a package has no
/// field for, such as the adapter, the model endpoint and the environment.
/// With `None` the package is verified and reported, and nothing is mapped.
///
/// A package that cannot be accepted is an error:
/// [`FabricError::AgentPackage`] with a stable code and the thing it names.
/// A load that the `if_unsupported` or `if_unverified` option stops is not an error:
/// the report has outcome `stopped` and no config.
///
/// The check happens once, at load time. It does not protect the files for
/// the life of a runtime.
pub fn load_agent_package(
    package_dir: &Path,
    config: Option<FabricConfig>,
    options: AgentPackageOptions,
    context: ResolveContext,
) -> Result<AgentPackageReport> {
    load_agent_package_with_adapter_directories(package_dir, config, options, context, &[])
}

/// Same as [`load_agent_package`], with extra adapter descriptor directories.
///
/// This is an internal integration surface for hosts that know about
/// environment-specific package data directories.
#[doc(hidden)]
pub fn load_agent_package_with_adapter_directories(
    package_dir: &Path,
    config: Option<FabricConfig>,
    options: AgentPackageOptions,
    context: ResolveContext,
    adapter_directories: &[PathBuf],
) -> Result<AgentPackageReport> {
    let base_dir = std::path::absolute(&context.base_dir).map_err(|source| {
        FabricError::ResolveBaseDirectory {
            path: context.base_dir.clone(),
            source,
        }
    })?;

    // The package, the entry document, the lock and the variables.
    let reading = read_package(
        package_dir,
        &ReadRequest {
            entry: options.entry.clone(),
            variables: options.variables.clone(),
            expected_lock_sha256: options.expected_lock_sha256.clone(),
        },
    )?;
    // The documents, the lock and every pin are checked. Only now is a binding
    // looked at, and only to see that it names a thing the package declares.
    check_bindings(&reading, &options.bindings)?;

    // Facts: probed only when asked, and the caller's facts win.
    let probed = if options.check_machine {
        probe_facts(&software_names(&reading), &base_dir)
    } else {
        Facts::default()
    };
    let mut facts = overlay_facts(probed, options.facts.as_ref())?;
    let support = config
        .as_ref()
        .map(|config| adapter_support(config, &context, adapter_directories))
        .transpose()?;
    // The harness names are needed only when a document states a harness need.
    if let (Some(config), Some(support)) = (&config, &support)
        && states_a_harness(&reading)
    {
        add_harness_names(&mut facts, config, support, &context, adapter_directories);
    }
    let needs = need_lines(&reading.agent, reading.definition.as_ref(), &facts);

    let mut report = AgentPackageReport {
        schema_version: REPORT_SCHEMA_VERSION.to_string(),
        format_version: format_version().to_string(),
        package_dir: reading.package_dir.clone(),
        base_dir,
        outcome: PackageOutcome::VerifiedOnly,
        if_unsupported: options.if_unsupported,
        if_unverified: options.if_unverified,
        stopped_by: Vec::new(),
        entry: reading.entry.clone(),
        agent: reading.agent_record.clone(),
        lock: reading.lock.clone(),
        adapter: None,
        variables: reading.variables.clone(),
        model: unmapped_model(&reading),
        declared: Vec::new(),
        digests: digest_checks(&reading),
        notes: reading
            .notes
            .iter()
            .chain(reading.called.iter().flat_map(|called| called.notes.iter()))
            .cloned()
            .collect(),
        config: None,
        config_sha256: None,
    };

    let (Some(config), Some(support)) = (config, support) else {
        report.declared = unmapped_items(&reading, &options, &needs, &facts);
        return Ok(report);
    };

    let mut mapped = map_package(&reading, config, &support, &options, &needs, &facts)?;
    // The mapped config must plan on the selected adapter. A failure here is
    // returned as it is.
    let planned = resolve_run_plan_from_config_with_adapter_directories(
        mapped.config.clone(),
        context.clone(),
        adapter_directories,
    )?;
    // The base directory the planner resolved is the one a later plan or start uses.
    report.base_dir = planned.base_dir;

    let (outcome, stopped_by) = apply_actions(
        &mut mapped.items,
        options.if_unsupported,
        options.if_unverified,
    );
    report.outcome = outcome;
    report.stopped_by = stopped_by;
    report.adapter = Some(AdapterReport {
        adapter_id: support.adapter_id.clone(),
        accepts: support.accepts.iter().copied().collect(),
        extensions: support.extension_keys().to_vec(),
        instruction_mode: mapped.instruction_mode,
    });
    report.model = mapped.model;
    report.declared = mapped.items;
    report.notes.extend(mapped.notes);
    if outcome == PackageOutcome::Ready {
        report.config_sha256 = Some(config_sha256(&mapped.config)?);
        report.config = Some(mapped.config);
    }
    Ok(report)
}

/// The program names the agent and its definition state under `runtime.software`.
fn software_names(reading: &PackageReading) -> Vec<String> {
    let mut names = Vec::new();
    for document in [Some(&reading.agent), reading.definition.as_ref()]
        .into_iter()
        .flatten()
    {
        let Some(software) = document
            .get("runtime")
            .and_then(|runtime| runtime.get("software"))
            .and_then(Value::as_object)
        else {
            continue;
        };
        for name in software.keys() {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    names
}

/// Whether the agent or its definition states `runtime.harness`.
fn states_a_harness(reading: &PackageReading) -> bool {
    [Some(&reading.agent), reading.definition.as_ref()]
        .into_iter()
        .flatten()
        .any(|document| document.pointer("/runtime/harness").is_some())
}

/// Tell the need check which names this harness goes by and which names NeMo
/// Fabric knows. A name the caller gave stays first.
fn add_harness_names(
    facts: &mut Facts,
    config: &FabricConfig,
    support: &AdapterSupport,
    context: &ResolveContext,
    adapter_directories: &[PathBuf],
) {
    let (own, known) = harness_names(config, support, context, adapter_directories);
    for name in own {
        if !facts.harness_aliases.contains(&name) {
            facts.harness_aliases.push(name);
        }
    }
    if facts.harness.is_none() {
        facts.harness = Some(support.adapter_id.clone());
    }
    for name in known {
        if !facts.known.harnesses.contains(&name) {
            facts.known.harnesses.push(name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_package::model::{DocumentKind, Integrity, PackageAction, StatedBy};
    use crate::agent_package::report::{
        DeclaredKind, DeclaredStatus, ItemEffect, MappingGap, ModelOrigin,
    };
    use crate::config::SkillDirectory;

    const SHIM: &str = "test.fabric.hermes_shim";
    /// A first-party adapter whose descriptor lists `skill` entries.
    const HERMES: &str = "nvidia.fabric.hermes";
    const VARIANT: &str = "variants/short/agent-definition.json";

    fn repository() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// The repository's example package: one skill, one carried connector, one variant.
    fn example() -> PathBuf {
        repository().join("examples/agent_package/release-notes-agent")
    }

    /// The test adapter. It accepts models, mcp and skills, and neither
    /// instructions nor a turn limit.
    fn context() -> ResolveContext {
        ResolveContext::new(repository().join("tests/fixtures/hermes-shim-agent"))
    }

    fn config(name: &str, adapter_id: &str) -> FabricConfig {
        serde_json::from_value(serde_json::json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": name},
            "harness": {"adapter_id": adapter_id},
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {},
        }))
        .expect("a config")
    }

    /// The same config with `skills.directories` set to an empty list.
    fn config_with_directories(name: &str, adapter_id: &str) -> FabricConfig {
        let mut config = config(name, adapter_id);
        config.skills =
            Some(serde_json::from_value(serde_json::json!({"directories": []})).expect("skills"));
        config
    }

    fn warn() -> AgentPackageOptions {
        AgentPackageOptions {
            if_unsupported: PackageAction::Warn,
            ..AgentPackageOptions::default()
        }
    }

    fn load(config: Option<FabricConfig>, options: AgentPackageOptions) -> AgentPackageReport {
        load_agent_package(&example(), config, options, context()).expect("the example loads")
    }

    #[test]
    fn a_package_without_a_config_is_verified_and_reported() {
        let report = load(None, AgentPackageOptions::default());

        assert_eq!(report.outcome, PackageOutcome::VerifiedOnly);
        assert_eq!(report.schema_version, REPORT_SCHEMA_VERSION);
        assert_eq!(report.format_version, "0.4.0");
        assert_eq!(report.entry.kind, DocumentKind::Agent);
        assert_eq!(report.entry.name, "release-notes-agent");
        assert_eq!(report.agent, report.entry);
        assert!(report.adapter.is_none());
        assert!(report.config.is_none());
        assert!(report.config_sha256.is_none());
        assert!(report.stopped_by.is_empty());
        assert_eq!(report.model.origin, ModelOrigin::NotEvaluated);
        assert_eq!(report.lock.entries, report.digests.len());
        assert!(!report.digests.is_empty());
        for check in &report.digests {
            assert_eq!(check.result, Integrity::Verified, "{}", check.address);
            assert!(check.digest.is_some());
            assert!(check.path.as_ref().is_some_and(|path| path.is_absolute()));
        }
        assert!(!report.declared.is_empty());
        for item in &report.declared {
            if item.kind != DeclaredKind::Need {
                assert_eq!(item.status, DeclaredStatus::NotEvaluated, "{}", item.id);
            }
            assert_eq!(item.effect, ItemEffect::None);
        }
        // Nothing was probed and no fact was given, so the need could not be checked.
        let software = report
            .declared
            .iter()
            .find(|item| item.id == "agent:/runtime/software/python3")
            .expect("the software need");
        assert_eq!(software.status, DeclaredStatus::CouldNotBeChecked);
        assert!(software.reason.is_some());
    }

    #[test]
    fn default_settings_stop_on_what_the_adapter_does_not_accept() {
        let report = load(
            Some(config("release-notes-agent", SHIM)),
            AgentPackageOptions::default(),
        );

        assert_eq!(report.outcome, PackageOutcome::Stopped);
        assert!(report.config.is_none());
        assert!(report.config_sha256.is_none());
        for id in ["agent:/instructions", "agent:/limits/max_turns"] {
            assert!(
                report.stopped_by.iter().any(|stopped| stopped == id),
                "{id}"
            );
            let item = report
                .declared
                .iter()
                .find(|item| item.id == id)
                .expect("a declared item");
            assert_eq!(item.status, DeclaredStatus::NotMapped);
            assert_eq!(item.effect, ItemEffect::Stopped);
        }
        assert_eq!(
            report.adapter.as_ref().expect("an adapter").adapter_id,
            SHIM
        );
    }

    #[test]
    fn a_caller_fact_decides_a_need() {
        let options = AgentPackageOptions {
            facts: serde_json::json!({"software": {"python3": "3.9.1"}})
                .as_object()
                .cloned(),
            ..warn()
        };

        let report = load(Some(config("release-notes-agent", SHIM)), options);

        let software = report
            .declared
            .iter()
            .find(|item| item.id == "agent:/runtime/software/python3")
            .expect("the software need");
        assert_eq!(software.status, DeclaredStatus::NotMet);
        assert_eq!(software.effect, ItemEffect::Warned);
        assert_eq!(report.outcome, PackageOutcome::Ready);
    }

    #[test]
    fn warn_gives_a_config_that_plans_and_a_stable_digest() {
        let report = load(Some(config("release-notes-agent", SHIM)), warn());
        let again = load(Some(config("release-notes-agent", SHIM)), warn());

        assert_eq!(report.outcome, PackageOutcome::Ready);
        assert!(report.stopped_by.is_empty());
        let mapped = report.config.clone().expect("a mapped config");
        assert_eq!(mapped.metadata.name, "release-notes-agent");
        assert!(mapped.metadata.description.is_some());
        assert_eq!(mapped.runtime.timeout_seconds, Some(600.0));
        // The adapter accepts neither, so neither was written.
        assert!(mapped.instructions.is_none());
        assert_eq!(mapped.runtime.max_turns, None);
        let skills = mapped.skills.as_ref().expect("skills").paths.clone();
        assert_eq!(skills.len(), 1);
        assert!(skills[0].is_absolute());
        assert!(skills[0].ends_with("skills/release-notes"));
        assert!(skills[0].join("SKILL.md").is_file());
        if cfg!(unix) {
            let servers = &mapped.mcp.as_ref().expect("mcp").servers;
            assert_eq!(servers.keys().collect::<Vec<_>>(), vec!["changelog"]);
            assert_eq!(servers["changelog"].url, "/bin/sh");
        }
        assert_eq!(
            report.base_dir,
            std::path::absolute(context().base_dir).unwrap()
        );
        // The same package, config and options give the same values.
        assert_eq!(report, again);
        assert_eq!(
            report.config_sha256,
            Some(config_sha256(&mapped).expect("a digest"))
        );
        // The mapped config is a normal config: the existing planner takes it.
        resolve_run_plan_from_config_with_adapter_directories(mapped, context(), &[])
            .expect("the mapped config plans");
    }

    #[test]
    fn a_config_with_skill_directories_gets_a_skill_entry_that_plans() {
        let report = load(
            Some(config_with_directories("release-notes-agent", HERMES)),
            warn(),
        );

        assert_eq!(report.outcome, PackageOutcome::Ready);
        let skill = report
            .declared
            .iter()
            .find(|item| item.id == "agent:/skills/release-notes")
            .expect("the skill");
        assert_eq!(skill.status, DeclaredStatus::Mapped);
        assert_eq!(skill.mapped_to, vec!["skills.directories.0.skill"]);
        let mapped = report.config.clone().expect("a mapped config");
        let skills = mapped.skills.as_ref().expect("skills");
        assert!(skills.paths.is_empty());
        let entries = skills.directories.clone().expect("directories");
        assert_eq!(entries.len(), 1);
        let SkillDirectory::Skill(folder) = &entries[0] else {
            panic!("expected a `skill` entry, got {:?}", entries[0]);
        };
        assert!(folder.is_absolute());
        assert!(folder.ends_with("skills/release-notes"));
        assert!(folder.join("SKILL.md").is_file());
        // The planner takes the entry and hands it to the adapter.
        let plan = resolve_run_plan_from_config_with_adapter_directories(mapped, context(), &[])
            .expect("the mapped config plans");
        assert_eq!(plan.capability_plan.skill_directories, Some(entries));
        assert!(plan.capability_plan.skill_paths.is_empty());
    }

    #[test]
    fn an_adapter_without_directory_loading_gets_no_skill_entry() {
        let caller = config_with_directories("release-notes-agent", SHIM);

        let stopped = load(Some(caller.clone()), AgentPackageOptions::default());
        let warned = load(Some(caller.clone()), warn());

        // The descriptor lists no `skill` entries, and planning refuses an
        // entry it does not list. The skill is reported, never written.
        assert_eq!(stopped.outcome, PackageOutcome::Stopped);
        assert!(
            stopped
                .stopped_by
                .iter()
                .any(|id| id == "agent:/skills/release-notes")
        );
        assert_eq!(warned.outcome, PackageOutcome::Ready);
        let skill = warned
            .declared
            .iter()
            .find(|item| item.id == "agent:/skills/release-notes")
            .expect("the skill");
        assert_eq!(skill.status, DeclaredStatus::NotMapped);
        assert_eq!(skill.gap, Some(MappingGap::AdapterDoesNotAccept));
        assert_eq!(skill.effect, ItemEffect::Warned);
        let mapped = warned.config.expect("a mapped config");
        assert_eq!(mapped.skills, caller.skills);
    }

    #[test]
    fn a_definition_kept_in_the_package_loads_through_the_entry_option() {
        let options = AgentPackageOptions {
            entry: Some(PathBuf::from(VARIANT)),
            ..warn()
        };

        let report = load(Some(config("release-notes-short", SHIM)), options);

        assert_eq!(report.outcome, PackageOutcome::Ready);
        assert_eq!(report.entry.kind, DocumentKind::AgentDefinition);
        assert_eq!(report.entry.name, "release-notes-short");
        assert_eq!(report.agent.kind, DocumentKind::Agent);
        assert_eq!(report.agent.name, "release-notes-agent");
        for check in &report.digests {
            assert_eq!(check.result, Integrity::Verified, "{}", check.address);
        }
        let added = report
            .declared
            .iter()
            .find(|item| item.id == "definition:/skills/short-form")
            .expect("the skill the definition adds");
        assert_eq!(added.stated_by, StatedBy::Definition);
        assert_eq!(added.document, "release-notes-short");
        assert_eq!(added.status, DeclaredStatus::Mapped);
        let mapped = report.config.expect("a mapped config");
        assert_eq!(mapped.metadata.name, "release-notes-short");
        assert_eq!(mapped.skills.expect("skills").paths.len(), 2);
    }

    #[test]
    fn refusals_carry_a_stable_code_and_the_thing_they_name() {
        let other_agent = load_agent_package(
            &example(),
            Some(config("another-agent", SHIM)),
            AgentPackageOptions::default(),
            context(),
        );
        let missing = load_agent_package(
            &example().join("no-such-folder"),
            None,
            AgentPackageOptions::default(),
            context(),
        );
        let unknown_fact = load_agent_package(
            &example(),
            None,
            AgentPackageOptions {
                facts: serde_json::json!({"ram": "64gb"}).as_object().cloned(),
                ..AgentPackageOptions::default()
            },
            context(),
        );

        // (result, code, subject when it is fixed)
        let table = [
            (
                other_agent,
                "package_config_conflict",
                Some("metadata.name"),
            ),
            (missing, "package_not_found", None),
            (unknown_fact, "package_options_invalid", Some("facts.ram")),
        ];
        for (result, expected_code, expected_subject) in table {
            match result {
                Err(FabricError::AgentPackage { code, subject, .. }) => {
                    assert_eq!(code, expected_code);
                    if let Some(expected) = expected_subject {
                        assert_eq!(subject, expected);
                    }
                }
                other => panic!("expected {expected_code}, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_config_error_passes_through_unchanged() {
        let result = load_agent_package(
            &example(),
            Some(config("release-notes-agent", "no.such.adapter")),
            AgentPackageOptions::default(),
            context(),
        );

        assert!(
            matches!(result, Err(FabricError::UnknownAdapter { ref adapter_id, .. }) if adapter_id == "no.such.adapter"),
            "{result:?}"
        );
    }

    #[test]
    fn harness_names_come_from_the_selected_and_the_discovered_adapters() {
        let config = config("release-notes-agent", SHIM);
        let support = adapter_support(&config, &context(), &[]).expect("adapter support");
        let mut facts = Facts {
            harness: Some("my-harness".to_string()),
            ..Facts::default()
        };

        add_harness_names(&mut facts, &config, &support, &context(), &[]);

        // The caller's name stays, and the adapter id and its last segment are added.
        assert_eq!(facts.harness.as_deref(), Some("my-harness"));
        assert_eq!(facts.harness_aliases, vec![SHIM, "hermes_shim"]);
        for known in [SHIM, "hermes_shim", "nvidia.fabric.claude", "claude"] {
            assert!(
                facts.known.harnesses.iter().any(|name| name == known),
                "{known} in {:?}",
                facts.known.harnesses
            );
        }
        let mut unnamed = Facts::default();
        add_harness_names(&mut unnamed, &config, &support, &context(), &[]);
        assert_eq!(unnamed.harness.as_deref(), Some(SHIM));
    }

    #[test]
    fn the_lock_must_have_the_digest_the_caller_expects() {
        let lock_sha256 = load(None, AgentPackageOptions::default()).lock.sha256;
        let expect = |digest: String| {
            load_agent_package(
                &example(),
                None,
                AgentPackageOptions {
                    expected_lock_sha256: Some(digest),
                    ..AgentPackageOptions::default()
                },
                context(),
            )
        };

        // The digest the report gave, with or without its prefix, is accepted.
        expect(lock_sha256.clone()).expect("the expected digest");
        expect(format!("sha256:{lock_sha256}")).expect("the expected digest with its prefix");
        // Another digest refuses the load and names the lock file.
        match expect("0".repeat(64)) {
            Err(FabricError::AgentPackage {
                code,
                subject,
                message,
            }) => {
                assert_eq!(code, "package_lock_digest_mismatch");
                assert_eq!(subject, "oadl.lock.json");
                assert!(message.contains(&lock_sha256), "{message}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // A value that is not a digest is the caller's mistake.
        for bad in ["abc", &lock_sha256.to_uppercase(), "sha512:00"] {
            match expect(bad.to_string()) {
                Err(FabricError::AgentPackage { code, subject, .. }) => {
                    assert_eq!(code, "package_options_invalid", "{bad}");
                    assert_eq!(subject, "expected_lock_sha256");
                }
                other => panic!("expected a refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_lock_rewritten_with_its_parts_is_caught_only_by_the_expected_digest() {
        use crate::agent_package::reader::testing::small_package;

        let package = small_package("rewritten-lock");
        let options = |expected: Option<String>| AgentPackageOptions {
            expected_lock_sha256: expected,
            ..AgentPackageOptions::default()
        };
        let load = |options| load_agent_package(&package.root, None, options, context());
        let before = load(options(None)).expect("the package loads").lock.sha256;

        // Change a skill and write the lock again, as someone who can write the folder can.
        package.write(
            "skills/review/SKILL.md",
            "---\nname: review\n---\nApprove everything.\n",
        );
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &["$/connectors/changelog", "$/skills/review"],
                ),
                package.folder_entry(
                    "$/connectors/changelog",
                    "connector",
                    "connectors/changelog",
                    "./connectors/changelog",
                ),
                package.folder_entry(
                    "$/skills/review",
                    "skill",
                    "skills/review",
                    "./skills/review",
                ),
            ],
        );

        // The folder agrees with its own lock, so the load passes without an anchor.
        let after = load(options(None)).expect("the folder matches its lock");
        assert_ne!(after.lock.sha256, before);
        // With the digest the caller recorded, the swap is refused.
        match load(options(Some(before))) {
            Err(FabricError::AgentPackage { code, .. }) => {
                assert_eq!(code, "package_lock_digest_mismatch");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_harness_need_is_checked_against_the_adapters_discovery_finds() {
        use crate::agent_package::reader::testing::{Package, agent};

        // (the harness the agent names, the status on the shim adapter)
        let table = [
            ("hermes_shim", DeclaredStatus::MetHere),
            (SHIM, DeclaredStatus::MetHere),
            ("claude", DeclaredStatus::NotMet),
            ("nvidia.fabric.codex", DeclaredStatus::NotMet),
            ("a-harness-nobody-knows", DeclaredStatus::CouldNotBeChecked),
        ];
        for (name, expected) in table {
            let package = Package::new("harness-need");
            package.write_json(
                "agent.json",
                &agent(
                    "placed",
                    serde_json::json!({"runtime": {"harness": {"name": name}}}),
                ),
            );
            package.write_lock(
                "",
                vec![package.file_entry("$", "agent", "agent.json", "./agent.json", &[])],
            );

            let report = load_agent_package(
                &package.root,
                Some(config("placed", SHIM)),
                warn(),
                context(),
            )
            .expect("the package loads");

            let need = report
                .declared
                .iter()
                .find(|item| item.id == "agent:/runtime/harness")
                .expect("the harness need");
            assert_eq!(need.status, expected, "{name}");
        }
    }

    /// A fixture adapter whose descriptor declares the three root extension
    /// keys for plugins, hooks and called agents. It is a descriptor only: a
    /// config plans on it and nothing starts.
    const PACKAGE_PARTS: &str = "test.fabric.package_parts";
    const HOOK_FORMAT: &str = "https://agent-plugins.org/schemas/1.1.0/hooks.schema.json";

    /// A video producer with one plugin, one hook and one called agent.
    fn video_producer(hook_format: &str) -> crate::agent_package::reader::testing::Package {
        use crate::agent_package::reader::testing::{Package, agent};

        let package = Package::new("video-producer");
        package.write_json(
            "agent.json",
            &agent(
                "video-producer",
                serde_json::json!({
                    "limits": {"max_turns": 60},
                    "plugins": {"remotion": {"source": "./plugins/remotion"}},
                    "hooks": {"tool-guard": {"source": "./hooks/tool-guard"}},
                    "agents": {"reviewer": {"source": "./agents/reviewer"}}
                }),
            ),
        );
        package.write("plugins/remotion/plugin.json", r#"{"name": "remotion"}"#);
        package.write(
            "hooks/tool-guard/hooks.json",
            serde_json::json!({"$schema": hook_format, "hooks": {}}).to_string(),
        );
        package.write_json(
            "agents/reviewer/agent.json",
            &agent(
                "reviewer",
                serde_json::json!({
                    "description": "Reviews a rendered video against the checklist.",
                    "instructions": "Check each item of the checklist and report what fails.",
                    "limits": {"max_turns": 80}
                }),
            ),
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
                        "$/agents/reviewer",
                        "$/hooks/tool-guard",
                        "$/plugins/remotion",
                    ],
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
        package
    }

    fn load_video_producer(
        package: &crate::agent_package::reader::testing::Package,
        adapter_id: &str,
    ) -> AgentPackageReport {
        load_agent_package(
            &package.root,
            Some(config("video-producer", adapter_id)),
            AgentPackageOptions::default(),
            ResolveContext::new(repository().join("tests/fixtures/agent_packages")),
        )
        .expect("the package loads")
    }

    fn line<'a>(
        report: &'a AgentPackageReport,
        id: &str,
    ) -> &'a crate::agent_package::report::DeclaredItem {
        report
            .declared
            .iter()
            .find(|item| item.id == id)
            .unwrap_or_else(|| panic!("no line {id}"))
    }

    #[test]
    fn plugins_hooks_and_called_agents_plan_on_an_adapter_that_declares_the_keys() {
        let package = video_producer(HOOK_FORMAT);

        let report = load_video_producer(&package, PACKAGE_PARTS);

        // The mapped config planned on the adapter's descriptor.
        assert_eq!(report.outcome, PackageOutcome::Ready);
        // A path is the package folder as the caller wrote it, and the part below it.
        let folder = |below: &str| package.root.join(below).to_string_lossy().into_owned();
        let config = report.config.as_ref().expect("a config");
        assert_eq!(
            serde_json::to_value(&config.extensions).expect("encode"),
            serde_json::json!({
                "nemo_fabric.plugins.v1": {"paths": [folder("plugins/remotion")]},
                "nemo_fabric.hooks.v1": {"entries": [
                    {"path": folder("hooks/tool-guard"), "format": HOOK_FORMAT}
                ]},
                "nemo_fabric.agents.v1": {"reviewer": {
                    "description": "Reviews a rendered video against the checklist.",
                    "instructions": "Check each item of the checklist and report what fails.",
                    "max_turns": 60
                }}
            })
        );
        assert_eq!(
            report.adapter.as_ref().expect("an adapter").extensions,
            [
                "nemo_fabric.agents.v1",
                "nemo_fabric.hooks.v1",
                "nemo_fabric.plugins.v1"
            ]
        );
        for (id, path) in [
            ("agent:/plugins/remotion", "nemo_fabric.plugins.v1.paths.0"),
            ("agent:/hooks/tool-guard", "nemo_fabric.hooks.v1.entries.0"),
            ("agent:/agents/reviewer", "nemo_fabric.agents.v1.reviewer"),
            (
                "agent:/agents/reviewer/limits/max_turns",
                "nemo_fabric.agents.v1.reviewer.max_turns",
            ),
        ] {
            let declared = line(&report, id);
            assert_eq!(declared.status, DeclaredStatus::Mapped, "{id}");
            assert_eq!(declared.mapped_to, vec![path], "{id}");
            assert_eq!(declared.effect, ItemEffect::None, "{id}");
        }
    }

    #[test]
    fn a_hook_format_the_adapter_does_not_list_stops_the_load_and_is_not_written() {
        let package = video_producer("https://hooks.example.com/schemas/2.0/hooks.schema.json");

        let report = load_video_producer(&package, PACKAGE_PARTS);

        assert_eq!(report.outcome, PackageOutcome::Stopped);
        assert_eq!(report.stopped_by, ["agent:/hooks/tool-guard"]);
        let hook = line(&report, "agent:/hooks/tool-guard");
        assert_eq!(hook.gap, Some(MappingGap::AdapterDoesNotAccept));
        assert!(report.config.is_none());
    }

    #[test]
    fn an_adapter_that_declares_no_key_reports_each_part_by_name() {
        let package = video_producer(HOOK_FORMAT);

        // Cline is a first-party adapter whose descriptor declares none of the keys.
        let report = load_video_producer(&package, "nvidia.fabric.cline");

        assert_eq!(report.outcome, PackageOutcome::Stopped);
        assert_eq!(
            report.stopped_by,
            [
                "agent:/limits/max_turns",
                "agent:/plugins/remotion",
                "agent:/hooks/tool-guard",
                "agent:/agents/reviewer"
            ]
        );
        assert!(
            report
                .adapter
                .as_ref()
                .expect("an adapter")
                .extensions
                .is_empty()
        );
        for (id, name) in [
            ("agent:/plugins/remotion", "remotion"),
            ("agent:/hooks/tool-guard", "tool-guard"),
            ("agent:/agents/reviewer", "reviewer"),
        ] {
            let declared = line(&report, id);
            assert_eq!(declared.name.as_deref(), Some(name));
            assert_eq!(declared.status, DeclaredStatus::NotMapped, "{id}");
            assert_eq!(declared.gap, Some(MappingGap::AdapterDoesNotAccept), "{id}");
            assert_eq!(declared.effect, ItemEffect::Stopped, "{id}");
        }
    }

    #[test]
    fn a_tool_definition_for_a_custom_tool_is_kept_on_an_adapter_that_takes_one() {
        use crate::agent_package::reader::testing::{Package, agent};

        let package = Package::new("custom-tool");
        package.write_json(
            "agent.json",
            &agent(
                "video-producer",
                serde_json::json!({"custom_tools": {"request_approval": {
                    "description": "Show the beat sheet to the person and wait for approval.",
                    "input_schema": {"type": "object"}
                }}}),
            ),
        );
        package.write_lock(
            "",
            vec![package.file_entry("$", "agent", "agent.json", "./agent.json", &[])],
        );
        // Pi is a first-party adapter that accepts `tools.definitions`: a
        // module in the workspace and the export that builds the tool.
        let mut config = config("video-producer", "nvidia.fabric.pi");
        let definition = serde_json::json!({
            "kind": "module",
            "ref": "tools/request_approval.ts#create"
        });
        config.tools = Some(
            serde_json::from_value(serde_json::json!({
                "definitions": {"request_approval": definition}
            }))
            .expect("tools"),
        );

        let report = load_agent_package(
            &package.root,
            Some(config.clone()),
            AgentPackageOptions::default(),
            context(),
        )
        .expect("the package loads");

        assert_eq!(report.outcome, PackageOutcome::Ready);
        let tool = line(&report, "agent:/custom_tools/request_approval");
        assert_eq!(tool.status, DeclaredStatus::CouldNotBeChecked);
        assert_eq!(tool.mapped_to, vec!["tools.definitions.request_approval"]);
        assert_eq!(tool.effect, ItemEffect::Warned);
        // The mapped config planned on Pi's descriptor with the caller's entry.
        assert_eq!(report.config.expect("a config").tools, config.tools);
    }

    #[test]
    fn the_settings_are_not_applied_when_no_config_is_given() {
        // With no config nothing starts, so nothing is stopped or warned: each
        // need keeps its status and the caller decides from the report.
        let options = AgentPackageOptions {
            facts: serde_json::json!({"software": {}}).as_object().cloned(),
            if_unsupported: PackageAction::Stop,
            if_unverified: PackageAction::Stop,
            ..AgentPackageOptions::default()
        };

        let report = load(None, options);

        assert_eq!(report.outcome, PackageOutcome::VerifiedOnly);
        assert!(report.stopped_by.is_empty());
        let software = report
            .declared
            .iter()
            .find(|item| item.id == "agent:/runtime/software/python3")
            .expect("the software need");
        assert_eq!(software.status, DeclaredStatus::NotMet);
        assert!(software.required);
        assert_eq!(software.effect, ItemEffect::None);
    }

    #[test]
    fn the_program_names_to_probe_come_from_both_documents() {
        let options = AgentPackageOptions {
            entry: Some(PathBuf::from(VARIANT)),
            ..AgentPackageOptions::default()
        };
        let reading = read_package(
            &example(),
            &ReadRequest {
                entry: options.entry.clone(),
                variables: options.variables.clone(),
                ..ReadRequest::default()
            },
        )
        .expect("the example reads");

        assert_eq!(software_names(&reading), vec!["python3"]);
    }
}
