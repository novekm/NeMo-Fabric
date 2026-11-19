// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The lock file: its shape and its own rules, before it is compared with any document.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::Value;

use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::model::{EntryType, LockInterface, SourceType};
use super::paths::is_safe_relative;
use super::schemas::{SchemaKind, first_failure, schema_id, unsupported_schema_message};

/// Where a lock entry says it was resolved from.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct LockSource {
    #[serde(rename = "type")]
    pub source_type: SourceType,
    pub url: Option<String>,
    pub subpath: Option<String>,
    /// A tag, a branch or a commit a developer fetched in place of the selected version.
    #[serde(rename = "ref")]
    pub reference: Option<String>,
    pub registry: Option<String>,
}

/// One entry of a lock.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct LockEntry {
    pub address: String,
    #[serde(rename = "type")]
    pub entry_type: EntryType,
    pub source: LockSource,
    pub version: String,
    pub constraint: Option<String>,
    pub commit: Option<String>,
    pub digest: Option<String>,
    pub interface: Option<LockInterface>,
    #[serde(default)]
    pub indirect: bool,
    pub retracted: Option<String>,
    pub dependencies: Vec<String>,
}

/// A lock that passed its schema and its own rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lock {
    pub generator: Option<String>,
    /// Every entry, by address.
    pub entries: BTreeMap<String, LockEntry>,
}

fn invalid(subject: impl Into<String>, message: impl Into<String>) -> PackageError {
    PackageError::new(PackageErrorCode::LockInvalid, subject, message)
}

/// The part of a local url after `./`: a relative path that may begin with `..` segments.
fn is_local_lock_path(path: &str) -> bool {
    if path == "." {
        return true;
    }
    let mut parts: Vec<&str> = path.split('/').collect();
    while parts.first() == Some(&"..") {
        parts.remove(0);
    }
    !parts.is_empty() && is_safe_relative(&parts.join("/")) && !path.contains('#')
}

/// A version text that selects one version: no range, no wildcard, no word that moves.
fn is_exact_version(version: &str) -> bool {
    version != "local"
        && version != "latest"
        && !version
            .chars()
            .any(|c| c.is_whitespace() || "*^~<>=|".contains(c))
}

/// Schema, then the lock's own rules: unique addresses, a root at "$" that is not indirect, safe local urls,
/// a local version is "local", exact other versions, pointer rules, no dangling dependency, no cycle, nothing unreachable.
///
/// This is the whole check of a lock on its own. The package reader makes it in two steps,
/// `check_lock_shape` and then `check_dependencies`, so only the tests call it.
#[cfg(test)]
pub(crate) fn check_lock(document: &Value, subject: &str) -> PackageResult<Lock> {
    let lock = check_lock_shape(document, subject)?;
    check_dependencies(&lock)?;
    Ok(lock)
}

/// Every dependency names an entry of the lock.
pub(crate) fn check_dependencies(lock: &Lock) -> PackageResult<()> {
    for (address, entry) in &lock.entries {
        if let Some(missing) = entry
            .dependencies
            .iter()
            .find(|dependency| !lock.entries.contains_key(*dependency))
        {
            return Err(invalid(
                address,
                format!("it depends on `{missing}`, and the lock has no such entry"),
            ));
        }
    }
    Ok(())
}

/// `check_lock` without the rule that every dependency names an entry.
///
/// The package reader makes that rule itself, after it has looked for a part the documents
/// reach and the lock does not pin: such a part is then named as missing from the lock.
pub(crate) fn check_lock_shape(document: &Value, subject: &str) -> PackageResult<Lock> {
    let Some(object) = document.as_object() else {
        return Err(invalid(subject, "the lock is not a JSON object"));
    };
    if object.get("$schema").and_then(Value::as_str) != Some(schema_id(SchemaKind::Lock)) {
        return Err(PackageError::new(
            PackageErrorCode::SchemaVersion,
            subject,
            unsupported_schema_message(object.get("$schema"), schema_id(SchemaKind::Lock)),
        ));
    }
    if let Some(failure) = first_failure(SchemaKind::Lock, document) {
        return Err(invalid(
            format!("{subject}#{}", failure.pointer),
            format!("{}: {}", failure.keyword, failure.message),
        ));
    }
    let listed: Vec<LockEntry> = serde_json::from_value(object["entries"].clone())
        .map_err(|error| invalid(subject, format!("an entry cannot be read: {error}")))?;

    let mut entries = BTreeMap::new();
    for entry in listed {
        let address = entry.address.clone();
        if entries.insert(address.clone(), entry).is_some() {
            return Err(invalid(address, "two entries have this address"));
        }
    }
    if !entries.contains_key("$") {
        return Err(invalid("$", "the lock has no root entry at `$`"));
    }
    for (address, entry) in &entries {
        check_entry(address, entry)?;
    }
    check_reach(&entries)?;
    Ok(Lock {
        generator: object
            .get("generator")
            .and_then(Value::as_str)
            .map(str::to_string),
        entries,
    })
}

fn check_entry(address: &str, entry: &LockEntry) -> PackageResult<()> {
    let source = &entry.source;
    if address == "$" && entry.indirect {
        return Err(invalid(address, "the root entry is marked indirect"));
    }
    if source.source_type == SourceType::Registry && source.registry.is_none() {
        return Err(invalid(
            address,
            "a registry source does not name the registry it was resolved through",
        ));
    }
    if source.source_type == SourceType::Local {
        // Relative to the folder that holds the lock file.
        let safe = source
            .url
            .as_deref()
            .and_then(|url| url.strip_prefix("./"))
            .is_some_and(is_local_lock_path);
        if !safe {
            return Err(invalid(
                address,
                "a local url begins with `./` and is a plain relative path",
            ));
        }
        if entry.version != "local" {
            return Err(invalid(
                address,
                "the version of a local source is the word `local`",
            ));
        }
    } else if !is_exact_version(&entry.version) {
        return Err(invalid(
            address,
            format!("`{}` is not one exact version", entry.version),
        ));
    }
    if let Some(subpath) = &source.subpath {
        let folder = subpath.split('#').next().unwrap_or("");
        if !is_safe_relative(folder) {
            return Err(invalid(address, "the subpath is not a plain relative path"));
        }
    }
    if source.source_type == SourceType::Pointer {
        if !matches!(
            entry.entry_type,
            EntryType::Model | EntryType::Connector | EntryType::Agent
        ) {
            return Err(invalid(
                address,
                "only a model, a connector and an agent can be a pointer",
            ));
        }
        // The schema requires an interface on a pointer.
        let interface = entry
            .interface
            .as_ref()
            .ok_or_else(|| invalid(address, "a pointer has no interface pin"))?;
        if interface.declared_version != entry.version {
            return Err(invalid(
                address,
                "the declared version of a pointer is not its version",
            ));
        }
        if entry.entry_type == EntryType::Connector && interface.tools_hash.is_none() {
            return Err(invalid(
                address,
                "a connector that is already running has no hash of its tools",
            ));
        }
        if entry.entry_type == EntryType::Agent && interface.card_digest.is_none() {
            return Err(invalid(
                address,
                "an agent that is already running has no digest of its card",
            ));
        }
    }
    if entry.entry_type == EntryType::Model && source.source_type != SourceType::Pointer {
        return Err(invalid(address, "a model is a pointer"));
    }
    Ok(())
}

/// No entry reaches itself, and the root reaches every entry. A dependency that names no
/// entry is left to `check_dependencies`.
fn check_reach(entries: &BTreeMap<String, LockEntry>) -> PackageResult<()> {
    let mut visited = BTreeSet::new();
    let mut active: Vec<&str> = Vec::new();
    // (address, the next dependency to look at). An explicit stack: a lock may be deep.
    let mut stack: Vec<(&str, usize)> = vec![("$", 0)];
    active.push("$");
    while let Some((address, next)) = stack.pop() {
        let dependencies = &entries[address].dependencies;
        if next == dependencies.len() {
            visited.insert(address);
            active.pop();
            continue;
        }
        stack.push((address, next + 1));
        let child = dependencies[next].as_str();
        if !entries.contains_key(child) {
            continue;
        }
        if active.contains(&child) {
            return Err(invalid(
                child,
                "the entry depends on itself through other entries",
            ));
        }
        if !visited.contains(child) {
            active.push(child);
            stack.push((child, 0));
        }
    }
    if let Some(unreached) = entries
        .keys()
        .find(|address| !visited.contains(address.as_str()))
    {
        return Err(invalid(
            unreached.clone(),
            "the root does not reach this entry",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const DIGEST: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn lock(entries: Value) -> Value {
        json!({"$schema": schema_id(SchemaKind::Lock), "type": "lock", "entries": entries})
    }

    fn root(dependencies: Value) -> Value {
        json!({
            "address": "$", "type": "agent", "source": {"type": "local", "url": "./agent.json"},
            "version": "local", "digest": DIGEST, "dependencies": dependencies
        })
    }

    fn skill(address: &str) -> Value {
        json!({
            "address": address, "type": "skill",
            "source": {"type": "local", "url": "./skills/review"},
            "version": "local", "digest": DIGEST, "dependencies": []
        })
    }

    fn refusal(document: Value) -> PackageError {
        check_lock(&document, "oadl.lock.json").expect_err("refused")
    }

    /// A lock with a root and one more entry, after `change` edits that entry.
    fn with_entry(mut entry: Value, change: impl FnOnce(&mut Value)) -> Value {
        change(&mut entry);
        let address = entry["address"].clone();
        lock(json!([root(json!([address])), entry]))
    }

    #[test]
    fn a_lock_with_every_kind_of_entry_is_read() {
        let document = json!({
            "$schema": schema_id(SchemaKind::Lock),
            "type": "lock",
            "generator": "example/1.0",
            "entries": [
                root(json!(["$/agents/translator", "$/connectors/assets", "$/model", "$/packages/whisper", "$/skills/review"])),
                {"address": "$/agents/translator", "type": "agent", "source": {"type": "pointer"}, "version": "2.1.0",
                 "constraint": "2.1.0", "interface": {"declared_version": "2.1.0", "card_digest": DIGEST}, "dependencies": []},
                {"address": "$/connectors/assets", "type": "connector", "source": {"type": "pointer"}, "version": "2.3.0",
                 "interface": {"declared_version": "2.3.0", "tools_hash": DIGEST}, "dependencies": []},
                {"address": "$/model", "type": "model", "source": {"type": "pointer"}, "version": "5a408ab55df5",
                 "interface": {"declared_version": "5a408ab55df5"}, "dependencies": []},
                {"address": "$/packages/whisper", "type": "package",
                 "source": {"type": "registry", "url": "faster-whisper", "registry": "pypi"},
                 "version": "1.2.1", "constraint": "1.2.1", "digest": DIGEST, "dependencies": []},
                {"address": "$/skills/review", "type": "skill",
                 "source": {"type": "git", "url": "https://git.example.com/parts.git", "subpath": "skills/review"},
                 "version": "1.7.2", "constraint": "~> 1.7", "commit": COMMIT, "digest": DIGEST,
                 "retracted": "superseded", "dependencies": []}
            ]
        });
        let lock = check_lock(&document, "oadl.lock.json").expect("valid lock");
        assert_eq!(lock.generator.as_deref(), Some("example/1.0"));
        assert_eq!(lock.entries.len(), 6);
        let skill = &lock.entries["$/skills/review"];
        assert_eq!(skill.entry_type, EntryType::Skill);
        assert_eq!(skill.source.source_type, SourceType::Git);
        assert_eq!(skill.source.subpath.as_deref(), Some("skills/review"));
        assert_eq!(skill.commit.as_deref(), Some(COMMIT));
        assert_eq!(skill.constraint.as_deref(), Some("~> 1.7"));
        assert_eq!(skill.retracted.as_deref(), Some("superseded"));
        assert!(!skill.indirect);
        let translator = &lock.entries["$/agents/translator"];
        assert_eq!(
            translator
                .interface
                .as_ref()
                .and_then(|i| i.card_digest.as_deref()),
            Some(DIGEST)
        );
    }

    #[test]
    fn the_schema_is_checked_first() {
        let mut other = lock(json!([root(json!([]))]));
        other["$schema"] = json!("https://example.com/lock.schema.json");
        assert_eq!(refusal(other).code, PackageErrorCode::SchemaVersion);

        let error = refusal(json!([]));
        assert_eq!(error.code, PackageErrorCode::LockInvalid);

        // No digest on an entry that is not a pointer.
        let error = refusal(with_entry(skill("$/skills/review"), |entry| {
            entry.as_object_mut().expect("entry").remove("digest");
        }));
        assert_eq!(error.code, PackageErrorCode::LockInvalid);
        assert_eq!(error.subject, "oadl.lock.json#/entries/1");

        // No root entry: the schema asks for one entry at `$`.
        let error = refusal(lock(json!([skill("$/skills/review")])));
        assert_eq!(error.code, PackageErrorCode::LockInvalid);
    }

    #[test]
    fn two_entries_with_one_address_are_refused() {
        let error = refusal(lock(json!([
            root(json!(["$/skills/review"])),
            skill("$/skills/review"),
            skill("$/skills/review")
        ])));
        assert_eq!(error.code, PackageErrorCode::LockInvalid);
        assert_eq!(error.subject, "$/skills/review");
    }

    #[test]
    fn an_indirect_root_is_refused() {
        let mut entry = root(json!([]));
        entry["indirect"] = json!(true);
        let error = refusal(lock(json!([entry])));
        assert_eq!(error.subject, "$");
        assert!(error.message.contains("indirect"), "{}", error.message);
    }

    #[test]
    fn a_local_entry_has_a_safe_url_and_the_version_local() {
        for url in [
            "skills/review",
            "/skills/review",
            "./",
            "./a/../b",
            "./a//b",
            "./a#b",
            "./..",
            "./a\\b",
        ] {
            let error = refusal(with_entry(skill("$/skills/review"), |entry| {
                entry["source"]["url"] = json!(url);
            }));
            assert_eq!(error.code, PackageErrorCode::LockInvalid, "{url}");
            assert_eq!(error.subject, "$/skills/review", "{url}");
        }
        for url in ["./.", "./../../agent.json", "./skills/review"] {
            check_lock(
                &with_entry(skill("$/skills/review"), |entry| {
                    entry["source"]["url"] = json!(url);
                }),
                "oadl.lock.json",
            )
            .unwrap_or_else(|error| panic!("{url}: {error}"));
        }
        let error = refusal(with_entry(skill("$/skills/review"), |entry| {
            entry["version"] = json!("1.0.0");
        }));
        assert!(error.message.contains("local"), "{}", error.message);
    }

    #[test]
    fn another_entry_has_one_exact_version() {
        let git = json!({
            "address": "$/skills/review", "type": "skill",
            "source": {"type": "git", "url": "https://git.example.com/parts.git", "subpath": "skills/review"},
            "version": "1.7.2", "commit": COMMIT, "digest": DIGEST, "dependencies": []
        });
        check_lock(&with_entry(git.clone(), |_| {}), "oadl.lock.json").expect("exact version");
        for version in [
            "local",
            "latest",
            "~> 1.7",
            ">= 1.0",
            "1.*",
            "^1.2",
            "1.0 || 2.0",
            "1.0 ",
        ] {
            let error = refusal(with_entry(git.clone(), |entry| {
                entry["version"] = json!(version);
            }));
            assert_eq!(error.code, PackageErrorCode::LockInvalid, "{version}");
        }
        let error = refusal(with_entry(git.clone(), |entry| {
            entry["source"]["subpath"] = json!("../outside");
        }));
        assert!(error.message.contains("subpath"), "{}", error.message);
        let error = refusal(with_entry(git, |entry| {
            entry["source"] = json!({"type": "registry", "url": "review"});
            entry.as_object_mut().expect("entry").remove("commit");
        }));
        assert!(error.message.contains("registry"), "{}", error.message);
    }

    #[test]
    fn a_pointer_follows_the_pointer_rules() {
        let pointer = json!({
            "address": "$/connectors/assets", "type": "connector", "source": {"type": "pointer"},
            "version": "2.3.0", "interface": {"declared_version": "2.3.0", "tools_hash": DIGEST},
            "dependencies": []
        });
        check_lock(&with_entry(pointer.clone(), |_| {}), "oadl.lock.json").expect("valid pointer");
        let cases: [(&str, Box<dyn Fn(&mut Value)>); 5] = [
            (
                "only a model",
                Box::new(|entry| entry["type"] = json!("skill")),
            ),
            (
                "declared version",
                Box::new(|entry| entry["version"] = json!("2.4.0")),
            ),
            (
                "hash of its tools",
                Box::new(|entry| {
                    entry["interface"] = json!({"declared_version": "2.3.0"});
                }),
            ),
            (
                "digest of its card",
                Box::new(|entry| entry["type"] = json!("agent")),
            ),
            (
                "a model is a pointer",
                Box::new(|entry| {
                    *entry = json!({
                        "address": "$/connectors/assets", "type": "model",
                        "source": {"type": "oci", "url": "registry.example.com/model"},
                        "version": "1.0", "digest": DIGEST, "dependencies": []
                    });
                }),
            ),
        ];
        for (words, change) in cases {
            let error = refusal(with_entry(pointer.clone(), change));
            assert_eq!(error.code, PackageErrorCode::LockInvalid, "{words}");
            assert!(error.message.contains(words), "{words}: {}", error.message);
        }
    }

    #[test]
    fn a_dangling_dependency_a_cycle_and_an_unreachable_entry_are_refused() {
        let error = refusal(lock(json!([root(json!(["$/skills/none"]))])));
        assert!(error.message.contains("no such entry"), "{}", error.message);

        let mut first = skill("$/skills/a");
        first["dependencies"] = json!(["$/skills/b"]);
        let mut second = skill("$/skills/b");
        second["dependencies"] = json!(["$/skills/a"]);
        let error = refusal(lock(json!([root(json!(["$/skills/a"])), first, second])));
        assert!(
            error.message.contains("depends on itself"),
            "{}",
            error.message
        );

        let error = refusal(lock(json!([root(json!([])), skill("$/skills/review")])));
        assert_eq!(error.subject, "$/skills/review");
        assert!(
            error.message.contains("does not reach"),
            "{}",
            error.message
        );
    }

    #[test]
    fn a_part_with_two_parents_is_not_a_cycle() {
        let mut helper = json!({
            "address": "$/agents/helper", "type": "agent",
            "source": {"type": "local", "url": "./agents/helper/agent.json"},
            "version": "local", "digest": DIGEST, "dependencies": ["$/agents/helper/skills/review"]
        });
        helper["dependencies"] = json!(["$/agents/helper/skills/review"]);
        let document = lock(json!([
            root(json!(["$/agents/helper", "$/agents/helper/skills/review"])),
            helper,
            skill("$/agents/helper/skills/review")
        ]));
        check_lock(&document, "oadl.lock.json").expect("a shared part");
    }
}
