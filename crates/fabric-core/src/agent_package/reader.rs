// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read one package: entry document, lock, parts, variables, definition.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde_json::{Map, Value};

use super::called::called_under;
use super::definition::read_together;
use super::digest::bytes_digest;
use super::document::{check_document, var_name};
use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::graph::{AGENT_FILE, GraphReading, LOCK_FILE, verify_graph};
use super::json::{MAX_DOCUMENT_BYTES, parse_document, read_regular_file};
use super::lock::check_lock_shape;
use super::model::{
    CalledAgent, DocumentKind, DocumentRecord, Integrity, ListedAgent, LockRecord, Note, NoteCode,
    PackageReading, ReadRequest, SourceType, ValueOrigin,
};
use super::paths::{address, display_inside, resolve_inside};
use super::variables::resolve_variables;

/// The entry document of a definition.
const DEFINITION_FILE: &str = "agent-definition.json";

fn refuse(
    code: PackageErrorCode,
    subject: impl Into<String>,
    message: impl Into<String>,
) -> PackageError {
    PackageError::new(code, subject, message)
}

fn name_of(document: &Value) -> String {
    document
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// A path the caller gave, as relative text with `/` between segments.
fn relative_text(path: &Path) -> Option<String> {
    let mut segments = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(name) => segments.push(name.to_str()?),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (!segments.is_empty()).then(|| segments.join("/"))
}

/// A package whose documents, lock and pins are checked, before any variable has a value.
pub(crate) struct CheckedPackage {
    /// Absolute. The boundary for every local path.
    pub package_dir: PathBuf,
    /// The folder of the entry document and its lock.
    pub entry_dir: PathBuf,
    pub entry: DocumentRecord,
    pub lock: LockRecord,
    pub graph: GraphReading,
    entry_document: Value,
}

/// Get the package, read and check the entry document and the lock, and check every part
/// against the lock. No variable is read, so no value from whoever starts the agent is needed.
pub(crate) fn check_package(
    package_dir: &Path,
    request: &ReadRequest,
) -> PackageResult<CheckedPackage> {
    // 1. The package folder. It is the boundary for every local path.
    let shown = package_dir.display().to_string();
    let package_dir = std::path::absolute(package_dir)
        .map_err(|error| refuse(PackageErrorCode::NotFound, &shown, error.to_string()))?;
    if !fs::metadata(&package_dir).is_ok_and(|metadata| metadata.is_dir()) {
        return Err(refuse(
            PackageErrorCode::NotFound,
            &shown,
            "the package folder is missing or is not a folder",
        ));
    }
    let inside = |path: &Path| display_inside(&package_dir, path);

    // 2. The lock sits beside the entry document.
    let named_entry = match &request.entry {
        None => None,
        Some(entry) => {
            let relative = relative_text(entry).ok_or_else(|| {
                refuse(
                    PackageErrorCode::PathUnsafe,
                    entry.display().to_string(),
                    "the entry document is named by a relative path inside the package",
                )
            })?;
            Some(resolve_inside(
                &package_dir,
                &package_dir,
                &relative,
                &relative,
            )?)
        }
    };
    let entry_dir = named_entry
        .as_deref()
        .and_then(Path::parent)
        .unwrap_or(&package_dir)
        .to_path_buf();
    let lock_path = entry_dir.join(LOCK_FILE);
    let lock_subject = inside(&lock_path);
    if fs::symlink_metadata(&lock_path).is_err() {
        return Err(refuse(
            PackageErrorCode::LockMissing,
            inside(&entry_dir),
            format!("the folder has no `{LOCK_FILE}` beside the entry document"),
        ));
    }
    let lock_bytes = read_regular_file(&lock_path, &lock_subject, MAX_DOCUMENT_BYTES)?;
    let lock_sha256 = bytes_digest(&lock_bytes)
        .trim_start_matches("sha256:")
        .to_string();
    // The lock sits in the folder it describes, so whoever can write a part can
    // write the lock. A digest the caller holds from elsewhere anchors it.
    if let Some(expected) = &request.expected_lock_sha256 {
        check_expected_lock(expected, &lock_sha256, &lock_subject)?;
    }
    let lock_document = parse_document(&lock_bytes, &lock_subject)?;
    // A dependency that names no entry is refused by the graph check, after it has looked
    // for parts the documents reach and the lock does not pin.
    let lock = check_lock_shape(&lock_document, &lock_subject)?;

    // 3. The entry document: the one the caller named, else the one the lock's root names.
    let entry_path = match named_entry {
        Some(path) => path,
        None => {
            let url = lock.entries["$"].source.url.as_deref().unwrap_or_default();
            let name = url.strip_prefix("./").unwrap_or(url);
            if name != AGENT_FILE && name != DEFINITION_FILE {
                return Err(refuse(
                    PackageErrorCode::LockEntryMismatch,
                    "$",
                    format!(
                        "the root entry names `{url}`: the entry document at the root of a package is `{AGENT_FILE}` or `{DEFINITION_FILE}`"
                    ),
                ));
            }
            resolve_inside(&package_dir, &entry_dir, name, name)?
        }
    };
    let entry_subject = inside(&entry_path);
    if !fs::symlink_metadata(&entry_path).is_ok_and(|metadata| metadata.is_file()) {
        return Err(refuse(
            PackageErrorCode::EntryMissing,
            &entry_subject,
            "the entry document is not in the package",
        ));
    }
    let entry_bytes = read_regular_file(&entry_path, &entry_subject, MAX_DOCUMENT_BYTES)?;
    let entry_digest = bytes_digest(&entry_bytes);
    let entry_document = parse_document(&entry_bytes, &entry_subject)?;
    let kind = check_document(&entry_document, &entry_subject)?;

    // 4. Every part against the lock.
    let graph = verify_graph(&package_dir, &entry_path, &entry_document, &lock)?;
    let root_digest = graph
        .parts
        .iter()
        .find(|part| part.address == "$")
        .and_then(|part| part.digest.as_deref());
    if root_digest != Some(entry_digest.as_str()) {
        // The graph check hashed the file itself. The bytes that were parsed must be those.
        return Err(refuse(
            PackageErrorCode::DigestMismatch,
            "$",
            format!("`{entry_subject}` changed while the package was read"),
        ));
    }
    Ok(CheckedPackage {
        package_dir,
        entry_dir,
        entry: DocumentRecord {
            kind,
            name: name_of(&entry_document),
            address: "$".to_string(),
            path: entry_path,
            digest: entry_digest,
        },
        lock: LockRecord {
            path: lock_path,
            sha256: lock_sha256,
            generator: lock.generator.clone(),
            entries: lock.entries.len(),
        },
        graph,
        entry_document,
    })
}

/// Steps 1 to 4 of what a receiver does: get the package, read and check the entry document,
/// check every part against the lock, take variable values and read a definition with its agent.
pub(crate) fn read_package(
    package_dir: &Path,
    request: &ReadRequest,
) -> PackageResult<PackageReading> {
    let CheckedPackage {
        package_dir,
        entry_dir,
        entry,
        lock,
        mut graph,
        entry_document,
    } = check_package(package_dir, request)?;
    let kind = entry.kind;

    // 5. Variable values, and a definition read together with its agent.
    let mut notes = std::mem::take(&mut graph.notes);
    let (agent_record, definition, agent_as_written, agent, variables, together) = match kind {
        DocumentKind::Agent => {
            let resolved = resolve_variables(
                &entry_document,
                &request.variables,
                None,
                ValueOrigin::Entry,
            )?;
            (
                entry.clone(),
                None,
                entry_document,
                resolved.agent,
                resolved.values,
                None,
            )
        }
        DocumentKind::AgentDefinition => {
            let pinned = graph.parts.iter().find(|part| part.address == "$/agent");
            let Some(agent_document) = graph.documents.get("$/agent").cloned() else {
                let why = match pinned.map(|part| part.integrity) {
                    Some(Integrity::NotVerifiable) => {
                        "it is pinned by an artifact digest, which cannot be worked out from files"
                    }
                    Some(Integrity::Absent) => {
                        "its content is not in the package, and the loader fetches nothing"
                    }
                    _ => "its document was not read",
                };
                return Err(refuse(
                    PackageErrorCode::AgentUnavailable,
                    "$/agent",
                    format!("the agent the definition points at cannot be used: {why}"),
                ));
            };
            let agent_path = pinned
                .and_then(|part| part.path.clone())
                .unwrap_or_default();
            // The same `./path` in the two documents leads to one folder only when both
            // documents are stored in the same one.
            let one_folder = pinned.is_some_and(|part| part.source_type == SourceType::Local)
                && agent_path.parent() == Some(entry_dir.as_path());
            let together = read_together(
                &entry_document,
                &agent_document,
                &request.variables,
                one_folder,
            )?;
            notes.extend(together.notes);
            (
                DocumentRecord {
                    kind: DocumentKind::Agent,
                    name: name_of(&agent_document),
                    address: "$/agent".to_string(),
                    path: agent_path,
                    digest: pinned
                        .and_then(|part| part.digest.clone())
                        .unwrap_or_default(),
                },
                Some(entry_document),
                agent_document,
                together.agent,
                together.values,
                Some(together.summary),
            )
        }
    };

    // The agents the agent that runs lists, and the agents those list.
    let mut called = Vec::new();
    let owners: Vec<&str> = if agent_record.address == entry.address {
        vec!["$"]
    } else {
        vec![agent_record.address.as_str(), "$"]
    };
    called_agents(&agent, &owners, &[], &graph, &mut called, &mut notes)?;

    // The values an entry gives are written out: a reference takes the value its owner has.
    let mut values_of: BTreeMap<&str, BTreeMap<&str, &Value>> = BTreeMap::new();
    let root_values = || -> BTreeMap<&str, &Value> {
        variables
            .iter()
            .map(|value| (value.name.as_str(), &value.value))
            .collect()
    };
    for owner in &owners {
        values_of.insert(owner, root_values());
    }
    for listed in &called {
        values_of.entry(listed.address.as_str()).or_insert_with(|| {
            listed
                .values
                .iter()
                .map(|value| (value.name.as_str(), &value.value))
                .collect()
        });
    }
    let mut parts = graph.parts;
    for part in &mut parts {
        let (Some(values), Some(owned)) = (&mut part.values, values_of.get(part.owner.as_str()))
        else {
            continue;
        };
        for value in values.values_mut() {
            if let Some(filled) = var_name(value).and_then(|name| owned.get(name)) {
                *value = (*filled).clone();
            }
        }
    }

    Ok(PackageReading {
        package_dir,
        entry_dir,
        entry,
        agent_record,
        lock,
        definition,
        agent_as_written,
        agent,
        variables,
        parts,
        left_out: graph.left_out,
        called,
        listings: graph.listings,
        together,
        notes,
    })
}

/// The lock file must have the sha256 the caller holds for it.
///
/// `expected` is 64 lowercase hex characters, with or without `sha256:` before them.
fn check_expected_lock(expected: &str, have: &str, lock_subject: &str) -> PackageResult<()> {
    let wanted = expected.strip_prefix("sha256:").unwrap_or(expected);
    let is_digest = wanted.len() == 64
        && wanted
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !is_digest {
        return Err(refuse(
            PackageErrorCode::OptionsInvalid,
            "expected_lock_sha256",
            "the expected lock digest is 64 lowercase hex characters, with or without `sha256:` before them",
        ));
    }
    if wanted != have {
        return Err(refuse(
            PackageErrorCode::LockDigestMismatch,
            lock_subject,
            format!("the lock file has sha256 {have}, and the caller expects {wanted}"),
        ));
    }
    Ok(())
}

/// What holds for each agent `caller` lists whose document was read, at every depth.
///
/// `owners` are the addresses the caller's references are found under. `place`
/// holds the names that lead to the caller from the agent that runs.
///
/// A called agent whose variable has no value, or a value that does not fit
/// its field, could not be started. That is a fact about the called agent, not
/// a reason to refuse its caller: it is noted, and the agents below it are not
/// walked. A value for an undeclared name, of another type or outside the
/// allowed values was already refused when the graph was checked.
fn called_agents(
    caller: &Value,
    owners: &[&str],
    place: &[String],
    graph: &GraphReading,
    called: &mut Vec<CalledAgent>,
    notes: &mut Vec<Note>,
) -> PackageResult<()> {
    let Some(listed) = caller.get("agents").and_then(Value::as_object) else {
        return Ok(());
    };
    // How many calls below the first agent the listed agents run.
    let depth = place.len() as u64 + 1;
    for (name, entry) in listed {
        let Some(target) = owners
            .iter()
            .find_map(|owner| graph.edges.get(&address(owner, &["agents", name])))
        else {
            continue;
        };
        // An agent that is already running, or one whose content is not here, has no document.
        let Some(document) = graph.documents.get(target) else {
            continue;
        };
        let given = entry.get("variables").and_then(Value::as_object);
        let started = called_under(caller, name, document, depth).and_then(|agent| {
            let below = resolve_variables(document, &Map::new(), given, ValueOrigin::Entry)?;
            Ok((agent, below.agent))
        });
        let (mut agent, mut below) = match started {
            Ok(started) => started,
            Err(error)
                if matches!(
                    error.code,
                    PackageErrorCode::VariableMissing | PackageErrorCode::VariableValueInvalid
                ) =>
            {
                notes.push(Note {
                    code: NoteCode::CalledAgentCannotStart,
                    subject: target.clone(),
                    text: format!(
                        "the called agent `{name}` could not be started: {}: {}",
                        error.subject, error.message
                    ),
                });
                continue;
            }
            Err(error) => return Err(error),
        };
        agent.address = target.clone();
        agent.place = place.iter().cloned().chain([name.clone()]).collect();
        agent.agents = listed_agents(&below, target, graph);

        // The called agent as it runs is the caller of the agents it lists: its limits are
        // the ones that apply to it, and its model is the one it takes.
        below["limits"] = Value::Object(agent.limits.clone());
        if below.pointer("/model/source").is_none()
            && let Some(source) = &agent.model_source
        {
            if !below.get("model").is_some_and(Value::is_object) {
                below["model"] = Value::Object(Map::new());
            }
            below["model"]["source"] = Value::String(source.clone());
        }
        let place = agent.place.clone();
        called.push(agent);
        called_agents(&below, &[target.as_str()], &place, graph, called, notes)?;
    }
    Ok(())
}

/// The agents one called agent lists, each with the lock entry it leads to.
fn listed_agents(document: &Value, owner: &str, graph: &GraphReading) -> Vec<ListedAgent> {
    let Some(listed) = document.get("agents").and_then(Value::as_object) else {
        return Vec::new();
    };
    listed
        .iter()
        .map(|(name, entry)| ListedAgent {
            name: name.clone(),
            optional: entry.get("optional").and_then(Value::as_bool) == Some(true),
            address: graph.edges.get(&address(owner, &["agents", name])).cloned(),
        })
        .collect()
}

/// Builds small packages for tests: files, and a lock with the digests of what was written.
#[cfg(test)]
pub(crate) mod testing {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::json;

    use super::super::digest::{bytes_digest, folder_listing};
    use super::super::schemas::{SchemaKind, schema_id};
    use super::*;

    pub(crate) const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    pub(crate) const MADE_UP: &str =
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    static NEXT: AtomicU64 = AtomicU64::new(0);

    /// A package folder that is removed when the value is dropped.
    pub(crate) struct Package {
        pub root: PathBuf,
    }

    impl Drop for Package {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    impl Package {
        pub(crate) fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "nemo-fabric-agent-package-{name}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).expect("create package folder");
            Self { root }
        }

        pub(crate) fn write(&self, path: &str, content: impl AsRef<[u8]>) {
            let file = self.root.join(path);
            fs::create_dir_all(file.parent().expect("parent")).expect("create folder");
            fs::write(file, content).expect("write file");
        }

        /// Writes a JSON document the way an author's tool would.
        pub(crate) fn write_json(&self, path: &str, document: &Value) {
            let mut text = serde_json::to_string_pretty(document).expect("serialize");
            text.push('\n');
            self.write(path, text);
        }

        pub(crate) fn file_digest(&self, path: &str) -> String {
            bytes_digest(&fs::read(self.root.join(path)).expect("read file"))
        }

        pub(crate) fn folder_digest(&self, path: &str) -> String {
            folder_listing(&self.root.join(path), None, path)
                .expect("folder listing")
                .digest
        }

        /// A lock entry for a document or a single file stored in the package.
        pub(crate) fn file_entry(
            &self,
            at: &str,
            kind: &str,
            path: &str,
            url: &str,
            dependencies: &[&str],
        ) -> Value {
            json!({
                "address": at, "type": kind, "source": {"type": "local", "url": url},
                "version": "local", "digest": self.file_digest(path), "dependencies": dependencies
            })
        }

        /// A lock entry for a folder stored in the package.
        pub(crate) fn folder_entry(&self, at: &str, kind: &str, path: &str, url: &str) -> Value {
            json!({
                "address": at, "type": kind, "source": {"type": "local", "url": url},
                "version": "local", "digest": self.folder_digest(path), "dependencies": []
            })
        }

        pub(crate) fn write_lock(&self, folder: &str, entries: Vec<Value>) {
            let path = if folder.is_empty() {
                LOCK_FILE.to_string()
            } else {
                format!("{folder}/{LOCK_FILE}")
            };
            self.write_json(
                &path,
                &json!({
                    "$schema": schema_id(SchemaKind::Lock),
                    "type": "lock",
                    "generator": "tests/1",
                    "entries": entries
                }),
            );
        }

        pub(crate) fn read(&self) -> PackageResult<PackageReading> {
            read_package(&self.root, &ReadRequest::default())
        }

        pub(crate) fn read_with(
            &self,
            entry: Option<&str>,
            variables: Value,
        ) -> PackageResult<PackageReading> {
            read_package(
                &self.root,
                &ReadRequest {
                    entry: entry.map(PathBuf::from),
                    variables: variables.as_object().cloned().unwrap_or_default(),
                    ..ReadRequest::default()
                },
            )
        }
    }

    pub(crate) fn agent(name: &str, more: Value) -> Value {
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

    pub(crate) fn definition(name: &str, more: Value) -> Value {
        let mut document = json!({
            "$schema": schema_id(SchemaKind::AgentDefinition),
            "type": "agent-definition",
            "name": name
        });
        document
            .as_object_mut()
            .expect("object")
            .extend(more.as_object().expect("object").clone());
        document
    }

    /// A lock entry for something fetched from git.
    pub(crate) fn git_entry(
        at: &str,
        kind: &str,
        subpath: &str,
        version: &str,
        constraint: &str,
        digest: &str,
    ) -> Value {
        json!({
            "address": at, "type": kind,
            "source": {"type": "git", "url": "https://git.example.com/parts.git", "subpath": subpath},
            "version": version, "constraint": constraint, "commit": COMMIT, "digest": digest,
            "dependencies": []
        })
    }

    pub(crate) const SKILL: &str =
        "---\nname: review\ndescription: Review a render\n---\nReport every item that fails.\n";

    /// One agent with one skill and one carried connector, and its lock.
    pub(crate) fn small_package(name: &str) -> Package {
        let package = Package::new(name);
        package.write_json(
            "agent.json",
            &agent(
                "release-notes",
                json!({
                    "instructions": "Draft release notes.",
                    "skills": {"review": {"source": "./skills/review"}},
                    "connectors": {"changelog": {"source": "./connectors/changelog"}},
                    "tools": [{"connector": "changelog", "tool": "list_changes"}]
                }),
            ),
        );
        package.write("skills/review/SKILL.md", SKILL);
        package.write(
            "connectors/changelog/mcp.json",
            r#"{"mcpServers": {"changelog": {"type": "stdio", "command": "python3", "args": ["server.py"]}}}"#,
        );
        package.write("connectors/changelog/server.py", "print('ready')\n");
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
        package
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::model::{DigestKind, EntryType, NoteCode, PartField, StatedBy};
    use super::testing::*;
    use super::*;

    fn code(result: PackageResult<PackageReading>) -> (PackageErrorCode, String) {
        let error = result.expect_err("refused");
        (error.code, error.subject)
    }

    #[test]
    fn a_package_is_read_and_every_local_part_is_verified() {
        let package = small_package("read");
        let reading = package.read().expect("package reads");
        assert_eq!(reading.entry.kind, DocumentKind::Agent);
        assert_eq!(reading.entry.name, "release-notes");
        assert_eq!(reading.entry.address, "$");
        assert_eq!(reading.entry.digest, package.file_digest("agent.json"));
        assert_eq!(reading.agent_record, reading.entry);
        assert_eq!(reading.entry_dir, reading.package_dir);
        assert!(reading.package_dir.is_absolute());
        assert_eq!(reading.lock.entries, 3);
        assert_eq!(reading.lock.generator.as_deref(), Some("tests/1"));
        assert_eq!(
            format!("sha256:{}", reading.lock.sha256),
            package.file_digest("oadl.lock.json")
        );
        assert!(reading.definition.is_none() && reading.together.is_none());
        assert_eq!(reading.agent, reading.agent_as_written);

        let addresses: Vec<&str> = reading
            .parts
            .iter()
            .map(|part| part.address.as_str())
            .collect();
        assert_eq!(
            addresses,
            ["$", "$/connectors/changelog", "$/skills/review"]
        );
        assert!(
            reading
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );

        let skill = reading.part(PartField::Skills, "review").expect("skill");
        assert_eq!(skill.entry_type, EntryType::Skill);
        assert_eq!(skill.field, Some(PartField::Skills));
        assert_eq!(skill.name.as_deref(), Some("review"));
        assert_eq!(skill.owner, "$");
        assert_eq!(skill.stated_by, StatedBy::Agent);
        assert_eq!(skill.pointer.as_deref(), Some("/skills/review"));
        assert_eq!(skill.source_text.as_deref(), Some("./skills/review"));
        assert_eq!(skill.digest_kind, Some(DigestKind::Folder));
        assert_eq!(
            skill.path.as_deref(),
            Some(reading.package_dir.join("skills/review").as_path())
        );
        assert!(!skill.optional);

        let listing = &reading.listings["$/connectors/changelog"];
        let files: Vec<&str> = listing.files.keys().map(String::as_str).collect();
        assert_eq!(files, ["mcp.json", "server.py"]);
        assert_eq!(
            listing.files["mcp.json"],
            package.file_digest("connectors/changelog/mcp.json")
        );
        let root = &reading.parts[0];
        assert_eq!(
            (root.field, root.digest_kind),
            (None, Some(DigestKind::File))
        );
        assert_eq!(root.path.as_deref(), Some(reading.entry.path.as_path()));
    }

    #[test]
    fn the_same_package_reads_the_same_way_twice() {
        let package = small_package("twice");
        assert_eq!(
            package.read().expect("first"),
            package.read().expect("second")
        );
    }

    #[test]
    fn a_changed_byte_is_refused_with_the_address_it_belongs_to() {
        let package = small_package("tamper-skill");
        package.write("skills/review/SKILL.md", SKILL.replace("fails", "failz"));
        assert_eq!(
            code(package.read()),
            (
                PackageErrorCode::DigestMismatch,
                "$/skills/review".to_string()
            )
        );

        let package = small_package("tamper-extra");
        package.write("skills/review/extra.md", "more");
        assert_eq!(
            code(package.read()),
            (
                PackageErrorCode::DigestMismatch,
                "$/skills/review".to_string()
            )
        );

        let package = small_package("tamper-entry");
        let text = fs::read_to_string(package.root.join("agent.json")).expect("read");
        package.write(
            "agent.json",
            text.replace("Draft release notes.", "Draft release notez."),
        );
        assert_eq!(
            code(package.read()),
            (PackageErrorCode::DigestMismatch, "$".to_string())
        );
    }

    #[test]
    fn a_ds_store_file_in_a_pinned_folder_does_not_change_its_digest() {
        // The lock is written first. Then macOS Finder leaves its file in two folders.
        let package = small_package("finder");
        package.write("skills/review/.DS_Store", b"\0\0\0\x01Bud1");
        package.write("connectors/changelog/assets/.DS_Store", b"Bud1");

        let reading = package.read().expect("the package still reads");

        assert!(
            reading
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );
        // The file is not one of the folder's files.
        let files =
            |at: &str| -> Vec<String> { reading.listings[at].files.keys().cloned().collect() };
        assert_eq!(files("$/skills/review"), ["SKILL.md"]);
        assert_eq!(files("$/connectors/changelog"), ["mcp.json", "server.py"]);

        // A file with any other name is one of the folder's files.
        for name in [".ds_store", ".DS_Store.bak", "Thumbs.db"] {
            let package = small_package("finder-other");
            package.write(&format!("skills/review/{name}"), b"Bud1");
            assert_eq!(
                code(package.read()),
                (
                    PackageErrorCode::DigestMismatch,
                    "$/skills/review".to_string()
                ),
                "{name}"
            );
        }
    }

    #[test]
    fn a_part_that_is_one_file_named_ds_store_is_not_in_the_package() {
        // A file of that name is never part of a package. A document that names one as a
        // part names nothing, whatever the lock pins: the format's reference checker reads
        // such a reference as missing.
        let written = |name: &str, file: &str, optional: bool| {
            let package = Package::new(name);
            let source = format!("./policies/{file}");
            package.write_json(
                "agent.json",
                &agent(
                    "release-notes",
                    json!({"policies": {"finder": {"source": source, "optional": optional}}}),
                ),
            );
            let stored = format!("policies/{file}");
            package.write(&stored, "permit(principal, action, resource);\n");
            // An optional reference is left out of the lock here.
            let pinned: &[&str] = if optional {
                &[]
            } else {
                &["$/policies/finder"]
            };
            let mut entries =
                vec![package.file_entry("$", "agent", "agent.json", "./agent.json", pinned)];
            if !optional {
                entries.push(package.file_entry(
                    "$/policies/finder",
                    "policy",
                    &stored,
                    &source,
                    &[],
                ));
            }
            package.write_lock("", entries);
            package
        };

        let error = written("finder-named", ".DS_Store", false)
            .read()
            .expect_err("refused");
        assert_eq!(
            (error.code, error.subject.as_str()),
            (PackageErrorCode::PartMissing, "$/policies/finder")
        );
        assert!(error.message.contains("`.DS_Store`"), "{}", error.message);

        // The same bytes under any other name are a part like any other.
        let reading = written("finder-other-name", ".DS_Store.cedar", false)
            .read()
            .expect("the package reads");
        let policy = reading.part(PartField::Policies, "finder").expect("policy");
        assert_eq!(policy.integrity, Integrity::Verified);
        assert_eq!(policy.digest_kind, Some(DigestKind::File));

        // An optional reference that the lock leaves out is left out, as when no file is there.
        let reading = written("finder-optional", ".DS_Store", true)
            .read()
            .expect("the package reads");
        assert_eq!(reading.left_out.len(), 1);
        assert_eq!(reading.parts.len(), 1);
    }

    #[test]
    fn a_pinned_part_that_is_gone_is_refused() {
        let package = small_package("gone");
        fs::remove_dir_all(package.root.join("skills/review")).expect("remove");
        assert_eq!(
            code(package.read()),
            (PackageErrorCode::PartMissing, "$/skills/review".to_string())
        );
    }

    #[test]
    fn the_package_the_lock_and_the_entry_document_must_be_there() {
        let package = small_package("missing");
        assert_eq!(
            code(read_package(
                &package.root.join("none"),
                &ReadRequest::default()
            ))
            .0,
            PackageErrorCode::NotFound
        );
        assert_eq!(
            code(read_package(
                &package.root.join("agent.json"),
                &ReadRequest::default()
            ))
            .0,
            PackageErrorCode::NotFound
        );
        fs::remove_file(package.root.join("agent.json")).expect("remove");
        assert_eq!(
            code(package.read()),
            (PackageErrorCode::EntryMissing, "agent.json".to_string())
        );
        fs::remove_file(package.root.join(LOCK_FILE)).expect("remove");
        assert_eq!(
            code(package.read()),
            (PackageErrorCode::LockMissing, ".".to_string())
        );
    }

    #[test]
    fn the_lock_names_the_entry_document_or_the_caller_does() {
        // A lock whose root names another file: the caller says which document to load.
        let package = Package::new("named");
        package.write_json("configs/video.json", &agent("video", json!({})));
        package.write_lock(
            "configs",
            vec![package.file_entry("$", "agent", "configs/video.json", "./video.json", &[])],
        );
        let reading = package
            .read_with(Some("configs/video.json"), json!({}))
            .expect("named entry");
        assert_eq!(reading.entry_dir, reading.package_dir.join("configs"));
        assert_eq!(
            reading.lock.path,
            reading.package_dir.join("configs").join(LOCK_FILE)
        );

        // The package root has no lock.
        assert_eq!(code(package.read()).0, PackageErrorCode::LockMissing);
        // A lock at the root that names a file with another name is not followed.
        package.write_json("video.json", &agent("video", json!({})));
        package.write_lock(
            "",
            vec![package.file_entry("$", "agent", "video.json", "./video.json", &[])],
        );
        assert_eq!(
            code(package.read()),
            (PackageErrorCode::LockEntryMismatch, "$".to_string())
        );
        package
            .read_with(Some("video.json"), json!({}))
            .expect("named entry at the root");
        // The entry option stays inside the package.
        assert_eq!(
            code(package.read_with(Some("../outside.json"), json!({}))).0,
            PackageErrorCode::PathUnsafe
        );
        // The lock must be about the document that is loaded.
        assert_eq!(
            code(package.read_with(Some("configs/../video.json"), json!({}))).0,
            PackageErrorCode::PathUnsafe
        );
    }

    #[test]
    fn a_document_that_is_not_valid_is_refused_before_its_parts_are_read() {
        let package = small_package("invalid");
        package.write("agent.json", "{\"type\": \"agent\", \"type\": \"agent\"}");
        assert_eq!(
            code(package.read()),
            (PackageErrorCode::JsonDuplicateKey, "agent.json".to_string())
        );
        package.write_json(
            "agent.json",
            &agent("release-notes", json!({"sandbox": true})),
        );
        let (refusal, subject) = code(package.read());
        assert_eq!(refusal, PackageErrorCode::SchemaInvalid);
        assert_eq!(subject, "agent.json#");
        package.write(LOCK_FILE, "{}");
        assert_eq!(code(package.read()).0, PackageErrorCode::SchemaVersion);
    }

    #[test]
    fn variables_take_their_values_and_entry_values_are_written_out() {
        let package = Package::new("variables");
        package.write_json(
            "agent.json",
            &agent(
                "release-notes",
                json!({
                    "instructions": {"var": "job"},
                    "limits": {"max_turns": {"var": "max_items"}},
                    "variables": {
                        "job": {"type": "string"},
                        "max_items": {"type": "number", "default": 20}
                    },
                    "skills": {"review": {"source": "./skills/review", "variables": {"depth": {"var": "max_items"}, "mode": "full"}}}
                }),
            ),
        );
        package.write("skills/review/SKILL.md", SKILL);
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &["$/skills/review"],
                ),
                package.folder_entry(
                    "$/skills/review",
                    "skill",
                    "skills/review",
                    "./skills/review",
                ),
            ],
        );
        assert_eq!(
            code(package.read()),
            (PackageErrorCode::VariableMissing, "job".to_string())
        );
        assert_eq!(
            code(package.read_with(None, json!({"job": "x", "nope": 1}))),
            (PackageErrorCode::VariableUndeclared, "nope".to_string())
        );
        let reading = package
            .read_with(None, json!({"job": "Draft release notes.", "max_items": 5}))
            .expect("values given");
        assert_eq!(reading.agent["instructions"], "Draft release notes.");
        assert_eq!(reading.agent["limits"]["max_turns"], 5);
        assert_eq!(
            reading.agent_as_written["instructions"],
            json!({"var": "job"})
        );
        let origins: Vec<(&str, ValueOrigin)> = reading
            .variables
            .iter()
            .map(|value| (value.name.as_str(), value.origin))
            .collect();
        assert_eq!(
            origins,
            [
                ("job", ValueOrigin::Start),
                ("max_items", ValueOrigin::Start)
            ]
        );
        let skill = reading.part(PartField::Skills, "review").expect("skill");
        assert_eq!(
            skill
                .values
                .as_ref()
                .map(|values| Value::Object(values.clone())),
            Some(json!({"depth": 5, "mode": "full"}))
        );
    }

    /// A definition kept in a folder of its agent's package: it gives the job, adds one skill
    /// and lowers a limit.
    fn variant_package(name: &str) -> Package {
        let package = Package::new(name);
        package.write_json(
            "agent.json",
            &agent(
                "release-notes",
                json!({
                    "instructions": {"var": "job"},
                    "limits": {"max_turns": 60},
                    "variables": {"job": {"type": "string"}},
                    "skills": {"review": {"source": "./skills/review"}}
                }),
            ),
        );
        package.write("skills/review/SKILL.md", SKILL);
        package.write(
            "variants/short/skills/glossary/SKILL.md",
            "---\nname: glossary\n---\nUse the glossary.\n",
        );
        package.write_json(
            "variants/short/agent-definition.json",
            &definition(
                "short-notes",
                json!({
                    "agent": {"source": "./../../agent.json"},
                    "variables": {"job": "Draft short release notes."},
                    "instructions": "Keep it under ten lines.",
                    "limits": {"max_turns": 25, "timeout_seconds": 600},
                    "skills": {"glossary": {"source": "./skills/glossary"}}
                }),
            ),
        );
        package.write_lock(
            "variants/short",
            vec![
                package.file_entry(
                    "$",
                    "agent-definition",
                    "variants/short/agent-definition.json",
                    "./agent-definition.json",
                    &["$/agent", "$/skills/glossary"],
                ),
                package.file_entry(
                    "$/agent",
                    "agent",
                    "agent.json",
                    "./../../agent.json",
                    &["$/agent/skills/review"],
                ),
                package.folder_entry(
                    "$/agent/skills/review",
                    "skill",
                    "skills/review",
                    "./../../skills/review",
                ),
                package.folder_entry(
                    "$/skills/glossary",
                    "skill",
                    "variants/short/skills/glossary",
                    "./skills/glossary",
                ),
            ],
        );
        package
    }

    #[test]
    fn a_definition_is_read_together_with_its_agent() {
        let package = variant_package("definition");
        let reading = package
            .read_with(Some("variants/short/agent-definition.json"), json!({}))
            .expect("definition reads");
        assert_eq!(reading.entry.kind, DocumentKind::AgentDefinition);
        assert_eq!(reading.entry.name, "short-notes");
        assert_eq!(reading.agent_record.kind, DocumentKind::Agent);
        assert_eq!(reading.agent_record.name, "release-notes");
        assert_eq!(reading.agent_record.address, "$/agent");
        assert_eq!(
            reading.agent_record.path,
            reading.package_dir.join("agent.json")
        );
        assert_eq!(
            reading.agent_record.digest,
            package.file_digest("agent.json")
        );
        assert_eq!(
            reading.entry_dir,
            reading.package_dir.join("variants/short")
        );

        assert_eq!(
            reading.agent["instructions"],
            "Draft short release notes.\n\nKeep it under ten lines."
        );
        assert_eq!(
            reading.agent["limits"],
            json!({"max_turns": 25, "timeout_seconds": 600})
        );
        assert!(reading.agent["skills"].get("glossary").is_some());
        assert_eq!(
            reading.agent_as_written["instructions"],
            json!({"var": "job"})
        );
        assert_eq!(reading.variables[0].origin, ValueOrigin::Definition);
        let together = reading.together.as_ref().expect("together");
        assert_eq!(together.added, ["skills.glossary"]);
        assert!(together.kept_once.is_empty());

        let review = reading
            .part(PartField::Skills, "review")
            .expect("the agent's skill");
        assert_eq!(
            (review.address.as_str(), review.owner.as_str()),
            ("$/agent/skills/review", "$/agent")
        );
        assert_eq!(review.stated_by, StatedBy::Agent);
        let glossary = reading
            .part(PartField::Skills, "glossary")
            .expect("the added skill");
        assert_eq!(
            (glossary.address.as_str(), glossary.owner.as_str()),
            ("$/skills/glossary", "$")
        );
        assert_eq!(glossary.stated_by, StatedBy::Definition);
        assert_eq!(
            glossary.path.as_deref(),
            Some(
                reading
                    .package_dir
                    .join("variants/short/skills/glossary")
                    .as_path()
            )
        );
        assert!(
            reading
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );

        // The start wins over the definition.
        let reading = package
            .read_with(
                Some("variants/short/agent-definition.json"),
                json!({"job": "From the start."}),
            )
            .expect("definition reads");
        assert_eq!(reading.variables[0].origin, ValueOrigin::Start);
    }

    #[test]
    fn a_definition_that_reuses_a_name_of_its_agent_is_refused() {
        let package = variant_package("clash");
        package.write(
            "variants/short/skills/review/SKILL.md",
            "---\nname: other\n---\nAnother.\n",
        );
        let mut document: Value = serde_json::from_slice(
            &fs::read(package.root.join("variants/short/agent-definition.json")).expect("read"),
        )
        .expect("parse");
        document["skills"]["review"] = json!({"source": "./skills/review"});
        package.write_json("variants/short/agent-definition.json", &document);
        assert_eq!(
            code(package.read_with(Some("variants/short/agent-definition.json"), json!({}))),
            (
                PackageErrorCode::DefinitionNameInUse,
                "skills.review".to_string()
            )
        );
    }

    #[test]
    fn a_definition_whose_agent_is_pinned_by_an_artifact_digest_is_refused() {
        let package = Package::new("artifact-agent");
        package.write_json(
            "agent-definition.json",
            &definition(
                "ada",
                json!({"agent": {"source": "oci::registry.example.com/platform/media-desk", "version": "~> 3.0"}}),
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
                    &["$/agent"],
                ),
                json!({
                    "address": "$/agent", "type": "agent",
                    "source": {"type": "oci", "url": "registry.example.com/platform/media-desk"},
                    "version": "3.2.0", "constraint": "~> 3.0", "digest": MADE_UP,
                    "dependencies": ["$/agent/skills/brand-kit"]
                }),
                git_entry(
                    "$/agent/skills/brand-kit",
                    "skill",
                    "skills/brand-kit",
                    "1.3.4",
                    "~> 1.3",
                    MADE_UP,
                ),
            ],
        );
        assert_eq!(
            code(package.read()),
            (PackageErrorCode::AgentUnavailable, "$/agent".to_string())
        );
    }

    #[test]
    fn a_called_agent_that_cannot_start_is_noted_and_its_caller_still_reads() {
        // (the helper's variables, the values on the caller's entry, optional)
        let table = [
            // No value and no default.
            (json!({"tone": {"type": "string"}}), json!({}), false),
            (json!({"tone": {"type": "string"}}), json!({}), true),
            // A value that does not fit the field it lands in: a turn limit of 0.
            (
                json!({"turns": {"type": "number"}}),
                json!({"turns": 0}),
                false,
            ),
        ];
        for (variables, given, optional) in table {
            let package = Package::new("called-cannot-start");
            let mut entry = json!({"source": "./agents/helper"});
            if given.as_object().is_some_and(|given| !given.is_empty()) {
                entry["variables"] = given;
            }
            if optional {
                entry["optional"] = json!(true);
            }
            package.write_json(
                "agent.json",
                &agent("lead", json!({"agents": {"helper": entry}})),
            );
            let mut helper = json!({"variables": variables});
            if helper["variables"].get("turns").is_some() {
                helper["limits"] = json!({"max_turns": {"var": "turns"}});
            } else {
                helper["instructions"] = json!({"var": "tone"});
            }
            package.write_json("agents/helper/agent.json", &agent("helper", helper));
            package.write_lock(
                "",
                vec![
                    package.file_entry(
                        "$",
                        "agent",
                        "agent.json",
                        "./agent.json",
                        &["$/agents/helper"],
                    ),
                    package.file_entry(
                        "$/agents/helper",
                        "agent",
                        "agents/helper/agent.json",
                        "./agents/helper/agent.json",
                        &[],
                    ),
                ],
            );

            let reading = package.read().expect("the caller reads");

            // The helper is pinned and verified. It is not listed as startable.
            assert!(reading.called.is_empty());
            assert!(reading.part(PartField::Agents, "helper").is_some());
            let note = reading
                .notes
                .iter()
                .find(|note| note.code == NoteCode::CalledAgentCannotStart)
                .expect("a note that names the called agent");
            assert_eq!(note.subject, "$/agents/helper");
            assert!(note.text.contains("helper"), "{}", note.text);
        }
    }

    #[test]
    fn called_agents_are_checked_and_a_shared_part_has_one_entry() {
        let package = Package::new("called");
        package.write_json(
            "agent.json",
            &agent(
                "lead",
                json!({
                    "model": {"source": "ollama::large"},
                    "limits": {"max_turns": 60, "max_depth": 1},
                    "variables": {"width": {"type": "number", "default": 37}},
                    "agents": {"helper": {"source": "./agents/helper", "variables": {"max_line_length": {"var": "width"}}}},
                    "skills": {"review": {"source": "./skills/review"}}
                }),
            ),
        );
        package.write_json(
            "agents/helper/agent.json",
            &agent(
                "helper",
                json!({
                    "limits": {"max_turns": 80},
                    "variables": {"max_line_length": {"type": "number", "default": 42}},
                    "skills": {"review": {"source": "./../../skills/review", "variables": {"length": {"var": "max_line_length"}}},
                               "shared": {"source": "./../../skills/review"}}
                }),
            ),
        );
        package.write("skills/review/SKILL.md", SKILL);
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &["$/agents/helper", "$/agents/helper/skills/shared"],
                ),
                package.file_entry(
                    "$/agents/helper",
                    "agent",
                    "agents/helper/agent.json",
                    "./agents/helper/agent.json",
                    &[
                        "$/agents/helper/skills/review",
                        "$/agents/helper/skills/shared",
                    ],
                ),
                package.folder_entry(
                    "$/agents/helper/skills/review",
                    "skill",
                    "skills/review",
                    "./skills/review",
                ),
                package.folder_entry(
                    "$/agents/helper/skills/shared",
                    "skill",
                    "skills/review",
                    "./skills/review",
                ),
            ],
        );
        let reading = package.read().expect("package reads");
        assert_eq!(reading.called.len(), 1);
        let helper = &reading.called[0];
        assert_eq!(
            (helper.name.as_str(), helper.address.as_str()),
            ("helper", "$/agents/helper")
        );
        assert_eq!(helper.under, "lead");
        assert_eq!(
            Value::Object(helper.limits.clone()),
            json!({"max_turns": 60, "max_depth": 0})
        );
        assert_eq!(helper.model_from, "caller");
        assert_eq!(helper.model_source.as_deref(), Some("ollama::large"));
        assert_eq!(helper.values[0].value, json!(37));
        assert_eq!(helper.notes[0].code, NoteCode::LimitLowered);

        // The lead's skill is the helper's skill with no values: one entry, at the first path.
        let review = reading
            .part(PartField::Skills, "review")
            .expect("shared skill");
        assert_eq!(review.address, "$/agents/helper/skills/shared");
        // The helper's entry with values is its own use of the same folder.
        let with_values = reading
            .parts
            .iter()
            .find(|part| part.address == "$/agents/helper/skills/review")
            .expect("entry with values");
        assert_eq!(
            with_values
                .values
                .as_ref()
                .map(|values| Value::Object(values.clone())),
            Some(json!({"length": 37}))
        );
        assert_eq!(with_values.owner, "$/agents/helper");
    }
}
