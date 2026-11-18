// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The selected graph: derived from the documents, compared with the lock, and hashed.
//!
//! The documents decide what is in the graph. The lock is checked against them, entry by
//! entry, and the bytes that are here are checked against the lock. Nothing is fetched.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::definition::check_together;
use super::digest::{UNLISTED_FILE_NAME, bytes_digest, file_digest, folder_listing};
use super::document::{Source, check_document, declared_variables, parse_source, references};
use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::json::{MAX_DOCUMENT_BYTES, parse_document, read_regular_file};
use super::lock::{Lock, LockEntry, check_dependencies};
use super::model::{
    DigestKind, DocumentKind, EntryType, FolderListing, Integrity, LeftOutPart, LockedPart, Note,
    NoteCode, PartField, SourceType, StatedBy,
};
use super::oci::{STORE_DIR, find_artifact};
use super::paths::{
    address, display_inside, json_pointer, layout_path, normalize_inside, refuse_links,
    resolve_inside,
};
use super::variables::check_entry_values;
use super::version::check_selected;

/// The name of the lock file. It sits beside the entry document.
pub(crate) const LOCK_FILE: &str = "oadl.lock.json";
/// The entry document of an agent. A source that is a folder leads to it.
pub(crate) const AGENT_FILE: &str = "agent.json";

/// The most paths the loader walks through one graph.
const MAX_PATHS: usize = 200_000;
/// The deepest agents may list agents that list agents.
const MAX_NESTING: usize = 64;

/// What the graph check established.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GraphReading {
    /// Every lock entry, sorted by address.
    pub parts: Vec<LockedPart>,
    pub left_out: Vec<LeftOutPart>,
    /// The typed document read at each address.
    pub documents: BTreeMap<String, Value>,
    /// The files of each verified folder, by address.
    pub listings: BTreeMap<String, FolderListing>,
    pub notes: Vec<Note>,
    /// The address of each reference a read document holds, and the address of the lock
    /// entry it leads to. The two differ for a part that an earlier path also reaches.
    pub edges: BTreeMap<String, String>,
}

/// What makes two references one part.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Identity {
    /// A typed document stored in the package: its file.
    Document(PathBuf),
    /// A part stored in the package: its file or folder, the values its entry gives, and
    /// the type its field asks for.
    Stored(PathBuf, String, EntryType),
    /// Something fetched: the source, the version as written, the values and the type.
    Fetched(String, String, String, EntryType),
    /// A local reference of a fetched document: that document's package, the path as
    /// written, the values and the type.
    Below(usize, String, String, EntryType),
    /// A connector that is already running.
    RunningConnector(String, String),
    /// An agent that is already running.
    RunningAgent(String, String, String),
    /// The default model, written with its source and its revision.
    Model(String, String),
}

/// How a node is compared with its lock entry.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Origin {
    Root,
    Local,
    Git {
        url: String,
        subpath: Option<String>,
    },
    Oci {
        url: String,
    },
    /// Resolved through a registry: the recorded source is accepted as it is.
    Registry,
    /// A local reference of a fetched document. It keeps that package's source and version.
    Below {
        parent: usize,
    },
    RunningConnector,
    RunningAgent,
    Model {
        revision: String,
    },
}

#[derive(Debug, Clone)]
enum Content {
    File(String),
    Folder(FolderListing),
}

impl Content {
    fn digest(&self) -> &str {
        match self {
            Self::File(digest) => digest,
            Self::Folder(listing) => &listing.digest,
        }
    }
}

#[derive(Debug, Clone)]
struct Edge {
    /// `[field, name]`, `["agent"]` or `["model"]`.
    segments: Vec<String>,
    child: usize,
    /// The version constraint written on a reference that is not local.
    wanted: Option<String>,
    source_is_git: bool,
    field: Option<PartField>,
    pointer: String,
    optional: bool,
    source_text: Option<String>,
    version_text: Option<String>,
    variables: Option<Map<String, Value>>,
    stated_by: StatedBy,
    /// An agent that the model calls: it has a description, and its entry gives it values.
    called: bool,
}

#[derive(Debug, Clone)]
struct Node {
    entry_type: EntryType,
    origin: Origin,
    edges: Vec<Edge>,
    /// The typed document read here.
    document: Option<Value>,
    /// Where the bytes are, for content stored in the package or read from a layout path.
    path: Option<PathBuf>,
    content: Option<Content>,
    /// What sits below is not derived: a plugin, a package, or a document that was not read.
    opaque: bool,
}

/// One reference that the lock leaves out, before the address of its owner is known.
struct Pruned {
    parent: usize,
    edge: Edge,
}

/// One document while its references are followed.
struct Place {
    node: usize,
    /// The address of the path being walked, for the subject of a refusal.
    address: String,
    /// The folder local references resolve from. None for a fetched document.
    dir: Option<PathBuf>,
    /// The fetched document whose package this document belongs to.
    package: Option<usize>,
    /// The path of this document inside that package, as text.
    prefix: String,
    stated_by: StatedBy,
}

/// How a reference is followed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// The agent a definition points at.
    Pointer,
    /// An entry under `agents`.
    Called,
    /// Stored bytes with no document type of this format.
    Leaf,
}

struct Link {
    segments: Vec<String>,
    role: Role,
    entry_type: EntryType,
    field: Option<PartField>,
    pointer: String,
    source: String,
    version: Option<String>,
    optional: bool,
    variables: Option<Map<String, Value>>,
}

struct Walk<'a> {
    package_dir: &'a Path,
    entry_dir: PathBuf,
    lock: &'a Lock,
    nodes: Vec<Node>,
    index: BTreeMap<Identity, usize>,
    active: BTreeSet<usize>,
    pruned: Vec<Pruned>,
    notes: Vec<Note>,
}

fn refuse(code: PackageErrorCode, subject: &str, message: impl Into<String>) -> PackageError {
    PackageError::new(code, subject, message)
}

/// The values an entry gives, as text that is the same for the same values.
fn values_text(variables: Option<&Map<String, Value>>) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            // `1` and `1.0` are one number.
            Value::Number(number) => match number.as_f64() {
                Some(float) if float.fract() == 0.0 && float.abs() < 9.0e15 => {
                    out.push_str(&format!("{}", float as i64));
                }
                _ => out.push_str(&number.to_string()),
            },
            Value::Array(items) => {
                out.push('[');
                for item in items {
                    write(item, out);
                    out.push(',');
                }
                out.push(']');
            }
            Value::Object(object) => {
                out.push('{');
                for (key, item) in object {
                    out.push_str(&Value::String(key.clone()).to_string());
                    out.push(':');
                    write(item, out);
                    out.push(',');
                }
                out.push('}');
            }
            other => out.push_str(&other.to_string()),
        }
    }
    let mut text = String::new();
    if let Some(variables) = variables {
        write(&Value::Object(variables.clone()), &mut text);
    }
    text
}

fn is_file(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_file())
}

fn is_dir(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_dir())
}

/// The file a part's folder must hold at its root, when the part has content here.
fn entry_file(entry_type: EntryType) -> Option<&'static str> {
    match entry_type {
        EntryType::Skill => Some("SKILL.md"),
        EntryType::Connector => Some("mcp.json"),
        EntryType::Hook => Some("hooks.json"),
        _ => None,
    }
}

/// A part's content has its own entry file. A connector and a hook are always a folder.
fn check_entry_file(entry_type: EntryType, content: &Content, subject: &str) -> PackageResult<()> {
    let Some(name) = entry_file(entry_type) else {
        return Ok(());
    };
    match content {
        Content::Folder(listing) if listing.files.contains_key(name) => Ok(()),
        Content::Folder(_) => Err(refuse(
            PackageErrorCode::PartEntrypoint,
            subject,
            format!("the folder has no `{name}` at its root"),
        )),
        // One file can be a skill. It cannot be a connector or a hook.
        Content::File(_) if entry_type == EntryType::Skill => Ok(()),
        Content::File(_) => Err(refuse(
            PackageErrorCode::PartEntrypoint,
            subject,
            format!("the source is one file, not a folder with `{name}` at its root"),
        )),
    }
}

impl<'a> Walk<'a> {
    fn lock_path(&self) -> PathBuf {
        self.entry_dir.join(LOCK_FILE)
    }

    fn node(&mut self, identity: Identity, node: Node) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        self.index.insert(identity, index);
        index
    }

    fn subject_path(&self, path: &Path) -> String {
        display_inside(self.package_dir, path)
    }

    /// Reads one typed document and checks that it is an agent.
    fn read_agent(&self, path: &Path, at: &str) -> PackageResult<(Value, String)> {
        let subject = self.subject_path(path);
        let bytes = read_regular_file(path, &subject, MAX_DOCUMENT_BYTES)?;
        let document = parse_document(&bytes, &subject)?;
        let kind = check_document(&document, &subject)?;
        if kind != DocumentKind::Agent {
            return Err(refuse(
                PackageErrorCode::ReferenceType,
                at,
                format!(
                    "`{subject}` is an agent-definition, and the reference must resolve to an agent"
                ),
            ));
        }
        Ok((document, bytes_digest(&bytes)))
    }

    /// Follows every reference of one document.
    fn follow(&mut self, place: &Place, document: &Value) -> PackageResult<()> {
        if self.active.len() > MAX_NESTING {
            return Err(too_deep(&place.address));
        }
        self.active.insert(place.node);
        let result = self.follow_links(place, document);
        self.active.remove(&place.node);
        result
    }

    fn follow_links(&mut self, place: &Place, document: &Value) -> PackageResult<()> {
        let is_definition =
            document.get("type").and_then(Value::as_str) == Some("agent-definition");
        // A definition declares no variable of its own.
        let own = (!is_definition).then(|| declared_variables(document));
        // The agent a definition points at, once it is read: the two are read together.
        let mut base: Option<usize> = None;

        if is_definition && let Some(reference) = document.get("agent") {
            let link = Link {
                segments: vec!["agent".to_string()],
                role: Role::Pointer,
                entry_type: EntryType::Agent,
                field: None,
                pointer: json_pointer(&["agent"]),
                source: reference
                    .get("source")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                version: reference
                    .get("version")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                optional: false,
                variables: None,
            };
            let at = address(&place.address, &["agent"]);
            if let Some(child) = self.link(place, &link, &at)? {
                // The pointer of a definition has one address, so a fetched agent is read now.
                self.read_fetched(child, &at)?;
                if let Some(agent) = self.nodes[child].document.clone() {
                    let given = document
                        .get("variables")
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default();
                    check_entry_values(&given, &declared_variables(&agent), None, &at)?;
                    let one_folder = self.nodes[child].origin == Origin::Local
                        && self.nodes[child].path.as_deref().and_then(Path::parent)
                            == place.dir.as_deref();
                    check_together(document, &agent, one_folder)?;
                    base = Some(child);
                }
            }
        }

        if !is_definition
            && let Some(model) = document.get("model")
            && let (Some(source), Some(revision)) = (
                model.get("source").and_then(Value::as_str),
                model.get("revision").and_then(Value::as_str),
            )
        {
            // A model has a lock entry only when it writes its source and its revision as text.
            let identity = Identity::Model(source.to_string(), revision.to_string());
            let child = match self.index.get(&identity) {
                Some(child) => *child,
                None => self.node(
                    identity,
                    Node {
                        entry_type: EntryType::Model,
                        origin: Origin::Model {
                            revision: revision.to_string(),
                        },
                        edges: Vec::new(),
                        document: None,
                        path: None,
                        content: None,
                        opaque: false,
                    },
                ),
            };
            self.nodes[place.node].edges.push(Edge {
                segments: vec!["model".to_string()],
                child,
                wanted: None,
                source_is_git: false,
                field: None,
                pointer: json_pointer(&["model"]),
                optional: false,
                source_text: Some(source.to_string()),
                version_text: None,
                variables: None,
                stated_by: place.stated_by,
                called: false,
            });
        }

        let mut links: Vec<(String, Link)> = references(document)
            .into_iter()
            .map(|reference| {
                let segments = vec![reference.field.as_str().to_string(), reference.name.clone()];
                let at = address(&place.address, &[reference.field.as_str(), &reference.name]);
                let role = if reference.field == PartField::Agents {
                    Role::Called
                } else {
                    Role::Leaf
                };
                let link = Link {
                    segments,
                    role,
                    entry_type: reference.field.entry_type(),
                    field: Some(reference.field),
                    pointer: reference.pointer,
                    source: reference.source,
                    version: reference.version,
                    optional: reference.optional,
                    variables: reference.variables,
                };
                (at, link)
            })
            .collect();
        links.sort_by(|left, right| left.0.cmp(&right.0));

        for (at, link) in links {
            if let Some(base) = base
                && let Some(agent) = &self.nodes[base].document
                && agent
                    .get(&link.segments[0])
                    .is_some_and(|entries| entries.get(&link.segments[1]).is_some())
            {
                // An entry identical to the agent's is the same part, reached by two paths.
                let copied: Vec<Edge> = self.nodes[base]
                    .edges
                    .iter()
                    .filter(|edge| edge.segments == link.segments)
                    .cloned()
                    .collect();
                self.nodes[place.node].edges.extend(copied);
                continue;
            }
            let Some(child) = self.link(place, &link, &at)? else {
                continue;
            };
            if link.role == Role::Called
                && let Some(agent) = &self.nodes[child].document
            {
                check_called(agent, link.variables.as_ref(), own.as_ref(), &at)?;
            }
        }
        Ok(())
    }

    /// Resolves one reference to a node and records the edge. None when the lock leaves an
    /// optional reference out because nothing is there to pin.
    fn link(&mut self, place: &Place, link: &Link, at: &str) -> PackageResult<Option<usize>> {
        let child = match parse_source(&link.source) {
            Source::Local { path } => match &place.dir {
                Some(dir) => match self.stored(place, link, dir, &path, at)? {
                    Some(child) => child,
                    None => return Ok(None),
                },
                None => self.below(place, link),
            },
            Source::Typed {
                kind,
                address: location,
                subpath,
            } => self.typed(link, &kind, location, subpath, at)?,
        };
        let local = link.source.starts_with("./");
        if local && link.version.is_some() {
            self.notes.push(Note {
                code: NoteCode::LocalVersionIgnored,
                subject: at.to_string(),
                text: format!("{at}: a version on a local source has no meaning and is ignored"),
            });
        }
        self.nodes[place.node].edges.push(Edge {
            segments: link.segments.clone(),
            child,
            wanted: if local { None } else { link.version.clone() },
            source_is_git: link.source.starts_with("git::"),
            field: link.field,
            pointer: link.pointer.clone(),
            optional: link.optional,
            source_text: Some(link.source.clone()),
            version_text: link.version.clone(),
            variables: link.variables.clone(),
            stated_by: place.stated_by,
            called: link.role == Role::Called,
        });
        Ok(Some(child))
    }

    /// A local reference of a document that is stored in the package.
    fn stored(
        &mut self,
        place: &Place,
        link: &Link,
        dir: &Path,
        relative: &str,
        at: &str,
    ) -> PackageResult<Option<usize>> {
        if link.source.contains('#') || link.source.starts_with(".//") {
            return Err(refuse(
                PackageErrorCode::PathUnsafe,
                at,
                format!("`{}` is not a plain path inside the package", link.source),
            ));
        }
        let path = resolve_inside(self.package_dir, dir, relative, at)?;
        let missing = |walk: &mut Self, what: String| -> PackageResult<Option<usize>> {
            // Nothing is there to pin. An optional reference that the lock leaves out is
            // reported and never stops a start. A pinned part that is gone fails its pin.
            if link.optional && !walk.lock.entries.contains_key(at) {
                walk.prune(place.node, link);
                return Ok(None);
            }
            Err(refuse(PackageErrorCode::PartMissing, at, what))
        };

        // A file of this name is never part of a package, so a reference to one finds
        // nothing, as it does in the format's reference checker.
        let unlisted = path.file_name() == Some(UNLISTED_FILE_NAME.as_ref());
        if unlisted && is_file(&path) {
            let what = format!(
                "`{}` is not in the package: a file named `{UNLISTED_FILE_NAME}` is never part of one",
                self.subject_path(&path)
            );
            return missing(self, what);
        }

        if link.role == Role::Leaf {
            let content = if is_file(&path) {
                Content::File(file_digest(&path, at)?)
            } else if is_dir(&path) {
                let listing = folder_listing(&path, Some(&self.lock_path()), at)?;
                if listing.files.is_empty() {
                    let what = format!("`{}` holds no file", self.subject_path(&path));
                    return missing(self, what);
                }
                Content::Folder(listing)
            } else {
                let what = format!("`{}` is not in the package", self.subject_path(&path));
                return missing(self, what);
            };
            check_entry_file(link.entry_type, &content, at)?;
            let identity = Identity::Stored(
                path.clone(),
                values_text(link.variables.as_ref()),
                link.entry_type,
            );
            if let Some(found) = self.index.get(&identity) {
                return Ok(Some(*found));
            }
            return Ok(Some(self.node(
                identity,
                Node {
                    entry_type: link.entry_type,
                    origin: Origin::Local,
                    edges: Vec::new(),
                    document: None,
                    path: Some(path),
                    content: Some(content),
                    opaque: matches!(link.entry_type, EntryType::Plugin | EntryType::Package),
                },
            )));
        }

        // A source that is a folder leads to the agent.json at its root.
        let file = if is_file(&path) {
            path
        } else {
            let inside = path.join(AGENT_FILE);
            refuse_links(self.package_dir, &inside, at)?;
            if !is_file(&inside) {
                let what = format!(
                    "`{}` is not a document, and it has no `{AGENT_FILE}` at its root",
                    self.subject_path(&path)
                );
                return missing(self, what);
            }
            inside
        };
        let identity = Identity::Document(file.clone());
        if let Some(found) = self.index.get(&identity).copied() {
            if link.role == Role::Called
                && let Some(document) = &self.nodes[found].document
            {
                check_called(document, None, None, at)?;
            }
            if self.active.contains(&found) {
                return Err(refuse(
                    PackageErrorCode::ReferenceCycle,
                    at,
                    format!("`{}` reaches itself", self.subject_path(&file)),
                ));
            }
            return Ok(Some(found));
        }
        let (document, digest) = self.read_agent(&file, at)?;
        if link.role == Role::Called {
            check_called(&document, None, None, at)?;
        }
        let child = self.node(
            identity,
            Node {
                entry_type: EntryType::Agent,
                origin: Origin::Local,
                edges: Vec::new(),
                document: Some(document.clone()),
                path: Some(file.clone()),
                content: Some(Content::File(digest)),
                opaque: false,
            },
        );
        let below = Place {
            node: child,
            address: at.to_string(),
            dir: file.parent().map(Path::to_path_buf),
            package: None,
            prefix: String::new(),
            stated_by: StatedBy::Agent,
        };
        self.follow(&below, &document)?;
        Ok(Some(child))
    }

    /// A local reference of a fetched document: it is inside that document's own package.
    fn below(&mut self, place: &Place, link: &Link) -> usize {
        let package = place.package.unwrap_or(place.node);
        let is_document = link.role != Role::Leaf;
        let identity = Identity::Below(
            package,
            format!("{}{}", place.prefix, link.source),
            values_text(link.variables.as_ref()),
            link.entry_type,
        );
        if let Some(found) = self.index.get(&identity) {
            return *found;
        }
        self.node(
            identity,
            Node {
                entry_type: link.entry_type,
                origin: Origin::Below { parent: package },
                edges: Vec::new(),
                document: None,
                path: None,
                content: None,
                opaque: is_document
                    || matches!(link.entry_type, EntryType::Plugin | EntryType::Package),
            },
        )
    }

    /// A reference that is not local: something fetched, or something already running.
    fn typed(
        &mut self,
        link: &Link,
        kind: &str,
        location: String,
        subpath: Option<String>,
        at: &str,
    ) -> PackageResult<usize> {
        let Some(version) = link.version.clone() else {
            return Err(refuse(
                PackageErrorCode::SchemaInvalid,
                at,
                "a source that is not local needs a version",
            ));
        };
        let values = values_text(link.variables.as_ref());
        let running_connector = kind == "mcp" && link.entry_type == EntryType::Connector;
        let running_agent = kind == "a2a" && link.role == Role::Called;
        let (identity, origin) = if running_connector {
            (
                Identity::RunningConnector(link.source.clone(), version),
                Origin::RunningConnector,
            )
        } else if running_agent {
            (
                Identity::RunningAgent(link.source.clone(), version, values),
                Origin::RunningAgent,
            )
        } else {
            let origin = match kind {
                "git" => Origin::Git {
                    url: location,
                    subpath,
                },
                "oci" => Origin::Oci { url: location },
                _ => Origin::Registry,
            };
            (
                Identity::Fetched(link.source.clone(), version, values, link.entry_type),
                origin,
            )
        };
        if let Some(found) = self.index.get(&identity) {
            return Ok(*found);
        }
        let fetched = !(running_connector || running_agent);
        Ok(self.node(
            identity,
            Node {
                entry_type: link.entry_type,
                origin,
                edges: Vec::new(),
                document: None,
                path: None,
                content: None,
                // A fetched agent is opaque until its document is read.
                opaque: fetched
                    && (link.role != Role::Leaf
                        || matches!(link.entry_type, EntryType::Plugin | EntryType::Package)),
            },
        ))
    }

    fn prune(&mut self, parent: usize, link: &Link) {
        self.pruned.push(Pruned {
            parent,
            edge: Edge {
                segments: link.segments.clone(),
                child: usize::MAX,
                wanted: None,
                source_is_git: false,
                field: link.field,
                pointer: link.pointer.clone(),
                optional: true,
                source_text: Some(link.source.clone()),
                version_text: link.version.clone(),
                variables: link.variables.clone(),
                stated_by: StatedBy::Agent,
                called: link.role == Role::Called,
            },
        });
    }

    /// Reads a fetched agent at `at` when the lock pins it there by a tree digest and its
    /// bytes are in the package. The bytes are checked against the pin before they are read.
    fn read_fetched(&mut self, node: usize, at: &str) -> PackageResult<bool> {
        if self.nodes[node].document.is_some() || self.nodes[node].entry_type != EntryType::Agent {
            return Ok(false);
        }
        if !matches!(
            self.nodes[node].origin,
            Origin::Git { .. } | Origin::Registry | Origin::Below { .. }
        ) {
            return Ok(false);
        }
        let Some(entry) = self.lock.entries.get(at) else {
            return Ok(false);
        };
        if entry.source.source_type != SourceType::Git {
            return Ok(false);
        }
        let file = layout_path(&self.entry_dir, at)?.join(AGENT_FILE);
        refuse_links(self.package_dir, &file, at)?;
        if !is_file(&file) {
            return Ok(false);
        }
        let subject = self.subject_path(&file);
        let bytes = read_regular_file(&file, &subject, MAX_DOCUMENT_BYTES)?;
        let digest = bytes_digest(&bytes);
        if entry.digest.as_deref() != Some(digest.as_str()) {
            return Err(digest_mismatch(at, entry, &digest, &subject));
        }
        let document = parse_document(&bytes, &subject)?;
        if check_document(&document, &subject)? != DocumentKind::Agent {
            return Err(refuse(
                PackageErrorCode::ReferenceType,
                at,
                format!(
                    "`{subject}` is an agent-definition, and the reference must resolve to an agent"
                ),
            ));
        }
        let package = match self.nodes[node].origin {
            Origin::Below { parent } => parent,
            _ => node,
        };
        let prefix = match self.nodes[node].origin {
            Origin::Below { .. } => format!("{at}|"),
            _ => String::new(),
        };
        self.nodes[node].document = Some(document.clone());
        self.nodes[node].path = Some(file);
        self.nodes[node].content = Some(Content::File(digest));
        self.nodes[node].opaque = false;
        let place = Place {
            node,
            address: at.to_string(),
            dir: None,
            package: Some(package),
            prefix,
            stated_by: StatedBy::Agent,
        };
        self.follow(&place, &document)?;
        Ok(true)
    }
}

/// A called agent has a description, and the values on its entry fit what it declares.
fn check_called(
    agent: &Value,
    given: Option<&Map<String, Value>>,
    own: Option<&Map<String, Value>>,
    at: &str,
) -> PackageResult<()> {
    let described = agent
        .get("description")
        .and_then(Value::as_str)
        .is_some_and(|text| !text.is_empty());
    if !described {
        return Err(refuse(
            PackageErrorCode::CalledAgentDescription,
            at,
            "the called agent has no description for the model to read",
        ));
    }
    match given {
        Some(given) => check_entry_values(given, &declared_variables(agent), own, at),
        None => Ok(()),
    }
}

fn digest_mismatch(at: &str, entry: &LockEntry, found: &str, path: &str) -> PackageError {
    refuse(
        PackageErrorCode::DigestMismatch,
        at,
        format!(
            "the lock pins {}, and the bytes at `{path}` have {found}",
            entry.digest.as_deref().unwrap_or("no digest")
        ),
    )
}

fn too_deep(at: &str) -> PackageError {
    refuse(
        PackageErrorCode::TooLarge,
        at,
        format!("agents list agents deeper than {MAX_NESTING} levels"),
    )
}

fn mismatch(at: &str, message: impl Into<String>) -> PackageError {
    refuse(PackageErrorCode::LockEntryMismatch, at, message)
}

/// The first address of each node, and the edge that ends there.
struct Paths {
    first: Vec<Option<String>>,
    /// The parent and the index of the edge whose path is the first address.
    home: Vec<Option<(usize, usize)>>,
}

impl Paths {
    fn of(nodes: &[Node], root: usize) -> PackageResult<Self> {
        let mut paths = Self {
            first: vec![None; nodes.len()],
            home: vec![None; nodes.len()],
        };
        let mut steps = 0usize;
        let mut on_path = vec![false; nodes.len()];
        paths.reach(
            nodes,
            root,
            "$".to_string(),
            None,
            0,
            &mut on_path,
            &mut steps,
        )?;
        Ok(paths)
    }

    #[allow(clippy::too_many_arguments)]
    fn reach(
        &mut self,
        nodes: &[Node],
        node: usize,
        at: String,
        through: Option<(usize, usize)>,
        depth: usize,
        on_path: &mut [bool],
        steps: &mut usize,
    ) -> PackageResult<()> {
        // A path is one step longer than the agents on it: its last step may be a part.
        if depth > MAX_NESTING + 2 {
            return Err(too_deep(&at));
        }
        *steps += 1;
        if *steps > MAX_PATHS {
            return Err(refuse(
                PackageErrorCode::TooLarge,
                "$",
                format!("the graph has more than {MAX_PATHS} paths"),
            ));
        }
        if on_path[node] {
            return Err(refuse(
                PackageErrorCode::ReferenceCycle,
                &at,
                "the reference reaches a document that is already on its own path",
            ));
        }
        if self.first[node].as_ref().is_none_or(|first| at < *first) {
            self.first[node] = Some(at.clone());
            self.home[node] = through;
        }
        on_path[node] = true;
        for (index, edge) in nodes[node].edges.iter().enumerate() {
            let segments: Vec<&str> = edge.segments.iter().map(String::as_str).collect();
            self.reach(
                nodes,
                edge.child,
                address(&at, &segments),
                Some((node, index)),
                depth + 1,
                on_path,
                steps,
            )?;
        }
        on_path[node] = false;
        Ok(())
    }

    /// Walks again from `node`, at an address that no path the documents show gives it.
    fn reach_at(&mut self, nodes: &[Node], node: usize, at: String) -> PackageResult<()> {
        let through = self.home[node];
        let mut on_path = vec![false; nodes.len()];
        self.reach(nodes, node, at, through, 1, &mut on_path, &mut 0)
    }
}

/// Derive the graph from the documents and compare it with the lock, entry by entry. Hash what is here.
pub(crate) fn verify_graph(
    package_dir: &Path,
    entry_path: &Path,
    entry: &Value,
    lock: &Lock,
) -> PackageResult<GraphReading> {
    let entry_dir = entry_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| package_dir.to_path_buf());
    let is_definition = entry.get("type").and_then(Value::as_str) == Some("agent-definition");
    let mut walk = Walk {
        package_dir,
        entry_dir,
        lock,
        nodes: Vec::new(),
        index: BTreeMap::new(),
        active: BTreeSet::new(),
        pruned: Vec::new(),
        notes: Vec::new(),
    };
    let root = walk.node(
        Identity::Document(entry_path.to_path_buf()),
        Node {
            entry_type: if is_definition {
                EntryType::AgentDefinition
            } else {
                EntryType::Agent
            },
            origin: Origin::Root,
            edges: Vec::new(),
            document: Some(entry.clone()),
            path: Some(entry_path.to_path_buf()),
            content: Some(Content::File(file_digest(entry_path, "$")?)),
            opaque: false,
        },
    );
    let place = Place {
        node: root,
        address: "$".to_string(),
        dir: Some(walk.entry_dir.clone()),
        package: None,
        prefix: String::new(),
        stated_by: if is_definition {
            StatedBy::Definition
        } else {
            StatedBy::Agent
        },
    };
    walk.follow(&place, entry)?;
    let paths = walk.settle(root)?;
    // Every part the documents reach has an entry. Now every dependency must name one.
    check_dependencies(lock)?;
    walk.compare(&paths)
}

impl Walk<'_> {
    /// Reads the fetched agents that are here, and leaves out the optional references the
    /// lock does not pin, until neither changes the graph any more.
    fn settle(&mut self, root: usize) -> PackageResult<Paths> {
        loop {
            let paths = self.paths(root)?;

            // A fetched agent is read at the address of its lock entry.
            let mut unread: Vec<(String, usize)> = (0..self.nodes.len())
                .filter(|node| self.nodes[*node].document.is_none())
                .filter_map(|node| Some((paths.first[node].clone()?, node)))
                .collect();
            unread.sort();
            let mut read = false;
            for (at, node) in unread {
                if !self.read_fetched(node, &at)? {
                    continue;
                }
                read = true;
                // Each entry that calls the agent gives it values, checked now that it is read.
                let agent = self.nodes[node].document.clone().unwrap_or(Value::Null);
                for parent in 0..self.nodes.len() {
                    let own = self.nodes[parent].document.as_ref().and_then(|document| {
                        (document.get("type").and_then(Value::as_str) == Some("agent"))
                            .then(|| declared_variables(document))
                    });
                    for edge in &self.nodes[parent].edges {
                        if edge.child == node && edge.called {
                            check_called(&agent, edge.variables.as_ref(), own.as_ref(), &at)?;
                        }
                    }
                }
                break;
            }
            if read {
                continue;
            }

            // What the lock does not pin: an optional reference is left out, and reported.
            let missing: Vec<usize> = (0..self.nodes.len())
                .filter(|node| {
                    paths.first[*node]
                        .as_ref()
                        .is_some_and(|at| !self.lock.entries.contains_key(at))
                })
                .collect();
            if missing.is_empty() {
                return Ok(paths);
            }
            let reached = |node: usize| paths.first[node].is_some();
            let optional_only = |node: usize| {
                (0..self.nodes.len())
                    .filter(|parent| reached(*parent))
                    .flat_map(|parent| self.nodes[parent].edges.iter())
                    .filter(|edge| edge.child == node)
                    .all(|edge| edge.optional)
            };
            let left_out: Vec<usize> = missing
                .iter()
                .copied()
                .filter(|node| *node != root && optional_only(*node))
                .collect();
            if left_out.is_empty() {
                let at = missing
                    .iter()
                    .filter_map(|node| paths.first[*node].as_deref())
                    .min()
                    .unwrap_or("$");
                return Err(refuse(
                    PackageErrorCode::LockEntryMissing,
                    at,
                    "the documents reach this part, and the lock has no entry for it",
                ));
            }
            for parent in 0..self.nodes.len() {
                if !reached(parent) {
                    continue;
                }
                let (kept, removed): (Vec<Edge>, Vec<Edge>) =
                    std::mem::take(&mut self.nodes[parent].edges)
                        .into_iter()
                        .partition(|edge| !left_out.contains(&edge.child));
                self.nodes[parent].edges = kept;
                self.pruned
                    .extend(removed.into_iter().map(|edge| Pruned { parent, edge }));
            }
        }
    }

    /// The first address of every node the documents reach.
    ///
    /// A part that two paths reach has one entry, at the first path in sort order. A path
    /// through an agent whose document is not in the package cannot be derived. A part with
    /// no entry at its first derived address is then taken at the address the lock records
    /// it at below such an agent. Once that agent's document is in the package, the path is
    /// derived like any other, and the lock must match it.
    fn paths(&self, root: usize) -> PackageResult<Paths> {
        let mut paths = Paths::of(&self.nodes, root)?;
        while let Some((node, at)) = self.recorded_below_unread(&paths) {
            paths.reach_at(&self.nodes, node, at)?;
        }
        Ok(paths)
    }

    /// A part with no entry at its first derived address, with the earlier address at
    /// which the lock records an entry it can have below an agent that is not read.
    fn recorded_below_unread(&self, paths: &Paths) -> Option<(usize, String)> {
        let entries = &self.lock.entries;
        let derived: BTreeSet<&str> = paths.first.iter().flatten().map(String::as_str).collect();
        // What the lock records below each agent that is not read: the entries a chain of
        // dependencies leads to from it, at addresses below its own.
        let mut below: BTreeSet<&str> = BTreeSet::new();
        for (agent, at) in paths.first.iter().enumerate() {
            let this = &self.nodes[agent];
            let unread =
                this.entry_type == EntryType::Agent && this.document.is_none() && this.opaque;
            let Some(at) = at
                .as_deref()
                .filter(|at| unread && entries.contains_key(*at))
            else {
                continue;
            };
            let prefix = format!("{at}/");
            let mut queue = vec![at];
            while let Some(parent) = queue.pop() {
                for dependency in &entries[parent].dependencies {
                    let dependency = dependency.as_str();
                    if dependency.starts_with(&prefix)
                        && entries.contains_key(dependency)
                        && !derived.contains(dependency)
                        && below.insert(dependency)
                    {
                        queue.push(dependency);
                    }
                }
            }
        }
        if below.is_empty() {
            return None;
        }
        let mut missing: Vec<(&str, usize)> = (0..self.nodes.len())
            .filter_map(|node| Some((paths.first[node].as_deref()?, node)))
            .filter(|(at, _)| !entries.contains_key(*at))
            // Only a part that documents of two packages can both name is shared this way.
            .filter(|(_, node)| {
                matches!(
                    self.nodes[*node].origin,
                    Origin::Git { .. }
                        | Origin::Oci { .. }
                        | Origin::Registry
                        | Origin::RunningConnector
                        | Origin::RunningAgent
                        | Origin::Model { .. }
                )
            })
            .collect();
        missing.sort();
        missing.into_iter().find_map(|(first, node)| {
            let home = paths.home[node].map(|(parent, index)| &self.nodes[parent].edges[index]);
            let incoming: Vec<(usize, usize)> = (0..self.nodes.len())
                .filter(|parent| paths.first[*parent].is_some())
                .flat_map(|parent| {
                    let edges = self.nodes[parent].edges.iter().enumerate();
                    edges.filter_map(move |(index, edge)| {
                        (edge.child == node).then_some((parent, index))
                    })
                })
                .collect();
            // The entry must be one this reference can have: its type, its source, the
            // selected version and the recorded constraint are checked as at any address.
            below
                .iter()
                .take_while(|at| **at < first)
                .find(|at| {
                    self.check_entry(
                        at,
                        node,
                        &entries[**at],
                        home,
                        &incoming,
                        paths,
                        &mut Vec::new(),
                    )
                    .is_ok()
                })
                .map(|at| (node, at.to_string()))
        })
    }

    /// Compares each node with its lock entry and hashes what is here.
    fn compare(mut self, paths: &Paths) -> PackageResult<GraphReading> {
        let lock = self.lock;
        let expected: BTreeMap<&str, usize> = (0..self.nodes.len())
            .filter_map(|node| Some((paths.first[node].as_deref()?, node)))
            .collect();

        // Entries below something that is not derived are accepted when a chain of
        // dependencies leads to them from it. Any other extra entry is refused.
        let mut below: BTreeMap<&str, &str> = BTreeMap::new();
        let mut queue: Vec<&str> = expected
            .iter()
            .filter(|(_, node)| self.nodes[**node].opaque)
            .map(|(at, _)| *at)
            .collect();
        while let Some(parent) = queue.pop() {
            for dependency in &lock.entries[parent].dependencies {
                let dependency = dependency.as_str();
                if !expected.contains_key(dependency) && !below.contains_key(dependency) {
                    below.insert(dependency, parent);
                    queue.push(dependency);
                }
            }
        }
        if let Some(extra) = lock
            .entries
            .keys()
            .find(|at| !expected.contains_key(at.as_str()) && !below.contains_key(at.as_str()))
        {
            return Err(refuse(
                PackageErrorCode::LockEntryExtra,
                extra,
                "the lock pins this entry, and no document reaches it",
            ));
        }

        let mut incoming: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.nodes.len()];
        for (parent, node) in self.nodes.iter().enumerate() {
            if paths.first[parent].is_none() {
                continue;
            }
            for (index, edge) in node.edges.iter().enumerate() {
                incoming[edge.child].push((parent, index));
            }
        }

        let mut parts = Vec::new();
        let mut listings = BTreeMap::new();
        let mut documents = BTreeMap::new();
        let mut notes = std::mem::take(&mut self.notes);
        for (at, node) in &expected {
            let (at, node) = (*at, *node);
            let entry = &lock.entries[at];
            let home =
                paths.home[node].map(|(parent, index)| (parent, &self.nodes[parent].edges[index]));
            self.check_entry(
                at,
                node,
                entry,
                home.map(|(_, edge)| edge),
                &incoming[node],
                paths,
                &mut notes,
            )?;
            let (integrity, path, digest_kind, listing) = self.integrity(at, node, entry)?;
            if !self.nodes[node].opaque {
                let derived: BTreeSet<&str> = self.nodes[node]
                    .edges
                    .iter()
                    .filter_map(|edge| paths.first[edge.child].as_deref())
                    .collect();
                let recorded: BTreeSet<&str> =
                    entry.dependencies.iter().map(String::as_str).collect();
                if derived != recorded {
                    return Err(mismatch(
                        at,
                        format!(
                            "the lock lists the dependencies {recorded:?}, and the document reaches {derived:?}"
                        ),
                    ));
                }
            }
            if let Some(listing) = listing {
                listings.insert(at.to_string(), listing);
            }
            if let Some(document) = &self.nodes[node].document {
                documents.insert(at.to_string(), document.clone());
            }
            if let Some(reason) = &entry.retracted {
                notes.push(Note {
                    code: NoteCode::Retracted,
                    subject: at.to_string(),
                    text: format!("{at}: the selected version is retracted: {reason}"),
                });
            }
            let owner = home
                .and_then(|(parent, _)| paths.first[parent].clone())
                .unwrap_or_else(|| "$".to_string());
            let edge = home.map(|(_, edge)| edge);
            let field = edge.and_then(|edge| edge.field);
            parts.push(LockedPart {
                address: at.to_string(),
                entry_type: entry.entry_type,
                field,
                name: field.and_then(|_| edge.and_then(|edge| edge.segments.get(1).cloned())),
                owner,
                stated_by: match edge {
                    Some(edge) if field.is_some() => edge.stated_by,
                    _ => StatedBy::Agent,
                },
                optional: edge.is_some_and(|edge| edge.optional),
                pointer: edge.map(|edge| edge.pointer.clone()),
                source_text: edge.and_then(|edge| edge.source_text.clone()),
                version_text: edge.and_then(|edge| edge.version_text.clone()),
                values: edge.and_then(|edge| edge.variables.clone()),
                path,
                integrity,
                digest_kind,
                ..locked(at, entry)
            });
        }
        for (at, parent) in &below {
            let entry = &lock.entries[*at];
            parts.push(LockedPart {
                owner: parent.to_string(),
                integrity: Integrity::NotChecked,
                digest_kind: None,
                ..locked(at, entry)
            });
        }
        parts.sort_by(|left, right| left.address.cmp(&right.address));

        let mut edges = BTreeMap::new();
        for (parent, node) in self.nodes.iter().enumerate() {
            let Some(owner) = paths.first[parent].as_deref() else {
                continue;
            };
            for edge in &node.edges {
                let segments: Vec<&str> = edge.segments.iter().map(String::as_str).collect();
                if let Some(target) = &paths.first[edge.child] {
                    edges.insert(address(owner, &segments), target.clone());
                }
            }
        }
        let mut left_out: Vec<LeftOutPart> = self
            .pruned
            .iter()
            .filter_map(|pruned| {
                Some(LeftOutPart {
                    field: pruned.edge.field?,
                    name: pruned.edge.segments.get(1)?.clone(),
                    owner: paths.first[pruned.parent].clone()?,
                    stated_by: self.nodes[pruned.parent]
                        .document
                        .as_ref()
                        .and_then(|document| document.get("type"))
                        .and_then(Value::as_str)
                        .map_or(StatedBy::Agent, |kind| {
                            if kind == "agent-definition" {
                                StatedBy::Definition
                            } else {
                                StatedBy::Agent
                            }
                        }),
                    pointer: pruned.edge.pointer.clone(),
                    source_text: pruned.edge.source_text.clone().unwrap_or_default(),
                })
            })
            .collect();
        left_out.sort_by(|left, right| {
            (&left.owner, &left.pointer).cmp(&(&right.owner, &right.pointer))
        });
        left_out.dedup();
        Ok(GraphReading {
            parts,
            left_out,
            documents,
            listings,
            notes,
            edges,
        })
    }

    /// The type, the source, the selected version and the recorded constraint of one entry.
    #[allow(clippy::too_many_arguments)]
    fn check_entry(
        &self,
        at: &str,
        node: usize,
        entry: &LockEntry,
        home: Option<&Edge>,
        incoming: &[(usize, usize)],
        paths: &Paths,
        notes: &mut Vec<Note>,
    ) -> PackageResult<()> {
        let this = &self.nodes[node];
        if entry.entry_type != this.entry_type {
            return Err(mismatch(
                at,
                format!(
                    "the lock records the type {:?}, and the document reaches a {:?}",
                    entry.entry_type, this.entry_type
                ),
            ));
        }
        let source = &entry.source;
        let kind = source.source_type;
        match &this.origin {
            Origin::Root | Origin::Local => {
                let recorded = source
                    .url
                    .as_deref()
                    .and_then(|url| url.strip_prefix("./"))
                    .and_then(|relative| {
                        normalize_inside(self.package_dir, &self.entry_dir, relative)
                    });
                if kind == SourceType::Local && source.url.is_some() && recorded.is_none() {
                    return Err(refuse(
                        PackageErrorCode::PathUnsafe,
                        at,
                        "the local url of the lock entry leaves the package folder",
                    ));
                }
                if kind != SourceType::Local
                    || recorded.as_deref() != this.path.as_deref()
                    || source.subpath.is_some()
                    || source.reference.is_some()
                    || source.registry.is_some()
                {
                    return Err(mismatch(
                        at,
                        format!(
                            "the lock records another source than the document: `{}` is at `{}`",
                            source.url.as_deref().unwrap_or(""),
                            this.path
                                .as_deref()
                                .map(|path| self.subject_path(path))
                                .unwrap_or_default()
                        ),
                    ));
                }
            }
            Origin::Git { url, subpath } => {
                if kind != SourceType::Git
                    || source.url.as_deref() != Some(url.as_str())
                    || source.subpath != *subpath
                {
                    return Err(mismatch(
                        at,
                        "the lock records another git repository or folder than the reference names",
                    ));
                }
            }
            Origin::Oci { url } => {
                if kind != SourceType::Oci
                    || source.url.as_deref() != Some(url.as_str())
                    || source.subpath.is_some()
                {
                    return Err(mismatch(
                        at,
                        "the lock records another artifact than the reference names",
                    ));
                }
            }
            Origin::Registry => {
                if !matches!(
                    kind,
                    SourceType::Git | SourceType::Oci | SourceType::Registry
                ) {
                    return Err(mismatch(
                        at,
                        "a part resolved through a registry is recorded as git, oci or registry",
                    ));
                }
                notes.push(Note {
                    code: NoteCode::SourceNotCompared,
                    subject: at.to_string(),
                    text: format!(
                        "{at}: the part was resolved through a registry; the source the lock records was not compared with the reference"
                    ),
                });
            }
            Origin::Below { parent } => {
                let outer = paths.first[*parent]
                    .as_deref()
                    .and_then(|at| self.lock.entries.get(at));
                let same = outer.is_some_and(|outer| {
                    outer.source.source_type == kind
                        && outer.source.url == source.url
                        && outer.version == entry.version
                        && outer.commit == entry.commit
                });
                if !same {
                    return Err(mismatch(
                        at,
                        "a local part of a fetched agent keeps that agent's source, version and commit",
                    ));
                }
            }
            Origin::RunningConnector | Origin::RunningAgent => {
                if kind != SourceType::Pointer {
                    return Err(mismatch(
                        at,
                        "something that is already running is recorded as a pointer",
                    ));
                }
            }
            Origin::Model { revision } => {
                if kind != SourceType::Pointer || entry.version != *revision {
                    return Err(mismatch(
                        at,
                        "a model is recorded as a pointer whose version is the revision",
                    ));
                }
            }
        }
        // Every reference that reaches the entry is checked against its own version.
        for (parent, index) in incoming {
            let edge = &self.nodes[*parent].edges[*index];
            if let Some(wanted) = &edge.wanted {
                check_selected(
                    edge.source_is_git,
                    wanted,
                    &entry.version,
                    entry.commit.as_deref(),
                    at,
                )?;
            }
        }
        // The lock may record the constraint the author wrote on the reference at this address.
        let written = home.and_then(|edge| edge.wanted.as_deref());
        match (written, entry.constraint.as_deref()) {
            (_, None) => {}
            (Some(written), Some(recorded)) if written == recorded => {}
            (Some(written), Some(recorded)) => {
                return Err(mismatch(
                    at,
                    format!(
                        "the lock records the constraint `{recorded}`, and the reference writes `{written}`"
                    ),
                ));
            }
            (None, Some(recorded)) => {
                return Err(mismatch(
                    at,
                    format!(
                        "the lock records the constraint `{recorded}`, and the reference selects no version"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// What the bytes of one entry are checked against, and where they are.
    #[allow(clippy::type_complexity)]
    fn integrity(
        &self,
        at: &str,
        node: usize,
        entry: &LockEntry,
    ) -> PackageResult<(
        Integrity,
        Option<PathBuf>,
        Option<DigestKind>,
        Option<FolderListing>,
    )> {
        let this = &self.nodes[node];
        let is_document = matches!(
            this.entry_type,
            EntryType::Agent | EntryType::AgentDefinition
        );
        match entry.source.source_type {
            SourceType::Pointer => Ok((Integrity::Pointer, None, None, None)),
            SourceType::Oci | SourceType::Registry => {
                // An artifact digest cannot be worked out from files alone. The bytes are used
                // only when the package keeps the artifact they were unpacked from.
                let unpacked = match entry.source.source_type {
                    SourceType::Oci if !is_document => self.unpacked_artifact(at, entry)?,
                    _ => None,
                };
                let Some((listing, path)) = unpacked else {
                    return Ok((
                        Integrity::NotVerifiable,
                        None,
                        Some(DigestKind::Artifact),
                        None,
                    ));
                };
                check_entry_file(this.entry_type, &Content::Folder(listing.clone()), at)?;
                Ok((
                    Integrity::Verified,
                    Some(path),
                    Some(DigestKind::Artifact),
                    Some(listing),
                ))
            }
            SourceType::Local | SourceType::Git => {
                let fetched = entry.source.source_type == SourceType::Git;
                let content = match &this.content {
                    Some(content) => Some((content.clone(), this.path.clone())),
                    None if fetched && !is_document => self.fetched_content(at)?,
                    None => None,
                };
                let Some((content, path)) = content else {
                    if fetched {
                        let kind = if is_document {
                            DigestKind::File
                        } else {
                            DigestKind::Folder
                        };
                        return Ok((Integrity::Absent, None, Some(kind), None));
                    }
                    return Err(refuse(
                        PackageErrorCode::PartMissing,
                        at,
                        "the lock pins this local part, and its bytes are not in the package",
                    ));
                };
                if entry.digest.as_deref() != Some(content.digest()) {
                    let shown = path
                        .as_deref()
                        .map(|path| self.subject_path(path))
                        .unwrap_or_default();
                    return Err(digest_mismatch(at, entry, content.digest(), &shown));
                }
                if fetched {
                    check_entry_file(this.entry_type, &content, at)?;
                }
                Ok(match content {
                    Content::File(_) => (Integrity::Verified, path, Some(DigestKind::File), None),
                    Content::Folder(listing) => (
                        Integrity::Verified,
                        path,
                        Some(DigestKind::Folder),
                        Some(listing),
                    ),
                })
            }
        }
    }

    /// The files of a part pinned as an OCI artifact, when the package keeps that artifact.
    ///
    /// The walk starts at the lock's digest and trusts nothing else: the
    /// manifest kept under that digest, the one layer it names, and the files
    /// in the layer. The files at the layout path of the address must be
    /// exactly those. None when the package keeps no manifest of that digest.
    fn unpacked_artifact(
        &self,
        at: &str,
        entry: &LockEntry,
    ) -> PackageResult<Option<(FolderListing, PathBuf)>> {
        let Some(digest) = entry.digest.as_deref() else {
            return Ok(None);
        };
        let store = self.package_dir.join(STORE_DIR);
        refuse_links(self.package_dir, &store.join("blobs").join("sha256"), at)?;
        let Some(artifact) = find_artifact(&store, digest, at)? else {
            return Ok(None);
        };
        let pinned = artifact.listing(at)?;
        let path = layout_path(&self.entry_dir, at)?;
        refuse_links(self.package_dir, &path, at)?;
        let found = match is_dir(&path) {
            true => Some(folder_listing(&path, Some(&self.lock_path()), at)?),
            false => None,
        };
        match found {
            Some(listing) if listing.files == pinned => Ok(Some((listing, path))),
            _ => Err(refuse(
                PackageErrorCode::DigestMismatch,
                at,
                format!(
                    "the files at `{}` are not the files of the artifact {digest} the lock pins",
                    self.subject_path(&path)
                ),
            )),
        }
    }

    /// The bytes of a fetched part, when they are at the layout path of its address.
    fn fetched_content(&self, at: &str) -> PackageResult<Option<(Content, Option<PathBuf>)>> {
        let path = layout_path(&self.entry_dir, at)?;
        refuse_links(self.package_dir, &path, at)?;
        if is_file(&path) {
            return Ok(Some((Content::File(file_digest(&path, at)?), Some(path))));
        }
        if is_dir(&path) {
            let listing = folder_listing(&path, Some(&self.lock_path()), at)?;
            if !listing.files.is_empty() {
                return Ok(Some((Content::Folder(listing), Some(path))));
            }
        }
        Ok(None)
    }
}

/// The fields of a locked part that come from the lock entry alone.
fn locked(at: &str, entry: &LockEntry) -> LockedPart {
    LockedPart {
        address: at.to_string(),
        entry_type: entry.entry_type,
        field: None,
        name: None,
        owner: "$".to_string(),
        stated_by: StatedBy::Agent,
        optional: false,
        pointer: None,
        source_text: None,
        version_text: None,
        source_type: entry.source.source_type,
        source_url: entry.source.url.clone(),
        source_subpath: entry.source.subpath.clone(),
        registry: entry.source.registry.clone(),
        version: entry.version.clone(),
        commit: entry.commit.clone(),
        digest: entry.digest.clone(),
        digest_kind: None,
        interface: entry.interface.clone(),
        retracted: entry.retracted.clone(),
        dependencies: entry.dependencies.clone(),
        path: None,
        integrity: Integrity::NotChecked,
        values: None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::model::{PackageReading, ReadRequest};
    use super::super::reader::read_package;
    use super::super::reader::testing::*;
    use super::*;

    fn refusal(package: &Package) -> (PackageErrorCode, String) {
        let error = package.read().expect_err("refused");
        (error.code, error.subject)
    }

    fn part<'a>(reading: &'a PackageReading, at: &str) -> &'a LockedPart {
        reading
            .parts
            .iter()
            .find(|part| part.address == at)
            .unwrap_or_else(|| panic!("no entry at {at}"))
    }

    /// Writes `agent.json` and a lock whose root lists `dependencies`, then `entries`.
    fn write_agent(
        package: &Package,
        document: &Value,
        dependencies: &[&str],
        entries: Vec<Value>,
    ) {
        package.write_json("agent.json", document);
        let mut all =
            vec![package.file_entry("$", "agent", "agent.json", "./agent.json", dependencies)];
        all.extend(entries);
        package.write_lock("", all);
    }

    fn pointer_entry(at: &str, kind: &str, version: &str, pin: &str) -> Value {
        json!({
            "address": at, "type": kind, "source": {"type": "pointer"}, "version": version,
            "constraint": version, "interface": {"declared_version": version, pin: MADE_UP},
            "dependencies": []
        })
    }

    const GIT_SKILL: &str = "git::https://git.example.com/parts.git//skills/review";

    #[test]
    fn an_optional_reference_the_lock_leaves_out_is_reported() {
        let package = Package::new("left-out");
        let document = agent(
            "lead",
            json!({
                "skills": {
                    "local": {"source": "./skills/none", "optional": true},
                    "fetched": {"source": GIT_SKILL, "version": "~> 1.7", "optional": true}
                },
                "agents": {"translator": {"source": "a2a::https://agents.example.com/translator", "version": "2.1.0", "optional": true}}
            }),
        );
        write_agent(&package, &document, &[], vec![]);
        let reading = package
            .read()
            .expect("optional references never stop a load");
        let left: Vec<(&str, &str, &str, &str)> = reading
            .left_out
            .iter()
            .map(|part| {
                (
                    part.field.as_str(),
                    part.name.as_str(),
                    part.owner.as_str(),
                    part.pointer.as_str(),
                )
            })
            .collect();
        assert_eq!(
            left,
            [
                ("agents", "translator", "$", "/agents/translator"),
                ("skills", "fetched", "$", "/skills/fetched"),
                ("skills", "local", "$", "/skills/local"),
            ]
        );
        assert_eq!(reading.left_out[1].source_text, GIT_SKILL);
        assert_eq!(reading.left_out[1].stated_by, StatedBy::Agent);
        assert_eq!(reading.parts.len(), 1);
        assert!(reading.part(PartField::Skills, "fetched").is_none());

        // The same references without `optional` are parts the lock must pin.
        let mut required = document.clone();
        required["skills"]["fetched"]
            .as_object_mut()
            .expect("entry")
            .remove("optional");
        write_agent(&package, &required, &[], vec![]);
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::LockEntryMissing,
                "$/skills/fetched".to_string()
            )
        );
        let mut required = document.clone();
        required["skills"]["local"]
            .as_object_mut()
            .expect("entry")
            .remove("optional");
        write_agent(&package, &required, &[], vec![]);
        assert_eq!(
            refusal(&package),
            (PackageErrorCode::PartMissing, "$/skills/local".to_string())
        );
    }

    #[test]
    fn an_optional_part_that_the_lock_pins_is_checked_like_any_other() {
        let package = Package::new("optional-pinned");
        package.write("skills/review/SKILL.md", SKILL);
        let document = agent(
            "lead",
            json!({"skills": {"review": {"source": "./skills/review", "optional": true}}}),
        );
        let entry = package.folder_entry(
            "$/skills/review",
            "skill",
            "skills/review",
            "./skills/review",
        );
        write_agent(
            &package,
            &document,
            &["$/skills/review"],
            vec![entry.clone()],
        );
        let reading = package.read().expect("pinned and present");
        assert!(part(&reading, "$/skills/review").optional);
        assert!(reading.left_out.is_empty());
        // A pinned part that is gone fails its pin, optional or not.
        fs::remove_dir_all(package.root.join("skills/review")).expect("remove");
        assert_eq!(
            refusal(&package),
            (PackageErrorCode::PartMissing, "$/skills/review".to_string())
        );
        // An optional part that is present and not pinned is left out: unpinned bytes are not used.
        package.write("skills/review/SKILL.md", SKILL);
        write_agent(&package, &document, &[], vec![]);
        let reading = package.read().expect("left out");
        assert_eq!(reading.left_out.len(), 1);
        assert_eq!(reading.parts.len(), 1);
    }

    #[test]
    fn fetched_content_is_absent_verified_or_not_verifiable() {
        let package = Package::new("fetched");
        let document = agent(
            "lead",
            json!({
                "model": {"source": "ollama::llama3.2:3b", "revision": "a80c4f17acd5"},
                "skills": {"review": {"source": GIT_SKILL, "version": "~> 1.7"}},
                "plugins": {"remotion": {"source": "oci::registry.example.com/plugins/remotion", "version": "1.2.0"}},
                "packages": {"whisper": {"source": "pypi::faster-whisper", "version": "1.2.1"}},
                "connectors": {"assets": {"source": "mcp::https://mcp.example.com/assets", "version": "2.3.0"}},
                "agents": {"translator": {"source": "a2a::https://agents.example.com/translator", "version": "2.1.0"}}
            }),
        );
        let dependencies = [
            "$/agents/translator",
            "$/connectors/assets",
            "$/model",
            "$/packages/whisper",
            "$/plugins/remotion",
            "$/skills/review",
        ];
        let entries = |skill_digest: &str| {
            vec![
                pointer_entry("$/agents/translator", "agent", "2.1.0", "card_digest"),
                pointer_entry("$/connectors/assets", "connector", "2.3.0", "tools_hash"),
                json!({"address": "$/model", "type": "model", "source": {"type": "pointer"}, "version": "a80c4f17acd5",
                       "interface": {"declared_version": "a80c4f17acd5"}, "dependencies": []}),
                json!({"address": "$/packages/whisper", "type": "package",
                       "source": {"type": "registry", "url": "faster-whisper", "registry": "pypi"},
                       "version": "1.2.1", "constraint": "1.2.1", "digest": MADE_UP, "dependencies": []}),
                json!({"address": "$/plugins/remotion", "type": "plugin",
                       "source": {"type": "oci", "url": "registry.example.com/plugins/remotion"},
                       "version": "1.2.0", "constraint": "1.2.0", "digest": MADE_UP, "dependencies": []}),
                git_entry(
                    "$/skills/review",
                    "skill",
                    "skills/review",
                    "1.7.2",
                    "~> 1.7",
                    skill_digest,
                ),
            ]
        };
        write_agent(&package, &document, &dependencies, entries(MADE_UP));
        let reading = package.read().expect("nothing fetched is here");
        let found: Vec<(&str, Integrity, Option<DigestKind>)> = reading
            .parts
            .iter()
            .map(|part| (part.address.as_str(), part.integrity, part.digest_kind))
            .collect();
        assert_eq!(
            found,
            [
                ("$", Integrity::Verified, Some(DigestKind::File)),
                ("$/agents/translator", Integrity::Pointer, None),
                ("$/connectors/assets", Integrity::Pointer, None),
                ("$/model", Integrity::Pointer, None),
                (
                    "$/packages/whisper",
                    Integrity::NotVerifiable,
                    Some(DigestKind::Artifact)
                ),
                (
                    "$/plugins/remotion",
                    Integrity::NotVerifiable,
                    Some(DigestKind::Artifact)
                ),
                (
                    "$/skills/review",
                    Integrity::Absent,
                    Some(DigestKind::Folder)
                ),
            ]
        );
        assert!(reading.parts.iter().skip(1).all(|part| part.path.is_none()));
        let skill = part(&reading, "$/skills/review");
        assert_eq!(skill.commit.as_deref(), Some(COMMIT));
        assert_eq!(skill.version, "1.7.2");
        assert_eq!(skill.version_text.as_deref(), Some("~> 1.7"));
        assert_eq!(skill.source_subpath.as_deref(), Some("skills/review"));
        assert_eq!(
            part(&reading, "$/connectors/assets")
                .interface
                .as_ref()
                .and_then(|pin| pin.tools_hash.as_deref()),
            Some(MADE_UP)
        );
        assert_eq!(
            part(&reading, "$/packages/whisper").registry.as_deref(),
            Some("pypi")
        );
        // A part resolved through a registry: its recorded source was accepted as it is.
        assert!(reading.notes.iter().any(|note| {
            note.code == NoteCode::SourceNotCompared && note.subject == "$/packages/whisper"
        }));
        assert!(reading.called.is_empty());

        // The fetched skill, placed at the layout path of its address, is checked like local bytes.
        package.write("skills/review/SKILL.md", SKILL);
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::DigestMismatch,
                "$/skills/review".to_string()
            )
        );
        write_agent(
            &package,
            &document,
            &dependencies,
            entries(&package.folder_digest("skills/review")),
        );
        let reading = package.read().expect("fetched bytes match the pin");
        let skill = part(&reading, "$/skills/review");
        assert_eq!(skill.integrity, Integrity::Verified);
        assert_eq!(
            skill.path.as_deref(),
            Some(reading.package_dir.join("skills/review").as_path())
        );
        assert!(reading.listings.contains_key("$/skills/review"));
        // Content under an artifact digest is never used, also when files sit at its layout path.
        package.write("plugins/remotion/plugin.json", "{}");
        let reading = package.read().expect("still loads");
        assert_eq!(
            part(&reading, "$/plugins/remotion").integrity,
            Integrity::NotVerifiable
        );
        assert!(part(&reading, "$/plugins/remotion").path.is_none());
    }

    #[test]
    fn the_selected_version_and_the_recorded_constraint_are_compared() {
        let package = Package::new("versions");
        let document = agent(
            "lead",
            json!({"skills": {"review": {"source": GIT_SKILL, "version": "~> 1.7"}}}),
        );
        let with = |change: &dyn Fn(&mut Value)| {
            let mut entry = git_entry(
                "$/skills/review",
                "skill",
                "skills/review",
                "1.7.2",
                "~> 1.7",
                MADE_UP,
            );
            change(&mut entry);
            write_agent(&package, &document, &["$/skills/review"], vec![entry]);
            package.read().map(|_| ()).map_err(|error| error.code)
        };
        assert_eq!(with(&|_| {}), Ok(()));
        // The lock may leave the constraint out.
        assert_eq!(
            with(&|entry| {
                entry.as_object_mut().expect("entry").remove("constraint");
            }),
            Ok(())
        );
        assert_eq!(
            with(&|entry| entry["constraint"] = json!("~> 1.6")),
            Err(PackageErrorCode::LockEntryMismatch)
        );
        assert_eq!(
            with(&|entry| entry["version"] = json!("2.0.0")),
            Err(PackageErrorCode::VersionNotSelected)
        );
        assert_eq!(
            with(&|entry| entry["version"] = json!("1.8.0-rc1")),
            Err(PackageErrorCode::VersionNotSelected)
        );
        assert_eq!(
            with(&|entry| entry["source"]["subpath"] = json!("skills/other")),
            Err(PackageErrorCode::LockEntryMismatch)
        );
        assert_eq!(
            with(&|entry| entry["type"] = json!("hook")),
            Err(PackageErrorCode::LockEntryMismatch)
        );

        // A full commit as the version of a git source.
        let pinned = agent(
            "lead",
            json!({"skills": {"review": {"source": GIT_SKILL, "version": COMMIT}}}),
        );
        let commit_entry = |version: &str, commit: &str| {
            let mut entry = git_entry(
                "$/skills/review",
                "skill",
                "skills/review",
                version,
                COMMIT,
                MADE_UP,
            );
            entry["commit"] = json!(commit);
            entry
        };
        write_agent(
            &package,
            &pinned,
            &["$/skills/review"],
            vec![commit_entry(COMMIT, COMMIT)],
        );
        package
            .read()
            .expect("the commit is the version and the commit");
        write_agent(
            &package,
            &pinned,
            &["$/skills/review"],
            vec![commit_entry(&"f".repeat(40), &"f".repeat(40))],
        );
        assert_eq!(refusal(&package).0, PackageErrorCode::VersionNotSelected);
        // A commit on a source that is not git is refused.
        let oci = agent(
            "lead",
            json!({"plugins": {"p": {"source": "oci::registry.example.com/p", "version": COMMIT}}}),
        );
        write_agent(
            &package,
            &oci,
            &["$/plugins/p"],
            vec![
                json!({"address": "$/plugins/p", "type": "plugin", "source": {"type": "oci", "url": "registry.example.com/p"},
                        "version": COMMIT, "digest": MADE_UP, "dependencies": []}),
            ],
        );
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::VersionNotSelected,
                "$/plugins/p".to_string()
            )
        );
    }

    #[test]
    fn a_version_on_a_local_source_is_ignored_and_never_recorded() {
        let package = Package::new("local-version");
        package.write("skills/review/SKILL.md", SKILL);
        let document = agent(
            "lead",
            json!({"skills": {"review": {"source": "./skills/review", "version": "1.0.0"}}}),
        );
        let entry = package.folder_entry(
            "$/skills/review",
            "skill",
            "skills/review",
            "./skills/review",
        );
        write_agent(
            &package,
            &document,
            &["$/skills/review"],
            vec![entry.clone()],
        );
        let reading = package.read().expect("the version is ignored");
        assert!(reading.notes.iter().any(|note| {
            note.code == NoteCode::LocalVersionIgnored && note.subject == "$/skills/review"
        }));
        let mut recorded = entry;
        recorded["constraint"] = json!("1.0.0");
        write_agent(&package, &document, &["$/skills/review"], vec![recorded]);
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::LockEntryMismatch,
                "$/skills/review".to_string()
            )
        );
    }

    #[test]
    fn entries_below_what_was_not_read_are_accepted_and_not_checked() {
        let package = Package::new("below");
        let document = agent(
            "lead",
            json!({
                "agents": {"reviewer": {"source": "git::https://git.example.com/parts.git//reviewer", "version": "~> 1.0"}},
                "plugins": {"remotion": {"source": "./plugins/remotion"}}
            }),
        );
        package.write("plugins/remotion/plugin.json", "{}");
        let mut reviewer = git_entry(
            "$/agents/reviewer",
            "agent",
            "reviewer",
            "1.0.3",
            "~> 1.0",
            MADE_UP,
        );
        reviewer["dependencies"] = json!(["$/agents/reviewer/skills/checklist"]);
        let mut plugin = package.folder_entry(
            "$/plugins/remotion",
            "plugin",
            "plugins/remotion",
            "./plugins/remotion",
        );
        plugin["dependencies"] = json!(["$/plugins/remotion/hooks/guard"]);
        let mut guard = git_entry(
            "$/plugins/remotion/hooks/guard",
            "hook",
            "hooks/guard",
            "1.0.0",
            "1.0.0",
            MADE_UP,
        );
        guard["indirect"] = json!(true);
        guard["retracted"] = json!("superseded by 1.0.1");
        let entries = vec![
            reviewer,
            git_entry(
                "$/agents/reviewer/skills/checklist",
                "skill",
                "skills/checklist",
                "2.0.0",
                "2.0.0",
                MADE_UP,
            ),
            plugin,
            guard,
        ];
        write_agent(
            &package,
            &document,
            &["$/agents/reviewer", "$/plugins/remotion"],
            entries.clone(),
        );
        let reading = package.read().expect("entries below are accepted");
        assert_eq!(
            part(&reading, "$/agents/reviewer").integrity,
            Integrity::Absent
        );
        assert_eq!(
            part(&reading, "$/agents/reviewer").digest_kind,
            Some(DigestKind::File)
        );
        let below = part(&reading, "$/agents/reviewer/skills/checklist");
        assert_eq!(below.integrity, Integrity::NotChecked);
        assert_eq!(
            (below.owner.as_str(), below.field, below.path.as_ref()),
            ("$/agents/reviewer", None, None)
        );
        assert_eq!(
            part(&reading, "$/plugins/remotion").integrity,
            Integrity::Verified
        );
        assert_eq!(
            part(&reading, "$/plugins/remotion/hooks/guard").integrity,
            Integrity::NotChecked
        );
        // The agent was not read, so nothing is known about it as a called agent.
        assert!(reading.called.is_empty());

        // The same entry below a skill is an entry no document reaches.
        let skill_document = agent(
            "lead",
            json!({"skills": {"review": {"source": GIT_SKILL, "version": "1.7.2"}}}),
        );
        let mut skill = git_entry(
            "$/skills/review",
            "skill",
            "skills/review",
            "1.7.2",
            "1.7.2",
            MADE_UP,
        );
        skill["dependencies"] = json!(["$/skills/review/hooks/guard"]);
        write_agent(
            &package,
            &skill_document,
            &["$/skills/review"],
            vec![
                skill,
                git_entry(
                    "$/skills/review/hooks/guard",
                    "hook",
                    "hooks/guard",
                    "1.0.0",
                    "1.0.0",
                    MADE_UP,
                ),
            ],
        );
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::LockEntryExtra,
                "$/skills/review/hooks/guard".to_string()
            )
        );
    }

    #[test]
    fn a_retracted_selection_is_noted_and_does_not_stop_a_load() {
        let package = Package::new("retracted");
        let document = agent(
            "lead",
            json!({"skills": {"review": {"source": GIT_SKILL, "version": "1.7.2"}}}),
        );
        let mut entry = git_entry(
            "$/skills/review",
            "skill",
            "skills/review",
            "1.7.2",
            "1.7.2",
            MADE_UP,
        );
        entry["retracted"] = json!("leaks a token");
        write_agent(&package, &document, &["$/skills/review"], vec![entry]);
        let reading = package.read().expect("a retraction is reported");
        assert_eq!(
            part(&reading, "$/skills/review").retracted.as_deref(),
            Some("leaks a token")
        );
        let note = reading
            .notes
            .iter()
            .find(|note| note.code == NoteCode::Retracted)
            .expect("note");
        assert_eq!(note.subject, "$/skills/review");
        assert!(note.text.contains("leaks a token"));
    }

    #[test]
    fn the_lock_must_match_the_documents_entry_by_entry() {
        let package = Package::new("match");
        package.write("skills/review/SKILL.md", SKILL);
        package.write("skills/other/SKILL.md", SKILL);
        let document = agent(
            "lead",
            json!({"skills": {"review": {"source": "./skills/review"}}}),
        );
        let good = package.folder_entry(
            "$/skills/review",
            "skill",
            "skills/review",
            "./skills/review",
        );
        let other =
            package.folder_entry("$/skills/other", "skill", "skills/other", "./skills/other");
        write_agent(
            &package,
            &document,
            &["$/skills/review"],
            vec![good.clone()],
        );
        package.read().expect("the lock matches");

        // Equal content at another path is another source.
        let mut moved = good.clone();
        moved["source"]["url"] = json!("./skills/other");
        write_agent(&package, &document, &["$/skills/review"], vec![moved]);
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::LockEntryMismatch,
                "$/skills/review".to_string()
            )
        );
        // An entry no document reaches.
        write_agent(
            &package,
            &document,
            &["$/skills/other", "$/skills/review"],
            vec![other.clone(), good.clone()],
        );
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::LockEntryExtra,
                "$/skills/other".to_string()
            )
        );
        // A part the lock does not pin.
        write_agent(&package, &document, &[], vec![]);
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::LockEntryMissing,
                "$/skills/review".to_string()
            )
        );
        // A dependency that names no entry: the part is named as missing first.
        write_agent(&package, &document, &["$/skills/review"], vec![]);
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::LockEntryMissing,
                "$/skills/review".to_string()
            )
        );
        write_agent(
            &package,
            &document,
            &["$/skills/ghost", "$/skills/review"],
            vec![good.clone()],
        );
        assert_eq!(
            refusal(&package),
            (PackageErrorCode::LockInvalid, "$".to_string())
        );
        // Dependencies that are not the edges of the document.
        let mut listing = good.clone();
        listing["dependencies"] = json!(["$"]);
        write_agent(&package, &document, &["$/skills/review"], vec![listing]);
        assert_eq!(refusal(&package).0, PackageErrorCode::LockInvalid);
        let two = agent(
            "lead",
            json!({"skills": {"review": {"source": "./skills/review"}, "other": {"source": "./skills/other"}}}),
        );
        let mut nested = good.clone();
        nested["dependencies"] = json!(["$/skills/other"]);
        write_agent(
            &package,
            &two,
            &["$/skills/other", "$/skills/review"],
            vec![other, nested],
        );
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::LockEntryMismatch,
                "$/skills/review".to_string()
            )
        );
        // The root entry is the document that was loaded.
        package.write_json("agent.json", &document);
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent-definition",
                    "agent.json",
                    "./agent.json",
                    &["$/skills/review"],
                ),
                good,
            ],
        );
        assert_eq!(
            refusal(&package),
            (PackageErrorCode::LockEntryMismatch, "$".to_string())
        );
    }

    #[test]
    fn a_part_with_content_has_its_own_entry_file() {
        let package = Package::new("entry-file");
        package.write("connectors/queue/server.py", "print('ready')\n");
        package.write("hooks/guard", "#!/bin/sh\n");
        package.write("skills/review/notes.md", "notes");
        package.write("skills/single.md", SKILL);
        let check = |field: &str, source: &str| {
            let document = agent("lead", json!({field: {"part": {"source": source}}}));
            write_agent(&package, &document, &[], vec![]);
            package
                .read()
                .map(|_| ())
                .map_err(|error| (error.code, error.subject))
        };
        let at = |field: &str| (PackageErrorCode::PartEntrypoint, format!("$/{field}/part"));
        assert_eq!(
            check("connectors", "./connectors/queue"),
            Err(at("connectors"))
        );
        assert_eq!(check("hooks", "./hooks/guard"), Err(at("hooks")));
        assert_eq!(check("skills", "./skills/review"), Err(at("skills")));
        // One file can be a skill. The lock has no entry for it here, so it is named as missing.
        assert_eq!(
            check("skills", "./skills/single.md"),
            Err((
                PackageErrorCode::LockEntryMissing,
                "$/skills/part".to_string()
            ))
        );
        let document = agent(
            "lead",
            json!({"skills": {"part": {"source": "./skills/single.md"}}}),
        );
        let entry = package.file_entry(
            "$/skills/part",
            "skill",
            "skills/single.md",
            "./skills/single.md",
            &[],
        );
        write_agent(&package, &document, &["$/skills/part"], vec![entry]);
        let reading = package.read().expect("a skill of one file");
        assert_eq!(
            part(&reading, "$/skills/part").digest_kind,
            Some(DigestKind::File)
        );
        assert!(reading.listings.is_empty());
    }

    #[test]
    fn a_source_that_leaves_the_package_is_refused() {
        let package = Package::new("escape");
        for source in [
            "./../outside",
            "./.",
            "./skills/../..",
            "./skills#part",
            ".//skills",
        ] {
            let document = agent("lead", json!({"skills": {"review": {"source": source}}}));
            write_agent(&package, &document, &[], vec![]);
            assert_eq!(
                refusal(&package),
                (PackageErrorCode::PathUnsafe, "$/skills/review".to_string()),
                "{source}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_link_on_the_way_to_a_part_or_inside_it_is_refused() {
        let package = Package::new("links");
        package.write("real/review/SKILL.md", SKILL);
        std::os::unix::fs::symlink(package.root.join("real"), package.root.join("skills"))
            .expect("create link");
        let document = agent(
            "lead",
            json!({"skills": {"review": {"source": "./skills/review"}}}),
        );
        write_agent(&package, &document, &[], vec![]);
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::SymlinkRefused,
                "$/skills/review".to_string()
            )
        );
        let inside = Package::new("link-inside");
        inside.write("skills/review/SKILL.md", SKILL);
        std::os::unix::fs::symlink(
            inside.root.join("skills/review/SKILL.md"),
            inside.root.join("skills/review/alias.md"),
        )
        .expect("create link");
        write_agent(&inside, &document, &[], vec![]);
        assert_eq!(
            refusal(&inside),
            (
                PackageErrorCode::SymlinkRefused,
                "$/skills/review".to_string()
            )
        );
    }

    #[test]
    fn a_called_agent_is_an_agent_with_a_description_that_does_not_reach_itself() {
        let package = Package::new("called-rules");
        let lead = agent(
            "lead",
            json!({"agents": {"helper": {"source": "./agents/helper", "variables": {"tone": "kind"}}}}),
        );
        let check = |helper: Value| {
            package.write_json("agents/helper/agent.json", &helper);
            write_agent(&package, &lead, &[], vec![]);
            let error = package.read().expect_err("refused");
            (error.code, error.subject)
        };
        let mut silent = agent("helper", json!({"variables": {"tone": {"type": "string"}}}));
        silent
            .as_object_mut()
            .expect("object")
            .remove("description");
        assert_eq!(
            check(silent),
            (
                PackageErrorCode::CalledAgentDescription,
                "$/agents/helper".to_string()
            )
        );
        assert_eq!(
            check(definition(
                "helper",
                json!({"agent": {"source": "./agent.json"}})
            )),
            (
                PackageErrorCode::ReferenceType,
                "$/agents/helper".to_string()
            )
        );
        assert_eq!(
            check(agent("helper", json!({}))),
            (PackageErrorCode::VariableUndeclared, "tone".to_string())
        );
        assert_eq!(
            check(agent(
                "helper",
                json!({"variables": {"tone": {"type": "number"}}})
            )),
            (PackageErrorCode::VariableType, "tone".to_string())
        );
        assert_eq!(
            check(agent(
                "helper",
                json!({"variables": {"tone": {"type": "string"}}, "agents": {"lead": {"source": "./../../agent.json"}}})
            )),
            (
                PackageErrorCode::ReferenceCycle,
                "$/agents/helper/agents/lead".to_string()
            )
        );
        assert_eq!(
            check(agent(
                "helper",
                json!({"tools": [{"connector": "none", "tool": "x"}]})
            )),
            (PackageErrorCode::ToolConnectorUnknown, "none.x".to_string())
        );
        // A folder with no agent.json is nothing to load.
        fs::remove_file(package.root.join("agents/helper/agent.json")).expect("remove");
        write_agent(&package, &lead, &[], vec![]);
        assert_eq!(
            refusal(&package),
            (PackageErrorCode::PartMissing, "$/agents/helper".to_string())
        );
    }

    /// A definition whose agent was fetched from git and placed at `agent/`.
    fn fetched_agent_package(name: &str) -> Package {
        let package = Package::new(name);
        package.write_json(
            "agent-definition.json",
            &definition(
                "ada-notes",
                json!({
                    "agent": {"source": "git::https://git.example.com/parts.git//team-assistant", "version": "~> 3.0"},
                    "variables": {"job": "Draft release notes."},
                    "limits": {"max_turns": 25}
                }),
            ),
        );
        package.write_json(
            "agent/agent.json",
            &agent(
                "team-assistant",
                json!({
                    "instructions": {"var": "job"},
                    "variables": {"job": {"type": "string"}},
                    "limits": {"max_turns": 60},
                    "skills": {
                        "glossary": {"source": "./parts/glossary"},
                        "review": {"source": GIT_SKILL, "version": "~> 1.7"}
                    }
                }),
            ),
        );
        package.write(
            "agent/skills/glossary/SKILL.md",
            "---\nname: glossary\n---\nUse the glossary.\n",
        );
        package.write("agent/skills/review/SKILL.md", SKILL);
        let mut agent_entry = git_entry(
            "$/agent",
            "agent",
            "team-assistant",
            "3.2.0",
            "~> 3.0",
            &package.file_digest("agent/agent.json"),
        );
        agent_entry["dependencies"] = json!(["$/agent/skills/glossary", "$/agent/skills/review"]);
        // A local part of a fetched agent keeps that agent's source, version and commit.
        let mut glossary = git_entry(
            "$/agent/skills/glossary",
            "skill",
            "team-assistant/parts/glossary",
            "3.2.0",
            "",
            &package.folder_digest("agent/skills/glossary"),
        );
        glossary
            .as_object_mut()
            .expect("entry")
            .remove("constraint");
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
                agent_entry,
                glossary,
                git_entry(
                    "$/agent/skills/review",
                    "skill",
                    "skills/review",
                    "1.7.2",
                    "~> 1.7",
                    &package.folder_digest("agent/skills/review"),
                ),
            ],
        );
        package
    }

    #[test]
    fn a_definition_reads_an_agent_fetched_from_git_at_its_layout_path() {
        let package = fetched_agent_package("fetched-agent");
        let reading = package.read().expect("the fetched agent is read");
        assert_eq!(reading.agent_record.name, "team-assistant");
        assert_eq!(
            reading.agent_record.path,
            reading.package_dir.join("agent/agent.json")
        );
        assert_eq!(reading.agent["instructions"], "Draft release notes.");
        assert_eq!(reading.agent["limits"]["max_turns"], 25);
        assert!(
            reading
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );
        let glossary = reading
            .part(PartField::Skills, "glossary")
            .expect("the agent's local skill");
        assert_eq!(glossary.address, "$/agent/skills/glossary");
        assert_eq!(glossary.source_type, SourceType::Git);
        assert_eq!(
            glossary.path.as_deref(),
            Some(reading.package_dir.join("agent/skills/glossary").as_path())
        );
        let review = reading
            .part(PartField::Skills, "review")
            .expect("the agent's fetched skill");
        assert_eq!(review.owner, "$/agent");
        assert_eq!(review.stated_by, StatedBy::Agent);

        // The fetched document is checked against its pin before it is read.
        let text = fs::read_to_string(package.root.join("agent/agent.json")).expect("read");
        package.write(
            "agent/agent.json",
            text.replace("team-assistant", "team-assistent"),
        );
        assert_eq!(
            refusal(&package),
            (PackageErrorCode::DigestMismatch, "$/agent".to_string())
        );
        // With no bytes the agent cannot be used, and its parts are not checked.
        fs::remove_file(package.root.join("agent/agent.json")).expect("remove");
        assert_eq!(
            refusal(&package),
            (PackageErrorCode::AgentUnavailable, "$/agent".to_string())
        );
    }

    #[test]
    fn a_local_part_of_a_fetched_agent_keeps_the_source_of_that_agent() {
        let package = fetched_agent_package("fetched-child");
        let path = package.root.join(LOCK_FILE);
        let mut lock: Value =
            serde_json::from_slice(&fs::read(&path).expect("read")).expect("parse");
        lock["entries"][2]["source"]["url"] = json!("https://git.example.com/other.git");
        package.write_json(LOCK_FILE, &lock);
        assert_eq!(
            refusal(&package),
            (
                PackageErrorCode::LockEntryMismatch,
                "$/agent/skills/glossary".to_string()
            )
        );
    }

    #[test]
    fn an_identical_entry_of_a_definition_is_the_part_of_its_agent() {
        let package = Package::new("identical");
        package.write("skills/review/SKILL.md", SKILL);
        package.write_json(
            "agent.json",
            &agent(
                "team-assistant",
                json!({"skills": {"review": {"source": "./skills/review"}}}),
            ),
        );
        package.write_json(
            "agent-definition.json",
            &definition(
                "ada-notes",
                json!({"agent": {"source": "./agent.json"}, "skills": {"review": {"source": "./skills/review"}}}),
            ),
        );
        let entries = |root_dependencies: &[&str]| {
            vec![
                package.file_entry(
                    "$",
                    "agent-definition",
                    "agent-definition.json",
                    "./agent-definition.json",
                    root_dependencies,
                ),
                package.file_entry(
                    "$/agent",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &["$/agent/skills/review"],
                ),
                package.folder_entry(
                    "$/agent/skills/review",
                    "skill",
                    "skills/review",
                    "./skills/review",
                ),
            ]
        };
        package.write_lock("", entries(&["$/agent", "$/agent/skills/review"]));
        let reading = package.read().expect("identical entries are one part");
        assert_eq!(reading.parts.len(), 3);
        assert_eq!(
            reading
                .part(PartField::Skills, "review")
                .map(|part| part.address.as_str()),
            Some("$/agent/skills/review")
        );
        assert_eq!(
            reading.together.as_ref().expect("together").kept_once,
            ["skills.review"]
        );
        assert!(
            reading
                .notes
                .iter()
                .any(|note| note.code == NoteCode::EntryKeptOnce)
        );
        // The definition's own entry is an edge of the root, so the root lists the part.
        package.write_lock("", entries(&["$/agent"]));
        assert_eq!(
            refusal(&package),
            (PackageErrorCode::LockEntryMismatch, "$".to_string())
        );
    }

    #[test]
    fn a_part_an_agent_that_is_not_here_also_reaches_is_at_the_address_the_lock_records() {
        const REVIEWER: &str = "git::https://git.example.com/parts.git//reviewer";
        const SHARED: &str = "$/agents/reviewer/skills/review";
        let package = Package::new("shared-with-absent");
        // The lead and its reviewer use one skill. The reviewer's path to it sorts first.
        package.write("agents/reviewer/skills/review/SKILL.md", SKILL);
        let skill = package.folder_digest("agents/reviewer/skills/review");
        let lead = agent(
            "code-review-lead",
            json!({
                "agents": {"reviewer": {"source": REVIEWER, "version": "~> 2.0"}},
                "skills": {"review": {"source": GIT_SKILL, "version": "~> 1.7"}}
            }),
        );
        let write = |reviewer_digest: &str, reviewer_needs: &[&str], skill_entry: Value| {
            let mut reviewer = git_entry(
                "$/agents/reviewer",
                "agent",
                "reviewer",
                "2.1.0",
                "~> 2.0",
                reviewer_digest,
            );
            reviewer["dependencies"] = json!(reviewer_needs);
            write_agent(
                &package,
                &lead,
                &["$/agents/reviewer", SHARED],
                vec![reviewer, skill_entry],
            );
        };
        let recorded = |subpath: &str, constraint: &str| {
            git_entry(SHARED, "skill", subpath, "1.7.2", constraint, &skill)
        };

        // The reviewer's document is not in the package, so its path cannot be derived.
        write(MADE_UP, &[SHARED], recorded("skills/review", "~> 1.7"));
        let reading = package.read().expect("the lock records the shared skill");
        assert_eq!(
            part(&reading, "$/agents/reviewer").integrity,
            Integrity::Absent
        );
        let shared = reading
            .part(PartField::Skills, "review")
            .expect("the lead's skill");
        assert_eq!(shared.address, SHARED);
        assert_eq!(shared.integrity, Integrity::Verified);
        assert_eq!(
            shared.path.as_deref(),
            Some(
                reading
                    .package_dir
                    .join("agents/reviewer/skills/review")
                    .as_path()
            )
        );

        // An entry that this reference cannot have, or that the reviewer does not need, is
        // not the lead's skill.
        let missing = (
            PackageErrorCode::LockEntryMissing,
            "$/skills/review".to_string(),
        );
        write(MADE_UP, &[SHARED], recorded("skills/checklist", "~> 1.7"));
        assert_eq!(refusal(&package), missing);
        write(MADE_UP, &[SHARED], recorded("skills/review", "~> 1.6"));
        assert_eq!(refusal(&package), missing);
        write(MADE_UP, &[], recorded("skills/review", "~> 1.7"));
        assert_eq!(refusal(&package), missing);

        // With the reviewer's document in the package, its path is derived like any other.
        let uses_it = agent(
            "reviewer",
            json!({"skills": {"review": {"source": GIT_SKILL, "version": "~> 1.7"}}}),
        );
        package.write_json("agents/reviewer/agent.json", &uses_it);
        let document = package.file_digest("agents/reviewer/agent.json");
        write(&document, &[SHARED], recorded("skills/review", "~> 1.7"));
        let reading = package.read().expect("both paths lead to the one entry");
        assert!(
            reading
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );
        assert_eq!(part(&reading, SHARED).owner, "$/agents/reviewer");
        // A reviewer that does not use the skill leaves the lead's skill with no entry.
        package.write_json("agents/reviewer/agent.json", &agent("reviewer", json!({})));
        let document = package.file_digest("agents/reviewer/agent.json");
        write(&document, &[SHARED], recorded("skills/review", "~> 1.7"));
        assert_eq!(refusal(&package), missing);
    }

    #[test]
    fn the_address_of_each_reference_leads_to_its_entry() {
        let package = small_package("edges");
        let entry_path = package.root.join(AGENT_FILE);
        let document: Value =
            serde_json::from_slice(&fs::read(&entry_path).expect("read")).expect("parse");
        let lock_document: Value =
            serde_json::from_slice(&fs::read(package.root.join(LOCK_FILE)).expect("read"))
                .expect("parse");
        let lock = super::super::lock::check_lock(&lock_document, LOCK_FILE).expect("lock");
        let graph = verify_graph(&package.root, &entry_path, &document, &lock).expect("graph");
        assert_eq!(
            graph.edges.into_iter().collect::<Vec<_>>(),
            [
                (
                    "$/connectors/changelog".to_string(),
                    "$/connectors/changelog".to_string()
                ),
                ("$/skills/review".to_string(), "$/skills/review".to_string()),
            ]
        );
        assert_eq!(graph.documents.keys().collect::<Vec<_>>(), ["$"]);
        let _ = read_package(&package.root, &ReadRequest::default()).expect("reads");
    }

    #[test]
    fn agents_that_nest_too_deep_are_refused() {
        let package = Package::new("deep");
        let mut folder = String::new();
        for level in 0..=MAX_NESTING + 1 {
            let document = agent(
                &format!("level-{level}"),
                json!({"agents": {"next": {"source": "./next"}}}),
            );
            package.write_json(&format!("{folder}agent.json"), &document);
            folder.push_str("next/");
        }
        package.write_lock(
            "",
            vec![package.file_entry("$", "agent", "agent.json", "./agent.json", &[])],
        );
        assert_eq!(refusal(&package).0, PackageErrorCode::TooLarge);
    }

    #[test]
    fn values_have_one_text_for_one_set_of_values() {
        let text = |value: Value| values_text(value.as_object());
        assert_eq!(values_text(None), "");
        assert_eq!(
            text(json!({"a": 1, "b": "x"})),
            text(json!({"b": "x", "a": 1.0}))
        );
        assert_ne!(text(json!({"a": 1})), text(json!({"a": "1"})));
        assert_ne!(text(json!({"a": 1})), text(json!({"a": true})));
        assert_ne!(text(json!({})), values_text(None));
    }
}
