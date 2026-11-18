// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fetch the content a package's lock pins and its folder does not hold.
//!
//! This is the only file of the loader that asks a remote for anything, and
//! nothing on the load path calls it. The remote decides which bytes arrive.
//! The digest in the lock decides whether they are used: what is placed is
//! checked by the same reader a load uses.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::digest::{MAX_FOLDER_BYTES, MAX_FOLDER_DEPTH, MAX_FOLDER_FILES};
use super::document::{Source, parse_source, secret, without_password};
use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::graph::AGENT_FILE;
use super::json::MAX_DOCUMENT_BYTES;
use super::model::{
    DocumentKind, EntryType, Integrity, LockRecord, LockedPart, ReadRequest, SourceType,
};
use super::oci::{STORE_DIR, find_artifact, write_layout};
use super::paths::{display_inside, is_safe_relative, layout_path};
use super::reader::{CheckedPackage, check_package};
use super::version::is_commit;
use crate::error::Result;

/// The transports git may use. Neither runs a program of the address's choosing
/// nor sends content as plain text.
const ALLOWED_PROTOCOLS: &str = "https:ssh:file";
/// The longest one `git fetch` may run.
const FETCH_LIMIT: Duration = Duration::from_secs(300);
/// The longest one read from a fetched repository may run.
const READ_LIMIT: Duration = Duration::from_secs(60);
const POLL_INTERVAL: Duration = Duration::from_millis(10);
/// The largest folder listing git may write.
const MAX_LISTING_BYTES: u64 = 64 * 1024 * 1024;

static NEXT_WORK_DIR: AtomicU64 = AtomicU64::new(0);

/// What the caller chooses for one fetch. Every member may be left out.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct AgentPackageFetchOptions {
    /// The entry document, as a path inside the package, when it is not at the root.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry: Option<PathBuf>,
    /// The sha256 the caller holds for the lock file, as 64 lowercase hex
    /// characters, with or without `sha256:` before them. A lock file with
    /// another digest is refused before anything is fetched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_lock_sha256: Option<String>,
    /// A folder in the OCI image layout form that holds the artifacts the lock
    /// pins from an OCI registry. No registry is asked: the caller fills the
    /// folder with the tool they sign in with. Left out: OCI content is not
    /// fetched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oci_layout: Option<PathBuf>,
}

/// What one fetch placed and what it left.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentPackageFetchReport {
    /// The folder to load, absolute: `out_dir` when something was placed, else
    /// the package folder as it was given.
    pub package_dir: PathBuf,
    /// The lock file of that folder.
    pub lock: LockRecord,
    /// Every lock entry whose content this fetch placed.
    pub fetched: Vec<FetchedContent>,
    /// Every lock entry whose content is still not in the folder, with the reason.
    pub not_fetched: Vec<NotFetchedContent>,
}

/// One lock entry whose content was fetched, placed and checked against its pin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FetchedContent {
    /// The lock address.
    pub address: String,
    /// The entry's type.
    pub entry_type: EntryType,
    /// The recorded source address, with any password left out.
    pub source_url: String,
    /// The recorded path below the source address.
    pub source_subpath: Option<String>,
    /// The git commit the content was read at. None for an OCI artifact.
    pub commit: Option<String>,
    /// The lock's digest, which the placed bytes have.
    pub digest: String,
    /// Where the content was placed: a file for a document, else a folder or a file.
    pub path: PathBuf,
}

/// One lock entry whose content this fetch did not place.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct NotFetchedContent {
    /// The lock address.
    pub address: String,
    /// The entry's type.
    pub entry_type: EntryType,
    /// Where the lock says the content comes from.
    pub source_type: SourceType,
    /// The recorded source address, with any password left out.
    pub source_url: Option<String>,
    /// Why the content was not placed.
    pub reason: String,
}

/// Fetch the content the lock of a package pins, into a complete copy of the package.
///
/// `package_dir` is read with the loader's own reader first, so a package that
/// a load would refuse is refused before any network use. Every lock entry
/// with a git source whose bytes are not in the folder is then fetched at its
/// pinned commit with the `git` program, read as the bytes git stores, and
/// placed at the path of its lock address in a copy of the package. The copy
/// is read again, so every placed entry has the digest its lock entry pins,
/// and is then moved to `out_dir`.
///
/// The copy holds what the lock pins: the lock file, the entry document, and
/// the content of every entry the folder already holds. A file of the package
/// folder that no entry pins is not copied, and the package folder is never
/// written to.
///
/// `out_dir` must be missing or empty, and outside the package folder. When
/// nothing needs fetching, the package folder is returned as it is and
/// nothing is written.
///
/// A lock entry with an OCI source is read from `options.oci_layout`, a folder
/// in the OCI image layout form, and from nowhere else: no registry is asked.
/// The artifact is the manifest the lock's digest names, with one layer that
/// is an uncompressed tar archive. Its files are unpacked at the path of the
/// lock address, and its blobs are kept in `.oci` at the root of the package,
/// so a load can check the files against the lock's digest itself.
///
/// An optional entry that cannot be fetched is listed in `not_fetched`. A
/// required one is an error: [`crate::FabricError::AgentPackage`] with the
/// code `package_fetch_failed`.
///
/// The fetch resolves no version, keeps no cache and handles no credential:
/// `git` uses the sign-in the caller already has.
pub fn fetch_agent_package(
    package_dir: &Path,
    out_dir: &Path,
    options: AgentPackageFetchOptions,
) -> Result<AgentPackageFetchReport> {
    Ok(fetch(package_dir, out_dir, &options)?)
}

/// Fetch an agent package that is itself a folder in a git repository, then
/// complete it from its own lock as [`fetch_agent_package`] does.
///
/// `source` is written the way the format writes a git source:
/// `git::<repository>//<folder>`. `reference` is a tag, a branch or a full
/// commit. The folder is read at that commit as the bytes git stores and
/// written whole to `out_dir`, which this function always writes.
///
/// Nothing the caller holds pins a package that is fetched this way, and a tag
/// or a branch is a name the repository can move. The first row of `fetched`
/// has the address `$` and the commit the folder was read at. To pin more,
/// pass `expected_lock_sha256`: a lock file with another digest is refused
/// and nothing is left in `out_dir`.
pub fn fetch_agent_package_from_git(
    source: &str,
    reference: &str,
    out_dir: &Path,
    options: AgentPackageFetchOptions,
) -> Result<AgentPackageFetchReport> {
    Ok(fetch_from_git(source, reference, out_dir, &options)?)
}

fn read_request(options: &AgentPackageFetchOptions) -> ReadRequest {
    ReadRequest {
        entry: options.entry.clone(),
        expected_lock_sha256: options.expected_lock_sha256.clone(),
        ..ReadRequest::default()
    }
}

/// Whether this fetch can place the content of a lock entry the package does not hold.
fn wanted(part: &LockedPart, options: &AgentPackageFetchOptions) -> bool {
    match (part.source_type, part.integrity) {
        (SourceType::Git, Integrity::Absent) => true,
        (SourceType::Oci, Integrity::NotVerifiable) => {
            options.oci_layout.is_some() && !is_document(part)
        }
        _ => false,
    }
}

fn fetch(
    package_dir: &Path,
    out_dir: &Path,
    options: &AgentPackageFetchOptions,
) -> PackageResult<AgentPackageFetchReport> {
    let request = read_request(options);
    let source = check_package(package_dir, &request)?;
    if !source.graph.parts.iter().any(|part| wanted(part, options)) {
        return Ok(nothing_placed(source, &BTreeMap::new()));
    }
    let target = Target::new(out_dir, Some(&source.package_dir))?;
    copy_pinned(&source, &target.staging)?;
    let mut git = Git::new(target.work.path.join("git"))?;
    let completed = complete(&target.staging, &request, options, &mut git)?;
    if completed.placed.is_empty() {
        return Ok(nothing_placed(source, &completed.unreachable));
    }
    target.finish(completed, Vec::new())
}

fn fetch_from_git(
    source: &str,
    reference: &str,
    out_dir: &Path,
    options: &AgentPackageFetchOptions,
) -> PackageResult<AgentPackageFetchReport> {
    let (url, subpath) = match parse_source(source) {
        Source::Typed {
            kind,
            address,
            subpath,
        } if kind == "git" => (address, subpath),
        _ => {
            return Err(PackageError::new(
                PackageErrorCode::OptionsInvalid,
                "source",
                "a package in a git repository is written `git::<repository>//<folder>`",
            ));
        }
    };
    let target = Target::new(out_dir, None)?;
    let mut git = Git::new(target.work.path.join("git"))?;
    let commit = git.package(&url, reference, subpath.as_deref(), &target.staging)?;
    let completed = complete(&target.staging, &read_request(options), options, &mut git)?;
    let package = FetchedContent {
        address: completed.package.entry.address.clone(),
        entry_type: match completed.package.entry.kind {
            DocumentKind::Agent => EntryType::Agent,
            DocumentKind::AgentDefinition => EntryType::AgentDefinition,
        },
        source_url: without_password(&url),
        source_subpath: subpath,
        commit: Some(commit),
        digest: completed.package.entry.digest.clone(),
        path: target.out_dir.clone(),
    };
    target.finish(completed, vec![package])
}

/// Where one fetch writes: `out_dir`, and the work folder beside it that holds
/// the package until every placed entry is checked.
struct Target {
    out_dir: PathBuf,
    /// `out_dir` as the caller gave it, for messages.
    shown: String,
    work: WorkDir,
    staging: PathBuf,
}

impl Target {
    /// `package_dir` is the package the fetch reads, when it is a folder: `out_dir`
    /// must not be inside it.
    fn new(out_dir: &Path, package_dir: Option<&Path>) -> PackageResult<Self> {
        let shown = out_dir.display().to_string();
        let out_dir = std::path::absolute(out_dir)
            .map_err(|error| in_use(&shown, format!("the path cannot be used: {error}")))?;
        check_out_dir(&out_dir, &shown)?;
        // Links and `..` are followed before the two are compared, so no other name of a
        // folder inside the package gets past. Nothing is made before this holds.
        if package_dir.is_some_and(|package| followed(&out_dir).starts_with(followed(package))) {
            return Err(in_use(&shown, "it is inside the package folder"));
        }
        let work = WorkDir::beside(&out_dir, &shown)?;
        let staging = work.path.join("package");
        Ok(Self {
            out_dir,
            shown,
            work,
            staging,
        })
    }

    /// Move the checked package to `out_dir` and report what was placed.
    /// `fetched` holds the rows the caller already has.
    fn finish(
        mut self,
        completed: Completed,
        mut fetched: Vec<FetchedContent>,
    ) -> PackageResult<AgentPackageFetchReport> {
        let Completed {
            package,
            placed,
            unreachable,
        } = completed;
        let moved = |path: &Path| {
            self.out_dir
                .join(path.strip_prefix(&self.staging).unwrap_or(path))
        };
        // The reader refuses bytes that do not have the pinned digest, so every placed
        // entry is verified here. Only now does the package become `out_dir`.
        for part in &package.graph.parts {
            if !placed.contains(&part.address) {
                continue;
            }
            let (Integrity::Verified, Some(path)) = (part.integrity, &part.path) else {
                return Err(failed(
                    &part.address,
                    "the placed content was not verified against the lock",
                ));
            };
            fetched.push(FetchedContent {
                address: part.address.clone(),
                entry_type: part.entry_type,
                source_url: without_password(part.source_url.as_deref().unwrap_or_default()),
                source_subpath: part.source_subpath.clone(),
                commit: part.commit.clone(),
                digest: part.digest.clone().unwrap_or_default(),
                path: moved(path),
            });
        }
        let artifacts = package
            .graph
            .parts
            .iter()
            .filter(|part| part.source_type == SourceType::Oci)
            .filter_map(|part| part.digest.clone())
            .collect();
        write_layout(&self.staging.join(STORE_DIR), &artifacts).map_err(|error| {
            failed(
                STORE_DIR,
                format!("the index of the kept artifacts cannot be written: {error}"),
            )
        })?;
        let not_fetched = left(&package.graph.parts, &unreachable);
        let lock = LockRecord {
            path: moved(&package.lock.path),
            ..package.lock
        };
        // An empty folder is in the way of the rename on some systems.
        let _ = fs::remove_dir(&self.out_dir);
        fs::rename(&self.staging, &self.out_dir).map_err(|error| {
            in_use(
                &self.shown,
                format!("the fetched package cannot be moved there: {error}"),
            )
        })?;
        // The folders this fetch made above `out_dir` now lead to the package.
        self.work.made.clear();
        Ok(AgentPackageFetchReport {
            package_dir: self.out_dir.clone(),
            lock,
            fetched,
            not_fetched,
        })
    }
}

/// A package after everything this fetch can get was placed in it.
struct Completed {
    /// The last reading of the package, with every placed entry verified.
    package: CheckedPackage,
    /// The addresses of the entries that were placed.
    placed: BTreeSet<String>,
    /// The optional entries that could not be fetched, each with the reason.
    unreachable: BTreeMap<String, String>,
}

/// Place what the lock of the package in `staging` pins and the package does not hold.
///
/// An agent's document shows its own parts only after it is placed, so the
/// package is read again until no entry is left to fetch.
fn complete(
    staging: &Path,
    request: &ReadRequest,
    options: &AgentPackageFetchOptions,
    git: &mut Git,
) -> PackageResult<Completed> {
    let store = staging.join(STORE_DIR);
    let mut placed = BTreeSet::new();
    let mut unreachable = BTreeMap::new();
    loop {
        let package = check_package(staging, request)?;
        let absent: Vec<&LockedPart> = package
            .graph
            .parts
            .iter()
            .filter(|part| wanted(part, options) && !unreachable.contains_key(&part.address))
            .collect();
        if absent.is_empty() {
            return Ok(Completed {
                package,
                placed,
                unreachable,
            });
        }
        for part in absent {
            if !placed.insert(part.address.clone()) {
                return Err(failed(
                    &part.address,
                    "the content was placed and the package still does not hold it",
                ));
            }
            let done = match (part.source_type, &options.oci_layout) {
                (SourceType::Oci, Some(layout)) => {
                    place_artifact(layout, part, &package.entry_dir, &store)
                }
                _ => git.place(part, &package.entry_dir),
            };
            match done {
                Ok(()) => {}
                Err(error) if part.optional => {
                    placed.remove(&part.address);
                    unreachable.insert(
                        part.address.clone(),
                        format!("{}: {}", error.code.as_str(), error.message),
                    );
                }
                Err(error) => return Err(error),
            }
        }
    }
}

/// The report of a fetch that placed nothing: the package folder as it was given.
fn nothing_placed(
    source: CheckedPackage,
    unreachable: &BTreeMap<String, String>,
) -> AgentPackageFetchReport {
    AgentPackageFetchReport {
        package_dir: source.package_dir,
        lock: source.lock,
        fetched: Vec::new(),
        not_fetched: left(&source.graph.parts, unreachable),
    }
}

/// An agent or a definition: an entry that is pinned by its document.
fn is_document(part: &LockedPart) -> bool {
    matches!(
        part.entry_type,
        EntryType::Agent | EntryType::AgentDefinition
    )
}

/// Unpack what one OCI lock entry pins at the path of its address, and keep
/// the blobs it came from in the package. A load then walks from the lock's
/// digest to the files itself, and trusts nothing this fetch said.
fn place_artifact(
    layout: &Path,
    part: &LockedPart,
    entry_dir: &Path,
    store: &Path,
) -> PackageResult<()> {
    let at = part.address.as_str();
    let digest = part.digest.as_deref().unwrap_or_default();
    let Some(artifact) = find_artifact(layout, digest, at)? else {
        return Err(failed(
            at,
            format!(
                "the layout folder `{}` holds no blob {digest}",
                layout.display()
            ),
        ));
    };
    let folder = layout_path(entry_dir, at)?;
    if fs::read_dir(&folder).is_ok_and(|mut items| items.next().is_some()) {
        return Err(failed(
            at,
            "the package holds files at the path of this address, and nothing pins them",
        ));
    }
    let placed = artifact
        .unpack(&folder, at)
        .and_then(|()| artifact.keep(store, at));
    if placed.is_err() {
        // An entry that is not whole is not left in the package.
        let _ = fs::remove_dir_all(&folder);
    }
    placed
}

/// The entries whose content is not in the folder, each with the reason.
fn left(parts: &[LockedPart], unreachable: &BTreeMap<String, String>) -> Vec<NotFetchedContent> {
    parts
        .iter()
        .filter_map(|part| {
            let reason = match (part.integrity, part.source_type) {
                (Integrity::Absent | Integrity::NotVerifiable, _)
                    if unreachable.contains_key(&part.address) =>
                {
                    format!(
                        "the entry is optional and could not be fetched: {}",
                        unreachable[&part.address]
                    )
                }
                (Integrity::NotVerifiable, SourceType::Oci) if is_document(part) => {
                    "an agent pinned as an OCI artifact is not read: the lock holds one digest for the artifact and none for the agent's document"
                        .to_string()
                }
                (Integrity::NotVerifiable, SourceType::Oci) => {
                    "it is pinned as an OCI artifact, and no `oci_layout` folder was given"
                        .to_string()
                }
                (Integrity::NotVerifiable, _) => {
                    "it comes from a package registry, and the tool of that registry installs it"
                        .to_string()
                }
                _ => return None,
            };
            Some(NotFetchedContent {
                address: part.address.clone(),
                entry_type: part.entry_type,
                source_type: part.source_type,
                source_url: part.source_url.as_deref().map(without_password),
                reason,
            })
        })
        .collect()
}

fn failed(subject: &str, message: impl Into<String>) -> PackageError {
    PackageError::new(PackageErrorCode::FetchFailed, subject, message)
}

fn in_use(subject: &str, message: impl Into<String>) -> PackageError {
    PackageError::new(PackageErrorCode::OutDirInUse, subject, message)
}

/// `out_dir` is missing, or it is a folder that holds nothing.
fn check_out_dir(out_dir: &Path, shown: &str) -> PackageResult<()> {
    match fs::read_dir(out_dir).map(|mut items| items.next().is_none()) {
        Ok(true) => Ok(()),
        Ok(false) => Err(in_use(shown, "the folder exists and is not empty")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(in_use(
            shown,
            format!("it cannot be used as a folder: {error}"),
        )),
    }
}

/// `path` with each link and each `..` followed. Its last folders need not be there yet.
fn followed(path: &Path) -> PathBuf {
    let mut followed = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                followed.pop();
            }
            _ => {
                followed.push(component);
                if let Ok(real) = fs::canonicalize(&followed) {
                    followed = real;
                }
            }
        }
    }
    followed
}

/// The work folder of one fetch, beside `out_dir` and on its file system. It
/// holds the package until it is checked and the scratch repositories. When
/// the value is dropped the folder is removed, and so is each folder above
/// `out_dir` that this fetch made and that holds nothing.
struct WorkDir {
    path: PathBuf,
    /// The folders above `out_dir` that this fetch made, deepest first.
    made: Vec<PathBuf>,
}

impl WorkDir {
    fn beside(out_dir: &Path, shown: &str) -> PackageResult<Self> {
        let (Some(parent), Some(name)) = (out_dir.parent(), out_dir.file_name()) else {
            return Err(in_use(
                shown,
                "it does not name a folder below another folder",
            ));
        };
        let unusable = |error: std::io::Error| {
            in_use(shown, format!("no folder can be made beside it: {error}"))
        };
        let work = Self {
            path: parent.join(format!(
                ".{}.fetch-{}-{}",
                name.to_string_lossy(),
                std::process::id(),
                NEXT_WORK_DIR.fetch_add(1, Ordering::Relaxed)
            )),
            made: parent
                .ancestors()
                .take_while(|folder| fs::symlink_metadata(folder).is_err())
                .map(Path::to_path_buf)
                .collect(),
        };
        fs::create_dir_all(parent).map_err(unusable)?;
        #[cfg_attr(not(unix), allow(unused_mut))]
        let mut builder = fs::DirBuilder::new();
        // Fetched content can be private. No other user reads it while it is checked.
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(&work.path).map_err(unusable)?;
        Ok(work)
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
        // A folder that now holds the fetched package is not empty, and stays.
        for folder in &self.made {
            let _ = fs::remove_dir(folder);
        }
    }
}

/// Copy what the lock of a package pins into `staging`: the lock file, and the content of
/// each entry the reader verified. The entry document is one of them.
///
/// Nothing else in the package folder is read or copied, so a file that no entry pins is
/// never in the way: a render beside the documents, a link, the folder of a repository.
fn copy_pinned(source: &CheckedPackage, staging: &Path) -> PackageResult<()> {
    let copy = |path: &Path| -> PackageResult<()> {
        let uncopied = |error: std::io::Error| {
            PackageError::new(
                PackageErrorCode::FileUnreadable,
                display_inside(&source.package_dir, path),
                format!("the file cannot be copied: {error}"),
            )
        };
        let Ok(inside) = path.strip_prefix(&source.package_dir) else {
            return Err(PackageError::new(
                PackageErrorCode::PathUnsafe,
                path.display().to_string(),
                "the path is not inside the package folder",
            ));
        };
        let target = staging.join(inside);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(uncopied)?;
        }
        fs::copy(path, target).map(drop).map_err(uncopied)
    };
    copy(&source.lock.path)?;
    let store = source.package_dir.join(STORE_DIR);
    for part in &source.graph.parts {
        let (Integrity::Verified, Some(path)) = (part.integrity, &part.path) else {
            continue;
        };
        match source.graph.listings.get(&part.address) {
            Some(listing) => {
                for file in listing.files.keys() {
                    copy(&path.join(file))?;
                }
            }
            None => copy(path)?,
        }
        // Files unpacked from an artifact are verified against the artifact, so it goes along.
        let at = part.address.as_str();
        if part.source_type == SourceType::Oci
            && let Some(digest) = part.digest.as_deref()
            && let Some(artifact) = find_artifact(&store, digest, at)?
        {
            artifact.keep(&staging.join(STORE_DIR), at)?;
        }
    }
    Ok(())
}

/// One thing a folder holds at a commit, as `git ls-tree -l` lists it.
struct TreeEntry {
    mode: String,
    kind: String,
    id: String,
    size: u64,
    path: String,
}

impl TreeEntry {
    /// A regular file, and whether it is executable. A link or a submodule is refused.
    fn regular(&self, at: &str) -> PackageResult<bool> {
        let path = &self.path;
        match (self.kind.as_str(), self.mode.as_str()) {
            ("blob", "100644") => Ok(false),
            ("blob", "100755") => Ok(true),
            ("blob", "120000") => Err(PackageError::new(
                PackageErrorCode::SymlinkRefused,
                at,
                format!("`{path}` is a symbolic link"),
            )),
            ("commit", _) => Err(PackageError::new(
                PackageErrorCode::FileUnreadable,
                at,
                format!("`{path}` is a submodule, and a submodule is not followed"),
            )),
            _ => Err(PackageError::new(
                PackageErrorCode::FileUnreadable,
                at,
                format!("`{path}` is not a regular file"),
            )),
        }
    }
}

/// The records of `git ls-tree -l -z`: `<mode> <type> <id> <size>`, a tab, the path, a NUL.
fn parse_tree(listed: &[u8], at: &str) -> PackageResult<Vec<TreeEntry>> {
    let unread = || {
        failed(
            at,
            "git listed the folder in a form this step does not read",
        )
    };
    let mut entries = Vec::new();
    for record in listed.split(|byte| *byte == 0) {
        if record.is_empty() {
            continue;
        }
        let tab = record
            .iter()
            .position(|byte| *byte == b'\t')
            .ok_or_else(unread)?;
        let Ok(path) = std::str::from_utf8(&record[tab + 1..]) else {
            return Err(PackageError::new(
                PackageErrorCode::PathUnsafe,
                at,
                "a file name in the fetched folder is not UTF-8",
            ));
        };
        let head = std::str::from_utf8(&record[..tab]).map_err(|_| unread())?;
        let mut fields = head.split_ascii_whitespace();
        let (Some(mode), Some(kind), Some(id), Some(size)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            return Err(unread());
        };
        entries.push(TreeEntry {
            mode: mode.to_string(),
            kind: kind.to_string(),
            id: id.to_string(),
            // A folder has no size: git writes `-`.
            size: size.parse().unwrap_or(0),
            path: path.to_string(),
        });
    }
    Ok(entries)
}

/// The `git` program, run on scratch repositories that are bare: nothing is
/// ever checked out, so no attribute or filter of a repository changes a byte.
struct Git {
    scratch: PathBuf,
    /// The scratch repository of each address, so one address is asked once for each commit.
    repositories: BTreeMap<String, PathBuf>,
    /// The addresses git could not fetch from, each with what git said. An
    /// address that did not answer is not asked again for the next entry.
    unanswered: BTreeMap<String, String>,
}

impl Git {
    fn new(scratch: PathBuf) -> PackageResult<Self> {
        fs::create_dir(&scratch).map_err(|error| {
            failed(
                &scratch.display().to_string(),
                format!("the scratch folder cannot be made: {error}"),
            )
        })?;
        Ok(Self {
            scratch,
            repositories: BTreeMap::new(),
            unanswered: BTreeMap::new(),
        })
    }

    /// Run one git command on a scratch repository, with a time limit.
    ///
    /// What the command writes goes to `output`, and its error text to a file,
    /// so a long answer never waits on a full pipe. Git's own prompt is off,
    /// and git uses only the allowed transports. A program git starts, such
    /// as `ssh`, can still ask on a terminal. The limit ends git, not a
    /// program git started.
    fn run(
        &self,
        repository: &Path,
        arguments: &[&str],
        output: Stdio,
        limit: Duration,
    ) -> std::result::Result<(), String> {
        let errors = self.scratch.join("errors");
        let error_text = fs::File::create(&errors)
            .map_err(|error| format!("`{}` cannot be written: {error}", errors.display()))?;
        let mut child = Command::new("git")
            .arg("--git-dir")
            .arg(repository)
            .args(arguments)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_ALLOW_PROTOCOL", ALLOWED_PROTOCOLS)
            .stdin(Stdio::null())
            .stdout(output)
            .stderr(error_text)
            .spawn()
            .map_err(|error| format!("the `git` program cannot be started: {error}"))?;
        let deadline = Instant::now() + limit;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() < deadline => std::thread::sleep(POLL_INTERVAL),
                waited => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(match waited {
                        Err(error) => error.to_string(),
                        Ok(_) => format!("git ran longer than {} seconds", limit.as_secs()),
                    });
                }
            }
        };
        if status.success() {
            return Ok(());
        }
        let text = fs::read(&errors).unwrap_or_default();
        let text = String::from_utf8_lossy(&text).trim().to_string();
        Err(if text.is_empty() {
            format!("git ended with {status}")
        } else {
            text
        })
    }

    /// Run one git command whose answer is read whole.
    fn answer(
        &self,
        repository: &Path,
        arguments: &[&str],
        limit: Duration,
    ) -> std::result::Result<Vec<u8>, String> {
        let output = self.scratch.join("answer");
        let answer = fs::File::create(&output).map_err(|error| error.to_string())?;
        self.run(repository, arguments, answer.into(), limit)?;
        let size = fs::metadata(&output)
            .map_err(|error| error.to_string())?
            .len();
        if size > MAX_LISTING_BYTES {
            return Err(format!("git wrote more than {MAX_LISTING_BYTES} bytes"));
        }
        fs::read(&output).map_err(|error| error.to_string())
    }

    /// Whether the repository holds exactly this commit. The id of a tag or of a
    /// folder is not one.
    fn has_commit(&self, repository: &Path, commit: &str) -> bool {
        let peeled = format!("{commit}^{{commit}}");
        self.answer(
            repository,
            &["rev-parse", "--verify", "--quiet", &peeled],
            READ_LIMIT,
        )
        .is_ok_and(|found| found.trim_ascii() == commit.as_bytes())
    }

    /// The scratch repository of `url`, made on first use.
    fn scratch(&mut self, url: &str, at: &str) -> PackageResult<PathBuf> {
        // The address is passed after `--`. An address that looks like an option is
        // refused as well, so git can never read it as one.
        if url.starts_with('-') {
            return Err(failed(
                at,
                format!("the address `{}` starts with `-`", without_password(url)),
            ));
        }
        if let Some(repository) = self.repositories.get(url) {
            return Ok(repository.clone());
        }
        let repository = self
            .scratch
            .join(format!("{}.git", self.repositories.len()));
        self.run(
            &repository,
            &["init", "--quiet", "--bare"],
            Stdio::null(),
            READ_LIMIT,
        )
        .map_err(|error| {
            failed(
                at,
                format!("git could not make a scratch repository: {error}"),
            )
        })?;
        self.repositories
            .insert(url.to_string(), repository.clone());
        Ok(repository)
    }

    /// Ask `url` for one commit, tag or branch, and nothing around it.
    fn fetch_one(
        &self,
        repository: &Path,
        url: &str,
        wanted: &str,
    ) -> std::result::Result<(), String> {
        self.run(
            repository,
            &[
                "fetch",
                "--quiet",
                "--depth",
                "1",
                "--no-tags",
                "--no-recurse-submodules",
                "--",
                url,
                wanted,
            ],
            Stdio::null(),
            FETCH_LIMIT,
        )
    }

    /// The scratch repository of `url`, holding `commit`.
    fn repository(&mut self, url: &str, commit: &str, at: &str) -> PackageResult<PathBuf> {
        let repository = self.scratch(url, at)?;
        if self.has_commit(&repository, commit) {
            return Ok(repository);
        }
        if let Some(error) = self.unanswered.get(url) {
            return Err(cannot_fetch(at, url, error));
        }
        // The pinned commit alone. A server that gives no commit by its id is asked
        // for every branch and tag instead.
        let alone = self.fetch_one(&repository, url, commit);
        if alone.is_err() || !self.has_commit(&repository, commit) {
            let all = self.run(
                &repository,
                &[
                    "fetch",
                    "--quiet",
                    "--tags",
                    "--no-recurse-submodules",
                    "--",
                    url,
                    "+refs/heads/*:refs/remotes/origin/*",
                ],
                Stdio::null(),
                FETCH_LIMIT,
            );
            if let Err(error) = all {
                let refusal = cannot_fetch(at, url, &error);
                self.unanswered.insert(url.to_string(), error);
                return Err(refusal);
            }
        }
        if !self.has_commit(&repository, commit) {
            return Err(failed(
                at,
                format!(
                    "`{}` does not have the commit {commit}",
                    without_password(url)
                ),
            ));
        }
        Ok(repository)
    }

    /// The commit a tag, a branch or a full commit of `url` names, with the
    /// scratch repository that now holds it.
    fn resolve(
        &mut self,
        url: &str,
        reference: &str,
        at: &str,
    ) -> PackageResult<(PathBuf, String)> {
        if is_commit(reference) {
            return Ok((self.repository(url, reference, at)?, reference.to_string()));
        }
        // A name git could read as an option or as a mapping between refs never reaches git.
        if reference.is_empty()
            || reference.starts_with(['-', '+'])
            || reference.contains([':', ' ', '\t', '\n'])
        {
            return Err(failed(
                at,
                format!("`{reference}` is not a tag, a branch or a full commit"),
            ));
        }
        let repository = self.scratch(url, at)?;
        self.fetch_one(&repository, url, reference)
            .map_err(|error| cannot_fetch(at, url, &error))?;
        let found = self
            .answer(
                &repository,
                &["rev-parse", "--verify", "--quiet", "FETCH_HEAD^{commit}"],
                READ_LIMIT,
            )
            .map(|found| String::from_utf8_lossy(found.trim_ascii()).into_owned())
            .unwrap_or_default();
        if !is_commit(&found) {
            return Err(failed(
                at,
                format!(
                    "`{reference}` does not name a commit of `{}`",
                    without_password(url)
                ),
            ));
        }
        Ok((repository, found))
    }

    /// Fetch the folder of a package at a tag, a branch or a full commit and
    /// write it whole to `folder`. Returns the commit the folder was read at.
    fn package(
        &mut self,
        url: &str,
        reference: &str,
        subpath: Option<&str>,
        folder: &Path,
    ) -> PackageResult<String> {
        let at = "$";
        let (repository, commit) = self.resolve(url, reference, at)?;
        let named = self.named(&repository, &commit, subpath, at)?;
        if named.kind != "tree" {
            return Err(failed(
                at,
                format!("`{}` is not a folder at the commit {commit}", named.path),
            ));
        }
        for (destination, file) in self.folder(&repository, &named.id, at, folder)? {
            self.write(&repository, &file, &destination, at)?;
        }
        Ok(commit)
    }

    /// What one folder holds, with or without what its folders hold.
    fn tree(
        &self,
        repository: &Path,
        tree: &str,
        recursive: bool,
        at: &str,
    ) -> PackageResult<Vec<TreeEntry>> {
        let mut arguments = vec!["ls-tree", "-l", "-z"];
        if recursive {
            arguments.push("-r");
        }
        arguments.push(tree);
        let listed = self
            .answer(repository, &arguments, READ_LIMIT)
            .map_err(|error| failed(at, format!("git could not list `{tree}`: {error}")))?;
        parse_tree(&listed, at)
    }

    /// What the subpath of a lock entry names at the commit. With no subpath it
    /// is the folder of the commit itself.
    fn named(
        &self,
        repository: &Path,
        commit: &str,
        subpath: Option<&str>,
        at: &str,
    ) -> PackageResult<TreeEntry> {
        let Some(subpath) = subpath else {
            return Ok(TreeEntry {
                mode: "040000".to_string(),
                kind: "tree".to_string(),
                id: format!("{commit}^{{tree}}"),
                size: 0,
                path: String::new(),
            });
        };
        if !is_safe_relative(subpath) {
            return Err(PackageError::new(
                PackageErrorCode::PathUnsafe,
                at,
                format!("the subpath `{subpath}` is not a relative path inside the repository"),
            ));
        }
        let (folder, name) = subpath.rsplit_once('/').unwrap_or(("", subpath));
        self.tree(repository, &format!("{commit}:{folder}"), false, at)?
            .into_iter()
            .find(|entry| entry.path == name)
            .ok_or_else(|| failed(at, format!("the commit {commit} has no `{subpath}`")))
    }

    /// The files one lock entry pins, each with the place it goes to.
    ///
    /// An agent is pinned by its document alone, whether the subpath names the
    /// document or the folder that holds it. Any other entry is one file or a
    /// whole folder. Everything is checked here, before a byte is written.
    fn files(
        &self,
        repository: &Path,
        named: TreeEntry,
        part: &LockedPart,
        destination: &Path,
    ) -> PackageResult<Vec<(PathBuf, TreeEntry)>> {
        let at = part.address.as_str();
        let too_large =
            |message: String| PackageError::new(PackageErrorCode::TooLarge, at, message);
        if part.entry_type == EntryType::Agent {
            let document = if named.kind == "tree" {
                self.tree(repository, &named.id, false, at)?
                    .into_iter()
                    .find(|entry| entry.path == AGENT_FILE)
                    .ok_or_else(|| failed(at, format!("the folder has no `{AGENT_FILE}`")))?
            } else {
                named
            };
            document.regular(at)?;
            if document.size > MAX_DOCUMENT_BYTES {
                return Err(too_large(format!(
                    "the document is larger than {MAX_DOCUMENT_BYTES} bytes"
                )));
            }
            return Ok(vec![(destination.join(AGENT_FILE), document)]);
        }
        if named.kind != "tree" {
            named.regular(at)?;
            if named.size > MAX_FOLDER_BYTES {
                return Err(too_large(format!(
                    "the file is larger than {MAX_FOLDER_BYTES} bytes"
                )));
            }
            return Ok(vec![(destination.to_path_buf(), named)]);
        }
        self.folder(repository, &named.id, at, destination)
    }

    /// The regular files of one folder at a commit, each with the place it goes
    /// to below `destination`. A link or a submodule is refused, and the
    /// loader's folder bounds hold.
    fn folder(
        &self,
        repository: &Path,
        tree: &str,
        at: &str,
        destination: &Path,
    ) -> PackageResult<Vec<(PathBuf, TreeEntry)>> {
        let too_large =
            |message: String| PackageError::new(PackageErrorCode::TooLarge, at, message);
        let listed = self.tree(repository, tree, true, at)?;
        if listed.is_empty() {
            return Err(failed(at, "the folder holds no file at the pinned commit"));
        }
        if listed.len() > MAX_FOLDER_FILES {
            return Err(too_large(format!(
                "the folder holds more than {MAX_FOLDER_FILES} files"
            )));
        }
        let mut bytes = 0u64;
        let mut files = Vec::with_capacity(listed.len());
        for file in listed {
            file.regular(at)?;
            if !is_safe_relative(&file.path) {
                return Err(PackageError::new(
                    PackageErrorCode::PathUnsafe,
                    at,
                    format!("`{}` is not a path a package can hold", file.path),
                ));
            }
            if file.path.split('/').count() > MAX_FOLDER_DEPTH {
                return Err(too_large(format!(
                    "the folder nests deeper than {MAX_FOLDER_DEPTH} levels"
                )));
            }
            bytes = bytes.saturating_add(file.size);
            if bytes > MAX_FOLDER_BYTES {
                return Err(too_large(format!(
                    "the folder holds more than {MAX_FOLDER_BYTES} bytes"
                )));
            }
            files.push((destination.join(&file.path), file));
        }
        Ok(files)
    }

    /// Write one file as the bytes git stores for it, and keep its executable mark.
    fn write(
        &self,
        repository: &Path,
        file: &TreeEntry,
        destination: &Path,
        at: &str,
    ) -> PackageResult<()> {
        let unwritten =
            |error: String| failed(at, format!("`{}` could not be written: {error}", file.path));
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|error| unwritten(error.to_string()))?;
        }
        // A file is never written over. On a file system that does not keep case, two
        // names of one folder are one file, and the second would replace the first.
        let output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)
            .map_err(|error| {
                unwritten(match error.kind() {
                    std::io::ErrorKind::AlreadyExists => {
                        "another file of the folder is already at that path on this file system"
                            .to_string()
                    }
                    _ => error.to_string(),
                })
            })?;
        self.run(
            repository,
            &["cat-file", "blob", &file.id],
            output.into(),
            READ_LIMIT,
        )
        .map_err(unwritten)?;
        let written = fs::metadata(destination)
            .map_err(|error| unwritten(error.to_string()))?
            .len();
        if written != file.size {
            return Err(unwritten(format!(
                "git wrote {written} bytes and listed {}",
                file.size
            )));
        }
        #[cfg(unix)]
        if file.mode == "100755" {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(destination, fs::Permissions::from_mode(0o755))
                .map_err(|error| unwritten(error.to_string()))?;
        }
        Ok(())
    }

    /// Fetch what one lock entry pins and place it at the path of its address.
    fn place(&mut self, part: &LockedPart, entry_dir: &Path) -> PackageResult<()> {
        let at = part.address.as_str();
        let (Some(url), Some(commit)) = (part.source_url.as_deref(), part.commit.as_deref()) else {
            return Err(failed(at, "the lock entry records no address or no commit"));
        };
        let repository = self.repository(url, commit, at)?;
        let named = self.named(&repository, commit, part.source_subpath.as_deref(), at)?;
        let files = self.files(&repository, named, part, &layout_path(entry_dir, at)?)?;
        for (index, (destination, file)) in files.iter().enumerate() {
            if let Err(error) = self.write(&repository, file, destination, at) {
                // An entry that is not whole is not left in the package.
                for (written, _) in &files[..=index] {
                    let _ = fs::remove_file(written);
                }
                return Err(error);
            }
        }
        Ok(())
    }
}

fn cannot_fetch(at: &str, url: &str, error: &str) -> PackageError {
    failed(
        at,
        format!(
            "git could not fetch from `{}`: {}",
            without_password(url),
            hide_password(error, url)
        ),
    )
}

/// The error text of git, with the password of the address it was given left out.
fn hide_password(text: &str, url: &str) -> String {
    match secret(url) {
        Some((start, end)) if end > start => text.replace(&url[start..end], "***"),
        _ => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::super::digest::bytes_digest;
    use super::super::model::PartField;
    use super::super::oci::testing::{TAR_LAYER, tar, write_artifact};
    use super::super::reader::read_package;
    use super::super::reader::testing::{
        COMMIT, MADE_UP, Package, agent, definition, small_package,
    };
    use super::*;

    const STORYBOARD: &str = "---\nname: storyboard\ndescription: Plan the shots of a video\n---\nList every shot before rendering.\n";
    const CHECKLIST: &str =
        "---\nname: checklist\ndescription: Review a change\n---\nReport every item that fails.\n";

    fn git(folder: &Path, arguments: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(folder)
            .args([
                "-c",
                "user.name=Tests",
                "-c",
                "user.email=tests@example.com",
            ])
            .args(["-c", "commit.gpgsign=false"])
            .args(arguments)
            .output()
            .expect("git runs");
        assert!(
            output.status.success(),
            "git {arguments:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    /// A bare git repository the test makes, with one commit, reached as `file://`.
    struct Repository {
        /// The files that were committed. The folder is removed with the value.
        work: Package,
        url: String,
        commit: String,
    }

    impl Repository {
        /// Commits everything `work` holds and makes the bare copy to fetch from.
        fn of(work: Package) -> Self {
            git(&work.root, &["init", "--quiet"]);
            git(&work.root, &["add", "--all"]);
            git(
                &work.root,
                &["commit", "--quiet", "--message", "Add the shared parts"],
            );
            let commit = git(&work.root, &["rev-parse", "HEAD"]);
            git(
                &work.root,
                &["clone", "--quiet", "--bare", ".", "shared.git"],
            );
            let url = format!("file://{}", work.root.join("shared.git").display());
            Self { work, url, commit }
        }

        /// A reference to a folder of the repository, selected by the full commit.
        fn reference(&self, subpath: &str) -> Value {
            json!({"source": format!("git::{}//{subpath}", self.url), "version": self.commit})
        }

        /// The lock entry for a folder or a document of the repository.
        fn entry(&self, at: &str, kind: &str, subpath: &str, digest: &str) -> Value {
            json!({
                "address": at, "type": kind,
                "source": {"type": "git", "url": self.url, "subpath": subpath},
                "version": self.commit, "commit": self.commit, "digest": digest,
                "dependencies": []
            })
        }
    }

    /// The shared parts of a media team: a `storyboard` skill with a script, and the skill's digest.
    fn storyboard_parts(name: &str) -> (Package, String) {
        let parts = Package::new(name);
        parts.write("README.md", "Shared parts of the media team.\n");
        parts.write("skills/storyboard/SKILL.md", STORYBOARD);
        parts.write(
            "skills/storyboard/scripts/contact-sheet.sh",
            "#!/bin/sh\nexit 0\n",
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(
                parts
                    .root
                    .join("skills/storyboard/scripts/contact-sheet.sh"),
                fs::Permissions::from_mode(0o755),
            )
            .expect("mark the script executable");
        }
        let digest = parts.folder_digest("skills/storyboard");
        (parts, digest)
    }

    /// A video producer whose `storyboard` skill is pinned from the repository and is not in the folder.
    fn video_producer(repository: &Repository, digest: &str) -> Package {
        let package = Package::new("fetch-video-producer");
        package.write_json(
            "agent.json",
            &agent(
                "video-producer",
                json!({"skills": {"storyboard": repository.reference("skills/storyboard")}}),
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
                    &["$/skills/storyboard"],
                ),
                repository.entry("$/skills/storyboard", "skill", "skills/storyboard", digest),
            ],
        );
        package
    }

    fn integrity(parts: &[LockedPart], at: &str) -> Integrity {
        parts
            .iter()
            .find(|part| part.address == at)
            .unwrap_or_else(|| panic!("no entry {at}"))
            .integrity
    }

    fn names(folder: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(folder)
            .expect("list the folder")
            .map(|item| {
                item.expect("item")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        names
    }

    fn refusal(package: &Package, out_dir: &Path) -> PackageError {
        fetch(&package.root, out_dir, &AgentPackageFetchOptions::default()).expect_err("refused")
    }

    #[test]
    fn a_skill_pinned_from_git_is_fetched_and_verified() {
        let (parts, digest) = storyboard_parts("fetch-skill");
        let repository = Repository::of(parts);
        let package = video_producer(&repository, &digest);
        let before = package.read().expect("the package reads");
        assert_eq!(
            integrity(&before.parts, "$/skills/storyboard"),
            Integrity::Absent
        );
        let out = Package::new("fetch-skill-out");
        let out_dir = out.root.join("video-producer");

        let report =
            fetch_agent_package(&package.root, &out_dir, AgentPackageFetchOptions::default())
                .expect("the skill is fetched");

        assert_eq!(report.package_dir, out_dir);
        assert_eq!(report.lock.path, out_dir.join("oadl.lock.json"));
        assert_eq!(report.lock.sha256, before.lock.sha256);
        assert!(report.not_fetched.is_empty());
        assert_eq!(
            report.fetched,
            [FetchedContent {
                address: "$/skills/storyboard".to_string(),
                entry_type: EntryType::Skill,
                source_url: repository.url.clone(),
                source_subpath: Some("skills/storyboard".to_string()),
                commit: Some(repository.commit.clone()),
                digest,
                path: out_dir.join("skills/storyboard"),
            }]
        );
        let after = read_package(&out_dir, &ReadRequest::default()).expect("the result loads");
        assert!(
            after
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |path: &str| {
                fs::metadata(out_dir.join(path))
                    .expect("placed")
                    .permissions()
                    .mode()
                    & 0o111
            };
            assert_ne!(mode("skills/storyboard/scripts/contact-sheet.sh"), 0);
            assert_eq!(mode("skills/storyboard/SKILL.md"), 0);
        }
        // Only what the entry pins is placed. The package that was given is not written to,
        // and nothing is left beside the result.
        assert_eq!(names(&out_dir), ["agent.json", "oadl.lock.json", "skills"]);
        assert_eq!(names(&package.root), ["agent.json", "oadl.lock.json"]);
        assert_eq!(names(&out.root), ["video-producer"]);
        assert!(repository.work.root.join("README.md").exists());
    }

    #[test]
    fn a_called_agent_pinned_from_git_is_fetched_with_its_own_parts() {
        let parts = Package::new("fetch-called");
        parts.write_json(
            "reviewer/agent.json",
            &agent(
                "reviewer",
                json!({
                    "description": "Reviews a change against the checklist.",
                    "skills": {"checklist": {"source": "./skills/checklist"}}
                }),
            ),
        );
        parts.write("reviewer/skills/checklist/SKILL.md", CHECKLIST);
        parts.write(
            "reviewer/README.md",
            "The reviewer of the code review team.\n",
        );
        let document = parts.file_digest("reviewer/agent.json");
        let checklist = parts.folder_digest("reviewer/skills/checklist");
        let repository = Repository::of(parts);
        let package = Package::new("fetch-code-review");
        package.write_json(
            "agent.json",
            &agent(
                "code-review-lead",
                json!({"agents": {"reviewer": repository.reference("reviewer")}}),
            ),
        );
        let mut reviewer = repository.entry("$/agents/reviewer", "agent", "reviewer", &document);
        reviewer["dependencies"] = json!(["$/agents/reviewer/skills/checklist"]);
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &["$/agents/reviewer"],
                ),
                reviewer,
                // A local part of a fetched agent keeps that agent's source, version and commit.
                repository.entry(
                    "$/agents/reviewer/skills/checklist",
                    "skill",
                    "reviewer/skills/checklist",
                    &checklist,
                ),
            ],
        );
        let out = Package::new("fetch-called-out");
        // A folder that is there and holds nothing can be used.
        let out_dir = out.root.join("code-review");
        fs::create_dir(&out_dir).expect("an empty folder");

        let report = fetch(
            &package.root,
            &out_dir,
            &AgentPackageFetchOptions::default(),
        )
        .expect("the agent and its skill are fetched");

        let placed: Vec<(&str, PathBuf)> = report
            .fetched
            .iter()
            .map(|row| (row.address.as_str(), row.path.clone()))
            .collect();
        assert_eq!(
            placed,
            [
                (
                    "$/agents/reviewer",
                    out_dir.join("agents/reviewer/agent.json")
                ),
                (
                    "$/agents/reviewer/skills/checklist",
                    out_dir.join("agents/reviewer/skills/checklist")
                ),
            ]
        );
        // The document is what the agent's entry pins. The file beside it is not placed.
        assert_eq!(
            names(&out_dir.join("agents/reviewer")),
            ["agent.json", "skills"]
        );
        let after = read_package(&out_dir, &ReadRequest::default()).expect("the result loads");
        assert!(
            after
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );
        assert_eq!(after.called.len(), 1);
        assert_eq!(after.called[0].name, "reviewer");
    }

    #[test]
    fn a_definition_whose_agent_is_in_git_is_completed_from_its_lock() {
        let parts = Package::new("fetch-shared-agent");
        parts.write_json(
            "video-producer/agent.json",
            &agent(
                "video-producer",
                json!({"skills": {"storyboard": {"source": "./skills/storyboard"}}}),
            ),
        );
        parts.write("video-producer/skills/storyboard/SKILL.md", STORYBOARD);
        let document = parts.file_digest("video-producer/agent.json");
        let storyboard = parts.folder_digest("video-producer/skills/storyboard");
        let repository = Repository::of(parts);
        let package = Package::new("fetch-definition");
        package.write_json(
            "agent-definition.json",
            &definition(
                "video-producer-short",
                json!({"agent": repository.reference("video-producer"), "limits": {"max_turns": 20}}),
            ),
        );
        let mut shared = repository.entry("$/agent", "agent", "video-producer", &document);
        shared["dependencies"] = json!(["$/agent/skills/storyboard"]);
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
                shared,
                repository.entry(
                    "$/agent/skills/storyboard",
                    "skill",
                    "video-producer/skills/storyboard",
                    &storyboard,
                ),
            ],
        );
        // Nothing placed: a load refuses the definition, and the skill of its agent is not seen yet.
        assert_eq!(
            package.read().expect_err("refused").code,
            PackageErrorCode::AgentUnavailable
        );
        let request = ReadRequest::default();
        let first = check_package(&package.root, &request).expect("the folder is a package");
        assert_eq!(integrity(&first.graph.parts, "$/agent"), Integrity::Absent);
        assert_eq!(
            integrity(&first.graph.parts, "$/agent/skills/storyboard"),
            Integrity::NotChecked
        );
        let out = Package::new("fetch-definition-out");
        let out_dir = out.root.join("video-producer-short");

        let report = fetch(
            &package.root,
            &out_dir,
            &AgentPackageFetchOptions::default(),
        )
        .expect("the agent is fetched, then its skill");

        // The agent's document is placed first. Reading it shows the skill, which is placed next.
        let placed: Vec<&str> = report
            .fetched
            .iter()
            .map(|row| row.address.as_str())
            .collect();
        assert_eq!(placed, ["$/agent", "$/agent/skills/storyboard"]);
        let after = read_package(&out_dir, &request).expect("the definition loads with its agent");
        assert!(
            after
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );
        assert_eq!(after.agent_record.path, out_dir.join("agent/agent.json"));
        assert_eq!(after.agent["limits"]["max_turns"], 20);
    }

    #[test]
    fn content_that_does_not_have_the_pinned_digest_is_refused_and_nothing_is_left() {
        let (parts, digest) = storyboard_parts("fetch-changed");
        // One byte of the skill changes after its digest went into the lock.
        parts.write(
            "skills/storyboard/SKILL.md",
            STORYBOARD.replace("every", "Every"),
        );
        let repository = Repository::of(parts);
        let package = video_producer(&repository, &digest);
        let out = Package::new("fetch-changed-out");

        // The folders above `out_dir` are not there yet. The fetch makes them, and removes them.
        let error = refusal(&package, &out.root.join("releases/2026/video-producer"));

        assert_eq!(error.code, PackageErrorCode::DigestMismatch);
        assert_eq!(error.subject, "$/skills/storyboard");
        assert!(names(&out.root).is_empty());
    }

    #[test]
    fn a_commit_the_repository_does_not_have_is_refused() {
        let (parts, digest) = storyboard_parts("fetch-no-commit");
        let mut repository = Repository::of(parts);
        repository.commit = COMMIT.to_string();
        let package = video_producer(&repository, &digest);
        let out = Package::new("fetch-no-commit-out");

        let error = refusal(&package, &out.root.join("video-producer"));

        assert_eq!(error.code, PackageErrorCode::FetchFailed);
        assert_eq!(error.subject, "$/skills/storyboard");
        assert!(
            error
                .message
                .ends_with(&format!("does not have the commit {COMMIT}")),
            "{}",
            error.message
        );
        assert!(names(&out.root).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn a_link_in_a_fetched_folder_is_refused() {
        let (parts, _) = storyboard_parts("fetch-link");
        std::os::unix::fs::symlink("SKILL.md", parts.root.join("skills/storyboard/latest.md"))
            .expect("a link");
        let repository = Repository::of(parts);
        let package = video_producer(&repository, MADE_UP);
        let out = Package::new("fetch-link-out");

        let error = refusal(&package, &out.root.join("video-producer"));

        assert_eq!(error.code, PackageErrorCode::SymlinkRefused);
        assert_eq!(error.subject, "$/skills/storyboard");
        assert!(error.message.contains("`latest.md`"), "{}", error.message);
        assert!(names(&out.root).is_empty());
    }

    #[test]
    fn an_address_git_must_not_follow_is_refused() {
        let (parts, digest) = storyboard_parts("fetch-transport");
        let mut repository = Repository::of(parts);
        let out = Package::new("fetch-transport-out");
        // Plain `http` is not an allowed transport, so git stops before it connects.
        repository.url = "http://127.0.0.1:9/media-team/parts.git".to_string();
        let error = refusal(&video_producer(&repository, &digest), &out.root.join("one"));
        assert_eq!(error.code, PackageErrorCode::FetchFailed);
        assert!(error.message.contains("not allowed"), "{}", error.message);

        // An address that git could read as an option never reaches git.
        repository.url = "--upload-pack=touch".to_string();
        let error = refusal(&video_producer(&repository, &digest), &out.root.join("two"));
        assert_eq!(error.code, PackageErrorCode::FetchFailed);
        assert!(
            error.message.ends_with("starts with `-`"),
            "{}",
            error.message
        );
        assert!(names(&out.root).is_empty());
    }

    #[test]
    fn an_out_dir_that_is_in_use_is_refused() {
        let (parts, digest) = storyboard_parts("fetch-in-use");
        let repository = Repository::of(parts);
        let package = video_producer(&repository, &digest);
        let out = Package::new("fetch-in-use-out");
        out.write("video-producer/notes.txt", "kept\n");

        // The package folder by two other names: through `..`, and through a link.
        let name = package.root.file_name().expect("a folder name");
        let through_parent = out.root.join("staging/../..").join(name).join("build");
        let mut inside = vec![package.root.join("build"), through_parent];
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&package.root, out.root.join("current")).expect("a link");
            inside.push(out.root.join("current/build"));
        }
        for out_dir in [
            out.root.join("video-producer"),
            out.root.join("video-producer/notes.txt"),
        ]
        .into_iter()
        .chain(inside)
        {
            let error = refusal(&package, &out_dir);
            assert_eq!(
                error.code,
                PackageErrorCode::OutDirInUse,
                "{}",
                out_dir.display()
            );
        }
        assert_eq!(names(&out.root.join("video-producer")), ["notes.txt"]);
        assert_eq!(names(&package.root), ["agent.json", "oadl.lock.json"]);
        // Nothing was made on the way to a refused `out_dir`.
        assert!(!out.root.join("staging").exists());
    }

    #[test]
    fn an_optional_entry_that_cannot_be_fetched_is_listed_and_does_not_stop_the_fetch() {
        let (parts, digest) = storyboard_parts("fetch-optional");
        let repository = Repository::of(parts);
        let package = Package::new("fetch-optional-package");
        let captions = json!({
            "source": format!("git::{}//skills/captions", repository.url),
            "version": COMMIT,
            "optional": true
        });
        package.write_json(
            "agent.json",
            &agent(
                "video-producer",
                json!({"skills": {
                    "captions": captions,
                    "storyboard": repository.reference("skills/storyboard")
                }}),
            ),
        );
        let mut unreachable =
            repository.entry("$/skills/captions", "skill", "skills/captions", MADE_UP);
        unreachable["version"] = json!(COMMIT);
        unreachable["commit"] = json!(COMMIT);
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &["$/skills/captions", "$/skills/storyboard"],
                ),
                unreachable,
                repository.entry("$/skills/storyboard", "skill", "skills/storyboard", &digest),
            ],
        );
        let out = Package::new("fetch-optional-out");
        let out_dir = out.root.join("video-producer");

        let report = fetch(
            &package.root,
            &out_dir,
            &AgentPackageFetchOptions::default(),
        )
        .expect("the required skill is fetched");

        let placed: Vec<&str> = report
            .fetched
            .iter()
            .map(|row| row.address.as_str())
            .collect();
        assert_eq!(placed, ["$/skills/storyboard"]);
        let [left] = report.not_fetched.as_slice() else {
            panic!("one row: {:?}", report.not_fetched);
        };
        assert_eq!(left.address, "$/skills/captions");
        assert_eq!(left.source_type, SourceType::Git);
        assert!(
            left.reason.contains("package_fetch_failed"),
            "{}",
            left.reason
        );
        assert!(
            left.reason.contains("does not have the commit"),
            "{}",
            left.reason
        );
        let after = read_package(&out_dir, &ReadRequest::default()).expect("the result loads");
        assert_eq!(
            integrity(&after.parts, "$/skills/captions"),
            Integrity::Absent
        );
        assert_eq!(
            integrity(&after.parts, "$/skills/storyboard"),
            Integrity::Verified
        );
    }

    #[test]
    fn only_what_the_lock_pins_goes_into_the_fetched_package() {
        let (parts, digest) = storyboard_parts("fetch-pinned");
        let repository = Repository::of(parts);
        // A package written by hand in a project folder: one skill of its own, a render
        // beside the documents, and a link to that render.
        let package = Package::new("fetch-pinned-package");
        package.write(
            "skills/captions/SKILL.md",
            "---\nname: captions\ndescription: Write captions\n---\nWrite one caption for each shot.\n",
        );
        package.write("renders/draft.mp4", "a render, not a part\n");
        #[cfg(unix)]
        std::os::unix::fs::symlink("renders/draft.mp4", package.root.join("latest.mp4"))
            .expect("a link");
        package.write_json(
            "agent.json",
            &agent(
                "video-producer",
                json!({"skills": {
                    "captions": {"source": "./skills/captions"},
                    "storyboard": repository.reference("skills/storyboard")
                }}),
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
                    &["$/skills/captions", "$/skills/storyboard"],
                ),
                package.folder_entry(
                    "$/skills/captions",
                    "skill",
                    "skills/captions",
                    "./skills/captions",
                ),
                repository.entry("$/skills/storyboard", "skill", "skills/storyboard", &digest),
            ],
        );
        let out = Package::new("fetch-pinned-out");
        let out_dir = out.root.join("video-producer");

        fetch(
            &package.root,
            &out_dir,
            &AgentPackageFetchOptions::default(),
        )
        .expect("what no entry pins is not in the way");

        assert_eq!(names(&out_dir), ["agent.json", "oadl.lock.json", "skills"]);
        assert_eq!(names(&out_dir.join("skills")), ["captions", "storyboard"]);
        let after = read_package(&out_dir, &ReadRequest::default()).expect("the result loads");
        assert!(
            after
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );
    }

    /// The digest of a folder that holds these files, each with its text.
    fn folder_digest_of(files: &[(&str, &str)]) -> String {
        let mut pairs: Vec<[String; 2]> = files
            .iter()
            .map(|(path, text)| [path.to_string(), bytes_digest(text.as_bytes())])
            .collect();
        pairs.sort();
        bytes_digest(&serde_json::to_vec(&pairs).expect("the pairs serialize"))
    }

    #[test]
    fn two_names_that_differ_only_in_case_never_become_one_file() {
        // The skill holds `notes.md` and `Notes.md`. A file system that does not keep case
        // cannot hold both, so the commit is written with no working folder.
        let work = Package::new("fetch-case");
        git(&work.root, &["init", "--quiet"]);
        for (path, text) in [
            ("SKILL.md", STORYBOARD),
            ("notes.md", "first\n"),
            ("Notes.md", "second\n"),
        ] {
            work.write("text", text);
            let blob = git(&work.root, &["hash-object", "-w", "text"]);
            let entry = format!("100644,{blob},skills/storyboard/{path}");
            let add = ["update-index", "--add", "--cacheinfo", &entry];
            git(
                &work.root,
                &[&["-c", "core.ignorecase=false"], &add[..]].concat(),
            );
        }
        let tree = git(&work.root, &["write-tree"]);
        let commit = git(
            &work.root,
            &["commit-tree", &tree, "-m", "Add the storyboard skill"],
        );
        git(&work.root, &["update-ref", "HEAD", &commit]);
        git(
            &work.root,
            &["clone", "--quiet", "--bare", ".", "shared.git"],
        );
        let url = format!("file://{}", work.root.join("shared.git").display());
        let repository = Repository { work, url, commit };
        let out = Package::new("fetch-case-out");
        fs::write(out.root.join("probe"), "").expect("a file");
        let keeps_case = !out.root.join("PROBE").exists();
        fs::remove_file(out.root.join("probe")).expect("remove the file");

        // The bytes of one file under the name of the other: a folder the commit does not hold.
        let mixed = folder_digest_of(&[("SKILL.md", STORYBOARD), ("Notes.md", "first\n")]);
        let error = refusal(
            &video_producer(&repository, &mixed),
            &out.root.join("mixed"),
        );
        if keeps_case {
            assert_eq!(error.code, PackageErrorCode::DigestMismatch);
        } else {
            assert_eq!(error.code, PackageErrorCode::FetchFailed);
            assert!(error.message.contains("`notes.md`"), "{}", error.message);
        }

        // The folder as the commit holds it is fetched where both names can be kept.
        let whole = folder_digest_of(&[
            ("SKILL.md", STORYBOARD),
            ("notes.md", "first\n"),
            ("Notes.md", "second\n"),
        ]);
        let fetched = fetch(
            &video_producer(&repository, &whole).root,
            &out.root.join("whole"),
            &AgentPackageFetchOptions::default(),
        );
        if keeps_case {
            fetched.expect("three files");
            assert_eq!(
                names(&out.root.join("whole/skills/storyboard")),
                ["Notes.md", "SKILL.md", "notes.md"]
            );
        } else {
            assert_eq!(
                fetched.expect_err("refused").code,
                PackageErrorCode::FetchFailed
            );
            assert!(names(&out.root).is_empty());
        }
    }

    #[test]
    fn an_address_that_did_not_answer_is_not_asked_again() {
        let (parts, _) = storyboard_parts("fetch-asked-once");
        let repository = Repository::of(parts);
        let scratch = Package::new("fetch-asked-once-scratch");
        let moved = scratch.root.join("parts.git");
        let url = format!("file://{}", moved.display());
        let mut git = Git::new(scratch.root.join("git")).expect("a scratch folder");

        let first = git
            .repository(&url, &repository.commit, "$/skills/storyboard")
            .expect_err("no repository is at the address");
        // The repository appears at the address. The next entry does not ask for it.
        fs::rename(repository.work.root.join("shared.git"), &moved).expect("move the repository");
        let second = git
            .repository(&url, &repository.commit, "$/skills/captions")
            .expect_err("the address is not asked again");

        assert_eq!(first.code, PackageErrorCode::FetchFailed);
        assert_eq!(second.subject, "$/skills/captions");
        assert_eq!(second.message, first.message);
    }

    #[test]
    fn a_part_a_called_agent_from_git_also_uses_is_fetched_where_the_lock_records_it() {
        // The lead and its reviewer use one checklist. One entry pins it, at the first
        // path in sort order, which runs through the reviewer.
        const SHARED: &str = "$/agents/reviewer/skills/checklist";
        let (parts, storyboard) = storyboard_parts("fetch-shared-parts");
        parts.write("skills/checklist/SKILL.md", CHECKLIST);
        let checklist = parts.folder_digest("skills/checklist");
        let parts = Repository::of(parts);
        let team = Package::new("fetch-shared-team");
        team.write_json(
            "reviewer/agent.json",
            &agent(
                "reviewer",
                json!({"skills": {"checklist": parts.reference("skills/checklist")}}),
            ),
        );
        let document = team.file_digest("reviewer/agent.json");
        let team = Repository::of(team);
        let package = Package::new("fetch-shared-lead");
        package.write_json(
            "agent.json",
            &agent(
                "code-review-lead",
                json!({
                    "agents": {"reviewer": team.reference("reviewer")},
                    "skills": {
                        "checklist": parts.reference("skills/checklist"),
                        "storyboard": parts.reference("skills/storyboard")
                    }
                }),
            ),
        );
        let mut reviewer = team.entry("$/agents/reviewer", "agent", "reviewer", &document);
        reviewer["dependencies"] = json!([SHARED]);
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &["$/agents/reviewer", SHARED, "$/skills/storyboard"],
                ),
                reviewer,
                parts.entry(SHARED, "skill", "skills/checklist", &checklist),
                parts.entry(
                    "$/skills/storyboard",
                    "skill",
                    "skills/storyboard",
                    &storyboard,
                ),
            ],
        );
        let out = Package::new("fetch-shared-out");
        let out_dir = out.root.join("code-review");

        let report = fetch(
            &package.root,
            &out_dir,
            &AgentPackageFetchOptions::default(),
        )
        .expect("the reviewer and the shared checklist are fetched");

        let placed: Vec<(&str, PathBuf)> = report
            .fetched
            .iter()
            .map(|row| (row.address.as_str(), row.path.clone()))
            .collect();
        assert_eq!(
            placed,
            [
                (
                    "$/agents/reviewer",
                    out_dir.join("agents/reviewer/agent.json")
                ),
                (SHARED, out_dir.join("agents/reviewer/skills/checklist")),
                ("$/skills/storyboard", out_dir.join("skills/storyboard")),
            ]
        );
        let after = read_package(&out_dir, &ReadRequest::default()).expect("the result loads");
        assert!(
            after
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );
        let lead_checklist = after
            .part(PartField::Skills, "checklist")
            .expect("the lead's checklist");
        assert_eq!(lead_checklist.address, SHARED);
    }

    #[test]
    fn a_part_a_definition_adds_that_its_agent_from_git_also_uses_is_fetched_once() {
        const SHARED: &str = "$/agent/skills/storyboard";
        let (parts, storyboard) = storyboard_parts("fetch-shared-definition-parts");
        let parts = Repository::of(parts);
        let agents = Package::new("fetch-shared-definition-agents");
        agents.write_json(
            "video-producer/agent.json",
            &agent(
                "video-producer",
                json!({"skills": {"storyboard": parts.reference("skills/storyboard")}}),
            ),
        );
        let document = agents.file_digest("video-producer/agent.json");
        let agents = Repository::of(agents);
        let package = Package::new("fetch-shared-definition");
        // The definition names the skill its agent already uses, with the same entry.
        package.write_json(
            "agent-definition.json",
            &definition(
                "video-producer-short",
                json!({
                    "agent": agents.reference("video-producer"),
                    "skills": {"storyboard": parts.reference("skills/storyboard")}
                }),
            ),
        );
        let mut shared = agents.entry("$/agent", "agent", "video-producer", &document);
        shared["dependencies"] = json!([SHARED]);
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent-definition",
                    "agent-definition.json",
                    "./agent-definition.json",
                    &["$/agent", SHARED],
                ),
                shared,
                parts.entry(SHARED, "skill", "skills/storyboard", &storyboard),
            ],
        );
        let out = Package::new("fetch-shared-definition-out");
        let out_dir = out.root.join("video-producer-short");

        let report = fetch(
            &package.root,
            &out_dir,
            &AgentPackageFetchOptions::default(),
        )
        .expect("the agent and the skill are fetched");

        let placed: Vec<&str> = report
            .fetched
            .iter()
            .map(|row| row.address.as_str())
            .collect();
        assert_eq!(placed, ["$/agent", SHARED]);
        let after = read_package(&out_dir, &ReadRequest::default()).expect("the result loads");
        assert!(
            after
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );
        assert_eq!(
            names(&out_dir),
            ["agent", "agent-definition.json", "oadl.lock.json"]
        );
    }

    #[test]
    fn a_package_with_nothing_to_fetch_is_returned_as_it_is() {
        let package = small_package("fetch-nothing");
        let out = Package::new("fetch-nothing-out");
        let out_dir = out.root.join("release-notes");

        let report = fetch(
            &package.root,
            &out_dir,
            &AgentPackageFetchOptions::default(),
        )
        .expect("nothing to fetch");

        assert_eq!(report.package_dir, package.root);
        assert_eq!(report.lock.path, package.root.join("oadl.lock.json"));
        assert!(report.fetched.is_empty());
        assert!(report.not_fetched.is_empty());
        assert!(names(&out.root).is_empty());
    }

    #[test]
    fn a_package_a_load_would_refuse_is_refused_before_anything_is_fetched() {
        let (parts, digest) = storyboard_parts("fetch-refused-first");
        let mut repository = Repository::of(parts);
        // No repository is at this address. The fetch never asks it.
        repository.url = "file:///nowhere/media-team/parts.git".to_string();
        let package = video_producer(&repository, &digest);
        let out = Package::new("fetch-refused-first-out");
        let options = AgentPackageFetchOptions {
            expected_lock_sha256: Some("0".repeat(64)),
            ..AgentPackageFetchOptions::default()
        };

        let error = fetch(&package.root, &out.root.join("video-producer"), &options)
            .expect_err("the lock is not the one the caller holds");

        assert_eq!(error.code, PackageErrorCode::LockDigestMismatch);
        assert!(names(&out.root).is_empty());
    }

    #[test]
    fn a_load_never_fetches_and_never_writes() {
        use crate::agent_package::{AgentPackageOptions, load_agent_package};
        use crate::config::ResolveContext;

        let (parts, digest) = storyboard_parts("fetch-load");
        let repository = Repository::of(parts);
        let package = video_producer(&repository, &digest);

        let report = load_agent_package(
            &package.root,
            None,
            AgentPackageOptions::default(),
            ResolveContext::new(&package.root),
        )
        .expect("the package loads with its skill absent");

        let skill = report
            .digests
            .iter()
            .find(|check| check.address == "$/skills/storyboard")
            .expect("the skill's entry");
        assert_eq!(skill.result, Integrity::Absent);
        assert!(skill.path.is_none());
        assert_eq!(names(&package.root), ["agent.json", "oadl.lock.json"]);
    }

    #[test]
    fn a_load_report_leaves_the_password_out_of_an_address() {
        use crate::agent_package::{AgentPackageOptions, load_agent_package};
        use crate::config::ResolveContext;

        let (parts, digest) = storyboard_parts("fetch-load-password");
        let mut repository = Repository::of(parts);
        repository.url = "https://build:hunter2@git.example.com/media-team/parts.git".to_string();
        let package = video_producer(&repository, &digest);

        let report = load_agent_package(
            &package.root,
            None,
            AgentPackageOptions::default(),
            ResolveContext::new(&package.root),
        )
        .expect("the package loads with its skill absent");

        let skill = report
            .digests
            .iter()
            .find(|check| check.address == "$/skills/storyboard")
            .expect("the skill's entry");
        assert_eq!(
            skill.source_url.as_deref(),
            Some("https://build:***@git.example.com/media-team/parts.git")
        );
        let written = serde_json::to_string(&report).expect("the report serializes");
        assert!(!written.contains("hunter2"), "{written}");
    }

    /// A repository of agents whose `video-producer` folder is a whole package: the agent,
    /// its lock, and a skill pinned from the repository of shared parts. It is tagged `v2.1.0`.
    fn agents_repository(parts: &Repository, storyboard: &str) -> Repository {
        let agents = Package::new("fetch-agents");
        agents.write("README.md", "Agents of the media team.\n");
        agents.write_json(
            "video-producer/agent.json",
            &agent(
                "video-producer",
                json!({"skills": {"storyboard": parts.reference("skills/storyboard")}}),
            ),
        );
        agents.write_lock(
            "video-producer",
            vec![
                agents.file_entry(
                    "$",
                    "agent",
                    "video-producer/agent.json",
                    "./agent.json",
                    &["$/skills/storyboard"],
                ),
                parts.entry(
                    "$/skills/storyboard",
                    "skill",
                    "skills/storyboard",
                    storyboard,
                ),
            ],
        );
        let repository = Repository::of(agents);
        let bare = repository.work.root.join("shared.git");
        git(&bare, &["tag", "v2.1.0", &repository.commit]);
        repository
    }

    #[test]
    fn a_package_that_is_itself_in_git_is_fetched_and_completed_from_its_lock() {
        let (parts, digest) = storyboard_parts("fetch-package-parts");
        let parts = Repository::of(parts);
        let agents = agents_repository(&parts, &digest);
        let source = format!("git::{}//video-producer", agents.url);
        let out = Package::new("fetch-package-out");

        for (reference, name) in [("v2.1.0", "by-tag"), (agents.commit.as_str(), "by-commit")] {
            let out_dir = out.root.join(name);

            let report = fetch_from_git(
                &source,
                reference,
                &out_dir,
                &AgentPackageFetchOptions::default(),
            )
            .expect("the package is fetched and completed");

            assert_eq!(report.package_dir, out_dir);
            // The package itself is the first row, with the commit the tag named.
            let rows: Vec<(&str, Option<&str>, PathBuf)> = report
                .fetched
                .iter()
                .map(|row| {
                    (
                        row.address.as_str(),
                        row.commit.as_deref(),
                        row.path.clone(),
                    )
                })
                .collect();
            assert_eq!(
                rows,
                [
                    ("$", Some(agents.commit.as_str()), out_dir.clone()),
                    (
                        "$/skills/storyboard",
                        Some(parts.commit.as_str()),
                        out_dir.join("skills/storyboard")
                    ),
                ]
            );
            assert_eq!(report.fetched[0].entry_type, EntryType::Agent);
            assert_eq!(
                report.fetched[0].source_subpath.as_deref(),
                Some("video-producer")
            );
            // Only the folder of the package is written, not what sits beside it in the repository.
            assert_eq!(names(&out_dir), ["agent.json", "oadl.lock.json", "skills"]);
            let after = read_package(&out_dir, &ReadRequest::default()).expect("the result loads");
            assert!(
                after
                    .parts
                    .iter()
                    .all(|part| part.integrity == Integrity::Verified)
            );
            assert_eq!(report.lock.sha256, after.lock.sha256);
        }
        assert_eq!(names(&out.root), ["by-commit", "by-tag"]);
    }

    #[test]
    fn a_package_in_git_that_cannot_be_used_is_refused_and_nothing_is_left() {
        let (parts, digest) = storyboard_parts("fetch-package-refused-parts");
        let parts = Repository::of(parts);
        let agents = agents_repository(&parts, &digest);
        let source = format!("git::{}//video-producer", agents.url);
        let out = Package::new("fetch-package-refused-out");
        let out_dir = out.root.join("video-producer");
        let defaults = AgentPackageFetchOptions::default();
        let other_lock = AgentPackageFetchOptions {
            expected_lock_sha256: Some("0".repeat(64)),
            ..AgentPackageFetchOptions::default()
        };
        let no_folder = format!("git::{}//storyboard-artist", agents.url);
        let a_file = format!("git::{}//README.md", agents.url);
        let not_a_package = format!("git::{}//skills/storyboard", parts.url);

        for (source, reference, options, code) in [
            // The lock is not the one the caller holds.
            (
                source.as_str(),
                "v2.1.0",
                &other_lock,
                PackageErrorCode::LockDigestMismatch,
            ),
            (
                source.as_str(),
                "v9.9.9",
                &defaults,
                PackageErrorCode::FetchFailed,
            ),
            (
                source.as_str(),
                "--upload-pack=touch",
                &defaults,
                PackageErrorCode::FetchFailed,
            ),
            (
                source.as_str(),
                "main:refs/heads/moved",
                &defaults,
                PackageErrorCode::FetchFailed,
            ),
            (
                no_folder.as_str(),
                "v2.1.0",
                &defaults,
                PackageErrorCode::FetchFailed,
            ),
            (
                a_file.as_str(),
                "v2.1.0",
                &defaults,
                PackageErrorCode::FetchFailed,
            ),
            // A folder with no lock is not a package.
            (
                not_a_package.as_str(),
                parts.commit.as_str(),
                &defaults,
                PackageErrorCode::LockMissing,
            ),
            (
                "oci::registry.example.com/agents/video-producer",
                "v2.1.0",
                &defaults,
                PackageErrorCode::OptionsInvalid,
            ),
            (
                "./video-producer",
                "v2.1.0",
                &defaults,
                PackageErrorCode::OptionsInvalid,
            ),
        ] {
            let error = fetch_from_git(source, reference, &out_dir, options).expect_err("refused");
            assert_eq!(
                error.code, code,
                "{source} at {reference}: {}",
                error.message
            );
        }
        assert!(names(&out.root).is_empty());
    }

    const PLUGIN: &[(&str, &str, bool)] = &[
        (
            ".claude-plugin/plugin.json",
            "{\"name\": \"remotion\"}\n",
            false,
        ),
        ("scripts/render.sh", "#!/bin/sh\nexit 0\n", true),
        (
            "skills/remotion/SKILL.md",
            "---\nname: remotion\n---\nRender with Remotion.\n",
            false,
        ),
    ];

    /// A video producer whose `remotion` plugin is pinned as the OCI artifact of this digest.
    fn video_producer_with_plugin(digest: &str, optional: bool) -> Package {
        let package = Package::new("fetch-oci-package");
        package.write_json(
            "agent.json",
            &agent(
                "video-producer",
                json!({"plugins": {"remotion": {
                    "source": "oci::registry.example.com/plugins/remotion",
                    "version": "1.2.0",
                    "optional": optional
                }}}),
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
                    &["$/plugins/remotion"],
                ),
                json!({
                    "address": "$/plugins/remotion", "type": "plugin",
                    "source": {"type": "oci", "url": "registry.example.com/plugins/remotion"},
                    "version": "1.2.0", "constraint": "1.2.0", "digest": digest,
                    "dependencies": []
                }),
            ],
        );
        package
    }

    fn with_layout(layout: &Package) -> AgentPackageFetchOptions {
        AgentPackageFetchOptions {
            oci_layout: Some(layout.root.clone()),
            ..AgentPackageFetchOptions::default()
        }
    }

    #[test]
    fn a_plugin_pinned_as_an_oci_artifact_is_unpacked_and_a_load_verifies_it() {
        let layout = Package::new("fetch-oci-layout");
        let digest = write_artifact(&layout.root, &tar(PLUGIN), TAR_LAYER);
        let package = video_producer_with_plugin(&digest, false);
        let before = package.read().expect("the package reads");
        assert_eq!(
            integrity(&before.parts, "$/plugins/remotion"),
            Integrity::NotVerifiable
        );
        let out = Package::new("fetch-oci-out");
        let out_dir = out.root.join("video-producer");

        // With no layout folder the artifact is left, and the reason is given.
        let left = fetch(
            &package.root,
            &out_dir,
            &AgentPackageFetchOptions::default(),
        )
        .expect("nothing to fetch");
        assert_eq!(left.package_dir, package.root);
        assert_eq!(left.not_fetched.len(), 1);
        assert!(
            left.not_fetched[0]
                .reason
                .contains("no `oci_layout` folder was given")
        );
        assert!(names(&out.root).is_empty());

        let report = fetch(&package.root, &out_dir, &with_layout(&layout)).expect("unpacked");

        assert!(report.not_fetched.is_empty());
        assert_eq!(
            report.fetched,
            [FetchedContent {
                address: "$/plugins/remotion".to_string(),
                entry_type: EntryType::Plugin,
                source_url: "registry.example.com/plugins/remotion".to_string(),
                source_subpath: None,
                commit: None,
                digest: digest.clone(),
                path: out_dir.join("plugins/remotion"),
            }]
        );
        // The package keeps the blobs the files came from, as a layout folder of its own.
        let store = out_dir.join(".oci");
        assert_eq!(names(&store), ["blobs", "index.json", "oci-layout"]);
        assert_eq!(names(&store.join("blobs/sha256")).len(), 3);
        let index: Value =
            serde_json::from_slice(&fs::read(store.join("index.json")).expect("the index"))
                .expect("JSON");
        assert_eq!(index["manifests"][0]["digest"], digest.as_str());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let script = out_dir.join("plugins/remotion/scripts/render.sh");
            assert_ne!(
                fs::metadata(script).expect("placed").permissions().mode() & 0o111,
                0
            );
        }

        // A load walks from the lock's digest to the files again.
        let request = ReadRequest::default();
        let after = read_package(&out_dir, &request).expect("the result loads");
        let plugin = after
            .part(PartField::Plugins, "remotion")
            .expect("the plugin");
        assert_eq!(plugin.integrity, Integrity::Verified);
        assert_eq!(
            plugin.path.as_deref(),
            Some(out_dir.join("plugins/remotion").as_path())
        );
        assert_eq!(
            after.listings["$/plugins/remotion"].files.len(),
            PLUGIN.len()
        );

        // A file changed, added or removed after unpacking is refused.
        let script = out_dir.join("plugins/remotion/scripts/render.sh");
        let extra = out_dir.join("plugins/remotion/scripts/upload.sh");
        fs::write(&script, "#!/bin/sh\nexit 1\n").expect("change one byte");
        let changed = read_package(&out_dir, &request).expect_err("refused");
        assert_eq!(changed.code, PackageErrorCode::DigestMismatch);
        assert_eq!(changed.subject, "$/plugins/remotion");
        fs::write(&script, "#!/bin/sh\nexit 0\n").expect("restore");
        fs::write(&extra, "#!/bin/sh\n").expect("add a file");
        let added = read_package(&out_dir, &request).expect_err("refused");
        assert_eq!(added.code, PackageErrorCode::DigestMismatch);
        fs::remove_file(&extra).expect("remove it again");
        read_package(&out_dir, &request).expect("the files are the artifact's again");
        // With the kept blobs gone the content cannot be checked, and is not used.
        fs::remove_dir_all(&store).expect("remove the store");
        let unkept = read_package(&out_dir, &request).expect("the package still reads");
        assert_eq!(
            integrity(&unkept.parts, "$/plugins/remotion"),
            Integrity::NotVerifiable
        );
    }

    #[test]
    fn a_ds_store_file_is_never_one_of_the_files_of_an_unpacked_artifact() {
        // An archive made on a Mac can hold the file, and Finder can add one after unpacking.
        let layout = Package::new("fetch-oci-finder-layout");
        let mut layer = PLUGIN.to_vec();
        layer.push(("skills/.DS_Store", "Bud1", false));
        let digest = write_artifact(&layout.root, &tar(&layer), TAR_LAYER);
        let out = Package::new("fetch-oci-finder-out");
        let out_dir = out.root.join("video-producer");

        fetch(
            &video_producer_with_plugin(&digest, false).root,
            &out_dir,
            &with_layout(&layout),
        )
        .expect("unpacked");

        let request = ReadRequest::default();
        let verified = || {
            let after = read_package(&out_dir, &request).expect("the result loads");
            assert_eq!(
                integrity(&after.parts, "$/plugins/remotion"),
                Integrity::Verified
            );
            assert_eq!(
                after.listings["$/plugins/remotion"].files.len(),
                PLUGIN.len()
            );
        };
        verified();
        // The file of the layer was unpacked with the others. With it gone, or with one more
        // at the root of the plugin, the files are still the artifact's.
        let plugin = out_dir.join("plugins/remotion");
        fs::remove_file(plugin.join("skills/.DS_Store")).expect("the file was unpacked");
        verified();
        fs::write(plugin.join(".DS_Store"), "Bud1").expect("add one");
        verified();
    }

    #[test]
    fn content_unpacked_from_an_artifact_stays_verified_in_the_next_fetch() {
        let layout = Package::new("fetch-oci-then-git-layout");
        let plugin = write_artifact(&layout.root, &tar(PLUGIN), TAR_LAYER);
        let out = Package::new("fetch-oci-then-git-out");
        let first = out.root.join("with-plugin");
        fetch(
            &video_producer_with_plugin(&plugin, false).root,
            &first,
            &with_layout(&layout),
        )
        .expect("the plugin is unpacked");
        // The package then gets a skill from git: its document and its lock are written again.
        let (parts, storyboard) = storyboard_parts("fetch-oci-then-git-parts");
        let repository = Repository::of(parts);
        let package = Package {
            root: first.clone(),
        };
        let mut lock: Value =
            serde_json::from_slice(&fs::read(first.join("oadl.lock.json")).expect("the lock"))
                .expect("JSON");
        let mut document: Value =
            serde_json::from_slice(&fs::read(first.join("agent.json")).expect("the agent"))
                .expect("JSON");
        document["skills"] = json!({"storyboard": repository.reference("skills/storyboard")});
        package.write_json("agent.json", &document);
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &["$/plugins/remotion", "$/skills/storyboard"],
                ),
                lock["entries"][1].take(),
                repository.entry(
                    "$/skills/storyboard",
                    "skill",
                    "skills/storyboard",
                    &storyboard,
                ),
            ],
        );
        let out_dir = out.root.join("with-skill");

        // No layout folder is given: the artifact the package keeps goes along with its files.
        let report = fetch(&first, &out_dir, &AgentPackageFetchOptions::default())
            .expect("the skill is fetched");

        let placed: Vec<&str> = report
            .fetched
            .iter()
            .map(|row| row.address.as_str())
            .collect();
        assert_eq!(placed, ["$/skills/storyboard"]);
        assert!(report.not_fetched.is_empty());
        assert_eq!(
            names(&out_dir),
            [".oci", "agent.json", "oadl.lock.json", "plugins", "skills"]
        );
        assert_eq!(
            names(&out_dir.join(".oci")),
            ["blobs", "index.json", "oci-layout"]
        );
        let after = read_package(&out_dir, &ReadRequest::default()).expect("the result loads");
        assert!(
            after
                .parts
                .iter()
                .all(|part| part.integrity == Integrity::Verified)
        );
    }

    #[test]
    fn an_artifact_that_cannot_be_used_is_refused_and_nothing_is_left() {
        let out = Package::new("fetch-oci-refused-out");
        let out_dir = out.root.join("video-producer");

        // The layer kept in the layout folder is not the one the manifest pins.
        let layout = Package::new("fetch-oci-wrong-layer");
        let layer = tar(PLUGIN);
        let digest = write_artifact(&layout.root, &layer, TAR_LAYER);
        let kept = layout
            .root
            .join("blobs/sha256")
            .join(bytes_digest(&layer).trim_start_matches("sha256:"));
        fs::write(kept, tar(&[("plugin.json", "{}", false)])).expect("other bytes under its name");
        let package = video_producer_with_plugin(&digest, false);
        let error = fetch(&package.root, &out_dir, &with_layout(&layout)).expect_err("refused");
        assert_eq!(error.code, PackageErrorCode::DigestMismatch);
        assert_eq!(error.subject, "$/plugins/remotion");

        // A compressed layer is not read.
        let layout = Package::new("fetch-oci-gzip");
        let gzip = format!("{TAR_LAYER}+gzip");
        let digest = write_artifact(&layout.root, b"\x1f\x8b compressed bytes", &gzip);
        let package = video_producer_with_plugin(&digest, false);
        let error = fetch(&package.root, &out_dir, &with_layout(&layout)).expect_err("refused");
        assert_eq!(error.code, PackageErrorCode::FileUnreadable);
        assert!(
            error.message.ends_with("a compressed layer is not read"),
            "{}",
            error.message
        );

        // The layout folder does not hold the artifact the lock pins.
        let layout = Package::new("fetch-oci-empty");
        let package = video_producer_with_plugin(MADE_UP, false);
        let error = fetch(&package.root, &out_dir, &with_layout(&layout)).expect_err("refused");
        assert_eq!(error.code, PackageErrorCode::FetchFailed);
        assert!(
            error.message.ends_with(&format!("holds no blob {MADE_UP}")),
            "{}",
            error.message
        );
        assert!(names(&out.root).is_empty());

        // The same for an optional plugin is a row of the report. Nothing was placed, so the
        // package is returned as it was given and nothing is written.
        let package = video_producer_with_plugin(MADE_UP, true);
        let report = fetch(&package.root, &out_dir, &with_layout(&layout)).expect("not an error");
        assert_eq!(report.package_dir, package.root);
        assert!(report.fetched.is_empty());
        assert_eq!(report.not_fetched.len(), 1);
        assert!(
            report.not_fetched[0]
                .reason
                .contains("package_fetch_failed")
        );
        assert!(names(&out.root).is_empty());
    }

    #[test]
    fn a_password_in_an_address_is_never_shown() {
        for (url, shown) in [
            (
                "https://build:hunter2@git.example.com/media-team/agents.git",
                "https://build:***@git.example.com/media-team/agents.git",
            ),
            // A token is often written in place of a user name.
            (
                "https://glpat-secret@git.example.com/media-team/agents.git",
                "https://***@git.example.com/media-team/agents.git",
            ),
            (
                "ssh://git@git.example.com/media-team/agents.git",
                "ssh://git@git.example.com/media-team/agents.git",
            ),
            (
                "git@git.example.com:media-team/agents.git",
                "git@git.example.com:media-team/agents.git",
            ),
            (
                "https://git.example.com/media-team/agents.git",
                "https://git.example.com/media-team/agents.git",
            ),
        ] {
            assert_eq!(without_password(url), shown);
        }
        assert_eq!(
            hide_password(
                "fatal: unable to access 'https://build:hunter2@git.example.com/': timeout",
                "https://build:hunter2@git.example.com/media-team/agents.git",
            ),
            "fatal: unable to access 'https://build:***@git.example.com/': timeout"
        );
    }
}
