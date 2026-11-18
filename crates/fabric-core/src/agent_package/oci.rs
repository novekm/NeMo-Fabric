// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content pinned as an OCI artifact, read from a folder in the OCI image layout form.
//!
//! No registry is asked. The lock entry of an OCI source holds one digest, the
//! digest of the artifact's manifest. This file leads from that digest to
//! files: the manifest, its one layer, and the files in the layer's tar
//! archive. Each step is checked against the digest the step before gives.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::digest::{
    MAX_FOLDER_BYTES, MAX_FOLDER_DEPTH, MAX_FOLDER_FILES, UNLISTED_FILE_NAME, bytes_digest,
    digest_text, file_digest,
};
use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::json::{MAX_DOCUMENT_BYTES, read_regular_file};
use super::paths::is_safe_relative;

/// The folder at the root of a package that keeps the blobs its OCI content was unpacked
/// from. It is itself a folder in the OCI image layout form.
pub(crate) const STORE_DIR: &str = ".oci";

const MANIFEST_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
const INDEX_TYPE: &str = "application/vnd.oci.image.index.v1+json";
/// The one layer form that is read: a tar archive that is not compressed.
const LAYER_TYPE: &str = "application/vnd.oci.image.layer.v1.tar";
/// The size of every block of a tar archive.
const BLOCK: usize = 512;
/// The longest path a tar archive may give in a record of its own.
const MAX_PATH_RECORD: u64 = 64 * 1024;
/// The most bytes one layer may hold: the bytes of a folder, and four blocks around each
/// of its files for what a tar archive writes there.
const MAX_LAYER_BYTES: u64 = MAX_FOLDER_BYTES + (MAX_FOLDER_FILES * 4 * BLOCK) as u64;

fn unread(at: &str, message: impl Into<String>) -> PackageError {
    PackageError::new(PackageErrorCode::FileUnreadable, at, message)
}

/// Where a layout folder keeps the blob of a `sha256:` digest. None for any other text.
fn blob_path(layout: &Path, digest: &str) -> Option<PathBuf> {
    let hex = digest.strip_prefix("sha256:")?;
    let is_hex = hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    is_hex.then(|| layout.join("blobs").join("sha256").join(hex))
}

/// One artifact in a layout folder: an image manifest with one tar layer.
#[derive(Debug)]
pub(crate) struct Artifact {
    layout: PathBuf,
    /// The manifest, then the config when the layout holds it, then the layer.
    blobs: Vec<String>,
    layer_digest: String,
    layer_size: u64,
}

/// The artifact a lock digest names, in a layout folder.
///
/// None when the folder has no blob of that digest. The manifest is the blob
/// the digest names, and its bytes must have that digest. An image index, a
/// manifest with more or fewer layers than one, and a layer that is not an
/// uncompressed tar archive are refused by name.
pub(crate) fn find_artifact(
    layout: &Path,
    digest: &str,
    at: &str,
) -> PackageResult<Option<Artifact>> {
    let Some(path) = blob_path(layout, digest) else {
        return Ok(None);
    };
    if fs::symlink_metadata(&path).is_err() {
        return Ok(None);
    }
    let bytes = read_regular_file(&path, at, MAX_DOCUMENT_BYTES)?;
    let found = bytes_digest(&bytes);
    if found != digest {
        return Err(PackageError::new(
            PackageErrorCode::DigestMismatch,
            at,
            format!("the lock pins {digest}, and the manifest kept under that name has {found}"),
        ));
    }
    let manifest: Value = serde_json::from_slice(&bytes)
        .map_err(|error| unread(at, format!("the manifest {digest} is not JSON: {error}")))?;
    let media_type = manifest.get("mediaType").and_then(Value::as_str);
    if media_type == Some(INDEX_TYPE) || manifest.get("manifests").is_some() {
        return Err(unread(
            at,
            format!("{digest} is an image index, and one image manifest is read"),
        ));
    }
    if let Some(other) = media_type.filter(|media_type| *media_type != MANIFEST_TYPE) {
        return Err(unread(
            at,
            format!("the manifest has media type `{other}`, and an OCI image manifest is read"),
        ));
    }
    let layers = manifest
        .get("layers")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let [layer] = layers else {
        return Err(unread(
            at,
            format!(
                "the artifact has {} layers, and an artifact with one layer is read",
                layers.len()
            ),
        ));
    };
    let layer_type = layer
        .get("mediaType")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if layer_type != LAYER_TYPE {
        let why = if layer_type.starts_with(LAYER_TYPE) {
            "a compressed layer is not read"
        } else {
            "a layer that is a tar archive is read"
        };
        return Err(unread(
            at,
            format!("the layer has media type `{layer_type}`: {why}"),
        ));
    }
    let layer_digest = layer.get("digest").and_then(Value::as_str);
    let layer_size = layer.get("size").and_then(Value::as_u64);
    let (Some(layer_digest), Some(layer_size)) = (layer_digest, layer_size) else {
        return Err(unread(at, "the layer has no digest or no size"));
    };
    if blob_path(layout, layer_digest).is_none() {
        return Err(unread(
            at,
            format!("the layer's digest `{layer_digest}` is not a sha256 digest"),
        ));
    }
    if layer_size > MAX_LAYER_BYTES {
        return Err(PackageError::new(
            PackageErrorCode::TooLarge,
            at,
            format!("the layer is larger than {MAX_LAYER_BYTES} bytes"),
        ));
    }
    let mut blobs = vec![digest.to_string()];
    let config = manifest.pointer("/config/digest").and_then(Value::as_str);
    if let Some(config) = config.filter(|config| {
        blob_path(layout, config).is_some_and(|path| fs::symlink_metadata(path).is_ok())
    }) {
        blobs.push(config.to_string());
    }
    blobs.push(layer_digest.to_string());
    Ok(Some(Artifact {
        layout: layout.to_path_buf(),
        blobs,
        layer_digest: layer_digest.to_string(),
        layer_size,
    }))
}

/// Reads a blob while it works out the blob's digest and its length.
struct Hashed<R> {
    inner: R,
    hasher: Sha256,
    length: u64,
}

impl<R: Read> Read for Hashed<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..read]);
        self.length += read as u64;
        Ok(read)
    }
}

/// The text of a header field: its bytes up to the first NUL.
fn field(bytes: &[u8]) -> &[u8] {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    &bytes[..end]
}

/// A number a header writes in octal. A number in the binary form is not read.
fn octal(bytes: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(field(bytes)).ok()?.trim();
    if text.is_empty() {
        return Some(0);
    }
    u64::from_str_radix(text, 8).ok()
}

/// The sum a header records over its own bytes, with the sum's field read as spaces.
fn checksum_holds(header: &[u8; BLOCK]) -> bool {
    let sum: u64 = header
        .iter()
        .enumerate()
        .map(|(index, byte)| match index {
            148..156 => u64::from(b' '),
            _ => u64::from(*byte),
        })
        .sum();
    octal(&header[148..156]) == Some(sum)
}

/// The path a header gives: its name, after its prefix when the archive is in the POSIX form.
fn header_path(header: &[u8; BLOCK]) -> Option<String> {
    let name = std::str::from_utf8(field(&header[..100])).ok()?;
    let prefix = if &header[257..263] == b"ustar\0" {
        std::str::from_utf8(field(&header[345..500])).ok()?
    } else {
        ""
    };
    Some(if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    })
}

/// The `path` a pax header gives for the entry after it. Its records are
/// `<length> <key>=<value>` and a newline, where the length counts the whole record.
fn pax_path(records: &[u8]) -> Option<String> {
    let mut rest = records;
    while !rest.is_empty() {
        let space = rest.iter().position(|byte| *byte == b' ')?;
        let length: usize = std::str::from_utf8(&rest[..space]).ok()?.parse().ok()?;
        if length <= space || length > rest.len() {
            return None;
        }
        let record = &rest[space + 1..length];
        if let Some(value) = record.strip_prefix(b"path=") {
            let value = value.strip_suffix(b"\n").unwrap_or(value);
            return std::str::from_utf8(value).ok().map(str::to_string);
        }
        rest = &rest[length..];
    }
    None
}

impl Artifact {
    /// Walk the regular files of the layer, in the order the archive holds them.
    ///
    /// `visit` gets the path, whether the file is executable, and the file's
    /// bytes. A link, a device, a path that is not a plain relative path and a
    /// path written twice are refused, and the loader's folder bounds hold.
    /// The layer's digest and size are checked when the walk ends, so a caller
    /// keeps nothing of a walk that failed.
    pub(crate) fn files(
        &self,
        at: &str,
        mut visit: impl FnMut(&str, bool, &mut dyn Read) -> PackageResult<()>,
    ) -> PackageResult<()> {
        let layer = &self.layer_digest;
        let cut =
            |error: io::Error| unread(at, format!("the layer {layer} cannot be read: {error}"));
        let too_large =
            |message: String| PackageError::new(PackageErrorCode::TooLarge, at, message);
        let path = blob_path(&self.layout, layer).unwrap_or_default();
        let kept = fs::symlink_metadata(&path)
            .ok()
            .filter(|metadata| metadata.is_file());
        let Some(kept) = kept else {
            return Err(unread(
                at,
                format!("the layer {layer} is not a file in the layout folder"),
            ));
        };
        // A blob of another length is not the layer, and is not read to find that out.
        if kept.len() != self.layer_size {
            return Err(PackageError::new(
                PackageErrorCode::DigestMismatch,
                at,
                format!(
                    "the manifest pins the layer {layer} with {} bytes, and the blob kept under that name has {} bytes",
                    self.layer_size,
                    kept.len()
                ),
            ));
        }
        let mut archive = Hashed {
            inner: io::BufReader::new(fs::File::open(&path).map_err(cut)?),
            hasher: Sha256::new(),
            length: 0,
        };
        let mut seen = BTreeSet::new();
        let mut bytes = 0u64;
        // A long path comes in a record of its own, before the entry it names.
        let mut named: Option<String> = None;
        let mut header = [0u8; BLOCK];
        loop {
            match archive.read_exact(&mut header) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(cut(error)),
            }
            // A block of zeros ends the archive.
            if header.iter().all(|byte| *byte == 0) {
                break;
            }
            let size = octal(&header[124..136]).filter(|_| checksum_holds(&header));
            let (Some(size), Some(written)) = (size, header_path(&header)) else {
                return Err(unread(
                    at,
                    format!("the layer {layer} is not a tar archive this loader reads"),
                ));
            };
            let padding = (BLOCK as u64 - size % BLOCK as u64) % BLOCK as u64;
            let kind = header[156];
            match kind {
                b'x' | b'L' => {
                    if size > MAX_PATH_RECORD {
                        return Err(too_large(format!(
                            "a path record in the layer is larger than {MAX_PATH_RECORD} bytes"
                        )));
                    }
                    let mut record = vec![0u8; size as usize];
                    archive.read_exact(&mut record).map_err(cut)?;
                    named = if kind == b'x' {
                        pax_path(&record).or(named)
                    } else {
                        std::str::from_utf8(field(&record)).ok().map(str::to_string)
                    };
                }
                // Global settings of the archive and folders: nothing in them is a file.
                b'g' | b'5' => {
                    // A path record names the entry right after it, and that is this folder.
                    if kind == b'5' {
                        named = None;
                    }
                    io::copy(&mut (&mut archive).take(size), &mut io::sink()).map_err(cut)?;
                }
                b'0' | 0 => {
                    let path = named.take().unwrap_or(written);
                    let path = path.strip_prefix("./").unwrap_or(&path);
                    if !is_safe_relative(path) {
                        return Err(PackageError::new(
                            PackageErrorCode::PathUnsafe,
                            at,
                            format!("`{path}` in the layer is not a path a package can hold"),
                        ));
                    }
                    if !seen.insert(path.to_string()) {
                        return Err(unread(at, format!("the layer holds `{path}` twice")));
                    }
                    if seen.len() > MAX_FOLDER_FILES {
                        return Err(too_large(format!(
                            "the layer holds more than {MAX_FOLDER_FILES} files"
                        )));
                    }
                    if path.split('/').count() > MAX_FOLDER_DEPTH {
                        return Err(too_large(format!(
                            "the layer nests deeper than {MAX_FOLDER_DEPTH} levels"
                        )));
                    }
                    bytes = bytes.saturating_add(size);
                    if bytes > MAX_FOLDER_BYTES {
                        return Err(too_large(format!(
                            "the layer holds more than {MAX_FOLDER_BYTES} bytes"
                        )));
                    }
                    let executable = octal(&header[100..108]).is_some_and(|mode| mode & 0o111 != 0);
                    let mut content = (&mut archive).take(size);
                    visit(path, executable, &mut content)?;
                    io::copy(&mut content, &mut io::sink()).map_err(cut)?;
                    if content.limit() != 0 {
                        return Err(unread(
                            at,
                            format!("the layer {layer} ends inside `{path}`"),
                        ));
                    }
                }
                b'1' | b'2' => {
                    return Err(PackageError::new(
                        PackageErrorCode::SymlinkRefused,
                        at,
                        format!("`{written}` in the layer is a link"),
                    ));
                }
                _ => {
                    return Err(unread(
                        at,
                        format!("`{written}` in the layer is not a regular file"),
                    ));
                }
            }
            io::copy(&mut (&mut archive).take(padding), &mut io::sink()).map_err(cut)?;
        }
        // What follows the last entry counts toward the digest of the layer.
        io::copy(&mut archive, &mut io::sink()).map_err(cut)?;
        let found = digest_text(archive.hasher.finalize());
        if found != *layer || archive.length != self.layer_size {
            return Err(PackageError::new(
                PackageErrorCode::DigestMismatch,
                at,
                format!(
                    "the manifest pins the layer {layer} with {} bytes, and the blob kept under that name has {found} with {} bytes",
                    self.layer_size, archive.length
                ),
            ));
        }
        Ok(())
    }

    /// The files of the layer, each with the digest of its bytes.
    ///
    /// A file named [`UNLISTED_FILE_NAME`] is left out, as a folder's listing
    /// leaves it out, so the files on disk and the files of the layer are
    /// compared by one rule.
    pub(crate) fn listing(&self, at: &str) -> PackageResult<BTreeMap<String, String>> {
        let mut files = BTreeMap::new();
        self.files(at, |path, _, content| {
            if path.rsplit('/').next() == Some(UNLISTED_FILE_NAME) {
                return Ok(());
            }
            let mut hasher = Sha256::new();
            io::copy(content, &mut hasher).map_err(|error| {
                unread(at, format!("`{path}` in the layer cannot be read: {error}"))
            })?;
            files.insert(path.to_string(), digest_text(hasher.finalize()));
            Ok(())
        })?;
        Ok(files)
    }

    /// Write the files of the layer below `folder`, which holds nothing yet.
    pub(crate) fn unpack(&self, folder: &Path, at: &str) -> PackageResult<()> {
        self.files(at, |path, executable, content| {
            let unwritten =
                |error: io::Error| unread(at, format!("`{path}` cannot be written: {error}"));
            let target = folder.join(path);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(unwritten)?;
            }
            // A file is never written over, so no path of the layer lands on another.
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&target)
                .map_err(unwritten)?;
            io::copy(content, &mut file).map_err(unwritten)?;
            #[cfg(unix)]
            if executable {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&target, fs::Permissions::from_mode(0o755))
                    .map_err(unwritten)?;
            }
            #[cfg(not(unix))]
            let _ = executable;
            Ok(())
        })
    }

    /// Copy the blobs of the artifact into the store of a package.
    ///
    /// Each blob is copied only when its bytes have its digest. The manifest
    /// goes last: a load looks for the manifest first, so a store never names
    /// an artifact it does not hold whole.
    pub(crate) fn keep(&self, store: &Path, at: &str) -> PackageResult<()> {
        for digest in self.blobs.iter().rev() {
            let (Some(from), Some(to)) =
                (blob_path(&self.layout, digest), blob_path(store, digest))
            else {
                continue;
            };
            let unkept =
                |error: io::Error| unread(at, format!("the blob {digest} cannot be kept: {error}"));
            if file_digest(&from, at)? != *digest {
                return Err(PackageError::new(
                    PackageErrorCode::DigestMismatch,
                    at,
                    format!("the blob kept as {digest} in the layout folder has other bytes"),
                ));
            }
            if let Some(parent) = to.parent() {
                fs::create_dir_all(parent).map_err(unkept)?;
            }
            fs::copy(&from, &to).map_err(unkept)?;
        }
        Ok(())
    }
}

/// Write what makes the store of a package a folder in the OCI image layout
/// form: the marker file, and an index of the manifests the store holds.
pub(crate) fn write_layout(store: &Path, manifests: &BTreeSet<String>) -> io::Result<()> {
    let descriptors: Vec<Value> = manifests
        .iter()
        .filter_map(|digest| {
            let size = fs::metadata(blob_path(store, digest)?).ok()?.len();
            Some(json!({"mediaType": MANIFEST_TYPE, "digest": digest, "size": size}))
        })
        .collect();
    if descriptors.is_empty() {
        return Ok(());
    }
    fs::write(
        store.join("oci-layout"),
        "{\"imageLayoutVersion\":\"1.0.0\"}\n",
    )?;
    let index = json!({"schemaVersion": 2, "mediaType": INDEX_TYPE, "manifests": descriptors});
    fs::write(store.join("index.json"), format!("{index}\n"))
}

/// Writes archives and layout folders for tests, byte by byte.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// One header block of a tar archive in the POSIX form.
    pub(crate) fn header(path: &str, size: usize, mode: u32, kind: u8) -> [u8; BLOCK] {
        let mut header = [0u8; BLOCK];
        header[..path.len()].copy_from_slice(path.as_bytes());
        header[100..107].copy_from_slice(format!("{mode:07o}").as_bytes());
        header[124..135].copy_from_slice(format!("{size:011o}").as_bytes());
        header[156] = kind;
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        header[148..156].fill(b' ');
        let sum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
        header[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
        header
    }

    /// One entry of an archive: its header, its bytes, and zeros up to the next block.
    pub(crate) fn entry(path: &str, content: &[u8], mode: u32, kind: u8) -> Vec<u8> {
        let mut bytes = header(path, content.len(), mode, kind).to_vec();
        bytes.extend_from_slice(content);
        bytes.resize(bytes.len().next_multiple_of(BLOCK), 0);
        bytes
    }

    /// A tar archive of regular files, each with its path, its text and whether it is executable.
    pub(crate) fn tar(files: &[(&str, &str, bool)]) -> Vec<u8> {
        let mut archive = Vec::new();
        for (path, content, executable) in files {
            let mode = if *executable { 0o755 } else { 0o644 };
            archive.extend(entry(path, content.as_bytes(), mode, b'0'));
        }
        archive.resize(archive.len() + 2 * BLOCK, 0);
        archive
    }

    pub(crate) fn write_blob(layout: &Path, bytes: &[u8]) -> String {
        let digest = bytes_digest(bytes);
        let path = blob_path(layout, &digest).expect("a sha256 digest");
        fs::create_dir_all(path.parent().expect("parent")).expect("create the blob folder");
        fs::write(path, bytes).expect("write the blob");
        digest
    }

    /// Write one manifest into a layout folder and return its digest: the digest a lock pins.
    pub(crate) fn write_manifest(layout: &Path, manifest: &Value) -> String {
        write_blob(layout, manifest.to_string().as_bytes())
    }

    /// Write one artifact into a layout folder: the layer, the empty config and the
    /// manifest. Returns the manifest's digest.
    pub(crate) fn write_artifact(layout: &Path, layer: &[u8], layer_type: &str) -> String {
        let config = write_blob(layout, b"{}");
        let layer_digest = write_blob(layout, layer);
        write_manifest(
            layout,
            &json!({
                "schemaVersion": 2,
                "mediaType": MANIFEST_TYPE,
                "config": {"mediaType": "application/vnd.oci.empty.v1+json", "digest": config, "size": 2},
                "layers": [{"mediaType": layer_type, "digest": layer_digest, "size": layer.len()}]
            }),
        )
    }

    pub(crate) const TAR_LAYER: &str = LAYER_TYPE;
}

#[cfg(test)]
mod tests {
    use super::super::reader::testing::Package;
    use super::testing::*;
    use super::*;

    const AT: &str = "$/plugins/remotion";

    fn artifact(layout: &Package, layer: &[u8]) -> Artifact {
        let digest = write_artifact(&layout.root, layer, TAR_LAYER);
        find_artifact(&layout.root, &digest, AT)
            .expect("the artifact is read")
            .expect("the layout holds it")
    }

    /// The paths of the layer, each with whether it is executable and its text.
    fn files(artifact: &Artifact) -> PackageResult<Vec<(String, bool, String)>> {
        let mut found = Vec::new();
        artifact.files(AT, |path, executable, content| {
            let mut text = String::new();
            content.read_to_string(&mut text).expect("text");
            found.push((path.to_string(), executable, text));
            Ok(())
        })?;
        Ok(found)
    }

    #[test]
    fn the_regular_files_of_a_layer_are_read_in_the_order_it_holds_them() {
        let layout = Package::new("oci-files");
        let folder = format!("{}/", "templates/lower-thirds".repeat(6));
        let long = format!("{folder}caption-styles.json");
        let pax = |path: &str| {
            let record = format!("path={path}\n");
            // The length a pax record starts with counts its own digits.
            let record = format!("{} {record}", record.len() + 4);
            assert_eq!(
                record.len(),
                record.len().to_string().len() + record.len() - 3
            );
            record
        };
        let mut layer = Vec::new();
        layer.extend(entry("./", b"", 0o755, b'5'));
        layer.extend(entry("./.claude-plugin/plugin.json", b"{}\n", 0o644, b'0'));
        layer.extend(entry("scripts/", b"", 0o755, b'5'));
        layer.extend(entry("scripts/render.sh", b"#!/bin/sh\n", 0o755, b'0'));
        layer.extend(entry(
            "PaxHeaders/caption-styles.json",
            pax(&long).as_bytes(),
            0o644,
            b'x',
        ));
        layer.extend(entry("templates/caption-styles.json", b"[]\n", 0o644, 0));
        // A folder with a long name has a path record too, in either form. The record
        // names the folder, not the file after it.
        layer.extend(entry(
            "PaxHeaders/lower-thirds",
            pax(&folder).as_bytes(),
            0o644,
            b'x',
        ));
        layer.extend(entry("templates/lower-thirds/", b"", 0o755, b'5'));
        layer.extend(entry("NOTICE", b"Fonts: OFL-1.1\n", 0o644, b'0'));
        layer.extend(entry(
            "././@LongLink",
            format!("{folder}\0").as_bytes(),
            0o644,
            b'L',
        ));
        layer.extend(entry("templates/lower-thirds/", b"", 0o755, b'5'));
        layer.extend(entry("LICENSE", b"Apache-2.0\n", 0o644, b'0'));
        layer.resize(layer.len() + 2 * BLOCK, 0);

        let found = files(&artifact(&layout, &layer)).expect("the layer is read");

        assert_eq!(
            found,
            [
                (
                    ".claude-plugin/plugin.json".to_string(),
                    false,
                    "{}\n".to_string()
                ),
                (
                    "scripts/render.sh".to_string(),
                    true,
                    "#!/bin/sh\n".to_string()
                ),
                (long, false, "[]\n".to_string()),
                ("NOTICE".to_string(), false, "Fonts: OFL-1.1\n".to_string()),
                ("LICENSE".to_string(), false, "Apache-2.0\n".to_string()),
            ]
        );
    }

    #[test]
    fn what_a_package_cannot_hold_is_refused_in_a_layer() {
        let layout = Package::new("oci-refused");
        let end = vec![0u8; 2 * BLOCK];
        let one = |path: &str, kind: u8| [entry(path, b"", 0o644, kind), end.clone()].concat();
        let twice = tar(&[("plugin.json", "{}", false), ("./plugin.json", "{}", false)]);
        let mut cut_short = entry("plugin.json", b"{}", 0o644, b'0');
        cut_short.truncate(BLOCK + 1);
        let mut not_tar = tar(&[("plugin.json", "{}", false)]);
        not_tar[0] ^= 1;
        for (layer, code) in [
            (one("latest", b'2'), PackageErrorCode::SymlinkRefused),
            (one("copy", b'1'), PackageErrorCode::SymlinkRefused),
            (one("tty", b'3'), PackageErrorCode::FileUnreadable),
            (one("../outside.json", b'0'), PackageErrorCode::PathUnsafe),
            (one("/etc/plugin.json", b'0'), PackageErrorCode::PathUnsafe),
            (twice, PackageErrorCode::FileUnreadable),
            (cut_short, PackageErrorCode::FileUnreadable),
            (not_tar, PackageErrorCode::FileUnreadable),
        ] {
            let error = files(&artifact(&layout, &layer)).expect_err("refused");
            assert_eq!(error.code, code, "{}", error.message);
            assert_eq!(error.subject, AT);
        }
    }

    #[test]
    fn a_layer_that_is_not_the_one_the_manifest_pins_is_refused() {
        let layout = Package::new("oci-layer-digest");
        let layer = tar(&[("plugin.json", "{}", false)]);
        let artifact = artifact(&layout, &layer);
        // Other bytes under the layer's name, with the same length.
        let other = tar(&[("plugin.json", "[]", false)]);
        fs::write(
            blob_path(&layout.root, &bytes_digest(&layer)).expect("path"),
            other,
        )
        .expect("write the blob");

        let error = artifact.listing(AT).expect_err("refused");

        assert_eq!(error.code, PackageErrorCode::DigestMismatch);
        assert_eq!(error.subject, AT);

        // A blob of another length is refused by its length.
        let mut longer = layer.clone();
        longer.resize(layer.len() + BLOCK, 0);
        fs::write(
            blob_path(&layout.root, &bytes_digest(&layer)).expect("path"),
            longer,
        )
        .expect("write the blob");
        let error = artifact.listing(AT).expect_err("refused");
        assert_eq!(error.code, PackageErrorCode::DigestMismatch);
        assert!(
            error
                .message
                .ends_with(&format!("name has {} bytes", layer.len() + BLOCK)),
            "{}",
            error.message
        );
    }

    #[test]
    fn an_artifact_that_is_not_one_manifest_with_one_tar_layer_is_refused_by_name() {
        let layout = Package::new("oci-shapes");
        let layer = json!({"mediaType": TAR_LAYER, "digest": bytes_digest(b"layer"), "size": 5});
        let manifest = |layers: Value| json!({"schemaVersion": 2, "mediaType": MANIFEST_TYPE, "layers": layers});
        let gzip = json!({"mediaType": format!("{TAR_LAYER}+gzip"), "digest": bytes_digest(b"layer"), "size": 5});
        for (document, named) in [
            (
                json!({"schemaVersion": 2, "mediaType": INDEX_TYPE, "manifests": []}),
                "is an image index",
            ),
            (manifest(json!([layer, layer])), "has 2 layers"),
            (manifest(json!([])), "has 0 layers"),
            (manifest(json!([gzip])), "a compressed layer is not read"),
            (
                json!({"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json", "layers": [layer]}),
                "an OCI image manifest is read",
            ),
        ] {
            let digest = write_manifest(&layout.root, &document);
            let error = find_artifact(&layout.root, &digest, AT).expect_err("refused");
            assert_eq!(error.code, PackageErrorCode::FileUnreadable);
            assert!(error.message.contains(named), "{}", error.message);
        }
        // A layer larger than a folder can be is refused before any of it is read.
        let huge = json!({"mediaType": TAR_LAYER, "digest": bytes_digest(b"layer"), "size": MAX_LAYER_BYTES + 1});
        let digest = write_manifest(&layout.root, &manifest(json!([huge])));
        let error = find_artifact(&layout.root, &digest, AT).expect_err("refused");
        assert_eq!(error.code, PackageErrorCode::TooLarge);
        // A blob that does not have the digest of its name is not a manifest the lock pins.
        let digest = write_manifest(&layout.root, &manifest(json!([layer])));
        fs::write(blob_path(&layout.root, &digest).expect("path"), "{}").expect("write");
        let error = find_artifact(&layout.root, &digest, AT).expect_err("refused");
        assert_eq!(error.code, PackageErrorCode::DigestMismatch);
        // A layout folder that has no blob of the digest holds no such artifact.
        let absent = find_artifact(&layout.root, &bytes_digest(b"nothing"), AT).expect("read");
        assert!(absent.is_none());
    }
}
