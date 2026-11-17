// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! File and folder digests.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::model::FolderListing;

/// The most files one hashed folder may hold.
pub(crate) const MAX_FOLDER_FILES: usize = 50_000;
/// The deepest a hashed folder may nest.
pub(crate) const MAX_FOLDER_DEPTH: usize = 64;
/// The most bytes one hashed folder may hold.
pub(crate) const MAX_FOLDER_BYTES: u64 = 1024 * 1024 * 1024;

/// The one file name a folder's listing leaves out, at any depth. The format's own reference
/// checker leaves out a regular file of this name and no other: macOS Finder writes it into a
/// folder it has shown, so it is never part of a package, and a reference to one finds nothing.
pub(crate) const UNLISTED_FILE_NAME: &str = ".DS_Store";

const CHUNK_BYTES: usize = 64 * 1024;

/// The given bytes as lowercase hex.
pub(crate) fn hex_text(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `sha256:<64 hex>` for the bytes of a finished SHA-256.
pub(crate) fn digest_text(digest: impl AsRef<[u8]>) -> String {
    format!("sha256:{}", hex_text(digest))
}

/// `sha256:<64 hex>` of the given bytes.
pub(crate) fn bytes_digest(bytes: &[u8]) -> String {
    digest_text(Sha256::digest(bytes))
}

/// `sha256:<64 hex>` of one file's bytes, read in chunks.
pub(crate) fn file_digest(path: &Path, subject: &str) -> PackageResult<String> {
    let unreadable = |error: std::io::Error| {
        PackageError::new(
            PackageErrorCode::FileUnreadable,
            subject,
            format!("`{}` cannot be read: {error}", path.display()),
        )
    };
    let mut file = fs::File::open(path).map_err(unreadable)?;
    let mut hasher = Sha256::new();
    let mut chunk = vec![0u8; CHUNK_BYTES];
    loop {
        let read = file.read(&mut chunk).map_err(unreadable)?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(digest_text(hasher.finalize()))
}

/// The digest of a folder from the digest of each of its files.
///
/// The value is the JSON array of `[path, file digest]` pairs sorted by the UTF-8 bytes of the
/// path, written with no white space. For an array of strings that is the same bytes as the
/// canonical form the format names (RFC 8785).
fn listing_digest(files: &BTreeMap<String, String>) -> String {
    // A BTreeMap of strings is ordered by the UTF-8 bytes of its keys.
    let pairs: Vec<[&str; 2]> = files
        .iter()
        .map(|(path, digest)| [path.as_str(), digest.as_str()])
        .collect();
    let encoded = serde_json::to_vec(&pairs).expect("an array of strings serializes");
    bytes_digest(&encoded)
}

/// Lists every regular file under `folder` and computes the folder's digest.
///
/// A link or a file that is not regular is refused. `leave_out` is the lock
/// file being checked, when the folder holds it. A regular file named
/// [`UNLISTED_FILE_NAME`] is left out too, so it is neither listed nor
/// hashed. The digest is the SHA-256 of a compact JSON array of
/// `[path, file digest]` pairs, sorted by the UTF-8 bytes of the path.
pub(crate) fn folder_listing(
    folder: &Path,
    leave_out: Option<&Path>,
    subject: &str,
) -> PackageResult<FolderListing> {
    let bounds = Bounds {
        files: MAX_FOLDER_FILES,
        depth: MAX_FOLDER_DEPTH,
        bytes: MAX_FOLDER_BYTES,
    };
    bounded_listing(folder, leave_out, subject, bounds)
}

/// How much one hashed folder may hold.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    files: usize,
    depth: usize,
    bytes: u64,
}

fn bounded_listing(
    folder: &Path,
    leave_out: Option<&Path>,
    subject: &str,
    bounds: Bounds,
) -> PackageResult<FolderListing> {
    let mut files = BTreeMap::new();
    let mut bytes = 0u64;
    let mut pending = vec![(folder.to_path_buf(), String::new(), 1usize)];
    while let Some((directory, prefix, depth)) = pending.pop() {
        if depth > bounds.depth {
            return Err(PackageError::new(
                PackageErrorCode::TooLarge,
                subject,
                format!("the folder nests deeper than {} levels", bounds.depth),
            ));
        }
        let unreadable = |error: std::io::Error| {
            PackageError::new(
                PackageErrorCode::FileUnreadable,
                subject,
                format!("`{}` cannot be read: {error}", directory.display()),
            )
        };
        for item in fs::read_dir(&directory).map_err(unreadable)? {
            let item = item.map_err(unreadable)?;
            let path = item.path();
            let Some(name) = item.file_name().to_str().map(str::to_string) else {
                return Err(PackageError::new(
                    PackageErrorCode::PathUnsafe,
                    subject,
                    format!("a file name under `{}` is not UTF-8", directory.display()),
                ));
            };
            let unlisted = name == UNLISTED_FILE_NAME;
            let relative = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            // The entry's own type: a link is seen as a link, never followed.
            let kind = item.file_type().map_err(unreadable)?;
            if kind.is_symlink() {
                return Err(PackageError::new(
                    PackageErrorCode::SymlinkRefused,
                    subject,
                    format!("`{relative}` is a symbolic link"),
                ));
            }
            if kind.is_dir() {
                pending.push((path, relative, depth + 1));
                continue;
            }
            if !kind.is_file() {
                return Err(PackageError::new(
                    PackageErrorCode::FileUnreadable,
                    subject,
                    format!("`{relative}` is not a regular file"),
                ));
            }
            // Only a regular file is left out for its name. A folder of that name was walked
            // above, and a link or a special file of that name was refused.
            if unlisted || leave_out.is_some_and(|skip| skip == path) {
                continue;
            }
            if files.len() >= bounds.files {
                return Err(PackageError::new(
                    PackageErrorCode::TooLarge,
                    subject,
                    format!("the folder holds more than {} files", bounds.files),
                ));
            }
            bytes = bytes.saturating_add(item.metadata().map_err(unreadable)?.len());
            if bytes > bounds.bytes {
                return Err(PackageError::new(
                    PackageErrorCode::TooLarge,
                    subject,
                    format!("the folder holds more than {} bytes", bounds.bytes),
                ));
            }
            files.insert(relative, file_digest(&path, subject)?);
        }
    }
    Ok(FolderListing {
        digest: listing_digest(&files),
        files,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!(
            "nemo-fabric-agent-package-digest-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&folder);
        fs::create_dir_all(&folder).expect("create scratch folder");
        folder
    }

    /// The skill file of the format's own cases, and the digests its lock entries pin.
    const SKILL_TEXT: &str = "---\nname: review\ndescription: Review a render\n---\nReport every item of the checklist that fails.\n";
    const SKILL_FOLDER_DIGEST: &str =
        "sha256:8f89569ad30f380b1b7cef127bee2f8a45d87bf242a994973a93713055601f5a";

    #[test]
    fn bytes_and_files_have_the_same_digest() {
        assert_eq!(
            bytes_digest(b""),
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let folder = scratch("file");
        let path = folder.join("big.bin");
        // More than one chunk.
        let content = vec![7u8; CHUNK_BYTES * 2 + 5];
        fs::write(&path, &content).expect("write file");
        assert_eq!(
            file_digest(&path, "$").expect("file digest"),
            bytes_digest(&content)
        );
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }

    #[test]
    fn a_folder_has_the_digest_the_format_pins() {
        let folder = scratch("folder");
        fs::write(folder.join("SKILL.md"), SKILL_TEXT).expect("write file");
        let listing = folder_listing(&folder, None, "$/skills/review").expect("listing");
        assert_eq!(listing.digest, SKILL_FOLDER_DIGEST);
        assert_eq!(
            listing.files.get("SKILL.md").map(String::as_str),
            Some(bytes_digest(SKILL_TEXT.as_bytes()).as_str())
        );
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }

    #[test]
    fn paths_are_relative_and_sorted_by_their_bytes() {
        let folder = scratch("names");
        fs::create_dir_all(folder.join("a")).expect("create folder");
        fs::write(folder.join("a/b.txt"), b"one").expect("write file");
        fs::write(folder.join("a-b.txt"), b"two").expect("write file");
        fs::write(folder.join("r\u{e9}sum\u{e9}.md"), b"three").expect("write file");
        let listing = folder_listing(&folder, None, "$/skills/a").expect("listing");
        let names: Vec<&str> = listing.files.keys().map(String::as_str).collect();
        // `-` sorts before `/`, and a non-ASCII name after both.
        assert_eq!(names, ["a-b.txt", "a/b.txt", "r\u{e9}sum\u{e9}.md"]);
        let expected = format!(
            "[[\"a-b.txt\",\"{}\"],[\"a/b.txt\",\"{}\"],[\"r\u{e9}sum\u{e9}.md\",\"{}\"]]",
            bytes_digest(b"two"),
            bytes_digest(b"one"),
            bytes_digest(b"three")
        );
        assert_eq!(listing.digest, bytes_digest(expected.as_bytes()));
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }

    #[test]
    fn names_are_written_the_way_the_canonical_form_writes_strings() {
        // RFC 8785 escapes the quote, the backslash and the control characters, with lowercase
        // hex, and writes everything else as it is.
        let mut files = BTreeMap::new();
        files.insert("a\"b\\c".to_string(), "sha256:00".to_string());
        files.insert("tab\there\u{1}\u{7f}".to_string(), "sha256:01".to_string());
        files.insert("\u{20ac}\u{1f600}".to_string(), "sha256:02".to_string());
        let expected = concat!(
            r#"[["a\"b\\c","sha256:00"],"#,
            r#"["tab\there\u0001"#,
            "\u{7f}",
            r#"","sha256:01"],"#,
            "[\"\u{20ac}\u{1f600}\",\"sha256:02\"]]"
        );
        assert_eq!(listing_digest(&files), bytes_digest(expected.as_bytes()));
    }

    #[test]
    fn the_lock_being_checked_is_left_out() {
        let folder = scratch("lock");
        fs::write(folder.join("SKILL.md"), SKILL_TEXT).expect("write file");
        fs::write(folder.join("oadl.lock.json"), b"{}").expect("write file");
        let with_lock = folder_listing(&folder, None, "$").expect("listing");
        assert_ne!(with_lock.digest, SKILL_FOLDER_DIGEST);
        let without =
            folder_listing(&folder, Some(&folder.join("oadl.lock.json")), "$").expect("listing");
        assert_eq!(without.digest, SKILL_FOLDER_DIGEST);
        assert!(!without.files.contains_key("oadl.lock.json"));
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }

    #[test]
    fn a_file_named_ds_store_is_left_out_at_any_depth() {
        let folder = scratch("finder");
        fs::write(folder.join("SKILL.md"), SKILL_TEXT).expect("write file");
        // macOS Finder writes this file into a folder it has shown.
        fs::create_dir_all(folder.join("references")).expect("create folder");
        fs::write(folder.join(UNLISTED_FILE_NAME), b"\0\0\0\x01Bud1").expect("write file");
        fs::write(folder.join("references").join(UNLISTED_FILE_NAME), b"Bud1").expect("write file");

        let listing = folder_listing(&folder, None, "$/skills/review").expect("listing");

        // The digest is the one the format pins for the skill file alone.
        assert_eq!(listing.digest, SKILL_FOLDER_DIGEST);
        let names: Vec<&str> = listing.files.keys().map(String::as_str).collect();
        assert_eq!(names, ["SKILL.md"]);
        // A file that is not listed counts toward no bound.
        let bounds = Bounds {
            files: 1,
            depth: 2,
            bytes: SKILL_TEXT.len() as u64,
        };
        bounded_listing(&folder, None, "$/skills/review", bounds).expect("at the bounds");
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }

    #[test]
    fn a_file_with_any_other_name_is_listed_and_changes_the_digest() {
        // Each name is close to the one that is left out, or is another file a desktop leaves.
        for name in [
            ".ds_store",
            ".DS_Store.bak",
            "._.DS_Store",
            "DS_Store",
            "Thumbs.db",
            ".gitignore",
        ] {
            let folder = scratch("finder-other");
            fs::write(folder.join("SKILL.md"), SKILL_TEXT).expect("write file");
            fs::write(folder.join(name), b"Bud1").expect("write file");
            let listing = folder_listing(&folder, None, "$/skills/review").expect("listing");
            assert_ne!(listing.digest, SKILL_FOLDER_DIGEST, "{name}");
            assert!(listing.files.contains_key(name), "{name}");
            fs::remove_dir_all(folder).expect("remove scratch folder");
        }
        // A folder of that name is a folder like any other: its files are listed.
        let folder = scratch("finder-folder");
        fs::write(folder.join("SKILL.md"), SKILL_TEXT).expect("write file");
        fs::create_dir_all(folder.join(UNLISTED_FILE_NAME)).expect("create folder");
        fs::write(folder.join(UNLISTED_FILE_NAME).join("notes.md"), b"more").expect("write file");
        let listing = folder_listing(&folder, None, "$/skills/review").expect("listing");
        let names: Vec<&str> = listing.files.keys().map(String::as_str).collect();
        assert_eq!(names, [".DS_Store/notes.md", "SKILL.md"]);
        assert_ne!(listing.digest, SKILL_FOLDER_DIGEST);
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }

    #[test]
    fn an_empty_folder_lists_nothing() {
        let folder = scratch("empty");
        fs::create_dir_all(folder.join("only/folders")).expect("create folders");
        let listing = folder_listing(&folder, None, "$").expect("listing");
        assert!(listing.files.is_empty());
        assert_eq!(listing.digest, bytes_digest(b"[]"));
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }

    #[test]
    fn a_folder_over_a_bound_is_refused() {
        let folder = scratch("bounds");
        fs::create_dir_all(folder.join("a/b")).expect("create folders");
        fs::write(folder.join("one"), b"12345").expect("write file");
        fs::write(folder.join("a/two"), b"12345").expect("write file");
        fs::write(folder.join("a/b/three"), b"12345").expect("write file");
        let wide = Bounds {
            files: 3,
            depth: 3,
            bytes: 15,
        };
        let listing = bounded_listing(&folder, None, "$/skills/big", wide).expect("at the bounds");
        assert_eq!(listing.files.len(), 3);
        for bounds in [
            Bounds { files: 2, ..wide },
            Bounds { depth: 2, ..wide },
            Bounds { bytes: 14, ..wide },
        ] {
            let error =
                bounded_listing(&folder, None, "$/skills/big", bounds).expect_err("over one bound");
            assert_eq!(error.code, PackageErrorCode::TooLarge, "{bounds:?}");
            assert_eq!(error.subject, "$/skills/big");
        }
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }

    #[cfg(unix)]
    #[test]
    fn a_link_inside_a_folder_is_refused() {
        let folder = scratch("link");
        fs::write(folder.join("SKILL.md"), SKILL_TEXT).expect("write file");
        std::os::unix::fs::symlink(folder.join("SKILL.md"), folder.join("alias.md"))
            .expect("create link");
        let error = folder_listing(&folder, None, "$/skills/review").expect_err("a link");
        assert_eq!(error.code, PackageErrorCode::SymlinkRefused);
        // A link is refused whatever its name is.
        fs::remove_file(folder.join("alias.md")).expect("remove link");
        std::os::unix::fs::symlink(folder.join("SKILL.md"), folder.join(UNLISTED_FILE_NAME))
            .expect("create link");
        let error = folder_listing(&folder, None, "$/skills/review").expect_err("a link");
        assert_eq!(error.code, PackageErrorCode::SymlinkRefused);
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }
}
