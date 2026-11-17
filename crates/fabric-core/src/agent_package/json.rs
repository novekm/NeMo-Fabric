// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded file reads and strict JSON parsing.

use std::fs;
use std::io::Read;
use std::path::Path;

use serde_json::Value;

use super::error::{PackageError, PackageErrorCode, PackageResult};

/// The largest document the loader reads.
pub(crate) const MAX_DOCUMENT_BYTES: u64 = 4 * 1024 * 1024;

/// A regular file, not a link, at most `limit` bytes.
pub(crate) fn read_regular_file(path: &Path, subject: &str, limit: u64) -> PackageResult<Vec<u8>> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        PackageError::new(
            PackageErrorCode::FileUnreadable,
            subject,
            format!("the file cannot be read: {error}"),
        )
    })?;
    if metadata.file_type().is_symlink() {
        return Err(PackageError::new(
            PackageErrorCode::SymlinkRefused,
            subject,
            "the file is a symbolic link",
        ));
    }
    if !metadata.is_file() {
        return Err(PackageError::new(
            PackageErrorCode::FileUnreadable,
            subject,
            "it is not a regular file",
        ));
    }
    if metadata.len() > limit {
        return Err(too_large(subject, limit));
    }
    let file = fs::File::open(path).map_err(|error| {
        PackageError::new(
            PackageErrorCode::FileUnreadable,
            subject,
            format!("the file cannot be opened: {error}"),
        )
    })?;
    // The length is checked again while reading: the file can grow after the first look.
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| {
            PackageError::new(
                PackageErrorCode::FileUnreadable,
                subject,
                format!("the file cannot be read: {error}"),
            )
        })?;
    if bytes.len() as u64 > limit {
        return Err(too_large(subject, limit));
    }
    Ok(bytes)
}

fn too_large(subject: &str, limit: u64) -> PackageError {
    PackageError::new(
        PackageErrorCode::TooLarge,
        subject,
        format!("the file is larger than {limit} bytes"),
    )
}

/// One JSON object. A key written twice, a root that is not an object, or bad syntax is refused.
pub(crate) fn parse_document(bytes: &[u8], subject: &str) -> PackageResult<Value> {
    let value: Value = serde_json::from_slice(bytes).map_err(|error| {
        PackageError::new(
            PackageErrorCode::JsonInvalid,
            subject,
            format!("it is not valid JSON: {error}"),
        )
    })?;
    // The text parsed, so a failure of the strict pass is a key written twice. A map keeps one
    // value for each key, so this pass reads the text itself.
    if let Err(error) = crate::strict_json::validate(bytes) {
        return Err(PackageError::new(
            PackageErrorCode::JsonDuplicateKey,
            subject,
            format!("a key is written twice in one object: {error}"),
        ));
    }
    if !value.is_object() {
        return Err(PackageError::new(
            PackageErrorCode::JsonInvalid,
            subject,
            "the root is not a JSON object",
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let folder = std::env::temp_dir().join(format!(
            "nemo-fabric-agent-package-json-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&folder);
        fs::create_dir_all(&folder).expect("create scratch folder");
        folder
    }

    #[test]
    fn an_object_is_read() {
        let value = parse_document(br#"{"type": "agent", "name": "a"}"#, "agent.json")
            .expect("valid document");
        assert_eq!(value["name"], "a");
    }

    #[test]
    fn a_key_written_twice_is_refused_at_any_depth() {
        for text in [
            r#"{"name": "a", "name": "b"}"#,
            r#"{"skills": {"review": {"source": "./a"}, "review": {"source": "./b"}}}"#,
            r#"{"tools": [{"builtin": "a", "builtin": "b"}]}"#,
        ] {
            let error = parse_document(text.as_bytes(), "agent.json").expect_err(text);
            assert_eq!(error.code, PackageErrorCode::JsonDuplicateKey, "{text}");
            assert_eq!(error.subject, "agent.json");
        }
    }

    #[test]
    fn a_root_that_is_not_an_object_is_refused() {
        for text in ["[]", "\"agent\"", "1", "null", "true"] {
            let error = parse_document(text.as_bytes(), "agent.json").expect_err(text);
            assert_eq!(error.code, PackageErrorCode::JsonInvalid, "{text}");
        }
    }

    #[test]
    fn bad_syntax_is_refused() {
        for text in ["", "{", "{\"a\": NaN}", "{\"a\": 1,}", "\u{feff}{}"] {
            let error = parse_document(text.as_bytes(), "agent.json").expect_err(text);
            assert_eq!(error.code, PackageErrorCode::JsonInvalid, "{text:?}");
        }
    }

    #[test]
    fn a_file_over_the_bound_is_refused() {
        let folder = scratch("oversize");
        let path = folder.join("agent.json");
        fs::write(&path, vec![b' '; 65]).expect("write file");
        let error = read_regular_file(&path, "agent.json", 64).expect_err("over the bound");
        assert_eq!(error.code, PackageErrorCode::TooLarge);
        assert_eq!(
            read_regular_file(&path, "agent.json", 65)
                .expect("at the bound")
                .len(),
            65
        );
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }

    #[test]
    fn a_missing_file_and_a_folder_are_refused() {
        let folder = scratch("missing");
        let error = read_regular_file(&folder.join("none.json"), "none.json", 64)
            .expect_err("missing file");
        assert_eq!(error.code, PackageErrorCode::FileUnreadable);
        let error = read_regular_file(&folder, "folder", 64).expect_err("a folder");
        assert_eq!(error.code, PackageErrorCode::FileUnreadable);
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }

    #[cfg(unix)]
    #[test]
    fn a_link_is_refused() {
        let folder = scratch("link");
        fs::write(folder.join("real.json"), b"{}").expect("write file");
        std::os::unix::fs::symlink(folder.join("real.json"), folder.join("link.json"))
            .expect("create link");
        let error =
            read_regular_file(&folder.join("link.json"), "link.json", 64).expect_err("a link");
        assert_eq!(error.code, PackageErrorCode::SymlinkRefused);
        fs::remove_dir_all(folder).expect("remove scratch folder");
    }
}
