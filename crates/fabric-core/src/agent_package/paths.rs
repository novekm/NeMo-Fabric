// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Paths inside a package, lock addresses and JSON Pointers.

use std::fs;
use std::path::{Component, Path, PathBuf};

use super::error::{PackageError, PackageErrorCode, PackageResult};

/// A relative path with `/` between segments: no empty, `.` or `..` segment, no leading `/`,
/// no `\` and no NUL.
pub(crate) fn is_safe_relative(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && !path.contains('\0')
        && !path.starts_with('/')
        && path.split('/').all(is_safe_segment)
}

/// One path segment that names a file or a folder and nothing else.
fn is_safe_segment(segment: &str) -> bool {
    !(segment.is_empty()
        || segment == "."
        || segment == ".."
        || segment.contains('/')
        || segment.contains('\\')
        || segment.contains('\0')
        || (cfg!(windows) && segment.contains(':')))
}

/// `from_dir/relative`, normalized by text, with no look at the file system.
///
/// `relative` uses `/` between segments. An empty segment and `.` are dropped, and `..` steps
/// up. None when `from_dir` is not inside `root`, when a segment is not a plain name, or when
/// the result leaves `root` or is `root` itself.
pub(crate) fn normalize_inside(root: &Path, from_dir: &Path, relative: &str) -> Option<PathBuf> {
    if relative.contains('\\') || relative.contains('\0') || relative.starts_with('/') {
        return None;
    }
    let mut segments: Vec<String> = Vec::new();
    for component in from_dir.strip_prefix(root).ok()?.components() {
        match component {
            Component::Normal(name) => segments.push(name.to_str()?.to_string()),
            Component::CurDir => {}
            _ => return None,
        }
    }
    for segment in relative.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop()?;
            }
            name if is_safe_segment(name) => segments.push(name.to_string()),
            _ => return None,
        }
    }
    if segments.is_empty() {
        return None;
    }
    let mut path = root.to_path_buf();
    path.extend(segments);
    Some(path)
}

/// Refuses a symbolic link on the way from `root` down to `path`.
///
/// Each component below `root` that exists is checked. `root` itself may be reached through
/// links, as a temporary folder on macOS or a container mount is.
pub(crate) fn refuse_links(root: &Path, path: &Path, subject: &str) -> PackageResult<()> {
    let Ok(below) = path.strip_prefix(root) else {
        return Err(PackageError::new(
            PackageErrorCode::PathUnsafe,
            subject,
            "the path is not inside the package folder",
        ));
    };
    let mut current = root.to_path_buf();
    for component in below.components() {
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(PackageError::new(
                    PackageErrorCode::SymlinkRefused,
                    subject,
                    format!("`{}` is a symbolic link", display_inside(root, &current)),
                ));
            }
            Ok(_) => {}
            // What is not there has nothing below it to check.
            Err(_) => break,
        }
    }
    Ok(())
}

/// Joins `relative` onto `from_dir` and normalizes the result by text.
///
/// A `..` segment is allowed only while the result stays inside `root`. Each
/// existing component below `root` is checked with `symlink_metadata`, and a
/// link is refused. `root` itself can be reached through links, such as
/// `/tmp` on macOS or a container mount.
pub(crate) fn resolve_inside(
    root: &Path,
    from_dir: &Path,
    relative: &str,
    subject: &str,
) -> PackageResult<PathBuf> {
    let path = normalize_inside(root, from_dir, relative).ok_or_else(|| {
        PackageError::new(
            PackageErrorCode::PathUnsafe,
            subject,
            format!("the path `{relative}` does not name a place inside the package folder"),
        )
    })?;
    refuse_links(root, &path, subject)?;
    Ok(path)
}

/// A path as text for messages: relative to `root` with `/` between segments when it is inside.
pub(crate) fn display_inside(root: &Path, path: &Path) -> String {
    match path.strip_prefix(root) {
        Ok(below) => {
            let segments: Vec<String> = below
                .components()
                .map(|component| component.as_os_str().to_string_lossy().into_owned())
                .collect();
            if segments.is_empty() {
                ".".to_string()
            } else {
                segments.join("/")
            }
        }
        Err(_) => path.display().to_string(),
    }
}

/// One segment of a lock address: `~` is written `~0` and `/` is written `~1`.
fn escape_segment(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

/// `parent` followed by one address segment for each of `segments`.
pub(crate) fn address(parent: &str, segments: &[&str]) -> String {
    let mut text = parent.to_string();
    for segment in segments {
        text.push('/');
        text.push_str(&escape_segment(segment));
    }
    text
}

/// The segments of a lock address after `$`, with `~0` and `~1` read back.
pub(crate) fn address_segments(address: &str) -> PackageResult<Vec<String>> {
    let invalid = || {
        PackageError::new(
            PackageErrorCode::LockInvalid,
            address,
            "it is not a lock address: `$`, then `/` and a segment for each step",
        )
    };
    let rest = address.strip_prefix('$').ok_or_else(invalid)?;
    if rest.is_empty() {
        return Ok(Vec::new());
    }
    let rest = rest.strip_prefix('/').ok_or_else(invalid)?;
    let mut segments = Vec::new();
    for raw in rest.split('/') {
        if raw.is_empty() {
            return Err(invalid());
        }
        let mut segment = String::with_capacity(raw.len());
        let mut characters = raw.chars();
        while let Some(character) = characters.next() {
            if character != '~' {
                segment.push(character);
                continue;
            }
            match characters.next() {
                Some('0') => segment.push('~'),
                Some('1') => segment.push('/'),
                _ => return Err(invalid()),
            }
        }
        segments.push(segment);
    }
    Ok(segments)
}

/// The folder or file a lock address names, below `entry_dir`.
///
/// The address `$/agents/a/skills/s` gives `<entry_dir>/agents/a/skills/s`,
/// and `$/agent` gives `<entry_dir>/agent`.
pub(crate) fn layout_path(entry_dir: &Path, address: &str) -> PackageResult<PathBuf> {
    let mut path = entry_dir.to_path_buf();
    for segment in address_segments(address)? {
        if !is_safe_segment(&segment) {
            return Err(PackageError::new(
                PackageErrorCode::PathUnsafe,
                address,
                format!("the address segment `{segment}` is not a folder name"),
            ));
        }
        path.push(segment);
    }
    Ok(path)
}

/// An RFC 6901 JSON Pointer for the given segments.
pub(crate) fn json_pointer(segments: &[&str]) -> String {
    segments
        .iter()
        .map(|segment| format!("/{}", escape_segment(segment)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!(
            "nemo-fabric-agent-package-paths-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&folder);
        fs::create_dir_all(&folder).expect("create scratch folder");
        folder
    }

    #[test]
    fn safe_relative_paths() {
        for path in ["a", "a/b", "skills/review/SKILL.md", "a b/c", "..a/b"] {
            assert!(is_safe_relative(path), "{path}");
        }
        for path in [
            "", "/a", "a/", "a//b", ".", "..", "a/./b", "a/../b", "a\\b", "a\0b",
        ] {
            assert!(!is_safe_relative(path), "{path:?}");
        }
    }

    #[test]
    fn a_relative_path_is_normalized_by_text() {
        let root = Path::new("/packages/video");
        let from = root.join("agents/helper");
        let inside = |relative: &str| normalize_inside(root, &from, relative);
        assert_eq!(
            inside("skills/review"),
            Some(root.join("agents/helper/skills/review"))
        );
        assert_eq!(
            inside("../../skills/review"),
            Some(root.join("skills/review"))
        );
        assert_eq!(inside("./a//b/../c"), Some(root.join("agents/helper/a/c")));
        assert_eq!(inside(".."), Some(root.join("agents")));
        // The package folder itself, and anything above it, is not a place inside it.
        assert_eq!(inside("../.."), None);
        assert_eq!(inside("../../.."), None);
        assert_eq!(inside("../../../video/skills"), None);
        assert_eq!(inside("/etc"), None);
        assert_eq!(inside("a\\b"), None);
        assert_eq!(normalize_inside(root, Path::new("/elsewhere"), "a"), None);
        assert_eq!(normalize_inside(root, root, "."), None);
        assert_eq!(
            normalize_inside(root, root, "agent.json"),
            Some(root.join("agent.json"))
        );
    }

    #[test]
    fn a_path_that_leaves_the_package_is_refused() {
        let root = scratch("escape");
        let error = resolve_inside(&root, &root, "../outside", "$/skills/a").expect_err("escape");
        assert_eq!(error.code, PackageErrorCode::PathUnsafe);
        assert_eq!(error.subject, "$/skills/a");
        let inside = resolve_inside(
            &root,
            &root.join("configs"),
            "../parts/review",
            "$/skills/a",
        )
        .expect("stays inside");
        assert_eq!(inside, root.join("parts/review"));
        fs::remove_dir_all(root).expect("remove scratch folder");
    }

    #[cfg(unix)]
    #[test]
    fn a_link_below_the_root_is_refused_and_a_root_reached_through_a_link_is_not() {
        let base = scratch("links");
        let real = base.join("real");
        fs::create_dir_all(real.join("skills/review")).expect("create folders");
        fs::write(real.join("skills/review/SKILL.md"), b"review").expect("write file");
        std::os::unix::fs::symlink(real.join("skills"), real.join("linked")).expect("create link");
        std::os::unix::fs::symlink(&real, base.join("alias")).expect("create link");

        let error = resolve_inside(&real, &real, "linked/review", "$/skills/review")
            .expect_err("a link below the root");
        assert_eq!(error.code, PackageErrorCode::SymlinkRefused);

        // The root itself is reached through a link here.
        let alias = base.join("alias");
        let path = resolve_inside(&alias, &alias, "skills/review", "$/skills/review")
            .expect("the root may be a link");
        assert_eq!(path, alias.join("skills/review"));
        let error = resolve_inside(&alias, &alias, "linked/review", "$/skills/review")
            .expect_err("a link below a linked root");
        assert_eq!(error.code, PackageErrorCode::SymlinkRefused);
        fs::remove_dir_all(base).expect("remove scratch folder");
    }

    #[test]
    fn addresses_escape_and_read_back() {
        assert_eq!(address("$", &["skills", "review"]), "$/skills/review");
        assert_eq!(
            address("$/agent", &["skills", "a/b~c"]),
            "$/agent/skills/a~1b~0c"
        );
        assert_eq!(address_segments("$").expect("root"), Vec::<String>::new());
        assert_eq!(
            address_segments("$/agent/skills/a~1b~0c").expect("escaped"),
            vec!["agent", "skills", "a/b~c"]
        );
        for bad in ["", "skills", "$skills", "$/", "$//a", "$/a~2", "$/a~"] {
            let error = address_segments(bad).expect_err(bad);
            assert_eq!(error.code, PackageErrorCode::LockInvalid, "{bad}");
        }
    }

    #[test]
    fn layout_paths_follow_the_address() {
        let entry_dir = Path::new("/packages/video");
        assert_eq!(
            layout_path(entry_dir, "$/agents/a/skills/s").expect("layout"),
            entry_dir.join("agents/a/skills/s")
        );
        assert_eq!(
            layout_path(entry_dir, "$/agent").expect("layout"),
            entry_dir.join("agent")
        );
        assert_eq!(layout_path(entry_dir, "$").expect("layout"), entry_dir);
        for bad in ["$/skills/..", "$/skills/a~1b", "$/skills/."] {
            let error = layout_path(entry_dir, bad).expect_err(bad);
            assert_eq!(error.code, PackageErrorCode::PathUnsafe, "{bad}");
        }
    }

    #[test]
    fn json_pointers_escape_their_segments() {
        assert_eq!(json_pointer(&[]), "");
        assert_eq!(json_pointer(&["skills", "review"]), "/skills/review");
        assert_eq!(json_pointer(&["tools", "a/b~c"]), "/tools/a~1b~0c");
    }

    #[test]
    fn paths_are_shown_relative_to_the_package() {
        let root = Path::new("/packages/video");
        assert_eq!(
            display_inside(root, &root.join("skills").join("review")),
            "skills/review"
        );
        assert_eq!(display_inside(root, root), ".");
    }
}
