// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version constraints: an exact version, `>= V` and `~> V`.

use std::cmp::Ordering;

use super::error::{PackageError, PackageErrorCode, PackageResult};

/// The length of the number part of a version when `text` is a version, else None.
///
/// A version is one or more numbers joined by dots. It may end with `-` and a pre-release
/// part, and with `+` and a build part.
fn version_number_length(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let digits = |from: usize| {
        bytes[from..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count()
    };
    let mut at = digits(0);
    if at == 0 {
        return None;
    }
    while at < bytes.len() && bytes[at] == b'.' {
        let more = digits(at + 1);
        if more == 0 {
            break;
        }
        at += 1 + more;
    }
    let numbers = at;
    let part = |from: usize| {
        bytes[from..]
            .iter()
            .take_while(|b| b.is_ascii_alphanumeric() || **b == b'.' || **b == b'-')
            .count()
    };
    for mark in [b'-', b'+'] {
        if at < bytes.len() && bytes[at] == mark {
            let length = part(at + 1);
            if length == 0 {
                return None;
            }
            at += 1 + length;
        }
    }
    (at == bytes.len()).then_some(numbers)
}

/// The numbers of a version. None when it has a `-` or `+` part: the format does not say how
/// such a version orders. None also for a number too large to hold.
fn version_numbers(text: &str) -> Option<Vec<u64>> {
    if version_number_length(text)? != text.len() {
        return None;
    }
    text.split('.').map(|part| part.parse().ok()).collect()
}

/// A constraint as (operator, version): the operator is `>=`, `~>` or none.
fn split_constraint(constraint: &str) -> Option<(Option<&'static str>, &str)> {
    for operator in [">=", "~>"] {
        if let Some(rest) = constraint.strip_prefix(operator) {
            let version = rest.strip_prefix(' ').unwrap_or(rest);
            version_number_length(version)?;
            return Some((Some(operator), version));
        }
    }
    version_number_length(constraint)?;
    Some((None, constraint))
}

fn compare_padded(left: &[u64], right: &[u64]) -> Ordering {
    let width = left.len().max(right.len());
    let at = |numbers: &[u64], index: usize| numbers.get(index).copied().unwrap_or(0);
    (0..width)
        .map(|index| at(left, index).cmp(&at(right, index)))
        .find(|order| *order != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

/// Exact as written, ">= V", "~> V". None when the two cannot be compared.
///
/// An exact version is met by that version and no other, compared as written: `1.2` is not
/// `1.2.0`. `>= V` is met by V or any later version. `~> V` is met by V or a later version in
/// which only the last number written has moved; with one number written, that number does
/// not move. Versions are ordered number by number from the left, and a missing number counts
/// as zero.
pub(crate) fn satisfies(have: &str, constraint: &str) -> Option<bool> {
    let (operator, wanted) = split_constraint(constraint)?;
    version_number_length(have)?;
    let Some(operator) = operator else {
        return Some(have == wanted);
    };
    let left = version_numbers(have)?;
    let floor = version_numbers(wanted)?;
    if compare_padded(&left, &floor) == Ordering::Less {
        return Some(false);
    }
    if operator == ">=" {
        return Some(true);
    }
    // The numbers that may not pass their next value.
    let held = if floor.len() > 1 {
        &floor[..floor.len() - 1]
    } else {
        &floor[..]
    };
    let mut ceiling = held.to_vec();
    let last = ceiling.last_mut()?;
    *last = last.checked_add(1)?;
    let width = left.len().max(floor.len());
    let mut padded = left;
    padded.resize(width, 0);
    Some(padded[..ceiling.len()].cmp(&ceiling[..]) == Ordering::Less)
}

/// 40 lowercase hex characters.
pub(crate) fn is_commit(text: &str) -> bool {
    text.len() == 40
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// One reference's `version` against what the lock selected.
/// Git source and a commit: it must equal the lock's `commit` and `version`.
/// Another source and a commit: refused. Otherwise `satisfies` must be Some(true); None is refused.
pub(crate) fn check_selected(
    source_is_git: bool,
    wanted: &str,
    lock_version: &str,
    lock_commit: Option<&str>,
    subject: &str,
) -> PackageResult<()> {
    if is_commit(wanted) {
        if !source_is_git {
            return Err(PackageError::new(
                PackageErrorCode::VersionNotSelected,
                subject,
                "the version is written as a commit, and the source is not a git source",
            ));
        }
        if lock_commit != Some(wanted) {
            return Err(PackageError::new(
                PackageErrorCode::VersionNotSelected,
                subject,
                format!(
                    "the reference asks for commit {wanted}, the lock pins commit {}",
                    lock_commit.unwrap_or("none")
                ),
            ));
        }
        if lock_version != wanted {
            return Err(PackageError::new(
                PackageErrorCode::LockEntryMismatch,
                subject,
                format!(
                    "the reference asks for a commit, and the lock records the version `{lock_version}` in place of the commit"
                ),
            ));
        }
        return Ok(());
    }
    match satisfies(lock_version, wanted) {
        Some(true) => Ok(()),
        Some(false) => Err(PackageError::new(
            PackageErrorCode::VersionNotSelected,
            subject,
            format!("the pinned version `{lock_version}` does not satisfy `{wanted}`"),
        )),
        None => Err(PackageError::new(
            PackageErrorCode::VersionNotSelected,
            subject,
            format!("the pinned version `{lock_version}` cannot be compared with `{wanted}`"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn an_exact_version_is_compared_as_written() {
        assert_eq!(satisfies("1.2.0", "1.2.0"), Some(true));
        assert_eq!(satisfies("1.2", "1.2.0"), Some(false));
        assert_eq!(satisfies("1.2.0", "1.2"), Some(false));
        assert_eq!(satisfies("1.2.0-rc1", "1.2.0-rc1"), Some(true));
        assert_eq!(satisfies("1.2.0+build.5", "1.2.0"), Some(false));
    }

    #[test]
    fn at_least_takes_that_version_or_a_later_one() {
        assert_eq!(satisfies("1.4.0", ">= 1.4.0"), Some(true));
        assert_eq!(satisfies("2.0", ">=1.4.0"), Some(true));
        assert_eq!(satisfies("1.4", ">= 1.4.0"), Some(true));
        assert_eq!(satisfies("1.3.9", ">= 1.4.0"), Some(false));
        assert_eq!(satisfies("12.4", ">= 12.0"), Some(true));
        assert_eq!(satisfies("11.8", ">= 12.0"), Some(false));
    }

    #[test]
    fn pessimistic_moves_only_the_last_number_written() {
        assert_eq!(satisfies("1.2", "~> 1.2"), Some(true));
        assert_eq!(satisfies("1.9.4", "~> 1.2"), Some(true));
        assert_eq!(satisfies("2.0", "~> 1.2"), Some(false));
        assert_eq!(satisfies("1.1", "~> 1.2"), Some(false));
        assert_eq!(satisfies("1.2.0", "~> 1.2.0"), Some(true));
        assert_eq!(satisfies("1.2.9", "~>1.2.0"), Some(true));
        assert_eq!(satisfies("1.3.0", "~> 1.2.0"), Some(false));
        assert_eq!(satisfies("1", "~> 1"), Some(true));
        assert_eq!(satisfies("1.9", "~> 1"), Some(true));
        assert_eq!(satisfies("2", "~> 1"), Some(false));
        assert_eq!(satisfies("0.9", "~> 1"), Some(false));
    }

    #[test]
    fn what_cannot_be_compared_gives_none() {
        // A pre-release or a build part has no order here.
        assert_eq!(satisfies("1.2.0-rc1", ">= 1.2.0"), None);
        assert_eq!(satisfies("1.2.0", ">= 1.2.0-rc1"), None);
        assert_eq!(satisfies("1.2.0+b1", "~> 1.2"), None);
        // Text that is not a version, or not a constraint.
        for (have, constraint) in [
            ("local", "1.0"),
            ("v1.2", "1.2"),
            ("1.2", "^1.2"),
            ("1.2", "> 1.2"),
            ("1.2", ">=  1.2"),
            ("1.2", "1.2, 1.3"),
            ("1.", "1."),
            ("", "1.0"),
            ("1.0", ""),
            ("1.2", "1.2-"),
            ("1.2", "latest"),
        ] {
            assert_eq!(
                satisfies(have, constraint),
                None,
                "{have} against {constraint}"
            );
        }
    }

    #[test]
    fn a_commit_is_forty_lowercase_hex_characters() {
        assert!(is_commit(COMMIT));
        assert!(!is_commit(&COMMIT[1..]));
        assert!(!is_commit(&COMMIT.to_uppercase()));
        assert!(!is_commit("g123456789abcdef0123456789abcdef01234567"));
    }

    #[test]
    fn a_commit_on_a_git_source_must_be_the_pinned_commit_and_version() {
        check_selected(true, COMMIT, COMMIT, Some(COMMIT), "$/skills/a").expect("the same commit");
        let other = "f".repeat(40);
        let error = check_selected(true, COMMIT, &other, Some(&other), "$/skills/a")
            .expect_err("another commit");
        assert_eq!(error.code, PackageErrorCode::VersionNotSelected);
        let error =
            check_selected(true, COMMIT, COMMIT, None, "$/skills/a").expect_err("no commit");
        assert_eq!(error.code, PackageErrorCode::VersionNotSelected);
        let error = check_selected(true, COMMIT, "1.0.0", Some(COMMIT), "$/skills/a")
            .expect_err("the commit is not recorded as the version");
        assert_eq!(error.code, PackageErrorCode::LockEntryMismatch);
    }

    #[test]
    fn a_commit_on_another_source_is_refused() {
        let error = check_selected(false, COMMIT, COMMIT, None, "$/plugins/a").expect_err("oci");
        assert_eq!(error.code, PackageErrorCode::VersionNotSelected);
        assert_eq!(error.subject, "$/plugins/a");
    }

    #[test]
    fn a_constraint_must_be_satisfied_by_the_pinned_version() {
        check_selected(true, "~> 1.7", "1.7.2", Some(COMMIT), "$/skills/a").expect("in range");
        check_selected(false, "2.3.0", "2.3.0", None, "$/connectors/a").expect("exact");
        let error =
            check_selected(false, "~> 1.7", "2.0.0", None, "$/skills/a").expect_err("out of range");
        assert_eq!(error.code, PackageErrorCode::VersionNotSelected);
        let error = check_selected(false, ">= 1.0", "local", None, "$/skills/a")
            .expect_err("cannot be compared");
        assert_eq!(error.code, PackageErrorCode::VersionNotSelected);
    }
}
