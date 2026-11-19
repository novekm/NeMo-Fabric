// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Refusals raised while a package is read, checked, mapped or fetched.

/// One stable refusal code. The text of a code never changes meaning.
///
/// Two more codes exist that are never an error in Rust. The Python SDK raises
/// `package_config_required` when a report with no config is turned into one,
/// and `package_requirements_unmet` for a load that the `if_unsupported` or
/// `if_unverified` setting stopped. The command bridge gives the second as its
/// cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PackageErrorCode {
    NotFound,
    EntryMissing,
    FileUnreadable,
    JsonInvalid,
    JsonDuplicateKey,
    DocumentType,
    SchemaVersion,
    SchemaInvalid,
    ToolConnectorUnknown,
    VariableReferenceUndeclared,
    VariableReferenceType,
    VariableDefaultNotAllowed,
    VariableBoundsInvalid,
    PathUnsafe,
    SymlinkRefused,
    TooLarge,
    LockMissing,
    LockInvalid,
    LockEntryMissing,
    LockEntryExtra,
    LockEntryMismatch,
    VersionNotSelected,
    DigestMismatch,
    PartMissing,
    PartEntrypoint,
    AgentUnavailable,
    ReferenceType,
    ReferenceCycle,
    CalledAgentDescription,
    VariableMissing,
    VariableUndeclared,
    VariableType,
    VariableNotAllowed,
    VariableValueInvalid,
    DefinitionNameInUse,
    OptionsInvalid,
    ConfigConflict,
    LockDigestMismatch,
    BindingUndeclared,
    FetchFailed,
    OutDirInUse,
}

impl PackageErrorCode {
    /// Every code. The tests walk it.
    #[cfg(test)]
    pub(crate) const ALL: &'static [Self] = &[
        Self::NotFound,
        Self::EntryMissing,
        Self::FileUnreadable,
        Self::JsonInvalid,
        Self::JsonDuplicateKey,
        Self::DocumentType,
        Self::SchemaVersion,
        Self::SchemaInvalid,
        Self::ToolConnectorUnknown,
        Self::VariableReferenceUndeclared,
        Self::VariableReferenceType,
        Self::VariableDefaultNotAllowed,
        Self::VariableBoundsInvalid,
        Self::PathUnsafe,
        Self::SymlinkRefused,
        Self::TooLarge,
        Self::LockMissing,
        Self::LockInvalid,
        Self::LockEntryMissing,
        Self::LockEntryExtra,
        Self::LockEntryMismatch,
        Self::VersionNotSelected,
        Self::DigestMismatch,
        Self::PartMissing,
        Self::PartEntrypoint,
        Self::AgentUnavailable,
        Self::ReferenceType,
        Self::ReferenceCycle,
        Self::CalledAgentDescription,
        Self::VariableMissing,
        Self::VariableUndeclared,
        Self::VariableType,
        Self::VariableNotAllowed,
        Self::VariableValueInvalid,
        Self::DefinitionNameInUse,
        Self::OptionsInvalid,
        Self::ConfigConflict,
        Self::LockDigestMismatch,
        Self::BindingUndeclared,
        Self::FetchFailed,
        Self::OutDirInUse,
    ];

    /// The stable code text.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::NotFound => "package_not_found",
            Self::EntryMissing => "package_entry_missing",
            Self::FileUnreadable => "package_file_unreadable",
            Self::JsonInvalid => "package_json_invalid",
            Self::JsonDuplicateKey => "package_json_duplicate_key",
            Self::DocumentType => "package_document_type",
            Self::SchemaVersion => "package_schema_version",
            Self::SchemaInvalid => "package_schema_invalid",
            Self::ToolConnectorUnknown => "package_tool_connector_unknown",
            Self::VariableReferenceUndeclared => "package_variable_reference_undeclared",
            Self::VariableReferenceType => "package_variable_reference_type",
            Self::VariableDefaultNotAllowed => "package_variable_default_not_allowed",
            Self::VariableBoundsInvalid => "package_variable_bounds_invalid",
            Self::PathUnsafe => "package_path_unsafe",
            Self::SymlinkRefused => "package_symlink_refused",
            Self::TooLarge => "package_too_large",
            Self::LockMissing => "package_lock_missing",
            Self::LockInvalid => "package_lock_invalid",
            Self::LockEntryMissing => "package_lock_entry_missing",
            Self::LockEntryExtra => "package_lock_entry_extra",
            Self::LockEntryMismatch => "package_lock_entry_mismatch",
            Self::VersionNotSelected => "package_version_not_selected",
            Self::DigestMismatch => "package_digest_mismatch",
            Self::PartMissing => "package_part_missing",
            Self::PartEntrypoint => "package_part_entrypoint",
            Self::AgentUnavailable => "package_agent_unavailable",
            Self::ReferenceType => "package_reference_type",
            Self::ReferenceCycle => "package_reference_cycle",
            Self::CalledAgentDescription => "package_called_agent_description",
            Self::VariableMissing => "package_variable_missing",
            Self::VariableUndeclared => "package_variable_undeclared",
            Self::VariableType => "package_variable_type",
            Self::VariableNotAllowed => "package_variable_not_allowed",
            Self::VariableValueInvalid => "package_variable_value_invalid",
            Self::DefinitionNameInUse => "package_definition_name_in_use",
            Self::OptionsInvalid => "package_options_invalid",
            Self::ConfigConflict => "package_config_conflict",
            Self::LockDigestMismatch => "package_lock_digest_mismatch",
            Self::BindingUndeclared => "package_binding_undeclared",
            Self::FetchFailed => "package_fetch_failed",
            Self::OutDirInUse => "package_out_dir_in_use",
        }
    }
}

/// One refusal: a stable code, the thing it names, and a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackageError {
    pub code: PackageErrorCode,
    pub subject: String,
    pub message: String,
}

impl PackageError {
    pub(crate) fn new(
        code: PackageErrorCode,
        subject: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            subject: subject.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for PackageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}: {}: {}",
            self.code.as_str(),
            self.subject,
            self.message
        )
    }
}

impl std::error::Error for PackageError {}

pub(crate) type PackageResult<T> = std::result::Result<T, PackageError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_is_unique_and_matches_the_stable_pattern() {
        let mut seen = std::collections::BTreeSet::new();
        for code in PackageErrorCode::ALL {
            let text = code.as_str();
            assert!(seen.insert(text), "duplicate code {text}");
            assert!(text.len() <= 64, "{text}");
            let mut characters = text.chars();
            assert!(characters.next().is_some_and(|c| c.is_ascii_lowercase()));
            assert!(
                characters.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "{text}"
            );
        }
        assert_eq!(seen.len(), 41);
    }

    #[test]
    fn display_gives_code_subject_and_message() {
        let error = PackageError::new(PackageErrorCode::DigestMismatch, "$/skills/a", "changed");
        assert_eq!(
            error.to_string(),
            "package_digest_mismatch: $/skills/a: changed"
        );
    }
}
