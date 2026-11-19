// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load an agent package: its entry document, its lock and its parts.

mod adapter;
mod bindings;
mod called;
mod checker;
mod connector;
mod definition;
mod digest;
mod document;
mod error;
mod facts;
mod fetch;
mod graph;
mod http_checker;
mod json;
mod load;
mod lock;
mod mapping;
mod model;
mod needs;
mod oci;
mod options;
mod paths;
mod reader;
mod report;
mod schemas;
mod variables;
mod version;

#[cfg(test)]
mod conformance;

pub use fetch::{
    AgentPackageFetchOptions, AgentPackageFetchReport, FetchedContent, NotFetchedContent,
    fetch_agent_package, fetch_agent_package_from_git,
};
pub use load::{load_agent_package, load_agent_package_with_adapter_directories};
pub use model::{
    DigestKind, DocumentKind, DocumentRecord, EntryType, Integrity, LockInterface, LockRecord,
    Note, NoteCode, PackageAction, SourceType, StatedBy, ValueOrigin, VariableValue,
};
pub use options::AgentPackageOptions;

/// The facts a checker URL is sent for a package, built from its load report: what `checker_url` posts.
///
/// Nothing is started and no checker is asked. `enforcers_present` is what the caller says runs where the agent
/// starts, as `checker_enforcers` gives it.
pub fn agent_package_start_facts(
    report: &AgentPackageReport,
    enforcers_present: &[String],
) -> serde_json::Value {
    http_checker::start_facts(report, enforcers_present)
}
pub use report::{
    AdapterReport, AgentPackageReport, CheckerDecision, CheckerFailure, CheckerResult,
    DeclaredItem, DeclaredKind, DeclaredStatus, DigestCheck, ItemEffect, MappingGap, ModelOrigin,
    ModelReport, PackageOutcome,
};
