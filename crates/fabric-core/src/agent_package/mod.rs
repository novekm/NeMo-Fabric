// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read an agent package: its entry document, its lock and its parts, each checked against the
//! digest the lock pins.
//!
//! The loader that maps a package onto a harness config builds on this reader. Until it does,
//! most of the reader is used by its own tests only.
#![allow(dead_code)]

mod called;
mod definition;
mod digest;
mod document;
mod error;
mod graph;
mod json;
mod lock;
mod model;
mod needs;
mod oci;
mod paths;
mod reader;
mod schemas;
mod variables;
mod version;

#[cfg(test)]
mod conformance;

pub use model::{
    DigestKind, DocumentKind, DocumentRecord, EntryType, Integrity, LockInterface, LockRecord,
    Note, NoteCode, PackageAction, SourceType, StatedBy, ValueOrigin, VariableValue,
};
