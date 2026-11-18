// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runs the package format's offline conformance cases and walks example packages.
//!
//! Both tests call the reader's functions directly. They need no adapter and no network.
//!
//! `conformance_cases_agree` reads a manifest and runs each case in its mode. The default root
//! is the vendored subset in `tests/fixtures/agent_packages/conformance`. Set
//! `NEMO_FABRIC_AGENT_PACKAGE_CONFORMANCE` to a full conformance root to run every case there.
//!
//! `example_packages_load` reads the package of every lock file under a folder. The default
//! folder is `examples/agent_package`. Set `NEMO_FABRIC_AGENT_PACKAGE_EXAMPLES` to a folder of
//! the format's own example packages to walk those instead.
//!
//! Each test prints one summary. Pass `-- --nocapture` to see it on a passing run.
//!
//! What a case decides. A manifest line with `error: null` must be accepted. Any other line
//! must be refused, and the refusal code must be in the class that [`code_class`] gives for the
//! finding. Expected texts and the format checker's own report ids are not compared.
//!
//! Three things differ from the format's own checker, and the runner allows for each:
//!
//! - The checker reports every defect of a case. The loader stops at the first one. A refusal
//!   agrees when it answers the manifest's finding or another finding the checker gives.
//! - The checker lists names in the order the keys are written in a file. The reader does not
//!   keep key order, so those lists are compared as sets.
//! - A graph case is about the lock, and a start needs a value for every variable. The runner
//!   gives a value for each variable the author left with no default.
//!
//! Every case that is not run is counted in the summary with its reason. So is every key of a
//! case's `expect` that is not compared.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Map, Value, json};

use super::called::check_called_agent;
use super::definition::{read_together, stored_beside};
use super::digest::bytes_digest;
use super::document::check_document;
use super::error::{PackageError, PackageErrorCode as Code};
use super::json::parse_document;
use super::lock::check_lock;
use super::model::{
    Decision, Facts, Integrity, NeedLine, PackageAction, PackageReading, ReadRequest, SourceType,
    ValueOrigin, VariableValue,
};
use super::needs::{decide, need_lines};
use super::reader::read_package;
use super::variables::resolve_variables;

const CONFORMANCE_ROOT_VARIABLE: &str = "NEMO_FABRIC_AGENT_PACKAGE_CONFORMANCE";
const EXAMPLES_ROOT_VARIABLE: &str = "NEMO_FABRIC_AGENT_PACKAGE_EXAMPLES";
const LOCK_FILE: &str = "oadl.lock.json";

/// Graph cases whose verdict comes from capture data that a folder of files does not hold.
const CAPTURE_ONLY_GRAPH_CASES: [&str; 2] =
    ["stored-part-not-stated", "stored-part-states-dependencies"];

const NOT_RUN_CAPTURE: &str = "graph: the verdict comes from capture data a folder does not hold";
const NOT_RUN_RESOLVER: &str =
    "graph: needs a resolver's capture (pins, selections or more than one package)";
const NOT_RUN_CANONICAL: &str = "canonical: needs a general canonical encoder";
const NOT_RUN_COMPATIBILITY: &str =
    "compatibility: six run in the mapping test, four comparison cases are not read";

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

// ---------------------------------------------------------------------------------------------
// JSON that keeps key order.
//
// A case stores a JSON file either as text or as a JSON object. An object stands for the bytes
// the format's checker writes for it: two-space indented, keys in the order written, non-ASCII
// characters as they are, and a final newline. `serde_json::Value` sorts keys, so the stored
// files of a case are read with this small reader instead.

#[derive(Debug, Clone, PartialEq)]
enum Stored {
    Null,
    Bool(bool),
    /// The number exactly as the case file writes it.
    Number(String),
    Text(String),
    List(Vec<Stored>),
    Map(Vec<(String, Stored)>),
}

impl Stored {
    fn parse(text: &str) -> Result<Self, String> {
        let mut reader = StoredReader {
            text: text.as_bytes(),
            at: 0,
        };
        let value = reader.value()?;
        reader.skip_space();
        if reader.at != reader.text.len() {
            return Err(format!("unexpected text at byte {}", reader.at));
        }
        Ok(value)
    }

    fn get(&self, key: &str) -> Option<&Stored> {
        match self {
            Self::Map(members) => members
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    /// The bytes a stored file stands for.
    fn file_bytes(&self) -> Vec<u8> {
        match self {
            Self::Text(text) => text.as_bytes().to_vec(),
            other => {
                let mut out = String::new();
                other.write_indented(0, &mut out);
                out.push('\n');
                out.into_bytes()
            }
        }
    }

    fn write_indented(&self, depth: usize, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Number(raw) => out.push_str(raw),
            Self::Text(text) => write_json_string(text, out),
            Self::List(items) if items.is_empty() => out.push_str("[]"),
            Self::List(items) => {
                out.push_str("[\n");
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(",\n");
                    }
                    out.push_str(&"  ".repeat(depth + 1));
                    item.write_indented(depth + 1, out);
                }
                out.push('\n');
                out.push_str(&"  ".repeat(depth));
                out.push(']');
            }
            Self::Map(members) if members.is_empty() => out.push_str("{}"),
            Self::Map(members) => {
                out.push_str("{\n");
                for (index, (key, value)) in members.iter().enumerate() {
                    if index > 0 {
                        out.push_str(",\n");
                    }
                    out.push_str(&"  ".repeat(depth + 1));
                    write_json_string(key, out);
                    out.push_str(": ");
                    value.write_indented(depth + 1, out);
                }
                out.push('\n');
                out.push_str(&"  ".repeat(depth));
                out.push('}');
            }
        }
    }
}

fn write_json_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

struct StoredReader<'a> {
    text: &'a [u8],
    at: usize,
}

impl StoredReader<'_> {
    fn skip_space(&mut self) {
        while self
            .text
            .get(self.at)
            .is_some_and(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.at += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.text.get(self.at) == Some(&byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!("expected `{}` at byte {}", byte as char, self.at))
        }
    }

    fn word(&mut self, word: &str, value: Stored) -> Result<Stored, String> {
        if self.text[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(format!("unexpected text at byte {}", self.at))
        }
    }

    fn value(&mut self) -> Result<Stored, String> {
        self.skip_space();
        match self.text.get(self.at) {
            Some(b'{') => {
                self.at += 1;
                let mut members = Vec::new();
                self.skip_space();
                if self.text.get(self.at) == Some(&b'}') {
                    self.at += 1;
                    return Ok(Stored::Map(members));
                }
                loop {
                    self.skip_space();
                    let key = self.string()?;
                    self.skip_space();
                    self.expect(b':')?;
                    members.push((key, self.value()?));
                    self.skip_space();
                    match self.text.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(Stored::Map(members));
                        }
                        _ => return Err(format!("expected `,` or `}}` at byte {}", self.at)),
                    }
                }
            }
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.skip_space();
                if self.text.get(self.at) == Some(&b']') {
                    self.at += 1;
                    return Ok(Stored::List(items));
                }
                loop {
                    items.push(self.value()?);
                    self.skip_space();
                    match self.text.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Stored::List(items));
                        }
                        _ => return Err(format!("expected `,` or `]` at byte {}", self.at)),
                    }
                }
            }
            Some(b'"') => Ok(Stored::Text(self.string()?)),
            Some(b't') => self.word("true", Stored::Bool(true)),
            Some(b'f') => self.word("false", Stored::Bool(false)),
            Some(b'n') => self.word("null", Stored::Null),
            Some(_) => {
                let start = self.at;
                while self.text.get(self.at).is_some_and(|byte| {
                    matches!(byte, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                }) {
                    self.at += 1;
                }
                if start == self.at {
                    return Err(format!("unexpected text at byte {start}"));
                }
                let raw = std::str::from_utf8(&self.text[start..self.at])
                    .map_err(|error| error.to_string())?;
                Ok(Stored::Number(raw.to_string()))
            }
            None => Err("unexpected end of text".to_string()),
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .text
            .get(self.at..self.at + 4)
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .ok_or_else(|| format!("short escape at byte {}", self.at))?;
        self.at += 4;
        u32::from_str_radix(digits, 16).map_err(|error| error.to_string())
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            let byte = *self
                .text
                .get(self.at)
                .ok_or_else(|| "unexpected end of text in a string".to_string())?;
            self.at += 1;
            match byte {
                b'"' => return String::from_utf8(out).map_err(|error| error.to_string()),
                b'\\' => {
                    let escape = *self
                        .text
                        .get(self.at)
                        .ok_or_else(|| "unexpected end of text in an escape".to_string())?;
                    self.at += 1;
                    let character = match escape {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let first = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&first) {
                                self.expect(b'\\')?;
                                self.expect(b'u')?;
                                let second = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&second) {
                                    return Err("a surrogate without its pair".to_string());
                                }
                                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else {
                                first
                            };
                            char::from_u32(code)
                                .ok_or_else(|| "a surrogate without its pair".to_string())?
                        }
                        other => return Err(format!("unknown escape `\\{}`", other as char)),
                    };
                    let mut buffer = [0u8; 4];
                    out.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
                }
                other => out.push(other),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The finding of a case and the refusal codes that answer it.

/// The class of a finding: `schema` for a schema finding, else the text before the first colon.
fn finding_class(finding: &str) -> &str {
    if finding.starts_with("schema:") {
        "schema"
    } else {
        finding.split(':').next().unwrap_or(finding)
    }
}

/// The refusal codes that answer one finding of the format's checker.
///
/// `run.not_started` is not here: it is a start decision, not a refusal.
fn code_class(mode: &str, finding: &str) -> Option<&'static [Code]> {
    Some(match finding_class(finding) {
        // A lock that fails its schema is refused as a lock.
        "schema" if mode == "lock" => &[Code::LockInvalid, Code::SchemaInvalid],
        "schema" => &[Code::SchemaInvalid],
        "json.duplicate_key" => &[Code::JsonDuplicateKey],
        "document.type" => &[Code::DocumentType],
        // A root that is not an object is refused when the text is read or when it is checked.
        "document.object" => &[Code::DocumentType, Code::JsonInvalid],
        "document.version" => &[Code::SchemaVersion],
        "document.tool_connector" | "definition.tool_connector" => &[Code::ToolConnectorUnknown],
        "document.variable_undeclared" => &[Code::VariableReferenceUndeclared],
        "document.variable_type" => &[Code::VariableReferenceType],
        "document.variable_default" => &[Code::VariableDefaultNotAllowed],
        "variable.missing" => &[Code::VariableMissing],
        "variable.undeclared"
        | "reference.variable_undeclared"
        | "definition.variable_undeclared" => &[Code::VariableUndeclared],
        "variable.type" | "reference.variable_type" | "definition.variable_type" => {
            &[Code::VariableType]
        }
        "variable.enum" | "reference.variable_enum" | "definition.variable_enum" => {
            &[Code::VariableNotAllowed]
        }
        "variable.value" => &[Code::VariableValueInvalid],
        "definition.name_in_use" => &[Code::DefinitionNameInUse],
        "definition.agent_type" | "called.type" | "reference.type" => &[Code::ReferenceType],
        "called.description" | "reference.agent_description" => &[Code::CalledAgentDescription],
        "lock.digest_binding" => &[Code::DigestMismatch],
        "lock.exact_closure" => &[Code::LockEntryMissing, Code::LockEntryExtra],
        "reference.unresolved" => &[Code::LockEntryMissing],
        "reference.missing" => &[Code::PartMissing],
        "reference.version_selection" => &[Code::VersionNotSelected],
        "lock.source_binding"
        | "lock.type_binding"
        | "lock.version_binding"
        | "lock.commit_binding"
        | "lock.constraint_binding"
        | "lock.interface_binding" => &[Code::LockEntryMismatch],
        "reference.connector_entrypoint" | "reference.hook_entrypoint" => &[Code::PartEntrypoint],
        "reference.package_boundary" => &[Code::PathUnsafe],
        "reference.cycle" => &[Code::ReferenceCycle],
        other if other.starts_with("lock.") => &[Code::LockInvalid],
        _ => return None,
    })
}

/// The thing a finding names after its colon, for the findings that name a variable or an
/// entry. The loader's refusal must name the same thing in its subject or its message.
fn finding_name(finding: &str) -> Option<&str> {
    let (class, name) = finding.split_once(':')?;
    let names_a_thing = class.starts_with("variable.")
        || class.starts_with("document.variable_")
        || class.starts_with("reference.variable_")
        || class.starts_with("definition.variable_")
        || class == "definition.name_in_use";
    (names_a_thing && !name.is_empty()).then_some(name)
}

/// Why a refusal does not answer a finding, or None when it does.
fn does_not_answer(mode: &str, finding: &str, error: &PackageError) -> Option<String> {
    let codes = code_class(mode, finding)?;
    if !codes.contains(&error.code) {
        let class: Vec<&str> = codes.iter().map(|code| code.as_str()).collect();
        return Some(format!("the code is not one of {class:?}"));
    }
    match finding_name(finding) {
        Some(name) if !error.subject.contains(name) && !error.message.contains(name) => {
            Some(format!("the refusal does not name `{name}`"))
        }
        _ => None,
    }
}

/// The findings the format's checker also gives for a lock that breaks one of its own rules.
///
/// The checker reports every defect of a graph case, and the manifest names one of them. The
/// loader checks the lock's own rules first and stops there. So a lock that names a dependency
/// with no entry, holds an entry nothing reaches, or writes an address twice is also answered
/// by the refusal for the lock's own rules.
fn lock_own_findings(lock: &Value) -> Vec<&'static str> {
    let Some(entries) = lock.get("entries").and_then(Value::as_array) else {
        return Vec::new();
    };
    let address = |entry: &Value| {
        entry
            .get("address")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let dependencies = |entry: &Value| -> Vec<String> {
        entry
            .get("dependencies")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    };
    let by_address: BTreeMap<String, Vec<String>> = entries
        .iter()
        .map(|entry| (address(entry), dependencies(entry)))
        .collect();
    let mut found = Vec::new();
    if by_address.len() != entries.len() {
        found.push("lock.address_unique");
    }
    if by_address
        .values()
        .flatten()
        .any(|target| !by_address.contains_key(target))
    {
        found.push("lock.dangling");
    }
    let mut reached = BTreeSet::new();
    let mut pending = vec!["$".to_string()];
    while let Some(next) = pending.pop() {
        if let Some(targets) = by_address.get(&next)
            && reached.insert(next)
        {
            pending.extend(targets.iter().cloned());
        }
    }
    if reached.len() != by_address.len() {
        found.push("lock.unreachable");
    }
    found
}

// ---------------------------------------------------------------------------------------------
// Running one case.

/// What the loader did with one case that it did not refuse.
#[derive(Default)]
struct Accepted {
    /// True when the start decision stops the run.
    stops: bool,
    /// The parts of the result that a case's `expect` is compared with. A key of `expect`
    /// that is not here is listed in the summary as not compared.
    result: Map<String, Value>,
}

/// Why a case did not reach a result.
enum Stop {
    /// The loader refused.
    Refused(PackageError),
    /// The case could not be given to the loader, or a result broke a rule the case states.
    Broken(String),
}

impl From<PackageError> for Stop {
    fn from(error: PackageError) -> Self {
        Self::Refused(error)
    }
}

enum Verdict {
    Agreed,
    Disagreed(String),
    NotRun(String),
}

fn object<'a>(case: &'a Value, key: &str) -> Result<&'a Value, Stop> {
    case.get(key)
        .ok_or_else(|| Stop::Broken(format!("the case has no `{key}`")))
}

/// The values whoever starts the agent gives in a case.
fn given(case: &Value) -> Map<String, Value> {
    case.get("values")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

/// One of the two settings of a case. The format's cases name them `on_unmet`
/// and `on_unchecked`; the options name them `if_unsupported` and `if_unverified`.
fn choice(case: &Value, key: &str, default: PackageAction) -> Result<PackageAction, Stop> {
    match case.get(key).and_then(Value::as_str) {
        None => Ok(default),
        Some("stop") => Ok(PackageAction::Stop),
        Some("warn") => Ok(PackageAction::Warn),
        Some(other) => Err(Stop::Broken(format!("`{key}` is `{other}`"))),
    }
}

fn facts(case: &Value) -> Result<Facts, Stop> {
    serde_json::from_value(object(case, "facts")?.clone()).map_err(|error| {
        Stop::Broken(format!(
            "the facts of the case do not fit the loader's facts: {error}"
        ))
    })
}

fn values_of(values: &[VariableValue]) -> (Value, Value) {
    let mut by_name = Map::new();
    let mut origin = Map::new();
    for value in values {
        by_name.insert(value.name.clone(), value.value.clone());
        origin.insert(
            value.name.clone(),
            serde_json::to_value(value.origin).expect("an origin is JSON"),
        );
    }
    (Value::Object(by_name), Value::Object(origin))
}

fn line_of(line: &NeedLine) -> Value {
    let mut out = Map::new();
    out.insert("need".to_string(), json!(line.need));
    out.insert(
        "status".to_string(),
        json!(line.status.format_text(line.connector.as_deref())),
    );
    out.insert("optional".to_string(), json!(line.optional));
    out.insert(
        "stated_by".to_string(),
        serde_json::to_value(line.stated_by).expect("a document kind is JSON"),
    );
    out.insert("document".to_string(), json!(line.document));
    if let Some(reason) = &line.reason {
        out.insert("reason".to_string(), json!(reason));
    }
    Value::Object(out)
}

/// The report of one run, in the shape a case's `expect` uses.
fn report_of(lines: &[NeedLine], decision: &Decision) -> Value {
    let statuses: Map<String, Value> = lines
        .iter()
        .map(|line| {
            (
                line.need.clone(),
                json!(line.status.format_text(line.connector.as_deref())),
            )
        })
        .collect();
    json!({
        "statuses": statuses,
        "starts": decision.starts,
        "required_not_met": decision.required_not_met,
        "required_not_checked": decision.required_not_checked,
        "lines": lines.iter().map(line_of).collect::<Vec<_>>(),
    })
}

fn run_document(case: &Value) -> Result<Accepted, Stop> {
    check_document(object(case, "document")?, "document")?;
    Ok(Accepted::default())
}

fn run_text(case: &Value) -> Result<Accepted, Stop> {
    let text = object(case, "text")?
        .as_str()
        .ok_or_else(|| Stop::Broken("`text` is not text".to_string()))?;
    let document = parse_document(text.as_bytes(), "document")?;
    check_document(&document, "document")?;
    Ok(Accepted::default())
}

fn run_lock(case: &Value) -> Result<Accepted, Stop> {
    check_lock(object(case, "lock")?, "lock")?;
    Ok(Accepted::default())
}

fn run_start(case: &Value) -> Result<Accepted, Stop> {
    let document = object(case, "document")?;
    check_document(document, "document")?;
    let resolved = resolve_variables(document, &given(case), None, ValueOrigin::Entry)?;
    let (values, came_from) = values_of(&resolved.values);
    let mut accepted = Accepted::default();
    accepted.result.insert("values".to_string(), values);
    accepted.result.insert("came_from".to_string(), came_from);
    accepted.result.insert("agent".to_string(), resolved.agent);
    Ok(accepted)
}

fn run_run(case: &Value) -> Result<Accepted, Stop> {
    let document = object(case, "document")?;
    check_document(document, "document")?;
    let resolved = resolve_variables(document, &given(case), None, ValueOrigin::Entry)?;
    let lines = need_lines(&resolved.agent, None, &facts(case)?);
    let decision = decide(
        &lines,
        choice(case, "on_unmet", PackageAction::Stop)?,
        choice(case, "on_unchecked", PackageAction::Warn)?,
    );
    let Value::Object(result) = report_of(&lines, &decision) else {
        unreachable!("a report is an object");
    };
    Ok(Accepted {
        stops: !decision.starts,
        result,
    })
}

fn run_definition(case: &Value) -> Result<Accepted, Stop> {
    let definition = object(case, "definition")?;
    let agent = object(case, "agent")?;
    check_document(definition, "definition")?;
    check_document(agent, "agent")?;
    // A case holds two documents and no folders, so where they are stored is
    // read from the pointer alone.
    let together = read_together(definition, agent, &given(case), stored_beside(definition))?;
    let (values, came_from) = values_of(&together.values);
    let mut accepted = Accepted::default();
    let sorted = |names: &[String]| {
        let mut names = names.to_vec();
        names.sort();
        json!(names)
    };
    accepted.result.insert("values".to_string(), values);
    accepted.result.insert("came_from".to_string(), came_from);
    accepted
        .result
        .insert("added".to_string(), sorted(&together.summary.added));
    accepted
        .result
        .insert("kept_once".to_string(), sorted(&together.summary.kept_once));
    accepted
        .result
        .insert("limits".to_string(), Value::Object(together.limits.clone()));
    accepted.result.insert(
        "instructions".to_string(),
        together
            .summary
            .instructions
            .iter()
            .map(|part| {
                json!({
                    "from": serde_json::to_value(part.from).expect("a document kind is JSON"),
                    "name": part.name,
                    "text": part.text,
                })
            })
            .collect(),
    );
    if case.get("facts").is_some() {
        let lines = need_lines(&together.agent, Some(definition), &facts(case)?);
        let decision = decide(
            &lines,
            choice(case, "on_unmet", PackageAction::Stop)?,
            choice(case, "on_unchecked", PackageAction::Warn)?,
        );
        accepted.stops = !decision.starts;
        accepted
            .result
            .insert("run".to_string(), report_of(&lines, &decision));
    }
    accepted.result.insert("agent".to_string(), together.agent);
    Ok(accepted)
}

fn run_called(case: &Value) -> Result<Accepted, Stop> {
    let caller = object(case, "caller")?;
    let document = object(case, "document")?;
    let name = object(case, "name")?
        .as_str()
        .ok_or_else(|| Stop::Broken("`name` is not text".to_string()))?;
    let depth = case.get("depth").and_then(Value::as_u64).unwrap_or(1);
    check_document(caller, "caller")?;
    check_document(document, "document")?;
    let called = check_called_agent(caller, name, document, depth)?;
    let (values, came_from) = values_of(&called.values);
    let mut accepted = Accepted::default();
    accepted
        .result
        .insert("under".to_string(), json!(called.under));
    accepted.result.insert(
        "model".to_string(),
        json!({"source": called.model_source, "from": called.model_from}),
    );
    accepted.result.insert("values".to_string(), values);
    accepted.result.insert("came_from".to_string(), came_from);
    // A case with `used` says what the call tree has used so far, and its limits are what is
    // left. The loader takes no such input, so those limits are not compared.
    if case.get("used").is_none() {
        accepted
            .result
            .insert("limits".to_string(), Value::Object(called.limits));
    }
    Ok(accepted)
}

/// A relative path with plain segments only.
fn plain_relative(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && path
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

/// `dir/relative`, with `.` and `..` worked out by text. None when it leaves the top.
fn join_text(dir: &str, relative: &str) -> Option<String> {
    let mut segments: Vec<&str> = dir
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    for segment in relative.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop()?;
            }
            other => segments.push(other),
        }
    }
    Some(segments.join("/"))
}

fn parent_text(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(dir, _)| dir)
}

/// A value for each variable that has no default and that `already_set` does not give.
///
/// A graph case and an example lock are about the lock. A start needs a value for every
/// variable, so the runner gives one where the author left the choice to whoever starts the
/// agent. Otherwise a missing start value would hide what the lock check says.
fn made_up_values(agent: &Value, already_set: Option<&Map<String, Value>>) -> Map<String, Value> {
    let mut values = Map::new();
    let Some(declared) = agent.get("variables").and_then(Value::as_object) else {
        return values;
    };
    for (name, variable) in declared {
        if variable.get("default").is_some()
            || already_set.is_some_and(|set| set.contains_key(name))
        {
            continue;
        }
        let value = match variable.get("enum").and_then(Value::as_array) {
            Some(allowed) if !allowed.is_empty() => allowed[0].clone(),
            _ => match variable.get("type").and_then(Value::as_str) {
                Some("number") => json!(1),
                Some("boolean") => json!(true),
                _ => json!("value"),
            },
        };
        values.insert(name.clone(), value);
    }
    values
}

/// The start values for the root document of a set of stored files.
///
/// `document_at` returns the JSON document stored at a path relative to the package, if any.
fn start_values(
    root_path: &str,
    document_at: &dyn Fn(&str) -> Option<Value>,
) -> Map<String, Value> {
    let Some(root) = document_at(root_path) else {
        return Map::new();
    };
    match root.get("type").and_then(Value::as_str) {
        Some("agent") => made_up_values(&root, None),
        Some("agent-definition") => {
            let Some(source) = root
                .pointer("/agent/source")
                .and_then(Value::as_str)
                .and_then(|source| source.strip_prefix("./"))
            else {
                return Map::new();
            };
            let Some(target) = join_text(parent_text(root_path), source) else {
                return Map::new();
            };
            let agent =
                document_at(&target).or_else(|| document_at(&format!("{target}/agent.json")));
            match agent {
                Some(agent) => {
                    made_up_values(&agent, root.get("variables").and_then(Value::as_object))
                }
                None => Map::new(),
            }
        }
        _ => Map::new(),
    }
}

static CASE_FOLDER: AtomicU64 = AtomicU64::new(0);

/// A fresh folder for one graph case. It is removed when the value is dropped.
struct CaseFolder(PathBuf);

impl CaseFolder {
    fn new() -> Result<Self, Stop> {
        let path = std::env::temp_dir().join(format!(
            "nemo-fabric-agent-package-case-{}-{}",
            std::process::id(),
            CASE_FOLDER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).map_err(|error| Stop::Broken(error.to_string()))?;
        Ok(Self(path))
    }
}

impl Drop for CaseFolder {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_file(folder: &Path, relative: &str, bytes: &[u8]) -> Result<(), Stop> {
    if !plain_relative(relative) {
        return Err(Stop::Broken(format!(
            "the case stores a file at `{relative}`"
        )));
    }
    let path = folder.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| Stop::Broken(error.to_string()))?;
    }
    fs::write(&path, bytes).map_err(|error| Stop::Broken(error.to_string()))
}

/// Whether a graph case is one local package with nothing fetched.
fn one_local_package(case: &Value) -> bool {
    let inventory = &case["inventory"];
    let empty = |key: &str| {
        inventory
            .get(key)
            .and_then(Value::as_object)
            .is_none_or(Map::is_empty)
    };
    inventory["packages"]
        .as_object()
        .is_some_and(|packages| packages.len() == 1)
        && empty("pins")
        && empty("selections")
}

fn addresses(case: &Value, key: &str) -> Vec<String> {
    case.get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn escaped_segment(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

fn run_graph(case: &Value, case_text: &str) -> Result<Accepted, Stop> {
    let broken = |text: &str| Stop::Broken(text.to_string());
    let root = &case["inventory"]["root"];
    let package = root["package"]
        .as_str()
        .ok_or_else(|| broken("no root package"))?;
    let root_path = root["path"]
        .as_str()
        .ok_or_else(|| broken("no root path"))?;
    let stored = Stored::parse(case_text).map_err(Stop::Broken)?;
    let Some(Stored::Map(files)) = stored
        .get("inventory")
        .and_then(|inventory| inventory.get("packages"))
        .and_then(|packages| packages.get(package))
        .and_then(|package| package.get("files"))
    else {
        return Err(broken("the root package has no files"));
    };

    let folder = CaseFolder::new()?;
    for (path, file) in files {
        write_file(&folder.0, path, &file.file_bytes())?;
    }
    let lock_path = match parent_text(root_path) {
        "" => LOCK_FILE.to_string(),
        dir => format!("{dir}/{LOCK_FILE}"),
    };
    let lock = serde_json::to_vec_pretty(&case["lock"])
        .map_err(|error| Stop::Broken(error.to_string()))?;
    write_file(&folder.0, &lock_path, &lock)?;

    let stored_files = &case["inventory"]["packages"][package]["files"];
    let document_at = |path: &str| {
        stored_files
            .get(path)
            .filter(|file| file.is_object())
            .cloned()
    };
    let request = ReadRequest {
        entry: Some(PathBuf::from(root_path)),
        variables: start_values(root_path, &document_at),
        ..ReadRequest::default()
    };
    let reading = read_package(&folder.0, &request)?;

    let pinned: BTreeSet<&str> = reading
        .parts
        .iter()
        .map(|part| part.address.as_str())
        .collect();
    for address in addresses(case, "required_addresses") {
        if !pinned.contains(address.as_str()) {
            return Err(Stop::Broken(format!(
                "`{address}` is not among the parts that were read"
            )));
        }
    }
    let left_out: BTreeSet<String> = reading
        .left_out
        .iter()
        .map(|part| {
            format!(
                "{}/{}/{}",
                part.owner,
                part.field.as_str(),
                escaped_segment(&part.name)
            )
        })
        .collect();
    for address in addresses(case, "absent_addresses") {
        if pinned.contains(address.as_str()) {
            return Err(Stop::Broken(format!(
                "`{address}` is among the parts and must not be"
            )));
        }
        // An optional reference of the root document that has no lock entry is reported.
        let segments: Vec<&str> = address.split('/').collect();
        let optional = segments.len() == 3
            && segments[0] == "$"
            && document_at(root_path).is_some_and(|document| {
                document
                    .get(segments[1])
                    .and_then(|field| field.get(segments[2]))
                    .and_then(|entry| entry.get("optional"))
                    == Some(&Value::Bool(true))
            });
        if optional && !left_out.contains(&address) {
            return Err(Stop::Broken(format!(
                "`{address}` is optional and has no lock entry, and it is not reported as left out"
            )));
        }
    }
    Ok(Accepted::default())
}

// ---------------------------------------------------------------------------------------------
// Comparing a result with what a case expects.

/// Where `expect` is not in `actual`, the way the format's checker compares: an object may be
/// a part of the actual object, a list matches item for item, and numbers match by value.
fn mismatch(expect: &Value, actual: &Value, path: &str) -> Option<String> {
    match expect {
        Value::Object(members) => {
            let Some(actual) = actual.as_object() else {
                return Some(path.to_string());
            };
            members.iter().find_map(|(key, value)| {
                let place = format!("{path}.{key}");
                match actual.get(key) {
                    None => Some(place),
                    Some(found) => mismatch(value, found, &place),
                }
            })
        }
        Value::Array(items) => {
            let Some(actual) = actual
                .as_array()
                .filter(|actual| actual.len() == items.len())
            else {
                return Some(path.to_string());
            };
            items
                .iter()
                .zip(actual)
                .enumerate()
                .find_map(|(index, (left, right))| {
                    mismatch(left, right, &format!("{path}.{index}"))
                })
        }
        Value::Number(left) => match actual {
            Value::Number(right) if left.as_f64() == right.as_f64() => None,
            _ => Some(path.to_string()),
        },
        other => (other != actual).then(|| path.to_string()),
    }
}

/// A list of names compared without its order.
///
/// The format's checker lists names in the order the keys are written in a file. The reader
/// does not keep that order, so these lists are compared as sets.
fn names_mismatch(expect: &Value, actual: &Value, path: &str) -> Option<String> {
    let names = |value: &Value| -> Option<Vec<String>> {
        let mut names: Vec<String> = value
            .as_array()?
            .iter()
            .map(|item| item.as_str().map(str::to_string))
            .collect::<Option<_>>()?;
        names.sort();
        Some(names)
    };
    (names(expect).is_none() || names(expect) != names(actual)).then(|| path.to_string())
}

/// The lines of a run report, matched by the need each line is about.
fn lines_mismatch(expect: &Value, actual: &Value, path: &str) -> Option<String> {
    let (Some(expect), Some(actual)) = (expect.as_array(), actual.as_array()) else {
        return Some(path.to_string());
    };
    if expect.len() != actual.len() {
        return Some(path.to_string());
    }
    expect.iter().find_map(|line| {
        let need = line.get("need").and_then(Value::as_str).unwrap_or_default();
        let place = format!("{path}[{need}]");
        match actual
            .iter()
            .find(|found| found.get("need") == line.get("need"))
        {
            None => Some(place),
            Some(found) => mismatch(line, found, &place),
        }
    })
}

const UNORDERED_NAMES: [&str; 4] = [
    "added",
    "kept_once",
    "required_not_met",
    "required_not_checked",
];

/// Compare the keys of `expect` that `actual` holds. Keys it does not hold go to `uncompared`.
fn compare(
    expect: &Map<String, Value>,
    actual: &Map<String, Value>,
    path: &str,
    uncompared: &mut Vec<String>,
) -> Option<String> {
    for (key, wanted) in expect {
        let place = format!("{path}.{key}");
        let Some(found) = actual.get(key) else {
            uncompared.push(place.trim_start_matches("expect.").to_string());
            continue;
        };
        let wrong = if key == "run" {
            match (wanted.as_object(), found.as_object()) {
                (Some(wanted), Some(found)) => compare(wanted, found, &place, uncompared),
                _ => Some(place.clone()),
            }
        } else if key == "lines" {
            lines_mismatch(wanted, found, &place)
        } else if UNORDERED_NAMES.contains(&key.as_str()) {
            names_mismatch(wanted, found, &place)
        } else {
            mismatch(wanted, found, &place)
        };
        if wrong.is_some() {
            return wrong;
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// The runner.

#[derive(Default)]
struct Summary {
    cases: usize,
    /// Mode to (cases run, cases that agreed).
    by_mode: BTreeMap<String, (usize, usize)>,
    not_run: BTreeMap<String, usize>,
    uncompared: BTreeMap<String, usize>,
    disagreements: Vec<String>,
}

impl Summary {
    fn run(&self) -> usize {
        self.by_mode.values().map(|(run, _)| run).sum()
    }

    fn agreed(&self) -> usize {
        self.by_mode.values().map(|(_, agreed)| agreed).sum()
    }

    fn print(&self, root: &Path) {
        println!("agent package conformance cases: root {}", root.display());
        println!(
            "  cases {}  run {}  agreed {}  disagreed {}  not run {}",
            self.cases,
            self.run(),
            self.agreed(),
            self.disagreements.len(),
            self.not_run.values().sum::<usize>()
        );
        for (mode, (run, agreed)) in &self.by_mode {
            println!("  mode {mode}: {agreed} of {run} agree");
        }
        for (reason, count) in &self.not_run {
            println!("  not run, {count}: {reason}");
        }
        for (key, count) in &self.uncompared {
            println!("  expectation not compared, {count}: {key}");
        }
        for line in &self.disagreements {
            println!("  DISAGREES {line}");
        }
    }
}

fn panic_text(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|text| (*text).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".to_string())
}

/// Run one case and say whether the loader agrees with the manifest line.
fn judge(
    name: &str,
    case: &Value,
    case_text: &str,
    expected: Option<&str>,
    uncompared: &mut Vec<String>,
) -> Verdict {
    let mode = case["mode"].as_str().unwrap_or_default();
    match mode {
        "document" | "text" | "lock" | "start" | "definition" | "called" | "run" => {}
        "graph" if !one_local_package(case) => {
            return Verdict::NotRun(NOT_RUN_RESOLVER.to_string());
        }
        "graph" if CAPTURE_ONLY_GRAPH_CASES.contains(&name) => {
            return Verdict::NotRun(NOT_RUN_CAPTURE.to_string());
        }
        "graph" => {}
        "canonical" => return Verdict::NotRun(NOT_RUN_CANONICAL.to_string()),
        "compatibility" => return Verdict::NotRun(NOT_RUN_COMPATIBILITY.to_string()),
        other => return Verdict::NotRun(format!("mode `{other}` is not known to the runner")),
    }
    // The findings a refusal may answer. The format's checker reports every defect of a case:
    // it lists every refusal in `expect.refused`, and for a graph case it also checks the
    // lock's own rules. The loader stops at the first defect it finds, so its refusal agrees
    // when it answers the manifest's finding or another finding the checker gives.
    let findings: Option<Vec<&str>> = match expected {
        None | Some("run.not_started") => None,
        Some(finding) => {
            if code_class(mode, finding).is_none() {
                return Verdict::NotRun(format!(
                    "finding `{}` is not in the runner's table",
                    finding_class(finding)
                ));
            }
            let listed = case
                .pointer("/expect/refused")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|refusal| refusal.get("id").and_then(Value::as_str))
                .filter(|id| *id != finding && code_class(mode, id).is_some());
            let lock_rules = if mode == "graph" {
                lock_own_findings(&case["lock"])
            } else {
                Vec::new()
            };
            Some(
                std::iter::once(finding)
                    .chain(listed)
                    .chain(lock_rules)
                    .collect(),
            )
        }
    };

    let outcome = catch_unwind(AssertUnwindSafe(|| match mode {
        "document" => run_document(case),
        "text" => run_text(case),
        "lock" => run_lock(case),
        "start" => run_start(case),
        "definition" => run_definition(case),
        "called" => run_called(case),
        "run" => run_run(case),
        _ => run_graph(case, case_text),
    }));
    let outcome = match outcome {
        Ok(outcome) => outcome,
        Err(panic) => {
            return Verdict::Disagreed(format!("the loader panicked: {}", panic_text(panic)));
        }
    };
    let expected_text = expected.unwrap_or("accepted");
    match (outcome, findings) {
        (Err(Stop::Broken(text)), _) => {
            Verdict::Disagreed(format!("expected {expected_text}: {text}"))
        }
        (Err(Stop::Refused(error)), None) => Verdict::Disagreed(format!(
            "expected {expected_text}, the loader refused: {error}"
        )),
        (Err(Stop::Refused(error)), Some(findings)) => {
            let reasons: Vec<String> = findings
                .iter()
                .map_while(|finding| does_not_answer(mode, finding, &error))
                .collect();
            if reasons.len() < findings.len() {
                Verdict::Agreed
            } else {
                Verdict::Disagreed(format!(
                    "expected {expected_text}, {}, the loader refused: {error}",
                    reasons[0]
                ))
            }
        }
        (Ok(_), Some(_)) => {
            Verdict::Disagreed(format!("expected {expected_text}, the loader accepted"))
        }
        (Ok(mut accepted), None) => {
            // Nothing was refused, so a case that expects no refusal can be compared on that.
            accepted
                .result
                .entry("refused".to_string())
                .or_insert_with(|| json!([]));
            let must_stop = expected == Some("run.not_started");
            if accepted.stops != must_stop {
                return Verdict::Disagreed(format!(
                    "expected {expected_text}, the start decision is `starts: {}`",
                    !accepted.stops
                ));
            }
            let wrong = case
                .get("expect")
                .and_then(Value::as_object)
                .and_then(|expect| compare(expect, &accepted.result, "expect", uncompared));
            match wrong {
                Some(place) => {
                    Verdict::Disagreed(format!("the result differs from the case at {place}"))
                }
                None => Verdict::Agreed,
            }
        }
    }
}

fn read_json(path: &Path) -> (String, Value) {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{} cannot be read: {error}", path.display()));
    let value = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{} is not JSON: {error}", path.display()));
    (text, value)
}

fn run_manifest(root: &Path) -> Summary {
    let (_, manifest) = read_json(&root.join("manifest.json"));
    let records = manifest["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("the manifest under {} has no cases", root.display()));
    let mut summary = Summary {
        cases: records.len(),
        ..Summary::default()
    };
    for record in records {
        let relative = record["path"].as_str().expect("a manifest line has a path");
        assert!(plain_relative(relative), "manifest path `{relative}`");
        let expected = record["error"].as_str();
        let (case_text, case) = read_json(&root.join(relative));
        let name = Path::new(relative)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or(relative);
        let mode = case["mode"].as_str().unwrap_or("unknown").to_string();
        let mut uncompared = Vec::new();
        match judge(name, &case, &case_text, expected, &mut uncompared) {
            Verdict::NotRun(reason) => *summary.not_run.entry(reason).or_default() += 1,
            verdict => {
                let counts = summary.by_mode.entry(mode.clone()).or_default();
                counts.0 += 1;
                match verdict {
                    Verdict::Disagreed(text) => {
                        summary
                            .disagreements
                            .push(format!("{relative} [{mode}]: {text}"));
                    }
                    _ => counts.1 += 1,
                }
            }
        }
        for key in uncompared {
            *summary
                .uncompared
                .entry(format!("{mode}: {key}"))
                .or_default() += 1;
        }
    }
    summary
}

#[test]
fn conformance_cases_agree() {
    let (root, vendored) = match std::env::var_os(CONFORMANCE_ROOT_VARIABLE) {
        Some(root) => (PathBuf::from(root), false),
        None => (
            repository_root().join("tests/fixtures/agent_packages/conformance"),
            true,
        ),
    };
    let summary = run_manifest(&root);
    summary.print(&root);
    assert!(
        summary.run() > 0,
        "no case under {} was run",
        root.display()
    );
    assert!(
        summary.disagreements.is_empty(),
        "{} of {} cases disagree:\n{}",
        summary.disagreements.len(),
        summary.run(),
        summary.disagreements.join("\n")
    );
    if vendored {
        // The vendored subset holds only cases the runner runs, and the cases of the mapping test.
        let not_run: Vec<&str> = summary.not_run.keys().map(String::as_str).collect();
        assert_eq!(not_run, [NOT_RUN_COMPATIBILITY], "cases that were not run");
    }
}

#[test]
fn stored_json_file_has_the_bytes_its_case_pins() {
    let root = repository_root().join("tests/fixtures/agent_packages/conformance");
    let (text, case) = read_json(&root.join("cases/graph-valid.json"));
    let stored = Stored::parse(&text).expect("the case is JSON");
    let file = stored
        .get("inventory")
        .and_then(|inventory| inventory.get("packages"))
        .and_then(|packages| packages.get("root"))
        .and_then(|package| package.get("files"))
        .and_then(|files| files.get("agent.json"))
        .expect("the case stores agent.json");
    let pinned = case["lock"]["entries"]
        .as_array()
        .and_then(|entries| entries.iter().find(|entry| entry["address"] == "$"))
        .and_then(|entry| entry["digest"].as_str())
        .expect("the lock pins its root");
    assert_eq!(bytes_digest(&file.file_bytes()), pinned);
}

#[test]
fn stored_json_writer_matches_the_checker_encoding() {
    let stored = Stored::parse(
        r#"{"b": [], "a": {}, "text": "café 😀 \"q\" \\ \n\u0001", "n": [1, -0.5, 1e-05, true, null]}"#,
    )
    .expect("the text is JSON");
    let written = String::from_utf8(stored.file_bytes()).expect("the bytes are UTF-8");
    let expected = "{\n  \"b\": [],\n  \"a\": {},\n  \"text\": \"caf\u{e9} \u{1f600} \\\"q\\\" \\\\ \\n\\u0001\",\n  \"n\": [\n    1,\n    -0.5,\n    1e-05,\n    true,\n    null\n  ]\n}\n";
    assert_eq!(written, expected);
    assert!(Stored::parse(r#"{"lone": "\ud800"}"#).is_err());
}

#[test]
fn every_finding_of_the_table_names_known_codes() {
    for finding in [
        "schema:/ required:",
        "json.duplicate_key",
        "document.type",
        "document.object",
        "document.version",
        "document.tool_connector",
        "definition.tool_connector:units.convert",
        "document.variable_undeclared:x",
        "document.variable_type:x",
        "document.variable_default:x",
        "variable.missing:x",
        "variable.undeclared:x",
        "variable.type:x",
        "variable.enum:x",
        "variable.value",
        "reference.variable_enum:x",
        "definition.variable_type:x",
        "definition.name_in_use:skills.review",
        "definition.agent_type",
        "called.type:helper",
        "reference.type",
        "called.description:helper",
        "reference.agent_description",
        "lock.digest_binding",
        "lock.exact_closure",
        "reference.unresolved",
        "reference.missing",
        "reference.version_selection",
        "lock.source_binding",
        "lock.constraint_binding",
        "lock.cycle",
        "reference.connector_entrypoint",
        "reference.hook_entrypoint",
        "reference.package_boundary",
        "reference.cycle",
    ] {
        let codes =
            code_class("graph", finding).unwrap_or_else(|| panic!("`{finding}` has no class"));
        assert!(!codes.is_empty(), "{finding}");
        for code in codes {
            assert!(Code::ALL.contains(code), "{finding}");
        }
    }
    assert_eq!(
        code_class("lock", "schema:/entries minItems:"),
        Some(&[Code::LockInvalid, Code::SchemaInvalid][..])
    );
    assert_eq!(code_class("graph", "support.required_feature:hooks"), None);
    assert_eq!(finding_name("variable.undeclared:tone"), Some("tone"));
    assert_eq!(
        finding_name("definition.name_in_use:skills.review"),
        Some("skills.review")
    );
    assert_eq!(finding_name("called.type:helper"), None);
    assert_eq!(finding_name("schema:/ required:"), None);
    let lock = |entries: Value| json!({"entries": entries});
    assert!(
        lock_own_findings(&lock(json!([
            {"address": "$", "dependencies": ["$/skills/a"]},
            {"address": "$/skills/a", "dependencies": []},
        ])))
        .is_empty()
    );
    assert_eq!(
        lock_own_findings(&lock(
            json!([{"address": "$", "dependencies": ["$/model"]}])
        )),
        ["lock.dangling"]
    );
    assert_eq!(
        lock_own_findings(&lock(json!([
            {"address": "$", "dependencies": []},
            {"address": "$/skills/a", "dependencies": []},
        ]))),
        ["lock.unreachable"]
    );
    assert_eq!(
        lock_own_findings(&lock(json!([
            {"address": "$", "dependencies": []},
            {"address": "$", "dependencies": []},
        ]))),
        ["lock.address_unique"]
    );
    let named = PackageError::new(Code::VariableUndeclared, "tone", "no variable of that name");
    assert_eq!(
        does_not_answer("start", "variable.undeclared:tone", &named),
        None
    );
    assert!(does_not_answer("start", "variable.undeclared:other", &named).is_some());
    assert!(does_not_answer("start", "variable.type:tone", &named).is_some());
    assert_eq!(code_class("run", "run.not_started"), None);
}

#[test]
fn a_result_is_compared_the_way_the_cases_expect() {
    // An object may be a part of the actual one. A list matches item for item.
    let actual =
        json!({"agent": {"name": "a", "tools": [{"builtin": "read"}], "limits": {"max_turns": 5}}});
    assert_eq!(
        mismatch(&json!({"agent": {"name": "a"}}), &actual, "expect"),
        None
    );
    assert_eq!(
        mismatch(
            &json!({"agent": {"limits": {"max_turns": 5.0}}}),
            &actual,
            "expect"
        ),
        None
    );
    assert_eq!(
        mismatch(
            &json!({"agent": {"limits": {"max_turns": 6}}}),
            &actual,
            "expect"
        ),
        Some("expect.agent.limits.max_turns".to_string())
    );
    assert_eq!(
        mismatch(&json!({"agent": {"tools": []}}), &actual, "expect"),
        Some("expect.agent.tools".to_string())
    );
    assert_eq!(
        mismatch(&json!({"agent": {"model": {}}}), &actual, "expect"),
        Some("expect.agent.model".to_string())
    );
    assert!(mismatch(&json!(true), &json!(1), "expect").is_some());

    // Names are compared without their order. Lines are matched by their need.
    assert_eq!(
        names_mismatch(&json!(["b", "a"]), &json!(["a", "b"]), "added"),
        None
    );
    assert!(names_mismatch(&json!(["a"]), &json!(["a", "b"]), "added").is_some());
    let lines = json!([
        {"need": "runtime.compute.memory", "status": "not met", "optional": false},
        {"need": "runtime.network", "status": "met here", "optional": false},
    ]);
    assert_eq!(
        lines_mismatch(
            &json!([{"need": "runtime.network", "status": "met here"}, {"need": "runtime.compute.memory"}]),
            &lines,
            "lines"
        ),
        None
    );
    assert_eq!(
        lines_mismatch(
            &json!([{"need": "runtime.network", "status": "not met"}, {"need": "runtime.compute.memory"}]),
            &lines,
            "lines"
        ),
        Some("lines[runtime.network].status".to_string())
    );
    assert!(lines_mismatch(&json!([{"need": "runtime.network"}]), &lines, "lines").is_some());

    // A key the result does not hold is listed, not compared. A key of `run` is compared inside it.
    let result = json!({"added": ["skills.a"], "run": {"starts": false, "statuses": {"runtime.network": "not met"}}});
    let expect = json!({
        "added": ["skills.a"],
        "reported": [{"id": "x"}],
        "run": {"starts": false, "statuses": {"runtime.network": "not met"}, "reported": []},
    });
    let mut uncompared = Vec::new();
    let wrong = compare(
        expect.as_object().expect("an object"),
        result.as_object().expect("an object"),
        "expect",
        &mut uncompared,
    );
    assert_eq!(wrong, None);
    assert_eq!(uncompared, ["reported", "run.reported"]);
    let expect = json!({"run": {"starts": true}});
    let wrong = compare(
        expect.as_object().expect("an object"),
        result.as_object().expect("an object"),
        "expect",
        &mut uncompared,
    );
    assert_eq!(wrong, Some("expect.run.starts".to_string()));
}

#[test]
fn start_values_cover_only_what_the_author_left_open() {
    let agent = json!({
        "type": "agent",
        "variables": {
            "job": {"type": "string"},
            "size": {"type": "number", "enum": [3, 5]},
            "tone": {"type": "string", "default": "plain"},
            "strict": {"type": "boolean"},
        }
    });
    let definition = json!({
        "type": "agent-definition",
        "agent": {"source": "./../agent.json"},
        "variables": {"job": "Write notes."},
    });
    let stored = |path: &str| match path {
        "agent.json" => Some(agent.clone()),
        "people/agent-definition.json" => Some(definition.clone()),
        _ => None,
    };
    assert_eq!(
        Value::Object(start_values("agent.json", &stored)),
        json!({"job": "value", "size": 3, "strict": true})
    );
    // A definition's own values are left to the definition.
    assert_eq!(
        Value::Object(start_values("people/agent-definition.json", &stored)),
        json!({"size": 3, "strict": true})
    );
    assert!(start_values("missing.json", &stored).is_empty());
    assert_eq!(join_text("a/b", "../c"), Some("a/c".to_string()));
    assert_eq!(join_text("", "../c"), None);
    assert!(plain_relative("a/b.json") && !plain_relative("../a") && !plain_relative("/a"));
}

// ---------------------------------------------------------------------------------------------
// Example packages.

fn find_locks(folder: &Path, found: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(folder)
        .unwrap_or_else(|error| panic!("{} cannot be listed: {error}", folder.display()));
    for entry in entries {
        let entry = entry.expect("a folder entry can be read");
        let kind = entry.file_type().expect("a folder entry has a type");
        if kind.is_dir() {
            find_locks(&entry.path(), found);
        } else if kind.is_file() && entry.file_name() == LOCK_FILE {
            found.push(entry.path());
        }
    }
}

fn forward_slashes(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// The integrity each part must have, from what its lock entry says about its source.
fn integrity_problems(reading: &PackageReading) -> Vec<String> {
    let by_address: BTreeMap<&str, Integrity> = reading
        .parts
        .iter()
        .map(|part| (part.address.as_str(), part.integrity))
        .collect();
    let mut problems = Vec::new();
    for part in &reading.parts {
        let expected = match part.source_type {
            SourceType::Local => Integrity::Verified,
            // Nothing is fetched, and the examples carry no fetched content.
            SourceType::Git => Integrity::Absent,
            SourceType::Oci | SourceType::Registry => Integrity::NotVerifiable,
            SourceType::Pointer => Integrity::Pointer,
        };
        if part.integrity == expected {
            continue;
        }
        // An entry below a document that was not read is not checked.
        let below_unread = part.integrity == Integrity::NotChecked
            && by_address.iter().any(|(address, integrity)| {
                part.address.starts_with(&format!("{address}/"))
                    && !matches!(integrity, Integrity::Verified | Integrity::Pointer)
            });
        if !below_unread {
            problems.push(format!(
                "{} is {:?} and its source is {:?}",
                part.address, part.integrity, part.source_type
            ));
        }
    }
    problems
}

#[test]
fn example_packages_load() {
    let root = match std::env::var_os(EXAMPLES_ROOT_VARIABLE) {
        Some(root) => PathBuf::from(root),
        None => repository_root().join("examples/agent_package"),
    };
    let root = std::path::absolute(&root)
        .unwrap_or_else(|error| panic!("{} cannot be opened: {error}", root.display()));
    let mut locks = Vec::new();
    find_locks(&root, &mut locks);
    locks.sort();
    assert!(!locks.is_empty(), "no {LOCK_FILE} under {}", root.display());

    let mut problems = Vec::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut unavailable = Vec::new();
    let mut loaded = 0usize;
    println!("agent package examples: root {}", root.display());
    for lock_path in &locks {
        let (_, lock) = read_json(lock_path);
        let entries = lock["entries"].as_array().expect("a lock has entries");
        let entry_at = |address: &str| entries.iter().find(|entry| entry["address"] == address);
        let root_entry = entry_at("$").expect("a lock has a root entry");
        let url = root_entry["source"]["url"]
            .as_str()
            .and_then(|url| url.strip_prefix("./"))
            .expect("the root entry of a lock is a local file");
        let folder = lock_path
            .parent()
            .and_then(|parent| parent.strip_prefix(&root).ok())
            .expect("a lock is under the root");
        let folder_text = forward_slashes(folder);
        let entry_text =
            join_text(&folder_text, url).expect("the entry document is under the root");
        let label = if folder_text.is_empty() {
            "."
        } else {
            folder_text.as_str()
        };

        // A definition whose agent is fetched: the agent's bytes are not in the package.
        let agent_is_fetched = root_entry["type"] == "agent-definition"
            && entry_at("$/agent").is_some_and(|agent| agent["source"]["type"] != "local");
        let document_at = |path: &str| {
            fs::read(root.join(path))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .filter(Value::is_object)
        };
        let request = ReadRequest {
            entry: Some(PathBuf::from(&entry_text)),
            variables: start_values(&entry_text, &document_at),
            ..ReadRequest::default()
        };
        match catch_unwind(AssertUnwindSafe(|| read_package(&root, &request))) {
            Err(panic) => problems.push(format!(
                "{label}: the loader panicked: {}",
                panic_text(panic)
            )),
            Ok(Err(error)) if agent_is_fetched && error.code == Code::AgentUnavailable => {
                println!("  {label}: refused, {}", error.code.as_str());
                unavailable.push(label.to_string());
            }
            Ok(Err(error)) => problems.push(format!("{label}: refused: {error}")),
            Ok(Ok(_)) if agent_is_fetched => problems.push(format!(
                "{label}: loaded, and the agent the definition points at is not in the package"
            )),
            Ok(Ok(reading)) => {
                if reading.parts.len() != entries.len() {
                    problems.push(format!(
                        "{label}: the lock has {} entries and {} parts were read",
                        entries.len(),
                        reading.parts.len()
                    ));
                }
                for problem in integrity_problems(&reading) {
                    problems.push(format!("{label}: {problem}"));
                }
                let mut here: BTreeMap<String, usize> = BTreeMap::new();
                for part in &reading.parts {
                    let name = serde_json::to_value(part.integrity)
                        .ok()
                        .and_then(|value| value.as_str().map(str::to_string))
                        .unwrap_or_default();
                    *here.entry(name.clone()).or_default() += 1;
                    *counts.entry(name).or_default() += 1;
                }
                loaded += 1;
                println!("  {label}: loaded, {here:?}");
            }
        }
    }
    println!(
        "  lock roots {}  loaded {loaded}  agent not in the package {}  problems {}  entries {counts:?}",
        locks.len(),
        unavailable.len(),
        problems.len()
    );
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
