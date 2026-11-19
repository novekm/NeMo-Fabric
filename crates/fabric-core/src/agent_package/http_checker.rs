// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ask a checker over HTTP whether a loaded package may start.
//!
//! The loader posts plain facts about the agent, gathered from the load
//! report, to the URL the caller named: the agent's name, each part with its
//! type, source, version, digest and whether the source is local, the models
//! named, and the policies with the enforcers each needs. The body is the
//! format-neutral request of the NeMo Guardrails hooks server's
//! `/v1/hooks/agent-start` endpoint; any server that answers it the same way
//! is a checker. The checker reads no package.
//!
//! Only `http` is spoken, with the standard library, so the loader takes no
//! new dependency: a checker on another host is reached through a local
//! address such as a sidecar.

use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use super::model::{EntryType, Integrity, SourceType};
use super::report::{AgentPackageReport, CheckerDecision, CheckerFailure, CheckerResult};

/// The most of a response the loader reads, as for a checker program's output.
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
/// The most of a response's text the report keeps, in characters.
const MAX_DETAIL_CHARS: usize = 2000;
/// The event name of the request.
const EVENT: &str = "AgentStart";

/// Where an HTTP checker listens: an `http` URL with a host, an optional
/// port and an optional path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Endpoint {
    url: String,
    host: String,
    port: u16,
    path: String,
}

impl Endpoint {
    /// The endpoint `url` names, or why it names none. Only `http` is
    /// spoken; a URL with a user name or password is refused, so no
    /// credential is written into the options.
    pub(super) fn parse(url: &str) -> Result<Self, &'static str> {
        if url.chars().any(|c| c.is_control() || c.is_whitespace()) {
            return Err("the checker URL holds a space or a control character");
        }
        let Some(rest) = url.strip_prefix("http://") else {
            return Err(
                "the checker URL must start with http://. https is not spoken: reach a remote checker through a local address",
            );
        };
        let (authority, path) = match rest.find(['/', '?', '#']) {
            Some(at) => (&rest[..at], &rest[at..]),
            None => (rest, "/"),
        };
        if path.starts_with('#') || path.contains('#') {
            return Err("the checker URL must not have a fragment");
        }
        let path = if path.starts_with('?') {
            format!("/{path}")
        } else {
            path.to_string()
        };
        if authority.contains('@') {
            return Err("the checker URL must not hold a user name or password");
        }
        let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
            let Some((host, after)) = bracketed.split_once(']') else {
                return Err("the checker URL has an unclosed [");
            };
            (host, after.strip_prefix(':'))
        } else {
            match authority.rsplit_once(':') {
                Some((host, port)) => (host, Some(port)),
                None => (authority, None),
            }
        };
        if host.is_empty() {
            return Err("the checker URL has no host");
        }
        let port = match port {
            None => 80,
            Some(port) => port
                .parse::<u16>()
                .ok()
                .filter(|port| *port != 0)
                .ok_or("the checker URL has a port that is not a number from 1 to 65535")?,
        };
        Ok(Self {
            url: url.to_string(),
            host: host.to_string(),
            port,
            path,
        })
    }

    /// The `Host` header: the host as the URL writes it, and its port.
    fn host_header(&self) -> String {
        if self.host.contains(':') {
            format!("[{}]:{}", self.host, self.port)
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

/// The facts the loader sends about a load that is ready to start.
///
/// A part is every lock entry below the entry document, called agents and
/// their parts included, except a definition document and a pinned model.
/// A pinned model is named under `models` with the default model of the
/// agent. A policy needs the enforcers its folder has a folder for, such as
/// `openshell/`. `enforcers_present` is what the caller says runs here.
pub(super) fn start_facts(report: &AgentPackageReport, enforcers_present: &[String]) -> Value {
    start_facts_with(report, enforcers_present, None)
}

/// `start_facts` with the `registration` object the platform supplied, passed through as it was given.
pub(super) fn start_facts_with(
    report: &AgentPackageReport,
    enforcers_present: &[String],
    registration: Option<&Map<String, Value>>,
) -> Value {
    let mut facts = base_facts(report, enforcers_present);
    if let (Some(registration), Some(object)) = (registration, facts.as_object_mut()) {
        object.insert(
            "registration".to_string(),
            Value::Object(registration.clone()),
        );
    }
    facts
}

fn base_facts(report: &AgentPackageReport, enforcers_present: &[String]) -> Value {
    let mut parts = Vec::new();
    let mut models: Vec<String> = Vec::new();
    let mut policies = Vec::new();
    let mut add_model = |model: &str| {
        if !model.is_empty() && !models.iter().any(|known| known == model) {
            models.push(model.to_string());
        }
    };
    if let Some(model) = &report.model.declared_source {
        add_model(model);
    }
    for entry in &report.digests {
        if entry.address == report.entry.address {
            continue;
        }
        let kind = match entry.entry_type {
            EntryType::Skill => "skill",
            EntryType::Plugin => "plugin",
            EntryType::Connector => "connector",
            EntryType::Hook => "hook",
            EntryType::Package => "package",
            EntryType::Policy => "policy",
            EntryType::Agent => "agent",
            EntryType::AgentDefinition => continue,
            EntryType::Model => {
                if let Some(model) = &entry.source_url {
                    add_model(model);
                }
                continue;
            }
        };
        parts.push(json!({
            "id": entry.address,
            "type": kind,
            "source": source_text(entry.source_type, entry.source_url.as_deref(), entry.source_subpath.as_deref()),
            "version": entry.version,
            "digest": entry.digest,
            "local": entry.source_type == SourceType::Local,
        }));
        if entry.entry_type == EntryType::Policy {
            let required = report
                .declared
                .iter()
                .find(|item| item.address.as_deref() == Some(entry.address.as_str()))
                .is_none_or(|item| item.required);
            let enforcers = match (&entry.result, &entry.path) {
                (Integrity::Verified, Some(path)) => folders(&report.package_dir.join(path)),
                _ => Vec::new(),
            };
            policies
                .push(json!({"id": entry.address, "required": required, "enforcers": enforcers}));
        }
    }
    // The models the package declares, then the model the config binds the run to: the one that
    // will really run. Both are in `models`, so a rule judges each; the two members say which is which.
    let declared_models = models.clone();
    let bound_model = bound_model(report);
    if let Some(bound) = &bound_model
        && !models.iter().any(|known| known == bound)
    {
        models.push(bound.clone());
    }
    json!({
        "hook_event_name": EVENT,
        "agent_name": report.entry.name,
        // The package's identity, with the name: a quarantine holds this digest, and a new version has another.
        "agent_digest": format!("sha256:{}", report.lock.sha256),
        "parts": parts,
        "models": models,
        "declared_models": declared_models,
        "bound_model": bound_model,
        "policies": policies,
        "enforcers_present": enforcers_present,
    })
}

/// The model the config binds the run to, as `<provider>::<model>`, the way a package names a
/// model. None when the config names no model and the harness picks its own.
fn bound_model(report: &AgentPackageReport) -> Option<String> {
    let model = report
        .model
        .model
        .as_deref()
        .filter(|model| !model.is_empty())?;
    Some(
        match report
            .model
            .provider
            .as_deref()
            .filter(|provider| !provider.is_empty())
        {
            Some(provider) if !model.starts_with(&format!("{provider}::")) => {
                format!("{provider}::{model}")
            }
            _ => model.to_string(),
        },
    )
}

/// A source as the package states it: `git::<url>//<subpath>` for a git
/// repository, the path itself for a local folder.
fn source_text(kind: SourceType, url: Option<&str>, subpath: Option<&str>) -> String {
    let url = url.unwrap_or_default();
    let base = match kind {
        SourceType::Local | SourceType::Pointer => return url.to_string(),
        SourceType::Git => format!("git::{url}"),
        SourceType::Oci => format!("oci::{url}"),
        SourceType::Registry => format!("registry::{url}"),
    };
    match subpath {
        Some(subpath) if !subpath.is_empty() => format!("{base}//{subpath}"),
        _ => base,
    }
}

/// The names of the folders directly in `folder`, sorted; none when it cannot be read.
fn folders(folder: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(folder)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| !name.starts_with('.'))
        .collect();
    names.sort();
    names
}

/// Ask the checker at `endpoint` about `report`. It never gives an error: a
/// checker that cannot be reached, does not answer in time, or does not
/// answer as it must is a deny with the failure named.
pub(super) fn ask(
    endpoint: &Endpoint,
    report: &AgentPackageReport,
    enforcers_present: &[String],
    registration: Option<&Map<String, Value>>,
    limit: Duration,
) -> CheckerResult {
    let failed = |failure, detail: String, answer| CheckerResult {
        decision: CheckerDecision::Deny,
        failure: Some(failure),
        exit_code: None,
        detail: Some(detail),
        answer,
    };
    let body = start_facts_with(report, enforcers_present, registration).to_string();
    let (status, response) = match post(endpoint, body.as_bytes(), limit) {
        Ok(answered) => answered,
        Err((failure, detail)) => return failed(failure, detail, None),
    };
    let answer = serde_json::from_slice::<Value>(&response)
        .ok()
        .and_then(|answer| match answer {
            Value::Object(answer) => Some(answer),
            _ => None,
        });
    if status != 200 {
        let text = String::from_utf8_lossy(&response);
        let text: String = text.trim().chars().take(MAX_DETAIL_CHARS).collect();
        let detail = format!(
            "the checker at {} answered HTTP {status}: {text}",
            endpoint.url
        );
        return failed(CheckerFailure::UnexpectedStatus, detail, answer);
    }
    let decision = match answer_decision(answer.as_ref()) {
        Some(decision) => decision,
        None => {
            let detail = format!(
                "the checker at {} answered with no decision of allow or deny",
                endpoint.url
            );
            return failed(CheckerFailure::InvalidAnswer, detail, answer);
        }
    };
    CheckerResult {
        decision,
        failure: None,
        exit_code: None,
        detail: None,
        answer,
    }
}

fn answer_decision(answer: Option<&Map<String, Value>>) -> Option<CheckerDecision> {
    match answer?.get("decision")?.as_str()? {
        "allow" => Some(CheckerDecision::Allow),
        "deny" => Some(CheckerDecision::Deny),
        _ => None,
    }
}

/// Post `body` to `endpoint` and read the status and body of the answer,
/// all within `limit`.
fn post(
    endpoint: &Endpoint,
    body: &[u8],
    limit: Duration,
) -> Result<(u16, Vec<u8>), (CheckerFailure, String)> {
    let started = Instant::now();
    let left = || {
        limit
            .checked_sub(started.elapsed())
            .filter(|left| !left.is_zero())
    };
    let unreachable = |error: std::io::Error| {
        let detail = format!("the checker at {} cannot be reached: {error}", endpoint.url);
        (CheckerFailure::Unreachable, detail)
    };
    let timed_out = || {
        let detail = format!("the checker at {} did not answer in time", endpoint.url);
        (CheckerFailure::TimedOut, detail)
    };
    let addresses = (endpoint.host.as_str(), endpoint.port)
        .to_socket_addrs()
        .map_err(unreachable)?;
    let mut last = std::io::Error::new(ErrorKind::NotFound, "the host has no address");
    let mut stream = None;
    for address in addresses {
        let Some(left) = left() else {
            return Err(timed_out());
        };
        match TcpStream::connect_timeout(&address, left) {
            Ok(connected) => {
                stream = Some(connected);
                break;
            }
            Err(error) if error.kind() == ErrorKind::TimedOut => return Err(timed_out()),
            Err(error) => last = error,
        }
    }
    let mut stream = stream.ok_or_else(|| unreachable(last))?;
    let mut request = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        endpoint.path,
        endpoint.host_header(),
        body.len()
    )
    .into_bytes();
    request.extend_from_slice(body);
    stream
        .set_write_timeout(Some(left().ok_or_else(timed_out)?))
        .map_err(unreachable)?;
    stream
        .write_all(&request)
        .map_err(|error| match error.kind() {
            ErrorKind::TimedOut | ErrorKind::WouldBlock => timed_out(),
            _ => unreachable(error),
        })?;
    // Headers, then a body of at most the limit.
    let most = MAX_RESPONSE_BYTES + 64 * 1024;
    let mut response = Vec::new();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        stream
            .set_read_timeout(Some(left().ok_or_else(timed_out)?))
            .map_err(unreachable)?;
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => response.extend_from_slice(&buffer[..read]),
            Err(error) if matches!(error.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) => {
                return Err(timed_out());
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(unreachable(error)),
        }
        if response.len() > most {
            let detail = format!(
                "the checker at {} answered with more than {MAX_RESPONSE_BYTES} bytes",
                endpoint.url
            );
            return Err((CheckerFailure::OutputTooLarge, detail));
        }
    }
    parse_response(&response).ok_or_else(|| {
        let detail = format!("the checker at {} did not answer in HTTP/1.1", endpoint.url);
        (CheckerFailure::InvalidAnswer, detail)
    })
}

/// The status and body of one HTTP/1.x response. A chunked body is joined.
fn parse_response(response: &[u8]) -> Option<(u16, Vec<u8>)> {
    let end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&response[..end]).ok()?;
    let body = &response[end + 4..];
    let mut lines = head.split("\r\n");
    let status_line = lines.next()?;
    let mut words = status_line.split(' ');
    if !words.next()?.starts_with("HTTP/1.") {
        return None;
    }
    let status = words.next()?.parse::<u16>().ok()?;
    let mut chunked = false;
    let mut length = None;
    for line in lines {
        let (name, value) = line.split_once(':')?;
        let value = value.trim();
        if name.eq_ignore_ascii_case("transfer-encoding") {
            chunked = value.eq_ignore_ascii_case("chunked");
        } else if name.eq_ignore_ascii_case("content-length") {
            length = Some(value.parse::<usize>().ok()?);
        }
    }
    if chunked {
        return dechunk(body).map(|body| (status, body));
    }
    match length {
        Some(length) if length <= body.len() => Some((status, body[..length].to_vec())),
        Some(_) => None,
        None => Some((status, body.to_vec())),
    }
}

fn dechunk(mut body: &[u8]) -> Option<Vec<u8>> {
    let mut joined = Vec::new();
    loop {
        let end = body.windows(2).position(|window| window == b"\r\n")?;
        let size = std::str::from_utf8(&body[..end]).ok()?;
        let size = usize::from_str_radix(size.split(';').next()?.trim(), 16).ok()?;
        body = &body[end + 2..];
        if size == 0 {
            return Some(joined);
        }
        joined.extend_from_slice(body.get(..size)?);
        body = body.get(size + 2..)?;
    }
}
