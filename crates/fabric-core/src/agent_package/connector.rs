// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connectors carried in a package: read `mcp.json` and build stdio server entries.
//!
//! `McpServerConfig` has no working-directory field, and a carried server is
//! written to start from its own folder. So the entry starts `/bin/sh` with a
//! constant script that changes directory and then replaces itself with the
//! server command. The folder, the command and its arguments are passed as
//! separate arguments and are never parsed by the shell.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::digest::bytes_digest;
use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::json::{MAX_DOCUMENT_BYTES, parse_document, read_regular_file};
use super::model::{FolderListing, LockedPart};
use super::paths::resolve_inside;
use super::report::MappingGap;
use crate::config::{McpExposure, McpServerConfig, McpTransport};

/// The file in a connector's folder that says how to start it.
const ENTRY_FILE: &str = "mcp.json";
/// The shell that starts a carried server from its folder.
const SHELL: &str = "/bin/sh";
/// `$1` is the working directory. The rest is the command and its arguments.
const START_SCRIPT: &str = "cd \"$1\" && shift && exec \"$@\"";
/// The keys of one server that the loader reads.
const SERVER_KEYS: [&str; 5] = ["type", "command", "args", "env", "cwd"];

/// One stdio server from a carried connector's `mcp.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CarriedServer {
    pub key: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub working_dir: PathBuf,
    /// The working directory as a path inside the connector's folder. Empty
    /// for the folder itself.
    pub working_dir_shown: String,
    /// Keys of the server entry that the loader does not read, in sorted order.
    pub keys_not_read: Vec<String>,
}

/// What reading a carried connector gave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CarriedConnector {
    Servers(Vec<CarriedServer>),
    NotMapped { gap: MappingGap, detail: String },
}

/// Read the `mcp.json` of a verified connector folder.
///
/// The bytes that are parsed are compared with the digest in the verified
/// folder listing, so the file that is parsed is the file that was hashed.
/// A file the loader cannot read in the expected shape makes the connector
/// not mapped. It is not a refusal of the package.
pub(super) fn read_carried_connector(
    part: &LockedPart,
    listing: &FolderListing,
) -> PackageResult<CarriedConnector> {
    let subject = part.address.as_str();
    let connector = part.name.as_deref().unwrap_or_default();
    let Some(folder) = part.path.as_deref() else {
        return Ok(not_mapped(
            MappingGap::ContentAbsent,
            "the connector's folder is not in the package",
        ));
    };
    let Some(listed) = listing.files.get(ENTRY_FILE) else {
        return Err(PackageError::new(
            PackageErrorCode::PartEntrypoint,
            subject,
            format!("the connector's folder has no {ENTRY_FILE}"),
        ));
    };
    let bytes = read_regular_file(&folder.join(ENTRY_FILE), subject, MAX_DOCUMENT_BYTES)?;
    let read = bytes_digest(&bytes);
    if read != *listed {
        return Err(PackageError::new(
            PackageErrorCode::DigestMismatch,
            subject,
            format!(
                "{ENTRY_FILE} changed after the connector was checked: checked {listed}, read {read}"
            ),
        ));
    }
    let document = match parse_document(&bytes, subject) {
        Ok(document) => document,
        Err(error) => {
            return Ok(not_mapped(
                MappingGap::UnsupportedShape,
                format!("{ENTRY_FILE} could not be read: {}", error.message),
            ));
        }
    };
    Ok(servers_from(&document, connector, folder, subject))
}

/// The servers a parsed `mcp.json` describes, or why the connector is not mapped.
fn servers_from(
    document: &Value,
    connector: &str,
    folder: &Path,
    subject: &str,
) -> CarriedConnector {
    let Some(servers) = document
        .get("mcpServers")
        .and_then(Value::as_object)
        .filter(|servers| !servers.is_empty())
    else {
        return not_mapped(
            MappingGap::UnsupportedShape,
            format!("{ENTRY_FILE} has no `mcpServers` object with at least one server"),
        );
    };
    let single = servers.len() == 1;
    let mut carried = Vec::new();
    for (name, server) in servers {
        let unsupported = |why: &str| {
            not_mapped(
                MappingGap::UnsupportedShape,
                format!("server `{name}` in {ENTRY_FILE} {why}"),
            )
        };
        let Some(server) = server.as_object() else {
            return unsupported("is not an object");
        };
        // A server with a command and no `type` is a stdio server.
        match server.get("type") {
            None => {}
            Some(Value::String(kind)) if kind == "stdio" => {}
            Some(_) => return unsupported("is not a stdio server"),
        }
        let Some(command) = server
            .get("command")
            .and_then(Value::as_str)
            .filter(|command| !command.trim().is_empty())
        else {
            return unsupported("has no `command`");
        };
        let args = match server.get("args") {
            None => Vec::new(),
            Some(Value::Array(args)) => {
                let texts: Vec<String> = args
                    .iter()
                    .filter_map(|arg| arg.as_str().map(str::to_string))
                    .collect();
                if texts.len() != args.len() {
                    return unsupported("has an `args` member that is not text");
                }
                texts
            }
            Some(_) => return unsupported("has `args` that is not a list"),
        };
        let env = match server.get("env") {
            None => BTreeMap::new(),
            Some(Value::Object(env)) => {
                let texts: BTreeMap<String, String> = env
                    .iter()
                    .filter_map(|(key, value)| Some((key.clone(), value.as_str()?.to_string())))
                    .collect();
                if texts.len() != env.len() {
                    return unsupported("has an `env` value that is not text");
                }
                texts
            }
            Some(_) => return unsupported("has `env` that is not an object"),
        };
        let working_dir = match server.get("cwd") {
            None => folder.to_path_buf(),
            // `.`, `./` and an empty value name the connector's folder itself.
            Some(Value::String(cwd)) if names_the_folder(cwd) => folder.to_path_buf(),
            Some(Value::String(cwd)) => match resolve_inside(folder, folder, cwd, subject) {
                Ok(path) if path.is_dir() => path,
                Ok(_) => return unsupported("has a `cwd` that is not a folder in the connector"),
                Err(error) => {
                    return unsupported(&format!(
                        "has a `cwd` that cannot be used: {}",
                        error.message
                    ));
                }
            },
            Some(_) => return unsupported("has a `cwd` that is not text"),
        };
        let working_dir_shown = working_dir
            .strip_prefix(folder)
            .map(|below| below.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut keys_not_read: Vec<String> = server
            .keys()
            .filter(|key| !SERVER_KEYS.contains(&key.as_str()))
            .cloned()
            .collect();
        keys_not_read.sort();
        carried.push(CarriedServer {
            key: server_key(connector, name, single),
            command: command.to_string(),
            args,
            env,
            working_dir,
            working_dir_shown,
            keys_not_read,
        });
    }
    CarriedConnector::Servers(carried)
}

/// The one place that builds the stdio entry. None on a platform without `/bin/sh`,
/// or when the working directory is not text.
pub(super) fn stdio_server(server: &CarriedServer) -> Option<McpServerConfig> {
    if !cfg!(unix) || !Path::new(SHELL).is_file() {
        return None;
    }
    let working_dir = server.working_dir.to_str()?;
    let mut args = vec![
        "-c".to_string(),
        START_SCRIPT.to_string(),
        "sh".to_string(),
        working_dir.to_string(),
        server.command.clone(),
    ];
    args.extend(server.args.iter().cloned());
    Some(McpServerConfig {
        transport: McpTransport::Stdio,
        url: SHELL.to_string(),
        args,
        env: server.env.clone(),
        authentication: None,
        exposure: McpExposure::HarnessNative,
        allowed_tools: None,
        blocked_tools: Vec::new(),
        extensions: BTreeMap::new(),
        custom_headers: BTreeMap::new(),
    })
}

/// A relative `cwd` made only of `.` and empty segments, such as `.`, `./` or the empty text.
fn names_the_folder(cwd: &str) -> bool {
    !cwd.starts_with('/')
        && !cwd.contains(['\\', '\0'])
        && cwd
            .split('/')
            .all(|segment| segment.is_empty() || segment == ".")
}

/// The connector's name when its file has one server, else `<connector>-<server>`.
pub(super) fn server_key(connector: &str, server: &str, single: bool) -> String {
    if single {
        connector.to_string()
    } else {
        format!("{connector}-{server}")
    }
}

fn not_mapped(gap: MappingGap, detail: impl Into<String>) -> CarriedConnector {
    CarriedConnector::NotMapped {
        gap,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn temp_folder(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos();
        let folder = std::env::temp_dir().join(format!(
            "nemo-fabric-agent-package-connector-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&folder).expect("temporary folder should be created");
        folder
            .canonicalize()
            .expect("temporary folder should resolve")
    }

    fn gap_of(connector: &CarriedConnector) -> MappingGap {
        match connector {
            CarriedConnector::NotMapped { gap, .. } => *gap,
            CarriedConnector::Servers(servers) => panic!("expected a gap, got {servers:?}"),
        }
    }

    #[test]
    fn one_server_takes_the_connector_name_and_starts_in_its_folder() {
        let folder = temp_folder("one");
        let document = json!({
            "mcpServers": {
                "render-queue": {
                    "type": "stdio",
                    "command": "uv",
                    "args": ["run", "server.py"],
                    "env": {"RENDER_CONCURRENCY": "2"}
                }
            }
        });

        let connector = servers_from(&document, "queue", &folder, "$/connectors/queue");

        assert_eq!(
            connector,
            CarriedConnector::Servers(vec![CarriedServer {
                key: "queue".to_string(),
                command: "uv".to_string(),
                args: vec!["run".to_string(), "server.py".to_string()],
                env: BTreeMap::from([("RENDER_CONCURRENCY".to_string(), "2".to_string())]),
                working_dir: folder.clone(),
                working_dir_shown: String::new(),
                keys_not_read: Vec::new(),
            }])
        );
        std::fs::remove_dir_all(folder).expect("temporary folder should be removed");
    }

    #[test]
    fn several_servers_get_the_connector_and_server_name() {
        let folder = temp_folder("several");
        let document = json!({
            "mcpServers": {
                "read": {"type": "stdio", "command": "node", "args": ["read.js"]},
                "write": {"command": "node", "args": ["write.js"]}
            }
        });

        let CarriedConnector::Servers(servers) =
            servers_from(&document, "files", &folder, "$/connectors/files")
        else {
            panic!("expected servers");
        };

        assert_eq!(
            servers
                .iter()
                .map(|server| server.key.as_str())
                .collect::<Vec<_>>(),
            vec!["files-read", "files-write"]
        );
        std::fs::remove_dir_all(folder).expect("temporary folder should be removed");
    }

    #[test]
    fn shapes_the_loader_cannot_read_are_not_mapped() {
        let folder = temp_folder("shapes");
        let cases = [
            json!({}),
            json!({"mcpServers": {}}),
            json!({"mcpServers": {"a": "uv run"}}),
            json!({"mcpServers": {"a": {"type": "http", "url": "https://example.com/mcp"}}}),
            json!({"mcpServers": {"a": {"type": "stdio"}}}),
            json!({"mcpServers": {"a": {"type": "stdio", "command": " "}}}),
            json!({"mcpServers": {"a": {"type": "stdio", "command": "uv", "args": "run"}}}),
            json!({"mcpServers": {"a": {"type": "stdio", "command": "uv", "args": [1]}}}),
            json!({"mcpServers": {"a": {"type": "stdio", "command": "uv", "env": {"N": 2}}}}),
            json!({"mcpServers": {"a": {"type": "stdio", "command": "uv", "cwd": 3}}}),
            json!({"mcpServers": {"a": {"type": "stdio", "command": "uv", "cwd": "missing"}}}),
        ];

        for document in cases {
            let connector = servers_from(&document, "c", &folder, "$/connectors/c");

            assert_eq!(
                gap_of(&connector),
                MappingGap::UnsupportedShape,
                "{document}"
            );
        }
        std::fs::remove_dir_all(folder).expect("temporary folder should be removed");
    }

    #[test]
    fn cwd_is_resolved_inside_the_connector_folder() {
        let folder = temp_folder("cwd");
        std::fs::create_dir_all(folder.join("server/bin")).expect("subfolder should be created");
        let with_cwd = |cwd: &str| json!({"mcpServers": {"a": {"type": "stdio", "command": "node", "cwd": cwd}}});

        let inside = servers_from(&with_cwd("server/bin"), "c", &folder, "$/connectors/c");
        let outside = servers_from(&with_cwd("../.."), "c", &folder, "$/connectors/c");
        let absolute = servers_from(&with_cwd("/usr/bin"), "c", &folder, "$/connectors/c");
        let root = servers_from(&with_cwd("/"), "c", &folder, "$/connectors/c");
        let root_twice = servers_from(&with_cwd("//"), "c", &folder, "$/connectors/c");

        let CarriedConnector::Servers(servers) = inside else {
            panic!("expected servers");
        };
        assert_eq!(servers[0].working_dir, folder.join("server/bin"));
        assert_eq!(servers[0].working_dir_shown, "server/bin");
        assert_eq!(gap_of(&outside), MappingGap::UnsupportedShape);
        assert_eq!(gap_of(&absolute), MappingGap::UnsupportedShape);
        assert_eq!(gap_of(&root), MappingGap::UnsupportedShape);
        assert_eq!(gap_of(&root_twice), MappingGap::UnsupportedShape);
        std::fs::remove_dir_all(folder).expect("temporary folder should be removed");
    }

    #[test]
    fn a_cwd_that_names_the_folder_itself_is_the_connector_folder() {
        let folder = temp_folder("cwd-self");
        for cwd in [".", "./", "", "./.", ".//"] {
            let document =
                json!({"mcpServers": {"a": {"type": "stdio", "command": "node", "cwd": cwd}}});

            let CarriedConnector::Servers(servers) =
                servers_from(&document, "c", &folder, "$/connectors/c")
            else {
                panic!("expected servers for cwd `{cwd}`");
            };

            assert_eq!(servers[0].working_dir, folder, "{cwd}");
            assert_eq!(servers[0].working_dir_shown, "", "{cwd}");
        }
        std::fs::remove_dir_all(folder).expect("temporary folder should be removed");
    }

    #[test]
    fn keys_the_loader_does_not_read_are_listed() {
        let folder = temp_folder("keys");
        let document = json!({
            "mcpServers": {
                "a": {
                    "type": "stdio", "command": "node", "args": [], "env": {}, "cwd": ".",
                    "timeout": 30, "disabled": true, "alwaysAllow": ["search"]
                }
            }
        });

        let CarriedConnector::Servers(servers) =
            servers_from(&document, "c", &folder, "$/connectors/c")
        else {
            panic!("expected servers");
        };

        assert_eq!(
            servers[0].keys_not_read,
            vec!["alwaysAllow", "disabled", "timeout"]
        );
        std::fs::remove_dir_all(folder).expect("temporary folder should be removed");
    }

    /// A verified connector part whose `mcp.json` holds the given bytes.
    fn carried_part(folder: &Path, bytes: &[u8]) -> (LockedPart, FolderListing) {
        use crate::agent_package::model::{EntryType, Integrity, PartField, SourceType, StatedBy};

        std::fs::write(folder.join(ENTRY_FILE), bytes).expect("mcp.json should be written");
        let part = LockedPart {
            address: "$/connectors/changelog".to_string(),
            entry_type: EntryType::Connector,
            field: Some(PartField::Connectors),
            name: Some("changelog".to_string()),
            owner: "$".to_string(),
            stated_by: StatedBy::Agent,
            optional: false,
            pointer: Some("/connectors/changelog".to_string()),
            source_text: Some("./connectors/changelog".to_string()),
            version_text: None,
            source_type: SourceType::Local,
            source_url: Some("./connectors/changelog".to_string()),
            source_subpath: None,
            registry: None,
            version: "local".to_string(),
            commit: None,
            digest: None,
            digest_kind: None,
            interface: None,
            retracted: None,
            dependencies: Vec::new(),
            path: Some(folder.to_path_buf()),
            integrity: Integrity::Verified,
            values: None,
        };
        let listing = FolderListing {
            digest: String::new(),
            files: BTreeMap::from([(ENTRY_FILE.to_string(), bytes_digest(bytes))]),
        };
        (part, listing)
    }

    #[test]
    fn the_file_that_is_parsed_is_the_file_that_was_hashed() {
        let folder = temp_folder("read");
        let bytes = br#"{"mcpServers": {"main": {"type": "stdio", "command": "node"}}}"#;
        let (part, mut listing) = carried_part(&folder, bytes);

        let read = read_carried_connector(&part, &listing).expect("the connector reads");

        let CarriedConnector::Servers(servers) = read else {
            panic!("expected servers");
        };
        assert_eq!(servers[0].key, "changelog");
        assert_eq!(servers[0].working_dir, folder);
        // The bytes changed after the folder was checked.
        listing
            .files
            .insert(ENTRY_FILE.to_string(), bytes_digest(b"other"));
        let error = read_carried_connector(&part, &listing).expect_err("a changed file");
        assert_eq!(error.code, PackageErrorCode::DigestMismatch);
        assert_eq!(error.subject, "$/connectors/changelog");
        // A folder listing without the entry file.
        listing.files.clear();
        let error = read_carried_connector(&part, &listing).expect_err("no entry file");
        assert_eq!(error.code, PackageErrorCode::PartEntrypoint);
        std::fs::remove_dir_all(folder).expect("temporary folder should be removed");
    }

    #[test]
    fn an_mcp_json_that_cannot_be_read_leaves_the_connector_unmapped() {
        let folder = temp_folder("unreadable");
        for bytes in [
            b"not json".as_slice(),
            br#"["mcpServers"]"#.as_slice(),
            br#"{"mcpServers": {}, "mcpServers": {}}"#.as_slice(),
        ] {
            let (part, listing) = carried_part(&folder, bytes);

            let read = read_carried_connector(&part, &listing).expect("not a refusal");

            assert_eq!(gap_of(&read), MappingGap::UnsupportedShape);
        }
        std::fs::remove_dir_all(folder).expect("temporary folder should be removed");
    }

    #[test]
    fn server_keys_follow_the_single_server_rule() {
        assert_eq!(server_key("changelog", "main", true), "changelog");
        assert_eq!(server_key("changelog", "main", false), "changelog-main");
    }

    #[cfg(unix)]
    #[test]
    fn the_stdio_entry_passes_folder_command_and_arguments_as_separate_arguments() {
        let server = CarriedServer {
            key: "queue".to_string(),
            command: "uv".to_string(),
            args: vec!["run".to_string(), "server.py; rm -rf /".to_string()],
            env: BTreeMap::from([("A".to_string(), "$HOME".to_string())]),
            working_dir: PathBuf::from("/sandbox/agent/connectors/queue with space"),
            working_dir_shown: String::new(),
            keys_not_read: Vec::new(),
        };

        let entry = stdio_server(&server).expect("this platform has /bin/sh");

        assert_eq!(entry.transport, McpTransport::Stdio);
        assert_eq!(entry.url, "/bin/sh");
        assert_eq!(
            entry.args,
            vec![
                "-c",
                "cd \"$1\" && shift && exec \"$@\"",
                "sh",
                "/sandbox/agent/connectors/queue with space",
                "uv",
                "run",
                "server.py; rm -rf /",
            ]
        );
        assert_eq!(entry.env, server.env);
        assert_eq!(entry.exposure, McpExposure::HarnessNative);
        assert!(entry.authentication.is_none());
        assert!(entry.allowed_tools.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn the_start_script_runs_the_command_in_the_working_directory() {
        let folder = temp_folder("script");
        let server = CarriedServer {
            key: "probe".to_string(),
            command: "/bin/sh".to_string(),
            args: vec![
                "-c".to_string(),
                "pwd; printf '%s\\n' \"$1\"".to_string(),
                "sh".to_string(),
                "a b; echo injected".to_string(),
            ],
            env: BTreeMap::new(),
            working_dir: folder.clone(),
            working_dir_shown: String::new(),
            keys_not_read: Vec::new(),
        };
        let entry = stdio_server(&server).expect("this platform has /bin/sh");

        let output = std::process::Command::new(&entry.url)
            .args(&entry.args)
            .output()
            .expect("the start script should run");

        assert!(output.status.success(), "{output:?}");
        let text = String::from_utf8(output.stdout).expect("text output");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            PathBuf::from(lines[0])
                .canonicalize()
                .expect("printed folder"),
            folder
        );
        assert_eq!(lines[1], "a b; echo injected");
        assert_eq!(lines.len(), 2);
        std::fs::remove_dir_all(folder).expect("temporary folder should be removed");
    }
}
