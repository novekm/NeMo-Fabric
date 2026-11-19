// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Facts about the place the agent runs: what the caller knows, and what the
//! loader can read from this machine when it is asked to.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::model::Facts;
use super::needs::{NETWORK_REACH, size_bytes};

/// Every key a caller may give under `facts`.
const FACT_KEYS: [&str; 12] = [
    "cpus",
    "memory",
    "storage",
    "network",
    "harness",
    "harness_aliases",
    "gpus",
    "remote_gpus",
    "software",
    "packages",
    "model",
    "known",
];

/// The members of one GPU fact. A remote GPU also names its connector.
const GPU_FACT_KEYS: [&str; 4] = ["interface", "memory", "family", "model"];
const REMOTE_GPU_FACT_KEYS: [&str; 5] = ["interface", "memory", "family", "model", "connector"];
/// The members of one interface that a GPU fact offers.
const INTERFACE_FACT_KEYS: [&str; 2] = ["version", "compute_capability"];
/// The members of the `model` fact.
const MODEL_FACT_KEYS: [&str; 4] = [
    "name",
    "capabilities",
    "context_tokens",
    "parameters_applied",
];
/// The members of the `known` fact.
const KNOWN_FACT_KEYS: [&str; 4] = ["interfaces", "families", "capabilities", "harnesses"];
/// A size as JSON Schema reads it: the text `size_bytes` accepts.
const SIZE_PATTERN: &str = r"^(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:kb|mb|gb|tb)$";

/// The JSON Schema properties of `facts`, for the options schema.
///
/// The schema states what `overlay_facts` accepts, so a caller can check an
/// options file before the load. A test holds the two together.
pub(super) fn facts_schema_properties() -> Value {
    let names = json!({"type": "array", "items": {"type": "string"}});
    let gpu = |remote: bool| {
        let mut members = json!({
            "interface": {
                "description": "The interfaces the GPU offers, by name, for example `cuda`.",
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "version": {"type": "string"},
                        "compute_capability": {"type": "string"}
                    }
                }
            },
            "memory": {"description": "A size, such as `24gb`.", "type": "string", "pattern": SIZE_PATTERN},
            "family": {"type": "string"},
            "model": {"type": "string"}
        });
        if remote {
            members["connector"] = json!({
                "description": "The connector the GPU is reached through.",
                "type": "string"
            });
        }
        json!({"type": "object", "additionalProperties": false, "properties": members})
    };
    json!({
        "cpus": {
            "description": "Processors the agent can use.",
            "type": ["integer", "null"],
            "minimum": 0
        },
        "memory": {
            "description": "Memory the agent can use: a number and a lower-case unit, such as `64gb`.",
            "type": ["string", "null"],
            "pattern": SIZE_PATTERN
        },
        "storage": {
            "description": "Free storage the agent can use, as a size.",
            "type": ["string", "null"],
            "pattern": SIZE_PATTERN
        },
        "network": {
            "description": "The network reach the agent is allowed.",
            "enum": ["none", "internal", "external", null]
        },
        "harness": {
            "description": "The name this harness goes by. Left out: the selected adapter's id.",
            "type": ["string", "null"]
        },
        "harness_aliases": {
            "description": "Other names of this harness. The loader adds the names of the selected adapter.",
            "type": "array",
            "items": {"type": "string"}
        },
        "gpus": {
            "description": "The GPUs on the machine, one entry for each.",
            "type": ["array", "null"],
            "items": gpu(false)
        },
        "remote_gpus": {
            "description": "The GPUs reached through a connector, one entry for each.",
            "type": ["array", "null"],
            "items": gpu(true)
        },
        "software": {
            "description": "Every program on the machine with its version, by name. A program left out is not on the machine. Null means the program is there and its version is not known.",
            "type": ["object", "null"],
            "additionalProperties": {"type": ["string", "null"]}
        },
        "packages": {
            "description": "The installed software packages, by registry and then by name, with the registry named as a lock names it, such as `pypi` or `npm`. A registry lists every package installed from it with its version. A package left out is not installed. Null means the package is installed and its version is not known. A registry left out was not looked at.",
            "type": ["object", "null"],
            "additionalProperties": {
                "type": "object",
                "additionalProperties": {"type": ["string", "null"]}
            }
        },
        "model": {
            "description": "The model that runs.",
            "type": ["object", "null"],
            "additionalProperties": false,
            "properties": {
                "name": {"type": "string"},
                "capabilities": names,
                "context_tokens": {"type": "integer", "minimum": 0},
                "parameters_applied": names
            }
        },
        "known": {
            "description": "Names this platform knows. A need on a name that is not known is could not be checked, never not met.",
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "interfaces": {"type": ["array", "null"], "items": {"type": "string"}},
                "families": names,
                "capabilities": names,
                "harnesses": names
            }
        }
    })
}

/// How long one version command may run.
const VERSION_COMMAND_LIMIT: Duration = Duration::from_secs(3);
/// How long the GPU query may run.
const GPU_QUERY_LIMIT: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// The loader's own table of version commands. The loader never runs a program
/// because a document names it: it runs only these, and only for a name that
/// is a key here.
const VERSION_COMMANDS: [(&str, &[&str]); 11] = [
    ("python3", &["--version"]),
    ("python", &["--version"]),
    ("node", &["--version"]),
    ("npm", &["--version"]),
    ("uv", &["--version"]),
    ("git", &["--version"]),
    ("docker", &["--version"]),
    ("ffmpeg", &["-version"]),
    ("java", &["-version"]),
    ("go", &["version"]),
    ("bash", &["--version"]),
];

/// Caller facts win, key by key. An unknown key is `package_options_invalid`.
///
/// `software` is laid over program by program, so a caller that states one
/// program does not hide the others that were probed.
pub(super) fn overlay_facts(
    probed: Facts,
    supplied: Option<&Map<String, Value>>,
) -> PackageResult<Facts> {
    let Some(supplied) = supplied else {
        return Ok(probed);
    };
    if let Some(unknown) = supplied
        .keys()
        .find(|key| !FACT_KEYS.contains(&key.as_str()))
    {
        return Err(PackageError::new(
            PackageErrorCode::OptionsInvalid,
            format!("facts.{unknown}"),
            "this is not a fact the loader knows",
        ));
    }
    let given: Facts =
        serde_json::from_value(Value::Object(supplied.clone())).map_err(|error| {
            PackageError::new(
                PackageErrorCode::OptionsInvalid,
                "facts",
                format!("the facts do not have the expected shape: {error}"),
            )
        })?;
    check_fact_values(&given, supplied)?;
    let mut facts = probed;
    let has = |key: &str| supplied.contains_key(key);
    if has("cpus") {
        facts.cpus = given.cpus;
    }
    if has("memory") {
        facts.memory = given.memory;
    }
    if has("storage") {
        facts.storage = given.storage;
    }
    if has("network") {
        facts.network = given.network;
    }
    if has("harness") {
        facts.harness = given.harness;
    }
    if has("harness_aliases") {
        facts.harness_aliases = given.harness_aliases;
    }
    if has("gpus") {
        facts.gpus = given.gpus;
    }
    if has("remote_gpus") {
        facts.remote_gpus = given.remote_gpus;
    }
    if has("software") {
        facts.software = match (facts.software.take(), given.software) {
            (Some(mut probed), Some(given)) => {
                probed.extend(given);
                Some(probed)
            }
            (_, given) => given,
        };
    }
    if has("packages") {
        facts.packages = given.packages;
    }
    if has("model") {
        facts.model = given.model;
    }
    if has("known") {
        facts.known = given.known;
    }
    Ok(facts)
}

/// A fact value the need check could not read is refused, naming the fact.
/// So is a member it does not read. Left alone, either would count as no fact
/// at all and the need would be reported as could not be checked.
fn check_fact_values(given: &Facts, supplied: &Map<String, Value>) -> PackageResult<()> {
    let invalid = |subject: String, message: &str| {
        Err(PackageError::new(
            PackageErrorCode::OptionsInvalid,
            subject,
            message,
        ))
    };
    let is_text_list = |value: &Value| {
        value
            .as_array()
            .is_some_and(|items| items.iter().all(Value::is_string))
    };
    const SIZE: &str =
        "a size is a number and a unit, in lower case: kb, mb, gb or tb, for example 16gb";
    for (key, value) in [("memory", &given.memory), ("storage", &given.storage)] {
        if value
            .as_deref()
            .is_some_and(|text| size_bytes(text).is_none())
        {
            return invalid(format!("facts.{key}"), SIZE);
        }
    }
    if given
        .network
        .as_deref()
        .is_some_and(|reach| !NETWORK_REACH.contains(&reach))
    {
        return invalid(
            "facts.network".to_string(),
            "the network reach is one of none, internal, external",
        );
    }
    for (key, gpus) in [("gpus", &given.gpus), ("remote_gpus", &given.remote_gpus)] {
        let members: &[&str] = if key == "gpus" {
            &GPU_FACT_KEYS
        } else {
            &REMOTE_GPU_FACT_KEYS
        };
        for (index, gpu) in gpus.iter().flatten().enumerate() {
            let subject = format!("facts.{key}.{index}");
            let Some(gpu) = gpu.as_object() else {
                return invalid(subject, "a GPU fact is an object");
            };
            if let Some(unknown) = gpu.keys().find(|name| !members.contains(&name.as_str())) {
                return invalid(
                    format!("{subject}.{unknown}"),
                    "this is not a member of a GPU fact",
                );
            }
            match gpu.get("memory") {
                None => {}
                Some(Value::String(text)) if size_bytes(text).is_some() => {}
                Some(_) => return invalid(format!("{subject}.memory"), SIZE),
            }
            for name in ["family", "model", "connector"] {
                if gpu.get(name).is_some_and(|value| !value.is_string()) {
                    return invalid(format!("{subject}.{name}"), "the value is text");
                }
            }
            const INTERFACE: &str = "the interfaces are an object by name, each with an object, for example {\"cuda\": {\"version\": \"12.4\"}}";
            let Some(interfaces) = gpu.get("interface") else {
                continue;
            };
            let Some(interfaces) = interfaces.as_object() else {
                return invalid(format!("{subject}.interface"), INTERFACE);
            };
            for (name, offered) in interfaces {
                let Some(offered) = offered.as_object() else {
                    return invalid(format!("{subject}.interface.{name}"), INTERFACE);
                };
                for (member, value) in offered {
                    if !INTERFACE_FACT_KEYS.contains(&member.as_str()) {
                        return invalid(
                            format!("{subject}.interface.{name}.{member}"),
                            "this is not a member of an interface fact",
                        );
                    }
                    if !value.is_string() {
                        return invalid(
                            format!("{subject}.interface.{name}.{member}"),
                            "the value is text",
                        );
                    }
                }
            }
        }
    }
    if let Some(model) = &given.model {
        let Some(model) = model.as_object() else {
            return invalid("facts.model".to_string(), "the model fact is an object");
        };
        for (member, value) in model {
            if !MODEL_FACT_KEYS.contains(&member.as_str()) {
                return invalid(
                    format!("facts.model.{member}"),
                    "this is not a member of the model fact",
                );
            }
            let readable = match member.as_str() {
                "name" => value.is_string(),
                "context_tokens" => value.is_u64(),
                _ => is_text_list(value),
            };
            if !readable {
                return invalid(
                    format!("facts.model.{member}"),
                    "the value does not have the type of this member",
                );
            }
        }
    }
    if let Some(unknown) = supplied
        .get("known")
        .and_then(Value::as_object)
        .and_then(|known| {
            known
                .keys()
                .find(|name| !KNOWN_FACT_KEYS.contains(&name.as_str()))
        })
    {
        return invalid(
            format!("facts.known.{unknown}"),
            "this is not a member of the known fact",
        );
    }
    Ok(())
}

/// Facts about this machine. Never fails: what cannot be read is left out.
///
/// `software`: the program names the documents state. `storage_dir`: a folder
/// on the file system the agent writes to.
pub(super) fn probe_facts(software: &[String], storage_dir: &Path) -> Facts {
    let mut facts = Facts {
        cpus: std::thread::available_parallelism()
            .ok()
            .map(|count| count.get() as u64),
        memory: memory_bytes().map(size_text),
        storage: free_storage_bytes(storage_dir).map(size_text),
        gpus: nvidia_gpus(),
        ..Facts::default()
    };
    if !software.is_empty() {
        let mut found = BTreeMap::new();
        for name in software {
            // Only a bare program name is looked up. A name with a path
            // separator is recorded with no version, so its need is reported
            // as could not be checked and never as not met.
            if name.contains(['/', '\\']) {
                found.insert(name.clone(), None);
            } else if crate::doctor::command_available(name) {
                found.insert(name.clone(), program_version(name));
            }
        }
        // An absent key means the program was looked for and is not on the machine.
        facts.software = Some(found);
    }
    facts
}

/// A byte count as the format's size text, in whole kb. Units are powers of 1024.
fn size_text(bytes: u64) -> String {
    format!("{}kb", bytes / 1024)
}

/// The version of a program in the loader's table, or None when it is not in
/// the table or its version cannot be read.
fn program_version(name: &str) -> Option<String> {
    let (_, arguments) = VERSION_COMMANDS
        .iter()
        .find(|(program, _)| *program == name)?;
    let output = run_bounded(name, arguments, VERSION_COMMAND_LIMIT)?;
    first_version_number(&output)
}

/// The first dotted number in the text: `3.12.4` in `Python 3.12.4`.
fn first_version_number(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut start = 0;
    while start < bytes.len() {
        if !bytes[start].is_ascii_digit() {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.') {
            end += 1;
        }
        let candidate = text[start..end].trim_end_matches('.');
        if candidate.contains('.') && !candidate.contains("..") {
            return Some(candidate.to_string());
        }
        start = end.max(start + 1);
    }
    None
}

/// Run one fixed command with a time limit. Returns its output and error text
/// joined, or None when it cannot be started, fails or runs too long.
fn run_bounded(program: &str, arguments: &[&str], limit: Duration) -> Option<String> {
    let mut child = Command::new(program)
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + limit;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(POLL_INTERVAL),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push('\n');
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Some(text)
}

/// NVIDIA GPUs through `nvidia-smi` with a fixed query: one entry per GPU, with
/// its memory. None when the program is not there or fails.
///
/// The probe states nothing about interfaces, families or models: a need on
/// one of those is reported as could not be checked unless the caller gives
/// facts. The program's model text (for example `NVIDIA GeForce RTX 4090`) is
/// not a name the format defines, so it is not compared with a need.
fn nvidia_gpus() -> Option<Vec<Value>> {
    if !crate::doctor::command_available("nvidia-smi") {
        return None;
    }
    let output = run_bounded(
        "nvidia-smi",
        &["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        GPU_QUERY_LIMIT,
    )?;
    Some(parse_nvidia_gpus(&output))
}

/// One GPU per line: `<memory in MiB>`.
fn parse_nvidia_gpus(output: &str) -> Vec<Value> {
    output
        .lines()
        .filter_map(|line| {
            let mebibytes: u64 = line.trim().parse().ok()?;
            Some(serde_json::json!({"memory": format!("{mebibytes}mb")}))
        })
        .collect()
}

/// Memory the run can use: on Linux the smaller of the cgroup limit and total
/// memory, on macOS total memory. Not probed elsewhere.
fn memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let total = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| meminfo_total_bytes(&text));
        let limit = crate::system_stats::cgroup::read_host(
            Path::new("/"),
            std::process::id(),
            &[crate::system_stats::SystemStatsMetricGroup::CgroupMemory],
        )
        .ok()
        .and_then(|snapshot| snapshot.visible_memory_limit_bytes.value().copied());
        match (total, limit) {
            (Some(total), Some(limit)) => Some(total.min(limit)),
            (total, limit) => total.or(limit),
        }
    }
    #[cfg(target_os = "macos")]
    {
        let mut bytes: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        // SAFETY: the name is a NUL-terminated literal, and the value and size
        // pointers refer to live locals of the size passed.
        let status = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&raw mut bytes).cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0 && bytes > 0).then_some(bytes)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// `MemTotal` of `/proc/meminfo`, in bytes.
#[cfg(any(target_os = "linux", test))]
fn meminfo_total_bytes(text: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kilobytes: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kilobytes.checked_mul(1024)
}

/// Free space, for an unprivileged user, of the file system that holds `folder`.
fn free_storage_bytes(folder: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;

        let path = std::ffi::CString::new(folder.as_os_str().as_bytes()).ok()?;
        let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        // SAFETY: the path is a NUL-terminated string and the buffer is the
        // size of the structure the call fills.
        let status = unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) };
        if status != 0 {
            return None;
        }
        // SAFETY: the call returned 0, so it filled the structure.
        let stats = unsafe { stats.assume_init() };
        (stats.f_bavail as u64).checked_mul(stats.f_frsize as u64)
    }
    #[cfg(not(unix))]
    {
        let _ = folder;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supplied(value: Value) -> Map<String, Value> {
        value.as_object().expect("an object").clone()
    }

    #[test]
    fn caller_facts_win_key_by_key() {
        let probed = Facts {
            cpus: Some(8),
            memory: Some("16gb".to_string()),
            storage: Some("100gb".to_string()),
            software: Some(BTreeMap::from([
                ("node".to_string(), Some("20.1.0".to_string())),
                ("uv".to_string(), None),
            ])),
            ..Facts::default()
        };
        let given = supplied(json!({
            "memory": "64gb",
            "network": "internal",
            "software": {"uv": "0.4.18", "ffmpeg": "6.1"},
            "packages": {"pypi": {"faster-whisper": "1.2.1", "mcp": null}},
            "known": {"harnesses": ["claude-code"]}
        }));

        let facts = overlay_facts(probed, Some(&given)).expect("facts");

        assert_eq!(facts.cpus, Some(8));
        assert_eq!(facts.memory.as_deref(), Some("64gb"));
        assert_eq!(facts.storage.as_deref(), Some("100gb"));
        assert_eq!(facts.network.as_deref(), Some("internal"));
        assert_eq!(
            facts.software,
            Some(BTreeMap::from([
                ("ffmpeg".to_string(), Some("6.1".to_string())),
                ("node".to_string(), Some("20.1.0".to_string())),
                ("uv".to_string(), Some("0.4.18".to_string())),
            ]))
        );
        // Nothing probes for packages, so the fact is the caller's as given.
        assert_eq!(
            facts.packages,
            Some(BTreeMap::from([(
                "pypi".to_string(),
                BTreeMap::from([
                    ("faster-whisper".to_string(), Some("1.2.1".to_string())),
                    ("mcp".to_string(), None),
                ])
            )]))
        );
        assert_eq!(facts.known.harnesses, vec!["claude-code".to_string()]);
    }

    #[test]
    fn no_caller_facts_keeps_the_probed_facts() {
        let probed = Facts {
            cpus: Some(2),
            ..Facts::default()
        };

        assert_eq!(overlay_facts(probed.clone(), None).expect("facts"), probed);
    }

    #[test]
    fn a_null_fact_states_that_the_value_is_not_known() {
        let probed = Facts {
            cpus: Some(2),
            ..Facts::default()
        };

        let facts = overlay_facts(probed, Some(&supplied(json!({"cpus": null})))).expect("facts");

        assert_eq!(facts.cpus, None);
    }

    #[test]
    fn an_unknown_fact_key_is_refused_by_name() {
        let error = overlay_facts(Facts::default(), Some(&supplied(json!({"ram": "64gb"}))))
            .expect_err("unknown key");

        assert_eq!(error.code, PackageErrorCode::OptionsInvalid);
        assert_eq!(error.subject, "facts.ram");
    }

    #[test]
    fn a_fact_of_the_wrong_shape_is_refused() {
        let error = overlay_facts(Facts::default(), Some(&supplied(json!({"cpus": "eight"}))))
            .expect_err("wrong shape");

        assert_eq!(error.code, PackageErrorCode::OptionsInvalid);
        assert_eq!(error.subject, "facts");
    }

    #[test]
    fn version_numbers_are_read_from_common_outputs() {
        let cases = [
            ("Python 3.12.4", Some("3.12.4")),
            ("v22.3.0\n", Some("22.3.0")),
            ("10.8.1", Some("10.8.1")),
            ("uv 0.4.18 (7b55e9790 2024-10-01)", Some("0.4.18")),
            ("git version 2.39.5 (Apple Git-154)", Some("2.39.5")),
            ("Docker version 27.0.3, build 7d4bcd8", Some("27.0.3")),
            (
                "ffmpeg version 6.1.1 Copyright (c) 2000-2023",
                Some("6.1.1"),
            ),
            ("\nopenjdk version \"21.0.2\" 2024-01-16", Some("21.0.2")),
            ("go version go1.22.3 darwin/arm64", Some("1.22.3")),
            ("GNU bash, version 5.2.26(1)-release", Some("5.2.26")),
            ("build 12 of 2024", None),
            ("", None),
        ];

        for (text, expected) in cases {
            assert_eq!(first_version_number(text).as_deref(), expected, "{text}");
        }
    }

    #[test]
    fn the_version_table_has_no_duplicate_and_only_bare_names() {
        let mut names = std::collections::BTreeSet::new();
        for (name, arguments) in VERSION_COMMANDS {
            assert!(names.insert(name), "{name}");
            assert!(!name.contains(['/', '\\', ' ']));
            assert!(!arguments.is_empty());
        }
    }

    #[test]
    fn a_program_outside_the_table_is_never_run() {
        assert_eq!(program_version("sh"), None);
        assert_eq!(program_version("definitely-not-a-program"), None);
    }

    #[test]
    fn gpu_lines_give_one_entry_with_its_memory_and_nothing_else() {
        let gpus = parse_nvidia_gpus("49140\n 24564 \n\n");

        // No interface, family or model: the probe has no fact about them.
        assert_eq!(
            gpus,
            vec![json!({"memory": "49140mb"}), json!({"memory": "24564mb"})]
        );
        assert!(parse_nvidia_gpus("not a gpu line").is_empty());
    }

    #[test]
    fn a_fact_value_that_cannot_be_read_is_refused_by_name() {
        // (facts, the fact the refusal names)
        let table = [
            (json!({"network": "Internal"}), "facts.network"),
            (json!({"network": "lan"}), "facts.network"),
            (json!({"memory": "16GB"}), "facts.memory"),
            (json!({"memory": "16"}), "facts.memory"),
            (json!({"storage": "a lot"}), "facts.storage"),
            (json!({"gpus": [{"memory": "24GB"}]}), "facts.gpus.0.memory"),
            (json!({"gpus": ["l4"]}), "facts.gpus.0"),
            (
                json!({"remote_gpus": [{"memory": "24gb"}, {"memory": 24}]}),
                "facts.remote_gpus.1.memory",
            ),
        ];
        for (facts, subject) in table {
            let error = overlay_facts(Facts::default(), Some(&supplied(facts)))
                .expect_err("a value that cannot be read");

            assert_eq!(error.code, PackageErrorCode::OptionsInvalid, "{subject}");
            assert_eq!(error.subject, subject);
        }
        let good = supplied(json!({
            "network": "none", "memory": "16gb", "storage": "1.5tb",
            "gpus": [{"memory": "24gb"}, {"model": "l4"}]
        }));
        overlay_facts(Facts::default(), Some(&good)).expect("values the need check reads");
    }

    #[test]
    fn a_member_the_need_check_does_not_read_is_refused_by_name() {
        // (facts, the member the refusal names)
        let table = [
            (
                json!({"gpus": [{"memory": "24gb", "interfaces": {}}]}),
                "facts.gpus.0.interfaces",
            ),
            (
                json!({"gpus": [{"connector": "render-queue"}]}),
                "facts.gpus.0.connector",
            ),
            (
                json!({"gpus": [{"interface": {"cuda": "12.4"}}]}),
                "facts.gpus.0.interface.cuda",
            ),
            (
                json!({"gpus": [{"interface": ["cuda"]}]}),
                "facts.gpus.0.interface",
            ),
            (
                json!({"gpus": [{"interface": {"cuda": {"release": "12.4"}}}]}),
                "facts.gpus.0.interface.cuda.release",
            ),
            (
                json!({"gpus": [{"interface": {"cuda": {"version": 12.4}}}]}),
                "facts.gpus.0.interface.cuda.version",
            ),
            (
                json!({"remote_gpus": [{"family": 4090}]}),
                "facts.remote_gpus.0.family",
            ),
            (json!({"model": {"context": 128000}}), "facts.model.context"),
            (
                json!({"model": {"context_tokens": "128000"}}),
                "facts.model.context_tokens",
            ),
            (
                json!({"model": {"capabilities": "vision"}}),
                "facts.model.capabilities",
            ),
            (json!({"model": "served-model"}), "facts.model"),
            (
                json!({"known": {"harness": ["claude-code"]}}),
                "facts.known.harness",
            ),
        ];
        for (facts, subject) in table {
            let error = overlay_facts(Facts::default(), Some(&supplied(facts)))
                .expect_err("a member that is not read");

            assert_eq!(error.code, PackageErrorCode::OptionsInvalid, "{subject}");
            assert_eq!(error.subject, subject);
        }
        let good = supplied(json!({
            "gpus": [{"interface": {"cuda": {"version": "12.4", "compute_capability": "8.0"}, "metal": {}},
                      "memory": "24gb", "family": "ada", "model": "l4"}],
            "remote_gpus": [{"memory": "48gb", "connector": "render-queue"}],
            "model": {"name": "served-model", "capabilities": ["vision"],
                      "context_tokens": 128000, "parameters_applied": ["temperature"]},
            "known": {"interfaces": ["cuda"], "families": ["ada"],
                      "capabilities": ["vision"], "harnesses": ["claude-code"]}
        }));
        overlay_facts(Facts::default(), Some(&good)).expect("members the need check reads");
    }

    #[test]
    fn the_options_schema_accepts_the_facts_the_loader_accepts() {
        let schema = crate::schema::generate_schema(crate::schema::SchemaName::AgentPackageOptions)
            .expect("options schema");
        let validator = jsonschema::validator_for(&schema).expect("valid options schema");
        let properties = facts_schema_properties();
        let stated: Vec<&str> = properties
            .as_object()
            .expect("an object")
            .keys()
            .map(String::as_str)
            .collect();
        let mut known = FACT_KEYS.to_vec();
        known.sort_unstable();
        assert_eq!(stated, known);

        // (facts, whether the loader accepts them)
        let table = [
            (json!({}), true),
            (
                json!({"cpus": 8, "memory": "64gb", "storage": "1.5tb"}),
                true,
            ),
            (json!({"cpus": null, "memory": null, "network": null}), true),
            (
                json!({"network": "internal", "harness": "my-harness"}),
                true,
            ),
            (json!({"harness_aliases": ["mine"]}), true),
            (json!({"software": {"python3": "3.12", "uv": null}}), true),
            (json!({"software": {}}), true),
            (
                json!({"packages": {"pypi": {"faster-whisper": "1.2.1", "mcp": null}, "npm": {}}}),
                true,
            ),
            (json!({"packages": null}), true),
            (json!({"gpus": [], "remote_gpus": null}), true),
            (
                json!({"gpus": [{"interface": {"cuda": {"version": "12.4"}, "metal": {}},
                                 "memory": "24gb", "family": "ada", "model": "l4"}]}),
                true,
            ),
            (
                json!({"remote_gpus": [{"memory": "48gb", "connector": "render-queue"}]}),
                true,
            ),
            (
                json!({"model": {"name": "served-model", "capabilities": ["vision"],
                                 "context_tokens": 128000, "parameters_applied": []}}),
                true,
            ),
            (json!({"model": null}), true),
            (
                json!({"known": {"interfaces": null, "families": ["ada"], "harnesses": []}}),
                true,
            ),
            (json!({"ram": "64gb"}), false),
            (json!({"cpus": "eight"}), false),
            (json!({"cpus": -1}), false),
            (json!({"memory": "16GB"}), false),
            (json!({"storage": "a lot"}), false),
            (json!({"network": "lan"}), false),
            (json!({"harness_aliases": "mine"}), false),
            (json!({"software": {"python3": 3.12}}), false),
            (json!({"packages": {"pypi": ["faster-whisper"]}}), false),
            (
                json!({"packages": {"pypi": {"faster-whisper": 1.2}}}),
                false,
            ),
            (json!({"gpus": 1}), false),
            (json!({"gpus": ["24gb"]}), false),
            (json!({"gpus": [{"memory": "24GB"}]}), false),
            (json!({"gpus": [{"connector": "render-queue"}]}), false),
            (json!({"gpus": [{"interface": {"cuda": "12.4"}}]}), false),
            (
                json!({"gpus": [{"interface": {"cuda": {"release": "12.4"}}}]}),
                false,
            ),
            (json!({"remote_gpus": [{"family": 4090}]}), false),
            (json!({"model": {"context": 128000}}), false),
            (json!({"model": {"context_tokens": "128000"}}), false),
            (json!({"model": {"capabilities": "vision"}}), false),
            (json!({"known": {"harness": ["claude-code"]}}), false),
            (json!({"known": null}), false),
        ];
        for (facts, accepted) in table {
            assert_eq!(
                validator.is_valid(&json!({"facts": facts})),
                accepted,
                "schema: {facts}"
            );
            assert_eq!(
                overlay_facts(Facts::default(), Some(&supplied(facts.clone()))).is_ok(),
                accepted,
                "loader: {facts}"
            );
        }
    }

    #[test]
    fn sizes_are_whole_kb() {
        assert_eq!(size_text(16 * 1024 * 1024 * 1024), "16777216kb");
        assert_eq!(size_text(1536), "1kb");
    }

    #[test]
    fn meminfo_total_is_read_in_bytes() {
        let text = "MemTotal:       16384000 kB\nMemFree:         1024000 kB\n";

        assert_eq!(meminfo_total_bytes(text), Some(16_384_000 * 1024));
        assert_eq!(meminfo_total_bytes("MemFree: 1 kB\n"), None);
    }

    #[test]
    fn probing_gives_only_well_formed_facts() {
        let folder = std::env::temp_dir();
        let software = vec!["definitely-not-a-program".to_string(), "../sh".to_string()];

        let facts = probe_facts(&software, &folder);

        assert!(facts.cpus.is_none_or(|count| count > 0));
        for size in [&facts.memory, &facts.storage].into_iter().flatten() {
            let digits = size.strip_suffix("kb").expect("a size in kb");
            assert!(!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()));
        }
        // A program that is not there has no key. A name with a path separator
        // is not looked up, so it has no version and is never "not there".
        assert_eq!(
            facts.software,
            Some(BTreeMap::from([("../sh".to_string(), None)]))
        );
        for gpu in facts.gpus.iter().flatten() {
            assert_eq!(
                gpu.as_object()
                    .expect("a GPU fact")
                    .keys()
                    .collect::<Vec<_>>(),
                vec!["memory"]
            );
        }
        assert!(facts.network.is_none());
        assert!(facts.harness.is_none());
        assert!(facts.model.is_none());
        assert_eq!(probe_facts(&[], &folder).software, None);
    }
}
