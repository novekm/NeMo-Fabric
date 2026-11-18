// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime needs: one line per need, and the start decision.
//!
//! A status is worked out from facts about one load, never from the document. A need that
//! could not be checked is never given as met.

use std::collections::BTreeMap;

use serde_json::Value;

#[cfg(test)]
use super::model::{Decision, PackageAction};
use super::model::{Facts, NeedLine, NeedStatus, StatedBy};
use super::paths::json_pointer;
use super::version::satisfies;

/// The interface names a receiver knows when its facts list none.
const WELL_KNOWN_INTERFACES: [&str; 4] = ["cuda", "metal", "rocm", "vulkan"];
/// Network reach, from the least to the most.
pub(super) const NETWORK_REACH: [&str; 3] = ["none", "internal", "external"];

/// A size such as `16gb`, in bytes. None for text that is not a size.
pub(super) fn size_bytes(text: &str) -> Option<f64> {
    let split = text.len().checked_sub(2)?;
    if !text.is_char_boundary(split) {
        return None;
    }
    let (amount, unit) = text.split_at(split);
    let unit = match unit {
        "kb" => 1024f64,
        "mb" => 1024f64.powi(2),
        "gb" => 1024f64.powi(3),
        "tb" => 1024f64.powi(4),
        _ => return None,
    };
    let plain = !amount.is_empty() && amount.bytes().all(|b| b.is_ascii_digit() || b == b'.');
    let amount: f64 = if plain {
        amount.parse().ok()?
    } else {
        return None;
    };
    Some(amount * unit)
}

/// All of them hold: false when one fails, unknown when none fails and one is unknown.
fn weigh(verdicts: impl IntoIterator<Item = Option<bool>>) -> Option<bool> {
    let mut unknown = false;
    for verdict in verdicts {
        match verdict {
            Some(false) => return Some(false),
            None => unknown = true,
            Some(true) => {}
        }
    }
    if unknown { None } else { Some(true) }
}

/// Any one listed name satisfies. Unknown when the receiver cannot tell: no fact, or a listed
/// name it does not recognize.
fn one_of(names: &Value, value: Option<&str>, known: Option<&[String]>) -> Option<bool> {
    let names: Vec<&str> = names
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let Some(value) = value else {
        return None;
    };
    if names.contains(&value) {
        return Some(true);
    }
    if known.is_some_and(|known| names.iter().any(|name| !known.iter().any(|k| k == name))) {
        return None;
    }
    Some(false)
}

/// Does one GPU meet every field of the need? Unknown when the receiver cannot tell.
fn gpu_fits(need: &Value, gpu: &Value, facts: &Facts) -> Option<bool> {
    let mut verdicts = Vec::new();
    if let Some(interfaces) = need.get("interface").and_then(Value::as_object) {
        // Any one listed interface at its floor meets the need. A GPU fact
        // with no `interface` member says nothing about interfaces, so there
        // is no fact to check with: that is unknown, never not met. A fact
        // that lists its interfaces and leaves a known name out does not
        // offer that interface.
        let offers = gpu.get("interface");
        let mut best = if offers.is_some() { Some(false) } else { None };
        for (name, floor) in interfaces {
            let Some(offers) = offers else {
                break;
            };
            let Some(offered) = offers.get(name) else {
                let known = match &facts.known.interfaces {
                    Some(known) => known.iter().any(|k| k == name),
                    None => WELL_KNOWN_INTERFACES.contains(&name.as_str()),
                };
                if !known {
                    best = None;
                }
                continue;
            };
            let met = weigh(
                ["version", "compute_capability"]
                    .into_iter()
                    .filter_map(|key| {
                        let wanted = floor.get(key)?.as_str()?;
                        Some(
                            offered
                                .get(key)
                                .and_then(Value::as_str)
                                .and_then(|have| satisfies(have, wanted)),
                        )
                    }),
            );
            match met {
                Some(true) => {
                    best = Some(true);
                    break;
                }
                None => best = None,
                Some(false) => {}
            }
        }
        verdicts.push(best);
    }
    if let Some(wanted) = need.get("memory").and_then(Value::as_str) {
        let have = gpu.get("memory").and_then(Value::as_str);
        verdicts.push(match (have.and_then(size_bytes), size_bytes(wanted)) {
            (Some(have), Some(wanted)) => Some(have >= wanted),
            _ => None,
        });
    }
    if let Some(families) = need.get("families") {
        verdicts.push(one_of(
            families,
            gpu.get("family").and_then(Value::as_str),
            Some(&facts.known.families),
        ));
    }
    if let Some(models) = need.get("models") {
        verdicts.push(one_of(
            models,
            gpu.get("model").and_then(Value::as_str),
            None,
        ));
    }
    weigh(verdicts)
}

/// `access` says which GPUs count. All `count` GPUs are in one place: here, or through one connector.
fn gpu_status(need: &Value, facts: &Facts) -> (NeedStatus, Option<String>, String) {
    if facts.gpus.is_none() && facts.remote_gpus.is_none() {
        return (
            NeedStatus::CouldNotBeChecked,
            None,
            "no fact about a GPU was given".to_string(),
        );
    }
    let access = need.get("access").and_then(Value::as_str).unwrap_or("any");
    let wanted = need.get("count").and_then(Value::as_u64).unwrap_or(1) as usize;
    let here: Vec<Option<bool>> = match (&facts.gpus, access) {
        (Some(gpus), access) if access != "remote" => {
            gpus.iter().map(|gpu| gpu_fits(need, gpu, facts)).collect()
        }
        _ => Vec::new(),
    };
    let mut by_connector: BTreeMap<&str, Vec<Option<bool>>> = BTreeMap::new();
    if access != "local" {
        for gpu in facts.remote_gpus.iter().flatten() {
            // A remote GPU counts through the connector that reaches it.
            if let Some(connector) = gpu.get("connector").and_then(Value::as_str) {
                by_connector
                    .entry(connector)
                    .or_default()
                    .push(gpu_fits(need, gpu, facts));
            }
        }
    }
    let fits = |place: &[Option<bool>]| place.iter().filter(|v| **v == Some(true)).count();
    if fits(&here) >= wanted {
        return (
            NeedStatus::MetHere,
            None,
            format!("{wanted} GPU that fits is in the agent's own environment"),
        );
    }
    for (connector, place) in &by_connector {
        if fits(place) >= wanted {
            return (
                NeedStatus::MetRemotely,
                Some(connector.to_string()),
                format!("{wanted} GPU that fits is reached through the connector {connector}"),
            );
        }
    }
    let could_fit = |place: &[Option<bool>]| place.iter().filter(|v| **v != Some(false)).count();
    if std::iter::once(&here)
        .chain(by_connector.values())
        .any(|place| could_fit(place) >= wanted)
    {
        return (
            NeedStatus::CouldNotBeChecked,
            None,
            "a GPU could not be compared with every field of the need".to_string(),
        );
    }
    (
        NeedStatus::NotMet,
        None,
        format!("no one place has {wanted} GPU that fits"),
    )
}

struct Stated<'a> {
    by: StatedBy,
    document: &'a str,
    /// `definition.` before the need text of a definition's need.
    prefix: &'a str,
    lines: &'a mut Vec<NeedLine>,
}

impl Stated<'_> {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        need: &str,
        pointer: &[&str],
        status: NeedStatus,
        connector: Option<String>,
        optional: bool,
        reason: Option<&Value>,
        detail: String,
    ) {
        self.lines.push(NeedLine {
            need: format!("{}{need}", self.prefix),
            pointer: json_pointer(pointer),
            status,
            connector,
            optional,
            stated_by: self.by,
            document: self.document.to_string(),
            reason: reason.and_then(Value::as_str).map(str::to_string),
            detail: Some(detail),
        });
    }
}

fn is_true(value: Option<&Value>) -> bool {
    value.and_then(Value::as_bool) == Some(true)
}

fn at_least(have: Option<f64>, wanted: Option<f64>) -> NeedStatus {
    match (have, wanted) {
        (Some(have), Some(wanted)) if have >= wanted => NeedStatus::MetHere,
        (Some(_), Some(_)) => NeedStatus::NotMet,
        _ => NeedStatus::CouldNotBeChecked,
    }
}

/// The needs under one document's `runtime` block.
fn runtime_lines(runtime: &Value, facts: &Facts, stated: &mut Stated<'_>) {
    let compute = runtime.get("compute");
    let reason = compute.and_then(|compute| compute.get("reason"));
    let spare = is_true(compute.and_then(|compute| compute.get("optional")));
    if let Some(wanted) = compute.and_then(|compute| compute.get("cpus")) {
        let (status, detail) = match facts.cpus {
            None => (
                NeedStatus::CouldNotBeChecked,
                "no fact about CPUs was given".to_string(),
            ),
            Some(have) => (
                at_least(Some(have as f64), wanted.as_f64()),
                format!("needs at least {wanted}, has {have}"),
            ),
        };
        stated.push(
            "runtime.compute.cpus",
            &["runtime", "compute", "cpus"],
            status,
            None,
            spare,
            reason,
            detail,
        );
    }
    for (key, fact) in [("memory", &facts.memory), ("storage", &facts.storage)] {
        let Some(wanted) = compute
            .and_then(|compute| compute.get(key))
            .and_then(Value::as_str)
        else {
            continue;
        };
        let (status, detail) = match fact {
            None => (
                NeedStatus::CouldNotBeChecked,
                format!("no fact about {key} was given"),
            ),
            Some(have) => (
                at_least(size_bytes(have), size_bytes(wanted)),
                format!("needs at least {wanted}, has {have}"),
            ),
        };
        stated.push(
            &format!("runtime.compute.{key}"),
            &["runtime", "compute", key],
            status,
            None,
            spare,
            reason,
            detail,
        );
    }
    if let Some(gpu) = compute.and_then(|compute| compute.get("gpu")) {
        let (status, connector, detail) = gpu_status(gpu, facts);
        stated.push(
            "runtime.compute.gpu",
            &["runtime", "compute", "gpu"],
            status,
            connector,
            spare || is_true(gpu.get("optional")),
            reason,
            detail,
        );
    }
    for (tool, entry) in runtime
        .get("software")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        let wanted = entry.get("version").and_then(Value::as_str).unwrap_or("");
        let (status, detail) = match facts.software.as_ref().map(|software| software.get(tool)) {
            None => (
                NeedStatus::CouldNotBeChecked,
                "no fact about software was given".to_string(),
            ),
            Some(None) => (
                NeedStatus::NotMet,
                format!("needs {tool} {wanted}, and {tool} is not there"),
            ),
            Some(Some(None)) => (
                NeedStatus::CouldNotBeChecked,
                format!("needs {tool} {wanted}, and the version that is there is not known"),
            ),
            Some(Some(Some(have))) => (
                match satisfies(have, wanted) {
                    Some(true) => NeedStatus::MetHere,
                    Some(false) => NeedStatus::NotMet,
                    None => NeedStatus::CouldNotBeChecked,
                },
                format!("needs {tool} {wanted}, has {have}"),
            ),
        };
        stated.push(
            &format!("runtime.software.{tool}"),
            &["runtime", "software", tool],
            status,
            None,
            is_true(entry.get("optional")),
            entry.get("reason"),
            detail,
        );
    }
    if let Some(wanted) = runtime.get("network").and_then(Value::as_str) {
        let reach = |name: &str| NETWORK_REACH.iter().position(|reach| *reach == name);
        let have = facts.network.as_deref();
        let (status, detail) = match (have.and_then(reach), reach(wanted)) {
            (Some(have_reach), Some(wanted_reach)) => (
                if have_reach >= wanted_reach {
                    NeedStatus::MetHere
                } else {
                    NeedStatus::NotMet
                },
                format!("asks for {wanted}, is allowed {}", have.unwrap_or("")),
            ),
            _ => (
                NeedStatus::CouldNotBeChecked,
                "no fact about the network reach was given".to_string(),
            ),
        };
        stated.push(
            "runtime.network",
            &["runtime", "network"],
            status,
            None,
            false,
            None,
            detail,
        );
    }
    if let Some(harness) = runtime.get("harness") {
        // A name means what the platform that reads it says it means. A name the receiver does
        // not know could not be checked.
        let wanted = harness.get("name").and_then(Value::as_str).unwrap_or("");
        let mut names: Vec<&str> = facts.harness.iter().map(String::as_str).collect();
        for alias in &facts.harness_aliases {
            if !names.contains(&alias.as_str()) {
                names.push(alias);
            }
        }
        let (status, detail) = if names.contains(&wanted) {
            (NeedStatus::MetHere, format!("this harness is {wanted}"))
        } else if !names.is_empty() && facts.known.harnesses.iter().any(|known| known == wanted) {
            (
                NeedStatus::NotMet,
                format!("asks for {wanted}, this harness is {}", names.join(" or ")),
            )
        } else {
            (
                NeedStatus::CouldNotBeChecked,
                format!("the harness name {wanted} is not one this receiver knows"),
            )
        };
        stated.push(
            "runtime.harness",
            &["runtime", "harness"],
            status,
            None,
            is_true(harness.get("optional")),
            harness.get("reason"),
            detail,
        );
    }
}

/// What the agent needs from any model.
fn model_lines(model: &Value, facts: &Facts, stated: &mut Stated<'_>) {
    let running = facts.model.as_ref();
    let has: Option<Vec<&str>> = running
        .and_then(|running| running.get("capabilities"))
        .and_then(Value::as_array)
        .map(|listed| listed.iter().filter_map(Value::as_str).collect());
    for (index, capability) in model
        .get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let Some(capability) = capability.as_str() else {
            continue;
        };
        let (status, detail) = match &has {
            None => (
                NeedStatus::CouldNotBeChecked,
                "no fact about the model's capabilities was given".to_string(),
            ),
            Some(has) if has.contains(&capability) => {
                (NeedStatus::MetHere, format!("the model has {capability}"))
            }
            Some(_)
                if facts
                    .known
                    .capabilities
                    .iter()
                    .any(|known| known == capability) =>
            {
                (
                    NeedStatus::NotMet,
                    format!("the model does not have {capability}"),
                )
            }
            Some(_) => (
                NeedStatus::CouldNotBeChecked,
                format!("the capability name {capability} is not one this receiver knows"),
            ),
        };
        stated.push(
            &format!("model.capabilities.{capability}"),
            &["model", "capabilities", &index.to_string()],
            status,
            None,
            false,
            None,
            detail,
        );
    }
    if let Some(wanted) = model.get("min_context_tokens") {
        let have = running.and_then(|running| running.get("context_tokens"));
        let (status, detail) = match have {
            None => (
                NeedStatus::CouldNotBeChecked,
                "no fact about the model's context was given".to_string(),
            ),
            Some(have) => (
                at_least(have.as_f64(), wanted.as_f64()),
                format!("needs at least {wanted} tokens, the model has {have}"),
            ),
        };
        stated.push(
            "model.min_context_tokens",
            &["model", "min_context_tokens"],
            status,
            None,
            false,
            None,
            detail,
        );
    }
}

/// One line per need of the agent and, separately, per need of the definition. Pure.
///
/// Nothing is combined or weakened: where both documents state memory, each need has a line
/// and a status of its own. A need a definition states is named `definition.runtime.<...>`.
pub(crate) fn need_lines(
    agent: &Value,
    definition: Option<&Value>,
    facts: &Facts,
) -> Vec<NeedLine> {
    let mut lines = Vec::new();
    let name = |document: &Value| {
        document
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let agent_name = name(agent);
    let mut stated = Stated {
        by: StatedBy::Agent,
        document: &agent_name,
        prefix: "",
        lines: &mut lines,
    };
    if let Some(runtime) = agent.get("runtime") {
        runtime_lines(runtime, facts, &mut stated);
    }
    if let Some(model) = agent.get("model") {
        model_lines(model, facts, &mut stated);
    }
    if let Some(definition) = definition {
        let definition_name = name(definition);
        let mut stated = Stated {
            by: StatedBy::Definition,
            document: &definition_name,
            prefix: "definition.",
            lines: &mut lines,
        };
        if let Some(runtime) = definition.get("runtime") {
            runtime_lines(runtime, facts, &mut stated);
        }
    }
    lines
}

/// Whether the needs let the agent start under the two settings.
///
/// A required need that is not met follows `if_unsupported`. A required need that could not be
/// checked follows `if_unverified`, and is never counted as not met. A need marked optional
/// never stops a start.
///
/// The mapping applies the same rule to every declared thing, need or not, so only the tests
/// call this form. It states the rule for needs alone, the way the format's cases do.
#[cfg(test)]
pub(crate) fn decide(
    lines: &[NeedLine],
    if_unsupported: PackageAction,
    if_unverified: PackageAction,
) -> Decision {
    let required = |status: NeedStatus| -> Vec<String> {
        lines
            .iter()
            .filter(|line| !line.optional && line.status == status)
            .map(|line| line.need.clone())
            .collect()
    };
    let required_not_met = required(NeedStatus::NotMet);
    let required_not_checked = required(NeedStatus::CouldNotBeChecked);
    let stops = (!required_not_met.is_empty() && if_unsupported == PackageAction::Stop)
        || (!required_not_checked.is_empty() && if_unverified == PackageAction::Stop);
    Decision {
        starts: !stops,
        required_not_met,
        required_not_checked,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn facts(value: Value) -> Facts {
        serde_json::from_value(value).expect("facts")
    }

    fn agent(runtime: Value) -> Value {
        json!({"type": "agent", "name": "video-producer", "runtime": runtime})
    }

    fn statuses(lines: &[NeedLine]) -> Vec<(String, String)> {
        lines
            .iter()
            .map(|line| {
                (
                    line.need.clone(),
                    line.status.format_text(line.connector.as_deref()),
                )
            })
            .collect()
    }

    fn status_of(document: &Value, facts_value: Value) -> String {
        let lines = need_lines(document, None, &facts(facts_value));
        assert_eq!(lines.len(), 1, "{lines:?}");
        lines[0].status.format_text(lines[0].connector.as_deref())
    }

    #[test]
    fn sizes_are_read_in_bytes() {
        assert_eq!(size_bytes("1kb"), Some(1024.0));
        assert_eq!(size_bytes("1.5gb"), Some(1.5 * 1024.0 * 1024.0 * 1024.0));
        assert_eq!(size_bytes("2tb"), Some(2.0 * 1024f64.powi(4)));
        for text in [
            "",
            "gb",
            "16",
            "16 gb",
            "16GB",
            "1e3gb",
            "-1gb",
            "16g\u{e9}",
        ] {
            assert_eq!(size_bytes(text), None, "{text:?}");
        }
    }

    #[test]
    fn a_number_is_a_floor_and_no_fact_means_not_checked() {
        let document = agent(json!({"compute": {
            "cpus": 8, "memory": "32gb", "storage": "100gb",
            "reason": "Generated scenes are large."
        }}));
        let lines = need_lines(
            &document,
            None,
            &facts(json!({"cpus": 8, "memory": "16gb"})),
        );
        assert_eq!(
            statuses(&lines),
            [
                ("runtime.compute.cpus".to_string(), "met here".to_string()),
                ("runtime.compute.memory".to_string(), "not met".to_string()),
                (
                    "runtime.compute.storage".to_string(),
                    "could not be checked".to_string()
                ),
            ]
        );
        assert_eq!(lines[1].pointer, "/runtime/compute/memory");
        assert_eq!(
            lines[1].reason.as_deref(),
            Some("Generated scenes are large.")
        );
        assert_eq!(
            lines[1].detail.as_deref(),
            Some("needs at least 32gb, has 16gb")
        );
        assert_eq!(lines[1].stated_by, StatedBy::Agent);
        assert_eq!(lines[1].document, "video-producer");
        assert!(!lines[1].optional);
        // A fact that is not a size cannot be compared.
        assert_eq!(
            status_of(
                &agent(json!({"compute": {"memory": "32gb"}})),
                json!({"memory": "plenty"})
            ),
            "could not be checked"
        );
    }

    #[test]
    fn optional_on_compute_covers_the_block_and_on_gpu_only_the_gpu() {
        let document =
            agent(json!({"compute": {"cpus": 2, "gpu": {"count": 1}, "optional": true}}));
        let lines = need_lines(&document, None, &Facts::default());
        assert!(lines.iter().all(|line| line.optional));
        let document =
            agent(json!({"compute": {"cpus": 2, "gpu": {"count": 1, "optional": true}}}));
        let lines = need_lines(&document, None, &Facts::default());
        assert_eq!(
            lines.iter().map(|line| line.optional).collect::<Vec<_>>(),
            [false, true]
        );
    }

    #[test]
    fn software_is_not_met_when_absent_and_not_checked_when_its_version_is_unknown() {
        let document = agent(json!({"software": {
            "ffmpeg": {"version": ">= 6.0", "reason": "Trims audio.", "optional": true},
            "node": {"version": ">= 20.0"},
            "python3": {"version": ">= 3.11"},
            "uv": {"version": ">= 0.4"}
        }}));
        let none = need_lines(&document, None, &Facts::default());
        assert!(
            none.iter()
                .all(|line| line.status == NeedStatus::CouldNotBeChecked)
        );
        let lines = need_lines(
            &document,
            None,
            &facts(json!({"software": {"ffmpeg": "7.1", "node": "18.2.0", "uv": null}})),
        );
        assert_eq!(
            statuses(&lines),
            [
                (
                    "runtime.software.ffmpeg".to_string(),
                    "met here".to_string()
                ),
                ("runtime.software.node".to_string(), "not met".to_string()),
                (
                    "runtime.software.python3".to_string(),
                    "not met".to_string()
                ),
                (
                    "runtime.software.uv".to_string(),
                    "could not be checked".to_string()
                ),
            ]
        );
        assert!(lines[0].optional);
        assert_eq!(lines[0].reason.as_deref(), Some("Trims audio."));
        assert_eq!(lines[1].pointer, "/runtime/software/node");
        // A version that has no order against the floor.
        assert_eq!(
            status_of(
                &agent(json!({"software": {"node": {"version": ">= 20.0"}}})),
                json!({"software": {"node": "22.0.0-nightly"}})
            ),
            "could not be checked"
        );
    }

    #[test]
    fn the_network_is_met_by_that_reach_or_more() {
        let document = agent(json!({"network": "internal"}));
        assert_eq!(
            status_of(&document, json!({"network": "external"})),
            "met here"
        );
        assert_eq!(
            status_of(&document, json!({"network": "internal"})),
            "met here"
        );
        assert_eq!(status_of(&document, json!({"network": "none"})), "not met");
        assert_eq!(status_of(&document, json!({})), "could not be checked");
        assert_eq!(
            status_of(&document, json!({"network": "open"})),
            "could not be checked"
        );
    }

    #[test]
    fn a_harness_name_is_met_not_met_or_unknown() {
        let document = agent(
            json!({"harness": {"name": "claude-code", "reason": "Evaluated on it.", "optional": true}}),
        );
        assert_eq!(
            status_of(&document, json!({"harness": "claude-code"})),
            "met here"
        );
        assert_eq!(
            status_of(
                &document,
                json!({"harness": "hermes", "harness_aliases": ["claude-code"]})
            ),
            "met here"
        );
        assert_eq!(
            status_of(
                &document,
                json!({"harness_aliases": ["nvidia.fabric.claude", "claude"]})
            ),
            "could not be checked"
        );
        assert_eq!(
            status_of(
                &document,
                json!({"harness": "hermes", "known": {"harnesses": ["claude-code"]}})
            ),
            "not met"
        );
        // A known name, and no name for this harness: nothing to compare it with.
        assert_eq!(
            status_of(&document, json!({"known": {"harnesses": ["claude-code"]}})),
            "could not be checked"
        );
        assert_eq!(
            status_of(&document, json!({"harness": "hermes"})),
            "could not be checked"
        );
        let lines = need_lines(&document, None, &Facts::default());
        assert!(lines[0].optional);
        assert_eq!(lines[0].reason.as_deref(), Some("Evaluated on it."));
    }

    #[test]
    fn a_gpu_is_met_here_remotely_not_met_or_unknown() {
        let need =
            json!({"interface": {"cuda": {"version": ">= 12.0"}}, "memory": "16gb", "count": 1});
        let document = agent(json!({"compute": {"gpu": need}}));
        let l4 = json!({"interface": {"cuda": {"version": "12.4"}}, "memory": "24gb", "family": "ampere", "model": "l4"});
        let small = json!({"interface": {"cuda": {"version": "12.4"}}, "memory": "8gb"});
        assert_eq!(status_of(&document, json!({})), "could not be checked");
        assert_eq!(status_of(&document, json!({"gpus": [l4]})), "met here");
        assert_eq!(status_of(&document, json!({"gpus": [small]})), "not met");
        assert_eq!(status_of(&document, json!({"gpus": []})), "not met");
        let mut remote = l4.clone();
        remote["connector"] = json!("transcription");
        assert_eq!(
            status_of(&document, json!({"gpus": [small], "remote_gpus": [remote]})),
            "met remotely through connector transcription"
        );
        // An interface version that was not reported cannot be compared.
        assert_eq!(
            status_of(
                &document,
                json!({"gpus": [{"interface": {"cuda": {}}, "memory": "24gb"}]})
            ),
            "could not be checked"
        );
        // A GPU that offers another well-known interface does not meet the need.
        assert_eq!(
            status_of(
                &document,
                json!({"gpus": [{"interface": {"metal": {"version": "3.0"}}, "memory": "24gb"}]})
            ),
            "not met"
        );
    }

    #[test]
    fn a_gpu_fact_with_no_interface_member_is_no_fact_about_interfaces() {
        let need = json!({"interface": {"cuda": {"version": ">= 12.0"}}, "memory": "16gb"});
        let document = agent(json!({"compute": {"gpu": need}}));

        // The machine probe gives this shape: memory and nothing else. There
        // is no fact to check the interface with, so the need is never "not met".
        assert_eq!(
            status_of(&document, json!({"gpus": [{"memory": "24gb"}]})),
            "could not be checked"
        );
        // A fact that lists its interfaces and has no cuda does not offer it.
        assert_eq!(
            status_of(
                &document,
                json!({"gpus": [{"interface": {}, "memory": "24gb"}]})
            ),
            "not met"
        );
        // Another field that fails still decides: 8gb is less than 16gb.
        assert_eq!(
            status_of(&document, json!({"gpus": [{"memory": "8gb"}]})),
            "not met"
        );
        // A models need is unknown too when the fact names no model.
        let pinned = agent(json!({"compute": {"gpu": {"models": ["l4"]}}}));
        assert_eq!(
            status_of(&pinned, json!({"gpus": [{"memory": "24gb"}]})),
            "could not be checked"
        );
    }

    #[test]
    fn gpu_access_and_count_say_which_gpus_count() {
        let fits = json!({"memory": "24gb"});
        let mut remote = fits.clone();
        remote["connector"] = json!("transcription");
        let with = |access: &str, count: u64| {
            agent(json!({"compute": {"gpu": {"memory": "16gb", "access": access, "count": count}}}))
        };
        let both = json!({"gpus": [fits], "remote_gpus": [remote]});
        assert_eq!(status_of(&with("local", 1), both.clone()), "met here");
        assert_eq!(
            status_of(&with("remote", 1), both.clone()),
            "met remotely through connector transcription"
        );
        assert_eq!(status_of(&with("any", 1), both.clone()), "met here");
        // Two GPUs must be in one place.
        assert_eq!(status_of(&with("any", 2), both.clone()), "not met");
        assert_eq!(
            status_of(
                &with("local", 1),
                json!({"remote_gpus": both["remote_gpus"]})
            ),
            "not met"
        );
    }

    #[test]
    fn gpu_families_and_models_take_any_one_listed_name() {
        let document = agent(
            json!({"compute": {"gpu": {"families": ["ampere", "ada"], "models": ["l4", "a10g"]}}}),
        );
        let gpu = |family: &str, model: &str| json!({"gpus": [{"family": family, "model": model}]});
        let mut known = gpu("ampere", "l4");
        assert_eq!(status_of(&document, known.clone()), "met here");
        assert_eq!(status_of(&document, gpu("ampere", "t4")), "not met");
        // A family name this receiver does not list as known cannot be ruled out.
        assert_eq!(
            status_of(&document, gpu("hopper", "l4")),
            "could not be checked"
        );
        known = gpu("hopper", "l4");
        known["known"] = json!({"families": ["ampere", "ada", "hopper"]});
        assert_eq!(status_of(&document, known), "not met");
        assert_eq!(
            status_of(&document, json!({"gpus": [{"family": "ampere"}]})),
            "could not be checked"
        );
    }

    #[test]
    fn what_the_agent_needs_from_a_model_is_checked_against_the_model_facts() {
        let document = json!({
            "type": "agent", "name": "video-producer",
            "model": {"capabilities": ["tool-use", "vision"], "min_context_tokens": 32000}
        });
        let lines = need_lines(&document, None, &Facts::default());
        assert!(
            lines
                .iter()
                .all(|line| line.status == NeedStatus::CouldNotBeChecked)
        );
        let lines = need_lines(
            &document,
            None,
            &facts(json!({
                "model": {"name": "ollama::small", "capabilities": ["tool-use"], "context_tokens": 8000},
                "known": {"capabilities": ["vision"]}
            })),
        );
        assert_eq!(
            statuses(&lines),
            [
                (
                    "model.capabilities.tool-use".to_string(),
                    "met here".to_string()
                ),
                (
                    "model.capabilities.vision".to_string(),
                    "not met".to_string()
                ),
                (
                    "model.min_context_tokens".to_string(),
                    "not met".to_string()
                ),
            ]
        );
        assert_eq!(lines[1].pointer, "/model/capabilities/1");
        // A capability name the receiver does not know is not counted as not met.
        let lines = need_lines(
            &document,
            None,
            &facts(json!({"model": {"capabilities": ["tool-use"], "context_tokens": 64000}})),
        );
        assert_eq!(lines[1].status, NeedStatus::CouldNotBeChecked);
        assert_eq!(lines[2].status, NeedStatus::MetHere);
    }

    #[test]
    fn the_needs_of_a_definition_have_lines_of_their_own() {
        let document = agent(json!({"compute": {"memory": "8gb"}}));
        let definition = json!({
            "type": "agent-definition", "name": "weather-large",
            "runtime": {"compute": {"memory": "48gb", "reason": "The larger model runs beside the agent."}}
        });
        let lines = need_lines(
            &document,
            Some(&definition),
            &facts(json!({"memory": "16gb"})),
        );
        assert_eq!(
            statuses(&lines),
            [
                ("runtime.compute.memory".to_string(), "met here".to_string()),
                (
                    "definition.runtime.compute.memory".to_string(),
                    "not met".to_string()
                ),
            ]
        );
        assert_eq!(lines[1].stated_by, StatedBy::Definition);
        assert_eq!(lines[1].document, "weather-large");
        assert_eq!(lines[1].pointer, "/runtime/compute/memory");
        assert_eq!(
            lines[1].reason.as_deref(),
            Some("The larger model runs beside the agent.")
        );
    }

    #[test]
    fn the_two_settings_decide_whether_the_agent_starts() {
        let document = agent(json!({
            "compute": {"memory": "32gb"},
            "software": {"ffmpeg": {"version": ">= 6.0", "optional": true}},
            "network": "internal"
        }));
        let lines = need_lines(
            &document,
            None,
            &facts(json!({"memory": "16gb", "software": {}})),
        );
        let decision = decide(&lines, PackageAction::Stop, PackageAction::Warn);
        assert!(!decision.starts);
        assert_eq!(decision.required_not_met, ["runtime.compute.memory"]);
        assert_eq!(decision.required_not_checked, ["runtime.network"]);
        assert!(decide(&lines, PackageAction::Warn, PackageAction::Warn).starts);
        assert!(!decide(&lines, PackageAction::Warn, PackageAction::Stop).starts);

        // An optional need that is not met never stops a start.
        let optional =
            agent(json!({"software": {"ffmpeg": {"version": ">= 6.0", "optional": true}}}));
        let lines = need_lines(&optional, None, &facts(json!({"software": {}})));
        assert_eq!(lines[0].status, NeedStatus::NotMet);
        let decision = decide(&lines, PackageAction::Stop, PackageAction::Stop);
        assert!(decision.starts);
        assert!(decision.required_not_met.is_empty());
    }

    #[test]
    fn an_agent_with_no_needs_has_no_lines() {
        let lines = need_lines(
            &json!({"type": "agent", "name": "plain"}),
            None,
            &Facts::default(),
        );
        assert!(lines.is_empty());
        assert!(decide(&lines, PackageAction::Stop, PackageAction::Stop).starts);
    }
}
