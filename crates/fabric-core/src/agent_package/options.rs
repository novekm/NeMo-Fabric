// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Options of one agent package load.

use std::collections::BTreeMap;
use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::model::PackageAction;
use crate::config::InstructionMode;

/// What the caller chooses for one agent package load.
///
/// Every member may be left out. The package has no field for any of these:
/// they belong to whoever starts the agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct AgentPackageOptions {
    /// The entry document, as a path inside the package, when it is not at the root
    /// (for example a definition kept at `variants/short/agent-definition.json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry: Option<PathBuf>,
    /// Values from whoever starts the agent, by variable name.
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub variables: serde_json::Map<String, Value>,
    /// How the instruction text is applied. Left out: replace when the adapter
    /// offers it, else append.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction_mode: Option<InstructionMode>,
    /// What to do about a required thing whose status is `not_mapped` or
    /// `not_met`. Default `stop`.
    pub if_unsupported: PackageAction,
    /// What to do about a required thing whose status is
    /// `could_not_be_checked`. Default `warn`.
    pub if_unverified: PackageAction,
    /// Probe this machine for runtime facts. Default false. The probe does
    /// not look for installed software packages.
    pub check_machine: bool,
    /// Facts the caller knows about the place the agent runs. They win over
    /// probed facts, key by key. The `packages` fact lists the installed
    /// software packages by registry.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(extend(
        "additionalProperties" = false,
        "properties" = super::facts::facts_schema_properties()
    ))]
    pub facts: Option<serde_json::Map<String, Value>>,
    /// The sha256 the caller holds for the lock file, as 64 lowercase hex
    /// characters, with or without `sha256:` before them. The load is refused
    /// when the lock file in the package has another digest.
    ///
    /// The lock sits in the folder it describes. Without this value the check
    /// shows that the parts match that lock, and no more.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_lock_sha256: Option<String>,
    /// Targets of the caller's own for custom tools and called agents the
    /// package declares, by the place of the declared thing.
    ///
    /// A key is `custom_tools.<name>` or `agents.<name>` for the agent that
    /// runs. For a thing a called agent declares, the same follows the called
    /// agent's key: `agents.<name>.custom_tools.<tool>`. A value is one line
    /// of text that names the caller's target, for example a Python reference
    /// or a registered name. The loader never imports, resolves or runs it.
    /// A key and a value each hold at most 1024 bytes and no control,
    /// separator or hidden format character.
    ///
    /// A bound thing has status `bound` in the report and does not stop the
    /// load. A key that names nothing the package declares is refused. A
    /// binding changes nothing else: no need, no pin, no schema finding and
    /// no other declared thing. Everything else a bound called agent declares
    /// keeps a line of its own in the report, and no binding accounts for it.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub bindings: BTreeMap<String, String>,
    /// A program that decides whether the loaded package may start: the
    /// program, then its arguments. Left out: no checker is asked. An empty
    /// list is refused.
    ///
    /// The checker is asked once, after the package passed its own checks and
    /// was mapped. It is started with no shell. It gets the package folder in
    /// `NEMO_FABRIC_PACKAGE_DIR`, the path of a copy of the load report
    /// without its config in `NEMO_FABRIC_PACKAGE_REPORT`, and the name of the
    /// entry document in `NEMO_FABRIC_AGENT_NAME`. The loader adds nothing to
    /// its arguments. Of the caller's environment it keeps operating-system
    /// variables such as `PATH` and `HOME`, and no other.
    ///
    /// The checker writes one JSON object to standard output. Exit code 0
    /// with `"decision": "allow"` lets the package start. Exit code 2 with
    /// `"decision": "deny"` does not. Anything else is a deny, reported as a
    /// checker that failed. The report holds the answer under
    /// `checker_result`, and a load the checker does not allow has outcome
    /// `denied` and no config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checker: Option<Vec<String>>,
    /// A checker reached over HTTP, in place of a program: an `http://` URL
    /// such as `http://127.0.0.1:8100/v1/hooks/agent-start`. Left out: no
    /// HTTP checker is asked. Naming it together with `checker` is refused.
    ///
    /// The loader posts plain facts about the agent, gathered from the load:
    /// the agent's name, each part with its type, source, version, digest
    /// and whether the source is local, the models named, and the policies
    /// with the enforcers each needs. The body is the request of the NeMo
    /// Guardrails hooks server's `/v1/hooks/agent-start` endpoint. A `200`
    /// answer with `"decision": "allow"` lets the package start, and one
    /// with `"decision": "deny"` does not. Anything else is a deny, reported
    /// as a checker that failed: a checker that cannot be reached, another
    /// status, an answer with no decision, or no answer in time. The address
    /// comes from these options, never from the package. Only `http` is
    /// spoken; reach a remote checker through a local address.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checker_url: Option<String>,
    /// The enforcers that run where the agent starts, such as `openshell`,
    /// sent to the checker URL as `enforcers_present`. Refused without
    /// `checker_url`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checker_enforcers: Vec<String>,
    /// The time limit of the checker, in seconds. Default 30. A checker that
    /// runs longer is ended, and the load is denied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checker_timeout_seconds: Option<f64>,
}

impl Default for AgentPackageOptions {
    fn default() -> Self {
        Self {
            entry: None,
            variables: serde_json::Map::new(),
            instruction_mode: None,
            if_unsupported: PackageAction::Stop,
            if_unverified: PackageAction::Warn,
            check_machine: false,
            facts: None,
            expected_lock_sha256: None,
            bindings: BTreeMap::new(),
            checker: None,
            checker_url: None,
            checker_enforcers: Vec::new(),
            checker_timeout_seconds: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_stop_if_unsupported_and_warn_if_unverified() {
        let options: AgentPackageOptions = serde_json::from_str("{}").expect("empty options");

        assert_eq!(options, AgentPackageOptions::default());
        assert_eq!(options.if_unsupported, PackageAction::Stop);
        assert_eq!(options.if_unverified, PackageAction::Warn);
        assert!(!options.check_machine);
        assert!(options.bindings.is_empty());
    }

    #[test]
    fn a_binding_target_is_text_and_nothing_else() {
        for target in [
            "3",
            "true",
            r#"{"ref": "app.tools:ask"}"#,
            r#"["app.tools:ask"]"#,
        ] {
            let text = format!(r#"{{"bindings": {{"custom_tools.ask": {target}}}}}"#);

            serde_json::from_str::<AgentPackageOptions>(&text).expect_err(target);
        }
    }

    #[test]
    fn a_checker_is_a_list_of_text_with_a_time_limit_in_seconds() {
        for text in [
            r#"{"checker": "/opt/policy/bin/check-package --policy /sandbox/policy"}"#,
            r#"{"checker": ["/opt/policy/bin/check-package", 3]}"#,
            r#"{"checker_timeout_seconds": "20"}"#,
        ] {
            serde_json::from_str::<AgentPackageOptions>(text).expect_err(text);
        }

        let options: AgentPackageOptions = serde_json::from_value(serde_json::json!({
            "checker": ["/opt/policy/bin/check-package", "--policy", "/sandbox/policy"],
            "checker_timeout_seconds": 20
        }))
        .expect("options");

        assert_eq!(
            options.checker.as_deref().expect("a checker"),
            [
                "/opt/policy/bin/check-package",
                "--policy",
                "/sandbox/policy"
            ]
        );
        assert_eq!(options.checker_timeout_seconds, Some(20.0));
        let encoded = serde_json::to_value(&options).expect("encode");
        assert_eq!(
            serde_json::from_value::<AgentPackageOptions>(encoded).expect("decode"),
            options
        );
        // An empty list is not the same as no checker: the load refuses it.
        let empty: AgentPackageOptions =
            serde_json::from_str(r#"{"checker": []}"#).expect("options");
        assert_eq!(empty.checker, Some(Vec::new()));
        assert_eq!(AgentPackageOptions::default().checker, None);
    }

    #[test]
    fn an_unknown_option_is_refused() {
        let error = serde_json::from_str::<AgentPackageOptions>(r#"{"caller_verified": true}"#)
            .expect_err("unknown option");

        assert!(error.to_string().contains("caller_verified"), "{error}");
    }

    #[test]
    fn options_round_trip_and_leave_unset_members_out() {
        let options: AgentPackageOptions = serde_json::from_value(serde_json::json!({
            "entry": "variants/short/agent-definition.json",
            "variables": {"max_items": 20},
            "instruction_mode": "append",
            "if_unsupported": "warn",
            "if_unverified": "stop",
            "check_machine": true,
            "facts": {"network": "internal"},
            "expected_lock_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "bindings": {"custom_tools.request_approval": "app.tools:request_approval"}
        }))
        .expect("options");

        assert_eq!(options.instruction_mode, Some(InstructionMode::Append));
        assert_eq!(options.if_unsupported, PackageAction::Warn);
        assert_eq!(options.if_unverified, PackageAction::Stop);
        assert_eq!(
            options.bindings["custom_tools.request_approval"],
            "app.tools:request_approval"
        );
        let encoded = serde_json::to_value(&options).expect("encode");
        assert_eq!(
            serde_json::from_value::<AgentPackageOptions>(encoded).expect("decode"),
            options
        );
        assert_eq!(
            serde_json::to_value(AgentPackageOptions::default()).expect("encode"),
            serde_json::json!({"if_unsupported": "stop", "if_unverified": "warn", "check_machine": false})
        );
    }
}
