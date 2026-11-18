// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ask the checker the caller named whether a loaded package may start.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::error::{PackageError, PackageErrorCode, PackageResult};
use super::http_checker::{self, Endpoint};
use super::json::{parse_document, read_regular_file};
use super::options::AgentPackageOptions;
use super::report::{AgentPackageReport, CheckerDecision, CheckerFailure, CheckerResult};

/// The variable that gives the checker the package folder.
const PACKAGE_DIR_VARIABLE: &str = "NEMO_FABRIC_PACKAGE_DIR";
/// The variable that gives the checker the path of the load report file.
const REPORT_VARIABLE: &str = "NEMO_FABRIC_PACKAGE_REPORT";
/// The variable that gives the checker the name of the entry document.
const AGENT_NAME_VARIABLE: &str = "NEMO_FABRIC_AGENT_NAME";
/// The operating-system variables a checker keeps. It gets no other value of
/// the caller's environment, so no credential held in a variable reaches it.
const KEPT_VARIABLES: &[&str] = &[
    "COMSPEC",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
    "USERPROFILE",
];
/// The time limit of a checker when the options give none.
const DEFAULT_LIMIT: Duration = Duration::from_secs(30);
/// The most a checker may write to standard output, and to standard error.
const MAX_OUTPUT_BYTES: u64 = 1024 * 1024;
/// The most of a checker's error text the report keeps, in characters.
const MAX_DETAIL_CHARS: usize = 2000;
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// The checker of one load: a program with its arguments, or a URL, and
/// its time limit.
#[derive(Debug)]
pub(super) struct Checker<'a> {
    asked: Asked<'a>,
    limit: Duration,
}

/// What a checker is.
#[derive(Debug)]
enum Asked<'a> {
    /// A program, then its arguments.
    Program {
        program: &'a str,
        arguments: &'a [String],
    },
    /// An HTTP checker, and the enforcers the caller says run here.
    Http {
        endpoint: Endpoint,
        enforcers: &'a [String],
    },
}

impl<'a> Checker<'a> {
    /// The checker the options name, if they name one. A time limit that is
    /// not a positive number of seconds is refused, with or without a
    /// checker. So is a checker with no program: an empty list never means
    /// that the package starts unchecked. So are a program and a URL
    /// together, a URL the loader cannot use, and enforcers with no URL.
    pub(super) fn from_options(options: &'a AgentPackageOptions) -> PackageResult<Option<Self>> {
        let limit = match options.checker_timeout_seconds {
            None => DEFAULT_LIMIT,
            Some(seconds) => Duration::try_from_secs_f64(seconds)
                .ok()
                .filter(|limit| !limit.is_zero())
                .ok_or_else(|| {
                    PackageError::new(
                        PackageErrorCode::OptionsInvalid,
                        "checker_timeout_seconds",
                        "the time limit of the checker must be a positive number of seconds",
                    )
                })?,
        };
        let invalid = |field, message: &str| {
            PackageError::new(PackageErrorCode::OptionsInvalid, field, message)
        };
        let command = match (&options.checker, &options.checker_url) {
            (Some(_), Some(_)) => {
                return Err(invalid(
                    "checker_url",
                    "name a checker program or a checker URL, not both",
                ));
            }
            (None, Some(url)) => {
                let endpoint =
                    Endpoint::parse(url).map_err(|message| invalid("checker_url", message))?;
                return Ok(Some(Self {
                    asked: Asked::Http {
                        endpoint,
                        enforcers: &options.checker_enforcers,
                    },
                    limit,
                }));
            }
            (_, None) if !options.checker_enforcers.is_empty() => {
                return Err(invalid(
                    "checker_enforcers",
                    "checker_enforcers is sent only to a checker URL; name checker_url",
                ));
            }
            (None, None) => return Ok(None),
            (Some(command), None) => command,
        };
        let Some((program, arguments)) = command.split_first() else {
            return Err(PackageError::new(
                PackageErrorCode::OptionsInvalid,
                "checker",
                "the checker must name a program. Leave the option out to ask no checker",
            ));
        };
        Ok(Some(Self {
            asked: Asked::Program { program, arguments },
            limit,
        }))
    }

    /// Ask the checker about a load that is ready to start. It never gives
    /// an error: a checker that cannot be asked, or that does not answer as
    /// it must, is a deny with the failure named.
    pub(super) fn ask(&self, report: &AgentPackageReport) -> CheckerResult {
        match &self.asked {
            Asked::Program { program, arguments } => self.ask_program(program, arguments, report),
            Asked::Http {
                endpoint,
                enforcers,
            } => http_checker::ask(endpoint, report, enforcers, self.limit),
        }
    }

    /// Ask a checker program, which reads a copy of `report` without its config.
    fn ask_program(
        &self,
        program: &str,
        arguments: &[String],
        report: &AgentPackageReport,
    ) -> CheckerResult {
        let failed = |failure, exit_code, detail, answer| CheckerResult {
            decision: CheckerDecision::Deny,
            failure: Some(failure),
            exit_code,
            detail,
            answer,
        };
        let scratch = match Scratch::with_report(report) {
            Ok(scratch) => scratch,
            Err(error) => {
                let detail = format!("the report for the checker cannot be written: {error}");
                return failed(CheckerFailure::NotStarted, None, Some(detail), None);
            }
        };
        let status = match self.run(program, arguments, report, &scratch) {
            Ok(status) => status,
            Err((failure, detail)) => {
                return failed(failure, None, detail.or_else(|| scratch.detail()), None);
            }
        };
        let exit_code = status.code();
        let answer = read_regular_file(&scratch.answer(), "checker", MAX_OUTPUT_BYTES)
            .and_then(|bytes| parse_document(&bytes, "checker"))
            .ok()
            .and_then(|answer| match answer {
                Value::Object(answer) => Some(answer),
                _ => None,
            });
        let decision = match exit_code {
            Some(0) => CheckerDecision::Allow,
            Some(2) => CheckerDecision::Deny,
            _ => {
                let failure = CheckerFailure::UnexpectedExit;
                return failed(failure, exit_code, scratch.detail(), answer);
            }
        };
        let stated = match decision {
            CheckerDecision::Allow => "allow",
            CheckerDecision::Deny => "deny",
        };
        let fits = answer
            .as_ref()
            .and_then(|answer| answer.get("decision"))
            .and_then(Value::as_str)
            == Some(stated);
        if !fits {
            let failure = CheckerFailure::InvalidAnswer;
            return failed(failure, exit_code, scratch.detail(), answer);
        }
        CheckerResult {
            decision,
            failure: None,
            exit_code,
            detail: None,
            answer,
        }
    }

    /// Run the checker until it ends, its time limit passes, or it wrote
    /// more than it may. An error is the failure, and its text when there
    /// is one to give.
    ///
    /// What the checker writes goes to two files, so an answer never waits
    /// on a full pipe and a program the checker started cannot hold the
    /// load. The limits end the checker, not a program it started.
    ///
    /// A checker that wrote more than it may has one result, however it
    /// ended: [`CheckerFailure::OutputTooLarge`], and no exit code. So what
    /// it wrote is looked at on every pass, also on the pass that finds it
    /// ended, and before its exit code is used. What a program it started
    /// wrote to the two files by then counts as the checker's own.
    fn run(
        &self,
        program: &str,
        arguments: &[String],
        report: &AgentPackageReport,
        scratch: &Scratch,
    ) -> Result<ExitStatus, (CheckerFailure, Option<String>)> {
        let not_started = |error: std::io::Error| {
            let detail = format!("the checker cannot be started: {error}");
            (CheckerFailure::NotStarted, Some(detail))
        };
        let answer = fs::File::create(scratch.answer()).map_err(not_started)?;
        let errors = fs::File::create(scratch.errors()).map_err(not_started)?;
        let mut child = Command::new(program)
            .args(arguments)
            .env_clear()
            .envs(
                KEPT_VARIABLES
                    .iter()
                    .filter_map(|name| Some((name, std::env::var_os(name)?))),
            )
            .env(PACKAGE_DIR_VARIABLE, &report.package_dir)
            .env(REPORT_VARIABLE, scratch.report())
            .env(AGENT_NAME_VARIABLE, &report.entry.name)
            .stdin(Stdio::null())
            .stdout(answer)
            .stderr(errors)
            .spawn()
            .map_err(not_started)?;
        let started = Instant::now();
        let ended = loop {
            // Whether it ended is asked first. A checker that ended writes no
            // more itself, so the look that follows is the last word on what it
            // wrote. A program it left behind can still write to the two files.
            let state = child.try_wait();
            if scratch.oversized() {
                let detail = format!("the checker wrote more than {MAX_OUTPUT_BYTES} bytes");
                break (CheckerFailure::OutputTooLarge, Some(detail));
            }
            match state {
                Ok(Some(status)) => return Ok(status),
                Ok(None) if started.elapsed() < self.limit => std::thread::sleep(POLL_INTERVAL),
                Ok(None) => break (CheckerFailure::TimedOut, None),
                Err(error) => break (CheckerFailure::UnexpectedExit, Some(error.to_string())),
            }
        };
        let _ = child.kill();
        let _ = child.wait();
        Err(ended)
    }
}

/// The private folder of one check: the report the checker reads, and what
/// it writes to standard output and to standard error. The folder is removed
/// when the value is dropped.
struct Scratch {
    path: PathBuf,
}

impl Scratch {
    /// Make the folder and write `report` into it without its config, which
    /// can hold values a checker has no need for.
    fn with_report(report: &AgentPackageReport) -> std::io::Result<Self> {
        let mut encoded = serde_json::to_value(report)?;
        if let Some(members) = encoded.as_object_mut() {
            members.remove("config");
        }
        let path = std::env::temp_dir().join(crate::runtime::new_id("fabric-package-check"));
        #[cfg_attr(not(unix), allow(unused_mut))]
        let mut builder = fs::DirBuilder::new();
        // The report names what the package declares. No other user reads it.
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(&path)?;
        let scratch = Self { path };
        fs::write(scratch.report(), serde_json::to_vec_pretty(&encoded)?)?;
        Ok(scratch)
    }

    fn report(&self) -> PathBuf {
        self.path.join("report.json")
    }

    fn answer(&self) -> PathBuf {
        self.path.join("answer.json")
    }

    fn errors(&self) -> PathBuf {
        self.path.join("errors")
    }

    /// Whether the checker wrote more than it may to either of its files.
    fn oversized(&self) -> bool {
        [self.answer(), self.errors()].iter().any(|file| {
            fs::symlink_metadata(file).is_ok_and(|written| written.len() > MAX_OUTPUT_BYTES)
        })
    }

    /// More about a checker that failed: the end of what it wrote to
    /// standard error.
    fn detail(&self) -> Option<String> {
        let errors = read_regular_file(&self.errors(), "checker", MAX_OUTPUT_BYTES).ok()?;
        let errors = String::from_utf8_lossy(&errors);
        let errors = errors.trim();
        let skipped = errors.chars().count().saturating_sub(MAX_DETAIL_CHARS);
        (!errors.is_empty()).then(|| errors.chars().skip(skipped).collect())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use serde_json::json;

    use super::super::load::load_agent_package;
    use super::super::model::PackageAction;
    use super::super::reader::testing::{Package, agent};
    use super::super::report::PackageOutcome;
    use super::*;
    use crate::config::{FabricConfig, ResolveContext};
    use crate::error::FabricError;

    /// How long a checker sleeps that a limit must end, in seconds, and the time limit of
    /// every test that is not about the time limit. No test here reads a clock: a load
    /// that returns while its checker sleeps is the proof that the loader ended it.
    const A_DAY: u32 = 86_400;
    const ALLOW: &str = r#"printf '{"decision": "allow", "rules": ["platform-base"]}'"#;
    const DENY: &str = r#"printf '{"decision": "deny", "reasons": ["the skill review is not on the approved list"]}'; exit 2"#;

    /// The test adapter. It accepts models, mcp and skills.
    fn context() -> ResolveContext {
        ResolveContext::new(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/hermes-shim-agent"),
        )
    }

    fn config() -> FabricConfig {
        serde_json::from_value(json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "code-reviewer"},
            "harness": {"adapter_id": "test.fabric.hermes_shim"},
            "discovery": {"local_paths": ["adapters"]},
            "environment": {
                "provider": "local",
                "env": {"REVIEW_API_TOKEN": "a-value-of-the-callers-config"},
            },
            "runtime": {},
        }))
        .expect("a config")
    }

    /// A code review agent with one skill.
    fn code_reviewer(name: &str) -> Package {
        let package = Package::new(name);
        package.write("skills/review/SKILL.md", "# Review\n");
        package.write_json(
            "agent.json",
            &agent(
                "code-reviewer",
                json!({"skills": {"review": {"source": "./skills/review"}}}),
            ),
        );
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &["$/skills/review"],
                ),
                package.folder_entry(
                    "$/skills/review",
                    "skill",
                    "skills/review",
                    "./skills/review",
                ),
            ],
        );
        package
    }

    /// Options that name a shell script as the checker. `more` follows the
    /// script as its arguments: the first one is `$0` in the script. The time
    /// limit is a day, so no test depends on how fast a shell starts.
    fn checked_by(script: &str, more: &[&str]) -> AgentPackageOptions {
        AgentPackageOptions {
            checker: Some(
                ["/bin/sh", "-c", script]
                    .into_iter()
                    .chain(more.iter().copied())
                    .map(str::to_string)
                    .collect(),
            ),
            checker_timeout_seconds: Some(f64::from(A_DAY)),
            ..AgentPackageOptions::default()
        }
    }

    fn load(package: &Package, options: AgentPackageOptions) -> AgentPackageReport {
        load_agent_package(&package.root, Some(config()), options, context())
            .expect("the package loads")
    }

    #[test]
    fn a_checker_that_allows_leaves_the_load_ready_and_its_answer_in_the_report() {
        let package = code_reviewer("checker-allows");

        let report = load(&package, checked_by(ALLOW, &[]));

        assert_eq!(report.outcome, PackageOutcome::Ready);
        assert!(report.config.is_some());
        assert_eq!(
            report.checker_result,
            Some(CheckerResult {
                decision: CheckerDecision::Allow,
                failure: None,
                exit_code: Some(0),
                detail: None,
                answer: json!({"decision": "allow", "rules": ["platform-base"]})
                    .as_object()
                    .cloned(),
            })
        );
        // The checker changes nothing else of the report.
        let unchecked = load(&package, AgentPackageOptions::default());
        assert_eq!(unchecked.checker_result, None);
        assert_eq!(
            AgentPackageReport {
                checker_result: None,
                ..report
            },
            unchecked
        );
    }

    #[test]
    fn a_checker_that_denies_gives_no_config_and_its_reasons_in_the_report() {
        let package = code_reviewer("checker-denies");

        let report = load(&package, checked_by(DENY, &[]));

        assert_eq!(report.outcome, PackageOutcome::Denied);
        assert!(report.config.is_none());
        // The digest of the config the checker did not allow stays.
        assert!(report.config_sha256.is_some());
        assert!(report.stopped_by.is_empty());
        let result = report.checker_result.expect("a result");
        assert_eq!(result.decision, CheckerDecision::Deny);
        assert_eq!(result.failure, None);
        assert_eq!(result.exit_code, Some(2));
        assert_eq!(result.detail, None);
        assert_eq!(
            result.answer.expect("an answer")["reasons"],
            json!(["the skill review is not on the approved list"])
        );
    }

    #[test]
    fn a_checker_that_does_not_answer_as_it_must_is_a_deny() {
        use CheckerFailure::{InvalidAnswer, UnexpectedExit};
        let package = code_reviewer("checker-fails");
        // (what the checker does, the failure, its exit code, whether an answer is kept)
        let table = [
            (
                r#"printf '{"decision": "allow"}'; exit 3"#,
                UnexpectedExit,
                Some(3),
                true,
            ),
            (
                r#"printf '{"decision": "deny"}'; exit 1"#,
                UnexpectedExit,
                Some(1),
                true,
            ),
            ("kill -KILL $$", UnexpectedExit, None, false),
            ("exit 0", InvalidAnswer, Some(0), false),
            (
                "printf 'the package may start'",
                InvalidAnswer,
                Some(0),
                false,
            ),
            (r#"printf '["allow"]'"#, InvalidAnswer, Some(0), false),
            (
                r#"printf '{"decision": "allow"} {"decision": "allow"}'"#,
                InvalidAnswer,
                Some(0),
                false,
            ),
            (
                r#"printf '{"decision": "deny", "decision": "allow"}'"#,
                InvalidAnswer,
                Some(0),
                false,
            ),
            (
                r#"printf '{"decision": "deny"}'"#,
                InvalidAnswer,
                Some(0),
                true,
            ),
            (
                r#"printf '{"decision": "allow"}'; exit 2"#,
                InvalidAnswer,
                Some(2),
                true,
            ),
            (
                r#"printf '{"decision": true}'"#,
                InvalidAnswer,
                Some(0),
                true,
            ),
            (
                r#"printf '{"reasons": []}'; exit 2"#,
                InvalidAnswer,
                Some(2),
                true,
            ),
        ];
        for (script, failure, exit_code, answered) in table {
            let report = load(&package, checked_by(script, &[]));

            assert_eq!(report.outcome, PackageOutcome::Denied, "{script}");
            assert!(report.config.is_none(), "{script}");
            let result = report.checker_result.expect("a result");
            assert_eq!(result.decision, CheckerDecision::Deny, "{script}");
            assert_eq!(result.failure, Some(failure), "{script}");
            assert_eq!(result.exit_code, exit_code, "{script}");
            assert_eq!(result.answer.is_some(), answered, "{script}");
        }
    }

    #[test]
    fn a_checker_that_writes_more_than_it_may_has_one_result_however_it_ended() {
        let package = code_reviewer("checker-writes-too-much");
        let allow = r#"printf '{"decision": "allow"}'"#;
        let scripts = [
            // Twice the limit, to standard output and then to standard error, and then the
            // checker sleeps for a day. The time limit is a day too, so the load returns
            // only because the write limit ended the checker. `exec` keeps one process, so
            // the limit ends what sleeps.
            format!("head -c 2097152 /dev/zero; exec sleep {A_DAY}; {ALLOW}"),
            format!("head -c 2097152 /dev/zero >&2; exec sleep {A_DAY}; {ALLOW}"),
            // One byte more than the limit, and the checker ends by itself. Its exit code and
            // what it answered are not used: a load must not depend on whether the checker
            // ended before the loader looked.
            format!("head -c 1048577 /dev/zero | tr '\\0' ' '; {allow}"),
            format!("head -c 1048577 /dev/zero >&2; {allow}"),
            format!("head -c 1048577 /dev/zero >&2; {DENY}"),
            "head -c 1048577 /dev/zero; exit 3".to_string(),
            "head -c 1048577 /dev/zero >&2; kill -KILL $$".to_string(),
            // What a program the checker started wrote counts as the checker's own.
            format!("head -c 1048577 /dev/zero & wait; {allow}"),
        ];
        for script in scripts {
            let report = load(&package, checked_by(&script, &[]));

            assert_eq!(report.outcome, PackageOutcome::Denied, "{script}");
            assert!(report.config.is_none(), "{script}");
            assert_eq!(
                report.checker_result,
                Some(CheckerResult {
                    decision: CheckerDecision::Deny,
                    failure: Some(CheckerFailure::OutputTooLarge),
                    exit_code: None,
                    detail: Some("the checker wrote more than 1048576 bytes".to_string()),
                    answer: None,
                }),
                "{script}"
            );
        }
    }

    #[test]
    fn a_checker_that_writes_as_much_as_it_may_is_answered_as_any_other() {
        let package = code_reviewer("checker-writes-the-limit");
        // The answer after white space fills standard output to the last byte it may hold,
        // and standard error holds as much.
        let padding = MAX_OUTPUT_BYTES as usize - r#"{"decision": "allow"}"#.len();
        let script = format!(
            "head -c {MAX_OUTPUT_BYTES} /dev/zero >&2; head -c {padding} /dev/zero | tr '\\0' ' '; printf '{{\"decision\": \"allow\"}}'"
        );

        let report = load(&package, checked_by(&script, &[]));

        assert_eq!(report.outcome, PackageOutcome::Ready);
        let result = report.checker_result.expect("a result");
        assert_eq!(result.decision, CheckerDecision::Allow);
        assert_eq!(result.failure, None);
        assert_eq!(result.exit_code, Some(0));
        assert_eq!(result.detail, None);
    }

    #[test]
    fn a_checker_that_fails_leaves_the_end_of_its_error_text_in_the_report() {
        let package = code_reviewer("checker-error-text");
        let said = "check-package: the policy folder /etc/platform/policy is not readable";
        let result = |script: &str| {
            load(&package, checked_by(script, &[]))
                .checker_result
                .expect("a result")
        };

        let crashed = result(&format!("echo '{said}' >&2; exit 1"));
        assert_eq!(crashed.failure, Some(CheckerFailure::UnexpectedExit));
        assert_eq!(crashed.exit_code, Some(1));
        assert_eq!(crashed.detail.as_deref(), Some(said));

        // Only the end of a long text is kept.
        let long = result(&format!(
            "head -c 5000 /dev/zero | tr '\\0' . >&2; echo '{said}' >&2"
        ));
        assert_eq!(long.failure, Some(CheckerFailure::InvalidAnswer));
        let detail = long.detail.expect("a detail");
        assert_eq!(detail.chars().count(), MAX_DETAIL_CHARS);
        assert!(detail.ends_with(said), "{detail}");

        // A checker that answered as it must has no detail, whatever it wrote there.
        let allowed = result(&format!("echo 'policy folder read' >&2; {ALLOW}"));
        assert_eq!(allowed.failure, None);
        assert_eq!(allowed.detail, None);
    }

    #[test]
    fn a_checker_that_cannot_be_started_is_a_deny() {
        let package = code_reviewer("checker-missing");
        let options = AgentPackageOptions {
            checker: Some(vec![
                package.root.join("no-such-checker").display().to_string(),
            ]),
            ..AgentPackageOptions::default()
        };

        let report = load(&package, options);

        assert_eq!(report.outcome, PackageOutcome::Denied);
        assert!(report.config.is_none());
        let result = report.checker_result.expect("a result");
        assert_eq!(result.decision, CheckerDecision::Deny);
        assert_eq!(result.failure, Some(CheckerFailure::NotStarted));
        assert_eq!(result.exit_code, None);
        assert_eq!(result.answer, None);
        // The operating system says why, in its own words.
        let detail = result.detail.expect("a detail");
        assert!(
            detail.starts_with("the checker cannot be started: ") && detail.contains("os error"),
            "{detail}"
        );
    }

    #[test]
    fn a_checker_that_runs_past_the_time_limit_is_ended_and_is_a_deny() {
        let package = code_reviewer("checker-slow");
        let scripts = [
            // The checker sleeps for a day, so the load returns only because the time limit
            // ended it. `exec` keeps one process, so the limit ends what sleeps.
            format!("exec sleep {A_DAY}; {ALLOW}"),
            // What it answered before the limit passed is not used.
            format!("{ALLOW}; exec sleep {A_DAY}"),
        ];
        for script in scripts {
            let options = AgentPackageOptions {
                checker_timeout_seconds: Some(0.2),
                ..checked_by(&script, &[])
            };

            let report = load(&package, options);

            assert_eq!(report.outcome, PackageOutcome::Denied, "{script}");
            assert_eq!(
                report.checker_result,
                Some(CheckerResult {
                    decision: CheckerDecision::Deny,
                    failure: Some(CheckerFailure::TimedOut),
                    exit_code: None,
                    detail: None,
                    answer: None,
                }),
                "{script}"
            );
        }
    }

    #[test]
    fn the_checker_gets_three_variables_and_nothing_of_the_callers_config() {
        let package = code_reviewer("checker-environment");
        let seen = Package::new("checker-environment-seen");
        // The script keeps what it was given: its variables, its arguments
        // after `$0`, and the report it was pointed at.
        let script = format!(
            r#"env > "$0/env"; printf '%s' "$#" > "$0/arguments"; cp "$NEMO_FABRIC_PACKAGE_REPORT" "$0/report.json"; {ALLOW}"#
        );
        let folder = seen.root.display().to_string();

        let report = load(&package, checked_by(&script, &[&folder]));

        assert_eq!(report.outcome, PackageOutcome::Ready);
        let read = |name: &str| fs::read_to_string(seen.root.join(name)).expect(name);
        let variables: std::collections::BTreeMap<String, String> = read("env")
            .lines()
            .filter_map(|line| line.split_once('='))
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        assert_eq!(
            variables[PACKAGE_DIR_VARIABLE],
            report.package_dir.display().to_string()
        );
        assert_eq!(variables[AGENT_NAME_VARIABLE], "code-reviewer");
        // The report file is gone with its folder once the checker answered.
        let report_file = PathBuf::from(&variables[REPORT_VARIABLE]);
        assert!(report_file.is_absolute());
        assert!(!report_file.parent().expect("a folder").exists());
        // Nothing follows the checker's own arguments.
        assert_eq!(read("arguments"), "0");
        let given = [PACKAGE_DIR_VARIABLE, REPORT_VARIABLE, AGENT_NAME_VARIABLE];
        // The shell adds these three itself.
        let shell = ["PWD", "SHLVL", "_"];
        // The test's own environment holds a variable the checker must not get, whatever
        // started the test: cargo sets some, and so does a shell.
        let kept_out = |name: &str| !KEPT_VARIABLES.contains(&name) && !shell.contains(&name);
        assert!(
            std::env::vars_os().any(|(name, _)| name.to_str().is_none_or(kept_out)),
            "this test needs a variable of its own that a checker must not get"
        );
        for name in variables.keys() {
            assert!(
                given.contains(&name.as_str())
                    || KEPT_VARIABLES.contains(&name.as_str())
                    || shell.contains(&name.as_str()),
                "{name} reached the checker"
            );
        }
        // The report the checker reads is the load report without its config.
        let text = read("report.json");
        assert!(!text.contains("a-value-of-the-callers-config"));
        assert!(!read("env").contains("a-value-of-the-callers-config"));
        let read_report: Value = serde_json::from_str(&text).expect("a report");
        assert_eq!(read_report["outcome"], "ready");
        assert_eq!(read_report["checker_result"], Value::Null);
        assert_eq!(read_report["config_sha256"], json!(report.config_sha256));
        assert_eq!(read_report["lock"]["sha256"], json!(report.lock.sha256));
        assert!(read_report.get("config").is_none());
    }

    #[test]
    fn the_checker_is_not_asked_about_a_package_that_cannot_start() {
        let marker = Package::new("checker-not-asked-marker");
        let asked = marker.root.join("asked");
        let options = || {
            checked_by(
                &format!(r#"touch "$0"; {ALLOW}"#),
                &[&asked.display().to_string()],
            )
        };

        // A required hook the test adapter does not take stops the load.
        let package = Package::new("checker-not-asked");
        package.write("hooks/guard/hooks.json", "{\"hooks\": {}}\n");
        package.write_json(
            "agent.json",
            &agent(
                "code-reviewer",
                json!({"hooks": {"guard": {"source": "./hooks/guard"}}}),
            ),
        );
        package.write_lock(
            "",
            vec![
                package.file_entry(
                    "$",
                    "agent",
                    "agent.json",
                    "./agent.json",
                    &["$/hooks/guard"],
                ),
                package.folder_entry("$/hooks/guard", "hook", "hooks/guard", "./hooks/guard"),
            ],
        );
        let stopped = load(&package, options());
        assert_eq!(stopped.outcome, PackageOutcome::Stopped);
        assert_eq!(stopped.checker_result, None);

        // With no config nothing can start.
        let verified = load_agent_package(&package.root, None, options(), context())
            .expect("the package is verified");
        assert_eq!(verified.outcome, PackageOutcome::VerifiedOnly);
        assert_eq!(verified.checker_result, None);

        // A package that fails its own checks is refused.
        package.write("hooks/guard/hooks.json", "{\"hooks\": {}}\n\n");
        let refused = load_agent_package(&package.root, Some(config()), options(), context());
        assert!(matches!(
            refused,
            Err(FabricError::AgentPackage { ref code, .. }) if code == "package_digest_mismatch"
        ));

        assert!(!asked.exists());
        // The same checker is asked once the load continues past the hook.
        package.write("hooks/guard/hooks.json", "{\"hooks\": {}}\n");
        let warned = load(
            &package,
            AgentPackageOptions {
                if_unsupported: PackageAction::Warn,
                ..options()
            },
        );
        assert_eq!(warned.outcome, PackageOutcome::Ready);
        assert!(asked.exists());
    }

    #[test]
    fn a_time_limit_that_is_not_a_positive_number_of_seconds_is_refused() {
        let package = code_reviewer("checker-limit");
        // The longest limit a duration holds is taken as it is.
        let longest = AgentPackageOptions {
            checker_timeout_seconds: Some(1e19),
            ..checked_by(ALLOW, &[])
        };
        assert_eq!(load(&package, longest).outcome, PackageOutcome::Ready);
        for seconds in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            for checker in [None, Some(vec!["/bin/sh".to_string()])] {
                let options = AgentPackageOptions {
                    checker,
                    checker_timeout_seconds: Some(seconds),
                    ..AgentPackageOptions::default()
                };

                let refused = load_agent_package(&package.root, Some(config()), options, context());

                assert!(
                    matches!(
                        refused,
                        Err(FabricError::AgentPackage { ref code, ref subject, .. })
                            if code == "package_options_invalid"
                                && subject == "checker_timeout_seconds"
                    ),
                    "{seconds}"
                );
            }
        }
    }

    #[test]
    fn a_checker_with_no_program_is_refused_and_nothing_starts_unchecked() {
        let package = code_reviewer("checker-empty");
        let options = AgentPackageOptions {
            checker: Some(Vec::new()),
            ..AgentPackageOptions::default()
        };

        let refused = load_agent_package(&package.root, Some(config()), options, context());

        assert!(matches!(
            refused,
            Err(FabricError::AgentPackage { ref code, ref subject, .. })
                if code == "package_options_invalid" && subject == "checker"
        ));
    }

    // -- A checker URL ----------------------------------------------------

    /// A server on a free local port that answers one request with `status`
    /// and `body`, and hands back the request it read. `chunked` sends the
    /// body in two chunks. `silent` reads the request and never answers.
    struct OneShot {
        url: String,
        request: std::sync::mpsc::Receiver<String>,
    }

    fn serve(status: &str, body: &str, chunked: bool, silent: bool) -> OneShot {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a local port");
        let url = format!(
            "http://{}/v1/hooks/agent-start",
            listener.local_addr().expect("an address")
        );
        let (sent, request) = std::sync::mpsc::channel();
        let status = status.to_string();
        let body = body.to_string();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("a connection");
            let mut read = Vec::new();
            let mut buffer = [0u8; 4096];
            // The head, then as many bytes as Content-Length says.
            loop {
                let count = stream.read(&mut buffer).expect("a read");
                read.extend_from_slice(&buffer[..count]);
                let text = String::from_utf8_lossy(&read).to_string();
                if let Some(end) = text.find("\r\n\r\n") {
                    let length: usize = text[..end]
                        .lines()
                        .find_map(|line| line.strip_prefix("Content-Length: "))
                        .and_then(|length| length.trim().parse().ok())
                        .unwrap_or(0);
                    if read.len() >= end + 4 + length {
                        break;
                    }
                }
                if count == 0 {
                    break;
                }
            }
            let _ = sent.send(String::from_utf8_lossy(&read).to_string());
            if silent {
                std::thread::sleep(Duration::from_secs(5));
                return;
            }
            let answer = if chunked {
                let (first, second) = body.split_at(body.len() / 2);
                format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n{:x}\r\n{first}\r\n{:x}\r\n{second}\r\n0\r\n\r\n",
                    first.len(),
                    second.len()
                )
            } else {
                format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                    body.len()
                )
            };
            let _ = stream.write_all(answer.as_bytes());
        });
        OneShot { url, request }
    }

    fn asked_at(url: &str) -> AgentPackageOptions {
        AgentPackageOptions {
            checker_url: Some(url.to_string()),
            checker_timeout_seconds: Some(f64::from(A_DAY)),
            ..AgentPackageOptions::default()
        }
    }

    /// The JSON body of a request the one-shot server read.
    fn body_of(request: &str) -> Value {
        let (_, body) = request.split_once("\r\n\r\n").expect("a body");
        serde_json::from_str(body).expect("a JSON body")
    }

    const GUARDRAILS_DENY: &str = r#"{"decision": "deny", "reason": "skill '$/skills/review' comes from a local folder, and hooks_server.agent_start.allow_local_sources is false"}"#;

    #[test]
    fn a_checker_url_gets_the_facts_and_its_allow_leaves_the_load_ready() {
        let package = code_reviewer("checker-url-allows");
        let server = serve("200 OK", r#"{"decision": "allow"}"#, false, false);

        let report = load(
            &package,
            AgentPackageOptions {
                checker_enforcers: vec!["openshell".to_string()],
                ..asked_at(&server.url)
            },
        );

        assert_eq!(report.outcome, PackageOutcome::Ready);
        assert!(report.config.is_some());
        assert_eq!(
            report.checker_result,
            Some(CheckerResult {
                decision: CheckerDecision::Allow,
                failure: None,
                exit_code: None,
                detail: None,
                answer: json!({"decision": "allow"}).as_object().cloned(),
            })
        );
        let request = server.request.recv().expect("the request");
        assert!(request.starts_with("POST /v1/hooks/agent-start HTTP/1.1\r\n"));
        assert!(request.contains("Content-Type: application/json\r\n"));
        let digest = report.digests[1].digest.clone();
        assert_eq!(
            body_of(&request),
            json!({
                "hook_event_name": "AgentStart",
                "agent_name": "code-reviewer",
                "parts": [{
                    "id": "$/skills/review",
                    "type": "skill",
                    "source": "./skills/review",
                    "version": report.digests[1].version,
                    "digest": digest,
                    "local": true,
                }],
                "models": [],
                "policies": [],
                "enforcers_present": ["openshell"],
            })
        );
        // Nothing of the caller's config is sent.
        assert!(!request.contains("a-value-of-the-callers-config"));
    }

    #[test]
    fn a_checker_url_that_denies_gives_no_config_and_its_reason_in_the_report() {
        let package = code_reviewer("checker-url-denies");
        let server = serve("200 OK", GUARDRAILS_DENY, true, false);

        let report = load(&package, asked_at(&server.url));

        assert_eq!(report.outcome, PackageOutcome::Denied);
        assert!(report.config.is_none());
        let result = report.checker_result.expect("a checker result");
        assert_eq!(result.decision, CheckerDecision::Deny);
        assert_eq!(result.failure, None);
        assert_eq!(
            result.answer,
            serde_json::from_str::<Value>(GUARDRAILS_DENY)
                .expect("json")
                .as_object()
                .cloned()
        );
    }

    #[test]
    fn a_checker_url_that_cannot_be_reached_is_a_deny() {
        let package = code_reviewer("checker-url-unreachable");
        // A port that was free a moment ago, with nothing listening on it.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("a local port")
            .local_addr()
            .expect("an address")
            .port();

        let report = load(
            &package,
            asked_at(&format!("http://127.0.0.1:{port}/v1/hooks/agent-start")),
        );

        assert_eq!(report.outcome, PackageOutcome::Denied);
        assert!(report.config.is_none());
        let result = report.checker_result.expect("a checker result");
        assert_eq!(result.decision, CheckerDecision::Deny);
        assert_eq!(result.failure, Some(CheckerFailure::Unreachable));
        assert!(
            result
                .detail
                .expect("a detail")
                .contains("cannot be reached")
        );
    }

    #[test]
    fn a_checker_url_that_does_not_answer_as_it_must_is_a_deny() {
        for (status, body, failure) in [
            (
                "422 Unprocessable Entity",
                r#"{"error": {"message": "body.parts: Field required"}}"#,
                CheckerFailure::UnexpectedStatus,
            ),
            (
                "404 Not Found",
                r#"{"error": {"code": "check_not_configured"}}"#,
                CheckerFailure::UnexpectedStatus,
            ),
            (
                "200 OK",
                r#"{"decision": "maybe"}"#,
                CheckerFailure::InvalidAnswer,
            ),
            ("200 OK", r#"["allow"]"#, CheckerFailure::InvalidAnswer),
            ("200 OK", "allow", CheckerFailure::InvalidAnswer),
        ] {
            let package = code_reviewer("checker-url-invalid");
            let server = serve(status, body, false, false);

            let report = load(&package, asked_at(&server.url));

            assert_eq!(report.outcome, PackageOutcome::Denied, "{status} {body}");
            let result = report.checker_result.expect("a checker result");
            assert_eq!(result.failure, Some(failure), "{status} {body}");
            if failure == CheckerFailure::UnexpectedStatus {
                assert!(
                    result
                        .detail
                        .expect("a detail")
                        .contains(status.split(' ').next().unwrap())
                );
                assert!(result.answer.is_some());
            }
        }
    }

    #[test]
    fn a_checker_url_that_does_not_answer_in_time_is_a_deny() {
        let package = code_reviewer("checker-url-silent");
        let server = serve("200 OK", "", false, true);

        let report = load(
            &package,
            AgentPackageOptions {
                checker_timeout_seconds: Some(0.3),
                ..asked_at(&server.url)
            },
        );

        assert_eq!(report.outcome, PackageOutcome::Denied);
        assert_eq!(
            report.checker_result.expect("a checker result").failure,
            Some(CheckerFailure::TimedOut)
        );
    }

    #[test]
    fn a_checker_url_the_loader_cannot_use_is_refused() {
        let package = code_reviewer("checker-url-refused");
        let refusals = [
            AgentPackageOptions {
                checker_url: Some("http://127.0.0.1:8100/v1/hooks/agent-start".to_string()),
                ..checked_by(ALLOW, &[])
            },
            asked_at("https://checker.example.com/v1/hooks/agent-start"),
            asked_at("http://user:secret@127.0.0.1:8100/"),
            asked_at("http://127.0.0.1:0/"),
            asked_at("http://127.0.0.1:99999/"),
            asked_at("http:///v1/hooks/agent-start"),
            asked_at("http://127.0.0.1:8100/a path"),
        ];
        for options in refusals {
            let named = options.checker_url.clone();
            let refused = load_agent_package(&package.root, Some(config()), options, context());
            assert!(
                matches!(
                    refused,
                    Err(FabricError::AgentPackage { ref code, ref subject, .. })
                        if code == "package_options_invalid" && subject == "checker_url"
                ),
                "{named:?}"
            );
        }
        let enforcers_alone = AgentPackageOptions {
            checker_enforcers: vec!["openshell".to_string()],
            ..AgentPackageOptions::default()
        };
        assert!(matches!(
            load_agent_package(&package.root, Some(config()), enforcers_alone, context()),
            Err(FabricError::AgentPackage { ref code, ref subject, .. })
                if code == "package_options_invalid" && subject == "checker_enforcers"
        ));
    }

    #[test]
    fn a_checker_url_is_not_asked_about_a_package_that_cannot_start() {
        let package = code_reviewer("checker-url-not-asked");
        // Nothing listens: a load that asked would be denied as unreachable.
        let options = asked_at("http://127.0.0.1:9/v1/hooks/agent-start");

        let report = load_agent_package(&package.root, None, options, context()).expect("a report");

        assert_eq!(report.checker_result, None);
    }
}
