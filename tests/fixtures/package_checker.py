# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A checker for agent package tests. Its first argument says how it answers.

With a second argument it first writes what it was given to that file: its
arguments, its environment and the load report it was pointed at.
"""

import json
import os
import sys
import time

ALLOW = json.dumps(
    {
        "decision": "allow",
        "rules": ["approved-hooks"],
        "checker": {"name": "test-checker", "version": "1"},
    }
)
DENY = json.dumps(
    {
        "decision": "deny",
        "reasons": ["The skill release-notes is not on the approved list."],
        "rules": ["approved-skills"],
    }
)
# What the checker writes, and its exit code.
ANSWERS = {
    "allow": (ALLOW, 0),
    "deny": (DENY, 2),
    "exit-3": (ALLOW, 3),
    "no-output": ("", 0),
    "no-json": ("The package may start.", 0),
    "two-objects": (f"{ALLOW}\n{ALLOW}", 0),
    "deny-with-exit-0": (DENY, 0),
    "allow-with-exit-2": (ALLOW, 2),
    # These two write too much first, then answer and end by themselves.
    "too-much-output-then-allow": (ALLOW, 0),
    "too-much-error-text-then-allow": (ALLOW, 0),
}
# Twice what a checker may write to one stream.
TOO_MUCH = " " * (2 * 1024 * 1024)
# These two write too much and then sleep, so the load has to end them.
SLEEPS_AFTER_TOO_MUCH = ("too-much-output", "too-much-error-text")
# How long a checker sleeps that a limit must end, in seconds. A test that names
# such a checker reads no clock: its load returns only because the checker was ended.
A_DAY = 24 * 60 * 60

answer, *record = sys.argv[1:]
if record:
    with open(os.environ["NEMO_FABRIC_PACKAGE_REPORT"], encoding="utf-8") as report:
        given = {
            "arguments": sys.argv[1:],
            "environment": dict(os.environ),
            "report": json.load(report),
        }
    with open(record[0], "w", encoding="utf-8") as file:
        json.dump(given, file)
if answer == "crash":
    raise RuntimeError("the policy folder is not readable")
if answer.startswith("too-much-output"):
    print(TOO_MUCH, flush=True)
if answer.startswith("too-much-error-text"):
    print(TOO_MUCH, file=sys.stderr, flush=True)
if answer == "hang" or answer in SLEEPS_AFTER_TOO_MUCH:
    time.sleep(A_DAY)
text, exit_code = ANSWERS[answer]
print(text)
sys.exit(exit_code)
