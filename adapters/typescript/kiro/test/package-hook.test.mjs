// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { answer, formatEvent } from "../dist/package-hook.js";
import { packageHooks } from "../dist/workspace-files.js";

const SCRIPT = fileURLToPath(new URL("../dist/package-hook.js", import.meta.url));
const FORMAT = "https://agent-plugins.org/schemas/1.1.0/hooks.schema.json";

test("reads only the closed answers the format defines for each event", () => {
  assert.deepEqual(answer("PreToolUse", { decision: "allow" }), {});
  assert.deepEqual(answer("PreToolUse", { decision: "deny", reason: "no" }), { block: "no" });
  assert.equal(answer("PreToolUse", { decision: "deny" }), null);
  assert.equal(answer("PreToolUse", { decision: "block", reason: "x" }), null);
  assert.deepEqual(answer("PostToolUse", { additional_context: "note" }), { context: "note" });
  assert.equal(answer("PreToolUse", { additional_context: "note" }), null);
  assert.equal(answer("Stop", { decision: "continue", reason: "x" }), null);
});

test("names the tool the way the format's Claude-style events do", () => {
  const event = formatEvent("PreToolUse", { session_id: "s", cwd: "/w", tool_name: "read_file", tool_input: { path: "/a" }, tool_use_id: "t" });
  assert.equal(event.tool_name, "Read");
  assert.equal(event.hook_event_name, "PreToolUse");
  assert.equal(event.tool_use_id, "t");
});

async function hookFolder(root, script) {
  const folder = join(root, "guard");
  await mkdir(folder);
  await writeFile(join(folder, "hooks.json"), JSON.stringify({ $schema: FORMAT, hooks: { PreToolUse: [{ command: "./check" }] } }));
  await writeFile(join(folder, "check"), script);
  await chmod(join(folder, "check"), 0o755);
  return folder;
}

function run(args, input) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [SCRIPT, ...args], { stdio: ["pipe", "pipe", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("close", (code) => resolve({ code, stderr }));
    child.stdin.end(JSON.stringify(input));
  });
}

test("runs the handler in its folder: a deny blocks, an allow and a handler failure follow the event", async () => {
  const root = await mkdtemp(join(tmpdir(), "kiro-pkg-hook-"));
  const folder = await hookFolder(
    root,
    '#!/bin/sh\ninput=$(cat)\ncase "$input" in *secret*) echo \'{"decision":"deny","reason":"names secret"}\';; *) echo \'{"decision":"allow"}\';; esac\n',
  );
  const base = ["--root", folder, "--event", "PreToolUse", "--command", join(folder, "check"), "--data", join(root, "data")];
  assert.equal((await run(base, { tool_name: "read_file", tool_input: { path: "/public" } })).code, 0);
  const denied = await run(base, { tool_name: "read_file", tool_input: { path: "/secret" } });
  assert.equal(denied.code, 2);
  assert.match(denied.stderr, /names secret/u);
  const missing = await run(["--root", folder, "--event", "PreToolUse", "--command", "/no/such/handler"], { tool_name: "x" });
  assert.equal(missing.code, 2);
  assert.match(missing.stderr, /could not check/u);
  await rm(root, { recursive: true, force: true });
});

test("a package's hooks become one Kiro command each, and a hook it cannot run stops the start", async () => {
  const root = await mkdtemp(join(tmpdir(), "kiro-pkg-hooks-"));
  const folder = await hookFolder(root, "#!/bin/sh\n");
  const config = { extensions: { "nemo_fabric.hooks.v1": { entries: [{ path: folder, format: FORMAT }] } } };
  const hooks = await packageHooks(config, root, "/x/package-hook.js", join(root, "data"));
  assert.equal(hooks.length, 1);
  assert.equal(hooks[0].event, "PreToolUse");
  assert.match(hooks[0].command, /package-hook\.js' --name 'guard' --root /u);
  await writeFile(join(folder, "hooks.json"), JSON.stringify({ $schema: FORMAT, hooks: { Stop: [{ command: "./check" }] } }));
  await assert.rejects(packageHooks(config, root, "/x/p.js", join(root, "data")), { code: "kiro_hook_event_unsupported" });
  await rm(root, { recursive: true, force: true });
});
