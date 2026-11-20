// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { claudeCall, verdict } from "../dist/guardrails-hook.js";
import { HOOKS_FILE, MARKER, prepareWorkspaceFiles } from "../dist/workspace-files.js";

const SCRIPT = fileURLToPath(new URL("../dist/guardrails-hook.js", import.meta.url));

test("names a Kiro call the way the Claude Code endpoint reads it", () => {
  assert.deepEqual(claudeCall("read_file", { path: "/a" }), { tool_name: "Read", tool_input: { path: "/a", file_path: "/a" } });
  assert.equal(claudeCall("execute_bash", { command: "ls" }).tool_name, "Bash");
  assert.equal(claudeCall("subagent_file-reader", {}).tool_name, "Task");
  assert.equal(claudeCall("words::word_of_the_day", {}).tool_name, "mcp__words__word_of_the_day");
  assert.equal(claudeCall("something_else", {}).tool_name, "something_else");
});

test("a deny, or an answer that ends the run, blocks the call; anything else lets it run", () => {
  assert.deepEqual(verdict({}), { code: 0, message: "" });
  assert.deepEqual(
    verdict({ hookSpecificOutput: { permissionDecision: "deny", permissionDecisionReason: "no Bash" } }),
    { code: 2, message: "no Bash" },
  );
  assert.equal(verdict({ continue: false, stopReason: "stop" }).code, 2);
});

function run(args, input) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [SCRIPT, ...args], { stdio: ["pipe", "pipe", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("close", (code) => resolve({ code, stderr }));
    child.stdin.end(JSON.stringify(input));
  });
}

test("the hook command asks the server, with the agent and run named, and blocks on a deny", async () => {
  const seen = [];
  const server = createServer((request, response) => {
    let body = "";
    request.on("data", (chunk) => (body += chunk));
    request.on("end", () => {
      const call = JSON.parse(body);
      seen.push({ path: request.url, agent: request.headers["x-nemo-agent-name"], run: request.headers["x-nemo-run-id"], call });
      response.setHeader("content-type", "application/json");
      response.end(
        call.tool_name === "Bash"
          ? JSON.stringify({ hookSpecificOutput: { permissionDecision: "deny", permissionDecisionReason: "Bash is not allowed" } })
          : "{}",
      );
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  const flags = ["--url", url, "--agent-name", "lead", "--run-id", "run-1", "--timeout", "5"];
  const allowed = await run(flags, { tool_name: "read_file", tool_input: { path: "/a" }, session_id: "s" });
  const denied = await run(flags, { tool_name: "execute_bash", tool_input: { command: "ls" }, session_id: "s" });
  assert.equal(allowed.code, 0);
  assert.equal(denied.code, 2);
  assert.match(denied.stderr, /Bash is not allowed/u);
  assert.deepEqual(seen.map((item) => [item.path, item.agent, item.run, item.call.tool_name]), [
    ["/v1/hooks/claude-code", "lead", "run-1", "Read"],
    ["/v1/hooks/claude-code", "lead", "run-1", "Bash"],
  ]);
  server.close();
  const silent = await run(["--url", "http://127.0.0.1:9", "--timeout", "2"], { tool_name: "read_file" });
  assert.equal(silent.code, 2);
});

test("writes the hook as a standalone file for the whole session and removes it", async () => {
  const root = await mkdtemp(join(tmpdir(), "kiro-hook-"));
  const workspace = join(root, "w");
  await mkdir(workspace);
  const files = await prepareWorkspaceFiles({}, workspace, root, "lead", [], { command: "node hook.js --url 'http://x'" });
  assert.equal(files.hooks, true);
  const text = await readFile(join(workspace, ".kiro", "hooks", HOOKS_FILE), "utf8");
  const file = JSON.parse(text);
  assert.equal(file.version, "v1");
  assert.equal(file.hooks[0].trigger, "PreToolUse");
  assert.equal(file.hooks[0].action.command, "node hook.js --url 'http://x'");
  assert.ok(text.includes(MARKER));
  await files.cleanup();
  await assert.rejects(readFile(join(workspace, ".kiro", "hooks", HOOKS_FILE)));
  await rm(root, { recursive: true, force: true });
});
