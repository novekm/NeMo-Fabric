// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { AGENT_MARKER_KEY, MARKER, prepareWorkspaceFiles } from "../dist/workspace-files.js";

async function scratch() {
  const root = await mkdtemp(join(tmpdir(), "kiro-files-"));
  const workspace = join(root, "workspace");
  await mkdir(workspace);
  return { root, workspace, done: () => rm(root, { recursive: true, force: true }) };
}

const present = (path) => stat(path).then(() => true, () => false);
const agentFile = async (workspace, name) =>
  JSON.parse(await readFile(join(workspace, ".kiro", "agents", `${name}.json`), "utf8"));
const append = (content) => ({ instructions: { system: { mode: "append", content } } });

async function skillFolder(root, name) {
  const folder = join(root, name);
  await mkdir(folder, { recursive: true });
  await writeFile(join(folder, "SKILL.md"), `---\nname: ${name}\ndescription: d\n---\nbody\n`);
  return folder;
}

test("writes the session's own agent with its instructions and selects it as the mode", async () => {
  const { root, workspace, done } = await scratch();
  const files = await prepareWorkspaceFiles(append("Answer in three lines."), workspace, root, "Engineering-Lead");
  assert.equal(files.leadMode, "engineering-lead");
  const lead = await agentFile(workspace, "engineering-lead");
  assert.equal(lead.prompt, "Answer in three lines.");
  assert.deepEqual(lead.tools, []);
  assert.equal(lead[AGENT_MARKER_KEY], MARKER);
  await files.cleanup();
  assert.equal(await present(join(workspace, ".kiro")), false);
  await done();
});

test("writes nothing, and selects no mode, for a package with nothing for Kiro to load", async () => {
  const { root, workspace, done } = await scratch();
  const files = await prepareWorkspaceFiles({}, workspace, root, "lead");
  assert.equal(files.leadMode, undefined);
  assert.equal(await present(join(workspace, ".kiro")), false);
  await done();
});

test("refuses replace instructions instead of ignoring them", async () => {
  const { root, workspace, done } = await scratch();
  const config = { instructions: { system: { mode: "replace", content: "x" } } };
  await assert.rejects(prepareWorkspaceFiles(config, workspace, root, "lead"), { code: "kiro_instruction_mode_unsupported" });
  await done();
});

test("maps tools.enabled and tools.blocked to the Kiro tools the agent sees and pre-allows only those", async () => {
  const { root, workspace, done } = await scratch();
  const config = { tools: { enabled: ["read_file", "shell", "grep"], blocked: ["shell"] } };
  await prepareWorkspaceFiles(config, workspace, root, "lead");
  const lead = await agentFile(workspace, "lead");
  assert.deepEqual(lead.tools, ["read", "grep"]);
  assert.deepEqual(lead.permissions.rules, [{ capability: "fs_read", effect: "allow" }]);
  await assert.rejects(prepareWorkspaceFiles({ tools: { enabled: ["teleport"] } }, workspace, root, "other"), {
    code: "kiro_tool_unsupported",
  });
  await done();
});

test("will not overwrite or remove an agent file it did not write", async () => {
  const { root, workspace, done } = await scratch();
  await mkdir(join(workspace, ".kiro", "agents"), { recursive: true });
  const path = join(workspace, ".kiro", "agents", "lead.json");
  await writeFile(path, "{}\n");
  await assert.rejects(prepareWorkspaceFiles(append("x"), workspace, root, "lead"), { code: "kiro_workspace_file_exists" });
  assert.equal(await readFile(path, "utf8"), "{}\n");
  await done();
});

test("lead skills are skill:// resources, allowed by name", async () => {
  const { root, workspace, done } = await scratch();
  const skill = await skillFolder(root, "release-checklist");
  await prepareWorkspaceFiles({ skills: { paths: [skill] } }, workspace, root, "lead");
  const lead = await agentFile(workspace, "lead");
  assert.deepEqual(lead.resources, [`skill://${join(await realpath(skill), "SKILL.md")}`]);
  assert.deepEqual(lead.permissions.rules, [{ capability: "skill", match: ["release-checklist"], effect: "allow" }]);
  await assert.rejects(
    prepareWorkspaceFiles({ skills: { paths: [join(root, "nope")] } }, workspace, root, "x"),
    { code: "kiro_called_agent_invalid" },
  );
  await assert.rejects(prepareWorkspaceFiles({ skills: { extensions: {} } }, workspace, root, "y"), {
    code: "kiro_skill_extensions_unsupported",
  });
  await done();
});

test("each called agent is a Kiro agent with its prompt, tools and skills; the lead may call them and carries their rules", async () => {
  const { root, workspace, done } = await scratch();
  const skill = await skillFolder(root, "release-checklist");
  const plugin = join(root, "review-kit");
  await skillFolder(join(plugin, "skills"), "style");
  const config = {
    ...append("Lead them."),
    extensions: {
      "nemo_fabric.agents.v1": {
        coder: { description: "Writes code.", instructions: "Write the change.", tools: ["read_file", "write_file"] },
        reviewer: {
          description: "Reviews.",
          instructions: "Review it.",
          skills: { paths: [skill] },
          plugins: { paths: [plugin] },
        },
      },
    },
  };
  const files = await prepareWorkspaceFiles(config, workspace, root, "lead");
  const coder = await agentFile(workspace, "coder");
  assert.equal(coder.prompt, "Write the change.");
  assert.deepEqual(coder.tools, ["read", "write"]);
  const reviewer = await agentFile(workspace, "reviewer");
  assert.deepEqual(reviewer.resources, [
    `skill://${join(await realpath(skill), "SKILL.md")}`,
    `skill://${join(await realpath(plugin), "skills", "*", "SKILL.md")}`,
  ]);
  const lead = await agentFile(workspace, "lead");
  assert.deepEqual(lead.tools, ["subagent/coder", "subagent/reviewer"]);
  assert.deepEqual(lead.permissions.rules, [
    { capability: "subagent", match: ["coder", "reviewer"], effect: "allow" },
    { capability: "fs_read", effect: "allow" },
    { capability: "fs_write", effect: "allow" },
    { capability: "skill", match: ["release-checklist", "style"], effect: "allow" },
  ]);
  await files.cleanup();
  assert.equal(await present(join(workspace, ".kiro")), false);
  await done();
});

test("refuses a called agent field Kiro cannot take, and a called agent named like the lead", async () => {
  const { root, workspace, done } = await scratch();
  const agent = (extra, name = "coder") => ({
    extensions: { "nemo_fabric.agents.v1": { [name]: { description: "d", instructions: "i", ...extra } } },
  });
  await assert.rejects(prepareWorkspaceFiles(agent({ max_turns: 3 }), workspace, root, "lead"), {
    code: "kiro_called_agent_invalid",
  });
  await assert.rejects(prepareWorkspaceFiles(agent({}, "lead"), workspace, root, "lead"), {
    code: "kiro_called_agent_invalid",
  });
  await done();
});

test("the lead sees each MCP server and may call its tools", async () => {
  const { root, workspace, done } = await scratch();
  await prepareWorkspaceFiles(append("x"), workspace, root, "lead", [{ name: "issues" }]);
  const lead = await agentFile(workspace, "lead");
  assert.deepEqual(lead.tools, ["@issues"]);
  assert.deepEqual(lead.permissions.rules, [{ capability: "mcp", match: ["issues/*"], effect: "allow" }]);
  await done();
});

test("a called agent's model, MCP servers and the lead's package tools reach the agent files", async () => {
  const { root, workspace, done } = await scratch();
  const config = {
    ...append("Lead."),
    extensions: {
      "nemo_fabric.tools.v1": { builtin: ["read_file"] },
      "nemo_fabric.agents.v1": {
        words: {
          description: "Asks the words server.",
          instructions: "Use the tool.",
          model: { provider: "kiro", model: "claude-haiku-4.5" },
          mcp: { servers: { words: { transport: "stdio", url: "/bin/sh", args: ["-c", "node server.js"], env: { A: "1" } } } },
        },
      },
    },
  };
  const files = await prepareWorkspaceFiles(config, workspace, root, "lead");
  assert.deepEqual(files.memberModels, [{ agent: "words", model: "claude-haiku-4.5" }]);
  const member = await agentFile(workspace, "words");
  assert.equal(member.model, "claude-haiku-4.5");
  assert.deepEqual(member.mcpServers, { words: { command: "/bin/sh", args: ["-c", "node server.js"], env: { A: "1" } } });
  assert.deepEqual(member.tools, ["@words"]);
  const lead = await agentFile(workspace, "lead");
  assert.deepEqual(lead.tools, ["read", "subagent/words"]);
  assert.deepEqual(lead.permissions.rules, [
    { capability: "fs_read", effect: "allow" },
    { capability: "subagent", match: ["words"], effect: "allow" },
    { capability: "mcp", match: ["words/*"], effect: "allow" },
  ]);
  await assert.rejects(
    prepareWorkspaceFiles(
      { extensions: { "nemo_fabric.agents.v1": { words: { description: "d", instructions: "i", model: { provider: "bedrock", model: "x" } } } } },
      workspace,
      root,
      "other",
    ),
    { code: "kiro_called_agent_invalid" },
  );
  await done();
});
