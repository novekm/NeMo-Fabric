// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { buildSpec, cedarText, writeProject } from "../dist/project.js";
import { selectSettings } from "../dist/configuration.js";

async function folder(t) {
  const dir = await mkdtemp(join(tmpdir(), "fabric-strands-project-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}

async function skill(dir, name) {
  const path = join(dir, "skills", name);
  await mkdir(path, { recursive: true });
  await writeFile(join(path, "SKILL.md"), `---\nname: ${name}\ndescription: ${name}\n---\n${name}\n`);
  return path;
}

const parts = (overrides = {}) => ({
  name: "lead",
  instructions: "Lead the team.",
  model: "bedrock/us.anthropic.claude-sonnet-5-5",
  builtinTools: ["shell", "read"],
  skills: [],
  guardrails: undefined,
  ...overrides,
});

test("a called agent becomes a member with its own instructions, tools, skills, servers and model", async (t) => {
  const dir = await folder(t);
  const own = await skill(dir, "own");
  const plugin = join(dir, "plugins", "kit");
  await mkdir(join(plugin, "skills", "inside"), { recursive: true });
  await writeFile(join(plugin, "skills", "inside", "SKILL.md"), "---\nname: inside\ndescription: i\n---\ni\n");
  const input = {
    baseDir: dir,
    config: {
      extensions: {
        "nemo_fabric.agents.v1": {
          coder: {
            description: "Writes code.",
            instructions: "Write the code.",
            tools: ["read_file", "edit_file"],
            skills: { paths: [own] },
            plugins: { paths: [plugin] },
            model: { provider: "bedrock", model: "us.anthropic.claude-haiku-5-5" },
            mcp: { servers: { words: { transport: "stdio", url: "/bin/echo", args: ["x"] } } },
            max_turns: 3,
          },
          plain: { description: "Plain.", instructions: "Do it." },
        },
      },
    },
  };
  const spec = await buildSpec(input, parts());
  assert.equal(spec.members.length, 2);
  const [coder, plain] = spec.members;
  assert.deepEqual(coder.builtinTools, ["read", "edit"]);
  assert.equal(coder.model, "bedrock/us.anthropic.claude-haiku-5-5");
  assert.equal(coder.maxTurns, 3);
  assert.deepEqual(coder.mcpServers.words, { command: "/bin/echo", args: ["x"] });
  assert.equal(coder.skills.length, 2);
  assert.ok(coder.skills[1].endsWith(join("skills", "inside")));
  // A member that lists no tools has none, and runs on the run's model.
  assert.deepEqual(plain.builtinTools, []);
  assert.equal(plain.model, undefined);
});

test("the package's own tool list sets the lead's tools, and a blocked tool is removed", async (t) => {
  const dir = await folder(t);
  const base = { extensions: { "nemo_fabric.tools.v1": { builtin: ["read_file", "shell", "write_file"] } } };
  const listed = await buildSpec({ baseDir: dir, config: base }, parts());
  assert.deepEqual(listed.lead.builtinTools, ["read", "shell", "write"]);
  const blocked = await buildSpec({ baseDir: dir, config: { ...base, tools: { blocked: ["shell"] } } }, parts());
  assert.deepEqual(blocked.lead.builtinTools, ["read", "write"]);
  const config = await buildSpec({ baseDir: dir, config: { ...base, tools: { enabled: ["read_file"] } } }, parts());
  assert.deepEqual(config.lead.builtinTools, ["read"]);
});

test("a package with called agents and no tool list gives the lead no built-in tools", async (t) => {
  const dir = await folder(t);
  const withAgents = { extensions: { "nemo_fabric.agents.v1": { a: { description: "d", instructions: "i" } } } };
  assert.deepEqual((await buildSpec({ baseDir: dir, config: withAgents }, parts())).lead.builtinTools, []);
  assert.deepEqual((await buildSpec({ baseDir: dir, config: {} }, parts())).lead.builtinTools, ["shell", "read"]);
});

test("a tool Strands has no built-in for is refused by name", async (t) => {
  const dir = await folder(t);
  await assert.rejects(
    buildSpec({ baseDir: dir, config: { tools: { enabled: ["grep"] } } }, parts()),
    (error) => error.code === "strands_tool_unsupported" && /grep/u.test(error.message),
  );
  await assert.rejects(
    buildSpec(
      { baseDir: dir, config: { extensions: { "nemo_fabric.agents.v1": { x: { description: "d", instructions: "i", tools: ["glob"] } } } } },
      parts(),
    ),
    (error) => error.code === "strands_tool_unsupported",
  );
});

test("a called agent is refused when it is malformed or runs on a model Strands cannot take here", async (t) => {
  const dir = await folder(t);
  const bad = (entry, name = "x") => buildSpec({ baseDir: dir, config: { extensions: { "nemo_fabric.agents.v1": { [name]: entry } } } }, parts());
  await assert.rejects(bad({ description: "d" }), (e) => e.code === "strands_called_agent_invalid");
  await assert.rejects(bad({ description: "d", instructions: "i", extra: 1 }), (e) => e.code === "strands_called_agent_invalid");
  await assert.rejects(bad({ description: "d", instructions: "i", max_turns: 0 }), (e) => e.code === "strands_called_agent_invalid");
  await assert.rejects(
    bad({ description: "d", instructions: "i", model: { provider: "openai", model: "m" } }),
    (e) => e.code === "strands_called_agent_invalid",
  );
  await assert.rejects(bad({ description: "d", instructions: "i" }, "Bad Name"), (e) => e.code === "strands_called_agent_invalid");
  await assert.rejects(
    bad({ description: "d", instructions: "i", skills: { paths: ["missing"] } }),
    (e) => e.code === "strands_called_agent_invalid",
  );
});

test("the lead's turn limit comes from runtime.max_turns", async (t) => {
  const dir = await folder(t);
  const spec = await buildSpec({ baseDir: dir, config: { runtime: { max_turns: 4 } } }, parts());
  assert.equal(spec.lead.maxTurns, 4);
  const none = await buildSpec({ baseDir: dir, config: {} }, parts());
  assert.equal(none.lead.maxTurns, undefined);
});

test("Cedar text forbids blocked tools, appends the operator's policy and is absent when there is neither", async (t) => {
  const dir = await folder(t);
  assert.equal(await cedarText({}, dir, undefined), undefined);
  const forbidden = await cedarText({ tools: { blocked: ["shell", "write_file"] } }, dir, undefined);
  assert.match(forbidden, /^permit\(principal, action, resource\);$/mu);
  assert.match(forbidden, /forbid\(principal, action == Action::"shell", resource\);/u);
  assert.match(forbidden, /forbid\(principal, action == Action::"write", resource\);/u);
  await writeFile(join(dir, "own.cedar"), 'forbid(principal, action == Action::"read", resource);\n');
  const both = await cedarText({}, dir, "own.cedar");
  assert.match(both, /Action::"read"/u);
  await assert.rejects(cedarText({}, dir, "missing.cedar"), (e) => e.code === "strands_policy_invalid");
});

test("the project holds static source and one spec, with no package value in the source", async (t) => {
  const dir = await folder(t);
  const marker = "INSTRUCTION-MARKER-`${process.exit(1)}`";
  const spec = await buildSpec({ baseDir: dir, config: {} }, parts({ instructions: marker }));
  const agent = await writeProject(join(dir, "project"), spec, "permit(principal, action, resource);\n");
  assert.ok(agent.endsWith(join("project", "agent", "agent.ts")));
  const source = await readFile(agent, "utf8");
  const governance = await readFile(join(dir, "project", "agent", "governance.ts"), "utf8");
  assert.equal(source.includes("INSTRUCTION-MARKER"), false);
  assert.equal(governance.includes("INSTRUCTION-MARKER"), false);
  const written = JSON.parse(await readFile(join(dir, "project", "spec.json"), "utf8"));
  assert.equal(written.lead.instructions, marker);
  assert.equal(written.cedarPolicy, join(dir, "project", "policy.cedar"));
  assert.equal(JSON.parse(await readFile(join(dir, "project", "package.json"), "utf8")).type, "module");
});

test("settings accept a Cedar policy and a Guardrails URL and refuse anything else", () => {
  const ok = selectSettings({
    harness: { settings: { cedar_policy: "p.cedar", nemo_guardrails: { url: "http://127.0.0.1:9", agent_name: "a" } } },
  });
  assert.equal(ok.cedarPolicy, "p.cedar");
  assert.equal(ok.guardrails.url, "http://127.0.0.1:9");
  for (const settings of [
    { cedar_policy: "" },
    { nemo_guardrails: { url: "ftp://x" } },
    { nemo_guardrails: { url: "http://x", extra: 1 } },
    { nemo_guardrails: "http://x" },
  ]) {
    assert.throws(() => selectSettings({ harness: { settings } }), (e) => e.code === "strands_settings_invalid");
  }
});
