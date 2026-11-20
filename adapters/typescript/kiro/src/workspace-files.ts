// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { mkdir, readdir, readFile, realpath, rm, rmdir, stat, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

import type { AgentConfig } from "nemo-fabric-adapter-contract";
import { LifecycleError } from "nemo-fabric-adapters-common";

/**
 * Kiro's v3 engine loads a custom agent from `<workspace>/.kiro/agents/<name>.json`. An
 * agent carries its prompt, the tools it can see, permission rules that pre-allow some of
 * them, `skill://` resources and an MCP tool list. The session can select one as its mode,
 * and Kiro's sub-agent tool hands a task to one by name. ACP `session/new` has no field for
 * any of this, so the adapter writes the agents a package names before the session starts
 * and removes them when it stops.
 *
 * Every file written here carries MARKER. A file of the same name without it was written
 * by someone else: startup fails rather than overwrite it, and cleanup never removes it.
 */
export const MARKER = "nemo-fabric:managed";
export const AGENTS_KEY = "nemo_fabric.agents.v1";
export const TOOLS_KEY = "nemo_fabric.tools.v1";
export const AGENT_MARKER_KEY = "x-nemo-fabric";
export const HOOKS_FILE = "nemo-fabric-guardrails.json";
export const PACKAGE_HOOKS_FILE = "nemo-fabric-package-hooks.json";
export const HOOKS_KEY = "nemo_fabric.hooks.v1";

interface Rule {
  capability: string;
  effect: "allow";
  match?: string[];
}

/** A Kiro tool: the name an agent lists, and the permission capability that pre-allows it. */
const KIRO_TOOLS: Record<string, { kiro: string; capability?: string }> = {
  read: { kiro: "read", capability: "fs_read" },
  write: { kiro: "write", capability: "fs_write" },
  shell: { kiro: "shell", capability: "shell" },
  grep: { kiro: "grep" },
  glob: { kiro: "glob" },
  web_fetch: { kiro: "web_fetch", capability: "web_fetch" },
  web_search: { kiro: "web_search", capability: "web_search" },
};

/** The portable names a package uses for built-in tools, and the Kiro tool each one is. */
const PORTABLE_TOOLS: Record<string, string> = {
  read_file: "read",
  write_file: "write",
  edit_file: "write",
  grep: "grep",
  glob: "glob",
  shell: "shell",
  web_fetch: "web_fetch",
  web_search: "web_search",
};

export interface PackageHook {
  /** The hook's event, and the command Kiro runs for it. */
  event: "UserPromptSubmit" | "PreToolUse" | "PostToolUse";
  name: string;
  command: string;
}

export interface HookCommand {
  /** The command Kiro runs for a PreToolUse event, with the event on standard input. */
  command: string;
}

export interface McpServerName {
  name: string;
}

export interface WorkspaceFiles {
  /** The model ids the called agents ask for, for the caller to check against what Kiro advertises. */
  readonly memberModels: ReadonlyArray<{ agent: string; model: string }>;
  /** The mode id of the custom agent written for the session's own agent, when there is one. */
  readonly leadMode?: string;
  /** True when an agent file carries a hook: the ACP client must then opt in to Kiro hooks. */
  readonly hooks: boolean;
  cleanup(): Promise<void>;
}

interface AgentSpec {
  name: string;
  description: string;
  prompt: string | undefined;
  tools: string[];
  rules: Rule[];
  resources: string[];
  /** The names of the skills `resources` carries: Kiro asks to load a skill by name. */
  skills: string[];
  /** The Kiro model id the agent runs on, when the package names one Kiro takes. */
  model?: string;
  /** The MCP servers the agent starts, as Kiro's agent file writes them. */
  mcpServers: Record<string, Record<string, unknown>>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function ours(path: string): Promise<boolean> {
  try {
    return (await readFile(path, "utf8")).includes(MARKER);
  } catch {
    return false;
  }
}

function instructionText(config: AgentConfig): string | undefined {
  const instruction = config.instructions?.system;
  if (instruction == null) return undefined;
  if (instruction.mode !== "append") {
    throw new LifecycleError(
      "kiro_instruction_mode_unsupported",
      "Kiro supports only append system instructions: an agent's prompt adds to its own",
    );
  }
  return instruction.content;
}

/** Kiro tool names for a list of portable or Kiro-native tool names. */
function kiroTools(names: readonly string[], where: string): string[] {
  const tools = new Set<string>();
  for (const name of names) {
    const kiro = PORTABLE_TOOLS[name] ?? (KIRO_TOOLS[name] ? name : undefined);
    if (kiro === undefined) {
      throw new LifecycleError("kiro_tool_unsupported", `${where}: Kiro has no tool for \`${name}\``);
    }
    tools.add(kiro);
  }
  return [...tools];
}

/** Rules that pre-allow exactly these tools, so nothing else the agent attempts is approved. */
function toolRules(tools: readonly string[]): Rule[] {
  const rules: Rule[] = [];
  for (const tool of tools) {
    const capability = KIRO_TOOLS[tool]?.capability;
    if (capability !== undefined) rules.push({ capability, effect: "allow" });
  }
  return rules;
}

async function folderPaths(entry: Record<string, unknown>, field: string, baseDir: string, where: string) {
  const value = entry[field];
  if (value === undefined) return [];
  const paths = isRecord(value) ? value.paths : undefined;
  if (!Array.isArray(paths) || paths.some((path) => typeof path !== "string" || path.length === 0)) {
    throw new LifecycleError("kiro_called_agent_invalid", `${where}.${field} needs a list of folder paths`);
  }
  const folders: string[] = [];
  for (const path of paths as string[]) {
    try {
      const folder = await realpath(resolve(baseDir, path));
      if (!(await stat(folder)).isDirectory()) throw new Error("not a folder");
      folders.push(folder);
    } catch {
      throw new LifecycleError("kiro_called_agent_invalid", `${where}.${field}: ${path} is not a folder`);
    }
  }
  return folders;
}

/** The name Kiro knows a skill folder by: its SKILL.md `name`, else the folder's name. */
async function skillName(folder: string): Promise<string> {
  const text = await readFile(join(folder, "SKILL.md"), "utf8");
  const front = /^---\r?\n([\s\S]*?)\r?\n---/u.exec(text)?.[1] ?? "";
  const named = /^name:\s*["']?([^"'\r\n]+?)["']?\s*$/mu.exec(front)?.[1];
  return named ?? basename(folder);
}

/**
 * `skill://` resources for skill folders (each holds a SKILL.md) and plugin folders
 * (`skills/*`), with the skill names Kiro will ask to load.
 */
async function skillResources(
  entry: Record<string, unknown>,
  baseDir: string,
  where: string,
  skillCode: string,
): Promise<{ resources: string[]; names: string[] }> {
  const resources: string[] = [];
  const names: string[] = [];
  for (const folder of await folderPaths(entry, "skills", baseDir, where)) {
    if (!(await exists(join(folder, "SKILL.md")))) {
      throw new LifecycleError(skillCode, `${where}.skills: ${folder} has no SKILL.md`);
    }
    resources.push(`skill://${join(folder, "SKILL.md")}`);
    names.push(await skillName(folder));
  }
  for (const folder of await folderPaths(entry, "plugins", baseDir, where)) {
    resources.push(`skill://${join(folder, "skills", "*", "SKILL.md")}`);
    const inside = join(folder, "skills");
    for (const child of (await exists(inside)) ? await readdir(inside) : []) {
      if (await exists(join(inside, child, "SKILL.md"))) names.push(await skillName(join(inside, child)));
    }
  }
  return { resources, names };
}

/** The Kiro model id a called agent asks for: Kiro runs models it advertises, so the provider is `kiro`. */
function memberModel(value: unknown, where: string): string | undefined {
  if (value === undefined) return undefined;
  if (!isRecord(value) || value.provider !== "kiro" || typeof value.model !== "string" || value.model.length === 0) {
    throw new LifecycleError("kiro_called_agent_invalid", `${where}.model needs the provider kiro and a model`);
  }
  return value.model;
}

/** Kiro's agent-file form of the MCP servers a called agent's entry lists. */
function memberServers(value: unknown, where: string): Record<string, Record<string, unknown>> {
  if (value === undefined) return {};
  const servers = isRecord(value) ? value.servers : undefined;
  if (!isRecord(servers)) throw new LifecycleError("kiro_called_agent_invalid", `${where}.mcp needs servers`);
  const written: Record<string, Record<string, unknown>> = {};
  for (const [name, server] of Object.entries(servers)) {
    if (!/^[A-Za-z0-9_-]+$/u.test(name) || !isRecord(server) || typeof server.url !== "string") {
      throw new LifecycleError("kiro_called_agent_invalid", `${where}.mcp.servers.${name} is not a valid server`);
    }
    const env = isRecord(server.env) ? server.env : undefined;
    if (server.transport === "stdio") {
      written[name] = {
        command: server.url,
        args: Array.isArray(server.args) ? server.args : [],
        ...(env === undefined ? {} : { env }),
      };
    } else if (server.transport === "streamable-http" || server.transport === "sse") {
      written[name] = {
        type: server.transport === "sse" ? "sse" : "http",
        url: server.url,
        ...(isRecord(server.custom_headers) ? { headers: server.custom_headers } : {}),
      };
    } else {
      throw new LifecycleError("kiro_called_agent_invalid", `${where}.mcp.servers.${name} has a transport Kiro cannot start`);
    }
  }
  return written;
}

async function calledAgents(config: AgentConfig, baseDir: string): Promise<AgentSpec[]> {
  const declared = config.extensions?.[AGENTS_KEY];
  if (declared === undefined || declared === null) return [];
  if (!isRecord(declared)) {
    throw new LifecycleError("kiro_called_agent_invalid", `${AGENTS_KEY} must be an object`);
  }
  const agents: AgentSpec[] = [];
  for (const [name, entry] of Object.entries(declared)) {
    const where = `${AGENTS_KEY}.${name}`;
    if (!/^[a-z0-9_][a-z0-9_-]*$/u.test(name) || !isRecord(entry)) {
      throw new LifecycleError("kiro_called_agent_invalid", `${where} is not a valid called agent`);
    }
    const { description, instructions } = entry;
    if (typeof description !== "string" || description.length === 0 || typeof instructions !== "string" || instructions.length === 0) {
      throw new LifecycleError("kiro_called_agent_invalid", `${where} needs a description and instructions`);
    }
    for (const field of Object.keys(entry)) {
      if (!["description", "instructions", "tools", "skills", "plugins", "model", "mcp"].includes(field)) {
        throw new LifecycleError("kiro_called_agent_invalid", `${where}.${field} is not supported`);
      }
    }
    const listed = entry.tools;
    if (listed !== undefined && (!Array.isArray(listed) || listed.some((tool) => typeof tool !== "string"))) {
      throw new LifecycleError("kiro_called_agent_invalid", `${where}.tools must be a list of tool names`);
    }
    const tools = kiroTools((listed as string[] | undefined) ?? [], where);
    const { resources, names } = await skillResources(entry, baseDir, where, "kiro_skill_not_found");
    const model = memberModel(entry.model, where);
    const mcpServers = memberServers(entry.mcp, where);
    // A server's tools are tools the agent sees, and the lead's rules pre-allow them (see leadAgent).
    for (const server of Object.keys(mcpServers)) tools.push(`@${server}`);
    // The agent sees only the tools its definition names. Kiro runs it under the permission
    // rules of the session's agent, so the lead carries the rules these produce.
    agents.push({
      name,
      description,
      prompt: instructions,
      tools,
      rules: toolRules(tools.filter((tool) => !tool.startsWith("@"))),
      resources,
      skills: names,
      ...(model === undefined ? {} : { model }),
      mcpServers,
    });
  }
  return agents;
}

/** The agent for the session itself: its instructions, skills, tools and the agents it may call. */
async function leadAgent(
  config: AgentConfig,
  baseDir: string,
  name: string,
  called: readonly AgentSpec[],
  mcpServers: readonly McpServerName[],
): Promise<AgentSpec | undefined> {
  if (config.skills?.extensions !== undefined) {
    throw new LifecycleError("kiro_skill_extensions_unsupported", "Kiro does not support skill extensions");
  }
  const prompt = instructionText(config);
  const { resources, names } = await skillResources(
    { skills: { paths: config.skills?.paths ?? [] } },
    baseDir,
    "skills",
    "kiro_skill_not_found",
  );
  const packageTools = (config.extensions?.[TOOLS_KEY] as { builtin?: unknown } | undefined)?.builtin;
  if (packageTools !== undefined && (!Array.isArray(packageTools) || packageTools.some((tool) => typeof tool !== "string"))) {
    throw new LifecycleError("kiro_tool_unsupported", `${TOOLS_KEY}.builtin must be a list of tool names`);
  }
  // The tools the config names win; else the ones the package's own agent lists.
  const enabled = config.tools?.enabled ?? (packageTools as string[] | undefined);
  const blocked = new Set(kiroTools(config.tools?.blocked ?? [], "tools.blocked"));
  const tools = enabled == null ? [] : kiroTools(enabled, "tools.enabled").filter((tool) => !blocked.has(tool));
  const rules = toolRules(tools);
  if (called.length > 0) {
    // `subagent/<name>` per agent: the agents the main agent may start (Kiro's configuration reference).
    for (const agent of called) tools.push(`subagent/${agent.name}`);
    rules.push({ capability: "subagent", match: called.map((agent) => agent.name), effect: "allow" });
  }
  // A called agent runs under the lead's rules: they allow what its definition names, and only that.
  for (const agent of called) {
    for (const rule of agent.rules) if (!rules.some((have) => have.capability === rule.capability && have.match === undefined)) rules.push(rule);
  }
  for (const agent of called) {
    for (const server of Object.keys(agent.mcpServers)) {
      rules.push({ capability: "mcp", match: [`${server}/*`], effect: "allow" });
    }
  }
  const skillNames = [...names, ...called.flatMap((agent) => agent.skills)];
  if (skillNames.length > 0) rules.push({ capability: "skill", match: [...new Set(skillNames)], effect: "allow" });
  if (mcpServers.length > 0) {
    for (const { name: server } of mcpServers) {
      tools.push(`@${server}`);
      rules.push({ capability: "mcp", match: [`${server}/*`], effect: "allow" });
    }
  }
  if (prompt === undefined && resources.length === 0 && tools.length === 0 && enabled == null) return undefined;
  return { name, description: `NeMo Fabric agent ${name}`, prompt, tools, rules, resources, skills: names, mcpServers: {} };
}

/**
 * Write the agents a package names where Kiro loads them from. Returns the mode id of the
 * session's own agent, when one was written, and a cleanup that removes what was written.
 */
export async function prepareWorkspaceFiles(
  config: AgentConfig,
  workspace: string,
  baseDir: string,
  agentName: string,
  mcpServers: readonly McpServerName[] = [],
  preToolUse?: HookCommand,
  packageHooks: readonly PackageHook[] = [],
): Promise<WorkspaceFiles> {
  const called = await calledAgents(config, baseDir);
  const leadName = agentName.toLowerCase().replace(/[^a-z0-9_-]/gu, "-").replace(/^-+/u, "") || "agent";
  if (called.some((agent) => agent.name === leadName)) {
    throw new LifecycleError("kiro_called_agent_invalid", `A called agent is named like the agent itself: ${leadName}`);
  }
  const lead = await leadAgent(config, baseDir, leadName, called, mcpServers);
  const written: string[] = [];
  const made: string[] = [];
  const ensure = async (path: string): Promise<void> => {
    if (await exists(path)) return;
    await mkdir(path);
    made.unshift(path);
  };
  const cleanup = async (): Promise<void> => {
    for (const path of written.splice(0)) await rm(path, { force: true });
    // Folders this call made go too, but only while empty: a folder with anything else in it stays.
    for (const path of made.splice(0)) await rmdir(path).catch(() => undefined);
  };
  try {
    for (const agent of lead === undefined ? called : [lead, ...called]) {
      const path = join(workspace, ".kiro", "agents", `${agent.name}.json`);
      if ((await exists(path)) && !(await ours(path))) {
        throw new LifecycleError(
          "kiro_workspace_file_exists",
          `The workspace already has .kiro/agents/${agent.name}.json, which NeMo Fabric did not write`,
        );
      }
      await ensure(join(workspace, ".kiro"));
      await ensure(join(workspace, ".kiro", "agents"));
      await writeFile(
        path,
        `${JSON.stringify(
          {
            name: agent.name,
            description: agent.description,
            ...(agent.prompt === undefined ? {} : { prompt: agent.prompt }),
            tools: agent.tools,
            ...(agent.model === undefined ? {} : { model: agent.model }),
            mcpServers: agent.mcpServers,
            resources: agent.resources,
            includeMcpJson: false,
            permissions: { rules: agent.rules },
            [AGENT_MARKER_KEY]: MARKER,
          },
          null,
          2,
        )}\n`,
      );
      written.push(path);
    }
    if (packageHooks.length > 0) {
      const path = join(workspace, ".kiro", "hooks", PACKAGE_HOOKS_FILE);
      if ((await exists(path)) && !(await ours(path))) {
        throw new LifecycleError(
          "kiro_workspace_file_exists",
          `The workspace already has .kiro/hooks/${PACKAGE_HOOKS_FILE}, which NeMo Fabric did not write`,
        );
      }
      await ensure(join(workspace, ".kiro"));
      await ensure(join(workspace, ".kiro", "hooks"));
      await writeFile(
        path,
        `${JSON.stringify(
          {
            version: "v1",
            hooks: packageHooks.map((hook, index) => ({
              name: `${hook.name}-${hook.event}-${index}`,
              description: `A hook of the agent package (${MARKER})`,
              trigger: hook.event,
              action: { type: "command", command: hook.command },
              timeout: 60,
              enabled: true,
            })),
          },
          null,
          2,
        )}\n`,
      );
      written.push(path);
    }
    if (preToolUse !== undefined) {
      // A standalone hook file covers every agent of the session, the called agents included.
      const path = join(workspace, ".kiro", "hooks", HOOKS_FILE);
      if ((await exists(path)) && !(await ours(path))) {
        throw new LifecycleError(
          "kiro_workspace_file_exists",
          `The workspace already has .kiro/hooks/${HOOKS_FILE}, which NeMo Fabric did not write`,
        );
      }
      await ensure(join(workspace, ".kiro"));
      await ensure(join(workspace, ".kiro", "hooks"));
      await writeFile(
        path,
        `${JSON.stringify(
          {
            version: "v1",
            hooks: [
              {
                name: "nemo-guardrails",
                description: `Asks NeMo Guardrails about every tool call (${MARKER})`,
                trigger: "PreToolUse",
                action: { type: "command", command: preToolUse.command },
                timeout: 30,
                enabled: true,
              },
            ],
          },
          null,
          2,
        )}\n`,
      );
      written.push(path);
    }
  } catch (error) {
    await cleanup();
    throw error;
  }
  const memberModels = called.flatMap((agent) => (agent.model === undefined ? [] : [{ agent: agent.name, model: agent.model }]));
  return { ...(lead === undefined ? {} : { leadMode: lead.name }), hooks: preToolUse !== undefined || packageHooks.length > 0, memberModels, cleanup };
}

const HOOKS_FORMAT = "https://agent-plugins.org/schemas/1.1.0/hooks.schema.json";
const SUPPORTED_EVENTS = ["UserPromptSubmit", "PreToolUse", "PostToolUse"];
const shellQuote = (value: string): string => `'${value.replaceAll("'", "'\\''")}'`;

/**
 * The hooks an agent package declares, as Kiro hook commands that run each handler through
 * package-hook.js. A hook the adapter cannot run as written stops the start with the reason.
 */
export async function packageHooks(config: AgentConfig, baseDir: string, script: string, data: string): Promise<PackageHook[]> {
  const declared = config.extensions?.[HOOKS_KEY] as { entries?: unknown } | undefined;
  if (declared === undefined || declared === null) return [];
  const entries = declared.entries;
  if (!Array.isArray(entries)) throw new LifecycleError("kiro_hook_invalid", `${HOOKS_KEY}.entries must be a list`);
  const hooks: PackageHook[] = [];
  for (const entry of entries) {
    if (!isRecord(entry) || typeof entry.path !== "string" || entry.format !== HOOKS_FORMAT) {
      throw new LifecycleError("kiro_hook_invalid", "a hook entry needs a path and the Agent Plugins 1.1.0 hooks format");
    }
    const root = await realpath(resolve(baseDir, entry.path)).catch(() => undefined);
    const name = basename(entry.path);
    if (root === undefined) throw new LifecycleError("kiro_hook_invalid", `hook ${name}: ${entry.path} is not a folder`);
    let document: unknown;
    try {
      document = JSON.parse(await readFile(join(root, "hooks.json"), "utf8"));
    } catch {
      throw new LifecycleError("kiro_hook_invalid", `hook ${name} has no readable hooks.json`);
    }
    const events = isRecord(document) && isRecord(document.hooks) ? document.hooks : undefined;
    if (events === undefined) throw new LifecycleError("kiro_hook_invalid", `hook ${name}: hooks.json has no hooks`);
    for (const [event, handlers] of Object.entries(events)) {
      if (!SUPPORTED_EVENTS.includes(event)) {
        throw new LifecycleError(
          "kiro_hook_event_unsupported",
          `hook ${name} declares ${event}, which the Kiro adapter cannot run; it runs ${SUPPORTED_EVENTS.join(", ")}`,
        );
      }
      if (!Array.isArray(handlers) || handlers.length === 0) {
        throw new LifecycleError("kiro_hook_invalid", `hook ${name}: ${event} must list at least one handler`);
      }
      for (const handler of handlers) {
        if (!isRecord(handler) || typeof handler.command !== "string" || handler.command.length === 0) {
          throw new LifecycleError("kiro_hook_invalid", `hook ${name}: a ${event} handler needs a command`);
        }
        const args = Array.isArray(handler.args) && handler.args.every((arg) => typeof arg === "string") ? handler.args : [];
        const command = handler.command.startsWith("./") ? join(root, handler.command) : handler.command;
        hooks.push({
          event: event as PackageHook["event"],
          name,
          command: [
            shellQuote(process.execPath),
            shellQuote(script),
            "--name", shellQuote(name),
            "--root", shellQuote(root),
            "--event", event,
            "--command", shellQuote(command),
            "--args", shellQuote(JSON.stringify(args)),
            "--data", shellQuote(join(data, name)),
          ].join(" "),
        });
      }
    }
  }
  return hooks;
}
