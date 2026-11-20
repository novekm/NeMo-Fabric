// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { mkdir, readdir, readFile, realpath, stat, writeFile } from "node:fs/promises";
import { basename, isAbsolute, join, resolve } from "node:path";

import type { AgentConfig } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterStartInput } from "nemo-fabric-adapters-common";

/**
 * The Strands CLI runs an agent project: `strands --agent <folder>/agent/agent.ts`. A project
 * is the only place a Strands agent can carry what the `strands` flags cannot say: named
 * called agents with their own instructions, tools, skills and MCP servers, a Cedar
 * authorization policy and a turn limit. The adapter writes one per runtime, from static
 * source templates (below) plus one `spec.json` that holds every value from the package. No
 * package value is ever spliced into source text.
 */
export const AGENTS_KEY = "nemo_fabric.agents.v1";
export const TOOLS_KEY = "nemo_fabric.tools.v1";

/** The Strands tool each portable or Strands-native tool name is. */
const STRANDS_TOOLS: Record<string, string> = {
  read_file: "read",
  write_file: "write",
  edit_file: "edit",
  shell: "shell",
  web_fetch: "web_fetch",
  web_search: "web_search",
  read: "read",
  write: "write",
  edit: "edit",
};

/** The built-in tools a Cedar policy can name: the ones a package can list. */
const POLICY_TOOLS = ["shell", "read", "write", "edit", "web_fetch", "web_search"];

export interface GuardrailsSetting {
  url: string;
  agent_name?: string;
  agent_digest?: string;
  timeout_seconds?: number;
}

export interface MemberSpec {
  name: string;
  description: string;
  instructions: string;
  model?: string;
  builtinTools: string[];
  skills: string[];
  mcpServers: Record<string, Record<string, unknown>>;
  maxTurns?: number;
}

export interface ProjectSpec {
  lead: {
    name: string;
    instructions?: string;
    model?: string;
    builtinTools: string[];
    skills: string[];
    maxTurns?: number;
  };
  members: MemberSpec[];
  /** Absolute path of the Cedar policy every agent is checked against. */
  cedarPolicy?: string;
  guardrails?: GuardrailsSetting;
  /** A file that gets one JSON line for each tool call and model call of every agent. */
  trace?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function isFile(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

/** Strands tool names for a list of portable or Strands-native tool names. */
export function strandsTools(names: readonly string[], where: string): string[] {
  const tools = new Set<string>();
  for (const name of names) {
    const strands = STRANDS_TOOLS[name];
    if (strands === undefined) {
      throw new LifecycleError("strands_tool_unsupported", `${where}: Strands has no built-in tool for \`${name}\``);
    }
    tools.add(strands);
  }
  return [...tools];
}

async function folders(entry: Record<string, unknown>, field: string, baseDir: string, where: string): Promise<string[]> {
  const value = entry[field];
  if (value === undefined) return [];
  const paths = isRecord(value) ? value.paths : undefined;
  if (!Array.isArray(paths) || paths.some((path) => typeof path !== "string" || path.length === 0)) {
    throw new LifecycleError("strands_called_agent_invalid", `${where}.${field} needs a list of folder paths`);
  }
  const found: string[] = [];
  for (const path of paths as string[]) {
    try {
      const folder = await realpath(resolve(baseDir, path));
      if (!(await stat(folder)).isDirectory()) throw new Error("not a folder");
      found.push(folder);
    } catch {
      throw new LifecycleError("strands_called_agent_invalid", `${where}.${field}: ${path} is not a folder`);
    }
  }
  return found;
}

/** Skill folders of a called agent: its `skills` folders and the skills inside each plugin folder. */
async function memberSkills(entry: Record<string, unknown>, baseDir: string, where: string): Promise<string[]> {
  const skills: string[] = [];
  for (const folder of await folders(entry, "skills", baseDir, where)) {
    if (!(await isFile(join(folder, "SKILL.md")))) {
      throw new LifecycleError("strands_skill_invalid", `${where}.skills: ${folder} has no SKILL.md`);
    }
    skills.push(folder);
  }
  for (const folder of await folders(entry, "plugins", baseDir, where)) {
    const inside = join(folder, "skills");
    let children: string[] = [];
    try {
      children = await readdir(inside);
    } catch {
      /* A plugin with no skills folder carries no skills. */
    }
    for (const child of children.sort()) {
      if (await isFile(join(inside, child, "SKILL.md"))) skills.push(join(inside, child));
    }
  }
  return skills;
}

/** The Strands model string a called agent's `{provider, model}` names. */
function memberModel(value: unknown, where: string): string | undefined {
  if (value === undefined) return undefined;
  if (!isRecord(value) || typeof value.provider !== "string" || typeof value.model !== "string" || value.model.length === 0) {
    throw new LifecycleError("strands_called_agent_invalid", `${where}.model needs a provider and a model`);
  }
  if (value.provider !== "bedrock") {
    throw new LifecycleError("strands_called_agent_invalid", `${where}.model: Strands runs a called agent on Bedrock only here`);
  }
  return `bedrock/${value.model}`;
}

/** Strands' form of the MCP servers a called agent's entry lists. */
function memberServers(value: unknown, where: string): Record<string, Record<string, unknown>> {
  if (value === undefined) return {};
  const servers = isRecord(value) ? value.servers : undefined;
  if (!isRecord(servers)) throw new LifecycleError("strands_called_agent_invalid", `${where}.mcp needs servers`);
  const written: Record<string, Record<string, unknown>> = {};
  for (const [name, server] of Object.entries(servers)) {
    if (!/^[A-Za-z0-9_-]+$/u.test(name) || !isRecord(server) || typeof server.url !== "string") {
      throw new LifecycleError("strands_called_agent_invalid", `${where}.mcp.servers.${name} is not a valid server`);
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
        url: server.url,
        transport: server.transport,
        ...(isRecord(server.custom_headers) ? { headers: server.custom_headers } : {}),
      };
    } else {
      throw new LifecycleError("strands_called_agent_invalid", `${where}.mcp.servers.${name} has a transport Strands cannot start`);
    }
  }
  return written;
}

export async function calledAgents(config: AgentConfig, baseDir: string): Promise<MemberSpec[]> {
  const declared = config.extensions?.[AGENTS_KEY];
  if (declared === undefined || declared === null) return [];
  if (!isRecord(declared)) {
    throw new LifecycleError("strands_called_agent_invalid", `${AGENTS_KEY} must be an object`);
  }
  const members: MemberSpec[] = [];
  for (const [name, entry] of Object.entries(declared)) {
    const where = `${AGENTS_KEY}.${name}`;
    if (!/^[a-z0-9_][a-z0-9_-]*$/u.test(name) || !isRecord(entry)) {
      throw new LifecycleError("strands_called_agent_invalid", `${where} is not a valid called agent`);
    }
    const { description, instructions } = entry;
    if (typeof description !== "string" || description.length === 0 || typeof instructions !== "string" || instructions.length === 0) {
      throw new LifecycleError("strands_called_agent_invalid", `${where} needs a description and instructions`);
    }
    for (const field of Object.keys(entry)) {
      if (!["description", "instructions", "tools", "skills", "plugins", "model", "mcp", "max_turns"].includes(field)) {
        throw new LifecycleError("strands_called_agent_invalid", `${where}.${field} is not supported`);
      }
    }
    const listed = entry.tools;
    if (listed !== undefined && (!Array.isArray(listed) || listed.some((tool) => typeof tool !== "string"))) {
      throw new LifecycleError("strands_called_agent_invalid", `${where}.tools must be a list of tool names`);
    }
    const maxTurns = entry.max_turns;
    if (maxTurns !== undefined && (!Number.isSafeInteger(maxTurns) || (maxTurns as number) < 1)) {
      throw new LifecycleError("strands_called_agent_invalid", `${where}.max_turns must be a positive integer`);
    }
    const model = memberModel(entry.model, where);
    members.push({
      name,
      description,
      instructions,
      ...(model === undefined ? {} : { model }),
      // An agent has the tools its definition names, and none when it names none.
      builtinTools: listed === undefined ? [] : strandsTools(listed as string[], where),
      skills: await memberSkills(entry, baseDir, where),
      mcpServers: memberServers(entry.mcp, where),
      ...(maxTurns === undefined ? {} : { maxTurns: maxTurns as number }),
    });
  }
  return members;
}

/**
 * The built-in tools the package or the config lists for the session's own agent. A package
 * that names called agents and lists no tools gives its agent none; any other config keeps
 * the adapter's default set.
 */
export function leadTools(config: AgentConfig, fallback: string[]): string[] {
  const packageTools = (config.extensions?.[TOOLS_KEY] as { builtin?: unknown } | undefined)?.builtin;
  if (packageTools !== undefined && (!Array.isArray(packageTools) || packageTools.some((tool) => typeof tool !== "string"))) {
    throw new LifecycleError("strands_tool_unsupported", `${TOOLS_KEY}.builtin must be a list of tool names`);
  }
  const enabled = config.tools?.enabled ?? (packageTools as string[] | undefined);
  const blocked = new Set(strandsTools(config.tools?.blocked ?? [], "tools.blocked"));
  const fromPackage = config.extensions?.[AGENTS_KEY] != null;
  const none = enabled == null && fromPackage;
  const tools = enabled == null ? (none ? [] : fallback) : strandsTools(enabled, "tools.enabled");
  return tools.filter((tool) => !blocked.has(tool));
}

/** The Cedar text for a policy that forbids tools by name, then the operator's own policy text. */
export async function cedarText(config: AgentConfig, baseDir: string, operatorPath: string | undefined): Promise<string | undefined> {
  const blocked = strandsTools(config.tools?.blocked ?? [], "tools.blocked");
  if (blocked.length === 0 && operatorPath === undefined) return undefined;
  const lines = ["// Written by the NeMo Fabric Strands adapter. Cedar denies a call that any forbid matches.", "permit(principal, action, resource);"];
  for (const tool of POLICY_TOOLS.filter((name) => blocked.includes(name))) {
    lines.push(`forbid(principal, action == Action::"${tool}", resource);`);
  }
  if (operatorPath !== undefined) {
    const path = isAbsolute(operatorPath) ? operatorPath : resolve(baseDir, operatorPath);
    let text: string;
    try {
      text = await readFile(path, "utf8");
    } catch {
      throw new LifecycleError("strands_policy_invalid", "harness.settings.cedar_policy must name a readable Cedar file");
    }
    lines.push(`// From ${basename(path)}`, text);
  }
  return `${lines.join("\n")}\n`;
}

const PACKAGE_JSON = `${JSON.stringify({ name: "fabric-strands-agent", private: true, type: "module" })}\n`;

const GOVERNANCE_TS = `import { appendFileSync } from 'node:fs'
import { InterventionHandler } from '@strands-agents/sdk'

const CLAUDE_NAMES: Record<string, string> = {
  read: 'Read', write: 'Write', edit: 'Edit', shell: 'Bash', web_fetch: 'WebFetch', web_search: 'WebSearch',
}

/** One agent's turn limit, and the NeMo Guardrails check of each of its tool calls. */
export class FabricGovernance extends InterventionHandler {
  readonly name = 'fabric-governance'
  private turns = 0
  constructor(
    private readonly agent: string,
    private readonly maxTurns: number | undefined,
    private readonly guardrails: { url: string; agent_name?: string; agent_digest?: string; timeout_seconds?: number } | undefined,
    private readonly trace: string | undefined,
    private readonly delegates: readonly string[]
  ) {
    super()
  }

  private record(entry: Record<string, unknown>): void {
    if (this.trace === undefined) return
    try {
      appendFileSync(this.trace, JSON.stringify({ agent: this.agent, ...entry }) + '\\n')
    } catch {
      /* A trace that cannot be written never stops a run. */
    }
  }

  override beforeInvocation() {
    this.turns = 0
    return { type: 'proceed' as const }
  }

  override beforeModelCall() {
    this.turns += 1
    this.record({ event: 'model_call', turn: this.turns })
    if (this.maxTurns !== undefined && this.turns > this.maxTurns) {
      return { type: 'deny' as const, reason: 'The turn limit of ' + this.maxTurns + ' is reached.' }
    }
    return { type: 'proceed' as const }
  }

  override async beforeToolCall(event: { toolUse: { name: string; input?: unknown; toolUseId?: string } }) {
    const guardrails = this.guardrails
    this.record({ event: 'tool_call', tool: event.toolUse.name, input: JSON.stringify(event.toolUse.input ?? {}).slice(0, 300) })
    if (guardrails === undefined) return { type: 'proceed' as const }
    const endpoint = guardrails.url.replace(/\\/+$/u, '') + '/v1/hooks/claude-code'
    const headers: Record<string, string> = { 'content-type': 'application/json' }
    if (guardrails.agent_name) headers['x-nemo-agent-name'] = guardrails.agent_name
    if (guardrails.agent_digest) headers['x-nemo-agent-digest'] = guardrails.agent_digest
    const name = event.toolUse.name
    const input = typeof event.toolUse.input === 'object' && event.toolUse.input !== null ? { ...(event.toolUse.input as object) } : {}
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          hook_event_name: 'PreToolUse',
          tool_name: this.delegates.includes(name) ? 'Task' : (CLAUDE_NAMES[name] ?? name),
          tool_input: input,
          tool_use_id: event.toolUse.toolUseId ?? '',
        }),
        signal: AbortSignal.timeout((guardrails.timeout_seconds ?? 30) * 1000),
      })
      if (!response.ok) throw new Error('HTTP ' + response.status)
      const text = await response.text()
      const body = (text.length === 0 ? {} : JSON.parse(text)) as Record<string, any>
      const decision = body.hookSpecificOutput ?? {}
      if (decision.permissionDecision === 'deny') {
        this.record({ event: 'guardrails_deny', tool: name })
        return { type: 'deny' as const, reason: String(decision.permissionDecisionReason ?? 'NeMo Guardrails denied the tool call') }
      }
      if (body.continue === false) {
        return { type: 'deny' as const, reason: String(body.stopReason ?? 'NeMo Guardrails ended the run') }
      }
      return { type: 'proceed' as const }
    } catch (error) {
      return { type: 'deny' as const, reason: 'NeMo Guardrails did not answer: ' + (error as Error).message }
    }
  }
}
`;

const AGENT_TS = `import { readFileSync } from 'node:fs'
import { createHarness } from '@strands-agents/harness'
import { FabricGovernance } from './governance.ts'

const spec = JSON.parse(readFileSync(new URL('../spec.json', import.meta.url), 'utf8'))

function interventions(agent: string, maxTurns?: number): unknown[] {
  const list: unknown[] = []
  if (maxTurns !== undefined || spec.guardrails !== undefined || spec.trace !== undefined) {
    const delegates = spec.members.map((member: { name: string }) => member.name)
    list.push(new FabricGovernance(agent, maxTurns, spec.guardrails, spec.trace, delegates))
  }
  if (spec.cedarPolicy) list.push(spec.cedarPolicy)
  return list
}

const members = []
for (const member of spec.members) {
  const called = await createHarness({
    name: member.name,
    description: member.description,
    instructions: member.instructions,
    ...(member.model ? { model: member.model } : {}),
    builtinTools: member.builtinTools,
    skills: member.skills.length > 0 ? member.skills : false,
    mcpServers: member.mcpServers,
    interventions: interventions(member.name, member.maxTurns) as any,
    builtinPlugins: ['environment'],
    session: false,
    memory: false,
    backgroundTasks: false,
  } as any)
  members.push(called.asTool())
}

export const agent = await createHarness({
  name: spec.lead.name,
  ...(spec.lead.instructions ? { instructions: spec.lead.instructions } : {}),
  ...(spec.lead.model ? { model: spec.lead.model } : {}),
  builtinTools: spec.lead.builtinTools,
  skills: spec.lead.skills.length > 0 ? spec.lead.skills : false,
  tools: members,
  interventions: interventions(spec.lead.name, spec.lead.maxTurns) as any,
  builtinPlugins: ['environment'],
  session: false,
  memory: false,
  backgroundTasks: false,
} as any)
`;

/** Writes the project under `dir` and returns the path of its `agent.ts`. */
export async function writeProject(dir: string, spec: ProjectSpec, cedar: string | undefined): Promise<string> {
  const source = join(dir, "agent");
  await mkdir(source, { recursive: true });
  const withPolicy: ProjectSpec = cedar === undefined ? spec : { ...spec, cedarPolicy: join(dir, "policy.cedar") };
  if (cedar !== undefined) await writeFile(join(dir, "policy.cedar"), cedar, { mode: 0o600 });
  await writeFile(join(dir, "package.json"), PACKAGE_JSON, { mode: 0o600 });
  await writeFile(join(dir, "spec.json"), `${JSON.stringify(withPolicy, null, 2)}\n`, { mode: 0o600 });
  await writeFile(join(source, "governance.ts"), GOVERNANCE_TS, { mode: 0o600 });
  await writeFile(join(source, "agent.ts"), AGENT_TS, { mode: 0o600 });
  return join(source, "agent.ts");
}

export async function buildSpec(
  input: AdapterStartInput,
  parts: {
    name: string;
    instructions: string | undefined;
    model: string | undefined;
    builtinTools: string[];
    skills: string[];
    guardrails: GuardrailsSetting | undefined;
    trace?: string | undefined;
  },
): Promise<ProjectSpec> {
  const members = await calledAgents(input.config, input.baseDir);
  const maxTurns = input.config.runtime?.max_turns;
  return {
    lead: {
      name: parts.name,
      ...(parts.instructions === undefined ? {} : { instructions: parts.instructions }),
      ...(parts.model === undefined ? {} : { model: parts.model }),
      builtinTools: leadTools(input.config, parts.builtinTools),
      skills: parts.skills,
      ...(typeof maxTurns === "number" ? { maxTurns } : {}),
    },
    members,
    ...(parts.guardrails === undefined ? {} : { guardrails: parts.guardrails }),
    ...(parts.trace === undefined ? {} : { trace: parts.trace }),
  };
}
