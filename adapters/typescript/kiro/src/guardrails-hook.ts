// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * A Kiro PreToolUse hook command that asks a NeMo Guardrails hooks server about the tool call.
 *
 * Kiro runs the command with the event on standard input and blocks the tool when the command
 * exits with a code other than 0, sending standard error to the agent. The server judges Claude
 * Code tool names on `/v1/hooks/claude-code`, so the call is named the way that endpoint reads
 * it. A server that does not answer denies the call, as the Claude adapter's hook does.
 */

const NAMES: Record<string, string> = {
  read_file: "Read",
  fs_read: "Read",
  write_file: "Write",
  fs_write: "Write",
  edit_file: "Edit",
  grep_search: "Grep",
  file_search: "Glob",
  execute_bash: "Bash",
  shell: "Bash",
  run_command: "Bash",
  web_fetch: "WebFetch",
  web_search: "WebSearch",
};

import { pathToFileURL } from "node:url";

type Json = Record<string, unknown>;

/** The tool name and input as the Claude Code endpoint reads them. */
export function claudeCall(toolName: string, toolInput: unknown): { tool_name: string; tool_input: Json } {
  const input: Json = typeof toolInput === "object" && toolInput !== null ? { ...(toolInput as Json) } : {};
  const mcp = /^([^:]+)::(.+)$/u.exec(toolName) ?? /^@?([^/]+)\/(.+)$/u.exec(toolName);
  if (mcp !== null) return { tool_name: `mcp__${mcp[1]}__${mcp[2]}`, tool_input: input };
  // Kiro's sub-agent tool is `subagent_<agent>` for each agent the session may start: Claude Code's Task.
  const name = /^(?:subagent|use_subagent)(?:[_/].*)?$/u.test(toolName) ? "Task" : (NAMES[toolName] ?? toolName);
  if (typeof input.path === "string" && input.file_path === undefined) input.file_path = input.path;
  return { tool_name: name, tool_input: input };
}

/** The exit code and the message for the server's answer. */
export function verdict(answer: unknown): { code: number; message: string } {
  if (typeof answer !== "object" || answer === null) return { code: 0, message: "" };
  const body = answer as Json;
  const decision = (body.hookSpecificOutput ?? {}) as Json;
  if (decision.permissionDecision === "deny") {
    return { code: 2, message: String(decision.permissionDecisionReason ?? "NeMo Guardrails denied the tool call") };
  }
  if (body.continue === false) {
    return { code: 2, message: String(body.stopReason ?? "NeMo Guardrails ended the run") };
  }
  return { code: 0, message: "" };
}

function option(args: string[], name: string): string | undefined {
  const at = args.indexOf(`--${name}`);
  return at === -1 ? undefined : args[at + 1];
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString("utf8");
}

export async function main(args: string[]): Promise<number> {
  const url = option(args, "url");
  if (url === undefined) {
    process.stderr.write("NeMo Guardrails hook: no --url\n");
    return 2;
  }
  const endpoint = `${url.replace(/\/+$/u, "")}/v1/hooks/claude-code`;
  const timeout = Number(option(args, "timeout") ?? "30") * 1000;
  const headers: Record<string, string> = { "content-type": "application/json" };
  for (const [flag, header] of [
    ["agent-name", "x-nemo-agent-name"],
    ["agent-digest", "x-nemo-agent-digest"],
    ["run-id", "x-nemo-run-id"],
  ] as const) {
    const value = option(args, flag);
    if (value !== undefined && value.length > 0) headers[header] = value;
  }
  let event: Json;
  try {
    event = JSON.parse(await readStdin()) as Json;
  } catch {
    process.stderr.write("NeMo Guardrails hook: the event is not JSON\n");
    return 2;
  }
  const call = claudeCall(String(event.tool_name ?? ""), event.tool_input);
  const body = {
    hook_event_name: "PreToolUse",
    session_id: event.session_id,
    cwd: event.cwd,
    ...call,
    tool_use_id: String(event.tool_use_id ?? ""),
  };
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeout),
    });
    const text = await response.text();
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { code, message } = verdict(text.length === 0 ? {} : JSON.parse(text));
    if (message.length > 0) process.stderr.write(`${message}\n`);
    return code;
  } catch (error) {
    process.stderr.write(`NeMo Guardrails at ${endpoint} did not answer: ${(error as Error).message}\n`);
    return 2;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    () => process.exit(2),
  );
}
