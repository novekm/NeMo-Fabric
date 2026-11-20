// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * A Kiro hook command that runs one handler of an agent package's hook, the way the Agent
 * Plugins 1.1.0 hooks format says a client does: a new process for the event, the hook folder as
 * its working directory, one JSON object on standard input and one answer on standard output.
 *
 * Kiro blocks a tool on a non-zero exit of a PreToolUse command and sends standard error to the
 * agent; standard output of a command that exits 0 is added to the agent's context. The
 * handler's answer is read into those: a PreToolUse `deny`, or a handler that fails, exits 2; an
 * `additional_context` answer is written to standard output.
 */

import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { claudeCall } from "./guardrails-hook.js";

type Json = Record<string, unknown>;

const HANDLER_TIMEOUT_MS = 45_000;
const ENV_NAMES = ["COMSPEC", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "PATH", "PATHEXT", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "USER", "USERPROFILE"];

/** The closed answers the format defines for each event; null when `value` is not one. */
export function answer(event: string, value: unknown): { block?: string; context?: string } | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const body = value as Json;
  const keys = Object.keys(body).sort().join(",");
  const text = (name: string): string | undefined =>
    typeof body[name] === "string" && body[name] !== "" ? (body[name] as string) : undefined;
  if (keys === "additional_context") {
    const context = text("additional_context");
    return event === "UserPromptSubmit" || event === "PostToolUse" ? (context === undefined ? null : { context }) : null;
  }
  if (keys === "decision") return event === "PreToolUse" && body.decision === "allow" ? {} : null;
  if (keys === "decision,reason") {
    const reason = text("reason");
    return event === "PreToolUse" && body.decision === "deny" && reason !== undefined ? { block: reason } : null;
  }
  return null;
}

/** The format's event, from the event Kiro gave the hook. */
export function formatEvent(event: string, kiro: Json): Json {
  const payload: Json = { hook_event_name: event, session_id: kiro.session_id, cwd: kiro.cwd };
  if (event === "UserPromptSubmit") payload.prompt = kiro.prompt ?? kiro.user_prompt ?? "";
  else {
    const call = claudeCall(String(kiro.tool_name ?? ""), kiro.tool_input);
    payload.tool_use_id = String(kiro.tool_use_id ?? "");
    payload.tool_name = call.tool_name;
    payload.tool_input = call.tool_input;
    if (event === "PostToolUse") payload.tool_result = kiro.tool_response ?? kiro.tool_result ?? {};
  }
  return payload;
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
  const root = option(args, "root");
  const event = option(args, "event");
  const command = option(args, "command");
  const handlerArgs = JSON.parse(option(args, "args") ?? "[]") as string[];
  const data = option(args, "data");
  const name = option(args, "name") ?? "hook";
  if (root === undefined || event === undefined || command === undefined) {
    process.stderr.write("package hook: --root, --event and --command are required\n");
    return 2;
  }
  const fail = (reason: string): number => {
    process.stderr.write(`The hook '${name}' could not check this ${event}: ${reason}\n`);
    return event === "PreToolUse" ? 2 : 0;
  };
  let kiro: Json;
  try {
    kiro = JSON.parse(await readStdin()) as Json;
  } catch {
    return fail("the Kiro event is not JSON");
  }
  const environment: NodeJS.ProcessEnv = {};
  for (const variable of ENV_NAMES) if (process.env[variable] !== undefined) environment[variable] = process.env[variable];
  environment.PLUGIN_ROOT = root;
  if (data !== undefined) {
    mkdirSync(data, { recursive: true });
    environment.PLUGIN_DATA = data;
  }
  const expand = (value: string): string => value.replace(/\$\{(PLUGIN_ROOT|PLUGIN_DATA)\}/gu, (_all, key: string) => environment[key] ?? "");
  const output = await new Promise<{ code: number | null; stdout: string }>((resolve) => {
    const child = spawn(expand(command), handlerArgs.map(expand), { cwd: root, env: environment, stdio: ["pipe", "pipe", "ignore"] });
    let stdout = "";
    const timer = setTimeout(() => child.kill("SIGKILL"), HANDLER_TIMEOUT_MS);
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.on("error", () => resolve({ code: -1, stdout: "" }));
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code, stdout });
    });
    child.stdin.on("error", () => undefined);
    child.stdin.end(JSON.stringify(formatEvent(event, kiro)));
  });
  if (output.code !== 0) return fail(`the handler exited with status ${output.code}`);
  if (output.stdout.trim().length === 0) return 0;
  let parsed: unknown;
  try {
    parsed = JSON.parse(output.stdout);
  } catch {
    return fail("the handler wrote output that is not JSON");
  }
  const result = answer(event, parsed);
  if (result === null) return fail(`the handler wrote an answer ${event} does not define`);
  if (result.block !== undefined) {
    process.stderr.write(`${result.block}\n`);
    return 2;
  }
  if (result.context !== undefined) process.stdout.write(`${result.context}\n`);
  return 0;
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    () => process.exit(2),
  );
}
