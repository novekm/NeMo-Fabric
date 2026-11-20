// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AdapterCheckResult,
  CheckLevel,
  AgentUsage,
} from "nemo-fabric-adapter-contract";
import {
  AcpRunnerError,
  LifecycleError,
  unobservedHealth,
  startAcpSession,
  type AcpPromptResult,
  type AcpSession,
  type AdapterStartInput,
} from "nemo-fabric-adapters-common";

import { selectMcpServers, selectModel, selectSettings, workspacePath, type KiroSettings } from "./configuration.js";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { packageHooks, prepareWorkspaceFiles, type HookCommand, type WorkspaceFiles } from "./workspace-files.js";

export interface KiroPromptOutcome {
  text: string;
  stopReason: AcpPromptResult["stopReason"];
  usage?: AgentUsage;
}

export interface KiroSessionHandle {
  readonly id: string;
  check(level: CheckLevel): Promise<AdapterCheckResult>;
  prompt(text: string, timeoutMs?: number): Promise<KiroPromptOutcome>;
  stop(): Promise<void>;
}

export interface KiroSessionFactory {
  create(input: AdapterStartInput): Promise<KiroSessionHandle>;
}

const INHERITED_ENVIRONMENT_NAMES = new Set([
  "APPDATA", "COMSPEC", "ComSpec", "HOME", "HTTP_PROXY", "HTTPS_PROXY", "KIRO_API_KEY", "KIRO_HOME", "LANG",
  "LC_ALL", "LC_CTYPE", "LOCALAPPDATA", "NO_COLOR", "NO_PROXY", "PATH", "PATHEXT", "Path", "PathExt",
  "SHELL", "SSL_CERT_DIR", "SSL_CERT_FILE", "SYSTEMROOT", "SystemRoot", "TEMP", "TERM", "TMP", "TMPDIR",
  "USER", "USERPROFILE", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR", "http_proxy",
  "https_proxy", "no_proxy",
]);

function childEnvironment(input: AdapterStartInput): NodeJS.ProcessEnv {
  const environment: NodeJS.ProcessEnv = {};
  for (const name of INHERITED_ENVIRONMENT_NAMES) {
    const value = process.env[name];
    if (value !== undefined) environment[name] = value;
  }
  Object.assign(environment, input.runtimeContext.environment.env ?? {});
  environment.KIRO_LOG_NO_COLOR = "1";
  environment.NO_COLOR = "1";
  return environment;
}

function translateRunnerError(error: AcpRunnerError, stage: "start" | "prompt", timeoutMs = 300_000): LifecycleError {
  if (error.code === "acp_process_unavailable") {
    return new LifecycleError("kiro_harness_unavailable", "The installed Kiro CLI could not be started");
  }
  if (error.code === "acp_protocol_unsupported") {
    return new LifecycleError("kiro_acp_protocol_unsupported", "The installed Kiro CLI selected an unsupported ACP version");
  }
  if (error.code === "acp_mode_unsupported") {
    return new LifecycleError("kiro_mode_unsupported", "The installed Kiro CLI does not advertise the configured mode");
  }
  if (error.code === "acp_config_option_unsupported") {
    return new LifecycleError(
      "kiro_session_option_unsupported",
      "The installed Kiro CLI does not advertise the configured model or effort",
    );
  }
  if (error.code === "acp_mcp_transport_unsupported") {
    return new LifecycleError(
      "kiro_mcp_transport_unsupported",
      "The installed Kiro CLI does not advertise the configured MCP transport",
    );
  }
  if (error.code === "acp_prompt_timeout") {
    return new LifecycleError(
      "timeout",
      `Kiro adapter invocation timed out after ${timeoutMs / 1000} seconds`,
      { metadata: {
        owner: "adapter",
        timeout_seconds: timeoutMs / 1000,
        cause_code: "kiro_prompt_timeout",
        native_code: error.code,
      } },
    );
  }
  return new LifecycleError(
    stage === "start" ? "kiro_start_failed" : "kiro_session_failed",
    stage === "start" ? "Kiro ACP initialization failed" : "Kiro ACP session communication failed",
    { retryable: stage === "start" && error.retryable },
  );
}

function validUsageValue(value: number | null | undefined): value is number {
  return value !== undefined && value !== null && Number.isSafeInteger(value) && value >= 0;
}

type AcpUsage = NonNullable<AcpPromptResult["usage"]>;

function kiroMcpReady(params: unknown, sessionId: string, serverNames: string[]): boolean {
  if (typeof params !== "object" || params === null) return false;
  const status = params as { sessionId?: unknown; servers?: unknown };
  const servers = status.servers;
  if (status.sessionId !== sessionId || !Array.isArray(servers)) return false;
  return serverNames.every((name) =>
    servers.some(
      (server: unknown) =>
        typeof server === "object" &&
        server !== null &&
        (server as { name?: unknown }).name === name &&
        (server as { status?: unknown }).status === "connected",
    ),
  );
}

function validateUsage(usage: AcpUsage): void {
  if (![usage.inputTokens, usage.outputTokens, usage.totalTokens].every(validUsageValue)) {
    throw new LifecycleError("kiro_malformed_result", "Kiro returned malformed ACP usage metadata");
  }
}

function normalizeUsage(current: AcpUsage, previous: AcpUsage | undefined): AgentUsage {
  validateUsage(current);
  const baseline = previous ?? { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
  validateUsage(baseline);
  const input = current.inputTokens - baseline.inputTokens;
  const output = current.outputTokens - baseline.outputTokens;
  const total = current.totalTokens - baseline.totalTokens;
  if (input < 0 || output < 0 || total < 0) {
    throw new LifecycleError("kiro_malformed_result", "Kiro returned decreasing cumulative ACP usage metadata");
  }
  const cachedCurrent = validUsageValue(current.cachedReadTokens) ? current.cachedReadTokens : undefined;
  const cachedPrevious = validUsageValue(previous?.cachedReadTokens) ? previous.cachedReadTokens : undefined;
  const cached =
    cachedCurrent === undefined || (previous !== undefined && cachedPrevious === undefined)
      ? undefined
      : cachedCurrent - (cachedPrevious ?? 0);
  if (cached !== undefined && cached < 0) {
    throw new LifecycleError("kiro_malformed_result", "Kiro returned decreasing cumulative ACP usage metadata");
  }
  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: total,
    ...(cached === undefined ? {} : { cached_input_tokens: cached }),
  };
}

const shellQuote = (value: string): string => `'${value.replaceAll("'", "'\\''")}'`;

/** The PreToolUse command that asks a NeMo Guardrails hooks server, when the settings name one. */
function guardrailsHook(settings: KiroSettings, input: AdapterStartInput): HookCommand | undefined {
  const setting = settings.guardrails;
  if (setting === undefined) return undefined;
  const script = fileURLToPath(new URL("./guardrails-hook.js", import.meta.url));
  const flags: Array<[string, string | undefined]> = [
    ["url", setting.url],
    ["agent-name", setting.agentName ?? input.agentName],
    ["agent-digest", setting.agentDigest],
    ["run-id", process.env.FABRIC_RUNTIME_ID],
    ["timeout", setting.timeoutSeconds === undefined ? undefined : String(setting.timeoutSeconds)],
  ];
  const args = flags.flatMap(([flag, value]) => (value === undefined ? [] : [`--${flag}`, shellQuote(value)]));
  return { command: [shellQuote(process.execPath), shellQuote(script), ...args].join(" ") };
}

class KiroAcpSession implements KiroSessionHandle {
  private stopped = false;
  readonly id: string;
  private readonly session: AcpSession;
  private previousUsage?: AcpPromptResult["usage"];
  private usageContinuityLost = false;

  private readonly files: WorkspaceFiles;

  constructor(session: AcpSession, files: WorkspaceFiles) {
    this.files = files;
    this.session = session;
    this.id = session.sessionId;
  }

  async check(level: CheckLevel): Promise<AdapterCheckResult> {
    if (this.stopped)
      return {
        ...unobservedHealth("session_closed"),
        outcome: "failed",
        activity: "stopped",
      };
    if (level !== "live")
      return {
        ...unobservedHealth("unsupported_level"),
        outcome: "unsupported",
      };
    const { alive } = this.session.observeProcess();
    const observed_at = new Date().toISOString();
    return alive
      ? {
          outcome: "passed",
          reason_code: null,
          activity: "unknown",
          source: "fabric.adapter.liveness",
          observed_at,
        }
      : {
          outcome: "failed",
          reason_code: "agent_exited",
          activity: "stopped",
          source: "fabric.adapter.liveness",
          observed_at,
        };
  }

  async prompt(text: string, timeoutMs?: number): Promise<KiroPromptOutcome> {
    let result: AcpPromptResult;
    try {
      result = await this.session.prompt(text, timeoutMs ?? null);
    } catch (error) {
      if (error instanceof AcpRunnerError) throw translateRunnerError(error, "prompt", timeoutMs);
      throw new LifecycleError("kiro_session_failed", "Kiro ACP session communication failed");
    }
    let usage: AgentUsage | undefined;
    try {
      if (result.usage === undefined) {
        this.previousUsage = undefined;
        this.usageContinuityLost = true;
      } else if (this.usageContinuityLost) {
        validateUsage(result.usage);
        this.previousUsage = result.usage;
        this.usageContinuityLost = false;
      } else {
        usage = normalizeUsage(result.usage, this.previousUsage);
        this.previousUsage = result.usage;
      }
    } catch {
      // Optional usage metadata must not discard a completed invocation.
      usage = undefined;
      this.previousUsage = undefined;
      this.usageContinuityLost = true;
    }
    return {
      text: result.text,
      stopReason: result.stopReason,
      ...(usage === undefined ? {} : { usage }),
    };
  }

  async stop(): Promise<void> {
    this.stopped = true;
    try {
      await this.session.stop();
    } finally {
      await this.files.cleanup();
    }
  }
}

export class KiroAcpSessionFactory implements KiroSessionFactory {
  async create(input: AdapterStartInput): Promise<KiroSessionHandle> {
    const workspace = await workspacePath(input);
    const settings = selectSettings(input.config);
    const model = selectModel(input.config);
    const configuredEnvironment = input.runtimeContext.environment.env ?? {};
    const mcpServers = selectMcpServers(input.config, configuredEnvironment, process.env);
    const configOptions = [
      ...(model === undefined ? [] : [{ id: "model", value: model }]),
      ...(settings.effort === undefined ? [] : [{ id: "effortLevel", value: settings.effort }]),
    ];
    const files = await prepareWorkspaceFiles(
      input.config,
      workspace,
      input.baseDir,
      input.agentName,
      mcpServers.map((server) => ({ name: server.name })),
      guardrailsHook(settings, input),
      await packageHooks(
        input.config,
        input.baseDir,
        fileURLToPath(new URL("./package-hook.js", import.meta.url)),
        join(tmpdir(), "nemo-fabric-kiro-hook-data"),
      ),
    );
    if (files.leadMode !== undefined && settings.modeId !== undefined) {
      await files.cleanup();
      throw new LifecycleError(
        "kiro_mode_conflict",
        "harness.settings.mode_id selects a Kiro mode, and the package's own agent needs the mode: set one or the other",
      );
    }
    const modeId = files.leadMode ?? settings.modeId;
    try {
      const start = (): Promise<AcpSession> => startAcpSession({
        command: "kiro-cli",
        args: ["acp", "--agent-engine=v3"],
        cwd: workspace,
        env: childEnvironment(input),
        mcpServers,
        ...(mcpServers.length === 0
          ? {}
          : {
              readinessNotification: {
                method: "_kiro/mcp/status",
                isReady: (params: unknown, sessionId: string) =>
                  kiroMcpReady(
                    params,
                    sessionId,
                    mcpServers.map((server) => server.name),
                  ),
              },
            }),
        ...(modeId === undefined ? {} : { modeId }),
        // Kiro runs hooks in an ACP session only for a client that opts in.
        ...(files.hooks ? { clientCapabilityMeta: { kiro: { hooks: { enabled: true, v2: true } } } } : {}),
        configOptions,
        permissionResolver: () => settings.permissionMode,
        clientName: "nemo-fabric-kiro",
        clientVersion: "0.6.0",
      });
      // Kiro now and then starts a session without the model list it advertises a moment later: ask once more.
      let session: AcpSession;
      try {
        session = await start();
      } catch (error) {
        if (!(error instanceof AcpRunnerError) || error.code !== "acp_config_option_unsupported") throw error;
        await new Promise((resolve) => setTimeout(resolve, 1000));
        session = await start();
      }
      const advertised = session.advertisedValues("model");
      const missing = files.memberModels.filter((member) => !advertised.includes(member.model));
      if (missing.length > 0) {
        await session.stop().catch(() => undefined);
        await files.cleanup();
        throw new LifecycleError(
          "kiro_member_model_unsupported",
          `Kiro does not advertise the model of called agent ${missing.map((member) => `${member.agent} (${member.model})`).join(", ")}`,
        );
      }
      return new KiroAcpSession(session, files);
    } catch (error) {
      await files.cleanup();
      if (error instanceof AcpRunnerError) throw translateRunnerError(error, "start");
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("kiro_start_failed", "Kiro ACP initialization failed", { retryable: true });
    }
  }
}
