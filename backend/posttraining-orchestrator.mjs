import process from "node:process";
import path from "node:path";
import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

import { recommendDatasets } from "./hf-dataset-recommender.mjs";
import {
  loadCompilerOverrides,
  runCompiler,
} from "./posttraining-spec-compiler.mjs";
import {
  buildTrainingMetricGraphArtifacts,
  parseStructuredTrainingMetricLine,
} from "./training-metrics.mjs";
import {
  buildUiProgressEvent,
  emitUiProgress,
} from "../src/lib/posttraining-progress.mjs";

const backendDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(backendDir, "..");
const jobsRoot = path.join(backendDir, "generated-posttraining-jobs");
const trainerScriptPath = path.join(backendDir, "modal_trl_posttrain.py");
const serveScriptPath = path.join(backendDir, "modal_vllm_serve.py");
const STRUCTURED_LIFECYCLE_EVENT_PREFIX = "PT_LIFECYCLE_EVENT::";
const DEFAULT_VLLM_PACKAGE_SPEC = "vllm==0.18.0";
const DEFAULT_SERVE_TRANSFORMERS_SPEC = "transformers>=4.56.0,<5";
const STRUCTURED_PROGRESS_PREFIX = "PT_PROGRESS::";
const ANSI_ESCAPE_SEQUENCE_RE = /\u001B\[[0-9;?]*[ -/]*[@-~]/g;
const TOKENIZER_MISMATCH_WARNING_CORE =
  "mismatch between tokenized prompt and the start of tokenized prompt+completion";
const TOKENIZER_MISMATCH_WARNING_CONTEXT = [
  "unexpected tokenizer behavior",
  "whitespace issues",
  "special token handling",
];

function nowIso() {
  return new Date().toISOString();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function truncateText(value, maxLength = 500) {
  const text = String(value ?? "");
  return text.length <= maxLength ? text : `${text.slice(0, maxLength - 3)}...`;
}

function readPositiveIntegerEnv(name, fallback) {
  const raw = process.env[name];
  if (raw == null || raw === "") {
    return fallback;
  }
  const parsed = Number.parseInt(String(raw), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function getServeStartupTimeoutSeconds() {
  return readPositiveIntegerEnv("POSTTRAINING_SERVE_STARTUP_TIMEOUT_SECONDS", 30 * 60);
}

function getSmokeTestConfig() {
  const serveStartupTimeoutSeconds = getServeStartupTimeoutSeconds();
  return {
    maxAttempts: readPositiveIntegerEnv("POSTTRAINING_SMOKE_TEST_MAX_ATTEMPTS", 12),
    retryDelayMs: readPositiveIntegerEnv("POSTTRAINING_SMOKE_TEST_RETRY_DELAY_MS", 15_000),
    initialModelsTimeoutMs: readPositiveIntegerEnv(
      "POSTTRAINING_SMOKE_TEST_INITIAL_MODELS_TIMEOUT_MS",
      (serveStartupTimeoutSeconds + 120) * 1000,
    ),
    modelsTimeoutMs: readPositiveIntegerEnv("POSTTRAINING_SMOKE_TEST_MODELS_TIMEOUT_MS", 90_000),
    chatTimeoutMs: readPositiveIntegerEnv("POSTTRAINING_SMOKE_TEST_CHAT_TIMEOUT_MS", 180_000),
  };
}

function summarizeData(value, depth = 0) {
  if (value == null || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    return truncateText(value, 500);
  }
  if (depth >= 2) {
    return truncateText(JSON.stringify(value), 500);
  }
  if (Array.isArray(value)) {
    return value.slice(0, 10).map((item) => summarizeData(item, depth + 1));
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 20)
        .map(([key, nested]) => [key, summarizeData(nested, depth + 1)]),
    );
  }
  return truncateText(String(value), 500);
}

function safeError(error) {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack ? truncateText(error.stack, 1500) : null,
    };
  }
  return {
    name: "Error",
    message: truncateText(String(error), 1000),
    stack: null,
  };
}

function buildTrainingMetricUiProgress(record) {
  const step = Number(record?.step);
  if (!Number.isFinite(step) || step <= 0) {
    return null;
  }

  return "I'm training the model";
}

export function detectTrainingStageUiProgress(line, stage) {
  const text = String(line ?? "");
  const structuredProgress = text.startsWith(STRUCTURED_PROGRESS_PREFIX)
    ? text.slice(STRUCTURED_PROGRESS_PREFIX.length).trim()
    : null;

  if (structuredProgress?.startsWith("synthesizing_generation_cases ")) {
    const counts = structuredProgress.slice("synthesizing_generation_cases ".length);
    return `I'm preparing the evaluation cases (${counts})`;
  }

  if (structuredProgress?.startsWith("running_baseline_generation_cases ")) {
    return "I'm testing the base model";
  }

  if (structuredProgress?.startsWith("running_candidate_generation_cases ")) {
    return "I'm testing the tuned model";
  }

  if (
    structuredProgress?.startsWith("scoring_generation_cases ") ||
    structuredProgress?.startsWith("judging_generation_cases ")
  ) {
    return "I'm scoring each model against the expected outputs";
  }

  if (structuredProgress === "preparing_merged_deployment_artifact") {
    return "I'm packaging the merged model for deployment";
  }

  if (structuredProgress === "prepared_merged_deployment_artifact") {
    return "I'm finishing the evaluation artifacts";
  }

  if (
    text.includes("Generating train split:") ||
    text.includes("Preparing ") ||
    text.includes("Annotating provenance") ||
    text.includes("Normalizing SFT text dataset")
  ) {
    return "I'm getting the training data ready";
  }

  if (text.includes("Fetching ") && text.includes(" files:")) {
    return "I'm downloading the model files";
  }

  if (text.includes("Loading weights:")) {
    return stage === "evaluating" ? "I'm loading the model for testing" : "I'm loading the model";
  }

  return null;
}

function detectDeploymentStageUiProgress(line) {
  const text = String(line ?? "");

  if (
    text.startsWith("Building image ") ||
    text.startsWith("=> Step ") ||
    text.startsWith("Saving image") ||
    text.startsWith("Built image ")
  ) {
    return "I'm getting the model ready to go live";
  }

  if (text.includes("✓ Created objects.") || text.includes("✓ App deployed in ")) {
    return "I'm setting up the live model service";
  }

  if (text.includes("Created web function serve")) {
    return "I'm creating the live model link";
  }

  return null;
}

function formatSmokeTestProbeLabel(probeId) {
  switch (String(probeId ?? "")) {
    case "role_demo":
      return "style example";
    case "representative_response":
      return "real-world example";
    default:
      return String(probeId ?? "").replace(/[_-]+/g, " ").trim() || "example";
  }
}

function normalizeModalTrainingGpuType(gpuType) {
  const normalized = String(gpuType ?? "").trim().toUpperCase();
  if (normalized === "A10" || normalized === "A10G") {
    return "A10G";
  }
  if (normalized === "L40S") {
    return "L40S";
  }
  if (normalized === "H100") {
    return "H100";
  }
  return null;
}

function isGenerationTaskSpec(taskSpec) {
  return String(taskSpec?.task_family ?? "").trim().toLowerCase() === "generation";
}

export function getRecommendationStageSelectedDatasets(
  recommendation,
  overriddenSftDataset = null,
) {
  if (overriddenSftDataset !== null && overriddenSftDataset !== undefined) {
    return [];
  }

  if (!Array.isArray(recommendation?.recommended_datasets)) {
    return [];
  }

  return recommendation.recommended_datasets
    .map((item) => (typeof item?.dataset === "string" ? item.dataset.trim() : ""))
    .filter(Boolean);
}

export function parseStructuredLifecycleEventLine(line) {
  const text = String(line ?? "");
  if (!text.startsWith(STRUCTURED_LIFECYCLE_EVENT_PREFIX)) {
    return null;
  }

  try {
    const parsed = JSON.parse(text.slice(STRUCTURED_LIFECYCLE_EVENT_PREFIX.length));
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

export function resolveDeploymentArtifact(trainingResult) {
  const merged = Boolean(trainingResult?.merged_dir);
  const artifactPath = merged ? trainingResult?.merged_dir : trainingResult?.final_adapter_dir;
  if (!artifactPath) {
    throw new Error("Training did not produce a deployable artifact.");
  }
  return {
    merged,
    path: artifactPath,
    relativePath: toRelativeAdapterPath(artifactPath),
  };
}

function getServePackageSpecs() {
  return {
    vllmPackageSpec:
      process.env.POSTTRAINING_VLLM_PACKAGE_SPEC ||
      process.env.VLLM_PACKAGE_SPEC ||
      DEFAULT_VLLM_PACKAGE_SPEC,
    transformersSpec:
      process.env.POSTTRAINING_SERVE_TRANSFORMERS_SPEC ||
      process.env.SERVE_TRANSFORMERS_SPEC ||
      DEFAULT_SERVE_TRANSFORMERS_SPEC,
  };
}

export function resolveDeploymentRuntimePolicy(trainingResult, deploymentArtifact = null) {
  const artifact = deploymentArtifact ?? resolveDeploymentArtifact(trainingResult);
  const packageSpecs = getServePackageSpecs();
  return {
    merged: artifact.merged,
    vllmPackageSpec: packageSpecs.vllmPackageSpec,
    transformersSpec: packageSpecs.transformersSpec,
    vllmUseV1: "default",
    modelImpl: "auto",
    languageModelOnly: false,
  };
}

export function buildDeploymentEnvironment({
  deploymentAppName,
  deploymentArtifact,
  adapterName,
  trainingResult,
  compiledConfig,
  serveGpuType,
  serveStartupTimeoutSeconds,
  runtimePolicy = resolveDeploymentRuntimePolicy(trainingResult, deploymentArtifact),
}) {
  return {
    APP_NAME: deploymentAppName,
    ADAPTER_PATH: deploymentArtifact.relativePath,
    ADAPTER_NAME: adapterName,
    BASE_MODEL: trainingResult.base_model,
    MERGED: deploymentArtifact.merged ? "1" : "0",
    MAX_MODEL_LEN: String(compiledConfig.max_length ?? 2048),
    GPU_TYPE: serveGpuType,
    SERVE_STARTUP_TIMEOUT_SECONDS: String(serveStartupTimeoutSeconds),
    VLLM_PACKAGE_SPEC: runtimePolicy.vllmPackageSpec,
    SERVE_TRANSFORMERS_SPEC: runtimePolicy.transformersSpec,
  };
}

export function buildDeploymentRecord({
  deploymentUrl,
  deploymentAppName,
  adapterName,
  deploymentArtifact,
  trainingResult,
  serveGpuType,
  serveStartupTimeoutSeconds,
  runtimePolicy,
}) {
  return {
    deployedAt: nowIso(),
    appName: deploymentAppName,
    url: deploymentUrl,
    adapterName,
    adapterPath: deploymentArtifact.relativePath,
    baseModel: trainingResult.base_model,
    gpuType: serveGpuType,
    startupTimeoutSeconds: serveStartupTimeoutSeconds,
    merged: deploymentArtifact.merged,
    runtimePolicy,
  };
}

function normalizeDescriptionForProbe(description) {
  return truncateText(String(description ?? "").replace(/\s+/g, " ").trim(), 700);
}

export function buildGenerationSmokeTestProbes(description) {
  const objective = normalizeDescriptionForProbe(description) || "Follow the user's requested role and style.";

  return [
    {
      id: "role_demo",
      prompt: [
        "You are validating a freshly fine-tuned model.",
        `Target behavior: ${objective}`,
        "In 3-5 sentences, demonstrate the intended role, tone, and style for a user.",
        "Do not mention training, system prompts, or evaluation.",
      ].join("\n\n"),
      minLength: 40,
      maxTokens: 160,
    },
    {
      id: "representative_response",
      prompt: [
        "You are validating a freshly fine-tuned model.",
        `Target behavior: ${objective}`,
        "Write one short representative response that this model might give to a realistic user request in that style.",
        "Keep it concise but substantive.",
      ].join("\n\n"),
      minLength: 40,
      maxTokens: 200,
    },
  ];
}

function extractChatCompletionText(chatPayload) {
  const content = chatPayload?.choices?.[0]?.message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((item) => (typeof item?.text === "string" ? item.text : ""))
      .join("")
      .trim();
  }
  if (typeof chatPayload?.choices?.[0]?.text === "string") {
    return chatPayload.choices[0].text;
  }
  return "";
}

function getJobPaths(jobId) {
  const jobDir = path.join(jobsRoot, jobId);
  return {
    jobDir,
    jobFilePath: path.join(jobDir, "job.json"),
    eventsFilePath: path.join(jobDir, "events.jsonl"),
    recommendationPath: path.join(jobDir, "recommendation.json"),
    specPath: path.join(jobDir, "post_training_job_spec.yaml"),
    compiledConfigPath: path.join(jobDir, "compiled_train_config.yaml"),
    manifestPath: path.join(jobDir, "prepared_dataset_manifest.json"),
    compilerTracePath: path.join(jobDir, "compiler_trace.json"),
    trainingResultPath: path.join(jobDir, "training_result.json"),
    trainingMetricsPath: path.join(jobDir, "training_metrics.jsonl"),
    trainingLossGraphPath: path.join(jobDir, "training_loss.svg"),
    learningRateGraphPath: path.join(jobDir, "learning_rate.svg"),
    evaluationPath: path.join(jobDir, "evaluation_result.json"),
    comparisonEvaluationPath: path.join(jobDir, "comparison_evaluation.json"),
    deploymentPath: path.join(jobDir, "deployment.json"),
    smokeTestPath: path.join(jobDir, "smoke_test.json"),
  };
}

async function readJsonFile(filePath) {
  const raw = await readFile(filePath, "utf8");
  return JSON.parse(raw);
}

async function writeJsonFile(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, JSON.stringify(value, null, 2) + "\n", "utf8");
}

async function readOptionalJsonFile(filePath) {
  try {
    return await readJsonFile(filePath);
  } catch (error) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      error.code === "ENOENT"
    ) {
      return null;
    }
    throw error;
  }
}

async function readJob(jobId) {
  return readJsonFile(getJobPaths(jobId).jobFilePath);
}

async function updateJob(jobId, updater) {
  const current = await readJob(jobId);
  const next = await updater(structuredClone(current));
  next.updatedAt = nowIso();
  await writeJsonFile(getJobPaths(jobId).jobFilePath, next);
  return next;
}

async function appendEvent(jobId, event) {
  const record = {
    timestamp: nowIso(),
    jobId,
    ...event,
  };
  console.log(JSON.stringify(record));
  await appendFile(getJobPaths(jobId).eventsFilePath, JSON.stringify(record) + "\n", "utf8");
  return record;
}

async function appendUiProgressEvent(jobId, stage, text, tone = "normal") {
  const event = buildUiProgressEvent({
    stageId: stage,
    text,
    tone,
  });
  if (!event) {
    return null;
  }
  return appendEvent(jobId, event);
}

function createStageLogger(jobId, stage) {
  let currentStage = stage;
  const lastUiProgressByStage = new Map();

  const handleAppendFailure = (targetStage, error) => {
    console.error(
      JSON.stringify({
        timestamp: nowIso(),
        jobId,
        stage: targetStage,
        level: "error",
        source: "orchestrator",
        message: "Failed to append log event.",
        data: safeError(error),
      }),
    );
  };

  return {
    setStage(nextStage) {
      currentStage = nextStage;
    },
    getStage() {
      return currentStage;
    },
    emitUiProgress(text, tone = "normal", stageOverride = null) {
      const targetStage = stageOverride ?? currentStage;
      const event = buildUiProgressEvent({
        stageId: targetStage,
        text,
        tone,
      });
      if (!event) {
        return false;
      }

      const dedupeKey = `${event.level}:${event.message}`;
      if (lastUiProgressByStage.get(targetStage) === dedupeKey) {
        return false;
      }
      lastUiProgressByStage.set(targetStage, dedupeKey);

      void appendEvent(jobId, event).catch((error) => {
        handleAppendFailure(targetStage, error);
      });
      return true;
    },
    emit(payload) {
      const targetStage = payload.stage ?? currentStage;
      void appendEvent(jobId, {
        stage: targetStage,
        level: payload.level ?? "info",
        source: payload.source ?? "orchestrator",
        message: payload.message ?? "",
        data: payload.data === undefined ? undefined : summarizeData(payload.data),
      }).catch((error) => {
        handleAppendFailure(targetStage, error);
      });
    },
  };
}

function emitLoggerUiProgress(logger, text, tone = "normal", stage = null) {
  if (!text) {
    return false;
  }

  if (logger && typeof logger.emitUiProgress === "function") {
    return logger.emitUiProgress(text, tone, stage);
  }

  return emitUiProgress(logger, {
    stageId: stage ?? (typeof logger?.getStage === "function" ? logger.getStage() : null),
    text,
    tone,
  });
}

async function appendTrainingMetricRecord(jobId, record) {
  await appendFile(
    getJobPaths(jobId).trainingMetricsPath,
    JSON.stringify(record) + "\n",
    "utf8",
  );
}

async function writeTrainingMetricGraphs(jobId, metricRecords) {
  const jobPaths = getJobPaths(jobId);
  const bundle = buildTrainingMetricGraphArtifacts(metricRecords);

  for (const artifact of bundle.artifacts) {
    await writeFile(path.join(jobPaths.jobDir, artifact.filename), artifact.content, "utf8");
  }

  return {
    summary: bundle.summary,
    trainingLossGraphPath: bundle.artifacts.some((artifact) => artifact.filename === "training_loss.svg")
      ? jobPaths.trainingLossGraphPath
      : null,
    learningRateGraphPath: bundle.artifacts.some((artifact) => artifact.filename === "learning_rate.svg")
      ? jobPaths.learningRateGraphPath
      : null,
  };
}

async function startStage(jobId, stage, message) {
  await updateJob(jobId, (job) => {
    job.status = stage;
    job.currentStage = stage;
    job.errorSummary = null;
    job.stageHistory.push({
      stage,
      status: "in_progress",
      startedAt: nowIso(),
      completedAt: null,
      errorSummary: null,
    });
    return job;
  });
  await appendEvent(jobId, {
    stage,
    level: "info",
    source: "orchestrator",
    message,
  });
}

async function completeStage(jobId, stage, message, updater = null) {
  await updateJob(jobId, (job) => {
    const entry = [...job.stageHistory].reverse().find((item) => item.stage === stage && item.status === "in_progress");
    if (entry) {
      entry.status = "completed";
      entry.completedAt = nowIso();
      entry.errorSummary = null;
    }
    if (typeof updater === "function") {
      return updater(job);
    }
    return job;
  });
  await appendEvent(jobId, {
    stage,
    level: "info",
    source: "orchestrator",
    message,
  });
}

async function failStage(jobId, stage, error) {
  const details = safeError(error);
  await updateJob(jobId, (job) => {
    const entry = [...job.stageHistory].reverse().find((item) => item.stage === stage && item.status === "in_progress");
    if (entry) {
      entry.status = "failed";
      entry.completedAt = nowIso();
      entry.errorSummary = details.message;
    }
    job.status = "failed";
    job.currentStage = "failed";
    job.errorSummary = details.message;
    return job;
  });
  await appendUiProgressEvent(jobId, stage, details.message, "error");
  await appendEvent(jobId, {
    stage,
    level: "error",
    source: "orchestrator",
    message: details.message,
    data: details,
  });
}

async function markReady(jobId, smokeTestResult) {
  await updateJob(jobId, (job) => {
    job.status = "ready";
    job.currentStage = "ready";
    job.errorSummary = null;
    job.smokeTest = smokeTestResult;
    job.stageHistory.push({
      stage: "ready",
      status: "completed",
      startedAt: nowIso(),
      completedAt: nowIso(),
      errorSummary: null,
    });
    return job;
  });
  await appendEvent(jobId, {
    stage: "ready",
    level: "info",
    source: "orchestrator",
    message: "Job is ready.",
    data: {
      deploymentUrl: smokeTestResult?.deploymentUrl ?? null,
      model: smokeTestResult?.model ?? null,
    },
  });
}

function parseCliArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index];
    const next = argv[index + 1];
    if (current === "--job-id") {
      parsed.jobId = next;
      index += 1;
    }
  }
  return parsed;
}

function resolveModalBin() {
  if (process.env.MODAL_BIN) return process.env.MODAL_BIN;
  const venvModal = path.join(repoRoot, ".venv", "bin", "modal");
  if (existsSync(venvModal)) return venvModal;
  return "modal";
}

function emitProcessLine(logger, source, line) {
  logger.emit({
    source,
    level: source === "stderr" ? "warn" : "info",
    message: line,
  });
}

function normalizeProcessOutputLineForFiltering(line) {
  return String(line ?? "").replace(ANSI_ESCAPE_SEQUENCE_RE, "").replace(/\r/g, "").trim();
}

export function buildCommandFailureMessage({ label, code, stdout = "", stderr = "" }) {
  const lines = [...String(stdout).split(/\r?\n/), ...String(stderr).split(/\r?\n/)]
    .map((line) => normalizeProcessOutputLineForFiltering(line))
    .filter(Boolean);
  const details = lines.length > 0 ? truncateText(lines.slice(-8).join(" | "), 1000) : null;
  return details
    ? `${label} failed with exit code ${code}. ${details}`
    : `${label} failed with exit code ${code}.`;
}

export function shouldSuppressProcessOutputLine(line) {
  const normalizedLine = normalizeProcessOutputLineForFiltering(line).toLowerCase();
  if (!normalizedLine.includes(TOKENIZER_MISMATCH_WARNING_CORE)) {
    return false;
  }

  return TOKENIZER_MISMATCH_WARNING_CONTEXT.some((fragment) => normalizedLine.includes(fragment));
}

export async function handleProcessOutputLine({ source, line, logger, onOutputLine = null }) {
  const handled = typeof onOutputLine === "function" ? await onOutputLine({ source, line, logger }) : false;
  if (!handled && !shouldSuppressProcessOutputLine(line)) {
    emitProcessLine(logger, source, line);
  }
  return handled;
}

async function runCommand({ command, args, env, cwd, logger, label, onOutputLine = null }) {
  logger.emit({
    source: "orchestrator",
    level: "info",
    message: `Running ${label}`,
    data: { command, args },
  });

  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let stdoutBuffer = "";
    let stderrBuffer = "";
    let settled = false;
    let pendingOutputWork = Promise.resolve();

    const settleResolve = (value) => {
      if (settled) {
        return;
      }
      settled = true;
      resolve(value);
    };

    const settleReject = (error) => {
      if (settled) {
        return;
      }
      settled = true;
      reject(error);
    };

    const enqueueLine = (source, rawLine) => {
      const line = String(rawLine ?? "").trim();
      if (!line) {
        return;
      }

      pendingOutputWork = pendingOutputWork.then(async () => {
        await handleProcessOutputLine({ source, line, logger, onOutputLine });
      });

      pendingOutputWork.catch((error) => {
        try {
          child.kill();
        } catch {
          // Best effort cleanup while surfacing the original failure.
        }
        settleReject(error);
      });
    };

    const pushChunk = (source, text) => {
      if (source === "stdout") {
        stdoutBuffer += text;
        const lines = stdoutBuffer.split(/\r?\n/);
        stdoutBuffer = lines.pop() ?? "";
        for (const line of lines) {
          enqueueLine(source, line);
        }
        return;
      }

      stderrBuffer += text;
      const lines = stderrBuffer.split(/\r?\n/);
      stderrBuffer = lines.pop() ?? "";
      for (const line of lines) {
        enqueueLine(source, line);
      }
    };

    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      pushChunk("stdout", text);
    });

    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      pushChunk("stderr", text);
    });

    child.on("error", (error) => {
      settleReject(error);
    });

    child.on("close", (code) => {
      enqueueLine("stdout", stdoutBuffer);
      enqueueLine("stderr", stderrBuffer);
      stdoutBuffer = "";
      stderrBuffer = "";

      pendingOutputWork.then(
        () => {
          if (code === 0) {
            settleResolve({ stdout, stderr, combined: `${stdout}\n${stderr}` });
            return;
          }
          settleReject(new Error(buildCommandFailureMessage({ label, code, stdout, stderr })));
        },
        (error) => {
          settleReject(error);
        },
      );
    });
  });
}

function toRelativeAdapterPath(finalAdapterDir) {
  const normalized = String(finalAdapterDir ?? "").replace(/\\/g, "/");
  const prefix = "/checkpoints/";
  return normalized.startsWith(prefix) ? normalized.slice(prefix.length) : normalized.replace(/^\/+/, "");
}

function buildServeAppName(jobId) {
  const normalized = String(jobId).toLowerCase().replace(/[^a-z0-9-]+/g, "-");
  return `vllm-lora-${normalized}`.slice(0, 60);
}

function collectModalRunUrls(text) {
  return [...String(text).matchAll(/https:\/\/[^\s"'<>]+modal\.run[^\s"'<>]*/g)]
    .map((match) => match[0].replace(/[),.;]+$/, ""))
    .filter((url) => !url.includes("/docs"));
}

export function extractModalRunUrl(text) {
  const rawText = String(text ?? "");
  const directMatches = collectModalRunUrls(rawText);
  if (directMatches.length > 0) {
    directMatches.sort((left, right) => left.length - right.length);
    return directMatches[0];
  }

  const lines = rawText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const reconstructedMatches = [];

  for (let index = 0; index < lines.length; index += 1) {
    if (!lines[index].includes("https://")) {
      continue;
    }

    let candidate = lines[index];
    for (let continuationIndex = index + 1; continuationIndex < Math.min(lines.length, index + 4); continuationIndex += 1) {
      const continuationToken = lines[continuationIndex].split(/\s+/)[0];
      if (!continuationToken || continuationToken.startsWith("https://")) {
        break;
      }

      candidate += continuationToken;
      const candidateMatches = collectModalRunUrls(candidate);
      if (candidateMatches.length > 0) {
        reconstructedMatches.push(...candidateMatches);
        break;
      }
    }
  }

  if (!reconstructedMatches.length) {
    return null;
  }

  reconstructedMatches.sort((left, right) => left.length - right.length);
  return reconstructedMatches[0];
}

async function fetchJson(url, options = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), options.timeoutMs ?? 30_000);

  try {
    const response = await fetch(url, {
      method: options.method ?? "GET",
      headers: {
        Accept: "application/json",
        ...(options.headers ?? {}),
      },
      body: options.body,
      signal: controller.signal,
    });

    const responseText = await response.text();
    let parsed = null;
    try {
      parsed = responseText ? JSON.parse(responseText) : null;
    } catch {
      parsed = null;
    }

    if (!response.ok) {
      throw new Error(`Request to ${url} failed with ${response.status}: ${truncateText(responseText, 500)}`);
    }

    return parsed;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function runSmokeTest({ deploymentUrl, model, logger, taskSpec = null, description = "" }) {
  const modelsUrl = `${deploymentUrl}/v1/models`;
  const chatUrl = `${deploymentUrl}/v1/chat/completions`;
  const attempts = [];
  const isGeneration = isGenerationTaskSpec(taskSpec);
  const probes = isGeneration ? buildGenerationSmokeTestProbes(description) : [];
  const smokeTestConfig = getSmokeTestConfig();

  for (let attempt = 1; attempt <= smokeTestConfig.maxAttempts; attempt += 1) {
    try {
      const modelsTimeoutMs =
        attempt === 1 ? smokeTestConfig.initialModelsTimeoutMs : smokeTestConfig.modelsTimeoutMs;
      emitLoggerUiProgress(
        logger,
        `I'm waiting for the model to start (attempt ${attempt}/${smokeTestConfig.maxAttempts})`,
      );
      logger.emit({
        source: "smoke-test",
        level: "info",
        message: `Smoke test attempt ${attempt}`,
        data: {
          modelsUrl,
          chatUrl,
          model,
          taskFamily: taskSpec?.task_family ?? null,
          probeIds: probes.map((probe) => probe.id),
          modelsTimeoutMs,
          chatTimeoutMs: smokeTestConfig.chatTimeoutMs,
        },
      });

      // The first /v1/models request is the cold-start trigger for Modal's
      // web_server wrapper, so it needs to be allowed to span the whole vLLM boot.
      emitLoggerUiProgress(logger, "I'm checking that the model is available");
      const modelsPayload = await fetchJson(modelsUrl, { timeoutMs: modelsTimeoutMs });
      const modelIds = Array.isArray(modelsPayload?.data)
        ? modelsPayload.data.map((entry) => entry?.id).filter(Boolean)
        : [];
      if (!modelIds.includes(model)) {
        throw new Error(`Expected model '${model}' in /v1/models, got ${JSON.stringify(modelIds)}.`);
      }

      if (isGeneration) {
        const probeResults = [];

        for (const probe of probes) {
          emitLoggerUiProgress(
            logger,
            `I'm trying a sample request (${formatSmokeTestProbeLabel(probe.id)})`,
          );
          const chatPayload = await fetchJson(chatUrl, {
            method: "POST",
            timeoutMs: smokeTestConfig.chatTimeoutMs,
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              model,
              messages: [
                {
                  role: "user",
                  content: probe.prompt,
                },
              ],
              chat_template_kwargs: {
                enable_thinking: false,
              },
              max_tokens: probe.maxTokens,
            }),
          });

          const completionText = extractChatCompletionText(chatPayload).trim();
          const passed = completionText.length >= probe.minLength;
          const probeResult = {
            id: probe.id,
            prompt: probe.prompt,
            passed,
            completionPreview: truncateText(completionText, 200),
            error: passed
              ? null
              : `Completion was shorter than ${probe.minLength} characters.`,
          };
          probeResults.push(probeResult);

          if (!passed) {
            throw new Error(
              `Generation smoke test probe '${probe.id}' returned an insufficient completion (${completionText.length} chars).`,
            );
          }
        }

        return {
          passed: true,
          checkedAt: nowIso(),
          deploymentUrl,
          model,
          attempt,
          modelIds,
          completionPreview: probeResults[0]?.completionPreview ?? null,
          probes: probeResults,
          attempts,
        };
      }

      emitLoggerUiProgress(logger, "I'm trying a quick sample request");
      const chatPayload = await fetchJson(chatUrl, {
        method: "POST",
        timeoutMs: smokeTestConfig.chatTimeoutMs,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          messages: [
            {
              role: "user",
              content: "Reply with the single word READY.",
            },
          ],
          chat_template_kwargs: {
            enable_thinking: false,
          },
          max_tokens: 16,
        }),
      });

      const completionText = extractChatCompletionText(chatPayload);

      return {
        passed: true,
        checkedAt: nowIso(),
        deploymentUrl,
        model,
        attempt,
        modelIds,
        completionPreview: truncateText(completionText, 200),
        attempts,
      };
    } catch (error) {
      const details = safeError(error);
      attempts.push({
        attempt,
        timestamp: nowIso(),
        error: details.message,
      });
      logger.emit({
        source: "smoke-test",
        level: "warn",
        message: `Smoke test attempt ${attempt} failed`,
        data: details,
      });
      if (attempt < smokeTestConfig.maxAttempts) {
        emitLoggerUiProgress(logger, "I'm waiting a moment before trying again");
        await sleep(smokeTestConfig.retryDelayMs);
      }
    }
  }

  throw new Error(
    `Smoke test did not succeed within the retry window after ${smokeTestConfig.maxAttempts} attempts.`,
  );
}

export async function waitForDeploymentReady({ deploymentUrl, model, logger, runtimePolicy = null }) {
  const modelsUrl = `${deploymentUrl}/v1/models`;
  const readinessConfig = getSmokeTestConfig();
  const attempts = [];

  for (let attempt = 1; attempt <= readinessConfig.maxAttempts; attempt += 1) {
    const modelsTimeoutMs =
      attempt === 1 ? readinessConfig.initialModelsTimeoutMs : readinessConfig.modelsTimeoutMs;
    try {
      emitLoggerUiProgress(logger, "I'm waiting for the live model to start");
      logger.emit({
        source: "deploy-readiness",
        level: "info",
        message: `Deployment readiness attempt ${attempt}`,
        data: {
          modelsUrl,
          model,
          modelsTimeoutMs,
          runtimePolicy,
        },
      });

      const modelsPayload = await fetchJson(modelsUrl, { timeoutMs: modelsTimeoutMs });
      const modelIds = Array.isArray(modelsPayload?.data)
        ? modelsPayload.data.map((entry) => entry?.id).filter(Boolean)
        : [];
      if (!modelIds.includes(model)) {
        throw new Error(`Expected model '${model}' in /v1/models, got ${JSON.stringify(modelIds)}.`);
      }

      return {
        readyAt: nowIso(),
        attempt,
        modelIds,
        attempts,
      };
    } catch (error) {
      const details = safeError(error);
      attempts.push({
        attempt,
        timestamp: nowIso(),
        error: details.message,
      });
      logger.emit({
        source: "deploy-readiness",
        level: "warn",
        message: `Deployment readiness attempt ${attempt} failed`,
        data: {
          ...details,
          runtimePolicy,
        },
      });
      if (attempt < readinessConfig.maxAttempts) {
        await sleep(readinessConfig.retryDelayMs);
      }
    }
  }

  throw new Error(
    `Deployment did not become ready within the retry window after ${readinessConfig.maxAttempts} attempts.`,
  );
}

async function runRecommendationStage(jobId, job) {
  const logger = createStageLogger(jobId, "recommending");
  try {
    const recommendation = await recommendDatasets(
      {
        description: job.input.description,
      },
      {
      logger,
      skipDebugWrite: true,
      },
    );
    const overriddenSftDataset = loadCompilerOverrides().sft_dataset;
    const recommendationStageSelectedDatasets = getRecommendationStageSelectedDatasets(
      recommendation,
      overriddenSftDataset,
    );
    await writeJsonFile(getJobPaths(jobId).recommendationPath, recommendation);
    await completeStage(jobId, "recommending", "Recommendation completed.", (draft) => {
      draft.selectedDatasets = recommendationStageSelectedDatasets;
      draft.artifacts.recommendationPath = getJobPaths(jobId).recommendationPath;
      return draft;
    });
    await logger.emit({
      source: "orchestrator",
      level: "info",
      message: "Generated and saved recommendation output from description.",
      data: {
        path: getJobPaths(jobId).recommendationPath,
        queryCount: recommendation.search_queries?.length ?? 0,
        taskSpec: recommendation.task_spec ?? null,
        recommendedDatasets: recommendation.recommended_datasets?.map((item) => item.dataset) ?? [],
      },
    });
  } catch (error) {
    if (error && typeof error === "object" && "recommendation" in error && error.recommendation) {
      await writeJsonFile(getJobPaths(jobId).recommendationPath, error.recommendation);
      await logger.emit({
        source: "orchestrator",
        level: "info",
        message: "Saved recommendation diagnostics.",
        data: {
          path: getJobPaths(jobId).recommendationPath,
        },
      });
    }
    throw error;
  }
}

async function runCompilerStage(jobId, job) {
  const logger = createStageLogger(jobId, "compiling");
  const result = await runCompiler(
    {
      inputPath: getJobPaths(jobId).recommendationPath,
      outputRoot: jobsRoot,
      jobId,
      objectiveSummary: job.input.description,
      seedArtifact: job.input.seedArtifact ?? undefined,
    },
    { logger },
  );

  await completeStage(jobId, "compiling", "Compilation completed.", (draft) => {
    draft.method = result.method;
    draft.selectedDatasets = result.selected_datasets.map((item) => item.dataset);
    draft.artifacts.specPath = result.spec_path;
    draft.artifacts.compiledConfigPath = result.compiled_config_path;
    draft.artifacts.manifestPath = result.manifest_path;
    draft.artifacts.compilerTracePath = result.trace_path;
    return draft;
  });

  await logger.emit({
    source: "orchestrator",
    level: "info",
    message: "Compiled PostTrainingJobSpec and trainer config.",
    data: {
      method: result.method,
      selectedDatasets: result.selected_datasets,
      compiledConfigPath: result.compiled_config_path,
    },
  });

  return result;
}

function summarizeTrainingFiltering(trainingResult) {
  const missingTargetFiltering =
    trainingResult?.preprocessing_diagnostics?.missing_target_label_filtering ?? null;
  const droppedMissingTargetRows = Number(missingTargetFiltering?.dropped_rows_missing_target ?? 0);
  const droppedDatasets = Array.isArray(missingTargetFiltering?.datasets)
    ? missingTargetFiltering.datasets
        .filter((entry) => Number(entry?.dropped_rows_missing_target ?? 0) > 0)
        .map((entry) => ({
          dataset: entry?.dataset ?? null,
          split: entry?.split ?? null,
          selectedTargetColumn: entry?.selected_target_column ?? null,
          droppedRowsMissingTarget: Number(entry?.dropped_rows_missing_target ?? 0),
        }))
    : [];
  const invalidSftFiltering =
    trainingResult?.preprocessing_diagnostics?.invalid_sft_example_filtering ?? null;
  const droppedInvalidSftRows = Number(invalidSftFiltering?.dropped_rows_invalid_examples ?? 0);
  const invalidSftDatasets = Array.isArray(invalidSftFiltering?.datasets)
    ? invalidSftFiltering.datasets
        .filter((entry) => Number(entry?.dropped_rows_invalid_examples ?? 0) > 0)
        .map((entry) => ({
          dataset: entry?.dataset ?? null,
          split: entry?.split ?? null,
          droppedRowsInvalidExamples: Number(entry?.dropped_rows_invalid_examples ?? 0),
          droppedRowsByField:
            entry && typeof entry === "object" && entry.dropped_rows_by_field
              ? entry.dropped_rows_by_field
              : null,
        }))
    : [];

  return {
    droppedMissingTargetRows,
    droppedDatasets,
    droppedInvalidSftRows,
    invalidSftDatasets,
  };
}

async function emitCapturedTrainingLogs({
  logger,
  trainingResult,
  compiledConfig,
  metricGraphArtifacts,
}) {
  const compiledGpuType = compiledConfig?.gpu_type ?? null;
  const modalGpuType = normalizeModalTrainingGpuType(compiledGpuType);
  const {
    droppedMissingTargetRows,
    droppedDatasets,
    droppedInvalidSftRows,
    invalidSftDatasets,
  } = summarizeTrainingFiltering(trainingResult);

  await logger.emit({
    source: "orchestrator",
    level: "info",
    message: "Captured training result.",
    data: {
      finalAdapterDir: trainingResult.final_adapter_dir,
      mergedDir: trainingResult.merged_dir ?? null,
      trainerType: trainingResult.trainer_type,
      selectedDatasets: trainingResult.selected_datasets,
      compiledGpuType,
      modalGpuType: trainingResult.modal_gpu_type ?? modalGpuType,
      globalStep: trainingResult.global_step,
      evalExamples: trainingResult.eval_examples,
      availableTrainingMetrics: metricGraphArtifacts.summary.availableMetrics,
      trainingLossGraphPath: metricGraphArtifacts.trainingLossGraphPath,
      learningRateGraphPath: metricGraphArtifacts.learningRateGraphPath,
      droppedRowsMissingTargetLabels: droppedMissingTargetRows,
      droppedRowsInvalidSftExamples: droppedInvalidSftRows,
    },
  });

  if (droppedMissingTargetRows > 0) {
    await logger.emit({
      source: "orchestrator",
      level: "warn",
      message: "Dropped rows with missing selected target labels before training.",
      data: {
        droppedRowsMissingTargetLabels: droppedMissingTargetRows,
        droppedDatasets,
      },
    });
  }

  if (droppedInvalidSftRows > 0) {
    await logger.emit({
      source: "orchestrator",
      level: "warn",
      message: "Dropped invalid SFT examples with blank required fields before training.",
      data: {
        droppedRowsInvalidSftExamples: droppedInvalidSftRows,
        invalidSftDatasets,
      },
    });
  }
}

async function runTrainingAndEvaluationStage(jobId, compiledConfig) {
  const logger = createStageLogger(jobId, "training");
  const jobPaths = getJobPaths(jobId);
  const compiledGpuType = compiledConfig?.gpu_type ?? null;
  const modalGpuType = normalizeModalTrainingGpuType(compiledGpuType);
  const metricRecords = [];
  let trainingResult = null;
  let trainingStageCompleted = false;
  let evaluationStageStarted = false;
  let metricGraphArtifacts = {
    summary: {
      availableMetrics: [],
      latest: {},
      series: {},
    },
    trainingLossGraphPath: null,
    learningRateGraphPath: null,
  };

  await writeFile(jobPaths.trainingMetricsPath, "", "utf8");
  emitLoggerUiProgress(logger, "I'm starting the training run");
  await logger.emit({
    source: "orchestrator",
    level: "info",
    message: "Launching same-container Modal train_then_evaluate run with resolved GPU types.",
    data: {
      compiledGpuType,
      modalGpuType,
      compiledConfigPath: jobPaths.compiledConfigPath,
    },
  });
  const transitionToEvaluating = async () => {
    if (!trainingResult) {
      throw new Error("Combined training run emitted evaluation transition without a training_result payload.");
    }
    if (!trainingStageCompleted) {
      emitLoggerUiProgress(logger, "I'm saving the trained model");
      await completeStage(jobId, "training", "Training completed.", (draft) => {
        draft.method = trainingResult.trainer_type ?? draft.method;
        draft.selectedDatasets = Array.isArray(trainingResult.selected_datasets)
          ? trainingResult.selected_datasets
          : draft.selectedDatasets;
        draft.artifacts.trainingResultPath = jobPaths.trainingResultPath;
        draft.artifacts.trainingMetricsPath = jobPaths.trainingMetricsPath;
        draft.artifacts.trainingLossGraphPath = metricGraphArtifacts.trainingLossGraphPath;
        draft.artifacts.learningRateGraphPath = metricGraphArtifacts.learningRateGraphPath;
        return draft;
      });
      await emitCapturedTrainingLogs({
        logger,
        trainingResult,
        compiledConfig,
        metricGraphArtifacts,
      });
      trainingStageCompleted = true;
    }
    if (!evaluationStageStarted) {
      logger.setStage("evaluating");
      await startStage(jobId, "evaluating", "Running offline evaluation.");
      emitLoggerUiProgress(logger, "I'm testing how the model performs");
      evaluationStageStarted = true;
    }
  };

  try {
    await runCommand({
      command: resolveModalBin(),
      args: ["run", trainerScriptPath, "--config", jobPaths.compiledConfigPath],
      env: {
        POSTTRAINING_RUN_MODE: "train_then_evaluate",
        TRAIN_RESULT_PATH: jobPaths.trainingResultPath,
        EVALUATION_RESULT_PATH: jobPaths.evaluationPath,
        COMPARISON_EVALUATION_PATH: jobPaths.comparisonEvaluationPath,
      },
      cwd: repoRoot,
      logger,
      label: "Modal train_then_evaluate run",
      onOutputLine: async ({ line }) => {
        const record = parseStructuredTrainingMetricLine(line);
        if (record) {
          metricRecords.push(record);
          await appendTrainingMetricRecord(jobId, record);
          metricGraphArtifacts = await writeTrainingMetricGraphs(jobId, metricRecords);
          const trainingProgress = buildTrainingMetricUiProgress(record);
          if (trainingProgress && logger.getStage() === "training") {
            emitLoggerUiProgress(logger, trainingProgress);
          }
          return true;
        }

        const stageProgress = detectTrainingStageUiProgress(line, logger.getStage());
        if (stageProgress) {
          emitLoggerUiProgress(logger, stageProgress);
          if (line.startsWith(STRUCTURED_PROGRESS_PREFIX)) {
            return true;
          }
        }

        const lifecycleEvent = parseStructuredLifecycleEventLine(line);
        if (!lifecycleEvent) {
          return false;
        }
        if (lifecycleEvent.event !== "training_complete") {
          return true;
        }

        trainingResult = lifecycleEvent.training_result ?? null;
        if (!trainingResult) {
          throw new Error("Structured lifecycle event omitted training_result.");
        }
        emitLoggerUiProgress(logger, "I'm saving the trained model");
        await writeJsonFile(jobPaths.trainingResultPath, trainingResult);
        await transitionToEvaluating();
        return true;
      },
    });
  } catch (error) {
    if (trainingResult) {
      await transitionToEvaluating();
    }
    throw error;
  }

  if (!trainingResult) {
    throw new Error("Combined training run completed without emitting a training_result lifecycle event.");
  }
  await transitionToEvaluating();

  return {
    trainingResult,
    compiledConfig,
  };
}

function isNonTrivialRun(trainingResult) {
  return Number(trainingResult?.train_examples ?? 0) >= 1000;
}

async function finalizeEvaluationStage(jobId, trainingResult) {
  const logger = createStageLogger(jobId, "evaluating");
  const jobPaths = getJobPaths(jobId);
  const evaluation = await readOptionalJsonFile(jobPaths.evaluationPath);
  const comparisonEvaluation = await readOptionalJsonFile(jobPaths.comparisonEvaluationPath);
  emitLoggerUiProgress(logger, "I'm checking the test results");
  if (!evaluation && !comparisonEvaluation) {
    throw new Error("Evaluation completed without writing any evaluation artifacts.");
  }

  if (isNonTrivialRun(trainingResult) && Number(trainingResult.global_step ?? 0) <= 1) {
    throw new Error(
      `Training run is degenerate: global_step=${trainingResult.global_step} for ${trainingResult.train_examples} train examples.`,
    );
  }

  const invalidLabelRate = Number(evaluation?.metrics?.invalid_label_rate ?? 0);
  if (evaluation && (!Number.isFinite(invalidLabelRate) || invalidLabelRate >= 1)) {
    throw new Error(
      `Evaluation indicates catastrophic label prediction failure (invalid_label_rate=${invalidLabelRate}).`,
    );
  }

  if (comparisonEvaluation) {
    emitLoggerUiProgress(logger, "I'm scoring each model against the expected outputs");
  }

  await completeStage(jobId, "evaluating", "Offline evaluation completed.", (draft) => {
    draft.artifacts.evaluationPath = evaluation ? jobPaths.evaluationPath : null;
    draft.artifacts.comparisonEvaluationPath = comparisonEvaluation ? jobPaths.comparisonEvaluationPath : null;
    draft.evaluation = evaluation ?? comparisonEvaluation;
    return draft;
  });

  await logger.emit({
    source: "orchestrator",
    level: "info",
    message: "Evaluation artifact captured.",
    data: {
      accuracy: evaluation?.metrics?.accuracy ?? null,
      macroF1: evaluation?.metrics?.macro_f1 ?? null,
      invalidLabelRate: evaluation ? invalidLabelRate : null,
      sampledExamples: evaluation?.sampled_examples ?? null,
      comparisonTaskFamily: comparisonEvaluation?.task_family ?? null,
      comparisonSampledCases: comparisonEvaluation?.sample_policy?.sampled_cases ?? null,
      comparisonEvaluationPath: comparisonEvaluation ? jobPaths.comparisonEvaluationPath : null,
    },
  });

  return evaluation ?? comparisonEvaluation;
}

async function runDeploymentStage(jobId, trainingResult, compiledConfig) {
  const logger = createStageLogger(jobId, "deploying");
  const deploymentArtifact = resolveDeploymentArtifact(trainingResult);
  const runtimePolicy = resolveDeploymentRuntimePolicy(trainingResult, deploymentArtifact);
  const deploymentAppName = buildServeAppName(jobId);
  const serveStartupTimeoutSeconds = getServeStartupTimeoutSeconds();
  const serveGpuType =
    process.env.POSTTRAINING_SERVE_GPU ||
    normalizeModalTrainingGpuType(compiledConfig.gpu_type) ||
    "A10G";
  const adapterName = jobId;
  const deploymentEnv = buildDeploymentEnvironment({
    deploymentAppName,
    deploymentArtifact,
    adapterName,
    trainingResult,
    compiledConfig,
    serveGpuType,
    serveStartupTimeoutSeconds,
    runtimePolicy,
  });

  await logger.emit({
    source: "orchestrator",
    level: "info",
    message: "Resolved deployment runtime policy.",
    data: {
      baseModel: trainingResult.base_model,
      adapterPath: deploymentArtifact.relativePath,
      ...runtimePolicy,
    },
  });

  emitLoggerUiProgress(logger, "I'm getting the model ready to go live");
  const commandResult = await runCommand({
    command: resolveModalBin(),
    args: ["deploy", serveScriptPath],
    env: deploymentEnv,
    cwd: repoRoot,
    logger,
    label: "Modal deployment",
    onOutputLine: async ({ line }) => {
      const stageProgress = detectDeploymentStageUiProgress(line);
      if (stageProgress) {
        emitLoggerUiProgress(logger, stageProgress);
      }
      return false;
    },
  });

  emitLoggerUiProgress(logger, "I'm creating the live model link");
  const deploymentUrl = extractModalRunUrl(commandResult.combined);
  if (!deploymentUrl) {
    throw new Error("Could not determine the deployed modal.run URL from Modal deploy output.");
  }

  const deployment = buildDeploymentRecord({
    deploymentUrl,
    deploymentAppName,
    adapterName,
    deploymentArtifact,
    trainingResult,
    serveGpuType,
    serveStartupTimeoutSeconds,
    runtimePolicy,
  });

  await writeJsonFile(getJobPaths(jobId).deploymentPath, deployment);
  const readiness = await waitForDeploymentReady({
    deploymentUrl,
    model: adapterName,
    logger,
    runtimePolicy,
  });
  await completeStage(jobId, "deploying", "Deployment completed.", (draft) => {
    draft.deployment = deployment;
    draft.artifacts.deploymentPath = getJobPaths(jobId).deploymentPath;
    return draft;
  });

  await logger.emit({
    source: "orchestrator",
    level: "info",
    message: "Deployment URL captured.",
    data: {
      ...deployment,
      ...runtimePolicy,
      readinessAttempt: readiness.attempt,
      readyAt: readiness.readyAt,
      modelIds: readiness.modelIds,
    },
  });

  return deployment;
}

async function runSmokeTestStage(jobId, deployment, { taskSpec = null, description = "" } = {}) {
  const logger = createStageLogger(jobId, "smoke_testing");
  const smokeTestResult = await runSmokeTest({
    jobId,
    deploymentUrl: deployment.url,
    model: deployment.adapterName,
    logger,
    taskSpec,
    description,
  });
  await writeJsonFile(getJobPaths(jobId).smokeTestPath, smokeTestResult);
  await completeStage(jobId, "smoke_testing", "Smoke test passed.", (draft) => {
    draft.artifacts.smokeTestPath = getJobPaths(jobId).smokeTestPath;
    draft.smokeTest = smokeTestResult;
    return draft;
  });
  return smokeTestResult;
}

async function runOrchestrator(jobId) {
  await mkdir(jobsRoot, { recursive: true });
  const job = await readJob(jobId);
  if (!job) {
    throw new Error(`Job not found: ${jobId}`);
  }

  await startStage(jobId, "recommending", "Starting dataset recommendation.");
  let compiledResult = null;
  let trainingBundle = null;
  let deployment = null;

  try {
    await runRecommendationStage(jobId, job);

    const jobAfterRecommendation = await readJob(jobId);
    await startStage(jobId, "compiling", "Starting spec compilation.");
    compiledResult = await runCompilerStage(jobId, jobAfterRecommendation);

    await startStage(jobId, "training", "Starting Modal training.");
    trainingBundle = await runTrainingAndEvaluationStage(jobId, compiledResult.compiled_config);
    await finalizeEvaluationStage(jobId, trainingBundle.trainingResult);

    await startStage(jobId, "deploying", "Starting stable vLLM deployment.");
    deployment = await runDeploymentStage(
      jobId,
      trainingBundle.trainingResult,
      trainingBundle.compiledConfig,
    );

    await startStage(jobId, "smoke_testing", "Starting deployment smoke test.");
    const smokeTestResult = await runSmokeTestStage(jobId, deployment, {
      taskSpec: trainingBundle.trainingResult?.task_spec ?? compiledResult.task_spec ?? null,
      description: job.input.description,
    });
    await markReady(jobId, {
      ...smokeTestResult,
      deploymentUrl: deployment.url,
    });
  } catch (error) {
    const currentJob = await readJob(jobId).catch(() => null);
    const stage =
      currentJob?.currentStage && currentJob.currentStage !== "failed"
        ? currentJob.currentStage
        : deployment
          ? "smoke_testing"
          : trainingBundle
            ? "deploying"
            : compiledResult
              ? "training"
              : "recommending";
    await failStage(jobId, stage, error);
    process.exitCode = 1;
  }
}

async function runCli() {
  const args = parseCliArgs(process.argv.slice(2));
  if (!args.jobId || typeof args.jobId !== "string") {
    throw new Error("Usage: node backend/posttraining-orchestrator.mjs --job-id <jobId>");
  }
  await runOrchestrator(args.jobId);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await runCli();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
