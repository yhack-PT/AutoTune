import process from "node:process";
import path from "node:path";
import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

import { recommendDatasets } from "./hf-dataset-recommender.mjs";
import { runCompiler } from "./posttraining-spec-compiler.mjs";

const backendDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(backendDir, "..");
const jobsRoot = path.join(backendDir, "generated-posttraining-jobs");
const trainerScriptPath = path.join(backendDir, "modal_trl_posttrain.py");
const serveScriptPath = path.join(backendDir, "modal_vllm_serve.py");

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

function createStageLogger(jobId, stage) {
  return {
    emit(payload) {
      void appendEvent(jobId, {
        stage,
        level: payload.level ?? "info",
        source: payload.source ?? "orchestrator",
        message: payload.message ?? "",
        data: payload.data === undefined ? undefined : summarizeData(payload.data),
      }).catch((error) => {
        console.error(
          JSON.stringify({
            timestamp: nowIso(),
            jobId,
            stage,
            level: "error",
            source: "orchestrator",
            message: "Failed to append log event.",
            data: safeError(error),
          }),
        );
      });
    },
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
  return process.env.MODAL_BIN || "modal";
}

function emitProcessOutput(logger, source, chunk) {
  const lines = String(chunk)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (const line of lines) {
    logger.emit({
      source,
      level: source === "stderr" ? "warn" : "info",
      message: line,
    });
  }
}

async function runCommand({ command, args, env, cwd, logger, label }) {
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

    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      emitProcessOutput(logger, "stdout", text);
    });

    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      emitProcessOutput(logger, "stderr", text);
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr, combined: `${stdout}\n${stderr}` });
        return;
      }
      reject(new Error(`${label} failed with exit code ${code}.`));
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

async function runSmokeTest({ jobId, deploymentUrl, model, logger }) {
  const modelsUrl = `${deploymentUrl}/v1/models`;
  const chatUrl = `${deploymentUrl}/v1/chat/completions`;
  const attempts = [];

  for (let attempt = 1; attempt <= 30; attempt += 1) {
    try {
      logger.emit({
        source: "smoke-test",
        level: "info",
        message: `Smoke test attempt ${attempt}`,
        data: { modelsUrl, chatUrl, model },
      });

      const modelsPayload = await fetchJson(modelsUrl, { timeoutMs: 60_000 });
      const modelIds = Array.isArray(modelsPayload?.data)
        ? modelsPayload.data.map((entry) => entry?.id).filter(Boolean)
        : [];
      if (!modelIds.includes(model)) {
        throw new Error(`Expected model '${model}' in /v1/models, got ${JSON.stringify(modelIds)}.`);
      }

      const chatPayload = await fetchJson(chatUrl, {
        method: "POST",
        timeoutMs: 120_000,
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
          max_tokens: 16,
        }),
      });

      const completionText =
        chatPayload?.choices?.[0]?.message?.content ??
        chatPayload?.choices?.[0]?.text ??
        null;

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
      await sleep(10_000);
    }
  }

  throw new Error("Smoke test did not succeed within the retry window.");
}

async function runRecommendationStage(jobId, job) {
  const logger = createStageLogger(jobId, "recommending");
  try {
    const recommendation = await recommendDatasets(
      {
        domain: job.input.domain,
        qualityTier: job.input.qualityTier,
        useCase: job.input.task,
      },
      {
        logger,
        skipDebugWrite: true,
      },
    );
    await writeJsonFile(getJobPaths(jobId).recommendationPath, recommendation);
    await completeStage(jobId, "recommending", "Recommendation completed.", (draft) => {
      draft.artifacts.recommendationPath = getJobPaths(jobId).recommendationPath;
      return draft;
    });
    await logger.emit({
      source: "orchestrator",
      level: "info",
      message: "Saved recommendation output.",
      data: {
        path: getJobPaths(jobId).recommendationPath,
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
      objectiveSummary: job.input.task,
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

async function runTrainingStage(jobId, compiledConfig) {
  const logger = createStageLogger(jobId, "training");
  await runCommand({
    command: resolveModalBin(),
    args: ["run", trainerScriptPath, "--config", getJobPaths(jobId).compiledConfigPath],
    env: {
      TRAIN_RESULT_PATH: getJobPaths(jobId).trainingResultPath,
    },
    cwd: repoRoot,
    logger,
    label: "Modal training run",
  });

  const trainingResult = await readJsonFile(getJobPaths(jobId).trainingResultPath);
  if (!trainingResult) {
    throw new Error("Training completed but training_result.json was not written.");
  }

  await completeStage(jobId, "training", "Training completed.", (draft) => {
    draft.method = trainingResult.trainer_type ?? draft.method;
    draft.selectedDatasets = Array.isArray(trainingResult.selected_datasets)
      ? trainingResult.selected_datasets
      : draft.selectedDatasets;
    draft.artifacts.trainingResultPath = getJobPaths(jobId).trainingResultPath;
    return draft;
  });

  await logger.emit({
    source: "orchestrator",
    level: "info",
    message: "Captured training result.",
    data: {
      finalAdapterDir: trainingResult.final_adapter_dir,
      trainerType: trainingResult.trainer_type,
      selectedDatasets: trainingResult.selected_datasets,
    },
  });

  return {
    trainingResult,
    compiledConfig,
  };
}

async function runDeploymentStage(jobId, trainingResult, compiledConfig) {
  const logger = createStageLogger(jobId, "deploying");
  const adapterPath = toRelativeAdapterPath(trainingResult.final_adapter_dir);
  const deploymentAppName = buildServeAppName(jobId);
  const serveGpuType =
    process.env.POSTTRAINING_SERVE_GPU ||
    (compiledConfig.gpu_type === "A10" ? "A10G" : compiledConfig.gpu_type || "A10G");
  const adapterName = jobId;

  const commandResult = await runCommand({
    command: resolveModalBin(),
    args: ["deploy", serveScriptPath],
    env: {
      APP_NAME: deploymentAppName,
      ADAPTER_PATH: adapterPath,
      ADAPTER_NAME: adapterName,
      BASE_MODEL: trainingResult.base_model,
      MAX_MODEL_LEN: String(compiledConfig.max_length ?? 2048),
      GPU_TYPE: serveGpuType,
    },
    cwd: repoRoot,
    logger,
    label: "Modal deployment",
  });

  const deploymentUrl = extractModalRunUrl(commandResult.combined);
  if (!deploymentUrl) {
    throw new Error("Could not determine the deployed modal.run URL from Modal deploy output.");
  }

  const deployment = {
    deployedAt: nowIso(),
    appName: deploymentAppName,
    url: deploymentUrl,
    adapterName,
    adapterPath,
    baseModel: trainingResult.base_model,
    gpuType: serveGpuType,
  };

  await writeJsonFile(getJobPaths(jobId).deploymentPath, deployment);
  await completeStage(jobId, "deploying", "Deployment completed.", (draft) => {
    draft.deployment = deployment;
    draft.artifacts.deploymentPath = getJobPaths(jobId).deploymentPath;
    return draft;
  });

  await logger.emit({
    source: "orchestrator",
    level: "info",
    message: "Deployment URL captured.",
    data: deployment,
  });

  return deployment;
}

async function runSmokeTestStage(jobId, deployment) {
  const logger = createStageLogger(jobId, "smoke_testing");
  const smokeTestResult = await runSmokeTest({
    jobId,
    deploymentUrl: deployment.url,
    model: deployment.adapterName,
    logger,
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
    trainingBundle = await runTrainingStage(jobId, compiledResult.compiled_config);

    await startStage(jobId, "deploying", "Starting stable vLLM deployment.");
    deployment = await runDeploymentStage(
      jobId,
      trainingBundle.trainingResult,
      trainingBundle.compiledConfig,
    );

    await startStage(jobId, "smoke_testing", "Starting deployment smoke test.");
    const smokeTestResult = await runSmokeTestStage(jobId, deployment);
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
