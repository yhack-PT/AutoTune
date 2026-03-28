import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";

export type JobStage =
  | "queued"
  | "recommending"
  | "compiling"
  | "training"
  | "deploying"
  | "smoke_testing"
  | "ready"
  | "failed";

export type JobHistoryEntry = {
  stage: JobStage;
  status: "completed" | "failed" | "in_progress";
  startedAt: string;
  completedAt?: string | null;
  errorSummary?: string | null;
};

export type PostTrainingJobRecord = {
  jobId: string;
  status: JobStage;
  currentStage: JobStage;
  createdAt: string;
  updatedAt: string;
  input: {
    domain: string;
    task: string;
    qualityTier: number;
    seedArtifact: string | null;
  };
  method: string | null;
  selectedDatasets: string[];
  errorSummary: string | null;
  artifacts: Record<string, string | null>;
  deployment: Record<string, unknown> | null;
  smokeTest: Record<string, unknown> | null;
  stageHistory: JobHistoryEntry[];
};

type CreateJobInput = {
  domain: string;
  task: string;
  qualityTier: number;
  seedArtifact?: string | null;
};

type JobEvent = {
  timestamp: string;
  jobId: string;
  stage: string;
  level: string;
  message: string;
  source?: string;
  data?: unknown;
};

const JOBS_ROOT = path.join(process.cwd(), "backend", "generated-posttraining-jobs");

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
}

function nowIso(): string {
  return new Date().toISOString();
}

function buildJobId(domain: string, task: string): string {
  const base = slugify(`${domain}-${task}`) || "posttraining-job";
  return `${base}-${randomUUID().slice(0, 8)}`;
}

export function getJobsRoot(): string {
  return JOBS_ROOT;
}

export function getJobDir(jobId: string): string {
  return path.join(JOBS_ROOT, jobId);
}

export function getJobFilePath(jobId: string): string {
  return path.join(getJobDir(jobId), "job.json");
}

export function getEventsFilePath(jobId: string): string {
  return path.join(getJobDir(jobId), "events.jsonl");
}

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    const raw = await readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch (error) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: string }).code === "ENOENT"
    ) {
      return null;
    }
    throw error;
  }
}

async function readEvents(jobId: string, limit = 200): Promise<JobEvent[]> {
  try {
    const raw = await readFile(getEventsFilePath(jobId), "utf8");
    const lines = raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    return lines.slice(-limit).map((line) => JSON.parse(line) as JobEvent);
  } catch (error) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: string }).code === "ENOENT"
    ) {
      return [];
    }
    throw error;
  }
}

function assertString(value: unknown, fieldName: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${fieldName} must be a non-empty string.`);
  }
  return value.trim();
}

function assertQualityTier(value: unknown): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 5) {
    throw new Error("qualityTier must be an integer between 1 and 5.");
  }
  return parsed;
}

function normalizeSeedArtifact(value: unknown): string | null {
  if (value == null) {
    return null;
  }
  if (typeof value !== "string") {
    throw new Error("seedArtifact must be a string when provided.");
  }
  const normalized = value.trim();
  return normalized || null;
}

export function validateCreateJobInput(body: unknown): CreateJobInput {
  if (!body || typeof body !== "object") {
    throw new Error("Request body must be an object.");
  }

  const payload = body as Record<string, unknown>;
  return {
    domain: assertString(payload.domain, "domain"),
    task: assertString(payload.task, "task"),
    qualityTier: assertQualityTier(payload.qualityTier),
    seedArtifact: normalizeSeedArtifact(payload.seedArtifact),
  };
}

export async function createPostTrainingJob(input: CreateJobInput): Promise<PostTrainingJobRecord> {
  await mkdir(JOBS_ROOT, { recursive: true });

  const createdAt = nowIso();
  const jobId = buildJobId(input.domain, input.task);
  const jobDir = getJobDir(jobId);
  await mkdir(jobDir, { recursive: true });

  const job: PostTrainingJobRecord = {
    jobId,
    status: "queued",
    currentStage: "queued",
    createdAt,
    updatedAt: createdAt,
    input: {
      domain: input.domain,
      task: input.task,
      qualityTier: input.qualityTier,
      seedArtifact: input.seedArtifact ?? null,
    },
    method: null,
    selectedDatasets: [],
    errorSummary: null,
    artifacts: {
      jobDir,
      recommendationPath: path.join(jobDir, "recommendation.json"),
      specPath: path.join(jobDir, "post_training_job_spec.yaml"),
      compiledConfigPath: path.join(jobDir, "compiled_train_config.yaml"),
      manifestPath: path.join(jobDir, "prepared_dataset_manifest.json"),
      compilerTracePath: path.join(jobDir, "compiler_trace.json"),
      trainingResultPath: path.join(jobDir, "training_result.json"),
      deploymentPath: path.join(jobDir, "deployment.json"),
      smokeTestPath: path.join(jobDir, "smoke_test.json"),
    },
    deployment: null,
    smokeTest: null,
    stageHistory: [
      {
        stage: "queued",
        status: "completed",
        startedAt: createdAt,
        completedAt: createdAt,
        errorSummary: null,
      },
    ],
  };

  await writeFile(getJobFilePath(jobId), JSON.stringify(job, null, 2) + "\n", "utf8");
  await writeFile(
    getEventsFilePath(jobId),
    JSON.stringify({
      timestamp: createdAt,
      jobId,
      stage: "queued",
      level: "info",
      source: "api",
      message: "Job created and queued.",
      data: {
        domain: input.domain,
        qualityTier: input.qualityTier,
      },
    }) + "\n",
    "utf8",
  );
  return job;
}

export function spawnPostTrainingOrchestrator(jobId: string): void {
  const scriptPath = path.join(process.cwd(), "backend", "posttraining-orchestrator.mjs");
  const child = spawn(process.execPath, [scriptPath, "--job-id", jobId], {
    cwd: process.cwd(),
    detached: true,
    stdio: "inherit",
    env: process.env,
  });
  child.unref();
}

export async function listPostTrainingJobs(): Promise<
  Array<{
    jobId: string;
    status: JobStage;
    createdAt: string;
    updatedAt: string;
    method: string | null;
    deploymentUrl: string | null;
    errorSummary: string | null;
  }>
> {
  await mkdir(JOBS_ROOT, { recursive: true });
  const entries = await readdir(JOBS_ROOT, { withFileTypes: true });

  const jobs = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map(async (entry) => {
        const job = await readJsonFile<PostTrainingJobRecord>(path.join(JOBS_ROOT, entry.name, "job.json"));
        return job;
      }),
  );

  return jobs
    .filter((job): job is PostTrainingJobRecord => Boolean(job))
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
    .map((job) => ({
      jobId: job.jobId,
      status: job.status,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      method: job.method,
      deploymentUrl:
        job.deployment && typeof job.deployment.url === "string" ? job.deployment.url : null,
      errorSummary: job.errorSummary,
    }));
}

export async function getPostTrainingJob(jobId: string): Promise<
  | (PostTrainingJobRecord & {
      logs: JobEvent[];
    })
  | null
> {
  const job = await readJsonFile<PostTrainingJobRecord>(getJobFilePath(jobId));
  if (!job) {
    return null;
  }

  return {
    ...job,
    logs: await readEvents(jobId),
  };
}
