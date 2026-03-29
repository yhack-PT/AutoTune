import { existsSync } from "node:fs";
import path from "node:path";

import type { PostTrainingJobRecord } from "@/lib/posttraining-server";

const CHECKPOINTS_PREFIX = "/checkpoints/";

export const POSTTRAINING_CHECKPOINTS_VOLUME_NAME = "trl-checkpoints";

type ResolveModalBinOptions = {
  env?: NodeJS.ProcessEnv;
  repoRoot?: string;
  existsSyncFn?: (filePath: string) => boolean;
};

type AdapterTrainingResult = {
  final_adapter_dir?: unknown;
};

export type AdapterWeightsDownloadSpec = {
  archiveFileName: string;
  modalRemotePath: string;
  trainingResultPath: string;
};

export function buildAdapterWeightsArchiveFileName(jobId: string): string {
  const normalizedJobId = String(jobId ?? "").trim();
  if (!normalizedJobId) {
    throw new Error("jobId is required to build the adapter weights archive name.");
  }

  return `${normalizedJobId}-adapter-weights.tar.gz`;
}

export function normalizeCheckpointsVolumePath(finalAdapterDir: unknown): string {
  if (typeof finalAdapterDir !== "string" || !finalAdapterDir.trim()) {
    throw new Error("Training result did not include final_adapter_dir.");
  }

  const normalizedPath = finalAdapterDir.trim().replace(/\\/g, "/");
  if (!normalizedPath.startsWith(CHECKPOINTS_PREFIX)) {
    throw new Error("final_adapter_dir must point inside /checkpoints.");
  }

  const relativePath = normalizedPath.slice(CHECKPOINTS_PREFIX.length).replace(/^\/+/, "");
  if (!relativePath) {
    throw new Error("final_adapter_dir must point to a directory inside /checkpoints.");
  }

  return relativePath;
}

export function resolveModalBin({
  env = process.env,
  repoRoot = process.cwd(),
  existsSyncFn = existsSync,
}: ResolveModalBinOptions = {}): string {
  if (env.MODAL_BIN) {
    return env.MODAL_BIN;
  }

  const venvModal = path.join(repoRoot, ".venv", "bin", "modal");
  if (existsSyncFn(venvModal)) {
    return venvModal;
  }

  return "modal";
}

export function resolveAdapterWeightsDownloadSpec({
  job,
  trainingResult,
}: {
  job: Pick<PostTrainingJobRecord, "jobId" | "artifacts">;
  trainingResult: AdapterTrainingResult;
}): AdapterWeightsDownloadSpec {
  const trainingResultPath =
    typeof job.artifacts.trainingResultPath === "string" && job.artifacts.trainingResultPath.trim()
      ? job.artifacts.trainingResultPath
      : null;

  if (!trainingResultPath) {
    throw new Error("Job record is missing trainingResultPath.");
  }

  return {
    archiveFileName: buildAdapterWeightsArchiveFileName(job.jobId),
    modalRemotePath: normalizeCheckpointsVolumePath(trainingResult.final_adapter_dir),
    trainingResultPath,
  };
}
