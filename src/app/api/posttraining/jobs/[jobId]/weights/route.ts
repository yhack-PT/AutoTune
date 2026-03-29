import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { NextResponse } from "next/server";

import {
  POSTTRAINING_CHECKPOINTS_VOLUME_NAME,
  resolveAdapterWeightsDownloadSpec,
  resolveModalBin,
} from "@/lib/posttraining-download";
import { getPostTrainingJob } from "@/lib/posttraining-server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{
    jobId: string;
  }>;
};

type TrainingResult = {
  final_adapter_dir?: unknown;
};

type BufferedCommandOptions = {
  command: string;
  args: string[];
  cwd?: string;
  signal: AbortSignal;
  label: string;
};

function jsonError(message: string, status: number) {
  return NextResponse.json({ error: message }, { status });
}

function buildCommandFailureMessage({
  label,
  code,
  stdout,
  stderr,
}: {
  label: string;
  code: number | null;
  stdout: string;
  stderr: string;
}): string {
  const details = [stdout, stderr]
    .flatMap((text) => String(text).split(/\r?\n/))
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-8)
    .join(" | ");

  return details
    ? `${label} failed with exit code ${code ?? "unknown"}. ${details}`
    : `${label} failed with exit code ${code ?? "unknown"}.`;
}

function createAbortError(): Error {
  return Object.assign(new Error("The request was aborted."), { name: "AbortError" });
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

async function readTrainingResult(trainingResultPath: string): Promise<TrainingResult> {
  try {
    const raw = await readFile(trainingResultPath, "utf8");
    return JSON.parse(raw) as TrainingResult;
  } catch (error) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: string }).code === "ENOENT"
    ) {
      throw new Error("Training result artifact not found.");
    }

    if (error instanceof SyntaxError) {
      throw new Error("Training result artifact is not valid JSON.");
    }

    throw error;
  }
}

async function runBufferedCommand({
  command,
  args,
  cwd = process.cwd(),
  signal,
  label,
}: BufferedCommandOptions): Promise<void> {
  if (signal.aborted) {
    throw createAbortError();
  }

  await new Promise<void>((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const settleResolve = () => {
      if (settled) {
        return;
      }
      settled = true;
      signal.removeEventListener("abort", handleAbort);
      resolve();
    };

    const settleReject = (error: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      signal.removeEventListener("abort", handleAbort);
      reject(error);
    };

    const handleAbort = () => {
      try {
        child.kill("SIGTERM");
      } catch {
        // Best-effort cancellation.
      }
      settleReject(createAbortError());
    };

    signal.addEventListener("abort", handleAbort, { once: true });

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      settleReject(error);
    });

    child.on("close", (code) => {
      if (signal.aborted) {
        settleReject(createAbortError());
        return;
      }

      if (code === 0) {
        settleResolve();
        return;
      }

      settleReject(
        new Error(
          buildCommandFailureMessage({
            label,
            code,
            stdout,
            stderr,
          }),
        ),
      );
    });
  });
}

function createTarArchiveStream({
  archiveRoot,
  archiveDirName,
  signal,
  onFinished,
}: {
  archiveRoot: string;
  archiveDirName: string;
  signal: AbortSignal;
  onFinished: () => Promise<void>;
}): ReadableStream<Uint8Array> {
  const tar = spawn("tar", ["-czf", "-", "-C", archiveRoot, archiveDirName], {
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stderr = "";
  let cleanedUp = false;

  const cleanupOnce = async () => {
    if (cleanedUp) {
      return;
    }

    cleanedUp = true;
    signal.removeEventListener("abort", handleAbort);
    await onFinished();
  };

  const handleAbort = () => {
    try {
      tar.kill("SIGTERM");
    } catch {
      // Best-effort cancellation.
    }
    void cleanupOnce();
  };

  signal.addEventListener("abort", handleAbort, { once: true });

  return new ReadableStream<Uint8Array>({
    start(controller) {
      let settled = false;

      const closeController = () => {
        if (settled) {
          return;
        }
        settled = true;
        controller.close();
      };

      const errorController = (error: Error) => {
        if (settled) {
          return;
        }
        settled = true;
        controller.error(error);
      };

      tar.stdout.on("data", (chunk: Buffer) => {
        controller.enqueue(new Uint8Array(chunk));
      });

      tar.stdout.on("end", () => {
        closeController();
      });

      tar.stdout.on("error", (error) => {
        errorController(error);
        void cleanupOnce();
      });

      tar.stderr.on("data", (chunk: Buffer) => {
        stderr += chunk.toString();
      });

      tar.on("error", (error) => {
        errorController(error);
        void cleanupOnce();
      });

      tar.on("close", (code) => {
        if (!signal.aborted && code !== 0) {
          errorController(
            new Error(
              buildCommandFailureMessage({
                label: "Creating adapter weights archive",
                code,
                stdout: "",
                stderr,
              }),
            ),
          );
        } else {
          closeController();
        }

        void cleanupOnce();
      });
    },
    cancel() {
      try {
        tar.kill("SIGTERM");
      } catch {
        // Best-effort cancellation.
      }

      void cleanupOnce();
    },
  });
}

export async function GET(request: Request, context: RouteContext) {
  let tempRoot: string | null = null;

  try {
    const { jobId } = await context.params;
    const job = await getPostTrainingJob(jobId);

    if (!job) {
      return jsonError("Job not found.", 404);
    }

    if (job.status !== "ready") {
      return jsonError("Model weights are only available after the job is ready.", 409);
    }

    const trainingResultPath =
      typeof job.artifacts.trainingResultPath === "string" ? job.artifacts.trainingResultPath : "";
    if (!trainingResultPath.trim()) {
      return jsonError("Training result artifact path is missing.", 500);
    }

    const trainingResult = await readTrainingResult(trainingResultPath);
    const downloadSpec = resolveAdapterWeightsDownloadSpec({
      job,
      trainingResult,
    });

    tempRoot = await mkdtemp(path.join(tmpdir(), "posttraining-weights-"));
    const archiveDirName = path.posix.basename(downloadSpec.modalRemotePath);
    const localAdapterDir = path.join(tempRoot, archiveDirName);
    await mkdir(localAdapterDir, { recursive: true });

    await runBufferedCommand({
      command: resolveModalBin(),
      args: [
        "volume",
        "get",
        POSTTRAINING_CHECKPOINTS_VOLUME_NAME,
        downloadSpec.modalRemotePath,
        localAdapterDir,
      ],
      signal: request.signal,
      label: "Downloading adapter weights from Modal volume",
    });

    const downloadedEntries = await readdir(localAdapterDir);
    if (downloadedEntries.length === 0) {
      throw new Error("Downloaded adapter directory was empty.");
    }

    const archiveStream = createTarArchiveStream({
      archiveRoot: tempRoot,
      archiveDirName,
      signal: request.signal,
      onFinished: async () => {
        if (tempRoot) {
          await rm(tempRoot, { recursive: true, force: true });
        }
      },
    });

    return new Response(archiveStream, {
      headers: {
        "Content-Type": "application/gzip",
        "Content-Disposition": `attachment; filename="${downloadSpec.archiveFileName}"`,
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    if (tempRoot) {
      await rm(tempRoot, { recursive: true, force: true });
    }

    if (isAbortError(error)) {
      return jsonError("The download was canceled.", 499);
    }

    const message =
      error instanceof Error ? error.message : "Failed to download fine-tuned weights.";
    return jsonError(message, 500);
  }
}
