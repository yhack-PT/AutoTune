import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";

const {
  buildAdapterWeightsArchiveFileName,
  normalizeCheckpointsVolumePath,
  resolveAdapterWeightsDownloadSpec,
  resolveModalBin,
} = await import(new URL("./posttraining-download.ts", import.meta.url).href);

test("normalizeCheckpointsVolumePath converts /checkpoints paths into Modal volume paths", () => {
  assert.equal(
    normalizeCheckpointsVolumePath("/checkpoints/experiments/demo/final_adapter"),
    "experiments/demo/final_adapter",
  );
});

test("normalizeCheckpointsVolumePath rejects blank, missing, and non-checkpoints paths", () => {
  assert.throws(() => normalizeCheckpointsVolumePath(""), /final_adapter_dir/);
  assert.throws(() => normalizeCheckpointsVolumePath(undefined), /final_adapter_dir/);
  assert.throws(() => normalizeCheckpointsVolumePath("/tmp/demo/final_adapter"), /\/checkpoints/);
});

test("buildAdapterWeightsArchiveFileName produces the expected tarball name", () => {
  assert.equal(
    buildAdapterWeightsArchiveFileName("demo-job-1234"),
    "demo-job-1234-adapter-weights.tar.gz",
  );
});

test("resolveAdapterWeightsDownloadSpec derives the adapter download inputs from the job", () => {
  const spec = resolveAdapterWeightsDownloadSpec({
    job: {
      jobId: "demo-job-1234",
      artifacts: {
        trainingResultPath: "/tmp/demo-job/training_result.json",
      },
    },
    trainingResult: {
      final_adapter_dir: "/checkpoints/experiments/demo-job-1234/final_adapter",
    },
  });

  assert.deepEqual(spec, {
    archiveFileName: "demo-job-1234-adapter-weights.tar.gz",
    modalRemotePath: "experiments/demo-job-1234/final_adapter",
    trainingResultPath: "/tmp/demo-job/training_result.json",
  });
});

test("resolveModalBin follows MODAL_BIN, then .venv/bin/modal, then modal", () => {
  const repoRoot = "/workspace/demo";
  const venvModal = path.join(repoRoot, ".venv", "bin", "modal");

  assert.equal(
    resolveModalBin({
      env: {
        ...process.env,
        MODAL_BIN: "/custom/modal",
      },
      repoRoot,
      existsSyncFn: () => true,
    }),
    "/custom/modal",
  );

  assert.equal(
    resolveModalBin({
      env: { ...process.env },
      repoRoot,
      existsSyncFn: (candidate: string) => candidate === venvModal,
    }),
    venvModal,
  );

  assert.equal(
    resolveModalBin({
      env: { ...process.env },
      repoRoot,
      existsSyncFn: () => false,
    }),
    "modal",
  );
});
