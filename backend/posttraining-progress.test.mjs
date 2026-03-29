import test from "node:test";
import assert from "node:assert/strict";

import {
  UI_PROGRESS_SOURCE,
  buildUiProgressEvent,
  getSidebarStageProgress,
  mergeStageProgressById,
  parseUiProgressEvent,
} from "../src/lib/posttraining-progress.mjs";

test("buildUiProgressEvent creates a stable ui-progress event payload", () => {
  assert.deepEqual(
    buildUiProgressEvent({
      stageId: "training",
      text: "I'm training the model (40 steps completed)",
    }),
    {
      stage: "training",
      source: UI_PROGRESS_SOURCE,
      level: "info",
      message: "I'm training the model (40 steps completed)",
      data: {
        tone: "normal",
      },
    },
  );
});

test("parseUiProgressEvent ignores non-ui events and normalizes valid ones", () => {
  assert.equal(parseUiProgressEvent({ source: "stdout", message: "hello" }), null);
  assert.deepEqual(
    parseUiProgressEvent({
      stage: "smoke_testing",
      source: UI_PROGRESS_SOURCE,
      level: "error",
      message: "Smoke test failed on attempt 3",
    }),
    {
      stageId: "smoke_testing",
      text: "Smoke test failed on attempt 3",
      tone: "error",
    },
  );
});

test("getSidebarStageProgress shows the latest recommendation query line for the active stage", () => {
  const logs = [
    buildUiProgressEvent({
      stageId: "recommending",
      text: "I'm figuring out what kind of data this model needs",
    }),
    buildUiProgressEvent({
      stageId: "recommending",
      text: "I'm searching through different datasets that could fit this request",
    }),
  ];

  assert.deepEqual(
    getSidebarStageProgress({
      logs,
      activeStageId: "recommending",
      completedStageIds: [],
      failedStageId: null,
      jobStatus: "recommending",
    }),
    {
      recommending: [
        {
          id: "recommending:0",
          text: "I'm figuring out what kind of data this model needs",
          tone: "normal",
        },
        {
          id: "recommending:1",
          text: "I'm searching through different datasets that could fit this request",
          tone: "normal",
        },
      ],
    },
  );
});

test("getSidebarStageProgress keeps completed stage history and active training history without adding done", () => {
  const logs = [
    buildUiProgressEvent({
      stageId: "compiling",
      text: "I'm saving the training setup",
    }),
    buildUiProgressEvent({
      stageId: "training",
      text: "I'm loading the model",
    }),
    buildUiProgressEvent({
      stageId: "training",
      text: "I'm training the model (30 steps completed)",
    }),
  ];

  assert.deepEqual(
    getSidebarStageProgress({
      logs,
      activeStageId: "training",
      completedStageIds: ["compiling"],
      failedStageId: null,
      jobStatus: "training",
    }),
    {
      compiling: [
        {
          id: "compiling:0",
          text: "I'm saving the training setup",
          tone: "normal",
        },
      ],
      training: [
        {
          id: "training:1",
          text: "I'm loading the model",
          tone: "normal",
        },
        {
          id: "training:2",
          text: "I'm training the model (30 steps completed)",
          tone: "normal",
        },
      ],
    },
  );
});

test("getSidebarStageProgress keeps the smoke-test history while active", () => {
  const logs = [
    buildUiProgressEvent({
      stageId: "compiling",
      text: "I'm saving the training setup",
    }),
    buildUiProgressEvent({
      stageId: "smoke_testing",
      text: "I'm waiting for the model to start (attempt 2/12)",
    }),
    buildUiProgressEvent({
      stageId: "smoke_testing",
      text: "I'm waiting a moment before trying again",
    }),
  ];

  assert.deepEqual(
    getSidebarStageProgress({
      logs,
      activeStageId: "smoke_testing",
      completedStageIds: ["compiling"],
      failedStageId: null,
      jobStatus: "smoke_testing",
    }),
    {
      compiling: [
        {
          id: "compiling:0",
          text: "I'm saving the training setup",
          tone: "normal",
        },
      ],
      smoke_testing: [
        {
          id: "smoke_testing:1",
          text: "I'm waiting for the model to start (attempt 2/12)",
          tone: "normal",
        },
        {
          id: "smoke_testing:2",
          text: "I'm waiting a moment before trying again",
          tone: "normal",
        },
      ],
    },
  );
});

test("getSidebarStageProgress preserves the failed stage error line", () => {
  const logs = [
    buildUiProgressEvent({
      stageId: "training",
      text: "I'm training the model (40 steps completed)",
    }),
    buildUiProgressEvent({
      stageId: "training",
      text: "Training failed: CUDA out of memory.",
      tone: "error",
    }),
  ];

  assert.deepEqual(
    getSidebarStageProgress({
      logs,
      activeStageId: null,
      completedStageIds: ["recommending", "compiling"],
      failedStageId: "training",
      jobStatus: "failed",
    }),
    {
      training: [
        {
          id: "training:0",
          text: "I'm training the model (40 steps completed)",
          tone: "normal",
        },
        {
          id: "training:1",
          text: "Training failed: CUDA out of memory.",
          tone: "error",
        },
      ],
    },
  );
});

test("getSidebarStageProgress does not trim stage history by default", () => {
  const logs = Array.from({ length: 8 }, (_, index) =>
    buildUiProgressEvent({
      stageId: "recommending",
      text: `progress-${index + 1}`,
    }),
  );

  assert.deepEqual(
    getSidebarStageProgress({
      logs,
      activeStageId: "recommending",
      completedStageIds: [],
      failedStageId: null,
      jobStatus: "recommending",
    }).recommending,
    [
      { id: "recommending:0", text: "progress-1", tone: "normal" },
      { id: "recommending:1", text: "progress-2", tone: "normal" },
      { id: "recommending:2", text: "progress-3", tone: "normal" },
      { id: "recommending:3", text: "progress-4", tone: "normal" },
      { id: "recommending:4", text: "progress-5", tone: "normal" },
      { id: "recommending:5", text: "progress-6", tone: "normal" },
      { id: "recommending:6", text: "progress-7", tone: "normal" },
      { id: "recommending:7", text: "progress-8", tone: "normal" },
    ],
  );
});

test("getSidebarStageProgress still supports an explicit maxItemsPerStage cap", () => {
  const logs = Array.from({ length: 4 }, (_, index) =>
    buildUiProgressEvent({
      stageId: "training",
      text: `step-${index + 1}`,
    }),
  );

  assert.deepEqual(
    getSidebarStageProgress({
      logs,
      activeStageId: "training",
      completedStageIds: [],
      failedStageId: null,
      jobStatus: "training",
      maxItemsPerStage: 2,
    }).training,
    [
      { id: "training:2", text: "step-3", tone: "normal" },
      { id: "training:3", text: "step-4", tone: "normal" },
    ],
  );
});

test("getSidebarStageProgress keeps consecutive duplicate ui-progress items", () => {
  const logs = [
    buildUiProgressEvent({
      stageId: "recommending",
      text: "I'm searching through different datasets that could fit this request",
    }),
    buildUiProgressEvent({
      stageId: "recommending",
      text: "I'm searching through different datasets that could fit this request",
    }),
  ];

  assert.deepEqual(
    getSidebarStageProgress({
      logs,
      activeStageId: "recommending",
      completedStageIds: [],
      failedStageId: null,
      jobStatus: "recommending",
    }).recommending,
    [
      {
        id: "recommending:0",
        text: "I'm searching through different datasets that could fit this request",
        tone: "normal",
      },
      {
        id: "recommending:1",
        text: "I'm searching through different datasets that could fit this request",
        tone: "normal",
      },
    ],
  );
});

test("mergeStageProgressById preserves previously seen bullets when a later snapshot is shorter", () => {
  const previousProgress = {
    recommending: [
      { id: "recommending:0", text: "progress-1", tone: "normal" },
      { id: "recommending:1", text: "progress-2", tone: "normal" },
    ],
    compiling: [
      { id: "compiling:0", text: "compile-1", tone: "normal" },
    ],
  };
  const nextProgress = {
    recommending: [
      { id: "recommending:0", text: "progress-1", tone: "normal" },
    ],
    training: [
      { id: "training:0", text: "training-1", tone: "normal" },
    ],
  };

  assert.deepEqual(mergeStageProgressById(previousProgress, nextProgress), {
    recommending: [
      { id: "recommending:0", text: "progress-1", tone: "normal" },
      { id: "recommending:1", text: "progress-2", tone: "normal" },
    ],
    compiling: [
      { id: "compiling:0", text: "compile-1", tone: "normal" },
    ],
    training: [
      { id: "training:0", text: "training-1", tone: "normal" },
    ],
  });
});
