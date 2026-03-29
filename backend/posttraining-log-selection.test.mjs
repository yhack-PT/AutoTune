import test from "node:test";
import assert from "node:assert/strict";

import { buildUiProgressEvent, getSidebarStageProgress } from "../src/lib/posttraining-progress.mjs";
import { selectJobEventsForApi } from "../src/lib/posttraining-log-selection.mjs";

test("selectJobEventsForApi keeps all ui-progress events while trimming raw logs to a tail window", () => {
  const events = [
    buildUiProgressEvent({
      stageId: "recommending",
      text: "I'm figuring out what kind of data this model needs",
    }),
    { source: "stdout", message: "raw-1" },
    { source: "stdout", message: "raw-2" },
    { source: "stdout", message: "raw-3" },
    buildUiProgressEvent({
      stageId: "compiling",
      text: "I'm putting together the training plan",
    }),
    { source: "stdout", message: "raw-4" },
    { source: "stdout", message: "raw-5" },
    { source: "stdout", message: "raw-6" },
  ];

  const selected = selectJobEventsForApi(events, { rawTailLimit: 2 });

  assert.deepEqual(
    selected.map((event) => String(event.message ?? "")),
    [
      "I'm figuring out what kind of data this model needs",
      "I'm putting together the training plan",
      "raw-5",
      "raw-6",
    ],
  );
});

test("sidebar progress persists completed-stage bullets even after many raw logs arrive later", () => {
  const events = [
    buildUiProgressEvent({
      stageId: "recommending",
      text: "I'm figuring out what kind of data this model needs",
    }),
    buildUiProgressEvent({
      stageId: "recommending",
      text: "I'm reviewing the most promising dataset options",
    }),
    ...Array.from({ length: 300 }, (_, index) => ({
      source: "stdout",
      message: `raw-${index + 1}`,
    })),
    buildUiProgressEvent({
      stageId: "deploying",
      text: "I'm getting the model ready to go live",
    }),
  ];

  const logs = selectJobEventsForApi(events, { rawTailLimit: 10 });
  const visibleProgress = getSidebarStageProgress({
    logs,
    activeStageId: "deploying",
    completedStageIds: ["recommending"],
    failedStageId: null,
    jobStatus: "deploying",
  });

  assert.deepEqual(visibleProgress.recommending, [
    {
      id: "recommending:0",
      text: "I'm figuring out what kind of data this model needs",
      tone: "normal",
    },
    {
      id: "recommending:1",
      text: "I'm reviewing the most promising dataset options",
      tone: "normal",
    },
  ]);
  assert.deepEqual(visibleProgress.deploying, [
    {
      id: "deploying:12",
      text: "I'm getting the model ready to go live",
      tone: "normal",
    },
  ]);
});
