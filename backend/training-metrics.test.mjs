import test from "node:test";
import assert from "node:assert/strict";

import {
  STRUCTURED_TRAINING_METRIC_PREFIX,
  buildTrainingMetricGraphArtifacts,
  parseStructuredTrainingMetricLine,
  summarizeTrainingMetricRecords,
} from "./training-metrics.mjs";

test("parseStructuredTrainingMetricLine extracts numeric metrics from prefixed output", () => {
  const parsed = parseStructuredTrainingMetricLine(
    `${STRUCTURED_TRAINING_METRIC_PREFIX}${JSON.stringify({
      timestamp: "2026-03-29T12:00:00.000Z",
      step: 10,
      epoch: 0.5,
      metrics: {
        loss: 1.234,
        eval_loss: "1.111",
        learning_rate: 0.0001,
        note: "skip me",
      },
    })}`,
  );

  assert.deepEqual(parsed, {
    timestamp: "2026-03-29T12:00:00.000Z",
    step: 10,
    epoch: 0.5,
    metrics: {
      loss: 1.234,
      eval_loss: 1.111,
      learning_rate: 0.0001,
    },
  });
  assert.equal(parseStructuredTrainingMetricLine("plain log line"), null);
});

test("summarizeTrainingMetricRecords sorts points and tracks latest values", () => {
  const summary = summarizeTrainingMetricRecords([
    {
      timestamp: "2026-03-29T12:00:02.000Z",
      step: 20,
      epoch: 1,
      metrics: { loss: 0.9, learning_rate: 0.00008 },
    },
    {
      timestamp: "2026-03-29T12:00:01.000Z",
      step: 10,
      epoch: 0.5,
      metrics: { loss: 1.2, eval_loss: 1.1, learning_rate: 0.0001 },
    },
  ]);

  assert.deepEqual(summary.availableMetrics, ["eval_loss", "learning_rate", "loss"]);
  assert.deepEqual(summary.series.loss.map((point) => point.step), [10, 20]);
  assert.equal(summary.latest.loss, 0.9);
  assert.equal(summary.latest.eval_loss, 1.1);
  assert.equal(summary.latest.learning_rate, 0.00008);
});

test("buildTrainingMetricGraphArtifacts returns SVG charts for saved training metrics", () => {
  const bundle = buildTrainingMetricGraphArtifacts([
    {
      timestamp: "2026-03-29T12:00:00.000Z",
      step: 10,
      epoch: 0.5,
      metrics: { loss: 1.2, learning_rate: 0.0001 },
    },
    {
      timestamp: "2026-03-29T12:00:10.000Z",
      step: 20,
      epoch: 1,
      metrics: { loss: 0.9, eval_loss: 0.95, learning_rate: 0.00008 },
    },
  ]);

  assert.deepEqual(bundle.summary.availableMetrics, ["eval_loss", "learning_rate", "loss"]);

  const filenames = bundle.artifacts.map((artifact) => artifact.filename).sort();
  assert.deepEqual(filenames, ["learning_rate.svg", "training_loss.svg"]);

  const lossChart = bundle.artifacts.find((artifact) => artifact.filename === "training_loss.svg");
  const learningRateChart = bundle.artifacts.find((artifact) => artifact.filename === "learning_rate.svg");
  assert.match(lossChart?.content ?? "", /<svg[\s\S]*Training Loss/u);
  assert.match(lossChart?.content ?? "", /eval loss/u);
  assert.match(learningRateChart?.content ?? "", /<svg[\s\S]*Learning Rate/u);
});
