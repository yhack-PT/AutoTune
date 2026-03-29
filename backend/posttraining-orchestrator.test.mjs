import test from "node:test";
import assert from "node:assert/strict";

import {
  buildCommandFailureMessage,
  buildDeploymentEnvironment,
  buildDeploymentRecord,
  buildGenerationSmokeTestProbes,
  handleProcessOutputLine,
  extractModalRunUrl,
  parseStructuredLifecycleEventLine,
  resolveDeploymentRuntimePolicy,
  resolveDeploymentArtifact,
  runSmokeTest,
  shouldSuppressProcessOutputLine,
  waitForDeploymentReady,
} from "./posttraining-orchestrator.mjs";

test("extractModalRunUrl returns a contiguous modal.run URL", () => {
  const output = `
✓ Created objects.
└── 🔨 Created web function serve =>
https://andrew-qian64--vllm-lora-customer-support-classify-support-ticket.modal.run
✓ App deployed in 2.914s! 🎉
`;

  assert.equal(
    extractModalRunUrl(output),
    "https://andrew-qian64--vllm-lora-customer-support-classify-support-ticket.modal.run",
  );
});

test("extractModalRunUrl reconstructs a wrapped modal.run URL from deploy output", () => {
  const output = `
✓ Created objects.
└── 🔨 Created web function serve =>
https://andrew-qian64--vllm-lora-customer-support-classify-suppo-6a9926.moda
l.run (label truncated)
✓ App deployed in 2.914s! 🎉
View Deployment:
https://modal.com/apps/andrew-qian64/main/deployed/vllm-lora-customer-support-cl
assify-support-ticket-dc6d7998
`;

  assert.equal(
    extractModalRunUrl(output),
    "https://andrew-qian64--vllm-lora-customer-support-classify-suppo-6a9926.modal.run",
  );
});

test("buildGenerationSmokeTestProbes creates deterministic role and response probes", () => {
  const probes = buildGenerationSmokeTestProbes(
    "Fine-tune a model to act as an AMC/AIME tutor with scaffolded, intuitive explanations.",
  );

  assert.equal(probes.length, 2);
  assert.equal(probes[0].id, "role_demo");
  assert.equal(probes[1].id, "representative_response");
  assert.match(probes[0].prompt, /AMC\/AIME tutor/i);
  assert.match(probes[1].prompt, /representative response/i);
});

test("parseStructuredLifecycleEventLine parses training lifecycle events", () => {
  const payload = parseStructuredLifecycleEventLine(
    'PT_LIFECYCLE_EVENT::{"event":"training_complete","training_result":{"global_step":5}}',
  );

  assert.equal(payload?.event, "training_complete");
  assert.equal(payload?.training_result?.global_step, 5);
  assert.equal(parseStructuredLifecycleEventLine("PT_METRIC_EVENT::{}"), null);
});

test("shouldSuppressProcessOutputLine matches the tokenizer mismatch warning family", () => {
  const warning =
    "[RANK 0] Mismatch between tokenized prompt and the start of tokenized prompt+completion. This may be due to unexpected tokenizer behavior, whitespace issues, or special token handling. Verify that the tokenizer is processing text consistently.";
  const ansiPrefixedWarning =
    "\u001b[1ATokenizing train dataset:   0%|          | 0/29970 [00:00<?, ? examples/s]" +
    warning;

  assert.equal(shouldSuppressProcessOutputLine(warning), true);
  assert.equal(shouldSuppressProcessOutputLine(ansiPrefixedWarning), true);
  assert.equal(shouldSuppressProcessOutputLine("Downloading model files"), false);
});

test("handleProcessOutputLine suppresses tokenizer mismatch warnings while preserving unrelated stdout and ui-progress", async () => {
  const warning =
    "[RANK 0] Mismatch between tokenized prompt and the start of tokenized prompt+completion. This may be due to unexpected tokenizer behavior, whitespace issues, or special token handling. Verify that the tokenizer is processing text consistently.";
  const ansiPrefixedWarning =
    "\u001b[1ATokenizing train dataset:   0%|          | 0/29970 [00:00<?, ? examples/s]" +
    warning;
  const emittedEvents = [];
  const logger = {
    emit(event) {
      emittedEvents.push(event);
    },
  };
  const onOutputLine = async ({ line, logger: activeLogger }) => {
    if (line !== "STRUCTURED_UI_PROGRESS") {
      return false;
    }

    activeLogger.emit({
      stage: "training",
      source: "ui-progress",
      level: "info",
      message: "I'm training the model",
    });
    return true;
  };

  await handleProcessOutputLine({ source: "stdout", line: warning, logger, onOutputLine });
  await handleProcessOutputLine({ source: "stdout", line: ansiPrefixedWarning, logger, onOutputLine });
  await handleProcessOutputLine({ source: "stdout", line: "normal stdout line", logger, onOutputLine });
  await handleProcessOutputLine({ source: "stdout", line: "STRUCTURED_UI_PROGRESS", logger, onOutputLine });

  const emittedMessages = emittedEvents.map((event) => String(event?.message ?? ""));
  assert.equal(
    emittedMessages.some((message) => message.includes("normal stdout line")),
    true,
  );
  assert.ok(emittedMessages.includes("I'm training the model"));
  assert.equal(
    emittedMessages.some((message) => message.includes("Mismatch between tokenized prompt")),
    false,
  );
});

test("buildCommandFailureMessage includes the actionable subprocess tail", () => {
  const message = buildCommandFailureMessage({
    label: "Modal deployment",
    code: 1,
    stdout: `
Building image
No solution found when resolving dependencies:
Because vllm==0.18.0 depends on transformers>=4.56.0,<5 and you
require transformers==5.2.0, we can conclude that your requirements and
vllm==0.18.0 are incompatible.
`,
    stderr: `
Image build failed.
`,
  });

  assert.match(message, /^Modal deployment failed with exit code 1\./);
  assert.match(message, /No solution found when resolving dependencies/);
  assert.match(message, /transformers>=4\.56\.0,<5/);
  assert.match(message, /Image build failed/);
});

test("resolveDeploymentArtifact uses merged_dir only when training produced one", () => {
  assert.deepEqual(
    resolveDeploymentArtifact({
      base_model: "Qwen/Qwen3-8B-Base",
      final_adapter_dir: "/checkpoints/experiments/demo/final_adapter",
      merged_dir: "/checkpoints/experiments/demo/merged",
    }),
    {
      merged: true,
      path: "/checkpoints/experiments/demo/merged",
      relativePath: "experiments/demo/merged",
    },
  );
  assert.deepEqual(
    resolveDeploymentArtifact({
      base_model: "Qwen/Qwen3-8B-Base",
      final_adapter_dir: "/checkpoints/experiments/demo/final_adapter",
      merged_dir: null,
    }),
    {
      merged: false,
      path: "/checkpoints/experiments/demo/final_adapter",
      relativePath: "experiments/demo/final_adapter",
    },
  );
});

test("resolveDeploymentRuntimePolicy keeps Qwen 3 on the standard serving path", () => {
  const policy = resolveDeploymentRuntimePolicy(
    {
      base_model: "Qwen/Qwen3-8B-Base",
      final_adapter_dir: "/checkpoints/experiments/demo/final_adapter",
    },
    {
      merged: false,
      path: "/checkpoints/experiments/demo/final_adapter",
      relativePath: "experiments/demo/final_adapter",
    },
  );

  assert.equal(policy.merged, false);
  assert.equal(policy.vllmPackageSpec, "vllm==0.18.0");
  assert.equal(policy.transformersSpec, "transformers>=4.56.0,<5");
  assert.equal(policy.vllmUseV1, "default");
  assert.equal(policy.modelImpl, "auto");
  assert.equal(policy.languageModelOnly, false);
});

test("buildDeploymentEnvironment keeps Qwen 3 on the standard LoRA serving env", () => {
  const deploymentArtifact = {
    merged: false,
    path: "/checkpoints/experiments/demo/final_adapter",
    relativePath: "experiments/demo/final_adapter",
  };
  const runtimePolicy = resolveDeploymentRuntimePolicy(
    {
      base_model: "Qwen/Qwen3-8B-Base",
      final_adapter_dir: "/checkpoints/experiments/demo/final_adapter",
    },
    deploymentArtifact,
  );

  const env = buildDeploymentEnvironment({
    deploymentAppName: "vllm-lora-demo",
    deploymentArtifact,
    adapterName: "demo-job",
    trainingResult: {
      base_model: "Qwen/Qwen3-8B-Base",
    },
    compiledConfig: {
      max_length: 1024,
    },
    serveGpuType: "H100",
    serveStartupTimeoutSeconds: 1800,
    runtimePolicy,
  });

  assert.deepEqual(env, {
    APP_NAME: "vllm-lora-demo",
    ADAPTER_PATH: "experiments/demo/final_adapter",
    ADAPTER_NAME: "demo-job",
    BASE_MODEL: "Qwen/Qwen3-8B-Base",
    MERGED: "0",
    MAX_MODEL_LEN: "1024",
    GPU_TYPE: "H100",
    SERVE_STARTUP_TIMEOUT_SECONDS: "1800",
    VLLM_PACKAGE_SPEC: "vllm==0.18.0",
    SERVE_TRANSFORMERS_SPEC: "transformers>=4.56.0,<5",
  });
});

test("buildDeploymentRecord captures the standard runtime policy for Qwen 3 and non-Qwen deployments", () => {
  const qwenPolicy = resolveDeploymentRuntimePolicy(
    {
      base_model: "Qwen/Qwen3-8B-Base",
      final_adapter_dir: "/checkpoints/experiments/demo/final_adapter",
    },
    {
      merged: false,
      path: "/checkpoints/experiments/demo/final_adapter",
      relativePath: "experiments/demo/final_adapter",
    },
  );
  const qwenRecord = buildDeploymentRecord({
    deploymentUrl: "https://demo.modal.run",
    deploymentAppName: "vllm-lora-demo",
    adapterName: "demo-job",
    deploymentArtifact: {
      merged: false,
      path: "/checkpoints/experiments/demo/final_adapter",
      relativePath: "experiments/demo/final_adapter",
    },
    trainingResult: {
      base_model: "Qwen/Qwen3-8B-Base",
    },
    serveGpuType: "H100",
    serveStartupTimeoutSeconds: 1800,
    runtimePolicy: qwenPolicy,
  });
  assert.equal(qwenRecord.merged, false);
  assert.equal(qwenRecord.runtimePolicy.merged, false);
  assert.equal(qwenRecord.runtimePolicy.vllmPackageSpec, "vllm==0.18.0");
  assert.equal(qwenRecord.runtimePolicy.transformersSpec, "transformers>=4.56.0,<5");
  assert.equal(qwenRecord.runtimePolicy.modelImpl, "auto");
  assert.equal(qwenRecord.runtimePolicy.languageModelOnly, false);

  const llamaPolicy = resolveDeploymentRuntimePolicy(
    {
      base_model: "meta-llama/Llama-3-8B",
      final_adapter_dir: "/checkpoints/experiments/demo/final_adapter",
    },
    {
      merged: false,
      path: "/checkpoints/experiments/demo/final_adapter",
      relativePath: "experiments/demo/final_adapter",
    },
  );
  const llamaRecord = buildDeploymentRecord({
    deploymentUrl: "https://demo-llama.modal.run",
    deploymentAppName: "vllm-lora-demo-llama",
    adapterName: "demo-llama-job",
    deploymentArtifact: {
      merged: false,
      path: "/checkpoints/experiments/demo/final_adapter",
      relativePath: "experiments/demo/final_adapter",
    },
    trainingResult: {
      base_model: "meta-llama/Llama-3-8B",
    },
    serveGpuType: "A10G",
    serveStartupTimeoutSeconds: 1800,
    runtimePolicy: llamaPolicy,
  });
  assert.equal(llamaRecord.merged, false);
  assert.equal(llamaRecord.runtimePolicy.merged, false);
  assert.equal(llamaRecord.runtimePolicy.vllmPackageSpec, "vllm==0.18.0");
  assert.equal(llamaRecord.runtimePolicy.transformersSpec, "transformers>=4.56.0,<5");
  assert.equal(llamaRecord.runtimePolicy.modelImpl, "auto");
  assert.equal(llamaRecord.runtimePolicy.languageModelOnly, false);
});

test("runSmokeTest records generation probes and completion previews", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  const emittedEvents = [];

  function jsonResponse(body) {
    return {
      ok: true,
      text: async () => JSON.stringify(body),
    };
  }

  try {
    globalThis.fetch = async (url, options = {}) => {
      requests.push({ url: String(url), body: String(options.body ?? "") });

      if (String(url).endsWith("/v1/models")) {
        return jsonResponse({
          data: [{ id: "demo-model" }],
        });
      }

      const requestBody = JSON.parse(String(options.body ?? "{}"));
      const prompt = requestBody.messages?.[0]?.content ?? "";
      return jsonResponse({
        choices: [
          {
            message: {
              content: `This is a sufficiently long completion for probe validation. Prompt excerpt: ${prompt.slice(0, 40)}`,
            },
          },
        ],
      });
    };

    const result = await runSmokeTest({
      jobId: "demo-job",
      deploymentUrl: "https://demo.modal.run",
      model: "demo-model",
      logger: {
        emit(event) {
          emittedEvents.push(event);
        },
        getStage() {
          return "smoke_testing";
        },
      },
      taskSpec: {
        task_family: "generation",
      },
      description:
        "Fine-tune a model to act as an AMC/AIME tutor with scaffolded, intuitive explanations.",
    });

    assert.equal(result.passed, true);
    assert.equal(result.probes.length, 2);
    assert.ok(result.probes.every((probe) => probe.passed));
    assert.ok(result.probes.every((probe) => typeof probe.completionPreview === "string"));
    assert.equal(
      requests.filter((request) => request.url.endsWith("/v1/chat/completions")).length,
      2,
    );
    const uiProgressMessages = emittedEvents
      .filter((event) => event?.source === "ui-progress")
      .map((event) => event.message);
    assert.ok(uiProgressMessages.includes("I'm checking that the model is available"));
    assert.ok(uiProgressMessages.includes("I'm trying a sample request (style example)"));
    assert.ok(uiProgressMessages.includes("I'm trying a sample request (real-world example)"));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("runSmokeTest survives transient startup failures before the model list becomes ready", async () => {
  const originalFetch = globalThis.fetch;
  const originalMaxAttempts = process.env.POSTTRAINING_SMOKE_TEST_MAX_ATTEMPTS;
  const originalRetryDelayMs = process.env.POSTTRAINING_SMOKE_TEST_RETRY_DELAY_MS;
  const originalInitialModelsTimeoutMs = process.env.POSTTRAINING_SMOKE_TEST_INITIAL_MODELS_TIMEOUT_MS;
  const originalModelsTimeoutMs = process.env.POSTTRAINING_SMOKE_TEST_MODELS_TIMEOUT_MS;
  const originalChatTimeoutMs = process.env.POSTTRAINING_SMOKE_TEST_CHAT_TIMEOUT_MS;
  let modelsCalls = 0;
  const emittedEvents = [];

  function jsonResponse(body) {
    return {
      ok: true,
      text: async () => JSON.stringify(body),
    };
  }

  try {
    process.env.POSTTRAINING_SMOKE_TEST_MAX_ATTEMPTS = "3";
    process.env.POSTTRAINING_SMOKE_TEST_RETRY_DELAY_MS = "1";
    process.env.POSTTRAINING_SMOKE_TEST_INITIAL_MODELS_TIMEOUT_MS = "5";
    process.env.POSTTRAINING_SMOKE_TEST_MODELS_TIMEOUT_MS = "5";
    process.env.POSTTRAINING_SMOKE_TEST_CHAT_TIMEOUT_MS = "5";

    globalThis.fetch = async (url, options = {}) => {
      if (String(url).endsWith("/v1/models")) {
        modelsCalls += 1;
        if (modelsCalls < 3) {
          const error = new Error("This operation was aborted");
          error.name = "AbortError";
          throw error;
        }
        return jsonResponse({
          data: [{ id: "demo-model" }],
        });
      }

      const requestBody = JSON.parse(String(options.body ?? "{}"));
      const prompt = requestBody.messages?.[0]?.content ?? "";
      return jsonResponse({
        choices: [
          {
            message: {
              content: `This is a sufficiently long completion for probe validation. Prompt excerpt: ${prompt.slice(0, 40)}`,
            },
          },
        ],
      });
    };

    const result = await runSmokeTest({
      deploymentUrl: "https://demo.modal.run",
      model: "demo-model",
      logger: {
        emit(event) {
          emittedEvents.push(event);
        },
        getStage() {
          return "smoke_testing";
        },
      },
      taskSpec: {
        task_family: "generation",
      },
      description: "Fine-tune a model to act as a step-by-step AMC/AIME tutor.",
    });

    assert.equal(result.passed, true);
    assert.equal(result.attempt, 3);
    assert.equal(modelsCalls, 3);
    assert.equal(result.attempts.length, 2);
    assert.equal(result.attempts[0].error, "This operation was aborted");
    const uiProgressMessages = emittedEvents
      .filter((event) => event?.source === "ui-progress")
      .map((event) => event.message);
    assert.ok(uiProgressMessages.includes("I'm waiting for the model to start (attempt 1/3)"));
    assert.ok(uiProgressMessages.includes("I'm waiting a moment before trying again"));
  } finally {
    globalThis.fetch = originalFetch;

    if (originalMaxAttempts == null) {
      delete process.env.POSTTRAINING_SMOKE_TEST_MAX_ATTEMPTS;
    } else {
      process.env.POSTTRAINING_SMOKE_TEST_MAX_ATTEMPTS = originalMaxAttempts;
    }

    if (originalRetryDelayMs == null) {
      delete process.env.POSTTRAINING_SMOKE_TEST_RETRY_DELAY_MS;
    } else {
      process.env.POSTTRAINING_SMOKE_TEST_RETRY_DELAY_MS = originalRetryDelayMs;
    }

    if (originalInitialModelsTimeoutMs == null) {
      delete process.env.POSTTRAINING_SMOKE_TEST_INITIAL_MODELS_TIMEOUT_MS;
    } else {
      process.env.POSTTRAINING_SMOKE_TEST_INITIAL_MODELS_TIMEOUT_MS = originalInitialModelsTimeoutMs;
    }

    if (originalModelsTimeoutMs == null) {
      delete process.env.POSTTRAINING_SMOKE_TEST_MODELS_TIMEOUT_MS;
    } else {
      process.env.POSTTRAINING_SMOKE_TEST_MODELS_TIMEOUT_MS = originalModelsTimeoutMs;
    }

    if (originalChatTimeoutMs == null) {
      delete process.env.POSTTRAINING_SMOKE_TEST_CHAT_TIMEOUT_MS;
    } else {
      process.env.POSTTRAINING_SMOKE_TEST_CHAT_TIMEOUT_MS = originalChatTimeoutMs;
    }
  }
});

test("waitForDeploymentReady retries until the deployed model is visible", async () => {
  const originalFetch = globalThis.fetch;
  const originalMaxAttempts = process.env.POSTTRAINING_SMOKE_TEST_MAX_ATTEMPTS;
  const originalRetryDelayMs = process.env.POSTTRAINING_SMOKE_TEST_RETRY_DELAY_MS;
  const originalInitialModelsTimeoutMs = process.env.POSTTRAINING_SMOKE_TEST_INITIAL_MODELS_TIMEOUT_MS;
  const originalModelsTimeoutMs = process.env.POSTTRAINING_SMOKE_TEST_MODELS_TIMEOUT_MS;
  let modelsCalls = 0;

  function jsonResponse(body) {
    return {
      ok: true,
      text: async () => JSON.stringify(body),
    };
  }

  try {
    process.env.POSTTRAINING_SMOKE_TEST_MAX_ATTEMPTS = "3";
    process.env.POSTTRAINING_SMOKE_TEST_RETRY_DELAY_MS = "1";
    process.env.POSTTRAINING_SMOKE_TEST_INITIAL_MODELS_TIMEOUT_MS = "5";
    process.env.POSTTRAINING_SMOKE_TEST_MODELS_TIMEOUT_MS = "5";

    globalThis.fetch = async (url) => {
      if (String(url).endsWith("/v1/models")) {
        modelsCalls += 1;
        if (modelsCalls < 3) {
          return jsonResponse({ data: [{ id: "wrong-model" }] });
        }
        return jsonResponse({
          data: [{ id: "demo-model" }],
        });
      }
      throw new Error(`Unexpected URL: ${url}`);
    };

    const result = await waitForDeploymentReady({
      deploymentUrl: "https://demo.modal.run",
      model: "demo-model",
      logger: { emit() {} },
    });

    assert.equal(result.attempt, 3);
    assert.equal(modelsCalls, 3);
    assert.equal(result.attempts.length, 2);
    assert.deepEqual(result.modelIds, ["demo-model"]);
  } finally {
    globalThis.fetch = originalFetch;

    if (originalMaxAttempts == null) {
      delete process.env.POSTTRAINING_SMOKE_TEST_MAX_ATTEMPTS;
    } else {
      process.env.POSTTRAINING_SMOKE_TEST_MAX_ATTEMPTS = originalMaxAttempts;
    }

    if (originalRetryDelayMs == null) {
      delete process.env.POSTTRAINING_SMOKE_TEST_RETRY_DELAY_MS;
    } else {
      process.env.POSTTRAINING_SMOKE_TEST_RETRY_DELAY_MS = originalRetryDelayMs;
    }

    if (originalInitialModelsTimeoutMs == null) {
      delete process.env.POSTTRAINING_SMOKE_TEST_INITIAL_MODELS_TIMEOUT_MS;
    } else {
      process.env.POSTTRAINING_SMOKE_TEST_INITIAL_MODELS_TIMEOUT_MS = originalInitialModelsTimeoutMs;
    }

    if (originalModelsTimeoutMs == null) {
      delete process.env.POSTTRAINING_SMOKE_TEST_MODELS_TIMEOUT_MS;
    } else {
      process.env.POSTTRAINING_SMOKE_TEST_MODELS_TIMEOUT_MS = originalModelsTimeoutMs;
    }
  }
});
