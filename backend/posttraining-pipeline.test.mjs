import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";

import {
  coerceTaskSpecForRawInput,
  recommendDatasets,
  RecommendationFailure,
} from "./hf-dataset-recommender.mjs";
import {
  filterSourceSchemaForTextOnlyTraining,
  inferDeterministicNormalization,
} from "./posttraining-normalization.mjs";
import {
  buildSpecPrompt,
  getCompilerOverridesConfigPath,
  resolveConfiguredSftGpuType,
  resolveConfiguredSftNumTrainEpochs,
  runCompiler,
} from "./posttraining-spec-compiler.mjs";

function buildPlan() {
  return {
    task_spec: {
      supported: true,
      task_family: "classification",
      target_policy: "single_target",
      output_shape_preference: "prompt_completion",
      objective_summary: "Classify support tickets by priority.",
      unsupported_reason: null,
    },
    analysis: {
      domain_summary: "Customer support ticket priority classification.",
      mapped_task_types: ["text-classification"],
      data_format_needed: "completion",
      quality_tier_strategy: "Prefer one strong dataset.",
    },
    search_queries: [
      {
        search: "customer support tickets",
        task_filter: "text-classification",
        sort: "downloads",
        min_rows: 1000,
        intent: "Find trainable support ticket data.",
      },
    ],
    ranking_criteria: [],
    recommendation_guidance: {
      ideal_dataset_count: 1,
      target_total_rows: "10K-100K",
      mixing_strategy: "Prefer a single strong dataset.",
      warnings: [],
    },
  };
}

function buildDirectCompatibleCandidate() {
  return {
    id: "acme/support-ticket-bodies",
    source_url: "https://huggingface.co/datasets/acme/support-ticket-bodies",
    description: "Support tickets with priority labels.",
    matched_queries: ["customer support tickets"],
    matched_tasks: ["text-classification"],
    downloads: 100,
    likes: 10,
    num_rows: 25000,
    license: "apache-2.0",
    splits: ["train"],
    schema_signals: ["has_text", "classification_ready"],
    viewer_accessible: true,
    size_partial: false,
    warnings: [],
    compatibility_status: "compatible",
    compatibility_reason: "Classification-style normalization is available via 'body' and 'priority'.",
    normalization_source: "deterministic",
    compatible_methods: ["sft"],
    source_schema: {
      available_columns: ["body", "priority"],
      sample_rows: [{ body: "My router is offline", priority: "high" }],
    },
    selected_target_column: "priority",
    target_selection_reason: "Selected 'priority' as the sole plausible native label column.",
    target_selection_confidence: 0.95,
    target_candidates: ["priority"],
    ambiguity_warnings: [],
    preferred_dataset_config: "default",
    preferred_train_split: "train",
    preferred_eval_split: null,
    normalization_proposal: {
      version: 1,
      shape: "prompt_completion",
      strategy: "classification_template",
      source_columns: ["body", "priority"],
      fields: {
        text: null,
        prompt: {
          source_column: null,
          template: "Classify the following example. Return only the label.\n\nInput:\n{body}\n\nLabel:",
          value_mapping: null,
        },
        completion: {
          source_column: "priority",
          template: null,
          value_mapping: null,
        },
      },
    },
  };
}

function buildStructuredCompatibleCandidate() {
  return {
    dataset: "acme/structured-ticket-metadata",
    source_url: "https://huggingface.co/datasets/acme/structured-ticket-metadata",
    score: 91,
    why: "Structured ticket metadata can be serialized into a classification prompt.",
    matched_queries: ["customer support tickets"],
    mapped_task_types: ["text-classification"],
    downloads: 80,
    likes: 6,
    num_rows: 50000,
    license: "apache-2.0",
    splits: ["train"],
    schema_signals: ["classification_ready"],
    compatibility_status: "compatible",
    compatibility_reason: "Template synthesis can serialize the structured row into a classification prompt.",
    normalization_source: "openai",
    compatible_methods: ["sft"],
    source_schema: {
      available_columns: ["operator", "issue_type", "priority", "status", "channel"],
      sample_rows: [
        {
          operator: "Airtel",
          issue_type: "network_issue",
          priority: "high",
          status: "open",
          channel: "email",
        },
      ],
    },
    preferred_dataset_config: "default",
    preferred_train_split: "train",
    preferred_eval_split: null,
    warnings: [],
    selected_target_column: "issue_type",
    target_selection_reason: "Selected 'issue_type' from the available native label columns for this dataset.",
    target_selection_confidence: 0.7,
    target_candidates: ["issue_type", "priority"],
    ambiguity_warnings: [
      "Multiple plausible target columns were detected ('issue_type', 'priority'); using 'issue_type' as the selected target.",
    ],
    normalization_proposal: {
      version: 1,
      shape: "prompt_completion",
      strategy: "classification_template",
      source_columns: ["operator", "issue_type", "priority", "status", "channel"],
      fields: {
        text: null,
        prompt: {
          source_column: null,
          template:
            "Classify the following example. Return only the label.\n\nOperator: {operator}\nChannel: {channel}\nStatus: {status}\nPriority: {priority}\n\nLabel:",
          value_mapping: null,
        },
        completion: {
          source_column: "issue_type",
          template: null,
          value_mapping: null,
        },
      },
    },
  };
}

function buildIdentityMappedClassificationCandidate() {
  return {
    dataset: "acme/priority-classification",
    source_url: "https://huggingface.co/datasets/acme/priority-classification",
    score: 92,
    why: "Priority labels are directly trainable from ticket bodies.",
    matched_queries: ["customer support tickets"],
    mapped_task_types: ["text-classification"],
    downloads: 150,
    likes: 12,
    num_rows: 12000,
    license: "apache-2.0",
    splits: ["train"],
    schema_signals: ["classification_ready"],
    compatibility_status: "compatible",
    compatibility_reason: "Classification-style normalization is available via 'body' and 'priority'.",
    normalization_source: "deterministic",
    compatible_methods: ["sft"],
    source_schema: {
      available_columns: ["body", "priority"],
      sample_rows: [
        { body: "Router offline", priority: "high" },
        { body: "Password reset request", priority: "medium" },
      ],
    },
    preferred_dataset_config: "default",
    preferred_train_split: "train",
    preferred_eval_split: null,
    warnings: [],
    selected_target_column: "priority",
    target_selection_reason: "Selected 'priority' as the sole plausible native label column.",
    target_selection_confidence: 0.95,
    target_candidates: ["priority"],
    ambiguity_warnings: [],
    normalization_proposal: {
      version: 1,
      shape: "prompt_completion",
      strategy: "classification_template",
      source_columns: ["body", "priority"],
      fields: {
        text: null,
        prompt: {
          source_column: null,
          template:
            "Classify the following example. Return only the label.\n\nAvailable labels: high, medium\n\nInput:\n{body}\n\nLabel:",
          value_mapping: null,
        },
        completion: {
          source_column: "priority",
          template: null,
          value_mapping: {
            high: "high",
            medium: "medium",
          },
        },
      },
    },
  };
}

function toRecommendedCandidate(candidate) {
  return {
    dataset: candidate.dataset ?? candidate.id,
    source_url: candidate.source_url,
    score: 95,
    why: candidate.why ?? "Best fit for training.",
    matched_queries: candidate.matched_queries,
    mapped_task_types: candidate.matched_tasks ?? candidate.mapped_task_types ?? ["text-classification"],
    downloads: candidate.downloads,
    likes: candidate.likes,
    num_rows: candidate.num_rows,
    license: candidate.license,
    splits: candidate.splits,
    schema_signals: candidate.schema_signals,
    compatibility_status: candidate.compatibility_status,
    compatibility_reason: candidate.compatibility_reason,
    normalization_source: candidate.normalization_source,
    normalization_proposal: candidate.normalization_proposal,
    compatible_methods: candidate.compatible_methods,
    selected_target_column: candidate.selected_target_column,
    target_selection_reason: candidate.target_selection_reason,
    target_selection_confidence: candidate.target_selection_confidence,
    target_candidates: candidate.target_candidates,
    ambiguity_warnings: candidate.ambiguity_warnings,
    source_schema: candidate.source_schema,
    preferred_dataset_config: candidate.preferred_dataset_config,
    preferred_train_split: candidate.preferred_train_split,
    preferred_eval_split: candidate.preferred_eval_split,
    warnings: candidate.warnings,
  };
}

function buildSpec(datasetId) {
  return {
    objective_summary: "Train a customer support adaptation.",
    method: "sft",
    adaptation_strategy: "lora",
    artifact_strategy: "adapter",
    base_model: {
      model_id: "Qwen/Qwen3.5-9B-Base",
      revision: null,
    },
    selected_datasets: [
      {
        dataset: datasetId,
        dataset_config: "default",
        train_split: "train",
        eval_split: null,
        weight: 1,
        warnings: [],
        include_reason: "Primary training dataset.",
      },
    ],
    compute_preset: {
      gpu_type: "A10",
      max_length: 1024,
      per_device_train_batch_size: 1,
      per_device_eval_batch_size: 1,
      gradient_accumulation_steps: 8,
    },
    training_params: {
      learning_rate: 0.0001,
      num_train_epochs: 1,
      max_steps: 20,
      lora_r: 16,
      lora_alpha: 32,
      lora_dropout: 0.05,
      target_modules: ["q_proj"],
      logging_steps: 10,
      save_steps: 10,
      eval_steps: 10,
    },
    seed_artifact: null,
    notes: [],
  };
}

async function withTempDir(callback) {
  const tempRoot = await mkdtemp(path.join(tmpdir(), "pt-pipeline-"));
  try {
    return await callback(tempRoot);
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
}

test("recommendation fails early with diagnostics when all candidates are incompatible", async () => {
  const incompatibleCandidate = {
    id: "acme/metadata-only",
    source_url: "https://huggingface.co/datasets/acme/metadata-only",
    description: "Ticket metadata only.",
    matched_queries: ["customer support tickets"],
    matched_tasks: ["text-classification"],
    downloads: 20,
    likes: 0,
    num_rows: 10000,
    license: "apache-2.0",
    splits: ["train"],
    schema_signals: ["classification_ready"],
    viewer_accessible: true,
    size_partial: false,
    warnings: [],
    compatibility_status: "incompatible",
    compatibility_reason: "No supported normalization proposal could be inferred.",
    normalization_source: null,
    normalization_proposal: null,
    compatible_methods: [],
    selected_target_column: null,
    target_selection_reason: null,
    target_selection_confidence: null,
    target_candidates: ["issue_type", "priority"],
    ambiguity_warnings: [],
    source_schema: {
      available_columns: ["issue_type", "priority"],
      sample_rows: [{ issue_type: "billing", priority: "high" }],
    },
    preferred_dataset_config: "default",
    preferred_train_split: "train",
    preferred_eval_split: null,
  };

  await assert.rejects(
    recommendDatasets(buildPlan(), {
      discoverCandidates: async () => [incompatibleCandidate],
      enrichCandidates: async () => [incompatibleCandidate],
      rankCandidates: async () => [],
      skipDebugWrite: true,
    }),
    (error) => {
      assert.ok(error instanceof RecommendationFailure);
      assert.equal(error.recommendation.fatal_error.code, "no_compatible_candidates");
      assert.equal(error.recommendation.recommended_datasets.length, 0);
      assert.equal(error.recommendation.ranked_datasets.length, 1);
      return true;
    },
  );
});

test("multi-target classification requests are coerced to a single target with warning", () => {
  const coerced = coerceTaskSpecForRawInput(
    {
      ...buildPlan(),
      task_spec: {
        supported: false,
        task_family: "unsupported",
        target_policy: "unsupported",
        output_shape_preference: "unsupported",
        objective_summary: "",
        selected_target_focus: null,
        requested_targets: [],
        task_warnings: [],
        target_selection_reason: null,
        unsupported_reason: "Request asks for two classification targets.",
      },
    },
    {
      domain: "customer support",
      useCase: "Classify support tickets by issue type and urgency.",
    },
  );

  assert.equal(coerced.task_spec.supported, true);
  assert.equal(coerced.task_spec.task_family, "classification");
  assert.equal(coerced.task_spec.target_policy, "single_target");
  assert.equal(coerced.task_spec.selected_target_focus, "issue type");
  assert.deepEqual(coerced.task_spec.requested_targets, ["issue type", "urgency"]);
  assert.match(coerced.task_spec.task_warnings[0], /continuing with 'issue type'/i);
  assert.match(coerced.recommendation_guidance.warnings[0], /continuing with 'issue type'/i);
});

test("single-text requests are also coerced to a single target with warning", () => {
  const coerced = coerceTaskSpecForRawInput(
    {
      ...buildPlan(),
      task_spec: {
        supported: false,
        task_family: "unsupported",
        target_policy: "unsupported",
        output_shape_preference: "unsupported",
        objective_summary: "",
        selected_target_focus: null,
        requested_targets: [],
        task_warnings: [],
        target_selection_reason: null,
        unsupported_reason: "Request asks for two classification targets.",
      },
    },
    {
      description: "Classify support tickets by issue type and urgency.",
    },
  );

  assert.equal(coerced.task_spec.supported, true);
  assert.equal(coerced.task_spec.selected_target_focus, "issue type");
  assert.deepEqual(coerced.task_spec.requested_targets, ["issue type", "urgency"]);
  assert.match(coerced.task_spec.task_warnings[0], /continuing with 'issue type'/i);
});

test("recommendDatasets accepts a single-text payload when a plan generator is provided", async () => {
  const recommendation = await recommendDatasets(
    {
      description: "Classify customer support tickets by priority.",
    },
    {
      planGenerator: async () => buildPlan(),
      discoverCandidates: async () => [buildDirectCompatibleCandidate()],
      enrichCandidates: async () => [buildDirectCompatibleCandidate()],
      rankCandidates: async () => [buildIdentityMappedClassificationCandidate()],
      skipDebugWrite: true,
    },
  );

  assert.equal(recommendation.recommended_datasets.length, 1);
  assert.equal(recommendation.recommended_datasets[0].dataset, "acme/priority-classification");
  assert.equal(recommendation.search_queries.length, 1);
});

test("recommendDatasets defaults dataset ranking to gpt-5.4", async () => {
  const previousApiKey = process.env.OPENAI_API_KEY;
  const previousGlobalModel = process.env.OPENAI_MODEL;
  const previousSelectionModel = process.env.OPENAI_DATASET_SELECTION_MODEL;
  const originalFetch = globalThis.fetch;
  const candidate = buildDirectCompatibleCandidate();
  let capturedModel = null;

  try {
    process.env.OPENAI_API_KEY = "test-openai-key";
    delete process.env.OPENAI_MODEL;
    delete process.env.OPENAI_DATASET_SELECTION_MODEL;

    globalThis.fetch = async (_url, options = {}) => {
      const requestBody = JSON.parse(String(options.body ?? "{}"));
      capturedModel = requestBody.model ?? null;

      return {
        ok: true,
        json: async () => ({
          id: "resp_test_dataset_selection",
          status: "completed",
          model: "gpt-5.4",
          output_text: JSON.stringify({
            recommended_datasets: [
              {
                dataset: candidate.id,
                score: 95,
                why: "Best fit for this classification task.",
                warnings: [],
              },
            ],
          }),
        }),
      };
    };

    const recommendation = await recommendDatasets(buildPlan(), {
      discoverCandidates: async () => [candidate],
      enrichCandidates: async () => [candidate],
      skipDebugWrite: true,
    });

    assert.equal(capturedModel, "gpt-5.4");
    assert.equal(recommendation.recommended_datasets.length, 1);
    assert.equal(recommendation.recommended_datasets[0].dataset, candidate.id);
  } finally {
    globalThis.fetch = originalFetch;

    if (previousApiKey === undefined) {
      delete process.env.OPENAI_API_KEY;
    } else {
      process.env.OPENAI_API_KEY = previousApiKey;
    }

    if (previousGlobalModel === undefined) {
      delete process.env.OPENAI_MODEL;
    } else {
      process.env.OPENAI_MODEL = previousGlobalModel;
    }

    if (previousSelectionModel === undefined) {
      delete process.env.OPENAI_DATASET_SELECTION_MODEL;
    } else {
      process.env.OPENAI_DATASET_SELECTION_MODEL = previousSelectionModel;
    }
  }
});

test("recommendDatasets planning prompt asks for time-based quality tier strategy", async () => {
  const previousApiKey = process.env.OPENAI_API_KEY;
  const previousGlobalModel = process.env.OPENAI_MODEL;
  const originalFetch = globalThis.fetch;
  const candidate = buildDirectCompatibleCandidate();
  let capturedPrompt = "";

  try {
    process.env.OPENAI_API_KEY = "test-openai-key";
    delete process.env.OPENAI_MODEL;

    globalThis.fetch = async (_url, options = {}) => {
      const requestBody = JSON.parse(String(options.body ?? "{}"));
      capturedPrompt = String(requestBody.input?.[1]?.content ?? "");

      return {
        ok: true,
        json: async () => ({
          id: "resp_test_plan_prompt",
          status: "completed",
          model: "gpt-5-mini",
          output_text: JSON.stringify(buildPlan()),
        }),
      };
    };

    await recommendDatasets(
      {
        description: "Classify customer support tickets by priority.",
      },
      {
        discoverCandidates: async () => [candidate],
        enrichCandidates: async () => [candidate],
        rankCandidates: async () => [toRecommendedCandidate(candidate)],
        skipDebugWrite: true,
      },
    );

    assert.match(capturedPrompt, /Infer a reasonable end-to-end run-time budget directly from the user's request\./);
    assert.match(capturedPrompt, /If the user states a time preference or deadline, honor it when possible\./);
    assert.match(capturedPrompt, /If the user does not specify a time, default to an 8-hour run budget\./);
    assert.match(capturedPrompt, /quality_tier_strategy.*wall-clock time-budget summary/i);
    assert.match(capturedPrompt, /Do not use a numeric 1-5 tier or score\./);
    assert.ok(!capturedPrompt.includes("under 10K rows"));
    assert.ok(!capturedPrompt.includes("4-6+ datasets"));
    assert.ok(!capturedPrompt.includes("1 = Fastest"));
    assert.ok(!capturedPrompt.includes("quality tier 3"));
  } finally {
    globalThis.fetch = originalFetch;

    if (previousApiKey === undefined) {
      delete process.env.OPENAI_API_KEY;
    } else {
      process.env.OPENAI_API_KEY = previousApiKey;
    }

    if (previousGlobalModel === undefined) {
      delete process.env.OPENAI_MODEL;
    } else {
      process.env.OPENAI_MODEL = previousGlobalModel;
    }
  }
});

test("spec planner prompt hardcodes one training epoch for backend SFT jobs", () => {
  const prompt = buildSpecPrompt({
    objectiveSummary: "Classify support tickets.",
    context: {
      task_spec: buildPlan().task_spec,
      analysis: buildPlan().analysis,
      recommendation_guidance: buildPlan().recommendation_guidance,
      search_queries: buildPlan().search_queries,
    },
    candidateProfiles: [buildDirectCompatibleCandidate()],
    platformConstraints: {
      supported_methods: ["sft"],
      supported_task_families: ["classification"],
      supported_target_policies: ["single_target"],
      allowed_base_models: [{ model_id: "Qwen/Qwen3.5-9B-Base", revision: null }],
      allowed_compute_gpus: ["A10", "L40S", "H100"],
    },
    jobId: "prompt-test-job",
  });

  assert.match(prompt, /training_params\.num_train_epochs to 1\.0/i);
});

test("compiler override can force a different SFT epoch count than the planner proposed", async () => {
  const previousOverridePath = process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;

  try {
    await withTempDir(async (tempRoot) => {
      const overridesPath = path.join(tempRoot, "compiler-overrides.yaml");
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = overridesPath;
      await writeFile(overridesPath, "sft_num_train_epochs: 2.0\n", "utf8");

      assert.equal(getCompilerOverridesConfigPath(), overridesPath);
      assert.equal(resolveConfiguredSftNumTrainEpochs(), 2.0);

      const enrichedCandidate = buildDirectCompatibleCandidate();
      const recommendation = await recommendDatasets(buildPlan(), {
        discoverCandidates: async () => [enrichedCandidate],
        enrichCandidates: async () => [enrichedCandidate],
        rankCandidates: async () => [toRecommendedCandidate(enrichedCandidate)],
        skipDebugWrite: true,
      });

      const inputPath = path.join(tempRoot, "recommendation.json");
      await writeFile(inputPath, `${JSON.stringify(recommendation, null, 2)}\n`, "utf8");

      const spec = buildSpec("acme/support-ticket-bodies");
      const result = await runCompiler(
        {
          inputPath,
          outputRoot: tempRoot,
          jobId: "epoch-override-job",
          objectiveSummary: "Classify support tickets.",
        },
        {
          specPlanner: async () => ({
            parsed: {
              ...spec,
              training_params: {
                ...spec.training_params,
                num_train_epochs: 3,
              },
            },
            model: "test-model",
            response_id: "resp_epoch_override",
          }),
        },
      );

      assert.equal(result.compiled_config.num_train_epochs, 2);
      assert.match(result.spec.notes.at(-1) ?? "", /Compiler override applied: num_train_epochs=2/);
    });
  } finally {
    if (previousOverridePath === undefined) {
      delete process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;
    } else {
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = previousOverridePath;
    }
  }
});

test("compiler override can force a different SFT GPU type than the planner proposed", async () => {
  const previousOverridePath = process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;

  try {
    await withTempDir(async (tempRoot) => {
      const overridesPath = path.join(tempRoot, "compiler-overrides.yaml");
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = overridesPath;
      await writeFile(overridesPath, "sft_gpu_type: l40s\n", "utf8");

      assert.equal(getCompilerOverridesConfigPath(), overridesPath);
      assert.equal(resolveConfiguredSftGpuType(), "L40S");

      const enrichedCandidate = buildDirectCompatibleCandidate();
      const recommendation = await recommendDatasets(buildPlan(), {
        discoverCandidates: async () => [enrichedCandidate],
        enrichCandidates: async () => [enrichedCandidate],
        rankCandidates: async () => [toRecommendedCandidate(enrichedCandidate)],
        skipDebugWrite: true,
      });

      const inputPath = path.join(tempRoot, "recommendation.json");
      await writeFile(inputPath, `${JSON.stringify(recommendation, null, 2)}\n`, "utf8");

      const result = await runCompiler(
        {
          inputPath,
          outputRoot: tempRoot,
          jobId: "gpu-override-job",
          objectiveSummary: "Classify support tickets.",
        },
        {
          specPlanner: async () => buildSpec("acme/support-ticket-bodies"),
        },
      );

      const compiledConfigRaw = await readFile(result.compiled_config_path, "utf8");
      assert.match(compiledConfigRaw, /gpu_type:\s*"L40S"/);

      const specRaw = await readFile(result.spec_path, "utf8");
      assert.match(specRaw, /Compiler override applied: gpu_type=L40S\./);
    });
  } finally {
    if (previousOverridePath === undefined) {
      delete process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;
    } else {
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = previousOverridePath;
    }
  }
});

test("deterministic classification normalization omits sampled label constraints", () => {
  const normalization = inferDeterministicNormalization(
    ["body", "priority"],
    [
      { body: "Billing issue", priority: "high" },
      { body: "Refund request", priority: "medium" },
    ],
  );

  assert.equal(normalization.compatibility_status, "compatible");
  assert.equal(normalization.normalization_proposal.shape, "prompt_completion");
  assert.equal(normalization.normalization_proposal.fields.completion.value_mapping, null);
  assert.doesNotMatch(
    normalization.normalization_proposal.fields.prompt.template,
    /Available labels:/,
  );
  assert.match(
    normalization.normalization_proposal.fields.prompt.template,
    /Label:\n$/,
  );
});

test("text-only schema filtering strips image blob columns from sampled source rows", () => {
  const filtered = filterSourceSchemaForTextOnlyTraining(
    ["image", "image_id", "caption", "cui"],
    [
      {
        image: { bytes: "iVBORw0KGgoAAAANSUhEUgAA", path: null },
        image_id: "study-1",
        caption: "Portable chest radiograph shows bibasilar opacities.",
        cui: "atelectasis",
      },
    ],
  );

  assert.deepEqual(filtered.excluded_columns, ["image", "image_id"]);
  assert.deepEqual(filtered.available_columns, ["caption", "cui"]);
  assert.deepEqual(filtered.sample_rows, [
    {
      caption: "Portable chest radiograph shows bibasilar opacities.",
      cui: "atelectasis",
    },
  ]);
});

test("a direct-compatible dataset survives recommendation and compiles", async () => {
  const enrichedCandidate = buildDirectCompatibleCandidate();
  const recommendation = await recommendDatasets(buildPlan(), {
    discoverCandidates: async () => [enrichedCandidate],
    enrichCandidates: async () => [enrichedCandidate],
    rankCandidates: async () => [toRecommendedCandidate(enrichedCandidate)],
    skipDebugWrite: true,
  });

  assert.equal(recommendation.recommended_datasets.length, 1);
  assert.equal(recommendation.recommended_datasets[0].normalization_proposal.shape, "prompt_completion");
  assert.equal(recommendation.recommended_datasets[0].selected_target_column, "priority");

  await withTempDir(async (tempRoot) => {
    const inputPath = path.join(tempRoot, "recommendation.json");
    await writeFile(inputPath, `${JSON.stringify(recommendation, null, 2)}\n`, "utf8");

    const result = await runCompiler(
      {
        inputPath,
        outputRoot: tempRoot,
        jobId: "direct-compatible-job",
        objectiveSummary: "Classify support tickets.",
      },
      {
        specPlanner: async () => ({
          parsed: buildSpec("acme/support-ticket-bodies"),
          model: "test-model",
          response_id: "resp_direct",
        }),
      },
    );

    assert.equal(result.selected_datasets[0].normalization_shape, "prompt_completion");
    const manifest = JSON.parse(await readFile(result.manifest_path, "utf8"));
    assert.equal(manifest.selected_datasets[0].normalization.shape, "prompt_completion");
    assert.equal(manifest.selected_datasets[0].selected_target_column, "priority");
    assert.equal(result.compiled_config.max_steps, -1);
  });
});

test("a structured ticket dataset can compile with a synthesized classification recipe", async () => {
  const recommendation = {
    task_spec: buildPlan().task_spec,
    analysis: buildPlan().analysis,
    search_queries: buildPlan().search_queries,
    ranking_criteria: [],
    recommendation_guidance: buildPlan().recommendation_guidance,
    compatibility_summary: {
      total_candidates: 1,
      compatible_candidates: 1,
      incompatible_candidates: 0,
    },
    ranked_datasets: [buildStructuredCompatibleCandidate()],
    recommended_datasets: [buildStructuredCompatibleCandidate()],
  };

  await withTempDir(async (tempRoot) => {
    const inputPath = path.join(tempRoot, "recommendation.json");
    await writeFile(inputPath, `${JSON.stringify(recommendation, null, 2)}\n`, "utf8");

    const result = await runCompiler(
      {
        inputPath,
        outputRoot: tempRoot,
        jobId: "structured-compatible-job",
        objectiveSummary: "Classify support tickets.",
      },
      {
        specPlanner: async () => ({
          parsed: buildSpec("acme/structured-ticket-metadata"),
          model: "test-model",
          response_id: "resp_structured",
        }),
      },
    );

    const manifest = JSON.parse(await readFile(result.manifest_path, "utf8"));
    assert.equal(manifest.selected_datasets[0].normalization.shape, "prompt_completion");
    assert.match(
      manifest.selected_datasets[0].normalization.fields.prompt.template,
      /Operator: \{operator\}/,
    );
    assert.match(
      manifest.selected_datasets[0].normalization.fields.prompt.template,
      /Label:\n$/,
    );
    assert.equal(manifest.selected_datasets[0].selected_target_column, "issue_type");
  });
});

test("invalid normalization proposals are rejected during compilation", async () => {
  const invalidCandidate = {
    ...buildStructuredCompatibleCandidate(),
    dataset: "acme/invalid-normalization",
    normalization_proposal: {
      version: 1,
      shape: "text",
      strategy: "template_synthesis",
      source_columns: ["missing_column"],
      fields: {
        text: {
          source_column: "missing_column",
          template: null,
          value_mapping: null,
        },
        prompt: null,
        completion: null,
      },
    },
  };

  const recommendation = {
    task_spec: buildPlan().task_spec,
    analysis: buildPlan().analysis,
    search_queries: buildPlan().search_queries,
    ranking_criteria: [],
    recommendation_guidance: buildPlan().recommendation_guidance,
    compatibility_summary: {
      total_candidates: 1,
      compatible_candidates: 1,
      incompatible_candidates: 0,
    },
    ranked_datasets: [invalidCandidate],
    recommended_datasets: [invalidCandidate],
  };

  await withTempDir(async (tempRoot) => {
    const inputPath = path.join(tempRoot, "recommendation.json");
    await writeFile(inputPath, `${JSON.stringify(recommendation, null, 2)}\n`, "utf8");

    await assert.rejects(
      runCompiler(
        {
          inputPath,
          outputRoot: tempRoot,
          jobId: "invalid-normalization-job",
          objectiveSummary: "Classify support tickets.",
        },
        {
          specPlanner: async () => ({
            parsed: buildSpec("acme/invalid-normalization"),
            model: "test-model",
            response_id: "resp_invalid",
          }),
        },
      ),
      /missing_column|unknown column|valid normalization proposal/,
    );
  });
});

test("compiler strips identity value_mapping on direct label-column completions", async () => {
  const candidate = buildIdentityMappedClassificationCandidate();
  const recommendation = {
    task_spec: buildPlan().task_spec,
    analysis: buildPlan().analysis,
    search_queries: buildPlan().search_queries,
    ranking_criteria: [],
    recommendation_guidance: buildPlan().recommendation_guidance,
    compatibility_summary: {
      total_candidates: 1,
      compatible_candidates: 1,
      incompatible_candidates: 0,
    },
    ranked_datasets: [candidate],
    recommended_datasets: [candidate],
  };

  await withTempDir(async (tempRoot) => {
    const inputPath = path.join(tempRoot, "recommendation.json");
    await writeFile(inputPath, `${JSON.stringify(recommendation, null, 2)}\n`, "utf8");

    const result = await runCompiler(
      {
        inputPath,
        outputRoot: tempRoot,
        jobId: "identity-mapping-job",
        objectiveSummary: "Classify support tickets.",
      },
      {
        specPlanner: async () => ({
          parsed: buildSpec("acme/priority-classification"),
          model: "test-model",
          response_id: "resp_identity_mapping",
        }),
      },
    );

    const manifest = JSON.parse(await readFile(result.manifest_path, "utf8"));
    assert.equal(
      manifest.selected_datasets[0].normalization.fields.completion.value_mapping,
      null,
    );
  });
});
