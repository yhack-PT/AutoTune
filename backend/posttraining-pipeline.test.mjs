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
  getSpecSchema,
  getCompilerOverridesConfigPath,
  resolveConfiguredSftDataset,
  resolveConfiguredSftGpuType,
  resolveConfiguredSftMaxLength,
  resolveConfiguredSftNumTrainEpochs,
  runCompiler,
  validateStrictStructuredOutputSchema,
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

function buildGenerationPlan() {
  return {
    task_spec: {
      supported: true,
      task_family: "generation",
      target_policy: "none",
      output_shape_preference: "text",
      objective_summary:
        "Fine-tune an LLM to act as a step-by-step AMC/AIME tutor with strong mathematical intuition.",
      selected_target_focus: null,
      requested_targets: [],
      task_warnings: [],
      target_selection_reason: null,
      unsupported_reason: null,
    },
    analysis: {
      domain_summary: "Contest-math tutoring with scaffolded, high-quality step-by-step explanations.",
      mapped_task_types: ["text-generation"],
      data_format_needed: "raw_text",
      quality_tier_strategy: "Prefer one strong raw-text tutor corpus for a short focused run.",
    },
    search_queries: [
      {
        search: "amc aime tutor text",
        task_filter: "text-generation",
        sort: "downloads",
        min_rows: 1000,
        intent: "Find raw tutor-style math text for generation SFT.",
      },
    ],
    ranking_criteria: [],
    recommendation_guidance: {
      ideal_dataset_count: 1,
      target_total_rows: "10K-100K",
      mixing_strategy: "Prefer a single strong raw-text dataset.",
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

function buildTextGenerationCandidate() {
  return {
    dataset: "acme/aime-tutor-text",
    source_url: "https://huggingface.co/datasets/acme/aime-tutor-text",
    score: 96,
    why: "Curated raw-text tutor transcripts closely match the desired AMC/AIME teaching style.",
    matched_queries: ["amc aime tutor text"],
    mapped_task_types: ["text-generation"],
    downloads: 140,
    likes: 9,
    num_rows: 18000,
    license: "apache-2.0",
    splits: ["train"],
    schema_signals: ["has_text"],
    compatibility_status: "compatible",
    compatibility_reason: "Direct text normalization is available via 'text'.",
    normalization_source: "deterministic",
    compatible_methods: ["sft"],
    source_schema: {
      available_columns: ["text"],
      sample_rows: [
        {
          text:
            "User: Help me solve this AMC problem.\nAssistant: Let's start by identifying the symmetry in the expression before we compute anything.",
        },
      ],
    },
    preferred_dataset_config: "default",
    preferred_train_split: "train",
    preferred_eval_split: null,
    warnings: [],
    selected_target_column: null,
    target_selection_reason: null,
    target_selection_confidence: null,
    target_candidates: [],
    ambiguity_warnings: [],
    normalization_proposal: {
      version: 1,
      shape: "text",
      strategy: "copy_column",
      source_columns: ["text"],
      fields: {
        text: {
          source_column: "text",
          template: null,
          value_mapping: null,
        },
        prompt: null,
        completion: null,
      },
    },
  };
}

function buildPromptCompletionGenerationCandidate() {
  return {
    dataset: "acme/aime-tutor-instruction-pairs",
    source_url: "https://huggingface.co/datasets/acme/aime-tutor-instruction-pairs",
    score: 88,
    why: "Instruction and response pairs directly supervise the target generation behavior.",
    matched_queries: ["amc aime tutor text"],
    mapped_task_types: ["text-generation"],
    downloads: 70,
    likes: 5,
    num_rows: 9000,
    license: "apache-2.0",
    splits: ["train"],
    schema_signals: ["instruction_ready"],
    compatibility_status: "compatible",
    compatibility_reason:
      "Candidate is compatible with paired prompt/completion generation SFT requirements.",
    normalization_source: "deterministic",
    compatible_methods: ["sft"],
    source_schema: {
      available_columns: ["instruction", "response"],
      sample_rows: [
        {
          instruction: "Guide a student through this AMC problem.",
          response: "Let's break the problem into two simpler subproblems first.",
        },
      ],
    },
    preferred_dataset_config: "default",
    preferred_train_split: "train",
    preferred_eval_split: null,
    warnings: [],
    selected_target_column: null,
    target_selection_reason: null,
    target_selection_confidence: null,
    target_candidates: [],
    ambiguity_warnings: [],
    normalization_proposal: {
      version: 1,
      shape: "prompt_completion",
      strategy: "copy_columns",
      source_columns: ["instruction", "response"],
      fields: {
        text: null,
        prompt: {
          source_column: "instruction",
          template: null,
          value_mapping: null,
        },
        completion: {
          source_column: "response",
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
    source_splits: candidate.source_splits ?? candidate.splits,
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
      model_id: "Qwen/Qwen3-8B-Base",
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

function buildGenerationSpec(datasetId) {
  return {
    objective_summary: "Train an AMC/AIME tutor adaptation.",
    method: "sft",
    adaptation_strategy: "lora",
    artifact_strategy: "adapter",
    base_model: {
      model_id: "Qwen/Qwen3-8B-Base",
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
        include_reason: "Primary raw-text tutor corpus.",
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

const NEUTRAL_COMPILER_OVERRIDES_YAML =
  "sft_num_train_epochs: null\nsft_gpu_type: null\nsft_max_length: null\nsft_dataset: null\n";

async function withCompilerOverrides(contents, callback) {
  const previousOverridePath = process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;
  const tempRoot = await mkdtemp(path.join(tmpdir(), "pt-compiler-overrides-"));
  const overridesPath = path.join(tempRoot, "compiler-overrides.yaml");
  await writeFile(overridesPath, contents, "utf8");
  process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = overridesPath;

  try {
    return await callback(overridesPath);
  } finally {
    if (previousOverridePath === undefined) {
      delete process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;
    } else {
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = previousOverridePath;
    }
    await rm(tempRoot, { recursive: true, force: true });
  }
}

async function runCompilerWithNeutralOverrides(args, options) {
  return withCompilerOverrides(NEUTRAL_COMPILER_OVERRIDES_YAML, () => runCompiler(args, options));
}

function buildMockJsonResponse(payload) {
  return {
    ok: true,
    headers: {
      get() {
        return null;
      },
    },
    json: async () => payload,
  };
}

function buildMockErrorResponse(status, body = "") {
  return {
    ok: false,
    status,
    headers: {
      get() {
        return null;
      },
    },
    text: async () => body,
  };
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

test("recommendDatasets accepts open-ended generation requests with raw-text candidates", async () => {
  const candidate = buildTextGenerationCandidate();
  const recommendation = await recommendDatasets(
    {
      description:
        "Fine-tune a language model to act as an expert AMC and AIME tutor who teaches step by step.",
    },
    {
      planGenerator: async () => buildGenerationPlan(),
      discoverCandidates: async () => [candidate],
      enrichCandidates: async () => [candidate],
      rankCandidates: async () => [toRecommendedCandidate(candidate)],
      skipDebugWrite: true,
    },
  );

  assert.equal(recommendation.task_spec.task_family, "generation");
  assert.equal(recommendation.task_spec.target_policy, "none");
  assert.equal(recommendation.task_spec.output_shape_preference, "text");
  assert.equal(recommendation.recommended_datasets.length, 1);
  assert.equal(recommendation.recommended_datasets[0].dataset, candidate.dataset);
  assert.equal(recommendation.recommended_datasets[0].normalization_proposal.shape, "text");
});

test("recommendDatasets accepts prompt-completion candidates for generation", async () => {
  const candidate = buildPromptCompletionGenerationCandidate();
  const recommendation = await recommendDatasets(buildGenerationPlan(), {
    discoverCandidates: async () => [candidate],
    enrichCandidates: async () => [candidate],
    rankCandidates: async () => [toRecommendedCandidate(candidate)],
    skipDebugWrite: true,
  });

  assert.equal(recommendation.task_spec.task_family, "generation");
  assert.equal(recommendation.recommended_datasets.length, 1);
  assert.equal(recommendation.recommended_datasets[0].dataset, candidate.dataset);
  assert.equal(recommendation.recommended_datasets[0].normalization_proposal.shape, "prompt_completion");
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

test("recommendDatasets retries planning with a smaller time budget after no compatible candidates", async () => {
  const previousApiKey = process.env.OPENAI_API_KEY;
  const previousGlobalModel = process.env.OPENAI_MODEL;
  const originalFetch = globalThis.fetch;
  const compatibleCandidate = buildDirectCompatibleCandidate();
  const incompatibleCandidate = {
    ...compatibleCandidate,
    id: "acme/incompatible-priority-classification",
    source_url: "https://huggingface.co/datasets/acme/incompatible-priority-classification",
    compatibility_status: "incompatible",
    compatibility_reason: "No supported normalization proposal could be inferred.",
    normalization_source: null,
    normalization_proposal: null,
    compatible_methods: [],
    selected_target_column: null,
    target_selection_reason: null,
    target_selection_confidence: null,
    target_candidates: ["priority"],
  };
  const firstPlan = {
    ...buildPlan(),
    analysis: {
      ...buildPlan().analysis,
      quality_tier_strategy: "About 8 hours.",
    },
    search_queries: [
      {
        ...buildPlan().search_queries[0],
        search: "broad support ticket classification",
        min_rows: 10000,
      },
    ],
    recommendation_guidance: {
      ...buildPlan().recommendation_guidance,
      target_total_rows: "100K-300K",
    },
  };
  const retryPlan = {
    ...buildPlan(),
    analysis: {
      ...buildPlan().analysis,
      quality_tier_strategy: "About 2-4 hours.",
    },
    search_queries: [
      {
        ...buildPlan().search_queries[0],
        search: "small support ticket classification",
        min_rows: 1000,
      },
    ],
    recommendation_guidance: {
      ...buildPlan().recommendation_guidance,
      target_total_rows: "10K-50K",
      warnings: [],
    },
  };
  const planningPrompts = [];
  let planningCallCount = 0;

  try {
    process.env.OPENAI_API_KEY = "test-openai-key";
    delete process.env.OPENAI_MODEL;

    globalThis.fetch = async (_url, options = {}) => {
      const requestBody = JSON.parse(String(options.body ?? "{}"));
      planningPrompts.push(String(requestBody.input?.[1]?.content ?? ""));
      planningCallCount += 1;

      return {
        ok: true,
        json: async () => ({
          id: `resp_test_retry_${planningCallCount}`,
          status: "completed",
          model: "gpt-5-mini",
          output_text: JSON.stringify(planningCallCount === 1 ? firstPlan : retryPlan),
        }),
      };
    };

    const recommendation = await recommendDatasets(
      {
        description: "Classify customer support tickets by priority.",
      },
      {
        discoverCandidates: async (context) =>
          context.analysis.quality_tier_strategy === retryPlan.analysis.quality_tier_strategy
            ? [compatibleCandidate]
            : [incompatibleCandidate],
        enrichCandidates: async (candidates) => candidates,
        rankCandidates: async () => [toRecommendedCandidate(compatibleCandidate)],
        skipDebugWrite: true,
      },
    );

    assert.equal(planningCallCount, 2);
    assert.equal(recommendation.search_queries[0].search, retryPlan.search_queries[0].search);
    assert.equal(recommendation.recommended_datasets[0].dataset, compatibleCandidate.id);
    assert.match(planningPrompts[1], /Retry context: the previous inferred time budget did not yield compatible public datasets\./);
    assert.match(planningPrompts[1], /Previous time-budget summary: About 8 hours\./);
    assert.match(planningPrompts[1], /Retry with a smaller inferred run-time budget than before\./);
    assert.match(
      recommendation.recommendation_guidance.warnings.join(" "),
      /smaller inferred run-time budget/i,
    );
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

test("recommendDatasets uses the expanded default smaller-time-budget retry budget for text input", async () => {
  const incompatibleCandidate = {
    ...buildDirectCompatibleCandidate(),
    id: "acme/never-compatible",
    source_url: "https://huggingface.co/datasets/acme/never-compatible",
    compatibility_status: "incompatible",
    compatibility_reason: "No supported normalization proposal could be inferred.",
    normalization_source: null,
    normalization_proposal: null,
    compatible_methods: [],
    selected_target_column: null,
    target_selection_reason: null,
    target_selection_confidence: null,
    target_candidates: ["priority"],
  };
  const planningHints = [];
  let planningCallCount = 0;

  await assert.rejects(
    recommendDatasets(
      {
        description: "Classify customer support tickets by priority.",
      },
      {
        planGenerator: async (_input, planningHintsForCall = {}) => {
          planningHints.push(planningHintsForCall);
          planningCallCount += 1;
          return {
            ...buildPlan(),
            analysis: {
              ...buildPlan().analysis,
              quality_tier_strategy: `Attempt ${planningCallCount}`,
            },
            search_queries: [
              {
                ...buildPlan().search_queries[0],
                search: `customer support tickets retry ${planningCallCount}`,
              },
            ],
          };
        },
        discoverCandidates: async () => [incompatibleCandidate],
        enrichCandidates: async (candidates) => candidates,
        rankCandidates: async () => [],
        skipDebugWrite: true,
      },
    ),
    (error) => {
      assert.ok(error instanceof RecommendationFailure);
      assert.equal(error.recommendation.fatal_error.code, "no_compatible_candidates");
      assert.equal(planningCallCount, 4);
      assert.equal(planningHints.length, 4);
      assert.equal(Boolean(planningHints[0]?.smaller_time_budget), false);
      assert.ok(planningHints.slice(1).every((hints) => hints?.smaller_time_budget === true));
      return true;
    },
  );
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

test("recommendDatasets falls back to Hub card row counts when viewer size fails", async () => {
  const originalFetch = globalThis.fetch;
  const logMessages = [];
  const candidate = {
    id: "acme/carddata-row-count",
    source_url: "https://huggingface.co/datasets/acme/carddata-row-count",
    description: "Raw-text tutoring transcripts.",
    matched_queries: ["amc aime tutor text"],
    matched_tasks: ["text-generation"],
    downloads: 42,
    likes: 3,
    gated: false,
    private: false,
    tags: ["task_categories:text-generation"],
    cardData: {
      license: "apache-2.0",
      dataset_info: {
        features: [{ name: "text", dtype: "string" }],
        splits: [{ name: "train", num_examples: 12345 }],
      },
    },
  };

  try {
    globalThis.fetch = async (url) => {
      const parsedUrl = new URL(String(url));

      if (parsedUrl.pathname.endsWith("/is-valid")) {
        return buildMockJsonResponse({ viewer: true, preview: true });
      }
      if (parsedUrl.pathname.endsWith("/size")) {
        return buildMockErrorResponse(500, "temporary viewer failure");
      }
      if (parsedUrl.pathname.endsWith("/splits")) {
        return buildMockJsonResponse({
          splits: [{ config: "default", split: "train" }],
        });
      }
      if (parsedUrl.pathname.endsWith("/first-rows")) {
        return buildMockJsonResponse({
          features: [{ name: "text" }],
          rows: [
            {
              row: {
                text: "User: Teach me this AMC problem. Assistant: Let's start by isolating the invariant.",
              },
            },
          ],
        });
      }

      throw new Error(`Unexpected fetch URL in test: ${url}`);
    };

    const recommendation = await recommendDatasets(buildGenerationPlan(), {
      discoverCandidates: async () => [candidate],
      rankCandidates: async (compatibleCandidates) => [
        toRecommendedCandidate(compatibleCandidates[0]),
      ],
      logger: {
        emit(event) {
          logMessages.push(String(event?.message ?? ""));
        },
      },
      skipDebugWrite: true,
    });

    assert.equal(recommendation.recommended_datasets.length, 1);
    assert.equal(recommendation.recommended_datasets[0].dataset, candidate.id);
    assert.equal(recommendation.recommended_datasets[0].num_rows, 12345);
    assert.match(
      logMessages.join("\n"),
      /HF viewer \/size for acme\/carddata-row-count failed with 500; retrying/i,
    );
    assert.match(
      logMessages.join("\n"),
      /HF viewer \/size unavailable for acme\/carddata-row-count; using Hub card metadata row count \(12345\)\./i,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("recommendDatasets emits ui-progress lines for query search and candidate review", async () => {
  const originalFetch = globalThis.fetch;
  const progressMessages = [];
  const candidate = buildTextGenerationCandidate();
  const plan = buildGenerationPlan();
  plan.search_queries = [
    {
      ...plan.search_queries[0],
      search: "clinical dialogue dataset",
    },
  ];

  try {
    globalThis.fetch = async (url) => {
      const parsedUrl = new URL(String(url));

      if (parsedUrl.hostname === "huggingface.co" && parsedUrl.pathname === "/api/datasets") {
        return buildMockJsonResponse([
          {
            id: candidate.dataset,
            description: candidate.description,
            downloads: candidate.downloads,
            likes: candidate.likes,
            gated: false,
            private: false,
            tags: candidate.matched_tasks,
            cardData: {},
          },
        ]);
      }

      throw new Error(`Unexpected fetch URL in test: ${url}`);
    };

    const recommendation = await recommendDatasets(plan, {
      enrichCandidates: async () => [candidate],
      rankCandidates: async () => [toRecommendedCandidate(candidate)],
      logger: {
        emit(event) {
          if (event?.source === "ui-progress") {
            progressMessages.push(String(event.message ?? ""));
          }
        },
      },
      skipDebugWrite: true,
    });

    assert.equal(recommendation.recommended_datasets.length, 1);
    assert.equal(recommendation.recommended_datasets[0].dataset, candidate.dataset);
    assert.equal(
      progressMessages.filter(
        (message) => message === "I'm searching through different datasets that could fit this request",
      ).length,
      1,
    );
    assert.ok(progressMessages.includes("I'm reviewing the most promising dataset options"));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("recommendDatasets planning prompt describes both classification and generation SFT paths", async () => {
  const previousApiKey = process.env.OPENAI_API_KEY;
  const previousGlobalModel = process.env.OPENAI_MODEL;
  const originalFetch = globalThis.fetch;
  const candidate = buildTextGenerationCandidate();
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
          id: "resp_test_generation_plan_prompt",
          status: "completed",
          model: "gpt-5-mini",
          output_text: JSON.stringify(buildGenerationPlan()),
        }),
      };
    };

    await recommendDatasets(
      {
        description:
          "Fine-tune a model to act as a contest-math tutor with step-by-step explanations.",
      },
      {
        discoverCandidates: async () => [candidate],
        enrichCandidates: async () => [candidate],
        rankCandidates: async () => [toRecommendedCandidate(candidate)],
        skipDebugWrite: true,
      },
    );

    assert.match(capturedPrompt, /supports two SFT task families/i);
    assert.match(capturedPrompt, /classification = finite-label prediction/i);
    assert.match(capturedPrompt, /generation = open-ended behavior, source-to-target generation, summarization, or domain adaptation/i);
    assert.match(capturedPrompt, /Use output_shape_preference 'prompt_completion' for source-to-target, instruction-following, or summarization tasks/i);
    assert.match(capturedPrompt, /Do not force open-ended behavior requests into classification/i);
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

test("spec planner prompt hardcodes one training epoch and sane learning-rate guidance for backend SFT jobs", () => {
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
      allowed_base_models: [{ model_id: "Qwen/Qwen3-8B-Base", revision: null }],
      allowed_compute_gpus: ["A10", "L40S", "H100"],
    },
    jobId: "prompt-test-job",
  });

  assert.match(prompt, /training_params\.num_train_epochs to 1\.0/i);
  assert.match(prompt, /leave training_params\.learning_rate null/i);
  assert.match(prompt, /never above 1e-3/i);
});

test("compiler override can hardcode a specific Hugging Face dataset URL", async () => {
  const previousOverridePath = process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;

  try {
    await withTempDir(async (tempRoot) => {
      const overridesPath = path.join(tempRoot, "compiler-overrides.yaml");
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = overridesPath;
      await writeFile(
        overridesPath,
        "sft_dataset: https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes\n",
        "utf8",
      );

      assert.equal(getCompilerOverridesConfigPath(), overridesPath);
      assert.equal(resolveConfiguredSftDataset(), "starmpcc/Asclepius-Synthetic-Clinical-Notes");

      const hardcodedCandidate = {
        ...buildPromptCompletionGenerationCandidate(),
        dataset: "starmpcc/Asclepius-Synthetic-Clinical-Notes",
        source_url: "https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes",
        why: "Clinical transcript-to-note pairs directly match the requested note-drafting task.",
      };
      const distractorCandidate = buildPromptCompletionGenerationCandidate();
      const recommendation = await recommendDatasets(buildGenerationPlan(), {
        discoverCandidates: async () => [distractorCandidate],
        enrichCandidates: async () => [distractorCandidate],
        rankCandidates: async () => [toRecommendedCandidate(distractorCandidate)],
        skipDebugWrite: true,
      });

      const inputPath = path.join(tempRoot, "recommendation.json");
      await writeFile(inputPath, `${JSON.stringify(recommendation, null, 2)}\n`, "utf8");

      const result = await runCompiler(
        {
          inputPath,
          outputRoot: tempRoot,
          jobId: "dataset-override-job",
          objectiveSummary: "Draft clinical notes from clinician-patient conversations.",
        },
        {
          specPlanner: async () => ({
            parsed: buildGenerationSpec(distractorCandidate.dataset),
            model: "test-model",
            response_id: "resp_dataset_override",
          }),
          datasetOverrideResolver: async (datasetId, context) => {
            assert.equal(datasetId, "starmpcc/Asclepius-Synthetic-Clinical-Notes");
            assert.equal(context.task_spec?.task_family, "generation");
            return hardcodedCandidate;
          },
        },
      );

      assert.equal(result.spec.selected_datasets.length, 1);
      assert.equal(result.spec.selected_datasets[0].dataset, hardcodedCandidate.dataset);
      assert.match(
        result.spec.notes.join("\n"),
        /Compiler override applied: dataset=starmpcc\/Asclepius-Synthetic-Clinical-Notes\./,
      );
    });
  } finally {
    if (previousOverridePath === undefined) {
      delete process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;
    } else {
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = previousOverridePath;
    }
  }
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

test("compiler epoch override scales merged holdout eval size for short SFT runs", async () => {
  const previousOverridePath = process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;

  try {
    await withTempDir(async (tempRoot) => {
      const overridesPath = path.join(tempRoot, "compiler-overrides.yaml");
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = overridesPath;
      await writeFile(overridesPath, "sft_num_train_epochs: 0.01\n", "utf8");

      assert.equal(getCompilerOverridesConfigPath(), overridesPath);
      assert.equal(resolveConfiguredSftNumTrainEpochs(), 0.01);

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
          jobId: "epoch-override-holdout-job",
          objectiveSummary: "Classify support tickets.",
        },
        {
          specPlanner: async () => ({
            parsed: {
              ...spec,
              training_params: {
                ...spec.training_params,
                num_train_epochs: 1,
                max_steps: -1,
              },
            },
            model: "test-model",
            response_id: "resp_epoch_override_holdout",
          }),
        },
      );

      assert.equal(result.compiled_config.num_train_epochs, 0.01);
      assert.equal(result.spec.evaluation_plan.holdout_fraction, 0.001);
      assert.equal(result.compiled_config.evaluation_plan.holdout_fraction, 0.001);
      assert.equal(result.compiled_config.training_estimate.estimated_eval_examples, 25);
      assert.match(result.spec.notes.join("\n"), /Compiler override applied: num_train_epochs=0.01/);
    });
  } finally {
    if (previousOverridePath === undefined) {
      delete process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;
    } else {
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = previousOverridePath;
    }
  }
});

test("compiler fallback floors epoch-based short runs to five training steps", async () => {
  const previousOverridePath = process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;

  try {
    await withTempDir(async (tempRoot) => {
      const overridesPath = path.join(tempRoot, "compiler-overrides.yaml");
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = overridesPath;
      await writeFile(overridesPath, "sft_num_train_epochs: 0.01\n", "utf8");

      const tinyCandidate = {
        ...buildDirectCompatibleCandidate(),
        id: "acme/tiny-support-ticket-bodies",
        num_rows: 1000,
        source_url: "https://huggingface.co/datasets/acme/tiny-support-ticket-bodies",
      };
      const recommendation = await recommendDatasets(buildPlan(), {
        discoverCandidates: async () => [tinyCandidate],
        enrichCandidates: async () => [tinyCandidate],
        rankCandidates: async () => [toRecommendedCandidate(tinyCandidate)],
        skipDebugWrite: true,
      });

      const inputPath = path.join(tempRoot, "recommendation.json");
      await writeFile(inputPath, `${JSON.stringify(recommendation, null, 2)}\n`, "utf8");

      const invalidSpec = {
        ...buildSpec(tinyCandidate.id),
        selected_datasets: [],
      };
      const result = await runCompiler(
        {
          inputPath,
          outputRoot: tempRoot,
          jobId: "fallback-min-steps-job",
          objectiveSummary: "Classify support tickets.",
        },
        {
          specPlanner: async () => ({
            parsed: invalidSpec,
            model: "test-model",
            response_id: "resp_force_fallback",
          }),
        },
      );

      assert.equal(result.compiled_config.num_train_epochs, 0.01);
      assert.equal(result.compiled_config.max_steps, 5);
      assert.equal(result.compiled_config.training_estimate.expected_total_steps, 5);
      assert.match(
        result.spec.notes.join("\n"),
        /Compiler safeguard applied: max_steps=5 because epoch-based expected_total_steps=1 fell below the minimum of 5\./,
      );
    });
  } finally {
    if (previousOverridePath === undefined) {
      delete process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;
    } else {
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = previousOverridePath;
    }
  }
});

test("compiler fallback resets absurd SFT learning rates to the backend default", async () => {
  await withTempDir(async (tempRoot) => {
    const candidate = buildDirectCompatibleCandidate();
    const recommendation = await recommendDatasets(buildPlan(), {
      discoverCandidates: async () => [candidate],
      enrichCandidates: async () => [candidate],
      rankCandidates: async () => [toRecommendedCandidate(candidate)],
      skipDebugWrite: true,
    });

    const inputPath = path.join(tempRoot, "recommendation.json");
    await writeFile(inputPath, `${JSON.stringify(recommendation, null, 2)}\n`, "utf8");

    const baseSpec = buildSpec(candidate.dataset);
    const invalidSpec = {
      ...baseSpec,
      training_params: {
        ...baseSpec.training_params,
        learning_rate: 2,
      },
    };
    const result = await runCompilerWithNeutralOverrides(
      {
        inputPath,
        outputRoot: tempRoot,
        jobId: "fallback-learning-rate-job",
        objectiveSummary: "Classify support tickets.",
      },
      {
        specPlanner: async () => ({
          parsed: invalidSpec,
          model: "test-model",
          response_id: "resp_force_learning_rate_fallback",
        }),
      },
    );

    assert.equal(result.compiled_config.learning_rate, 0.0001);
    assert.equal(result.spec.training_params.learning_rate, 0.0001);
  });
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

test("compiler override accepts lowercase h100 and canonicalizes it to H100", async () => {
  const previousOverridePath = process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;

  try {
    await withTempDir(async (tempRoot) => {
      const overridesPath = path.join(tempRoot, "compiler-overrides.yaml");
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = overridesPath;
      await writeFile(overridesPath, "sft_gpu_type: h100\n", "utf8");

      assert.equal(getCompilerOverridesConfigPath(), overridesPath);
      assert.equal(resolveConfiguredSftGpuType(), "H100");

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
          jobId: "gpu-override-h100-job",
          objectiveSummary: "Classify support tickets.",
        },
        {
          specPlanner: async () => buildSpec("acme/support-ticket-bodies"),
        },
      );

      const compiledConfigRaw = await readFile(result.compiled_config_path, "utf8");
      assert.match(compiledConfigRaw, /gpu_type:\s*"H100"/);
    });
  } finally {
    if (previousOverridePath === undefined) {
      delete process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;
    } else {
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = previousOverridePath;
    }
  }
});

test("compiler override accepts a10g and canonicalizes it back to public A10", async () => {
  const previousOverridePath = process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;

  try {
    await withTempDir(async (tempRoot) => {
      const overridesPath = path.join(tempRoot, "compiler-overrides.yaml");
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = overridesPath;
      await writeFile(overridesPath, "sft_gpu_type: a10g\n", "utf8");

      assert.equal(getCompilerOverridesConfigPath(), overridesPath);
      assert.equal(resolveConfiguredSftGpuType(), "A10");

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
          jobId: "gpu-override-a10g-job",
          objectiveSummary: "Classify support tickets.",
        },
        {
          specPlanner: async () => ({
            ...buildSpec("acme/support-ticket-bodies"),
            compute_preset: {
              ...buildSpec("acme/support-ticket-bodies").compute_preset,
              gpu_type: "L40S",
            },
          }),
        },
      );

      const compiledConfigRaw = await readFile(result.compiled_config_path, "utf8");
      assert.match(compiledConfigRaw, /gpu_type:\s*"A10"/);

    });
  } finally {
    if (previousOverridePath === undefined) {
      delete process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;
    } else {
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = previousOverridePath;
    }
  }
});

test("compiler override can force a different SFT max_length than the planner proposed", async () => {
  const previousOverridePath = process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH;

  try {
    await withTempDir(async (tempRoot) => {
      const overridesPath = path.join(tempRoot, "compiler-overrides.yaml");
      process.env.POSTTRAINING_COMPILER_OVERRIDES_PATH = overridesPath;
      await writeFile(overridesPath, "sft_max_length: 1024\n", "utf8");

      assert.equal(getCompilerOverridesConfigPath(), overridesPath);
      assert.equal(resolveConfiguredSftMaxLength(), 1024);

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
          jobId: "max-length-override-job",
          objectiveSummary: "Classify support tickets.",
        },
        {
          specPlanner: async () => ({
            parsed: {
              ...spec,
              compute_preset: {
                ...spec.compute_preset,
                max_length: 2048,
              },
            },
            model: "test-model",
            response_id: "resp_max_length_override",
          }),
        },
      );

      assert.equal(result.compiled_config.max_length, 1024);
      assert.match(result.spec.notes.at(-1) ?? "", /Compiler override applied: max_length=1024/);
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

test("inferDeterministicNormalization recognizes prompt and soap columns as prompt-completion data", () => {
  const normalization = inferDeterministicNormalization(
    ["prompt", "soap"],
    [
      {
        prompt: "Create a Medical SOAP note summary from the dialogue.",
        soap: "S: ...\nO: ...\nA: ...\nP: ...",
      },
    ],
  );

  assert.equal(normalization.compatibility_status, "compatible");
  assert.equal(normalization.normalization_proposal.shape, "prompt_completion");
  assert.equal(normalization.normalization_proposal.fields.prompt.source_column, "prompt");
  assert.equal(normalization.normalization_proposal.fields.completion.source_column, "soap");
});

test("OpenAI structured-output schema marks every property as required", () => {
  const schema = getSpecSchema(["sft"]);
  const errors = validateStrictStructuredOutputSchema(schema);

  assert.deepEqual(errors, []);
  assert.deepEqual(
    schema.properties.selected_datasets.items.required,
    [
      "dataset",
      "dataset_config",
      "train_split",
      "eval_split",
      "source_splits",
      "weight",
      "warnings",
      "include_reason",
    ],
  );
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

    const result = await runCompilerWithNeutralOverrides(
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
    assert.deepEqual(result.selected_datasets[0].source_splits, ["train"]);
    assert.equal(result.spec.evaluation_plan.strategy, "merged_sft_holdout");
    assert.equal(result.spec.evaluation_plan.comparison_max_examples, 15);
    assert.equal(result.spec.evaluation_plan.split_style, "stratified_completion_label");
    const manifest = JSON.parse(await readFile(result.manifest_path, "utf8"));
    assert.equal(manifest.selected_datasets[0].normalization.shape, "prompt_completion");
    assert.equal(manifest.selected_datasets[0].selected_target_column, "priority");
    assert.deepEqual(manifest.selected_datasets[0].source_splits, ["train"]);
    assert.equal(manifest.evaluation_plan.strategy, "merged_sft_holdout");
    assert.equal(result.compiled_config.max_steps, -1);
  });
});

test("a raw-text generation dataset survives recommendation and compiles", async () => {
  const candidate = buildTextGenerationCandidate();
  const recommendation = await recommendDatasets(buildGenerationPlan(), {
    discoverCandidates: async () => [candidate],
    enrichCandidates: async () => [candidate],
    rankCandidates: async () => [toRecommendedCandidate(candidate)],
    skipDebugWrite: true,
  });

  assert.equal(recommendation.recommended_datasets.length, 1);
  assert.equal(recommendation.recommended_datasets[0].normalization_proposal.shape, "text");
  assert.equal(recommendation.task_spec.task_family, "generation");

  await withTempDir(async (tempRoot) => {
    const inputPath = path.join(tempRoot, "recommendation.json");
    await writeFile(inputPath, `${JSON.stringify(recommendation, null, 2)}\n`, "utf8");

    const result = await runCompilerWithNeutralOverrides(
      {
        inputPath,
        outputRoot: tempRoot,
        jobId: "generation-compatible-job",
        objectiveSummary:
          "Fine-tune a model to act as an AMC/AIME tutor with scaffolded explanations.",
      },
      {
        specPlanner: async () => ({
          parsed: buildGenerationSpec(candidate.dataset),
          model: "test-model",
          response_id: "resp_generation_direct",
        }),
      },
    );

    assert.equal(result.selected_datasets[0].normalization_shape, "text");
    assert.deepEqual(result.selected_datasets[0].source_splits, ["train"]);
    assert.equal(result.spec.task_spec.task_family, "generation");
    assert.equal(result.spec.evaluation_plan.strategy, "merged_sft_holdout");
    assert.equal(result.spec.evaluation_plan.comparison_max_examples, 15);
    assert.equal(result.spec.evaluation_plan.split_style, "deterministic_random");
    assert.equal(result.compiled_config.evaluation_plan.strategy, "merged_sft_holdout");

    const manifest = JSON.parse(await readFile(result.manifest_path, "utf8"));
    assert.equal(manifest.selected_datasets[0].normalization.shape, "text");
    assert.deepEqual(manifest.selected_datasets[0].source_splits, ["train"]);
    assert.equal(manifest.evaluation_plan.strategy, "merged_sft_holdout");
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

    const result = await runCompilerWithNeutralOverrides(
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
      runCompilerWithNeutralOverrides(
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

test("generation compilation accepts prompt-completion candidates", async () => {
  const candidate = buildPromptCompletionGenerationCandidate();
  const recommendation = {
    task_spec: buildGenerationPlan().task_spec,
    analysis: buildGenerationPlan().analysis,
    search_queries: buildGenerationPlan().search_queries,
    ranking_criteria: [],
    recommendation_guidance: buildGenerationPlan().recommendation_guidance,
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

    const result = await runCompilerWithNeutralOverrides(
      {
        inputPath,
        outputRoot: tempRoot,
        jobId: "generation-prompt-completion-job",
        objectiveSummary: "Train a contest-math tutor.",
      },
      {
        specPlanner: async () => ({
          parsed: buildGenerationSpec(candidate.dataset),
          model: "test-model",
          response_id: "resp_generation_prompt_completion",
        }),
      },
    );

    assert.equal(result.selected_datasets[0].normalization_shape, "prompt_completion");
    assert.equal(result.spec.task_spec.task_family, "generation");
    assert.equal(result.spec.task_spec.output_shape_preference, "prompt_completion");

    const manifest = JSON.parse(await readFile(result.manifest_path, "utf8"));
    assert.equal(manifest.selected_datasets[0].normalization.shape, "prompt_completion");
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

    const result = await runCompilerWithNeutralOverrides(
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
