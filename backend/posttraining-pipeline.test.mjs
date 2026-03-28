import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";

import { recommendDatasets, RecommendationFailure } from "./hf-dataset-recommender.mjs";
import { inferDeterministicNormalization } from "./posttraining-normalization.mjs";
import { runCompiler } from "./posttraining-spec-compiler.mjs";

function buildPlan() {
  return {
    analysis: {
      domain_summary: "Customer support ticket routing and issue classification.",
      mapped_task_types: ["text-classification"],
      data_format_needed: "raw_text",
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
    description: "Support tickets with message bodies.",
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
    compatibility_reason: "Direct text normalization is available via 'body'.",
    normalization_source: "deterministic",
    compatible_methods: ["sft"],
    source_schema: {
      available_columns: ["body", "issue_type"],
      sample_rows: [{ body: "My router is offline", issue_type: "technical" }],
    },
    preferred_dataset_config: "default",
    preferred_train_split: "train",
    preferred_eval_split: null,
    normalization_proposal: {
      version: 1,
      shape: "text",
      strategy: "copy_column",
      source_columns: ["body"],
      fields: {
        text: {
          source_column: "body",
          template: null,
          value_mapping: null,
        },
        prompt: null,
        completion: null,
      },
    },
  };
}

function buildStructuredCompatibleCandidate() {
  return {
    dataset: "acme/structured-ticket-metadata",
    source_url: "https://huggingface.co/datasets/acme/structured-ticket-metadata",
    score: 91,
    why: "Structured ticket metadata can be serialized into domain text.",
    matched_queries: ["customer support tickets"],
    mapped_task_types: ["text-classification"],
    downloads: 80,
    likes: 6,
    num_rows: 50000,
    license: "apache-2.0",
    splits: ["train"],
    schema_signals: ["classification_ready"],
    compatibility_status: "compatible",
    compatibility_reason: "Template synthesis can serialize the structured row into text.",
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
    normalization_proposal: {
      version: 1,
      shape: "text",
      strategy: "template_synthesis",
      source_columns: ["operator", "issue_type", "priority", "status", "channel"],
      fields: {
        text: {
          source_column: null,
          template:
            "Operator: {operator}\nChannel: {channel}\nStatus: {status}\nIssue type: {issue_type}\nPriority: {priority}",
          value_mapping: null,
        },
        prompt: null,
        completion: null,
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
      beta: 0.1,
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
  assert.equal(recommendation.recommended_datasets[0].normalization_proposal.shape, "text");

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

    assert.equal(result.selected_datasets[0].normalization_shape, "text");
    const manifest = JSON.parse(await readFile(result.manifest_path, "utf8"));
    assert.equal(manifest.selected_datasets[0].normalization.shape, "text");
    assert.ok(!("transform_preset" in manifest.selected_datasets[0]));
  });
});

test("a structured ticket dataset can compile with a synthesized text normalization recipe", async () => {
  const recommendation = {
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
    assert.equal(manifest.selected_datasets[0].normalization.shape, "text");
    assert.match(
      manifest.selected_datasets[0].normalization.fields.text.template,
      /Issue type: \{issue_type\}/,
    );
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
