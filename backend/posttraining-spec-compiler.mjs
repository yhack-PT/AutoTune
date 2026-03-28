import process from "node:process";
import path from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

import {
  inferOutputKindFromNormalization as inferOutputKindFromNormalizationShared,
  validateNormalizationProposal as validateNormalizationProposalShared,
} from "./posttraining-normalization.mjs";

const OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses";
const DEFAULT_OPENAI_MODEL = "gpt-5-mini";
const DEFAULT_INPUT_PATH = "backend/datasets.json";
const DEFAULT_OUTPUT_ROOT = "backend/generated-posttraining-jobs";
const REQUEST_TIMEOUT_MS = 10_000;
const MAX_REPAIR_ATTEMPTS = 3;
const LOG_PREFIX = "[posttraining-spec-compiler]";
const TRANSIENT_STATUS_CODES = new Set([408, 409, 429, 500, 502, 503, 504]);

const DEFAULT_BASE_MODEL = {
  model_id: "Qwen/Qwen3-8B-Base",
  revision: "7b8a267e13df1a9427e7dfa2691f69a417c58d94",
};

const DEFAULT_TRAINING_PARAMS = {
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  target_modules: [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
  ],
  beta: 0.1,
  logging_steps: 10,
  save_steps: 200,
  eval_steps: 200,
};

const ALLOWED_GPUS = ["A10", "L40S", "H100"];

let dotEnvLoaded = false;
let activeLogger = null;

function logInfo(message) {
  if (activeLogger && typeof activeLogger.emit === "function") {
    activeLogger.emit({ source: "posttraining-spec-compiler", level: "info", message });
    return;
  }
  console.log(`${LOG_PREFIX} ${message}`);
}

async function withLoggerContext(logger, callback) {
  const previousLogger = activeLogger;
  activeLogger = logger ?? previousLogger;
  try {
    return await callback();
  } finally {
    activeLogger = previousLogger;
  }
}

function normalizeOptionalString(value) {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  return null;
}

function uniqueStrings(values) {
  return [...new Set(values.filter(Boolean))];
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isNoOpValueMapping(valueMapping) {
  if (!isPlainObject(valueMapping)) {
    return false;
  }

  const entries = Object.entries(valueMapping);
  return (
    entries.length === 0 ||
    entries.every(([rawValue, mappedValue]) => String(rawValue) === String(mappedValue))
  );
}

function canonicalizeNormalizationField(fieldSpec) {
  if (!isPlainObject(fieldSpec)) {
    return fieldSpec;
  }

  const sourceColumn = normalizeOptionalString(fieldSpec.source_column);
  const template = normalizeOptionalString(fieldSpec.template);
  const shouldStripValueMapping = sourceColumn && !template && isNoOpValueMapping(fieldSpec.value_mapping);

  return {
    ...fieldSpec,
    value_mapping: shouldStripValueMapping ? null : fieldSpec.value_mapping ?? null,
  };
}

function canonicalizeNormalizationProposal(normalizationProposal) {
  if (!isPlainObject(normalizationProposal)) {
    return null;
  }

  const fields = isPlainObject(normalizationProposal.fields)
    ? { ...normalizationProposal.fields }
    : normalizationProposal.fields;

  if (isPlainObject(fields)) {
    for (const fieldName of ["text", "prompt", "completion"]) {
      if (isPlainObject(fields[fieldName])) {
        fields[fieldName] = canonicalizeNormalizationField(fields[fieldName]);
      }
    }
  }

  return {
    ...normalizationProposal,
    fields,
  };
}

function slugify(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

function truncateText(value, maxLength = 240) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 3)}...`;
}

function safeUrlLabel(url) {
  try {
    const parsed = new URL(url);
    return `${parsed.hostname}${parsed.pathname}`;
  } catch {
    return String(url);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getRetryDelayMs(attempt, response) {
  const retryAfterHeader = response?.headers?.get?.("retry-after");
  const retryAfterSeconds = Number(retryAfterHeader);
  if (Number.isFinite(retryAfterSeconds) && retryAfterSeconds >= 0) {
    return Math.min(retryAfterSeconds * 1000, 10_000);
  }
  return Math.min(1000 * 2 ** attempt, 10_000);
}

function isTransientStatusError(error) {
  return Boolean(error && typeof error === "object" && TRANSIENT_STATUS_CODES.has(error.status));
}

async function safeReadResponseText(response) {
  try {
    const text = await response.text();
    return text.slice(0, 500);
  } catch {
    return "";
  }
}

function buildHttpErrorMessage(label, url, status, responseText) {
  const prefix = `${label} failed with ${status} (${safeUrlLabel(url)})`;
  if (!responseText) {
    return prefix;
  }
  return `${prefix}: ${responseText}`;
}

async function fetchJson(url, options = {}) {
  const maxRetries = options.maxRetries ?? 0;
  const requestLabel = options.requestLabel ?? safeUrlLabel(url);

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeoutMs ?? REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(url, {
        method: options.method ?? "GET",
        headers: {
          Accept: "application/json",
          ...(options.headers ?? {}),
        },
        body: options.body,
        signal: controller.signal,
      });

      if (!response.ok) {
        const responseText = await safeReadResponseText(response);
        const error = new Error(buildHttpErrorMessage(requestLabel, url, response.status, responseText));
        error.status = response.status;

        if (attempt < maxRetries && TRANSIENT_STATUS_CODES.has(response.status)) {
          const delayMs = getRetryDelayMs(attempt, response);
          logInfo(
            `${requestLabel} failed with ${response.status}; retrying in ${delayMs}ms (${attempt + 1}/${maxRetries})`,
          );
          await sleep(delayMs);
          continue;
        }

        throw error;
      }

      return await response.json();
    } catch (error) {
      const isAbort = error instanceof Error && error.name === "AbortError";
      const isRetryableNetworkError =
        error instanceof TypeError || isAbort || isTransientStatusError(error);

      if (attempt < maxRetries && isRetryableNetworkError) {
        const delayMs = getRetryDelayMs(attempt);
        logInfo(
          `${requestLabel} failed with ${error instanceof Error ? error.message : String(error)}; retrying in ${delayMs}ms (${attempt + 1}/${maxRetries})`,
        );
        await sleep(delayMs);
        continue;
      }

      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  throw new Error(`${requestLabel} failed unexpectedly without returning a response.`);
}

async function mapWithConcurrency(items, concurrency, mapper) {
  const results = new Array(items.length);
  let nextIndex = 0;

  async function worker() {
    while (true) {
      const currentIndex = nextIndex;
      nextIndex += 1;

      if (currentIndex >= items.length) {
        return;
      }

      results[currentIndex] = await mapper(items[currentIndex], currentIndex);
    }
  }

  const workers = Array.from(
    { length: Math.max(1, Math.min(concurrency, items.length || 1)) },
    () => worker(),
  );
  await Promise.all(workers);
  return results;
}

async function ensureDotEnvLoaded() {
  if (dotEnvLoaded) {
    return;
  }
  dotEnvLoaded = true;

  if (process.env.OPENAI_API_KEY) {
    return;
  }

  try {
    const envContents = await readFile(new URL("../.env", import.meta.url), "utf8");
    for (const rawLine of envContents.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) {
        continue;
      }

      const separatorIndex = line.indexOf("=");
      if (separatorIndex <= 0) {
        continue;
      }

      const key = line.slice(0, separatorIndex).trim();
      if (!key || process.env[key]) {
        continue;
      }

      let value = line.slice(separatorIndex + 1).trim();
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }

      process.env[key] = value;
    }
  } catch {
    // Missing .env is fine if the environment is already configured.
  }
}

function extractOpenAIOutputText(response) {
  if (!response || !Array.isArray(response.output)) {
    return "";
  }

  for (const outputItem of response.output) {
    if (!Array.isArray(outputItem?.content)) {
      continue;
    }
    for (const contentItem of outputItem.content) {
      if (contentItem?.type === "output_text" && typeof contentItem.text === "string") {
        return contentItem.text;
      }
    }
  }

  if (typeof response.output_text === "string") {
    return response.output_text;
  }

  return "";
}

function parseCliArgs(argv) {
  const parsed = {
    inputPath: DEFAULT_INPUT_PATH,
    outputRoot: DEFAULT_OUTPUT_ROOT,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index];
    const next = argv[index + 1];

    if (current === "--input") {
      parsed.inputPath = next;
      index += 1;
    } else if (current === "--context") {
      parsed.contextPath = next;
      index += 1;
    } else if (current === "--objective-summary") {
      parsed.objectiveSummary = next;
      index += 1;
    } else if (current === "--seed-artifact") {
      parsed.seedArtifact = next;
      index += 1;
    } else if (current === "--output-dir") {
      parsed.outputRoot = next;
      index += 1;
    } else if (current === "--job-id") {
      parsed.jobId = next;
      index += 1;
    } else if (current === "--enable-wandb") {
      parsed.enableWandb = true;
    }
  }

  return parsed;
}

async function loadJsonFile(filePath) {
  const contents = await readFile(filePath, "utf8");
  return JSON.parse(contents);
}

function normalizeCandidate(candidate) {
  if (!candidate || typeof candidate !== "object") {
    return null;
  }

  const dataset = normalizeOptionalString(candidate.dataset ?? candidate.id);
  if (!dataset) {
    return null;
  }

  return {
    dataset,
    source_url: normalizeOptionalString(candidate.source_url) ?? `https://huggingface.co/datasets/${dataset}`,
    score: Number(candidate.score ?? 0),
    why: normalizeOptionalString(candidate.why) ?? "",
    matched_queries: Array.isArray(candidate.matched_queries)
      ? uniqueStrings(candidate.matched_queries.map((value) => String(value).trim()))
      : [],
    mapped_task_types: Array.isArray(candidate.mapped_task_types)
      ? uniqueStrings(candidate.mapped_task_types.map((value) => String(value).trim()))
      : [],
    downloads: Number(candidate.downloads ?? 0),
    likes: Number(candidate.likes ?? 0),
    num_rows: Number(candidate.num_rows ?? 0),
    license: normalizeOptionalString(candidate.license),
    splits: Array.isArray(candidate.splits)
      ? uniqueStrings(candidate.splits.map((value) => String(value).trim()))
      : [],
    schema_signals: Array.isArray(candidate.schema_signals)
      ? uniqueStrings(candidate.schema_signals.map((value) => String(value).trim()))
      : [],
    compatibility_status: normalizeOptionalString(candidate.compatibility_status) ?? "unknown",
    compatibility_reason: normalizeOptionalString(candidate.compatibility_reason) ?? "",
    source_schema:
      candidate.source_schema && typeof candidate.source_schema === "object"
        ? {
            available_columns: Array.isArray(candidate.source_schema.available_columns)
              ? uniqueStrings(candidate.source_schema.available_columns.map((value) => String(value).trim()))
              : [],
            sample_rows: Array.isArray(candidate.source_schema.sample_rows)
              ? candidate.source_schema.sample_rows
              : [],
          }
        : { available_columns: [], sample_rows: [] },
    preferred_dataset_config: normalizeOptionalString(candidate.preferred_dataset_config),
    preferred_train_split: normalizeOptionalString(candidate.preferred_train_split) ?? "train",
    preferred_eval_split: normalizeOptionalString(candidate.preferred_eval_split),
    normalization_source: normalizeOptionalString(candidate.normalization_source),
    normalization_proposal:
      candidate.normalization_proposal && typeof candidate.normalization_proposal === "object"
        ? canonicalizeNormalizationProposal(candidate.normalization_proposal)
        : null,
    compatible_methods: Array.isArray(candidate.compatible_methods)
      ? uniqueStrings(candidate.compatible_methods.map((value) => String(value).trim()))
      : [],
    warnings: Array.isArray(candidate.warnings)
      ? uniqueStrings(candidate.warnings.map((value) => String(value).trim()))
      : [],
  };
}

function normalizeCompilerInput(rawInput, contextOverride = null) {
  const baseContext =
    rawInput && typeof rawInput === "object" && !Array.isArray(rawInput)
      ? {
          analysis:
            rawInput.analysis && typeof rawInput.analysis === "object" ? rawInput.analysis : {},
          recommendation_guidance:
            rawInput.recommendation_guidance && typeof rawInput.recommendation_guidance === "object"
              ? rawInput.recommendation_guidance
              : {},
          search_queries: Array.isArray(rawInput.search_queries) ? rawInput.search_queries : [],
          ranking_criteria: Array.isArray(rawInput.ranking_criteria) ? rawInput.ranking_criteria : [],
        }
      : {
          analysis: {},
          recommendation_guidance: {},
          search_queries: [],
          ranking_criteria: [],
        };

  const candidates = Array.isArray(rawInput)
    ? rawInput
    : Array.isArray(rawInput?.recommended_datasets)
      ? rawInput.recommended_datasets
      : Array.isArray(rawInput?.ranked_datasets)
        ? rawInput.ranked_datasets
        : Array.isArray(rawInput?.datasets)
          ? rawInput.datasets
          : [];

  if (!candidates.length) {
    throw new Error(
      "Compiler input must be either an array of ranked dataset objects or an object containing recommended_datasets / ranked_datasets.",
    );
  }

  const normalizedCandidates = candidates.map((candidate) => normalizeCandidate(candidate)).filter(Boolean);
  if (!normalizedCandidates.length) {
    throw new Error("No usable dataset candidates were found in the compiler input.");
  }

  return {
    context: {
      ...baseContext,
      ...(contextOverride && typeof contextOverride === "object" ? contextOverride : {}),
    },
    candidates: normalizedCandidates,
  };
}

function buildObjectiveSummary(context, args, candidates) {
  const explicit = normalizeOptionalString(args.objectiveSummary);
  if (explicit) {
    return explicit;
  }

  const domainSummary = normalizeOptionalString(context?.analysis?.domain_summary);
  if (domainSummary) {
    return domainSummary;
  }

  const searchQuery = Array.isArray(context?.search_queries) ? context.search_queries[0] : null;
  const searchText = normalizeOptionalString(searchQuery?.search);
  if (searchText) {
    return `Create a post-trained model for ${searchText}.`;
  }

  const topCandidate = candidates[0];
  const matchedQuery = topCandidate?.matched_queries?.[0];
  if (matchedQuery) {
    return `Create a post-trained model using datasets related to ${matchedQuery}.`;
  }

  return "Create a useful text-only post-trained model from the selected Hugging Face datasets.";
}

function buildPlatformConstraints(seedArtifact) {
  return {
    text_only: true,
    supported_methods: ["sft"],
    default_adaptation_strategy: "lora",
    default_artifact_strategy: "adapter",
    allowed_normalization_shapes: ["text", "prompt_completion"],
    allowed_compute_gpus: ALLOWED_GPUS,
    allowed_base_models: [DEFAULT_BASE_MODEL],
    seed_artifact_available: Boolean(normalizeOptionalString(seedArtifact)),
    seed_artifact: normalizeOptionalString(seedArtifact),
    trainer_notes: [
      "This normalization redesign currently supports SFT only.",
      "All selected datasets must normalize to the same final SFT shape: either text or prompt-completion.",
      "LoRA is the default adaptation strategy and should usually remain unchanged.",
    ],
  };
}

function buildCandidatePromptPayload(candidateProfiles) {
  return candidateProfiles.map((candidate) => ({
    dataset: candidate.dataset,
    source_url: candidate.source_url,
    score: candidate.score,
    why: candidate.why,
    matched_queries: candidate.matched_queries,
    mapped_task_types: candidate.mapped_task_types,
    num_rows: candidate.num_rows,
    license: candidate.license,
    splits: candidate.splits,
    warnings: candidate.warnings,
    preferred_dataset_config: candidate.preferred_dataset_config,
    preferred_train_split: candidate.preferred_train_split,
    preferred_eval_split: candidate.preferred_eval_split,
    source_schema: candidate.source_schema,
    normalization_proposal: candidate.normalization_proposal,
    compatibility_status: candidate.compatibility_status,
    compatibility_reason: candidate.compatibility_reason,
    normalization_source: candidate.normalization_source,
    compatible_methods: candidate.compatible_methods,
  }));
}

function buildSpecPrompt({
  objectiveSummary,
  context,
  candidateProfiles,
  platformConstraints,
  jobId,
  previousSpec = null,
  validationErrors = [],
}) {
  const lines = [
    "Produce a compact PostTrainingJobSpec for text-only LLM post-training.",
    "You must select the training method and the dataset mix.",
    "Use only the provided dataset candidates. Do not invent dataset ids, columns, or normalization recipes.",
    "Each candidate already includes a canonical normalization proposal and its compatible methods.",
    "Method and LoRA are different axes: the method is the trainer choice, while LoRA is the default adaptation strategy.",
    "Choose 1-3 datasets only when mixing materially improves coverage or robustness.",
    "All selected datasets must be executable by the current trainer backend.",
    "Do not mix text-style outputs with prompt-completion-style outputs in the same job.",
    "The current normalization redesign supports SFT-compatible datasets only.",
    `Job id: ${jobId}`,
    `Objective summary: ${objectiveSummary}`,
    `Recommender analysis: ${JSON.stringify(context.analysis ?? {})}`,
    `Recommendation guidance: ${JSON.stringify(context.recommendation_guidance ?? {})}`,
    `Search queries: ${JSON.stringify(context.search_queries ?? [])}`,
    `Platform constraints: ${JSON.stringify(platformConstraints)}`,
    `Dataset candidates: ${JSON.stringify(buildCandidatePromptPayload(candidateProfiles))}`,
  ];

  if (previousSpec) {
    lines.push(`Previous invalid spec: ${JSON.stringify(previousSpec)}`);
  }
  if (validationErrors.length > 0) {
    lines.push(`Validation errors to repair: ${JSON.stringify(validationErrors)}`);
    lines.push("Repair only the invalid parts and keep the rest coherent.");
  }

  return lines.join("\n");
}

function getSpecSchema(supportedMethods = ["sft"]) {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      objective_summary: { type: "string" },
      method: { type: "string", enum: supportedMethods },
      adaptation_strategy: { type: "string", enum: ["lora"] },
      artifact_strategy: { type: "string", enum: ["adapter", "merged"] },
      base_model: {
        type: "object",
        additionalProperties: false,
        properties: {
          model_id: { type: "string" },
          revision: { type: ["string", "null"] },
        },
        required: ["model_id", "revision"],
      },
      selected_datasets: {
        type: "array",
        minItems: 1,
        maxItems: 3,
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            dataset: { type: "string" },
            dataset_config: { type: ["string", "null"] },
            train_split: { type: "string" },
            eval_split: { type: ["string", "null"] },
            weight: { type: "number", exclusiveMinimum: 0 },
            warnings: {
              type: "array",
              items: { type: "string" },
            },
            include_reason: { type: "string" },
          },
          required: [
            "dataset",
            "dataset_config",
            "train_split",
            "eval_split",
            "weight",
            "warnings",
            "include_reason",
          ],
        },
      },
      compute_preset: {
        type: "object",
        additionalProperties: false,
        properties: {
          gpu_type: { type: "string", enum: ALLOWED_GPUS },
          max_length: { type: "integer", minimum: 256 },
          per_device_train_batch_size: { type: "integer", minimum: 1 },
          per_device_eval_batch_size: { type: "integer", minimum: 1 },
          gradient_accumulation_steps: { type: "integer", minimum: 1 },
        },
        required: [
          "gpu_type",
          "max_length",
          "per_device_train_batch_size",
          "per_device_eval_batch_size",
          "gradient_accumulation_steps",
        ],
      },
      training_params: {
        type: "object",
        additionalProperties: false,
        properties: {
          learning_rate: { type: ["number", "null"], exclusiveMinimum: 0 },
          num_train_epochs: { type: "number", exclusiveMinimum: 0 },
          max_steps: { type: "integer", minimum: -1 },
          beta: { type: "number", minimum: 0 },
          lora_r: { type: "integer", minimum: 1 },
          lora_alpha: { type: "integer", minimum: 1 },
          lora_dropout: { type: "number", minimum: 0, maximum: 1 },
          target_modules: {
            type: "array",
            minItems: 1,
            items: { type: "string" },
          },
          logging_steps: { type: "integer", minimum: 1 },
          save_steps: { type: "integer", minimum: 1 },
          eval_steps: { type: "integer", minimum: 1 },
        },
        required: [
          "learning_rate",
          "num_train_epochs",
          "max_steps",
          "beta",
          "lora_r",
          "lora_alpha",
          "lora_dropout",
          "target_modules",
          "logging_steps",
          "save_steps",
          "eval_steps",
        ],
      },
      seed_artifact: { type: ["string", "null"] },
      notes: {
        type: "array",
        items: { type: "string" },
      },
    },
    required: [
      "objective_summary",
      "method",
      "adaptation_strategy",
      "artifact_strategy",
      "base_model",
      "selected_datasets",
      "compute_preset",
      "training_params",
      "seed_artifact",
      "notes",
    ],
  };
}

async function callOpenAISpecPlanner(prompt, specSchema) {
  await ensureDotEnvLoaded();

  const apiKey = normalizeOptionalString(process.env.OPENAI_API_KEY);
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required to generate a post-training job spec.");
  }

  const model = normalizeOptionalString(process.env.OPENAI_MODEL) ?? DEFAULT_OPENAI_MODEL;
  logInfo(`calling OpenAI spec planner with model ${model}`);

  const response = await fetchJson(OPENAI_RESPONSES_API_URL, {
    method: "POST",
    timeoutMs: 60_000,
    maxRetries: 3,
    requestLabel: "OpenAI post-training spec request",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      store: false,
      input: [
        {
          role: "system",
          content:
            "You plan post-training jobs for a text-only TRL + Modal backend. Return only the requested JSON structure and use only the provided datasets and normalization proposals.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "post_training_job_spec",
          schema: specSchema,
          strict: true,
        },
      },
    }),
  });

  const outputText = extractOpenAIOutputText(response);
  if (!outputText) {
    throw new Error("OpenAI returned an empty PostTrainingJobSpec response.");
  }

  try {
    return {
      parsed: JSON.parse(outputText),
      model: response.model ?? model,
      response_id: response.id ?? null,
    };
  } catch (error) {
    throw new Error(
      `OpenAI returned invalid JSON for PostTrainingJobSpec: ${
        error instanceof Error ? error.message : "unknown parse error"
      }`,
    );
  }
}

function normalizeWeights(selectedDatasets) {
  const total = selectedDatasets.reduce((sum, dataset) => sum + Number(dataset.weight ?? 0), 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new Error("Selected dataset weights must sum to a positive number.");
  }
  return selectedDatasets.map((dataset) => ({
    ...dataset,
    weight: Number((Number(dataset.weight) / total).toFixed(6)),
  }));
}

function validateAndFinalizeSpec(rawSpec, candidateProfiles, platformConstraints, seedArtifact, jobId) {
  const errors = [];
  const candidateMap = new Map(candidateProfiles.map((candidate) => [candidate.dataset, candidate]));
  const modelIds = new Set(platformConstraints.allowed_base_models.map((model) => model.model_id));
  const spec = {
    ...rawSpec,
    job_id: jobId,
    adaptation_strategy: normalizeOptionalString(rawSpec?.adaptation_strategy) ?? "lora",
    artifact_strategy: normalizeOptionalString(rawSpec?.artifact_strategy) ?? "adapter",
    seed_artifact: normalizeOptionalString(rawSpec?.seed_artifact) ?? normalizeOptionalString(seedArtifact),
  };

  if (!platformConstraints.supported_methods.includes(spec.method)) {
    errors.push(`method must be one of ${platformConstraints.supported_methods.join(", ")}.`);
  }
  if (spec.adaptation_strategy !== "lora") {
    errors.push("adaptation_strategy must be 'lora' for the current backend.");
  }
  if (!["adapter", "merged"].includes(spec.artifact_strategy)) {
    errors.push("artifact_strategy must be either 'adapter' or 'merged'.");
  }

  if (!spec.base_model || !modelIds.has(spec.base_model.model_id)) {
    errors.push(
      `base_model.model_id must be one of ${JSON.stringify(platformConstraints.allowed_base_models)}.`,
    );
  }

  const compute = spec.compute_preset ?? {};
  if (!ALLOWED_GPUS.includes(compute.gpu_type)) {
    errors.push(`compute_preset.gpu_type must be one of ${ALLOWED_GPUS.join(", ")}.`);
  }
  for (const fieldName of [
    "max_length",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
  ]) {
    if (!Number.isInteger(compute[fieldName]) || compute[fieldName] <= 0) {
      errors.push(`compute_preset.${fieldName} must be a positive integer.`);
    }
  }

  const params = spec.training_params ?? {};
  if (
    !(
      params.learning_rate === null ||
      (typeof params.learning_rate === "number" && params.learning_rate > 0)
    )
  ) {
    errors.push("training_params.learning_rate must be null or a positive number.");
  }
  if (!(typeof params.num_train_epochs === "number" && params.num_train_epochs > 0)) {
    errors.push("training_params.num_train_epochs must be a positive number.");
  }
  if (!Number.isInteger(params.max_steps) || params.max_steps < -1) {
    errors.push("training_params.max_steps must be an integer >= -1.");
  }
  if (!(typeof params.lora_dropout === "number" && params.lora_dropout >= 0 && params.lora_dropout <= 1)) {
    errors.push("training_params.lora_dropout must be between 0 and 1.");
  }
  for (const fieldName of [
    "beta",
    "lora_r",
    "lora_alpha",
    "logging_steps",
    "save_steps",
    "eval_steps",
  ]) {
    if (!(typeof params[fieldName] === "number" && params[fieldName] > 0)) {
      errors.push(`training_params.${fieldName} must be a positive number.`);
    }
  }
  if (!Array.isArray(params.target_modules) || params.target_modules.length === 0) {
    errors.push("training_params.target_modules must be a non-empty array.");
  }

  if (!Array.isArray(spec.selected_datasets) || spec.selected_datasets.length === 0) {
    errors.push("selected_datasets must be a non-empty array.");
  }

  const outputKinds = new Set();
  const selectedDatasets = [];

  for (const selectedDataset of spec.selected_datasets ?? []) {
    const datasetId = normalizeOptionalString(selectedDataset?.dataset);
    if (!datasetId) {
      errors.push("Each selected dataset must include a dataset id.");
      continue;
    }

    const candidate = candidateMap.get(datasetId);
    if (!candidate) {
      errors.push(`Selected dataset '${datasetId}' was not present in the enriched candidate set.`);
      continue;
    }

    if (!candidate.normalization_proposal) {
      errors.push(`${datasetId}: candidate is missing a normalization proposal.`);
      continue;
    }

    const normalizationErrors = validateNormalizationProposalShared(
      candidate.normalization_proposal,
      candidate.source_schema?.available_columns ?? [],
    );
    if (normalizationErrors.length > 0) {
      errors.push(...normalizationErrors.map((error) => `${datasetId}: ${error}`));
    }

    if (!candidate.compatible_methods.includes(spec.method)) {
      errors.push(`${datasetId}: normalization proposal is incompatible with method '${spec.method}'.`);
    }

    const rawWeight = Number(selectedDataset.weight);
    if (!Number.isFinite(rawWeight) || rawWeight <= 0) {
      errors.push(`${datasetId}: weight must be a positive number.`);
    }

    const outputKind = inferOutputKindFromNormalizationShared(spec.method, candidate.normalization_proposal);
    if (outputKind === "unsupported") {
      errors.push(`${datasetId}: normalization proposal does not support method '${spec.method}'.`);
    }
    outputKinds.add(outputKind);

    selectedDatasets.push({
      dataset: datasetId,
      dataset_config: normalizeOptionalString(selectedDataset.dataset_config) ?? candidate.preferred_dataset_config,
      train_split:
        normalizeOptionalString(selectedDataset.train_split) ?? candidate.preferred_train_split ?? "train",
      eval_split:
        normalizeOptionalString(selectedDataset.eval_split) ?? candidate.preferred_eval_split ?? null,
      weight: rawWeight,
      source_schema: candidate.source_schema,
      normalization: candidate.normalization_proposal,
      compatible_methods: candidate.compatible_methods,
      warnings: uniqueStrings([
        ...(candidate.warnings ?? []),
        ...(Array.isArray(selectedDataset.warnings) ? selectedDataset.warnings.map((value) => String(value)) : []),
      ]),
      include_reason: normalizeOptionalString(selectedDataset.include_reason) ?? candidate.why,
    });
  }

  if (outputKinds.size > 1) {
    errors.push(
      `All selected datasets must normalize to the same output shape for a mixed run. Found: ${JSON.stringify(
        [...outputKinds],
      )}.`,
    );
  }

  if (errors.length > 0) {
    return { ok: false, errors };
  }

  const finalized = {
    job_id: jobId,
    objective_summary: normalizeOptionalString(spec.objective_summary) ?? "",
    method: spec.method,
    adaptation_strategy: "lora",
    artifact_strategy: spec.artifact_strategy,
    base_model: {
      model_id: spec.base_model.model_id,
      revision: normalizeOptionalString(spec.base_model.revision) ?? DEFAULT_BASE_MODEL.revision,
    },
    selected_datasets: normalizeWeights(selectedDatasets),
    compute_preset: {
      gpu_type: compute.gpu_type,
      max_length: compute.max_length,
      per_device_train_batch_size: compute.per_device_train_batch_size,
      per_device_eval_batch_size: compute.per_device_eval_batch_size,
      gradient_accumulation_steps: compute.gradient_accumulation_steps,
    },
    training_params: {
      learning_rate: params.learning_rate,
      num_train_epochs: params.num_train_epochs,
      max_steps: params.max_steps,
      beta: params.beta,
      lora_r: params.lora_r,
      lora_alpha: params.lora_alpha,
      lora_dropout: params.lora_dropout,
      target_modules: params.target_modules,
      logging_steps: params.logging_steps,
      save_steps: params.save_steps,
      eval_steps: params.eval_steps,
    },
    seed_artifact: spec.seed_artifact,
    notes: Array.isArray(spec.notes) ? spec.notes.map((note) => String(note)) : [],
  };

  return { ok: true, spec: finalized };
}

function buildPreparedDatasetManifest(spec) {
  return {
    format_version: 1,
    job_id: spec.job_id,
    objective_summary: spec.objective_summary,
    trainer_type: spec.method,
    adaptation_strategy: spec.adaptation_strategy,
    artifact_strategy: spec.artifact_strategy,
    selected_datasets: spec.selected_datasets.map((dataset) => ({
      dataset: dataset.dataset,
      dataset_config: dataset.dataset_config,
      train_split: dataset.train_split,
      eval_split: dataset.eval_split,
      weight: dataset.weight,
      source_schema: dataset.source_schema,
      normalization: dataset.normalization,
      compatible_methods: dataset.compatible_methods,
      warnings: dataset.warnings,
      include_reason: dataset.include_reason,
    })),
    notes: spec.notes,
  };
}

function compileTrainingConfig(spec, preparedDatasetManifest, enableWandb) {
  return {
    trainer_type: spec.method,
    base_model: spec.base_model.model_id,
    base_model_revision: spec.base_model.revision,
    dataset_name: "prepared_manifest",
    dataset_config: null,
    dataset_source_type: "prepared_manifest",
    prepared_dataset_manifest: preparedDatasetManifest,
    train_split: "train",
    eval_split: null,
    output_name: spec.job_id,
    seed_artifact: spec.seed_artifact,
    gpu_type: spec.compute_preset.gpu_type,
    max_length: spec.compute_preset.max_length,
    per_device_train_batch_size: spec.compute_preset.per_device_train_batch_size,
    per_device_eval_batch_size: spec.compute_preset.per_device_eval_batch_size,
    gradient_accumulation_steps: spec.compute_preset.gradient_accumulation_steps,
    num_train_epochs: spec.training_params.num_train_epochs,
    max_steps: spec.training_params.max_steps,
    use_peft: true,
    load_in_4bit: true,
    lora_r: spec.training_params.lora_r,
    lora_alpha: spec.training_params.lora_alpha,
    lora_dropout: spec.training_params.lora_dropout,
    target_modules: spec.training_params.target_modules,
    learning_rate: spec.training_params.learning_rate,
    beta: spec.training_params.beta,
    logging_steps: spec.training_params.logging_steps,
    save_steps: spec.training_params.save_steps,
    eval_steps: spec.training_params.eval_steps,
    merge_after_train: spec.artifact_strategy === "merged",
    enable_wandb: Boolean(enableWandb),
  };
}

function isScalar(value) {
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

function yamlScalar(value) {
  if (value === null) {
    return "null";
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  return String(value);
}

function toYaml(value, indent = 0) {
  const pad = "  ".repeat(indent);

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return `${pad}[]`;
    }
    return value
      .map((item) => {
        if (isScalar(item)) {
          return `${pad}- ${yamlScalar(item)}`;
        }
        if (Array.isArray(item)) {
          return `${pad}-\n${toYaml(item, indent + 1)}`;
        }
        const entries = Object.entries(item);
        if (entries.length === 0) {
          return `${pad}- {}`;
        }
        const [firstKey, firstValue] = entries[0];
        const lines = [];
        if (isScalar(firstValue)) {
          lines.push(`${pad}- ${firstKey}: ${yamlScalar(firstValue)}`);
        } else {
          lines.push(`${pad}- ${firstKey}:`);
          lines.push(toYaml(firstValue, indent + 2));
        }
        for (const [key, nestedValue] of entries.slice(1)) {
          if (isScalar(nestedValue)) {
            lines.push(`${"  ".repeat(indent + 1)}${key}: ${yamlScalar(nestedValue)}`);
          } else {
            lines.push(`${"  ".repeat(indent + 1)}${key}:`);
            lines.push(toYaml(nestedValue, indent + 2));
          }
        }
        return lines.join("\n");
      })
      .join("\n");
  }

  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    if (entries.length === 0) {
      return `${pad}{}`;
    }
    return entries
      .map(([key, nestedValue]) => {
        if (isScalar(nestedValue)) {
          return `${pad}${key}: ${yamlScalar(nestedValue)}`;
        }
        return `${pad}${key}:\n${toYaml(nestedValue, indent + 1)}`;
      })
      .join("\n");
  }

  return `${pad}${yamlScalar(value)}`;
}

function buildDefaultTrainingParams(method) {
  return {
    learning_rate:
      method === "sft" ? 1e-4 : method === "bco" ? 5e-7 : 1e-6,
    num_train_epochs: 1.0,
    max_steps: 200,
    beta: DEFAULT_TRAINING_PARAMS.beta,
    lora_r: DEFAULT_TRAINING_PARAMS.lora_r,
    lora_alpha: DEFAULT_TRAINING_PARAMS.lora_alpha,
    lora_dropout: DEFAULT_TRAINING_PARAMS.lora_dropout,
    target_modules: DEFAULT_TRAINING_PARAMS.target_modules,
    logging_steps: DEFAULT_TRAINING_PARAMS.logging_steps,
    save_steps: DEFAULT_TRAINING_PARAMS.save_steps,
    eval_steps: DEFAULT_TRAINING_PARAMS.eval_steps,
  };
}

function buildFallbackSpec(objectiveSummary, jobId, candidateProfiles, seedArtifact) {
  const firstCandidate = candidateProfiles.find((candidate) => candidate.compatible_methods.includes("sft"));
  if (!firstCandidate || !firstCandidate.normalization_proposal) {
    throw new Error("Unable to build even a fallback SFT spec because no compatible SFT dataset candidate was found.");
  }

  return {
    job_id: jobId,
    objective_summary: objectiveSummary,
    method: "sft",
    adaptation_strategy: "lora",
    artifact_strategy: "adapter",
    base_model: DEFAULT_BASE_MODEL,
    selected_datasets: [
      {
        dataset: firstCandidate.dataset,
        dataset_config: firstCandidate.preferred_dataset_config,
        train_split: firstCandidate.preferred_train_split,
        eval_split: firstCandidate.preferred_eval_split,
        weight: 1,
        source_schema: firstCandidate.source_schema,
        normalization: firstCandidate.normalization_proposal,
        compatible_methods: firstCandidate.compatible_methods,
        warnings: firstCandidate.warnings,
        include_reason: firstCandidate.why || "Fallback compatible dataset selection.",
      },
    ],
    compute_preset: {
      gpu_type: "A10",
      max_length: 2048,
      per_device_train_batch_size: 1,
      per_device_eval_batch_size: 1,
      gradient_accumulation_steps: 16,
    },
    training_params: buildDefaultTrainingParams("sft"),
    seed_artifact: normalizeOptionalString(seedArtifact),
    notes: [
      "Fallback spec generated locally because no valid LLM-generated spec could be repaired.",
    ],
  };
}

async function generateValidatedSpec({
  objectiveSummary,
  context,
  candidateProfiles,
  platformConstraints,
  seedArtifact,
  jobId,
  specPlanner,
}) {
  const attempts = [];
  let previousSpec = null;
  let validationErrors = [];
  const specSchema = getSpecSchema(platformConstraints.supported_methods);

  for (let attempt = 1; attempt <= MAX_REPAIR_ATTEMPTS; attempt += 1) {
    const prompt = buildSpecPrompt({
      objectiveSummary,
      context,
      candidateProfiles,
      platformConstraints,
      jobId,
      previousSpec,
      validationErrors,
    });

    const openAIResult = await (
      typeof specPlanner === "function"
        ? specPlanner({ prompt, specSchema, attempt, previousSpec, validationErrors })
        : callOpenAISpecPlanner(prompt, specSchema)
    );
    const validation = validateAndFinalizeSpec(
      openAIResult.parsed,
      candidateProfiles,
      platformConstraints,
      seedArtifact,
      jobId,
    );

    attempts.push({
      attempt,
      response_id: openAIResult.response_id,
      model: openAIResult.model,
      spec: openAIResult.parsed,
      validation_errors: validation.ok ? [] : validation.errors,
    });

    if (validation.ok) {
      return {
        spec: validation.spec,
        attempts,
      };
    }

    previousSpec = openAIResult.parsed;
    validationErrors = validation.errors;
  }

  logInfo("LLM spec generation could not be repaired after retries; falling back to a deterministic SFT plan.");
  const fallbackSpec = buildFallbackSpec(objectiveSummary, jobId, candidateProfiles, seedArtifact);
  const fallbackValidation = validateAndFinalizeSpec(
    fallbackSpec,
    candidateProfiles,
    platformConstraints,
    seedArtifact,
    jobId,
  );
  if (!fallbackValidation.ok) {
    throw new Error(
      `Fallback spec generation failed validation: ${JSON.stringify(fallbackValidation.errors, null, 2)}`,
    );
  }

  attempts.push({
    attempt: "fallback",
    response_id: null,
    model: null,
    spec: fallbackSpec,
    validation_errors: [],
  });

  return {
    spec: fallbackValidation.spec,
    attempts,
  };
}

async function writeArtifacts(outputDir, artifacts) {
  await mkdir(outputDir, { recursive: true });

  const writes = [
    writeFile(
      path.join(outputDir, "post_training_job_spec.yaml"),
      `${toYaml(artifacts.spec)}\n`,
      "utf8",
    ),
    writeFile(
      path.join(outputDir, "compiled_train_config.yaml"),
      `${toYaml(artifacts.compiledConfig)}\n`,
      "utf8",
    ),
    writeFile(
      path.join(outputDir, "prepared_dataset_manifest.json"),
      `${JSON.stringify(artifacts.preparedDatasetManifest, null, 2)}\n`,
      "utf8",
    ),
    writeFile(
      path.join(outputDir, "compiler_trace.json"),
      `${JSON.stringify(artifacts.trace, null, 2)}\n`,
      "utf8",
    ),
  ];

  await Promise.all(writes);
}

export async function runCompiler(args, options = {}) {
  return withLoggerContext(options.logger, async () => {
    const rawInput = await loadJsonFile(args.inputPath);
    const contextOverride = args.contextPath ? await loadJsonFile(args.contextPath) : null;
    const normalizedInput = normalizeCompilerInput(rawInput, contextOverride);
    const sortedCandidates = [...normalizedInput.candidates].sort((left, right) => right.score - left.score);
    const candidateProfiles = sortedCandidates.slice(0, 8);

    logInfo(`loaded ${candidateProfiles.length} ranked dataset candidates from recommendation output`);
    const candidateDiagnostics = candidateProfiles.map((candidate) => {
      const validationErrors = candidate.normalization_proposal
        ? validateNormalizationProposalShared(
            candidate.normalization_proposal,
            candidate.source_schema?.available_columns ?? [],
          )
        : ["candidate is missing a normalization proposal."];
      return {
        candidate,
        validationErrors,
      };
    });
    const usableCandidates = candidateDiagnostics
      .filter(
        ({ candidate, validationErrors }) =>
          validationErrors.length === 0 && candidate.compatible_methods.includes("sft"),
      )
      .map(({ candidate }) => candidate);
    if (!usableCandidates.length) {
      const diagnosticMessage = candidateDiagnostics
        .map(({ candidate, validationErrors }) =>
          `${candidate.dataset}: ${validationErrors.join(" ") || "not compatible with sft."}`,
        )
        .join(" ; ");
      throw new Error(
        `None of the recommended dataset candidates exposed a valid normalization proposal for the current backend. ${diagnosticMessage}`,
      );
    }

    const objectiveSummary = buildObjectiveSummary(normalizedInput.context, args, usableCandidates);
    const timestamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
    const explicitJobId = normalizeOptionalString(args.jobId)?.replace(/[\\/]+/g, "-");
    const generatedJobId = explicitJobId
      ? explicitJobId
      : `${slugify(objectiveSummary || usableCandidates[0].dataset || "job")}-${timestamp}`;
    const platformConstraints = buildPlatformConstraints(args.seedArtifact);

    const { spec, attempts } = await generateValidatedSpec({
      objectiveSummary,
      context: normalizedInput.context,
      candidateProfiles: usableCandidates,
      platformConstraints,
      seedArtifact: args.seedArtifact,
      jobId: generatedJobId,
      specPlanner: options.specPlanner,
    });

    const preparedDatasetManifest = buildPreparedDatasetManifest(spec);
    const compiledConfig = compileTrainingConfig(spec, preparedDatasetManifest, args.enableWandb);

    const outputDir = path.resolve(args.outputRoot, spec.job_id);
    const trace = {
      created_at: new Date().toISOString(),
      objective_summary: objectiveSummary,
      input_path: path.resolve(args.inputPath),
      context_path: args.contextPath ? path.resolve(args.contextPath) : null,
      usable_candidates: usableCandidates.map((candidate) => ({
        dataset: candidate.dataset,
        compatibility_status: candidate.compatibility_status,
        compatibility_reason: candidate.compatibility_reason,
        normalization_source: candidate.normalization_source,
        compatible_methods: candidate.compatible_methods,
        normalization_proposal: candidate.normalization_proposal,
        source_schema: candidate.source_schema,
        preferred_dataset_config: candidate.preferred_dataset_config,
        preferred_train_split: candidate.preferred_train_split,
        preferred_eval_split: candidate.preferred_eval_split,
        warnings: candidate.warnings,
      })),
      attempts,
      output_files: {
        spec_yaml: path.join(outputDir, "post_training_job_spec.yaml"),
        compiled_yaml: path.join(outputDir, "compiled_train_config.yaml"),
        manifest_json: path.join(outputDir, "prepared_dataset_manifest.json"),
        trace_json: path.join(outputDir, "compiler_trace.json"),
      },
    };

    await writeArtifacts(outputDir, {
      spec,
      compiledConfig,
      preparedDatasetManifest,
      trace,
    });

    return {
      job_id: spec.job_id,
      objective_summary: spec.objective_summary,
      output_dir: outputDir,
      method: spec.method,
      compiled_config: compiledConfig,
      spec,
      selected_datasets: spec.selected_datasets.map((dataset) => ({
        dataset: dataset.dataset,
        weight: dataset.weight,
        normalization_shape: dataset.normalization?.shape ?? null,
      })),
      compiled_config_path: path.join(outputDir, "compiled_train_config.yaml"),
      spec_path: path.join(outputDir, "post_training_job_spec.yaml"),
      manifest_path: path.join(outputDir, "prepared_dataset_manifest.json"),
      trace_path: path.join(outputDir, "compiler_trace.json"),
    };
  });
}

async function runCli() {
  try {
    const args = parseCliArgs(process.argv.slice(2));
    const result = await runCompiler(args);
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await runCli();
}
