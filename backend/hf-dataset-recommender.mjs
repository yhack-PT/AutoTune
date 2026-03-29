import process from "node:process";
import { readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

import {
  extractRowsFromPreview,
  inferCompatibleMethodsFromNormalization,
  inferClassificationTargetCandidates,
  inferDeterministicNormalization,
  normalizeOptionalString,
  selectPreferredEvalSplit,
  selectPreferredTrainSplit,
  summarizeRow,
  uniqueStrings,
  validateNormalizationProposal,
} from "./posttraining-normalization.mjs";

const HUB_API_URL = "https://huggingface.co/api/datasets";
const DATASET_VIEWER_URL = "https://datasets-server.huggingface.co";
const OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses";
const RANKED_DATASETS_DEBUG_PATH = new URL("./datasets.json", import.meta.url);
const HUB_SEARCH_LIMIT = 25;
const SHORTLIST_LIMIT = 20;
const REQUEST_TIMEOUT_MS = 8000;
const ENRICHMENT_CONCURRENCY = 5;
const DEFAULT_OPENAI_MODEL = "gpt-5-mini";
const LOG_PREFIX = "[hf-dataset-recommender]";
const TRANSIENT_STATUS_CODES = new Set([408, 409, 429, 500, 502, 503, 504]);

let dotEnvLoaded = false;
let activeLogger = null;

const DEFAULT_ANALYSIS = {
  domain_summary: "",
  mapped_task_types: [],
  data_format_needed: "mixed",
  quality_tier_strategy: "",
};

const DEFAULT_TASK_SPEC = {
  supported: false,
  task_family: "unsupported",
  target_policy: "unsupported",
  output_shape_preference: "unsupported",
  objective_summary: "",
  selected_target_focus: null,
  requested_targets: [],
  task_warnings: [],
  target_selection_reason: null,
  unsupported_reason: "Task spec was not provided.",
};

const DEFAULT_GUIDANCE = {
  ideal_dataset_count: 3,
  target_total_rows: "",
  mixing_strategy: "",
  warnings: [],
};

const DEFAULT_RANKING_CRITERIA = [
  {
    criterion: "Search relevance",
    weight: "high",
    description:
      "Prefer datasets whose name, description, tags, and previewed schema closely match the GPT-generated Hugging Face search queries.",
  },
  {
    criterion: "Task and schema fit",
    weight: "high",
    description:
      "Prefer datasets whose task tags or sample fields clearly support the intended fine-tuning format.",
  },
  {
    criterion: "Dataset size",
    weight: "medium",
    description:
      "Favor datasets that meet the requested minimum row counts and sit near the desired target row band when one is provided.",
  },
  {
    criterion: "Accessibility and licensing",
    weight: "medium",
    description:
      "Prefer public, viewer-accessible datasets with train splits and explicit licensing information.",
  },
  {
    criterion: "Community validation",
    weight: "low",
    description:
      "Downloads and likes are useful tie-breakers when multiple candidates are otherwise similar.",
  },
];

const QUERY_NOISE_TERMS = new Set([
  "dataset",
  "datasets",
  "labeled",
  "labelled",
  "labels",
  "examples",
  "small",
  "quick",
  "rapid",
]);

/**
 * Accepts a GPT-produced search plan and returns one or a few recommended Hugging Face datasets.
 */
async function withLoggerContext(logger, callback) {
  const previousLogger = activeLogger;
  activeLogger = logger ?? previousLogger;
  try {
    return await callback();
  } finally {
    activeLogger = previousLogger;
  }
}

export class RecommendationFailure extends Error {
  constructor(message, recommendation) {
    super(message);
    this.name = "RecommendationFailure";
    this.recommendation = recommendation;
  }
}

export async function recommendDatasets(input, options = {}) {
  return withLoggerContext(options.logger, async () => {
    logInfo(
      `starting recommendDatasets in ${isSearchPlanInput(input) ? "search-plan" : "text"} mode`,
    );
    logJson("input", input);

    const normalizedPlan = isSearchPlanInput(input)
      ? validatePlan(input)
      : await createPlanFromUserInputs(input, options);

    if (!normalizedPlan.task_spec.supported) {
      const recommendation = {
        analysis: normalizedPlan.analysis,
        task_spec: normalizedPlan.task_spec,
        search_queries: normalizedPlan.search_queries,
        ranking_criteria: normalizedPlan.ranking_criteria,
        recommendation_guidance: normalizedPlan.recommendation_guidance,
        compatibility_summary: {
          total_candidates: 0,
          compatible_candidates: 0,
          incompatible_candidates: 0,
        },
        ranked_datasets: [],
        recommended_datasets: [],
        fatal_error: {
          code: "unsupported_task",
          message:
            normalizedPlan.task_spec.unsupported_reason ||
            "Only single-target classification SFT jobs are supported right now.",
        },
      };
      if (!options.skipDebugWrite) {
        await writeRankedDatasetsDebugFile(recommendation, options.debugOutputPath);
      }
      throw new RecommendationFailure(recommendation.fatal_error.message, recommendation);
    }

    logJson("normalized plan", normalizedPlan);
    const context = buildContext(normalizedPlan);
    logInfo(
      `running ${context.search_queries.length} HF searches with min row floor ${context.min_rows_floor || 0}`,
    );
    const discoveredCandidates = await (
      typeof options.discoverCandidates === "function"
        ? options.discoverCandidates(context)
        : discoverCandidates(context)
    );
    logInfo(`discovered ${discoveredCandidates.length} candidate datasets before enrichment`);
    const enrichedCandidates = await (
      typeof options.enrichCandidates === "function"
        ? options.enrichCandidates(discoveredCandidates, context, options)
        : enrichCandidates(discoveredCandidates, context, options)
    );
    logInfo(`enriched ${enrichedCandidates.length} candidate datasets`);
    const publicCandidates = enrichedCandidates.filter((candidate) => !candidate.private && !candidate.gated);
    const rankedDatasets = sortCandidatesForDiagnostics(publicCandidates);
    const compatibleCandidates = publicCandidates.filter((candidate) => candidate.compatibility_status === "compatible");

    if (compatibleCandidates.length === 0) {
      const recommendation = {
        analysis: normalizedPlan.analysis,
        task_spec: normalizedPlan.task_spec,
        search_queries: normalizedPlan.search_queries,
        ranking_criteria: normalizedPlan.ranking_criteria,
        recommendation_guidance: buildRecommendationGuidance([], normalizedPlan),
        compatibility_summary: buildCompatibilitySummary(publicCandidates),
        ranked_datasets: rankedDatasets,
        recommended_datasets: [],
        fatal_error: {
          code: "no_compatible_candidates",
          message:
            "No ranked dataset candidates could be normalized into a backend-supported training shape during recommendation.",
        },
      };
      if (!options.skipDebugWrite) {
        await writeRankedDatasetsDebugFile(recommendation, options.debugOutputPath);
      }
      throw new RecommendationFailure(recommendation.fatal_error.message, recommendation);
    }

    const recommendedDatasets = dedupeRankedDatasets(
      await (
        typeof options.rankCandidates === "function"
          ? options.rankCandidates(compatibleCandidates, context)
          : rankCandidatesWithOpenAI(compatibleCandidates, context)
      ),
    );
    logInfo(`returning ${recommendedDatasets.length} recommended datasets`);

    const result = {
      analysis: normalizedPlan.analysis,
      task_spec: normalizedPlan.task_spec,
      search_queries: normalizedPlan.search_queries,
      ranking_criteria: normalizedPlan.ranking_criteria,
      recommendation_guidance: buildRecommendationGuidance(recommendedDatasets, normalizedPlan),
      compatibility_summary: buildCompatibilitySummary(publicCandidates),
      ranked_datasets: rankedDatasets,
      recommended_datasets: recommendedDatasets,
    };
    if (!options.skipDebugWrite) {
      await writeRankedDatasetsDebugFile(result, options.debugOutputPath);
    }
    return result;
  });
}

export default recommendDatasets;

function sortCandidatesForDiagnostics(candidates) {
  return [...candidates].sort((left, right) => {
    const leftCompatibility = left.compatibility_status === "compatible" ? 0 : 1;
    const rightCompatibility = right.compatibility_status === "compatible" ? 0 : 1;
    if (leftCompatibility !== rightCompatibility) {
      return leftCompatibility - rightCompatibility;
    }
    const leftSource = left.normalization_source === "deterministic" ? 0 : 1;
    const rightSource = right.normalization_source === "deterministic" ? 0 : 1;
    if (leftSource !== rightSource) {
      return leftSource - rightSource;
    }
    return Number(right.score ?? 0) - Number(left.score ?? 0);
  });
}

function buildCompatibilitySummary(candidates) {
  return {
    total_candidates: candidates.length,
    compatible_candidates: candidates.filter((candidate) => candidate.compatibility_status === "compatible").length,
    incompatible_candidates: candidates.filter((candidate) => candidate.compatibility_status !== "compatible").length,
  };
}

function isSearchPlanInput(input) {
  return Boolean(input && typeof input === "object" && Array.isArray(input.search_queries));
}

function validatePlan(plan) {
  if (!plan || typeof plan !== "object") {
    throw new Error("Input must be an object containing `search_queries`.");
  }

  const searchQueries = Array.isArray(plan.search_queries)
    ? plan.search_queries
      .map((query) => normalizeSearchQuery(query))
      .filter((query) => query.search)
    : [];

  if (searchQueries.length === 0) {
    throw new Error("`search_queries` must be a non-empty array of query objects.");
  }

  const analysis = {
    ...DEFAULT_ANALYSIS,
    ...(plan.analysis && typeof plan.analysis === "object" ? plan.analysis : {}),
  };

  analysis.domain_summary = String(analysis.domain_summary ?? "").trim();
  analysis.mapped_task_types = uniqueStrings(
    Array.isArray(analysis.mapped_task_types)
      ? analysis.mapped_task_types.map((task) => String(task).trim()).filter(Boolean)
      : searchQueries.map((query) => query.task_filter).filter(Boolean),
  );
  analysis.data_format_needed = normalizeOptionalString(analysis.data_format_needed) ?? "mixed";
  analysis.quality_tier_strategy = String(analysis.quality_tier_strategy ?? "").trim();

  const taskSpec = {
    ...DEFAULT_TASK_SPEC,
    ...(plan.task_spec && typeof plan.task_spec === "object" ? plan.task_spec : {}),
  };

  taskSpec.supported = Boolean(taskSpec.supported);
  taskSpec.task_family = normalizeOptionalString(taskSpec.task_family) ?? "unsupported";
  taskSpec.target_policy = normalizeOptionalString(taskSpec.target_policy) ?? "unsupported";
  taskSpec.output_shape_preference =
    normalizeOptionalString(taskSpec.output_shape_preference) ?? "unsupported";
  taskSpec.objective_summary = String(taskSpec.objective_summary ?? "").trim();
  taskSpec.selected_target_focus = normalizeOptionalString(taskSpec.selected_target_focus);
  taskSpec.requested_targets = uniqueStrings(
    Array.isArray(taskSpec.requested_targets)
      ? taskSpec.requested_targets.map((target) => String(target).trim()).filter(Boolean)
      : [],
  );
  taskSpec.task_warnings = uniqueStrings(
    Array.isArray(taskSpec.task_warnings)
      ? taskSpec.task_warnings.map((warning) => String(warning).trim()).filter(Boolean)
      : [],
  );
  taskSpec.target_selection_reason = normalizeOptionalString(taskSpec.target_selection_reason);
  taskSpec.unsupported_reason = normalizeOptionalString(taskSpec.unsupported_reason);

  const guidance = {
    ...DEFAULT_GUIDANCE,
    ...(plan.recommendation_guidance && typeof plan.recommendation_guidance === "object"
      ? plan.recommendation_guidance
      : {}),
  };

  guidance.ideal_dataset_count = normalizePositiveInt(guidance.ideal_dataset_count) ?? 3;
  guidance.target_total_rows = String(guidance.target_total_rows ?? "").trim();
  guidance.mixing_strategy = String(guidance.mixing_strategy ?? "").trim();
  guidance.warnings = uniqueStrings(
    Array.isArray(guidance.warnings)
      ? guidance.warnings.map((warning) => String(warning).trim()).filter(Boolean)
      : [],
  );

  const rankingCriteria =
    Array.isArray(plan.ranking_criteria) && plan.ranking_criteria.length > 0
      ? plan.ranking_criteria
        .map((criterion) => normalizeCriterion(criterion))
        .filter((criterion) => criterion.criterion)
      : DEFAULT_RANKING_CRITERIA;

  return {
    analysis,
    task_spec: taskSpec,
    search_queries: searchQueries,
    ranking_criteria: rankingCriteria,
    recommendation_guidance: guidance,
  };
}

async function createPlanFromUserInputs(input, options = {}) {
  const normalizedInput = validateUserInputs(input);
  if (typeof options.planGenerator === "function") {
    const generatedPlan = await options.planGenerator(normalizedInput);
    return coerceTaskSpecForRawInput(validatePlan(generatedPlan), normalizedInput);
  }

  await ensureDotEnvLoaded();
  logJson("validated text input", normalizedInput);

  const apiKey = normalizeOptionalString(process.env.OPENAI_API_KEY);
  if (!apiKey) {
    throw new Error(
      "OPENAI_API_KEY is required to generate a search plan from a text description.",
    );
  }

  const model = normalizeOptionalString(process.env.OPENAI_MODEL) ?? DEFAULT_OPENAI_MODEL;
  const prompt = buildOpenAIPlanningPrompt(normalizedInput);
  const openAIRequestBody = {
    model,
    store: false,
    input: [
      {
        role: "system",
        content:
          "You generate search plans for finding Hugging Face datasets for language-model post-training. Return only the requested JSON structure.",
      },
      {
        role: "user",
        content: prompt,
      },
    ],
    text: {
      format: {
        type: "json_schema",
        name: "hf_dataset_search_plan",
        schema: getOpenAIPlanSchema(),
        strict: true,
      },
    },
  };
  logInfo(`creating search plan with OpenAI model ${model}`);
  logMultiline("openai prompt", prompt);

  const response = await fetchJson(OPENAI_RESPONSES_API_URL, {
    method: "POST",
    timeoutMs: 60_000,
    maxRetries: 3,
    requestLabel: "OpenAI planning request",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(openAIRequestBody),
  });

  logJson("openai response metadata", {
    id: response.id ?? null,
    status: response.status ?? null,
    model: response.model ?? null,
    usage: response.usage ?? null,
  });

  const outputText = extractOpenAIOutputText(response);
  let plan;
  let parseError = null;
  if (outputText) {
    logMultiline("openai output text", outputText);
  }

  try {
    if (outputText) {
      plan = JSON.parse(outputText);
    }
  } catch (error) {
    parseError =
      error instanceof Error ? error.message : "unknown parse error";
  }

  if (!outputText) {
    throw new Error("OpenAI returned an empty planning response.");
  }
  if (parseError) {
    throw new Error(`OpenAI returned invalid JSON for the search plan: ${parseError}`);
  }

  logJson("openai parsed plan", plan);

  return coerceTaskSpecForRawInput(validatePlan(plan), normalizedInput);
}

function extractRequestedTargetsFromText(inputText) {
  const text = String(inputText ?? "").trim().replace(/[.?!]+$/, "");
  if (!text) {
    return [];
  }

  const byMatch = text.match(/\bby\s+(.+)$/i);
  const targetClause = byMatch ? byMatch[1] : text;
  const parts = targetClause
    .split(/\s*(?:,|\/|&|\band\b|\bplus\b)\s*/i)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => part.replace(/^(the|a|an)\s+/i, "").trim())
    .filter(Boolean);

  return uniqueStrings(parts);
}

function looksLikeClassificationRequest(inputText) {
  return /\bclassif(?:y|ication|ies|ied)\b/i.test(String(inputText ?? ""));
}

function buildObjectiveSummary(rawInput, selectedTargetFocus) {
  const domain = normalizeOptionalString(rawInput?.domain);
  if (domain) {
    return `Classify ${domain} examples by ${selectedTargetFocus}.`;
  }
  return `Classify examples by ${selectedTargetFocus}.`;
}

export function coerceTaskSpecForRawInput(plan, rawInput) {
  if (plan.task_spec.supported) {
    return plan;
  }

  const normalizedInput = validateUserInputs(rawInput);
  const requestText = normalizedInput.useCase ?? normalizedInput.description;
  const requestedTargets = extractRequestedTargetsFromText(requestText);
  if (!looksLikeClassificationRequest(requestText) || requestedTargets.length < 2) {
    return plan;
  }

  const selectedTargetFocus = requestedTargets[0];
  const warning =
    `Requested multiple targets (${requestedTargets.join(", ")}); continuing with '${selectedTargetFocus}' as the single-target focus for this run.`;

  return {
    ...plan,
    task_spec: {
      supported: true,
      task_family: "classification",
      target_policy: "single_target",
      output_shape_preference: "prompt_completion",
      objective_summary: buildObjectiveSummary(normalizedInput, selectedTargetFocus),
      selected_target_focus: selectedTargetFocus,
      requested_targets: requestedTargets,
      task_warnings: uniqueStrings([...(plan.task_spec.task_warnings ?? []), warning]),
      target_selection_reason:
        `Defaulted to the first requested target '${selectedTargetFocus}' so the single-target backend can continue.`,
      unsupported_reason: null,
    },
    recommendation_guidance: {
      ...plan.recommendation_guidance,
      warnings: uniqueStrings([...(plan.recommendation_guidance.warnings ?? []), warning]),
    },
  };
}

function extractOpenAIOutputText(response) {
  const topLevelText = normalizeOptionalString(response?.output_text);
  if (topLevelText) {
    return topLevelText;
  }

  const outputItems = Array.isArray(response?.output) ? response.output : [];
  for (const item of outputItems) {
    const content = Array.isArray(item?.content) ? item.content : [];
    for (const part of content) {
      if (part?.type === "output_text") {
        const text = normalizeOptionalString(part.text);
        if (text) {
          return text;
        }
      }
    }
  }

  return null;
}

function validateUserInputs(input) {
  if (typeof input === "string") {
    const description = input.trim();
    if (!description) {
      throw new Error("Input text must be a non-empty string.");
    }
    return {
      description,
      domain: null,
      qualityTier: null,
      useCase: null,
    };
  }

  if (!input || typeof input !== "object") {
    throw new Error(
      "Input must either include `search_queries` or a single text field such as `description`.",
    );
  }

  const description =
    normalizeOptionalString(input.description) ??
    normalizeOptionalString(input.text) ??
    normalizeOptionalString(input.prompt) ??
    null;
  if (description) {
    const qualityTier = normalizeQualityTier(input.qualityTier);
    return {
      description,
      domain: normalizeOptionalString(input.domain),
      qualityTier,
      useCase: normalizeOptionalString(input.useCase),
    };
  }

  const domain = normalizeOptionalString(input.domain);
  const useCase = normalizeOptionalString(input.useCase);
  const qualityTier = normalizeQualityTier(input.qualityTier);

  if (!domain) {
    throw new Error("Legacy raw input requires `domain`, or provide a single `description` field.");
  }
  if (!useCase) {
    throw new Error("Legacy raw input requires `useCase`, or provide a single `description` field.");
  }

  return {
    description: buildLegacyDescription({ domain, qualityTier, useCase }),
    domain,
    qualityTier,
    useCase,
  };
}

function buildLegacyDescription(input) {
  const lines = [`Domain: ${input.domain}`, `Use case: ${input.useCase}`];
  if (input.qualityTier != null) {
    lines.splice(1, 0, `Quality tier: ${input.qualityTier}`);
  }
  return lines.join("\n");
}

function normalizeQualityTier(value) {
  if (value == null || value === "") {
    return null;
  }

  const qualityTier = Number(value);
  if (!Number.isInteger(qualityTier) || qualityTier < 1 || qualityTier > 5) {
    throw new Error("`qualityTier` must be an integer between 1 and 5 when provided.");
  }
  return qualityTier;
}

function buildOpenAIPlanningPrompt(input) {
  const promptLines = [
    "Create a Hugging Face dataset search plan for post-training a language model.",
    "This backend v1 supports only single-target classification SFT jobs.",
    `User request: ${input.description}`,
    "Quality tier definitions:",
    "1 = Fastest: under 10K rows, one tightly matched dataset, optimize for minutes to a few hours.",
    "2 = Fast: 10K-50K rows, 1-2 complementary datasets, moderate filtering.",
    "3 = Balanced: 50K-500K rows, 2-3 datasets, core use case plus edge cases.",
    "4 = Quality: 500K-2M rows, 3-5 diverse high-quality datasets.",
    "5 = Maximum Quality: 2M+ rows, 4-6+ datasets, broad foundational and domain-specialized coverage.",
    input.qualityTier == null
      ? "If the request does not specify a desired data volume or quality tier, default to a balanced tier-3 plan."
      : `The user explicitly requested quality tier ${input.qualityTier}; set min_rows and guidance accordingly.`,
    "Return task_spec, analysis, search_queries, ranking_criteria, and recommendation_guidance.",
    "search_queries must contain concise Hugging Face search strings, usually 2-6 words, like something typed directly into the Hugging Face search bar.",
    "Use task_filter values only from: text-classification, question-answering, summarization, text-generation, translation, conversational, token-classification, or null.",
    "Use sort values only from: downloads, likes, trending, created.",
    "Set min_rows based on the explicit or inferred quality tier.",
    "Set data_format_needed to exactly one of: instruction, completion, preference, raw_text, mixed.",
    "Keep mapped_task_types narrowly focused on the main fine-tuning objective, usually 1-2 task types.",
    "Keep warnings focused on practical dataset-selection risks.",
    "If the request mentions multiple classification targets, choose one primary target to optimize for, keep task_spec.supported=true, and record a warning plus the selected_target_focus.",
    "Set task_spec.supported=false only when the request is non-classification or otherwise cannot be reduced to a single-target classification SFT job.",
  ];

  return promptLines.filter(Boolean).join("\n");
}

function getOpenAIPlanSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      task_spec: {
        type: "object",
        additionalProperties: false,
        properties: {
          supported: { type: "boolean" },
          task_family: {
            type: "string",
            enum: ["classification", "unsupported"],
          },
          target_policy: {
            type: "string",
            enum: ["single_target", "unsupported"],
          },
          output_shape_preference: {
            type: "string",
            enum: ["prompt_completion", "unsupported"],
          },
          objective_summary: { type: "string" },
          selected_target_focus: { type: ["string", "null"] },
          requested_targets: {
            type: "array",
            items: { type: "string" },
          },
          task_warnings: {
            type: "array",
            items: { type: "string" },
          },
          target_selection_reason: { type: ["string", "null"] },
          unsupported_reason: { type: ["string", "null"] },
        },
        required: [
          "supported",
          "task_family",
          "target_policy",
          "output_shape_preference",
          "objective_summary",
          "selected_target_focus",
          "requested_targets",
          "task_warnings",
          "target_selection_reason",
          "unsupported_reason",
        ],
      },
      analysis: {
        type: "object",
        additionalProperties: false,
        properties: {
          domain_summary: { type: "string" },
          mapped_task_types: {
            type: "array",
            maxItems: 3,
            items: {
              type: "string",
              enum: [
                "text-classification",
                "question-answering",
                "summarization",
                "text-generation",
                "translation",
                "conversational",
                "token-classification",
              ],
            },
          },
          data_format_needed: {
            type: "string",
            enum: ["instruction", "completion", "preference", "raw_text", "mixed"],
          },
          quality_tier_strategy: { type: "string" },
        },
        required: [
          "domain_summary",
          "mapped_task_types",
          "data_format_needed",
          "quality_tier_strategy",
        ],
      },
      search_queries: {
        type: "array",
        minItems: 1,
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            search: { type: "string" },
            task_filter: {
              type: ["string", "null"],
            },
            sort: {
              type: "string",
              enum: ["downloads", "likes", "trending", "created"],
            },
            min_rows: { type: "integer", minimum: 0 },
            intent: { type: "string" },
          },
          required: ["search", "task_filter", "sort", "min_rows", "intent"],
        },
      },
      ranking_criteria: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            criterion: { type: "string" },
            weight: { type: "string", enum: ["high", "medium", "low"] },
            description: { type: "string" },
          },
          required: ["criterion", "weight", "description"],
        },
      },
      recommendation_guidance: {
        type: "object",
        additionalProperties: false,
        properties: {
          ideal_dataset_count: { type: "integer", minimum: 1 },
          target_total_rows: { type: "string" },
          mixing_strategy: { type: "string" },
          warnings: {
            type: "array",
            items: { type: "string" },
          },
        },
        required: [
          "ideal_dataset_count",
          "target_total_rows",
          "mixing_strategy",
          "warnings",
        ],
      },
    },
    required: [
      "task_spec",
      "analysis",
      "search_queries",
      "ranking_criteria",
      "recommendation_guidance",
    ],
  };
}

function normalizeSearchQuery(query) {
  if (!query || typeof query !== "object") {
    return {
      search: "",
      task_filter: null,
      sort: "downloads",
      min_rows: 0,
      intent: "",
    };
  }

  const sort = ["downloads", "likes", "trending", "created"].includes(query.sort)
    ? query.sort
    : "downloads";

  return {
    search: String(query.search ?? "").trim(),
    task_filter: normalizeOptionalString(query.task_filter),
    sort,
    min_rows: normalizeNonNegativeInt(query.min_rows) ?? 0,
    intent: String(query.intent ?? "").trim(),
  };
}

function normalizeCriterion(criterion) {
  return {
    criterion: String(criterion?.criterion ?? "").trim(),
    weight: ["high", "medium", "low"].includes(criterion?.weight) ? criterion.weight : "medium",
    description: String(criterion?.description ?? "").trim(),
  };
}

function buildContext(plan) {
  const queryTokens = uniqueStrings(
    plan.search_queries.flatMap((query) => tokenize([query.search, query.intent].join(" "))),
  );
  const rowFloors = plan.search_queries
    .map((query) => query.min_rows)
    .filter((value) => Number.isFinite(value) && value > 0);

  return {
    ...plan,
    query_tokens: queryTokens,
    min_rows_floor: rowFloors.length > 0 ? Math.min(...rowFloors) : 0,
    target_row_band: parseTargetRowBand(plan.recommendation_guidance.target_total_rows),
  };
}

async function discoverCandidates(context) {
  const candidateMap = new Map();

  await Promise.all(
    context.search_queries.map(async (query) => {
      logJson("searching HF datasets for query", query);
      const datasets = await searchHubDatasets(query);
      logInfo(`HF search "${query.search}" returned ${datasets.length} unique candidates`);

      for (const dataset of datasets) {
        const id = dataset.id ?? dataset._id ?? dataset.name;
        if (!id) {
          continue;
        }

        const metadata = {
          id,
          source_url: `https://huggingface.co/datasets/${id}`,
          description:
            dataset.description ??
            dataset.cardData?.description ??
            dataset.cardData?.pretty_name ??
            "",
          downloads: toNumber(dataset.downloads),
          likes: toNumber(dataset.likes),
          gated: Boolean(dataset.gated),
          private: Boolean(dataset.private),
          tags: Array.isArray(dataset.tags) ? dataset.tags : [],
          cardData: dataset.cardData ?? {},
        };

        const existing = candidateMap.get(id);
        if (existing) {
          existing.matched_queries.push(query.search);
          existing.matched_tasks = uniqueStrings([
            ...existing.matched_tasks,
            ...(query.task_filter ? [query.task_filter] : []),
          ]);
          existing.downloads = Math.max(existing.downloads, metadata.downloads);
          existing.likes = Math.max(existing.likes, metadata.likes);
          existing.tags = uniqueStrings([...existing.tags, ...metadata.tags]);
          continue;
        }

        candidateMap.set(id, {
          ...metadata,
          matched_queries: [query.search],
          matched_tasks: query.task_filter ? [query.task_filter] : [],
        });
      }
    }),
  );

  return [...candidateMap.values()].slice(0, SHORTLIST_LIMIT);
}

async function searchHubDatasets(query) {
  const results = new Map();

  for (const variant of buildSearchVariants(query)) {
    logInfo(`trying HF search variant: ${variant}`);
    const url = new URL(HUB_API_URL);
    url.searchParams.set("search", variant);
    url.searchParams.set("limit", String(HUB_SEARCH_LIMIT));
    url.searchParams.set("sort", query.sort);
    url.searchParams.set("direction", "-1");
    url.searchParams.set("full", "true");

    try {
      const response = await fetchJson(url.toString(), {
        timeoutMs: REQUEST_TIMEOUT_MS,
        maxRetries: 2,
        requestLabel: `HF search for "${query.search}"`,
      });
      if (!Array.isArray(response) || response.length === 0) {
        logInfo(`HF search variant produced no results: ${variant}`);
        continue;
      }
      logInfo(`HF search variant produced ${response.length} results: ${variant}`);

      for (const dataset of response) {
        const id = dataset.id ?? dataset._id ?? dataset.name;
        if (!id || results.has(id)) {
          continue;
        }
        results.set(id, dataset);
      }

      if (results.size >= 10) {
        logInfo(`stopping early after collecting ${results.size} results for query "${query.search}"`);
        break;
      }
    } catch {
      logInfo(`HF search variant failed: ${variant}`);
      continue;
    }
  }

  return [...results.values()];
}

function buildSearchVariants(searchText) {
  const query = typeof searchText === "string" ? { search: searchText, task_filter: null } : searchText;
  const normalized = normalizeText(query.search);
  const tokens = tokenize(query.search).filter((token) => !QUERY_NOISE_TERMS.has(token));
  const compact = tokens.slice(0, 5).join(" ");
  const broad = tokens.slice(0, 3).join(" ");
  const withTask = query.task_filter && compact ? `${compact} ${taskSearchLabel(query.task_filter)}` : null;

  return uniqueStrings([query.search, normalized, compact, broad, withTask].filter(Boolean));
}

async function enrichCandidates(candidates, context, options = {}) {
  logInfo(`starting enrichment for ${candidates.length} candidates`);
  return mapWithConcurrency(candidates, ENRICHMENT_CONCURRENCY, async (candidate) => {
    logInfo(`enriching dataset ${candidate.id}`);
    const warnings = [];

    if (candidate.gated) {
      warnings.push("Gated dataset; excluded from public-only recommendations.");
    }
    if (candidate.private) {
      warnings.push("Private dataset; excluded from public-only recommendations.");
    }

    const validity = await fetchViewerEndpoint("/is-valid", { dataset: candidate.id }).catch(() => null);
    if (validity?.error) {
      warnings.push("Dataset viewer reported the dataset as inaccessible or unsupported.");
    }

    const viewerAccessible =
      validity && !validity.error && (validity.viewer === true || validity.preview === true);

    const sizePayload = await fetchViewerEndpoint("/size", { dataset: candidate.id }).catch(() => null);
    const sizeInfo = parseSizeInfo(sizePayload);
    if (sizeInfo.partial) {
      warnings.push("Dataset size is partial or approximate in the dataset viewer.");
    }

    const splitsPayload = await fetchViewerEndpoint("/splits", { dataset: candidate.id }).catch(() => null);
    const splitsInfo = parseSplitsInfo(splitsPayload);
    const previewTarget = choosePreviewTarget(splitsInfo.rawSplits);
    const preferredTrainSplit = selectPreferredTrainSplit(splitsInfo.rawSplits);
    const preferredEvalSplit = selectPreferredEvalSplit(splitsInfo.rawSplits);

    let schemaPreview = null;
    if (viewerAccessible && previewTarget) {
      schemaPreview = await fetchViewerEndpoint("/first-rows", {
        dataset: candidate.id,
        config: previewTarget.config,
        split: previewTarget.split,
      }).catch(() => null);
    }

    const schemaSignals = parseSchemaSignals(schemaPreview, candidate.tags);
    const featureNames = Array.isArray(schemaPreview?.features)
      ? schemaPreview.features.map((feature) => String(feature?.name ?? "")).filter(Boolean)
      : [];
    const sampleRows = extractRowsFromPreview(schemaPreview).slice(0, 3);
    const rowKeys = uniqueStrings(sampleRows.flatMap((row) => Object.keys(row ?? {})));
    const availableColumns = uniqueStrings([...featureNames, ...rowKeys]);
    const sourceSchema = {
      available_columns: availableColumns,
      sample_rows: sampleRows.map((row) => summarizeRow(row)),
    };
    const directCompatibility = inferDeterministicNormalization(availableColumns, sampleRows);
    const directTargetCandidates = inferClassificationTargetCandidates(availableColumns, sampleRows);

    let compatibilityStatus = directCompatibility.compatibility_status;
    let compatibilityReason = directCompatibility.compatibility_reason;
    let normalizationProposal = directCompatibility.normalization_proposal;
    let normalizationSource = normalizationProposal ? "deterministic" : null;
    let selectedTargetColumn = directCompatibility.selected_target_column ?? null;
    let targetSelectionReason = directCompatibility.target_selection_reason ?? null;
    let targetSelectionConfidence = directCompatibility.target_selection_confidence ?? null;
    let targetCandidates = Array.isArray(directCompatibility.target_candidates)
      ? directCompatibility.target_candidates
      : [];
    let ambiguityWarnings = Array.isArray(directCompatibility.ambiguity_warnings)
      ? directCompatibility.ambiguity_warnings
      : [];

    const needsTargetDisambiguation =
      context.task_spec?.task_family === "classification" &&
      directTargetCandidates.length > 1 &&
      normalizationProposal?.strategy === "classification_template";

    if ((!normalizationProposal || needsTargetDisambiguation) && viewerAccessible && availableColumns.length > 0) {
      const gptNormalization = await inferNormalizationProposal(candidate, context, {
        ...options,
        sourceSchema,
      }).catch((error) => {
        warnings.push(
          `Normalization inference failed: ${error instanceof Error ? error.message : String(error)}`,
        );
        return null;
      });

      const proposalErrors = validateNormalizationProposal(
        gptNormalization?.normalization_proposal,
        availableColumns,
      );
      if (gptNormalization?.normalization_proposal && proposalErrors.length === 0) {
        normalizationProposal = gptNormalization.normalization_proposal;
        compatibilityStatus = "compatible";
        compatibilityReason = gptNormalization.compatibility_reason;
        normalizationSource = "openai";
        selectedTargetColumn = normalizeOptionalString(gptNormalization.selected_target_column);
        targetSelectionReason = normalizeOptionalString(gptNormalization.target_selection_reason);
        targetSelectionConfidence =
          typeof gptNormalization.target_selection_confidence === "number"
            ? gptNormalization.target_selection_confidence
            : null;
        targetCandidates = Array.isArray(gptNormalization.target_candidates)
          ? uniqueStrings(gptNormalization.target_candidates.map((value) => String(value).trim()))
          : targetCandidates;
        ambiguityWarnings = Array.isArray(gptNormalization.ambiguity_warnings)
          ? uniqueStrings(gptNormalization.ambiguity_warnings.map((value) => String(value).trim()))
          : ambiguityWarnings;
      } else if (gptNormalization?.normalization_proposal) {
        warnings.push(
          `Discarded invalid normalization proposal: ${proposalErrors.join(" ")}`,
        );
      } else if (gptNormalization?.compatibility_reason) {
        compatibilityReason = gptNormalization.compatibility_reason;
      }
    }

    const compatibleMethods = inferCompatibleMethodsFromNormalization(normalizationProposal);
    const classificationCompatible =
      context.task_spec?.task_family === "classification" &&
      normalizationProposal?.shape === "prompt_completion" &&
      normalizeOptionalString(selectedTargetColumn);
    const license = extractLicense(candidate);

    if (!license) {
      warnings.push("License is missing or unclear in the available Hub metadata.");
    }
    if (!splitsInfo.names.includes("train") && !splitsInfo.names.includes("training")) {
      warnings.push("No obvious train split was detected.");
    }

    return {
      ...candidate,
      viewer_accessible: viewerAccessible,
      num_rows: sizeInfo.numRows,
      size_partial: sizeInfo.partial,
      splits: splitsInfo.names,
      schema_signals: schemaSignals.signals,
      schema_details: { feature_names: availableColumns },
      source_schema: sourceSchema,
      selected_target_column: selectedTargetColumn,
      target_selection_reason: targetSelectionReason,
      target_selection_confidence: targetSelectionConfidence,
      target_candidates: targetCandidates,
      ambiguity_warnings: ambiguityWarnings,
      preferred_dataset_config: normalizeOptionalString(preferredTrainSplit?.config) ?? null,
      preferred_train_split: normalizeOptionalString(preferredTrainSplit?.split) ?? "train",
      preferred_eval_split: normalizeOptionalString(preferredEvalSplit?.split),
      compatibility_status:
        compatibleMethods.length > 0 && classificationCompatible ? "compatible" : "incompatible",
      compatibility_reason:
        compatibleMethods.length > 0 && classificationCompatible
          ? compatibilityReason
          : context.task_spec?.task_family === "classification"
            ? "Candidate is not compatible with single-target classification SFT requirements."
            : compatibilityReason ||
            "No supported normalization proposal could be inferred for this dataset.",
      normalization_source: normalizationSource,
      normalization_proposal: normalizationProposal,
      compatible_methods: classificationCompatible ? compatibleMethods : [],
      license,
      warnings,
    };
  });
}

async function inferNormalizationProposal(candidate, context, options = {}) {
  if (typeof options.normalizationPlanner === "function") {
    return options.normalizationPlanner(candidate, context, options.sourceSchema);
  }
  return callOpenAINormalizationPlanner(candidate, context, options.sourceSchema);
}

async function fetchViewerEndpoint(path, params) {
  const url = new URL(path, DATASET_VIEWER_URL);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  }
  return fetchJson(url.toString(), {
    timeoutMs: REQUEST_TIMEOUT_MS,
    maxRetries: 2,
    requestLabel: `HF viewer ${path}`,
  });
}

function parseSizeInfo(payload) {
  return {
    numRows: toNumber(payload?.size?.dataset?.num_rows),
    partial: Boolean(payload?.partial),
  };
}

function parseSplitsInfo(payload) {
  const rawSplits = Array.isArray(payload?.splits) ? payload.splits : [];
  return {
    rawSplits,
    names: uniqueStrings(
      rawSplits.map((split) => String(split.split ?? "").toLowerCase().trim()).filter(Boolean),
    ),
  };
}

function choosePreviewTarget(splits) {
  if (!Array.isArray(splits) || splits.length === 0) {
    return null;
  }

  const preferredOrder = ["train", "training", "default", "validation", "dev", "test"];
  for (const preferred of preferredOrder) {
    const match = splits.find((split) => String(split.split ?? "").toLowerCase() === preferred);
    if (match) {
      return { config: match.config, split: match.split };
    }
  }

  return { config: splits[0].config, split: splits[0].split };
}

function parseSchemaSignals(previewPayload, tags) {
  const features = Array.isArray(previewPayload?.features) ? previewPayload.features : [];
  const rows = Array.isArray(previewPayload?.rows) ? previewPayload.rows : [];
  const featureNames = features.map((feature) => String(feature.name ?? "").toLowerCase()).filter(Boolean);
  const rowKeys = uniqueStrings(
    rows
      .flatMap((rowEntry) => (rowEntry?.row && typeof rowEntry.row === "object" ? Object.keys(rowEntry.row) : []))
      .map((key) => key.toLowerCase()),
  );
  const allKeys = uniqueStrings([...featureNames, ...rowKeys]);
  const tagText = (Array.isArray(tags) ? tags : []).join(" ").toLowerCase();
  const signals = [];

  if (allKeys.includes("text")) signals.push("has_text");
  if (allKeys.includes("label") || tagText.includes("text-classification")) signals.push("classification_ready");
  if (allKeys.includes("question") || allKeys.includes("answer") || allKeys.includes("answers") || tagText.includes("question-answering")) signals.push("qa_ready");
  if (allKeys.includes("summary") || allKeys.includes("abstract") || allKeys.includes("highlights") || tagText.includes("summarization")) signals.push("summarization_ready");
  if (allKeys.includes("instruction") || allKeys.includes("prompt") || allKeys.includes("response") || allKeys.includes("output")) signals.push("instruction_ready");
  if (allKeys.includes("messages") || allKeys.includes("conversation") || allKeys.includes("dialog") || allKeys.includes("dialogue")) signals.push("conversation_ready");
  if (allKeys.includes("translation") || (allKeys.includes("source") && allKeys.includes("target")) || tagText.includes("translation")) signals.push("translation_ready");
  if (allKeys.includes("tokens") || allKeys.includes("ner_tags") || allKeys.includes("pos_tags") || tagText.includes("token-classification")) signals.push("token_label_ready");

  return {
    signals: uniqueStrings(signals),
    details: { feature_names: allKeys },
  };
}

function extractLicense(candidate) {
  const cardData = candidate.cardData ?? {};
  const tagLicense = candidate.tags
    .find((tag) => typeof tag === "string" && tag.startsWith("license:"))
    ?.replace("license:", "");

  return (
    normalizeOptionalString(cardData.license) ??
    normalizeOptionalString(Array.isArray(cardData.licenses) ? cardData.licenses[0] : null) ??
    normalizeOptionalString(tagLicense) ??
    null
  );
}

function getNormalizationFieldSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      source_column: { type: ["string", "null"] },
      template: { type: ["string", "null"] },
      value_mapping: {
        type: ["object", "null"],
        additionalProperties: { type: "string" },
      },
    },
    required: ["source_column", "template", "value_mapping"],
  };
}

function getNormalizationProposalSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      usable: { type: "boolean" },
      compatibility_reason: { type: "string" },
      selected_target_column: { type: ["string", "null"] },
      target_selection_reason: { type: ["string", "null"] },
      target_selection_confidence: { type: ["number", "null"] },
      target_candidates: {
        type: "array",
        items: { type: "string" },
      },
      ambiguity_warnings: {
        type: "array",
        items: { type: "string" },
      },
      normalization_proposal: {
        anyOf: [
          { type: "null" },
          {
            type: "object",
            additionalProperties: false,
            properties: {
              version: { type: "integer", enum: [1] },
              shape: { type: "string", enum: ["text", "prompt_completion"] },
              strategy: {
                type: "string",
                enum: [
                  "copy_column",
                  "copy_columns",
                  "qa_template",
                  "classification_template",
                  "template_synthesis",
                ],
              },
              source_columns: {
                type: "array",
                minItems: 1,
                items: { type: "string" },
              },
              fields: {
                type: "object",
                additionalProperties: false,
                properties: {
                  text: { anyOf: [{ type: "null" }, getNormalizationFieldSchema()] },
                  prompt: { anyOf: [{ type: "null" }, getNormalizationFieldSchema()] },
                  completion: { anyOf: [{ type: "null" }, getNormalizationFieldSchema()] },
                },
                required: ["text", "prompt", "completion"],
              },
            },
            required: ["version", "shape", "strategy", "source_columns", "fields"],
          },
        ],
      },
    },
    required: [
      "usable",
      "compatibility_reason",
      "selected_target_column",
      "target_selection_reason",
      "target_selection_confidence",
      "target_candidates",
      "ambiguity_warnings",
      "normalization_proposal",
    ],
  };
}

function buildOpenAINormalizationPrompt(candidate, context, sourceSchema) {
  return [
    "Infer a deterministic normalization recipe for a single Hugging Face dataset so a text-only SFT backend can train on it.",
    "This backend v1 supports only single-target classification SFT jobs.",
    "Do not invent columns.",
    "Use the native dataset field names exactly as provided.",
    "Return usable=false when the schema cannot support a meaningful deterministic text or prompt/completion normalization.",
    "Use template_synthesis only when direct column mapping is impossible.",
    "Avoid likely identifier columns such as ticket ids, customer ids, phone numbers, and other obvious PII unless they are essential.",
    "When multiple plausible label columns exist, choose one target column and record ambiguity_warnings instead of failing.",
    `Objective: ${context.analysis?.domain_summary || ""}`,
    `Task spec: ${JSON.stringify(context.task_spec ?? {})}`,
    `Mapped task types: ${JSON.stringify(context.analysis?.mapped_task_types ?? [])}`,
    `Dataset id: ${candidate.id}`,
    `Description: ${candidate.description ?? ""}`,
    `Warnings: ${JSON.stringify(candidate.warnings ?? [])}`,
    `Schema signals: ${JSON.stringify(candidate.schema_signals ?? [])}`,
    `Available columns: ${JSON.stringify(sourceSchema?.available_columns ?? [])}`,
    `Sample rows: ${JSON.stringify(sourceSchema?.sample_rows ?? [])}`,
  ].join("\n");
}

async function callOpenAINormalizationPlanner(candidate, context, sourceSchema) {
  await ensureDotEnvLoaded();
  const apiKey = normalizeOptionalString(process.env.OPENAI_API_KEY);
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required to infer dataset normalization proposals.");
  }

  const model = normalizeOptionalString(process.env.OPENAI_MODEL) ?? DEFAULT_OPENAI_MODEL;
  const prompt = buildOpenAINormalizationPrompt(candidate, context, sourceSchema);
  logMultiline(`openai normalization prompt for ${candidate.id}`, prompt);

  const response = await fetchJson(OPENAI_RESPONSES_API_URL, {
    method: "POST",
    timeoutMs: 60_000,
    maxRetries: 3,
    requestLabel: `OpenAI normalization request for ${candidate.id}`,
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
            "You infer deterministic dataset normalization recipes for text-only SFT training. Return only the requested JSON and never invent schema fields.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "hf_dataset_normalization",
          schema: getNormalizationProposalSchema(),
          strict: true,
        },
      },
    }),
  });

  const outputText = extractOpenAIOutputText(response);
  if (!outputText) {
    throw new Error("OpenAI returned an empty normalization response.");
  }

  try {
    return JSON.parse(outputText);
  } catch (error) {
    throw new Error(
      `OpenAI returned invalid JSON for dataset normalization: ${error instanceof Error ? error.message : "unknown parse error"
      }`,
    );
  }
}

async function rankCandidatesWithOpenAI(candidates, context) {
  if (candidates.length === 0) {
    return [];
  }

  await ensureDotEnvLoaded();
  const apiKey = normalizeOptionalString(process.env.OPENAI_API_KEY);
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required to rank fetched Hugging Face datasets.");
  }

  const model = normalizeOptionalString(process.env.OPENAI_MODEL) ?? DEFAULT_OPENAI_MODEL;
  const rankingInput = buildOpenAIRankingInput(candidates, context);
  const prompt = buildOpenAIRankingPrompt(context, rankingInput);
  logInfo(`selecting recommended datasets from ${candidates.length} candidates with OpenAI model ${model}`);
  logMultiline("openai dataset-selection prompt", prompt);

  const response = await fetchJson(OPENAI_RESPONSES_API_URL, {
    method: "POST",
    timeoutMs: 60_000,
    maxRetries: 3,
    requestLabel: "OpenAI dataset ranking request",
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
            "You select the smallest useful set of already-fetched Hugging Face datasets for language model post-training. Use only the candidate datasets provided. Return only the requested JSON structure.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "hf_dataset_recommendations",
          schema: getOpenAIRankingSchema(candidates.length),
          strict: true,
        },
      },
    }),
  });

  logJson("openai dataset-selection response metadata", {
    id: response.id ?? null,
    status: response.status ?? null,
    model: response.model ?? null,
    usage: response.usage ?? null,
  });

  const outputText = extractOpenAIOutputText(response);
  if (!outputText) {
    throw new Error("OpenAI returned an empty dataset-selection response.");
  }
  logMultiline("openai dataset-selection output text", outputText);

  let ranking;
  try {
    ranking = JSON.parse(outputText);
  } catch (error) {
    throw new Error(
      `OpenAI returned invalid JSON for dataset selection: ${error instanceof Error ? error.message : "unknown parse error"
      }`,
    );
  }

  logJson("openai parsed dataset-selection", ranking);
  const ranked = mergeOpenAIRanking(candidates, ranking, context);

  logJson(
    "recommended datasets selected by OpenAI",
    ranked.map((candidate) => ({
      dataset: candidate.dataset,
      score: candidate.score,
      matched_queries: candidate.matched_queries,
      warnings: candidate.warnings,
    })),
  );

  return ranked;
}

function buildOpenAIRankingInput(candidates, context) {
  return candidates.map((candidate) => ({
    dataset: candidate.id,
    source_url: candidate.source_url,
    matched_queries: candidate.matched_queries,
    matched_tasks: candidate.matched_tasks,
    description: candidate.description ?? "",
    downloads: candidate.downloads,
    likes: candidate.likes,
    num_rows: candidate.num_rows,
    license: candidate.license,
    splits: candidate.splits,
    schema_signals: candidate.schema_signals,
    viewer_accessible: candidate.viewer_accessible,
    size_partial: candidate.size_partial,
    compatibility_status: candidate.compatibility_status,
    compatibility_reason: candidate.compatibility_reason,
    normalization_source: candidate.normalization_source,
    compatible_methods: candidate.compatible_methods,
    normalization_shape: candidate.normalization_proposal?.shape ?? null,
    normalization_strategy: candidate.normalization_proposal?.strategy ?? null,
    source_columns: candidate.normalization_proposal?.source_columns ?? [],
    selected_target_column: candidate.selected_target_column ?? null,
    target_selection_reason: candidate.target_selection_reason ?? null,
    target_selection_confidence: candidate.target_selection_confidence ?? null,
    target_candidates: candidate.target_candidates ?? [],
    ambiguity_warnings: candidate.ambiguity_warnings ?? [],
    warnings: candidate.warnings ?? [],
    tags: (candidate.tags ?? []).slice(0, 12),
    feature_names: candidate.source_schema?.available_columns?.slice(0, 20) ?? [],
    sample_rows: candidate.source_schema?.sample_rows ?? [],
  }));
}

function buildOpenAIRankingPrompt(context, rankingInput) {
  return [
    "Choose the smallest useful set of already-fetched Hugging Face datasets for the user's post-training use case.",
    "Use only the provided candidates. Do not invent new dataset ids.",
    "Return 1 dataset when a single clear winner is enough.",
    "Return 2-3 datasets only when combining them would materially improve coverage, robustness, or task fit.",
    "Do not return more than 3 datasets.",
    "Prefer domain relevance, task/schema fit, target row-band fit, public usability, license clarity, and overall data usefulness.",
    "Only choose candidates whose compatibility_status is compatible.",
    "Prefer candidates with an explicit selected_target_column and lower ambiguity.",
    "Prefer deterministic direct normalization over template_synthesis when task fit is otherwise similar.",
    "The candidates have already been fetched from Hugging Face; you are only choosing the recommended set.",
    `Task spec: ${JSON.stringify(context.task_spec)}`,
    `Analysis: ${JSON.stringify(context.analysis)}`,
    `Search queries: ${JSON.stringify(context.search_queries)}`,
    `Recommendation guidance: ${JSON.stringify(context.recommendation_guidance)}`,
    `Target row band: ${JSON.stringify(context.target_row_band)}`,
    `Minimum row floor: ${context.min_rows_floor}`,
    `Candidates: ${JSON.stringify(rankingInput)}`,
  ].join("\n");
}

function getOpenAIRankingSchema(candidateCount) {
  const maxRecommendations = Math.max(1, Math.min(candidateCount, 3));
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      recommended_datasets: {
        type: "array",
        minItems: 1,
        maxItems: maxRecommendations,
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            dataset: { type: "string" },
            score: { type: "integer", minimum: 0, maximum: 100 },
            why: { type: "string" },
            warnings: {
              type: "array",
              items: { type: "string" },
            },
          },
          required: ["dataset", "score", "why", "warnings"],
        },
      },
    },
    required: ["recommended_datasets"],
  };
}

function mergeOpenAIRanking(candidates, ranking, context) {
  const candidateMap = new Map(candidates.map((candidate) => [candidate.id, candidate]));
  const merged = [];
  const seen = new Set();

  for (const rankedCandidate of ranking.recommended_datasets ?? []) {
    const candidate = candidateMap.get(rankedCandidate.dataset);
    if (!candidate || seen.has(rankedCandidate.dataset)) {
      continue;
    }

    merged.push({
      dataset: candidate.id,
      source_url: candidate.source_url,
      score: rankedCandidate.score,
      why: rankedCandidate.why,
      matched_queries: candidate.matched_queries,
      mapped_task_types: context.analysis.mapped_task_types,
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
      selected_target_column: candidate.selected_target_column ?? null,
      target_selection_reason: candidate.target_selection_reason ?? null,
      target_selection_confidence: candidate.target_selection_confidence ?? null,
      target_candidates: candidate.target_candidates ?? [],
      ambiguity_warnings: candidate.ambiguity_warnings ?? [],
      source_schema: candidate.source_schema,
      preferred_dataset_config: candidate.preferred_dataset_config ?? null,
      preferred_train_split: candidate.preferred_train_split ?? "train",
      preferred_eval_split: candidate.preferred_eval_split ?? null,
      warnings: uniqueStrings([...(candidate.warnings ?? []), ...(rankedCandidate.warnings ?? [])]),
    });
    seen.add(rankedCandidate.dataset);
  }

  if (merged.length === 0) {
    throw new Error("OpenAI dataset-selection response did not contain any valid candidate dataset ids.");
  }

  return merged;
}

function buildRecommendationGuidance(rankedDatasets, plan) {
  const mergedWarnings = uniqueStrings([
    ...plan.recommendation_guidance.warnings,
    ...rankedDatasets.flatMap((dataset) => dataset.warnings.slice(0, 2)),
  ]).slice(0, 6);
  const selectedCount = rankedDatasets.length;

  return {
    ideal_dataset_count: selectedCount
      ? selectedCount
      : (plan.recommendation_guidance.ideal_dataset_count ??
        inferIdealDatasetCount(rankedDatasets)),
    target_total_rows: plan.recommendation_guidance.target_total_rows,
    mixing_strategy:
      selectedCount > 0
        ? inferMixingStrategy(rankedDatasets, plan)
        : plan.recommendation_guidance.mixing_strategy || inferMixingStrategy(rankedDatasets, plan),
    warnings: mergedWarnings,
  };
}

function inferIdealDatasetCount(rankedDatasets) {
  if (rankedDatasets.length === 0) return 1;
  return Math.min(3, rankedDatasets.length);
}

function inferMixingStrategy(rankedDatasets, plan) {
  if (rankedDatasets.length === 0) {
    return "No strong public candidates were found, so start with the top GPT-generated search query and refine the prompt or filters.";
  }
  if (rankedDatasets.length === 1) {
    return "Use the selected dataset as the primary source and only add a second dataset if GPT identifies a clear coverage gap.";
  }
  if (rankedDatasets.length === 2) {
    return "Use a 70/30 mix: primary dataset first, then one complementary dataset with a different schema or source profile.";
  }
  return "Use a 70/20/10 mix across the best direct match, one broader supporting dataset, and one smaller edge-case set.";
}

function parseTargetRowBand(target) {
  const normalized = String(target ?? "")
    .toUpperCase()
    .replace(/[,\s]+/g, "")
    .replace(/[–—]/g, "-")
    .replace(/ROWS?$/, "");
  if (!normalized) return null;

  const plusMatch = normalized.match(/^([0-9.]+)([KMB]?)\+$/);
  if (plusMatch) {
    return [parseScaledNumber(plusMatch[1], plusMatch[2]), Number.POSITIVE_INFINITY];
  }

  const rangeMatch = normalized.match(/^([0-9.]+)([KMB]?)-([0-9.]+)([KMB]?)$/);
  if (rangeMatch) {
    return [
      parseScaledNumber(rangeMatch[1], rangeMatch[2]),
      parseScaledNumber(rangeMatch[3], rangeMatch[4]),
    ];
  }

  return null;
}

function parseScaledNumber(value, suffix) {
  const number = Number(value);
  const scale =
    suffix === "K" ? 1_000 : suffix === "M" ? 1_000_000 : suffix === "B" ? 1_000_000_000 : 1;
  return Number.isFinite(number) ? number * scale : 0;
}

function dedupeRankedDatasets(rankedDatasets) {
  const selected = [];

  for (const candidate of rankedDatasets) {
    if (selected.some((existing) => areNearDuplicateDatasets(existing.dataset, candidate.dataset))) {
      continue;
    }
    selected.push(candidate);
  }

  return selected;
}

function areNearDuplicateDatasets(leftId, rightId) {
  const left = extractDatasetIdentity(leftId);
  const right = extractDatasetIdentity(rightId);
  const leftNormalizedRepo = normalizeRepoIdentity(left.repo);
  const rightNormalizedRepo = normalizeRepoIdentity(right.repo);

  if (!leftNormalizedRepo || !rightNormalizedRepo) {
    return false;
  }

  if (leftNormalizedRepo === rightNormalizedRepo) {
    return true;
  }

  const leftTokens = datasetIdentityTokens(left.repo);
  const rightTokens = datasetIdentityTokens(right.repo);
  const sharedTokens = leftTokens.filter((token) => rightTokens.includes(token));

  if (!sharedTokens.length) {
    return false;
  }

  const overlap = sharedTokens.length / Math.max(leftTokens.length || 1, rightTokens.length || 1);
  if (left.owner === right.owner && overlap >= 0.75) {
    return true;
  }

  return sharedTokens.length >= 3 && overlap >= 0.85;
}

function extractDatasetIdentity(datasetId) {
  const [owner = "", repo = ""] = String(datasetId ?? "").toLowerCase().split("/");
  return { owner, repo };
}

function datasetIdentityTokens(repo) {
  return uniqueStrings(
    normalizeText(repo)
      .split(/[\s_-]+/)
      .filter(
        (token) =>
          token &&
          token.length > 2 &&
          !["data", "dataset", "datasets", "cleaned", "processed", "archive", "raw", "exp"].includes(
            token,
          ),
      ),
  );
}

function normalizeRepoIdentity(repo) {
  return normalizeText(repo).replace(/[\s_-]+/g, "");
}

function matchesTask(tags, task) {
  return (tags ?? []).some((tag) => {
    const lowerTag = String(tag).toLowerCase();
    return (
      lowerTag === task ||
      lowerTag.includes(task) ||
      lowerTag === `task_categories:${task}` ||
      lowerTag.includes(`task_categories:${task}`)
    );
  });
}

function isTransientStatusError(error) {
  return Boolean(error && typeof error === "object" && TRANSIENT_STATUS_CODES.has(error.status));
}

function getRetryDelayMs(attempt, response) {
  const retryAfterHeader = response?.headers?.get?.("retry-after");
  const retryAfterSeconds = Number(retryAfterHeader);
  if (Number.isFinite(retryAfterSeconds) && retryAfterSeconds >= 0) {
    return Math.min(retryAfterSeconds * 1000, 10_000);
  }

  return Math.min(1000 * 2 ** attempt, 10_000);
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

      try {
        results[currentIndex] = await mapper(items[currentIndex], currentIndex);
      } catch (error) {
        results[currentIndex] = {
          ...items[currentIndex],
          warnings: uniqueStrings([
            ...(items[currentIndex].warnings ?? []),
            error instanceof Error ? error.message : "Unknown enrichment error.",
          ]),
        };
      }
    }
  }

  const workers = Array.from({ length: Math.max(1, Math.min(concurrency, items.length)) }, () => worker());
  await Promise.all(workers);
  return results;
}

function normalizeText(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s/-]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(value) {
  return uniqueStrings(
    normalizeText(value)
      .split(/[\s/-]+/)
      .map((token) => token.trim())
      .filter((token) => token.length > 1),
  );
}

function overlapScore(tokens, haystack) {
  if (!tokens.length || !haystack) return 0;

  let matched = 0;
  for (const token of tokens) {
    if (haystack.includes(token)) {
      matched += 1;
    }
  }

  return matched / tokens.length;
}

function normalizePositiveInt(value) {
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? number : null;
}

function normalizeNonNegativeInt(value) {
  const number = Number(value);
  return Number.isInteger(number) && number >= 0 ? number : null;
}

function toNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : 0;
}

function logInfo(message) {
  if (activeLogger && typeof activeLogger.emit === "function") {
    activeLogger.emit({ source: "hf-dataset-recommender", level: "info", message });
    return;
  }
  console.log(`${LOG_PREFIX} ${message}`);
}

function logJson(label, value) {
  if (activeLogger && typeof activeLogger.emit === "function") {
    activeLogger.emit({
      source: "hf-dataset-recommender",
      level: "info",
      message: label,
      data: value,
    });
    return;
  }
  console.log(`${LOG_PREFIX} ${label}:`);
  console.log(JSON.stringify(value, null, 2));
}

function logMultiline(label, value) {
  if (activeLogger && typeof activeLogger.emit === "function") {
    activeLogger.emit({
      source: "hf-dataset-recommender",
      level: "info",
      message: label,
      data: String(value ?? ""),
    });
    return;
  }
  console.log(`${LOG_PREFIX} ${label}:`);
  console.log(String(value ?? ""));
}

async function writeRankedDatasetsDebugFile(recommendedDatasets, debugOutputPath) {
  const targetPath = debugOutputPath ?? RANKED_DATASETS_DEBUG_PATH;
  await writeFile(targetPath, JSON.stringify(recommendedDatasets, null, 2));
  logInfo(
    `wrote recommended datasets JSON to ${typeof targetPath === "string" ? targetPath : targetPath.pathname
    }`,
  );
}

function taskSearchLabel(task) {
  switch (task) {
    case "text-classification":
      return "classification";
    case "question-answering":
      return "question answering";
    case "token-classification":
      return "token classification";
    case "text-generation":
      return "instruction";
    default:
      return String(task ?? "").replace(/-/g, " ");
  }
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
    // Ignore missing .env files and rely on existing environment variables.
  }
}

function parseCliArgs(argv) {
  const parsed = {};

  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index];
    const next = argv[index + 1];

    if (current === "--input") {
      parsed.inputPath = next;
      index += 1;
    } else if (current === "--json") {
      parsed.inlineJson = next;
      index += 1;
    }
  }

  return parsed;
}

async function loadCliInput(args) {
  if (args.inlineJson) {
    return JSON.parse(args.inlineJson);
  }
  if (args.inputPath) {
    const fileContents = await readFile(args.inputPath, "utf8");
    return JSON.parse(fileContents);
  }
  throw new Error(
    "Usage: node backend/hf-dataset-recommender.mjs --input input.json or --json '{\"description\":\"...\"}' or --json '{\"search_queries\":[...]}'",
  );
}

async function runCli() {
  try {
    const args = parseCliArgs(process.argv.slice(2));
    const input = await loadCliInput(args);
    const result = await recommendDatasets(input);
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await runCli();
}
