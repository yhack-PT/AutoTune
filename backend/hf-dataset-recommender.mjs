import process from "node:process";
import { readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

import {
  extractRowsFromPreview,
  filterSourceSchemaForTextOnlyTraining,
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
import { emitUiProgress } from "../src/lib/posttraining-progress.mjs";

const HUB_API_URL = "https://huggingface.co/api/datasets";
const DATASET_VIEWER_URL = "https://datasets-server.huggingface.co";
const OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses";
const RANKED_DATASETS_DEBUG_PATH = new URL("./datasets.json", import.meta.url);
const HUB_SEARCH_LIMIT = 25;
const SHORTLIST_LIMIT = 20;
const REQUEST_TIMEOUT_MS = 8000;
const ENRICHMENT_CONCURRENCY = 5;
const DEFAULT_OPENAI_MODEL = "gpt-5-mini";
const DEFAULT_DATASET_SELECTION_MODEL = "gpt-5.4";
const DEFAULT_SMALLER_TIME_BUDGET_RETRIES = 3;
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

const SUPPORTED_TASKS_ERROR_MESSAGE =
  "Only single-target classification and generation SFT jobs are supported right now.";

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
    const inputIsSearchPlan = isSearchPlanInput(input);
    logInfo(
      `starting recommendDatasets in ${inputIsSearchPlan ? "search-plan" : "text"} mode`,
    );
    logJson("input", input);

    let normalizedPlan = inputIsSearchPlan
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
            SUPPORTED_TASKS_ERROR_MESSAGE,
        },
      };
      if (!options.skipDebugWrite) {
        await writeRankedDatasetsDebugFile(recommendation, options.debugOutputPath);
      }
      throw new RecommendationFailure(recommendation.fatal_error.message, recommendation);
    }

    const maxSmallerTimeBudgetRetries =
      inputIsSearchPlan
        ? 0
        : normalizeNonNegativeInt(options.maxSmallerTimeBudgetRetries) ??
          DEFAULT_SMALLER_TIME_BUDGET_RETRIES;
    const retryWarnings = [];

    for (let attemptIndex = 0; attemptIndex <= maxSmallerTimeBudgetRetries; attemptIndex += 1) {
      if (!normalizedPlan.task_spec.supported) {
        const recommendation = {
          analysis: normalizedPlan.analysis,
          task_spec: normalizedPlan.task_spec,
          search_queries: normalizedPlan.search_queries,
          ranking_criteria: normalizedPlan.ranking_criteria,
          recommendation_guidance: {
            ...normalizedPlan.recommendation_guidance,
            warnings: uniqueStrings([
              ...(retryWarnings ?? []),
              ...(normalizedPlan.recommendation_guidance?.warnings ?? []),
            ]),
          },
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
              SUPPORTED_TASKS_ERROR_MESSAGE,
          },
        };
        if (!options.skipDebugWrite) {
          await writeRankedDatasetsDebugFile(recommendation, options.debugOutputPath);
        }
        throw new RecommendationFailure(recommendation.fatal_error.message, recommendation);
      }

      logJson("normalized plan", normalizedPlan);
      const attempt = await executeRecommendationAttempt(normalizedPlan, options);

      if (attempt.compatibleCandidates.length > 0) {
        const recommendedDatasets = dedupeRankedDatasets(
          await (
            typeof options.rankCandidates === "function"
              ? options.rankCandidates(attempt.compatibleCandidates, attempt.context)
              : rankCandidatesWithOpenAI(attempt.compatibleCandidates, attempt.context)
          ),
        );
        logInfo(`returning ${recommendedDatasets.length} recommended datasets`);

        const recommendationGuidance = buildRecommendationGuidance(recommendedDatasets, normalizedPlan);
        const result = {
          analysis: normalizedPlan.analysis,
          task_spec: normalizedPlan.task_spec,
          search_queries: normalizedPlan.search_queries,
          ranking_criteria: normalizedPlan.ranking_criteria,
          recommendation_guidance:
            retryWarnings.length > 0
              ? {
                ...recommendationGuidance,
                warnings: uniqueStrings([...(retryWarnings ?? []), ...(recommendationGuidance.warnings ?? [])]),
              }
              : recommendationGuidance,
          compatibility_summary: buildCompatibilitySummary(attempt.publicCandidates),
          ranked_datasets: attempt.rankedDatasets,
          recommended_datasets: recommendedDatasets,
        };
        if (!options.skipDebugWrite) {
          await writeRankedDatasetsDebugFile(result, options.debugOutputPath);
        }
        return result;
      }

      if (attemptIndex === maxSmallerTimeBudgetRetries) {
        const emptyGuidance = buildRecommendationGuidance([], normalizedPlan);
        const recommendation = {
          analysis: normalizedPlan.analysis,
          task_spec: normalizedPlan.task_spec,
          search_queries: normalizedPlan.search_queries,
          ranking_criteria: normalizedPlan.ranking_criteria,
          recommendation_guidance: {
            ...emptyGuidance,
            warnings: uniqueStrings([
              ...(retryWarnings ?? []),
              ...(emptyGuidance.warnings ?? []),
            ]),
          },
          compatibility_summary: buildCompatibilitySummary(attempt.publicCandidates),
          ranked_datasets: attempt.rankedDatasets,
          recommended_datasets: [],
          fatal_error: {
            code: "no_compatible_candidates",
            message:
              attemptIndex > 0
                ? "No ranked dataset candidates could be normalized into a backend-supported training shape during recommendation, even after retrying with a smaller inferred run-time budget."
                : "No ranked dataset candidates could be normalized into a backend-supported training shape during recommendation.",
          },
        };
        if (!options.skipDebugWrite) {
          await writeRankedDatasetsDebugFile(recommendation, options.debugOutputPath);
        }
        throw new RecommendationFailure(recommendation.fatal_error.message, recommendation);
      }

      const previousTimeBudget = normalizeOptionalString(normalizedPlan.analysis?.quality_tier_strategy);
      const retryWarning = buildSmallerTimeBudgetRetryWarning(previousTimeBudget);
      retryWarnings.push(retryWarning);
      logUiProgress("I'm narrowing the search to find a better match");
      logInfo("no compatible candidates found; retrying planning with a smaller inferred run-time budget");
      normalizedPlan = await createPlanFromUserInputs(input, options, {
        retry_reason: "no_compatible_candidates",
        smaller_time_budget: true,
        previous_plan: normalizedPlan,
        previous_time_budget: previousTimeBudget,
      });
    }
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

async function createPlanFromUserInputs(input, options = {}, planningHints = {}) {
  const normalizedInput = validateUserInputs(input);
  if (typeof options.planGenerator === "function") {
    const generatedPlan = await options.planGenerator(normalizedInput, planningHints);
    return coerceTaskSpecForRawInput(validatePlan(generatedPlan), normalizedInput);
  }

  await ensureDotEnvLoaded();
  logJson("validated text input", normalizedInput);
  logUiProgress("I'm figuring out what kind of data this model needs");

  const apiKey = normalizeOptionalString(process.env.OPENAI_API_KEY);
  if (!apiKey) {
    throw new Error(
      "OPENAI_API_KEY is required to generate a search plan from a text description.",
    );
  }

  const model = normalizeOptionalString(process.env.OPENAI_MODEL) ?? DEFAULT_OPENAI_MODEL;
  const prompt = buildOpenAIPlanningPrompt(normalizedInput, planningHints);
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

async function executeRecommendationAttempt(plan, options = {}) {
  const context = buildContext(plan);
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

  return {
    context,
    publicCandidates,
    rankedDatasets,
    compatibleCandidates,
  };
}

function buildSmallerTimeBudgetRetryWarning(previousTimeBudget) {
  if (previousTimeBudget) {
    return `Initial time-budget estimate (${previousTimeBudget}) did not yield compatible public datasets; retried with a smaller inferred run-time budget.`;
  }
  return "Initial time-budget estimate did not yield compatible public datasets; retried with a smaller inferred run-time budget.";
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

function isClassificationTaskSpec(taskSpec) {
  return normalizeOptionalString(taskSpec?.task_family) === "classification";
}

function isGenerationTaskSpec(taskSpec) {
  return normalizeOptionalString(taskSpec?.task_family) === "generation";
}

function buildClassificationObjectiveSummary(rawInput, selectedTargetFocus) {
  const domain = normalizeOptionalString(rawInput?.domain);
  if (domain) {
    return `Classify ${domain} examples by ${selectedTargetFocus}.`;
  }
  return `Classify examples by ${selectedTargetFocus}.`;
}

function inferCompatibilityForTask(taskSpec, normalizationProposal, selectedTargetColumn, fallbackReason) {
  const shape = normalizeOptionalString(normalizationProposal?.shape);
  const normalizedTargetColumn = normalizeOptionalString(selectedTargetColumn);

  if (isClassificationTaskSpec(taskSpec)) {
    if (shape === "prompt_completion" && normalizedTargetColumn) {
      return {
        compatible: true,
        reason: fallbackReason || "Candidate is compatible with classification SFT requirements.",
      };
    }
    if (!shape) {
      return {
        compatible: false,
        reason: fallbackReason || "No supported normalization proposal could be inferred for this dataset.",
      };
    }
    if (shape !== "prompt_completion") {
      return {
        compatible: false,
        reason: `Candidate normalizes to '${shape}', but classification SFT requires prompt_completion examples.`,
      };
    }
    return {
      compatible: false,
      reason: "Candidate is missing a selected target column for classification SFT.",
    };
  }

  if (isGenerationTaskSpec(taskSpec)) {
    if (shape === "text") {
      return {
        compatible: true,
        reason: fallbackReason || "Candidate is compatible with raw-text generation SFT requirements.",
      };
    }
    if (shape === "prompt_completion") {
      return {
        compatible: true,
        reason:
          fallbackReason ||
          "Candidate is compatible with paired prompt/completion generation SFT requirements.",
      };
    }
    return {
      compatible: false,
      reason:
        fallbackReason ||
        "No supported raw-text normalization proposal could be inferred for this dataset.",
    };
  }

  return {
    compatible: false,
    reason: fallbackReason || "Task family is not supported by the current backend.",
  };
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
      objective_summary: buildClassificationObjectiveSummary(normalizedInput, selectedTargetFocus),
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
    return {
      description,
      domain: normalizeOptionalString(input.domain),
      useCase: normalizeOptionalString(input.useCase),
    };
  }

  const domain = normalizeOptionalString(input.domain);
  const useCase = normalizeOptionalString(input.useCase);

  if (!domain) {
    throw new Error("Legacy raw input requires `domain`, or provide a single `description` field.");
  }
  if (!useCase) {
    throw new Error("Legacy raw input requires `useCase`, or provide a single `description` field.");
  }

  return {
    description: buildLegacyDescription({ domain, useCase }),
    domain,
    useCase,
  };
}

function buildLegacyDescription(input) {
  return [`Domain: ${input.domain}`, `Use case: ${input.useCase}`].join("\n");
}

function buildOpenAIPlanningPrompt(input, planningHints = {}) {
  const retryLines = planningHints.smaller_time_budget
    ? [
      "Retry context: the previous inferred time budget did not yield compatible public datasets.",
      planningHints.previous_time_budget
        ? `Previous time-budget summary: ${planningHints.previous_time_budget}`
        : null,
      "Retry with a smaller inferred run-time budget than before.",
      "Reduce the required data volume and simplify the plan enough to improve the odds of finding compatible datasets.",
      "Prefer more common, public, clearly structured text datasets if needed.",
    ]
    : [];

  const promptLines = [
    "Create a Hugging Face dataset search plan for post-training a language model.",
    "This backend v1 supports two SFT task families.",
    "classification = finite-label prediction with prompt/completion examples and one selected target column.",
    "generation = open-ended behavior, source-to-target generation, summarization, or domain adaptation using either raw text datasets or prompt/completion pairs.",
    `User request: ${input.description}`,
    ...retryLines,
    "Infer a reasonable end-to-end run-time budget directly from the user's request.",
    "If the user states a time preference or deadline, honor it when possible.",
    "If the user does not specify a time, default to an 8-hour run budget.",
    "Return task_spec, analysis, search_queries, ranking_criteria, and recommendation_guidance.",
    "search_queries must contain concise Hugging Face search strings, usually 2-6 words, like something typed directly into the Hugging Face search bar.",
    "Use task_filter values only from: text-classification, question-answering, summarization, text-generation, translation, conversational, token-classification, or null.",
    "Use sort values only from: downloads, likes, trending, created.",
    "Set min_rows and recommendation guidance based on the inferred time budget and task complexity.",
    "Set data_format_needed to exactly one of: instruction, completion, preference, raw_text, mixed.",
    "Set analysis.quality_tier_strategy to a concise wall-clock time-budget summary for the recommended run.",
    "Do not use a numeric 1-5 tier or score.",
    "Do not describe analysis.quality_tier_strategy in terms of row counts, corpus size, or number of datasets.",
    "Keep mapped_task_types narrowly focused on the main fine-tuning objective, usually 1-2 task types.",
    "Keep warnings focused on practical dataset-selection risks.",
    "Use classification only when the user wants explicit label prediction or category assignment.",
    "Use generation for tutor, expert, writer, assistant, style, tone, behavior, summarization, translation, or other open-ended response tasks.",
    "For classification, set task_spec.target_policy to 'single_target' and output_shape_preference to 'prompt_completion'.",
    "For generation, set task_spec.target_policy to 'none'. Use output_shape_preference 'prompt_completion' for source-to-target, instruction-following, or summarization tasks, and 'text' for open-ended style, tone, role, or domain adaptation tasks.",
    "If the request mentions multiple classification targets, choose one primary target to optimize for, keep task_spec.supported=true, and record a warning plus the selected_target_focus.",
    "Do not force open-ended behavior requests into classification.",
    "Set task_spec.supported=false only when the request cannot reasonably be handled as either single-target classification or generation SFT.",
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
            enum: ["classification", "generation", "unsupported"],
          },
          target_policy: {
            type: "string",
            enum: ["single_target", "none", "unsupported"],
          },
          output_shape_preference: {
            type: "string",
            enum: ["prompt_completion", "text", "unsupported"],
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
  logUiProgress("I'm searching through different datasets that could fit this request");

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

        const metadata = buildDiscoveredCandidateSeed(dataset, {
          matchedQueries: [query.search],
          matchedTasks: query.task_filter ? [query.task_filter] : [],
        });

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

        candidateMap.set(id, metadata);
      }
    }),
  );

  return [...candidateMap.values()].slice(0, SHORTLIST_LIMIT);
}

function buildDiscoveredCandidateSeed(dataset, options = {}) {
  const id = dataset?.id ?? dataset?._id ?? dataset?.name;
  if (!id) {
    return null;
  }

  return {
    id,
    source_url: `https://huggingface.co/datasets/${id}`,
    description:
      dataset?.description ??
      dataset?.cardData?.description ??
      dataset?.cardData?.pretty_name ??
      "",
    downloads: toNumber(dataset?.downloads),
    likes: toNumber(dataset?.likes),
    gated: Boolean(dataset?.gated),
    private: Boolean(dataset?.private),
    tags: Array.isArray(dataset?.tags) ? dataset.tags : [],
    cardData: dataset?.cardData ?? {},
    matched_queries: uniqueStrings(
      Array.isArray(options.matchedQueries) ? options.matchedQueries.map((value) => String(value).trim()).filter(Boolean) : [],
    ),
    matched_tasks: uniqueStrings(
      Array.isArray(options.matchedTasks) ? options.matchedTasks.map((value) => String(value).trim()).filter(Boolean) : [],
    ),
  };
}

export async function resolveDatasetOverrideCandidate(datasetId, planLikeContext, options = {}) {
  const normalizedDatasetId = normalizeOptionalString(datasetId);
  if (!normalizedDatasetId) {
    throw new Error("dataset override id must be a non-empty string.");
  }

  const normalizedPlan = {
    analysis: {
      ...DEFAULT_ANALYSIS,
      ...(planLikeContext?.analysis && typeof planLikeContext.analysis === "object" ? planLikeContext.analysis : {}),
    },
    task_spec: {
      ...DEFAULT_TASK_SPEC,
      ...(planLikeContext?.task_spec && typeof planLikeContext.task_spec === "object" ? planLikeContext.task_spec : {}),
    },
    search_queries:
      Array.isArray(planLikeContext?.search_queries) && planLikeContext.search_queries.length > 0
        ? planLikeContext.search_queries.map((query) => normalizeSearchQuery(query)).filter((query) => query.search)
        : [
          normalizeSearchQuery({
            search: normalizedDatasetId,
            task_filter: null,
            sort: "downloads",
            min_rows: 0,
            intent: "Resolve an explicitly overridden dataset.",
          }),
        ],
    ranking_criteria:
      Array.isArray(planLikeContext?.ranking_criteria) && planLikeContext.ranking_criteria.length > 0
        ? planLikeContext.ranking_criteria.map((criterion) => normalizeCriterion(criterion))
        : DEFAULT_RANKING_CRITERIA,
    recommendation_guidance: {
      ...DEFAULT_GUIDANCE,
      ...(planLikeContext?.recommendation_guidance && typeof planLikeContext.recommendation_guidance === "object"
        ? planLikeContext.recommendation_guidance
        : {}),
    },
  };
  const context = buildContext(normalizedPlan);

  const matchedQueries = uniqueStrings(
    normalizedPlan.search_queries.map((query) => normalizeOptionalString(query.search)).filter(Boolean),
  );
  const matchedTasks = uniqueStrings(
    Array.isArray(normalizedPlan.analysis.mapped_task_types)
      ? normalizedPlan.analysis.mapped_task_types.map((value) => String(value).trim()).filter(Boolean)
      : [],
  );

  let candidateSeed = {
    id: normalizedDatasetId,
    source_url: `https://huggingface.co/datasets/${normalizedDatasetId}`,
    description: "",
    downloads: 0,
    likes: 0,
    gated: false,
    private: false,
    tags: [],
    cardData: {},
    matched_queries: matchedQueries,
    matched_tasks: matchedTasks,
  };

  const exactMatch = await searchHubDatasets({
    search: normalizedDatasetId,
    task_filter: null,
    sort: "downloads",
    min_rows: 0,
    intent: "Resolve an explicitly overridden dataset.",
  })
    .then((results) => results.find((dataset) => (dataset.id ?? dataset._id ?? dataset.name) === normalizedDatasetId) ?? null)
    .catch(() => null);

  if (exactMatch) {
    candidateSeed = buildDiscoveredCandidateSeed(exactMatch, {
      matchedQueries,
      matchedTasks,
    }) ?? candidateSeed;
  }

  const [enrichedCandidate] = await enrichCandidates([candidateSeed], context, options);
  return enrichedCandidate ?? null;
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

    let sizeRequestFailed = false;
    const sizePayload = await fetchViewerEndpoint("/size", { dataset: candidate.id }).catch(() => {
      sizeRequestFailed = true;
      return null;
    });
    const fallbackNumRows = inferNumRowsFromHubCardData(candidate.cardData);
    const sizeInfo = parseSizeInfo(sizePayload, fallbackNumRows);
    if (sizeRequestFailed && fallbackNumRows > 0) {
      logInfo(
        `HF viewer /size unavailable for ${candidate.id}; using Hub card metadata row count (${fallbackNumRows}).`,
      );
    }
    if (sizeInfo.partial) {
      warnings.push("Dataset size is partial or approximate in the dataset viewer.");
    }

    const splitsPayload = await fetchViewerEndpoint("/splits", { dataset: candidate.id }).catch(() => null);
    const splitsInfo = parseSplitsInfo(splitsPayload);
    const previewTarget = choosePreviewTarget(splitsInfo.rawSplits);
    const preferredTrainSplit = selectPreferredTrainSplit(splitsInfo.rawSplits);
    const preferredEvalSplit = selectPreferredEvalSplit(splitsInfo.rawSplits);
    const sourceSplits = resolveSourceSplits(splitsInfo.rawSplits, preferredTrainSplit);

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
    const textOnlySourceSchema = filterSourceSchemaForTextOnlyTraining(featureNames, sampleRows);
    const availableColumns = textOnlySourceSchema.available_columns;
    const sourceSchema = {
      available_columns: availableColumns,
      sample_rows: textOnlySourceSchema.sample_rows.map((row) => summarizeRow(row)),
      excluded_columns: textOnlySourceSchema.excluded_columns,
    };
    if (textOnlySourceSchema.excluded_columns.length > 0) {
      warnings.push(
        `Excluded non-text image/blob columns for text-only training: ${textOnlySourceSchema.excluded_columns.join(", ")}.`,
      );
    }
    const directCompatibility = inferDeterministicNormalization(
      availableColumns,
      textOnlySourceSchema.sample_rows,
    );
    const directTargetCandidates = inferClassificationTargetCandidates(
      availableColumns,
      textOnlySourceSchema.sample_rows,
    );

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
    const taskCompatibility = inferCompatibilityForTask(
      context.task_spec,
      normalizationProposal,
      selectedTargetColumn,
      compatibilityReason,
    );
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
      source_splits: sourceSplits,
      compatibility_status:
        compatibleMethods.length > 0 && taskCompatibility.compatible ? "compatible" : "incompatible",
      compatibility_reason: taskCompatibility.reason,
      normalization_source: normalizationSource,
      normalization_proposal: normalizationProposal,
      compatible_methods: taskCompatibility.compatible ? compatibleMethods : [],
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
    requestLabel: buildViewerRequestLabel(path, params),
  });
}

function buildViewerRequestLabel(path, params) {
  const label = [`HF viewer ${path}`];
  const dataset = normalizeOptionalString(params?.dataset);
  const config = normalizeOptionalString(params?.config);
  const split = normalizeOptionalString(params?.split);

  if (dataset) {
    label.push(`for ${dataset}`);
  }
  if (config && split) {
    label.push(`(${config}/${split})`);
  } else if (split) {
    label.push(`(split ${split})`);
  } else if (config) {
    label.push(`(config ${config})`);
  }

  return label.join(" ");
}

function parseSizeInfo(payload, fallbackNumRows = 0) {
  const viewerNumRows = toNumber(payload?.size?.dataset?.num_rows);
  return {
    numRows: viewerNumRows > 0 ? viewerNumRows : toNumber(fallbackNumRows),
    partial: Boolean(payload?.partial),
  };
}

function inferNumRowsFromHubCardData(cardData) {
  const splits = Array.isArray(cardData?.dataset_info?.splits) ? cardData.dataset_info.splits : [];
  const totalExamples = splits.reduce((sum, split) => {
    const numExamples = normalizeNonNegativeInt(split?.num_examples);
    return numExamples === null ? sum : sum + numExamples;
  }, 0);

  return totalExamples > 0 ? totalExamples : 0;
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

function resolveSourceSplits(rawSplits, preferredTrainSplit) {
  const discoveredSplits = uniqueStrings(
    (Array.isArray(rawSplits) ? rawSplits : [])
      .map((split) => normalizeOptionalString(split?.split))
      .filter(Boolean),
  );
  if (discoveredSplits.length > 0) {
    return discoveredSplits;
  }

  const fallbackSplit = normalizeOptionalString(preferredTrainSplit?.split) ?? "train";
  return [fallbackSplit];
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
  const taskSpecificLines = isGenerationTaskSpec(context.task_spec)
    ? [
      "This request is for generation SFT.",
      "Return usable=true only when the schema supports a meaningful deterministic generation normalization.",
      "Prefer direct prompt/completion normalization for source-to-target, instruction-following, summarization, or transformation datasets.",
      "Prefer direct text normalization for open-ended style, tone, role, or domain adaptation datasets.",
      "Do not convert prompt/completion pairs into raw text when direct prompt/completion normalization is available.",
      "Do not convert conversational `messages` arrays into text for generation SFT v1.",
      "selected_target_column must be null for generation tasks.",
    ]
    : [
      "This request is for classification SFT.",
      "When multiple plausible label columns exist, choose one target column and record ambiguity_warnings instead of failing.",
    ];

  return [
    "Infer a deterministic normalization recipe for a single Hugging Face dataset so the SFT backend can train on it.",
    "This backend supports both single-target classification SFT and generation SFT with raw-text or prompt/completion normalization.",
    "Do not invent columns.",
    "Use the native dataset field names exactly as provided.",
    "Return usable=false when the schema cannot support a meaningful deterministic text or prompt/completion normalization.",
    "Use template_synthesis only when direct column mapping is impossible.",
    "Do not use image, pixel, scan, DICOM, or other binary/media columns in prompt or completion text.",
    "Avoid likely identifier columns such as ticket ids, customer ids, phone numbers, and other obvious PII unless they are essential.",
    ...taskSpecificLines,
    `Objective: ${context.analysis?.domain_summary || ""}`,
    `Task spec: ${JSON.stringify(context.task_spec ?? {})}`,
    `Mapped task types: ${JSON.stringify(context.analysis?.mapped_task_types ?? [])}`,
    `Dataset id: ${candidate.id}`,
    `Description: ${candidate.description ?? ""}`,
    `Warnings: ${JSON.stringify(candidate.warnings ?? [])}`,
    `Schema signals: ${JSON.stringify(candidate.schema_signals ?? [])}`,
    `Excluded non-text columns: ${JSON.stringify(sourceSchema?.excluded_columns ?? [])}`,
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

  const model =
    normalizeOptionalString(process.env.OPENAI_DATASET_SELECTION_MODEL) ??
    DEFAULT_DATASET_SELECTION_MODEL;
  const rankingInput = buildOpenAIRankingInput(candidates);
  const prompt = buildOpenAIRankingPrompt(context, rankingInput);
  logUiProgress("I'm narrowing it down to the best training data");
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

function buildOpenAIRankingInput(candidates) {
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
  const taskSpecificLines = isGenerationTaskSpec(context.task_spec)
    ? [
      "Prefer prompt/completion datasets when the task is a source-to-target transformation, instruction-following task, or summarization task.",
      "Prefer raw-text datasets when the task is primarily style adaptation, role adaptation, or open-ended domain adaptation without explicit targets.",
      "Prefer datasets whose normalized inputs and targets closely match the requested behavior and output format.",
      "selected_target_column is not relevant for generation tasks.",
    ]
    : [
      "Prefer candidates with an explicit selected_target_column and lower ambiguity.",
      "Prefer deterministic direct normalization over template_synthesis when task fit is otherwise similar.",
    ];

  return [
    "Choose the smallest useful set of already-fetched Hugging Face datasets for the user's post-training use case.",
    "Use only the provided candidates. Do not invent new dataset ids.",
    "Return 1 dataset when a single clear winner is enough.",
    "Return 2-3 datasets only when combining them would materially improve coverage, robustness, or task fit.",
    "Do not return more than 3 datasets.",
    "Prefer domain relevance, task/schema fit, target row-band fit, public usability, license clarity, and overall data usefulness.",
    "Only choose candidates whose compatibility_status is compatible.",
    ...taskSpecificLines,
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
      source_splits: Array.isArray(candidate.source_splits) ? candidate.source_splits : candidate.splits,
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
        ? inferMixingStrategy(rankedDatasets)
        : plan.recommendation_guidance.mixing_strategy || inferMixingStrategy(rankedDatasets),
    warnings: mergedWarnings,
  };
}

function inferIdealDatasetCount(rankedDatasets) {
  if (rankedDatasets.length === 0) return 1;
  return Math.min(3, rankedDatasets.length);
}

function inferMixingStrategy(rankedDatasets) {
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

function logUiProgress(message, tone = "normal") {
  emitUiProgress(activeLogger, {
    stageId: "recommending",
    text: message,
    tone,
  });
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
      return "text generation";
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
