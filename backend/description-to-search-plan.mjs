import process from "node:process";
import { readFile } from "node:fs/promises";

const OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses";
const DEFAULT_OPENAI_MODEL = "gpt-5-mini";
const REQUEST_TIMEOUT_MS = 60_000;
const MAX_RETRIES = 3;
const TRANSIENT_STATUS_CODES = new Set([408, 409, 429, 500, 502, 503, 504]);
const LOG_PREFIX = "[description-to-search-plan]";

let dotEnvLoaded = false;
let activeLogger = null;

export async function generateSearchPlan(description, options = {}) {
  const previousLogger = activeLogger;
  activeLogger = options.logger ?? previousLogger;
  try {
    return await _generateSearchPlan(description);
  } finally {
    activeLogger = previousLogger;
  }
}

async function _generateSearchPlan(description) {
  await ensureDotEnvLoaded();

  const apiKey = normalizeOptionalString(process.env.OPENAI_API_KEY);
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required to generate a search plan from a description.");
  }

  const model = normalizeOptionalString(process.env.OPENAI_MODEL) ?? DEFAULT_OPENAI_MODEL;
  const prompt = buildPrompt(description);
  logInfo(`generating search plan with OpenAI model ${model}`);
  logMultiline("prompt", prompt);

  const response = await fetchJson(OPENAI_RESPONSES_API_URL, {
    method: "POST",
    timeoutMs: REQUEST_TIMEOUT_MS,
    maxRetries: MAX_RETRIES,
    requestLabel: "OpenAI search-plan generation",
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
            "You generate search plans for finding Hugging Face datasets for language-model post-training. Return only the requested JSON structure.",
        },
        { role: "user", content: prompt },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "hf_dataset_search_plan",
          schema: getSearchPlanSchema(),
          strict: true,
        },
      },
    }),
  });

  logJson("openai response metadata", {
    id: response.id ?? null,
    status: response.status ?? null,
    model: response.model ?? null,
    usage: response.usage ?? null,
  });

  const outputText = extractOutputText(response);
  if (!outputText) {
    throw new Error("OpenAI returned an empty search-plan response.");
  }

  logMultiline("openai output text", outputText);

  let plan;
  try {
    plan = JSON.parse(outputText);
  } catch (error) {
    throw new Error(
      `OpenAI returned invalid JSON for the search plan: ${error instanceof Error ? error.message : "unknown parse error"}`,
    );
  }

  logJson("parsed search plan", plan);
  return plan;
}

function buildPrompt(description) {
  return [
    "Create a Hugging Face dataset search plan for post-training a language model.",
    `Description: ${description}`,
    "Return analysis, search_queries, ranking_criteria, and recommendation_guidance.",
    "search_queries must contain concise Hugging Face search strings, usually 2-6 words, like something typed directly into the Hugging Face search bar.",
    "Use task_filter values only from: text-classification, question-answering, summarization, text-generation, translation, conversational, token-classification, or null.",
    "Use sort values only from: downloads, likes, trending, created.",
    "Set data_format_needed to exactly one of: instruction, completion, preference, raw_text, mixed.",
    "Keep mapped_task_types narrowly focused on the main fine-tuning objective, usually 1-2 task types.",
    "Keep warnings focused on practical dataset-selection risks.",
  ].join("\n");
}

function getSearchPlanSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
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
            task_filter: { type: ["string", "null"] },
            sort: {
              type: "string",
              enum: ["downloads", "likes", "trending", "created"],
            },
            intent: { type: "string" },
          },
          required: ["search", "task_filter", "sort", "intent"],
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
          warnings: { type: "array", items: { type: "string" } },
        },
        required: ["ideal_dataset_count", "target_total_rows", "mixing_strategy", "warnings"],
      },
    },
    required: ["analysis", "search_queries", "ranking_criteria", "recommendation_guidance"],
  };
}

// ---------------------------------------------------------------------------
// Utilities (self-contained to avoid coupling with hf-dataset-recommender)
// ---------------------------------------------------------------------------

function extractOutputText(response) {
  const topLevelText = normalizeOptionalString(response?.output_text);
  if (topLevelText) return topLevelText;

  const outputItems = Array.isArray(response?.output) ? response.output : [];
  for (const item of outputItems) {
    const content = Array.isArray(item?.content) ? item.content : [];
    for (const part of content) {
      if (part?.type === "output_text") {
        const text = normalizeOptionalString(part.text);
        if (text) return text;
      }
    }
  }
  return null;
}

function normalizeOptionalString(value) {
  if (value == null || typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed || null;
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

async function safeReadResponseText(response) {
  try {
    const text = await response.text();
    return text.slice(0, 500);
  } catch {
    return "";
  }
}

function safeUrlLabel(url) {
  try {
    const parsed = new URL(url);
    return `${parsed.hostname}${parsed.pathname}`;
  } catch {
    return String(url);
  }
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
        headers: { Accept: "application/json", ...(options.headers ?? {}) },
        body: options.body,
        signal: controller.signal,
      });

      if (!response.ok) {
        const responseText = await safeReadResponseText(response);
        const error = new Error(
          `${requestLabel} failed with ${response.status} (${safeUrlLabel(url)})${responseText ? `: ${responseText}` : ""}`,
        );
        error.status = response.status;

        if (attempt < maxRetries && TRANSIENT_STATUS_CODES.has(response.status)) {
          const delayMs = getRetryDelayMs(attempt, response);
          logInfo(`${requestLabel} failed with ${response.status}; retrying in ${delayMs}ms (${attempt + 1}/${maxRetries})`);
          await sleep(delayMs);
          continue;
        }
        throw error;
      }

      return await response.json();
    } catch (error) {
      const isAbort = error instanceof Error && error.name === "AbortError";
      const isRetryable = error instanceof TypeError || isAbort || Boolean(error?.status && TRANSIENT_STATUS_CODES.has(error.status));

      if (attempt < maxRetries && isRetryable) {
        const delayMs = getRetryDelayMs(attempt);
        logInfo(`${requestLabel} failed; retrying in ${delayMs}ms (${attempt + 1}/${maxRetries})`);
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

async function ensureDotEnvLoaded() {
  if (dotEnvLoaded) return;
  dotEnvLoaded = true;
  if (process.env.OPENAI_API_KEY) return;

  try {
    const envContents = await readFile(new URL("../.env", import.meta.url), "utf8");
    for (const rawLine of envContents.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) continue;

      const separatorIndex = line.indexOf("=");
      if (separatorIndex <= 0) continue;

      const key = line.slice(0, separatorIndex).trim();
      if (!key || process.env[key]) continue;

      let value = line.slice(separatorIndex + 1).trim();
      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      process.env[key] = value;
    }
  } catch {
    // Ignore missing .env
  }
}

function logInfo(message) {
  if (activeLogger && typeof activeLogger.emit === "function") {
    activeLogger.emit({ source: "description-to-search-plan", level: "info", message });
    return;
  }
  console.log(`${LOG_PREFIX} ${message}`);
}

function logJson(label, value) {
  if (activeLogger && typeof activeLogger.emit === "function") {
    activeLogger.emit({ source: "description-to-search-plan", level: "info", message: label, data: value });
    return;
  }
  console.log(`${LOG_PREFIX} ${label}:`);
  console.log(JSON.stringify(value, null, 2));
}

function logMultiline(label, value) {
  if (activeLogger && typeof activeLogger.emit === "function") {
    activeLogger.emit({ source: "description-to-search-plan", level: "info", message: label, data: String(value ?? "") });
    return;
  }
  console.log(`${LOG_PREFIX} ${label}:`);
  console.log(String(value ?? ""));
}
