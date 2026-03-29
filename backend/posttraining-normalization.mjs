export const ALLOWED_NORMALIZATION_SHAPES = ["text", "prompt_completion"];
export const SFT_COMPATIBLE_METHODS = ["sft"];
export const NORMALIZATION_VERSION = 1;

const IDENTIFIER_COLUMN_PATTERNS = [
  /(^|_)(id|uuid|guid)(_|$)/i,
  /ticket[_-]?id/i,
  /customer[_-]?id/i,
  /conversation[_-]?id/i,
  /phone/i,
  /email/i,
];

const IDENTIFIER_VALUE_PATTERNS = [
  /^[0-9]{6,}$/,
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
  /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
];

const NON_TEXT_COLUMN_PATTERNS = [
  /(^|_)(image|images|img|photo|picture|pixel|pixels|frame|frames)(_|$)/i,
  /(^|_)(scan|dicom|xray|x_ray|mammogram|thumbnail)(_|$)/i,
];

const TEXT_COLUMN_CANDIDATES = [
  "text",
  "body",
  "message",
  "description",
  "details",
  "content",
  "subject",
  "ticket_text",
  "ticket_body",
  "prompt",
  "input",
  "utterance",
  "query",
];

const LABEL_COLUMN_CANDIDATES = [
  "label",
  "labels",
  "category",
  "class",
  "intent",
  "topic",
  "issue_type",
  "priority",
  "urgency",
  "severity",
  "type",
  "queue",
];

const PROMPT_COLUMN_CANDIDATES = [
  "prompt",
  "instruction",
  "input",
  "dialog",
  "dialogue",
  "conversation",
  "transcript",
];

const COMPLETION_COLUMN_CANDIDATES = [
  "completion",
  "response",
  "output",
  "answer",
  "summary",
  "soap",
  "soap_summary",
  "clinical_note",
  "clinical_note_text",
  "note",
  "notes",
  "note_text",
  "section_text",
  "target",
];

const QUESTION_COLUMN_CANDIDATES = [
  "question",
  "query",
];

const ANSWER_COLUMN_CANDIDATES = [
  "answer",
  "answers",
  "response",
];

const CONTEXT_COLUMN_CANDIDATES = [
  "context",
  "passage",
  "document",
];

export function uniqueStrings(values) {
  return [...new Set((values ?? []).filter(Boolean))];
}

function isBinaryArrayLike(value) {
  if (typeof Buffer !== "undefined" && typeof Buffer.isBuffer === "function" && Buffer.isBuffer(value)) {
    return true;
  }
  if (typeof ArrayBuffer !== "undefined") {
    if (value instanceof ArrayBuffer) {
      return true;
    }
    if (ArrayBuffer.isView(value)) {
      return true;
    }
  }
  return false;
}

function looksLikeImagePayloadValue(value) {
  if (value === null || value === undefined) {
    return false;
  }
  if (isBinaryArrayLike(value)) {
    return true;
  }
  if (typeof value === "string") {
    return /^data:image\//i.test(value.trim());
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const normalizedKeys = new Set(Object.keys(value).map((key) => normalizeKey(key)));
    if (normalizedKeys.has("bytes") || normalizedKeys.has("blob") || normalizedKeys.has("pixel_values")) {
      return true;
    }
    if (
      (normalizedKeys.has("path") || normalizedKeys.has("src") || normalizedKeys.has("url")) &&
      (normalizedKeys.has("bytes") || normalizedKeys.has("height") || normalizedKeys.has("width"))
    ) {
      return true;
    }
  }
  return false;
}

function looksLikeNonTextColumn(columnName, sampleRows) {
  if (NON_TEXT_COLUMN_PATTERNS.some((pattern) => pattern.test(String(columnName ?? "")))) {
    return true;
  }
  return sampleColumnValues(sampleRows, columnName).some((value) => looksLikeImagePayloadValue(value));
}

export function filterSourceSchemaForTextOnlyTraining(featureNames, sampleRows) {
  const normalizedRows = Array.isArray(sampleRows)
    ? sampleRows.filter((row) => row && typeof row === "object" && !Array.isArray(row))
    : [];
  const allColumns = uniqueStrings([
    ...(Array.isArray(featureNames) ? featureNames.map((value) => String(value).trim()).filter(Boolean) : []),
    ...normalizedRows.flatMap((row) => Object.keys(row)),
  ]);
  const excludedColumns = allColumns.filter((columnName) => looksLikeNonTextColumn(columnName, normalizedRows));
  const excludedColumnSet = new Set(excludedColumns);

  return {
    available_columns: allColumns.filter((columnName) => !excludedColumnSet.has(columnName)),
    sample_rows: normalizedRows.map((row) =>
      Object.fromEntries(
        Object.entries(row).filter(([columnName]) => !excludedColumnSet.has(columnName)),
      )
    ),
    excluded_columns: excludedColumns,
  };
}

export function normalizeOptionalString(value) {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  return null;
}

export function ensureClassificationPromptSeparator(template) {
  const normalized = normalizeOptionalString(template);
  if (!normalized) {
    return normalized;
  }

  if (/(?:^|\n)Label:\s*$/i.test(normalized)) {
    return normalized.replace(/Label:\s*$/i, "Label:\n");
  }

  return `${normalized}\n\nLabel:\n`;
}

export function truncateText(value, maxLength = 240) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 3)}...`;
}

export function normalizeKey(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

export function firstMatchingColumn(columnMap, candidates) {
  for (const candidate of candidates) {
    const normalized = normalizeKey(candidate);
    if (columnMap.has(normalized)) {
      return columnMap.get(normalized);
    }
  }
  return null;
}

export function extractRowsFromPreview(previewPayload) {
  if (!Array.isArray(previewPayload?.rows)) {
    return [];
  }
  return previewPayload.rows
    .map((rowEntry) => (rowEntry?.row && typeof rowEntry.row === "object" ? rowEntry.row : null))
    .filter(Boolean);
}

export function looksLikeMessages(value) {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((item) => item && typeof item === "object" && "role" in item)
  );
}

export function summarizeRow(row) {
  const summarized = {};
  for (const [key, value] of Object.entries(row ?? {})) {
    if (looksLikeMessages(value)) {
      summarized[key] = value.slice(0, 2).map((message) => ({
        role: message.role,
        content: truncateText(message.content, 120),
      }));
    } else if (Array.isArray(value)) {
      summarized[key] = value.slice(0, 4).map((item) =>
        typeof item === "string" ? truncateText(item, 120) : item,
      );
    } else if (value && typeof value === "object") {
      summarized[key] = truncateText(JSON.stringify(value), 160);
    } else {
      summarized[key] = truncateText(value, 160);
    }
  }
  return summarized;
}

export function selectPreferredTrainSplit(splits) {
  const preferredOrder = ["train", "training", "default"];
  for (const preferred of preferredOrder) {
    const match = splits.find((split) => normalizeOptionalString(split.split)?.toLowerCase() === preferred);
    if (match) {
      return match;
    }
  }
  return splits[0] ?? null;
}

export function selectPreferredEvalSplit(splits) {
  const preferredOrder = ["validation", "dev", "test", "eval"];
  for (const preferred of preferredOrder) {
    const match = splits.find((split) => normalizeOptionalString(split.split)?.toLowerCase() === preferred);
    if (match) {
      return match;
    }
  }
  return null;
}

function getSampleValue(rows, columnName) {
  for (const row of rows) {
    if (row && Object.prototype.hasOwnProperty.call(row, columnName)) {
      return row[columnName];
    }
  }
  return null;
}

function buildCopyField(sourceColumn) {
  return {
    source_column: sourceColumn,
    template: null,
    value_mapping: null,
  };
}

function buildTemplateField(template, valueMapping = null) {
  return {
    source_column: null,
    template,
    value_mapping: valueMapping,
  };
}

function buildTextNormalization(sourceColumn, strategy = "copy_column") {
  return {
    version: NORMALIZATION_VERSION,
    shape: "text",
    strategy,
    source_columns: [sourceColumn],
    fields: {
      text: buildCopyField(sourceColumn),
      prompt: null,
      completion: null,
    },
  };
}

function buildPromptCompletionNormalization({
  strategy,
  sourceColumns,
  promptField,
  completionField,
}) {
  return {
    version: NORMALIZATION_VERSION,
    shape: "prompt_completion",
    strategy,
    source_columns: uniqueStrings(sourceColumns),
    fields: {
      text: null,
      prompt: promptField,
      completion: completionField,
    },
  };
}

function isScalarValue(value) {
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

function looksIdentifierLikeColumn(columnName) {
  return IDENTIFIER_COLUMN_PATTERNS.some((pattern) => pattern.test(String(columnName ?? "")));
}

function looksIdentifierLikeValue(value) {
  const text = String(value ?? "").trim();
  if (!text) {
    return false;
  }
  return IDENTIFIER_VALUE_PATTERNS.some((pattern) => pattern.test(text));
}

function sampleColumnValues(sampleRows, columnName) {
  return (sampleRows ?? [])
    .map((row) =>
      row && Object.prototype.hasOwnProperty.call(row, columnName) ? row[columnName] : undefined,
    )
    .filter((value) => value !== undefined);
}

function isScalarCompatibleColumn(columnName, sampleRows) {
  const observedValues = sampleColumnValues(sampleRows, columnName).filter((value) => value !== null);
  if (observedValues.length === 0) {
    return true;
  }
  return observedValues.every((value) => isScalarValue(value));
}

function analyzeClassificationTargetColumn(columnName, sampleRows) {
  const observedValues = sampleColumnValues(sampleRows, columnName).filter((value) => value !== null);
  const scalarValues = observedValues.filter((value) => isScalarValue(value));
  const scalar = scalarValues.length === observedValues.length;
  const identifierLikeValues =
    scalarValues.length > 0 && scalarValues.every((value) => looksIdentifierLikeValue(value));
  const identifierLike = looksIdentifierLikeColumn(columnName) || identifierLikeValues;

  return {
    column: columnName,
    scalar,
    identifier_like: identifierLike,
    observed_values: scalarValues.map((value) => String(value)),
  };
}

export function inferClassificationTargetCandidates(featureNames, sampleRows) {
  const availableColumns = uniqueStrings(featureNames ?? []);
  const columnMap = new Map(availableColumns.map((name) => [normalizeKey(name), name]));
  const candidates = [];

  for (const candidateName of LABEL_COLUMN_CANDIDATES) {
    const columnName = firstMatchingColumn(columnMap, [candidateName]);
    if (!columnName || candidates.some((candidate) => candidate.column === columnName)) {
      continue;
    }
    const analysis = analyzeClassificationTargetColumn(columnName, sampleRows);
    if (!analysis.scalar || analysis.identifier_like) {
      continue;
    }
    candidates.push(analysis);
  }

  return candidates;
}

export function extractTemplateVariables(template) {
  const matches = [...String(template ?? "").matchAll(/\{([^{}]+)\}/g)];
  return uniqueStrings(
    matches
      .map((match) => normalizeOptionalString(match[1]))
      .filter(Boolean),
  );
}

export function inferCompatibleMethodsFromNormalization(normalizationProposal) {
  if (!normalizationProposal || typeof normalizationProposal !== "object") {
    return [];
  }
  if (ALLOWED_NORMALIZATION_SHAPES.includes(normalizationProposal.shape)) {
    return [...SFT_COMPATIBLE_METHODS];
  }
  return [];
}

export function inferOutputKindFromNormalization(method, normalizationProposal) {
  if (method !== "sft") {
    return "unsupported";
  }
  if (!normalizationProposal || typeof normalizationProposal !== "object") {
    return "unsupported";
  }
  return normalizationProposal.shape === "prompt_completion" ? "prompt_completion" : "text";
}

export function validateNormalizationProposal(normalizationProposal, availableColumns) {
  const errors = [];
  const validColumns = new Set(Array.isArray(availableColumns) ? availableColumns : []);

  if (!normalizationProposal || typeof normalizationProposal !== "object" || Array.isArray(normalizationProposal)) {
    return ["normalization proposal must be an object."];
  }

  if (!ALLOWED_NORMALIZATION_SHAPES.includes(normalizationProposal.shape)) {
    errors.push(
      `normalization.shape must be one of ${ALLOWED_NORMALIZATION_SHAPES.join(", ")}.`,
    );
  }

  const sourceColumns = Array.isArray(normalizationProposal.source_columns)
    ? uniqueStrings(normalizationProposal.source_columns.map((value) => String(value).trim()))
    : [];
  if (sourceColumns.length === 0) {
    errors.push("normalization.source_columns must be a non-empty array.");
  }
  for (const column of sourceColumns) {
    if (!validColumns.has(column)) {
      errors.push(`normalization.source_columns references unknown column '${column}'.`);
    }
  }

  const fields =
    normalizationProposal.fields && typeof normalizationProposal.fields === "object"
      ? normalizationProposal.fields
      : null;
  if (!fields) {
    errors.push("normalization.fields must be an object.");
    return errors;
  }

  const fieldNames =
    normalizationProposal.shape === "text"
      ? ["text"]
      : normalizationProposal.shape === "prompt_completion"
        ? ["prompt", "completion"]
        : [];

  for (const fieldName of fieldNames) {
    const fieldValue = fields[fieldName];
    if (!fieldValue || typeof fieldValue !== "object" || Array.isArray(fieldValue)) {
      errors.push(`normalization.fields.${fieldName} must be an object.`);
      continue;
    }

    const sourceColumn = normalizeOptionalString(fieldValue.source_column);
    const template = normalizeOptionalString(fieldValue.template);
    if (!sourceColumn && !template) {
      errors.push(
        `normalization.fields.${fieldName} must define either source_column or template.`,
      );
      continue;
    }
    if (sourceColumn && template) {
      errors.push(
        `normalization.fields.${fieldName} cannot define both source_column and template.`,
      );
    }
    if (sourceColumn && !validColumns.has(sourceColumn)) {
      errors.push(`normalization.fields.${fieldName}.source_column '${sourceColumn}' does not exist.`);
    }
    if (sourceColumn && !sourceColumns.includes(sourceColumn)) {
      errors.push(
        `normalization.fields.${fieldName}.source_column '${sourceColumn}' must also appear in normalization.source_columns.`,
      );
    }
    for (const variable of extractTemplateVariables(template)) {
      if (!validColumns.has(variable)) {
        errors.push(
          `normalization.fields.${fieldName}.template references unknown column '${variable}'.`,
        );
      } else if (!sourceColumns.includes(variable)) {
        errors.push(
          `normalization.fields.${fieldName}.template column '${variable}' must also appear in normalization.source_columns.`,
        );
      }
    }

    const valueMapping = fieldValue.value_mapping;
    if (
      valueMapping !== null &&
      valueMapping !== undefined &&
      (typeof valueMapping !== "object" || Array.isArray(valueMapping))
    ) {
      errors.push(`normalization.fields.${fieldName}.value_mapping must be an object when provided.`);
    }
  }

  return errors;
}

export function inferDeterministicNormalization(featureNames, sampleRows) {
  const textOnlySchema = filterSourceSchemaForTextOnlyTraining(featureNames, sampleRows);
  const availableColumns = textOnlySchema.available_columns;
  const filteredSampleRows = textOnlySchema.sample_rows;
  const columnMap = new Map(availableColumns.map((name) => [normalizeKey(name), name]));

  const promptColumn = firstMatchingColumn(columnMap, PROMPT_COLUMN_CANDIDATES);
  const completionColumn = firstMatchingColumn(columnMap, COMPLETION_COLUMN_CANDIDATES);
  if (
    promptColumn &&
    completionColumn &&
    isScalarCompatibleColumn(promptColumn, filteredSampleRows) &&
    isScalarCompatibleColumn(completionColumn, filteredSampleRows)
  ) {
    return {
      compatibility_status: "compatible",
      compatibility_reason: `Direct prompt/completion normalization is available via '${promptColumn}' and '${completionColumn}'.`,
      normalization_proposal: buildPromptCompletionNormalization({
        strategy: "copy_columns",
        sourceColumns: [promptColumn, completionColumn],
        promptField: buildCopyField(promptColumn),
        completionField: buildCopyField(completionColumn),
      }),
      compatible_methods: [...SFT_COMPATIBLE_METHODS],
    };
  }

  const questionColumn = firstMatchingColumn(columnMap, QUESTION_COLUMN_CANDIDATES);
  const answerColumn = firstMatchingColumn(columnMap, ANSWER_COLUMN_CANDIDATES);
  const contextColumn = firstMatchingColumn(columnMap, CONTEXT_COLUMN_CANDIDATES);
  if (questionColumn && answerColumn) {
    const promptTemplate = contextColumn
      ? `Context:\n{${contextColumn}}\n\nQuestion:\n{${questionColumn}}\n\nAnswer:`
      : `Question:\n{${questionColumn}}\n\nAnswer:`;
    return {
      compatibility_status: "compatible",
      compatibility_reason: `Question/answer normalization is available via '${questionColumn}' and '${answerColumn}'.`,
      normalization_proposal: buildPromptCompletionNormalization({
        strategy: "qa_template",
        sourceColumns: uniqueStrings([questionColumn, answerColumn, contextColumn].filter(Boolean)),
        promptField: buildTemplateField(promptTemplate),
        completionField: buildCopyField(answerColumn),
      }),
      compatible_methods: [...SFT_COMPATIBLE_METHODS],
    };
  }

  const inputColumn = firstMatchingColumn(columnMap, TEXT_COLUMN_CANDIDATES);
  const targetCandidates = inferClassificationTargetCandidates(availableColumns, filteredSampleRows);
  const labelColumn = targetCandidates[0]?.column ?? null;
  if (inputColumn && labelColumn) {
    const ambiguityWarnings =
      targetCandidates.length > 1
        ? [
            `Multiple plausible target columns were detected (${targetCandidates
              .map((candidate) => `'${candidate.column}'`)
              .join(", ")}); using '${labelColumn}' as the provisional target.`,
          ]
        : [];
    const promptTemplate = ensureClassificationPromptSeparator(
      `Classify the following example. Return only the label.\n\nInput:\n{${inputColumn}}\n\nLabel:`,
    );
    return {
      compatibility_status: "compatible",
      compatibility_reason: `Classification-style normalization is available via '${inputColumn}' and '${labelColumn}'.`,
      selected_target_column: labelColumn,
      target_selection_reason:
        targetCandidates.length > 1
          ? `Selected '${labelColumn}' from multiple plausible native label columns.`
          : `Selected '${labelColumn}' as the sole plausible native label column.`,
      target_selection_confidence: targetCandidates.length > 1 ? 0.6 : 0.95,
      target_candidates: targetCandidates.map((candidate) => candidate.column),
      ambiguity_warnings: ambiguityWarnings,
      normalization_proposal: buildPromptCompletionNormalization({
        strategy: "classification_template",
        sourceColumns: [inputColumn, labelColumn],
        promptField: buildTemplateField(promptTemplate),
        completionField: {
          source_column: labelColumn,
          template: null,
          value_mapping: null,
        },
      }),
      compatible_methods: [...SFT_COMPATIBLE_METHODS],
    };
  }

  if (inputColumn) {
    return {
      compatibility_status: "compatible",
      compatibility_reason: `Direct text normalization is available via '${inputColumn}'.`,
      selected_target_column: null,
      target_selection_reason: null,
      target_selection_confidence: null,
      target_candidates: [],
      ambiguity_warnings: [],
      normalization_proposal: buildTextNormalization(inputColumn),
      compatible_methods: [...SFT_COMPATIBLE_METHODS],
    };
  }

  const messagesColumn = firstMatchingColumn(columnMap, [
    "messages",
    "conversation",
    "conversations",
    "dialog",
    "dialogue",
    "chat",
  ]);
  if (messagesColumn && looksLikeMessages(getSampleValue(filteredSampleRows, messagesColumn))) {
    return {
      compatibility_status: "incompatible",
      compatibility_reason:
        "Conversational message arrays are only supported through the legacy preset path today.",
      selected_target_column: null,
      target_selection_reason: null,
      target_selection_confidence: null,
      target_candidates: [],
      ambiguity_warnings: [],
      normalization_proposal: null,
      compatible_methods: [],
    };
  }

  return {
    compatibility_status: "incompatible",
    compatibility_reason:
      "No deterministic SFT normalization could be inferred from the available native dataset fields.",
    selected_target_column: null,
    target_selection_reason: null,
    target_selection_confidence: null,
    target_candidates: targetCandidates.map((candidate) => candidate.column),
    ambiguity_warnings: [],
    normalization_proposal: null,
    compatible_methods: [],
  };
}
