export const UI_PROGRESS_SOURCE = "ui-progress";

/**
 * @typedef {"normal" | "error"} UiProgressTone
 */

/**
 * @typedef {{
 *   stageId: string;
 *   text: string;
 *   tone: UiProgressTone;
 * }} ParsedUiProgressEvent
 */

/**
 * @typedef {{
 *   id: string;
 *   text: string;
 *   tone: UiProgressTone;
 * }} StageProgressItem
 */

function normalizeTone(rawTone, rawLevel = "info") {
  if (rawTone === "error") {
    return "error";
  }
  return rawLevel === "error" ? "error" : "normal";
}

function normalizeStageId(input) {
  return typeof input === "string" ? input.trim() : "";
}

function normalizeProgressText(input) {
  return typeof input === "string" ? input.trim() : "";
}

export function buildUiProgressEvent(input = {}) {
  const stageId = normalizeStageId(input.stageId ?? input.stage);
  const text = normalizeProgressText(input.text ?? input.message);
  const tone = normalizeTone(input.tone);

  if (!stageId || !text) {
    return null;
  }

  return {
    stage: stageId,
    source: UI_PROGRESS_SOURCE,
    level: tone === "error" ? "error" : "info",
    message: text,
    data: {
      tone,
    },
  };
}

export function emitUiProgress(logger, input = {}) {
  const event = buildUiProgressEvent(input);
  if (!event || !logger || typeof logger.emit !== "function") {
    return false;
  }

  logger.emit(event);
  return true;
}

export function parseUiProgressEvent(event) {
  if (!event || typeof event !== "object") {
    return null;
  }

  const record = /** @type {{ stage?: unknown; source?: unknown; message?: unknown; level?: unknown; data?: { tone?: unknown } }} */ (event);
  if (record.source !== UI_PROGRESS_SOURCE) {
    return null;
  }

  const stageId = normalizeStageId(record.stage);
  const text = normalizeProgressText(record.message);
  if (!stageId || !text) {
    return null;
  }

  return {
    stageId,
    text,
    tone: normalizeTone(record.data?.tone, typeof record.level === "string" ? record.level : "info"),
  };
}

function normalizeMaxItemsPerStage(value) {
  return Number.isInteger(value) && value > 0 ? value : null;
}

export function getUiProgressHistoryByStage(logs, options = {}) {
  const maxItemsPerStage = normalizeMaxItemsPerStage(options.maxItemsPerStage);

  /** @type {Record<string, StageProgressItem[]>} */
  const historyByStage = {};

  for (const [eventIndex, event] of (Array.isArray(logs) ? logs : []).entries()) {
    const parsed = parseUiProgressEvent(event);
    if (!parsed) {
      continue;
    }

    const stageHistory = historyByStage[parsed.stageId] ?? [];
    stageHistory.push({
      id: `${parsed.stageId}:${eventIndex}`,
      text: parsed.text,
      tone: parsed.tone,
    });

    if (maxItemsPerStage !== null && stageHistory.length > maxItemsPerStage) {
      stageHistory.shift();
    }

    historyByStage[parsed.stageId] = stageHistory;
  }

  return historyByStage;
}

export function getLatestUiProgressByStage(logs, options = {}) {
  const historyByStage = getUiProgressHistoryByStage(logs, options);
  /** @type {Record<string, StageProgressItem>} */
  const latestByStage = {};

  for (const [stageId, history] of Object.entries(historyByStage)) {
    const latestItem = history[history.length - 1];
    if (latestItem) {
      latestByStage[stageId] = latestItem;
    }
  }

  return latestByStage;
}

export function mergeStageProgressById(previousProgress = {}, nextProgress = {}) {
  const mergedProgress = {};
  const stageIds = new Set([
    ...Object.keys(previousProgress ?? {}),
    ...Object.keys(nextProgress ?? {}),
  ]);

  for (const stageId of stageIds) {
    const previousItems = Array.isArray(previousProgress?.[stageId])
      ? previousProgress[stageId]
      : [];
    const nextItems = Array.isArray(nextProgress?.[stageId]) ? nextProgress[stageId] : [];

    if (nextItems.length === 0) {
      if (previousItems.length > 0) {
        mergedProgress[stageId] = previousItems;
      }
      continue;
    }

    mergedProgress[stageId] =
      nextItems.length >= previousItems.length ? nextItems : previousItems;
  }

  return mergedProgress;
}

export function getSidebarStageProgress({
  logs,
  activeStageId = null,
  completedStageIds = [],
  failedStageId = null,
  jobStatus = null,
  maxItemsPerStage = null,
}) {
  const historyByStage = getUiProgressHistoryByStage(logs, { maxItemsPerStage });
  const completedStageSet = new Set(
    Array.isArray(completedStageIds)
      ? completedStageIds.map((stageId) => normalizeStageId(stageId)).filter(Boolean)
      : [],
  );

  /** @type {Record<string, StageProgressItem[]>} */
  const visibleProgress = {};

  for (const completedStageId of completedStageSet) {
    const stageHistory = historyByStage[completedStageId];
    if (stageHistory?.length) {
      visibleProgress[completedStageId] = stageHistory;
    }
  }

  const normalizedFailedStageId = normalizeStageId(failedStageId);
  if (
    jobStatus === "failed" &&
    normalizedFailedStageId &&
    !completedStageSet.has(normalizedFailedStageId) &&
    historyByStage[normalizedFailedStageId]?.length
  ) {
    visibleProgress[normalizedFailedStageId] = historyByStage[normalizedFailedStageId];
    return visibleProgress;
  }

  const normalizedActiveStageId = normalizeStageId(activeStageId);
  if (
    normalizedActiveStageId &&
    !completedStageSet.has(normalizedActiveStageId) &&
    historyByStage[normalizedActiveStageId]?.length
  ) {
    visibleProgress[normalizedActiveStageId] = historyByStage[normalizedActiveStageId];
  }

  return visibleProgress;
}
