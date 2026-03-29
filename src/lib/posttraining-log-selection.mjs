import { UI_PROGRESS_SOURCE } from "./posttraining-progress.mjs";

function normalizeRawTailLimit(value) {
  if (!Number.isInteger(value) || value < 0) {
    return 200;
  }
  return value;
}

function isUiProgressEvent(event) {
  return event?.source === UI_PROGRESS_SOURCE;
}

export function selectJobEventsForApi(events, options = {}) {
  const rawTailLimit = normalizeRawTailLimit(options.rawTailLimit);
  const normalizedEvents = Array.isArray(events) ? events : [];
  const selectedIndexes = new Set();
  let rawEventsSelected = 0;

  normalizedEvents.forEach((event, index) => {
    if (isUiProgressEvent(event)) {
      selectedIndexes.add(index);
    }
  });

  for (let index = normalizedEvents.length - 1; index >= 0; index -= 1) {
    if (selectedIndexes.has(index)) {
      continue;
    }
    if (rawEventsSelected >= rawTailLimit) {
      break;
    }
    selectedIndexes.add(index);
    rawEventsSelected += 1;
  }

  return normalizedEvents.filter((_, index) => selectedIndexes.has(index));
}
