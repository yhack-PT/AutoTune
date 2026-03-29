const DEFAULT_WIDTH = 760;
const DEFAULT_HEIGHT = 320;
const PADDING = {
  top: 56,
  right: 24,
  bottom: 36,
  left: 56,
};

export const STRUCTURED_TRAINING_METRIC_PREFIX = "PT_METRIC_EVENT::";

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function formatMetricValue(value) {
  if (!isFiniteNumber(value)) {
    return "n/a";
  }
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(2);
  }
  return value.toFixed(4).replace(/0+$/u, "").replace(/\.$/u, "");
}

function buildLineConfig(summary, key, label, color) {
  return {
    key,
    label,
    color,
    points: Array.isArray(summary.series[key]) ? summary.series[key] : [],
  };
}

function computeDomain(lines) {
  const allPoints = lines.flatMap((line) => line.points);
  if (allPoints.length === 0) {
    return null;
  }

  const xValues = allPoints.map((point) => point.step);
  const yValues = allPoints.map((point) => point.value);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const paddedYRange = maxY === minY ? Math.max(Math.abs(maxY) * 0.1, 1) : (maxY - minY) * 0.12;

  return {
    minX,
    maxX,
    minY: minY - paddedYRange,
    maxY: maxY + paddedYRange,
  };
}

function scaleX(step, domain, width) {
  const chartWidth = width - PADDING.left - PADDING.right;
  if (domain.maxX === domain.minX) {
    return PADDING.left + chartWidth / 2;
  }
  return (
    PADDING.left +
    ((step - domain.minX) / (domain.maxX - domain.minX)) * chartWidth
  );
}

function scaleY(value, domain, height) {
  const chartHeight = height - PADDING.top - PADDING.bottom;
  if (domain.maxY === domain.minY) {
    return PADDING.top + chartHeight / 2;
  }
  return (
    height -
    PADDING.bottom -
    ((value - domain.minY) / (domain.maxY - domain.minY)) * chartHeight
  );
}

function renderGrid(domain, width, height) {
  const chartHeight = height - PADDING.top - PADDING.bottom;
  const rows = 4;
  const segments = [];

  for (let index = 0; index <= rows; index += 1) {
    const ratio = index / rows;
    const y = PADDING.top + ratio * chartHeight;
    const value = domain.maxY - ratio * (domain.maxY - domain.minY);
    segments.push(
      `<line x1="${PADDING.left}" y1="${y}" x2="${width - PADDING.right}" y2="${y}" stroke="#e5e7eb" stroke-width="1" />`,
    );
    segments.push(
      `<text x="${PADDING.left - 10}" y="${y + 4}" font-size="11" text-anchor="end" fill="#6b7280">${escapeXml(
        formatMetricValue(value),
      )}</text>`,
    );
  }

  const xStart = escapeXml(String(domain.minX));
  const xEnd = escapeXml(String(domain.maxX));
  segments.push(
    `<text x="${PADDING.left}" y="${height - 10}" font-size="11" fill="#6b7280">${xStart}</text>`,
  );
  segments.push(
    `<text x="${width - PADDING.right}" y="${height - 10}" font-size="11" text-anchor="end" fill="#6b7280">${xEnd}</text>`,
  );

  return segments.join("");
}

function renderLegend(lines, summary, width) {
  const segments = [];
  let x = PADDING.left;

  for (const line of lines) {
    const latestValue = summary.latest[line.key];
    segments.push(
      `<rect x="${x}" y="18" width="12" height="12" rx="3" fill="${line.color}" />`,
    );
    segments.push(
      `<text x="${x + 18}" y="28" font-size="12" fill="#111827">${escapeXml(
        `${line.label}: ${formatMetricValue(latestValue)}`,
      )}</text>`,
    );
    x += Math.min(width * 0.36, 190);
  }

  return segments.join("");
}

function renderPolyline(points, domain, width, height) {
  return points
    .map((point) => `${scaleX(point.step, domain, width)},${scaleY(point.value, domain, height)}`)
    .join(" ");
}

export function parseStructuredTrainingMetricLine(line) {
  const text = String(line ?? "").trim();
  if (!text.startsWith(STRUCTURED_TRAINING_METRIC_PREFIX)) {
    return null;
  }

  const payloadText = text.slice(STRUCTURED_TRAINING_METRIC_PREFIX.length);
  let payload;
  try {
    payload = JSON.parse(payloadText);
  } catch {
    return null;
  }

  const metrics = {};
  for (const [key, rawValue] of Object.entries(payload?.metrics ?? {})) {
    const numericValue = Number(rawValue);
    if (Number.isFinite(numericValue)) {
      metrics[key] = numericValue;
    }
  }

  if (Object.keys(metrics).length === 0) {
    return null;
  }

  const step = Number(payload?.step);
  if (!Number.isFinite(step)) {
    return null;
  }

  const epochValue = payload?.epoch;
  const epoch =
    epochValue === null || epochValue === undefined || epochValue === ""
      ? null
      : Number.isFinite(Number(epochValue))
        ? Number(epochValue)
        : null;

  return {
    timestamp:
      typeof payload?.timestamp === "string" && payload.timestamp.trim()
        ? payload.timestamp
        : new Date().toISOString(),
    step,
    epoch,
    metrics,
  };
}

export function summarizeTrainingMetricRecords(records) {
  const series = {};

  for (const record of Array.isArray(records) ? records : []) {
    const step = Number(record?.step);
    if (!Number.isFinite(step)) {
      continue;
    }

    const epochValue = record?.epoch;
    const epoch =
      epochValue === null || epochValue === undefined || epochValue === ""
        ? null
        : Number.isFinite(Number(epochValue))
          ? Number(epochValue)
          : null;

    const timestamp =
      typeof record?.timestamp === "string" && record.timestamp.trim()
        ? record.timestamp
        : new Date().toISOString();

    for (const [key, rawValue] of Object.entries(record?.metrics ?? {})) {
      const value = Number(rawValue);
      if (!Number.isFinite(value)) {
        continue;
      }
      if (!series[key]) {
        series[key] = [];
      }
      series[key].push({
        step,
        epoch,
        timestamp,
        value,
      });
    }
  }

  for (const points of Object.values(series)) {
    points.sort((left, right) => {
      if (left.step !== right.step) {
        return left.step - right.step;
      }
      return left.timestamp.localeCompare(right.timestamp);
    });
  }

  const availableMetrics = Object.keys(series).sort();
  const latest = Object.fromEntries(
    availableMetrics.map((metric) => [metric, series[metric].at(-1)?.value ?? null]),
  );

  return {
    availableMetrics,
    latest,
    series,
  };
}

export function renderTrainingMetricChartSvg({
  title,
  subtitle = "",
  lines,
  summary,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
}) {
  const activeLines = (Array.isArray(lines) ? lines : []).filter(
    (line) => Array.isArray(line.points) && line.points.length > 0,
  );
  if (activeLines.length === 0) {
    return null;
  }

  const domain = computeDomain(activeLines);
  if (!domain) {
    return null;
  }

  const polylines = activeLines
    .map((line) => {
      const polyline = renderPolyline(line.points, domain, width, height);
      return `<polyline fill="none" stroke="${line.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${polyline}" />`;
    })
    .join("");

  const lastPoints = activeLines
    .map((line) => {
      const point = line.points.at(-1);
      if (!point) {
        return "";
      }
      return `<circle cx="${scaleX(point.step, domain, width)}" cy="${scaleY(point.value, domain, height)}" r="4" fill="${line.color}" />`;
    })
    .join("");

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(title)}">`,
    `<rect width="${width}" height="${height}" rx="18" fill="#ffffff" />`,
    `<text x="${PADDING.left}" y="26" font-size="18" font-weight="700" fill="#111827">${escapeXml(title)}</text>`,
    subtitle
      ? `<text x="${PADDING.left}" y="44" font-size="12" fill="#6b7280">${escapeXml(subtitle)}</text>`
      : "",
    renderLegend(activeLines, summary, width),
    renderGrid(domain, width, height),
    polylines,
    lastPoints,
    `</svg>`,
  ].join("");
}

export function buildTrainingMetricGraphArtifacts(records) {
  const summary = summarizeTrainingMetricRecords(records);
  const artifacts = [];

  const lossSvg = renderTrainingMetricChartSvg({
    title: "Training Loss",
    subtitle: "Training loss with validation loss when available.",
    summary,
    lines: [
      buildLineConfig(summary, "loss", "train loss", "#0f766e"),
      buildLineConfig(summary, "eval_loss", "eval loss", "#dc2626"),
      buildLineConfig(summary, "train_loss", "final train loss", "#2563eb"),
    ],
  });
  if (lossSvg) {
    artifacts.push({
      filename: "training_loss.svg",
      content: lossSvg,
    });
  }

  const learningRateSvg = renderTrainingMetricChartSvg({
    title: "Learning Rate",
    subtitle: "Captured from trainer log events.",
    summary,
    lines: [
      buildLineConfig(summary, "learning_rate", "learning rate", "#7c3aed"),
    ],
  });
  if (learningRateSvg) {
    artifacts.push({
      filename: "learning_rate.svg",
      content: learningRateSvg,
    });
  }

  return {
    summary,
    artifacts,
  };
}
