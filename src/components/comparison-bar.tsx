export type ComparisonEvaluationSummary =
  | {
    mode: "match_rate";
    candidateRate: number;
    baselineRate: number;
    totalCases: number;
    baseModelName: string;
    matchThresholdScore: number;
  }
  | {
    mode: "legacy_win_rate";
    candidateRate: number;
    baselineRate: number;
    totalCases: number;
    baseModelName: string;
    ties: number;
  };

function formatThresholdScore(score: number) {
  return Number.isInteger(score) ? String(score) : score.toFixed(1).replace(/\.0$/, "");
}

export function ComparisonBar({
  data,
}: {
  data: ComparisonEvaluationSummary;
}) {
  const candidatePercent = Math.round(data.candidateRate * 100);
  const baselinePercent = Math.round(data.baselineRate * 100);

  return (
    <div className="flex justify-start my-1">
      <div className="w-full max-w-md rounded-xl border border-border bg-card px-5 py-5 space-y-2">
        <h3 className="text-sm font-semibold text-foreground">
          Evaluation Results
        </h3>

        <div className="space-y-1">
          {/* Post-trained bar */}
          <div className="space-y-1.5">
            <span className="text-xs font-medium text-foreground">
              Post-trained{data.mode === "match_rate" ? " match rate" : ""} — {candidatePercent}%
            </span>
            <div className="h-2.5 w-full rounded-full bg-muted">
              <div
                className="h-2.5 rounded-full bg-foreground transition-all duration-700 ease-out"
                style={{ width: `${candidatePercent}%` }}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <span
              className="text-xs font-medium text-muted-foreground"
              title={data.baseModelName}
            >
              Base model{data.mode === "match_rate" ? " match rate" : ""} — {baselinePercent}%
            </span>
            <div className="h-2.5 w-full rounded-full bg-muted">
              <div
                className="h-2.5 rounded-full bg-muted-foreground/40 transition-all duration-700 ease-out"
                style={{ width: `${baselinePercent}%` }}
              />
            </div>
          </div>
        </div>

        {data.mode === "match_rate" ? (
          <p className="text-xs text-muted-foreground">
            Match = judge score &gt;= {formatThresholdScore(data.matchThresholdScore)}/10
          </p>
        ) : data.ties > 0 ? (
          <p className="text-xs text-muted-foreground">
            {data.ties} tie{data.ties !== 1 ? "s" : ""}
          </p>
        ) : null}
      </div>
    </div>
  );
}
