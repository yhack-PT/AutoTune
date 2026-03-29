export type ComparisonEvaluationSummary = {
  candidateWins: number;
  baselineWins: number;
  ties: number;
  totalCases: number;
  baseModelName: string;
};

export function ComparisonBar({
  data,
}: {
  data: ComparisonEvaluationSummary;
}) {
  const candidatePercent =
    data.totalCases > 0
      ? Math.round((data.candidateWins / data.totalCases) * 100)
      : 0;
  const baselinePercent =
    data.totalCases > 0
      ? Math.round((data.baselineWins / data.totalCases) * 100)
      : 0;

  return (
    <div className="flex justify-start">
      <div className="w-full max-w-md rounded-xl border border-border bg-card p-5 space-y-4">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-foreground">
            Evaluation Results
          </h3>
          <p className="text-xs text-muted-foreground">
            Judged on {data.totalCases} held-out example
            {data.totalCases !== 1 ? "s" : ""}
          </p>
        </div>

        <div className="space-y-3">
          {/* Post-trained bar */}
          <div className="space-y-1.5">
            <span className="text-xs font-medium text-foreground">
              Post-trained — {candidatePercent}%
            </span>
            <div className="h-2.5 w-full rounded-full bg-muted">
              <div
                className="h-2.5 rounded-full bg-foreground transition-all duration-700 ease-out"
                style={{ width: `${candidatePercent}%` }}
              />
            </div>
          </div>

          {/* Base model bar */}
          <div className="space-y-1.5">
            <span className="text-xs font-medium text-muted-foreground">
              Base model — {baselinePercent}%
            </span>
            <div className="h-2.5 w-full rounded-full bg-muted">
              <div
                className="h-2.5 rounded-full bg-muted-foreground/40 transition-all duration-700 ease-out"
                style={{ width: `${baselinePercent}%` }}
              />
            </div>
          </div>
        </div>

        {data.ties > 0 && (
          <p className="text-xs text-muted-foreground">
            {data.ties} tie{data.ties !== 1 ? "s" : ""}
          </p>
        )}
      </div>
    </div>
  );
}
