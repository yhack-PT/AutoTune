"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

interface PromptPillProps {
  gistTitle?: string;
  rawPrompt?: string;
  loading?: boolean;
}

export function PromptPill({ gistTitle, rawPrompt, loading }: PromptPillProps) {
  const [expanded, setExpanded] = useState(false);
  const hasExpandablePrompt =
    gistTitle && rawPrompt && rawPrompt.trim() !== gistTitle;

  return (
    <div
      className={cn(
        "w-full rounded-3xl border border-border bg-background/40 backdrop-blur-sm text-left transition-all duration-300 ease-in-out overflow-hidden cursor-default",
        hasExpandablePrompt && "hover:bg-background/50"
      )}
      onMouseEnter={() => hasExpandablePrompt && setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
      role={hasExpandablePrompt ? "button" : undefined}
      aria-label={gistTitle ?? "Loading title"}
    >
      <div className="px-4 py-2.5">
        {loading && !gistTitle ? (
          <Skeleton className="h-6 w-48 rounded-md" />
        ) : (
          <span className="text-lg font-semibold tracking-tight text-foreground">
            {gistTitle}
          </span>
        )}
      </div>
      {hasExpandablePrompt && (
        <div
          className={cn(
            "overflow-hidden transition-all duration-300 ease-in-out",
            expanded ? "max-h-48 opacity-100" : "max-h-0 opacity-0"
          )}
        >
          <p className="px-4 pb-3 pt-0 text-sm italic text-muted-foreground whitespace-pre-wrap break-words overflow-y-auto max-h-44">
            {rawPrompt}
          </p>
        </div>
      )}
    </div>
  );
}
