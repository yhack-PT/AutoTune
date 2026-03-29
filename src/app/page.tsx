"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  ArrowUp,
  Check,
  ChevronDown,
  Copy,
  Download,
  Play,
  Rewind,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import { getSidebarStageProgress, mergeStageProgressById } from "@/lib/posttraining-progress.mjs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  ComparisonBar,
  type ComparisonEvaluationSummary,
} from "@/components/comparison-bar";

// =============================================================================
// Types
// =============================================================================

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  deploymentUrl?: string;
  deploymentModel?: string;
  deploymentJobId?: string;
  comparisonEvaluation?: ComparisonEvaluationSummary;
}

interface PipelineStage {
  id: string;
  label: string;
}

interface StageProgressItem {
  id: string;
  text: string;
  tone: "normal" | "error";
}

const getSidebarStageProgressTyped = getSidebarStageProgress as unknown as (input: {
  logs: unknown[];
  activeStageId: string | null;
  completedStageIds: string[];
  failedStageId: string | null;
  jobStatus: string;
}) => Record<string, StageProgressItem[]>;

const mergeStageProgressByIdTyped = mergeStageProgressById as unknown as (
  previousProgress: Record<string, StageProgressItem[]>,
  nextProgress: Record<string, StageProgressItem[]>,
) => Record<string, StageProgressItem[]>;

// =============================================================================
// Pipeline stages (mapped to real backend stages)
// =============================================================================

const PIPELINE_STAGES: readonly PipelineStage[] = [
  {
    id: "recommending",
    label: "Finding training data...",
  },
  {
    id: "compiling",
    label: "Preparing the training plan...",
  },
  {
    id: "training",
    label: "Training your model...",
  },
  {
    id: "evaluating",
    label: "Checking model quality...",
  },
  {
    id: "deploying",
    label: "Getting the model ready to use...",
  },
  {
    id: "smoke_testing",
    label: "Running smoke tests...",
  },
];

function stageIndexById(id: string): number {
  return PIPELINE_STAGES.findIndex((s) => s.id === id);
}

function getStageById(id: string): PipelineStage | null {
  return PIPELINE_STAGES.find((stage) => stage.id === id) ?? null;
}

function getVisiblePipelineStageIds(job: {
  currentStage?: string;
  stageHistory?: Array<{ stage?: string }>;
}): string[] {
  const visible = new Set(
    Array.isArray(job.stageHistory)
      ? job.stageHistory
        .map((entry) => String(entry?.stage ?? ""))
        .filter((stage) => stageIndexById(stage) >= 0)
      : [],
  );

  const currentStage = String(job.currentStage ?? "");
  if (stageIndexById(currentStage) >= 0) {
    visible.add(currentStage);
  }

  return PIPELINE_STAGES.map((stage) => stage.id).filter((stageId) => visible.has(stageId));
}

function getCompletedPipelineStageIds(job: {
  stageHistory?: Array<{ stage?: string; status?: string }>;
}): string[] {
  const completed = new Set(
    Array.isArray(job.stageHistory)
      ? job.stageHistory
        .filter((entry) => entry?.status === "completed")
        .map((entry) => String(entry?.stage ?? ""))
        .filter((stage) => stageIndexById(stage) >= 0)
      : [],
  );

  return PIPELINE_STAGES.map((stage) => stage.id).filter((stageId) => completed.has(stageId));
}

function getInProgressPipelineStageId(job: {
  currentStage?: string;
  stageHistory?: Array<{ stage?: string; status?: string }>;
}): string | null {
  const inProgressStage = Array.isArray(job.stageHistory)
    ? [...job.stageHistory]
      .reverse()
      .find(
        (entry) =>
          entry?.status === "in_progress" && stageIndexById(String(entry?.stage ?? "")) >= 0,
      )
    : null;

  if (inProgressStage?.stage && stageIndexById(String(inProgressStage.stage)) >= 0) {
    return String(inProgressStage.stage);
  }

  const currentStage = String(job.currentStage ?? "");
  return stageIndexById(currentStage) >= 0 ? currentStage : null;
}

function getFailedPipelineStageId(job: {
  stageHistory?: Array<{ stage?: string; status?: string }>;
}): string | null {
  const failedStage = Array.isArray(job.stageHistory)
    ? [...job.stageHistory]
      .reverse()
      .find(
        (entry) =>
          entry?.status === "failed" && stageIndexById(String(entry?.stage ?? "")) >= 0,
      )
    : null;

  if (!failedStage?.stage) {
    return null;
  }

  return String(failedStage.stage);
}

function getDeploymentLinkInfo(job: {
  deployment?: Record<string, unknown> | null;
}): {
  deploymentUrl: string | null;
  deploymentModel: string | null;
} {
  const deployment =
    typeof job.deployment === "object" && job.deployment !== null
      ? job.deployment
      : null;

  return {
    deploymentUrl: typeof deployment?.url === "string" ? deployment.url : null,
    deploymentModel: typeof deployment?.adapterName === "string" ? deployment.adapterName : null,
  };
}

function getSelectedDatasetIds(job: {
  selectedDatasets?: unknown;
}): string[] {
  if (!Array.isArray(job.selectedDatasets)) {
    return [];
  }

  return job.selectedDatasets
    .filter((datasetId): datasetId is string => typeof datasetId === "string")
    .map((datasetId) => datasetId.trim())
    .filter(Boolean);
}

function buildDeploymentApiEndpoint(deploymentUrl: string): string {
  return `${deploymentUrl.replace(/\/+$/, "")}/v1/chat/completions`;
}

async function copyTextToClipboard(value: string): Promise<void> {
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  if (typeof document === "undefined") {
    throw new Error("Clipboard is unavailable.");
  }

  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";

  document.body.appendChild(textarea);
  textarea.select();

  const didCopy = document.execCommand("copy");
  document.body.removeChild(textarea);

  if (!didCopy) {
    throw new Error("Copy failed.");
  }
}

// =============================================================================
// Suggestions (empty state)
// =============================================================================

const SUGGESTIONS = [
  {
    label: "Foreign Law Expert",
    prompt:
      "I want to fine-tune a model to be an expert in Nepali and South Asian law. It should accurately interpret and explain Nepalese statutes, constitutional provisions, case law, and legal procedures — helping users navigate legal queries while always noting when professional legal counsel is required.",
  },
  {
    label: "AIME Tutor",
    prompt:
      "I want to fine-tune a model to tutor students preparing for AMC and AIME competitions. It should guide students through challenging problems step by step, building deep intuition in number theory, combinatorics, geometry, and algebra rather than just providing answers.",
  },
  {
    label: "Clinical Document Processor",
    prompt:
      "I want to fine-tune a model to process and summarize clinical healthcare documents. It should extract key information from physician notes, discharge summaries, lab reports, and EHRs — identifying diagnoses, medications, procedures, and follow-up actions while handling sensitive data responsibly.",
  },
];

// =============================================================================
// Message Bubble
// =============================================================================

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-3xl bg-muted px-5 py-3">
          <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="min-w-0 max-w-[80%] px-1">
        <div className="prose prose-sm prose-neutral break-words dark:prose-invert prose-p:my-1 prose-p:leading-relaxed prose-headings:my-2 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 max-w-none">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

function ProcessingIndicator({
  activeStageIndex,
  isSidebarOpen,
  onOpenDetails,
}: {
  activeStageIndex: number;
  isSidebarOpen: boolean;
  onOpenDetails: () => void;
}) {
  const activeStage = PIPELINE_STAGES[activeStageIndex];
  if (!activeStage) return null;

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] px-1">
        <button
          type="button"
          onClick={onOpenDetails}
          aria-expanded={isSidebarOpen}
          aria-controls="processing-sidebar"
          className="flex items-center gap-2 text-left text-sm font-medium text-foreground transition-opacity hover:opacity-70"
        >
          <span
            className="h-2 w-2 rounded-full bg-foreground"
            style={{ animation: "pulse-dot 1.5s ease-in-out infinite" }}
          />
          <span>{activeStage.label}</span>
        </button>
      </div>
    </div>
  );
}

function ProcessingSidebar({
  isOpen,
  selectedDatasets,
  visibleStageIds,
  stageProgressById,
  onClose,
}: {
  isOpen: boolean;
  selectedDatasets: string[];
  visibleStageIds: string[];
  stageProgressById: Record<string, StageProgressItem[]>;
  onClose: () => void;
}) {
  const visibleStages = visibleStageIds
    .map((stageId) => getStageById(stageId))
    .filter((stage): stage is PipelineStage => Boolean(stage));

  return (
    <aside
      id="processing-sidebar"
      aria-hidden={!isOpen}
      className="shrink-0 overflow-hidden bg-background transition-[width] duration-300 ease-out"
      style={{ width: isOpen ? "clamp(18rem, 38vw, 24rem)" : 0 }}
    >
      <div className="flex h-full min-h-0 w-full min-w-0 flex-col">
        <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-5">
          <div>
            <h2 className="text-base font-semibold text-foreground">
              Activity
            </h2>
          </div>

          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-5">
          <div className="space-y-8">
            <section className="space-y-3 rounded-2xl border border-border/70 bg-card/60 px-4 py-4">
              <div className="space-y-1">
                <h3 className="text-sm font-semibold text-foreground">
                  Training data
                </h3>
                <p className="text-xs leading-relaxed text-foreground/60">
                  {selectedDatasets.length > 0
                    ? `Using ${selectedDatasets.length === 1 ? "this dataset" : "these datasets"} to train your model.`
                    : "We'll show the selected datasets here as soon as they're chosen."}
                </p>
              </div>

              {selectedDatasets.length > 0 ? (
                <ul className="space-y-2">
                  {selectedDatasets.map((datasetId) => (
                    <li
                      key={datasetId}
                      className="rounded-xl border border-border/70 bg-background px-3 py-2"
                    >
                      <code className="block break-all font-mono text-xs leading-relaxed text-foreground">
                        {datasetId}
                      </code>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm leading-relaxed text-foreground/65">
                  
                </p>
              )}
            </section>

            {isOpen &&
              visibleStages.map((stage) => {
                const progressItems = stageProgressById[stage.id] ?? [];

                return (
                  <section key={stage.id} className="space-y-2.5">
                    <h3 className="text-sm font-semibold text-foreground">
                      {stage.label}
                    </h3>

                    {progressItems.length > 0 && (
                      <ul className="space-y-0.5">
                        {progressItems.map((progressItem, index) => (
                          <li
                            key={progressItem.id}
                            className="grid grid-cols-[0.875rem_1fr] gap-3 pb-3 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-top-1 motion-safe:duration-500 last:pb-0"
                          >
                            <div className="flex h-full min-h-6 flex-col items-center">
                              <span
                                className={cn(
                                  "mt-1.5 h-2 w-2 rounded-full border bg-background",
                                  progressItem.tone === "error"
                                    ? "border-destructive/60"
                                    : "border-foreground/30",
                                )}
                              />
                              {index < progressItems.length - 1 && (
                                <span className="mt-2 w-px flex-1 bg-border" />
                              )}
                            </div>

                            <p
                              className={cn(
                                "text-sm leading-relaxed",
                                progressItem.tone === "error"
                                  ? "text-destructive"
                                  : "text-foreground/65",
                              )}
                            >
                              {progressItem.text}
                            </p>
                          </li>
                        ))}
                      </ul>
                    )}
                  </section>
                );
              })}
          </div>
        </div>
      </div>
    </aside>
  );
}

// =============================================================================
// Chat Page
// =============================================================================

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [isResponding, setIsResponding] = useState(false);
  const [activeStageIndex, setActiveStageIndex] = useState<number | null>(null);
  const [visibleStageIds, setVisibleStageIds] = useState<string[]>([]);
  const [, setCompletedStageIds] = useState<string[]>([]);
  const [, setFailedStageId] = useState<string | null>(null);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [stageProgressById, setStageProgressById] = useState<Record<string, StageProgressItem[]>>(
    {},
  );
  const [isProcessSidebarOpen, setIsProcessSidebarOpen] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [fineTunedEndpoint, setFineTunedEndpoint] = useState<string | null>(null);
  const [fineTunedModel, setFineTunedModel] = useState<string | null>(null);
  const [copiedDeploymentUrl, setCopiedDeploymentUrl] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<"openai" | "finetuned">("openai");
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeJobIdRef = useRef<string | null>(null);
  const lastStageRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [showIndicator, setShowIndicator] = useState(false);
  const indicatorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const copiedIndicatorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const evalShownRef = useRef(false);
  const terminalStatusByJobRef = useRef<Record<string, "ready" | "failed">>({});

  // Scroll to bottom when new messages appear
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isResponding, activeStageIndex]);

  // Delay showing the processing indicator slightly to avoid flicker
  useEffect(() => {
    if (!activeJobId) {
      return;
    }

    indicatorTimerRef.current = setTimeout(() => setShowIndicator(true), 1500);
    return () => {
      if (indicatorTimerRef.current) clearTimeout(indicatorTimerRef.current);
    };
  }, [activeJobId]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const scrollH = textarea.scrollHeight;
      textarea.style.height = `${Math.min(Math.max(scrollH, 52), 200)}px`;
    }
  }, [input]);

  useEffect(() => {
    return () => {
      if (copiedIndicatorTimerRef.current) {
        clearTimeout(copiedIndicatorTimerRef.current);
      }
    };
  }, []);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    activeJobIdRef.current = null;
    lastStageRef.current = null;
    setActiveJobId(null);
    setIsResponding(false);
    setShowIndicator(false);
    inputRef.current?.focus();
  }, []);

  // Job polling
  useEffect(() => {
    if (!activeJobId) return;

    const jobId = activeJobId;

    const poll = async () => {
      try {
        const res = await fetch(`/api/posttraining/jobs/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();
        if (activeJobIdRef.current !== jobId) {
          return;
        }

        const currentStageId: string = String(job.currentStage ?? "");
        const nextActiveStageId =
          job.status === "ready" || job.status === "failed"
            ? null
            : getInProgressPipelineStageId(job);
        const idx = nextActiveStageId ? stageIndexById(nextActiveStageId) : -1;
        const nextVisibleStageIds = getVisiblePipelineStageIds(job);
        const nextCompletedStageIds = getCompletedPipelineStageIds(job);
        const nextFailedStageId = getFailedPipelineStageId(job);
        const nextSelectedDatasets = getSelectedDatasetIds(job);
        const nextStageProgressById = getSidebarStageProgressTyped({
          logs: Array.isArray(job.logs) ? job.logs : [],
          activeStageId: nextActiveStageId,
          completedStageIds: nextCompletedStageIds,
          failedStageId: nextFailedStageId,
          jobStatus: String(job.status ?? ""),
        });

        // Update sidebar
        if (idx >= 0) {
          setActiveStageIndex(idx);
        } else {
          setActiveStageIndex(null);
        }
        setVisibleStageIds(nextVisibleStageIds);
        setCompletedStageIds(nextCompletedStageIds);
        setFailedStageId(nextFailedStageId);
        setSelectedDatasets(nextSelectedDatasets);
        setStageProgressById((previousStageProgressById) =>
          mergeStageProgressByIdTyped(
            previousStageProgressById,
            nextStageProgressById,
          ),
        );

        // Track stage changes (no chat bubbles — ProcessingIndicator handles it)
        if (currentStageId !== lastStageRef.current) {
          lastStageRef.current = currentStageId;
        }

        // Show comparison evaluation bar when evaluating stage completes
        if (
          !evalShownRef.current &&
          nextCompletedStageIds.includes("evaluating") &&
          job.evaluation?.summary &&
          job.evaluation?.show_evaluation_component !== false
        ) {
          evalShownRef.current = true;
          const evalSummary = job.evaluation.summary;
          const evalCases = job.evaluation.sample_policy?.sampled_cases;
          if (typeof evalCases === "number") {
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: "",
                comparisonEvaluation: {
                  candidateWins: evalSummary.candidate_wins ?? 0,
                  baselineWins: evalSummary.baseline_wins ?? 0,
                  ties: evalSummary.ties ?? 0,
                  totalCases: evalCases,
                  baseModelName:
                    job.evaluation.baseline?.model_id ?? "Base model",
                },
              },
            ]);
          }
        }

        // Terminal states
        if (job.status === "ready") {
          const { deploymentUrl, deploymentModel } = getDeploymentLinkInfo(job);
          if (deploymentUrl) {
            setFineTunedEndpoint(deploymentUrl);
            setFineTunedModel(deploymentModel);
          }
          setCompletedStageIds(nextVisibleStageIds);
          setActiveStageIndex(null);
          setFailedStageId(null);
          setStageProgressById((previousStageProgressById) =>
            mergeStageProgressByIdTyped(
              previousStageProgressById,
              nextStageProgressById,
            ),
          );
          if (terminalStatusByJobRef.current[jobId] !== "ready") {
            terminalStatusByJobRef.current[jobId] = "ready";
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: "Your model is ready!",
                deploymentJobId: jobId,
                ...(deploymentUrl
                  ? {
                    deploymentUrl,
                    ...(deploymentModel ? { deploymentModel } : {}),
                  }
                  : {}),
              },
            ]);
          }
          stopPolling();
        } else if (job.status === "failed") {
          setActiveStageIndex(null);
          setStageProgressById((previousStageProgressById) =>
            mergeStageProgressByIdTyped(
              previousStageProgressById,
              nextStageProgressById,
            ),
          );
          if (terminalStatusByJobRef.current[jobId] !== "failed") {
            terminalStatusByJobRef.current[jobId] = "failed";
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: `The training job failed: ${job.errorSummary || "Unknown error."}`,
              },
            ]);
          }
          stopPolling();
        }
      } catch {
        // Ignore transient polling errors
      }
    };

    pollingRef.current = setInterval(poll, 3000);
    poll(); // immediate first poll

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [activeJobId, stopPolling]);

  const sendToChat = useCallback(
    async (chatMessages: ChatMessage[]) => {
      setIsResponding(true);

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: chatMessages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
            ...(activeModel === "finetuned" && fineTunedEndpoint
              ? {
                customEndpoint: fineTunedEndpoint,
                ...(fineTunedModel ? { customModel: fineTunedModel } : {}),
              }
              : {}),
          }),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          const errorText = await res.text().catch(() => "Unknown error");
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: `Something went wrong: ${errorText}`,
            },
          ]);
          setIsResponding(false);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let jobStarted = false;

        // Create a streaming assistant message that we'll update in place
        const streamingMsgId = crypto.randomUUID();
        let streamingContent = "";
        let messageAdded = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data: ")) continue;
            const json = trimmed.slice(6);
            if (!json) continue;

            try {
              const event = JSON.parse(json);

              if (event.type === "text_delta" && event.content) {
                streamingContent += event.content;

                if (!messageAdded) {
                  // Add the message for the first token
                  messageAdded = true;
                  setMessages((prev) => [
                    ...prev,
                    {
                      id: streamingMsgId,
                      role: "assistant",
                      content: streamingContent,
                    },
                  ]);
                } else {
                  // Update the existing message in place
                  const snapshot = streamingContent;
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === streamingMsgId
                        ? { ...m, content: snapshot }
                        : m,
                    ),
                  );
                }
              } else if (event.type === "job_started" && event.jobId) {
                jobStarted = true;
                setShowIndicator(false);
                activeJobIdRef.current = event.jobId;
                setActiveJobId(event.jobId);
                setActiveStageIndex(0);
                setVisibleStageIds(["recommending"]);
                setCompletedStageIds([]);
                setFailedStageId(null);
                setSelectedDatasets([]);
                setStageProgressById({});
                setIsProcessSidebarOpen(true);
                lastStageRef.current = null;
                evalShownRef.current = false;
              } else if (event.type === "error") {
                streamingContent += event.message || "An error occurred.";
                if (!messageAdded) {
                  messageAdded = true;
                  setMessages((prev) => [
                    ...prev,
                    {
                      id: streamingMsgId,
                      role: "assistant",
                      content: streamingContent,
                    },
                  ]);
                } else {
                  const snapshot = streamingContent;
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === streamingMsgId
                        ? { ...m, content: snapshot }
                        : m,
                    ),
                  );
                }
              }
            } catch {
              // Skip malformed SSE events
            }
          }
        }

        // If no job was started, we're done responding
        if (!jobStarted) {
          setIsResponding(false);
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "Failed to reach the server. Please try again.",
          },
        ]);
        setIsResponding(false);
      }
    },
    [activeModel, fineTunedEndpoint, fineTunedModel],
  );

  const handleSubmit = useCallback(
    (text?: string) => {
      const message = (text || input).trim();
      if (!message || isResponding) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: message,
      };

      const updatedMessages = [...messages, userMsg];
      setMessages(updatedMessages);
      setInput("");
      sendToChat(updatedMessages);
    },
    [input, isResponding, messages, sendToChat],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const handleCopyDeploymentEndpoint = useCallback(async (deploymentUrl: string) => {
    try {
      await copyTextToClipboard(buildDeploymentApiEndpoint(deploymentUrl));
      setCopiedDeploymentUrl(deploymentUrl);

      if (copiedIndicatorTimerRef.current) {
        clearTimeout(copiedIndicatorTimerRef.current);
      }

      copiedIndicatorTimerRef.current = setTimeout(() => {
        setCopiedDeploymentUrl((currentUrl) =>
          currentUrl === deploymentUrl ? null : currentUrl,
        );
        copiedIndicatorTimerRef.current = null;
      }, 2000);
    } catch {
      setCopiedDeploymentUrl(null);
    }
  }, []);

  const isEmpty = messages.length === 0;
  const currentStage =
    activeStageIndex === null ? null : PIPELINE_STAGES[activeStageIndex];

  return (
    <div className="flex h-screen overflow-hidden bg-background transition-colors duration-300" data-model-mode={activeModel}>
      <div className="relative flex min-w-0 flex-1 flex-col">
        {/* Top bar */}
        <header className="shrink-0 flex items-center justify-between px-5 py-3 border-b border-border/50">
          <div className="flex items-center gap-3">
            {fineTunedEndpoint ? (
              <DropdownMenu>
                <DropdownMenuTrigger className="flex items-center gap-1 text-sm font-medium text-foreground/80 hover:text-foreground outline-hidden">
                  {activeModel === "finetuned" ? "Fine-tuned Model" : "AutoTune"}
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="center" style={{ borderRadius: '16px', padding: '6px' }} className="bg-white border border-border/50 text-foreground shadow-lg min-w-[13rem] z-50">
                  {activeModel !== "openai" && (
                    <DropdownMenuItem onClick={() => setActiveModel("openai")} style={{ borderRadius: '10px', padding: '8px 10px' }} className="flex justify-between items-center w-full cursor-pointer hover:bg-black/5 transition-colors text-sm font-medium">
                      AutoTune
                    </DropdownMenuItem>
                  )}
                  {activeModel !== "finetuned" && (
                    <DropdownMenuItem onClick={() => setActiveModel("finetuned")} style={{ borderRadius: '10px', padding: '8px 10px' }} className="flex justify-between items-center w-full cursor-pointer hover:bg-black/5 transition-colors text-sm font-medium">
                      Post-trained Model
                    </DropdownMenuItem>
                  )}
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <span className="text-sm font-medium text-foreground/80">AutoTune</span>
            )}
          </div>
          <div className="flex items-center gap-2" />
        </header>

        {/* Messages area */}
        <main
          ref={scrollContainerRef}
          className="flex-1 overflow-y-auto"
        >
          {isEmpty ? (
            /* Empty state */
            <div className="h-full flex flex-col items-center justify-center px-6 pb-24">
              <div className="max-w-2xl w-full space-y-5">
                <div className="text-center space-y-1">
                  <h2 className="text-2xl font-semibold tracking-tight">
                    What should your model be an expert at?
                  </h2>
                  <p className="text-sm text-muted-foreground max-w-md mx-auto">
                    Create a custom model using public datasets.
                  </p>
                </div>

                {/* Suggestions */}
                <div className="flex flex-wrap gap-2 justify-center">
                  {SUGGESTIONS.map((s) => (
                    <Button
                      key={s.label}
                      variant="outline"
                      size="sm"
                      className="text-muted-foreground hover:text-foreground"
                      onClick={() => handleSubmit(s.prompt)}
                    >
                      <Sparkles className="h-3 w-3" />
                      {s.label}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            /* Message list */
            <div className="max-w-3xl mx-auto px-4 pt-6 pb-36 space-y-6">
              {messages.map((msg) => (
                <div key={msg.id}>
                  {msg.comparisonEvaluation ? (
                    <ComparisonBar data={msg.comparisonEvaluation} />
                  ) : (
                    <MessageBubble message={msg} />
                  )}
                  {msg.deploymentUrl && (
                    <div className="flex justify-start pl-1 pt-3">
                      <div className="w-full max-w-xl space-y-3">
                        <Button
                          variant="outline"
                          size="sm"
                          className={cn(
                            "gap-2",
                            activeModel === "finetuned"
                              ? "border-transparent bg-white text-black hover:bg-white/80"
                              : "border-green-300 bg-green-50 text-green-700 hover:bg-green-100 hover:text-green-800",
                          )}
                          onClick={() => {
                            if (activeModel === "finetuned") {
                              setActiveModel("openai");
                            } else {
                              setFineTunedEndpoint(msg.deploymentUrl!);
                              setFineTunedModel(msg.deploymentModel ?? null);
                              setActiveModel("finetuned");
                            }
                          }}
                        >
                          {activeModel === "finetuned" ? (
                            <>
                              <Rewind className="h-3 w-3" />
                              Switch to AutoTune
                            </>
                          ) : (
                            <>
                              <Play className="h-3 w-3" />
                              Try this model now
                            </>
                          )}
                        </Button>

                        <div className="rounded-2xl border border-border/70 bg-card/90 px-4 py-3">
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0 space-y-1">
                              <p className="text-[0.65rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                                API endpoint
                              </p>
                              <code className="block break-all font-mono text-xs leading-relaxed text-foreground">
                                {buildDeploymentApiEndpoint(msg.deploymentUrl)}
                              </code>
                            </div>

                            <div className="flex shrink-0 items-center gap-2">
                              <Button
                                type="button"
                                variant="outline"
                                size="xs"
                                className="shrink-0"
                                onClick={() => handleCopyDeploymentEndpoint(msg.deploymentUrl!)}
                              >
                                {copiedDeploymentUrl === msg.deploymentUrl ? (
                                  <>
                                    <Check className="h-3 w-3" />
                                    Copied
                                  </>
                                ) : (
                                  <>
                                    <Copy className="h-3 w-3" />
                                    Copy
                                  </>
                                )}
                              </Button>

                              {msg.deploymentJobId && (
                                <Button asChild variant="outline" size="xs" className="gap-2">
                                  <a
                                    href={`/api/posttraining/jobs/${msg.deploymentJobId}/weights`}
                                    download
                                  >
                                    <Download className="h-3 w-3" />
                                    Download weights
                                  </a>
                                </Button>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {isResponding && activeStageIndex !== null && showIndicator && (
                <div className="-mt-6">
                  <ProcessingIndicator
                    activeStageIndex={activeStageIndex}
                    isSidebarOpen={isProcessSidebarOpen}
                    onOpenDetails={() => setIsProcessSidebarOpen(true)}
                  />
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </main>

        {/* Input bar — pinned at bottom */}
        <div className="absolute bottom-0 left-0 right-0 px-4 pb-8 pt-8 bg-gradient-to-t from-background via-background/95 to-transparent pointer-events-none z-10">
          <div className="max-w-3xl mx-auto pointer-events-auto">
            <div className="relative">
              <Textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  isResponding && currentStage
                    ? currentStage.label
                    : activeModel === "finetuned"
                      ? "Try your post-trained model..."
                      : "e.g. Build me an AI expert at drafting clinical notes from visit transcripts..."
                }
                rows={1}
                className={cn(
                  "resize-none shadow-lg border-border/60 focus-visible:ring-0 focus-visible:border-border min-h-0 pr-12 bg-card text-sm",
                  isResponding && "opacity-70",
                )}
                style={{
                  minHeight: "52px",
                  maxHeight: "200px",
                  overflow: "auto",
                  borderRadius: "1.625rem",
                  paddingLeft: "1.25rem",
                  paddingRight: "3.5rem",
                  paddingTop: "14px",
                  paddingBottom: "14px",
                }}
                disabled={isResponding}
                autoFocus
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2">
                <Button
                  size="icon-sm"
                  className="rounded-full"
                  onClick={() => handleSubmit()}
                  disabled={!input.trim() || isResponding}
                >
                  <ArrowUp className="h-4 w-4" />
                </Button>
              </div>
            </div>

          </div>
        </div>
      </div>

      <ProcessingSidebar
        isOpen={isProcessSidebarOpen && visibleStageIds.length > 0}
        selectedDatasets={selectedDatasets}
        visibleStageIds={visibleStageIds}
        stageProgressById={stageProgressById}
        onClose={() => setIsProcessSidebarOpen(false)}
      />
    </div>
  );
}
