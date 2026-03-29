"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ArrowUp, Sparkles, Plus, Play, Rewind, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

// =============================================================================
// Types
// =============================================================================

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  deploymentUrl?: string;
}

interface PipelineStage {
  id: string;
  label: string;
  detail: string;
}

// =============================================================================
// Pipeline stages (mapped to real backend stages)
// =============================================================================

const PIPELINE_STAGES: readonly PipelineStage[] = [
  {
    id: "recommending",
    label: "Finding datasets...",
    detail: "Searching Hugging Face for relevant training data.",
  },
  {
    id: "compiling",
    label: "Compiling training spec...",
    detail: "Building the training configuration from recommended datasets.",
  },
  {
    id: "training",
    label: "Training the model...",
    detail: "Fine-tuning the model on Modal GPUs.",
  },
  {
    id: "deploying",
    label: "Deploying the model...",
    detail: "Spinning up a vLLM inference server on Modal.",
  },
  {
    id: "smoke_testing",
    label: "Running smoke tests...",
    detail: "Verifying the deployed model responds correctly.",
  },
];

function stageIndexById(id: string): number {
  return PIPELINE_STAGES.findIndex((s) => s.id === id);
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
      <div className="max-w-[80%] px-1 prose prose-sm prose-neutral dark:prose-invert prose-p:leading-relaxed prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 max-w-none">
        <ReactMarkdown>{message.content}</ReactMarkdown>
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
  activeStageIndex,
  completedStageCount,
  onClose,
}: {
  isOpen: boolean;
  activeStageIndex: number | null;
  completedStageCount: number;
  onClose: () => void;
}) {
  const visibleStageCount =
    activeStageIndex === null
      ? completedStageCount
      : Math.max(completedStageCount, activeStageIndex + 1);

  return (
    <aside
      id="processing-sidebar"
      aria-hidden={!isOpen}
      className={cn(
        "shrink-0 overflow-hidden border-l bg-background transition-[width,border-color] duration-300 ease-out",
        isOpen ? "border-border" : "border-transparent",
      )}
      style={{ width: isOpen ? "min(24rem, 38vw)" : 0 }}
    >
      <div className="flex h-full min-h-0 w-[min(24rem,38vw)] min-w-[18rem] flex-col">
        <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-5">
          <div>
            <h2 className="text-base font-semibold text-foreground">
              What&apos;s Happening
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              A simple view of the steps being completed for this request.
            </p>
          </div>

          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-5">
          <div className="space-y-8">
            {isOpen &&
              PIPELINE_STAGES.slice(0, visibleStageCount).map((stage, index) => {
                const isComplete = index < completedStageCount;
                const isActive = index === activeStageIndex;
                const status = isComplete ? "Done" : "Working on this now";

                return (
                  <section key={stage.id} className="space-y-2.5">
                    <div className="space-y-1">
                      <h3 className="text-sm font-semibold text-foreground">
                        {stage.label}
                      </h3>
                      <p
                        className={cn(
                          "text-[11px] font-medium uppercase tracking-[0.16em]",
                          isComplete && "text-foreground/75",
                          isActive && "text-foreground",
                          !isComplete && !isActive && "text-muted-foreground/55",
                        )}
                      >
                        {status}
                      </p>
                    </div>

                    <p className="text-sm leading-relaxed text-muted-foreground">
                      {stage.detail}
                    </p>
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
  const [completedStageCount, setCompletedStageCount] = useState(0);
  const [isProcessSidebarOpen, setIsProcessSidebarOpen] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [fineTunedEndpoint, setFineTunedEndpoint] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<"openai" | "finetuned">("openai");
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastStageRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [showIndicator, setShowIndicator] = useState(false);
  const indicatorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Scroll to bottom when new messages appear
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isResponding, activeStageIndex]);

  // Delay showing the processing indicator by 3 seconds
  useEffect(() => {
    if (activeJobId) {
      indicatorTimerRef.current = setTimeout(() => setShowIndicator(true), 1500);
    } else {
      setShowIndicator(false);
    }
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

  // Job polling
  useEffect(() => {
    if (!activeJobId) return;

    const poll = async () => {
      try {
        const res = await fetch(`/api/posttraining/jobs/${activeJobId}`);
        if (!res.ok) return;
        const job = await res.json();

        const currentStageId: string = job.currentStage;
        const idx = stageIndexById(currentStageId);

        // Update sidebar
        if (idx >= 0) {
          setActiveStageIndex(idx);
          setCompletedStageCount(idx);
        }

        // Track stage changes (no chat bubbles — ProcessingIndicator handles it)
        if (currentStageId !== lastStageRef.current) {
          lastStageRef.current = currentStageId;
        }

        // Terminal states
        if (job.status === "ready") {
          const deploymentUrl =
            job.deployment && typeof job.deployment.url === "string"
              ? job.deployment.url
              : null;
          if (deploymentUrl) {
            setFineTunedEndpoint(deploymentUrl);
          }
          setCompletedStageCount(PIPELINE_STAGES.length);
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: "Your model is ready!",
              ...(deploymentUrl ? { deploymentUrl } : {}),
            },
          ]);
          stopPolling();
        } else if (job.status === "failed") {
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: `The training job failed: ${job.errorSummary || "Unknown error."}`,
            },
          ]);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJobId]);

  function stopPolling() {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setActiveJobId(null);
    setIsResponding(false);
    setActiveStageIndex(null);
    inputRef.current?.focus();
  }

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
              ? { customEndpoint: fineTunedEndpoint }
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
                setActiveJobId(event.jobId);
                setActiveStageIndex(0);
                setCompletedStageCount(0);
                setIsProcessSidebarOpen(true);
                lastStageRef.current = null;
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
    [activeModel, fineTunedEndpoint],
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

  const handleNewChat = useCallback(() => {
    abortRef.current?.abort();
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setMessages([]);
    setInput("");
    setIsResponding(false);
    setActiveStageIndex(null);
    setCompletedStageCount(0);
    setIsProcessSidebarOpen(false);
    setActiveJobId(null);
    setActiveModel("openai");
    setFineTunedEndpoint(null);
    lastStageRef.current = null;
    inputRef.current?.focus();
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
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={handleNewChat}
              title="New chat"
            >
              <Plus className="h-4 w-4" />
            </Button>
            {fineTunedEndpoint ? (
              <DropdownMenu>
                <DropdownMenuTrigger className="flex items-center gap-1 text-sm font-medium text-foreground/80 hover:text-foreground outline-hidden">
                  {activeModel === "finetuned" ? "Fine-tuned Model" : "ChatGPT"}
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="center" style={{ borderRadius: '16px', padding: '6px' }} className="bg-white border border-border/50 text-foreground shadow-lg min-w-[13rem] z-50">
                  {activeModel !== "openai" && (
                    <DropdownMenuItem onClick={() => setActiveModel("openai")} style={{ borderRadius: '10px', padding: '8px 10px' }} className="flex justify-between items-center w-full cursor-pointer hover:bg-black/5 transition-colors text-sm font-medium">
                      ChatGPT
                    </DropdownMenuItem>
                  )}
                  {activeModel !== "finetuned" && (
                    <DropdownMenuItem onClick={() => setActiveModel("finetuned")} style={{ borderRadius: '10px', padding: '8px 10px' }} className="flex justify-between items-center w-full cursor-pointer hover:bg-black/5 transition-colors text-sm font-medium">
                      Fine-tuned Model
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
                    What should your AI do?
                  </h2>
                  <p className="text-sm text-muted-foreground max-w-md mx-auto">
                    Build a specialized AI for any use case.
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
                  <MessageBubble message={msg} />
                  {msg.deploymentUrl && (
                    <div className="flex justify-start pl-1 pt-3">
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
                            setActiveModel("finetuned");
                          }
                        }}
                      >
                        {activeModel === "finetuned" ? (
                          <>
                            <Rewind className="h-3 w-3" />
                            Switch to ChatGPT
                          </>
                        ) : (
                          <>
                            <Play className="h-3 w-3" />
                            Try this model now
                          </>
                        )}
                      </Button>
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
                    : "Describe the AI you want to build..."
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
        isOpen={isProcessSidebarOpen && activeStageIndex !== null}
        activeStageIndex={activeStageIndex}
        completedStageCount={completedStageCount}
        onClose={() => setIsProcessSidebarOpen(false)}
      />
    </div>
  );
}
