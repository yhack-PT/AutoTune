"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ArrowUp, Sparkles, Plus } from "lucide-react";
import { cn } from "@/lib/utils";

// =============================================================================
// Types
// =============================================================================

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface ProcessStage {
  id: string;
  label: string;
  detail: string;
  sidebarInfo: string;
  durationMs: number;
}

// =============================================================================
// Dummy responses (no API wired up yet)
// =============================================================================

const DUMMY_RESPONSES = [
  "I'm a placeholder response — no model is connected yet! Once an API is wired up, you'll see real answers here.",
  "This is a static demo of the chat interface. Everything you see is running locally with no backend.",
  "Great question! Unfortunately I'm just a dummy response for now. The real magic comes when we connect a model.",
  "Hey! I'm not ChatGPT (yet). This is a UI prototype to nail the look and feel first.",
  "Roger that. I've noted your message but can't actually process it — I'm just a mock response.",
];

function getDummyResponse(): string {
  return DUMMY_RESPONSES[Math.floor(Math.random() * DUMMY_RESPONSES.length)]!;
}

const FAKE_PROCESS_STAGES: readonly ProcessStage[] = [
  {
    id: "thinking",
    label: "Reading your request...",
    detail: "We are figuring out what you want this AI to do.",
    sidebarInfo:
      "This placeholder step stands in for the first pass over your request. Later, the backend can use this area to explain how it understood your goal, what kind of assistant you are asking for, and any assumptions it is making before moving on.",
    durationMs: 2800,
  },
  {
    id: "planning",
    label: "Gathering examples...",
    detail: "We are deciding what kind of examples this AI should learn from.",
    sidebarInfo:
      "This filler text represents the step where the system chooses the kinds of examples, topics, and edge cases that should shape the AI's behavior. In the future, this section can show sample content, categories, or sources being considered.",
    durationMs: 3000,
  },
  {
    id: "tuning",
    label: "Teaching the AI...",
    detail: "We are shaping the AI so it responds in the way you want.",
    sidebarInfo:
      "This is a stand-in for the part where the system applies the examples and instructions to steer the model. Once the backend is connected, this panel can show more useful progress details about how the AI is being adapted.",
    durationMs: 3600,
  },
  {
    id: "evaluating",
    label: "Checking the results...",
    detail: "We are making sure the AI's answers look accurate and consistent.",
    sidebarInfo:
      "This placeholder section represents the review step. Later, it can show example outputs, quality checks, and simple summaries of whether the current setup is meeting the expected standard before the response is returned.",
    durationMs: 2800,
  },
  {
    id: "finalizing",
    label: "Getting your reply ready...",
    detail: "We are wrapping things up before showing the response in chat.",
    sidebarInfo:
      "This filler copy stands in for the last cleanup step before the answer appears. In a real version, this area could explain any final formatting, packaging, or response preparation that happens just before the reply is shown.",
    durationMs: 2800,
  },
];

const TOTAL_FAKE_PROCESS_MS = FAKE_PROCESS_STAGES.reduce(
  (total, stage) => total + stage.durationMs,
  0,
);

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
      <div className="max-w-[80%] px-1">
        <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap">
          {message.content}
        </p>
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
  const activeStage = FAKE_PROCESS_STAGES[activeStageIndex]!;

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
          <span className="h-2 w-2 rounded-full bg-foreground motion-safe:animate-pulse" />
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
              FAKE_PROCESS_STAGES.slice(0, visibleStageCount).map((stage, index) => {
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
                    <p className="text-sm leading-relaxed text-muted-foreground">
                      {stage.sidebarInfo}
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
  const responseTimeoutsRef = useRef<number[]>([]);
  const activeResponseRunRef = useRef(0);

  // Scroll to bottom when new messages appear
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isResponding, activeStageIndex]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const scrollH = textarea.scrollHeight;
      textarea.style.height = `${Math.min(Math.max(scrollH, 52), 200)}px`;
    }
  }, [input]);

  const clearResponseTimers = useCallback(() => {
    responseTimeoutsRef.current.forEach((timeoutId) => {
      window.clearTimeout(timeoutId);
    });
    responseTimeoutsRef.current = [];
  }, []);

  const cancelActiveResponse = useCallback(
    (resetState = true) => {
      activeResponseRunRef.current += 1;
      clearResponseTimers();

      if (resetState) {
        setIsResponding(false);
        setActiveStageIndex(null);
        setCompletedStageCount(0);
        setIsProcessSidebarOpen(false);
      }
    },
    [clearResponseTimers],
  );

  const startFakeResponseProcess = useCallback(() => {
    const runId = activeResponseRunRef.current + 1;

    activeResponseRunRef.current = runId;
    setIsResponding(true);
    setActiveStageIndex(0);
    setCompletedStageCount(0);
    setIsProcessSidebarOpen(false);

    let elapsedBeforeStage = 0;

    for (let index = 1; index < FAKE_PROCESS_STAGES.length; index += 1) {
      elapsedBeforeStage += FAKE_PROCESS_STAGES[index - 1]!.durationMs;

      const timeoutId = window.setTimeout(() => {
        if (activeResponseRunRef.current !== runId) return;

        setActiveStageIndex(index);
        setCompletedStageCount(index);
      }, elapsedBeforeStage);

      responseTimeoutsRef.current.push(timeoutId);
    }

    const completeTimeoutId = window.setTimeout(() => {
      if (activeResponseRunRef.current !== runId) return;

      setCompletedStageCount(FAKE_PROCESS_STAGES.length);
    }, TOTAL_FAKE_PROCESS_MS);

    const finishTimeoutId = window.setTimeout(() => {
      if (activeResponseRunRef.current !== runId) return;

      clearResponseTimers();
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: getDummyResponse(),
        },
      ]);
      setIsResponding(false);
      setActiveStageIndex(null);
      setCompletedStageCount(0);
      setIsProcessSidebarOpen(false);
      inputRef.current?.focus();
    }, TOTAL_FAKE_PROCESS_MS + 220);

    responseTimeoutsRef.current.push(completeTimeoutId);
    responseTimeoutsRef.current.push(finishTimeoutId);
  }, [clearResponseTimers]);

  useEffect(() => {
    return () => {
      activeResponseRunRef.current += 1;
      clearResponseTimers();
    };
  }, [clearResponseTimers]);

  const handleSubmit = useCallback(
    (text?: string) => {
      const message = (text || input).trim();
      if (!message || isResponding || responseTimeoutsRef.current.length > 0) {
        return;
      }

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: message,
      };

      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      startFakeResponseProcess();
    },
    [input, isResponding, startFakeResponseProcess],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const handleNewChat = useCallback(() => {
    cancelActiveResponse();
    setMessages([]);
    setInput("");
    inputRef.current?.focus();
  }, [cancelActiveResponse]);

  const isEmpty = messages.length === 0;
  const currentStage =
    activeStageIndex === null ? null : FAKE_PROCESS_STAGES[activeStageIndex];

  return (
    <div className="flex h-screen overflow-hidden bg-background">
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
            <span className="text-sm font-medium text-foreground/80">
              ChatGPT
            </span>
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
            <div className="h-full flex flex-col items-center justify-center px-6">
              <div className="max-w-2xl w-full space-y-8">
                <div className="text-center space-y-3">
                  <div className="mx-auto w-12 h-12 rounded-2xl bg-foreground flex items-center justify-center mb-4">
                    <Sparkles className="h-6 w-6 text-background" />
                  </div>
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
                <MessageBubble key={msg.id} message={msg} />
              ))}

              {isResponding && activeStageIndex !== null && (
                <ProcessingIndicator
                  activeStageIndex={activeStageIndex}
                  isSidebarOpen={isProcessSidebarOpen}
                  onOpenDetails={() => setIsProcessSidebarOpen(true)}
                />
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
                    : "Message ChatGPT..."
                }
                rows={1}
                className={cn(
                  "resize-none shadow-lg border-border/60 focus-visible:ring-0 focus-visible:border-border min-h-0 pr-12 bg-card text-sm",
                  isResponding && "opacity-70"
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
        isOpen={isResponding && isProcessSidebarOpen}
        activeStageIndex={activeStageIndex}
        completedStageCount={completedStageCount}
        onClose={() => setIsProcessSidebarOpen(false)}
      />
    </div>
  );
}
