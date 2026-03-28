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

// =============================================================================
// Suggestions (empty state)
// =============================================================================

const SUGGESTIONS = [
  {
    label: "Brainstorm names",
    prompt: "Help me brainstorm creative names for a new product",
  },
  {
    label: "Explain a concept",
    prompt: "Explain quantum computing in simple terms",
  },
  {
    label: "Write an email",
    prompt: "Draft a professional email declining a meeting",
  },
  {
    label: "Debug code",
    prompt: "Help me debug this Python function that's returning None",
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

  // Scroll to bottom when new messages appear
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const scrollH = textarea.scrollHeight;
      textarea.style.height = `${Math.min(Math.max(scrollH, 52), 200)}px`;
    }
  }, [input]);

  const handleSubmit = useCallback(
    async (text?: string) => {
      const message = (text || input).trim();
      if (!message || isResponding) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: message,
      };

      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setIsResponding(true);

      // Simulate a short delay for the "thinking" response
      await new Promise((r) => setTimeout(r, 800 + Math.random() * 1200));

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: getDummyResponse(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
      setIsResponding(false);
    },
    [input, isResponding]
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
    setMessages([]);
    setInput("");
    inputRef.current?.focus();
  }, []);

  const isEmpty = messages.length === 0;

  return (
    <div className="h-screen flex flex-col overflow-hidden relative bg-background">
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

            {/* Thinking indicator */}
            {isResponding && (
              <div className="flex justify-start">
                <div className="px-1">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      {/* Input bar — pinned at bottom */}
      <div className="absolute bottom-0 left-0 right-0 px-4 pb-6 pt-8 bg-gradient-to-t from-background via-background/95 to-transparent pointer-events-none z-10">
        <div className="max-w-3xl mx-auto pointer-events-auto">
          <div className="relative">
            <Textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isResponding
                  ? "Waiting for response..."
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

          <p className="text-[11px] text-muted-foreground/60 text-center mt-2.5">
            Demo mode — responses are placeholders. No data is sent anywhere.
          </p>
        </div>
      </div>
    </div>
  );
}
