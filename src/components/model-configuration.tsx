"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

const levels = [
  { label: "Max Reasoning", sublabel: "Slowest", value: 1 },
  { label: "High Reasoning", sublabel: "Slow", value: 2 },
  { label: "Balanced", sublabel: "Moderate", value: 3 },
  { label: "Fast", sublabel: "Lower Reasoning", value: 4 },
  { label: "Fastest", sublabel: "Min Reasoning", value: 5 },
];

export function ModelConfiguration() {
  const [domain, setDomain] = useState("");
  const [task, setTask] = useState("");
  const [reasoningLevel, setReasoningLevel] = useState(3);

  const handleSubmit = () => {
    const config = {
      domain,
      task,
      reasoningLevel: levels.find((l) => l.value === reasoningLevel)?.label,
    };
    console.log("Configuration:", config);
    alert("Configuration saved! Check console for details.");
  };

  return (
    <div className="w-full max-w-lg overflow-hidden rounded-xl border border-border bg-card text-card-foreground shadow-sm transition-colors">
      <div className="border-b border-border p-6">
        <h1 className="text-xl font-semibold text-foreground">
          Model Configuration
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Define your model parameters and reasoning preferences
        </p>
      </div>

      <div className="p-8">
        <div className="space-y-6">
          <div className="space-y-2">
            <Label
              htmlFor="domain"
              className="text-sm font-medium text-foreground"
            >
              Domain
            </Label>
            <Input
              id="domain"
              placeholder="e.g., Healthcare, Finance, Legal..."
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              className="h-11"
            />
          </div>

          <div className="space-y-2">
            <Label
              htmlFor="task"
              className="text-sm font-medium text-foreground"
            >
              What should the model do?
            </Label>
            <Textarea
              id="task"
              placeholder="Describe the task or capability you want the model to perform..."
              value={task}
              onChange={(e) => setTask(e.target.value)}
              className="min-h-[120px] resize-none"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium text-foreground">
              Reasoning vs Speed
            </Label>

            <div className="relative flex rounded-lg border border-neutral-200 bg-neutral-100 p-1">
              {/* Sliding indicator */}
              <div
                className="absolute top-1 bottom-1 rounded-md bg-white shadow-sm ring-1 ring-neutral-200 transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]"
                style={{
                  width: `calc((100% - 8px) / ${levels.length})`,
                  left: `calc(4px + ${(reasoningLevel - 1)} * (100% - 8px) / ${levels.length})`,
                }}
              />
              {levels.map((level) => (
                <button
                  key={level.value}
                  onClick={() => setReasoningLevel(level.value)}
                  className={`relative z-10 flex-1 rounded-md px-3 py-2.5 text-center transition-colors duration-200 ${reasoningLevel === level.value
                      ? "text-neutral-900"
                      : "text-neutral-500"
                    }`}
                  aria-label={`Select ${level.label}`}
                >
                  <div className="text-xs font-medium leading-tight">{level.label}</div>
                  <div className="mt-0.5 text-[10px] leading-tight opacity-60">{level.sublabel}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex justify-end border-t border-border bg-muted/30 p-6">
        <Button onClick={handleSubmit} className="px-6">
          Save Configuration
        </Button>
      </div>
    </div>
  );
}
