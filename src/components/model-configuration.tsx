"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
    <div className="w-full max-w-5xl rounded-xl border border-neutral-200 bg-white shadow-sm overflow-hidden">
      <div className="border-b border-neutral-200 p-6">
        <h1 className="text-xl font-semibold text-neutral-900">
          Model Configuration
        </h1>
        <p className="mt-1 text-sm text-neutral-500">
          Define your model parameters and reasoning preferences
        </p>
      </div>

      <div className="flex">
        {/* Left Half - Input Fields */}
        <div className="w-1/2 border-r border-neutral-200 p-8">
          <div className="space-y-6">
            <div className="space-y-2">
              <Label
                htmlFor="domain"
                className="text-sm font-medium text-neutral-700"
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
                className="text-sm font-medium text-neutral-700"
              >
                What should the model do?
              </Label>
              <textarea
                id="task"
                placeholder="Describe the task or capability you want the model to perform..."
                value={task}
                onChange={(e) => setTask(e.target.value)}
                className="w-full min-h-[120px] resize-none rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm placeholder:text-neutral-400 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-neutral-900"
              />
            </div>
          </div>
        </div>

        {/* Right Half - Vertical Slider */}
        <div className="w-1/2 p-8">
          <div className="space-y-4">
            <Label className="text-sm font-medium text-neutral-700">
              Reasoning vs Speed
            </Label>

            <div className="flex gap-6">
              {/* Labels column */}
              <div className="flex flex-col justify-between py-2 text-right">
                {levels.map((level) => (
                  <div
                    key={level.value}
                    className={`cursor-pointer transition-colors ${
                      reasoningLevel === level.value
                        ? "text-neutral-900"
                        : "text-neutral-400 hover:text-neutral-600"
                    }`}
                    onClick={() => setReasoningLevel(level.value)}
                  >
                    <div className="text-sm font-medium">{level.label}</div>
                    <div className="text-xs">{level.sublabel}</div>
                  </div>
                ))}
              </div>

              {/* Slider track */}
              <div className="relative flex flex-col items-center">
                <div className="relative h-64 w-2 rounded-full bg-neutral-200">
                  {/* Active track */}
                  <div
                    className="absolute w-full rounded-full bg-neutral-900 transition-all duration-200"
                    style={{
                      top: `${((reasoningLevel - 1) / 4) * 100}%`,
                      bottom: 0,
                    }}
                  />

                  {/* Snap points */}
                  {levels.map((level, index) => (
                    <button
                      key={level.value}
                      onClick={() => setReasoningLevel(level.value)}
                      className={`absolute -left-1 h-4 w-4 rounded-full border-2 transition-all duration-200 ${
                        reasoningLevel === level.value
                          ? "scale-125 border-neutral-900 bg-neutral-900"
                          : "border-neutral-400 bg-white hover:border-neutral-600"
                      }`}
                      style={{
                        top: `${(index / 4) * 100}%`,
                        transform: "translateY(-50%)",
                      }}
                      aria-label={`Select ${level.label}`}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex justify-end border-t border-neutral-200 bg-neutral-50 p-6">
        <Button onClick={handleSubmit} className="px-6">
          Save Configuration
        </Button>
      </div>
    </div>
  );
}
