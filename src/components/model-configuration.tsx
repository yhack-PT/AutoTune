"use client";

import { useEffect, useState } from "react";
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

const networkNodes = [
  { id: "input-1", x: 16, y: 18 },
  { id: "input-2", x: 16, y: 44 },
  { id: "input-3", x: 16, y: 70 },
  { id: "hidden-1", x: 54, y: 12 },
  { id: "hidden-2", x: 54, y: 31 },
  { id: "hidden-3", x: 54, y: 57 },
  { id: "hidden-4", x: 54, y: 76 },
  { id: "output-1", x: 92, y: 26 },
  { id: "output-2", x: 92, y: 62 },
  { id: "sink", x: 118, y: 44 },
] as const;

const networkEdges = [
  { id: "e-1", from: "input-1", to: "hidden-1" },
  { id: "e-2", from: "input-1", to: "hidden-2" },
  { id: "e-3", from: "input-2", to: "hidden-2" },
  { id: "e-4", from: "input-2", to: "hidden-3" },
  { id: "e-5", from: "input-3", to: "hidden-3" },
  { id: "e-6", from: "input-3", to: "hidden-4" },
  { id: "e-7", from: "hidden-1", to: "output-1" },
  { id: "e-8", from: "hidden-2", to: "output-1" },
  { id: "e-9", from: "hidden-2", to: "output-2" },
  { id: "e-10", from: "hidden-3", to: "output-1" },
  { id: "e-11", from: "hidden-3", to: "output-2" },
  { id: "e-12", from: "hidden-4", to: "output-2" },
  { id: "e-13", from: "output-1", to: "sink" },
  { id: "e-14", from: "output-2", to: "sink" },
] as const;

type NetworkNode = (typeof networkNodes)[number];
type NetworkNodeId = NetworkNode["id"];
type NetworkEdgeId = (typeof networkEdges)[number]["id"];
type NetworkPath = {
  nodes: readonly NetworkNodeId[];
  edges: readonly NetworkEdgeId[];
};

const networkPaths: readonly NetworkPath[] = [
  {
    nodes: ["input-1", "hidden-1", "output-1", "sink"],
    edges: ["e-1", "e-7", "e-13"],
  },
  {
    nodes: ["input-2", "hidden-2", "output-1", "sink"],
    edges: ["e-3", "e-8", "e-13"],
  },
  {
    nodes: ["input-2", "hidden-3", "output-2", "sink"],
    edges: ["e-4", "e-11", "e-14"],
  },
  {
    nodes: ["input-3", "hidden-4", "output-2", "sink"],
    edges: ["e-6", "e-12", "e-14"],
  },
] as const;

function NeuralNetworkPreview() {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return;
    }

    const intervalId = window.setInterval(() => {
      setPhase((current) => (current + 1) % networkPaths.length);
    }, 1400);

    return () => window.clearInterval(intervalId);
  }, []);

  const activePath = networkPaths[phase];
  const nodeById = networkNodes.reduce<Record<NetworkNodeId, NetworkNode>>(
    (nodes, node) => {
      nodes[node.id] = node;
      return nodes;
    },
    {} as Record<NetworkNodeId, NetworkNode>,
  );

  return (
    <div
      aria-hidden="true"
      className="shrink-0 self-start rounded-lg border border-border/80 bg-muted/40 p-3 text-neutral-700"
    >
      <svg
        viewBox="0 0 132 88"
        className="h-[84px] w-[126px]"
        role="presentation"
      >
        {networkEdges.map((edge) => {
          const start = nodeById[edge.from];
          const end = nodeById[edge.to];
          const isActive = activePath.edges.includes(edge.id);

          return (
            <line
              key={edge.id}
              x1={start.x}
              y1={start.y}
              x2={end.x}
              y2={end.y}
              stroke="currentColor"
              strokeLinecap="round"
              strokeOpacity={isActive ? 0.72 : 0.16}
              strokeWidth={isActive ? 1.8 : 1.15}
              style={{
                transition: "stroke-opacity 450ms ease, stroke-width 450ms ease",
              }}
            />
          );
        })}

        {networkNodes.map((node) => {
          const isActive = activePath.nodes.includes(node.id);

          return (
            <g key={node.id} transform={`translate(${node.x} ${node.y})`}>
              <circle
                r={isActive ? 6 : 4}
                fill="currentColor"
                fillOpacity={isActive ? 0.1 : 0.04}
                style={{ transition: "fill-opacity 450ms ease, r 450ms ease" }}
              />
              <circle
                r={isActive ? 2.8 : 2}
                fill="currentColor"
                fillOpacity={isActive ? 0.9 : 0.38}
                style={{ transition: "fill-opacity 450ms ease, r 450ms ease" }}
              />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

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
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h1 className="text-xl font-semibold text-foreground">
              Model Configuration
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Define your model parameters and reasoning preferences
            </p>
          </div>

          <NeuralNetworkPreview />
        </div>
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
