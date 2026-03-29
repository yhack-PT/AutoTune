import {
  createPostTrainingJob,
  spawnPostTrainingOrchestrator,
  validateCreateJobInput,
} from "@/lib/posttraining-server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses";
const DEFAULT_MODEL = "gpt-5.4";

const SYSTEM_PROMPT = `You are an assistant that helps users fine-tune language models. You can have a normal conversation, answer questions about fine-tuning, and when the user wants to train a model, use the train_model tool.

When the user describes a model they want to fine-tune, call the train_model tool with a detailed description that captures their domain, use case, desired model behavior, and any other relevant details they mentioned.

After starting a training job, let the user know it has started and they can watch its progress.`;

const TOOLS = [
  {
    type: "function" as const,
    name: "train_model",
    description:
      "Start a post-training job to fine-tune a language model. Call this when the user wants to train or fine-tune a model.",
    parameters: {
      type: "object",
      properties: {
        description: {
          type: "string",
          description:
            "A detailed description covering the domain, use case, desired model behavior, and any other relevant details about the model the user wants to fine-tune.",
        },
      },
      required: ["description"],
      additionalProperties: false,
    },
  },
];

type ChatMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};

export async function POST(request: Request) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return new Response(JSON.stringify({ error: "OPENAI_API_KEY is not configured." }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  let body: { messages: ChatMessage[] };
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body." }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return new Response(JSON.stringify({ error: "messages must be a non-empty array." }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const model = process.env.CHAT_OPENAI_MODEL || DEFAULT_MODEL;

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      function send(event: Record<string, unknown>) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      }

      try {
        const inputMessages = [
          { role: "system", content: SYSTEM_PROMPT },
          ...body.messages.map((m) => ({ role: m.role, content: m.content })),
        ];

        // First call — may produce text or a tool call
        const firstResponse = await callOpenAI(apiKey, model, inputMessages);
        const toolCall = extractToolCall(firstResponse);

        if (toolCall && toolCall.name === "train_model") {
          // Execute the tool
          const args = JSON.parse(toolCall.arguments);
          const input = validateCreateJobInput({ description: args.description });
          const job = await createPostTrainingJob(input);
          spawnPostTrainingOrchestrator(job.jobId);

          send({ type: "job_started", jobId: job.jobId });

          // Feed tool result back to OpenAI using Responses API format
          const toolResultMessages = [
            ...inputMessages,
            ...extractAssistantOutput(firstResponse),
            {
              type: "function_call_output",
              call_id: toolCall.id,
              output: JSON.stringify({
                jobId: job.jobId,
                status: "queued",
                message: "Post-training job created and queued.",
              }),
            },
          ];

          const secondResponse = await callOpenAI(apiKey, model, toolResultMessages);
          const text = extractTextContent(secondResponse);
          if (text) {
            send({ type: "text_delta", content: text });
          }
        } else {
          // Plain text response
          const text = extractTextContent(firstResponse);
          if (text) {
            send({ type: "text_delta", content: text });
          }
        }

        send({ type: "done" });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        send({ type: "error", message });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

async function callOpenAI(
  apiKey: string,
  model: string,
  messages: Array<Record<string, unknown>>,
) {
  const response = await fetch(OPENAI_RESPONSES_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      input: messages,
      tools: TOOLS,
    }),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`OpenAI API error ${response.status}: ${text.slice(0, 500)}`);
  }

  return response.json();
}

function extractToolCall(
  response: Record<string, unknown>,
): { id: string; name: string; arguments: string } | null {
  const output = Array.isArray(response.output) ? response.output : [];
  for (const item of output) {
    if (item?.type === "function_call") {
      return {
        id: item.call_id ?? item.id ?? "",
        name: item.name ?? "",
        arguments: item.arguments ?? "{}",
      };
    }
  }
  return null;
}

function extractAssistantOutput(response: Record<string, unknown>): Array<Record<string, unknown>> {
  // Return the output items as-is for feeding back into the conversation
  const output = Array.isArray(response.output) ? response.output : [];
  return output;
}

function extractTextContent(response: Record<string, unknown>): string | null {
  // Try top-level output_text first
  if (typeof response.output_text === "string" && response.output_text.trim()) {
    return response.output_text.trim();
  }

  const output = Array.isArray(response.output) ? response.output : [];
  for (const item of output) {
    const content = Array.isArray(item?.content) ? item.content : [];
    for (const part of content) {
      if (part?.type === "output_text" && typeof part.text === "string" && part.text.trim()) {
        return part.text.trim();
      }
    }
  }
  return null;
}
