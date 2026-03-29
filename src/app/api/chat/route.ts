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

type RequestBody = {
  messages: ChatMessage[];
  customEndpoint?: string;
};

export async function POST(request: Request) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return new Response(JSON.stringify({ error: "OPENAI_API_KEY is not configured." }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  let body: RequestBody;
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

  // If a custom vLLM endpoint is provided, proxy to it instead of OpenAI
  if (body.customEndpoint) {
    if (
      !body.customEndpoint.startsWith("https://") ||
      !body.customEndpoint.includes("modal.run")
    ) {
      return new Response(
        JSON.stringify({ error: "Invalid custom endpoint. Must be a modal.run HTTPS URL." }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    const endpoint = body.customEndpoint;
    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        function send(event: Record<string, unknown>) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
        }
        try {
          await streamVLLM(
            endpoint,
            body.messages.map((m) => ({ role: m.role, content: m.content })),
            send,
          );
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

        // First call (streamed) — may produce text or a tool call
        const firstResult = await streamOpenAI(apiKey, model, { input: inputMessages }, send);

        if (firstResult.toolCall && firstResult.toolCall.name === "train_model") {
          const args = JSON.parse(firstResult.toolCall.arguments);
          const input = validateCreateJobInput({ description: args.description });
          const job = await createPostTrainingJob(input);
          spawnPostTrainingOrchestrator(job.jobId);

          send({ type: "job_started", jobId: job.jobId });

          // Chain via previous_response_id and send only the tool output
          await streamOpenAI(
            apiKey,
            model,
            {
              input: [
                {
                  type: "function_call_output",
                  call_id: firstResult.toolCall.id,
                  output: JSON.stringify({
                    jobId: job.jobId,
                    status: "queued",
                    message: "Post-training job created and queued.",
                  }),
                },
              ],
              previousResponseId: firstResult.responseId,
            },
            send,
          );
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

type ToolCallResult = {
  id: string;
  name: string;
  arguments: string;
};

type StreamResult = {
  responseId: string | null;
  toolCall: ToolCallResult | null;
};

type StreamOpenAIOptions = {
  input: Array<Record<string, unknown>>;
  previousResponseId?: string | null;
};

/**
 * Streams an OpenAI Responses API call. Text deltas are forwarded to the
 * client via `send()` as they arrive. Tool calls are buffered and returned
 * so the caller can execute them. Uses `previous_response_id` for chaining.
 */
async function streamOpenAI(
  apiKey: string,
  model: string,
  options: StreamOpenAIOptions,
  send: (event: Record<string, unknown>) => void,
): Promise<StreamResult> {
  const requestBody: Record<string, unknown> = {
    model,
    input: options.input,
    tools: TOOLS,
    stream: true,
  };

  if (options.previousResponseId) {
    requestBody.previous_response_id = options.previousResponseId;
  }

  const response = await fetch(OPENAI_RESPONSES_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`OpenAI API error ${response.status}: ${text.slice(0, 500)}`);
  }

  if (!response.body) {
    throw new Error("OpenAI returned no response body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // Tool call state — capture name from output_item.added, args from .done
  let toolCallId = "";
  let toolCallName = "";
  let toolCallArgs = "";
  let hasToolCall = false;

  // Response ID from the completed event for chaining
  let responseId: string | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;
      const data = trimmed.slice(6);
      if (!data || data === "[DONE]") continue;

      try {
        const event = JSON.parse(data);
        const eventType: string = event.type ?? "";

        if (eventType === "response.output_text.delta") {
          const delta: string = event.delta ?? "";
          if (delta) {
            send({ type: "text_delta", content: delta });
          }
        } else if (eventType === "response.output_item.added") {
          const item = event.item;
          if (item?.type === "function_call") {
            hasToolCall = true;
            toolCallId = item.call_id ?? item.id ?? toolCallId;
            toolCallName = item.name ?? toolCallName;
          }
        } else if (eventType === "response.function_call_arguments.done") {
          // Use the authoritative done event for the complete arguments
          toolCallArgs = event.arguments ?? toolCallArgs;
        } else if (eventType === "response.function_call_arguments.delta") {
          // Fallback: accumulate deltas in case .done doesn't fire
          if (!toolCallArgs) {
            toolCallArgs += event.delta ?? "";
          }
        } else if (eventType === "response.completed") {
          responseId = event.response?.id ?? null;
        }
      } catch {
        // Skip malformed SSE lines
      }
    }
  }

  const toolCall: ToolCallResult | null = hasToolCall
    ? { id: toolCallId, name: toolCallName, arguments: toolCallArgs }
    : null;

  return { responseId, toolCall };
}

/**
 * Streams a vLLM Chat Completions API call. The Modal-deployed vLLM endpoint
 * is OpenAI-compatible, so we use the standard chat completions streaming format.
 */
async function streamVLLM(
  endpointUrl: string,
  messages: Array<{ role: string; content: string }>,
  send: (event: Record<string, unknown>) => void,
): Promise<void> {
  const url = `${endpointUrl.replace(/\/+$/, "")}/v1/chat/completions`;

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "default",
      messages,
      stream: true,
    }),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`vLLM API error ${response.status}: ${text.slice(0, 500)}`);
  }

  if (!response.body) {
    throw new Error("vLLM returned no response body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;
      const data = trimmed.slice(6);
      if (!data || data === "[DONE]") continue;

      try {
        const chunk = JSON.parse(data);
        const delta = chunk.choices?.[0]?.delta?.content;
        if (delta) {
          send({ type: "text_delta", content: delta });
        }
      } catch {
        // Skip malformed SSE lines
      }
    }
  }
}
