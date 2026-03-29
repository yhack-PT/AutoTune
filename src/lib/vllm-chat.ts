export type VLLMChatMessage = {
  role: string;
  content: string;
};

type FetchLike = typeof fetch;

function normalizeEndpointUrl(endpointUrl: string): string {
  return endpointUrl.replace(/\/+$/, "");
}

async function ensureOk(response: Response): Promise<void> {
  if (response.ok) {
    return;
  }

  const text = await response.text().catch(() => "");
  throw new Error(`vLLM API error ${response.status}: ${text.slice(0, 500)}`);
}

export async function resolveVLLMModel(
  endpointUrl: string,
  {
    preferredModel,
    fetchImpl = fetch,
  }: {
    preferredModel?: string | null;
    fetchImpl?: FetchLike;
  } = {},
): Promise<string> {
  const normalizedPreferredModel = String(preferredModel ?? "").trim();
  if (normalizedPreferredModel) {
    return normalizedPreferredModel;
  }

  const response = await fetchImpl(`${normalizeEndpointUrl(endpointUrl)}/v1/models`);
  await ensureOk(response);

  const payload = await response.json().catch(() => ({}));
  const modelIds: string[] = Array.isArray(payload?.data)
    ? payload.data
      .map((entry: unknown) =>
        typeof entry === "object" && entry !== null && "id" in entry
          ? String((entry as { id?: unknown }).id ?? "").trim()
          : "",
      )
      .filter((entry: string): entry is string => Boolean(entry))
    : [];

  const resolvedModel = [...new Set(modelIds)][0] ?? "";
  if (!resolvedModel) {
    throw new Error("vLLM endpoint did not report any models from /v1/models.");
  }

  return resolvedModel;
}

export async function streamVLLM({
  endpointUrl,
  model,
  messages,
  send,
  fetchImpl = fetch,
}: {
  endpointUrl: string;
  model?: string | null;
  messages: VLLMChatMessage[];
  send: (event: Record<string, unknown>) => void;
  fetchImpl?: FetchLike;
}): Promise<void> {
  const normalizedEndpointUrl = normalizeEndpointUrl(endpointUrl);
  const resolvedModel = await resolveVLLMModel(normalizedEndpointUrl, {
    preferredModel: model,
    fetchImpl,
  });

  const response = await fetchImpl(`${normalizedEndpointUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: resolvedModel,
      messages,
      stream: true,
    }),
  });

  await ensureOk(response);

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
        // Skip malformed SSE lines.
      }
    }
  }
}
