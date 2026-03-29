import test from "node:test";
import assert from "node:assert/strict";

import { resolveVLLMModel, streamVLLM } from "./vllm-chat.ts";

function jsonResponse(body, init = {}) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

function sseResponse(chunks) {
  return new Response(
    chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join(""),
    {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    },
  );
}

test("resolveVLLMModel returns the preferred model without querying /v1/models", async () => {
  let fetchCalls = 0;

  const resolved = await resolveVLLMModel("https://demo.modal.run", {
    preferredModel: "job-123",
    fetchImpl: async () => {
      fetchCalls += 1;
      throw new Error("fetch should not have been called");
    },
  });

  assert.equal(resolved, "job-123");
  assert.equal(fetchCalls, 0);
});

test("streamVLLM resolves the served model from /v1/models before calling chat completions", async () => {
  const seenUrls = [];
  const seenBodies = [];
  const streamedEvents = [];

  const fetchImpl = async (input, init) => {
    const url = String(input);
    seenUrls.push(url);

    if (url.endsWith("/v1/models")) {
      return jsonResponse({
        data: [
          { id: "job-123" },
          { id: "job-123" },
        ],
      });
    }

    if (url.endsWith("/v1/chat/completions")) {
      seenBodies.push(JSON.parse(String(init?.body ?? "{}")));
      return sseResponse([
        { choices: [{ delta: { content: "Hair loss " } }] },
        { choices: [{ delta: { content: "can have many causes." } }] },
      ]);
    }

    throw new Error(`Unexpected URL: ${url}`);
  };

  await streamVLLM({
    endpointUrl: "https://demo.modal.run/",
    messages: [{ role: "user", content: "What is hair loss caused by?" }],
    send: (event) => {
      streamedEvents.push(event);
    },
    fetchImpl,
  });

  assert.deepEqual(seenUrls, [
    "https://demo.modal.run/v1/models",
    "https://demo.modal.run/v1/chat/completions",
  ]);
  assert.deepEqual(seenBodies, [
    {
      model: "job-123",
      messages: [{ role: "user", content: "What is hair loss caused by?" }],
      stream: true,
    },
  ]);
  assert.deepEqual(streamedEvents, [
    { type: "text_delta", content: "Hair loss " },
    { type: "text_delta", content: "can have many causes." },
  ]);
});

test("resolveVLLMModel fails clearly when /v1/models returns no ids", async () => {
  await assert.rejects(
    () =>
      resolveVLLMModel("https://demo.modal.run", {
        fetchImpl: async () => jsonResponse({ data: [] }),
      }),
    /did not report any models/,
  );
});
