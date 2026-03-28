import test from "node:test";
import assert from "node:assert/strict";

import { extractModalRunUrl } from "./posttraining-orchestrator.mjs";

test("extractModalRunUrl returns a contiguous modal.run URL", () => {
  const output = `
✓ Created objects.
└── 🔨 Created web function serve =>
https://andrew-qian64--vllm-lora-customer-support-classify-support-ticket.modal.run
✓ App deployed in 2.914s! 🎉
`;

  assert.equal(
    extractModalRunUrl(output),
    "https://andrew-qian64--vllm-lora-customer-support-classify-support-ticket.modal.run",
  );
});

test("extractModalRunUrl reconstructs a wrapped modal.run URL from deploy output", () => {
  const output = `
✓ Created objects.
└── 🔨 Created web function serve =>
https://andrew-qian64--vllm-lora-customer-support-classify-suppo-6a9926.moda
l.run (label truncated)
✓ App deployed in 2.914s! 🎉
View Deployment:
https://modal.com/apps/andrew-qian64/main/deployed/vllm-lora-customer-support-cl
assify-support-ticket-dc6d7998
`;

  assert.equal(
    extractModalRunUrl(output),
    "https://andrew-qian64--vllm-lora-customer-support-classify-suppo-6a9926.modal.run",
  );
});
