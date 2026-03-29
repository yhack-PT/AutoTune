# Backend Flow

This repo's backend is a file-backed post-training pipeline. A user prompt eventually becomes a deployed Modal-hosted model by moving through a small set of API entrypoints, a detached orchestrator process, and a per-job artifact directory under `backend/generated-posttraining-jobs/`.

## At a Glance

```text
User prompt
  -> POST /api/chat
  -> OpenAI Responses API decides to call train_model
  -> createPostTrainingJob()
  -> backend/generated-posttraining-jobs/<jobId>/
  -> spawn detached posttraining-orchestrator.mjs
  -> recommending
  -> compiling
  -> training
  -> evaluating
  -> deploying
  -> smoke_testing
  -> ready
  -> deployment.json contains the live modal.run URL
```

There is also a direct non-chat entrypoint:

```text
POST /api/posttraining/jobs
  -> createPostTrainingJob()
  -> spawn detached posttraining-orchestrator.mjs
  -> same pipeline as above
```

## 1. Entry Points

### A. Conversational entry: `src/app/api/chat/route.ts`

This is the main "user prompt" path.

1. The frontend sends chat history to `POST /api/chat`.
2. The route calls the OpenAI Responses API with:
   - a short system prompt that tells the assistant to start training immediately
   - a single tool: `train_model`
3. If OpenAI emits a `train_model` tool call, the route:
   - validates the `description`
   - creates a job on disk
   - spawns the detached orchestrator
   - streams a `job_started` event back to the client
4. The same route then continues the OpenAI response so the user gets a short confirmation message.

Important detail: `POST /api/chat` also has a second mode. Once a model is deployed, the frontend can pass `customEndpoint`, and the route will proxy chat traffic directly to the deployed vLLM endpoint instead of OpenAI.

### B. Direct job entry: `src/app/api/posttraining/jobs/route.ts`

This route skips the chat/tool-call step. It:

1. accepts `{ description, seedArtifact? }`
2. validates input
3. creates the same job record on disk
4. spawns the same detached orchestrator
5. returns `202 Accepted` with the `jobId`

## 2. Job Creation And Persistence

Job state is managed in `src/lib/posttraining-server.ts`.

When `createPostTrainingJob()` runs, it:

1. creates `backend/generated-posttraining-jobs/<jobId>/`
2. writes `job.json` as the canonical state record
3. writes `events.jsonl` with the first `queued` event
4. precomputes artifact paths that later stages will fill in

`spawnPostTrainingOrchestrator()` then launches:

```bash
node backend/posttraining-orchestrator.mjs --job-id <jobId>
```

The orchestrator is detached, so the request can return immediately while the long-running work continues in the background.

## 3. The Orchestrator Stages

`backend/posttraining-orchestrator.mjs` is the backbone of the backend. It updates `job.json`, appends structured log lines to `events.jsonl`, and runs the pipeline stages in order.

### Stage 1: `recommending`

Code path:

- `runRecommendationStage()`
- `backend/hf-dataset-recommender.mjs`

What happens:

1. The user's natural-language description is analyzed.
2. OpenAI generates a Hugging Face search plan and a `task_spec`.
3. The recommender searches and enriches candidate HF datasets.
4. Each candidate is checked for schema compatibility and normalization feasibility.
5. OpenAI ranks the compatible candidates and selects the final dataset set.

Main output:

- `recommendation.json`

Important scope note:

- The recommender currently supports only single-target classification SFT and generation SFT.
- Even though the Python trainer has broader TRL support, the live backend path here compiles only SFT jobs.

### Stage 2: `compiling`

Code path:

- `runCompilerStage()`
- `backend/posttraining-spec-compiler.mjs`
- `backend/posttraining-normalization.mjs`

What happens:

1. The compiler reads `recommendation.json`.
2. It filters to usable dataset candidates.
3. It generates a validated post-training spec.
4. It converts that spec into a concrete trainer config plus a prepared dataset manifest.
5. It writes trace artifacts so the decision path is inspectable.

Main outputs:

- `post_training_job_spec.yaml`
- `compiled_train_config.yaml`
- `prepared_dataset_manifest.json`
- `compiler_trace.json`

This is the point where the backend turns "what the user wants" into "the exact training job we will run."

### Stage 3: `training`

Code path:

- `runTrainingAndEvaluationStage()`
- `backend/modal_trl_posttrain.py`

What happens:

1. The orchestrator launches Modal with the compiled config:

   ```bash
   modal run backend/modal_trl_posttrain.py --config <compiled_train_config.yaml>
   ```

2. The trainer runs in `train_then_evaluate` mode.
3. The Python process emits:
   - structured metric events
   - a structured `training_complete` lifecycle event
4. The orchestrator parses those lines in real time and persists them.

Main outputs:

- `training_result.json`
- `training_metrics.jsonl`
- `training_loss.svg`
- `learning_rate.svg` when available

Training details worth knowing:

- The compiled config uses a prepared manifest rather than raw direct HF loading at runtime.
- The trainer is LoRA/PEFT-first.
- The trainer may produce both `final_adapter_dir` and a merged model directory.

### Stage 4: `evaluating`

Code path:

- tail end of `backend/modal_trl_posttrain.py`
- `finalizeEvaluationStage()`

What happens:

1. Offline evaluation runs immediately after training.
2. The exact artifact depends on task type:
   - classification can write `evaluation_result.json`
   - generation can write `comparison_evaluation.json`
3. The orchestrator reads whichever evaluation artifact exists.
4. It rejects obviously bad runs, for example:
   - degenerate low-step runs for non-trivial jobs
   - catastrophic invalid-label rates

Main outputs:

- `evaluation_result.json` when present
- `comparison_evaluation.json` when present

The job record's `evaluation` field is populated from these artifacts so the UI can show summary results during polling.

### Stage 5: `deploying`

Code path:

- `runDeploymentStage()`
- `backend/modal_vllm_serve.py`

What happens:

1. The orchestrator chooses the deployable artifact:
   - prefer `merged_dir` when training produced one
   - otherwise use `final_adapter_dir`
2. It builds the serving environment variables:
   - base model
   - adapter path
   - adapter name
   - GPU type
   - max model length
   - serving package versions
3. It deploys the vLLM server to Modal:

   ```bash
   modal deploy backend/modal_vllm_serve.py
   ```

4. It parses the `modal.run` URL from deploy output.
5. It waits for `/v1/models` on the live endpoint to report the expected model.

Main output:

- `deployment.json`

`deployment.json` is the moment the backend has a real serving endpoint, but the job is not marked ready until smoke testing passes.

### Stage 6: `smoke_testing`

Code path:

- `runSmokeTestStage()`
- `runSmokeTest()`

What happens:

1. The backend probes `/v1/models`.
2. It sends a test request to `/v1/chat/completions`.
3. For generation jobs, it uses deterministic sample probes and checks for non-trivial completions.
4. It retries until success or timeout.

Main output:

- `smoke_test.json`

If smoke testing passes, `markReady()` moves the job into `ready`.

## 4. What Lives In A Job Directory

Each job directory is both the queue item and the audit trail.

| File | Produced by | Purpose |
| --- | --- | --- |
| `job.json` | API + orchestrator | Canonical job state, stage history, selected datasets, deployment info |
| `events.jsonl` | API + orchestrator + stage loggers | Append-only log stream, including UI progress messages |
| `recommendation.json` | recommender | Dataset search plan, task spec, ranked candidates, final picks |
| `post_training_job_spec.yaml` | compiler | High-level validated post-training spec |
| `compiled_train_config.yaml` | compiler | Concrete config consumed by the Modal trainer |
| `prepared_dataset_manifest.json` | compiler | Normalized dataset recipe for runtime loading |
| `compiler_trace.json` | compiler | Why the compiler chose this plan |
| `training_result.json` | trainer/orchestrator | Output artifact paths and training summary |
| `training_metrics.jsonl` | orchestrator | Structured training metrics captured live |
| `training_loss.svg` | orchestrator | Graph generated from captured metric lines |
| `learning_rate.svg` | orchestrator | Optional graph generated from captured metric lines |
| `evaluation_result.json` | trainer | Offline eval artifact for classification-style jobs |
| `comparison_evaluation.json` | trainer | Base-vs-tuned comparison artifact for generation jobs |
| `deployment.json` | orchestrator | Final Modal deployment metadata and URL |
| `smoke_test.json` | orchestrator | Proof that the live endpoint answered basic requests |

## 5. How The UI Watches Progress

The frontend does not stream stage state directly from the orchestrator. Instead:

1. `/api/chat` returns the `job_started` event.
2. The client begins polling `GET /api/posttraining/jobs/[jobId]`.
3. `getPostTrainingJob()` reads:
   - `job.json`
   - the last chunk of `events.jsonl`
4. `src/lib/posttraining-progress.mjs` extracts the UI-specific progress lines from the logs.

That means the source of truth for progress is still the job directory on disk. The UI is just reconstructing it by polling.

## 6. Ready State And Serving Path

A job is only effectively "done" when all of these are true:

1. deployment succeeded
2. the live Modal endpoint reports the expected model under `/v1/models`
3. smoke tests passed
4. `job.json` is marked `ready`

Once that happens:

- `job.deployment.url` contains the live model URL
- the frontend stores it as the fine-tuned endpoint
- future `POST /api/chat` calls can send `customEndpoint` and chat with the deployed model through the same backend route

## 7. Mental Model

The easiest way to think about this backend is:

1. `chat route` decides whether the user wants a model built
2. `job store` turns that into a durable work item on disk
3. `orchestrator` turns the work item into staged artifacts
4. `Modal trainer` creates the model artifacts
5. `Modal vLLM deploy` turns those artifacts into a live endpoint
6. `smoke test` is the final gate before the job is considered ready

So the real handoff chain is:

```text
prompt -> OpenAI tool call -> local job record -> detached orchestrator
-> recommendation/compilation artifacts -> Modal training artifacts
-> Modal deployment URL -> smoke-tested live model
```
