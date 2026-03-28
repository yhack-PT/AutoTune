import { NextResponse } from "next/server";

import {
  createPostTrainingJob,
  listPostTrainingJobs,
  spawnPostTrainingOrchestrator,
  validateCreateJobInput,
} from "@/lib/posttraining-server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const jobs = await listPostTrainingJobs();
    return NextResponse.json({ jobs });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to list post-training jobs.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const input = validateCreateJobInput(body);
    const job = await createPostTrainingJob(input);
    spawnPostTrainingOrchestrator(job.jobId);

    return NextResponse.json(
      {
        jobId: job.jobId,
        status: job.status,
      },
      { status: 202 },
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to create post-training job.";
    const status = /must be|Request body/i.test(message) ? 400 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
