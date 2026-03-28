import { NextResponse } from "next/server";

import { getPostTrainingJob } from "@/lib/posttraining-server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{
    jobId: string;
  }>;
};

export async function GET(_: Request, context: RouteContext) {
  try {
    const { jobId } = await context.params;
    const job = await getPostTrainingJob(jobId);
    if (!job) {
      return NextResponse.json({ error: "Job not found." }, { status: 404 });
    }
    return NextResponse.json(job);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to load post-training job.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
