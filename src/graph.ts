// The master pipeline: wires every previously-built, individually-verified
// piece into one StateGraph. Ingestion (acquire/chunk/embed) happens once in
// plain code before the graph runs -- it's a one-time setup step, not
// something that needs to loop or branch per query. The graph itself is the
// per-query decision path: agent -> grounding check -> human review -> merge,
// with the review and grounding steps each able to hard-stop the run.

import "dotenv/config";
import {
  StateGraph,
  StateSchema,
  MessagesValue,
  START,
  END,
  MemorySaver,
  Command,
  interrupt,
  type GraphNode,
  type ConditionalEdgeRouter,
} from "@langchain/langgraph";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { z } from "zod";
import { writeFile, mkdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";

import { acquirePds } from "./acquirePds.js";
import { extractAndChunkPds } from "./chunkPds.js";
import { buildIndex, type IndexedChunk } from "./embedIndex.js";
import { buildAgentGraph } from "./agent.js";
import { checkGrounding } from "./groundingCheck.js";
import { extractMergeValues } from "./extractMergeValues.js";
import { mergeLetter, type LetterData } from "./mergeLetter.js";

const State = new StateSchema({
  messages: MessagesValue,
  letterData: z.custom<LetterData | null>(() => true).default(() => null),
  groundedOk: z.boolean().optional(),
  reviewDecision: z.enum(["approve", "reject"]).optional(),
});

// .invoke()'s return type only reflects the fields declared in `State`.
// When a node calls interrupt(), the runtime adds an extra `__interrupt__`
// field that TypeScript can't know about statically -- this documents that
// shape so we can safely check for it.
type InvokeResultWithInterrupt = typeof State.State & {
  __interrupt__?: { id: string; value: unknown }[];
};

export function buildMasterGraph(index: IndexedChunk[]) {
  const runAgent: GraphNode<typeof State> = async (state) => {
    const agentGraph = buildAgentGraph(index);
    const result = await agentGraph.invoke({ messages: state.messages });
    // Return only the NEW messages -- the outer `messages` field appends
    // whatever we return, so returning the whole inner history again would
    // duplicate everything already in state.
    return { messages: result.messages.slice(state.messages.length) };
  };

  const groundingCheckNode: GraphNode<typeof State> = async (state) => {
    const result = checkGrounding(state.messages);
    if (!result.grounded) {
      console.warn("Grounding check failed. Ungrounded values:", result.ungroundedValues);
    }
    return { groundedOk: result.grounded };
  };

  const afterGroundingCheck: ConditionalEdgeRouter<typeof State, Record<string, any>, "humanReview"> = (state) => {
    return state.groundedOk ? "humanReview" : END;
  };

  const humanReviewNode: GraphNode<typeof State> = async (state) => {
    const letterData = extractMergeValues(state.messages);
    // Genuinely pauses graph execution here. With a checkpointer attached,
    // the first invoke() call returns normally (with __interrupt__ set)
    // instead of throwing -- execution resumes exactly at this line when
    // the graph is invoked again with Command({ resume }).
    const decision = interrupt({
      question: "Approve these values for the member letter?",
      letterData,
    });
    return { letterData, reviewDecision: decision === "approve" ? "approve" : "reject" };
  };

  const afterHumanReview: ConditionalEdgeRouter<typeof State, Record<string, any>, "merge"> = (state) => {
    return state.reviewDecision === "approve" ? "merge" : END;
  };

  const mergeNode: GraphNode<typeof State> = async (state) => {
    const buffer = await mergeLetter("templates/Cover_Letter_Template.docx", state.letterData!);
    await mkdir("output", { recursive: true });
    const outPath = "output/final_member_letter.docx";
    await writeFile(outPath, buffer);
    console.log(`Wrote ${outPath}`);
    return {};
  };

  return new StateGraph(State)
    .addNode("agent", runAgent)
    .addNode("groundingCheck", groundingCheckNode)
    .addNode("humanReview", humanReviewNode)
    .addNode("merge", mergeNode)
    .addEdge(START, "agent")
    .addEdge("agent", "groundingCheck")
    .addConditionalEdges("groundingCheck", afterGroundingCheck, ["humanReview", END])
    .addConditionalEdges("humanReview", afterHumanReview, ["merge", END])
    .addEdge("merge", END)
    .compile({ checkpointer: new MemorySaver() });
}

async function main() {
  console.log("--- ACQUIRE / CHUNK / EMBED (live, at runtime) ---");
  const pdsBuffer = await acquirePds();
  const chunks = await extractAndChunkPds(pdsBuffer);
  const index = await buildIndex(chunks);

  const graph = buildMasterGraph(index);
  // The checkpointer needs a thread id to know which paused run to resume --
  // a fixed id is fine here since this script only ever runs one thread.
  const config = { configurable: { thread_id: "sandpit-super-run-1" } };

  const systemMessage = new SystemMessage(
    "You are answering a Sandpit Super member's question. Use the search_pds tool to find " +
      "relevant PDS content -- do not rely on prior knowledge about superannuation fees. " +
      "Sandpit Super has both current and legacy fee structures; check the member's join date " +
      "against each section's eligibility criteria before deciding which applies. Always cite " +
      "the section number(s) you used to answer."
  );
  const humanMessage = new HumanMessage(
    "A member joined the fund on 15 December 2025. What are the administration fees and investment fees for their account?"
  );

  const firstResult = (await graph.invoke(
    { messages: [systemMessage, humanMessage] },
    config
  )) as InvokeResultWithInterrupt;

  if (firstResult.__interrupt__) {
    console.log("\n--- HUMAN REVIEW REQUIRED ---");
    console.log(JSON.stringify(firstResult.__interrupt__[0].value, null, 2));

    // A real deployment would surface this interrupt to a reviewer's UI and
    // resume later from a webhook/API call. This readline prompt simulates
    // that "later" synchronously, in the same process, for this demo.
    const readline = await import("node:readline/promises");
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const answer = await rl.question("Approve these values? (yes/no): ");
    rl.close();

    const decision = answer.trim().toLowerCase().startsWith("y") ? "approve" : "reject";
    const finalResult = await graph.invoke(new Command({ resume: decision }), config);
    console.log("\n--- FINAL STATE ---");
    console.log(finalResult);
  } else {
    // Reached END without an interrupt -- grounding check must have failed
    // and hard-stopped the run before review was ever reached.
    console.log("\nFinished without requiring review (likely flagged by grounding check):", firstResult);
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}