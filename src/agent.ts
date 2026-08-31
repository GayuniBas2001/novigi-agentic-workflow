// Agent node: the genuinely agentic part of the system. Given a member's
// details, the model must decide for itself to call the search tool
// (possibly more than once), read what comes back, and reason about which
// PDS section actually applies -- current vs legacy -- before answering.
// Nothing here hardcodes which section is correct; that's the model's job.

import "dotenv/config";
import {
  StateGraph,
  StateSchema,
  MessagesValue,
  START,
  END,
  type GraphNode,
  type ConditionalEdgeRouter,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { buildIndex, search, type IndexedChunk } from "./embedIndex.js";
import type { PdsChunk } from "./chunkPds.js";

const MODEL_ID = process.env.ANTHROPIC_MODEL ?? "claude-sonnet-4-5-20250929";

const State = new StateSchema({
  messages: MessagesValue,
});

export function buildAgentGraph(index: IndexedChunk[]) {
  // The tool closes over `index`. The model never sees embeddings directly --
  // it only ever gets back plain text excerpts through this tool call.
  const searchPds = tool(
    async ({ query }: { query: string }) => {
      const results = await search(index, query, 3);
      return results
        .map((r) => `<pds_excerpt section="${r.sectionNumber}">\n${r.text}\n</pds_excerpt>`)
        .join("\n\n");
    },
    {
      name: "search_pds",
      description:
        "Semantically searches the Sandpit Super Product Disclosure Statement and returns the most relevant excerpts, each tagged with its section number.",
      schema: z.object({
        query: z.string().describe("A natural-language question about fees, eligibility, or any other PDS topic."),
      }),
    }
  );

  const model = new ChatAnthropic({ model: MODEL_ID, temperature: 0 });
  const modelWithTools = model.bindTools([searchPds]);

  const callAgent: GraphNode<typeof State> = async (state) => {
    const response = await modelWithTools.invoke(state.messages);
    return { messages: [response] };
  };

  const shouldContinue: ConditionalEdgeRouter<typeof State, Record<string, any>, "tools"> = (state) => {
    const lastMessage = state.messages[state.messages.length - 1];
    if (lastMessage instanceof AIMessage && lastMessage.tool_calls?.length) {
      return "tools";
    }
    return END;
  };

  return new StateGraph(State)
    .addNode("agent", callAgent)
    .addNode("tools", new ToolNode([searchPds]))
    .addEdge(START, "agent")
    .addConditionalEdges("agent", shouldContinue, ["tools", END])
    .addEdge("tools", "agent")
    .compile();
}

async function main() {
  const chunksJson = await readFile("output/pds_chunks.json", "utf-8");
  const chunks: PdsChunk[] = JSON.parse(chunksJson);

  console.log("Building index...");
  const index = await buildIndex(chunks);

  const graph = buildAgentGraph(index);

  const systemMessage = new SystemMessage(
    "You are answering a Sandpit Super member's question. Use the search_pds tool to find " +
      "relevant PDS content -- do not rely on prior knowledge about superannuation fees. " +
      "Sandpit Super has both current and legacy fee structures; check the member's join date " +
      "against each section's eligibility criteria before deciding which applies. Always cite " +
      "the section number(s) you used to answer."
  );

  // TODO: replace with the exact fixed member query text from the brief,
  // verbatim -- not paraphrased.
  const memberQuery = "A member joined the fund on 15 December 2025. What are the administration fees and investment fees for their account?";
  const humanMessage = new HumanMessage(memberQuery);

  const result = await graph.invoke({ messages: [systemMessage, humanMessage] });

  const finalMessage = result.messages[result.messages.length - 1];
  console.log("\n--- FINAL ANSWER ---");
  console.log(finalMessage.content);

  console.log("\n--- FULL MESSAGE TRACE (for observability) ---");
  result.messages.forEach((m, i) => {
    console.log(`[${i}] ${m.getType()}:`, JSON.stringify(m.content).slice(0, 200));
  });
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}