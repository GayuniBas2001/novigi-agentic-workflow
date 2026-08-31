// Grounding check: deterministic, no LLM. Every numeric value asserted in
// the agent's final answer must actually appear somewhere the agent could
// have seen it -- either the member's own query, or a PDS excerpt returned
// by the search tool. If a number shows up that isn't grounded in either,
// that's a hard failure, not a warning: for financial figures we don't want
// the system silently trusting a plausible-looking number the model invented.

import { BaseMessage, HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { buildIndex } from "./embedIndex.js";
import { buildAgentGraph } from "./agent.js";
import type { PdsChunk } from "./chunkPds.js";

export type GroundingResult =
  | { grounded: true }
  | { grounded: false; ungroundedValues: string[] };

const NUMBER_PATTERN = /\$?\d+(\.\d+)?%?/g;

export function checkGrounding(messages: BaseMessage[]): GroundingResult {
  const finalAnswer = messages[messages.length - 1];
  const answerText =
    typeof finalAnswer.content === "string" ? finalAnswer.content : JSON.stringify(finalAnswer.content);

  // Everything the agent actually had access to: its own system instructions
  // don't contain member facts, so only the human query and tool results count.
  const groundingText = messages
    .filter((m) => m.getType() === "human" || m.getType() === "tool")
    .map((m) => (typeof m.content === "string" ? m.content : JSON.stringify(m.content)))
    .join("\n");

  const numbersInAnswer = answerText.match(NUMBER_PATTERN) ?? [];
  const ungroundedValues = [...new Set(numbersInAnswer)].filter((value) => !groundingText.includes(value));

  if (ungroundedValues.length > 0) {
    return { grounded: false, ungroundedValues };
  }
  return { grounded: true };
}

async function main() {
  // Test 1: a genuine run of the real agent. Should pass -- proves the
  // check doesn't false-positive on the system's actual behaviour.
  const chunksJson = await readFile("output/pds_chunks.json", "utf-8");
  const chunks: PdsChunk[] = JSON.parse(chunksJson);
  const index = await buildIndex(chunks);
  const graph = buildAgentGraph(index);

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

  const realResult = await graph.invoke({ messages: [systemMessage, humanMessage] });
  console.log("Test 1 (real agent run):", checkGrounding(realResult.messages));

  // Test 2: a synthetic, deliberately fabricated answer -- proves the check
  // actually catches a number that was never in the retrieved text, not
  // just that it never happens to trigger.
  const fakeMessages: BaseMessage[] = [
    humanMessage,
    new AIMessage("The administration fee is 2.75% p.a., which is not mentioned anywhere in the PDS."),
  ];
  console.log("Test 2 (fabricated number):", checkGrounding(fakeMessages));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}