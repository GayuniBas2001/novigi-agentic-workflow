// Deterministic extraction: pulls the four merge-letter values directly out
// of source text the agent actually retrieved -- never out of the agent's
// own paraphrased prose. This is what makes the merge step genuinely
// "merged, not generated": every value placed in the letter is a literal
// substring of either the member's query or the PDS excerpt the agent cited.

import type { BaseMessage } from "@langchain/core/messages";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import type { LetterData } from "./mergeLetter.js";
import { mergeLetter } from "./mergeLetter.js";
import { buildIndex } from "./embedIndex.js";
import { buildAgentGraph } from "./agent.js";
import { checkGrounding } from "./groundingCheck.js";
import type { PdsChunk } from "./chunkPds.js";

function getTextContent(message: BaseMessage): string {
  return typeof message.content === "string" ? message.content : JSON.stringify(message.content);
}

export function extractMergeValues(messages: BaseMessage[]): LetterData {
  const humanMessage = messages.find((m) => m.getType() === "human");
  const finalAnswer = messages[messages.length - 1];
  if (!humanMessage) throw new Error("No member query found in message history.");

  const queryText = getTextContent(humanMessage);
  const answerText = getTextContent(finalAnswer);

  // Join date: parsed from the member's own query, not from the PDS.
  const joinDateMatch = queryText.match(/joined the fund on ([^.]+)\./i);
  if (!joinDateMatch) throw new Error("Could not find a join date in the member query.");
  const joinDate = joinDateMatch[1].trim();

  // Which section did the agent actually cite as applicable?
  const sectionMatch = answerText.match(/Section\s+(\d+)/i);
  if (!sectionMatch) throw new Error("The agent's answer did not cite a section number.");
  const sectionNumber = sectionMatch[1];

  // Find the raw excerpt for that exact section among the tool results --
  // the real source text, not the agent's summary of it.
  const excerptMessage = messages.find(
    (m) => m.getType() === "tool" && getTextContent(m).includes(`section="${sectionNumber}"`)
  );
  if (!excerptMessage) {
    throw new Error(`No retrieved excerpt found for the cited Section ${sectionNumber}.`);
  }
  const excerptText = getTextContent(excerptMessage);

  // Each fee sits on its own line in the source text, so "everything after
  // the label to the end of that line" is all we need -- `.` never matches
  // a newline by default, so this can't accidentally run past the line.
  const investmentFeeMatch = excerptText.match(/Investment Fees:\s*(.+)/);
  const adminFeeMatch = excerptText.match(/Administration Fees:\s*(.+)/);

  if (!investmentFeeMatch || !adminFeeMatch) {
    throw new Error(
      `Could not deterministically extract fee values from Section ${sectionNumber}'s text -- ` +
        `its layout doesn't match the expected "Investment Fees: ... / Administration Fees: ..." pattern.`
    );
  }

  const clean = (s: string) => s.trim().replace(/\.$/, "");

  return {
    JOIN_DATE: joinDate,
    INVESTMENT_FEE: clean(investmentFeeMatch[1]),
    ADMINISTRATION_FEE: clean(adminFeeMatch[1]),
    PDS_SECTION_REFERENCE: `Section ${sectionNumber}`,
  };
}

async function main() {
  // Full manual chain, end to end: agent -> grounding check -> extraction ->
  // merge -> real .docx with real values. This previews what graph.ts will
  // formalize as one StateGraph once the human review node is added.
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

  const result = await graph.invoke({ messages: [systemMessage, humanMessage] });

  const grounding = checkGrounding(result.messages);
  if (!grounding.grounded) {
    throw new Error(`Grounding check failed -- ungrounded values: ${grounding.ungroundedValues.join(", ")}`);
  }

  const letterData = extractMergeValues(result.messages);
  console.log("Extracted merge values:", letterData);

  const buffer = await mergeLetter("templates/Cover_Letter_Template.docx", letterData);
  await mkdir("output", { recursive: true });
  await writeFile("output/filled_letter_real.docx", buffer);
  console.log("Wrote output/filled_letter_real.docx");
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}