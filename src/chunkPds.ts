// Chunk node: splits the PDS's raw text into per-section chunks. Plain code,
// no LLM -- the document's own numbering ("7. Fees and Other Costs") is the
// deterministic split point, since the whole task hinges on which *section*
// a value comes from. Requires a capital letter right after "<number>. " so
// that fee figures like "4.5%" or "3.0% p.a." are never mistaken for a new
// section heading.

import mammoth from "mammoth";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";

export type PdsChunk = {
  sectionNumber: number;
  text: string;
};

const SECTION_HEADING = /^(\d+)\.\s+[A-Z]/;

export function chunkPdsText(rawText: string): PdsChunk[] {
  const lines = rawText.split("\n").map((line) => line.trim()).filter(Boolean);

  const chunks: PdsChunk[] = [];
  let current: PdsChunk | null = null;

  for (const line of lines) {
    const match = line.match(SECTION_HEADING);
    if (match) {
      if (current) chunks.push(current);
      current = { sectionNumber: Number(match[1]), text: line };
    } else if (current) {
      current.text += "\n" + line;
    }
    // Lines before the first numbered section (the title page, version
    // line) are dropped -- nothing there is something the agent needs to cite.
  }
  if (current) chunks.push(current);

  return chunks;
}

export async function extractAndChunkPds(docxBuffer: Buffer): Promise<PdsChunk[]> {
  const { value: rawText } = await mammoth.extractRawText({ buffer: docxBuffer });
  return chunkPdsText(rawText);
}

async function main() {
  const docxBuffer = await readFile("output/pds_downloaded.docx");
  const chunks = await extractAndChunkPds(docxBuffer);

  console.log(`Found ${chunks.length} chunks:`);
  chunks.forEach((c) => {
    const preview = c.text.length > 60 ? c.text.slice(0, 60) + "..." : c.text;
    console.log(`  Section ${c.sectionNumber}: ${preview}`);
  });

  await mkdir("output", { recursive: true });
  await writeFile("output/pds_chunks.json", JSON.stringify(chunks, null, 2));
  console.log("Wrote output/pds_chunks.json");
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}