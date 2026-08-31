// Embed & index node: turns each chunk's text into a vector so the agent's
// search tool can do semantic lookup ("fees for someone who joined in 2025")
// rather than requiring an exact keyword match. Runs entirely locally, no
// API key -- a document with 9 short chunks doesn't need a real vector
// database, just an array and cosine similarity.

import { pipeline, type FeatureExtractionPipeline } from "@xenova/transformers";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import type { PdsChunk } from "./chunkPds.js";

export type IndexedChunk = PdsChunk & {
  embedding: number[];
};

// The model is loaded once and reused -- loading it is the slow part
// (downloading/initializing), embedding a single string afterward is fast.
let extractor: FeatureExtractionPipeline | null = null;

async function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (!extractor) {
    extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  return extractor;
}

async function embedText(text: string): Promise<number[]> {
  const model = await getExtractor();
  const output = await model(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

export async function buildIndex(chunks: PdsChunk[]): Promise<IndexedChunk[]> {
  const indexed: IndexedChunk[] = [];
  for (const chunk of chunks) {
    const embedding = await embedText(chunk.text);
    indexed.push({ ...chunk, embedding });
  }
  return indexed;
}

// Both vectors are already unit-normalized (normalize: true above), so a
// plain dot product *is* the cosine similarity -- no need to divide by
// magnitudes separately.
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

export async function search(
  index: IndexedChunk[],
  query: string,
  topK = 3
): Promise<IndexedChunk[]> {
  const queryEmbedding = await embedText(query);
  return [...index]
    .sort(
      (a, b) =>
        cosineSimilarity(b.embedding, queryEmbedding) -
        cosineSimilarity(a.embedding, queryEmbedding)
    )
    .slice(0, topK);
}

async function main() {
  const chunksJson = await readFile("output/pds_chunks.json", "utf-8");
  const chunks: PdsChunk[] = JSON.parse(chunksJson);

  console.log("Building index (downloads the embedding model on first run)...");
  const index = await buildIndex(chunks);
  console.log(`Indexed ${index.length} chunks.`);

  const testQuery = "What fees apply to a member who joined in 2025?";
  const results = await search(index, testQuery, 3);

  console.log(`\nTop matches for: "${testQuery}"`);
  results.forEach((r) => console.log(`  Section ${r.sectionNumber}`));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}