// Deterministic merge node: fills the supplied cover letter template with
// four values. No LLM involved -- this is plain substitution, which is
// exactly why it's plain code and not a model call. Hardcoded placeholder
// values here stand in for what the agent stage will eventually produce.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import PizZip from "pizzip";
import Docxtemplater from "docxtemplater";

export type LetterData = {
  JOIN_DATE: string;
  ADMINISTRATION_FEE: string;
  INVESTMENT_FEE: string;
  PDS_SECTION_REFERENCE: string;
};

export async function mergeLetter(templatePath: string, data: LetterData): Promise<Buffer> {
  const templateBuffer = await readFile(templatePath);
  const zip = new PizZip(templateBuffer);

  const doc = new Docxtemplater(zip, {
    paragraphLoop: true,
    linebreaks: true,
    delimiters: { start: "{{", end: "}}" },
  });

  doc.render(data);

  return doc.getZip().generate({ type: "nodebuffer" }) as Buffer;
}

async function main() {
  const placeholder: LetterData = {
    JOIN_DATE: "PLACEHOLDER",
    ADMINISTRATION_FEE: "PLACEHOLDER",
    INVESTMENT_FEE: "PLACEHOLDER",
    PDS_SECTION_REFERENCE: "PLACEHOLDER",
  };

  const buffer = await mergeLetter(
    "templates/Cover_Letter_Template.docx",
    placeholder
  );

  await mkdir("output", { recursive: true });
  const outPath = "output/filled_letter.docx";
  await writeFile(outPath, buffer);
  console.log(`Wrote ${outPath}`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}