// Acquire node: programmatically retrieves the current PDS document from the
// live Sandpit Super site at runtime. The URL is discovered by rendering the
// page and reading its actual links -- not hardcoded -- because the site is
// a client-rendered SPA (raw HTML has no content until JS runs) and the
// brief requires genuine retrieval, not a pasted-in path.

import { chromium } from "playwright";
import { writeFile, mkdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const FORMS_PAGE_URL = "https://sandpitsuper.replit.app/forms";

export async function acquirePds(): Promise<Buffer> {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  await page.goto(FORMS_PAGE_URL, { waitUntil: "networkidle" });

  // The SPA renders links client-side, so we search the rendered DOM for
  // the document -- this is what makes the retrieval "real" rather than us
  // having found the URL once and typed it in here ourselves.
  const pdsHref = await page.evaluate(() => {
    const link = Array.from(document.querySelectorAll("a"))
      .find((a) => a.href.toLowerCase().endsWith(".docx"));
    return link?.href ?? null;
  });

  if (!pdsHref) {
    await browser.close();
    throw new Error("Could not find a .docx link on the forms page.");
  }

  // Fetch the bytes using the browser's own request context -- the same
  // "session" that just rendered the page does the retrieval, rather than
  // handing the URL off to an unrelated HTTP client.
  const response = await page.request.get(pdsHref);
  const buffer = await response.body();

  await browser.close();
  return buffer;
}

async function main() {
  const buffer = await acquirePds();
  await mkdir("output", { recursive: true });
  const outPath = "output/pds_downloaded.docx";
  await writeFile(outPath, buffer);
  console.log(`Wrote ${outPath} (${buffer.length} bytes)`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}