/**
 * CLI: re-embed an existing index in place, without re-downloading anything.
 *
 *   npm run index:reembed
 *   npm run index:reembed -- --dry-run
 *
 * Why this exists rather than just re-running `npm run index:reset`:
 *
 *   1. **It is the honest way to compare embedding models.** A full re-index
 *      re-downloads and re-chunks the dataset, so any change in retrieval
 *      quality afterwards is confounded by a different corpus. This rewrites
 *      *only* the dense vectors, leaving chunk ids, text, BM25 postings and
 *      metadata byte-identical — so a benchmark before and after differs by
 *      exactly one variable.
 *   2. **It needs no network.** The store already carries every chunk's text,
 *      and re-downloading 14 languages from HuggingFace is both slow and, in
 *      practice, flaky enough to drop languages mid-run.
 *
 * Writes through a temp file and renames, so an interrupted run leaves the
 * existing index intact rather than half-rewritten.
 */
import { createReadStream } from 'node:fs';
import { rename, stat, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { config, REPO_ROOT } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { getEmbeddingProvider } from '../rag/embeddings/index.js';

/** Matches EmbeddedVectorStore's on-disk line format. */
interface SerializedRecord {
  id: string;
  chunkKey: string;
  text: string;
  metadata: Record<string, unknown>;
  isParent: boolean;
  sparse: { indices: number[]; values: number[] };
  vector: number[];
}

/** How many chunks to embed per forward pass. */
const BATCH = 32;

function dataFilePath(): string {
  return path.join(
    REPO_ROOT,
    'storage',
    'vectors',
    `${config.vectorStore.collection}.jsonl`,
  );
}

async function main(): Promise<void> {
  const dryRun = process.argv.includes('--dry-run');
  const source = dataFilePath();
  const provider = getEmbeddingProvider();

  const stats = await stat(source).catch(() => null);
  if (!stats) {
    throw new Error(
      `No index at ${source}. Run \`npm run index\` first — this script rewrites an existing one.`,
    );
  }

  process.stdout.write(
    [
      '',
      '  GoaRAG re-embed',
      '  ───────────────',
      `  index        ${path.relative(REPO_ROOT, source)} (${(stats.size / 1e6).toFixed(1)} MB)`,
      `  model        ${provider.model} · ${provider.dimensions}d`,
      `  mode         ${dryRun ? 'DRY RUN (nothing is written)' : 'rewrite in place'}`,
      '',
    ].join('\n'),
  );

  await provider.warmup();

  const reader = createInterface({
    input: createReadStream(source, 'utf8'),
    crlfDelay: Infinity,
  });

  const out: string[] = [];
  let batch: SerializedRecord[] = [];
  let processed = 0;
  let previousDimensions: number | null = null;
  const started = Date.now();

  const flush = async (): Promise<void> => {
    if (batch.length === 0) return;
    // Every stored record is a corpus passage, never a query — the asymmetric
    // prefix matters here and getting it backwards would poison the index.
    const vectors = await provider.embed(
      batch.map((record) => record.text),
      'passage',
    );
    batch.forEach((record, i) => {
      const vector = vectors[i];
      if (!vector) throw new Error(`Embedding returned no vector for chunk ${record.id}`);
      out.push(JSON.stringify({ ...record, vector: Array.from(vector) }));
    });
    processed += batch.length;
    batch = [];
    if (processed % (BATCH * 20) === 0) {
      const rate = processed / ((Date.now() - started) / 1000);
      process.stdout.write(`  ${processed} chunks · ${rate.toFixed(0)}/s\n`);
    }
  };

  for await (const line of reader) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const record = JSON.parse(trimmed) as SerializedRecord;
    previousDimensions ??= record.vector?.length ?? null;
    batch.push(record);
    if (batch.length >= BATCH) await flush();
  }
  await flush();

  const elapsed = (Date.now() - started) / 1000;
  process.stdout.write(
    [
      '',
      `  re-embedded  ${processed} chunks in ${elapsed.toFixed(1)}s (${(processed / elapsed).toFixed(0)}/s)`,
      `  dimensions   ${previousDimensions ?? '?'} → ${provider.dimensions}`,
      '',
    ].join('\n'),
  );

  if (dryRun) {
    process.stdout.write('  Dry run — index left untouched.\n\n');
    return;
  }

  // Temp file + rename: an interrupted write must not leave a half-rewritten
  // index behind, since there is no way to tell which lines carry which
  // model's vectors once they are mixed.
  const temp = `${source}.rebuild`;
  await writeFile(temp, `${out.join('\n')}\n`, 'utf8');
  await rename(temp, source);

  process.stdout.write(`  Wrote ${path.relative(REPO_ROOT, source)}\n\n`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    logger.error({ error: (error as Error).message }, 'Re-embed failed');
    process.stderr.write(`\n✖ ${(error as Error).message}\n\n`);
    process.exit(1);
  });
