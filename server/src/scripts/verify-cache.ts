/**
 * Ad-hoc check that the two-tier retrieval cache behaves as designed.
 *
 * Asserts the three things that actually matter and are easy to get wrong:
 *   1. a cold query misses and pays the full path;
 *   2. the identical query hits L1 and skips the encoder entirely;
 *   3. a reworded query hits L2 — paying for an embedding but nothing else —
 *      and returns the same chunks as the query it matched.
 */
import { StageTimer } from '../utils/async.js';
import { retrieve } from '../rag/retriever/index.js';
import { retrievalCache } from '../rag/retriever/cache.js';
import { warmPipeline } from '../services/warmup.service.js';
import { closeVectorStore } from '../rag/vector/index.js';

const COLD = 'What is a corporation?';
/** Same question, different punctuation and case — must normalise to COLD. */
const SAME = '  what is a CORPORATION  ';
/**
 * Contracted, so normalisation cannot collapse it onto COLD — "what's" and
 * "what is" are different strings under any case/whitespace/punctuation rule.
 * Only a vector comparison can match this one, so it is what actually
 * exercises L2.
 */
const SIMILAR = "what's a corporation";

async function once(query: string): Promise<{ ms: number; path: string; top: string }> {
  const timer = new StageTimer();
  const started = performance.now();
  const outcome = await retrieve({ query, options: {}, timer });
  const ms = performance.now() - started;
  return {
    ms,
    path: outcome.cache ? `${outcome.cache.tier} (${outcome.cache.similarity.toFixed(4)})` : 'full',
    top: outcome.chunks[0]?.id ?? '—',
  };
}

async function main(): Promise<void> {
  await warmPipeline();
  retrievalCache.clear();

  const cold = await once(COLD);
  const exact = await once(SAME);
  const similar = await once(SIMILAR);

  const rows = [
    ['cold (first ask)', COLD, cold],
    ['same, re-punctuated', SAME.trim(), exact],
    ['reworded', SIMILAR, similar],
  ] as const;

  process.stdout.write('\n  Retrieval cache\n  ───────────────\n');
  for (const [label, query, result] of rows) {
    process.stdout.write(
      `    ${label.padEnd(22)} ${result.ms.toFixed(2).padStart(8)}ms  ${result.path.padEnd(24)} top=${result.top.slice(0, 8)}\n`,
    );
    void query;
  }

  const stats = retrievalCache.stats();
  process.stdout.write(
    `\n    stats  l1=${stats.l1Hits} l2=${stats.l2Hits} misses=${stats.misses} size=${stats.size}\n`,
  );

  const failures: string[] = [];
  if (cold.path !== 'full') failures.push('cold query should not have been a cache hit');
  if (!exact.path.startsWith('l1')) failures.push('re-punctuated query should hit L1');
  if (exact.top !== cold.top) failures.push('L1 hit returned different chunks');
  // Specifically L2: if this lands on L1 the test is not exercising the
  // semantic tier at all, which is the half most likely to be broken.
  if (!similar.path.startsWith('l2')) {
    failures.push(`contracted query should hit L2, got "${similar.path}"`);
  }
  if (similar.top !== cold.top) failures.push('cache hit returned different chunks');
  if (exact.ms > cold.ms) failures.push('L1 hit was slower than the cold path');

  if (failures.length > 0) {
    process.stdout.write(`\n  ✖ ${failures.join('\n  ✖ ')}\n\n`);
    process.exitCode = 1;
    return;
  }
  process.stdout.write('\n  ✔ all checks passed\n\n');
}

main()
  .then(() => closeVectorStore())
  .catch(async (error) => {
    process.stderr.write(`\n✖ ${(error as Error).message}\n\n`);
    await closeVectorStore().catch(() => undefined);
    process.exitCode = 1;
  });
