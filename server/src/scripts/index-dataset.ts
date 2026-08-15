/**
 * CLI: chunk, embed and index the dataset.
 *
 *   npm run index                # incremental (deterministic ids upsert in place)
 *   npm run index:reset          # wipe the collection first
 *   npm run index -- --download  # force a fresh dataset download
 */
import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { runIndexing } from '../services/indexing.service.js';
import { closeVectorStore } from '../rag/vector/index.js';

function renderProgressBar(processed: number, total: number, width = 30): string {
  if (total === 0) return '';
  const ratio = Math.min(1, processed / total);
  const filled = Math.round(ratio * width);
  return `[${'█'.repeat(filled)}${'░'.repeat(width - filled)}] ${(ratio * 100).toFixed(0)}%`;
}

async function main(): Promise<void> {
  const reset = process.argv.includes('--reset');
  const forceDownload = process.argv.includes('--download');

  process.stdout.write(
    [
      '',
      '  GoaRAG indexer',
      '  ──────────────',
      `  embeddings   ${config.embedding.provider} · ${config.embedding.model} (${config.embedding.dimensions}d)`,
      `  vector store ${config.vectorStore.driver}`,
      `  collection   ${config.vectorStore.collection}`,
      `  dataset      ${config.dataset.repo} [${config.dataset.split}] · ${config.dataset.maxRows} rows`,
      reset ? '  mode         RESET (existing vectors will be deleted)' : '  mode         incremental upsert',
      '',
    ].join('\n'),
  );

  let lastLine = '';
  const result = await runIndexing({
    reset,
    forceDownload,
    onProgress: (progress) => {
      const bar =
        progress.phase === 'embedding' ? ` ${renderProgressBar(progress.processed, progress.total)}` : '';
      const line = `  ${progress.phase.padEnd(12)} ${progress.message}${bar}`;
      if (line !== lastLine) {
        // Rewrite in place during embedding so the log stays one line.
        if (progress.phase === 'embedding' && process.stdout.isTTY) {
          process.stdout.write(`\r${line.padEnd(100)}`);
        } else {
          process.stdout.write(`${line}\n`);
        }
        lastLine = line;
      }
    },
  });

  if (process.stdout.isTTY) process.stdout.write('\n');

  process.stdout.write(
    [
      '',
      '  Index built',
      '  ───────────',
      `  passages indexed   ${result.documents.toLocaleString()}`,
      `  retrievable chunks ${result.chunks.toLocaleString()}`,
      `  parent chunks      ${result.parents.toLocaleString()}`,
      `  vectors written    ${result.vectors.toLocaleString()}`,
      `  avg chunk tokens   ${result.averageChunkTokens}`,
      `  embedding time     ${(result.embeddingMs / 1000).toFixed(1)}s`,
      `  total time         ${(result.durationMs / 1000).toFixed(1)}s`,
      '',
      '  chunking strategies',
      ...result.strategies
        .sort((a, b) => b.count - a.count)
        .map((entry) => `    ${entry.strategy.padEnd(16)} ${entry.count.toLocaleString()}`),
      '',
      '  languages',
      ...result.languages
        .slice(0, 12)
        .map((entry) => `    ${entry.language.padEnd(16)} ${entry.count.toLocaleString()}`),
      '',
      '  Next: npm run dev',
      '',
    ].join('\n'),
  );
}

main()
  .then(() => closeVectorStore())
  .then(() => process.exit(0))
  .catch(async (error) => {
    logger.error({ error: (error as Error).message }, 'Indexing failed');
    process.stderr.write(`\n✖ ${(error as Error).message}\n\n`);
    await closeVectorStore().catch(() => undefined);
    process.exit(1);
  });
