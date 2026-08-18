/**
 * CLI: run the benchmark and write a JSON report.
 *
 *   npm run benchmark
 *   npm run benchmark -- --sample 25
 *   npm run benchmark -- --sample 10 --generation      # includes the LLM (slow)
 *   npm run benchmark -- --language hin_Deva
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import type { LatencyPercentiles } from '@goarag/shared';
import { config, REPO_ROOT } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { runBenchmark } from '../services/benchmark.service.js';
import { closeVectorStore } from '../rag/vector/index.js';

/** Read `--flag value` from argv. */
function argValue(flag: string): string | undefined {
  const index = process.argv.indexOf(flag);
  return index !== -1 ? process.argv[index + 1] : undefined;
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

function percentileRow(label: string, stats: LatencyPercentiles): string {
  if (stats.count === 0) return `    ${label.padEnd(14)} —`;
  return (
    `    ${label.padEnd(14)}` +
    `${formatMs(stats.p50).padStart(9)}` +
    `${formatMs(stats.p70).padStart(9)}` +
    `${formatMs(stats.p95).padStart(9)}` +
    `${formatMs(stats.p99).padStart(9)}` +
    `${formatMs(stats.p100).padStart(9)}` +
    `${formatMs(stats.mean).padStart(9)}`
  );
}

function bar(value: number, width = 24): string {
  const filled = Math.round(Math.max(0, Math.min(1, value)) * width);
  return `${'█'.repeat(filled)}${'░'.repeat(width - filled)}`;
}

/**
 * The task spec's budget, in milliseconds, for chunking + vector retrieval +
 * everything through to the final ranked context.
 */
const LATENCY_BUDGET_MS = 50;

/**
 * Pass/fail against the budget, judged at p100 rather than p50.
 *
 * "Under 50ms" is only a meaningful claim if the slowest measured query also
 * clears it — a p50 that passes while the tail runs at 300ms describes a
 * pipeline most users experience as slow. p50 and p70 are printed alongside
 * because the spec asks for them, but p100 is what decides the verdict.
 */
function budgetSection(stats: LatencyPercentiles): string[] {
  if (stats.count === 0) return [];
  const pass = stats.p100 <= LATENCY_BUDGET_MS;
  const headroom = LATENCY_BUDGET_MS - stats.p100;
  return [
    `  Latency budget  (query text in → ranked context out, ≤ ${LATENCY_BUDGET_MS}ms)`,
    '  ─────────────────────────────────────────────────────────────',
    `    p50   ${formatMs(stats.p50).padStart(9)}`,
    `    p70   ${formatMs(stats.p70).padStart(9)}`,
    `    p100  ${formatMs(stats.p100).padStart(9)}   ← the verdict is taken here`,
    '',
    pass
      ? `    ✔ PASS — ${formatMs(headroom)} of headroom at the slowest query (n=${stats.count})`
      : `    ✖ FAIL — over by ${formatMs(-headroom)} at the slowest query (n=${stats.count})`,
    '',
  ];
}

async function main(): Promise<void> {
  const sampleSize = Number(argValue('--sample') ?? 10);
  const generation = process.argv.includes('--generation');
  const language = argValue('--language');
  const concurrency = Number(argValue('--concurrency') ?? (generation ? 2 : 4));

  process.stdout.write(
    [
      '',
      '  GoaRAG benchmark',
      '  ────────────────',
      `  sample size   ${sampleSize} queries`,
      `  generation    ${generation ? 'enabled (slow — invokes the LLM per query)' : 'disabled (retrieval only)'}`,
      `  language      ${language ?? 'all'}`,
      `  embeddings    ${config.embedding.provider} · ${config.embedding.model}`,
      `  reranker      ${config.reranker.provider}`,
      '',
      '  Running…',
      '',
    ].join('\n'),
  );

  const result = await runBenchmark({
    sampleSize,
    generation,
    concurrency,
    ...(language ? { language } : {}),
  });

  const { quality } = result;
  const metrics: Array<[string, number]> = [
    ['Hit rate', quality.hitRate],
    ['Recall@5', quality.recallAt5],
    ['Recall@10', quality.recallAt10],
    ['Precision@5', quality.precisionAt5],
    ['MRR', quality.mrr],
    ['nDCG@5', quality.ndcgAt5],
  ];

  process.stdout.write(
    [
      '  Retrieval quality  (scored against the dataset’s is_selected labels)',
      '  ───────────────────────────────────────────────────────────────────',
      ...metrics.map(
        ([label, value]) =>
          `    ${label.padEnd(14)} ${bar(value)}  ${(value * 100).toFixed(1)}%`,
      ),
      '',
      '  Latency percentiles',
      '  ───────────────────',
      `    ${'stage'.padEnd(14)}${'p50'.padStart(9)}${'p70'.padStart(9)}${'p95'.padStart(9)}${'p99'.padStart(9)}${'p100'.padStart(9)}${'mean'.padStart(9)}`,
      percentileRow('embedding', result.latency.embedding),
      percentileRow('retrieval', result.latency.retrieval),
      percentileRow('reranking', result.latency.reranking),
      ...(generation ? [percentileRow('generation', result.latency.generation)] : []),
      percentileRow('retrieval path', result.latency.retrievalPath),
      ...(generation ? [percentileRow('total (+LLM)', result.latency.total)] : []),
      '',
      ...budgetSection(result.latency.retrievalPath),
      '  Summary',
      '  ───────',
      `    queries evaluated  ${quality.evaluatedQueries}`,
      `    wall clock         ${(result.durationMs / 1000).toFixed(1)}s`,
      `    avg confidence     ${(result.averageConfidence * 100).toFixed(1)}%`,
      ...(generation ? [`    tokens used        ${result.tokensUsed.totalTokens.toLocaleString()}`] : []),
      `    vector store       ${result.config.vectorStore}`,
      '',
    ].join('\n'),
  );

  // Persist the full report so runs can be compared over time.
  const outputDir = path.join(REPO_ROOT, 'docs', 'benchmarks');
  await mkdir(outputDir, { recursive: true });
  const filename = `benchmark-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  const outputPath = path.join(outputDir, filename);
  await writeFile(outputPath, JSON.stringify(result, null, 2), 'utf8');

  process.stdout.write(`  Full report: docs/benchmarks/${filename}\n\n`);
}

main()
  .then(() => closeVectorStore())
  .then(() => process.exit(0))
  .catch(async (error) => {
    logger.error({ error: (error as Error).message }, 'Benchmark failed');
    process.stderr.write(`\n✖ ${(error as Error).message}\n\n`);
    await closeVectorStore().catch(() => undefined);
    process.exit(1);
  });
