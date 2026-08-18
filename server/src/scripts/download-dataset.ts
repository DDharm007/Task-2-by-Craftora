/**
 * CLI: load the MSMARCO-XI dataset. Virtual mode keeps it in memory only.
 *
 *   npm run dataset:download            # use the optional cache if enabled
 *   npm run dataset:download -- --force # bypass the optional cache
 */
import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { downloadDataset, toEvaluationCases } from '../services/dataset.service.js';

async function main(): Promise<void> {
  const force = process.argv.includes('--force');

  const bundle = await downloadDataset({ force });
  const passages = bundle.records.reduce((sum, record) => sum + record.passages.length, 0);
  const selected = bundle.records.reduce(
    (sum, record) => sum + record.passages.filter((passage) => passage.isSelected).length,
    0,
  );
  const languages = new Map<string, number>();
  for (const record of bundle.records) {
    languages.set(record.targetLanguage, (languages.get(record.targetLanguage) ?? 0) + 1);
  }
  const evaluationCases = toEvaluationCases(bundle);

  process.stdout.write(
    [
      '',
      '  Dataset ready',
      '  ─────────────',
      `  repo               ${bundle.repo}`,
      `  split              ${bundle.split}`,
      `  rows downloaded    ${bundle.records.length.toLocaleString()} of ${bundle.totalAvailable.toLocaleString()} available`,
      `  passages           ${passages.toLocaleString()}`,
      `  labelled relevant  ${selected.toLocaleString()} (is_selected = 1)`,
      `  benchmark cases    ${evaluationCases.length.toLocaleString()}`,
      `  languages          ${[...languages.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([tag, count]) => `${tag} (${count})`)
        .join(', ')}`,
      `  english included   ${config.dataset.includeEnglish ? 'yes' : 'no'}`,
      '',
      '  Next: npm run index',
      '',
    ].join('\n'),
  );
}

main().catch((error) => {
  logger.error({ error: (error as Error).message }, 'Dataset download failed');
  process.stderr.write(`\n✖ ${(error as Error).message}\n\n`);
  process.exit(1);
});
