/**
 * MSMARCO-XI dataset loader.
 *
 * `ai4bharat/MSMARCO-XI` is MS MARCO translated into Indic languages. Each row
 * holds one query in both English and a target language, its answer in both,
 * and the candidate passages — with an `is_selected` flag marking which
 * passages actually answer the query.
 *
 * That flag is the reason this dataset is worth the trouble: it gives us
 * ground-truth relevance labels, so `/api/benchmark` can report real retrieval
 * quality (recall@k, MRR, nDCG) instead of only latency.
 *
 * Two sources, because the obvious one does not work:
 *
 *   parquet (default) — the repo stores one Parquet file per language
 *     (`validation/hinval.parquet` etc.). We download the file once, cache it
 *     under dataset/raw/, and decode only the first N rows. ~440MB per
 *     language, paid once.
 *
 *   api (fallback) — HuggingFace's datasets-server. Its paginated `/rows`
 *     endpoint is permanently broken for this dataset: the nested `passages`
 *     struct trips an ArrowNotImplementedError during Parquet conversion, so
 *     `/rows`, `/search` and `/filter` all return HTTP 500. Only `/first-rows`
 *     responds, and it caps out at ~18 rows. Useful as a smoke test, too small
 *     to benchmark against.
 *
 * Set DATASET_SOURCE to pick. `auto` tries parquet and falls back to the API.
 */
import { createWriteStream } from 'node:fs';
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';
import { downloadFile, listFiles } from '@huggingface/hub';
import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects } from 'hyparquet';
import type { SourceDocument } from '@voxrag/shared';
import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { retry, withTimeout } from '../utils/async.js';
import { normalizeWhitespace } from '../utils/text.js';

const FIRST_ROWS_ENDPOINT = 'https://datasets-server.huggingface.co/first-rows';

/** The repo, in the shape @huggingface/hub expects. */
function hubRepo() {
  return { type: 'dataset' as const, name: config.dataset.repo };
}

/**
 * Language tag → the 3-letter stem the repo uses in its filenames
 * (`hin_Deva` → `hinval.parquet`). Only the stem is hardcoded; the actual
 * filename is matched against the repo listing, so a renamed or differently
 * suffixed file still resolves.
 */
const LANGUAGE_STEMS: Record<string, string> = {
  asm_Beng: 'asm',
  ben_Beng: 'ben',
  guj_Gujr: 'guj',
  hin_Deva: 'hin',
  kan_Knda: 'kan',
  mal_Mlym: 'mal',
  mar_Deva: 'mar',
  npi_Deva: 'nep',
  ory_Orya: 'ori',
  pan_Guru: 'pan',
  san_Deva: 'san',
  tam_Taml: 'tam',
  tel_Telu: 'tel',
  urd_Arab: 'urd',
};

export const SUPPORTED_DATASET_LANGUAGES = Object.keys(LANGUAGE_STEMS);

interface RepoFile {
  path: string;
  size: number;
}

let fileListCache: { split: string; files: RepoFile[] } | null = null;

/** List the Parquet files in a split, via the Hub API. Cached per process. */
async function listSplitFiles(split: string): Promise<RepoFile[]> {
  if (fileListCache?.split === split) return fileListCache.files;

  const files: RepoFile[] = [];
  for await (const entry of listFiles({ repo: hubRepo(), path: split, recursive: true })) {
    if (entry.type !== 'file' || !entry.path.endsWith('.parquet')) continue;
    files.push({ path: entry.path, size: entry.size ?? 0 });
  }

  if (files.length === 0) {
    throw new Error(`No Parquet files found under "${split}" in ${config.dataset.repo}`);
  }

  fileListCache = { split, files };
  return files;
}

/**
 * Resolve a language tag to its file in the repo listing.
 *
 * Matching against the real listing rather than constructing a filename means
 * we fail with a useful message if the repo layout changes, instead of 404ing.
 */
async function resolveParquetFile(language: string, split: string): Promise<RepoFile> {
  const files = await listSplitFiles(split);
  const stem = LANGUAGE_STEMS[language];

  if (!stem) {
    throw new Error(
      `Unknown dataset language "${language}". Supported: ${SUPPORTED_DATASET_LANGUAGES.join(', ')}`,
    );
  }

  const match = files.find((file) => {
    const base = file.path.slice(file.path.lastIndexOf('/') + 1).toLowerCase();
    return base.startsWith(stem);
  });

  if (!match) {
    throw new Error(
      `No Parquet file for "${language}" (expected a name starting with "${stem}") in ${split}/. ` +
        `Found: ${files.map((file) => file.path).join(', ')}`,
    );
  }
  return match;
}

interface RawRow {
  source_lang?: string;
  target_lang?: string;
  query_id?: number;
  query_type?: string;
  query?: string;
  Answer?: string;
  Eng_Query?: string;
  Eng_Answer?: string;
  passages?: {
    English_passages?: string[];
    Translated_passages?: string[];
    is_selected?: number[];
  };
}

interface RowsResponse {
  rows?: Array<{ row_idx: number; row: RawRow }>;
  num_rows_total?: number;
  error?: string;
}

/** One dataset row, normalised. */
export interface DatasetRecord {
  queryId: string;
  queryType: string;
  targetLanguage: string;
  /** Query in the target (Indic) language. */
  query: string;
  /** The same query in English. */
  englishQuery: string;
  answer: string;
  englishAnswer: string;
  passages: Array<{
    index: number;
    english: string;
    translated: string;
    isSelected: boolean;
  }>;
}

export interface DatasetBundle {
  repo: string;
  split: string;
  downloadedAt: string;
  /** Rows present in the source files, not rows kept — see `records.length`. */
  totalAvailable: number;
  records: DatasetRecord[];
  /**
   * The settings this bundle was built with.
   *
   * Without them a cached bundle looks equally valid whatever the current
   * config says, so raising DATASET_MAX_ROWS appeared to do nothing: the
   * download short-circuited and returned the smaller bundle forever.
   */
  builtWith?: {
    maxRows: number;
    languages: string[];
    includeEnglish: boolean;
  };
  /** Languages that failed, so a partial bundle is not mistaken for complete. */
  failedLanguages?: Array<{ language: string; error: string }>;
}

function bundlePath(): string {
  return path.join(config.dataset.dir, 'processed', `${config.dataset.split}.json`);
}

/** Fetch the sample rows the datasets-server will still serve. */
async function fetchFirstRows(): Promise<RowsResponse> {
  const url =
    `${FIRST_ROWS_ENDPOINT}?dataset=${encodeURIComponent(config.dataset.repo)}` +
    `&config=default&split=${encodeURIComponent(config.dataset.split)}`;

  return retry(
    async () => {
      const response = await withTimeout(fetch(url, { headers: { Accept: 'application/json' } }), 60_000);
      const body = (await response.json().catch(() => ({}))) as RowsResponse;
      if (!response.ok) {
        const error = new Error(
          `HuggingFace datasets-server returned HTTP ${response.status}: ${body.error ?? 'unknown error'}`,
        );
        (error as unknown as { status: number }).status = response.status;
        throw error;
      }
      return body;
    },
    {
      retries: 3,
      baseDelayMs: 1_000,
      onRetry: (error, attempt, delayMs) =>
        logger.warn({ attempt, delayMs, error: (error as Error).message }, 'Retrying dataset fetch'),
    },
  );
}

/**
 * Download a language's Parquet file into dataset/raw/, skipping the transfer
 * when a complete copy is already cached.
 */
async function ensureParquetFile(
  language: string,
  onProgress?: (received: number, total: number) => void,
): Promise<string> {
  const remote = await resolveParquetFile(language, config.dataset.split);
  const destination = path.join(
    config.dataset.dir,
    'raw',
    `${language}.${config.dataset.split}.parquet`,
  );

  // The listing gives the authoritative size, so a partial file left behind by
  // an interrupted download is re-fetched instead of silently decoded.
  try {
    const existing = await stat(destination);
    if (existing.size > 0 && (remote.size === 0 || existing.size === remote.size)) {
      logger.info(
        { language, file: destination, mb: (existing.size / 1048576).toFixed(1) },
        'Using cached Parquet file',
      );
      return destination;
    }
    logger.warn(
      { language, have: existing.size, want: remote.size },
      'Cached Parquet is incomplete — re-downloading',
    );
  } catch {
    // Not cached yet.
  }

  await mkdir(path.dirname(destination), { recursive: true });
  logger.info(
    { language, path: remote.path, mb: (remote.size / 1048576).toFixed(1) },
    'Downloading Parquet file (one-time)',
  );

  // `downloadFile` resolves to a Blob; stream it rather than buffering ~440MB.
  const blob = await downloadFile({ repo: hubRepo(), path: remote.path });
  if (!blob) {
    throw new Error(`Failed to download ${remote.path} from ${config.dataset.repo}`);
  }

  const temporary = `${destination}.part`;
  let received = 0;
  const source = Readable.fromWeb(blob.stream() as Parameters<typeof Readable.fromWeb>[0]);
  source.on('data', (piece: Buffer) => {
    received += piece.length;
    onProgress?.(received, remote.size);
  });

  try {
    await pipeline(source, createWriteStream(temporary));
  } catch (error) {
    // Otherwise a dropped transfer leaves a stray `.part` behind that no later
    // run ever looks at or cleans up.
    await rm(temporary, { force: true }).catch(() => {});
    throw error;
  }
  await rename(temporary, destination);

  logger.info({ language, mb: (received / 1048576).toFixed(1) }, 'Parquet download complete');
  return destination;
}

/** Total rows in a Parquet file, read from its footer — cheap, no decode. */
async function countParquetRows(filePath: string): Promise<number> {
  try {
    const file = await asyncBufferFromFile(filePath);
    const metadata = await parquetMetadataAsync(file);
    return Number(metadata.num_rows);
  } catch {
    return 0;
  }
}

/**
 * Decode the first `limit` rows of a cached Parquet file.
 *
 * `limit` bounds what is *returned*, not what is read: these files store all
 * ~98,000 rows in a single row group, and a row group is the unit of
 * decompression, so even 22 rows costs a full decode — measured at ~9.5s and
 * ~2.4 GB of RSS per language. Only the columns actually used are requested,
 * which is the one lever that reduces it.
 */
async function readParquetRows(filePath: string, limit: number): Promise<RawRow[]> {
  const file = await asyncBufferFromFile(filePath);
  const rows = (await parquetReadObjects({
    file,
    rowStart: 0,
    rowEnd: limit,
    columns: [
      'query_id',
      'query_type',
      'target_lang',
      'query',
      'Answer',
      'Eng_Query',
      'Eng_Answer',
      'passages',
    ],
  })) as unknown as RawRow[];
  return rows;
}

/**
 * Parquet decodes integers as BigInt and may hand back typed arrays.
 * Normalise both so the rest of the pipeline sees plain numbers.
 */
function toNumberArray(value: unknown): number[] {
  if (!value) return [];
  const list = Array.isArray(value) ? value : Array.from(value as ArrayLike<unknown>);
  return list.map((item) => (typeof item === 'bigint' ? Number(item) : Number(item ?? 0)));
}

function toStringArray(value: unknown): string[] {
  if (!value) return [];
  const list = Array.isArray(value) ? value : Array.from(value as ArrayLike<unknown>);
  return list.map((item) => (item == null ? '' : String(item)));
}

function normalizeRow(raw: RawRow): DatasetRecord | null {
  const english = toStringArray(raw.passages?.English_passages);
  const translated = toStringArray(raw.passages?.Translated_passages);
  const selected = toNumberArray(raw.passages?.is_selected);

  const count = Math.max(english.length, translated.length);
  if (count === 0) return null;

  const passages: DatasetRecord['passages'] = [];
  for (let i = 0; i < count; i += 1) {
    const englishText = normalizeWhitespace(english[i] ?? '');
    const translatedText = normalizeWhitespace(translated[i] ?? '');
    if (!englishText && !translatedText) continue;
    passages.push({
      index: i,
      english: englishText,
      translated: translatedText,
      isSelected: (selected[i] ?? 0) === 1,
    });
  }
  if (passages.length === 0) return null;

  return {
    queryId: String(raw.query_id ?? '').replace(/n$/, ''),
    queryType: raw.query_type ?? 'UNKNOWN',
    targetLanguage: raw.target_lang ?? 'unknown',
    query: normalizeWhitespace(raw.query ?? ''),
    englishQuery: normalizeWhitespace(raw.Eng_Query ?? '').replace(/^[.\s]+/, ''),
    answer: normalizeWhitespace(raw.Answer ?? ''),
    englishAnswer: normalizeWhitespace(raw.Eng_Answer ?? ''),
    passages,
  };
}

export interface DownloadOptions {
  force?: boolean;
  onProgress?: (message: string) => void;
}

/**
 * Why a cached bundle can no longer be used, or `null` if it still matches.
 *
 * A bundle smaller than the current target is the case that matters: it is what
 * made raising DATASET_MAX_ROWS look like a no-op.
 */
function cacheMismatch(
  cached: DatasetBundle,
  current: { wanted: number; languages: string[] },
): string | null {
  if (cached.repo !== config.dataset.repo) return 'repo changed';
  if (cached.split !== config.dataset.split) return 'split changed';

  const built = cached.builtWith;
  // Bundles written before `builtWith` existed can only be judged by size.
  if (!built) {
    return cached.records.length < current.wanted ? 'bundle predates the current settings' : null;
  }

  if (built.includeEnglish !== config.dataset.includeEnglish) return 'includeEnglish changed';

  const sameLanguages =
    built.languages.length === current.languages.length &&
    built.languages.every((language) => current.languages.includes(language));
  if (!sameLanguages) return 'languages changed';

  // Only a raised target forces a rebuild; lowering it is satisfied by slicing.
  if (current.wanted > built.maxRows && cached.records.length < current.wanted) {
    return `row target raised (${built.maxRows} → ${current.wanted})`;
  }

  return null;
}

/**
 * Fetch dataset rows and cache the normalised bundle on disk.
 *
 * Rows are split evenly across the configured languages so a multilingual
 * index does not end up dominated by whichever file was read first.
 */
export async function downloadDataset(options: DownloadOptions = {}): Promise<DatasetBundle> {
  const destination = bundlePath();
  const report = (message: string) => options.onProgress?.(message);

  const wanted = config.dataset.maxRows;
  const languages =
    config.dataset.languages.length > 0 ? config.dataset.languages : SUPPORTED_DATASET_LANGUAGES;
  const source = config.dataset.source;

  if (!options.force) {
    const cached = await loadDataset().catch(() => null);
    if (cached && cached.records.length > 0) {
      const stale = cacheMismatch(cached, { wanted, languages });
      if (!stale) {
        logger.info(
          { records: cached.records.length, file: destination },
          'Using cached dataset (pass --force to re-download)',
        );
        return cached;
      }
      logger.warn(
        { reason: stale, cached: cached.records.length, wanted },
        'Cached dataset no longer matches the configuration — rebuilding',
      );
      report(`Cached dataset is stale (${stale}) — rebuilding`);
    }
  }

  logger.info(
    { repo: config.dataset.repo, split: config.dataset.split, rows: wanted, languages, source },
    'Preparing dataset',
  );

  let records: DatasetRecord[] = [];
  let totalAvailable = 0;
  const failedLanguages: Array<{ language: string; error: string }> = [];

  if (source === 'parquet' || source === 'auto') {
    const perLanguage = Math.max(1, Math.ceil(wanted / languages.length));

    for (const language of languages) {
      if (records.length >= wanted) break;

      // Isolated per language. A single file that 404s, or a transfer that
      // drops on the eighth of fourteen ~450 MB downloads, must not discard
      // the languages that already succeeded.
      try {
        let lastPercent = -1;
        const file = await ensureParquetFile(language, (received, total) => {
          if (total <= 0) return;
          const percent = Math.floor((received / total) * 100);
          if (percent !== lastPercent && percent % 5 === 0) {
            lastPercent = percent;
            report(
              `Downloading ${language}: ${percent}% (${(received / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)} MB)`,
            );
          }
        });

        report(`Decoding ${language} (${perLanguage} rows)`);
        totalAvailable += await countParquetRows(file);

        const rows = await readParquetRows(file, perLanguage);
        for (const row of rows) {
          if (records.length >= wanted) break;
          const record = normalizeRow(row);
          if (record) records.push(record);
        }
        logger.info({ language, kept: records.length }, 'Decoded Parquet rows');
      } catch (error) {
        const message = (error as Error).message;
        failedLanguages.push({ language, error: message });
        logger.warn({ language, error: message }, 'Language failed — continuing with the rest');
        report(`Skipped ${language}: ${message.slice(0, 120)}`);
      }
    }

    if (records.length === 0 && failedLanguages.length > 0 && source === 'parquet') {
      throw new Error(
        `Every language failed. First error: ${failedLanguages[0]?.error ?? 'unknown'}`,
      );
    }
    if (records.length === 0) {
      logger.warn(
        { failed: failedLanguages.length },
        'Parquet source yielded nothing — falling back to the datasets-server API',
      );
    }
  }

  if (records.length === 0) {
    report('Fetching sample rows from the HuggingFace datasets-server');
    const page = await fetchFirstRows();
    const rows = page.rows ?? [];
    totalAvailable = page.num_rows_total ?? rows.length;
    for (const { row } of rows) {
      if (records.length >= wanted) break;
      const record = normalizeRow(row);
      if (record) records.push(record);
    }
    if (records.length > 0) {
      logger.warn(
        { records: records.length },
        'Using the datasets-server sample — it caps at ~18 rows, so the index will be small',
      );
    }
  }

  if (records.length === 0) {
    throw new Error(
      'Could not load any dataset rows. Check the network, or set DATASET_SOURCE=parquet with DATASET_LANGUAGES.',
    );
  }

  const bundle: DatasetBundle = {
    repo: config.dataset.repo,
    split: config.dataset.split,
    downloadedAt: new Date().toISOString(),
    totalAvailable,
    records,
    builtWith: {
      maxRows: wanted,
      languages,
      includeEnglish: config.dataset.includeEnglish,
    },
    ...(failedLanguages.length > 0 ? { failedLanguages } : {}),
  };

  if (failedLanguages.length > 0) {
    logger.warn(
      { failed: failedLanguages.map((f) => f.language) },
      'Dataset built with some languages missing — re-run to retry them',
    );
  }

  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, JSON.stringify(bundle), 'utf8');

  const passageCount = records.reduce((sum, record) => sum + record.passages.length, 0);
  logger.info(
    { records: records.length, passages: passageCount, file: destination },
    'Dataset downloaded',
  );

  return bundle;
}

/** Read the cached bundle from disk. */
export async function loadDataset(): Promise<DatasetBundle> {
  const raw = await readFile(bundlePath(), 'utf8');
  return JSON.parse(raw) as DatasetBundle;
}

export async function datasetExists(): Promise<boolean> {
  try {
    const bundle = await loadDataset();
    return bundle.records.length > 0;
  } catch {
    return false;
  }
}

/**
 * Flatten records into indexable documents.
 *
 * Every passage becomes one document. The translated passage and its English
 * original are indexed as separate documents sharing a `documentId` prefix, so
 * a Hindi question can match the Hindi text directly *or* reach the English
 * text through BGE-M3's shared multilingual space.
 */
export function toSourceDocuments(bundle: DatasetBundle): SourceDocument[] {
  const documents: SourceDocument[] = [];

  for (const record of bundle.records) {
    const source = `${bundle.repo}/${bundle.split}#${record.queryId}`;

    for (const passage of record.passages) {
      const base = `${record.queryId}-${passage.index}`;

      if (passage.translated) {
        documents.push({
          documentId: `${base}-${record.targetLanguage}`,
          passageId: String(passage.index),
          text: passage.translated,
          language: record.targetLanguage,
          source,
          topic: record.query || record.englishQuery,
          isSelected: passage.isSelected,
          queryId: record.queryId,
          queryText: record.query || record.englishQuery,
        });
      }

      if (config.dataset.includeEnglish && passage.english) {
        documents.push({
          documentId: `${base}-eng_Latn`,
          passageId: String(passage.index),
          text: passage.english,
          language: 'eng_Latn',
          source,
          topic: record.englishQuery || record.query,
          isSelected: passage.isSelected,
          queryId: record.queryId,
          queryText: record.englishQuery || record.query,
        });
      }
    }
  }

  return documents;
}

/**
 * Ground-truth evaluation set built from `is_selected`.
 *
 * Each case is a query plus the document ids that genuinely answer it, which
 * is exactly what recall@k and MRR need.
 */
export interface EvaluationCase {
  queryId: string;
  query: string;
  englishQuery: string;
  language: string;
  expectedDocumentIds: string[];
  answer: string;
}

export function toEvaluationCases(bundle: DatasetBundle): EvaluationCase[] {
  const cases: EvaluationCase[] = [];

  for (const record of bundle.records) {
    const expected: string[] = [];
    for (const passage of record.passages) {
      if (!passage.isSelected) continue;
      const base = `${record.queryId}-${passage.index}`;
      if (passage.translated) expected.push(`${base}-${record.targetLanguage}`);
      if (config.dataset.includeEnglish && passage.english) expected.push(`${base}-eng_Latn`);
    }

    // A query with no selected passage cannot be scored.
    if (expected.length === 0) continue;

    cases.push({
      queryId: record.queryId,
      query: record.query || record.englishQuery,
      englishQuery: record.englishQuery,
      language: record.targetLanguage,
      expectedDocumentIds: expected,
      answer: record.englishAnswer || record.answer,
    });
  }

  return cases;
}
