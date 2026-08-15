/**
 * Web search service — DuckDuckGo Instant Answer API + HTML snippet scraping.
 *
 * Used as a fallback when the RAG index has insufficient evidence for a query.
 * No API key required. Returns structured snippets that are injected into the
 * LLM prompt as "web context" so the model can answer general questions.
 *
 * Two tiers:
 *   1. DuckDuckGo Instant Answer API (fast, structured, covers facts)
 *   2. DuckDuckGo HTML search → top-3 result snippets (richer, slower)
 */
import { logger } from '../utils/logger.js';

export interface WebSnippet {
  title: string;
  url: string;
  snippet: string;
  source: 'ddg_instant' | 'ddg_html';
}

export interface WebSearchResult {
  snippets: WebSnippet[];
  query: string;
  latencyMs: number;
}

const DDG_INSTANT_URL = 'https://api.duckduckgo.com/';
const DDG_HTML_URL = 'https://html.duckduckgo.com/html/';
const TIMEOUT_MS = 8_000;
const MAX_SNIPPETS = 5;
const MAX_SNIPPET_CHARS = 800;

/** Strip HTML tags from a string. */
function stripHtml(html: string): string {
  return html
    .replace(/<[^>]+>/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#039;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

/** Trim snippet to a readable length. */
function trimSnippet(text: string, max = MAX_SNIPPET_CHARS): string {
  const clean = stripHtml(text).trim();
  if (clean.length <= max) return clean;
  const cut = clean.slice(0, max);
  return cut.slice(0, cut.lastIndexOf(' ')) + '…';
}

/**
 * DuckDuckGo Instant Answer API.
 * Returns an abstract + related topics for factual queries.
 */
async function ddgInstant(query: string): Promise<WebSnippet[]> {
  const url = new URL(DDG_INSTANT_URL);
  url.searchParams.set('q', query);
  url.searchParams.set('format', 'json');
  url.searchParams.set('no_html', '1');
  url.searchParams.set('no_redirect', '1');
  url.searchParams.set('skip_disambig', '1');

  const response = await fetch(url.toString(), {
    headers: { 'User-Agent': 'GoaRAG/1.0 (+https://github.com/goarag)' },
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });

  if (!response.ok) return [];

  const data = await response.json() as {
    AbstractText?: string;
    AbstractURL?: string;
    AbstractSource?: string;
    Answer?: string;
    RelatedTopics?: Array<{
      Text?: string;
      FirstURL?: string;
      Name?: string;
      Topics?: Array<{ Text?: string; FirstURL?: string }>;
    }>;
  };

  const snippets: WebSnippet[] = [];

  // Main abstract
  if (data.AbstractText) {
    snippets.push({
      title: data.AbstractSource ?? 'Wikipedia',
      url: data.AbstractURL ?? 'https://duckduckgo.com',
      snippet: trimSnippet(data.AbstractText),
      source: 'ddg_instant',
    });
  }

  // Instant Answer (calculator, currency, etc.)
  if (data.Answer && data.Answer !== data.AbstractText) {
    snippets.push({
      title: 'DuckDuckGo Answer',
      url: 'https://duckduckgo.com',
      snippet: trimSnippet(data.Answer),
      source: 'ddg_instant',
    });
  }

  // Related topics
  const topics = data.RelatedTopics ?? [];
  for (const topic of topics) {
    if (snippets.length >= MAX_SNIPPETS) break;
    if (topic.Text && topic.FirstURL) {
      snippets.push({
        title: topic.Name ?? 'Related',
        url: topic.FirstURL,
        snippet: trimSnippet(topic.Text),
        source: 'ddg_instant',
      });
    }
    // Sub-topics
    for (const sub of topic.Topics ?? []) {
      if (snippets.length >= MAX_SNIPPETS) break;
      if (sub.Text && sub.FirstURL) {
        snippets.push({
          title: topic.Name ?? 'Related',
          url: sub.FirstURL,
          snippet: trimSnippet(sub.Text),
          source: 'ddg_instant',
        });
      }
    }
  }

  return snippets;
}

/**
 * DuckDuckGo HTML search — scrape result snippets from the lite HTML page.
 * Slower but richer than the Instant Answer API.
 */
async function ddgHtml(query: string): Promise<WebSnippet[]> {
  const form = new URLSearchParams({ q: query, kl: 'us-en' });

  const response = await fetch(DDG_HTML_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'User-Agent':
        'Mozilla/5.0 (compatible; GoaRAG/1.0; +https://github.com/goarag)',
      Accept: 'text/html',
    },
    body: form.toString(),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });

  if (!response.ok) return [];

  const html = await response.text();

  // Extract result blocks: <div class="result__body"> ... </div>
  const snippets: WebSnippet[] = [];
  const resultPattern =
    /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/gi;

  let match: RegExpExecArray | null;
  while ((match = resultPattern.exec(html)) !== null && snippets.length < MAX_SNIPPETS) {
    const [, url, titleHtml, snippetHtml] = match;
    if (!url || !snippetHtml) continue;
    const title = stripHtml(titleHtml ?? '');
    const snippet = trimSnippet(snippetHtml);
    if (snippet.length < 30) continue;
    snippets.push({ title, url, snippet, source: 'ddg_html' });
  }

  return snippets;
}

/**
 * Search the web for snippets relevant to a query.
 *
 * Runs both DDG tiers in parallel; deduplicates results; caps at MAX_SNIPPETS.
 */
export async function webSearch(query: string): Promise<WebSearchResult> {
  const started = Date.now();
  const results: WebSnippet[] = [];

  try {
    const [instant, html] = await Promise.allSettled([ddgInstant(query), ddgHtml(query)]);

    const instantSnippets = instant.status === 'fulfilled' ? instant.value : [];
    const htmlSnippets = html.status === 'fulfilled' ? html.value : [];

    // Merge, de-dup by url, prefer instant answers.
    const seen = new Set<string>();
    for (const s of [...instantSnippets, ...htmlSnippets]) {
      if (seen.has(s.url) || results.length >= MAX_SNIPPETS) continue;
      seen.add(s.url);
      results.push(s);
    }
  } catch (error) {
    logger.warn({ error: (error as Error).message }, 'Web search failed');
  }

  const latencyMs = Date.now() - started;
  logger.debug({ query, snippets: results.length, latencyMs }, 'Web search complete');

  return { snippets: results, query, latencyMs };
}

/**
 * Format web snippets as a context block for injection into an LLM prompt.
 */
export function formatWebContext(result: WebSearchResult): string {
  if (result.snippets.length === 0) return '';

  const lines = [
    '## Web Search Results',
    `Query: "${result.query}"`,
    '',
    ...result.snippets.map(
      (s, i) =>
        `[Web ${i + 1}] **${s.title}**\n${s.snippet}\nSource: ${s.url}`,
    ),
  ];

  return lines.join('\n\n');
}
