/**
 * Multilingual text utilities.
 *
 * MSMARCO-XI spans Latin, Devanagari, Bengali, Tamil, Telugu, Arabic and other
 * scripts, so every routine here is Unicode-aware. Nothing assumes whitespace
 * word boundaries or ASCII punctuation.
 */

/** Sentence terminators across the scripts present in the dataset. */
const SENTENCE_TERMINATORS = /([.!?。！？…]|॥|।|؟|۔)+/u;

/** Characters that count as word constituents in any script. */
const WORD_CHAR = /[\p{L}\p{N}\p{M}]/u;

/**
 * Approximate token count.
 *
 * A real BPE count would need the tokenizer loaded, which is far too heavy for
 * chunk-boundary decisions made millions of times during indexing. This
 * heuristic blends character and word counts and is calibrated against BGE-M3's
 * XLM-R tokenizer: Latin text averages ~4 chars/token while Indic abugidas
 * average ~2.6, so we weight by the script actually present.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  const chars = text.length;
  if (chars === 0) return 0;

  // Fraction of characters in scripts that tokenize densely.
  let dense = 0;
  for (const ch of text) {
    if (/[\p{Script=Devanagari}\p{Script=Bengali}\p{Script=Tamil}\p{Script=Telugu}\p{Script=Kannada}\p{Script=Malayalam}\p{Script=Gujarati}\p{Script=Gurmukhi}\p{Script=Oriya}\p{Script=Arabic}\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u.test(ch)) {
      dense += 1;
    }
  }
  const denseRatio = dense / chars;
  const charsPerToken = 4.0 - denseRatio * 1.4; // 4.0 for Latin → 2.6 for Indic
  return Math.max(1, Math.ceil(chars / charsPerToken));
}

/** Split text into sentences, preserving their terminators and offsets. */
export interface Sentence {
  text: string;
  start: number;
  end: number;
}

export function splitSentences(text: string): Sentence[] {
  const out: Sentence[] = [];
  if (!text.trim()) return out;

  // A hand-rolled splitter rather than Intl.Segmenter: the segmenter does not
  // break on the Devanagari danda (।) or the Arabic full stop (۔), both of
  // which are the primary sentence terminators across this dataset.
  let start = 0;
  let buffer = '';
  for (let i = 0; i < text.length; i += 1) {
    buffer += text[i];
    const ch = text[i] ?? '';
    if (SENTENCE_TERMINATORS.test(ch)) {
      // Consume any run of terminators plus trailing quotes/brackets.
      let j = i + 1;
      while (j < text.length && /[.!?。！？…॥।؟۔"'”’)\]]/u.test(text[j] ?? '')) {
        buffer += text[j];
        j += 1;
      }
      i = j - 1;
      const trimmed = buffer.trim();
      if (trimmed) out.push({ text: trimmed, start, end: i + 1 });
      buffer = '';
      start = i + 1;
    }
  }
  const tail = buffer.trim();
  if (tail) out.push({ text: tail, start, end: text.length });
  return mergeTinySentences(out);
}

/** Fold very short fragments (abbreviations, stray numerals) into their neighbour. */
function mergeTinySentences(sentences: Sentence[]): Sentence[] {
  const out: Sentence[] = [];
  for (const s of sentences) {
    const prev = out[out.length - 1];
    if (prev && estimateTokens(s.text) < 4) {
      prev.text = `${prev.text} ${s.text}`.trim();
      prev.end = s.end;
    } else {
      out.push({ ...s });
    }
  }
  return out;
}

/**
 * Unicode-aware tokenizer used by BM25 and the lexical guardrails.
 * Splits on non-word characters and lowercases via full case folding.
 */
export function tokenize(text: string): string[] {
  if (!text) return [];
  const tokens: string[] = [];
  let current = '';
  for (const ch of text.toLowerCase()) {
    if (WORD_CHAR.test(ch)) {
      current += ch;
    } else if (current) {
      tokens.push(current);
      current = '';
    }
  }
  if (current) tokens.push(current);
  return tokens;
}

/**
 * English stopwords only. Indic stopword lists vary by language and removing
 * them wrongly hurts recall, so non-Latin tokens are always kept.
 */
const STOPWORDS = new Set([
  'a','an','the','and','or','but','if','then','than','so','because','as','of','at','by','for','with',
  'about','against','between','into','through','during','before','after','above','below','to','from',
  'up','down','in','out','on','off','over','under','again','further','once','here','there','when',
  'where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor',
  'not','only','own','same','too','very','can','will','just','should','now','is','are','was','were',
  'be','been','being','have','has','had','having','do','does','did','doing','would','could','shall',
  'may','might','must','i','you','he','she','it','we','they','me','him','her','them','my','your',
  'his','its','our','their','this','that','these','those','what','which','who','whom','whose',
]);

export function isStopword(token: string): boolean {
  return STOPWORDS.has(token);
}

/** Tokens with English stopwords and 1-character noise removed. */
export function contentTokens(text: string): string[] {
  return tokenize(text).filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

/** Jaccard overlap between two token sets, 0-1. */
export function jaccard(a: Iterable<string>, b: Iterable<string>): number {
  const setA = a instanceof Set ? a : new Set(a);
  const setB = b instanceof Set ? b : new Set(b);
  if (setA.size === 0 || setB.size === 0) return 0;
  let intersection = 0;
  for (const t of setA) if (setB.has(t)) intersection += 1;
  return intersection / (setA.size + setB.size - intersection);
}

/** Fraction of `needles` present in `haystack`, 0-1. */
export function coverage(needles: Iterable<string>, haystack: Set<string>): number {
  const list = [...needles];
  if (list.length === 0) return 0;
  let hit = 0;
  for (const t of list) if (haystack.has(t)) hit += 1;
  return hit / list.length;
}

/** Collapse runs of whitespace and strip zero-width characters. */
export function normalizeWhitespace(text: string): string {
  return text
    .replace(/[\u200B-\u200D\uFEFF]/g, '')
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/** Truncate on a word boundary, appending an ellipsis when cut. */
export function truncate(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const cut = text.slice(0, maxChars);
  const lastSpace = cut.lastIndexOf(' ');
  return `${(lastSpace > maxChars * 0.6 ? cut.slice(0, lastSpace) : cut).trimEnd()}…`;
}

/** Detect the dominant Unicode script, used to label auto-detected text. */
export function detectScript(text: string): string {
  const counts = new Map<string, number>();
  const scripts: Array<[string, RegExp]> = [
    ['Latin', /\p{Script=Latin}/u],
    ['Devanagari', /\p{Script=Devanagari}/u],
    ['Bengali', /\p{Script=Bengali}/u],
    ['Tamil', /\p{Script=Tamil}/u],
    ['Telugu', /\p{Script=Telugu}/u],
    ['Kannada', /\p{Script=Kannada}/u],
    ['Malayalam', /\p{Script=Malayalam}/u],
    ['Gujarati', /\p{Script=Gujarati}/u],
    ['Gurmukhi', /\p{Script=Gurmukhi}/u],
    ['Oriya', /\p{Script=Oriya}/u],
    ['Arabic', /\p{Script=Arabic}/u],
  ];
  for (const ch of text) {
    for (const [name, re] of scripts) {
      if (re.test(ch)) {
        counts.set(name, (counts.get(name) ?? 0) + 1);
        break;
      }
    }
  }
  let best = 'Unknown';
  let bestCount = 0;
  for (const [name, count] of counts) {
    if (count > bestCount) {
      best = name;
      bestCount = count;
    }
  }
  return best;
}
