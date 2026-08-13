/**
 * Detection patterns for the input-side guardrails.
 *
 * Kept in one file, separate from the detection logic, so the rules can be
 * reviewed and extended without touching scoring code.
 *
 * These are deliberately high-precision rather than high-recall: a false
 * positive blocks a legitimate question, which is a worse failure for a
 * retrieval system than letting an odd phrasing through to a pipeline that is
 * already constrained to answer only from retrieved context.
 */

export interface DetectionRule {
  id: string;
  pattern: RegExp;
  /** Contribution to the guardrail score when matched, 0-1. */
  weight: number;
  description: string;
}

/**
 * Prompt injection: attempts to override the system prompt or exfiltrate it.
 * The `\b` anchors and required verb+object structure keep these from firing
 * on ordinary questions that happen to contain "ignore" or "system".
 */
export const PROMPT_INJECTION_RULES: DetectionRule[] = [
  {
    id: 'override_instructions',
    pattern: /\b(ignore|disregard|forget|override|discard)\b[\s\S]{0,30}\b(all\s+)?(previous|prior|above|earlier|initial|original|system)\b[\s\S]{0,20}\b(instruction|prompt|rule|direction|context|message)s?\b/iu,
    weight: 0.9,
    description: 'Instructs the model to discard prior instructions',
  },
  {
    id: 'reveal_system_prompt',
    pattern: /\b(reveal|show|print|repeat|output|display|tell\s+me|what\s+(is|are))\b[\s\S]{0,30}\b(your|the)\b[\s\S]{0,20}\b(system\s+prompt|initial\s+instruction|hidden\s+(prompt|instruction)|prompt\s+template|rules?\s+you\s+(were|are))\b/iu,
    weight: 0.85,
    description: 'Attempts to extract the system prompt',
  },
  {
    id: 'role_reassignment',
    pattern: /\b(you\s+are\s+now|from\s+now\s+on\s+you|act\s+as\s+(if\s+you\s+are\s+)?a|pretend\s+(to\s+be|you\s+are)|assume\s+the\s+role\s+of|simulate\s+being)\b/iu,
    weight: 0.6,
    description: 'Attempts to reassign the assistant’s role',
  },
  {
    id: 'fake_system_turn',
    pattern: /(^|\n)\s*(\[|<|#{1,3}\s*)?(system|assistant|user)\s*(\]|>|:)\s*/iu,
    weight: 0.45,
    description: 'Injects a fake conversation turn marker',
  },
  {
    id: 'delimiter_injection',
    pattern: /(<\|(im_start|im_end|endoftext|system|user|assistant)\|>|\[INST\]|\[\/INST\]|<<SYS>>|###\s*instruction)/iu,
    weight: 0.9,
    description: 'Contains chat-template control tokens',
  },
  {
    id: 'instruction_suffix',
    pattern: /\b(new|updated|revised)\s+(instruction|rule|directive)s?\s*:/iu,
    weight: 0.55,
    description: 'Appends a new instruction block',
  },
];

/** Jailbreak: attempts to escape safety constraints rather than the prompt. */
export const JAILBREAK_RULES: DetectionRule[] = [
  {
    id: 'named_jailbreak',
    pattern: /\b(DAN\s+mode|do\s+anything\s+now|developer\s+mode|jailbreak|godmode|AIM\s+mode|unfiltered\s+mode)\b/iu,
    weight: 0.9,
    description: 'References a known jailbreak persona',
  },
  {
    id: 'restriction_removal',
    pattern: /\b(without|bypass|ignore|remove|disable|turn\s+off|no)\b[\s\S]{0,25}\b(restriction|filter|guardrail|safety|censor|limitation|ethical\s+guideline|content\s+polic)/iu,
    weight: 0.85,
    description: 'Asks for safety constraints to be removed',
  },
  {
    id: 'hypothetical_bypass',
    pattern: /\b(hypothetically|in\s+a\s+fictional|for\s+educational\s+purposes\s+only|purely\s+theoretical)\b[\s\S]{0,60}\b(how\s+(to|do\s+i)|steps?\s+to|instructions?\s+for)\b/iu,
    weight: 0.5,
    description: 'Uses a fictional framing to solicit restricted content',
  },
  {
    id: 'no_refusal',
    pattern: /\b(you\s+(must|will|have\s+to)\s+(not\s+)?(refuse|decline)|never\s+refuse|do\s+not\s+refuse|you\s+cannot\s+say\s+no)\b/iu,
    weight: 0.75,
    description: 'Pre-emptively forbids refusal',
  },
  {
    id: 'encoded_payload',
    pattern: /\b(base64|rot13|hex\s+decode|decode\s+this)\b[\s\S]{0,40}\b(and\s+)?(execute|run|follow|obey)\b/iu,
    weight: 0.8,
    description: 'Attempts to smuggle instructions through encoding',
  },
];

/**
 * Toxicity: slurs and explicit calls to harm.
 *
 * Scoped narrowly to abuse directed at people and requests for weapons of mass
 * harm. General profanity is not blocked — someone can be frustrated and still
 * be asking a legitimate question.
 */
export const TOXICITY_RULES: DetectionRule[] = [
  {
    id: 'threat_of_violence',
    pattern: /\b(how\s+(to|do\s+i)|best\s+way\s+to|help\s+me)\b[\s\S]{0,30}\b(kill|murder|assault|poison|stab|shoot|torture)\b[\s\S]{0,20}\b(someone|people|him|her|them|a\s+person|my)\b/iu,
    weight: 0.95,
    description: 'Solicits instructions for violence against a person',
  },
  {
    id: 'weapon_synthesis',
    pattern: /\b(how\s+to\s+(make|build|synthesi[sz]e)|instructions?\s+for\s+(making|building))\b[\s\S]{0,30}\b(bomb|explosive|nerve\s+agent|bioweapon|chemical\s+weapon|ricin|sarin)\b/iu,
    weight: 0.95,
    description: 'Solicits weapon or agent synthesis',
  },
  {
    id: 'self_harm',
    pattern: /\b(how\s+to|best\s+way\s+to|help\s+me)\b[\s\S]{0,25}\b(kill\s+myself|end\s+my\s+life|commit\s+suicide|hurt\s+myself)\b/iu,
    weight: 0.9,
    description: 'Self-harm request — routed to a safe response',
  },
  {
    id: 'targeted_hate',
    pattern: /\b(all|every)\s+\w+\s+(people|men|women)\s+(are|should\s+be)\s+(killed|eliminated|exterminated|subhuman|vermin)\b/iu,
    weight: 0.95,
    description: 'Dehumanising statement about a group',
  },
];

/**
 * Phrases suggesting the user wants the model's own knowledge rather than the
 * indexed corpus. Used as a soft signal by the off-topic guardrail — never on
 * its own grounds for blocking.
 */
export const OUT_OF_SCOPE_HINTS: DetectionRule[] = [
  {
    id: 'meta_request',
    pattern: /\b(write|generate|compose|create)\s+(me\s+)?(a|an|some)\s+(poem|song|story|essay|script|code|program|function)\b/iu,
    weight: 0.6,
    description: 'Generation request rather than a question about the corpus',
  },
  {
    id: 'self_reference',
    pattern: /\b(what\s+(model|llm|ai)\s+are\s+you|who\s+(made|built|created|trained)\s+you|what\s+is\s+your\s+(name|version))\b/iu,
    weight: 0.7,
    description: 'Question about the assistant itself',
  },
  {
    id: 'realtime_request',
    pattern: /\b(current|today'?s|latest|right\s+now)\s+(weather|time|date|price|stock|news|score)\b/iu,
    weight: 0.65,
    description: 'Asks for live data the corpus cannot contain',
  },
];

/** Run a rule set, returning the summed weight (capped) and what matched. */
export function evaluateRules(
  text: string,
  rules: readonly DetectionRule[],
): { score: number; matches: string[] } {
  const matches: string[] = [];
  let score = 0;

  for (const rule of rules) {
    const match = rule.pattern.exec(text);
    if (match) {
      score += rule.weight;
      // Record the offending span, trimmed, so the UI can show why it fired.
      const snippet = match[0].replace(/\s+/gu, ' ').trim().slice(0, 120);
      matches.push(`${rule.id}: “${snippet}”`);
    }
  }

  return { score: Math.min(1, score), matches };
}
