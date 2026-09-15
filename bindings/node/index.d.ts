// Type definitions for rwkv-router (Node.js native binding).

export interface RoutingDecision {
  /** Classification tier: "R0" | "R1" | "R2" | "R3" */
  route: string
  confidence: number
  /** Per-tier probabilities [R0, R1, R2, R3] */
  probabilities: [number, number, number, number]
  margin: number
  sticky_applied: boolean
  safety_applied: boolean
  /** Decision source: "Classifier" | "Rules" | "TrivialAck" | "Sticky" | "Fallback" */
  source: string
}

export interface GenerationOutput {
  text: string
  input_tokens: number
  output_tokens: number
  stop_hit: string | null
  stopped_by_eos: boolean
}

export interface CaptureStats {
  total: number
  labeled: number
  /** Per-tier labelled counts keyed "R0".."R3" */
  [tier: string]: unknown
}

export interface EvolveResult {
  [key: string]: unknown
}

export class RouterSession {
  constructor()

  /** One routing decision (sticky context keyed by sessionId). */
  route(
    input: string,
    summary?: string | null,
    sessionId?: string | null,
    turnIndex?: number | null,
  ): RoutingDecision

  /** Stateless decision (no sticky table, no capture) — testing entry. */
  routePreview(input: string, summary?: string | null): RoutingDecision

  /** Attaches the built-in RWKV classifier (resident 0.1B + MLP head). */
  loadClassifier(
    model: string,
    vocab: string,
    head: string,
    timeoutMs?: number | null,
  ): void

  /** Maps a tier to a local RWKV model (loads lazily, LRU-pooled). */
  attachGeneration(
    tier: 'R0' | 'R1' | 'R2' | 'R3' | string,
    vocab: string,
    model: string,
    maxLoaded?: number | null,
  ): void

  /** Generates with the tier's attached model. */
  generate(
    tier: 'R0' | 'R1' | 'R2' | 'R3' | string,
    prompt: string,
    maxTokens?: number | null,
    temperature?: number | null,
    topP?: number | null,
    topK?: number | null,
    presencePenalty?: number | null,
    frequencyPenalty?: number | null,
    stop?: string[] | null,
  ): GenerationOutput

  /** Enables the self-evolution loop (capture → label → fine-tune → gate). */
  configureEvolution(
    dataDir: string,
    headPath?: string | null,
    packsDir?: string | null,
    captureLimit?: number | null,
    minLabeledForEvolve?: number | null,
    autoEvolveStep?: number | null,
  ): void

  /** Capture-store statistics (throws when evolution is unconfigured). */
  captureStats(): CaptureStats

  /** Captured samples for labeling flows (throws when unconfigured). */
  captureList(offset?: number | null, limit?: number | null): unknown[]

  /** Labels (tier "R0".."R3") or clears (null) the sample at idx. */
  captureLabel(idx: number, tier?: 'R0' | 'R1' | 'R2' | 'R3' | null): void

  /** One evolution cycle (blocking). */
  evolve(): EvolveResult
}
