export interface RuntimeHealth {
  status: string;
  profile: string;
  provider: string;
  model: string;
  persona_version: string;
  warmup?: {
    preloaded?: boolean;
    wall_clock_ms?: number | null;
    error?: string;
  };
}

export interface StoredMessage {
  message_id: string;
  role: string;
  text: string;
  status: string;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  status: 'committed' | 'streaming' | 'cancelled' | 'failed';
  turnId?: string;
}

export interface AISHAEvent {
  type: string;
  session_id?: string;
  turn_id?: string | null;
  timestamp?: string;
  payload: Record<string, unknown>;
}

export interface ModelRun {
  run_id: string;
  turn_id: string;
  provider: string;
  model: string;
  status: string;
  first_token_ms: number | null;
  total_ms: number | null;
  output_chars: number;
  backend_metrics: Record<string, unknown>;
}

export interface TurnLatency {
  firstTokenMs: number | null;
  totalMs: number | null;
}

export interface MemoryRecord {
  memory_id: string;
  text: string;
  source_type: string;
  source_session_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface AutomaticMemoryStatus {
  enabled: boolean;
  active_reflections: number;
  last_error: string | null;
}

export interface LearnedBelief {
  belief_id: string;
  topic_key: string;
  text: string;
  epistemic_status: 'stated' | 'inferred' | 'uncertain';
  source_quote: string;
  source_session_id: string;
  source_turn_id: string;
  open_question: string | null;
  revision: number;
  updated_at: string;
}

export interface BeliefVersion {
  version_id: string;
  belief_id: string;
  revision: number;
  text: string;
  epistemic_status: string;
  source_quote: string;
  source_session_id: string;
  source_turn_id: string;
  open_question: string | null;
  recorded_at: string;
}
