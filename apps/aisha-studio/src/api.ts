import type { AISHAEvent, ModelRun, RuntimeHealth, StoredMessage, TranscriptionResult, VoiceStatus } from './types';

async function json<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error('AISHA Core returned HTTP ' + response.status);
  return response.json() as Promise<T>;
}

export function getHealth(): Promise<RuntimeHealth> {
  return json<RuntimeHealth>('/v1/health');
}

export async function createSession(): Promise<string> {
  const created = await json<{ session_id: string }>('/v1/sessions', {
    method: 'POST',
  });
  return created.session_id;
}

export function getMessages(sessionId: string): Promise<StoredMessage[]> {
  return json<StoredMessage[]>('/v1/sessions/' + encodeURIComponent(sessionId) + '/messages');
}

export function getEvents(sessionId: string): Promise<AISHAEvent[]> {
  return json<AISHAEvent[]>('/v1/sessions/' + encodeURIComponent(sessionId) + '/events?limit=1000');
}

export function getRuns(sessionId: string): Promise<ModelRun[]> {
  return json<ModelRun[]>('/v1/sessions/' + encodeURIComponent(sessionId) + '/model-runs');
}

export function getVoiceStatus(): Promise<VoiceStatus> {
  return json<VoiceStatus>('/v1/voice/status');
}

async function voiceError(response: Response): Promise<Error> {
  const body = await response.json().catch(() => ({})) as { detail?: string };
  return new Error(body.detail ?? ('Voice service returned HTTP ' + response.status));
}

export async function transcribeAudio(blob: Blob): Promise<TranscriptionResult> {
  const type = blob.type.split(';')[0] || 'audio/webm';
  const extension = type === 'audio/mp4' ? 'mp4'
    : type === 'audio/ogg' ? 'ogg' : type.includes('wav') ? 'wav' : 'webm';
  const form = new FormData();
  form.append('file', new File([blob], 'speech.' + extension, { type }));
  const response = await fetch('/v1/voice/transcribe', { method: 'POST', body: form });
  if (!response.ok) throw await voiceError(response);
  return response.json() as Promise<TranscriptionResult>;
}

export async function synthesize(
  text: string,
  signal?: AbortSignal,
): Promise<{ audio: Blob; durationMs: number | null }> {
  const response = await fetch('/v1/voice/synthesize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    signal,
  });
  if (!response.ok) throw await voiceError(response);
  const header = response.headers.get('x-aisha-tts-ms');
  return {
    audio: await response.blob(),
    durationMs: header === null ? null : Number(header),
  };
}
