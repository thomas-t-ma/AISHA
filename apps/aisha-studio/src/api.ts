import type { AISHAEvent, ModelRun, RuntimeHealth, StoredMessage } from './types';

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
