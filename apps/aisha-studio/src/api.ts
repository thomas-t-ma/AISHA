import type { AISHAEvent, MemoryRecord, ModelRun, RuntimeHealth, StoredMessage } from './types';

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

export function getMemories(): Promise<MemoryRecord[]> {
  return json<MemoryRecord[]>('/v1/memories');
}

export function addMemory(text: string, sourceSessionId: string): Promise<MemoryRecord> {
  return json<MemoryRecord>('/v1/memories', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, source_session_id: sourceSessionId || null }),
  });
}

export function editMemory(memoryId: string, text: string): Promise<MemoryRecord> {
  return json<MemoryRecord>('/v1/memories/' + encodeURIComponent(memoryId), {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
}

export async function removeMemory(memoryId: string): Promise<void> {
  const response = await fetch('/v1/memories/' + encodeURIComponent(memoryId), {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Cannot remove memory: HTTP ' + response.status);
}
