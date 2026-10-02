import type { AISHAEvent, AutomaticMemoryStatus, BeliefVersion, CameraStatus, EmbodimentState, LearnedBelief, MemoryRecord, ModelRun, PerceptionSummary, RuntimeHealth, StoredMessage } from './types';

async function json<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error('AISHA Core returned HTTP ' + response.status);
  return response.json() as Promise<T>;
}

export function getHealth(): Promise<RuntimeHealth> {
  return json<RuntimeHealth>('/v1/health');
}

export function getEmbodimentState(): Promise<EmbodimentState> {
  return json<EmbodimentState>('/v1/embodiment/state');
}

export function getCameraStatus(): Promise<CameraStatus> {
  return json<CameraStatus>('/v1/perception/camera');
}

export function getPerceptionSummary(): Promise<PerceptionSummary> {
  return json<PerceptionSummary>('/v1/perception/summary');
}

export function setCameraEnabled(enabled: boolean): Promise<CameraStatus> {
  return json<CameraStatus>('/v1/perception/camera', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
}

export function setEmbodimentAffect(
  affect: EmbodimentState['affect'],
  intensity: number,
): Promise<EmbodimentState> {
  return json<EmbodimentState>('/v1/embodiment/affect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ affect, intensity }),
  });
}

export async function createSession(): Promise<string> {
  const created = await json<{ session_id: string }>('/v1/sessions', {
    method: 'POST',
  });
  return created.session_id;
}

export interface SessionSettings {
  session_id: string;
  memory_mode: 'normal' | 'test';
}

export function getSessionSettings(sessionId: string): Promise<SessionSettings> {
  return json<SessionSettings>('/v1/sessions/' + encodeURIComponent(sessionId));
}

export function setSessionMemoryMode(
  sessionId: string,
  memoryMode: 'normal' | 'test',
): Promise<SessionSettings> {
  return json<SessionSettings>('/v1/sessions/' + encodeURIComponent(sessionId), {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ memory_mode: memoryMode }),
  });
}

export function getMessages(sessionId: string): Promise<StoredMessage[]> {
  return json<StoredMessage[]>('/v1/sessions/' + encodeURIComponent(sessionId) + '/messages');
}

export function getEvents(sessionId: string): Promise<AISHAEvent[]> {
  return json<AISHAEvent[]>('/v1/sessions/' + encodeURIComponent(sessionId) + '/events?limit=2000');
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

export function getLearnedBeliefs(): Promise<LearnedBelief[]> {
  return json<LearnedBelief[]>('/v1/memory/beliefs');
}

export function getBeliefVersions(id: string): Promise<BeliefVersion[]> {
  return json<BeliefVersion[]>('/v1/memory/beliefs/' + encodeURIComponent(id) + '/versions');
}

export function getAutomaticMemoryStatus(): Promise<AutomaticMemoryStatus> {
  return json<AutomaticMemoryStatus>('/v1/memory/status');
}

export async function forgetLearnedBelief(id: string): Promise<void> {
  const response = await fetch('/v1/memory/beliefs/' + encodeURIComponent(id), {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Could not forget learned belief: HTTP ' + response.status);
}
