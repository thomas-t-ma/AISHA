import type { EmbodimentState, PerceptionSummary, RuntimeHealth } from './types';

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error('AISHA Core returned HTTP ' + response.status);
  return response.json() as Promise<T>;
}

export function getEmbodiment(): Promise<EmbodimentState> {
  return getJson<EmbodimentState>('/v1/embodiment/state');
}

export function getPerception(): Promise<PerceptionSummary> {
  return getJson<PerceptionSummary>('/v1/perception/summary');
}

export function getHealth(): Promise<RuntimeHealth> {
  return getJson<RuntimeHealth>('/v1/health');
}
