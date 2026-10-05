import type { BodyState } from './types';

export async function getBodyState(): Promise<BodyState> {
  const response = await fetch('/v1/body/state', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error('AISHA Core returned HTTP ' + response.status);
  }
  return response.json() as Promise<BodyState>;
}
