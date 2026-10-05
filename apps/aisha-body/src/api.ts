import type { BodyState } from './types';

export async function getBodyState(): Promise<BodyState> {
  const response = await fetch('/v1/body/state', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error('AISHA Core returned HTTP ' + response.status);
  }
  return response.json() as Promise<BodyState>;
}

export function connectBodyStream(
  onState: (state: BodyState) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  const source = new EventSource('/v1/body/stream');

  source.onopen = () => onConnection(true);
  source.onmessage = (event) => {
    try {
      onState(JSON.parse(event.data) as BodyState);
      onConnection(true);
    } catch {
      onConnection(false);
    }
  };
  source.onerror = () => onConnection(false);

  return () => source.close();
}
