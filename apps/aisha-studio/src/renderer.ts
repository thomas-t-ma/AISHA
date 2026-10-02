import type { EmbodimentState } from './types';

export interface RendererInput {
  sequence: number;
  activity: EmbodimentState['activity'];
  activityIntent: EmbodimentState['expression'];
  activityIntensity: number;
  affect: EmbodimentState['affect'];
  affectIntensity: number;
}

export interface EmbodimentRendererAdapter {
  readonly name: string;
  update(input: RendererInput): void;
  dispose(): void;
}

export function toRendererInput(state: EmbodimentState): RendererInput {
  return {
    sequence: state.sequence,
    activity: state.activity,
    activityIntent: state.expression,
    activityIntensity: state.intensity,
    affect: state.affect,
    affectIntensity: state.affect_intensity,
  };
}
