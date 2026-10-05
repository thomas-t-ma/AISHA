import type { EmbodimentState, PerceptionSummary } from './types';

export interface RendererInput {
  sequence: number;
  activity: EmbodimentState['activity'];
  activityIntent: EmbodimentState['expression'];
  activityIntensity: number;
  affect: EmbodimentState['affect'];
  affectIntensity: number;
  attentionTarget: 'viewer' | 'ambient';
  viewerPosition: { x: number; y: number } | null;
}

export interface EmbodimentRendererAdapter {
  readonly name: string;
  update(input: RendererInput): void;
  dispose(): void;
}

export function toRendererInput(
  state: EmbodimentState,
  perception?: PerceptionSummary | null,
): RendererInput {
  return {
    sequence: state.sequence,
    activity: state.activity,
    activityIntent: state.expression,
    activityIntensity: state.intensity,
    affect: state.affect,
    affectIntensity: state.affect_intensity,
    attentionTarget: perception?.person_present ? 'viewer' : 'ambient',
    viewerPosition: (
      perception?.person_present
      && perception.primary_person_x != null
      && perception.primary_person_y != null
    )
      ? { x: perception.primary_person_x, y: perception.primary_person_y }
      : null,
  };
}
