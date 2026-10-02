import type { EmbodimentState, PerceptionSummary } from './types';

export interface RendererInput {
  sequence: number;
  activity: EmbodimentState['activity'];
  activityIntent: EmbodimentState['expression'];
  activityIntensity: number;
  affect: EmbodimentState['affect'];
  affectIntensity: number;
  attentionTarget: 'viewer' | 'ambient';
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
    attentionTarget: (
      perception?.person_present && perception.gaze_toward_camera
        ? 'viewer'
        : 'ambient'
    ),
  };
}
