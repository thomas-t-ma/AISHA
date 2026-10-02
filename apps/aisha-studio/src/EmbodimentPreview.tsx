import type { EmbodimentState } from './types';

interface EmbodimentPreviewProps {
  state: EmbodimentState;
  compact?: boolean;
}

const LABELS: Record<EmbodimentState['activity'], string> = {
  idle: 'At rest',
  listening: 'Listening',
  thinking: 'Thinking',
  speaking: 'Speaking',
};

export default function EmbodimentPreview({
  state,
  compact = false,
}: EmbodimentPreviewProps) {
  return (
    <div
      className={
        'embodiment-preview activity-' + state.activity
        + ' expression-' + state.expression
        + ' affect-' + state.affect
        + (compact ? ' compact' : '')
      }
      aria-label={'AISHA is ' + state.activity}
      title={
        'Embodiment state: ' + state.activity
        + ' / ' + state.expression
        + ' · affect ' + state.affect
        + ' ' + state.affect_intensity.toFixed(2)
        + ' · activity intensity ' + state.intensity.toFixed(2)
      }
    >
      <div className="embodiment-stage" aria-hidden="true">
        <div className="embodiment-halo" />
        <div className="embodiment-head">
          <div className="embodiment-brow brow-left" />
          <div className="embodiment-brow brow-right" />
          <div className="embodiment-eyes">
            <span className="embodiment-eye eye-left"><i /></span>
            <span className="embodiment-eye eye-right"><i /></span>
          </div>
          <div className="embodiment-mouth"><span /></div>
        </div>
      </div>

      {!compact && (
        <div className="embodiment-copy">
          <span className="eyebrow">EMBODIMENT PREVIEW</span>
          <strong>{LABELS[state.activity]}</strong>
          <small>
            {state.expression} · {Math.round(state.intensity * 100)}% activity
            {' · '}{state.affect} · {Math.round(state.affect_intensity * 100)}% affect
          </small>
        </div>
      )}
      {compact && (
        <div className="embodiment-compact-copy">
          <strong>{LABELS[state.activity]}</strong>
          <span>{state.expression} · {state.affect}</span>
        </div>
      )}
    </div>
  );
}
