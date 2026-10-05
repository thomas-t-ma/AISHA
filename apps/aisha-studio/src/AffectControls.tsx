import type { EmbodimentState } from './types';

interface AffectControlsProps {
  state: EmbodimentState;
  disabled?: boolean;
  onChange: (
    affect: EmbodimentState['affect'],
    intensity: number,
  ) => void;
}

const AFFECTS: EmbodimentState['affect'][] = [
  'neutral',
  'warm',
  'amused',
  'curious',
  'concerned',
  'surprised',
];

export default function AffectControls({
  state,
  disabled = false,
  onChange,
}: AffectControlsProps) {
  const intensity = Math.round(state.affect_intensity * 100);

  return (
    <div className="affect-controls">
      <div className="affect-control-head">
        <span className="eyebrow">AFFECT DEBUG · 4S PULSE</span>
        <span>{state.affect} · {intensity}%</span>
      </div>
      <div className="affect-buttons" role="group" aria-label="Affect preview">
        {AFFECTS.map((affect) => (
          <button
            key={affect}
            type="button"
            className={state.affect === affect ? 'selected' : ''}
            disabled={disabled}
            onClick={() => onChange(
              affect,
              affect === 'neutral' ? 0 : Math.max(state.affect_intensity, 0.5),
            )}
          >
            {affect}
          </button>
        ))}
      </div>
      <label className="affect-intensity">
        <span>Intensity</span>
        <input
          type="range"
          min="0"
          max="100"
          step="5"
          value={intensity}
          disabled={disabled || state.affect === 'neutral'}
          onChange={(event) => onChange(
            state.affect,
            Number(event.target.value) / 100,
          )}
        />
      </label>
    </div>
  );
}
