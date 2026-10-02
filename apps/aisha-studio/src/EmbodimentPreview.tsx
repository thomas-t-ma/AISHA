import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';

import type { EmbodimentState } from './types';

interface EmbodimentPreviewProps {
  state: EmbodimentState;
  compact?: boolean;
}

interface Point {
  x: number;
  y: number;
}

interface HeadPose extends Point {
  tilt: number;
}

const LABELS: Record<EmbodimentState['activity'], string> = {
  idle: 'At rest',
  listening: 'Listening',
  thinking: 'Thinking',
  speaking: 'Speaking',
};

function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function gazeBias(activity: EmbodimentState['activity']): Point {
  if (activity === 'thinking') return { x: 1.4, y: -1.7 };
  if (activity === 'listening') return { x: 0, y: -0.3 };
  if (activity === 'speaking') return { x: 0.4, y: 0 };
  return { x: 0, y: 0 };
}

export default function EmbodimentPreview({
  state,
  compact = false,
}: EmbodimentPreviewProps) {
  const [blinking, setBlinking] = useState(false);
  const [gaze, setGaze] = useState<Point>({ x: 0, y: 0 });
  const [head, setHead] = useState<HeadPose>({ x: 0, y: 0, tilt: 0 });

  useEffect(() => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (reducedMotion.matches) {
      setBlinking(false);
      return;
    }

    let blinkTimer: number | undefined;
    let reopenTimer: number | undefined;
    let cancelled = false;

    const scheduleBlink = () => {
      blinkTimer = window.setTimeout(() => {
        if (cancelled) return;
        setBlinking(true);
        reopenTimer = window.setTimeout(() => {
          if (cancelled) return;
          setBlinking(false);
          scheduleBlink();
        }, randomBetween(90, 145));
      }, randomBetween(2300, 6500));
    };

    scheduleBlink();
    return () => {
      cancelled = true;
      if (blinkTimer !== undefined) window.clearTimeout(blinkTimer);
      if (reopenTimer !== undefined) window.clearTimeout(reopenTimer);
    };
  }, []);

  useEffect(() => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (reducedMotion.matches) {
      setGaze({ x: 0, y: 0 });
      return;
    }

    let gazeTimer: number | undefined;
    let cancelled = false;
    const bias = gazeBias(state.activity);
    const amplitude = compact ? 1.25 : 2.15;

    const scheduleGaze = () => {
      gazeTimer = window.setTimeout(() => {
        if (cancelled) return;
        setGaze({
          x: bias.x + randomBetween(-amplitude, amplitude),
          y: bias.y + randomBetween(-amplitude * 0.65, amplitude * 0.65),
        });
        scheduleGaze();
      }, randomBetween(650, 2100));
    };

    setGaze(bias);
    scheduleGaze();
    return () => {
      cancelled = true;
      if (gazeTimer !== undefined) window.clearTimeout(gazeTimer);
    };
  }, [compact, state.activity]);

  useEffect(() => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (reducedMotion.matches) {
      setHead({ x: 0, y: 0, tilt: 0 });
      return;
    }

    let headTimer: number | undefined;
    let cancelled = false;
    const amplitude = compact ? 0.45 : 1;

    const scheduleHeadPose = () => {
      headTimer = window.setTimeout(() => {
        if (cancelled) return;
        setHead({
          x: randomBetween(-1.4, 1.4) * amplitude,
          y: randomBetween(-0.9, 0.9) * amplitude,
          tilt: randomBetween(-1.15, 1.15) * amplitude,
        });
        scheduleHeadPose();
      }, randomBetween(2800, 5600));
    };

    scheduleHeadPose();
    return () => {
      cancelled = true;
      if (headTimer !== undefined) window.clearTimeout(headTimer);
    };
  }, [compact]);

  const proceduralStyle = {
    '--gaze-x': gaze.x.toFixed(2) + 'px',
    '--gaze-y': gaze.y.toFixed(2) + 'px',
    '--head-x': head.x.toFixed(2) + 'px',
    '--head-y': head.y.toFixed(2) + 'px',
    '--head-tilt': head.tilt.toFixed(2) + 'deg',
  } as CSSProperties;

  return (
    <div
      className={
        'embodiment-preview activity-' + state.activity
        + ' expression-' + state.expression
        + ' affect-' + state.affect
        + (blinking ? ' is-blinking' : '')
        + (compact ? ' compact' : '')
      }
      style={proceduralStyle}
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
