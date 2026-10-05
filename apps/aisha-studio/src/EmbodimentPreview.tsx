import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';

import { toRendererInput } from './renderer';
import type { EmbodimentState, PerceptionSummary } from './types';

interface EmbodimentPreviewProps {
  state: EmbodimentState;
  perception?: PerceptionSummary | null;
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

function gazeBias(
  activity: EmbodimentState['activity'],
  attentionTarget: 'viewer' | 'ambient',
  viewerPosition: { x: number; y: number } | null,
  compact: boolean,
): Point {
  if (attentionTarget === 'viewer') {
    if (!viewerPosition) return { x: 0, y: 0 };
    const horizontalRange = compact ? 1.8 : 3.2;
    const verticalRange = compact ? 1.25 : 2.2;
    return {
      // Webcam image-space X is mirrored into screen-space so the avatar
      // visually looks toward the viewer's physical side of the display.
      x: (0.5 - viewerPosition.x) * horizontalRange * 2,
      y: (viewerPosition.y - 0.5) * verticalRange * 2,
    };
  }
  if (activity === 'thinking') return { x: 1.4, y: -1.7 };
  if (activity === 'listening') return { x: 0, y: -0.3 };
  if (activity === 'speaking') return { x: 0.4, y: 0 };
  return { x: 0, y: 0 };
}

export default function EmbodimentPreview({
  state,
  perception = null,
  compact = false,
}: EmbodimentPreviewProps) {
  const renderer = toRendererInput(state, perception);
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
    const bias = gazeBias(
      renderer.activity,
      renderer.attentionTarget,
      renderer.viewerPosition,
      compact,
    );
    const amplitude = renderer.attentionTarget === 'viewer'
      ? (compact ? 0.35 : 0.55)
      : (compact ? 1.25 : 2.15);

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
  }, [
    compact,
    renderer.activity,
    renderer.attentionTarget,
    renderer.viewerPosition?.x,
    renderer.viewerPosition?.y,
  ]);

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
        'embodiment-preview activity-' + renderer.activity
        + ' expression-' + renderer.activityIntent
        + ' affect-' + renderer.affect
        + (blinking ? ' is-blinking' : '')
        + (compact ? ' compact' : '')
      }
      style={proceduralStyle}
      aria-label={'AISHA is ' + renderer.activity}
      title={
        'Embodiment state: ' + renderer.activity
        + ' / ' + renderer.activityIntent
        + ' · affect ' + renderer.affect
        + ' ' + renderer.affectIntensity.toFixed(2)
        + ' · activity intensity ' + renderer.activityIntensity.toFixed(2)
        + ' · attention ' + renderer.attentionTarget
        + (renderer.viewerPosition
          ? ' @ ' + renderer.viewerPosition.x.toFixed(2)
            + ',' + renderer.viewerPosition.y.toFixed(2)
          : '')
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
          <strong>{LABELS[renderer.activity]}</strong>
          <small>
            {renderer.activityIntent} · {Math.round(renderer.activityIntensity * 100)}% activity
            {' · '}{renderer.affect} · {Math.round(renderer.affectIntensity * 100)}% affect
          </small>
        </div>
      )}
      {compact && (
        <div className="embodiment-compact-copy">
          <strong>{LABELS[renderer.activity]}</strong>
          <span>{renderer.activityIntent} · {renderer.affect}</span>
        </div>
      )}
    </div>
  );
}
