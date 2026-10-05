import { useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';

import { connectBodyStream, getBodyState } from './api';
import type { BodyPerception, EmbodimentState } from './types';

const DEFAULT_STATE: EmbodimentState = {
  sequence: 0,
  activity: 'idle',
  expression: 'neutral',
  intensity: 0.2,
  affect: 'neutral',
  affect_intensity: 0,
  affect_expires_at: null,
  updated_at: '',
};

const EMPTY_PERCEPTION: BodyPerception = {
  person_present: false,
  person_count: 0,
  gaze_toward_camera: false,
  head_approximately_frontal: false,
  primary_person_x: null,
  primary_person_y: null,
  visible_objects: [],
};

interface Point {
  x: number;
  y: number;
}

interface HeadMotion extends Point {
  tilt: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function ambientGaze(
  activity: EmbodimentState['activity'],
  affect: EmbodimentState['affect'],
): Point {
  if (affect === 'curious') return { x: -0.7, y: -0.9 };
  if (affect === 'concerned') return { x: 0, y: 0.35 };
  if (activity === 'thinking') return { x: 1.5, y: -1.45 };
  if (activity === 'listening') return { x: 0, y: -0.25 };
  if (activity === 'speaking') return { x: 0.35, y: 0 };
  return { x: 0, y: 0 };
}

export default function App() {
  const [state, setState] = useState(DEFAULT_STATE);
  const [perception, setPerception] = useState(EMPTY_PERCEPTION);
  const [online, setOnline] = useState(false);
  const [blinking, setBlinking] = useState(false);
  const [saccade, setSaccade] = useState<Point>({ x: 0, y: 0 });
  const [head, setHead] = useState<HeadMotion>({ x: 0, y: 0, tilt: 0 });

  useEffect(() => {
    let cancelled = false;

    void getBodyState()
      .then((bodyState) => {
        if (cancelled) return;
        setState(bodyState.embodiment);
        setPerception(bodyState.perception);
      })
      .catch(() => {
        if (!cancelled) setOnline(false);
      });

    const disconnect = connectBodyStream(
      (bodyState) => {
        if (cancelled) return;
        setState(bodyState.embodiment);
        setPerception(bodyState.perception);
      },
      (connected) => {
        if (!cancelled) setOnline(connected);
      },
    );

    return () => {
      cancelled = true;
      disconnect();
    };
  }, []);

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
      }, randomBetween(2400, 6800));
    };

    scheduleBlink();
    return () => {
      cancelled = true;
      if (blinkTimer !== undefined) window.clearTimeout(blinkTimer);
      if (reopenTimer !== undefined) window.clearTimeout(reopenTimer);
    };
  }, []);

  const gazeTarget = useMemo(() => {
    if (
      perception.person_present
      && perception.primary_person_x != null
      && perception.primary_person_y != null
    ) {
      return {
        x: clamp((0.5 - perception.primary_person_x) * 18, -7, 7),
        y: clamp((perception.primary_person_y - 0.5) * 12, -5, 5),
      };
    }
    return ambientGaze(state.activity, state.affect);
  }, [
    perception.person_present,
    perception.primary_person_x,
    perception.primary_person_y,
    state.activity,
    state.affect,
  ]);

  useEffect(() => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (reducedMotion.matches) {
      setSaccade({ x: 0, y: 0 });
      return;
    }

    let timer: number | undefined;
    let cancelled = false;
    const amplitude = perception.person_present
      ? 0.45
      : state.activity === 'thinking'
        ? 1.15
        : 1.55;

    const schedule = () => {
      timer = window.setTimeout(() => {
        if (cancelled) return;
        setSaccade({
          x: randomBetween(-amplitude, amplitude),
          y: randomBetween(-amplitude * 0.62, amplitude * 0.62),
        });
        schedule();
      }, randomBetween(620, 1850));
    };

    setSaccade({ x: 0, y: 0 });
    schedule();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [perception.person_present, state.activity]);

  useEffect(() => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (reducedMotion.matches) {
      setHead({ x: 0, y: 0, tilt: 0 });
      return;
    }

    let timer: number | undefined;
    let cancelled = false;
    const amplitude = perception.person_present ? 0.55 : 1;
    const affectTilt = state.affect === 'curious'
      ? 1.1 * state.affect_intensity
      : state.affect === 'amused'
        ? -0.55 * state.affect_intensity
        : state.affect === 'concerned'
          ? 0.4 * state.affect_intensity
          : 0;

    const schedule = () => {
      timer = window.setTimeout(() => {
        if (cancelled) return;
        setHead({
          x: randomBetween(-1.4, 1.4) * amplitude,
          y: randomBetween(-0.85, 0.85) * amplitude,
          tilt: affectTilt + randomBetween(-1.05, 1.05) * amplitude,
        });
        schedule();
      }, randomBetween(2800, 5600));
    };

    schedule();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [
    perception.person_present,
    state.affect,
    state.affect_intensity,
  ]);

  const gaze = {
    x: clamp(gazeTarget.x + saccade.x, -7.5, 7.5),
    y: clamp(gazeTarget.y + saccade.y, -5.5, 5.5),
  };

  const style = {
    '--gaze-x': gaze.x.toFixed(2) + 'px',
    '--gaze-y': gaze.y.toFixed(2) + 'px',
    '--head-x': head.x.toFixed(2) + 'px',
    '--head-y': head.y.toFixed(2) + 'px',
    '--head-tilt': head.tilt.toFixed(2) + 'deg',
    '--activity': state.intensity.toFixed(3),
    '--affect': state.affect_intensity.toFixed(3),
  } as CSSProperties;

  return (
    <main
      className={[
        'body-shell',
        'activity-' + state.activity,
        'affect-' + state.affect,
        perception.person_present ? 'viewer-present' : '',
        blinking ? 'is-blinking' : '',
      ].join(' ')}
      style={style}
    >
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <section className="presence" aria-label={'AISHA is ' + state.activity}>
        <div className="aura outer-aura" />
        <div className="aura inner-aura" />
        <div className="face">
          <div className="brow brow-left" />
          <div className="brow brow-right" />
          <div className="eyes">
            <span className="eye"><i /></span>
            <span className="eye"><i /></span>
          </div>
          <div className="mouth"><i /></div>
        </div>
      </section>

      <div className="status">
        <div>
          <span className={'dot ' + (online ? 'online' : 'offline')} />
          <strong>AISHA</strong>
          <span>{state.activity}</span>
          <span>·</span>
          <span>{state.affect}</span>
        </div>
        <small>
          {online
            ? 'local renderer' + (perception.person_present ? ' · viewer present' : '')
            : 'Core unavailable'}
        </small>
      </div>
    </main>
  );
}
