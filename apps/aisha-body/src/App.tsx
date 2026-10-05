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

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export default function App() {
  const [state, setState] = useState(DEFAULT_STATE);
  const [perception, setPerception] = useState(EMPTY_PERCEPTION);
  const [online, setOnline] = useState(false);

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

  const gaze = useMemo(() => {
    if (
      !perception.person_present
      || perception.primary_person_x == null
      || perception.primary_person_y == null
    ) {
      return { x: 0, y: 0 };
    }
    return {
      x: clamp((0.5 - perception.primary_person_x) * 18, -7, 7),
      y: clamp((perception.primary_person_y - 0.5) * 12, -5, 5),
    };
  }, [
    perception.person_present,
    perception.primary_person_x,
    perception.primary_person_y,
  ]);

  const style = {
    '--gaze-x': gaze.x.toFixed(2) + 'px',
    '--gaze-y': gaze.y.toFixed(2) + 'px',
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
