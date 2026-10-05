import { useEffect, useMemo, useState } from 'react';

import { getEmbodiment, getHealth, getPerception } from './api';
import type { EmbodimentState, PerceptionSummary, RuntimeHealth } from './types';

const DEFAULT_STATE: EmbodimentState = {
  sequence: 0,
  activity: 'idle',
  expression: 'neutral',
  intensity: 0.2,
  affect: 'neutral',
  affect_intensity: 0,
  updated_at: '',
};

const EMPTY_PERCEPTION: PerceptionSummary = {
  person_present: false,
  person_count: 0,
  gaze_toward_camera: false,
  head_approximately_frontal: false,
  head_frontal_score: null,
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
  const [health, setHealth] = useState<RuntimeHealth | null>(null);
  const [online, setOnline] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const refresh = async () => {
      const results = await Promise.allSettled([
        getEmbodiment(),
        getPerception(),
        getHealth(),
      ]);
      if (cancelled) return;

      if (results[0].status === 'fulfilled') setState(results[0].value);
      if (results[1].status === 'fulfilled') setPerception(results[1].value);
      if (results[2].status === 'fulfilled') setHealth(results[2].value);
      setOnline(results.every((result) => result.status === 'fulfilled'));
    };

    void refresh();
    const timer = window.setInterval(() => void refresh(), 250);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
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
  } as React.CSSProperties;

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
            ? (health?.profile ?? 'local') + (perception.person_present ? ' · viewer present' : '')
            : 'Core unavailable'}
        </small>
      </div>
    </main>
  );
}
