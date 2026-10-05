export interface EmbodimentState {
  sequence: number;
  activity: 'idle' | 'listening' | 'thinking' | 'speaking';
  expression: 'neutral' | 'attentive' | 'focused' | 'engaged';
  intensity: number;
  affect: 'neutral' | 'warm' | 'amused' | 'curious' | 'concerned' | 'surprised';
  affect_intensity: number;
  updated_at: string;
}

export interface PerceptionSummary {
  person_present: boolean;
  person_count: number;
  gaze_toward_camera: boolean;
  head_approximately_frontal: boolean;
  head_frontal_score: number | null;
  primary_person_x: number | null;
  primary_person_y: number | null;
  visible_objects: string[];
}

export interface RuntimeHealth {
  status: string;
  profile: string;
  provider: string;
  model: string;
}
