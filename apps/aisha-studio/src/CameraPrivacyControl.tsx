import type { CameraStatus } from './types';

interface CameraPrivacyControlProps {
  status: CameraStatus | null;
  disabled?: boolean;
  onToggle: (enabled: boolean) => void;
}

export default function CameraPrivacyControl({
  status,
  disabled = false,
  onToggle,
}: CameraPrivacyControlProps) {
  const available = status?.available ?? false;
  const enabled = status?.enabled ?? false;

  return (
    <div className={'camera-privacy ' + (enabled ? 'active' : 'inactive')}>
      <div className="camera-privacy-head">
        <div>
          <span className="eyebrow">VISION PRIVACY</span>
          <strong>{enabled ? 'Camera on' : available ? 'Camera off' : 'Camera unavailable'}</strong>
        </div>
        <span className={'camera-state-pill ' + (enabled ? 'on' : 'off')}>
          {enabled ? 'ON' : 'OFF'}
        </span>
      </div>

      <div className="camera-privacy-facts">
        <span>Raw pixels in Core <strong>no</strong></span>
        <span>Capture persisted <strong>no</strong></span>
      </div>

      <button
        type="button"
        disabled={disabled || !available}
        onClick={() => onToggle(!enabled)}
      >
        {enabled ? 'Disable camera' : available ? 'Enable camera' : 'No camera source attached'}
      </button>
    </div>
  );
}
