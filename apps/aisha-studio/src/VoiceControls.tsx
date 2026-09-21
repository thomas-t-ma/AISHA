import { useEffect, useRef, useState } from 'react';
import { getVoiceStatus, synthesize, transcribeAudio } from './api';
import type { VoiceStatus } from './types';

interface Props {
  onTranscript: (text: string) => void;
  latestReply: string;
  busy: boolean;
  connected: boolean;
}

const MAX_RECORDING_MS = 30_000;

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function VoiceControls({ onTranscript, latestReply, busy, connected }: Props) {
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [synthesizing, setSynthesizing] = useState(false);
  const [audioReady, setAudioReady] = useState(false);
  const [voiceNotice, setVoiceNotice] = useState('');
  const [lastSttMs, setLastSttMs] = useState<number | null>(null);
  const [lastTtsMs, setLastTtsMs] = useState<number | null>(null);

  const mediaRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const aliveRef = useRef(true);

  async function refreshStatus() {
    try {
      setStatus(await getVoiceStatus());
    } catch {
      setStatus(null);
      setVoiceNotice('Voice service unavailable. Restart AISHA Core.');
    }
  }

  useEffect(() => {
    aliveRef.current = true;
    void refreshStatus();
    return () => {
      aliveRef.current = false;
      requestRef.current?.abort();
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      mediaRef.current?.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      audioRef.current?.pause();
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    };
  }, []);

  function stopPlayback() {
    requestRef.current?.abort();
    requestRef.current = null;
    audioRef.current?.pause();
    if (audioRef.current) audioRef.current.currentTime = 0;
    setSpeaking(false);
    setSynthesizing(false);
  }

  async function startRecording() {
    if (!status?.stt.ready || !connected || busy || recording || transcribing) return;
    stopPlayback();
    setAudioReady(false);
    setVoiceNotice('');
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setVoiceNotice('This browser cannot record microphone audio.');
      return;
    }

    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!aliveRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
        .find((candidate) => MediaRecorder.isTypeSupported(candidate));
      const recorder = mime
        ? new MediaRecorder(stream, { mimeType: mime })
        : new MediaRecorder(stream);
      const chunks: BlobPart[] = [];
      streamRef.current = stream;
      mediaRef.current = recorder;
      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size) chunks.push(event.data);
      };
      recorder.onstop = () => {
        if (timerRef.current !== null) window.clearTimeout(timerRef.current);
        timerRef.current = null;
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        mediaRef.current = null;
        if (!aliveRef.current) return;
        setRecording(false);

        const mediaType = recorder.mimeType.split(';')[0];
        const audio = new Blob(chunks, { type: mediaType });
        if (!audio.size) {
          setVoiceNotice('No audio captured. Try recording again.');
          return;
        }

        setTranscribing(true);
        setVoiceNotice('Transcribing locally…');
        void transcribeAudio(audio)
          .then((result) => {
            if (!aliveRef.current) return;
            setLastSttMs(result.transcribe_ms);
            if (result.text.trim()) {
              onTranscript(result.text.trim());
              setVoiceNotice('Transcript added to the composer. Review it, then Send.');
            } else {
              setVoiceNotice('No clear speech detected. Try again.');
            }
          })
          .catch((error: unknown) => {
            if (aliveRef.current) setVoiceNotice(message(error));
          })
          .finally(() => {
            if (aliveRef.current) setTranscribing(false);
          });
      };
      recorder.start();
      setRecording(true);
      timerRef.current = window.setTimeout(() => {
        if (recorder.state === 'recording') recorder.stop();
      }, MAX_RECORDING_MS);
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop());
      setVoiceNotice('Microphone access failed: ' + message(error));
    }
  }

  function stopRecording() {
    if (mediaRef.current?.state === 'recording') mediaRef.current.stop();
  }

  async function speak() {
    if (!status?.tts.ready || !latestReply || synthesizing || !connected) return;
    stopPlayback();
    setVoiceNotice('');
    setSynthesizing(true);
    const controller = new AbortController();
    requestRef.current = controller;
    try {
      const text = latestReply.slice(0, status.max_speech_text);
      const { audio, durationMs } = await synthesize(text, controller.signal);
      if (!aliveRef.current || controller.signal.aborted) return;
      setLastTtsMs(durationMs);
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
      const url = URL.createObjectURL(audio);
      audioUrlRef.current = url;
      const player = audioRef.current;
      if (player) {
        player.src = url;
        setAudioReady(true);
        try {
          await player.play();
          if (aliveRef.current) setSpeaking(true);
        } catch {
          setVoiceNotice('Audio ready. Press Play below to hear AISHA.');
        }
      }
      if (latestReply.length > text.length) {
        setVoiceNotice('Speaking only the first ' + status.max_speech_text + ' characters.');
      }
    } catch (error) {
      if (aliveRef.current && !controller.signal.aborted) setVoiceNotice(message(error));
    } finally {
      if (aliveRef.current) setSynthesizing(false);
      if (requestRef.current === controller) requestRef.current = null;
    }
  }

  return (
    <div className="voice-wrap">
      <div className="voice-actions">
        <button type="button" className={'voice-button ' + (recording ? 'recording' : '')}
          onClick={recording ? stopRecording : () => void startRecording()}
          disabled={!recording && (!status?.stt.ready || !connected || busy || transcribing)}>
          {recording ? '■ Stop recording' : transcribing ? '◌ Transcribing…' : '● Record'}
        </button>
        <button type="button" className="voice-button"
          onClick={speaking || synthesizing ? stopPlayback : () => void speak()}
          disabled={!speaking && !synthesizing && (!status?.tts.ready || !latestReply || !connected)}>
          {synthesizing ? '■ Cancel speech' : speaking ? '■ Stop audio' : '♫ Speak latest'}
        </button>
        <button type="button" className="voice-refresh" onClick={() => void refreshStatus()}>
          ↻ Voice status
        </button>
        <span className="voice-summary">
          STT {status?.stt.ready ? 'ready' : 'off'} · TTS {status?.tts.ready ? 'ready' : 'off'}
        </span>
      </div>
      <audio ref={audioRef} controls={audioReady}
        onPlay={() => setSpeaking(true)}
        onPause={() => setSpeaking(false)}
        onEnded={() => setSpeaking(false)}
        aria-label="AISHA spoken reply"
        className={audioReady ? 'voice-player' : 'voice-hidden-audio'} />
      {voiceNotice && <div className="voice-message" role="status">{voiceNotice}</div>}
      {status && (!status.stt.ready || !status.tts.ready) && (
        <div className="voice-message">
          {status.stt.missing.length > 0 && <>STT needs {status.stt.missing.join(', ')}. </>}
          {status.tts.missing.length > 0 && <>TTS needs {status.tts.missing.join(', ')}.</>}
        </div>
      )}
      {(lastSttMs != null || lastTtsMs != null) && (
        <div className="voice-measure">
          {lastSttMs != null && <>Last STT: {Math.round(lastSttMs)} ms</>}
          {lastTtsMs != null && <> · Last TTS: {Math.round(lastTtsMs)} ms</>}
        </div>
      )}
    </div>
  );
}
