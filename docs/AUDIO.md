# AISHA local audio architecture

AISHA audio is split into semantic conversation state, local speech synthesis,
ephemeral playback artifacts, and renderer-local waveform animation.

## Text-to-speech flow

Current output flow:

    completed assistant text
        -> TTSProvider
        -> GeneratedSpeech (WAV bytes, process-local)
        -> EphemeralAudioStore
        -> aisha.audio.ready (transient event)
        -> GET /v1/audio/{utterance_id}
        -> Studio HTMLAudioElement
        -> Web Audio AnalyserNode
        -> renderer mouth amplitude

`aisha.turn.finished` is emitted before speech synthesis completes. Text therefore
does not wait for TTS. A later `aisha.audio.ready` event contains only playback
metadata and an opaque short-lived utterance ID.

The audio-ready event is deliberately **not persisted**. Its playback URL would
be meaningless after the underlying artifact expires.

## Persistence and privacy

The following are persisted as part of the normal conversation system:

- user and assistant text;
- model-run metadata;
- semantic embodiment state events;
- ordinary memory state according to AISHA's memory policy.

The following are **not** persisted:

- synthesized WAV bytes;
- Web Audio amplitude samples;
- renderer mouth motion;
- transient `aisha.audio.ready` events.

The in-memory audio store is bounded and TTL-expiring (120 seconds by default)
and is cleared on Core shutdown.

## Local provider

The optional Windows development path uses `kokoro-onnx` with the versioned
Kokoro v1.0 English model and voice bundle. It is disabled by default.

Enable it with:

    .\Start-AISHA.ps1 -Voice

The first launch installs the optional `voice-local` dependencies and downloads
the model assets into AISHA's local data directory. The setup helper verifies the
upstream SHA-256 digests before accepting them.

The default voice is `af_heart`; profile/environment settings can select another
installed Kokoro voice later.

TTS is CPU-first so it does not unnecessarily compete with the conversation
model for GPU memory. GPU execution can be evaluated separately after the final
workstation environment is stable.

## Renderer behavior

When real speech output is enabled, Studio does not use the old procedural mouth
loop as lip sync. During playback, an `AnalyserNode` measures the actual waveform
energy and maps that amplitude to mouth geometry.

This preserves the intended control boundary:

    LLM -> words
    TTS -> waveform
    waveform -> mouth motion

The LLM never emits visemes, blendshape values, or mouth commands.

## Interruption behavior

Studio tracks the most recently started turn. Audio that finishes synthesis for
an older turn is ignored if a newer turn has already begun, preventing stale
speech from talking over the current interaction. Starting a new local turn also
stops current playback.

## Next audio milestones

1. Add a user-visible voice selector and mute/output controls without changing
   the provider contract.
2. Add chunked/streaming TTS so long answers can begin speaking before the full
   utterance is synthesized.
3. Add a microphone-source abstraction with an explicit privacy indicator.
4. Add local STT behind the same disabled-by-default pattern.
5. Drive `listening` embodiment state from actual microphone capture.
6. Add interruption/barge-in so detected user speech can stop AISHA's playback
   and cancel or pause the active response.
