# AISHA embodiment and perception architecture

This layer is deliberately hardware- and renderer-independent. The goal is to let
AISHA gain a visible body and local perception without making cognition depend on
Unity, Three.js, Live2D, OpenCV, CUDA, MLX, or any particular vision model.

## Embodiment

AISHA Core emits **semantic embodiment intent**:

- `idle / neutral`
- `listening / attentive`
- `thinking / focused`
- `speaking / engaged`

The first implementation is deterministic and tied only to conversation lifecycle.
It does **not** ask an LLM to choose emotions.

Current turn flow:

    turn starts
        -> thinking / focused
    first assistant text arrives
        -> speaking / engaged
    turn completes, fails, or is cancelled
        -> idle / neutral

These states are persisted as `aisha.embodiment.state` events and are available
from `GET /v1/embodiment/state`.

A future renderer owns the visual interpretation:

    semantic intent
        -> renderer policy
        -> animation clip / blendshapes / gaze / body pose / procedural motion

This separation means a renderer can move from a browser prototype to Unity or
another engine without changing AISHA cognition.

### Procedural renderer layer

Studio now adds low-level life entirely on the renderer side: randomized blink
timing, eye micro-saccades, activity-biased gaze, subtle head drift, breathing,
and smooth interpolation between poses. These signals are intentionally not
Core events and are never written to the AISHA database.

This separation is important:

    Core semantic state:
      activity + affect + intensity

    Renderer-local presentation:
      blink + gaze + breathing + head drift + future lip sync

The same semantic state can therefore drive a CSS development face today and a
future Live2D, Three.js, Unity, or other avatar renderer without changing
cognition or persistence.

### Affect layer

Affect is now a separate semantic channel from lifecycle activity. Current affect
intents are neutral, warm, amused, curious, concerned, and surprised, each with a
bounded intensity from 0 to 1.

For example:

    activity = speaking
    affect = amused
    affect_intensity = 0.35

The renderer blends activity and affect rather than replacing one with the other.
Studio exposes a local developer control for exercising these combinations without
an LLM. Automatic affect inference is intentionally deferred; raw model prose must
not directly control facial blendshapes.

## Perception

AISHA's semantic perception layer does not contain camera pixels. A local
capture/analyzer backend may hold a raw frame briefly in bounded provider-local
memory while converting it into a `VisionFrame` containing structured
`VisionObservation` records.

Example observations:

- person present;
- face location;
- gaze direction;
- hand/gesture;
- object;
- scene state;
- screen-region observation.

A `VisionObservation` may contain a normalized bounding box, confidence, label,
and provider-specific attributes. `VisionFrame.image_ref` is only an opaque
provider-local reference; raw image bytes are intentionally absent from the Core
contract.

Current flow:

    camera / screen source
        -> replaceable vision provider
        -> VisionFrame
        -> PerceptionHub
        -> latest structured scene state
        -> future cognition policy

`PerceptionHub` retains only the latest structured frame in memory. The default
runtime starts with vision disabled. On Windows, `Start-AISHA.ps1 -Vision`
selects the optional OpenCV + MediaPipe backend, but the physical camera still
starts off and must be enabled explicitly in Studio. Visual captures are never
persisted.

Developer endpoints:

- `GET /v1/perception/status`
- `GET /v1/perception/latest`

## Why this is portable

The eventual workstation can independently choose:

- camera hardware;
- frame-capture implementation;
- face/gaze/pose model;
- object detector;
- GPU runtime;
- avatar renderer.

AISHA Core only depends on the contracts. The same tests therefore run with mock
providers on Windows, macOS, Linux, or CI.

## Next milestones

The browser embodiment preview, renderer contract, explicit camera lifecycle,
structured perception runtime, and first local face backend are now scaffolded.

Next:

1. Calibrate observable head/eye geometry into a conservative viewer-attention
   signal; do not infer emotion or mental state.
2. Add optional local person/object detection when useful, likely through a
   GPU-capable ONNX Runtime backend.
3. Add perception-to-cognition policy deciding which observations are relevant
   enough to influence a response.
4. Add gesture/pose observations where they materially improve interaction.
5. Add optional automatic affect planning using the existing bounded affect
   channel.
6. Replace the CSS development face with a full avatar renderer when the semantic
   interface is stable.

Camera and screen observations should remain local by default, bounded in
retention, and explicit about when they are active. Current local camera frames
are ephemeral provider memory only; raw pixels never enter semantic state,
events, long-term memory, or SQLite.


### Camera and analysis privacy boundary

The camera path is now explicitly split into three layers:

    CameraSource
        -> CameraFrameDescriptor
        -> VisionAnalyzer
        -> VisionFrame / VisionObservation
        -> PerceptionHub

`CameraFrameDescriptor` contains only an opaque provider-local frame reference,
source ID, capture time, and optional dimensions. It does not contain image bytes.

The camera controller defaults to off. Studio exposes the current camera state
visibly, and camera enable/disable is restricted to the local Studio origin.
Disabling the camera clears both the latest capture reference and the latest
structured perception state.

The default development runtime uses `DisabledCameraSource` and
`DisabledVisionAnalyzer`, so no physical camera is opened. The optional local
backend uses OpenCV capture plus MediaPipe Face Landmarker and, by default,
EfficientDet-Lite0 object detection. One ephemeral frame is converted once and
shared by both analyses before it is discarded. Raw frames are kept only in a
small bounded `EphemeralFrameStore`, removed when consumed, and cleared when
the camera is disabled. Object labels are normalized, deduplicated, capped, and
strictly sanitized before they may enter the cognition summary. CI exercises
the lifecycle with mocks and hardware-free fake OpenCV/MediaPipe results.

AISHA also exposes an observable-only `PerceptionSummary` containing:

- person presence and count;
- primary visible-person position for renderer gaze;
- conservative head-frontal geometry;
- whether a reliable gaze observation is labeled `toward_camera`;
- up to six sanitized visible object labels;
- observed object kinds;
- source/frame metadata.

The summary intentionally does not infer facial emotion, identity, demographic
attributes, health state, or other unobserved internal traits.



### Current local face geometry

The optional local MediaPipe backend is intentionally geometry-only:

- face bounding boxes provide normalized face centers for renderer attention;
- facial transformation matrices are reduced immediately to a conservative
  forward-axis alignment score;
- scores at or above 0.90 are labeled `approximately_frontal`;
- this is **not** eye contact, gaze, attention, intent, or emotion;
- MediaPipe blendshape output is disabled because AISHA does not currently need
  to classify the user's facial expression.

Studio may use the normalized face center to bias AISHA's eye position toward the
viewer. The cognition whitelist does not receive those coordinates or the
frontal score; it receives only coarse observable facts such as "one person is
visible" or "a visible face is approximately oriented toward the camera."

Structured perception summaries expire after two seconds. The latest developer
frame may remain inspectable in memory, but stale observations no longer affect
the renderer or enter a new cognition turn. This prevents old camera state from
becoming a false current observation.


### Object awareness

The first environmental-awareness backend uses the official MediaPipe
EfficientDet-Lite0 model. It runs locally against the same transient frame used
for face/head analysis. The detector produces only structured `object`
observations (label, confidence, normalized bounding box). The raw image is
never placed in a prompt, event, memory record, or SQLite.

`PerceptionPromptPolicy` may surface at most four visible object labels to the
conversation model as explicitly transient system-generated context. Labels are
restricted to a short alphanumeric/space/hyphen vocabulary before prompt
construction, and the transient context is never copied into the memory episode
used by the learned-memory reflector.
