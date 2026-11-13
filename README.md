# Active and Memory-Augmented Few-Shot Personal Object Recognition for the Visually Impaired

A local, privacy-preserving assistive app. A user registers personal objects (keys, wallet, mug, medicine box)
from 3-5 views **without training anything**, and the app recognises them from a live camera and speaks guidance such as
*"Keys, slightly left, about one metre."*

Four features beyond a plain detect-embed-match pipeline:

1. **Active re-view** - low-quality or ambiguous views produce one concise hint ("Hold the camera steady.", "Move closer.", "Move the camera left.", "Not sure. Try a different angle.").
2. **Object memory** - confident sightings are logged (time, bounding box, CLIP scene label) so "Where did I leave my keys?" works for objects not in view.
3. **Look-alike handling** - objects marked as confusers are re-ranked with patch-token matching, asynchronously.
4. **Safe gallery updates** - very confident, stable matches are added automatically behind many guards; enrolment examples are never removed.

## Verification status (read this first)

Sections marked **[BUILD ENV]** were verified in the original Linux build environment. Sections marked **[THIS MACHINE]** were verified on the target machine (Windows 11, AMD64 Family 23 Model 160, 8 logical cores, AMD Radeon 610M GPU — CPU-only PyTorch, no CUDA).

| Item | Status |
|---|---|
| Unit + behavioural tests (`pytest`, 255 collected, no camera/GPU/mic/model/network) | **Passing [THIS MACHINE]** — exit code 0; all 255 reached 100% in `.venv` |
| CLI benchmark — synthetic stub (pipeline overhead only) | **Passing [THIS MACHINE]** — 1020 FPS overhead-only, clearly labelled SYNTHETIC |
| CLI benchmark — real YOLOv8n + real DINOv2 ViT-S/14 weights | **Measured [THIS MACHINE]** — see table below |
| YOLOv8n weights (`yolov8n.pt`, 6.2 MB) | **Downloaded [THIS MACHINE]** via Ultralytics; cached in `fewshot_vi/` |
| DINOv2 ViT-S/14 weights (~330 MB) | **Downloaded [THIS MACHINE]** via `torch.hub`; cached at `C:\Users\LENOVO\.cache\torch\hub\` |
| DINOv2 loads and runs a forward pass | **Verified [THIS MACHINE]** — 22 M parameters, forward pass runs on CPU |
| Any real-dataset accuracy / ablation result | **NOT produced.** No personal-object dataset was provided. See the eval section for dataset layout requirements. |
| CUDA / FP16 / MPS paths | **Not verified** (AMD Radeon 610M has no PyTorch backend; CPU-only). Device selection logic is unit-tested with mocks. |
| Real camera live loop | **Verified [THIS MACHINE]** — `python run.py --mode live --keyboard-only --max-frames 60` ran 60 frames from camera 0, **0 errors, 12.0 FPS rolling**, 92.9 ms mean total/frame, 185.8 ms mean detect (p95 266.9 ms). Gallery empty so no recognition events (expected). Clean shutdown. |
| Microphone STT (faster-whisper / sounddevice) | **Not verified** — optional packages not installed; falls back to keyboard input (logged, not silent). |
| pyttsx3 speech output | **Installed and importable [THIS MACHINE]**; actual audio playback not verified headlessly. |
| CLIP scene labels, MediaPipe hands | **Not installed** (optional). Disabled with logged warning; system continues without them. |
| Target: ≥ 15 FPS end to end | **Not met on this CPU** — static-scene (embeddings cached): **9.8 FPS**; stress (re-embed every frame): **1.8 FPS**. See table. |
| Target: < 150 ms recognition → speech | **Not verified** end-to-end; software enqueue path is < 1 ms (stub models); real DINOv2 embed adds 218–930 ms per frame on this CPU. |

### Measured performance — **[THIS MACHINE]** Windows 11, AMD64 (8 cores), CPU-only, Python 3.13.5, torch 2.15.0+cpu

Command: `python run.py --mode benchmark` (real YOLOv8n + real DINOv2 ViT-S/14 pretrained weights), `bus.jpg` 640×480, `detect_every_n_frames = 2`, 30 iterations per stage, 60 frames end-to-end.

| Stage | mean ms | p95 ms | Notes |
|---|---|---|---|
| YOLOv8n detect (1 frame) | **137.6** | 212.1 | CPU only, 7 torch threads |
| DINOv2 ViT-S/14 embed, 1 crop | **218.6** | 341.4 | Real pretrained weights |
| DINOv2 ViT-S/14 embed, batch of 4 | **734.2** | 928.3 | Real pretrained weights |
| match (synthetic gallery 20×20, dim 384) | 1.60 | 2.16 | |
| patch re-rank (synthetic, 2 candidates) | 1.48 | 1.83 | |
| guidance (5 tracks) | < 0.01 | < 0.01 | |
| **End-to-end, static scene** (embeddings cached per track) | **101.9** | 201.8 | **9.8 frames/s** |
| **End-to-end, stress** (re-embed every track every frame) | **559.7** | 1243.0 | **1.8 frames/s** |

**Live-camera run (real hardware, 60 frames, camera 0, empty gallery, `--keyboard-only`):**

| Measured | Value |
|---|---|
| Frames processed | 60 (0 errors) |
| Rolling FPS | **12.0** |
| Mean total latency/frame | 92.9 ms |
| Mean detect latency (every 2nd frame) | 185.8 ms (p95 266.9 ms) |
| Guidance latency | 0.01 ms |
| Recognition events | None (gallery empty — expected) |
| Camera | Index 0, 480×640 |
| TTS | pyttsx3 (Windows SAPI) initialised, beeps disabled (sounddevice not installed) |
| Scene classifier | Disabled (open_clip not installed) |

**Reading it honestly:** The embedder is the bottleneck. At the default input size, the 15 FPS target is not met on this CPU. This machine's AMD Radeon 610M is not exposed to PyTorch, which selected CPU; GPU performance has not been measured.

### Lower-resolution CPU profile — measured [THIS MACHINE]

Command: `.venv\Scripts\python.exe run.py --mode benchmark --config config.realtime-cpu.yaml --iterations 20 --bench-frames 60`, using real pretrained YOLOv8n and DINOv2 weights on the `bus.jpg` still image (640×480). The pipeline reuses embeddings in the static case and forces re-embedding in the stress case.

| Stage / scenario | Mean ms | p95 ms | Result |
|---|---:|---:|---|
| YOLOv8n detect, `imgsz=320` | 50.2 | 64.2 | Real model, CPU |
| DINOv2, one crop at input size 140 | 91.8 | 103.8 | Real pretrained weights |
| End-to-end static pipeline, 60 still-image frames | 37.7 | 56.8 | 26.5 FPS; includes cached embeddings |
| End-to-end stress, re-embed each track every frame | 182.5 | 387.5 | 5.5 FPS; embed batch averaged 306.9 ms |
| Recognition-to-speech start | — | — | Not measured; benchmark produced no confirmed speech event |

This profile passes 15 FPS only in the cached still-image case. It has **not** been tested with a live camera or for recognition accuracy. The smaller detector and embedder inputs can miss small objects or make similar objects harder to distinguish; validate on your own objects before relying on it. The measured live-camera result above (12 FPS) used the default profile.

**Synthetic benchmark (pipeline overhead only, stub detector + embedder):**

| Scenario | Throughput |
|---|---|
| Static scene (embeddings cached) | **1020 FPS** overhead |
| Stress (re-embed every track) | **362 FPS** overhead |

These confirm the pipeline glue (tracking, matching, guidance, queues) adds < 1 ms/frame; the bottleneck is exclusively the model inference time.

### Measured performance — **[BUILD ENV]** Linux x86_64, Intel Xeon @ 2.10 GHz, 2 cores, CPU-only, random-init weights

| Stage | mean ms | p95 ms |
|---|---|---|
| YOLOv8n detect (1 frame) | 86.2 | 88.1 |
| ViT-S/14 embed, 1 crop (**random weights — same latency as DINOv2**) | 134.5 | 160.6 |
| ViT-S/14 embed, batch of 4 (**random weights**) | 499.8 | 536.1 |
| End-to-end, static scene | 63.5 | 95.9 |
| End-to-end throughput, static scene | **15.8 frames/s** | |
| End-to-end, stress | 362 | 744 |
| End-to-end throughput, stress | **2.8 frames/s** | |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
# 1) PyTorch first (platform specific):
pip install torch --index-url https://download.pytorch.org/whl/cpu        # CPU only
#    CUDA: use the selector at https://pytorch.org/get-started/locally/
# 2) the rest
pip install -r requirements.txt
```

`requirements.txt` installs the base runtime and offline TTS. Optional integrations are separated into `requirements-optional.txt`, so a platform without their wheels or device libraries can still install the core app. On Linux, pyttsx3 needs `espeak`, and sounddevice needs PortAudio (`sudo apt install espeak libportaudio2`). Use the GUI `opencv-python` wheel instead of `-headless` if you want the `--show` window.

### Windows setup (verified on this machine)

From PowerShell in this `fewshot_vi` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CPU-only PyTorch (what this machine uses — AMD Radeon 610M has no PyTorch backend).
# For NVIDIA GPU, use the CUDA build from https://pytorch.org/get-started/locally/
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

> **Verified packages (`.venv` on this machine):** torch 2.15.0+cpu, torchvision 0.29.2+cpu, numpy 2.5.4, pillow 13.0.0, PyYAML 6.0.3, pytest 9.2.0, pyttsx3 2.99, opencv-python 5.0.0.93, ultralytics 8.4.211.

> **Note:** Use `opencv-python` (the GUI wheel), not `opencv-python-headless`, if you want the `--show` debug overlay window. The headless wheel was replaced during setup on this machine.

To enable microphone input, stereo beeps, CLIP scene labels, and optional hand guidance, install `requirements-optional.txt` too.

### First run (network needed once)

* **YOLO**: `yolov8n.pt` is downloaded by Ultralytics on first use.
* **DINOv2 ViT-S/14**: fetched via `torch.hub` (`facebookresearch/dinov2`) on first use, then cached by PyTorch. If your network blocks it, download once elsewhere and copy the torch hub cache (`~/.cache/torch/hub`) over.
* **CLIP** (scene labels) and **faster-whisper** (microphone) download their weights on first use. Without them the feature is disabled with a logged warning - no fake labels or fake transcripts.

After the first run everything works offline.

## What you need to provide manually

* **Live recognition and enrollment do not need a downloaded dataset.** You need a working camera and the personal object in view; guided enrollment captures its 3-5 views locally.
* **Model weights are required.** With internet access, the first run downloads `yolov8n.pt` and the DINOv2 ViT-S/14 Torch Hub code/weights. If the target PC is offline, copy the Ultralytics `yolov8n.pt` file and the complete PyTorch Torch Hub cache for `facebookresearch/dinov2` from a connected machine. The app does not silently substitute random weights.
* **YOLO can only propose regions it learned to detect.** Ignoring COCO class names does not make YOLOv8n a universal object detector; small or uncommon items such as loose keys may produce no boxes. Enrollment now gives a spoken repositioning hint and explains this if no boxes are proposed. If it still cannot see the item, enrollment cannot proceed with the current detector.
* **A dataset is only needed for evaluation and threshold calibration.** Create folders such as `DATA/keys/*.jpg`, `DATA/wallet/*.jpg`, with each image showing one object crop and the folder name as its class. The default full evaluation over 1/3/5-shot episodes needs at least 8 classes: 5 classes with at least 10 images each, and 3 additional classes with at least 5 images each. You can lower `--n-way`, `--n-unknown`, `--n-query`, and `--k-shot` for a smaller dataset. Do not use enrollment photos as both support and query images; the evaluator samples disjoint images within each episode.
* **Microphone, speaker, and GPU are hardware checks you must do on the target PC.** The application has keyboard and console fallbacks, but those fallbacks are not spoken guidance. Run `python run.py --mode benchmark` after installing weights to measure that machine; synthetic benchmark numbers do not measure DINOv2 or the camera.

For a CPU speed experiment, use `config.realtime-cpu.yaml` (`YOLO imgsz=320`, `DINOv2 input_size=140`): run `python run.py --mode benchmark --config config.realtime-cpu.yaml`, or `python run.py --mode live --config config.realtime-cpu.yaml --show`. This trades image detail for speed and has not been evaluated for recognition accuracy. Check recall and look-alike behavior on your objects before relying on this profile.

## Usage

```bash
python run.py --mode live                       # camera 0, microphone if available else keyboard
python run.py --mode live --show                # + debug overlay (boxes, ids, scores, FPS, timings)
python run.py --mode live --camera clip.mp4     # video file instead of a camera
python run.py --mode live --keyboard-only       # never touch the microphone
python run.py --mode enrol                      # guided enrolment of one object, then exit
python run.py --mode eval --ablation --dataset DATA --embedder dino --out results/ablation.csv
python run.py --mode benchmark                  # real detector + DINOv2 if loadable; reports NOT MEASURED otherwise
python run.py --mode benchmark --synthetic      # pipeline overhead only (stub models, clearly labelled)
python -m pytest                                # tests
```

Exit codes: 0 ok, 1 runtime failure (camera, enrolment), 2 usage/config/dataset error, 3 missing dependency or model that cannot be loaded.

### Keyboard fallback

If no microphone (or `faster-whisper`/`sounddevice`) is usable, the app logs a warning and reads typed commands at a `>` prompt. Enrolment questions ("What should I call this object?") are answered the same way. Speech output falls back to printing `[speech] ...` lines if `pyttsx3` is unavailable. Beeps are silently disabled if there is no audio output.

### Voice / typed commands

| Say | Effect |
|---|---|
| what is this / identify | IDENTIFY mode: announces each newly confirmed object once per cooldown |
| find my X / where is my X | FIND mode: only X; direction + distance + stereo beeps (faster/higher when closer); active hints if not visible; after `find_memory_wait_s` it answers from memory |
| where did I leave my X | Memory answer: "Your keys were last seen 12 minutes ago, on the desk." |
| register object / add object | Guided enrolment |
| delete X / forget X | Asks for a spoken "yes" before deleting |
| list objects | Lists gallery names |
| stop / quiet, resume | Mute / unmute guidance |
| this is similar to X | Marks the most recently enrolled object as a look-alike of X |

Names are fuzzy-matched against the gallery, and misspelled verbs are corrected.

### Guided enrolment (`--mode enrol` or "register object")

1. Hold the object steadily; a large, central, stable detection is selected. If no usable boxes appear, enrollment gives a spoken hint and the final error reports proposal/quality counts. YOLOv8n may not propose small or uncommon personal items.
2. 3-5 quality-approved views are captured within 30 s ("Rotate it slowly."). Blurred, edge-cut, too-small and near-duplicate views are rejected with a spoken hint.
3. Each view gets 8 deterministic augmentations (flip, brightness/contrast, colour gain, mild crop+resize, mild blur); everything is embedded in a forward pass only - **no gradient updates**.
4. You give a name and confirm it; examples are saved as `source='enrol'` with up to 32 salient patch tokens each.
5. It offers to mark an existing object as a confuser.

## How it works

```
Camera thread (Queue(2), drop-oldest)
  -> detect every N frames (YOLO, class-agnostic) / IoU tracker predicts on skipped frames
  -> embed only NEW or CHANGED tracks, batched in one DINOv2 pass (CLS + patch tokens)
  -> matcher: cosine vs gallery -> ACCEPT / AMBIGUOUS / UNKNOWN
       AMBIGUOUS between marked look-alikes -> async patch re-rank worker (deadline-bound)
  -> temporal vote (window/min_votes) -> only SMOOTHED ACCEPT results may be named
  -> quality (blur, edge-cut, size, margin) -> active hints / gallery-update queue / sighting writer queue
  -> guidance engine -> prioritised, interruptible TTS thread + stereo beeps
Voice thread: STT or keyboard -> intent parser -> AppState (thread-safe) -> vision loop
```

Threads: capture, vision (main), patch re-rank, gallery-update writer, sighting DB writer, TTS, voice, scene classifier. All work queues are bounded; database updates do not block the vision loop. A per-frame exception is logged and counted, never fatal. Ctrl+C or `q` stops threads and releases the camera, models, audio and DB.

Decision rules (`matching.*`): per-object score = max(prototype similarity, mean top-k similarity). ACCEPT needs score >= `accept_sim` and a top-2 margin >= `min_margin` and the top two not being confusers of each other. AMBIGUOUS is the band `[ambiguous_sim, accept_sim)`, a too-small margin, or confusers. Otherwise UNKNOWN. **Single-object rule:** with one object there is no runner-up, so the margin is `+inf`; `accept_sim` alone decides, and an absent runner-up can never turn ACCEPT into AMBIGUOUS or a low score into ACCEPT.

Safe updates add an `auto` embedding only if: smoothed ACCEPT for a full temporal window, score >= `update_min_sim`, margin >= `update_min_margin`, quality OK, cooldown passed, not a near-duplicate (`duplicate_sim`), confuser re-rank agrees (for look-alike objects), and drift guard not tripped. At the cap it replaces the most redundant **auto** embedding; enrolment embeddings are never deleted (the store refuses).

## Configuration (`config.yaml`)

Validated on load (unknown keys, wrong types, out-of-range values and inconsistent combinations are rejected). Sections:

* `camera` - index, width, height, target_fps.
* `detector` - backend (`yolo`), weights, conf, detect_every_n_frames, max_objects, `imgsz` (YOLO input resolution; smaller can improve CPU speed while reducing small-object recall).
* `embedder` - model (`dinov2_vits14`), input_size, fp16 (applied on CUDA only).
* `matching` - accept_sim, ambiguous_sim, min_margin, knn_k.
* `temporal` - window, min_votes.
* `gallery` - max_embeddings_per_object, update_min_sim, update_min_margin, update_cooldown_s.
* `quality` - blur_var_min, edge_margin_px, min_box_area_frac.
* `guidance` - hint_cooldown_s, announce_cooldown_s, known_object_width_cm_default, camera_hfov_deg.
* `memory` - min_conf_to_log, log_cooldown_s, scene_every_n_s.
* `features` - self_improve, active_hints, memory, patch_rerank, hand_guidance (off), depth_model (see limitations).
* `tracker` - reembed_iou, reembed_every_frames, match_iou, max_missed_frames.
* `rerank` - deadline_s, top_patches, min_gain, accept_patch_sim, min_patch_margin.
* `gallery_policy` - duplicate_sim, drift_min_sim.
* `enrol` - min_views, max_views, timeout_s, augmentations_per_view, seed, stable_frames, min_box_area_frac, stable_iou.
* `guidance_policy` - find_memory_wait_s, near_zone_frac, slight_zone_frac, stale_speech_s, beep_*.
* `voice` - fuzzy_cutoff, confirm_timeout_s.
* `logging` - level.

**Thresholds are untuned defaults.** `accept_sim` and `accept_patch_sim` in particular depend on the embedding model and must be calibrated on your own data before relying on the safety behaviour.

## Evaluation

Dataset layout: `DATA/<class_name>/*.jpg` (`.jpeg/.png/.bmp` also read). Each image is one object crop (the whole image is embedded). Episodes are N-way K-shot with reproducible sampling (`--seed`): `n_way` known classes (K support images enrolled, `n_query` disjoint queries each) plus `n_unknown` unknown classes used only as open-set queries.

```bash
python run.py --mode eval --ablation --dataset DATA --embedder dino --k-shot 1 3 5 --episodes 20 --out results/ablation.csv
```

Metrics: top-1 accuracy (accepted as the right class), closed-set top-1 (argmax ignoring thresholds), FAR (unknown accepted), FRR (known reported "not sure"), mis-ID (known accepted wrongly), decision latency, slow-path fraction. The five ablations use the production matcher/re-ranker/updater: (1) baseline; (2) cascade, with the most similar class pairs auto-marked as confusers, plus an "always re-rank" latency reference; (3) look-alike subset; (4) degraded (blurred / low-resolution) queries with hint coverage and a *simulated* re-view (re-capturing the clean image); (5) sequential sessions with self-update off vs on (offline simulation: each accepted query is fed to the updater as a stable match; cooldown disabled; reports wrong-class auto-adds). If the dataset is missing or too small the command fails with an explanation. `--embedder color-hist` is a non-learned baseline for plumbing checks only and is labelled as such in every report; `random-init` is refused for evaluation.

Observation from the plumbing run (synthetic data, colour-histogram embedder, **not** a result): when FAR was above zero, every automatic gallery addition was a wrong-class one. Safe updates are only as safe as the model's open-set rejection - check FAR and `wrong_add_rate` on your data, or set `features.self_improve: false`.

## Privacy

Everything runs locally. No image, audio or text is uploaded or sent for remote inference. The only network use is the first-run model/weights download. The gallery and sightings live in the local SQLite file (`data/gallery.db`); sightings store a time, box, scene label and scene embedding - not images.

## Known limitations

* Not tuned or validated on real data; see the verification table. No claim is made about real-world accuracy.
* CPU-only throughput is limited by embedding (see measurements); a GPU path exists but is unmeasured.
* **Implemented differently from the original plan:** tracker is a self-contained IoU tracker with velocity prediction (not ByteTrack). **Not implemented:** the optional depth model (`features.depth_model: true` is rejected by config validation; distance uses the box-size heuristic with a default 8 cm object width, so it is rough for objects of other sizes) and per-object threshold calibration.
* Look-alike re-ranking only runs for objects explicitly marked as confusers. A re-rank that misses its deadline is ignored (the provisional result stands) until the track is next re-embedded.
* Gallery updates and sightings use separate bounded writer queues; if the gallery-update queue is full, that update attempt is skipped without slowing frame processing.
* While a new object is being embedded on CPU the loop pauses; guidance is delayed accordingly.
* Motion blur, clutter, lighting changes and near-identical objects remain hard; a wrong name is worse than silence, so the system prefers "not sure", but it cannot guarantee it.
* The microphone path listens while the speaker talks; use headphones or push-to-talk-style quiet to avoid self-triggering.
