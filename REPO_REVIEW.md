# Source repository review and chosen architecture

This project is a computer-vision system. Camera frames are detected, cropped, embedded, tracked, matched, and temporally confirmed locally. Text-to-speech is the accessibility output for those vision results.

## Repositories reviewed

| Archive | What it contains | Useful idea | Why its code was not merged |
|---|---|---|---|
| `Real-Time-Object-Detection-and-Recognition-for-Blind-People-main` | Webcam script using TensorFlow Lite SSD MobileNet, a label map, gTTS, and pygame | A local detector can provide category labels without a personal gallery; label-map lookup is straightforward | It synthesizes and plays network TTS inside the per-detection frame loop, has limited lifecycle/error handling, and targets an older Raspberry Pi/TFLite setup rather than the current PyTorch/YOLO pipeline |
| `object-recognition-for-visually-challenged-main` | A notebook using YOLOv8m, OpenCV, keyboard input, and pyttsx3 | Keep pretrained detector class names and use offline speech output | Speech is invoked synchronously for every frame detection, which stalls the camera loop and repeats announcements; it has little tracking, temporal filtering, or runtime structure |
| `Object-Detection-master` | Older SIFT/ORB matching, hard-coded image paths, a small Keras CNN, Haar cascade, and serialized datasets/models | Classical feature matching and custom datasets are useful teaching examples | It mixes unrelated face detection and object classification, relies on obsolete APIs and fixed local paths, and does not provide a reliable maintained real-time few-shot pipeline |
| `assistive-ai-main` | Expo mobile app and FastAPI image-upload backend with captioning, CLIP, SVM, and k-NN modules | Image embeddings can support few-shot similarity matching; accessible audio output matters | The supplied backend handles uploaded still images rather than a live camera stream. Its upload route trains on CIFAR-10 at import time, and the README's OWL-ViT description does not match the included inference route. It is a separate mobile/server product, not a drop-in real-time local CV pipeline |

## Architecture selected for this project

1. **Detection and category labels:** YOLO proposes object boxes and preserves the pretrained model's category prediction. Supported categories can be recognized with an empty personal gallery.
2. **Few-shot visual matching:** when the user adds a small number of personal items, DINOv2 converts their reference views and live object crops into embeddings. The gallery matcher compares them; this stores examples and does not fine-tune the model.
3. **Tracking and confirmation:** an IoU tracker maintains object tracks, and temporal voting reduces one-frame label changes before speech.
4. **Assistive output:** the guidance engine speaks a category or accepted personal name with approximate direction and distance. Personal-item memory and look-alike re-ranking require a personal gallery.
5. **Evaluation:** the repository includes open-set few-shot evaluation, ablations, and benchmark commands. Reported hardware/accuracy limits remain explicit in `README.md`.

## What the user needs to enroll

Nothing for categories the pretrained detector supports. Enroll only a few personal items for which a custom name or find/memory behavior is useful. Each such item needs a few reference views. Categories outside the pretrained label set, such as loose keys, will not be named as a category unless a suitable detector/model is added; personal enrollment still depends on a live detector box proposal.

## Scope and verification

The submitted mobile app, old datasets, model binaries, and legacy scripts are not copied into this project. The selected repo remains one local Python application with a single real-time vision pipeline. Category-label integration was added after the earlier recorded test and camera runs; rerun verification on the target PC before describing the new category behavior as verified.
