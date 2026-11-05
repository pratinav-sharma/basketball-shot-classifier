# Basketball shot classifier

Learn convolutional neural networks (CNNs) by classifying **one trimmed basketball shot video** as `layup` or `jump_shot`. This is a learning baseline, not a detector for full games. There is no trained basketball model or real dataset yet.

We use PyTorch and TorchVision's pretrained **R3D-18 video CNN**, keep its feature extractor frozen, and learn a new two-class final layer. The first experiment runs on CPU. Python 3.12 is the tested version; the direct dependencies are pinned in `pyproject.toml`.

## Start here in the cloud terminal

The prepared cloud environment has a virtual environment outside the checkout. From the repository:

```bash
cd /workspace/basketball-shot-classifier
source /workspace/.venvs/basketball-shot-classifier/bin/activate
python -m shot_classifier doctor
python -m shot_classifier smoke
python -m pytest -q
```

`doctor` checks CUDA at runtime. `smoke` uses random weights and a synthetic video to check a real CNN forward pass and classifier update without downloading weights. Passing it means the plumbing works, not that the model understands basketball.

For **Part 1: how a video becomes a prediction**, run `python -m shot_classifier inspect` and follow [the walkthrough](docs/part1.md). The command reports input dimensions, CNN features, logits, softmax scores, and a demonstration label. It also accepts a trimmed clip. Its classification head is untrained.

For a fresh Linux or Windows CPU environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

On an Apple Silicon Mac, activate the environment and skip the CPU-index command; install with `python -m pip install -e ".[dev]"` from PyPI. This starter uses CPU on Macs; MPS acceleration is a later experiment. These local Mac/Windows instructions have not been tested in this Linux workspace. Legacy Intel Macs may require a separately compatible PyTorch/Python combination.

Cloud execution uses the cloud machine's hardware, regardless of whether your browser is on Mac or Windows. Local checkouts and virtual environments are separate: use Git commits/push/pull to transfer code, and a deliberate separate transfer for ignored datasets and checkpoints. Do not copy `.venv` between operating systems.

## How the CNN works

A video is a stack of images. Each image contains red, green, and blue numbers. A convolution slides a small learned filter over those numbers; its output describes where a pattern occurs. Training adjusts the filter values so useful patterns become easier to recognize. Early layers can represent simple edges, while deeper layers combine patterns into richer features.

A **3D convolution** slides over time as well as image height and width, so it can represent motion across frames. Our input is 16 frames. Preprocessing produces a tensor shaped `[batch, 3, 16, 112, 112]`: videos, color channels, frames, height, width.

The pretrained CNN learned from Kinetics-400 videos. Transfer learning reuses those learned filters as a starting point. We replace its original classifier with a layer that maps **512 features to 2 scores**. Freezing the CNN means its weights and BatchNorm statistics stay unchanged. We compute its features once per clip, then train the small final layer from that in-memory cache. This is manageable on CPU, although decoding and feature extraction still take time.

The two raw scores are called **logits**. Cross-entropy loss measures how poorly they match the correct label. Backpropagation computes how each trainable weight affects that loss; the optimizer updates those weights. An epoch here is one full pass over the cached training features. Softmax converts prediction logits to scores summing to one; those scores are not calibrated confidence.

Read the implementation in this order: [clip loading](shot_classifier/data.py), [model and metrics](shot_classifier/model.py), then [training and commands](shot_classifier/__main__.py).

## Collect the first dataset

1. Start with about **20–30 clips per class** to debug the workflow. Expand toward 100 or more per class across different players, courts, camera angles, and sessions before drawing strong conclusions. These are practical starting targets, not guarantees of accuracy.
2. Trim each clip to roughly **1–3 seconds**, showing one player approaching or gathering, the release, and a short follow-through. Keep the shooter near the center because the baseline uses a center crop. Keep clip durations similar across classes.
3. Label a layup as a close-range finishing action following an approach; label a jump shot as a shooting action with a jump and shooting release. Write down your exact boundary cases. Set aside dunks, hooks, free throws, unclear footage, and mixed-action clips for now. Make/miss is not the target label.
4. Collect both classes in each recording setting. Otherwise the model might learn the gym or camera instead of shot mechanics. Use recordings you have permission to use.
5. Split by **recording session or original source video**, approximately 70% training, 15% validation, 15% test. Related clips, replays, or alternate views of the same action must share a group. For an unseen-player evaluation, group by player instead (combine overlapping groups when necessary). Do not randomly split frames from the same clip.

The training set teaches the final layer. Validation chooses the checkpoint. Test clips are held aside until the experiment decisions are finished. Every split needs both classes. With a tiny number of independent sessions, any test estimate will be unstable.

Copy `data/manifest.example.csv` to `data/manifest.csv` and replace its example rows with your actual clips. Paths are relative to the CSV directory. The example names are placeholders; no videos are bundled.

```csv
path,label,split,group
clips/session01_layup.mp4,layup,train,session01
clips/session01_jump.mp4,jump_shot,train,session01
clips/session02_layup.mp4,layup,val,session02
clips/session02_jump.mp4,jump_shot,val,session02
clips/session03_layup.mp4,layup,test,session03
clips/session03_jump.mp4,jump_shot,test,session03
```

The validator rejects missing files, duplicate paths, bad labels, missing classes, and groups that cross splits. It cannot discover mislabeled footage, duplicated content under different filenames, or related videos given different group IDs; review those yourself. Record clip provenance and labeling decisions alongside the private dataset. Videos, checkpoints, and caches are ignored by Git.

## Train and evaluate

```bash
python -m shot_classifier check-data data/manifest.csv
python -m shot_classifier train data/manifest.csv --epochs 30 --output runs/baseline
python -m shot_classifier evaluate data/manifest.csv runs/baseline/best.pt
python -m shot_classifier predict data/clips/new_shot.mp4 runs/baseline/best.pt
```

The first training/evaluation/prediction run downloads the pretrained weights (about 127 MB). The default cache is `.cache/torch` under your current directory; set `TORCH_HOME` to use another location. Smoke tests and automated tests are offline. OpenCV supplies video decoding; codec support varies, so `check-data` decodes every clip before training.

Training saves `best.pt`, `history.json`, and `config.json` under the new output directory. Existing run directories are refused to avoid overwrites. The checkpoint contains the learned head and preprocessing identifiers; restoring it reconstructs the frozen pretrained encoder. Keep your manifest and source clips unchanged to reproduce a run. A fixed seed helps repeatability, but does not guarantee identical floating-point results across hardware.

Evaluation reports accuracy, per-class recall, balanced accuracy (mean class recall), and a confusion matrix whose **rows are actual labels and columns are predictions**, ordered `layup`, `jump_shot`. Compare against a majority-class baseline. Review wrong predictions and report clip counts plus independent session counts; a high score on a few similar clips is weak evidence. Repeatedly using test errors to tune the model turns that test set into validation data; reserve new groups for a final assessment.

Limitations: uniform sampling across each trimmed clip differs from the pretrained model's fixed-rate evaluation protocol; the fixed resize and center crop may lose useful context; no augmentation or class weighting is included. The model must choose one of the two classes, even for an unrelated video. This implementation is intentionally small and does not yet provide unknown-action detection, full-game localization, or persistent feature caching.

## Incremental learning plan

1. **Understand the tensor:** run `doctor` and `smoke`; trace frames → features → logits → loss in the code.
2. **Check the data:** collect the pilot clips, inspect labels and crops, then run `check-data`.
3. **Establish a baseline:** train only the head; compare train/validation losses and save the experiment settings. If it cannot fit a few training examples, inspect labels and preprocessing before expanding.
4. **Evaluate honestly:** use held-out sessions, balanced accuracy, the confusion matrix, and manual error review. Collect more diverse examples to address gaps.
5. **Improve one thing at a time:** consider temporal sampling, augmentation, and then unfreezing the last CNN block with a smaller learning rate. Unfreezing requires a different training loop; cached frozen features cannot train the encoder.
6. **Measure performance:** use [the profiling guide](docs/profiling.md) once representative clips and suitable hardware are available. Optimize only measured bottlenecks.

## Verified environment and references

On November 3, 2026 this Linux cloud workspace had Python 3.12.14, no `nvidia-smi`, no NVIDIA device nodes, and no Nsight binaries. Installed PyTorch is `2.9.1+cpu`, TorchVision `0.24.1+cpu`; `torch.cuda.is_available()` is false. CPU development is available; CUDA/Nsight execution is not verified here.

- [TorchVision R3D-18: weights, expected transforms, and model details](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html)
- [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [NVIDIA Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
