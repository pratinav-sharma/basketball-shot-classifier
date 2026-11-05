# Part 1: from a video to a prediction

Run these commands from the repository with your Python environment active. This
walkthrough runs on CPU. When using the cloud workspace from a Mac, computation
happens on the cloud machine.

## 1. Trace the complete path

```bash
python -m shot_classifier inspect
```

This uses synthetic input and random model weights, so it works offline. It runs
inference without updating weights. The report follows this path:

```text
video tensor         [1, 3, 16, 112, 112]
    R3D-18 encoder
feature tensor       [1, 512]
    linear classification head
raw scores (logits)  [1, 2]
    softmax / choose the largest score
scores and a demo label: layup or jump_shot
```

The head has not been trained on basketball. The demonstration label illustrates
the computation; it does not tell us whether a shot is classified correctly.

## 2. Read the input dimensions

A tensor is a multidimensional array of numbers. PyTorch's video CNN expects
dimensions in this order:

| Dimension | Example | Meaning |
| --- | --- | --- |
| Batch | 1 | Number of videos processed together |
| Channels | 3 | Red, green, and blue |
| Frames | 16 | Sampled moments from each video |
| Height | 112 | Pixels vertically in each processed frame |
| Width | 112 | Pixels horizontally in each processed frame |

Four videos would have input shape `[4, 3, 16, 112, 112]` and produce features
shaped `[4, 512]`. Each video gets its own feature vector.

`torch.randn(...)` generates random floating-point values with this arrangement.
The synthetic input has no basketball content and skips video preprocessing.
The report also shows `torch.float32` (32-bit floating-point values) and `cpu`
(where the tensor lives).

## 3. Follow an actual video through preprocessing

If you have a short, trimmed clip available, run:

```bash
python -m shot_classifier inspect /path/to/shot.mp4
```

No example basketball footage is included. This command uses `load_clip` in
[`data.py`](../shot_classifier/data.py), the same preprocessing used for training
and prediction:

1. OpenCV decodes the frames and resizes each to width 171 and height 128.
2. Convert OpenCV's blue/green/red channel order to red/green/blue.
3. Select 16 evenly spaced frames across the clip. Short clips repeat frames.
4. Stack the frames and rearrange dimensions into `[time, channels, height, width]`.
5. Apply the pretrained weights' transforms: resize, center crop to 112 by 112,
   scale pixel values from 0–255 to 0–1, and normalize channels using the expected
   means and standard deviations. The result is `[3, 16, 112, 112]`.
6. `unsqueeze(0)` adds the batch dimension: `[1, 3, 16, 112, 112]`.

Trim clips to one action, roughly 1–3 seconds. The center crop can remove action
near the frame edges. This baseline samples across the whole clip rather than
using a fixed sampling frame rate.

## 4. Understand the 512 features

R3D-18 uses convolutions across time, height, and width to process video patterns.
Its layers transform those patterns, and pooling combines the final information
into 512 numbers per video.

In [`model.py`](../shot_classifier/model.py), `encoder.fc = nn.Identity()` removes
the original classification layer. That exposes the 512 features instead of the
original action-class scores. Features have no individual names such as "ball"
or "arm"; their meaning depends on the learned weights and their combinations.

To use features learned from Kinetics-400 action videos, run:

```bash
python -m shot_classifier inspect --pretrained
# Or use the same encoder with your own clip:
python -m shot_classifier inspect /path/to/shot.mp4 --pretrained
```

The first use downloads about 127 MB from PyTorch if the weights are not cached.
The shapes remain the same. The encoder now has learned weights; the new
basketball classification head still has random weights.

## 5. Turn features into scores and a label

`make_head()` creates `nn.Linear(512, 2)`. Each output combines all 512 features
using its own weights and adds a bias. For one video, the result has shape
`[1, 2]`. Class order is always `layup`, then `jump_shot`.

Those two numbers are logits: raw scores that can be positive or negative.
`softmax(dim=1)` converts them to positive scores summing to one across classes.
`argmax(dim=1)` finds the index of the largest logit, which maps to a class name.
Softmax preserves the ordering, so the largest softmax score gives the same label.

For example, logits `[0.2, 1.4]` map to scores of approximately `[0.23, 0.77]` and
select `jump_shot`. These illustrative numbers do not establish a 77% chance of
being correct. Model accuracy requires a trained head and held-out real clips.

Read `inspect_video` in [`__main__.py`](../shot_classifier/__main__.py) to connect
each report field to the operation producing it. `torch.inference_mode()` skips
gradient tracking for this forward pass. Training and gradients are Part 2.

## 6. Check your understanding

- What does each input dimension represent?
- Why do we convert BGR to RGB and normalize real frames?
- Why does the encoder return `[1, 512]` instead of an image?
- How does the head convert 512 numbers into two logits?
- Why can an untrained head output a label without having learned basketball?

The existing `python -m shot_classifier smoke` command also prints input and
feature shapes, then verifies a training update. The `inspect` command focuses
on the forward path. Run `python -m pytest -q` to check the pipeline, including
the inspection command with synthetic input and a generated video file.
