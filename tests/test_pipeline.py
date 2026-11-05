import csv
import json

import cv2
import numpy as np
import pytest
import torch

from shot_classifier.data import CLASSES, load_clip, read_manifest
from shot_classifier.model import choose_device, make_encoder, make_head, metrics


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


def manifest(tmp_path):
    rows = []
    for split in ('train', 'val', 'test'):
        for label in CLASSES:
            clip = tmp_path / f'{split}_{label}.avi'
            clip.touch()
            rows.append(dict(path=clip.name, label=label, split=split, group=split))
    path = tmp_path / 'clips.csv'
    write_manifest(path, rows)
    return path, rows


def write_manifest(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'split', 'group'])
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_resolves_relative_paths(tmp_path):
    path, _ = manifest(tmp_path)
    rows = read_manifest(path)
    assert len(rows) == 6
    assert all(row['path'].is_absolute() for row in rows)


@pytest.mark.parametrize('problem,match', [
    ('leak', 'leaks across'), ('duplicate', 'Duplicate clip'),
    ('missing', 'does not exist'), ('label', 'unknown label'),
    ('class', 'must contain both'), ('empty', 'must be filled'),
])
def test_manifest_rejects_bad_data(tmp_path, problem, match):
    path, rows = manifest(tmp_path)
    if problem == 'leak':
        rows[2]['group'] = 'train'
    elif problem == 'duplicate':
        rows[2]['path'] = rows[0]['path']
    elif problem == 'missing':
        rows[0]['path'] = 'absent.avi'
    elif problem == 'label':
        rows[0]['label'] = 'dunk'
    elif problem == 'class':
        rows.pop()
    else:
        rows[0]['group'] = ''
    write_manifest(path, rows)
    with pytest.raises(ValueError, match=match):
        read_manifest(path)


def test_short_video_rgb_normalization_and_repeat(tmp_path):
    path = tmp_path / 'red.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 15, (64, 64))
    assert writer.isOpened()
    for _ in range(3):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # OpenCV writes BGR; network needs RGB.
        writer.write(frame)
    writer.release()
    clip = load_clip(path)
    assert clip.shape == (3, 16, 112, 112)
    assert clip.dtype == torch.float32 and torch.isfinite(clip).all()
    assert clip[0].mean() > 2 and clip[2].mean() < -1
    assert torch.equal(clip[:, 0], clip[:, -1])


def test_corrupt_video_fails(tmp_path):
    path = tmp_path / 'bad.avi'
    path.write_text('not video')
    with pytest.raises(ValueError, match='Cannot open|no decodable'):
        load_clip(path)


def test_frozen_backbone_and_trainable_head():
    encoder, head = make_encoder(pretrained=False), make_head()
    running_mean = encoder.stem[1].running_mean.clone()
    with torch.no_grad():
        features = encoder(torch.randn(1, 3, 16, 112, 112))
    before = head.weight.detach().clone()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
    logits = head(features)
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    optimizer.step()
    assert features.shape == (1, 512) and logits.shape == (1, 2)
    assert not torch.equal(before, head.weight)
    assert all(p.grad is None for p in encoder.parameters())
    assert torch.equal(running_mean, encoder.stem[1].running_mean)


def test_metrics_orientation_and_imbalance():
    result = metrics(torch.tensor([[3., 0.], [3., 0.], [3., 0.], [3., 0.]]),
                     torch.tensor([0, 0, 0, 1]))
    assert result['accuracy'] == 0.75
    assert result['balanced_accuracy'] == 0.5
    assert result['confusion_matrix_actual_rows_predicted_columns'] == [[3, 0], [1, 0]]


def test_cuda_request_does_not_silently_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert str(choose_device('auto')) == 'cpu'
    with pytest.raises(ValueError, match='CUDA is unavailable'):
        choose_device('cuda')


@pytest.mark.parametrize('use_clip', [False, True])
def test_inspect_command_traces_video_to_prediction(tmp_path, monkeypatch, capsys, use_clip):
    from shot_classifier.__main__ import main

    command = ['shot_classifier', 'inspect']
    if use_clip:
        path = tmp_path / 'clip.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 15, (64, 64))
        assert writer.isOpened()
        for _ in range(3):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        command.append(str(path))
    monkeypatch.setattr('sys.argv', command)
    main()
    report = json.loads(capsys.readouterr().out)
    assert report['input']['source'] == (str(path) if use_clip else 'synthetic random numbers')
    assert report['input']['shape'] == [1, 3, 16, 112, 112]
    assert report['input']['dtype'] == 'torch.float32'
    assert report['input']['device'] == 'cpu'
    assert report['encoder']['weights'] == 'random'
    assert report['encoder']['feature_shape'] == [1, 512]
    classifier = report['classifier']
    assert classifier['logit_shape'] == [1, 2]
    assert classifier['class_order'] == list(CLASSES)
    logits = torch.tensor(classifier['logits'])
    scores = torch.tensor([classifier['softmax_scores'][label] for label in CLASSES])
    assert torch.isfinite(logits).all()
    assert torch.allclose(scores, logits.softmax(0))
    assert classifier['prediction'] == CLASSES[int(logits.argmax())]
    assert 'untrained' in report['note']
