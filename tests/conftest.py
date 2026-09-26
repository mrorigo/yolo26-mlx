"""Shared fixtures: a tiny synthetic detection dataset."""

# tests import ``Hyp`` and ``shapes_root`` from here; pytest makes conftest importable as a module

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture(scope="session")
def shapes_root(tmp_path_factory):
    """A 2-class dataset of coloured squares; boxes are deterministic per image."""
    root = tmp_path_factory.mktemp("shapes")
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(4):
            image = (rng.random((64, 64, 3)) * 40).astype("uint8")
            draw = ImageDraw.Draw(Image.fromarray(image))
            cx, cy, s = 18 + 9 * i, 32, 14
            draw.rectangle([cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2], fill=(230, 70, 40))
            Image.fromarray(image).save(root / "images" / split / f"{i}.png")
            (root / "labels" / split / f"{i}.txt").write_text(
                f"0 {cx / 64:.4f} {cy / 64:.4f} {s / 64:.4f} {s / 64:.4f}\n"
            )
    return root


class Hyp:
    """Augmentation gains object, duck-typed like the trainer's."""

    mosaic = 0.0
    degrees = 0.0
    translate = 0.1
    scale = 0.5
    shear = 0.0
    flipud = 0.0
    fliplr = 0.5
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
