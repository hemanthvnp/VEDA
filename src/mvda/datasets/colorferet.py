"""ColorFERET multi-view face loader.

ColorFERET stores one or more images per subject under different *poses*
(frontal, quarter-left, profile, ...). We treat each pose as a *view* and each
subject as a *class*, giving a genuine multi-view problem: the same person seen
from several angles, with instance correspondence by subject.

The loader is **path-agnostic**: point ``root`` at any directory that contains
the image files -- a local copy, an ``rclone`` mount, or a Google-Drive mount in
Colab (``/content/drive/MyDrive/...``). See ``docs/COLORFERET.md``.

FERET image filenames encode subject and pose, e.g. ``00123_940928_fa.ppm.bz2``
(``00123`` = subject id, ``fa`` = pose). Supported image types: ``.ppm``,
``.ppm.bz2``, ``.png``, ``.jpg``/``.jpeg`` (bz2-compressed variants too).
"""

from __future__ import annotations

import bz2
import io
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

import numpy as np

# FERET pose codes.
POSE_CODES = ["pl", "pr", "hl", "hr", "ql", "qr", "ra", "rb", "rc", "rd", "re", "fa", "fb"]
_POSE_SET = set(POSE_CODES)


def _parse_name(filename: str):
    """Extract (subject, pose) from a FERET filename, or None.

    Filenames look like ``<subject>_<date>_<pose>[_<variant>].<ext>[.bz2]``,
    e.g. ``00001_930831_fa.ppm.bz2`` or ``00001_930831_fa_a.ppm.bz2`` (the
    ``_a``/``_b`` variants must not be skipped). We split on the extension and
    on ``_`` and take the first token that is a known pose code, which is
    robust to such trailing variant suffixes.
    """
    stem = filename.split(".")[0]
    parts = stem.split("_")
    if not parts or len(parts[0]) != 5 or not parts[0].isdigit():
        return None
    for tok in parts[1:]:
        if tok in _POSE_SET:
            return parts[0], tok
    return None
_IMAGE_EXT = (".ppm", ".png", ".jpg", ".jpeg")


def _is_image(name: str) -> bool:
    low = name.lower()
    if low.endswith(".bz2"):
        low = low[:-4]
    return low.endswith(_IMAGE_EXT)


def _read_image(path: str, size: Tuple[int, int], grayscale: bool) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "Reading ColorFERET images requires Pillow. Install it with "
            "`pip install pillow`."
        ) from exc

    if path.lower().endswith(".bz2"):
        with bz2.open(path, "rb") as f:
            raw = f.read()
        img = Image.open(io.BytesIO(raw))
    else:
        img = Image.open(path)

    img = img.convert("L" if grayscale else "RGB").resize(size)
    return np.asarray(img, dtype=np.float64).reshape(-1)


def _load_subject_poses(args):
    """Worker: load all poses for one subject. Used by ThreadPoolExecutor."""
    subject, label, poses, paths_by_pose, image_size, grayscale = args
    return label, {
        pose: np.asarray([_read_image(p, image_size, grayscale) for p in paths_by_pose[pose]])
        for pose in poses
    }


def _n_workers():
    return min(32, (os.cpu_count() or 4) * 2)


def _scan(root: str):
    """Return {subject: {pose: [filepaths]}} by recursively scanning ``root``."""
    table = defaultdict(lambda: defaultdict(list))
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not _is_image(name):
                continue
            parsed = _parse_name(name)
            if not parsed:
                continue
            subject, pose = parsed
            table[subject][pose].append(os.path.join(dirpath, name))
    return table


def load_colorferet(
    root: str,
    poses: Sequence[str] = ("fa", "fb", "hl", "hr"),
    image_size: Tuple[int, int] = (64, 64),
    grayscale: bool = True,
    max_subjects: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build multi-view face data from a ColorFERET image tree.

    Each *pose* is a view and each *subject* is a class. Every image is kept as a
    separate sample, so a view holds all images captured in that pose across all
    subjects -- views therefore have different sample counts and no per-row
    correspondence (MvDA pools classes by label). Subjects must appear in *every*
    requested pose so all classes are present in all views.

    Parameters
    ----------
    root : directory containing the FERET image files (scanned recursively).
    poses : pose codes used as views; subjects missing any pose are dropped.
    image_size : (width, height) each image is resized to before flattening.
    grayscale : load as grayscale (1 channel) vs RGB (3 channels).
    max_subjects : optionally cap the number of classes (useful for quick runs).
    cache_path : if given, assembled arrays are cached to/loaded from this .npz.

    Returns ``(views, ys)``: a list of ``(n_v, d)`` feature arrays and a list of
    matching per-view label arrays (encoded subject ids).
    """
    poses = list(poses)
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return ([data[f"view_{i}"] for i in range(len(poses))],
                [data[f"y_{i}"] for i in range(len(poses))])

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"ColorFERET root not found: {root!r}. Point it at a local copy or a "
            f"mounted Drive path (see docs/COLORFERET.md)."
        )

    table = _scan(root)
    subjects = sorted(s for s, p in table.items() if all(pose in p for pose in poses))
    if not subjects:
        raise RuntimeError(
            f"No subjects have all requested poses {poses}. Found {len(table)} "
            f"subjects total under {root}. Try fewer/different poses."
        )
    if max_subjects:
        subjects = subjects[:max_subjects]
    label_of = {s: i for i, s in enumerate(subjects)}

    task_args = [
        (subject, label_of[subject], poses, table[subject], image_size, grayscale)
        for subject in subjects
    ]
    print(f"  loading {len(subjects)} subjects × {len(poses)} poses "
          f"({len(task_args)} tasks, {_n_workers()} threads) ...")
    per_subject = {}
    with ThreadPoolExecutor(max_workers=_n_workers()) as ex:
        for label, data in ex.map(_load_subject_poses, task_args):
            per_subject[label] = data

    views, ys = [], []
    for pose in poses:
        feats, labels = [], []
        for subject in subjects:
            lab = label_of[subject]
            for img in per_subject[lab][pose]:
                feats.append(img)
                labels.append(lab)
        views.append(np.asarray(feats))
        ys.append(np.asarray(labels, dtype=int))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            **{f"view_{i}": v for i, v in enumerate(views)},
            **{f"y_{i}": y for i, y in enumerate(ys)},
        )

    return views, ys


def load_colorferet_grouped(
    root: str,
    poses: Sequence[str] = ("pl", "hl", "ql", "fa", "qr", "hr", "pr"),
    image_size: Tuple[int, int] = (64, 64),
    grayscale: bool = True,
    max_subjects: Optional[int] = None,
    cache_path: Optional[str] = None,
):
    """Load ColorFERET grouped per subject and pose (for the paper's protocol).

    Returns ``(groups, poses)`` where ``groups[label][pose]`` is an
    ``(n_images, d)`` array. Only subjects present in every requested pose are
    kept, and ``label`` is the encoded subject id (0..C-1, sorted by subject).

    This grouping enables the subject-disjoint gallery/probe protocol used by the
    MvDA paper (Kan et al. 2016): train on one set of identities, recognize a
    disjoint set across poses.
    """
    poses = list(poses)
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        keys = [k for k in data.files if "|" in k]
        groups: dict = {}
        for key in keys:
            lab, pose = key.split("|")
            groups.setdefault(int(lab), {})[pose] = data[key]
        return groups, poses

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"ColorFERET root not found: {root!r} (see docs/COLORFERET.md)."
        )

    table = _scan(root)
    subjects = sorted(s for s, p in table.items() if all(pose in p for pose in poses))
    if not subjects:
        raise RuntimeError(f"No subjects have all requested poses {poses}.")
    if max_subjects:
        subjects = subjects[:max_subjects]

    task_args = [
        (subject, label, poses, table[subject], image_size, grayscale)
        for label, subject in enumerate(subjects)
    ]
    print(f"  loading {len(subjects)} subjects × {len(poses)} poses "
          f"({len(task_args)} tasks, {_n_workers()} threads) ...")
    groups = {}
    with ThreadPoolExecutor(max_workers=_n_workers()) as ex:
        for label, data in ex.map(_load_subject_poses, task_args):
            groups[label] = data

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        flat = {f"{lab}|{pose}": arr for lab, pp in groups.items() for pose, arr in pp.items()}
        np.savez_compressed(cache_path, **flat)

    return groups, poses
