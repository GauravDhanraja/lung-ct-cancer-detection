"""
checkpoint_utils.py
────────────────────
Shared async, atomic checkpoint saving for train_detector.py and
train_classifier.py.

Why this exists: torch.save() can be surprisingly slow on Windows —
slower filesystem metadata handling than Linux, real-time antivirus
scanning the freshly written .pth file, or a project folder that lives
inside a OneDrive/Dropbox sync root are all common culprits. Whatever the
cause, the fix is the same: get the tensors off the GPU (fast, and has to
happen synchronously — the tensors would otherwise keep changing under the
writer's feet mid-save) and do the slow part (disk I/O) on a background
thread so it never blocks training.

Writing to a .tmp path and os.replace()-ing it into place afterwards makes
the save atomic: a crash or Ctrl+C mid-write can never leave a truncated
"best" checkpoint behind — only ever the old good one or the new good one.
"""

import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _move_to_cpu(obj):
    """Recursively clone tensors in a (possibly nested) state dict to CPU."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cpu(v) for v in obj)
    return obj


def _write_atomically(obj, path):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)   # atomic on both Windows and POSIX


def _write_to_all(obj, paths):
    for path in paths:
        _write_atomically(obj, path)


def save_checkpoint_async(ckpt: dict, paths, prev_thread: Optional[threading.Thread]) -> threading.Thread:
    """
    Snapshot ckpt's tensors to CPU synchronously (fast — a D2H copy, not a
    disk write), then write to disk on a background thread (slow, but no
    longer blocks the training loop). Joins prev_thread first so writes
    never overlap — in the normal case prev_thread is already done and this
    join is instant.

    `paths` can be a single Path or a list of Paths — pass a list (e.g.
    [last_path, best_path] on an improving epoch) rather than calling this
    function twice in a row: two separate calls means the *second* call's
    prev_thread.join() waits for the *first* call's write to finish, which
    can silently block the training loop on a slow disk. Writing to all
    destinations from one background thread avoids that trap entirely.

    Call save_thread.join() once more after your training loop ends, to
    make sure the very last checkpoint has actually landed on disk before
    you report "done" or return control to the caller.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if prev_thread is not None and prev_thread.is_alive():
        prev_thread.join()
    cpu_ckpt = _move_to_cpu(ckpt)   # synchronous but cheap (D2H copy only)
    t = threading.Thread(target=_write_to_all, args=(cpu_ckpt, paths), daemon=True)
    t.start()
    return t


# ═══════════════════════════════════════════════════════
# RNG STATE — so a resumed run reproduces the same augmentation
# sequence as an uninterrupted one, not just the same weights
# ═══════════════════════════════════════════════════════

def capture_rng_state() -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]):
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])
