"""Answer the same pictures twice, before and after a change (Problem 7).

Problem 7 asks what the model looks at, and the way to test a guess is to change
one thing in some pictures and see whether the answers move. This measures two
folders that hold the same file names, one folder per class in each, and prints
one line per picture: what the model said about the original and what it said
about the changed copy.

    uv run python src/changed_answers.py --original data/clean --changed data/changed

The model is docs/model.onnx, the same file the web page runs, and the pictures
are prepared by check.py's own code, so these answers are the answers
src/check.py and the web page give. Use a tag to keep the output for a report:

    uv run python src/changed_answers.py --original data/clean \
        --changed data/changed | tee results/changed_answers.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import forward, image_files, prepare  # noqa: E402  the given tool's code

ROOT = Path(__file__).resolve().parents[1]


def load_web_model():
    """docs/model.onnx, the file the web page loads, and its model.json."""
    import onnxruntime

    model = json.loads((ROOT / "docs" / "model.json").read_text())
    session = onnxruntime.InferenceSession(
        str(ROOT / "docs" / "model.onnx"), providers=["CPUExecutionProvider"])
    return model, session


def probabilities(logits):
    shifted = logits - logits.max()
    return np.exp(shifted) / np.exp(shifted).sum()


def answer(session, model, path):
    """What the model says about one picture: label and its probability."""
    logits = forward(session, prepare(Image.open(path), model))
    probabilities_of = probabilities(logits)
    guess = int(np.argmax(logits))
    return model["labels"][guess], float(probabilities_of[guess])


def companions(original, changed):
    """Files that are in both folders, under the same class and file name."""
    found = []
    for class_dir in sorted(d for d in changed.iterdir() if d.is_dir()):
        twin = original / class_dir.name
        if not twin.is_dir():
            print(f"  {class_dir.name} is not a folder in {original}, skipped")
            continue
        for path in image_files(class_dir):
            other = twin / path.name
            if other.exists():
                found.append((class_dir.name, other, path))
            else:
                print(f"  {class_dir.name}/{path.name} is not in {original}")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True,
                        help="the folder the copies were made from, "
                             "one sub-folder per class, for example data/clean")
    parser.add_argument("--changed", required=True,
                        help="the folder with the changed copies, the same "
                             "sub-folder and file names, for example data/changed")
    args = parser.parse_args()

    original = Path(args.original)
    changed = Path(args.changed)
    model, session = load_web_model()
    labels = model["labels"]

    print("The same pictures, before and after the change")
    print(f"  original  {original}")
    print(f"  changed   {changed}")
    print(f"  model     docs/model.onnx, {len(labels)} classes")

    found = companions(original, changed)
    if not found:
        print("\nnothing to compare: no file name appears in both folders")
        return 1

    print(f"\n  {len(found)} picture(s) appear in both folders\n")

    head = (f"  {'#':>3s}  {'class':20s} {'file':32s} "
            f"{'original answer':22s} {'changed answer':22s}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    moved = []
    right_original = right_changed = 0
    per_class = {}
    for number, (cls, before, after) in enumerate(found, start=1):
        label_before, chance_before = answer(session, model, before)
        label_after, chance_after = answer(session, model, after)
        same = "same" if label_before == label_after else "MOVED"
        if label_before != label_after:
            moved.append((cls, before.name, label_before, chance_before,
                          label_after, chance_after))
        right_original += label_before == cls
        right_changed += label_after == cls
        counts = per_class.setdefault(cls, [0, 0, 0, 0])
        counts[0] += 1
        counts[1] += label_before == cls
        counts[2] += label_after == cls
        counts[3] += label_before != label_after
        print(f"  {number:3d}  {cls:20s} {before.name:32s} "
              f"{label_before + f' ({chance_before:.0%})':22s} "
              f"{label_after + f' ({chance_after:.0%})':22s} {same}")

    total = len(found)
    print(f"\n  accuracy on the originals: {right_original}/{total} = "
          f"{right_original / total:.1%}")
    print(f"  accuracy on the changed copies: {right_changed}/{total} = "
          f"{right_changed / total:.1%}")

    print("\n  per class")
    print(f"  {'class':20s} {'pictures':>8s} {'right before':>13s} "
          f"{'right after':>12s} {'answers moved':>14s}")
    for cls, (n, before_right, after_right, changes) in sorted(per_class.items()):
        print(f"  {cls:20s} {n:8d} {before_right:13d} {after_right:12d} "
              f"{changes:14d}")

    if not moved:
        print("\n  no answer moved")
        return 0

    print(f"\n  answers that moved ({len(moved)} of {total})")
    for cls, name, label_before, chance_before, label_after, chance_after in moved:
        print(f"    {cls}/{name}: {label_before} ({chance_before:.0%}) -> "
              f"{label_after} ({chance_after:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())