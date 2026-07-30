# Notes on the code

This repository preserves the code exactly as it was submitted, then applies a small,
explicitly scoped set of fixes on top. The two states are separated in git history:

| Commit | Contents |
| --- | --- |
| `Add final project source code as submitted (June 2022)` | byte-for-byte identical to the submission |
| `Fix five defects in the submitted code` | the five changes listed below, nothing else |

To read the original, unmodified code:

```bash
git show $(git log --reverse --format=%H | head -1):src/train.py
```

---

## Defects fixed

Five genuine bugs, five changed lines. Line numbers refer to the original files.

### 1. `train.py:276`. Wrong argument count, prevents training from running at all

`train_valid_one_epoch` is defined at line 135 with four parameters, but the call inside
`train_loop` passed five:

```python
train_valid_one_epoch(model, loaders, criterion, optimizer, augmentation)
```

This raises `TypeError` on the first epoch, so `train.py` as submitted cannot complete a single
training step. Augmentation already reaches the model through the dataset transforms configured in
`get_loaders`, so the extra argument is dropped rather than added to the signature.

### 2. `data.py:56`. Assignment to the wrong variable

```python
if not isinstance(data_paths, list):
    test_paths = [data_paths]      # should be data_paths
```

When `data_paths` was passed as a bare string, this silently overwrote the test paths with the
training path and left `data_paths` unwrapped. The project always called it with a list, so the
branch never fired in practice, but the intent is unambiguous.

### 3. `data.py:137`. Hardcoded phase in the split log

Inside the loop over `['train', 'valid']`, every row was appended with `'phase': 'test'` instead of
the loop variable. The exported split CSV therefore labelled all training and validation samples as
test rows. Affects logging only, not what the model trained on.

### 4. `model.py:25`. The freeze assertion never checked anything

```python
for param in self.model.parameters():
    if 'fc' not in name:            # `name` leaks from the loop above
```

`self.model.parameters()` yields only tensors, so `name` still held the last value from the
preceding `named_parameters()` loop. The guard evaluated the same constant for every parameter,
making the assertion meaningless. Fixed by iterating `named_parameters()`.

### 5. `anot_mosaic.py:70`. Dead branch, `none` tiles rendered wrong

```python
if classes == class_to_id['none']:   # str compared to int 0
```

`anot_tile` receives `classes` as a class-name string, so this compared `'none'` against `0` and was
never true. `none` tiles fell through to the colouring path and received a black overlay at
`alpha=0.6`, darkening background tiles that should have been left untouched. Fixed by comparing
against `'none'`.

---

## Known issues, deliberately left as-is

These are documented rather than fixed, so the code stays close to what was submitted.

### Hardcoded absolute paths

The code was written to run on one specific machine. Dataset roots are hardcoded and there is no
configuration layer:

| File | Paths |
| --- | --- |
| `train.py:355-359` | `/home/linuxu/Desktop/testset3`, `/home/linuxu/Desktop/miriam`, `/home/linuxu/Desktop/prev_normal_dataset2` |
| `data.py:271-272`, `296-297` | same roots, inside the `test_data` / `test_dataset` sanity checks |
| `anot_mosaic.py:10-15` | four dated capture folders plus an output directory |
| `mosaic_annotator.py:29`, `:265` | dataset root and the legend image |
| `inference.ipynb` | run directory and dataset roots throughout |

`train.py:414` also hardcodes the Weights & Biases entity (`guykabiri`). Anyone re-running the code
needs to edit these by hand.

### API drift since 2022

`requirements.txt` pins mid-2022 versions because several APIs the code depends on have since
changed or been removed:

- **`DataFrame.append`** (`data.py:89`, `:137`, `:227`; `mosaic_annotator.py:109`, `:117`) was
  deprecated in pandas 1.4 and removed in pandas 2.0. Modern equivalent is `pd.concat`. Appending
  row-by-row in a loop is also quadratic, which matters at 12k tiles.
- **`torchmetrics.functional.accuracy` / `f1_score`** (`train.py:187-188`) changed signature; current
  versions require an explicit `task='multiclass'` argument.
- **`from kornia.losses.focal import FocalLoss`** (`train.py:22`) uses a submodule path that later
  kornia releases stopped exposing. `cost_sensitive_loss.py:8` already uses the top-level import
  and wraps it in `try/except`.
- **`A.Flip`** (`data.py:155`, `:170`) is deprecated in albumentations 1.4+.
- **PySimpleGUI** 4.60.x is the last release usable without a licence key; `mosaic_annotator.py`
  targets that generation of the API.

### Unused code

- `train.py:181 calc_metrics` is superseded by `calc_metrics_new`, which applies the clinical
  penalty matrix. Only the latter is called.
- `train.py:51 test_model` and `train.py:86 test_time_aug_model` accept an `augmentation` parameter
  they never use.
- `train.py:8` imports `bgr_to_grayscale` and `train.py:22` imports `KFocalLoss`; neither is
  referenced. Grayscale conversion happens in `TilesDataset.__getitem__` via OpenCV, and the focal
  loss actually used is constructed inside `CostSensitiveRegularizedLoss`.
- Stray imports left by the editor: `data.py:1-3` (`itertools.count`, `random.shuffle`, `re.M`) and
  `mosaic_annotator.py:1-2` (`concurrent.futures.thread`, `email.policy.default`).

### Structural notes

- `mosaic_annotator.py` runs its GUI event loop at module scope rather than under
  `if __name__ == '__main__':`, so importing it launches the application.
- `TilesData.load_data` caps the `mesh` class at 650 samples (`data.py:64`, `:112`) to blunt the
  class imbalance. The cap is a default argument rather than a documented hyperparameter.
- `data.py:112` compares `cls == 'mesh'` while incrementing `counts[label]`, where `mesh` and
  `mesh_ring` share label `2`. The cap therefore counts both variants but only skips `mesh`.
