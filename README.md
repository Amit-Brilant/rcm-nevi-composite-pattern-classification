<h1 align="center">Automated Classification of Reflectance Confocal Microscopy (RCM) Images of Nevi with Composite Patterns using Deep Learning and Computer Vision Algorithms</h1>

<p align="center">
  <img src="assets/poster.png" alt="Project poster summarising background, goals, methods, results and conclusions" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-1.11-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 1.11">
  <img src="https://img.shields.io/badge/backbone-ConvNeXt--Tiny-005F73" alt="ConvNeXt-Tiny">
  <img src="https://img.shields.io/badge/task-medical%20image%20classification-0A9396" alt="Medical image classification">
  <img src="https://img.shields.io/badge/domain-dermatology%20%2F%20RCM-94D2BD" alt="Dermatology and RCM">
  <img src="https://img.shields.io/badge/weighted%20accuracy-90.7%25-2A9D8F" alt="Weighted accuracy 90.7 percent">
</p>

---

## About

B.Sc. final project in Medical Engineering at Afeka Tel-Aviv Academic College of Engineering,
carried out in collaboration with **Sheba Medical Center**, July 2022. The clinical question, the
data and the definition of success all came from the dermatology research team at Sheba, and the
project's main design decisions were made jointly with them.

---

## Overview

Reflectance Confocal Microscopy images skin at cellular resolution without cutting it, a "virtual
biopsy". One lesion is captured as a **mosaic** of 35 to 196 small **tiles**, each covering
0.5 mm x 0.5 mm at 1000 x 1000 pixels. Every tile shows one of three cellular patterns (**clod**,
**mesh**, **ring**), a combination of them, or background. Reading those patterns across a lesion is
how a dermatologist assesses a mole.

**🔬 Challenge:** In real nevi the patterns nest inside one another, so a single tile routinely
carries two or three at once. Composite patterns are hard for the medical expert to label
consistently, which makes them hard to learn and hard to evaluate. The labelled data is small
(3,057 tiles) and severely imbalanced (47% background, 7% for the rarest composite class).

**💡 Solution:** A ConvNeXt-Tiny classifier over single tiles, whose predictions are painted back
onto the mosaic to produce a tile-level segmentation of the whole lesion. Two decisions taken with
the clinicians shaped the model: eight pattern combinations were merged into five clinically
equivalent classes, and a **penalty matrix encoding the clinical severity of each confusion** was
built into both the loss function and the evaluation metric.

**📈 Results:** 90.7% clinically weighted accuracy (73.6% plain top-1) on a held-out set of 717
tiles, plus a mosaic-level irregularity index that flags heterogeneous regions for review.

This project continues an earlier one under the same supervisors that classified **pure** patterns
at 94% accuracy. The step taken here is composite patterns, a substantially harder problem.

---

## Background: the pattern vocabulary

Three pure patterns, plus a background class ("none"):

<p align="center">
  <img src="assets/07-pure-patterns.jpg" alt="Clod, mesh and ring pure patterns in RCM tiles" width="72%">
</p>

They rarely appear alone. Most tiles in a real lesion look like this:

<p align="center">
  <img src="assets/08-composite-patterns.jpg" alt="Clod-Mesh, Clod-Mesh-Ring and Mesh-Ring composite patterns" width="72%">
</p>

Three patterns plus background give **eight** possible combinations. Training on all eight produced
poor results, and reviewing the confusions with the medical team led to a merge: combinations that
carry the same clinical meaning were collapsed into a single class. The grouping comes from the
team's prior published work, not from convenience.

| Original (8) | Merged (5) |
| --- | --- |
| Clod-Mesh, Clod-Mesh-Ring | **Clod-Mesh / Clod-Mesh-Ring** |
| Mesh, Mesh-Ring | **Mesh / Mesh-Ring** |
| Clod, Clod-Ring | **Clod / Clod-Ring** |
| Ring | **Ring** |
| None | **None** |

Fewer, clinically meaningful classes improved accuracy *and* made a predicted mosaic legible to a
clinician instead of a confetti of eight colours. It was the single most valuable change in the
project.

A second departure from the literature: published work segments RCM mosaics **per pixel**, but the
clinicians asked for **per tile**. The tile is the unit they label and reason about, so a tile-level
map is one they can act on directly.

---

## Data

Mosaics arrived from the hospital in batches every few months. A script split each into tiles for
labelling, and reassembled labelled tiles back into a coloured mosaic.

| | |
| --- | ---: |
| Mosaics received | 98 |
| Tiles total | 12,709 |
| **Tiles labelled** | **3,057** (~24%), from 83 mosaics |
| Tile | 0.5 mm x 0.5 mm at 1000 x 1000 px |
| Mosaic | 5 x 7 to 16 x 16 tiles |

Labelling coverage was uneven: 13 mosaics reached 91 to 100%, 7 reached 51 to 90%, and 63 sat below
50%, mostly the batches that arrived late. To widen the trainable pool the previous project's tiles
were folded in, bringing it to roughly 5,000 images.

Split: 10% held out for test, then 75/25 train/validation on the remainder, stratified so each phase
carries the class distribution of the whole.

| Class | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| None | 854 | 265 | 334 |
| Mesh / Mesh-Ring | 357 | 107 | 164 |
| Clod / Clod-Ring | 223 | 61 | 86 |
| Ring | 221 | 76 | 90 |
| Clod-Mesh / Clod-Mesh-Ring | 132 | 44 | 43 |
| **Total** | **1,787** | **553** | **717** |

Labelling quality was a problem in its own right. The team was annotating one tile at a time with no
view of its surroundings, so tiles were being judged out of context. A desktop viewer was built that
overlays their labels back onto the full mosaic, with per-class toggles and adjustable opacity.
Seeing a lone `Ring` tile sitting inside a field of `Mesh` changed how some tiles were labelled, so
the tool altered the dataset itself, not just the review process.

---

## Method

<p align="center">
  <img src="assets/16-pipeline.jpg" alt="Pipeline from RCM acquisition through tile splitting, training and prediction back onto the mosaic" width="100%">
</p>

**Backbone.** `convnext_tiny` from [`timm`](https://github.com/huggingface/pytorch-image-models),
ImageNet-pretrained and adapted to single-channel input. ConvNeXt was chosen after comparing several
architectures; it had been published only months earlier and rebuilds a ResNet with design choices
borrowed from vision transformers, which suited a small dataset better than a transformer trained
from scratch.

```python
create_model('convnext_tiny', pretrained=True, num_classes=5, in_chans=1, drop_rate=0.85)
```

| | |
| --- | --- |
| Input | 224 x 224, greyscale |
| Optimizer | AdamW, initial lr 1e-6 |
| Scheduler | cosine decay with warm-up, max 1e-4, min 1e-8, 30 steps down |
| Epochs / batch | 100 / 32 |
| Dropout | 0.85 |
| Seed | 42 |

Augmentation is aggressive by design, since 1,787 training tiles is not much: full rotation, resize
to 254 then random crop to 224, random flips, brightness and contrast jitter, coarse dropout, and
Gaussian noise. Rotation is unconstrained because a nevus has no canonical orientation.

**Cost-sensitive loss.** Not every mistake costs the same. Confusing `Mesh` with `Ring` is minor;
confusing `Clod` with `None` is not. The clinicians assigned a penalty to every possible confusion:

<p align="center">
  <img src="assets/18-cost-sensitive-loss.jpg" alt="Penalty matrix assigning a clinical severity to each pairwise confusion" width="92%">
</p>

Green is correct, light green a tolerable confusion, red a serious one. This matrix appears verbatim
in [`src/train.py`](src/train.py) as `punish_matrix`, normalised to 0 / 0.5 / 1 and applied on top of
a focal loss base (α = 0.5, γ = 2.0) with λ = 10:

```math
\mathcal{L} = \mathcal{L}_{focal}(\hat{y}, y) + \lambda \cdot \sum_{c} M_{y,c} \cdot \text{softmax}(\hat{y})_c
```

The second term charges the model for the probability mass it places on classes that are clinically
distant from the truth, so the optimiser is pushed away from *costly* errors rather than merely
wrong ones.

---

## Evaluation metric

The same matrix defines the reported metric. Per class, false positives and false negatives are
weighted by their clinical penalty before precision and recall are computed:

```math
FN_c = \sum_{i \neq c} M_{c,i} \cdot \text{cm}[c, i]
\qquad
FP_c = \sum_{i \neq c} M_{c,i} \cdot \text{cm}[i, c]
```

Per-class scores are then averaged weighted by class support. A confusion the clinicians consider
harmless counts half; a serious one counts fully.

**This is not standard top-1 accuracy, and the difference is large.** The same predictions score:

| | Weighted (reported) | Plain |
| --- | ---: | ---: |
| Accuracy | **90.7%** | 73.6% |
| Precision | 77.0% | 73.2% |
| Recall | 76.9% | 73.6% |
| F1 | 76.7% | 73.0% |

Both numbers are reproducible from the confusion matrix below. The weighted figure is the one the
project reports because it answers the question the clinicians actually asked: not "how often is the
model right", but "how often is it wrong in a way that matters". Of the 189 test errors, **43% fall
in the low-penalty band**, so the metric is measurably more generous than plain accuracy, though not
overwhelmingly so.

---

## Results

<p align="center">
  <img src="assets/19-test-confusion-matrix.jpg" alt="Test confusion matrix across the five merged classes" width="62%">
</p>

Per class on the 717-tile test set, unweighted:

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| None | 334 | 88.8% | 90.1% | 89.5% |
| Mesh / Mesh-Ring | 164 | 67.5% | 65.9% | 66.7% |
| Ring | 90 | 61.0% | 52.2% | 56.3% |
| Clod / Clod-Ring | 86 | 53.8% | 73.3% | 62.1% |
| Clod-Mesh / Clod-Mesh-Ring | 43 | 37.5% | 20.9% | 26.9% |

Performance tracks class support almost exactly. `None` is both the largest class and the most
visually distinct, and the model handles it well. The composite class `Clod-Mesh / Clod-Mesh-Ring`
is the weakest by a wide margin at 20.9% recall: it is the rarest class at 43 test samples, and it is
also the hardest one for a human to label, since it means "two or three patterns are entangled here".
Most of its misses go to `Clod / Clod-Ring`, a confusion the penalty matrix rates as low severity,
which is why the weighted score absorbs it.

The honest read: the model is reliable at separating lesion from background and competent on mesh,
but it has not solved the composite classes. Fixing that needs more labelled composite tiles, not a
different architecture.

### Segmentation at the mosaic level

Tile predictions painted back onto the mosaic give the clinician one image of the whole lesion.

<p align="center">
  <img src="assets/28-class-legend.jpg" alt="Colour legend for the five classes" width="55%">
</p>

<p align="center">
  <img src="assets/23-segmented-mosaic-2.jpg" alt="Ground truth and prediction for a mosaic labelled 98 percent, at 93.25 percent accuracy" width="100%">
</p>

<p align="center">
  <img src="assets/22-segmented-mosaic-1.jpg" alt="Ground truth and prediction for a second mosaic, at 89.06 percent accuracy" width="100%">
</p>

Two test mosaics, ground truth on the left and prediction on the right, at 93.3% and 89.1% tile
accuracy. The shape of the lesion survives intact in both. Where the model drifts it does so at the
boundaries, extending a `Mesh` field by a tile or two, rather than inventing structure in the middle
of the mole.

### Irregularity index

A fully coloured mosaic can still be too much information at once. A second layer flags
*disagreement between neighbours*: a 3 x 3 window slides across the mosaic, and where the nine tiles
carry three or more distinct predictions (ignoring `None`, which is background rather than a
pattern), the centre tile is circled in red.

<p align="center">
  <img src="assets/20-mosaic-prediction.jpg" alt="Predicted mosaic with class colours" width="49%">
  <img src="assets/21-mosaic-irregularities.jpg" alt="The same mosaic with irregularity regions circled in red" width="49%">
</p>

The circled band is where the lesion changes character. Heterogeneous, asymmetric regions are exactly
what clinicians are trained to treat as suspicious, so this reduces a wall of colour to a short list
of places to look. Feasibility of identifying pattern prototypes at the mosaic level was also
demonstrated, as a third output.

---

## Conclusions and future work

**Conclusions**

- **Clinical reasoning beat validation curves.** Merging eight classes into five was proposed by the
  medical team on clinical grounds and improved the model as a side effect. The largest single gain
  in the project came from a domain decision, not an architectural one.
- **The metric is part of the model.** Encoding error severity into both loss and evaluation changed
  what the network optimised for. A flat accuracy score would have hidden that most remaining errors
  are the tolerable kind.
- **Tile-level beats pixel-level here.** Not because it is technically superior, but because it
  matches the unit clinicians label and act on. The literature's pixel-level maps were harder for
  them to use.
- **Visualisation is not decoration.** The segmented mosaic and the irregularity overlay were what
  made the output usable at all, and they only converged after repeated rounds with the clinicians.
- **The bottleneck is labels, not capacity.** Every weak class is a small class.

**Future work**

- Extend to malignant lesions (melanoma) and measure performance there.
- Label in batches of four tiles rather than one, so spatial context enters the labels themselves.
- Lower the labelling decision threshold to surface many more composite tiles and enrich the rarest
  classes.
- Explore ensembles, attention, and class activation maps to expose *which* cellular structures drive
  a prediction, which is what would make the output auditable by a clinician.

---

## Notes and known limitations

This repository preserves the code as submitted. The first commit is byte-for-byte identical to the
submission; a second commit fixes five defects, one of which prevented `train.py` from running at
all. [NOTES.md](NOTES.md) documents both, along with what was deliberately left untouched.

- **The dataset is not here and cannot be shared.** The RCM images are patient-derived clinical data.
- **Absolute paths.** The code is written against one specific machine and pins mid-2022 library
  versions. Several APIs it uses have since been removed.
- **`drop_rate=0.85`** is unusually high dropout, a response to the small training set.
- **The `mesh` class is capped at 650 samples** during loading to blunt the imbalance, a default
  argument rather than a documented hyperparameter.

---

## Credits

Supervised by Dr. Eyal Katz (Afeka) and Prof. Alon Scope (Sheba Medical Center). Thanks to
Prof. Scope's research team for their work on data collection and labelling, and to project partner
Guy Kabiri.

Published as an academic portfolio record. All rights reserved.
