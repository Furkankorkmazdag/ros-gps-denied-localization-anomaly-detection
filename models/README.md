# Model checkpoint

The trained `model.ckpt` is intentionally not included because it is approximately 281 MB and exceeds GitHub's 100 MB per-file limit.

Train a new model using `scripts/train_patchcore.py` and normal training images. Copy the resulting checkpoint to `models/model.ckpt`, or pass an absolute checkpoint path through the `checkpoint` argument of `start.launch`.

The checkpoint must match the `Patchcore` architecture and the anomalib version used by the inference script.
