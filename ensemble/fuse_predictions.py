"""Fusion of U-Net and DeepLabV3 foreground probabilities for binary leaf segmentation.

Both models produce a foreground probability map:
    unet_prob     = softmax(unet_logits)[:, 1]      (or sigmoid of single channel)
    deeplab_prob  = softmax(deeplab_logits)[:, 1]

Probabilities are resized to a common spatial size (bilinear for probabilities,
NEAREST for the final binary mask), weighted-averaged with alpha, thresholded,
and evaluated against ground truth with foreground IoU/Dice/Precision/Recall.

alpha is the weight of U-Net: ensemble_prob = alpha*unet + (1-alpha)*deeplab.
"""

import numpy as np
from pathlib import Path
from PIL import Image

from predictors.predictor import Predictor
from data_generators.data_generator import load_split_names, PROJECT_ROOT
from utils.metrics import Evaluator


def _interp_np_same_size(arr, h, w):
    """Bilinear resize a (H0, W0) probability array to (h, w)."""
    return np.asarray(
        Image.fromarray(arr.astype(np.float32)).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    )


def _resize_pil(arr, size):
    return np.asarray(Image.fromarray(arr.astype(np.uint8)).resize(size, resample=Image.NEAREST))


class EnsembleFusion:
    """Loads DeepLabV3 via the project Predictor and a U-Net probability function.

    unet_prob_fn: callable(image_path) -> np.ndarray float32 (H, W) in [0, 1] at
                  original image resolution. Write it so U-Net applies its own
                  preprocessing and returns the foreground prob.
    """

    def __init__(self, config, deeplab_checkpoint, unet_prob_fn):
        self.config = config
        self.deeplab = Predictor(config, checkpoint_path=deeplab_checkpoint)
        self.unet_prob_fn = unet_prob_fn

    def predict_probabilities(self, image_path):
        """Return (unet_prob, deeplab_prob) at original image resolution."""
        deep_prob = self.deeplab.predict_probability(str(image_path))
        unet_prob = self.unet_prob_fn(str(image_path))
        unet_prob = np.asarray(unet_prob, dtype=np.float32)
        if unet_prob.shape != deep_prob.shape:
            h, w = deep_prob.shape
            unet_prob = _interp_np_same_size(unet_prob, h, w)
        return unet_prob, deep_prob

    def fuzz_probabilities(self, unet_prob, deep_prob, alpha=0.5):
        return alpha * unet_prob + (1.0 - alpha) * deep_prob

    def predict_mask(self, image_path, alpha=0.5, threshold=0.5, size=None):
        unet_prob, deep_prob = self.predict_probabilities(image_path)
        prob = self.fuzz_probabilities(unet_prob, deep_prob, alpha)
        mask = (prob >= threshold).astype(np.uint8)
        if size is not None:
            mask = _resize_pil(mask, size)
        return mask, prob


def evaluate_masks(gt_dir, stems, mask_fn):
    """Compute foreground IoU/Dice/Precision/Recall + Acc over stems.

    mask_fn(stem) -> uint8 (H, W) mask at the same size as ground truth.
    """
    ev = Evaluator(2)
    for stem in stems:
        gt = np.asarray(Image.open(Path(gt_dir) / f'{stem}.png'))
        if gt.ndim == 3:
            gt = np.any(gt != 0, axis=2)
        gt = gt.astype(np.int64)
        pred = mask_fn(stem).astype(np.int64)
        # align shapes
        if pred.shape != gt.shape:
            pred = _resize_pil(pred, (gt.shape[1], gt.shape[0]))
        ev.add_batch(gt, pred)

    cm = ev.confusion_matrix
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (cm[0, 0] + tp) / cm.sum() if cm.sum() > 0 else 0.0
    return {'IoU_leaf': float(iou), 'Dice_leaf': float(dice),
            'Precision': float(prec), 'Recall': float(rec), 'Acc': float(acc),
            'confusion': cm}


def search_alpha_threshold(config, deeplab_checkpoint, unet_prob_fn,
                            alphas=None, thresholds=None,
                            split_dir=None, data_dir=None):
    """Grid search alpha and threshold on the validation split, return best combo.

    Only the validation set is used here. The test set must stay untouched
    for the final evaluation.
    """
    alphas = alphas if alphas is not None else [round(a, 1) for a in np.arange(0, 1.01, 0.1)]
    thresholds = thresholds if thresholds is not None else [round(t, 5) for t in np.arange(0.3, 0.71, 0.05)]
    split_dir = Path(split_dir) if split_dir else PROJECT_ROOT / 'splits'
    data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / 'data'
    gt_dir = data_dir / 'masks'

    val_stems = load_split_names(split_dir, 'val')

    fuser = EnsembleFusion(config, deeplab_checkpoint, unet_prob_fn)
    gt = {}
    masks = {}
    for stem in val_stems:
        g = np.asarray(Image.open(gt_dir / f'{stem}.png'))
        if g.ndim == 3:
            g = np.any(g != 0, axis=2)
        gt[stem] = g.astype(np.int64)
        unet_p, deep_p = fuser.predict_probabilities(data_dir / 'imgs' / f'{stem}.png')
        masks[stem] = (unet_p, deep_p)

    best = None
    for alpha in alphas:
        for thr in thresholds:
            results = []
            for stem in val_stems:
                unet_p, deep_p = masks[stem]
                prob = alpha * unet_p + (1 - alpha) * deep_p
                pred = (prob >= thr).astype(np.int64)
                if pred.shape != gt[stem].shape:
                    pred = _resize_pil(pred, (gt[stem].shape[1], gt[stem].shape[0]))
                tp = int(((pred == 1) & (gt[stem] == 1)).sum())
                fp = int(((pred == 1) & (gt[stem] == 0)).sum())
                fn = int(((pred == 0) & (gt[stem] == 1)).sum())
                results.append((tp, fp, fn))
            tp = sum(r[0] for r in results)
            fp = sum(r[1] for r in results)
            fn = sum(r[2] for r in results)
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            if best is None or dice > best[0]:
                best = (dice, alpha, thr)

    return {'best_dice': best[0], 'alpha': best[1], 'threshold': best[2]}