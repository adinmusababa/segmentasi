import os
import random
from pathlib import Path
import argparse

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SegmentationDataset(Dataset):
    """Simple segmentation dataset that pairs images and masks.

    Masks are grayscale with class indices (0..nclass-1).
    Background is 0, other classes are 1..nclass-1.
    Returns sample dict: {'image': image_tensor, 'label': label_tensor}
    """

    def __init__(self, imgs_dir, masks_dir, crop_size=513, transforms=None, nclass=2):
        self.imgs_dir = Path(imgs_dir)
        self.masks_dir = Path(masks_dir)
        self.crop_size = crop_size
        self.nclass = nclass

        # find matching stems
        img_files = {p.stem: p for p in self.imgs_dir.glob('*.png')}
        mask_files = {p.stem: p for p in self.masks_dir.glob('*.png')}
        common = sorted(set(img_files.keys()) & set(mask_files.keys()))

        self.samples = [(img_files[s], mask_files[s]) for s in common]

        # transforms for image
        if transforms is not None:
            self.img_transforms = transforms
        else:
            self.img_transforms = T.Compose([
                T.Resize((self.crop_size, self.crop_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # process image
        image = self.img_transforms(image)

        # process mask -> keep class indices (0..nclass-1)
        mask = np.array(mask)
        if mask.ndim == 3:
            # If RGB, convert to grayscale (assuming single channel info)
            mask = np.mean(mask, axis=2).astype(np.uint8)

        # resize label to crop_size if necessary
        if mask.shape[0] != self.crop_size or mask.shape[1] != self.crop_size:
            mask = np.array(Image.fromarray(mask).resize((self.crop_size, self.crop_size), resample=Image.NEAREST))

        # Clamp to valid class range
        mask = np.clip(mask, 0, self.nclass - 1)

        label = torch.from_numpy(mask).long()

        return {'image': image, 'label': label}

    def shuffle_dataset(self):
        random.shuffle(self.samples)


def initialize_data_loader(config):
    """Create train/val/test DataLoaders.

    Heuristics:
    - If configs point to a dataset base path that exists, use it.
    - Otherwise fall back to project/data/imgs and project/data/masks.

    Returns: train_loader, val_loader, test_loader, nclass
    """
    # config values
    batch_size = int(config['training'].get('batch_size', 2))
    num_workers = int(config['training'].get('workers', 4))
    crop_size = int(config['image'].get('crop_size', config['image'].get('base_size', 513)))
    nclass = int(config['network'].get('num_classes', 2))  # from config

    # try config dataset base_path first
    cfg_base = config.get('dataset', {}).get('base_path', None)
    candidate_img = None
    candidate_mask = None

    # prefer repo data/ if present
    repo_imgs = PROJECT_ROOT / 'data' / 'imgs'
    repo_masks = PROJECT_ROOT / 'data' / 'masks'
    if repo_imgs.exists() and repo_masks.exists():
        candidate_img = repo_imgs
        candidate_mask = repo_masks
    elif cfg_base:
        base_path = (PROJECT_ROOT / cfg_base).resolve()
        # if config points to a dataset root containing imgs/masks
        if (base_path / 'imgs').exists() and (base_path / 'masks').exists():
            candidate_img = base_path / 'imgs'
            candidate_mask = base_path / 'masks'

    if candidate_img is None or candidate_mask is None:
        raise RuntimeError('Could not find dataset dirs. Put images in data/imgs/ and color masks in data/masks/ or update config.dataset.base_path')

    # list paired stems
    img_files = sorted([p for p in (candidate_img).glob('*.png')])
    mask_files = sorted([p for p in (candidate_mask).glob('*.png')])
    stems = sorted({p.stem for p in img_files} & {p.stem for p in mask_files})

    if len(stems) == 0:
        raise RuntimeError(f'No matching image/mask pairs found in {candidate_img} and {candidate_mask}')

    # build full list of pairs
    pairs = [(candidate_img / f'{s}.png', candidate_mask / f'{s}.png') for s in stems]

    # split
    random.seed(int(config.get('seed', 1)))
    random.shuffle(pairs)
    n = len(pairs)
    val_count = max(1, int(0.1 * n))
    test_count = max(1, int(0.1 * n))
    train_pairs = pairs[: n - val_count - test_count]
    val_pairs = pairs[n - val_count - test_count: n - test_count]
    test_pairs = pairs[n - test_count:]

    # create dataset objects (we'll create small wrapper datasets using the selected pairs)
    train_ds = SegmentationDatasetForPairs(train_pairs, crop_size=crop_size, nclass=nclass)
    val_ds = SegmentationDatasetForPairs(val_pairs, crop_size=crop_size, nclass=nclass)
    test_ds = SegmentationDatasetForPairs(test_pairs, crop_size=crop_size, nclass=nclass)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers//2))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers//2))

    return train_loader, val_loader, test_loader, nclass


class SegmentationDatasetForPairs(Dataset):
    """Helper dataset that accepts explicit (img,mask) pairs list.
    Implements same interface as SegmentationDataset.
    Supports multi-class masks (grayscale with class IDs 0..nclass-1).
    """
    def __init__(self, pairs, crop_size=513, nclass=2):
        self.samples = pairs
        self.crop_size = crop_size
        self.nclass = nclass
        self.img_transforms = T.Compose([
            T.Resize((self.crop_size, self.crop_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = self.img_transforms(image)

        # Process mask: keep as class indices (0..nclass-1)
        mask = np.array(mask)
        if mask.ndim == 3:
            # If RGB, convert to grayscale first (assuming single channel info)
            mask = np.mean(mask, axis=2).astype(np.uint8)

        # Mask values should already be class indices (0..nclass-1)
        # Just resize if needed
        if mask.shape[0] != self.crop_size or mask.shape[1] != self.crop_size:
            mask = np.array(Image.fromarray(mask).resize((self.crop_size, self.crop_size), resample=Image.NEAREST))

        # Clamp to valid class range
        mask = np.clip(mask, 0, self.nclass - 1)

        label = torch.from_numpy(mask).long()
        return {'image': image, 'label': label}

    def shuffle_dataset(self):
        random.shuffle(self.samples)


def convert_masks_to_single_class(masks_dir, out_dir):
    masks_dir = Path(masks_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(masks_dir.glob('*.png')):
        mask = np.array(Image.open(p))
        if mask.ndim == 3:
            leaf = np.any(mask != 0, axis=2).astype(np.uint8) * 255
        else:
            leaf = (mask != 0).astype(np.uint8) * 255
        out_p = out_dir / p.name
        Image.fromarray(leaf).convert('L').save(out_p)

    print(f'Converted {len(list(masks_dir.glob("*.png")))} masks -> {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert', action='store_true', help='Convert color masks -> single class masks in data/masks_encoded')
    parser.add_argument('--masks', help='masks directory (default: data/masks)', default=str(PROJECT_ROOT / 'data' / 'masks'))
    parser.add_argument('--out', help='output masks dir (default: data/masks_encoded)', default=str(PROJECT_ROOT / 'data' / 'masks_encoded'))
    args = parser.parse_args()

    if args.convert:
        convert_masks_to_single_class(args.masks, args.out)
    else:
        print('This module provides initialize_data_loader(config). Use --convert to pre-encode masks.')
