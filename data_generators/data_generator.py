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


def read_mask_binary(mask_path, size):
    """Read mask and convert to binary (0/1) int64.

    Instance masks (values 1,2,3,...) indicate leaf identity, not semantic
    class. Correct conversion is (mask != 0), never mean over RGB channels.
    """
    mask = np.asarray(Image.open(mask_path))

    if mask.ndim == 3:
        mask = np.any(mask != 0, axis=2)
    else:
        mask = mask != 0

    mask = mask.astype(np.int64)
    mask = np.array(
        Image.fromarray(mask.astype(np.uint8)).resize(size, resample=Image.NEAREST),
        copy=True,
    )
    return mask


class SegmentationDataset(Dataset):
    """Simple segmentation dataset that pairs images and masks.

    Binary masks with class indices 0 (background) and 1 (leaf).
    Returns sample dict: {'image': image_tensor, 'label': label_tensor}
    """

    def __init__(self, imgs_dir, masks_dir, crop_size=256, transforms=None, nclass=2):
        self.imgs_dir = Path(imgs_dir)
        self.masks_dir = Path(masks_dir)
        self.crop_size = crop_size
        self.nclass = nclass
        self.size = (crop_size, crop_size)

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
                T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.img_transforms(image)

        # process mask -> binary 0/1
        mask = read_mask_binary(mask_path, self.size)
        assert set(np.unique(mask)).issubset({0, 1}), \
            f"Mask {mask_path.name} contains non-binary values: {np.unique(mask)}"

        label = torch.from_numpy(mask).long()

        return {'image': image, 'label': label}

    def shuffle_dataset(self):
        random.shuffle(self.samples)


def load_split_names(split_dir, split_name):
    """Load stem names from splits/{split_name}.txt, one stem per line."""
    split_file = Path(split_dir) / f'{split_name}.txt'
    if not split_file.exists():
        raise RuntimeError(f'Split manifest not found: {split_file}')
    names = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
    if not names:
        raise RuntimeError(f'Split manifest is empty: {split_file}')
    return names


def make_split_manifests(data_dir, split_dir, frac=None, seed=42):
    """Create train/val/test manifests once from data/imgs + data/masks.

    Stems are matched (image and mask must share the same base name).
    Default split: 70% train, 15% val, 15% test.
    """
    imgs_dir = Path(data_dir) / 'imgs'
    masks_dir = Path(data_dir) / 'masks'
    img_stems = {p.stem for p in imgs_dir.glob('*.png')}
    mask_stems = {p.stem for p in masks_dir.glob('*.png')}
    unmatched = (img_stems - mask_stems) | (mask_stems - img_stems)
    if unmatched:
        names = ', '.join(sorted(unmatched)[:10])
        raise RuntimeError(f'Unmatched image/mask files (fix dataset first): {names}')

    stems = sorted(img_stems & mask_stems)
    if not stems:
        raise RuntimeError('No image/mask pairs found.')

    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    val_count = max(1, int(round(0.15 * n)))
    test_count = max(1, int(round(0.15 * n)))
    # ensure at least 1 train sample
    train_count = max(1, n - val_count - test_count)
    val_count = min(val_count, n - 1)
    test_count = min(test_count, n - train_count - val_count)

    split_map = {
        'train': stems[:train_count],
        'val': stems[train_count:train_count + val_count],
        'test': stems[train_count + val_count:],
    }

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, items in split_map.items():
        out = split_dir / f'{name}.txt'
        out.write_text('\n'.join(items) + '\n')
        print(f'  {name}: {len(items)} -> {out}')

    return split_map


def initialize_data_loader(config):
    """Create train/val/test DataLoaders.

    Splits are read from manifest files at splits/{train,val,test}.txt so the
    same train/val/test lists are shared with the U-Net pipeline. If manifests
    are missing, they are created once with a fixed seed.

    Returns: train_loader, val_loader, test_loader, nclass
    """
    # config values
    batch_size = int(config['training'].get('batch_size', 2))
    num_workers = int(config['training'].get('workers', 0))
    crop_size = int(config['image'].get('crop_size', config['image'].get('base_size', 256)))
    nclass = int(config['network'].get('num_classes', 2))  # from config
    base_path = PROJECT_ROOT / 'data'

    candidate_img = base_path / 'imgs'
    candidate_mask = base_path / 'masks'
    if not (candidate_img.exists() and candidate_mask.exists()):
        raise RuntimeError('Could not find dataset dirs. Put images in data/imgs/ and color masks in data/masks/ or update config.dataset.base_path')

    # ensure split manifests exist (shared with U-Net pipeline)
    split_dir = PROJECT_ROOT / 'splits'
    if not all((split_dir / f'{name}.txt').exists() for name in ('train', 'val', 'test')):
        print('Split manifests not found, creating them once at:', split_dir)
        make_split_manifests(base_path, split_dir, seed=int(config.get('seed', 42)))

    train_stems = load_split_names(split_dir, 'train')
    val_stems = load_split_names(split_dir, 'val')
    test_stems = load_split_names(split_dir, 'test')

    def pairs_from(stems):
        pairs = []
        missing = []
        for s in stems:
            ip = candidate_img / f'{s}.png'
            mp = candidate_mask / f'{s}.png'
            if ip.exists() and mp.exists():
                pairs.append((ip, mp))
            else:
                missing.append(s)
        if missing:
            raise RuntimeError(f'Files listed in split manifest missing from dataset: {missing[:10]}')
        return pairs

    train_pairs = pairs_from(train_stems)
    val_pairs = pairs_from(val_stems)
    test_pairs = pairs_from(test_stems)

    # create dataset objects
    train_ds = SegmentationDatasetForPairs(train_pairs, crop_size=crop_size, nclass=nclass)
    val_ds = SegmentationDatasetForPairs(val_pairs, crop_size=crop_size, nclass=nclass)
    test_ds = SegmentationDatasetForPairs(test_pairs, crop_size=crop_size, nclass=nclass)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers // 2))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers // 2))

    return train_loader, val_loader, test_loader, nclass


class SegmentationDatasetForPairs(Dataset):
    """Helper dataset that accepts explicit (img,mask) pairs list.
    Implements same interface as SegmentationDataset.
    Binary masks with class indices 0 (background) and 1 (leaf).
    """
    def __init__(self, pairs, crop_size=256, nclass=2):
        self.samples = pairs
        self.crop_size = crop_size
        self.nclass = nclass
        self.size = (crop_size, crop_size)
        self.img_transforms = T.Compose([
            T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.img_transforms(image)

        # process mask -> binary 0/1
        mask = read_mask_binary(mask_path, self.size)
        assert set(np.unique(mask)).issubset({0, 1}), \
            f"Mask {mask_path.name} contains non-binary values: {np.unique(mask)}"

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
    parser.add_argument('--make-splits', action='store_true', help='Create split manifests once from data/')
    args = parser.parse_args()

    if args.convert:
        convert_masks_to_single_class(args.masks, args.out)
    elif args.make_splits:
        make_split_manifests(PROJECT_ROOT / 'data', PROJECT_ROOT / 'splits')
    else:
        print('This module provides initialize_data_loader(config). Use --make-splits to create split manifests or --convert to pre-encode masks.')