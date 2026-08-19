"""
Download script for Plant Phenotyping Dataset from Kaggle.

This script downloads the dataset and organizes it into:
  - data/imgs/   : RGB images (renamed to {plant_id}.png)
  - data/masks/  : Label masks (renamed to matching {plant_id}.png)

Usage:
    python data/download_dataset.py

The script uses kagglehub to download the dataset from Kaggle.
Images and masks are renamed so they share the same base name,
which is required by the UNet training pipeline (BasicDataset).
"""

import sys
import shutil
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
# Target directories (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_IMGS_DIR = PROJECT_ROOT / 'data' / 'imgs'
DATA_MASKS_DIR = PROJECT_ROOT / 'data' / 'masks'

# Kaggle dataset identifier
DATASET_NAME = 'pillisiddharth/plant-phenotyping-dataset'

# Which sub-datasets to include
# Each entry: (source_subfolder, label_pattern)
SUBSETS = [
    ('Plant/Ara2012', 'label'),
    ('Plant/Ara2013-Canon', 'label'),
    ('Plant/Tobacco', 'label'),
]

# ── Step 1: Download via kagglehub ─────────────────────────────────────────────


def _find_dataset_root(download_path: Path) -> Path:
    """Find the actual root containing the Plant/ subdirectories.

    kagglehub wraps the dataset in version folders and the data itself
    may be nested inside a 'Plant_Phenotyping_Datasets' wrapper folder.
    """
    for candidate in download_path.rglob('Plant/Ara2012'):
        if candidate.is_dir():
            return candidate.parent.parent
    return download_path


def download_dataset():
    """Download dataset from Kaggle using kagglehub."""
    print('=' * 60)
    print(f'Downloading dataset: {DATASET_NAME}')
    print('=' * 60)

    try:
        import kagglehub
    except ImportError:
        print('Error: kagglehub is not installed.')
        print('Install it with: pip install kagglehub')
        sys.exit(1)

    print('Downloading... (this may take a while for large datasets)')
    download_path = kagglehub.dataset_download(DATASET_NAME)
    download_path = Path(download_path)
    print(f'Dataset downloaded to: {download_path}')

    download_path = _find_dataset_root(download_path)
    print(f'Dataset root: {download_path}')
    return download_path


def is_dataset_ready() -> bool:
    """Check if data/imgs/ and data/masks/ already have a complete matching set.

    Returns True only if both directories contain PNG files and every image
    has a corresponding mask (i.e. the dataset is already organized).
    """
    if not DATA_IMGS_DIR.is_dir() or not DATA_MASKS_DIR.is_dir():
        return False

    img_stems = {f.stem for f in DATA_IMGS_DIR.glob('*.png')}
    mask_stems = {f.stem for f in DATA_MASKS_DIR.glob('*.png')}

    if not img_stems or not mask_stems:
        return False

    # Every image must have a matching mask, and vice versa
    if img_stems != mask_stems:
        return False

    return True


# ── Step 2: Organize into imgs/ and masks/ ─────────────────────────────────────


def organize_dataset(download_path: Path):
    """Copy RGB images to data/imgs/ and label masks to data/masks/.

    Images and masks are renamed to share the same stem:
      - Source:  ara2012_plant001_rgb.png  →  data/imgs/ara2012_plant001.png
      - Source:  ara2012_plant001_label.png →  data/masks/ara2012_plant001.png

    The BasicDataset in the UNet pipeline requires matching filenames
    between data/imgs/ and data/masks/ (different extensions only).
    """
    print('\n' + '=' * 60)
    print('Organizing dataset into data/imgs/ and data/masks/')
    print('=' * 60)

    # Create target directories
    DATA_IMGS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    total_skipped = 0

    for subset_rel, mask_type in SUBSETS:
        subset_path = download_path / subset_rel
        if not subset_path.exists():
            print(f'  [WARN] Subset not found: {subset_rel}')
            continue

        print(f'\n  Processing: {subset_rel}')

        # Find all RGB images
        rgb_files = sorted(subset_path.glob('*_rgb.png'))
        if len(rgb_files) == 0:
            print(f'    No RGB files found, skipping.')
            continue

        print(f'    Found {len(rgb_files)} RGB images')

        copied = 0
        skipped = 0
        for rgb_path in rgb_files:
            # Derive shared base name (remove _rgb suffix)
            base_name = rgb_path.stem.replace('_rgb', '')
            label_path = subset_path / f'{base_name}_{mask_type}.png'

            if not label_path.exists():
                skipped += 1
                continue

            # Copy & rename: strip _rgb and _label so both share the same stem
            dest_img = DATA_IMGS_DIR / f'{base_name}.png'
            shutil.copy2(rgb_path, dest_img)

            dest_mask = DATA_MASKS_DIR / f'{base_name}.png'
            shutil.copy2(label_path, dest_mask)

            copied += 1

        total_copied += copied
        total_skipped += skipped
        print(f'    Copied: {copied} image/mask pairs, Skipped: {skipped} (no label)')

    print(f'\n  Total: {total_copied} image/mask pairs copied')
    if total_skipped > 0:
        print(f'  Skipped: {total_skipped} images without matching labels')

    return total_copied


# ── Step 3: Verify ─────────────────────────────────────────────────────────────


def verify_dataset():
    """Verify that images and masks match."""
    print('\n' + '=' * 60)
    print('Verifying dataset')
    print('=' * 60)

    img_stems = {f.stem for f in DATA_IMGS_DIR.glob('*.png')}
    mask_stems = {f.stem for f in DATA_MASKS_DIR.glob('*.png')}

    imgs_without_masks = img_stems - mask_stems
    masks_without_imgs = mask_stems - img_stems

    if imgs_without_masks:
        print(f'  Warning: {len(imgs_without_masks)} images have no matching mask')
        for name in sorted(list(imgs_without_masks))[:5]:
            print(f'    - {name}')

    if masks_without_imgs:
        print(f'  Warning: {len(masks_without_imgs)} masks have no matching image')
        for name in sorted(list(masks_without_imgs))[:5]:
            print(f'    - {name}')

    matched = len(img_stems & mask_stems)
    print(f'  Images: {len(img_stems)}')
    print(f'  Masks:  {len(mask_stems)}')
    print(f'  Matched pairs: {matched}')

    return matched


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    print('Plant Phenotyping Dataset Downloader')
    print('====================================\n')

    # Guard: skip download & organize if data is already complete
    if is_dataset_ready():
        print('✓ Dataset sudah lengkap di data/imgs/ dan data/masks/ — tidak perlu download ulang.\n')
        verify_dataset()
        return

    # Step 1: Download
    download_path = download_dataset()

    # Step 2: Organize
    total = organize_dataset(download_path)

    if total == 0:
        print('\nNo data was copied. Something may be wrong.')
        sys.exit(1)

    # Step 3: Verify
    matched = verify_dataset()

    print('\n' + '=' * 60)
    print('Done! Dataset is ready for training.')
    print(f'  Images: {DATA_IMGS_DIR}  ({total} files)')
    print(f'  Masks:  {DATA_MASKS_DIR}  ({total} files)')
    print(f'  Matched pairs: {matched}')
    print('=' * 60)

    print('\nTo train the model, run:')
    print('  python train.py')


if __name__ == '__main__':
    main()
    