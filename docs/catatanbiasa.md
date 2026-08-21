Hal yang harus ditinjau/atur

1. Config:
• configs/config.yml: set network.num_classes: 2 (background + leaf).
• set training.weights_initialization.use_pretrained_weights: false (atau berikan checkpoint valid pada restore_from).
• sesuaikan training.batch_size dan training.workers sesuai GPU/CPU.
• set dataset.base_path ke './data' atau biarkan (loader pakai data/imgs & data/masks jika ada).
• jika pakai GPU, use_cuda: true; jika tidak, false.
2. Validasi file:
• Pastikan setiap image di data\imgs*.png punya mask dengan stem sama di data\masks*.png.
• Pastikan background pada mask memang zero (hitam). Loader menganggap “non-zero = leaf”.
3. Quick checks (jalankan di environment Python Anda):
• Hitung pasangan image/mask:
python - <<PY
from pathlib import Path
imgs= {p.stem for p in Path('data/imgs').glob('.png')}
masks= {p.stem for p in Path('data/masks').glob('.png')}
print('images',len(imgs),'masks',len(masks),'pairs',len(imgs & masks))
PY
• Cek warna unik contoh mask (pastikan background is 0):
python - <<PY
import numpy as np
from PIL import Image
m=np.array(Image.open('data/masks/ara2012_plant001.png'))
print('shape',m.shape,'unique colors',np.unique(m.reshape(-1,m.shape[-1]) if m.ndim==3 else m.reshape(-1),axis=0)[:10])
PY
4. Gunakan data loader baru (sudah ditambahkan): trainers.Trainer memanggil initialize_data_loader(config) — tidak perlu perubahan kode lain.
5. Direktori output: pastikan ./experiments, ./snapshots, ./tensorboard ada/ditulis.