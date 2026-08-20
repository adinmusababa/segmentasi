# Rancangan Penyesuaian DeepLabV3 untuk Ensemble dengan U-Net

Dokumen ini menjadi acuan persiapan dataset, training, evaluasi, dan integrasi
DeepLabV3 dengan model U-Net pada repository:

`https://github.com/HaikalFK/segmentasi-unet`

Tujuan akhirnya adalah membangun ensemble untuk **segmentasi biner daun**:

- `0` = background
- `1` = leaf/daun

Model U-Net yang sudah tersedia tidak perlu diubah. Penyesuaian dilakukan pada
pipeline DeepLabV3 dan pada tahap inference/ensemble agar keluaran kedua model
dapat dibandingkan dan digabungkan secara konsisten.



---

## 1. Keputusan desain yang harus digunakan

| Komponen | Keputusan |
|---|---|
| Tugas | Semantic segmentation biner |
| Kelas | 2 kelas: background dan leaf |
| Sumber label | Mask instance daun yang dikonversi menjadi biner |
| Ukuran input training | 256 x 256 |
| Format gambar | RGB |
| Format mask | PNG, satu channel, nilai 0 atau 1 |
| Model pertama | U-Net yang sudah dilatih |
| Model kedua | DeepLabV3 dengan `num_classes=2` |
| Ensemble | menyesuaikan |
| Pemilihan threshold dan bobot | Validation set, bukan test set |
| Evaluasi akhir | Test set yang sama untuk kedua model |

Checkpoint U-Net **tidak dapat dimuat langsung** ke arsitektur DeepLabV3.
Keduanya dilatih sebagai model terpisah, kemudian hasil prediksinya digabungkan
pada tahap inference.

---

## 2. Struktur dataset yang wajib digunakan

Gunakan struktur data yang sama untuk U-Net dan DeepLabV3:

```text
data/
├── imgs/
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
└── masks/
    ├── sample_001.png
    ├── sample_002.png
    └── ...
```

Ketentuan:

1. Nama dasar gambar dan mask harus sama.
2. Setiap gambar harus memiliki satu mask yang sesuai.
3. Gambar dibaca sebagai RGB.
4. Mask dibaca sebagai grayscale atau array integer.
5. Mask tidak boleh menggunakan interpolasi bilinear.
6. Mask akhir hanya boleh memiliki nilai `0` dan `1`.
7. Jangan mencampur gambar dari dataset lain pada folder tersebut.
8. Jangan memasukkan file hasil prediksi ke folder training.

Repository U-Net menggunakan gambar PNG dan melakukan konversi mask instance
ke mask biner dengan aturan:

```python
binary_mask = (mask > 0).astype(np.int64)
```

Aturan yang sama harus diterapkan pada DeepLabV3. Nilai instance `1, 2, 3, ...`
menunjukkan identitas daun, bukan kelas semantic yang berbeda.

### 2.1 Validasi pasangan gambar dan mask

Sebelum training, jumlah pasangan harus diperiksa. Secara konsep:

```text
jumlah gambar = jumlah mask = jumlah pasangan
nama gambar tanpa ekstensi = nama mask tanpa ekstensi
ukuran gambar = ukuran mask sebelum resize
```

Jika terdapat file tanpa pasangan, proses training harus dihentikan dan masalah
data diperbaiki terlebih dahulu.

---

## 3. Gunakan split dataset yang sama

U-Net dan DeepLabV3 harus menggunakan daftar train, validation, dan test yang
identik. Jangan membiarkan masing-masing repository membuat split secara
independen.

Buat manifest seperti berikut:

```text
splits/
├── train.txt
├── val.txt
└── test.txt
```

Setiap baris berisi stem file, tanpa ekstensi:

```text
sample_001
sample_002
sample_003
```

Contoh proporsi:

```text
70% train
15% validation
15% test
```

Proporsi boleh disesuaikan dengan jumlah data, tetapi harus tetap sama untuk
kedua model.

### 3.1 Aturan split

- Split dilakukan satu kali sebelum training.
- Gunakan seed tetap.
- Simpan manifest ke repository atau arsip penelitian.
- Jangan mengubah split ketika membandingkan U-Net, DeepLabV3, dan ensemble.
- Test set hanya digunakan untuk evaluasi final.
- Threshold ensemble dan bobot model hanya boleh dicari dari validation set.

Hal ini mencegah perbedaan data evaluasi dan data leakage pada penelitian.

---

## 4. Penyesuaian preprocessing

### 4.1 Preprocessing U-Net yang sudah ada

U-Net menggunakan:

```text
resize gambar ke 256 x 256 dengan BICUBIC
resize mask ke 256 x 256 dengan NEAREST
ubah RGB menjadi channel-first
nilai pixel gambar dibagi 255
mask dikonversi menjadi 0/1
```

Pipeline ini tidak perlu diubah jika model U-Net sudah memberikan hasil yang
baik.

### 4.2 Preprocessing DeepLabV3

DeepLabV3 harus menggunakan ukuran spasial yang sama:

```text
resize gambar ke 256 x 256
resize mask ke 256 x 256
gambar menggunakan RGB
gambar dinormalisasi menggunakan mean dan std yang konsisten dengan training
mask menggunakan NEAREST
mask diubah menjadi long tensor
```

DeepLabV3 pada project ini menggunakan normalisasi ImageNet:

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

Normalisasi tersebut boleh tetap berbeda dari U-Net. Model memang dapat
memiliki preprocessing internal yang berbeda selama setiap model dilatih dan
di-inference dengan preprocessing yang sama. Yang harus sama adalah gambar
yang masuk, geometri resize, orientasi, dan ukuran keluaran.

### 4.3 Aturan mask DeepLabV3

Pada `data_generators/data_generator.py`, mask perlu diproses secara eksplisit:

```python
mask = np.asarray(Image.open(mask_path))

if mask.ndim == 3:
    mask = np.any(mask != 0, axis=2)
else:
    mask = mask != 0

mask = mask.astype(np.int64)
mask = np.asarray(
    Image.fromarray(mask.astype(np.uint8)).resize(
        (256, 256),
        resample=Image.NEAREST
    )
)

label = torch.from_numpy(mask).long()
```

Jangan mengubah mask RGB menjadi grayscale menggunakan rata-rata channel,
karena cara tersebut dapat menghasilkan nilai label yang tidak bermakna.
Konversi yang benar adalah `mask != 0`.

---

## 5. Konfigurasi DeepLabV3 yang disarankan

Sesuaikan `configs/config.yml` menjadi dasar berikut:

```yaml
dataset:
    base_path: "./data"
    dataset_name: "plant_phenotyping"

image:
    out_stride: 16
    base_size: 256
    crop_size: 256

network:
    backbone: "resnet"
    sync_bn: false
    freeze_bn: true
    use_cuda: true
    num_classes: 2

training:
    workers: 0
    loss_type: "ce"
    epochs: 50
    start_epoch: 0
    batch_size: 2
    use_balanced_weights: true
    lr: 0.0005
    lr_scheduler: "poly"
    momentum: 0.9
    weight_decay: 0.0005
    nesterov: false
    no_val: false
    val_interval: 1
    weights_initialization:
        use_pretrained_weights: false
        restore_from: ""
    train_on_subset:
        enabled: false
        dataset_fraction: 1.0
    tensorboard:
        enabled: false
        log_dir: "./tensorboard/"
```

### 5.1 Penjelasan perubahan penting

#### `num_classes: 2`

U-Net menghasilkan dua kelas. DeepLabV3 juga wajib menghasilkan dua channel
logit:

```text
output shape = (batch, 2, height, width)
```

#### `crop_size: 256`

Ukuran ini mengikuti U-Net. DeepLabV3 tetap dapat menggunakan ukuran lain saat
eksperimen, tetapi hasilnya harus di-resize sebelum ensemble. Menggunakan ukuran
yang sama sejak training mengurangi perbedaan geometri prediksi.

#### `freeze_bn: true`

Batch size kecil dapat membuat statistik BatchNorm tidak stabil. Jika GPU
memungkinkan batch size besar dan hasil eksperimen menunjukkan BatchNorm
stabil, opsi ini dapat diuji sebagai eksperimen terpisah.

#### `use_balanced_weights: true`

Background biasanya jauh lebih banyak daripada daun. Bobot kelas membantu
mencegah model memprediksi semua pixel sebagai background.

#### `train_on_subset: false`

Training ensemble harus menggunakan seluruh train set. Konfigurasi subset yang
aktif tidak diperlukan untuk training final.

#### Checkpoint

Checkpoint lama dengan `num_classes: 14` tidak kompatibel dengan konfigurasi
dua kelas. Setelah mengubah jumlah kelas, mulai training baru atau gunakan
pretrained backbone ImageNet saja.

---

## 6. Loss function dan strategi training

### 6.1 Minimum yang harus dilakukan

Gunakan weighted Cross Entropy:

```yaml
training:
    loss_type: "ce"
    use_balanced_weights: true
```

### 6.2 Rekomendasi utama

Karena U-Net menggunakan kombinasi loss untuk mengatasi ketidakseimbangan
kelas, DeepLabV3 sebaiknya menggunakan:

```text
Loss = Weighted Cross Entropy + Dice Loss
```

Weighted Cross Entropy memberikan sinyal pixel-level, sedangkan Dice Loss
mengoptimalkan overlap area daun. Implementasi dapat ditambahkan pada
`losses/loss.py`, dengan target berbentuk:

```text
target shape = (B, H, W)
target values = 0 atau 1
logit shape = (B, 2, H, W)
```

Contoh formula:

```python
ce_loss = cross_entropy(logits, target, weight=class_weights)
prob_leaf = torch.softmax(logits, dim=1)[:, 1]
dice_loss = 1 - dice_coefficient(prob_leaf, target)
loss = ce_loss + dice_loss
```

Gunakan loss yang sama untuk seluruh eksperimen DeepLabV3 agar perbandingan
hasil tetap adil.

### 6.3 Hal yang tidak perlu disamakan

Optimizer dan learning rate U-Net tidak harus sama dengan DeepLabV3. U-Net
referensi menggunakan RMSprop dengan learning rate kecil, sedangkan project
DeepLabV3 menggunakan SGD. Keduanya boleh berbeda karena karakteristik
arsitekturnya berbeda.

Yang harus dicatat dalam laporan:

- optimizer
- learning rate
- scheduler
- batch size
- jumlah epoch
- loss function
- pretrained weight
- seed
- split dataset

---

## 7. Proses training DeepLabV3

Urutan training yang disarankan:

1. Pastikan manifest split sudah dibuat.
2. Pastikan semua mask bernilai 0/1.
3. Jalankan pemeriksaan pasangan gambar dan mask.
4. Uji satu batch dari DataLoader.
5. Pastikan bentuk tensor:
   ```text
   image = (B, 3, 256, 256)
   label = (B, 256, 256)
   ```
6. Pastikan label hanya berisi `{0, 1}`.
7. Jalankan training DeepLabV3.
8. Simpan checkpoint terbaik berdasarkan validation Dice atau validation IoU.
9. Jangan memilih checkpoint berdasarkan test set.
10. Simpan konfigurasi dan hasil training bersama checkpoint.

Periksa visualisasi minimal untuk:

- gambar asli
- ground-truth biner
- prediksi U-Net
- prediksi DeepLabV3
- area false positive
- area false negative

---

## 8. Evaluasi individual sebelum ensemble

U-Net dan DeepLabV3 harus dievaluasi secara terpisah pada test set yang sama.

Metric minimum:

| Metric | Keterangan |
|---|---|
| IoU foreground | Overlap area daun |
| Dice/F1 foreground | Kesamaan area daun |
| Precision | Ketepatan pixel yang diprediksi sebagai daun |
| Recall | Kemampuan menemukan pixel daun |
| Pixel accuracy | Akurasi semua pixel |
| Confusion matrix | Rincian TP, TN, FP, FN |

Untuk penelitian daun, jangan hanya menggunakan pixel accuracy. Nilai tersebut
dapat terlihat tinggi walaupun model memprediksi sebagian besar gambar sebagai
background.

Laporkan setidaknya:

```text
U-Net:
  IoU, Dice, Precision, Recall

DeepLabV3:
  IoU, Dice, Precision, Recall

Ensemble:
  IoU, Dice, Precision, Recall
```

---

## 9. Format output yang diperlukan untuk ensemble

Jangan langsung menggabungkan hasil `argmax` kedua model. Simpan probabilitas
foreground dari masing-masing model.

### 9.1 Output U-Net

```python
unet_logits = unet(image)
unet_prob = torch.softmax(unet_logits, dim=1)[:, 1]
```

### 9.2 Output DeepLabV3

```python
deeplab_logits = deeplab(image)
deeplab_prob = torch.softmax(deeplab_logits, dim=1)[:, 1]
```

Jika salah satu model hanya menghasilkan satu channel sigmoid, gunakan:

```python
prob_leaf = torch.sigmoid(logit)
```

Namun, untuk menjaga format yang sama dengan DeepLabV3, dua channel dengan
softmax lebih mudah digunakan.

### 9.3 Penyamaan ukuran output

Sebelum penggabungan:

```python
unet_prob = torch.nn.functional.interpolate(
    unet_prob.unsqueeze(1),
    size=deeplab_prob.shape[-2:],
    mode="bilinear",
    align_corners=False
).squeeze(1)
```

Untuk mask biner akhir, gunakan interpolasi `NEAREST`, bukan bilinear.

---

## 10. Metode ensemble yang direkomendasikan

Gunakan weighted probability averaging:

```python
ensemble_prob = (
    alpha * unet_prob
    + (1.0 - alpha) * deeplab_prob
)

ensemble_mask = (ensemble_prob >= threshold).to(torch.uint8)
```

Parameter awal:

```text
alpha = 0.5
threshold = 0.5
```

Interpretasi:

- `alpha` besar berarti lebih percaya U-Net.
- `alpha` kecil berarti lebih percaya DeepLabV3.
- `threshold` mengatur keseimbangan false positive dan false negative.

### 10.1 Pencarian bobot dan threshold

Gunakan validation set untuk mencoba kombinasi:

```text
alpha:
  0.0, 0.1, 0.2, ..., 1.0

threshold:
  0.3, 0.35, 0.4, ..., 0.7
```

Pilih kombinasi dengan Dice atau IoU terbaik pada validation set. Setelah
parameter ditetapkan, jalankan satu kali evaluasi final pada test set.

Jangan mencari `alpha` atau `threshold` menggunakan test set karena hal tersebut
menjadikan test set ikut memengaruhi desain model.

### 10.2 Majority voting

Majority voting tidak menjadi pilihan utama untuk dua model karena hasilnya
hanya menjadi aturan biner. Jika tetap digunakan, aturan dapat berupa:

```python
ensemble_mask = ((unet_mask + deeplab_mask) >= 1).to(torch.uint8)
```

Metode ini kurang disarankan karena tidak menggunakan confidence/probabilitas
model.

---

## 11. Pencegahan masalah saat ensemble

### 11.1 Prediksi tampak bergeser

Penyebab umum:

- resize berbeda
- crop berbeda
- orientasi gambar berbeda
- padding berbeda
- mask menggunakan interpolasi yang salah

Solusi: gunakan transformasi geometrik yang sama dan pastikan ukuran prediksi
identik sebelum penggabungan.

### 11. Hasil ensemble semuanya background

Periksa:

- mask training benar-benar mengandung nilai `1`
- `num_classes` bernilai `2`
- class weight aktif
- threshold tidak terlalu tinggi
- probabilitas yang digunakan adalah channel foreground
- model tidak collapse ke background

### 11. Hasil ensemble semuanya daun

Periksa:

- threshold terlalu rendah
- mask ground-truth salah dikonversi
- normalisasi inference berbeda dari training
- probabilitas channel yang dipakai salah

### 11. Checkpoint tidak dapat dimuat

Periksa kesesuaian:

- jumlah kelas
- backbone
- output stride
- struktur `state_dict`
- penggunaan `DataParallel`

Checkpoint DeepLabV3 dengan output 14 kelas harus dianggap tidak berlaku untuk
model biner dua kelas.

### 11. DataLoader Windows bermasalah

Gunakan:

```yaml
training:
    workers: 0
```

Setelah pipeline stabil, jumlah worker dapat diuji kembali secara bertahap.

---

## 12. Perubahan file DeepLabV3 yang diperlukan

### 12.1 `configs/config.yml`

Perubahan minimum:

- `num_classes` menjadi `2`
- `crop_size` menjadi `256`
- `base_size` menjadi `256`
- `use_balanced_weights` menjadi `true`
- `train_on_subset.enabled` menjadi `false`
- checkpoint lama 14 kelas tidak digunakan

### 12.2 `data_generators/data_generator.py`

Perubahan minimum:

- membaca split manifest yang sama dengan U-Net
- mengonversi mask menggunakan `mask > 0`
- tidak menggunakan rata-rata channel untuk mask RGB
- mempertahankan mask sebagai integer `long`
- menggunakan `NEAREST` saat resize mask
- memvalidasi nilai mask agar hanya `0` dan `1`
- memastikan gambar dan mask berukuran sama setelah transformasi

### 12.3 `losses/loss.py`

Rekomendasi:

- pertahankan weighted Cross Entropy
- tambahkan Dice Loss
- gunakan kombinasi weighted Cross Entropy + Dice Loss

### 12.4 `trainers/trainer.py`

Perubahan yang perlu diperhatikan:

- checkpoint terbaik dipilih berdasarkan Dice atau IoU foreground
- jangan menggunakan test loader untuk pemilihan model
- simpan konfigurasi, seed, split, dan metric
- pastikan target bertipe `long`
- pastikan output model memiliki dua channel

### 12.5 `predictors/predictor.py`

Tambahkan dua jenis output:

1. `predict_mask()` untuk mask biner.
2. `predict_probability()` untuk probabilitas foreground.

Ensemble harus menggunakan `predict_probability()`, bukan hanya hasil
`argmax`.

Selain itu, fungsi inference test sebaiknya benar-benar menggunakan `test_loader`.
Jangan memberi nama fungsi test tetapi mengevaluasi `val_loader`.

### 12.6 Modul ensemble baru

Buat modul khusus, misalnya:

```text
ensemble/
├── __init__.py
└── fuse_predictions.py
```

Modul tersebut bertanggung jawab untuk:

- memuat checkpoint U-Net
- memuat checkpoint DeepLabV3
- menerapkan preprocessing masing-masing model
- menyamakan ukuran probabilitas
- menghitung weighted average
- menerapkan threshold
- menyimpan mask hasil ensemble
- menghitung metric pada test set

---

## 13. Protokol eksperimen penelitian

Setiap eksperimen sebaiknya dicatat dengan format:

```text
experiment_id:
dataset_version:
split_version:
seed:
unet_checkpoint:
deeplab_checkpoint:
image_size:
unet_preprocessing:
deeplab_preprocessing:
loss:
optimizer:
learning_rate:
alpha:
threshold:
validation_dice:
test_dice:
test_iou:
```

Eksperimen minimum:

1. U-Net saja.
2. DeepLabV3 saja.
3. Ensemble dengan `alpha=0.5`, `threshold=0.5`.
4. Ensemble dengan alpha terbaik dari validation set.
5. Ensemble dengan threshold terbaik dari validation set.

Kesimpulan ensemble hanya dapat dinyatakan lebih baik jika metric pada test set
meningkat dibandingkan masing-masing model secara individual.

---

## 14. Checklist sebelum training final

### Dataset

- [ ] Semua gambar dan mask memiliki pasangan nama.
- [ ] Tidak ada file yang tidak berpasangan.
- [ ] Semua mask sudah menjadi biner `0/1`.
- [ ] Split train/val/test sudah disimpan.
- [ ] U-Net dan DeepLabV3 menggunakan split yang sama.
- [ ] Test set tidak digunakan saat tuning.

### DeepLabV3

- [ ] `num_classes: 2`.
- [ ] Ukuran training `256 x 256`.
- [ ] Mask menggunakan interpolasi `NEAREST`.
- [ ] Target bertipe `long`.
- [ ] `use_balanced_weights: true`.
- [ ] Checkpoint 14 kelas tidak digunakan.
- [ ] `train_on_subset.enabled: false`.
- [ ] BatchNorm stabil atau dibekukan.

### Ensemble

- [ ] Kedua model menghasilkan probabilitas foreground.
- [ ] Probabilitas memiliki ukuran spasial yang sama.
- [ ] Penggabungan dilakukan sebelum threshold.
- [ ] Alpha dipilih dari validation set.
- [ ] Threshold dipilih dari validation set.
- [ ] Evaluasi final memakai test set yang sama.
- [ ] IoU dan Dice foreground dilaporkan.

---

## 15. Target hasil dan kriteria keberhasilan

Ensemble tidak otomatis lebih baik hanya karena menggabungkan dua model.
Ensemble dianggap berhasil jika:

1. Dataset dan label kedua model identik.
2. Prediksi kedua model berada pada koordinat pixel yang sama.
3. Ensemble menghasilkan metric test yang lebih baik atau lebih stabil.
4. False positive dan false negative berkurang dibandingkan model tunggal.
5. Hasil visual menunjukkan batas daun lebih lengkap dan tidak banyak noise.

Prioritas utama adalah konsistensi pipeline. Jangan melakukan tuning arsitektur
atau ensemble sebelum label biner, split dataset, dan preprocessing sudah
terverifikasi.