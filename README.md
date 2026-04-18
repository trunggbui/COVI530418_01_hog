# HOG + Linear SVM — Nhận dạng người

Cài đặt thuật toán HOG (Histogram of Oriented Gradients) từ đầu và huấn luyện Linear SVM để nhận dạng người trong ảnh.

---

## Cấu trúc thư mục

```
hog/
├── src/
│   ├── hog.py            # Cài đặt HOG: manual (minh họa) + fast (NumPy/SciPy)
│   ├── person_detect.py  # Split / Train / Mine / Evaluate / Detect
│   └── test_hog.py       # Unit tests (30 test cases)
├── resources/
│   └── images/
│       ├── pos/          # Ảnh chứa người (label +1) — 1038 ảnh
│       └── neg/          # Ảnh không có người (label -1) — 1544 ảnh
├── models/               # Model .npz sau khi train
├── outputs/              # Kết quả detect
└── requirements.txt
```

---

## Cài đặt môi trường

```bash
pip install -r requirements.txt
```

---

## Quy trình đầy đủ

### Bước 1 — Chia dataset 80/20

Chia toàn bộ ảnh gốc thành tập train (80%) và tập test (20%) độc lập, stratified theo từng class.

```bash
cd src
python person_detect.py split \
    --data-dir ../resources/images \
    --out-dir  ../resources/split \
    --test-ratio 0.2 \
    --seed 42
```

Kết quả tạo ra:

```
resources/split/
├── train/
│   ├── pos/   # 831 ảnh
│   └── neg/   # 1235 ảnh
└── test/
    ├── pos/   # 207 ảnh
    └── neg/   # 309 ảnh
```

---

### Bước 2 — Train lần 1 (model v1)

```bash
python person_detect.py train \
    --data-dir ../resources/split/train \
    --model-out ../models/hog_svm_v1.npz
```

```
Train acc: 1.0000
Val acc:   0.9443
Saved model to: ../models/hog_svm_v1.npz
```

---

### Bước 3 — Evaluate model v1 trên tập test độc lập

```bash
python person_detect.py evaluate \
    --model    ../models/hog_svm_v1.npz \
    --data-dir ../resources/split/test
```

```
========================================
  Tổng số mẫu  : 516
  Accuracy     : 0.9341  (482/516 đúng)
  Precision    : 0.8844
  Recall       : 0.9614
  F1-score     : 0.9213
========================================

Confusion Matrix:
                   Predicted +1  Predicted -1
Actual +1 (person)        199           8
Actual -1 (no person)      26         283
```

---

### Bước 4 — Hard Negative Mining

Chạy sliding window trên `train/neg/`, thu thập các patch bị nhận nhầm là người (false positives).

```bash
python person_detect.py mine \
    --model    ../models/hog_svm_v1.npz \
    --neg-dir  ../resources/split/train/neg \
    --out-dir  ../resources/split/neg_hard \
    --score-thr 0.0 \
    --max-patches 2000
```

```
Scanning 1235 ảnh negative để tìm hard negatives...
Tìm được 18 hard negative patches → lưu vào: ../resources/split/neg_hard
```

---

### Bước 5 — Train lần 2 (model v2, có hard negatives)

```bash
python person_detect.py train \
    --data-dir      ../resources/split/train \
    --extra-neg-dir ../resources/split/neg_hard \
    --model-out     ../models/hog_svm_v2.npz
```

```
  Hard negatives thêm vào: 18 mẫu từ ../resources/split/neg_hard
Train acc: 1.0000
Val acc:   0.9591
Saved model to: ../models/hog_svm_v2.npz
```

---

### Bước 6 — Evaluate model v2 trên tập test độc lập

```bash
python person_detect.py evaluate \
    --model    ../models/hog_svm_v2.npz \
    --data-dir ../resources/split/test
```

```
========================================
  Tổng số mẫu  : 516
  Accuracy     : 0.9360  (483/516 đúng)
  Precision    : 0.8919
  Recall       : 0.9565
  F1-score     : 0.9231
========================================

Confusion Matrix:
                   Predicted +1  Predicted -1
Actual +1 (person)        198           9
Actual -1 (no person)      24         285
```

---

## Kết quả so sánh v1 vs v2

> Đánh giá trên **516 ảnh test hoàn toàn độc lập** (không có trong quá trình train).

| Chỉ số | Model v1 | Model v2 (+HNM) | Thay đổi |
|--------|:--------:|:---------------:|:--------:|
| Val acc (trong train) | 0.9443 | **0.9591** | ↑ +0.0148 |
| **Accuracy** | 0.9341 | **0.9360** | ↑ +0.0019 |
| **Precision** | 0.8844 | **0.8919** | ↑ +0.0075 |
| Recall | **0.9614** | 0.9565 | ↓ −0.0049 |
| **F1-score** | 0.9213 | **0.9231** | ↑ +0.0018 |
| False Positive (nhận nhầm neg→pos) | 26 | **24** | ↓ −2 |
| False Negative (bỏ sót người) | **8** | 9 | ↑ +1 |

**Nhận xét:**

- Hard Negative Mining với 18 mẫu cải thiện rõ **Val acc** (+1.48%) và **Precision** (+0.75%): model ít nhận nhầm ảnh không có người hơn (FP: 26 → 24).
- **Recall** giảm nhẹ (bỏ sót thêm 1 người: FN 8 → 9) — đây là đánh đổi điển hình khi tăng precision.
- **Train acc = 1.0000** ở cả hai lần cho thấy model overfit hoàn toàn trên tập train; kết quả thực tế phải xem trên tập test độc lập.
- Hiệu quả của HNM sẽ rõ hơn khi dataset gốc có nhiều hard case hơn hoặc tập negative đa dạng hơn.

---

## Các lệnh khác

### Detect (phát hiện người trong ảnh mới)

```bash
cd src
python person_detect.py detect \
    --model  ../models/hog_svm_v2.npz \
    --image  ../resources/images/jiahao.jpeg \
    --output ../outputs/result.jpg
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--stride` | `8` | Bước trượt cửa sổ (px) |
| `--scales` | `1.0 0.9 0.8 0.7` | Các tỉ lệ scale ảnh |
| `--score-thr` | `0.0` | Ngưỡng điểm SVM để chấp nhận |
| `--iou-thr` | `0.3` | Ngưỡng IoU cho NMS |
| `--max-side` | `512` | Resize cạnh dài nhất về giá trị này |

### Unit Test

```bash
cd src
python -m unittest test_hog -v
# Ran 30 tests — OK
```

---

## Tham số train đầy đủ

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--data-dir` | _(bắt buộc)_ | Thư mục chứa `pos/` và `neg/` |
| `--model-out` | `models/hog_svm_model.npz` | Đường dẫn lưu model |
| `--extra-neg-dir` | _(trống)_ | Thư mục hard negatives (từ lệnh `mine`) |
| `--win-w` | `72` | Chiều rộng cửa sổ HOG (px) |
| `--win-h` | `96` | Chiều cao cửa sổ HOG (px) |
| `--cell-size` | `8` | Kích thước cell (px) |
| `--block-size` | `2` | Kích thước block (số cell) |
| `--n-bins` | `9` | Số bin histogram |
| `--c` | `1.0` | Hệ số regularization LinearSVC |
| `--val-ratio` | `0.2` | Tỉ lệ tập validation (trong train) |
| `--seed` | `42` | Random seed |

---

## Pipeline tổng quát

```
Ảnh RGB
   ↓ rgb2gray
Grayscale
   ↓ compute_gradients  (kernel [-1,0,1], scipy.ndimage.convolve)
Magnitude + Orientation (0–180°, unsigned)
   ↓ build_hog_cells    (cell 8×8, bilinear vote, np.bincount)
Cell histograms (n_cells_y × n_cells_x × 9)
   ↓ normalize_blocks   (block 2×2, stride=1, L2-norm)
HOG feature vector (3780 chiều cho cửa sổ 72×96)
   ↓ LinearSVC (sklearn)
Person (+1) / No Person (-1)
```

> `hog.py` có 2 bản cài đặt song song:
> - **Manual** (`fast=False`): vòng lặp Python thuần, để minh họa từng bước thuật toán
> - **Fast** (`fast=True`, mặc định): NumPy + SciPy, nhanh hơn ~74× so với manual

---

## Tài liệu tham khảo

- N. Dalal and B. Triggs, *"Histograms of Oriented Gradients for Human Detection"*, CVPR 2005.
- Dataset: [nishai/Person-Detection-SVM-HOG](https://github.com/nishai/Person-Detection-SVM-HOG/tree/master)
