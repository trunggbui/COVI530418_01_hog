# HOG + Linear SVM — Nhận dạng người

Cài đặt thuật toán HOG (Histogram of Oriented Gradients) từ đầu và huấn luyện Linear SVM để nhận dạng người trong ảnh.

---

## Cấu trúc thư mục

```
hog/
├── src/
│   ├── hog.py            # Cài đặt HOG (rgb2gray, gradient, histogram, normalize)
│   └── person_detect.py  # Train / Evaluate / Detect
├── resources/
│   └── images/
│       ├── pos/          # Ảnh chứa người (label +1)
│       └── neg/          # Ảnh không có người (label -1)
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

## 1. Train

Huấn luyện model từ tập dữ liệu. Dữ liệu cần có cấu trúc `pos/` và `neg/` bên trong thư mục chỉ định.

```bash
cd src
python person_detect.py train \
    --data-dir ../resources/images \
    --model-out ../models/hog_svm_model.npz
```

**Tham số tùy chọn:**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--data-dir` | `resources/images` | Thư mục chứa `pos/` và `neg/` |
| `--model-out` | `models/hog_svm_model.npz` | Đường dẫn lưu model |
| `--win-w` | `72` | Chiều rộng cửa sổ HOG (px) |
| `--win-h` | `96` | Chiều cao cửa sổ HOG (px) |
| `--cell-size` | `8` | Kích thước cell (px) |
| `--block-size` | `2` | Kích thước block (số cell) |
| `--n-bins` | `9` | Số bin histogram |
| `--c` | `1.0` | Hệ số regularization LinearSVC |
| `--val-ratio` | `0.2` | Tỉ lệ tập validation (0–1) |
| `--seed` | `42` | Random seed |

**Kết quả in ra:**

```
Train acc: 0.9823
Val acc:   0.9512
Saved model to: ../models/hog_svm_model.npz
```

---

## 2. Evaluate (Đánh giá model)

Đánh giá model trên tập dữ liệu có nhãn (cùng cấu trúc `pos/` / `neg/`).  
Dùng để kiểm tra trên **tập test** riêng biệt sau khi train.

```bash
cd src
python person_detect.py evaluate \
    --model ../models/hog_svm_model.npz \
    --data-dir ../resources/images
```

**Kết quả in ra:**

```
========================================
  Tổng số mẫu  : 2583
  Accuracy     : 0.9601  (2480/2583 đúng)
  Precision    : 0.9712
  Recall       : 0.9438
  F1-score     : 0.9573
========================================

Confusion Matrix:
                   Predicted +1  Predicted -1
Actual +1 (person)        981          59
Actual -1 (no person)      44        1499

Classification Report (sklearn):
                  precision    recall  f1-score   support
no person (-1)       0.96      0.97      0.97      1543
   person (+1)       0.96      0.94      0.95      1040
      accuracy                           0.96      2583
```

> **Giải thích chỉ số:**
> - **Accuracy**: Tỉ lệ phân loại đúng tổng thể
> - **Precision**: Trong số ảnh model dự đoán là "có người", bao nhiêu % đúng
> - **Recall**: Trong số ảnh thực sự có người, model phát hiện được bao nhiêu %
> - **F1-score**: Trung bình điều hòa của Precision và Recall
> - **Confusion Matrix**: Ma trận nhầm lẫn — hàng là nhãn thực, cột là nhãn dự đoán

---

## 3. Detect (Phát hiện người trong ảnh mới)

Chạy sliding window trên một ảnh bất kỳ để phát hiện vị trí người.

```bash
cd src
python person_detect.py detect \
    --model ../models/hog_svm_model.npz \
    --image ../resources/images/jiahao.jpeg \
    --output ../outputs/result.jpg
```

**Tham số tùy chọn:**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--stride` | `8` | Bước trượt cửa sổ (px) |
| `--scales` | `1.0 0.9 0.8 0.7` | Các tỉ lệ scale ảnh |
| `--score-thr` | `0.0` | Ngưỡng điểm SVM để chấp nhận |
| `--iou-thr` | `0.3` | Ngưỡng IoU cho NMS |
| `--max-side` | `512` | Resize cạnh dài nhất về giá trị này |

**Kết quả in ra:**

```
Detections before NMS: 142
Detections after NMS:  3
Saved result to: ../outputs/result.jpg
```

Ảnh kết quả được lưu tại `--output` với bounding box và điểm confidence.

---

## Pipeline tổng quát

```
Ảnh RGB
   ↓ rgb2gray
Grayscale
   ↓ compute_gradients  (kernel [-1,0,1])
Magnitude + Orientation (0–180°)
   ↓ build_hog_cells    (cell 8×8)
Cell histograms (n_cells_y × n_cells_x × 9)
   ↓ normalize_blocks   (block 2×2, L2-norm)
HOG feature vector (3780 chiều cho cửa sổ 72×96)
   ↓ LinearSVC
Person / No Person
```

---

## Tài liệu tham khảo

- N. Dalal and B. Triggs, *"Histograms of Oriented Gradients for Human Detection"*, CVPR 2005.
