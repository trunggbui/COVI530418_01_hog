from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from hog import extract_hog


@dataclass
class DetectorConfig:
    win_w: int = 72
    win_h: int = 96
    cell_size: int = 8
    block_size: int = 2
    n_bins: int = 9


@dataclass
class LinearModel:
    w: np.ndarray
    b: float

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(x) >= 0.0, 1, -1)


def _iter_image_files(folder: Path) -> Iterable[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    for ext in exts:
        for p in folder.glob(ext):
            if p.is_file():
                yield p


def _extract_feat(img_np: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    return extract_hog(
        img_np,
        cell_size=cfg.cell_size,
        block_size=cfg.block_size,
        n_bins=cfg.n_bins,
    ).astype(np.float32)


def load_dataset(
    data_dir: Path,
    cfg: DetectorConfig,
    extra_neg_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pos_dir = data_dir / "pos"
    neg_dir = data_dir / "neg"
    if not pos_dir.exists() or not neg_dir.exists():
        raise FileNotFoundError(f"Expected {pos_dir} and {neg_dir}")

    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    for label, cls_dir in [(1, pos_dir), (-1, neg_dir)]:
        for img_path in sorted(_iter_image_files(cls_dir)):
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                if rgb.size != (cfg.win_w, cfg.win_h):
                    rgb = rgb.resize((cfg.win_w, cfg.win_h), Image.Resampling.BILINEAR)
                img_np = np.array(rgb, dtype=np.uint8)
            x_list.append(_extract_feat(img_np, cfg))
            y_list.append(label)

    if extra_neg_dir is not None and extra_neg_dir.exists():
        hard_neg_files = sorted(_iter_image_files(extra_neg_dir))
        for img_path in hard_neg_files:
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                if rgb.size != (cfg.win_w, cfg.win_h):
                    rgb = rgb.resize((cfg.win_w, cfg.win_h), Image.Resampling.BILINEAR)
                img_np = np.array(rgb, dtype=np.uint8)
            x_list.append(_extract_feat(img_np, cfg))
            y_list.append(-1)
        print(f"  Hard negatives thêm vào: {len(hard_neg_files)} mẫu từ {extra_neg_dir}")

    if not x_list:
        raise RuntimeError("No training images found")

    x = np.stack(x_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return x, y


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)

    n_val = int(round(x.shape[0] * val_ratio))
    n_val = max(1, min(n_val, x.shape[0] - 1))

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def standardize_fit(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def save_model(
    out_path: Path,
    model: LinearModel,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: DetectorConfig,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        w=model.w.astype(np.float32),
        b=np.float32(model.b),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        win_w=np.int32(cfg.win_w),
        win_h=np.int32(cfg.win_h),
        cell_size=np.int32(cfg.cell_size),
        block_size=np.int32(cfg.block_size),
        n_bins=np.int32(cfg.n_bins),
    )


def load_model(model_path: Path) -> tuple[LinearModel, np.ndarray, np.ndarray, DetectorConfig]:
    data = np.load(model_path)
    cfg = DetectorConfig(
        win_w=int(data["win_w"]),
        win_h=int(data["win_h"]),
        cell_size=int(data["cell_size"]),
        block_size=int(data["block_size"]),
        n_bins=int(data["n_bins"]),
    )
    svm = LinearModel(w=data["w"].astype(np.float32), b=float(data["b"]))
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    return svm, mean, std, cfg


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    iou_thr: float = 0.3,
) -> list[int]:
    if not boxes:
        return []

    order = np.argsort(np.array(scores))[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        remain: list[int] = []
        for j in order[1:]:
            if iou(boxes[i], boxes[int(j)]) < iou_thr:
                remain.append(int(j))

        order = np.array(remain, dtype=np.int64)

    return keep


def sliding_window_detect(
    image_rgb: np.ndarray,
    svm: LinearModel,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: DetectorConfig,
    stride: int = 8,
    scales: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7),
    score_thr: float = 0.0,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    orig_h, orig_w = image_rgb.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []

    for s in scales:
        scaled_w = max(1, int(round(orig_w * s)))
        scaled_h = max(1, int(round(orig_h * s)))
        if scaled_w < cfg.win_w or scaled_h < cfg.win_h:
            continue

        scaled_img = np.array(
            Image.fromarray(image_rgb).resize((scaled_w, scaled_h), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

        for y in range(0, scaled_h - cfg.win_h + 1, stride):
            for x in range(0, scaled_w - cfg.win_w + 1, stride):
                patch = scaled_img[y:y + cfg.win_h, x:x + cfg.win_w, :]
                feat = extract_hog(
                    patch,
                    cell_size=cfg.cell_size,
                    block_size=cfg.block_size,
                    n_bins=cfg.n_bins,
                ).astype(np.float32)
                feat = standardize_apply(feat[None, :], mean, std)
                score = float(svm.decision_function(feat)[0])

                if score >= score_thr:
                    x1 = int(round(x / s))
                    y1 = int(round(y / s))
                    x2 = int(round((x + cfg.win_w) / s))
                    y2 = int(round((y + cfg.win_h) / s))
                    boxes.append((x1, y1, x2, y2))
                    scores.append(score)

    return boxes, scores


def cmd_train(args: argparse.Namespace) -> None:
    try:
        from sklearn.svm import LinearSVC
    except ImportError as e:
        raise ImportError(
            "Thiếu scikit-learn. Cài bằng: pip install scikit-learn"
        ) from e

    cfg = DetectorConfig(
        win_w=args.win_w,
        win_h=args.win_h,
        cell_size=args.cell_size,
        block_size=args.block_size,
        n_bins=args.n_bins,
    )

    extra_neg = Path(args.extra_neg_dir) if args.extra_neg_dir else None
    x, y = load_dataset(Path(args.data_dir), cfg, extra_neg_dir=extra_neg)
    x_train, y_train, x_val, y_val = train_val_split(x, y, val_ratio=args.val_ratio, seed=args.seed)

    mean, std = standardize_fit(x_train)
    x_train_s = standardize_apply(x_train, mean, std)
    x_val_s = standardize_apply(x_val, mean, std)

    # Dùng sklearn LinearSVC, chỉ lưu lại hyperplane để detect.
    svc = LinearSVC(
        C=args.c,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    svc.fit(x_train_s, y_train)
    svm = LinearModel(
        w=svc.coef_.reshape(-1).astype(np.float32),
        b=float(svc.intercept_[0]),
    )

    train_pred = svm.predict(x_train_s)
    val_pred = svm.predict(x_val_s)
    print(f"Train acc: {accuracy(y_train, train_pred):.4f}")
    print(f"Val acc:   {accuracy(y_val, val_pred):.4f}")

    save_model(Path(args.model_out), svm, mean, std, cfg)
    print(f"Saved model to: {args.model_out}")


def cmd_split(args: argparse.Namespace) -> None:
    """Chia dataset pos/ neg/ thành train/ và test/ theo tỉ lệ chỉ định."""
    src = Path(args.data_dir)
    out = Path(args.out_dir)
    test_ratio = args.test_ratio
    seed = args.seed

    rng = np.random.default_rng(seed)

    for split in ("train", "test"):
        for cls in ("pos", "neg"):
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    total_copied = {"train": 0, "test": 0}

    for cls in ("pos", "neg"):
        cls_dir = src / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy: {cls_dir}")

        files = sorted(_iter_image_files(cls_dir))
        if not files:
            raise RuntimeError(f"Không có ảnh trong {cls_dir}")

        indices = np.arange(len(files))
        rng.shuffle(indices)
        n_test = max(1, int(round(len(files) * test_ratio)))
        test_idx = set(indices[:n_test].tolist())

        for i, f in enumerate(files):
            split = "test" if i in test_idx else "train"
            shutil.copy2(f, out / split / cls / f.name)
            total_copied[split] += 1

    n_train_pos = len(list((out / "train" / "pos").iterdir()))
    n_train_neg = len(list((out / "train" / "neg").iterdir()))
    n_test_pos  = len(list((out / "test"  / "pos").iterdir()))
    n_test_neg  = len(list((out / "test"  / "neg").iterdir()))

    print(f"Split hoàn tất → {out}")
    print(f"  train: {n_train_pos} pos  +  {n_train_neg} neg  =  {n_train_pos + n_train_neg} ảnh")
    print(f"  test:  {n_test_pos}  pos  +  {n_test_neg}  neg  =  {n_test_pos  + n_test_neg}  ảnh")


def cmd_mine(args: argparse.Namespace) -> None:
    """Hard Negative Mining: chạy sliding window trên ảnh negative, lưu false positive patches."""
    svm, mean, std, cfg = load_model(Path(args.model))

    neg_dir = Path(args.neg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    neg_files = sorted(_iter_image_files(neg_dir))
    if not neg_files:
        print(f"Không tìm thấy ảnh nào trong {neg_dir}")
        return

    print(f"Scanning {len(neg_files)} ảnh negative để tìm hard negatives...")
    count = 0

    for img_path in neg_files:
        with Image.open(img_path) as im:
            image_rgb = np.array(im.convert("RGB"), dtype=np.uint8)

        orig_h, orig_w = image_rgb.shape[:2]

        for s in args.scales:
            scaled_w = max(1, int(round(orig_w * s)))
            scaled_h = max(1, int(round(orig_h * s)))
            if scaled_w < cfg.win_w or scaled_h < cfg.win_h:
                continue

            scaled_img = np.array(
                Image.fromarray(image_rgb).resize((scaled_w, scaled_h), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )

            for y in range(0, scaled_h - cfg.win_h + 1, args.stride):
                for x in range(0, scaled_w - cfg.win_w + 1, args.stride):
                    patch = scaled_img[y:y + cfg.win_h, x:x + cfg.win_w]
                    feat = _extract_feat(patch, cfg)
                    feat_s = standardize_apply(feat[None, :], mean, std)
                    score = float(svm.decision_function(feat_s)[0])

                    if score >= args.score_thr:
                        Image.fromarray(patch).save(out_dir / f"hardneg_{count:06d}.jpg")
                        count += 1
                        if 0 < args.max_patches <= count:
                            break
                if 0 < args.max_patches <= count:
                    break
            if 0 < args.max_patches <= count:
                break
        if 0 < args.max_patches <= count:
            break

    print(f"Tìm được {count} hard negative patches → lưu vào: {out_dir}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    svm, mean, std, cfg = load_model(Path(args.model))

    print(f"Loading dataset from: {args.data_dir}")
    x, y = load_dataset(Path(args.data_dir), cfg)
    x_s = standardize_apply(x, mean, std)

    y_pred = svm.predict(x_s)

    n = len(y)
    tp = int(((y == 1) & (y_pred == 1)).sum())
    fp = int(((y == -1) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == -1)).sum())
    tn = int(((y == -1) & (y_pred == -1)).sum())

    acc = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print()
    print("=" * 40)
    print(f"  Tổng số mẫu  : {n}")
    print(f"  Accuracy     : {acc:.4f}  ({tp + tn}/{n} đúng)")
    print(f"  Precision    : {precision:.4f}")
    print(f"  Recall       : {recall:.4f}")
    print(f"  F1-score     : {f1:.4f}")
    print("=" * 40)
    print()
    print("Confusion Matrix:")
    print(f"{'':18s} Predicted +1  Predicted -1")
    print(f"{'Actual +1 (person)':18s}   {tp:8d}    {fn:8d}")
    print(f"{'Actual -1 (no person)':21s}{fp:8d}    {tn:8d}")
    print()

    try:
        from sklearn.metrics import classification_report
        target_names = ["no person (-1)", "person (+1)"]
        print("Classification Report (sklearn):")
        print(classification_report(y, y_pred, target_names=target_names))
    except ImportError:
        pass


def cmd_detect(args: argparse.Namespace) -> None:
    svm, mean, std, cfg = load_model(Path(args.model))

    with Image.open(args.image) as im:
        image_rgb = np.array(im.convert("RGB"), dtype=np.uint8)

    h, w = image_rgb.shape[:2]
    max_side = max(h, w)
    if args.max_side > 0 and max_side > args.max_side:
        scale = args.max_side / max_side
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        image_rgb = np.array(
            Image.fromarray(image_rgb).resize((new_w, new_h), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        print(f"Resized input for detection: ({w}, {h}) -> ({new_w}, {new_h})")

    boxes, scores = sliding_window_detect(
        image_rgb,
        svm,
        mean,
        std,
        cfg,
        stride=args.stride,
        scales=tuple(args.scales),
        score_thr=args.score_thr,
    )

    keep = nms(boxes, scores, iou_thr=args.iou_thr)
    kept_boxes = [boxes[i] for i in keep]
    kept_scores = [scores[i] for i in keep]

    vis = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(vis)
    for (x1, y1, x2, y2), s in zip(kept_boxes, kept_scores):
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1, max(0, y1 - 10)), f"{s:.2f}", fill=(255, 0, 0))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)

    print(f"Detections before NMS: {len(boxes)}")
    print(f"Detections after NMS:  {len(kept_boxes)}")
    print(f"Saved result to: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HOG + Linear SVM person detector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("split", help="Chia dataset thành train/ và test/")
    p_split.add_argument("--data-dir", type=str, required=True,
                         help="Thư mục gốc chứa pos/ và neg/")
    p_split.add_argument("--out-dir", type=str, required=True,
                         help="Thư mục đầu ra (sẽ tạo train/ và test/ bên trong)")
    p_split.add_argument("--test-ratio", type=float, default=0.2,
                         help="Tỉ lệ tập test (mặc định 0.2 = 20%%)")
    p_split.add_argument("--seed", type=int, default=42)
    p_split.set_defaults(func=cmd_split)

    p_train = sub.add_parser("train", help="Train Linear SVM on HOG features")
    p_train.add_argument("--data-dir", type=str, default="resources/images")
    p_train.add_argument("--model-out", type=str, default="models/hog_svm_model.npz")
    p_train.add_argument("--win-w", type=int, default=72)
    p_train.add_argument("--win-h", type=int, default=96)
    p_train.add_argument("--cell-size", type=int, default=8)
    p_train.add_argument("--block-size", type=int, default=2)
    p_train.add_argument("--n-bins", type=int, default=9)
    p_train.add_argument("--c", type=float, default=1.0)
    p_train.add_argument("--max-iter", type=int, default=2000)
    p_train.add_argument("--val-ratio", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--extra-neg-dir", type=str, default="",
                         help="Thư mục chứa hard negative patches (tùy chọn)")
    p_train.set_defaults(func=cmd_train)

    p_mine = sub.add_parser("mine", help="Hard Negative Mining: tìm false positives trên tập negative")
    p_mine.add_argument("--model", type=str, required=True)
    p_mine.add_argument("--neg-dir", type=str, required=True,
                        help="Thư mục ảnh negative gốc")
    p_mine.add_argument("--out-dir", type=str, default="resources/images/neg_hard",
                        help="Thư mục lưu hard negative patches")
    p_mine.add_argument("--stride", type=int, default=8)
    p_mine.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7])
    p_mine.add_argument("--score-thr", type=float, default=0.0,
                        help="Ngưỡng score SVM để coi là false positive")
    p_mine.add_argument("--max-patches", type=int, default=2000,
                        help="Giới hạn số hard negative lưu (0 = không giới hạn)")
    p_mine.set_defaults(func=cmd_mine)

    p_eval = sub.add_parser("evaluate", help="Evaluate model on a labeled dataset (pos/ neg/)")
    p_eval.add_argument("--model", type=str, required=True)
    p_eval.add_argument("--data-dir", type=str, required=True,
                        help="Thư mục chứa pos/ và neg/")
    p_eval.set_defaults(func=cmd_evaluate)

    p_detect = sub.add_parser("detect", help="Run sliding-window detector on one image")
    p_detect.add_argument("--model", type=str, required=True)
    p_detect.add_argument("--image", type=str, required=True)
    p_detect.add_argument("--output", type=str, default="outputs/detect_result.jpg")
    p_detect.add_argument("--stride", type=int, default=8)
    p_detect.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7])
    p_detect.add_argument("--score-thr", type=float, default=0.0)
    p_detect.add_argument("--iou-thr", type=float, default=0.3)
    p_detect.add_argument("--max-side", type=int, default=512)
    p_detect.set_defaults(func=cmd_detect)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
