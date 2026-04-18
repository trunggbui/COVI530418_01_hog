import numpy as np
from numpy.typing import NDArray

def rgb2gray(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh RGB sang grayscale"""
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {image.shape}")

    img: NDArray[np.float32] = image.astype(np.float32)
    weights: NDArray[np.float32] = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    gray = img @ weights
    return gray


def _pad_zero(img: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    h, w = img.shape
    out = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            out[y + pad_h, x + pad_w] = float(img[y, x])
    return out

def _conv2d_manual(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = img.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = _pad_zero(img, ph, pw)
    out = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            s = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    s += padded[y + ky, x + kx] * kernel[ky, kx]
            out[y, x] = s
    return out

def compute_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Tính gradient theo x và y bằng kernel Sobel (hoặc [-1,0,1])
    Returns: magnitude, orientation (độ, 0-180 unsigned)
    """
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError(f"Expected grayscale 2D, got {gray.shape}")

    kx = np.array([[-1, 0, 1]], dtype=np.float32)  # 1x3
    ky = kx.T  # 3x1

    gx = _conv2d_manual(gray, kx)
    gy = _conv2d_manual(gray, ky)

    magnitude = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    orientation = np.degrees(np.arctan2(gy, gx))
    orientation = ((orientation + 180.0) % 180.0).astype(np.float32)

    return magnitude, orientation

def compute_cell_histogram(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    n_bins: int = 9
) -> np.ndarray:
    """
    Tạo histogram cho 1 cell (thường 8x8 px)
    - Bin range: 0-180 độ (unsigned), mỗi bin = 20 độ
    - Vote có trọng số = magnitude (bilinear interpolation nếu muốn chuẩn)
    Returns: histogram shape (n_bins,)
    """
    mag = np.asarray(magnitude, dtype=np.float32)
    ori = np.asarray(orientation, dtype=np.float32)

    if mag.shape != ori.shape:
        raise ValueError(f"magnitude và orientation phải cùng shape, got {mag.shape} vs {ori.shape}")
    if mag.ndim != 2:
        raise ValueError(f"Expected 2D cell, got {mag.shape}")

    hist = np.zeros(n_bins, dtype=np.float32)
    bin_width = 180.0 / n_bins  # 20 độ nếu n_bins=9

    h, w = mag.shape
    for y in range(h):
        for x in range(w):
            m = mag[y, x]
            a = ori[y, x]

            # vị trí bin thực (vd 2.3 nghĩa là giữa bin 2 và 3)
            pos = a / bin_width
            left_bin = int(np.floor(pos)) % n_bins
            right_bin = (left_bin + 1) % n_bins

            right_w = pos - np.floor(pos)  # phần gần bin phải
            left_w = 1.0 - right_w  # phần gần bin trái

            hist[left_bin] += m * left_w
            hist[right_bin] += m * right_w

    return hist

def build_hog_cells(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    cell_size: int = 8,
    n_bins: int = 9
) -> np.ndarray:
    """
    Chia ảnh thành lưới cell, tính histogram cho từng cell
    Returns: cell_histogram shape (n_cells_y, n_cells_x, n_bins)
    """
    mag = np.asarray(magnitude, dtype=np.float32)
    ori = np.asarray(orientation, dtype=np.float32)

    if mag.shape != ori.shape:
        raise ValueError(f"magnitude và orientation phải cùng shape, got {mag.shape} vs {ori.shape}")
    if mag.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {mag.shape}")
    if cell_size <= 0:
        raise ValueError(f"cell_size must be > 0, got {cell_size}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins}")

    h, w = mag.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    if n_cells_y == 0 or n_cells_x == 0:
        raise ValueError(
            f"Image too small for cell_size={cell_size}: image shape={mag.shape}"
        )

    cell_hist = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float32)

    for cy in range(n_cells_y):
        y0 = cy * cell_size
        y1 = y0 + cell_size
        for cx in range(n_cells_x):
            x0 = cx * cell_size
            x1 = x0 + cell_size

            mag_cell = mag[y0:y1, x0:x1]
            ori_cell = ori[y0:y1, x0:x1]
            cell_hist[cy, cx, :] = compute_cell_histogram(mag_cell, ori_cell, n_bins)

    return cell_hist

def normalize_blocks(
    cell_hist: np.ndarray,
    block_size: int = 2,         # block = 2x2 cells
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Trượt block (stride=1 cell) qua lưới cell, L2-normalize từng block
    Returns: feature vector 1D đã flatten
    """
    cells = np.asarray(cell_hist, dtype=np.float32)

    if cells.ndim != 3:
        raise ValueError(f"Expected cell_hist with shape (n_cells_y, n_cells_x, n_bins), got {cells.shape}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    n_cells_y, n_cells_x, n_bins = cells.shape
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    if n_blocks_y <= 0 or n_blocks_x <= 0:
        raise ValueError(
            f"block_size={block_size} is too large for cell grid {(n_cells_y, n_cells_x)}"
        )

    block_feature_len = block_size * block_size * n_bins
    total_blocks = n_blocks_y * n_blocks_x
    features = np.zeros(total_blocks * block_feature_len, dtype=np.float32)

    offset = 0
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = cells[by:by + block_size, bx:bx + block_size, :]
            vec = block.reshape(-1).astype(np.float32)

            norm = np.sqrt(np.sum(vec * vec) + epsilon * epsilon)
            vec_norm = vec / norm

            features[offset:offset + block_feature_len] = vec_norm
            offset += block_feature_len

    return features

def extract_hog(
    image: np.ndarray,
    cell_size: int = 8,
    block_size: int = 2,
    n_bins: int = 9
) -> np.ndarray:
    """Pipeline chính: gọi tuần tự các hàm trên"""
    gray = rgb2gray(image)
    mag, ori = compute_gradients(gray)
    cells = build_hog_cells(mag, ori, cell_size, n_bins)
    features = normalize_blocks(cells, block_size)
    return features
