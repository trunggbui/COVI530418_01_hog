"""Unit tests cho hog.py — chạy bằng: python -m pytest src/ hoặc python -m unittest src/test_hog.py"""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import hog as H

CELL = 8
BLOCK = 2
N_BINS = 9
_SRC_DIR = Path(__file__).parent
_IMG_PATH = _SRC_DIR / ".." / "resources" / "images" / "jiahao.jpeg"


class TestRgb2Gray(unittest.TestCase):
    def test_output_shape(self):
        img = np.random.randint(0, 256, (128, 64, 3), dtype=np.uint8)
        gray = H.rgb2gray(img)
        self.assertEqual(gray.shape, (128, 64))

    def test_black_image(self):
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        self.assertTrue(np.all(H.rgb2gray(black) == 0))

    def test_white_image(self):
        white = np.full((10, 10, 3), 255, dtype=np.uint8)
        self.assertTrue(np.allclose(H.rgb2gray(white), 255, atol=1))

    def test_pure_red_pixel_rec709(self):
        red = np.zeros((1, 1, 3), dtype=np.uint8)
        red[0, 0] = [255, 0, 0]
        val = float(H.rgb2gray(red)[0, 0])
        self.assertAlmostEqual(val, 255 * 0.2126, delta=1.5)

    def test_output_is_2d(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        self.assertEqual(H.rgb2gray(img).ndim, 2)


class TestComputeGradients(unittest.TestCase):
    def setUp(self):
        self.gray = np.random.rand(128, 64).astype(np.float64)
        self.mag, self.ori = H.compute_gradients(self.gray)

    def test_output_shape_matches_input(self):
        self.assertEqual(self.mag.shape, self.gray.shape)
        self.assertEqual(self.ori.shape, self.gray.shape)

    def test_magnitude_non_negative(self):
        self.assertTrue(np.all(self.mag >= 0))

    def test_orientation_range(self):
        self.assertTrue(np.all(self.ori >= 0))
        self.assertTrue(np.all(self.ori < 180))

    def test_flat_image_zero_gradient(self):
        flat = np.full((16, 16), 128.0)
        mag_flat, _ = H.compute_gradients(flat)
        self.assertTrue(np.allclose(mag_flat[1:-1, 1:-1], 0, atol=1e-6))

    def test_vertical_edge_has_gradient(self):
        edge = np.zeros((16, 16), dtype=np.float64)
        edge[:, 8:] = 255.0
        mag_e, _ = H.compute_gradients(edge)
        self.assertGreater(mag_e[8, 8], 0)


class TestComputeCellHistogram(unittest.TestCase):
    def test_output_shape(self):
        hist = H.compute_cell_histogram(np.ones((8, 8)), np.zeros((8, 8)), N_BINS)
        self.assertEqual(hist.shape, (N_BINS,))

    def test_sum_equals_magnitude_sum(self):
        mag = np.abs(np.random.rand(8, 8)) * 100
        ori = np.random.rand(8, 8) * 179.9
        hist = H.compute_cell_histogram(mag, ori, N_BINS)
        self.assertAlmostEqual(float(hist.sum()), float(mag.sum()), places=2)

    def test_bilinear_vote_at_10_degrees(self):
        mag = np.ones((8, 8))
        ori = np.full((8, 8), 10.0)
        hist = H.compute_cell_histogram(mag, ori, N_BINS)
        expected = mag.sum() * 0.5
        self.assertAlmostEqual(float(hist[0]), expected, places=4)
        self.assertAlmostEqual(float(hist[1]), expected, places=4)

    def test_histogram_non_negative(self):
        mag = np.abs(np.random.rand(8, 8)) * 100
        ori = np.random.rand(8, 8) * 179.9
        hist = H.compute_cell_histogram(mag, ori, N_BINS)
        self.assertTrue(np.all(hist >= 0))

    def test_zero_magnitude_gives_zero_histogram(self):
        ori = np.random.rand(8, 8) * 179.9
        hist = H.compute_cell_histogram(np.zeros((8, 8)), ori, N_BINS)
        self.assertTrue(np.allclose(hist, 0))


class TestBuildHogCells(unittest.TestCase):
    def setUp(self):
        self.H, self.W = 128, 64
        self.mag = np.random.rand(self.H, self.W)
        self.ori = np.random.rand(self.H, self.W) * 179.9

    def test_output_shape(self):
        cells = H.build_hog_cells(self.mag, self.ori, CELL, N_BINS)
        n_cy, n_cx = self.H // CELL, self.W // CELL
        self.assertEqual(cells.shape, (n_cy, n_cx, N_BINS))

    def test_all_values_non_negative(self):
        cells = H.build_hog_cells(self.mag, self.ori, CELL, N_BINS)
        self.assertTrue(np.all(cells >= 0))

    def test_flat_image_identical_cells(self):
        flat_mag = np.ones((self.H, self.W))
        flat_ori = np.full((self.H, self.W), 45.0)
        cells = H.build_hog_cells(flat_mag, flat_ori, CELL, N_BINS)
        ref = cells[0, 0]
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                self.assertTrue(np.allclose(cells[i, j], ref))

    def test_sum_equals_magnitude_sum(self):
        cells = H.build_hog_cells(self.mag, self.ori, CELL, N_BINS)
        self.assertAlmostEqual(float(cells.sum()), float(self.mag.sum()), places=3)

    def test_small_image_shape(self):
        cells = H.build_hog_cells(np.ones((32, 32)), np.ones((32, 32)) * 90, CELL, N_BINS)
        self.assertEqual(cells.shape, (4, 4, N_BINS))


class TestNormalizeBlocks(unittest.TestCase):
    def setUp(self):
        self.n_cy, self.n_cx = 16, 8
        self.cell_hist = np.random.rand(self.n_cy, self.n_cx, N_BINS)
        self.feat = H.normalize_blocks(self.cell_hist, block_size=BLOCK)

    def test_output_is_1d(self):
        self.assertEqual(self.feat.ndim, 1)

    def test_feature_length(self):
        n_by = self.n_cy - BLOCK + 1
        n_bx = self.n_cx - BLOCK + 1
        expected = n_by * n_bx * BLOCK * BLOCK * N_BINS
        self.assertEqual(self.feat.shape[0], expected)

    def test_each_block_norm_leq_1(self):
        block_len = BLOCK * BLOCK * N_BINS
        n_blocks = (self.n_cy - BLOCK + 1) * (self.n_cx - BLOCK + 1)
        for idx in range(n_blocks):
            vec = self.feat[idx * block_len:(idx + 1) * block_len]
            self.assertLessEqual(float(np.linalg.norm(vec)), 1.0 + 1e-5)

    def test_zero_input_gives_zero_output(self):
        feat_zero = H.normalize_blocks(np.zeros((self.n_cy, self.n_cx, N_BINS)), BLOCK)
        self.assertTrue(np.allclose(feat_zero, 0))

    def test_constant_input_block_norm_equals_1(self):
        feat_const = H.normalize_blocks(np.ones((self.n_cy, self.n_cx, N_BINS)), BLOCK)
        block_len = BLOCK * BLOCK * N_BINS
        n_blocks = (self.n_cy - BLOCK + 1) * (self.n_cx - BLOCK + 1)
        for idx in range(n_blocks):
            vec = feat_const[idx * block_len:(idx + 1) * block_len]
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)


class TestExtractHog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _IMG_PATH.exists():
            raise unittest.SkipTest(f"Không tìm thấy {_IMG_PATH}")
        img_rgb = np.array(Image.open(_IMG_PATH).convert("RGB"), dtype=np.uint8)
        cls.img_64x128 = np.array(
            Image.fromarray(img_rgb).resize((64, 128)), dtype=np.uint8
        )
        cls.img_128x128 = np.array(
            Image.fromarray(img_rgb).resize((128, 128)), dtype=np.uint8
        )

    def test_feature_length_64x128(self):
        feat = H.extract_hog(self.img_64x128, CELL, BLOCK, N_BINS)
        n_cy, n_cx = 128 // CELL, 64 // CELL
        expected = (n_cy - BLOCK + 1) * (n_cx - BLOCK + 1) * BLOCK * BLOCK * N_BINS
        self.assertEqual(feat.shape, (expected,))

    def test_output_is_1d(self):
        feat = H.extract_hog(self.img_64x128, CELL, BLOCK, N_BINS)
        self.assertEqual(feat.ndim, 1)

    def test_feature_values_in_range(self):
        feat = H.extract_hog(self.img_64x128, CELL, BLOCK, N_BINS)
        self.assertGreaterEqual(float(feat.min()), 0.0)
        self.assertLessEqual(float(feat.max()), 1.0 + 1e-5)

    def test_deterministic(self):
        feat1 = H.extract_hog(self.img_64x128.copy(), CELL, BLOCK, N_BINS)
        feat2 = H.extract_hog(self.img_64x128.copy(), CELL, BLOCK, N_BINS)
        self.assertTrue(np.allclose(feat1, feat2))

    def test_feature_length_128x128(self):
        feat = H.extract_hog(self.img_128x128, CELL, BLOCK, N_BINS)
        n_cy, n_cx = 128 // CELL, 128 // CELL
        expected = (n_cy - BLOCK + 1) * (n_cx - BLOCK + 1) * BLOCK * BLOCK * N_BINS
        self.assertEqual(feat.shape, (expected,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
