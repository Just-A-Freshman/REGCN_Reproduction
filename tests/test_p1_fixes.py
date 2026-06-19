"""Tests for P1 (high severity) bug fixes documented in docs/tasks/todo/10-p1-high-crashes.md."""

import unittest
import os
import sys
import numpy as np


class TestIssue4ResultKeep2D(unittest.TestCase):
    """Verify Issue 4 fix: result[:, -1, :] keeps 2D shape for downstream functions."""

    def test_last_timestep_slice_keeps_last_dim(self):
        """result[:, -1, :] on (n, seq_len, 1) should give (n, 1), not (n,) nor (n, 1, 1)."""
        arr = np.random.randn(10, 30, 1)
        sliced = arr[:, -1, :]
        self.assertEqual(sliced.shape, (10, 1))

    def test_get_trend_works_with_2d_input(self):
        """get_trend() uses [i, 0] indexing — must work with (n, 1) input."""
        sys.path.insert(0, 'REGCN')
        from utils import get_trend

        pre = np.array([[100], [101], [102]])
        cur = np.array([[101], [102], [100]])
        trends = get_trend(pre, cur)
        expected = np.array([1, 1, 0])
        np.testing.assert_array_equal(trends, expected)

    def test_avg_relative_error_works_with_2d_input(self):
        """avg_relative_error() uses [i, 0] indexing — must work with (n, 1) input."""
        sys.path.insert(0, 'REGCN')
        from utils import avg_relative_error

        actual = np.array([[100], [200]])
        pred = np.array([[110], [180]])
        err = avg_relative_error(actual, pred)
        expected = (abs(110 - 100) / 100 + abs(180 - 200) / 200) / 2
        self.assertAlmostEqual(err, expected, places=6)


class TestIssue5VmdFilename(unittest.TestCase):
    """Verify Issue 5 fix: VMD CSV filenames have underscore after stock index."""

    def test_filename_contains_underscore(self):
        """Filename should include '_' between stock index and K value."""
        with open('dataprecossing/data_VMD.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Check the savetxt line uses underscore format
        for line in content.split('\n'):
            if 'savetxt' in line and '.csv' in line:
                self.assertIn('"_', line,
                              "VMD filename should have underscore after stock index")
                # Should match pattern like "0_3-1.csv" not "03-1.csv"
                self.assertNotIn(
                    'str(j) + str(K)', line,
                    "Should not concatenate stock index and K without separator"
                )
                break


class TestIssue6DatasetOverride(unittest.TestCase):
    """Verify Issue 6 fix: main.py run_training overrides datasets."""

    def test_main_py_has_override_logic(self):
        """run_training() should override regcn.datasets after import."""
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('regcn.datasets = dataset', content,
                      "run_training must override regcn.datasets")
        self.assertIn('regcn.data_addr = os.path.join(base_data_addr, dataset + ".npy")', content,
                      "run_training must rebuild data path from base + dataset")
        self.assertIn('regcn.data = np.load(', content,
                      "run_training must reload data for the correct dataset")

    def test_regcn_datasets_comes_from_config(self):
        """REGCN.py still reads datasets from config.ini as default."""
        with open('REGCN/REGCN.py', 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('datasets = config["hyper"]["datasets"]', content)


class TestIssue3DenormaliseFromTrain(unittest.TestCase):
    """Verify B4 fix: VMD component predictions summed without per-component scaling."""

    def test_main_sums_raw_predictions(self):
        """main() should sum raw VMD predictions, not apply unautoNorm per component."""
        with open('REGCN/REGCN.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Each component prediction is appended as-is (no unautoNorm wrapper)
        self.assertIn('result.append(result1)', content)
        self.assertNotIn('result.append(unautoNorm(result1', content)


if __name__ == '__main__':
    unittest.main()
