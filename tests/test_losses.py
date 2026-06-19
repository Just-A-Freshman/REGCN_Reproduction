import unittest
import numpy as np
import tensorflow as tf
import sys

sys.path.insert(0, '.')  # project root: 期末/

from REGCN.losses import trend_regularized_loss


class TestTrendRegularizedLoss(unittest.TestCase):
    """Unit tests for the trend-regularized loss function (paper Eq. 15)."""

    def test_loss_equals_mse_when_trend_matches_perfectly(self):
        """When trends match perfectly, penalty is 0, loss = MSE."""
        y_true = tf.constant([[[1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[1.1], [2.1], [3.1], [4.1], [5.1]]], dtype=tf.float32)

        loss_fn = trend_regularized_loss(r_acc=0.1)
        l = loss_fn(y_true, y_pred).numpy()

        mse_expected = np.mean((np.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
                                - np.array([[[1.1], [2.1], [3.1], [4.1], [5.1]]])) ** 2)
        self.assertAlmostEqual(l, mse_expected, places=5)

    def test_loss_penalizes_wrong_trend_direction(self):
        """All trends wrong → penalty = 1.0 (max), so loss = MSE + r_acc."""
        y_true = tf.constant([[[1.0], [3.0], [2.0], [4.0], [3.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[3.0], [1.0], [4.0], [2.0], [5.0]]], dtype=tf.float32)

        loss_fn = trend_regularized_loss(r_acc=0.1)
        l = loss_fn(y_true, y_pred).numpy()

        mse_expected = np.mean((np.array([[[1.0], [3.0], [2.0], [4.0], [3.0]]])
                                - np.array([[[3.0], [1.0], [4.0], [2.0], [5.0]]])) ** 2)
        self.assertAlmostEqual(l, mse_expected + 0.1, places=5)

    def test_loss_partial_trend_penalty(self):
        """Half trends correct, half wrong."""
        y_true = tf.constant([[[1.0], [2.0], [3.0], [2.0], [1.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[1.0], [2.1], [2.9], [2.1], [0.9]]], dtype=tf.float32)

        loss_fn = trend_regularized_loss(r_acc=0.1)
        l = loss_fn(y_true, y_pred).numpy()

        # True diffs: [+1,+1,-1,-1], Pred diffs: [+1,+1,-1,-1]
        # All 4 match → penalty = 0.0
        mse_expected = np.mean((np.array([1.0, 2.0, 3.0, 2.0, 1.0])
                                - np.array([1.0, 2.1, 2.9, 2.1, 0.9])) ** 2)
        self.assertAlmostEqual(l, mse_expected, places=5)

    def test_loss_handles_batched_input(self):
        """Loss works with (batch, 1) shaped inputs from model."""
        y_true = tf.constant([[[1.0], [2.0], [3.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[1.2], [1.8], [3.3]]], dtype=tf.float32)

        loss_fn = trend_regularized_loss(r_acc=0.1)
        l = loss_fn(y_true, y_pred).numpy()

        # True diffs: [+1, +1], Pred diffs: [+0.6, +1.5] → both match
        # penalty = 0
        mse_expected = np.mean((np.array([1.0, 2.0, 3.0])
                                - np.array([1.2, 1.8, 3.3])) ** 2)
        self.assertAlmostEqual(l, mse_expected, places=5)

    def test_loss_zero_trend_for_flat_sequences(self):
        """When values don't change, sign is 0, which doesn't match ±1."""
        y_true = tf.constant([[[5.0], [5.0], [5.0], [5.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[5.0], [5.0], [5.0], [5.0]]], dtype=tf.float32)

        loss_fn = trend_regularized_loss(r_acc=0.1)
        l = loss_fn(y_true, y_pred).numpy()
        self.assertAlmostEqual(l, 0.0, places=5)

    def test_different_r_acc_values(self):
        """Changing r_acc scales the penalty proportionally."""
        y_true = tf.constant([[[1.0], [3.0], [2.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[3.0], [1.0], [4.0]]], dtype=tf.float32)

        mse_val = np.mean((np.array([[[1.0], [3.0], [2.0]]])
                           - np.array([[[3.0], [1.0], [4.0]]])) ** 2)

        l1 = trend_regularized_loss(r_acc=0.0)(y_true, y_pred).numpy()
        l2 = trend_regularized_loss(r_acc=0.5)(y_true, y_pred).numpy()
        l3 = trend_regularized_loss(r_acc=1.0)(y_true, y_pred).numpy()

        self.assertAlmostEqual(l1, mse_val, places=5)
        self.assertAlmostEqual(l2, mse_val + 0.5, places=5)
        self.assertAlmostEqual(l3, mse_val + 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
