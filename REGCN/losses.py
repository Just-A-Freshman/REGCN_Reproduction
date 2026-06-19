import tensorflow as tf


def trend_regularized_loss(r_acc=0.1):
    """Custom loss with trend direction penalty.

    Implements the regularized loss from the paper (Eq. 15):
        L = MSE(y, ŷ) + λ₂ · (1/n) Σ 1[sign(Δy) ≠ sign(Δŷ)]

    The L1 weight regularization (λ₁·‖w‖₁) is applied separately via
    kernel_regularizer on the Dense output layer, since Keras's
    model.fit() handles those losses automatically.
    """
    def loss(y_true, y_pred):
        # y_true / y_pred shape: (batch, seq_len, 1)

        # MSE component
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Trend direction component — within each sequence
        # Differences along the seq_len axis, never across batch boundary
        true_diff = y_true[:, 1:, :] - y_true[:, :-1, :]  # (batch, seq_len-1, 1)
        pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]

        true_sign = tf.sign(true_diff)
        pred_sign = tf.sign(pred_diff)

        trend_penalty = tf.reduce_mean(
            tf.cast(tf.not_equal(true_sign, pred_sign), tf.float32)
        )

        return mse + r_acc * trend_penalty

    return loss
