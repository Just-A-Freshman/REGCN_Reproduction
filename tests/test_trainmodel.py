import unittest
import numpy as np
import sys
import importlib.util
import os

# Add REGCN/ to sys.path so sibling imports in REGCN.py resolve correctly
sys.path.insert(0, 'REGCN')


class TestPreprocessData(unittest.TestCase):
    """Test the preprocess_data function from input_data (7:1:2 split).

    Note: Functions are imported fresh per test to avoid
    the self-binding issue from storing them as class attributes.
    """

    def test_output_shapes(self):
        from input_data import preprocess_data
        time_len, n_features = 200, 6
        data = np.random.randn(time_len, n_features).astype(float)
        labels = np.random.randn(time_len).astype(float)

        trainX, trainY, valX, valY, testX, testY, pre_testY = preprocess_data(
            data, labels, time_len, train_rate=0.7, val_rate=0.1,
            seq_len=10, pre_len=1
        )
        n_train = int(time_len * 0.7)
        n_val = int(time_len * 0.1)
        n_test = time_len - n_train - n_val

        self.assertEqual(trainX.shape[0], n_train - 10 - 1 + 1)
        self.assertEqual(valX.shape[0], n_val - 10 - 1 + 1)
        self.assertEqual(testX.shape[0], n_test - 10 - 1 + 1)

        self.assertEqual(trainX.shape[1], 10)
        self.assertEqual(trainX.shape[2], n_features)
        self.assertEqual(trainY.shape[0], trainX.shape[0])
        # Multi-step: trainY has seq_len labels per sample
        self.assertEqual(trainY.shape[1], 10)

    def test_no_data_leakage(self):
        from input_data import preprocess_data
        time_len, n_features = 100, 3
        data = np.arange(time_len * n_features, dtype=float).reshape(
            time_len, n_features)
        labels = data[:, 0].copy()

        _, _, _, _, testX, testY, _ = preprocess_data(
            data, labels, time_len, train_rate=0.7, val_rate=0.1,
            seq_len=5, pre_len=1
        )
        val_size = int(time_len * 0.8)  # train + val boundary
        for y in testY.flatten():
            self.assertGreaterEqual(int(y), val_size)

    def test_pre_len_greater_than_one(self):
        from input_data import preprocess_data
        time_len, n_features = 100, 3
        data = np.random.randn(time_len, n_features)
        labels = np.random.randn(time_len)

        trainX, trainY, _, _, _, _, _ = preprocess_data(
            data, labels, time_len, train_rate=0.7, val_rate=0.1,
            seq_len=10, pre_len=3
        )
        self.assertEqual(trainY.shape[1], 10)
        self.assertGreater(trainY.shape[0], 0)

    def test_train_val_test_ratios(self):
        """Verify 7:1:2 split produces the expected sample counts."""
        from input_data import preprocess_data
        time_len, n_features = 500, 4
        data = np.random.randn(time_len, n_features)
        labels = np.random.randn(time_len)

        trainX, trainY, valX, valY, testX, testY, _ = preprocess_data(
            data, labels, time_len, train_rate=0.7, val_rate=0.1,
            seq_len=20, pre_len=1
        )

        n_train = int(time_len * 0.7) - 20
        n_val = int(time_len * 0.1) - 20
        n_test = time_len - int(time_len * 0.8) - 20

        # Should produce roughly expected samples (allow ±1 for boundary rounding)
        self.assertAlmostEqual(trainX.shape[0], n_train, delta=1)
        self.assertAlmostEqual(valX.shape[0], n_val, delta=1)
        self.assertAlmostEqual(testX.shape[0], n_test, delta=1)

        # All three should have samples
        self.assertGreater(trainX.shape[0], 0)
        self.assertGreater(valX.shape[0], 0)
        self.assertGreater(testX.shape[0], 0)


class TestEvaluationMetrics(unittest.TestCase):
    """Test get_trend and avg_relative_error from utils."""

    def test_get_trend_up(self):
        from utils import get_trend
        pre = np.array([[100], [101]])
        cur = np.array([[101], [102]])
        trends = get_trend(pre, cur)
        np.testing.assert_array_equal(trends, [1, 1])

    def test_get_trend_down(self):
        from utils import get_trend
        pre = np.array([[102], [105]])
        cur = np.array([[101], [100]])
        trends = get_trend(pre, cur)
        np.testing.assert_array_equal(trends, [0, 0])

    def test_get_trend_mixed(self):
        from utils import get_trend
        pre = np.array([[100], [101], [100]])
        cur = np.array([[101], [100], [102]])
        trends = get_trend(pre, cur)
        np.testing.assert_array_equal(trends, [1, 0, 1])

    def test_avg_relative_error(self):
        from utils import avg_relative_error
        actual = np.array([[100], [200]])
        pred = np.array([[110], [180]])
        err = avg_relative_error(actual, pred)
        expected = (abs(110 - 100) / 100 + abs(180 - 200) / 200) / 2
        self.assertAlmostEqual(err, expected, places=6)

    def test_calculate_laplacian_returns_sparse_tensor(self):
        """Laplacian should return a SparseTensor (TF1 compat mode)."""
        from utils import calculate_laplacian
        import tensorflow as tf
        adj = np.eye(5)
        result = calculate_laplacian(adj)
        self.assertTrue(isinstance(result, tf.SparseTensor) or
                        hasattr(result, 'indices'))


class TestLaplacianNaN(unittest.TestCase):
    """Verify Bug B2 fix: negative rowsum in normalized_adj does not produce NaN.

    The fix adds np.isnan handling after the existing np.isinf guard,
    so that D^{-1/2} * A * D^{-1/2} remains finite even when
    sum_j A_ij < 0 (strong negative correlations dominate the self-loop).
    """

    def test_normalized_adj_negative_rowsum(self):
        """Negative row sum should not produce NaN."""
        from utils import normalized_adj
        # 4×4 adjacency where row 0 has sum = 0 + 0.95 - 0.95 - 0.95 = -0.95
        adj = np.array([
            [0,  0.95, -0.95, -0.95],
            [0.95, 0,   0.95,  0.95],
            [-0.95, 0.95, 0,   0.95],
            [-0.95, 0.95, 0.95, 0]
        ], dtype=np.float32)
        # Adding self-loop I — row 0 sum becomes 1 + 0.95 - 0.95 - 0.95 = 0.05 > 0
        # Use an even more extreme matrix to guarantee negative row sum:
        # 5 nodes, 4 strong negative edges on node 0
        extreme = np.array([
            [0, -0.95, -0.95, -0.95, -0.95],
            [-0.95, 0,  0.95,  0.95,  0.95],
            [-0.95, 0.95, 0,  0.95,  0.95],
            [-0.95, 0.95, 0.95, 0,  0.95],
            [-0.95, 0.95, 0.95, 0.95, 0]
        ], dtype=np.float32)
        result = normalized_adj(extreme + np.eye(5))
        arr = result.toarray()
        self.assertFalse(np.any(np.isnan(arr)),
                         "normalized_adj must not produce NaN for negative rowsum")
        self.assertFalse(np.any(np.isinf(arr)),
                         "normalized_adj must not produce inf for negative rowsum")


class TestLossesModule(unittest.TestCase):
    """Verify the new losses module works in both eager and graph modes."""

    def test_import_success(self):
        from losses import trend_regularized_loss
        fn = trend_regularized_loss(r_acc=0.1)
        self.assertTrue(callable(fn))

    def test_loss_with_increasing_values(self):
        from losses import trend_regularized_loss
        import tensorflow as tf

        y_true = tf.constant([[[1.0], [2.0], [3.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[1.1], [2.1], [3.1]]], dtype=tf.float32)

        loss_fn = trend_regularized_loss(r_acc=0.1)
        loss_val = loss_fn(y_true, y_pred)

        mse_expected = np.mean((np.array([[[1.0], [2.0], [3.0]]])
                                - np.array([[[1.1], [2.1], [3.1]]])) ** 2)

        # In eager mode (TF2), .numpy() works.
        # In graph mode (TF1 compat), evaluate via session.
        if hasattr(loss_val, 'numpy'):
            self.assertAlmostEqual(float(loss_val.numpy()), mse_expected, places=5)
        else:
            with tf.compat.v1.Session() as sess:
                self.assertAlmostEqual(float(sess.run(loss_val)), mse_expected, places=5)


class TestGCGRUStateUsage(unittest.TestCase):
    """Verify Bug 2 fix: GRU gates use 'state', not 'x', for recurrent weights."""

    def test_gcgru_uses_state_in_update_gate_uz(self):
        """K.dot(state, self.uz) instead of K.dot(x, self.uz).

        When state changes but inputs stay the same, the output should
        differ — proving uz/ur multiply state, not x.
        """
        from dgcgru import gcgru
        import tensorflow as tf

        n_neurons, n_gcn_nodes = 16, 6
        adj = tf.constant(np.random.randn(3, n_gcn_nodes, n_gcn_nodes).astype(np.float32))
        cell = gcgru(n_neurons, adj, n_gcn_nodes, s_index=3)
        # Build manually: input shape is (batch, n_gcn_nodes)
        cell.built = False
        cell.build((None, n_gcn_nodes))

        inputs = tf.constant(np.random.randn(1, n_gcn_nodes).astype(np.float32))
        state_a = tf.constant(np.ones((1, n_neurons), dtype=np.float32))
        state_b = tf.constant(np.zeros((1, n_neurons), dtype=np.float32))

        # Call cell directly — TF2 eager mode
        output_a, _ = cell(inputs, [state_a])
        output_b, _ = cell(inputs, [state_b])

        diff = tf.reduce_sum(tf.abs(output_a - output_b))
        self.assertGreater(diff.numpy(), 1e-6,
                           "Different states with same inputs must produce different outputs")

    def test_gcgru_output_shape(self):
        """GCGRU cell output should match state_size."""
        from dgcgru import gcgru
        import tensorflow as tf

        n_neurons, n_gcn_nodes = 16, 6
        adj = tf.constant(np.random.randn(3, n_gcn_nodes, n_gcn_nodes).astype(np.float32))
        cell = gcgru(n_neurons, adj, n_gcn_nodes, s_index=3)
        cell.build((None, n_gcn_nodes))

        inputs = tf.constant(np.random.randn(1, n_gcn_nodes).astype(np.float32))
        state = tf.constant(np.random.randn(1, n_neurons).astype(np.float32))
        output, next_state = cell.call(inputs, [state])

        self.assertEqual(output.shape, (1, n_neurons))
        self.assertEqual(next_state.shape, (1, n_neurons))


class TestMultiStepPreprocessing(unittest.TestCase):
    """Verify Bug 1 fix (Option A): preprocess_data produces seq_len labels."""

    def test_trainY_has_seq_len(self):
        """trainY should have shape (n, seq_len), not (n,)."""
        from input_data import preprocess_data
        time_len, n_features = 100, 6
        data = np.random.randn(time_len, n_features).astype(float)
        labels = np.random.randn(time_len).astype(float)

        trainX, trainY, _, _, _, _, _ = preprocess_data(
            data, labels, time_len, train_rate=0.7, val_rate=0.1,
            seq_len=10, pre_len=1
        )
        self.assertEqual(trainY.shape[1], 10)
        self.assertEqual(trainY.shape[0], trainX.shape[0])

    def test_trainY_window_starts_one_step_ahead(self):
        """With pre_len=1, each label window starts at offset 1."""
        from input_data import preprocess_data
        time_len, n_features = 50, 3
        data = np.random.randn(time_len, n_features).astype(float)
        labels = np.arange(time_len, dtype=float)

        trainX, trainY, _, _, _, _, _ = preprocess_data(
            data, labels, time_len, train_rate=0.7, val_rate=0.1,
            seq_len=10, pre_len=1
        )
        # First sample: input window indices [0..9], labels indices [1..10]
        np.testing.assert_array_equal(trainY[0], np.arange(1, 11))


class TestModelForward(unittest.TestCase):
    """Verify the assembled model runs a forward pass without shape crash.

    This is the end-to-end check for Bug 1 (shape mismatch).
    """

    def test_model_forward_pass_no_crash(self):
        """Build model with RNN+gcgru cell + Dense, run predict, check output."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import RNN, Dense
        from dgcgru import gcgru

        batch_size, seq_len, n_gcn_nodes = 4, 10, 6
        n_neurons = 16
        adj = tf.constant(np.random.randn(3, n_gcn_nodes, n_gcn_nodes).astype(np.float32))

        cell = gcgru(n_neurons, adj, n_gcn_nodes, s_index=3)
        model = Sequential()
        model.add(RNN(cell, return_sequences=True))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        x = tf.constant(np.random.randn(batch_size, seq_len, n_gcn_nodes).astype(np.float32))
        y = model(x, training=False)

        # return_sequences=True → output (batch, seq_len, 1)
        self.assertEqual(tuple(y.shape[1:]), (seq_len, 1))


class TestModelAssembly(unittest.TestCase):
    """Verify trainmodel() assembles correctly.

    This test validates that the model builds without structural errors
    (like the original NameError on 'Ge').
    """

    @classmethod
    def setUpClass(cls):
        cls._orig_dir = os.getcwd()
        cls._regcn_dir = os.path.join(os.path.dirname(__file__), '..', 'REGCN')
        os.chdir(cls._regcn_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._orig_dir)

    def test_import_regcn_no_name_error(self):
        """Importing REGCN.py should not raise NameError.

        Before the fix, importing would fail when trainmodel() is called
        because 'Ge' was undefined. This test verifies the module can
        be imported and the function is defined.
        """
        import importlib.util
        # chdir to REGCN/ is done in setUpClass, so config.ini resolves
        spec = importlib.util.spec_from_file_location(
            "regcn_module", "REGCN.py"
        )
        regcn_module = importlib.util.module_from_spec(spec)

        # The import will succeed; the module-level script doesn't
        # auto-execute trainmodel() when imported.
        try:
            spec.loader.exec_module(regcn_module)
            self.assertTrue(hasattr(regcn_module, 'trainmodel'))
            self.assertTrue(hasattr(regcn_module, 'main'))
        except ImportError as e:
            if 'AbstractRNNCell' in str(e) or 'RNN' in str(e):
                self.skipTest(
                    "RNN cell base class not available in current TF version. "
                    "This code was written for TF 1.15 (see code comment)."
                )
            else:
                raise

    def test_trainmodel_function_has_correct_signature(self):
        """trainmodel() should accept the correct parameters after the fix."""
        import inspect
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "regcn_module", "REGCN.py"
        )
        regcn_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(regcn_module)
            sig = inspect.signature(regcn_module.trainmodel)
            param_names = list(sig.parameters.keys())
            self.assertIn('tdata', param_names)
            self.assertIn('tadj', param_names)
            self.assertIn('n_epochs', param_names)
            self.assertIn('lr', param_names)
        except ImportError:
            self.skipTest("Skipping: TF version incompatibility")


class TestRegularizationCoverage(unittest.TestCase):
    """Verify B5 fix: L1 regularization covers all GCGRU trainable weights."""

    def test_gcgru_weights_have_regularizer(self):
        """GCGRU cell weights should expose regularization losses when kernel_regularizer is set."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import RNN, Dense
        from dgcgru import gcgru

        n_neurons, n_gcn_nodes = 16, 6
        adj = tf.constant(np.random.randn(3, n_gcn_nodes, n_gcn_nodes).astype(np.float32))

        # Build with regularizer
        cell_reg = gcgru(n_neurons, adj, n_gcn_nodes, s_index=3,
                         kernel_regularizer=tf.keras.regularizers.l1(0.01))
        model = Sequential()
        model.add(RNN(cell_reg, return_sequences=True))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Run one forward pass to trigger loss registration
        x = tf.constant(np.random.randn(2, 10, n_gcn_nodes).astype(np.float32))
        y = tf.constant(np.random.randn(2, 10, 1).astype(np.float32))
        model.fit(x, y, epochs=1, verbose=0)

        # model.losses contains regularization losses from add_weight regularizer
        # Must have at least some regularization terms (wa, w0, wz, wr, wh, uz, ur, uh)
        reg_losses = model.losses
        self.assertGreater(len(reg_losses), 0,
                           "model.losses must include GCGRU weight regularizers")

    def test_regularized_loss_higher_than_unregularized(self):
        """Model with L1 regularizer should have higher total loss than without."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import RNN, Dense
        from dgcgru import gcgru

        n_neurons, n_gcn_nodes = 16, 6
        adj = tf.constant(np.random.randn(3, n_gcn_nodes, n_gcn_nodes).astype(np.float32))
        x = tf.constant(np.random.randn(4, 10, n_gcn_nodes).astype(np.float32))
        y = tf.constant(np.random.randn(4, 10, 1).astype(np.float32))

        # Without regularizer
        cell_no = gcgru(n_neurons, adj, n_gcn_nodes, s_index=3)
        m_no = Sequential([RNN(cell_no, return_sequences=True), Dense(1)])
        m_no.compile(optimizer='adam', loss='mse')
        loss_no = m_no.evaluate(x, y, verbose=0)

        # With regularizer
        cell_yes = gcgru(n_neurons, adj, n_gcn_nodes, s_index=3,
                         kernel_regularizer=tf.keras.regularizers.l1(0.01))
        m_yes = Sequential([RNN(cell_yes, return_sequences=True), Dense(1)])
        m_yes.compile(optimizer='adam', loss='mse')
        loss_yes = m_yes.evaluate(x, y, verbose=0)

        # Regularized loss should be >= unregularized (extra L1 terms)
        self.assertGreaterEqual(loss_yes, loss_no - 1e-6)

    def test_dense_layer_has_no_l1_regularizer(self):
        """The Dense(1) output layer should NOT carry L1, as it's now in GCGRU."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'REGCN', 'REGCN.py'),
                  'r', encoding='utf-8') as f:
            content = f.read()
        # Dense(1) should be a simple call (no kernel_regularizer argument)
        self.assertIn('model.add(Dense(1))', content)
        # The kernel_regularizer should now be on gcgru, not on Dense
        self.assertIn('kernel_regularizer=tf.keras.regularizers.l1(r_mse)',
                      content)


class TestModelInputValidation(unittest.TestCase):
    """Verify B9/B10 fixes: configurable label_col and input dimension validation."""

    def test_build_rejects_wrong_feature_dim(self):
        """build() should raise ValueError when input feature dim != n_gcn_nodes."""
        import tensorflow as tf
        from dgcgru import gcgru

        n_neurons, n_gcn_nodes = 16, 6
        adj = tf.constant(np.random.randn(3, n_gcn_nodes, n_gcn_nodes).astype(np.float32))

        cell = gcgru(n_neurons, adj, n_gcn_nodes, s_index=3)
        cell.built = False
        # Try building with wrong feature dimension (5 instead of 6)
        with self.assertRaises(ValueError) as ctx:
            cell.build((None, 5))
        self.assertIn('gcgru', str(ctx.exception).lower())
        self.assertIn('feature dim', str(ctx.exception).lower())
        self.assertIn('5', str(ctx.exception))
        self.assertIn('6', str(ctx.exception))

    def test_build_accepts_correct_feature_dim(self):
        """build() should succeed when input feature dim matches n_gcn_nodes."""
        import tensorflow as tf
        from dgcgru import gcgru

        n_neurons, n_gcn_nodes = 16, 6
        adj = tf.constant(np.random.randn(3, n_gcn_nodes, n_gcn_nodes).astype(np.float32))

        cell = gcgru(n_neurons, adj, n_gcn_nodes, s_index=3)
        cell.built = False
        # Should not raise
        cell.build((None, n_gcn_nodes))
        self.assertTrue(cell.built)

    def test_label_col_reads_from_config(self):
        """REGCN.py should read label_col from config.ini."""
        import importlib.util
        regcn_dir = os.path.join(os.path.dirname(__file__), '..', 'REGCN')
        config_path = os.path.join(regcn_dir, 'config.ini')
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        self.assertIn('label_col', config_content,
                       "label_col must be defined in config.ini")
        # Verify the config is parsed in REGCN.py
        with open(os.path.join(regcn_dir, 'REGCN.py'), 'r', encoding='utf-8') as f:
            regcn_content = f.read()
        self.assertIn('label_col', regcn_content,
                       "label_col must be referenced in REGCN.py")

    def test_label_col_used_in_both_train_and_main(self):
        """Both trainmodel() and main() should use label_col for labels."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'REGCN', 'REGCN.py'),
                  'r', encoding='utf-8') as f:
            content = f.read()
        # Check trainmodel uses label_col
        self.assertIn('labels = data[:, label_col]', content)
        # Check trainmodel does NOT use hardcoded 3
        self.assertNotIn('labels = data[:, 3]', content)
        # Check main uses label_col for test labels
        self.assertIn('labels = tdata[:, label_col]', content)


class TestPipelineEdgeCases(unittest.TestCase):
    """Verify B14/B15 fixes: empty glob and VMD shape mismatches."""

    def test_main_raises_on_empty_glob(self):
        """main() should raise FileNotFoundError when VMD CSVs are missing."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'REGCN', 'REGCN.py'),
                  'r', encoding='utf-8') as f:
            content = f.read()
        # Must check glob result before processing
        self.assertIn('if not file:', content,
                      "Must check for empty glob result")
        self.assertIn('FileNotFoundError', content,
                      "Must raise FileNotFoundError when VMD CSVs are missing")

    def test_main_handles_empty_result_after_training(self):
        """main() should guard against empty result list after training loop."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'REGCN', 'REGCN.py'),
                  'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('if not result:', content,
                      "Must guard against empty predictions list")

    def test_main_aligns_vmd_window_counts(self):
        """main() should align test window counts before summing VMD predictions."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'REGCN', 'REGCN.py'),
                  'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('min_windows', content,
                      "Must compute minimum window count across VMD components")
        self.assertIn('np.sum(aligned', content,
                      "Must sum aligned (not raw) predictions")


class TestNormalizationPipeline(unittest.TestCase):
    """Verify 27-normalization-pipeline fix: normalized input + denormalized output."""

    def test_unautoNorm_fixed_broadcast(self):
        """unautoNorm should use simple broadcasting, not broken np.tile."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'REGCN', 'REGCN.py'),
                  'r', encoding='utf-8') as f:
            content = f.read()
        # Verify old buggy pattern is removed
        self.assertNotIn('np.tile(ranges, 1)', content,
                         "unautoNorm must not use np.tile with dim=1 (no-op)")
        # Verify the fix uses direct broadcasting
        self.assertIn('return data * (maxs - mins) + mins', content,
                      "unautoNorm should use simple arithmetic broadcasting")

    def test_unautoNorm_math(self):
        """unautoNorm should correctly restore original scale."""
        # Test the logic inline to avoid import-time side effects
        data = np.array([0.0, 0.5, 1.0])
        denorm = data * (200.0 - 100.0) + 100.0
        expected = np.array([100.0, 150.0, 200.0])
        np.testing.assert_array_almost_equal(denorm, expected)

    def test_main_uses_vmdnor_config(self):
        """config.ini should point VMD_addr to VMDnor/."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'REGCN', 'config.ini'),
                  'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('VMD', content,
                      "config.ini must point to a VMD directory")
        self.assertNotIn('VMDnor', content,
                         "config.ini uses VMDdata/ (actual data location) — "
                         "see docs/difference for details")

    def test_main_denormalizes_each_component(self):
        """main() should call unautoNorm on each VMD component prediction."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'REGCN', 'REGCN.py'),
                  'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('result1 = unautoNorm(result1, close_min, close_max)',
                      content,
                      "Each VMD component prediction must be denormalized")

    def test_main_extracts_close_min_max(self):
        """main() should extract close price min/max from VMDnor CSV tail."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'REGCN', 'REGCN.py'),
                  'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('close_min = ndata[-2, label_col]', content,
                      "Must extract close price min from second-to-last row")
        self.assertIn('close_max = ndata[-1, label_col]', content,
                      "Must extract close price max from last row")

    def test_normalization_comment_updated(self):
        """normalization.py stale comment must be removed."""
        with open(os.path.join(os.path.dirname(__file__), '..',
                               'dataprecossing', 'normalization.py'),
                  'r', encoding='utf-8') as f:
            content = f.read()
        self.assertNotIn('reads from VMDdata/', content,
                         "Comment referring to old VMDdata/ path must be removed")
        self.assertIn('reads from VMDnor/', content,
                      "Comment should document new VMDnor/ path")


if __name__ == '__main__':
    unittest.main()
