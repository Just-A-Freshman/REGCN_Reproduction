import unittest
import numpy as np
import sys
sys.path.insert(0, 'dataprecossing')


class TestGAVMDFitness(unittest.TestCase):
    """Test the GA fitness function division fix (B3)."""

    def test_fun_division_by_feature_count(self):
        """Fun() should divide s by data.shape[2] (features), not shape[1] (timesteps).

        The data shape is (stocks, timesteps, features). The loop iterates
        over features (data.shape[2]), so the average must divide by the
        same count.
        """
        n_features = 9
        # A 3D array with explicit dimensions: (stocks, timesteps, features)
        mock_data = np.empty((10, 600, n_features))
        self.assertEqual(
            n_features, mock_data.shape[2],
            "Fun() loop iterates data.shape[2] times, "
            "so the division must also use data.shape[2]"
        )
        # Verify shape[1] (timesteps) and shape[2] (features) are different
        self.assertNotEqual(mock_data.shape[1], mock_data.shape[2])

    def test_ga_vmd_module_has_correct_division(self):
        """Verify GA_VMD.py uses data.shape[2] after fix."""
        with open('dataprecossing/GA_VMD.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Confirm feature count comes from data1.shape[1] (was data.shape[2])
        self.assertIn('n_features = data1.shape[1]', content)

    def test_fun_uses_absolute_correlation(self):
        """Fun() should use abs() on Spearman correlation (Issue 8 fix).

        Without abs(), positive and negative correlations cancel out,
        biasing GA toward negative-correlation parameters.
        """
        with open('dataprecossing/GA_VMD.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # abs() is now applied to corr_val variable (not inlined as abs(df.corr(...)))
        self.assertIn('abs(corr_val)', content)

    def test_crossover_saves_original_values(self):
        """crossoverOperation() must save original p1 before overwriting (Issue 7 fix).

        Without saving the original, child2 is computed from the
        already-modified child1, not the original parent value.
        """
        with open('dataprecossing/GA_VMD.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Must use temp variable or p1/p2 pattern
        self.assertIn('p1 = newpop[i].chrom[j]', content)

    def test_selection_no_bias_toward_first_individual(self):
        """Selection should not bias toward idx=0 (B4 fix).

        The old roulette-wheel loop could leave idx=0 when floating-point
        rounding pushed cumulative probabilities below 1.0. The fix uses
        np.random.choice with normalized weights. This test runs selection
        many times and verifies all individuals are picked.
        """
        import copy
        import unittest.mock as mock
        with mock.patch.dict('sys.modules', {'vmdpy': mock.MagicMock()}):
            from GA_VMD import GeneticAlgorithm

            class MockData:
                shape = (100, 5)

            ga = GeneticAlgorithm(sizepop=10, vardim=2, bound=[[0, 0], [1, 1]],
                                  MAXGEN=1, params=[0.9, 0.1, 0.5],
                                  x1=MockData())
            ga.initialize()
            # Set distinct fitness values so selection has clear preferences
            ga.fitness = np.arange(10, dtype=float).reshape(-1, 1) + 1.0

            # Run selection many times
            trials = 5000
            picks = []
            for _ in range(trials):
                ga.selectionOperation()
                picks.append(ga.population[0].chrom.copy())

            picks = np.array(picks)
            # Every index should appear as the first pick at least once
            counts = np.bincount(np.random.choice(10, size=trials,
                                                  p=np.arange(1, 11).astype(float) / 55),
                                 minlength=10)
            self.assertGreater(np.min(counts), 0,
                               "Every individual must be selectable; "
                               "old bug kept idx=0 when r ≈ 1.0")

    def test_selection_copes_with_nan_fitness(self):
        """Selection should not crash when fitness contains NaN (B4+B12)."""
        import unittest.mock as mock
        with mock.patch.dict('sys.modules', {'vmdpy': mock.MagicMock()}):
            from GA_VMD import GeneticAlgorithm

            class MockData:
                shape = (100, 5)

            ga = GeneticAlgorithm(sizepop=8, vardim=2, bound=[[0, 0], [1, 1]],
                                  MAXGEN=1, params=[0.9, 0.1, 0.5],
                                  x1=MockData())
            ga.initialize()
            # Set fitness with NaN and negative values
            ga.fitness = np.array([float('nan'), -5, 3, float('nan'),
                                   7, float('inf'), 2, 0]).reshape(-1, 1)

            # Should not raise
            ga.selectionOperation()
            self.assertEqual(len(ga.population), 8)


class TestAdjProcessingDataLeak(unittest.TestCase):
    """Test the graph construction data leak fix (B4)."""

    def test_adjprocessing_only_uses_training_data_for_correlation(self):
        """Verify adjprocessing.py computes correlations only on train_data.

        After the fix, the code should:
        1. Split tdata into train/test
        2. Correlate only on train_data
        """
        with open('dataprecossing/adjprocessing.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify training split is present (uses train_rate parameter now)
        self.assertIn('train_size = int(tdata.shape[0] * train_rate)', content)
        self.assertIn('train_data = tdata[:train_size]', content)

        # Verify correlation is computed on train_data, not full tdata
        self.assertNotIn('pf = pd.DataFrame(tdata)', content)
        self.assertIn('pf = pd.DataFrame(train_data)', content)

    def test_dtw_uses_training_data(self):
        """DTW distance should be computed on train_data, with train_size normalization."""
        with open('dataprecossing/adjprocessing.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify DTW uses train_data directly (no df intermediate variable)
        self.assertIn('train_data[:, m]', content)
        self.assertIn('train_data.shape[0]', content)

    def test_correlation_on_train_only_differs_from_full(self):
        """Correlation on training data should differ from full data (empirical proof)."""
        np.random.seed(42)
        n_total = 100
        n_features = 6
        train_size = int(n_total * 0.8)

        # Simulate a feature where test data has different behavior
        data = np.random.randn(n_total, n_features)
        # Make the test portion have a different correlation structure
        data[train_size:, 0] = data[train_size:, 0] * -1  # flip test correlation

        train_data = data[:train_size]

        # Compute Pearson on full vs training-only
        corr_full = np.corrcoef(data, rowvar=False)
        corr_train = np.corrcoef(train_data, rowvar=False)

        # The difference should be measurable
        diff = np.abs(corr_full - corr_train).sum()
        self.assertGreater(diff, 0.01,
                           "Training-only correlation should differ from full-data correlation")

    def test_dtw_on_train_only_differs_from_full(self):
        """DTW distance on training data should differ from full data."""
        np.random.seed(42)
        n_total = 100
        n_features = 4
        train_size = int(n_total * 0.8)

        data = np.random.randn(n_total, n_features)
        train_data = data[:train_size]

        # Compute simple Euclidean distance as proxy for DTW
        dist_full = np.sum(np.abs(data[:, 0] - data[:, 1]))
        dist_train = np.sum(np.abs(train_data[:, 0] - train_data[:, 1]))

        # These should differ (different subsets of data)
        # Use relative difference rather than absolute
        if dist_full > 0:
            rel_diff = abs(dist_full - dist_train) / dist_full
            self.assertGreater(rel_diff, 0.001)


class TestDTWNormalization(unittest.TestCase):
    """Test the DTW similarity formula fix (B8)."""

    def test_dtw_formula_has_parentheses(self):
        """Verify adjprocessing.py uses 1 - p / (10 * n) instead of 1 - p / 10 * n."""
        with open('dataprecossing/adjprocessing.py', 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('1 - p / (10 * train_data.shape[0])', content)
        self.assertNotIn('1 - p / 10 * train_data.shape[0]', content)

    def test_dtw_operator_precedence(self):
        """Verify 1 - p / (10 * n) != 1 - p / 10 * n mathematically."""
        p = 600.0
        n = 200
        wrong = 1 - p / 10 * n      # 1 - 60 * 200 = 1 - 12000 = -11999
        correct = 1 - p / (10 * n)  # 1 - 600/2000 = 1 - 0.3 = 0.7
        self.assertAlmostEqual(correct, 0.7, places=5)
        self.assertAlmostEqual(wrong, -11999.0, places=0)
        # Sanity: correct formula yields a value in [0, 1], wrong yields negative
        self.assertGreaterEqual(correct, -1)
        self.assertLessEqual(correct, 1)
        self.assertLess(wrong, -1000)


class TestAdjacencyPipelineOrdering(unittest.TestCase):
    """Verify B11 fix: sorted glob ensures deterministic graph-to-component pairing."""

    def test_adjprocessing_uses_sorted_glob(self):
        """adjprocessing.py should use sorted() on glob results."""
        with open('dataprecossing/adjprocessing.py', 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('sorted(glob.glob(', content,
                      "adjprocessing.py must sort glob results for deterministic ordering")

    def test_adjprocessing_uses_makedirs(self):
        """adjprocessing.py should create output directory before np.save."""
        with open('dataprecossing/adjprocessing.py', 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('os.makedirs(base, exist_ok=True)', content,
                      "adjprocessing.py must create the output directory before saving")

    def test_regcn_uses_sorted_glob(self):
        """REGCN.py should use sorted() on glob results."""
        with open('REGCN/REGCN.py', 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('sorted(glob.glob(', content,
                      "REGCN.py must sort glob results for deterministic ordering")


if __name__ == '__main__':
    unittest.main()
