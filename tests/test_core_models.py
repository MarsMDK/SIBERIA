"""
Basic correctness tests for the TSeries core pipeline on all implemented models.

This script checks that, for a small synthetic time-series matrix, the TSeries
class can:
- compute signatures
- fit bSRGM and bSCM
- predict probabilities
- check distribution signatures (KS score)
- build the filtered graph

Models tested: 'naive', 'bSRGM', 'bSCM'.
"""

import unittest
import numpy as np

from siberia import TSeries


RANDOM_SEED = 123
N = 6
T = 80


class TestTSeriesCoreModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(RANDOM_SEED)
        # Synthetic standardized-ish data; TSeries will standardize rows anyway
        data = rng.normal(size=(N, T)).astype(float)
        cls.data = data

    def _check_graph_properties(self, ts):
        self.assertIsNotNone(ts.graph)
        self.assertEqual(ts.graph.shape, (N, N))
        self.assertTrue(np.all(np.isin(ts.graph, [-1, 0, 1])))

    def test_compute_signature(self):
        ts = TSeries(self.data, n_jobs=1)
        sig = ts.compute_signature()
        self.assertEqual(sig.shape, (N, N))
        self.assertTrue(np.all(np.isfinite(sig)))
        
    def test_naive_model_graph(self):
        """
        Naive model doesn't fit or predict, but uses the binary signature directly.
        """
        ts = TSeries(self.data, n_jobs=1)
        ts.compute_signature()
        ts.model = "naive"
        graph = ts.build_graph(fdr_correction_flag=False, alpha=0.05)
        self.assertEqual(graph.shape, (N, N))
        self._check_graph_properties(ts)
        
    def test_bSRGM_pipeline(self):
        """
        Full pipeline for bSRGM:
        - compute_signature
        - fit
        - predict
        - check_distribution_signature
        - build_graph
        """
        ts = TSeries(self.data, n_jobs=1)
        ts.compute_signature()

        ts.fit(
            model="bSRGM",
            maxiter=500,
            max_nfev=500,
            verbose=0,
            tol=1e-8,
            eps=1e-8,
            solver_type="fixed_point",  # not used by bSRGM but accepted
        )
        self.assertIsNotNone(ts.params)
        self.assertGreater(len(ts.params), 0)
        self.assertLessEqual(ts.norm_rel_error, 1e-6)
        if hasattr(ts, 'norm_rel_error') and ts.norm_rel_error < 1e-6:
            print("bSRGM fit converged with norm_rel_error:", ts.norm_rel_error)

        pit_plus, pit_minus = ts.predict()
        self.assertEqual(pit_plus.shape, (N, T))
        self.assertEqual(pit_minus.shape, (N, T))
        
        # Lightweight ensemble to keep test fast
        ks_score = ts.check_distribution_signature(
            n_ensemble=100,
            ks_score=True,
            alpha=0.05,
        )
        self.assertGreaterEqual(ks_score, 0.0)
        self.assertLessEqual(ks_score, 1.0)
        print("bSRGM KS score:", ks_score)

    def test_bSCM_pipeline(self):
        """
        Full pipeline for bSCM:
        - compute_signature
        - fit (fixed-point solver)
        - predict
        - check_distribution_signature
        - build_graph
        """
        ts = TSeries(self.data, n_jobs=1)
        ts.compute_signature()

        ts.fit(
            model="bSCM",
            maxiter=2000,
            max_nfev=2000,
            verbose=0,
            tol=1e-8,
            eps=1e-8,
            solver_type="fixed_point",
        )
        self.assertIsNotNone(ts.params)
        self.assertGreater(len(ts.params), 0)
        self.assertLessEqual(ts.norm_rel_error, 1e-6)
        if hasattr(ts, 'norm_rel_error') and ts.norm_rel_error < 1e-6:
            print("bSCM fit converged with norm_rel_error:", ts.norm_rel_error)

        pit_plus, pit_minus = ts.predict()
        self.assertEqual(pit_plus.shape, (N, T))
        self.assertEqual(pit_minus.shape, (N, T))

        ks_score = ts.check_distribution_signature(
            n_ensemble=100,
            ks_score=True,
            alpha=0.05,
        )
        self.assertGreaterEqual(ks_score, 0.0)
        self.assertLessEqual(ks_score, 1.0)
        print("bSCM KS score:", ks_score)

        graph = ts.build_graph(fdr_correction_flag=True, alpha=0.05)
        self.assertEqual(graph.shape, (N, N))
        self._check_graph_properties(ts)
        
        self.assertGreaterEqual(ks_score, 0.0)
        self.assertLessEqual(ks_score, 1.0)

        graph = ts.build_graph(fdr_correction_flag=True, alpha=0.05)
        self.assertEqual(graph.shape, (N, N))
        self._check_graph_properties(ts)


if __name__ == "__main__":
    unittest.main()
