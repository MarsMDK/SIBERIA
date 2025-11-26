"""
Community detection and plotting tests for TSeries on all implemented models.

This script checks that, for a slightly structured synthetic time-series matrix,
the TSeries class can:
- build the graph (for each model: naive, bSRGM, bSCM)
- run community_detection (bic objective, few trials)
- produce plots without error: plot_graph, plot_communities, plot_block_matrix
"""

import unittest
import numpy as np

from siberia import TSeries


RANDOM_SEED = 321
N = 10
T = 120


class TestTSeriesCommunitiesAndPlots(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(RANDOM_SEED)

        # Build a weakly modular synthetic dataset:
        # first half somewhat correlated, second half somewhat correlated,
        # with some differences between the two groups.
        base = rng.normal(size=(N, T))
        base[: N // 2] += 0.5 * base[0]        # reinforce group 1
        base[N // 2 :] += 0.5 * base[N // 2]   # reinforce group 2

        cls.data = base.astype(float)

    def _run_full_pipeline_for_model(self, model: str):
        """
        Helper that runs:
        - init + compute_signature
        - fit & predict (for bSRGM / bSCM)
        - build_graph
        - plot_graph (show=False)
        - community_detection
        - plot_communities, plot_block_matrix
        """

        ts = TSeries(self.data, n_jobs=1)
        ts.compute_signature()

        if model == "naive":
            ts.model = "naive"
        else:
            ts.fit(
                model=model,
                maxiter=1000,
                max_nfev=1000,
                verbose=0,
                tol=1e-8,
                eps=1e-8,
                solver_type="fixed_point",
            )
            ts.predict()

        # Build graph and basic sanity check
        ts.build_graph(fdr_correction_flag=True, alpha=0.05)
        self.assertIsNotNone(ts.graph)
        self.assertEqual(ts.graph.shape, (N, N))

        # Plot adjacency (just check it runs)
        ts.plot_graph(export_path="", show=False)

        # Community detection: few trials to keep test quick
        comm = ts.community_detection(
            trials=20,
            n_jobs=1,
            method="bic",
            show=False,
            random_state=RANDOM_SEED,
            starter="mixture",
        )

        self.assertEqual(len(comm), N)
        self.assertGreaterEqual(np.min(comm), 0)
        self.assertLess(np.max(comm), N)  # labels compact

        # Reordered graph + community boxes
        ts.plot_communities(export_path="", show=False)

        # Block matrix based on communities
        M = ts.plot_block_matrix(export_path="", show=False)
        k = len(np.unique(comm))
        self.assertEqual(M.shape, (k, k))

    def test_naive_model_communities_and_plots(self):
        self._run_full_pipeline_for_model("naive")

    def test_bSRGM_model_communities_and_plots(self):
        self._run_full_pipeline_for_model("bSRGM")

    def test_bSCM_model_communities_and_plots(self):
        self._run_full_pipeline_for_model("bSCM")


if __name__ == "__main__":
    unittest.main()
