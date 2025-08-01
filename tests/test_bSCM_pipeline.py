#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline test for the bSCM model.
Runs all steps: load data, standardize, compute signature,
fit the model, predict, validate signature, build graphs, and community detection.
"""

import os
import time
import numpy as np
import unittest
import TSeriesClass_new as tsc


DATA_DIRECTORY = 'data_new'
FILE_NAME = "106016_all_rois_ts.txt"
ALPHA = 0.05
COMMUNITY_TRIALS = 20
MODEL = 'bSCM'


class TestbSCM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load and standardize matrix
        file_path = os.path.join(DATA_DIRECTORY, FILE_NAME)
        matrix = np.loadtxt(file_path).reshape(2400, 116).T
        for i in range(matrix.shape[0]):
            row_mean = np.mean(matrix[i, :])
            row_std = np.std(matrix[i, :]) or 1e-8
            matrix[i, :] = (matrix[i, :] - row_mean) / row_std
        cls.matrix = matrix

    def test_full_pipeline(self):
        TS = tsc.TSeries(self.matrix, n_jobs=4)

        # Signature
        signature = TS.compute_signature()
        self.assertEqual(signature.shape, (116, 116))

        # Fit model
        start = time.time()
        TS.fit(MODEL, verbose=0)
        runtime_fit = time.time() - start
        self.assertGreater(TS.ll, -1e9)
        print(f"[{MODEL}] Fitting completed in {runtime_fit:.2f}s. AIC={TS.aic:.2f}")

        # Predict
        TS.predict()
        self.assertIsNotNone(TS.pit_plus)

        # Validate + Graphs
        start = time.time()
        filtered_signature = TS.validate_signature(alpha=ALPHA)
        naive_graph, filtered_graph = TS.build_graph()
        runtime_val = time.time() - start
        self.assertEqual(naive_graph.shape, (116, 116))
        self.assertEqual(filtered_graph.shape, (116, 116))
        print(f"[{MODEL}] Validation completed in {runtime_val:.2f}s.")

        # Community Detection
        start = time.time()
        comm_stats = TS.community_detection(
            trials=COMMUNITY_TRIALS,
            n_jobs=8,
            method="bic",
            show=False
        )
        runtime_comm = time.time() - start
        self.assertIn('naive', comm_stats)
        self.assertIn('filtered', comm_stats)
        print(f"[{MODEL}] Community detection in {runtime_comm:.2f}s.")
        print(f"    Naive min_loss={comm_stats['naive']['min_loss']:.2f}")
        print(f"    Filtered min_loss={comm_stats['filtered']['min_loss']:.2f}")


if __name__ == "__main__":
    unittest.main()
