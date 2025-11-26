SIBERIA: SIgned BEnchmarks foR tIme series Analysis
===================================================

SIBERIA is a Python 3 package for rigorous **signed time-series network analysis**.

Starting from an :math:`N \times T` matrix of standardized time series, it builds **signed co-fluctuation signatures**, fits **maximum-entropy null models**, and produces **validated signed graphs** that can be analyzed via community detection and block-structure inspection.

SIBERIA implements advanced null models (**bSRGM**, **bSCM**, plus a simple **naive** projection) to distinguish meaningful mesoscale structure from noise, supporting reproducible and interpretable time-series network analysis.

Main Features
=============

SIBERIA includes methods to:

- Compute binary **signature matrices** for co-fluctuations.
- Fit maximum-entropy **null models** (``bSRGM``, ``bSCM``) with LSQ or fixed-point solvers.
- Predict event probabilities (``pit_plus``, ``pit_minus``) from fitted parameters.
- Compare empirical signatures with null-model signatures via **ensemble and analytical distributions** and **KS scores**.
- Build **signed graphs** via analytical p-values and optional **FDR correction**.
- Perform **community detection** minimizing either
  - the BIC of a signed SBM (``method="bic"``), or
  - the signed network frustration (``method="frustration"``).
- Visualize results as adjacency heatmaps, community-reordered matrices, and community-level block matrices.

For more information about maximum-entropy methods, visit  
`Maximum Entropy Hub <https://meh.imtlucca.it>`_.

Citation
========

If you use **SIBERIA** in your research, please cite the following paper:

.. code-block:: bibtex

    @misc{divece2025assessingimbalancesignedbrain,
          title={Assessing (im)balance in signed brain networks}, 
          author={Marzio Di Vece and Emanuele Agrimi and Samuele Tatullo and Tommaso Gili and Miguel Ibáñez-Berganza and Tiziano Squartini},
          year={2025},
          eprint={2508.00542},
          archivePrefix={arXiv},
          primaryClass={physics.soc-ph},
          url={https://arxiv.org/abs/2508.00542}, 
    }

Installation
============

SIBERIA can be installed via ``pip``:

.. code-block:: bash

    pip install siberia

If you already installed the package and want to upgrade it:

.. code-block:: bash

    pip install siberia --upgrade

Dependencies
============

SIBERIA uses the following dependencies:

- **numpy** for numerical operations
- **scipy** for optimization and statistical functions
- **pandas** for structured data handling
- **fast-poibin** for Poisson–Binomial distributions
- **joblib** for parallel computation
- **statsmodels** for multiple testing corrections (FDR)
- **matplotlib** and **seaborn** for visualization
- **tqdm** for progress bars
- **numba** for accelerating heavy computations

They can be easily installed via ``pip``:

.. code-block:: bash

    pip install numpy scipy pandas fast-poibin joblib statsmodels matplotlib seaborn tqdm numba

How-to Guidelines
=================

The main entry point of SIBERIA is the :class:`TSeries` class, initialized with an
:math:`N \times T` float matrix representing :math:`N` time series of length :math:`T`.

If the rows are not standardized (mean ≈ 0, std ≈ 1), they are standardized internally.

Initialization
--------------

.. code-block:: python

    from siberia import TSeries
    import numpy as np

    # Tij is a 2D numpy array of shape (N, T) with float values
    Tij = np.asarray(Tij, dtype=float)

    T = TSeries(data=Tij, n_jobs=4)

After initialization you can explore marginal statistics of the binarized series:

.. code-block:: python

    T.ai_plus, T.ai_minus   # row-wise positive / negative counts
    T.kt_plus, T.kt_minus   # column-wise positive / negative counts
    T.a_plus,  T.a_minus    # total positive / negative counts

Computing the Signature
-----------------------

The signature captures concordant and discordant co-fluctuation motifs:

.. code-block:: python

    binary_signature = T.compute_signature()

Internally:

- Concordant motifs = positive–positive + negative–negative
- Discordant motifs = positive–negative + negative–positive
- Binary signature = concordant − discordant

The result is stored in:

.. code-block:: python

    T.binary_signature

Fitting Null Models
-------------------

You can list the available models:

.. code-block:: python

    T.implemented_models
    # ['naive', 'bSRGM', 'bSCM']

Choose and fit a maximum-entropy model:

.. code-block:: python

    T.fit(
        model="bSCM",          # 'naive', 'bSRGM', or 'bSCM'
        maxiter=1000,
        max_nfev=1000,
        tol=1e-8,
        eps=1e-8,
        solver_type="fixed_point"  # 'fixed_point' or 'lsq' for bSCM
    )

After fitting, the following attributes become available:

.. code-block:: python

    T.params            # fitted parameters
    T.ll                # log-likelihood
    T.jac               # Jacobian of the constraints
    T.norm              # infinite norm of the Jacobian
    T.norm_rel_error    # relative norm of the constraint error
    T.aic               # Akaike Information Criterion

Predicting Event Probabilities
------------------------------

Compute the expected probability of observing positive and negative events in each :math:`(i, t)` entry:

.. code-block:: python

    pit_plus, pit_minus = T.predict()

These matrices are stored in:

.. code-block:: python

    T.pit_plus
    T.pit_minus

and represent the null-model probabilities for positive and negative events for each time series and time step.

Checking Signature Distributions
--------------------------------

You can compare the empirical signature with the null-model signature distributions using ensemble sampling and analytical calculations, summarized by a Kolmogorov–Smirnov score:

.. code-block:: python

    ks_score = T.check_distribution_signature(
        n_ensemble=1000,
        ks_score=True,
        alpha=0.05
    )

This method:

- Generates an ensemble of signatures from the fitted model.
- Computes analytical signature distributions (binomial / Poisson–Binomial).
- Performs KS tests for node pairs and returns a **normalized KS score** in :math:`[0, 1]`.

The outputs are stored as:

.. code-block:: python

    T.ks_score                # fraction of pairs where p_KS >= alpha
    T.ensemble_signature      # N × N × n_ensemble
    T.analytical_signature    # N × N × n_ensemble
    T.analytical_signature_dist  # N × N × (T+1) PMFs (bSCM) or equivalent

A higher ``ks_score`` indicates better agreement between ensemble and analytical signatures.

Building Signed Graphs
----------------------

From the empirical signature and the fitted model, you can construct a signed adjacency matrix using analytical p-values and optional FDR correction:

.. code-block:: python

    graph = T.build_graph(
        fdr_correction_flag=True,
        alpha=0.05
    )

- For ``model='naive'``, the graph is simply the sign of the empirical binary signature.
- For ``model='bSRGM'`` and ``model='bSCM'``, SIBERIA:
  - Computes analytical p-values for concordant motifs under the fitted model.
  - Optionally applies Benjamini–Hochberg FDR correction on the upper triangle.
  - Builds a signed adjacency matrix where only significant links are kept, with the sign indicating a concordant excess or deficit.

The validated projection is stored as:

.. code-block:: python

    T.graph   # N × N signed adjacency matrix with entries in {-1, 0, 1}

Community Detection
-------------------

SIBERIA provides community detection based on greedy minimization of:

- **BIC** of a signed stochastic block model (``method="bic"``), or
- **Frustration** of the signed network (``method="frustration"``).

.. code-block:: python

    communities = T.community_detection(
        trials=500,
        n_jobs=4,
        method="bic",       # or "frustration"
        show=False,
        random_state=42,
        starter="uniform"   # or "mixture"
    )

This:

- Runs multiple randomized greedy trials in parallel.
- Minimizes the chosen objective for each trial.
- Returns the best partition and stores it as:

.. code-block:: python

    T.communities   # length-N array of community labels (0, 1, 2, …)

Graph and Community Plots
-------------------------

Plotting the Projection Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plot the projected signed adjacency matrix:

.. code-block:: python

    T.plot_graph(
        export_path="results/adjacency",  # saves results/adjacency_adjacency.pdf
        show=True
    )

This displays a heatmap with discrete values in ``{-1, 0, 1}``.

Plotting Communities
^^^^^^^^^^^^^^^^^^^^

Visualize the adjacency matrix reordered by detected communities, with blocks highlighted:

.. code-block:: python

    T.plot_communities(
        export_path="results/communities",  # saves results/communities_communities.pdf
        show=True
    )

This produces:

- A reordered adjacency heatmap.
- Lines separating communities, based on ``T.communities``.

Block Matrix of Community Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also inspect a coarse-grained block matrix summarizing dominant link signs between communities:

.. code-block:: python

    M = T.plot_block_matrix(
        export_path="results/block_matrix",  # saves results/block_matrix_block_matrix.pdf
        show=True
    )

``M`` is a :math:`K \times K` matrix (where :math:`K` is the number of communities) with entries in ``{-1, 0, 1}``, indicating whether positive or negative links dominate each intra- and inter-community block.

Documentation
=============

You can find the complete documentation of the **SIBERIA** library at:  
`https://siberia.readthedocs.io/en/latest/ <https://siberia.readthedocs.io/en/latest/>`_

Guide
=====

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   siberia
   license
   contacts


