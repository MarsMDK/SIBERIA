SIBERIA: SIgned BEnchmarks foR tIme series Analysis
===================================================

SIBERIA is a Python3 package for rigorous **signed time-series network analysis**.  
It provides **maximum-entropy null models** and validation methods to identify 
statistically significant patterns of co-fluctuations in time series data.

SIBERIA provides solvers for the binary models **bSRGM** and **bSCM**.  
These models enable researchers to distinguish meaningful signals from noise, 
build signed adjacency matrices, and explore the structure of time-series-derived networks.

Main features include:

- Computing binary signature matrices for co-fluctuations.
- Fitting maximum-entropy null models (bSRGM, bSCM).
- Predicting event probabilities from model parameters.
- Validating signatures with analytical statistics and FDR correction.
- Building signed graphs and filtering them.
- Performing community detection via BIC or frustration minimization.
- Visualizing results with heatmaps and adjacency plots.

For more information about Maximum Entropy methods, visit 
`Maximum Entropy Hub <https://meh.imtlucca.it>`_.

Citation
====================================

If you use SIBERIA in your research, please cite the following paper:

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
====================================

SIBERIA can be installed via pip:

.. code-block:: bash

    pip install Siberia

If you already installed the package and want to upgrade it:

.. code-block:: bash

    pip install Siberia --upgrade

Dependencies
====================================

SIBERIA uses the following dependencies:

- **numpy** for numerical operations
- **scipy** for optimization and statistical functions
- **pandas** for structured data handling
- **fast-poibin** for Poisson-Binomial distributions
- **joblib** for parallel computation
- **statsmodels** for multiple testing corrections
- **matplotlib** for visualization
- **tqdm** for progress bars
- **numba** (optional, recommended) for acceleration
- **seaborn** for advanced plotting

They can be easily installed via pip:

.. code-block:: bash

    pip install numpy scipy pandas fast-poibin joblib statsmodels matplotlib tqdm numba seaborn

How-to Guidelines
====================================

The module contains the class **TSeries**, initiated with a 2D time series matrix (N × T).

Class Instance and Empirical Statistics
---------------------------------------

To initialize a TSeries instance:

.. code-block:: python

    from siberia import TSeries
    T = TSeries(data=Tij, n_jobs=4)

Where ``Tij`` is a float matrix of shape (N, T).  
After initialization you can explore core statistics:

.. code-block:: python

    T.ai_plus, T.ai_minus
    T.kt_plus, T.kt_minus
    T.a_plus, T.a_minus

Computing the Signature
-----------------------

.. code-block:: python

    binary_signature = T.compute_signature()

The signature captures concordant and discordant motifs:

- Concordant = positive-positive + negative-negative
- Discordant = positive-negative + negative-positive
- Signature = concordant − discordant

Fitting the Models
------------------

List available models:

.. code-block:: python

    T.implemented_models   # ['bSRGM', 'bSCM']

Fit a model:

.. code-block:: python

    T.fit(model="bSCM")

After fitting, you can inspect:

.. code-block:: python

    T.ll, T.jacobian, T.norm, T.norm_rel_error, T.aic

Predicting Event Probabilities
------------------------------

.. code-block:: python

    pit_plus, pit_minus = T.predict()

Validating the Signature
------------------------

.. code-block:: python

    filtered_signature = T.validate_signature(fdr_correction_flag=True, alpha=0.05)

Building Signed Graphs
----------------------

.. code-block:: python

    naive_graph, filtered_graph = T.build_graph()

Plotting Signatures and Graphs
------------------------------

.. code-block:: python

    T.plot_signature(export_path="results/signature", show=True)
    T.plot_graph(export_path="results/adjacency", show=True)

Community Detection
-------------------

.. code-block:: python

    stats = T.community_detection(
        trials=500,
        method="bic",   # or "frustration"
        show=True
    )

You can access:

.. code-block:: python

    T.naive_communities
    T.filtered_communities

Documentation
====================================

You can find the complete documentation in 
`ReadTheDocs <https://siberia.readthedocs.io/en/latest/`_.

Guide
^^^^^

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   Siberia
   license
   contacts
