![PyPI](https://img.shields.io/badge/pypi-v0.1.0-blue) [![License:GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Python Version](https://img.shields.io/badge/Python-3.10.12-blue)
[![Paper](https://img.shields.io/badge/arXiv-2508.00542-b31b1b.svg)](https://arxiv.org/abs/2508.00542)


# SIBERIA: SIgned BEnchmarks foR tIme series Analysis

SIBERIA provides **maximum-entropy null models and validation methods for signed networks derived from time series**.  
Starting from an `N × T` matrix of standardized time series, it builds signed co-fluctuation signatures, fits maximum-entropy models, and produces validated signed graphs that can be analyzed via community detection.

The library implements advanced null models (`bSRGM` and `bSCM`, plus the `naive` projection) to distinguish meaningful mesoscale structure from noise, supporting reproducible and interpretable time-series network analysis.

---

## Quick Start (Teaser)

Here is a minimal end-to-end example that generates synthetic data, fits a null model, validates signatures, builds a graph, and detects communities:

```python
import numpy as np
from siberia import TSeries

# 1. Generate synthetic standardized time series (N nodes, T time steps)
N, Tlen = 50, 500
rng = np.random.default_rng(42)
data = rng.normal(size=(N, Tlen)).astype(float)

# 2. Initialize TSeries
T = TSeries(data=data, n_jobs=4)

# 3. Compute signature
T.compute_signature()

# 4. Fit a null model (bSCM with fixed-point solver)
T.fit(
    model="bSCM",
    maxiter=1000,
    max_nfev=1000,
    tol=1e-8,
    eps=1e-8,
    solver_type="fixed_point"
)

# 5. Predict event probabilities under the null model
T.predict()

# 6. Check signature distributions (ensemble vs analytical, KS score)
ks_score = T.check_distribution_signature(
    n_ensemble=500,
    ks_score=True,
    alpha=0.05
)
print("KS score:", ks_score)

# 7. Build a validated signed graph with FDR correction
graph = T.build_graph(
    fdr_correction_flag=True,
    alpha=0.05
)

# 8. Detect communities via signed SBM BIC minimization
communities = T.community_detection(
    trials=200,
    n_jobs=4,
    method="bic",
    show=False,
    random_state=42,
    starter="mixture"
)
print("Detected communities:", np.unique(communities))

# 9. Plot adjacency and community structure
T.plot_graph(export_path="results/adjacency", show=True)
T.plot_communities(export_path="results/communities", show=True)
T.plot_block_matrix(export_path="results/block_matrix", show=True)
```

---

## Citation

If you use **SIBERIA** in your research, please cite the following paper:

```bibtex
@misc{divece2025assessingimbalancesignedbrain,
      title={Assessing (im)balance in signed brain networks}, 
      author={Marzio Di Vece and Emanuele Agrimi and Samuele Tatullo and Tommaso Gili and Miguel Ibáñez-Berganza and Tiziano Squartini},
      year={2025},
      eprint={2508.00542},
      archivePrefix={arXiv},
      primaryClass={physics.soc-ph},
      url={https://arxiv.org/abs/2508.00542}, 
}
```

## Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [How-to Guidelines](#how-to-guidelines)
  - [Initialization](#initialization)
  - [Compute Signatures](#compute-signatures)
  - [Fit Null Models](#fit-null-models)
  - [Predict Event Probabilities](#predict-event-probabilities)
  - [Check Signature Distributions](#check-signature-distributions)
  - [Build Graphs](#build-graphs)
  - [Community Detection](#community-detection)
  - [Graph and Community Plots](#graph-and-community-plots)
- [Documentation](#documentation)
- [Credits](#credits)

---

## Installation

SIBERIA can be installed via [pip](https://pypi.org/project/siberia/). Run:

```bash
pip install siberia
```

To upgrade to the latest version:

```bash
pip install siberia --upgrade
```

## Dependencies

SIBERIA requires the following libraries:

- **numpy** for numerical operations  
- **scipy** for optimization and statistical functions  
- **pandas** for structured data handling  
- **fast-poibin** for Poisson–Binomial distributions  
- **joblib** for parallel computation  
- **statsmodels** for multiple testing corrections (FDR)  
- **matplotlib** and **seaborn** for visualization  
- **tqdm** for progress bars  
- **numba** for accelerating heavy computations  

Install them via:

```bash
pip install numpy scipy pandas fast-poibin joblib statsmodels matplotlib seaborn tqdm numba
```

---

## How-to Guidelines

The main entry point of SIBERIA is the `TSeries` class, initialized with an `N × T` float matrix representing `N` time series of length `T`.  
If the rows are not standardized (mean ≈ 0, std ≈ 1), they are standardized internally.

### Initialization

```python
from siberia import TSeries
import numpy as np

# Tij is a 2D numpy array of shape (N, T) with float values
Tij = np.asarray(Tij, dtype=float)

T = TSeries(data=Tij, n_jobs=4)
```

After initialization, you can explore marginal statistics of the binarized series:

```python
T.ai_plus, T.ai_minus   # row-wise positive / negative counts
T.kt_plus, T.kt_minus   # column-wise positive / negative counts
T.a_plus,  T.a_minus    # total positive / negative counts
```

### Compute Signatures

The signature captures concordant and discordant co-fluctuation motifs:

```python
binary_signature = T.compute_signature()
```

Internally:

- Concordant motifs = positive–positive + negative–negative  
- Discordant motifs = positive–negative + negative–positive  
- Binary signature = concordant − discordant  

The result is stored in:

```python
T.binary_signature
```

### Fit Null Models

You can list the available models:

```python
T.implemented_models
# ['naive', 'bSRGM', 'bSCM']
```

Choose and fit a maximum-entropy model:

```python
T.fit(
    model="bSCM",          # 'bSRGM' or 'bSCM' or 'naive'
    maxiter=1000,
    max_nfev=1000,
    tol=1e-8,
    eps=1e-8,
    solver_type="fixed_point"  # 'fixed_point' or 'lsq' for bSCM
)
```

After fitting, the following attributes become available:

```python
T.params            # fitted parameters
T.ll                # log-likelihood
T.jac               # Jacobian of the constraints
T.norm              # infinite norm of the Jacobian
T.norm_rel_error    # relative norm of the constraint error
T.aic               # Akaike Information Criterion
```

### Predict Event Probabilities

Compute the expected probability of observing positive and negative events in each `i, t` entry:

```python
pit_plus, pit_minus = T.predict()
```

These matrices are stored in:

```python
T.pit_plus
T.pit_minus
```

and represent the null-model probabilities for positive and negative events for each time series and time step.

### Check Signature Distributions

You can compare the empirical signature with the null-model signature distributions using ensemble sampling and analytical calculations, summarized by a Kolmogorov–Smirnov score:

```python
ks_score = T.check_distribution_signature(
    n_ensemble=1000,
    ks_score=True,
    alpha=0.05
)
```

This method:

- Generates an ensemble of signatures from the fitted model via Monte Carlo sampling.  
- Computes analytical signature distributions (binomial / Poisson–Binomial) under the null model.  
- Performs KS tests pairwise and returns a **normalized KS score** in `[0, 1]`, stored as:

```python
T.ks_score
T.ensemble_signature        # N × N × n_ensemble
T.analytical_signature      # N × N × n_ensemble (or equivalent)
T.analytical_signature_dist # N × N × (T+1) PMFs for bSCM; analogous for bSRGM
```

A higher `ks_score` indicates better agreement between ensemble and analytical signatures at the chosen `alpha` threshold.

### Build Graphs

From the empirical signature and the fitted model, you can construct a signed adjacency matrix using analytical p-values and (optionally) FDR correction:

```python
graph = T.build_graph(
    fdr_correction_flag=True,
    alpha=0.05
)
```

- For `model='naive'`, the graph is simply the sign of the empirical binary signature.  
- For `model='bSRGM'` and `model='bSCM'`, SIBERIA:
  - Computes analytical p-values for concordant motifs under the fitted model.  
  - Optionally applies Benjamini–Hochberg FDR correction on the upper triangle (`fdr_correction_flag=True`).  
  - Builds a signed adjacency matrix where only significant links are kept, with sign indicating concordant excess or deficit.

The validated projection is stored as:

```python
T.graph  # N × N signed adjacency matrix with entries in {-1, 0, 1}
```

### Community Detection

SIBERIA provides community detection based on greedy minimization of:

- **BIC** of a signed stochastic block model (`method="bic"`)  
- **Frustration** of the signed network (`method="frustration"`)

```python
communities = T.community_detection(
    trials=500,
    n_jobs=4,
    method="bic",       # or "frustration"
    show=False,
    random_state=42,
    starter="uniform"   # or "mixture"
)
```

This:

- Runs multiple randomized greedy trials in parallel.  
- Minimizes the chosen objective for each trial.  
- Returns the best partition and stores it as:

```python
T.communities   # length-N array of community labels (0, 1, 2, …)
```

### Graph and Community Plots

#### Plotting the Projection Matrix

Plot the projected signed adjacency matrix:

```python
T.plot_graph(
    export_path="results/adjacency",  # saves results/adjacency_adjacency.pdf
    show=True
)
```

This displays a heatmap with discrete values in {-1, 0, 1}.

#### Plotting Communities

Visualize the adjacency matrix reordered by detected communities, with blocks highlighted:

```python
T.plot_communities(
    export_path="results/communities",  # saves results/communities_communities.pdf
    show=True
)
```

This produces:

- A reordered adjacency heatmap.  
- Lines separating communities, based on `T.communities`.

#### Block Matrix of Community Structure

You can also inspect a coarse-grained block matrix summarizing dominant link signs between communities:

```python
M = T.plot_block_matrix(
    export_path="results/block_matrix",  # saves results/block_matrix_block_matrix.pdf
    show=True
)
```

`M` is a `K × K` matrix (where `K` is the number of communities) with entries in {-1, 0, 1}, indicating whether positive or negative links dominate each intra- and inter-community block.

---

## Documentation

You can find the complete documentation of the **SIBERIA** library at:  
[https://siberia.readthedocs.io/en/latest/](https://siberia.readthedocs.io/en/latest/)

## Credits

*Authors*:

- [Marzio Di Vece](https://www.sns.it/it/persona/marzio-di-vece) (a.k.a. [MarsMDK](https://github.com/MarsMDK))  
- [Emanuele Agrimi](https://www.imtlucca.it/it/people/emanuele-agrimi) (a.k.a. [Emaagr](https://github.com/Emaagr))  
- [Samuele Tatullo](https://www.imtlucca.it/it/people/samuele-tatullo)

*Acknowledgments*:

The module was developed under the supervision of  
[Tiziano Squartini](https://www.imtlucca.it/it/tiziano.squartini),  
[Miguel Ibáñez Berganza](https://networks.imtlucca.it/people/miguel), and  
[Tommaso Gili](https://networks.imtlucca.it/people/tommaso).  

It was developed at the [IMT School for Advanced Studies Lucca](https://www.imtlucca.it/en) and [Scuola Normale Superiore of Pisa](https://www.sns.it/it).  

This work is supported by the PNRR-M4C2-Investimento 1.3, Partenariato Esteso PE00000013 “FAIR–Future Artificial Intelligence Research” – Spoke 1 “Human-centered AI”, funded by the European Commission under the NextGeneration EU programme. MDV also acknowledges support by the European Community programme under the funding scheme ERC-2018-ADG G.A. 834756 “XAI: Science and technology for the eXplanation of AI decision making”.