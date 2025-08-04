![PyPI](https://img.shields.io/badge/pypi-v0.1.0-blue) [![License:GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Python Version](https://img.shields.io/badge/Python-3.10.12-blue)
 [![Paper](https://img.shields.io/badge/arXiv-2508.00542-b31b1b.svg)](https://arxiv.org/abs/2508.00542)


# SIBERIA: SIgned BEnchmarks foR tIme series Analysis

SIBERIA provides **maximum-entropy null models and validation methods for signed networks derived from time series**.  
It enables the construction, filtering, and community detection of signed adjacency matrices based on validated co-fluctuations.  
The library implements advanced null models (`bSRGM`, `bSCM`) to rigorously distinguish meaningful patterns from noise, supporting reproducible and interpretable time-series network analysis.



SIBERIA includes methods to:

- Compute binary **signature matrices** for co-fluctuations.  
- Fit maximum-entropy **null models** (`bSRGM`, `bSCM`).  
- Predict event probabilities from model parameters.  
- Validate signatures using **analytical statistics** and **FDR correction**.  
- Build **signed graphs** and filter them.  
- Perform **community detection** minimizing BIC or frustration.  
- Visualize results via heatmaps and network adjacency plots.  

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
- [Initialization](#initialization)
- [Core Functionalities](#core-functionalities)
  - [Compute Signatures](#compute-signatures)
  - [Fit Null Models](#fit-null-models)
  - [Predict Event Probabilities](#predict-event-probabilities)
  - [Validate Signatures](#validate-signatures)
  - [Build and Filter Graphs](#build-and-filter-graphs)
  - [Community Detection](#community-detection)
  - [Plotting](#plotting)
- [Documentation](#documentation)
- [Credits](#credits)

---


## Installation

siberia can be installed via [pip](https://pypi.org/project/siberia/). Run:

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
- **fast-poibin** for Poisson-Binomial distributions
- **joblib** for parallel computation
- **statsmodels** for multiple testing corrections
- **matplotlib** and **seaborn** for visualization
- **tqdm** for progress bars
- **numba** for accelerating heavy computations

Install them via:

```bash
pip install numpy scipy pandas fast-poibin joblib statsmodels matplotlib tqdm numba
```

## How-to Guidelines

The main entry point of Siberia is the `TSeries` class, initialized with an `N × T` float matrix representing `N` time series of length `T`.

### Initialization

```python
from siberia import TSeries

# Tij is a 2D numpy array of shape (N, T) with float values
T = TSeries(data=Tij, n_jobs=4)
```

After initialization, you can already explore marginal statistics:

```python
T.ai_plus, T.ai_minus   # row-wise positive/negative counts
T.kt_plus, T.kt_minus   # column-wise positive/negative counts
T.a_plus, T.a_minus     # total positive/negative counts
```

### Computing the Signature

The signature captures concordant and discordant motifs in your time series:

```python
binary_signature = T.compute_signature()
```

The signature matrix is computed as:

- Concordant motifs = positive-positive + negative-negative
- Discordant motifs = positive-negative + negative-positive
- Binary signature = concordant − discordant

### Fitting the Models

You can list available models:

```python
T.implemented_models
```

which returns:

```python
['bSRGM', 'bSCM']
```

Choose and fit a model:

```python
T.fit(model="bSCM")
```

After fitting, the following attributes become available:

```python
T.ll                 # log-likelihood
T.jac                # Jacobian
T.norm               # infinite norm of the Jacobian
T.norm_rel_error     # relative error
T.aic                # Akaike Information Criterion
```

### Predicting Event Probabilities

Compute the expected probability of observing positive and negative events:

```python
pit_plus, pit_minus = T.predict()
```

These matrices provide event probabilities under the null model.

### Validating the Signature

Statistical validation with False Discovery Rate (FDR) correction:

```python
filtered_signature = T.validate_signature(fdr_correction_flag=True, alpha=0.05)
```

This filters out non-significant entries from the signature matrix.

### Building Signed Graphs

Convert signatures into signed adjacency matrices:

```python
naive_graph, filtered_graph = T.build_graph()
```

- **Naive Graph**: raw signature signs
- **Filtered Graph**: FDR-validated signature signs

### Plotting Signatures

Visualize empirical vs. filtered signature matrices:

```python
T.plot_signature(export_path="results/signature", show=True)
```

Visualize naive vs. filtered adjacency matrices:

```python
T.plot_graph(export_path="results/adjacency", show=True)
```

### Community Detection

SIBERIA includes community detection routines based on:

- **BIC minimization**
- **Frustration minimization**

Run detection:

```python
stats = T.community_detection(
    trials=500,
    method="bic",   # or "frustration"
    show=True
)
```

The output dictionary contains:

```python
stats['naive']        # stats for naive graph
stats['filtered']     # stats for filtered graph
stats['method']       # chosen minimization method
```

Best community assignments:

```python
T.naive_communities
T.filtered_communities
```

### Plotting Communities

The `plot_communities` function provides a visualization of adjacency matrices reordered by detected communities.  
It highlights community partitions by drawing boxes around groups of nodes assigned to the same community.  

#### Usage

```python
T.plot_communities(graph_type="filtered", export_path="results/communities", show=True)
```


## Documentation
You can find the complete documentation of the MaxEntSeries library in [documentation](https://maxentseries.readthedocs.io/en/latest/index.html)

## Credits

*Author*:

[Marzio Di Vece](https://www.sns.it/it/persona/marzio-di-vece) (a.k.a. [MarsMDK](https://github.com/MarsMDK))
[Emanuele Agrimi](https://www.imtlucca.it/it/people/emanuele-agrimi) (a.k.a. [Emaagr](https://github.com/Emaagr))
[Samuele Tatullo](https://www.imtlucca.it/it/people/samuele-tatullo)


*Acknowledgments*:
The module was developed under the supervision of [Tiziano Squartini](https://www.imtlucca.it/it/tiziano.squartini) and [Miguel Ibáñez Berganza](https://networks.imtlucca.it/people/miguel) and [Tommaso Gili](https://networks.imtlucca.it/people/tommaso).
It was developed at [IMT School for Advanced Studies Lucca](https://www.imtlucca.it/en) and [Scuola Normale Superiore of Pisa](https://www.sns.it/it).
This work is supported by the PNRR-M4C2-Investimento 1.3, Partenariato Esteso PE00000013-“FAIR-Future Artificial Intelligence Research”-Spoke 1 “Human-centered AI”, funded by the European Commission under the NextGeneration EU programme. MDV also acknowledges support by the European Community programme under the funding schemes: ERC-2018-ADG G.A. 834756 “XAI: Science and technology for the eXplanation of AI decision making”.
