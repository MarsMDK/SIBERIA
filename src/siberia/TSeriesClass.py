import numpy as np
from scipy.optimize import least_squares as lsq
from scipy.stats import binom
import pandas as pd
from fast_poibin import PoiBin
from joblib import Parallel, delayed
from time import time
import logging
import numba
from numba import jit,prange
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import seaborn as sns


"""
TSeries class for analyzing time series data using various models.

The TSeries class provides methods for initializing with a time series matrix, computing signatures, fitting models, 
predicting probabilities, validating signatures, filtering signature matrices, building graphs, performing community detection, 
and plotting results.

Attributes:
    n_jobs (int): Number of threads for parallel computation.
    params (np.ndarray): Model parameters.
    args (tuple): Arguments for the model.
    ll (float): Log-likelihood value.of the error.
    jacobian (np.ndarray): Jacobian matrix. matrix.
    norm (float): Infinite norm of the Jacobian.
    aic (float): Akaike Information Criterion value.rix.
    args (tuple): Arguments for the model.
    norm_rel_error (float): Relative norm of the error.arameters for optimization.
    weighted_signature (np.ndarray): Weighted signature matrix.
    binary_signature (np.ndarray): Binary signature matrix.
    ensemble_signature (np.ndarray): Ensemble signature matrix.optimization.
    model (str): Chosen model.
    x0 (np.ndarray): Initial parameters for optimization. events.
    tol (float): Tolerance for optimization..
    eps (float): Epsilon for optimization.vents.
    maxiter (int): Maximum iterations for optimization..
    verbose (int): Verbosity level for optimization.
    pit_plus (np.ndarray): Probabilities of positive events.optimization.
    pit_minus (np.ndarray): Probabilities of negative events.
    n_ensemble (int): Number of ensemble samples.ix.
    bounds_lsq (tuple): Bounds for least squares optimization.
    N (int): Number of rows in the time series matrix.e series data.
    T (int): Number of columns in the time series matrix.ies data for positive weights.
    tseries (np.ndarray): Time series data.
    binary_tseries (np.ndarray): Binary time series data.
    binary_tseries_positive (np.ndarray): Binary time series data for positive weights.ts.
    binary_tseries_negative (np.ndarray): Binary time series data for negative weights.
    ai_plus (np.ndarray): Row-wise sum of binary positive weights.egative weights.
    kt_plus (np.ndarray): Column-wise sum of binary positive weights.ts.
    a_plus (float): Sum of binary positive weights.
    ai_minus (np.ndarray): Row-wise sum of binary negative weights.for positive weights.
    kt_minus (np.ndarray): Column-wise sum of binary negative weights.
    a_minus (float): Sum of binary negative weights.
    implemented_models (list): List of implemented models.ts.
    p_values_corrected (np.ndarray): Corrected p-values.
    filtered_signature (np.ndarray): Filtered signature matrix.ts.
    naive_graph (np.ndarray): Naive adjacency matrix.
    filtered_graph (np.ndarray): Filtered adjacency matrix..
    naive_communities (np.ndarray): Community assignments for the naive graph.ented models.
    filtered_communities (np.ndarray): Community assignments for the filtered graph. matrix.
    comm_stats (dict): Statistics for community detection.
matrix.
Methods: signature matrix.
    __init__(self, data=None, n_jobs=1):
        Initialize the TSeries instance with the time series matrix.
    compute_signature(self):hted triplet counts.
        Compute the binary and weighted signatures of time series data.
    fit(self, model, x0=None, maxiter=1000, max_nfev=1000, verbose=0, tol=1e-8, eps=1e-8, output_params_path=None, imported_params=None, solver_type='fixed_point'):it__(self, data=None, n_jobs=4):
        Fit the specified model to the data.with the weighted adjacency matrix.
    predict(self):
        Predict the probabilities of the occurrence of the events for the chosen model.ion matrices and the co-fluctuation matrices for the Time Series instance.
    validate_signature(self, fdr_correction_flag=True, alpha=0.05):s_path=None):
        Validate the signature of the model using analytical methods.elative norms.
    build_graph(self):
        Build naive and filtered graphs based on the filtered signature matrix.e probabilities of the occurrence of the events and their conditional weights for the chosen model.
    plot_signature(self, export_path='', show=True):
        Plot the empirical and filtered signature matrices as heatmaps.tion matrices with the chosen model.
    plot_graph(self, export_path='', show=True):
        Plot the naive and filtered adjacency matrices as heatmaps with discrete values.on matrix after the use of null model of choice.
    community_detection(self, trials=500, n_jobs=None, method="bic", show=False):
        Perform community detection on naive and filtered graphs using the chosen loss function.trices.
"""



#set number of threads for parallel computation in input
# Set number of threads for parallel computation in input

class TSeries:
    """TSeries class for analyzing time series data using graph-based methods.
    The TSeries class is designed to process a weighted adjacency matrix in 2D numpy array format. 
    It computes various graph-based statistics, including in-degrees, out-degrees, reciprocated degrees, 
    out-strengths, in-strengths, reciprocated strengths, and triadic statistics such as occurrences, 
    intensities, and fluxes.
        n_jobs (int): Number of parallel jobs to use for computations.
        params (numpy.ndarray): Fitted model parameters.
        ll (float): Log-likelihood of the fitted model.
        jacobian (numpy.ndarray): Jacobian of the fitted model.
        norm (float): Norm of the Jacobian.
        aic (float): Akaike Information Criterion of the fitted model.
        args (tuple): Arguments for the model.
        norm_rel_error (float): Relative error of the fitted model.
        weighted_signature (numpy.ndarray): Weighted signature matrix.
        binary_signature (numpy.ndarray): Binary signature matrix.
        ensemble_signature (numpy.ndarray): Ensemble signature matrix.
        model (str): Name of the model being used.
        x0 (numpy.ndarray): Initial guess for model parameters.
        tol (float): Tolerance for optimization.
        eps (float): Step size for numerical approximation.
        maxiter (int): Maximum number of iterations for optimization.
        verbose (int): Verbosity level for optimization.
        pit_plus (numpy.ndarray): Predicted probabilities for positive events.
        pit_minus (numpy.ndarray): Predicted probabilities for negative events.
        naive_graph (numpy.ndarray): Naive adjacency matrix.
        filtered_graph (numpy.ndarray): Filtered adjacency matrix.
        naive_communities (numpy.ndarray): Community assignments for the naive graph.
        filtered_communities (numpy.ndarray): Community assignments for the filtered graph.
        comm_stats (dict): Statistics for community detection.
    
    TSeries instance must be initialized with the weighted adjacency matrix in 2D numpy array format.
    On initialization, it computes in-degrees, out-degrees, reciprocated degrees, out-strengths, in-strengths, 
    reciprocated strengths, and triadic statistics such as occurrences, intensities, and fluxes.

    Methods:
        __init__(self, data=None, n_jobs=1):
            Initialize the TSeries instance with the time series matrix.
        compute_signature(self):
            Compute the binary and weighted signatures of time series data.
        fit(self, model, x0=None, maxiter=1000, max_nfev=1000, verbose=0, tol=1e-8, eps=1e-8, 
            output_params_path=None, imported_params=None, solver_type='fixed_point'):
            Fit the specified model to the data.
        predict(self):
            Predict the probabilities of the occurrence of the events for the chosen model.
        validate_signature(self, fdr_correction_flag=True, alpha=0.05):
            Validate the signature of the model using analytical methods.
        build_graph(self):
            Build naive and filtered graphs based on the filtered signature matrix.
        plot_signature(self, export_path='', show=True):
            Plot the empirical and filtered signature matrices as heatmaps.
        plot_graph(self, export_path='', show=True):
            Plot the naive and filtered adjacency matrices as heatmaps with discrete values.
        community_detection(self, trials=500, n_jobs=None, method="bic", show=False):
            Perform community detection on naive and filtered graphs using the chosen loss function.
    """
    
    
    def __init__(
        self,
        data = None, n_jobs=1
    ):
        #Initialization
        self.n_jobs = n_jobs
        self.params = None
        self.ll = None
        self.jacobian = None
        self.norm = None
        self.aic = None
        self.args = None
        self.norm_rel_error = None
        
        self.weighted_signature = None
        self.binary_signature = None
        self.ensemble_signature = None

        self.model = None
        self.args = None
        self.x0 = None
        self.tol = None
        self.eps = None
        self.maxiter = None
        self.verbose = None

        self.pit_plus = None
        self.pit_minus = None
        self.wit_plus = None
        self.wit_minus = None

        self.n_ensemble = None
        self.bounds_lsq = None

        #Check on data
        if data is None:
            raise ValueError('Time Series matrix is missing!')
        elif type(data) != np.ndarray:
            raise TypeError('Time Series matrix must be a numpy array!')


        if np.issubdtype(data.dtype, np.integer):
            raise ValueError('Time Series matrix must be a float matrix!')
        
        numba.set_num_threads(self.n_jobs)

        #Implemented models
        self.implemented_models = ['bSRGM','bSCM']
        


        # Inizialization of data and computation of marginals
        self.N = data.shape[0]
        self.T = data.shape[1]

        #Binary time series
        self.tseries = data
        self.binary_tseries = np.sign(data)

        

        #Marginals for positive weights
        self.binary_tseries_positive = np.where(self.binary_tseries > 0, self.binary_tseries, 0)
        self.binary_tseries_negative = np.abs(np.where(self.binary_tseries < 0, self.binary_tseries, 0))

        self.ai_plus = self.binary_tseries_positive.sum(axis=1).astype(float) # row-wise sum of binary positive weights
        self.kt_plus = self.binary_tseries_positive.sum(axis=0).astype(float) # column-wise sum of binary positive weights
        self.a_plus = self.binary_tseries_positive.sum().astype(float)
        self.ai_minus = self.binary_tseries_negative.sum(axis=1).astype(float) # row-wise sum of binary negative weights
        self.kt_minus = self.binary_tseries_negative.sum(axis=0).astype(float) # column-wise sum of binary negative weights
        self.a_minus = self.binary_tseries_negative.sum().astype(float)
        

    def compute_signature(self):
        """
        Computes the binary signatures of time series data.
        This method calculates the concordant and discordant motifs for binary time series data.
        It then computes the binary signature by subtracting the discordant motifs from the concordant motifs.
        The method performs the following steps:
        1. Computes pairwise motifs for binary time series data (positive-positive, positive-negative, negative-positive, negative-negative).
        2. Calculates the binary concordant motifs as the sum of positive-positive and negative-negative motifs.
        3. Calculates the binary discordant motifs as the sum of positive-negative and negative-positive motifs.
        4. Computes the binary signature as the difference between binary concordant and discordant motifs.
        Attributes:
            binary_concordant_motifs (int): Sum of concordant motifs for binary time series data.
            binary_discordant_motifs (int): Sum of discordant motifs for binary time series data.
            binary_signature (int): Difference between binary concordant and discordant motifs.
        """
        
        @staticmethod       
        @jit(nopython=True) 
        def pairwise_motif(data1, data2):
            """
            Compute the cofluctuation dynamic matrix for two time series datasets.
            This function calculates the cofluctuation dynamic matrix, which is an NxN matrix for each time interval. 
            The matrix element C_ij is defined as:
            - 1 if series 'i' and series 'j' fluctuate with the same sign,
            - -1 if they fluctuate with opposite signs,
            - 0 otherwise.
            Parameters:
            data1 (numpy.ndarray): A 2D array of shape (N, T) representing the first time series dataset.
            data2 (numpy.ndarray): A 2D array of shape (N, T) representing the second time series dataset.
            Returns:
            numpy.ndarray: An NxN matrix representing the cofluctuation dynamic matrix.
            """
            # Use matrix multiplication for efficient computation
            motif = np.dot(data1, data2.T)
            return motif
        
        motif_plus_plus = pairwise_motif(self.binary_tseries_positive,self.binary_tseries_positive)
        motif_plus_minus = pairwise_motif(self.binary_tseries_positive,self.binary_tseries_negative)
        motif_minus_plus = pairwise_motif(self.binary_tseries_negative,self.binary_tseries_positive)
        motif_minus_minus = pairwise_motif(self.binary_tseries_negative,self.binary_tseries_negative)

        self.binary_concordant_motifs = motif_plus_plus + motif_minus_minus
        self.binary_discordant_motifs = motif_plus_minus + motif_minus_plus
        self.binary_signature = self.binary_concordant_motifs - self.binary_discordant_motifs

        return self.binary_signature
        
    def fit(
        self,
        model,
        x0 = None,
        maxiter = 1000,
        max_nfev = 1000,
        verbose= 0,
        tol = 1e-8,
        eps = 1e-8,
        output_params_path = None, imported_params = None, solver_type = 'fixed_point'
        
        ):
        
        """
        Fit the specified model to the data.
        Parameters:
        -----------
        model : str
            The model to be fitted. Must be one of the implemented models: 'bSRGM', 'bSCM'.
        x0 : array-like, optional
            Initial guess for the parameters. If None, a random initialization will be used.
        maxiter : int, optional
            Maximum number of iterations for the optimization algorithm. Default is 1000.
        max_nfev : int, optional
            Maximum number of function evaluations for the optimization algorithm. Default is 1000.
        verbose : int, optional
            Verbosity level of the optimization algorithm. Default is 0.
        tol : float, optional
            Tolerance for termination by the optimization algorithm. Default is 1e-8.
        eps : float, optional
            Step size used for numerical approximation of the Jacobian. Default is 1e-8.
        output_params_path : str, optional
            Path to save the fitted parameters. If None, the parameters will not be saved.
        Raises:
        -------
        ValueError
            If the model is not initialized or not implemented.
        TypeError
            If output_params_path is not a string.
        Returns:
        --------
        None
        """
        
        ### Initialization
        
        self.x0 = x0
        self.tol = tol
        self.eps = eps
        self.maxiter = maxiter
        self.verbose = verbose
        self.model = model
        self.solver_type = solver_type

        

        ### Input Validation
        if model is None:
            raise ValueError('model must be initialized')
        elif model not in self.implemented_models:
            raise ValueError('model is not implemented! Inspect "self.implemented_models"')

        ### Inizialization of arguments for each model and of x0 if not initialized in input.
        if self.model == 'bSRGM': # Binary Bipartite-Signed Random Graph Model (for Time Series)
            self.args = (self.a_plus,self.a_minus, (self.N,self.T))
            self.x0 = np.random.random(2)
        elif self.model == 'bSCM': # Binary Bipartite-Signed Configuration Model (for Time Series)
            self.args = (self.ai_plus,self.kt_plus,self.ai_minus,self.kt_minus)
            self.x0 = np.random.random(2*self.N + 2*self.T)
        else:
            raise ValueError('Model not implemented!')
        
        if self.model == 'bSRGM':

            @staticmethod
            @jit(nopython=True)
            def loglikelihood_bsr_model(params,a_plus,a_minus,shape):
                """Log-likelihood function for the Binary Bipartite-Signed Random Graph Model (for Time Series)"""
                N = shape[0]
                T = shape[1]

                alpha = params[0]
                gamma = params[1]

                ll = - alpha * a_plus - gamma * a_minus - N*T*np.log(np.exp(-alpha) + np.exp(-gamma))

                return - ll
            
            @staticmethod
            @jit(nopython=True)
            def jacobian_bsr_model(params,a_plus,a_minus,shape):
                """Jacobian function for the Binary Bipartite-Signed Random Graph Model (for Time Series)"""
                N = shape[0]
                T = shape[1]

                alpha = params[0]
                gamma = params[1]
                au_alpha = np.exp(-alpha)
                au_gamma = np.exp(-gamma)
                jac = np.empty(len(params))
                
                a_plus_th = N*T*au_alpha/(au_alpha+au_gamma)
                a_minus_th = N*T*au_gamma/(au_alpha+au_gamma)
                
                jac[0] = -a_plus + a_plus_th
                jac[1] = -a_minus + a_minus_th
                
                return - jac
            
            @staticmethod
            @jit(nopython=True)
            def relative_error_bsr_model(params,a_plus,a_minus,shape,tol=1e-10):
                """Relative error function for the Binary Bipartite-Signed Random Graph Model (for Time Series)"""
                N = shape[0]
                T = shape[1]

                alpha = params[0]
                gamma = params[1]
                au_alpha = np.exp(-alpha)
                au_gamma = np.exp(-gamma)
                jac = np.empty(len(params))
                
                a_plus_th = N*T*au_alpha/(au_alpha+au_gamma)
                a_minus_th = N*T*au_gamma/(au_alpha+au_gamma)
                
                jac[0] = (-a_plus + a_plus_th)/(a_plus+tol)
                jac[1] = (-a_minus + a_minus_th)/(a_minus+tol)
                
                return - jac
            
            if imported_params is None:
                self.params = lsq(relative_error_bsr_model,x0=self.x0,args=self.args,
                                    gtol=self.tol,xtol=self.eps,max_nfev=max_nfev,verbose=self.verbose,tr_solver='lsmr').x
            else:
                self.params = imported_params
            self.ll = - loglikelihood_bsr_model(self.params,*self.args)
            self.jac = - jacobian_bsr_model(self.params,*self.args)
            self.norm = np.linalg.norm(self.jac,ord=np.inf)
            rel_error = - relative_error_bsr_model(self.params,*self.args)
            self.norm_rel_error = np.linalg.norm(rel_error,ord=np.inf)
            self.aic = 2*len(self.params) - 2*self.ll


        elif self.model == 'bSCM':

            @staticmethod
            @jit(nopython=True)
            def loglikelihood_bscm_model(params,ai_plus,kt_plus,ai_minus,kt_minus):
                """Log-likelihood function for the Binary Bipartite Signed Configuration Model (for Time Series)"""
                N = len(ai_plus)
                T = len(kt_plus)

                alphai = params[:N]
                betat = params[N:N+T]
                gammai = params[N+T:N+T+N]
                deltat = params[N+N+T:]

                H =  np.sum(alphai*ai_plus+gammai*ai_minus) + np.sum(betat*kt_plus + deltat*kt_minus)
                
                lnZ = 0

                for i in range(N):
                    for t in range(T):
                        aut1 = np.exp(-alphai[i]-betat[t])
                        aut2 = np.exp(-gammai[i]-deltat[t])

                        lnZ += np.log(aut1+aut2)

                ll = - H - lnZ
                
                return - ll
            
            @staticmethod
            @jit(nopython=True, parallel=True)
            def jacobian_bscm_model(params,ai_plus,kt_plus,ai_minus,kt_minus):
                """Jacobian function for the Binary Bipartite-Signed Configuration Model (for Time Series)"""
                N = len(ai_plus)
                T = len(kt_plus)

                alphai = params[:N]
                betat = params[N:N+T]
                gammai = params[N+T:N+T+N]
                deltat = params[N+N+T:]

                jac = np.empty(len(params))
                jac[:N] = -ai_plus
                jac[N:N+T] = -kt_plus
                jac[N+T:N+N+T] = -ai_minus
                jac[N+N+T:] = -kt_minus
                


                for i in prange(N):
                    for t in range(T):
                        aut1 = np.exp(-alphai[i]-betat[t])
                        aut2 = np.exp(-gammai[i]-deltat[t])

                        jac[i] += aut1 / ((aut1 + aut2))
                        jac[N+T+i] += aut2 / (aut1 + aut2)

                for t in prange(T):
                    for i in range(N):
                        aut1 = np.exp(-alphai[i]-betat[t])
                        aut2 = np.exp(-gammai[i]-deltat[t])

                        jac[N+t] += aut1 / ((aut1 + aut2))
                        jac[N+N+T+t] += aut2 / (aut1 + aut2)

                return - jac
            
            #@staticmethod
            @jit(nopython=True, parallel=True)
            def relative_error_bscm_model(params,ai_plus,kt_plus,ai_minus,kt_minus,tol=1e-10):
                """Relative error function for the Binary Bipartite-Signed Configuration Model (for Time Series)"""
                N = len(ai_plus)
                T = len(kt_plus)

                alphai = params[:N]
                betat = params[N:N+T]
                gammai = params[N+T:N+T+N]
                deltat = params[N+N+T:]

                jac = np.empty(len(params))
                jac[:N] = -ai_plus
                jac[N:N+T] = -kt_plus
                jac[N+T:N+N+T] = -ai_minus
                jac[N+N+T:] = -kt_minus

                for i in prange(N):
                    for t in range(T):
                        aut1 = np.exp(-alphai[i]-betat[t])
                        aut2 = np.exp(-gammai[i]-deltat[t])

                        jac[i] += aut1 / ((aut1 + aut2))
                        jac[N+T+i] += aut2 / (aut1 + aut2)

                for t in prange(T):
                    for i in range(N):
                        aut1 = np.exp(-alphai[i]-betat[t])
                        aut2 = np.exp(-gammai[i]-deltat[t])

                        jac[N+t] += aut1 / ((aut1 + aut2))
                        jac[N+N+T+t] += aut2 / (aut1 + aut2)
                
                jac[:N] /= (ai_plus+tol)
                jac[N:N+T] /= (kt_plus+tol)
                jac[N+T:N+N+T] /= (ai_minus+tol)
                jac[N+N+T:] /= (kt_minus+tol)
                
                return - jac

            

            @staticmethod
            @jit(nopython=True)
            def fixed_point_solver_bscm_model(ai_plus, kt_plus, ai_minus, kt_minus, max_iterations=10000, diff=1e-08, tol=1e-06, print_steps=100):
                """Jacobian function for the Binary Bipartite-Signed Configuration Model (for Time Series)"""
                N = len(ai_plus)
                T = len(kt_plus)

                xi = ai_plus / np.sqrt(np.sum(ai_plus))
                zt = kt_plus / np.sqrt(np.sum(kt_plus))
                yi = ai_minus / np.sqrt(np.sum(ai_minus))
                vt = kt_minus / np.sqrt(np.sum(kt_minus))

                xi_new = xi.copy()
                zt_new = zt.copy()
                yi_new = yi.copy()
                vt_new = vt.copy()

                for it in range(max_iterations):
                    xi = xi_new.copy()
                    yi = yi_new.copy()
                    zt = zt_new.copy()
                    vt = vt_new.copy()

                    for i in range(N):
                        den_xi_new = np.sum(zt / (1. + xi[i] * zt + yi[i] * vt))
                        den_yi_new = np.sum(vt / (1. + xi[i] * zt + yi[i] * vt))

                        xi_new[i] = ai_plus[i] / den_xi_new
                        yi_new[i] = ai_minus[i] / den_yi_new

                    for t in range(T):
                        den_zt_new = np.sum(xi / (1. + xi * zt[t] + yi * vt[t]))
                        den_vt_new = np.sum(yi / (1. + xi * zt[t] + yi * vt[t]))

                        zt_new[t] = kt_plus[t] / den_zt_new
                        vt_new[t] = kt_minus[t] / den_vt_new

                    
                    normies = np.empty(4)
                    normies[0] = np.linalg.norm(xi - xi_new)
                    normies[1] = np.linalg.norm(yi - yi_new)
                    normies[2] = np.linalg.norm(zt - zt_new)
                    normies[3] = np.linalg.norm(vt - vt_new)
                    
                    
                    if it % print_steps == 0:
                        alphai = -np.log(xi_new)
                        betat = -np.log(zt_new)
                        gammai = -np.log(yi_new)
                        deltat = -np.log(vt_new)
                        whole_params = np.concatenate((alphai, betat, gammai, deltat))
                        
                        rel_error = np.linalg.norm(relative_error_bscm_model(whole_params, ai_plus,kt_plus,ai_minus,kt_minus),ord=np.inf)
                        if rel_error < tol:
                            # print(f"Convergence reached at Iteration {it} for gtol.")
                            break

                        
                    if np.max(normies) < diff:
                        # print(f"Convergence reached at Iteration {it} for xtol.")

                        break

                alphai = -np.log(xi_new)
                betat = -np.log(zt_new)
                gammai = -np.log(yi_new)
                deltat = -np.log(vt_new)

                whole_params = np.concatenate((alphai, betat, gammai, deltat))

                return whole_params

                        



            if imported_params is None:
                if self.solver_type == 'lsq':
                    self.params = lsq(relative_error_bscm_model,x0=self.x0,args=self.args,
                                    gtol=self.tol,xtol=self.eps,max_nfev=max_nfev,verbose=self.verbose,tr_solver='lsmr').x
                elif self.solver_type == 'fixed_point':
                    self.params = fixed_point_solver_bscm_model(*self.args)
                else:
                    raise TypeError('wrong solver type!')
            else:
                self.params = imported_params

            self.ll = - loglikelihood_bscm_model(self.params,*self.args)
            self.jac = - jacobian_bscm_model(self.params,*self.args)
            self.norm = np.linalg.norm(self.jac,ord=np.inf)
            rel_error = - relative_error_bscm_model(self.params,*self.args)
            self.norm_rel_error = np.linalg.norm(rel_error,ord=np.inf)
            self.aic = 2*len(self.params) - 2*self.ll

        
        
            
        if output_params_path is not None:
            if isinstance(output_params_path,str):
                output_path = output_params_path
                params = pd.DataFrame(self.params)
                params.to_csv(output_path)
                
            else:
                raise TypeError('output_params_path must be a string')
            
    def predict(self):
        """
        Predict the probabilities of events based on the specified model.
        This method computes the probabilities of the occurrence of events for the implemented models:
        - binary Signed Random Graph Model (bSRGM)
        - binary Signed Configuration Model (bSCM)
        Returns:
            tuple: For "bSRGM" and "bSCM", returns the computed probabilities:
                - (pit_plus, pit_minus)
        """

        

        if self.model == "bSRGM":
            @staticmethod
            def bsr_model_proba_events(params, shape):
                """Compute the probabilities of the occurrence of the events for the Binary Bipartite-Signed Random Graph Model (for Time Series)"""
                alpha = np.exp(-params[0])
                gamma = np.exp(-params[1])

                N = shape[0]
                T = shape[1]

                pit_plus = np.ones((N,T))*alpha/(alpha+gamma)
                pit_minus = np.ones((N,T))*gamma/(alpha+gamma)

                return pit_plus,pit_minus
            self.pit_plus,self.pit_minus = bsr_model_proba_events(self.params,(self.N,self.T))

            return self.pit_plus,self.pit_minus
            
        
        elif self.model == "bSCM":
            @staticmethod
            def bscm_model_proba_events(params,shape):
                """Compute the probabilities of the occurrence of the events for the Binary Bipartite-Signed Configuration Model (for Time Series)"""
                N = shape[0]
                T = shape[1]
                
                exp_betai = np.exp(-params[:N])
                exp_deltat = np.exp(-params[N:N+T])
                exp_gammai = np.exp(-params[N+T:N+N+T])
                exp_etat = np.exp(-params[N+N+T:])
                pit_plus = np.empty((N,T))
                pit_minus = np.empty((N,T))

                for i in range(N):
                    for t in range(T):
                        aut1 = exp_betai[i]*exp_deltat[t]
                        aut2 = exp_gammai[i]*exp_etat[t]
                        
                        pit_plus[i,t] = aut1/(aut1+aut2)
                        pit_minus[i,t] = aut2/(aut1+aut2)

                return pit_plus,pit_minus
            self.pit_plus,self.pit_minus = bscm_model_proba_events(self.params,(self.N,self.T))

            return self.pit_plus,self.pit_minus

            

    def validate_signature(self, fdr_correction_flag = True, alpha = 0.05):
        """
        This function validates the signature of the model by computing p-values and applying 
        False Discovery Rate (FDR) correction. Depending on the model type, it uses analytical 
        methods for validation. The function supports two model types: 'bSRGM' and 'bSCM'.
        --------
        numpy.ndarray
            A filtered signature matrix where elements are retained based on the significance level.
        - For the 'bSRGM' model, p-values are computed using a binomial cumulative distribution function.
        - For the 'bSCM' model, p-values are computed using the Poisson Binomial distribution.
        - The FDR correction is applied to the upper triangular part of the p-values matrix, and the 
          corrected matrix is made symmetric.
        - The filtered signature matrix is computed by retaining elements of the empirical signature 
          matrix where the corrected p-values are below the significance level.
        
        Validate the signature of the model using analytical methods.
        Parameters:
        -----------
        fdr_correction_flag : bool, optional
            Flag to indicate whether to apply False Discovery Rate (FDR) correction. Default is True.
        alpha : float, optional
            Significance level for statistical tests. Default is 0.05.
        Raises:
        -------
        ValueError
            If the predicted probabilities and conditional weights are not computed before validation.
            If the model specified is not valid.
        Notes:
        ------
        This function validates the signature of the model by computing p-values and applying FDR correction.
        Depending on the model type, it uses analytical methods for validation:
        - It computes p-values using specific analytical models for different types of models.
        """
    
        if self.pit_plus is None:
            raise ValueError("Predict probabilities and conditional weights first!")
        
        @staticmethod
        def fdr_correction(p_values, alpha=0.05):
            
            
            """
            Apply False Discovery Rate (FDR) correction to the upper triangular part of a matrix of p-values,
            and ensure the corrected matrix is symmetric.
            
            Parameters:
            p_values (numpy.ndarray): A square numpy matrix of p-values to be corrected.
            alpha (float, optional): Significance level for the FDR correction. Default is 0.05.
            
            Returns:
            numpy.ndarray: A symmetric numpy array of p-values after FDR correction, with the same shape as the input array.
            """
            # Get the upper triangular indices (excluding the diagonal)
            triu_indices = np.triu_indices(p_values.shape[0], k=1)
            
            # Flatten the upper triangular part of the p-values matrix
            p_values_upper = p_values[triu_indices]
            
            # Apply the FDR correction using multipletests on the upper triangular part
            _, p_values_corrected, _, _ = multipletests(p_values_upper, alpha=alpha, method='fdr_bh')
            
            # Rebuild the corrected matrix
            corrected_p_values = p_values.copy()
            corrected_p_values[triu_indices] = p_values_corrected
            
            # Make the matrix symmetric by copying the upper triangular part to the lower triangular part
            corrected_p_values.T[triu_indices] = p_values_corrected
            
            return corrected_p_values
        self.alpha = alpha
        self.fdr_correction_flag = fdr_correction_flag
        
        if self.model == 'bSRGM':
            def p_values_analytical_bsr_model(pit_plus,pit_minus,concordant_motifs):
                """
                Compute the p-values for the Binary Bipartite-Signed Random Graph Model (for Time Series).
                This function calculates the p-values for a given binary bipartite-signed random graph model 
                using analytical methods. It evaluates the statistical significance of concordant motifs 
                between pairs of nodes in a time series dataset.
                Args:
                    pit_plus (numpy.ndarray): A 2D array of shape (N, T) representing the positive PIT (Probability Integral Transform) values 
                                              for N nodes over T time steps.
                    pit_minus (numpy.ndarray): A 2D array of shape (N, T) representing the negative PIT values 
                                               for N nodes over T time steps.
                    concordant_motifs (numpy.ndarray): A 2D array of shape (N, N) representing the number of concordant motifs 
                                                       between pairs of nodes.
                Returns:
                    numpy.ndarray: A 2D array of shape (N, N) containing the computed p-values for each pair of nodes. 
                                   The diagonal elements are set to 1.0 as self-comparisons are not meaningful.
                Notes:
                    - The p-values are computed using the cumulative distribution function (CDF) of the binomial distribution.
                    - The function assumes that the input arrays `pit_plus` and `pit_minus` are properly normalized.
                    - The p-values are two-tailed, calculated as `2 * min(CDF, 1 - CDF)`.
                """
                N = pit_plus.shape[0]
                T = pit_plus.shape[1]

                q_plus = (pit_plus**2 + pit_minus**2)[0]
                
                p_values = np.empty((N,N))
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            cdfx = binom.cdf(concordant_motifs[i,j],T,q_plus[0])
                            p_values[i,j] = 2.*min(cdfx,1.-cdfx)
                            
                        else:
                            p_values[i,j] = 1.0
                return p_values
            

            model_p_values = p_values_analytical_bsr_model(self.pit_plus,self.pit_minus,self.binary_concordant_motifs)
            self.p_values_corrected = fdr_correction(model_p_values,alpha=self.alpha)
            

        elif self.model == 'bSCM':
            
            @staticmethod
            def p_values_analytical_bscm_model(pit_plus,pit_minus,concordant_motifs):
                def p_values_analytical_bscm_model(pit_plus, pit_minus, concordant_motifs):
                    """
                    Compute the p-values for a given analytical bSCM (binary Signed Configuration Model) model.
                    This function calculates the p-values for concordant motifs between pairs of nodes
                    based on the provided PIT (Probability Integral Transform) matrices and concordant motifs matrix.
                    Args:
                        pit_plus (numpy.ndarray): A 2D array of shape (N, T) representing the PIT values for positive motifs.
                                                  N is the number of nodes, and T is the number of time steps.
                        pit_minus (numpy.ndarray): A 2D array of shape (N, T) representing the PIT values for negative motifs.
                                                   N is the number of nodes, and T is the number of time steps.
                        concordant_motifs (numpy.ndarray): A 2D array of shape (N, N) representing the concordant motifs
                                                           between pairs of nodes.
                    Returns:
                        numpy.ndarray: A 2D array of shape (N, N) containing the computed p-values for each pair of nodes.
                                       The diagonal elements are set to 1.0 as self-comparisons are not meaningful.
                    """
                
                N = pit_plus.shape[0]
                T = pit_plus.shape[1]

                
                p_values = np.empty((N,N))
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            probabilities = pit_plus[i,:] * pit_plus[j,:] + pit_minus[i,:] * pit_minus[j,:]
                            pb = PoiBin(list(probabilities))
                            cdfx = pb.cdf[int(concordant_motifs[i,j])]
                            p_values[i,j] = 2.*min(cdfx,1.-cdfx)
                            
                        else:
                            p_values[i,j] = 1.0
                return p_values

            
            model_p_values = p_values_analytical_bscm_model(self.pit_plus,self.pit_minus,self.binary_concordant_motifs)
            self.p_values_corrected = fdr_correction(model_p_values,alpha=self.alpha)
            
        else:
            raise ValueError('The model is not valid!')


        @staticmethod
        def filter_statistic(emp_stat,p_values, alpha):
            """
            Filters the empirical statistics based on p-values and a significance level.
            Parameters:
            emp_stat (array-like): The array of empirical statistics to be filtered.
            p_values (array-like): The array of p-values corresponding to the empirical statistics.
            alpha (float): The significance level threshold. Values in `p_values` less than `alpha` 
                       will retain the corresponding value in `emp_stat`, otherwise replaced with 0.
            Returns:
            numpy.ndarray: An array where values from `emp_stat` are retained if the corresponding 
                       `p_values` are less than `alpha`, otherwise replaced with 0.
            """
            
            filtered_stat = np.where(p_values < alpha, emp_stat, 0)
            return filtered_stat
    
        self.filtered_signature = filter_statistic(self.binary_signature, self.p_values_corrected, self.alpha)
        
        return self.filtered_signature


    def build_graph(self):
        """
        Constructs and returns two graph representations based on the signature matrices.
        This method generates two graph representations:
        1. A naive graph based on the binary signature matrix.
        2. A filtered graph based on the filtered signature matrix.
        Returns:
            tuple: A tuple containing:
                - naive_graph (numpy.ndarray): The graph representation derived from the binary signature matrix.
                - filtered_graph (numpy.ndarray): The graph representation derived from the filtered signature matrix.
        Raises:
            ValueError: If the filtered signature matrix is not available (i.e., `self.filtered_signature` is None).
        """

        if self.filtered_signature is None:
            raise ValueError("Filter the signature matrix first!")
        
        self.naive_graph = np.sign(self.binary_signature)
        self.filtered_graph = np.sign(self.filtered_signature)
        return self.naive_graph,self.filtered_graph



    def plot_signature(self,export_path='',show=True):
        """
        Plots the binary and filtered signature matrices as heatmaps.
        This method visualizes the binary signature matrix and the filtered 
        signature matrix side by side using heatmaps. It also provides options 
        to export the plot to a file and display it.
        Parameters:
        -----------
        export_path : str, optional
            The file path (excluding extension) where the plot will be saved 
            as a PDF. If empty, the plot will not be saved. Default is ''.
        show : bool, optional
            If True, the plot will be displayed. Default is True.
        Raises:
        -------
        ValueError
            If `binary_signature` or `filtered_signature` is None, indicating 
            that the signature has not been computed.
        Notes:
        ------
        - The heatmaps use the 'coolwarm_r' colormap.
        - The exported file will have '_signature.pdf' appended to the 
          provided `export_path`.
        Example:
        --------
        To save the plot to a file and display it:
            obj.plot_signature(export_path='/path/to/save/plot', show=True)
        To only save the plot without displaying it:
            obj.plot_signature(export_path='/path/to/save/plot', show=False)
        """
        
        
        


        if self.binary_signature is None or self.filtered_signature is None:
            raise ValueError("Compute the signature first!")

        def plot_heatmap(matrix, title, ax):
            cax = ax.matshow(matrix, cmap='coolwarm_r')
            plt.title(title)

            plt.colorbar(cax, ax=ax)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        plot_heatmap(self.binary_signature, 'Empirical Signature Matrix', ax1)
        plot_heatmap(self.filtered_signature, 'Filtered Signature Matrix', ax2)
        
        plt.tight_layout()
        if export_path != '':
            export_path_corr_matrix = export_path + '_signature.pdf'
            plt.savefig(export_path_corr_matrix,dpi=300)
        if show == True:
            plt.show()
        plt.close()

    def plot_graph(self, export_path='', show=True):
        """
        Plots the naive and filtered adjacency matrices as heatmaps.
        Parameters:
        -----------
        export_path : str, optional
            The file path (excluding extension) where the plot will be saved as a PDF.
            If not provided, the plot will not be saved. Default is an empty string.
        show : bool, optional
            If True, displays the plot. Default is True.
        Raises:
        -------
        ValueError
            If `self.filtered_graph` is None, indicating that the graph has not been built.
        Notes:
        ------
        - The naive adjacency matrix is plotted on the left, and the filtered adjacency 
          matrix is plotted on the right.
        - The heatmaps use a discrete colormap with three colors: red (-1), white (0), 
          and blue (1).
        - If `export_path` is provided, the plot is saved as a PDF with the suffix 
          "_adjacency.pdf".
        """
        
        if self.filtered_graph is None:
            raise ValueError("Build the graph first!")

        # Define a discrete colormap
        colors = ["red", "white", "blue"]  # Colors for -1, 0, and 1
        cmap = ListedColormap(colors)
        bounds = [-1.5, -0.5, 0.5, 1.5]  # Boundaries for discrete values
        norm = BoundaryNorm(bounds, cmap.N)

        def plot_heatmap(matrix, title, ax):
            # Plot heatmap using the discrete colormap
            cax = ax.matshow(matrix, cmap=cmap, norm=norm)
            ax.set_title(title)
            plt.colorbar(cax, ax=ax, ticks=[-1, 0, 1])  # Add colorbar with discrete ticks

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        plot_heatmap(self.naive_graph, 'Naive Adjacency Matrix', ax1)
        plot_heatmap(self.filtered_graph, 'Filtered Adjacency Matrix', ax2)

        plt.tight_layout()
        if export_path:
            export_path_corr_matrix = f"{export_path}_adjacency.pdf"
            plt.savefig(export_path_corr_matrix, dpi=600)
        if show:
            plt.show()
        plt.close()


    def community_detection(self,trials: int = 500,n_jobs: int = None,method: str = "bic",show: bool = False):
        """
        Perform community detection on the naive and filtered graphs using a specified method.
        This method applies a greedy minimization routine to detect communities in the graphs
        associated with the object. It supports two optimization criteria: Bayesian Information
        Criterion (BIC) and network frustration.
        -----------
        trials : int, optional
            The number of trials to run for the community detection algorithm. Default is 500.
        n_jobs : int, optional
            The number of parallel jobs to use for running trials. If None, the value of `self.n_jobs` is used.
        method : str, optional
            The optimization criterion to use for community detection. Must be either "bic" (Bayesian Information Criterion)
            or "frustration". Default is "bic".
        show : bool, optional
            If True, logs additional information about the progress of the trials. Default is False.
        Raises:
        -------
        ValueError
            If `self.naive_graph` or `self.filtered_graph` is None, indicating that `.build_graph()` must be called first.
        ValueError
            If the `method` parameter is not one of "bic" or "frustration".
        --------
        dict
            A dictionary containing the community detection results for both the naive and filtered graphs:
            - 'naive': A dictionary with the best community assignment, minimum loss, number of infinite losses, and runtime.
            - 'filtered': A dictionary with the same structure as 'naive', but for the filtered graph.
            - 'method': The optimization method used ("bic" or "frustration").
        Notes:
        ------
        - The method uses a greedy algorithm to iteratively reassign nodes to communities to minimize the specified criterion.
        - The results are stored in `self.naive_communities`, `self.filtered_communities`, and `self.comm_stats` for later use.
        """
        

        if self.naive_graph is None or self.filtered_graph is None:
            raise ValueError("You must call .build_graph() before running community detection.")
        
        if method not in ["bic", "frustration"]:
            raise ValueError('method must be either "bic" or "frustration".')

        if n_jobs is None:
            n_jobs = self.n_jobs

        @staticmethod
        @jit(nopython=True)
        def compute_edges_probabilities(adj, C, sign):
            """
            Compute the probabilities of edges between communities in a graph.
            
            Parameters:
            adj (numpy.ndarray): Adjacency matrix of the graph.
            C (list or array): List where the index represents the node and the value represents the community.
            sign (int): The sign of the edges to consider (e.g., 1 for positive edges, -1 for negative edges).
            
            Returns:
            tuple: (probability_matrix, total_links, count_matrix) where:
                probability_matrix: Matrix of edge probabilities between communities.
                total_links: Matrix of total possible links between communities.
                count_matrix: Matrix of actual edge counts between communities.
            """
            # Convert community assignment C to list of unique communities and its index vector.
            unique_communities = list(set(C))
            C_index = np.empty(len(C), dtype=np.int64)
            for i in range(len(C)):
                # Find index of C[i] in unique_communities
                for j in range(len(unique_communities)):
                    if C[i] == unique_communities[j]:
                        C_index[i] = j
                        break

            N = adj.shape[0]
            k = len(unique_communities)
            
            # Compute the counts for each community.
            com_counts = np.zeros(k, dtype=np.int64)
            for i in range(len(C_index)):
                com_counts[C_index[i]] += 1
            
            # Precompute the total possible links between communities.
            total_links = np.zeros((k, k), dtype=np.int64)
            for i in range(k):
                for j in range(k):
                    if i == j:
                        total_links[i, j] = com_counts[i] * (com_counts[i] - 1) // 2
                    else:
                        total_links[i, j] = com_counts[i] * com_counts[j]
            
            # Count the actual number of edges with the given sign.
            count_matrix = np.zeros((k, k), dtype=np.int64)
            for u in range(N):
                for v in range(u + 1, N):
                    if adj[u, v] == sign:
                        c1, c2 = C_index[u], C_index[v]
                        count_matrix[c1, c2] += 1
                        if c1 != c2:
                            count_matrix[c2, c1] += 1
                            
            # Compute the probability of edges between communities.
            probability_matrix = np.zeros((k, k), dtype=np.float64)
            for i in range(k):
                for j in range(k):
                    if total_links[i, j] > 0:
                        probability_matrix[i, j] = count_matrix[i, j] / total_links[i, j]
                    else:
                        probability_matrix[i, j] = 0.0
            
            return probability_matrix, total_links, count_matrix

        @staticmethod
        def UpdateBIC(adj, C):
            """
            Calculate the Bayesian Information Criterion (BIC) for a given adjacency matrix and community assignment.
            
            Parameters:
            adj (numpy.ndarray): The adjacency matrix representing the graph.
            C (list or array): Community assignment; index represents the node and value is the community.
            
            Returns:
            float: The BIC value.
            """
            # Create community index twice (keeping structure similar to the original code)
            unique_communities = list(set(C))
            C_index = np.array([unique_communities.index(c) for c in C])
            unique_communities = list(set(C))
            C_index = np.array([unique_communities.index(c) for c in C])
            
            # Compute community counts.
            com_counts = np.zeros(len(unique_communities), dtype=np.int64)
            for c in C_index:
                com_counts[c] += 1

            N = adj.shape[0]
            k = len(unique_communities)
            
            # Compute edge probabilities for negative and positive edges.
            edge_minus = compute_edges_probabilities(adj, C_index, -1)
            edge_plus = compute_edges_probabilities(adj, C_index, 1)

            P_minus = edge_minus[0]
            P_plus = edge_plus[0]
            L_minus = edge_minus[2]
            L_plus = edge_plus[2]
            
            n = com_counts

            def compute_log_likelihood(k, n, L_minus, L_plus, P_minus, P_plus):
                epsilon = 1e-10  # Small value to avoid divide by zero
                log_likelihood = 0.0
                for c in range(k):
                    nc = n[c]
                    if nc > 1:
                        Lmc = L_minus[c][c]
                        Lpc = L_plus[c][c]
                        Pmc = max(P_minus[c][c], epsilon)
                        Ppc = max(P_plus[c][c], epsilon)
                        one_minus = max(1 - Pmc - Ppc, epsilon)

                        log_likelihood += (Lmc * np.log(Pmc) + Lpc * np.log(Ppc) +
                                           ((nc * (nc - 1)) / 2 - Lmc - Lpc) * np.log(one_minus))

                        for d in range(c + 1, k):
                            Lmd = L_minus[c][d]
                            Lpd = L_plus[c][d]
                            Pmd = max(P_minus[c][d], epsilon)
                            Ppd = max(P_plus[c][d], epsilon)
                            one_minus_d = max(1 - Pmd - Ppd, epsilon)

                            log_likelihood += (Lmd * np.log(Pmd) + Lpd * np.log(Ppd) +
                                               (nc * n[d] - Lmd - Lpd) * np.log(one_minus_d))
                return log_likelihood

            log_likelihood = compute_log_likelihood(k, n, L_minus, L_plus, P_minus, P_plus)
            BIC = k * (k + 1) * np.log(N * (N - 1) / 2) - 2 * log_likelihood
            
            return BIC

        # ---------------------------- Frustration-related functions ----------------------------
        @staticmethod
        @jit(nopython=True)
        def UpdateF(adj, C):
            """
            Calculate the frustration F (with weights) for the given community partition.
            
            For each pair of nodes:
            - If an edge is negative (adj < 0) and the nodes are in the same community, add the absolute weight.
            - If an edge is positive (adj > 0) and the nodes are in different communities, add the edge weight.
            
            Returns:
            float: The frustration value, divided by 2 to avoid double counting.
            """
            F = 0.0
            N = len(C)
            for i in range(N):
                for j in range(N):
                    if adj[i, j] < 0 and C[i] == C[j]:
                        F += abs(adj[i, j])
                    elif adj[i, j] > 0 and C[i] != C[j]:
                        F += adj[i, j]
            return F / 2.0  # To avoid double counting

        # ------------------------ Greedy Minimization Routine ------------------------

        def greedy_min(K_up, C, adjmat, method="bic"):
            """
            Greedy algorithm that iteratively reassigns nodes to communities in order to 
            minimize a given criterion. The criterion is chosen by the parameter 'method'.
            
            Parameters:
            K_up (int): Initial maximum number of clusters.
            C (numpy.ndarray): Initial community assignment (1D array).
            adjmat (numpy.ndarray): Adjacency matrix of the graph.
            method (str): "bic" to minimize Bayesian Information Criterion, or "frustration" 
                            to minimize network frustration.
            
            Returns:
            tuple: (C, final_value)
                - C: The updated community assignment after optimization.
                - final_value: The final BIC value (if method=="bic") or the final frustration value (if method=="frustration").
            """
            # Choose the appropriate update function.
            if method.lower() == "bic":
                update = UpdateBIC
            elif method.lower() == "frustration":
                update = UpdateF
            else:
                raise ValueError("Unknown method. Use 'bic' or 'frustration'.")

            K = K_up
            stop = 0
            iteration = 0  # Iteration counter

            while stop != 1:
                iteration += 1
                V = np.arange(len(C))  # Node indices {0, 1, ..., N-1}
                stop = 1

                while len(V) > 0:
                    # Select a random node from V and remove it.
                    i = np.random.choice(V)
                    V = np.delete(V, np.where(V == i))
                    g = C[i]
                    current_val = update(adjmat, C)
                    delta = {}

                    # Consider all unique community labels.
                    unique_C = np.unique(C)
                    for cl in unique_C:
                        if cl != g:
                            C[i] = cl
                            candidate_val = update(adjmat, C)
                            delta[cl] = current_val - candidate_val  # Positive if candidate_val is lower.
                            C[i] = g  # restore original assignment

                    # If any candidate improves the criterion, choose the best one.
                    if any(val > 0 for val in delta.values()):
                        stop = 0
                        # Select the candidate with maximum improvement.
                        best_candidate = max(delta, key=delta.get)
                        C[i] = best_candidate

                        # If the original community becomes empty, reassign labels.
                        if np.sum(C == g) == 0:
                            for j in range(len(C)):
                                if C[j] > g:
                                    C[j] -= 1
                            K -= 1

            final_value = update(adjmat, C)
            return C, final_value

        def run_single_trial(adj_matrix: np.ndarray):
            comm, loss = greedy_min(
                len(adj_matrix),
                np.arange(len(adj_matrix)),
                adj_matrix,
                method=method
            )
            return comm, loss

        def run_trials(adj_matrix: np.ndarray, graph_type: str):
            if show:
                logging.info(f"Running {trials} {method.upper()} trials on {graph_type} graph...")
            start_time = time()

            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_trial)(adj_matrix) for _ in tqdm(range(trials), desc=f"Running trials for {graph_type} graph"))
            

            losses = [r[1] for r in results]
            min_loss = np.min(losses)
            best_idx = np.argmin(losses)
            best_comm = results[best_idx][0]
            num_infs = sum(np.isinf(l) for l in losses)
            runtime = time() - start_time

            return {
                'best_comm': best_comm,
                'min_loss': min_loss,
                'num_infs': num_infs,
                'runtime': runtime
            }

        # Run for both graphs
        naive_stats = run_trials(self.naive_graph, "naive")
        filtered_stats = run_trials(self.filtered_graph, "filtered")

        # Save results
        self.naive_communities = naive_stats['best_comm']
        self.filtered_communities = filtered_stats['best_comm']
        self.comm_stats = {'naive': naive_stats, 'filtered': filtered_stats, 'method': method}

        return self.comm_stats
    

    def _reorder_graph(self, graph, labels):
            """Reorder adjacency matrix by community labels."""
            sorted_idx = np.argsort(labels)
            return graph[sorted_idx][:, sorted_idx], labels[sorted_idx]

    def _draw_community_blocks(self, ax, labels, color='black', linewidth=1.5):
        """Draw lines to separate communities."""
        boundaries = np.cumsum(np.unique(labels, return_counts=True)[1])[:-1]
        for b in boundaries:
            ax.axhline(b - 0.5, color=color, linewidth=linewidth)
            ax.axvline(b - 0.5, color=color, linewidth=linewidth)

    def plot_communities(self, graph_type="filtered", export_path="", show=True):
        """
        Plot reordered adjacency matrix by community labels with boxes.

        Parameters:
        -----------
        graph_type : str, optional
            Either "naive" or "filtered" (default="filtered").
        export_path : str, optional
            Path to save the PDF figure. If empty, the plot is not saved.
        show : bool, optional
            If True, display the figure.
        """

        if graph_type not in ["naive", "filtered"]:
            raise ValueError("graph_type must be either 'naive' or 'filtered'")

        if graph_type == "naive":
            graph = self.naive_graph
            labels = self.naive_communities
        else:
            graph = self.filtered_graph
            labels = self.filtered_communities

        if graph is None or labels is None:
            raise ValueError(f"{graph_type}_graph or {graph_type}_communities not available. Run build_graph() and community_detection().")

        reordered_graph, reordered_labels = self._reorder_graph(graph, labels)

        cmap = ListedColormap(["red", "white", "blue"])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(reordered_graph, cmap=cmap, norm=norm, 
                    cbar_kws={'ticks': [-1, 0, 1]}, ax=ax, square=True)
        self._draw_community_blocks(ax, reordered_labels, color='black', linewidth=1.5)

        ax.set_title(f"{graph_type.capitalize()} Graph Reordered by Communities", fontsize=14, weight='bold')
        ax.set_xlabel("ROI (Reordered)", fontsize=12)
        ax.set_ylabel("ROI (Reordered)", fontsize=12)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()
        if export_path:
            plt.savefig(f"{export_path}_{graph_type}_communities.pdf", dpi=600)
        if show:
            plt.show()
        plt.close()
