import numpy as np
from scipy.optimize import least_squares as lsq
from scipy.stats import binom
import pandas as pd
from fast_poibin import PoiBin
from joblib import Parallel, delayed
import numba
from numba import jit,prange
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from scipy.stats import ks_2samp


#Set number of threads for parallel computation in input

class TSeries:
    """TSeries class for analyzing time series data using graph-based methods.
    The TSeries class is designed to process a weighted adjacency matrix in 2D numpy array format. 
    It computes various graph-based statistics, including in-degrees, out-degrees, reciprocated degrees, 
    out-strengths, in-strengths, reciprocated strengths, and triadic statistics such as occurrences, 
    intensities, and fluxes.
        n_jobs (int): Number of parallel jobs to use for computations.
        params (numpy.ndarray): Fitted model parameters.
        ll (float): Log-likelihood of the fitted model.
        jac (numpy.ndarray): Jacobian of the fitted model.
        norm (float): Norm of the Jacobian.
        aic (float): Akaike Information Criterion of the fitted model.
        args (tuple): Arguments for the model.
        norm_rel_error (float): Relative error of the fitted model.
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
        check_distribution_signature(self, n_ensemble=1000, ks_score=True, alpha=0.05):
            Validate the distribution of the signature using ensemble simulations and, optionally, a KS score.
        build_graph(self):
            Build naive and filtered graphs based on the filtered signature matrix.
        plot_graph(self, export_path='', show=True):
            Plot the naive/filtered adjacency matrix as a heatmap with discrete values.
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
        self.jac = None
        self.norm = None
        self.aic = None
        self.args = None
        self.norm_rel_error = None
        
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

        self.n_ensemble = None

        #Check on data
        if data is None:
            raise ValueError('Time Series matrix is missing!')
        elif type(data) != np.ndarray:
            raise TypeError('Time Series matrix must be a numpy array!')


        if np.issubdtype(data.dtype, np.integer):
            raise ValueError('Time Series matrix must be a float matrix!')
        
        numba.set_num_threads(self.n_jobs)

        #Implemented models
        self.implemented_models = ['naive','bSRGM','bSCM']
        
        # Check if data is standardized on rows (mean ~0, std ~1), if not, standardize
        row_means = np.mean(data, axis=1)
        row_stds = np.std(data, axis=1)
        if not (np.allclose(row_means, 0, atol=1e-6) and np.allclose(row_stds, 1, atol=1e-6)):
            data = (data - row_means[:, None]) / (row_stds[:, None] + 1e-12)
        # print("Data standardized on rows.")

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
        
        #@staticmethod       
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
        elif model == 'naive':
            pass
        ### Inizialization of arguments for each model and of x0 if not initialized in input.
        if self.model == 'bSRGM': # Binary Bipartite-Signed Random Graph Model (for Time Series)
            self.args = (self.a_plus,self.a_minus, (self.N,self.T))
            self.x0 = np.random.random(2)
        elif self.model == 'bSCM': # Binary Bipartite-Signed Configuration Model (for Time Series)
            self.args = (self.ai_plus,self.kt_plus,self.ai_minus,self.kt_minus)
            self.x0 = np.random.random(2*self.N + 2*self.T)
        
        if self.model == 'bSRGM':

            #@staticmethod
            @jit(nopython=True)
            def loglikelihood_bsr_model(params,a_plus,a_minus,shape):
                """Log-likelihood function for the Binary Bipartite-Signed Random Graph Model (for Time Series)"""
                N = shape[0]
                T = shape[1]

                alpha = params[0]
                gamma = params[1]

                ll = - alpha * a_plus - gamma * a_minus - N*T*np.log(np.exp(-alpha) + np.exp(-gamma))

                return - ll
            
            #@staticmethod
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
            
            #@staticmethod
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

            #@staticmethod
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
            
            #@staticmethod
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

            

            #@staticmethod
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
            if verbose > 0:
                print('Fitting Completed with infinite norm:',self.norm, 'and infinite relative norm:',self.norm_rel_error)


        
        
            
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
            #@staticmethod
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
            #@staticmethod
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
        elif self.model == "naive":pass
            

    def check_distribution_signature(self, n_ensemble = 1000, ks_score=True, alpha = 0.05):
        """
        Validate the signature of the model using either ensemble or analytical methods.
        Parameters:
        -----------
        n_ensemble : int, optional
            Number of ensemble realizations used to build the empirical signature distribution. Default is 1000.
        ks_score : bool, optional
            If True, compute the Kolmogorov–Smirnov agreement score between empirical and analytical
            signature distributions. Default is True.
        alpha : float, optional
            Significance level used in the KS test when computing the KS score. Default is 0.05.
    
            Flag to indicate whether to use analytical methods for validation. Default is True.
        Raises:
        -------
        ValueError
            If the predicted probabilities and conditional weights are not computed before validation.
            If the model specified is not valid.
        Notes:
        ------
        This function validates the signature of the model by computing p-values and applying FDR correction.
        Depending on the model type and the analytical flag, it uses different methods for validation:
        - For ensemble-based validation, it computes ensemble signatures and elaborates statistics.
        - For analytical validation, it computes p-values using specific analytical models for different types of models.
        """
    
        if self.pit_plus is None:
            raise ValueError("Predict probabilities and conditional weights first!")
        
        self.n_ensemble = n_ensemble

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
        
        @jit(nopython=True)
        def sample_single_realization_binary(pit_plus,pit_minus):
            """
            Generates a single realization of binary data based on the given probabilities.
            Parameters:
            pit_plus (numpy.ndarray): A 2D array of shape (N, T) containing the probabilities of the positive outcome (1) for each element.
            pit_minus (numpy.ndarray): A 2D array of shape (N, T) containing the probabilities of the negative outcome (-1) for each element.
            Returns:
            numpy.ndarray: A 2D array of shape (N, T) containing the generated binary data, where each element is either 1 or -1 based on the given probabilities.
            """

            N = pit_plus.shape[0]
            T = pit_plus.shape[1]

            realization_data = np.zeros((N,T))
            
            for i in range(N):
                for t in range(T):
                    p_plus = pit_plus[i,t]
                    
                    ran = np.random.rand()
                    if ran < p_plus:
                        realization_data[i,t] = 1
                    else:
                        realization_data[i,t] = -1
            return realization_data

        @jit(nopython=True,parallel=True)
        def ensemble_signature_computation_binary(pit_plus,pit_minus,n_ensemble):
            """Compute the ensemble statistics for the co-fluctuation matrices and correlation matrices."""

            N = pit_plus.shape[0]
            T = pit_plus.shape[1]
            
            ensemble_signature = np.empty((n_ensemble,N,N))

            
            for n_ens in prange(n_ensemble):
                realization_data = sample_single_realization_binary(pit_plus,pit_minus)
                positive_realization = np.where(realization_data > 0, realization_data, 0)
                negative_realization = np.abs(np.where(realization_data < 0, realization_data, 0))
                motif_plus_plus = pairwise_motif(positive_realization,positive_realization)
                motif_plus_minus = pairwise_motif(positive_realization,negative_realization)
                motif_minus_plus = pairwise_motif(negative_realization,positive_realization)
                motif_minus_minus = pairwise_motif(negative_realization,negative_realization)
                ensemble_signature[n_ens,:,:] = motif_plus_plus + motif_minus_minus - motif_plus_minus - motif_minus_plus
                
                
            return ensemble_signature        
                                
        if self.model == 'bSRGM':
            #@staticmethod       
            @jit(nopython=True,parallel=True)
            def sample_analytical_bsr_model(pit_plus,pit_minus,n_ensemble):
                """Compute the p-values for the Binary Bipartite-Signed Random Graph Model (for Time Series)"""
                N = pit_plus.shape[0]
                T = pit_plus.shape[1]

                q_plus = (pit_plus**2 + pit_minus**2)[0]
                
                signature_ens = np.empty((N,N,n_ensemble))
                for i in prange(N):
                    for j in range(N):
                        if j != i:
                            c_ens = np.random.binomial(T,q_plus[0],n_ensemble)
                            signature_ens[i,j,:] = 2*c_ens - T
                            
                return signature_ens
            
            #@staticmethod       
            def compute_c_ens(i, j, T, ensemble_signature, q_plus):
                """Helper function to compute c_ens for a specific (i, j)."""
                ensemble_cij = (T + ensemble_signature[i, j, :]) / 2
                return binom.pmf(ensemble_cij.astype(int), T, q_plus[0])

            #@staticmethod       
            def sample_analytical_bsr_model_2(pit_plus, pit_minus, ensemble_signature, n_jobs):
                """Compute the p-values for the Binary Bipartite-Signed Random Graph Model (for Time Series)"""
                N = pit_plus.shape[0]
                T = pit_plus.shape[1]

                q_plus = (pit_plus**2 + pit_minus**2)[0]
                signature_ens = ensemble_signature.copy()

                # Parallel computation for (i, j) pairs where i != j
                results = Parallel(n_jobs=n_jobs)(
                    delayed(compute_c_ens)(i, j, T, ensemble_signature, q_plus)
                    for i in range(N) for j in range(N) if i != j
                )

                # Update signature_ens with the computed results
                index = 0
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            signature_ens[i, j, :] = results[index]
                            index += 1

                return signature_ens
            
            ensemble_signature = ensemble_signature_computation_binary(self.pit_plus,self.pit_minus,self.n_ensemble).transpose(1,2,0)
            analytical_signature = sample_analytical_bsr_model(self.pit_plus,self.pit_minus,self.n_ensemble)
            analytical_signature_dist = sample_analytical_bsr_model_2(self.pit_plus,self.pit_minus,ensemble_signature, self.n_jobs)
            
            
        elif self.model == 'bSCM':

            
            #@staticmethod
            def compute_pair_signature(i, j, pit_plus, pit_minus, n_ensemble, T):
                """
                Compute the ensemble signature for a single node pair (i, j).

                Parameters:
                    i (int): Row index.
                    j (int): Column index.
                    pit_plus (numpy.ndarray): N x T matrix of probabilities for positive interactions.
                    pit_minus (numpy.ndarray): N x T matrix of probabilities for negative interactions.
                    n_ensemble (int): Number of ensemble samples to generate.
                    T (int): Number of trials (time steps).

                Returns:
                    tuple: (i, j, ensemble_signature), where ensemble_signature is an array of size n_ensemble.
                """
                if i == j:
                    # Self-loop case: Set signature to T
                    return i, j, np.full(n_ensemble, T)

                # Calculate Poisson Binomial probabilities
                probabilities = pit_plus[i, :] * pit_plus[j, :] + pit_minus[i, :] * pit_minus[j, :]

                # Initialize the Poisson Binomial distribution
                pb = PoiBin(probabilities)

                # Precompute the CDF for all possible outcomes
                cdf_values = pb.cdf

                # Generate uniform random numbers
                uniform_samples = np.random.uniform(0, 1, n_ensemble)

                # Map uniform samples to Poisson Binomial outcomes
                poibin_samples = np.searchsorted(cdf_values, uniform_samples)

                # Compute ensemble signature
                ensemble_signature = 2 * poibin_samples - T

                return i, j, ensemble_signature


            #@staticmethod
            def sample_analytical_bscm_model_poibin(pit_plus, pit_minus, n_ensemble, n_jobs=-1):
                """
                Compute ensemble signatures for the Binary Bipartite-Signed Random Graph Model with parallelization.

                Parameters:
                    pit_plus (numpy.ndarray): N x T matrix of probabilities for positive interactions.
                    pit_minus (numpy.ndarray): N x T matrix of probabilities for negative interactions.
                    n_ensemble (int): Number of ensemble samples to generate.
                    n_jobs (int): Number of parallel jobs (-1 to use all available cores).

                Returns:
                    numpy.ndarray: N x N x n_ensemble array of ensemble signatures.
                """
                N, T = pit_plus.shape
                ensemble_signature = np.empty((N, N, n_ensemble))

                # Outer loop over i with progress tracking
                for i in range(N):
                    # Parallel processing of inner loop over j
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(compute_pair_signature)(i, j, pit_plus, pit_minus, n_ensemble, T)
                        for j in range(N)
                    )
                    for _, j, signature in results:
                        ensemble_signature[i, j, :] = signature

                return ensemble_signature
            
            #@staticmethod
            def compute_pair_iteration(i, j, pit_plus, pit_minus, T):
                """
                Compute the ensemble PMF for a single node pair (i, j).

                Parameters:
                    i (int): Row index.
                    j (int): Column index.
                    pit_plus (numpy.ndarray): N x T matrix of probabilities for positive interactions.
                    pit_minus (numpy.ndarray): N x T matrix of probabilities for negative interactions.
                    T (int): Number of trials (time steps).

                Returns:
                    tuple: (i, j, pmf_values), where pmf_values is the PMF array.
                """
                if i == j:
                    # Self-loop case: Set PMF to T
                    return i, j, np.full(T + 1, T)

                # Calculate Poisson Binomial probabilities
                probabilities = pit_plus[i, :] * pit_plus[j, :] + pit_minus[i, :] * pit_minus[j, :]

                # Initialize the Poisson Binomial distribution
                pb = PoiBin(probabilities)

                # Compute the PMF
                pmf_values = pb.pmf

                return i, j, pmf_values


            #@staticmethod
            def sample_analytical_bscm_model_poibin_dist(pit_plus, pit_minus, ensemble_signature, n_jobs=-1):
                """
                Compute ensemble PMFs for the Binary Bipartite-Signed Random Graph Model with parallelization.

                Parameters:
                    pit_plus (numpy.ndarray): N x T matrix of probabilities for positive interactions.
                    pit_minus (numpy.ndarray): N x T matrix of probabilities for negative interactions.
                    ensemble_signature (numpy.ndarray): Precomputed ensemble signature array.
                    n_jobs (int): Number of parallel jobs (-1 to use all available cores).

                Returns:
                    numpy.ndarray: N x N x (T+1) array of PMFs.
                """
                N, T = pit_plus.shape
                dist_signature = np.empty((N, N, T + 1), dtype=float)

                # Outer loop over i with progress tracking
                for i in range(N):
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(compute_pair_iteration)(i, j, pit_plus, pit_minus, T)
                        for j in range(N)
                    )
                    for _, j, pmf_values in results:
                        dist_signature[i, j, :] = pmf_values

                return dist_signature



            
            ensemble_signature = ensemble_signature_computation_binary(self.pit_plus,self.pit_minus,self.n_ensemble).transpose(1,2,0)
            analytical_signature = np.array(sample_analytical_bscm_model_poibin(self.pit_plus,self.pit_minus,self.n_ensemble))
            analytical_signature_dist = np.array(sample_analytical_bscm_model_poibin_dist(self.pit_plus,self.pit_minus,ensemble_signature,self.n_jobs))
        
        self.ensemble_signature = ensemble_signature
        self.analytical_signature = analytical_signature
        self.analytical_signature_dist = analytical_signature_dist

        if ks_score==True:
            ### Statistical KS_scores
            #@staticmethod
            def compute_ks_score(ensemble_signature, analytical_signature, alpha):
                """
                Compute the Kolmogorov-Smirnov (KS) scores between ensemble and analytical signatures.
                Parameters:
                    ensemble_signature (numpy.ndarray): N x N x n_ensemble array of ensemble signatures.
                    analytical_signature (numpy.ndarray): N x N x n_analytical array of analytical signatures.
                    alpha (float): threshold for kS test
                Returns:
                    Returns (float): fraction of node pairs (i, j) for which the KS test p-value ≥ alpha
                """
                N = ensemble_signature.shape[0]
                ks_score = 0
                num_tot = 0
                for i in range(N):
                    for j in range(i,N):
                        if i != j:
                            num_tot += 1
                            _, p_KS = ks_2samp(ensemble_signature[i, j, :], analytical_signature[i, j, :], alternative='two-sided', mode='auto')
                            if p_KS >= alpha:
                                ks_score += 1
                
                ks_score_normalized = ks_score / num_tot
                return ks_score_normalized

            self.ks_score = compute_ks_score(self.ensemble_signature, self.analytical_signature, alpha)

            return self.ks_score

    def build_graph(self, fdr_correction_flag = True, alpha = 0.05):
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
        if self.model == 'naive':
            self.graph = np.sign(self.binary_signature)
            return self.graph
        
        if self.pit_plus is None:
            raise ValueError("Predict probabilities and conditional weights first!")
        
        #@staticmethod
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
                cdfx_condition = np.zeros((N,N))
                p_values = np.empty((N,N))
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            cdfx = binom.cdf(concordant_motifs[i,j],T,q_plus[0])                            
                            p_values[i,j] = 2.*min(cdfx,1.-cdfx)
                            cdfx_condition[i,j] = 1 if cdfx > 0.5 else -1

                        else:
                            p_values[i,j] = 1.0
                return p_values, cdfx_condition
            
            model_p_values, cdfx_condition = p_values_analytical_bsr_model(self.pit_plus,self.pit_minus,self.binary_concordant_motifs)
            if self.fdr_correction_flag:                                                   #the flag was not used otherwise
                self.p_values_corrected = fdr_correction(model_p_values, alpha=self.alpha)
            else:
                self.p_values_corrected = model_p_values
            self.cdfx_condition = cdfx_condition

        elif self.model == 'bSCM':
            
            #@staticmethod
            def p_values_analytical_bscm_model(pit_plus,pit_minus,concordant_motifs):
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

                cdfx_condition = np.zeros((N,N))
                p_values = np.empty((N,N))
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            probabilities = pit_plus[i,:] * pit_plus[j,:] + pit_minus[i,:] * pit_minus[j,:]
                            pb = PoiBin(list(probabilities))
                            cdfx = pb.cdf[int(concordant_motifs[i,j])]
                            if cdfx > 0.5:
                                cdfx_condition[i,j] = 1
                            else:
                                cdfx_condition[i,j] = -1
                            p_values[i,j] = 2.*min(cdfx,1.-cdfx)
                            
                        else:
                            p_values[i,j] = 1.0
                return p_values, cdfx_condition

            
            model_p_values, cdfx_condition = p_values_analytical_bscm_model(self.pit_plus,self.pit_minus,self.binary_concordant_motifs)
            if self.fdr_correction_flag:
                self.p_values_corrected = fdr_correction(model_p_values, alpha=self.alpha)
            else:
                self.p_values_corrected = model_p_values
            self.cdfx_condition = cdfx_condition
        else:
            raise ValueError('The model is not valid!')

        
        self.graph = np.where(self.p_values_corrected < self.alpha, self.cdfx_condition, 0) # filtered graph without sign information
        
        return self.graph


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
        
        if self.graph is None:
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
            # Place colorbar horizontally below the figure
            plt.colorbar(
                cax, ax=ax, orientation='horizontal', ticks=[-1, 0, 1],
                fraction=0.046, pad=0.15, label='Link Value'
            )

        fig, ax = plt.subplots(figsize=(5, 5))

        plot_heatmap(self.graph, 'Projection Matrix', ax)

        plt.tight_layout()
        if export_path:
            export_path_corr_matrix = f"{export_path}_adjacency.pdf"
            plt.savefig(export_path_corr_matrix, dpi=600)
        if show:
            plt.show()
        plt.close()


    def community_detection(
        self,
        trials: int = 500,
        n_jobs: int = 1,
        method: str = "bic",
        show: bool = False,
        random_state: int = 42,
        starter: str = "uniform"):
        """
        Detect communities in the current graph via greedy minimization with multiple randomized restarts.

        This method partitions the nodes of ``self.graph`` into communities by greedily minimizing
        an objective function. Two objectives are supported:

        • ``"bic"``: Bayesian Information Criterion of a signed stochastic block model (separate
            probabilities for positive and negative edges in each block);
        • ``"frustration"``: signed network frustration, penalizing negative edges inside
            communities and positive edges across communities.

        For robustness to local minima, the algorithm performs several independent trials, each
        starting from a different random community assignment. Trials are run in parallel and
        the partition with the lowest objective value is returned.

        Parameters
        ----------
        trials : int, optional
            Number of independent random restarts (trials) of the greedy algorithm.
            Each trial starts from a different initial community assignment.
            Default is 500.
        n_jobs : int or None, optional
            Number of parallel jobs used to run the trials. If ``None``, uses ``self.n_jobs``.
            Default is 1.
        method : {"bic", "frustration"}, optional
            Objective to minimize. ``"bic"`` uses the BIC of a signed SBM; ``"frustration"``
            uses network frustration. Default is ``"bic"``.
        show : bool, optional
            If ``True``, passes a verbose flag to the underlying parallel execution
            to log progress information. Default is ``False``.
        random_state : int or None, optional
            Seed for the global random number generator that produces per-trial seeds.
            Use this for reproducible community assignments. Default is 42.
        starter : str, optional
            Strategy used to generate initial community labels for each trial.
            If ``"uniform"``, each trial starts from a shuffled identity labeling
            (one unique label per node). Any other value triggers a mixture strategy
            that randomly chooses between shuffled identity and a random partition
            into ``k`` communities (with 2 ≤ k ≤ min(10, N)). Default is ``"uniform"``.

        Returns
        -------
        np.ndarray
            One-dimensional array of length ``N`` with the community label of each node
            (labels are relabeled to be contiguous integers starting at 0). The same
            array is also stored in ``self.communities``.

        Raises
        ------
        ValueError
            If ``self.graph`` is ``None`` (i.e., ``.build_graph()`` must be called first),
            or if ``method`` is not one of ``"bic"`` or ``"frustration"``.

        Notes
        -----
        The underlying graph is represented by ``self.graph``, a signed adjacency matrix.
        For the BIC objective, a signed stochastic block model with separate probabilities
        for positive and negative edges in each community pair is fitted, and the BIC
        is computed as a penalized negative log-likelihood. For the frustration objective,
        the loss counts (with weights) negative edges inside communities and positive
        edges between communities.

        During optimization, if a community becomes empty after a node move, the labels
        are renumbered so that community indices remain compact (0, 1, ..., K-1).
        """

        if self.graph is None:
            raise ValueError("You must call .build_graph() before running community detection.")
        
        if method not in ["bic", "frustration"]:
            raise ValueError('method must be either "bic" or "frustration".')
        
        if starter not in ["uniform", "mixture"]:
            raise ValueError('starter must be either "uniform" or "mixture".')

        if n_jobs is None:
            n_jobs = self.n_jobs

        #@staticmethod
        @jit(nopython=True)
        def _updateF(adj, C):
            """
            Frustration: penalize negative edges within a community and positive edges across communities.
            """
            F = 0.0
            N = len(C)
            for i in range(N):
                for j in range(N):
                    if adj[i, j] < 0 and C[i] == C[j]:
                        F += abs(adj[i, j])
                    elif adj[i, j] > 0 and C[i] != C[j]:
                        F += adj[i, j]
            return F / 2.0

        
        #@staticmethod
        @jit(nopython=True)
        def _compute_edges_probabilities(adj, C_index, sign):
            """
            Returns (prob_matrix, total_links, count_matrix) for edges of a given sign.
            """
            N = adj.shape[0]
            k = np.max(C_index) + 1

            # community counts
            com_counts = np.zeros(k, dtype=np.int64)
            for idx in C_index:
                com_counts[idx] += 1

            # total possible links between communities
            total_links = np.zeros((k, k), dtype=np.int64)
            for i in range(k):
                for j in range(k):
                    if i == j:
                        total_links[i, j] = com_counts[i] * (com_counts[i] - 1) // 2
                    else:
                        total_links[i, j] = com_counts[i] * com_counts[j]

            # count actual edges with given sign
            count_matrix = np.zeros((k, k), dtype=np.int64)
            for u in range(N):
                for v in range(u + 1, N):
                    if adj[u, v] == sign:
                        c1 = C_index[u]
                        c2 = C_index[v]
                        count_matrix[c1, c2] += 1
                        if c1 != c2:
                            count_matrix[c2, c1] += 1

            # probabilities
            prob = np.zeros((k, k), dtype=np.float64)
            for i in range(k):
                for j in range(k):
                    if total_links[i, j] > 0:
                        prob[i, j] = count_matrix[i, j] / total_links[i, j]
                    else:
                        prob[i, j] = 0.0
            return prob, total_links, count_matrix

        #@staticmethod
        # @jit(nopython=True)
        #@staticmethod
        def _updateBIC(adj, C):
            """
            BIC for signed SBM-like model with separate P(+) and P(-) per (community, community).
            """
            # normalize labels to 0..k-1
            unique = np.unique(C)
            remap = {c: i for i, c in enumerate(unique)}
            C_index = np.array([remap[c] for c in C], dtype=np.int64)

            N = adj.shape[0]
            k = len(unique)

            P_minus, _, L_minus = _compute_edges_probabilities(adj, C_index, -1)
            P_plus,  _, L_plus  = _compute_edges_probabilities(adj, C_index,  1)

            # counts per community
            n = np.zeros(k, dtype=np.int64)
            for idx in C_index:
                n[idx] += 1

            def _loglike(k, n, Lm, Lp, Pm, Pp):
                eps = 1e-10
                ll = 0.0
                for c in range(k):
                    nc = n[c]
                    if nc > 1:
                        Lmc = Lm[c, c]
                        Lpc = Lp[c, c]
                        Pmc = max(Pm[c, c], eps)
                        Ppc = max(Pp[c, c], eps)
                        one_minus = max(1 - Pmc - Ppc, eps)
                        ll += (Lmc * np.log(Pmc) + Lpc * np.log(Ppc)
                            + ((nc * (nc - 1)) / 2 - Lmc - Lpc) * np.log(one_minus))
                        for d in range(c + 1, k):
                            Lmd = Lm[c, d]
                            Lpd = Lp[c, d]
                            Pmd = max(Pm[c, d], eps)
                            Ppd = max(Pp[c, d], eps)
                            one_minus_d = max(1 - Pmd - Ppd, eps)
                            ll += (Lmd * np.log(Pmd) + Lpd * np.log(Ppd)
                                + (nc * n[d] - Lmd - Lpd) * np.log(one_minus_d))
                return ll

            ll = _loglike(k, n, L_minus, L_plus, P_minus, P_plus)
            # simple BIC form: k*(k+1) params (Pm and Pp per block), penalized by log(#pairs)
            bic = k * (k + 1) * np.log(N * (N - 1) / 2) - 2 * ll
            return bic

        #@staticmethod
        def _greedy_min(adj, C0, method="bic", rng=None):
            """Performs greedy node reassignment to minimize a specified objective function (BIC or Frustration) for community detection.

            Parameters
            ----------
            adj : np.ndarray
                Adjacency matrix representing the network/graph.
            C0 : np.ndarray
                Initial community assignments for each node.
            method : str, optional
                Objective function to minimize; either 'bic' (Bayesian Information Criterion) or 'frustration'. Default is 'bic'.
            rng : np.random.Generator or None, optional
                Random number generator for shuffling node order. If None, a new default generator is created.

            Returns
            -------
            C : np.ndarray
                Final community assignments for each node after optimization.
            final_val : float
                Value of the objective function (BIC or Frustration) after optimization.

            Raises
            ------
            ValueError
                If `method` is not 'bic' or 'frustration'.

            Notes
            -----
            - The function iteratively reassigns nodes to communities to greedily minimize the chosen objective.
            - If a community becomes empty after reassignment, community labels are relabeled compactly.
            """
            if method == "bic":
                update = _updateBIC
            elif method == "frustration":
                update = _updateF
            else:
                raise ValueError("method must be 'bic' or 'frustration'.")

            if rng is None:
                rng = np.random.default_rng()

            C = C0.copy()
            K = len(np.unique(C))
            stop = False

            while not stop:
                stop = True
                # Main difference with old version: we shuffle all nodes at once
                V = np.arange(len(C))
                rng.shuffle(V)

                for i in V:
                    g = C[i]
                    current_val = update(adj, C)
                    delta_best = 0.0
                    best_label = g

                    for cl in np.unique(C):
                        if cl == g:
                            continue
                        C[i] = cl
                        cand = update(adj, C)
                        gain = current_val - cand
                        if gain > delta_best:
                            delta_best = gain
                            best_label = cl
                        C[i] = g  # restore

                    if delta_best > 0:
                        C[i] = best_label
                        stop = False
                        # if old community becomes empty, relabel compactly
                        if np.sum(C == g) == 0:
                            # compress labels to 0..K'-1
                            uniq = np.unique(C)
                            remap = {c: ix for ix, c in enumerate(uniq)}
                            C = np.array([remap[x] for x in C], dtype=np.int64)
                            K = len(uniq)

            final_val = update(adj, C)
            return C, final_val

        # --- initialisation strategies to diversify starts
        def _make_start(rng: np.random.Generator) -> np.ndarray:
            # strategy 1: shuffled identity
            C = np.arange(self.N, dtype=np.int64)
            rng.shuffle(C)
            return C

        def _make_start_random_k(rng: np.random.Generator) -> np.ndarray:
            k = int(rng.integers(2, min(10, self.N) + 1))
            C = rng.integers(0, k, size=self.N, dtype=np.int64)
            # canonicalize labels
            uniq = np.unique(C)
            remap = {c: i for i, c in enumerate(uniq)}
            return np.array([remap[x] for x in C], dtype=np.int64)

        # pick a start generator at random per trial
        def _make_start_mixture(rng: np.random.Generator) -> np.ndarray:
            if rng.random() < 0.5:
                return _make_start(rng)
            return _make_start_random_k(rng)

        # --- per-trial runners
        def _single_trial_mixture(seed: int):
            rng = np.random.default_rng(seed)
            C0 = _make_start_mixture(rng)
            # expects your improved greedy with seedable rng
            C, loss = _greedy_min(self.graph, C0, method=method, rng=rng)
            return C, float(loss)

        def _single_trial(seed):
            rng = np.random.default_rng(seed)
            C0 = np.arange(self.N, dtype=np.int64)  # identity labeling as a safe start
            # Optional: you could also randomize starting labels for more diversity:
            rng.shuffle(C0)
            comm, loss = _greedy_min(self.graph, C0, method=method, rng=rng)
            return comm, loss
        rng_global = np.random.default_rng(random_state)
        seeds = rng_global.integers(0, 2**32 - 1, size=trials, dtype=np.uint32)
        
        def _run():
            return Parallel(n_jobs=n_jobs, backend='loky', batch_size='auto', verbose=2 if show==True else 0)(
                delayed(_single_trial)(int(s)) for s in seeds
            )
        
        def _run_mixture():
            return Parallel(n_jobs=n_jobs, backend='loky', batch_size='auto', verbose=2 if show==True else 0)(
                delayed(_single_trial_mixture)(int(s)) for s in seeds
            )

        if starter == 'uniform':
            results = _run()
        elif starter == 'mixture':
            results = _run_mixture()
        losses = np.array([loss for _, loss in results], dtype=float)
        best_idx = int(np.nanargmin(losses))
        best_comm = results[best_idx][0]

        self.communities= best_comm

        return self.communities


    def _reorder_graph(self, graph, labels):
            """Reorder adjacency matrix by community labels."""
            sorted_idx = np.argsort(labels)
            return graph[sorted_idx][:, sorted_idx], labels[sorted_idx]

    def _draw_community_blocks(self, ax, labels, color='black', linewidth=1.5):
        """Draw lines to separate communities."""
        boundaries = np.cumsum(np.unique(labels, return_counts=True)[1])[:-1]
        for b in boundaries:
            ax.axhline(b - 0.1, color=color, linewidth=linewidth)
            ax.axvline(b - 0.1, color=color, linewidth=linewidth)

    def plot_communities(self, export_path="", show=True):
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

        
        graph = self.graph
        labels = self.communities
        if graph is None or labels is None:
            raise ValueError(f"Graph or communities not available. Run build_graph() and community_detection().")

        reordered_graph, reordered_labels = self._reorder_graph(graph, labels)

        cmap = ListedColormap(["red", "white", "blue"])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(reordered_graph, cmap=cmap, norm=norm, 
                    cbar_kws={'ticks': [-1, 0, 1], 'orientation':'horizontal','fraction':0.046}, ax=ax, square=True)

        # ax.set_title(title)
        #     # Place colorbar horizontally below the figure
        #     plt.colorbar(
        #         cax, ax=ax, orientation='horizontal', ticks=[-1, 0, 1],
        #         fraction=0.046, pad=0.15, label='Link Value'
        #     )

        self._draw_community_blocks(ax, reordered_labels, color='black', linewidth=1.5)

        ax.set_title(f"Graph Reordered by Communities", fontsize=14, weight='bold')
        ax.set_xlabel("ROI (Reordered)", fontsize=12)
        ax.set_ylabel("ROI (Reordered)", fontsize=12)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()
        if export_path:
            plt.savefig(f"{export_path}_communities.pdf", dpi=600)
        if show:
            plt.show()
        plt.close()

    def plot_block_matrix(self, export_path="", show=True):
        """
        Plot block matrix of the graph based on detected communities.

        Parameters:
        -----------
        export_path : str, optional
            Path to save the PDF figure. If empty, the plot is not saved.
        show : bool, optional
            If True, display the figure.
        """

        if self.graph is None or self.communities is None:
            raise ValueError(f"Graph or communities not available. Run build_graph() and community_detection().")
        #@staticmethod
        def compute_diagonal_block_probabilities(adj_matrix, community_labels):
            """
            Compute the probabilities of positive and negative links in the diagonal blocks of an adjacency matrix.

            Parameters:
            - adj_matrix: numpy array, adjacency matrix of the graph
            - community_labels: numpy array, community labels for each node

            Returns:
            - dict: A dictionary with probabilities of positive and negative links for each community
            """
            unique_labels = np.unique(community_labels)
            probabilities = {}
            
            for label in unique_labels:
                # Get indices of nodes in the current community
                community_indices = np.where(community_labels == label)[0]
                N_c = len(community_indices)

                if N_c > 1:  # Only compute if the community has more than one node
                    # Extract the diagonal block for the current community
                    community_block = adj_matrix[np.ix_(community_indices, community_indices)]

                    # Count positive and negative links
                    num_positive_links = np.sum(community_block == 1)
                    num_negative_links = np.sum(community_block == -1)

                    # Total possible links in the diagonal block
                    total_links = N_c * (N_c - 1)

                    # Compute probabilities
                    prob_positive_links = num_positive_links / total_links
                    prob_negative_links = num_negative_links / total_links

                    probabilities[label] = {
                        'prob_positive_links': prob_positive_links,
                        'prob_negative_links': prob_negative_links
                    }
                else:
                    probabilities[label] = {
                        'prob_positive_links': 0,
                        'prob_negative_links': 0
                    }


            return probabilities

        #@staticmethod
        def compute_off_diagonal_block_probabilities(adj_matrix, community_labels):
            """
            Compute the probabilities of positive and negative links in the off-diagonal blocks of an adjacency matrix.

            Parameters:
            - adj_matrix: numpy array, adjacency matrix of the graph
            - community_labels: numpy array, community labels for each node

            Returns:
            - dict: A dictionary with probabilities of positive and negative links for each pair of communities
            """
            unique_labels = np.unique(community_labels)
            probabilities = {}

            for i, label_i in enumerate(unique_labels):
                for j, label_j in enumerate(unique_labels):
                    if i < j:  # Only consider off-diagonal blocks
                        # Get indices of nodes in the two communities
                        indices_i = np.where(community_labels == label_i)[0]
                        indices_j = np.where(community_labels == label_j)[0]

                        # Extract the off-diagonal block
                        block = adj_matrix[np.ix_(indices_i, indices_j)]

                        # Count positive and negative links
                        num_positive_links = np.sum(block == 1)
                        num_negative_links = np.sum(block == -1)

                        # Total possible links in the off-diagonal block
                        total_links = len(indices_i) * len(indices_j)

                        # Compute probabilities
                        prob_positive_links = num_positive_links / total_links if total_links > 0 else 0
                        prob_negative_links = num_negative_links / total_links if total_links > 0 else 0

                        probabilities[(label_i, label_j)] = {
                            'prob_positive_links': prob_positive_links,
                            'prob_negative_links': prob_negative_links
                        }

            return probabilities

        
        diag_probs = compute_diagonal_block_probabilities(self.graph, self.communities)
        off_probs = compute_off_diagonal_block_probabilities(self.graph, self.communities)

        unique_labels = np.unique(self.communities)
        M = np.empty((len(unique_labels), len(unique_labels)))

        for i, li in enumerate(unique_labels):
            for j, lj in enumerate(unique_labels):
                if i == j:
                    p_pos = diag_probs[li]['prob_positive_links']
                    p_neg = diag_probs[li]['prob_negative_links']
                    

                else:
                    key = (li, lj) if (li, lj) in off_probs else (lj, li)
                    p_pos = off_probs[key]['prob_positive_links']
                    p_neg = off_probs[key]['prob_negative_links']
                M[i, j] = 1 if p_pos > p_neg else -1 if p_pos < p_neg else 0

        fig,ax = plt.subplots(figsize=(5,5))
        cmap = ListedColormap(["red", "white", "blue"])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)
        sns.heatmap(M, cmap=cmap, norm=norm, linecolor='gray', linewidths=0.5,
                    cbar_kws={'ticks': [-1, 0, 1], 'orientation':'horizontal','fraction':0.046,'label':'Dominant Link Type'}, ax=ax, square=True)
        ax.set_title(f"Block Matrix based on Communities", fontsize=14, weight='bold')
        ax.set_xlabel("Community", fontsize=12)
        ax.set_ylabel("Community", fontsize=12)
        print(M.shape)
        num_labels = M.shape[0]
        ax.xaxis.set_major_locator(plt.FixedLocator(range(num_labels)))
        ax.yaxis.set_major_locator(plt.FixedLocator(range(num_labels)))
        ax.set_xticklabels([str(j+1) for j in range(num_labels)])
        ax.set_yticklabels([str(j+1) for j in range(num_labels)], rotation=0)

        plt.tight_layout()
        if export_path:
            plt.savefig(f"{export_path}_block_matrix.pdf", dpi=600)
        if show:
            plt.show()
        plt.close()
        return M
    
