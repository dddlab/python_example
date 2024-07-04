import _gaccord as _accord
import numpy as np
import pandas as pd

def temp(S):

    assert (type(S) == np.ndarray and S.dtype == 'float64')

    _accord.accord(S, lam_mat, lam2, epstol, maxitr, tau, penalize_diag, hist_norm_diff, hist_hn)

def check_symmetry(a, rtol=1e-5, atol=1e-8):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_lambda_matrix(lam1, S):
    if isinstance(lam1, float) | isinstance(lam1, int):
        lam_mat = np.full_like(S, lam1, order='F', dtype='float64')
    elif isinstance(lam1, np.ndarray):
        lam_mat = lam1
    return lam_mat

def accord(S, lam1=0.1, lam2=0.0, epstol=1e-5, maxitr=100, penalize_diag=True):
    """
    ACCORD algorithm for precision matrix estimation
    
    Parameters
    ----------
    S : ndarray of shape (n_features, n_features)
        Sample covariance matrix
    lam1 : float
        The l1-regularization parameter
    lam2 : float
        The l2-regularization parameter
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements
    
    Returns
    -------
    Theta : ndarray of shape (n_features, n_features)
        Estimated precision matrix
    hist : ndarray of shape (n_iters, 2)
        The list of values of (successive_norm_diff, objective) at each iteration until convergence
    """
    assert (type(S) == np.ndarray and S.dtype == 'float64')

    lam_mat = get_lambda_matrix(lam1, S)
    
    assert type(lam_mat) == np.ndarray and lam_mat.dtype == 'float64'
    assert check_symmetry(lam_mat)

    if not penalize_diag:
        np.fill_diagonal(lam_mat, 0)

    hist_norm_diff = np.full((maxitr, 1), -1, order='F', dtype='float64')
    hist_hn = np.full((maxitr, 1), -1, order='F', dtype='float64')

    tau = 1/np.linalg.svd(S)[1][0]

    Theta = _accord.accord(S, lam_mat, lam2, epstol, maxitr, tau, penalize_diag, hist_norm_diff, hist_hn)
    hist = np.hstack([hist_norm_diff, hist_hn])
    hist = hist[np.where(hist[:,0]!=-1)]

    return Theta, hist

def pseudo_BIC(X, Theta):
    n, p = X.shape
    Theta_reg = Theta/Theta.diagonal()[None,:]
    
    RSS = (X @ Theta_reg)**2
    RSS_i = RSS.sum(axis=0)
    num_nonzero = len(np.flatnonzero(Theta_reg))
    BIC = (np.log(n) * num_nonzero) + np.inner(np.diag(Theta), RSS_i) - n*np.sum(np.log(np.diag(Theta)))
    
    return BIC

class GraphicalAccord:
    """
    ACCORD algorithm for precision matrix estimation
    
    Parameters
    ----------
    lam1 : float
        The l1-regularization parameter
    lam2 : float
        The l2-regularization parameter
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements
    
    Attributes
    ----------
    precision_ : sparse matrix of shape (n_features, n_features)
        Estimated precision matrix
    hist_ : ndarray of shape (n_iters, 2)
        The list of values of (successive_norm_diff, objective) at each iteration until convergence
    """
    def __init__(self, lam1=0.1, lam2=0.0, epstol=1e-5, maxitr=100, penalize_diag=True):
        self.lam1 = lam1
        self.lam2 = lam2
        self.epstol = epstol
        self.maxitr = maxitr
        self.penalize_diag = penalize_diag
    
    def fit(self, X, y=None):
        """
        Fit ACCORD

        Parameters
        ----------
        X : ndarray, shape (n_samples, p_features)
            Data from which to compute the inverse covariance matrix
        y : (ignored)
        """
        S = np.cov(X, rowvar=False)

        self.precision_, self.hist_ = accord(S,
                                             lam1=self.lam1,
                                             lam2=self.lam2,
                                             epstol=self.epstol,
                                             maxitr=self.maxitr,
                                             penalize_diag=self.penalize_diag)
        
        return self