"""Kalman Filter implementation for Stock-Watson temporal disaggregation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Kalman Filter implementation for Stock-Watson temporal disaggregation.
    
    This class implements the Kalman Filter and Smoother for interpolating quarterly data
    into monthly estimates, following the methodology in Stock and Watson (2010).
    """
    
    def __init__(self, 
                 y_reg: np.ndarray = None, 
                 x_reg: Dict[str, np.ndarray] = None,
                 y_m: np.ndarray = None, 
                 y_q: np.ndarray = None):
        """
        Initialize the Kalman Filter.
        
        Args:
            y_reg: Detrended quarterly observations
            x_reg: Dictionary of monthly indicators (regressors)
            y_m: Monthly trend
            y_q: Quarterly trend
        """
        self.y_reg = y_reg
        self.x_reg = x_reg
        self.y_m = y_m
        self.y_q = y_q
        
    def log_likelihood(self, params: np.ndarray, series_idx: int, element: str) -> float:
        """
        Calculate the log-likelihood for the Kalman filter.
        
        Args:
            params: Parameters for optimization
            series_idx: Index of the quarterly series
            element: Name of the element in x_reg dictionary
            
        Returns:
            Log-likelihood value
        """
        # Remove NaN values if params is 1D
        if params.ndim == 1:
            params = params[~np.isnan(params)]
        else:
            params = params[~np.isnan(params).any(axis=1)]
        
        # Print debugging information
        print(f"Initial params shape: {params.shape}")
        
        # If y_m or y_q are None, compute them
        if self.y_m is None or self.y_q is None:
            print("Computing monthly and quarterly trends")
            # Get first valid indicator for dimensions
            first_key = None
            for key in self.x_reg:
                if isinstance(key, str) and element in self.x_reg[key] and self.x_reg[key][element].shape[1] > 0:
                    first_key = key
                    break
            
            if first_key is None:
                raise ValueError(f"No valid indicators found for element {element}")
            
            # Assuming the first column is time in months
            n_months = self.x_reg[first_key][element].shape[0]
            n_quarters = self.y_reg.shape[0]
            
            # Compute monthly trend (placeholder for actual trend calculation)
            if self.y_m is None:
                self.y_m = np.ones((n_months, self.y_reg.shape[1]))
            
            # Compute quarterly trend (placeholder for actual trend calculation)
            if self.y_q is None:
                self.y_q = np.ones((n_quarters, self.y_reg.shape[1]))
        
        # Calculate vague prior
        tmp = np.max(np.abs(self.y_reg[:, series_idx]))
        vague = 1000 * tmp * tmp
        
        # Initialize observation vector
        h = np.zeros(4)  # Observation vector
        r = 1.0e-08  # Small observation noise
        
        # Create transition matrix F
        f = np.zeros((4, 4))
        f[0, 3] = 1
        f[1:3, 0:2] = np.eye(2)
        
        # Initialize process noise covariance
        q = np.zeros((4, 4))
        
        # Count number of monthly indicators and create a mapping of valid indicators
        num_ind = 0
        valid_keys = []
        for key in self.x_reg:
            if isinstance(key, str) and element in self.x_reg[key] and self.x_reg[key][element].shape[1] > 0:
                valid_keys.append(key)
                num_ind += 1
        
        print(f"Number of valid indicators: {num_ind}")
        print(f"Valid keys: {valid_keys}")
        
        # Split parameters into components for each monthly indicator
        b = [{} for _ in range(num_ind)]
        index_size = 0
        for i, key in enumerate(valid_keys):
            n_coef = self.x_reg[key][element].shape[1]
            # Check if we have enough parameters
            if params.size < n_coef + 2:
                print(f"Warning: Not enough parameters. Expected {n_coef + 2}, got {params.size}")
                # Use available parameters and pad with defaults
                available_params = min(params.size, n_coef)
                b[i][element] = np.zeros(n_coef + 2)
                b[i][element][:available_params] = params[:available_params]
                # Set default values for rho and sigma if needed
                if params.size <= n_coef:
                    b[i][element][n_coef] = 0.9  # Default rho
                    b[i][element][n_coef + 1] = 0.1  # Default sigma
                elif params.size == n_coef + 1:
                    b[i][element][n_coef + 1] = 0.1  # Default sigma
                index_size = params.size
            else:
                # Normal case - we have enough parameters
                n_params = n_coef + 2  # Number of coefficients + rho + sigma
                print(f"Current index_size: {index_size}, Accessing params[{index_size}:{index_size + n_params}]")
                b[i][element] = params[index_size:index_size + n_params]
                index_size += n_params
        
        # Create Z matrix (fitted values of regression)
        first_key = valid_keys[0]
        z_reg = np.full((self.x_reg[first_key][element].shape[0], num_ind), np.nan)
        
        for i, key in enumerate(valid_keys):
            n_coef = self.x_reg[key][element].shape[1]
            z_reg[:, i] = self.x_reg[key][element] @ b[i][element][:n_coef]
        
        # Initial values for Kalman filter
        n_state = f.shape[0]
        n_m = z_reg.shape[0]
        p1 = vague * np.eye(n_state)
        x1 = np.zeros((n_state, 1))  # Make x1 a column vector
        
        # Storage for filter results
        x1t = np.zeros((n_m + 1, n_state))
        p1t = np.zeros((n_m + 1, n_state * n_state))
        x2t = np.zeros((n_m + 1, n_state))
        p2t = np.zeros((n_m + 1, n_state * n_state))
        
        # Initialize first row
        x1t[0, :] = x1.reshape(-1)
        p1t[0, :] = p1.flatten()
        
        # Initialize log-likelihood
        log_lik = 0.0
        
        # Main Kalman filter loop
        im = 0
        while im < z_reg.shape[0]:
            iq = im / 3  # Convert to quarters
            z = np.full((4, 1), np.nan)
            filled = False
            
            # Find appropriate monthly indicator
            ind_idx = num_ind - 1
            while not filled and ind_idx >= 0:
                if not np.isnan(z_reg[im, ind_idx]):
                    print(f"Found valid indicator at index {ind_idx} for month {im}")
                    # Reshape z_reg value to be a column vector
                    z_val = np.array([[z_reg[im, ind_idx]]])
                    z = np.vstack((z_val, np.zeros((3, 1))))
                    
                    # Get parameters for this indicator
                    n_coef = self.x_reg[valid_keys[ind_idx]][element].shape[1]
                    print(f"Using indicator {valid_keys[ind_idx]} with {n_coef} coefficients")
                    print(f"b[{ind_idx}][{element}] shape: {b[ind_idx][element].shape}")
                    
                    # Safely access parameters
                    if len(b[ind_idx][element]) > n_coef:
                        f[3, 3] = b[ind_idx][element][n_coef]  # rho parameter
                    else:
                        f[3, 3] = 0.9  # Default rho
                        
                    if len(b[ind_idx][element]) > n_coef + 1:
                        q[3, 3] = b[ind_idx][element][n_coef + 1] ** 2  # sigma^2 parameter
                    else:
                        q[3, 3] = 0.01  # Default sigma^2
                    filled = True
                else:
                    ind_idx -= 1
            
            if not filled:
                raise ValueError("Could not find valid indicator")
            
            # Check for missing data
            if np.isnan(z).any():
                raise ValueError("Missing data in z ... stopping")
            
            # For quarterly data points, compute log-likelihood
            if iq == np.floor(iq) and int(iq) < len(self.y_reg):  # If month is 3, 6, 9, or 12 and within bounds
                y = self.y_reg[int(iq), series_idx]
                
                # Simplified observation vector for quarterly aggregation
                h[0] = 1.0/3.0  # Equal weight for each month in the quarter
                h[1] = 1.0/3.0
                h[2] = 1.0/3.0
                
                # Kalman filter prediction and update
                x1, x2, p1, p2, v, s, k = self._kalman_filter_step(y, x1, p1, h, f, r, q, z)
                
                # Update log-likelihood
                log_lik += -0.5 * (np.log(2 * np.pi * s) + (v**2) / s)
            else:
                # Prediction only for non-quarterly points
                # Ensure proper dimensions
                x1_reshaped = np.atleast_2d(x1).reshape(-1, 1)  # Make x1 a column vector
                z_reshaped = np.atleast_2d(z).reshape(-1, 1)    # Make z a column vector
                
                # Prediction step
                x2 = f @ x1_reshaped + z_reshaped
                p2 = f @ p1 @ f.T + q
                x1 = x2
                p1 = p2
            
            # Store results (flatten vectors for storage)
            x1t[im + 1, :] = x1.reshape(-1)
            p1t[im + 1, :] = p1.flatten()
            x2t[im + 1, :] = x2.reshape(-1)
            p2t[im + 1, :] = p2.flatten()
            
            im += 1
        
        return -log_lik  # Return negative log-likelihood for minimization
    
    def _kalman_filter_step(self, y: float, x1: np.ndarray, p1: np.ndarray, 
                           h: np.ndarray, f: np.ndarray, r: float, q: np.ndarray, z: np.ndarray) -> tuple:
        """
        Perform one step of the Kalman filter.
        
        Args:
            y: Observation
            x1: Prior state estimate
            p1: Prior state covariance
            h: Observation matrix
            f: Transition matrix
            r: Observation noise variance
            q: Process noise covariance
            z: Exogenous input
            
        Returns:
            Tuple of (x1, x2, p1, p2, v, s, k)
        """
        # Ensure proper dimensions
        x1 = np.atleast_2d(x1).reshape(-1, 1)  # Make x1 a column vector
        z = np.atleast_2d(z).reshape(-1, 1)    # Make z a column vector
        h = np.atleast_2d(h).reshape(1, -1)    # Make h a row vector
        
        # Prediction step
        x2 = f @ x1 + z
        p2 = f @ p1 @ f.T + q
        
        # Update step
        v = y - float(h @ x2)  # Scalar observation
        s = float(h @ p2 @ h.T) + r  # Innovation covariance (scalar)
        k = p2 @ h.T / s  # Kalman gain
        
        # Updated state and covariance
        x1 = x2 + k * v
        p1 = p2 - k @ h @ p2
        
        return x1, x2, p1, p2, v, s, k
    
    def kalman_smoother(self, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray,
                        p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                        f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one step of the Kalman smoother.
        
        This is equivalent to the MATLAB ksmooth.m function.
        
        Args:
            x1: Filtered state estimate at t
            x2: Predicted state estimate at t+1
            x3: Smoothed state estimate at t+1
            p1: Filtered state covariance at t
            p2: Predicted state covariance at t+1
            p3: Smoothed state covariance at t+1
            f: Transition matrix
            
        Returns:
            Tuple containing smoothed state and covariance at time t
        """
        # Smoothing gain
        a_s = p1 @ f.T @ np.linalg.inv(p2)
        
        # Smoothed state covariance
        p3_new = p1 + a_s @ (p3 - p2) @ a_s.T
        
        # Smoothed state
        x3_new = x1 + a_s @ (x3 - x2)
        
        return x3_new, p3_new
    
    def interpolate_and_smooth(self, params: np.ndarray, series_idx: int, element: str) -> np.ndarray:
        """
        Perform interpolation and smoothing using the Kalman filter.
        
        This is equivalent to the MATLAB interpolation_and_smoothing.m function.
        
        Args:
            params: Optimized parameters
            series_idx: Index of the quarterly series
            element: Name of the element in x_reg dictionary
            
        Returns:
            Interpolated monthly series
        """
        # Remove NaN values if params is 1D
        if params.ndim == 1:
            params = params[~np.isnan(params)]
        else:
            params = params[~np.isnan(params).any(axis=1)]
        
        # If y_m or y_q are None, compute them
        if self.y_m is None or self.y_q is None:
            print("Computing monthly and quarterly trends for interpolation")
            # Get first valid indicator for dimensions
            first_key = None
            for key in self.x_reg:
                if isinstance(key, str) and element in self.x_reg[key] and self.x_reg[key][element].shape[1] > 0:
                    first_key = key
                    break
            
            if first_key is None:
                raise ValueError(f"No valid indicators found for element {element}")
            
            # Assuming the first column is time in months
            n_months = self.x_reg[first_key][element].shape[0]
            n_quarters = self.y_reg.shape[0]
            
            # Compute monthly trend (placeholder for actual trend calculation)
            if self.y_m is None:
                self.y_m = np.ones((n_months, self.y_reg.shape[1]))
            
            # Compute quarterly trend (placeholder for actual trend calculation)
            if self.y_q is None:
                self.y_q = np.ones((n_quarters, self.y_reg.shape[1]))
        
        # Calculate vague prior
        tmp = np.max(np.abs(self.y_reg[:, series_idx]))
        vague = 1000 * tmp * tmp
        
        # Initialize observation vector
        h = np.zeros(4)
        r = 1.0e-08  # Small observation noise
        
        # Create transition matrix F
        f = np.zeros((4, 4))
        f[0, 3] = 1
        f[1:3, 0:2] = np.eye(2)
        
        # Initialize process noise covariance
        q = np.zeros((4, 4))
        
        # Count number of monthly indicators and create a mapping of valid indicators
        num_ind = 0
        valid_keys = []
        for key in self.x_reg:
            if isinstance(key, str) and element in self.x_reg[key] and self.x_reg[key][element].shape[1] > 0:
                valid_keys.append(key)
                num_ind += 1
        
        # Split parameters into components for each monthly indicator
        b = [{} for _ in range(num_ind)]
        index_size = 0
        for i, key in enumerate(valid_keys):
            n_coef = self.x_reg[key][element].shape[1]
            # Check if we have enough parameters
            if params.size < n_coef + 2:
                print(f"Warning: Not enough parameters. Expected {n_coef + 2}, got {params.size}")
                # Use available parameters and pad with defaults
                available_params = min(params.size, n_coef)
                b[i][element] = np.zeros(n_coef + 2)
                b[i][element][:available_params] = params[:available_params]
                # Set default values for rho and sigma if needed
                if params.size <= n_coef:
                    b[i][element][n_coef] = 0.9  # Default rho
                    b[i][element][n_coef + 1] = 0.1  # Default sigma
                elif params.size == n_coef + 1:
                    b[i][element][n_coef + 1] = 0.1  # Default sigma
                index_size = params.size
            else:
                # Normal case - we have enough parameters
                n_params = n_coef + 2  # Number of coefficients + rho + sigma
                b[i][element] = params[index_size:index_size + n_params]
                index_size += n_params
        
        # Create Z matrix (fitted values of regression)
        first_key = valid_keys[0]
        z_reg = np.full((self.x_reg[first_key][element].shape[0], num_ind), np.nan)
        
        for i, key in enumerate(valid_keys):
            n_coef = self.x_reg[key][element].shape[1]
            z_reg[:, i] = self.x_reg[key][element] @ b[i][element][:n_coef]
        
        # Initial values for Kalman filter
        n_state = f.shape[0]
        n_m = z_reg.shape[0]
        p1 = vague * np.eye(n_state)
        x1 = np.zeros((n_state, 1))  # Make x1 a column vector
        
        # Storage for filter and smoother results
        x1t = np.zeros((n_m + 1, n_state))
        p1t = np.zeros((n_m + 1, n_state * n_state))
        x2t = np.zeros((n_m + 1, n_state))
        p2t = np.zeros((n_m + 1, n_state * n_state))
        
        # Initialize first row
        x1t[0, :] = x1.reshape(-1)
        p1t[0, :] = p1.flatten()
        
        # Main Kalman filter loop
        im = 0
        while im < z_reg.shape[0]:
            iq = im / 3  # Convert to quarters
            z = np.full((4, 1), np.nan)
            filled = False
            
            # Find appropriate monthly indicator
            ind_idx = num_ind - 1
            while not filled and ind_idx >= 0:
                if not np.isnan(z_reg[im, ind_idx]):
                    # Reshape z_reg value to be a column vector
                    z_val = np.array([[z_reg[im, ind_idx]]])
                    z = np.vstack((z_val, np.zeros((3, 1))))
                    
                    # Get parameters for this indicator
                    n_coef = self.x_reg[valid_keys[ind_idx]][element].shape[1]
                    
                    # Safely access parameters
                    if len(b[ind_idx][element]) > n_coef:
                        f[3, 3] = b[ind_idx][element][n_coef]  # rho parameter
                    else:
                        f[3, 3] = 0.9  # Default rho
                        
                    if len(b[ind_idx][element]) > n_coef + 1:
                        q[3, 3] = b[ind_idx][element][n_coef + 1] ** 2  # sigma^2 parameter
                    else:
                        q[3, 3] = 0.01  # Default sigma^2
                    filled = True
                else:
                    ind_idx -= 1
            
            if not filled:
                raise ValueError("Could not find valid indicator")
            
            # Check for missing data
            if np.isnan(z).any():
                raise ValueError("Missing data in z ... stopping")
            
            # For quarterly data points, compute log-likelihood
            if iq == np.floor(iq) and int(iq) < len(self.y_reg):  # If month is 3, 6, 9, or 12 and within bounds
                y = self.y_reg[int(iq), series_idx]
                
                # Simplified observation vector for quarterly aggregation
                h[0] = 1.0/3.0  # Equal weight for each month in the quarter
                h[1] = 1.0/3.0
                h[2] = 1.0/3.0
                
                # Kalman filter prediction and update
                x1, x2, p1, p2, _, _, _ = self._kalman_filter_step(y, x1, p1, h, f, r, q, z)
            else:
                # Prediction only for non-quarterly points
                # Ensure proper dimensions
                x1_reshaped = np.atleast_2d(x1).reshape(-1, 1)  # Make x1 a column vector
                z_reshaped = np.atleast_2d(z).reshape(-1, 1)    # Make z a column vector
                
                # Prediction step
                x2 = f @ x1_reshaped + z_reshaped
                p2 = f @ p1 @ f.T + q
                x1 = x2
                p1 = p2
            
            # Store results (flatten vectors for storage)
            x1t[im + 1, :] = x1.reshape(-1)
            p1t[im + 1, :] = p1.flatten()
            x2t[im + 1, :] = x2.reshape(-1)
            p2t[im + 1, :] = p2.flatten()
            
            im += 1
        
        # Store F matrices for all indicators
        f_base = np.zeros((4 * num_ind, 4))
        
        # Update F matrices with parameters
        for i in range(num_ind):
            # Copy the base F matrix
            f_base[i*4:(i+1)*4, :] = f.copy()
            
            # Update the rho parameter
            n_coef = self.x_reg[valid_keys[i]][element].shape[1]
            if len(b[i][element]) > n_coef:
                f_base[i*4 + 3, 3] = b[i][element][n_coef]  # rho parameter
            else:
                f_base[i*4 + 3, 3] = 0.9  # Default rho
        
        # Kalman smoother
        x3t = np.zeros((n_m + 1, n_state))
        p3t = np.zeros((n_m + 1, n_state * n_state))
        
        # Initialize with last filtered values
        x3t[n_m, :] = x1.reshape(-1)
        p3t[n_m, :] = p1.flatten()
        x3 = x1
        p3 = p1
        
        # Backward recursion
        j = n_m - 1
        while j >= 0:
            # Recover stored values
            x2 = x2t[j + 1, :].reshape(-1, 1)
            p2 = p2t[j + 1, :].reshape(n_state, n_state)
            x1 = x1t[j, :].reshape(-1, 1)
            p1 = p1t[j, :].reshape(n_state, n_state)
            
            # Find appropriate F matrix
            filled = False
            ind_idx = num_ind - 1
            while not filled and ind_idx >= 0:
                if not np.isnan(z_reg[j, ind_idx]):
                    filled = True
                    ft = f_base[ind_idx*4:(ind_idx+1)*4, :]
                else:
                    ind_idx -= 1
            
            if not filled:
                raise ValueError("Could not find valid indicator")
            
            # Apply smoother
            x3, p3 = self.kalman_smoother(x1, x2, x3, p1, p2, p3, ft)
            
            # Store results (flatten vectors for storage)
            x3t[j, :] = x3.reshape(-1)
            p3t[j, :] = p3.flatten()
            
            j -= 1
        
        # Extract smoothed estimates and multiply by monthly trend
        result = x3t[1:, 0] * self.y_m[:, series_idx]
        
        return result
    
    def optimize(self, initial_params: np.ndarray, series_idx: int, element: str) -> np.ndarray:
        """
        Optimize the Kalman filter parameters using MLE.
        
        Args:
            initial_params: Initial parameter values
            series_idx: Index of the quarterly series
            element: Name of the element in x_reg dictionary
            
        Returns:
            Optimized parameters
        """
        # Define objective function for minimization
        def objective(params):
            return self.log_likelihood(params, series_idx, element)
        
        # Run optimization
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'disp': True, 'maxiter': 1000}
        )
        
        return result.x
