import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Regressors:
    """
    Constructs regressors for the Stock-Watson interpolation procedure.
    
    This class implements the functionality of the MATLAB regressors.m file,
    which constructs an array structure where exogenous regressors are stored
    as matrices and calculates starting values for parameter estimation.
    
    The class organizes regressors for each economic indicator (PCE, Investment, etc.)
    and prepares them for use in the interpolation procedure.
    """
    
    def __init__(self):
        """Initialize the Regressors class."""
        # Store the regressors and starting values
        self.regressors = {}
        self.b_start = None
        
        # Define the indicators (matching the MATLAB code structure)
        self.indicators = [
            "PCE",          # Personal Consumption Expenditures
            "I_NS",         # Investment in Non-residential Structures
            "I_ES",         # Investment in Equipment and Software
            "I_RS",         # Residential Structures
            "I_chPI",       # Change in Private Inventories
            "X",            # Exports
            "IM",           # Imports
            "G",            # Government Spending
            "PGDP"          # GDP Price Index
        ]
        
        # Define the MATLAB element names for dictionary access
        self.element_names = {
            "PCE": "first",
            "I_NS": "second",
            "I_ES": "third",
            "I_RS": "four",
            "I_chPI": "five",
            "X": "six",
            "IM": "seven",
            "G": "eight",
            "PGDP": "nine"
        }
    
    def prepare(self, 
               Xreg: np.ndarray, 
               Y_ext: np.ndarray, 
               X_tr: np.ndarray,
               Y_m: Optional[np.ndarray] = None,
               debug: bool = False) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Prepare regressors and calculate starting values for parameter estimation.
        
        This method implements the functionality of the MATLAB regressors.m function,
        organizing the regressors for each economic indicator and calculating
        starting values for parameter estimation.
        
        Args:
            Xreg: Detrended monthly indicators
            Y_ext: Extended quarterly series
            X_tr: Transformed monthly indicators (before detrending)
            Y_m: Monthly data (used for calculating starting values)
            debug: If True, print debug information
            
        Returns:
            Tuple containing:
                - Dictionary of regressors for each indicator
                - Array of starting values for parameter estimation
        """
        if debug:
            logger.debug("Preparing regressors for Stock-Watson interpolation")
        
        # Initialize b_start with NaNs (matching MATLAB)
        self.b_start = np.full((20, len(self.indicators)), np.nan)
        
        # Create empty dictionary to store regressors
        self.regressors = {}
        
        # Get the component name from the shape of Xreg
        n_rows, n_cols = Xreg.shape
        
        # Process PCE (first indicator)
        # In MATLAB: X(1).first=[];
        self.regressors["first"] = []
        
        # Process I_NS (second indicator - Investment in Non-residential Structures)
        # In MATLAB: X(1).second=[ones(size(Xreg,1),1), Xreg(:,2)];
        #            X(2).second=[ones(size(Xreg,1),1), Xreg(:,3)];
        if n_cols > 3:  # Ensure we have enough columns
            self.regressors["second"] = []
            
            # Create regressors
            regressor1 = np.column_stack([np.ones(n_rows), Xreg[:, 1]])
            regressor2 = np.column_stack([np.ones(n_rows), Xreg[:, 2]])
            self.regressors["second"] = [regressor1, regressor2]
            
            # Calculate starting values (if Y_m is provided)
            if Y_m is not None and Y_ext is not None and Y_ext.shape[1] > 1:
                # Starting values for first regressor
                bx_start1 = np.array([0, 1])
                rho_start1 = 0.2
                
                # Calculate standard deviation of residuals
                try:
                    tmp2 = Y_ext[:, 1] - X_tr[:, 1]
                    # Remove rows with NaN values
                    tmp = tmp2[~np.isnan(tmp2)]
                    seps_start1 = np.std(tmp) / np.mean(Y_m[:, 1])
                except (IndexError, ValueError):
                    seps_start1 = 0.3  # Default value
                
                # Starting values for second regressor
                bx_start2 = np.array([0, 1])
                rho_start2 = 0.2
                
                try:
                    # Calculate standard deviation of residuals
                    tmp2 = Y_ext[:, 1] - X_tr[:, 2]
                    # Remove rows with NaN values
                    tmp = tmp2[~np.isnan(tmp2)]
                    seps_start2 = np.std(tmp) / np.mean(Y_m[:, 1])
                except (IndexError, ValueError):
                    seps_start2 = 0.3  # Default value
                
                # Combine starting values
                b_start1 = np.concatenate([bx_start1, [rho_start1, seps_start1], bx_start2, [rho_start2, seps_start2]])
                self.b_start[:len(b_start1), 1] = b_start1
        
        # Process I_ES (third indicator - Investment in Equipment and Software)
        # In MATLAB: X(1).third=[ones(size(Xreg,1),1), Xreg(:,4)];
        #            X(2).third=[ones(size(Xreg,1),1), Xreg(:,5:6)];
        #            X(3).third=[ones(size(Xreg,1),1), Xreg(:,7:8)];
        if n_cols > 8:  # Ensure we have enough columns
            self.regressors["third"] = []
            
            # Create regressors
            regressor1 = np.column_stack([np.ones(n_rows), Xreg[:, 3]])
            regressor2 = np.column_stack([np.ones(n_rows), Xreg[:, 4:6]])
            regressor3 = np.column_stack([np.ones(n_rows), Xreg[:, 6:8]])
            self.regressors["third"] = [regressor1, regressor2, regressor3]
            
            # Starting values
            bx_start1 = np.array([0, 1])
            bx_start2 = np.array([0, 1, 1])
            bx_start3 = np.array([0, 1, 1])
            rho_start1 = 0.2
            rho_start2 = 0.2
            rho_start3 = 0.2
            seps_start1 = 0.3
            seps_start2 = 0.3
            seps_start3 = 0.3
            
            b_start1 = np.concatenate([
                bx_start1, [rho_start1, seps_start1], 
                bx_start2, [rho_start2, seps_start2],
                bx_start3, [rho_start3, seps_start3]
            ])
            self.b_start[:len(b_start1), 2] = b_start1
        
        # Process I_RS (fourth indicator - Residential Structures)
        # In MATLAB: X(1).four=[ones(size(Xreg,1),1), Xreg(:,9)];
        #            X(2).four=[ones(size(Xreg,1),1), Xreg(:,10)];
        if n_cols > 10:  # Ensure we have enough columns
            self.regressors["four"] = []
            
            # Create regressors
            regressor1 = np.column_stack([np.ones(n_rows), Xreg[:, 8]])
            regressor2 = np.column_stack([np.ones(n_rows), Xreg[:, 9]])
            self.regressors["four"] = [regressor1, regressor2]
            
            # Calculate starting values (if Y_m is provided)
            if Y_m is not None and Y_ext is not None and Y_ext.shape[1] > 3:
                # Starting values
                bx_start1 = np.array([0, 1])
                bx_start2 = np.array([0, 1])
                rho_start1 = 0.2
                rho_start2 = 0.2
                
                try:
                    # Calculate standard deviation of residuals
                    tmp1 = Y_ext[:, 3] - X_tr[:, 8]
                    # Remove rows with NaN values
                    tmp = tmp1[~np.isnan(tmp1)]
                    seps_start1 = np.std(tmp) / np.mean(Y_m[:, 3])
                    
                    tmp1 = Y_ext[:, 3] - X_tr[:, 9]
                    # Remove rows with NaN values
                    tmp = tmp1[~np.isnan(tmp1)]
                    seps_start2 = np.std(tmp) / np.mean(Y_m[:, 3])
                except (IndexError, ValueError):
                    seps_start1 = 0.3  # Default value
                    seps_start2 = 0.3  # Default value
                
                # Combine starting values
                b_start1 = np.concatenate([
                    bx_start1, [rho_start1, seps_start1], 
                    bx_start2, [rho_start2, seps_start2]
                ])
                self.b_start[:len(b_start1), 3] = b_start1
        
        # Process I_chPI (fifth indicator - Change in Private Inventories)
        # In MATLAB: X(1).five=[ones(size(Xreg,1),1), Xreg(:,11)];
        #            X(2).five=[ones(size(Xreg,1),1), Xreg(:,12)];
        if n_cols > 12:  # Ensure we have enough columns
            self.regressors["five"] = []
            
            # Create regressors
            regressor1 = np.column_stack([np.ones(n_rows), Xreg[:, 10]])
            regressor2 = np.column_stack([np.ones(n_rows), Xreg[:, 11]])
            self.regressors["five"] = [regressor1, regressor2]
            
            # Starting values
            bx_start1 = np.array([0, 1])
            bx_start2 = np.array([0, 1])
            rho_start1 = 0.2
            rho_start2 = 0.2
            seps_start1 = 0.2
            seps_start2 = 0.2
            
            b_start1 = np.concatenate([
                bx_start1, [rho_start1, seps_start1], 
                bx_start2, [rho_start2, seps_start2]
            ])
            self.b_start[:len(b_start1), 4] = b_start1
        
        # Process X (sixth indicator - Exports)
        # In MATLAB: X(1).six=[ones(size(Xreg,1),1), Xreg(:,16)];
        #            X(2).six=[ones(size(Xreg,1),1), Xreg(:,14:16)];
        #            X(3).six=[ones(size(Xreg,1),1), Xreg(:,13)];
        if n_cols > 16:  # Ensure we have enough columns
            self.regressors["six"] = []
            
            # Create regressors
            regressor1 = np.column_stack([np.ones(n_rows), Xreg[:, 15]])
            regressor2 = np.column_stack([np.ones(n_rows), Xreg[:, 13:16]])
            regressor3 = np.column_stack([np.ones(n_rows), Xreg[:, 12]])
            self.regressors["six"] = [regressor1, regressor2, regressor3]
            
            # Starting values
            bx_start1 = np.array([0, 1])
            bx_start2 = np.array([0, 1, 1, 1])
            bx_start3 = np.array([0, 1])
            rho_start1 = 0.2
            rho_start2 = 0.2
            rho_start3 = 0.2
            seps_start1 = 0.3
            seps_start2 = 0.3
            seps_start3 = 0.3
            
            b_start1 = np.concatenate([
                bx_start1, [rho_start1, seps_start1], 
                bx_start2, [rho_start2, seps_start2],
                bx_start3, [rho_start3, seps_start3]
            ])
            self.b_start[:len(b_start1), 5] = b_start1
        
        # Process IM (seventh indicator - Imports)
        # In MATLAB: X(1).seven=[ones(size(Xreg,1),1), Xreg(:,20)];
        #            X(2).seven=[ones(size(Xreg,1),1), Xreg(:,18:20)];
        #            X(3).seven=[ones(size(Xreg,1),1), Xreg(:,17)];
        if n_cols > 20:  # Ensure we have enough columns
            self.regressors["seven"] = []
            
            # Create regressors
            regressor1 = np.column_stack([np.ones(n_rows), Xreg[:, 19]])
            regressor2 = np.column_stack([np.ones(n_rows), Xreg[:, 17:20]])
            regressor3 = np.column_stack([np.ones(n_rows), Xreg[:, 16]])
            self.regressors["seven"] = [regressor1, regressor2, regressor3]
            
            # Calculate starting values (if Y_m is provided)
            if Y_m is not None and Y_ext is not None and Y_ext.shape[1] > 6:
                # Starting values
                bx_start1 = np.array([0, 1])
                bx_start2 = np.array([0, 1, 1, 1])
                bx_start3 = np.array([0, 1])
                rho_start1 = 0.2
                rho_start2 = 0.2
                rho_start3 = 0.2
                
                try:
                    # Calculate standard deviation of residuals
                    tmp1 = Y_ext[:, 6] - X_tr[:, 19]
                    # Remove rows with NaN values
                    tmp = tmp1[~np.isnan(tmp1)]
                    seps_start1 = np.std(tmp) / np.mean(Y_m[:, 6])
                    
                    seps_start2 = seps_start1
                    
                    tmp1 = Y_ext[:, 6] - X_tr[:, 16]
                    # Remove rows with NaN values
                    tmp = tmp1[~np.isnan(tmp1)]
                    seps_start3 = np.std(tmp) / np.mean(Y_m[:, 6])
                except (IndexError, ValueError):
                    seps_start1 = 0.3  # Default value
                    seps_start2 = 0.3  # Default value
                    seps_start3 = 0.3  # Default value
                
                # Combine starting values
                b_start1 = np.concatenate([
                    bx_start1, [rho_start1, seps_start1], 
                    bx_start2, [rho_start2, seps_start2],
                    bx_start3, [rho_start3, seps_start3]
                ])
                self.b_start[:len(b_start1), 6] = b_start1
            else:
                # Default starting values
                bx_start1 = np.array([0, 1])
                bx_start2 = np.array([0, 1, 1, 1])
                bx_start3 = np.array([0, 1])
                rho_start1 = 0.2
                rho_start2 = 0.2
                rho_start3 = 0.2
                seps_start1 = 0.3
                seps_start2 = 0.3
                seps_start3 = 0.3
                
                b_start1 = np.concatenate([
                    bx_start1, [rho_start1, seps_start1], 
                    bx_start2, [rho_start2, seps_start2],
                    bx_start3, [rho_start3, seps_start3]
                ])
                self.b_start[:len(b_start1), 6] = b_start1
        
        # Process G (eighth indicator - Government Spending)
        # In MATLAB: X(1).eight=[ones(size(Xreg,1),1), Xreg(:,21), Xreg(:,22)];
        #            X(2).eight=[ones(size(Xreg,1),1), Xreg(:,21), Xreg(:,22), Xreg(:,24)];
        #            X(3).eight=[ones(size(Xreg,1),1), Xreg(:,21), Xreg(:,23), Xreg(:,25)];
        if n_cols > 25:  # Ensure we have enough columns
            self.regressors["eight"] = []
            
            # Create regressors
            regressor1 = np.column_stack([np.ones(n_rows), Xreg[:, 20], Xreg[:, 21]])
            regressor2 = np.column_stack([np.ones(n_rows), Xreg[:, 20], Xreg[:, 21], Xreg[:, 23]])
            regressor3 = np.column_stack([np.ones(n_rows), Xreg[:, 20], Xreg[:, 22], Xreg[:, 24]])
            self.regressors["eight"] = [regressor1, regressor2, regressor3]
            
            # Starting values
            bx_start1 = np.array([0, 1, 1])
            bx_start2 = np.array([0, 1, 1, 1])
            bx_start3 = np.array([0, 1, 1, 1])
            rho_start1 = 0.2
            rho_start2 = 0.2
            rho_start3 = 0.2
            seps_start1 = 0.3
            seps_start2 = 0.3
            seps_start3 = 0.3
            
            b_start1 = np.concatenate([
                bx_start1, [rho_start1, seps_start1], 
                bx_start2, [rho_start2, seps_start2],
                bx_start3, [rho_start3, seps_start3]
            ])
            self.b_start[:len(b_start1), 7] = b_start1
        
        # Process PGDP (ninth indicator - GDP Deflator)
        # In MATLAB: X(1).nine=[ones(size(Xreg,1),1), Xreg(:,26)];
        if n_cols > 26:  # Ensure we have enough columns
            self.regressors["nine"] = []
            
            # Create regressors
            regressor1 = np.column_stack([np.ones(n_rows), Xreg[:, 25]])
            self.regressors["nine"] = [regressor1]
            
            # Starting values
            bx_start = np.array([0, 1])
            rho_start = 0.2
            seps_start = 0.2
            
            b_start1 = np.concatenate([bx_start, [rho_start, seps_start]])
            self.b_start[:len(b_start1), 8] = b_start1
        
        if debug:
            logger.debug("Regressors preparation complete")
            
        # Map component names to MATLAB element names for compatibility
        component_regressors = {}
        for component in self.indicators:
            matlab_element = self.element_names.get(component)
            if matlab_element in self.regressors:
                component_regressors[component] = self.regressors[matlab_element]
        
        return component_regressors, self.b_start
    
    def construct_regressors(self, 
                            Xreg: np.ndarray, 
                            Y_ext: np.ndarray, 
                            X_tr: np.ndarray,
                            Y_m: Optional[np.ndarray] = None,
                            debug: bool = False) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Construct regressors and calculate starting values.
        
        This method implements the functionality of the MATLAB regressors.m function,
        organizing the regressors for each economic indicator and calculating
        starting values for parameter estimation.
        
        Args:
            Xreg: Detrended monthly indicators
            Y_ext: Extended quarterly series
            X_tr: Transformed monthly indicators (before detrending)
            Y_m: Monthly data (used for calculating starting values)
            debug: If True, print debug information
            
        Returns:
            Tuple containing:
                - Dictionary of regressors for each indicator
                - Array of starting values for parameter estimation
        """
        # For backward compatibility, call the prepare method
        return self.prepare(Xreg, Y_ext, X_tr, Y_m, debug) 