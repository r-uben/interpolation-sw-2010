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
        if debug:
            logger.debug("Constructing regressors for Stock-Watson interpolation")
        
        # Initialize b_start with NaNs (matching MATLAB)
        self.b_start = np.full((20, len(self.indicators)), np.nan)
        
        # Create empty dictionary to store regressors
        self.regressors = {indicator: [] for indicator in self.indicators}
        
        # Process each indicator
        self._process_pce(Xreg)
        self._process_nonres_investment(Xreg, Y_ext, X_tr, Y_m)
        self._process_equipment_investment(Xreg)
        self._process_residential_investment(Xreg, Y_ext, X_tr, Y_m)
        self._process_inventories(Xreg)
        self._process_exports(Xreg)
        self._process_imports(Xreg, Y_ext, X_tr, Y_m)
        self._process_government(Xreg)
        self._process_gdp_deflator(Xreg)
        
        if debug:
            logger.debug("Regressors construction complete")
            
        return self.regressors, self.b_start
    
    def _process_pce(self, Xreg: np.ndarray):
        """Process Personal Consumption Expenditures (PCE) indicator."""
        # First indicator: PCE - monthly averages to quarterly - no interpolation needed
        # In MATLAB: X(1).first=[];
        self.regressors["PCE"] = []
    
    def _process_nonres_investment(self, Xreg: np.ndarray, Y_ext: np.ndarray, X_tr: np.ndarray, Y_m: np.ndarray):
        """Process Investment in Non-residential Structures indicator."""
        # Second indicator: Inv in non-res structures
        # In MATLAB: X(1).second=[ones(size(Xreg,1),1), Xreg(:,2)];
        #            X(2).second=[ones(size(Xreg,1),1), Xreg(:,3)];
        
        # Create regressors
        regressor1 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 1]])
        regressor2 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 2]])
        self.regressors["I_NS"] = [regressor1, regressor2]
        
        # Calculate starting values (if Y_m is provided)
        if Y_m is not None:
            # Starting values for first regressor
            bx_start1 = np.array([0, 1])
            rho_start1 = 0.2
            
            # Calculate standard deviation of residuals
            tmp2 = Y_ext[:, 1] - X_tr[:, 1]
            # Remove rows with NaN values
            tmp = tmp2[~np.isnan(tmp2)]
            seps_start1 = np.std(tmp) / np.mean(Y_m[:, 1])
            
            # Starting values for second regressor
            bx_start2 = np.array([0, 1])
            rho_start2 = 0.2
            
            # Calculate standard deviation of residuals
            tmp2 = Y_ext[:, 1] - X_tr[:, 2]
            # Remove rows with NaN values
            tmp = tmp2[~np.isnan(tmp2)]
            seps_start2 = np.std(tmp) / np.mean(Y_m[:, 1])
            
            # Combine starting values
            b_start1 = np.concatenate([bx_start1, [rho_start1, seps_start1], bx_start2, [rho_start2, seps_start2]])
            self.b_start[:len(b_start1), 1] = b_start1
    
    def _process_equipment_investment(self, Xreg: np.ndarray):
        """Process Investment in Equipment and Software indicator."""
        # Third indicator: Inv, Equipment and Software
        # In MATLAB: X(1).third=[ones(size(Xreg,1),1), Xreg(:,4)];
        #            X(2).third=[ones(size(Xreg,1),1), Xreg(:,5:6)];
        #            X(3).third=[ones(size(Xreg,1),1), Xreg(:,7:8)];
        
        # Create regressors
        regressor1 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 3]])
        regressor2 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 4:6]])
        regressor3 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 6:8]])
        self.regressors["I_ES"] = [regressor1, regressor2, regressor3]
        
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
        
        # Combine starting values
        b_start1 = np.concatenate([
            bx_start1, [rho_start1, seps_start1], 
            bx_start2, [rho_start2, seps_start2],
            bx_start3, [rho_start3, seps_start3]
        ])
        self.b_start[:len(b_start1), 2] = b_start1
    
    def _process_residential_investment(self, Xreg: np.ndarray, Y_ext: np.ndarray, X_tr: np.ndarray, Y_m: np.ndarray):
        """Process Residential Structures indicator."""
        # Fourth indicator: Residential Structures
        # In MATLAB: X(1).four=[ones(size(Xreg,1),1), Xreg(:,9)];
        #            X(2).four=[ones(size(Xreg,1),1), Xreg(:,10)];
        
        # Create regressors
        regressor1 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 8]])
        regressor2 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 9]])
        self.regressors["I_RS"] = [regressor1, regressor2]
        
        # Calculate starting values (if Y_m is provided)
        if Y_m is not None:
            # Starting values
            bx_start1 = np.array([0, 1])
            bx_start2 = np.array([0, 1])
            rho_start1 = 0.2
            rho_start2 = 0.2
            
            # Calculate standard deviation of residuals
            tmp1 = Y_ext[:, 3] - X_tr[:, 8]
            # Remove rows with NaN values
            tmp = tmp1[~np.isnan(tmp1)]
            seps_start1 = np.std(tmp) / np.mean(Y_m[:, 3])
            
            tmp1 = Y_ext[:, 3] - X_tr[:, 9]
            # Remove rows with NaN values
            tmp = tmp1[~np.isnan(tmp1)]
            seps_start2 = np.std(tmp) / np.mean(Y_m[:, 3])
            
            # Combine starting values
            b_start1 = np.concatenate([
                bx_start1, [rho_start1, seps_start1], 
                bx_start2, [rho_start2, seps_start2]
            ])
            self.b_start[:len(b_start1), 3] = b_start1
    
    def _process_inventories(self, Xreg: np.ndarray):
        """Process Change in Inventories indicator."""
        # Fifth indicator: Change in Inventories
        # In MATLAB: X(1).five=[ones(size(Xreg,1),1), Xreg(:,11)];
        #            X(2).five=[ones(size(Xreg,1),1), Xreg(:,12)];
        
        # Create regressors
        regressor1 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 10]])
        regressor2 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 11]])
        self.regressors["I_chPI"] = [regressor1, regressor2]
        
        # Starting values
        bx_start1 = np.array([0, 1])
        bx_start2 = np.array([0, 1])
        rho_start1 = 0.2
        rho_start2 = 0.2
        seps_start1 = 0.2
        seps_start2 = 0.2
        
        # Combine starting values
        b_start1 = np.concatenate([
            bx_start1, [rho_start1, seps_start1], 
            bx_start2, [rho_start2, seps_start2]
        ])
        self.b_start[:len(b_start1), 4] = b_start1
    
    def _process_exports(self, Xreg: np.ndarray):
        """Process Exports indicator."""
        # Sixth indicator: Exports
        # In MATLAB: X(1).six=[ones(size(Xreg,1),1), Xreg(:,16)];
        #            X(2).six=[ones(size(Xreg,1),1), Xreg(:,14:16)];
        #            X(3).six=[ones(size(Xreg,1),1), Xreg(:,13)];
        
        # Create regressors
        regressor1 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 15]])
        regressor2 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 13:16]])
        regressor3 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 12]])
        self.regressors["X"] = [regressor1, regressor2, regressor3]
        
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
        
        # Combine starting values
        b_start1 = np.concatenate([
            bx_start1, [rho_start1, seps_start1], 
            bx_start2, [rho_start2, seps_start2],
            bx_start3, [rho_start3, seps_start3]
        ])
        self.b_start[:len(b_start1), 5] = b_start1
    
    def _process_imports(self, Xreg: np.ndarray, Y_ext: np.ndarray, X_tr: np.ndarray, Y_m: np.ndarray):
        """Process Imports indicator."""
        # Seventh indicator: Imports
        # In MATLAB: X(1).seven=[ones(size(Xreg,1),1), Xreg(:,20)];
        #            X(2).seven=[ones(size(Xreg,1),1), Xreg(:,18:20)];
        #            X(3).seven=[ones(size(Xreg,1),1), Xreg(:,17)];
        
        # Create regressors
        regressor1 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 19]])
        regressor2 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 17:20]])
        regressor3 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 16]])
        self.regressors["IM"] = [regressor1, regressor2, regressor3]
        
        # Calculate starting values (if Y_m is provided)
        if Y_m is not None:
            # Starting values
            bx_start1 = np.array([0, 1])
            bx_start2 = np.array([0, 1, 1, 1])
            bx_start3 = np.array([0, 1])
            rho_start1 = 0.2
            rho_start2 = 0.2
            rho_start3 = 0.2
            
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
            
            # Combine starting values
            b_start1 = np.concatenate([
                bx_start1, [rho_start1, seps_start1], 
                bx_start2, [rho_start2, seps_start2],
                bx_start3, [rho_start3, seps_start3]
            ])
            self.b_start[:len(b_start1), 6] = b_start1
    
    def _process_government(self, Xreg: np.ndarray):
        """Process Government Spending indicator."""
        # Eighth indicator: Government
        # In MATLAB: X(1).eight=[ones(size(Xreg,1),1), Xreg(:,21), Xreg(:,22)];
        #            X(2).eight=[ones(size(Xreg,1),1), Xreg(:,21), Xreg(:,22), Xreg(:,24)];
        #            X(3).eight=[ones(size(Xreg,1),1), Xreg(:,21), Xreg(:,23), Xreg(:,25)];
        
        # Create regressors
        regressor1 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 20], Xreg[:, 21]])
        regressor2 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 20], Xreg[:, 21], Xreg[:, 23]])
        regressor3 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 20], Xreg[:, 22], Xreg[:, 24]])
        self.regressors["G"] = [regressor1, regressor2, regressor3]
        
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
        
        # Combine starting values
        b_start1 = np.concatenate([
            bx_start1, [rho_start1, seps_start1], 
            bx_start2, [rho_start2, seps_start2],
            bx_start3, [rho_start3, seps_start3]
        ])
        self.b_start[:len(b_start1), 7] = b_start1
    
    def _process_gdp_deflator(self, Xreg: np.ndarray):
        """Process GDP Price Index indicator."""
        # Ninth indicator: GDP deflator
        # In MATLAB: X(1).nine=[ones(size(Xreg,1),1), Xreg(:,26)];
        
        # Create regressors
        regressor1 = np.column_stack([np.ones(Xreg.shape[0]), Xreg[:, 25]])
        self.regressors["PGDP"] = [regressor1]
        
        # Starting values
        bx_start = np.array([0, 1])
        rho_start = 0.2
        seps_start = 0.2
        
        # Combine starting values
        b_start1 = np.concatenate([bx_start, [rho_start, seps_start]])
        self.b_start[:len(b_start1), 8] = b_start1 