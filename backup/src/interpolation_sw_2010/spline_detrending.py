import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from typing import Tuple, List, Dict, Optional, Union, NamedTuple
from dataclasses import dataclass, field
import logging


logger = logging.getLogger(__name__)


@dataclass
class QuarterlySeriesConfig:
    """
    Configuration for a quarterly series and its related monthly indicators.
    
    This class encapsulates all the configuration parameters needed for a quarterly
    economic series and its associated monthly indicators in the Stock-Watson
    interpolation procedure.
    
    Attributes:
        name: Human-readable name of the quarterly series
        nknots_y: Number of knots for detrending the quarterly series
        nknots_x: Number of knots for detrending the monthly indicators
        monthly_indicators: List of names of related monthly indicators
    """
    name: str
    nknots_y: int  # Number of knots for detrending quarterly series
    nknots_x: int  # Number of knots for detrending monthly indicators
    monthly_indicators: List[str]  # Names of related monthly indicators
    
    @property
    def num_monthly_indicators(self) -> int:
        """Number of monthly indicators for this quarterly series."""
        return len(self.monthly_indicators)


class SplineDetrending:
    """
    Handles detrending of economic data using cubic splines.
    
    This class implements the cubic spline detrending approach used in the
    Stock-Watson procedure for temporal disaggregation of quarterly GDP to monthly values.
    It is a Python implementation of the MATLAB code in Main_US.m, cspline.m, and cspline_qm2.m.
    
    The class provides methods to:
    1. Detrend quarterly and monthly economic data using cubic splines
    2. Convert quarterly series to monthly frequency
    3. Prepare data for the Stock-Watson interpolation procedure
    
    Attributes:
        series_config: Dictionary mapping series keys to their configurations
    """
    
    def __init__(self):
        """Initialize the SplineDetrending class with default configuration."""
        # Define configuration for each quarterly series
        self.series_config = {
            "PCE": QuarterlySeriesConfig(
                name="Personal Consumption Expenditures",
                nknots_y=0,
                nknots_x=1,
                monthly_indicators=["PCE1"]
            ),
            "I_NS": QuarterlySeriesConfig(
                name="Non-residential Investment",
                nknots_y=5,
                nknots_x=1,
                monthly_indicators=["I_NS1", "I_NS2"]
            ),
            "I_ES": QuarterlySeriesConfig(
                name="Equipment & Software Investment",
                nknots_y=5,
                nknots_x=4,
                monthly_indicators=["I_ES1", "I_ES2", "I_ES3", "I_ES4", "I_ES5"]
            ),
            "I_RS": QuarterlySeriesConfig(
                name="Residential Investment",
                nknots_y=5,
                nknots_x=1,
                monthly_indicators=["I_RS1", "I_RS2"]
            ),
            "I_chPI": QuarterlySeriesConfig(
                name="Change in Private Inventories",
                nknots_y=5,
                nknots_x=1,
                monthly_indicators=["I_chPI1", "I_chPI2"]
            ),
            "X": QuarterlySeriesConfig(
                name="Exports",
                nknots_y=4,
                nknots_x=4,
                monthly_indicators=["X1", "X2", "X3", "X4"]
            ),
            "IM": QuarterlySeriesConfig(
                name="Imports",
                nknots_y=4,
                nknots_x=4,
                monthly_indicators=["IM1", "IM2", "IM3", "IM4"]
            ),
            "G": QuarterlySeriesConfig(
                name="Government Spending",
                nknots_y=5,
                nknots_x=4,
                monthly_indicators=["G1", "G2", "G3", "G4", "G5"]
            ),
            "PGDP": QuarterlySeriesConfig(
                name="GDP Price Index",
                nknots_y=5,
                nknots_x=5,
                monthly_indicators=["PGDP1"]
            )
        }
        
        # The ordered list of series keys (important for indexing)
        self.series_keys = ["PCE", "I_NS", "I_ES", "I_RS", "I_chPI", "X", "IM", "G", "PGDP"]
        
        # Define the mapping between quarterly column names in input data and series keys
        self.quarterly_mapping = {
            "PCEC": "PCE",
            "NR_STRUC_Q": "I_NS",
            "equip_q": "I_ES", 
            "RES_Q": "I_RS",
            "INVTCHANGE": "I_chPI",
            "exports_q": "X",
            "imports_q": "IM",
            "GOV_Q": "G",
            "GDPDEF": "PGDP"
        }
        
        # Results storage
        self.detrended_quarterly = None  # Detrended quarterly data
        self.detrended_monthly = None    # Detrended monthly data
        self.monthly_expanded = None     # Quarterly data expanded to monthly frequency
        
    def detrend_data(self, 
                    df_Y: pd.DataFrame, 
                    df_X: pd.DataFrame, 
                    debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detrend quarterly and monthly data using cubic splines.
        
        This method applies cubic spline detrending to both quarterly series and their
        associated monthly indicators according to the configuration in series_config.
        
        Args:
            df_Y: DataFrame of quarterly data with columns matching series_keys order
            df_X: DataFrame of monthly data with columns matching the flattened monthly_indicators
            debug: If True, print debug information during processing
            
        Returns:
            Tuple containing:
                - df_Yreg: DataFrame of detrended quarterly data
                - df_Xreg: DataFrame of detrended monthly data
        """
        if debug:
            logger.debug("Starting detrending with configuration:")
            for key, config in self.series_config.items():
                logger.debug(f"  {key} ({config.name}): {config.nknots_y} quarterly knots, {config.nknots_x} monthly knots")
        
        # Map column names if needed
        df_Y_mapped = self._map_quarterly_columns(df_Y, debug)
        
        # Convert DataFrames to numpy arrays for processing
        Y = df_Y_mapped.values
        X = df_X.values
        
        # Initialize output arrays
        Yreg = np.zeros_like(Y)
        Xreg = np.zeros_like(X)
        
        # Process each quarterly series
        for i, key in enumerate(self.series_keys):
            config = self.series_config[key]
            
            if debug:
                logger.debug(f"Processing {key} ({config.name})")
                
            # Detrend quarterly series if nknots_y > 0
            if config.nknots_y > 0:
                # Extract the quarterly series
                q = Y[:, i]
                
                # Fit cubic spline to quarterly data
                sqt = self._fit_cubic_spline(q, config.nknots_y)
                
                # Store detrended quarterly data
                Yreg[:, i] = Y[:, i] - sqt
                
                if debug:
                    logger.debug(f"  Detrended quarterly series {key} with {config.nknots_y} knots")
            else:
                # If nknots_y = 0, no detrending
                Yreg[:, i] = Y[:, i]
                
                if debug:
                    logger.debug(f"  No detrending for quarterly series {key}")
        
        # Process monthly indicators
        col_idx = 0
        for key in self.series_keys:
            config = self.series_config[key]
            
            if debug:
                logger.debug(f"Processing monthly indicators for {key} ({config.name})")
                
            # Process each monthly indicator for this quarterly series
            for j, indicator_name in enumerate(config.monthly_indicators):
                if config.nknots_x > 0:
                    # Extract the monthly indicator
                    m = X[:, col_idx]
                    
                    # Fit cubic spline to monthly data
                    smt = self._fit_cubic_spline(m, config.nknots_x)
                    
                    # Store detrended monthly data
                    Xreg[:, col_idx] = X[:, col_idx] - smt
                    
                    if debug:
                        logger.debug(f"  Detrended {indicator_name} with {config.nknots_x} knots")
                else:
                    # If nknots_x = 0, no detrending
                    Xreg[:, col_idx] = X[:, col_idx]
                    
                    if debug:
                        logger.debug(f"  No detrending for {indicator_name}")
                
                # Move to next column
                col_idx += 1
        
        # Convert back to DataFrames
        df_Yreg = pd.DataFrame(Yreg, columns=df_Y_mapped.columns)
        df_Xreg = pd.DataFrame(Xreg, columns=df_X.columns)
        
        # Store results
        self.detrended_quarterly = df_Yreg
        self.detrended_monthly = df_Xreg
        
        return df_Yreg, df_Xreg
    
    def expand_quarterly_to_monthly(self, 
                                   df_Y: pd.DataFrame, 
                                   debug: bool = False) -> pd.DataFrame:
        """
        Expand quarterly data to monthly frequency.
        
        This method converts quarterly data to monthly frequency by assigning
        the same value to each month in the quarter.
        
        Args:
            df_Y: DataFrame of quarterly data
            debug: If True, print debug information
            
        Returns:
            DataFrame of monthly frequency data expanded from quarterly data
        """
        if debug:
            logger.debug(f"Expanding quarterly data of shape {df_Y.shape} to monthly frequency")
        
        # Get the number of quarters and variables
        n_quarters, n_vars = df_Y.shape
        
        # Calculate the number of months
        n_months = n_quarters * 3
        
        # Initialize the expanded DataFrame
        df_Y_monthly = pd.DataFrame(
            np.zeros((n_months, n_vars)),
            columns=df_Y.columns
        )
        
        # Expand each variable
        for i in range(n_vars):
            # Get the quarterly series
            q_series = df_Y.iloc[:, i].values
            
            # Expand to monthly
            m_series = self._expand_series(q_series, n_months)
            
            # Store in the DataFrame
            df_Y_monthly.iloc[:, i] = m_series
        
        if debug:
            logger.debug(f"Expanded to monthly data of shape {df_Y_monthly.shape}")
        
        # Store the result
        self.monthly_expanded = df_Y_monthly
        
        return df_Y_monthly
    
    def quarterly_to_monthly_spline(self, 
                                   df_Y: pd.DataFrame, 
                                   nknots: int = 4,
                                   debug: bool = False) -> pd.DataFrame:
        """
        Convert quarterly data to monthly frequency using cubic splines.
        
        This method uses cubic splines to interpolate quarterly data to monthly frequency,
        resulting in a smoother conversion than simple expansion.
        
        Args:
            df_Y: DataFrame of quarterly data
            nknots: Number of knots to use for the cubic spline
            debug: If True, print debug information
            
        Returns:
            DataFrame of monthly frequency data interpolated from quarterly data
        """
        if debug:
            logger.debug(f"Converting quarterly data of shape {df_Y.shape} to monthly using splines with {nknots} knots")
        
        # Get the number of quarters and variables
        n_quarters, n_vars = df_Y.shape
        
        # Calculate the number of months
        n_months = n_quarters * 3
        
        # Initialize the monthly DataFrame
        df_Y_monthly = pd.DataFrame(
            np.zeros((n_months, n_vars)),
            columns=df_Y.columns
        )
        
        # Process each variable
        for i in range(n_vars):
            # Get the quarterly series
            q_series = df_Y.iloc[:, i].values
            
            # Convert to monthly using cubic spline
            m_series, _ = self._quarterly_to_monthly_spline(q_series, nknots)
            
            # Store in the DataFrame
            df_Y_monthly.iloc[:, i] = m_series
        
        if debug:
            logger.debug(f"Converted to monthly data of shape {df_Y_monthly.shape}")
        
        return df_Y_monthly
    
    def _fit_cubic_spline(self, y: np.ndarray, nknots: int) -> np.ndarray:
        """
        Fit a cubic spline to the data and return fitted values.
        
        This method implements the cubic spline fitting procedure used in the
        Stock-Watson interpolation method, equivalent to the MATLAB cspline.m function.
        
        Args:
            y: Input data array
            nknots: Number of knots to use for the spline
            
        Returns:
            Array of fitted values from the cubic spline
        """
        n = len(y)
        x = np.arange(1, n + 1)
        
        # Create knots evenly spaced across the domain
        knots = np.linspace(1, n, nknots + 2)[1:-1]
        
        # Create design matrix with polynomial terms
        X = np.column_stack([np.ones(n), x, x**2, x**3])
        
        # Add spline basis functions for each knot
        for knot in knots:
            X = np.column_stack([X, (np.maximum(0, x - knot))**3])
        
        # Fit the model using least squares
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Compute fitted values
        yhat = X @ beta
        
        return yhat
    
    def _quarterly_to_monthly_spline(self, q: np.ndarray, nknots: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert quarterly data to monthly frequency using cubic splines.
        
        This method implements the MATLAB cspline_qm2.m function, which fits a cubic
        spline to quarterly data and then evaluates it at monthly frequency.
        
        Args:
            q: Quarterly time series
            nknots: Number of knots for the spline
            
        Returns:
            Tuple containing:
                - Monthly frequency time series
                - Spline coefficients
        """
        # Number of quarters
        n_quarters = len(q)
        
        # Create quarterly time indices (0, 1, 2, ...)
        t_quarterly = np.arange(n_quarters)
        
        # Create knot points evenly spaced across the domain
        knots = np.linspace(0, n_quarters-1, nknots+2)[1:-1]
        
        # Create design matrix for quarterly data
        X_q = np.column_stack([
            np.ones(n_quarters),
            t_quarterly,
            t_quarterly**2,
            t_quarterly**3
        ])
        
        # Add spline basis functions for each knot
        for knot in knots:
            X_q = np.column_stack([X_q, (np.maximum(0, t_quarterly - knot))**3])
        
        # Estimate coefficients using least squares
        beta = np.linalg.lstsq(X_q, q, rcond=None)[0]
        
        # Create monthly time indices (0, 1/3, 2/3, 1, 4/3, ...)
        t_monthly = np.arange(3*n_quarters) / 3
        
        # Create design matrix for monthly data
        X_m = np.column_stack([
            np.ones(3*n_quarters),
            t_monthly,
            t_monthly**2,
            t_monthly**3
        ])
        
        # Add spline basis functions for each knot
        for knot in knots:
            X_m = np.column_stack([X_m, (np.maximum(0, t_monthly - knot))**3])
        
        # Compute fitted values at monthly frequency
        y_monthly = X_m @ beta
        
        return y_monthly, beta
    
    def _expand_series(self, y: np.ndarray, n_months: int) -> np.ndarray:
        """
        Expand a quarterly series to monthly frequency.
        
        This method assigns the same value to each month in a quarter.
        
        Args:
            y: Quarterly data array
            n_months: Total number of months to expand to
            
        Returns:
            Monthly frequency array expanded from quarterly data
        """
        # Initialize expanded series
        y_expanded = np.zeros(n_months)
        
        # For each quarter, assign the same value to each month in that quarter
        for i in range(len(y)):
            idx_start = 3*i
            idx_end = min(idx_start + 3, n_months)
            y_expanded[idx_start:idx_end] = y[i]
        
        return y_expanded
    
    def get_all_monthly_indicators(self) -> List[str]:
        """
        Get a flattened list of all monthly indicators in the correct order.
        
        Returns:
            List of all monthly indicator names in the order they should appear in the data
        """
        all_indicators = []
        for key in self.series_keys:
            all_indicators.extend(self.series_config[key].monthly_indicators)
        return all_indicators
    
    def _map_quarterly_columns(self, df_Y: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
        """
        Map quarterly column names from input data to the expected series names.
        
        Args:
            df_Y: DataFrame of quarterly data with original column names
            debug: If True, print debug information
            
        Returns:
            DataFrame with mapped column names matching series_keys
        """
        # Copy the DataFrame to avoid modifying the original
        df_mapped = df_Y.copy()
        
        # Preserve date columns
        date_cols = {}
        for col in ['Year', 'Quarter', 'Month']:
            if col in df_mapped.columns:
                date_cols[col] = df_mapped[col].copy()
                df_mapped = df_mapped.drop(columns=[col])
        
        if debug:
            logger.debug(f"Original quarterly columns: {df_Y.columns.tolist()}")
            
        # Create a new DataFrame with series_keys
        result_columns = []
        data_dict = {}
        
        # Map the columns using quarterly_mapping
        for key in self.series_keys:
            found = False
            # Try to find the corresponding original column
            for orig_col, mapped_key in self.quarterly_mapping.items():
                if mapped_key == key and orig_col in df_mapped.columns:
                    data_dict[key] = df_mapped[orig_col].values
                    result_columns.append(key)
                    found = True
                    if debug:
                        logger.debug(f"Mapped column: {orig_col} -> {key}")
                    break
            
            if not found:
                # Try to use the exact key name if it exists
                if key in df_mapped.columns:
                    data_dict[key] = df_mapped[key].values
                    result_columns.append(key)
                    if debug:
                        logger.debug(f"Using column as-is: {key}")
                else:
                    logger.warning(f"Could not find matching column for series key: {key}")
                    # Add an empty column (all zeros) to maintain structure
                    data_dict[key] = np.zeros(len(df_mapped))
                    result_columns.append(key)
        
        # Create a new DataFrame with mapped columns
        result_df = pd.DataFrame(data_dict)
        
        # Restore date columns
        for col, data in date_cols.items():
            if col == 'Year':
                result_df.insert(0, col, data)
            elif col == 'Quarter':
                loc = 1 if 'Year' in result_df.columns else 0
                result_df.insert(loc, col, data)
            elif col == 'Month':
                loc = 1 if 'Year' in result_df.columns else 0
                loc = loc + 1 if 'Quarter' in result_df.columns else loc
                result_df.insert(loc, col, data)
        
        if debug:
            logger.debug(f"Mapped quarterly columns: {result_df.columns.tolist()}")
            
        return result_df


def create_spline_qm2(y: np.ndarray, nknots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create cubic spline for quarterly data and convert to monthly frequency.
    
    Args:
        y: Quarterly data array
        nknots: Number of knots for the spline
        
    Returns:
        Tuple containing:
            - detrended quarterly data
            - detrended monthly data
    """
    spline = SplineDetrending()
    return spline._quarterly_to_monthly_spline(y, nknots)


def create_spline(x: np.ndarray, nknots: int) -> np.ndarray:
    """
    Create cubic spline for monthly data.
    
    Args:
        x: Monthly data array
        nknots: Number of knots for the spline
        
    Returns:
        Detrended monthly data
    """
    spline = SplineDetrending()
    return spline._fit_cubic_spline(x, nknots)


def expand_data(y: np.ndarray, n_months: int) -> np.ndarray:
    """
    Expand quarterly data to monthly frequency.
    
    Args:
        y: Quarterly data array
        n_months: Number of months to expand to
        
    Returns:
        Monthly frequency data expanded from quarterly data
    """
    spline = SplineDetrending()
    return spline._expand_series(y, n_months)


def detrend_data(y: np.ndarray, x: np.ndarray, y_m: np.ndarray, 
                 y_q: np.ndarray, x_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detrend quarterly and monthly data.
    
    Args:
        y: Quarterly data array
        x: Monthly data array
        y_m: Monthly trend for quarterly data
        y_q: Quarterly trend
        x_m: Monthly trend for monthly data
        
    Returns:
        Tuple containing:
            - detrended quarterly data
            - detrended monthly data
    """
    # Convert arrays to DataFrames for compatibility with SplineDetrending
    df_y = pd.DataFrame(y)
    df_x = pd.DataFrame(x)
    
    # Create SplineDetrending instance
    spline = SplineDetrending()
    
    # Detrend the data
    y_reg, x_reg = spline.detrend_data(df_y, df_x)
    
    return y_reg.values, x_reg.values 