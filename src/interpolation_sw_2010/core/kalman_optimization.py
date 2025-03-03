"""
Script to perform Kalman filter optimization and interpolation.

This script implements the optimization of Kalman filter parameters using Maximum Likelihood
Estimation (MLE) and performs interpolation of quarterly data into monthly estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path

from .kalman_filter import KalmanFilter
from .data_manager import DataManager

logger = logging.getLogger(__name__)


def run_kalman_optimization(
    y_reg: np.ndarray,
    x_reg: Dict[str, np.ndarray],
    y_m: np.ndarray,
    y_q: np.ndarray,
    b_start: np.ndarray,
    output_path: Optional[str] = None,
    dates: Optional[List[str]] = None,
    variable_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Run Kalman filter optimization and interpolation.
    
    Args:
        y_reg: Detrended quarterly observations
        x_reg: Dictionary of monthly indicators (regressors)
        y_m: Monthly trend
        y_q: Quarterly trend
        b_start: Initial parameter values
        output_path: Path to save results (optional)
        dates: List of date strings in YYYY-MM format
        variable_names: Names of the economic indicators being interpolated
        
    Returns:
        Interpolated monthly series
    """
    # Initialize Kalman filter
    kf = KalmanFilter(y_reg, x_reg, y_m, y_q)
    
    # Define element names (equivalent to namevec in MATLAB)
    element_names = ['first', 'second', 'third', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
    # Initialize results array
    results = []
    
    # Loop through quarterly variables (starting from index 1 as in MATLAB)
    for i in range(1, b_start.shape[1]):
        logger.info(f"Processing quarterly variable {i}")
        
        # Get element name
        element = element_names[i]
        
        # Get initial parameters for this variable
        b_st = b_start[:, i]
        # Remove NaN values if b_st is 1D
        if b_st.ndim == 1:
            b_st = b_st[~np.isnan(b_st)]
        else:
            b_st = b_st[~np.isnan(b_st).any(axis=1)]
        
        # Optimize parameters
        logger.info(f"Optimizing parameters for variable {i}")
        bmax = kf.optimize(b_st, i, element)
        
        # For problematic cases, repeat optimization
        if i in [2, 4, 5]:  # Equivalent to i==3, i==5, i==6 in MATLAB (0-indexed vs 1-indexed)
            logger.info(f"Repeating optimization for problematic variable {i}")
            bmax = kf.optimize(bmax, i, element)
        
        # Interpolation and smoothing
        logger.info(f"Performing interpolation and smoothing for variable {i}")
        tmp = kf.interpolate_and_smooth(bmax, i, element)
        
        # Save results
        results.append(tmp)
    
    # Convert results to numpy array
    results = np.column_stack(results)
    
    # Save results if output path is provided
    if output_path:
        output_dir = Path(output_path).parent
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame with dates and variable names
        if dates is not None:
            # Create a DataFrame with dates as the first column
            df = pd.DataFrame()
            df['Date'] = dates
            
            # Add the interpolated results
            if variable_names and len(variable_names) == results.shape[1]:
                for i, name in enumerate(variable_names):
                    df[name] = results[:, i]
            else:
                # Use default column names if variable_names not provided or incorrect length
                logger.warning("Variable names not provided or incorrect length. Using default names.")
                for i in range(results.shape[1]):
                    df[f"Variable_{i}"] = results[:, i]
        else:
            # No dates provided, just use the results
            if variable_names and len(variable_names) == results.shape[1]:
                df = pd.DataFrame(results, columns=variable_names)
            else:
                df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    # Log shapes and variable names for debugging
    logger.info(f"Results shape: {results.shape}")
    logger.info(f"Number of variable names: {len(variable_names)}")
    logger.info(f"Variable names: {variable_names}")
    
    return results


def main():
    """Main function to run the Kalman filter optimization."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define data path
    data_path = Path("data/raw_data.xlsx")
    
    # Get variable names from the quarterly data (excluding date columns)
    quarterly_df = pd.read_excel(data_path, sheet_name="Quarterly")
    variable_names = quarterly_df.columns[2:].tolist()  # Skip Year and Quarter columns
    
    # Load data using DataManager
    data_manager = DataManager()
    qdata, qtxt = data_manager.load_data(data_path, "Quarterly")
    mdata, mtxt = data_manager.load_data(data_path, "Monthly")
    
    # Transform data
    y, x, _ = data_manager.transform_data(qdata, mdata, data_manager.default_voa, qtxt, mtxt)
    
    # Create proper dates from the monthly data
    # First, extract Year and Month from the raw monthly data
    monthly_df = pd.read_excel(data_path, sheet_name="Monthly")
    years = monthly_df["Year"].values
    months = monthly_df["Month"].values
    
    # Create date strings in YYYY-MM format
    date_strings = [f"{int(year)}-{int(month):02d}" for year, month in zip(years, months)]
    
    # Get required data for Kalman filter
    y_reg = y[:, 2:]  # Remove year and month columns
    x_reg = {}
    
    # Define element names (equivalent to namevec in MATLAB)
    element_names = ['first', 'second', 'third', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
    # Set up x_reg with nested dictionaries
    x_reg = {
        'first': {element: x[:, 0:1] if element == 'first' else np.zeros((x.shape[0], 0)) for element in element_names},
        'second': {element: x[:, 1:3] if element == 'second' else np.zeros((x.shape[0], 0)) for element in element_names},
        'third': {element: x[:, 3:8] if element == 'third' else np.zeros((x.shape[0], 0)) for element in element_names},
        'four': {element: x[:, 8:10] if element == 'four' else np.zeros((x.shape[0], 0)) for element in element_names},
        'five': {element: x[:, 10:12] if element == 'five' else np.zeros((x.shape[0], 0)) for element in element_names},
        'six': {element: x[:, 12:16] if element == 'six' else np.zeros((x.shape[0], 0)) for element in element_names},
        'seven': {element: x[:, 16:20] if element == 'seven' else np.zeros((x.shape[0], 0)) for element in element_names},
        'eight': {element: x[:, 20:25] if element == 'eight' else np.zeros((x.shape[0], 0)) for element in element_names},
        'nine': {element: x[:, 25:26] if element == 'nine' else np.zeros((x.shape[0], 0)) for element in element_names}
    }
    
    # Get initial parameters
    b_start = get_initial_parameters(y_reg, x_reg)
    
    # Run Kalman filter optimization
    results = run_kalman_optimization(
        y_reg=y_reg,
        x_reg=x_reg,
        y_m=None,  # These will be computed in the Kalman filter
        y_q=None,  # These will be computed in the Kalman filter
        b_start=b_start,
        output_path="results/interpolated_data.csv",
        dates=date_strings,
        variable_names=variable_names[:-1]  # Exclude GDPDEF since it's used as a divisor for GDP_real
    )
    
    # Calculate GDP (similar to MATLAB code)
    gdp_nominal = np.sum(results[:, :-1], axis=1) - 2 * results[:, 6]  # Subtract imports twice
    gdp_real = gdp_nominal / results[:, -1] * 100
    
    # Combine GDP results
    gdp_results = np.column_stack([gdp_nominal, gdp_real, results[:, -1]])
    
    # Create output directory if it doesn't exist
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Create DataFrame with dates for GDP results
    gdp_df = pd.DataFrame()
    gdp_df['Date'] = date_strings
    gdp_df['GDP_nominal'] = gdp_nominal
    gdp_df['GDP_real'] = gdp_real
    gdp_df['Deflator'] = results[:, -1]
    
    # Save GDP results
    gdp_df.to_csv(output_dir / "gdp_results.csv", index=False)
    
    logger.info("Kalman filter optimization and interpolation completed successfully")


def get_initial_parameters(y_reg: np.ndarray, x_reg: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    """
    Get initial parameters for Kalman filter optimization.
    
    This is a placeholder function that would need to be implemented based on
    the MATLAB regressors.m function.
    
    Args:
        y_reg: Detrended quarterly observations
        x_reg: Dictionary of monthly indicators (regressors)
        
    Returns:
        Initial parameter values
    """
    # Initialize parameters array with the correct shape
    n_params = 10  # Maximum number of parameters for any series
    n_series = y_reg.shape[1]
    b_start = np.ones((n_params, n_series))
    
    # Set specific parameters for each series based on the number of indicators
    for i, (key, indicators) in enumerate(x_reg.items()):
        # Find the corresponding element in the indicators dictionary
        element_data = indicators[key]  # Each regressor only has data for its own element
        n_indicators = element_data.shape[1]
        # Fill with NaN after the actual parameters
        b_start[n_indicators:, i] = np.nan
    
    return b_start


def optimize_parameters(kf: KalmanFilter, b_st: np.ndarray, series_idx: int, element: str) -> np.ndarray:
    """
    Optimize Kalman filter parameters using Maximum Likelihood Estimation.
    
    Args:
        kf: KalmanFilter instance
        b_st: Initial parameter values
        series_idx: Index of the quarterly series
        element: Name of the element in x_reg dictionary
        
    Returns:
        Optimized parameters
    """
    # Remove NaN values if b_st is 1D
    if b_st.ndim == 1:
        b_st = b_st[~np.isnan(b_st)]
    else:
        b_st = b_st[~np.isnan(b_st).any(axis=1)]
    
    # Optimize parameters
    bmax = kf.optimize(b_st, series_idx, element)
    
    return bmax


if __name__ == "__main__":
    main() 