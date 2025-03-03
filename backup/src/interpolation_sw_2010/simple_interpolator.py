#!/usr/bin/env python
"""
Simplified Stock-Watson Interpolation Script

This script loads raw economic data and produces monthly interpolated values 
for quarterly GDP components with no configuration needed.

Input: data/raw_data.xlsx (always)
Output: data/monthly_gdp.csv (always)
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import os
from typing import Optional, Tuple, Dict, Any, List

from .data_manager import DataManager, TransformationResult
from .spline_detrending import SplineDetrending
from .regressors import Regressors
from .kalman_filter import KalmanFilter
from .kalman_optimization import run_kalman_optimization

# Set up simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def interpolate_gdp_monthly(debug: bool = False) -> pd.DataFrame:
    """
    Simple function to interpolate quarterly GDP to monthly frequency.
    Uses fixed input (data/raw_data.xlsx) and output (data/monthly_gdp.csv) paths.
    
    Args:
        debug: Whether to enable debug output
    
    Returns:
        DataFrame with monthly interpolated GDP data
    """
    # Fixed input and output paths
    input_file = "data/raw_data.xlsx"
    output_file = "data/monthly_gdp.csv"
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {input_file}")
    
    # Initialize data manager and load data
    data_manager = DataManager(data_path=input_file)
    
    # Transform the data to prepare it for interpolation
    result = data_manager.transform(debug=debug)
    
    # Save transformed data to get properly formatted DataFrames
    temp_dir = output_path.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_excel = temp_dir / "temp_transformed.xlsx"
    
    try:
        df_Y, df_X = data_manager.save_transformed_data(str(temp_excel))
        logger.info(f"Quarterly data shape: {df_Y.shape}, Monthly data shape: {df_X.shape}")
        
        # Log column names for debugging
        logger.info(f"Quarterly data columns: {df_Y.columns.tolist()}")
        logger.info(f"Monthly data columns: {df_X.columns.tolist()}")
        
        # Extract date columns
        date_cols_Y = [col for col in df_Y.columns if col in ['Year', 'Quarter']]
        date_cols_X = [col for col in df_X.columns if col in ['Year', 'Month']]
        
        Y_dates = df_Y[date_cols_Y] if date_cols_Y else None
        X_dates = df_X[date_cols_X] if date_cols_X else None
        
        # Create date columns if missing
        if X_dates is None or len(date_cols_X) < 2:
            logger.warning("Date columns not found in monthly data, creating default dates")
            num_months = len(df_X)
            years = [1959 + i//12 for i in range(num_months)]
            months = [(i % 12) + 1 for i in range(num_months)]
            X_dates = pd.DataFrame({'Year': years, 'Month': months})
        
        # Create SplineDetrending instance and detrend the data
        logger.info("Detrending data with cubic splines")
        spline_detrending = SplineDetrending()
        
        # Remove date columns before detrending
        df_Y_for_detrend = df_Y.drop(date_cols_Y, axis=1, errors='ignore')
        df_X_for_detrend = df_X.drop(date_cols_X, axis=1, errors='ignore')
        
        df_Y_detrended, df_X_detrended = spline_detrending.detrend_data(
            df_Y_for_detrend, df_X_for_detrend, debug=debug
        )
        
        # Expand quarterly data to monthly frequency for reference
        try:
            logger.info("Expanding quarterly data to monthly frequency")
            df_Y_monthly = spline_detrending.expand_quarterly_to_monthly(df_Y_for_detrend)
            logger.info(f"Expanded quarterly data shape: {df_Y_monthly.shape}")
        except Exception as e:
            logger.warning(f"Error expanding quarterly data: {e}")
            logger.warning("Using simple expansion method")
            # Simple expansion - replicate each quarterly value for 3 months
            quarterly_values = df_Y_for_detrend.values
            expanded = np.repeat(quarterly_values, 3, axis=0)
            # Trim to match monthly data length
            expanded = expanded[:len(df_X_for_detrend)]
            df_Y_monthly = pd.DataFrame(
                expanded, 
                columns=df_Y_for_detrend.columns
            )
        
        # Construct regressors for Kalman Filter
        logger.info("Constructing regressors for Kalman Filter")
        regressors = Regressors()
        
        try:
            regressors_dict, b_start = regressors.construct_regressors(
                df_X_detrended.values, 
                df_Y_monthly.values, 
                df_X_for_detrend.values,
                df_Y_monthly.values,
                debug=debug
            )
            logger.info(f"Successfully constructed regressors for {len(regressors_dict)} components")
        except Exception as e:
            logger.error(f"Error constructing regressors: {e}")
            logger.warning("Using simplified approach without regressors")
            # Create empty regressors dict
            regressors_dict = {}
            b_start = np.array([])
        
        # Identify components to interpolate
        components = []
        for col in df_Y_detrended.columns:
            if col not in date_cols_Y:
                components.append(col)
        
        if not components:
            raise ValueError("No components found for interpolation")
        
        logger.info(f"Components to interpolate: {components}")
        
        # Run Kalman Filter for each GDP component
        logger.info("Running Kalman Filter interpolation")
        results = []
        component_names = []
        
        # Process each component
        for i, component in enumerate(components):
            logger.info(f"Processing component {i+1}/{len(components)}: {component}")
            component_names.append(component)
            
            # For PCE, we may already have monthly data
            pce_monthly_cols = [col for col in df_X.columns if col.startswith('PCE')]
            if component == 'PCE' and pce_monthly_cols:
                logger.info(f"Using existing monthly data for {component}")
                monthly_component = df_X[pce_monthly_cols[0]].values
            else:
                # Try to find the corresponding regressors for this component
                if component in regressors_dict and len(regressors_dict[component]) > 0:
                    component_regressors = regressors_dict[component]
                    
                    # Find the correct index in b_start
                    component_idx = i
                    if b_start.size > 0:
                        if i >= b_start.shape[1]:
                            component_idx = 0  # Fallback
                        
                        component_b_start = b_start[:, component_idx]
                        component_b_start = component_b_start[~np.isnan(component_b_start)]
                    else:
                        # Create default starting values if none available
                        component_b_start = np.array([0, 1, 0.2, 0.3])
                    
                    # Optimize parameters and interpolate
                    logger.info(f"Optimizing parameters for {component}")
                    try:
                        # If we have Y_monthly data, use it
                        y_monthly_component = None
                        if component in df_Y_monthly.columns:
                            y_monthly_component = df_Y_monthly[component].values
                        
                        # Run Kalman optimization and interpolation
                        monthly_component = run_kalman_optimization(
                            df_Y_detrended[component].values,
                            component_regressors,
                            y_monthly_component,
                            df_Y_detrended[component].values,
                            component_b_start,
                            variable_names=[component]
                        )
                    except Exception as e:
                        logger.error(f"Error interpolating {component}: {e}")
                        logger.warning(f"Using simple expansion for {component}")
                        # Simple expansion - replicate each quarterly value for 3 months
                        quarterly_values = df_Y_detrended[component].values
                        monthly_component = np.repeat(quarterly_values, 3)[:len(df_X)]
                else:
                    logger.warning(f"No regressors found for {component}, using simple expansion")
                    # Simple expansion - replicate each quarterly value for 3 months
                    quarterly_values = df_Y_detrended[component].values
                    monthly_component = np.repeat(quarterly_values, 3)[:len(df_X)]
            
            results.append(monthly_component)
        
        # Create DataFrame with results
        results_array = np.column_stack(results)
        interpolated_df = pd.DataFrame(
            data=results_array,
            columns=component_names
        )
        
        # Add date columns
        for col in X_dates.columns:
            interpolated_df.insert(list(X_dates.columns).index(col), col, X_dates[col])
        
        # Calculate GDP (nominal and real)
        logger.info("Calculating GDP from components")
        
        # Identify imports column
        imports_col = next((col for col in component_names if col in ['IM', 'imports']), None)
        price_col = next((col for col in component_names if col in ['PGDP', 'GDPDEF']), None)
        
        # Calculate nominal GDP
        if imports_col:
            # Sum all components except imports (which are subtracted) and price index
            gdp_components = interpolated_df.drop(
                [col for col in interpolated_df.columns if col in date_cols_X + [imports_col, price_col]], 
                axis=1, 
                errors='ignore'
            )
            nominal_gdp = gdp_components.sum(axis=1) - 2 * interpolated_df[imports_col]
        else:
            # Sum all components except price index
            gdp_components = interpolated_df.drop(
                [col for col in interpolated_df.columns if col in date_cols_X + [price_col]], 
                axis=1, 
                errors='ignore'
            )
            nominal_gdp = gdp_components.sum(axis=1)
        
        # Calculate real GDP using price index
        if price_col:
            price_index = interpolated_df[price_col]
            real_gdp = nominal_gdp / price_index * 100
        else:
            # Without price index, nominal = real
            real_gdp = nominal_gdp
        
        # Add GDP to results
        interpolated_df['GDP_nominal'] = nominal_gdp
        interpolated_df['GDP_real'] = real_gdp
        
        # Save results to CSV
        logger.info(f"Saving interpolated data to {output_path}")
        interpolated_df.to_csv(output_path, index=False)
        
        # Clean up temporary files
        if temp_excel.exists():
            try:
                os.remove(temp_excel)
            except:
                pass
        
        return interpolated_df
        
    except Exception as e:
        logger.error(f"Error during interpolation: {e}", exc_info=True)
        raise


def main():
    """Simple entry point with no arguments needed."""
    try:
        interpolate_gdp_monthly(debug=False)
        logger.info("Interpolation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error during interpolation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 