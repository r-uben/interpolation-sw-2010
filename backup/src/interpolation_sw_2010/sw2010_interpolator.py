#!/usr/bin/env python
"""
Stock-Watson (2010) GDP Interpolation

This script implements the Stock-Watson (2010) interpolation procedure
for temporal disaggregation of quarterly GDP data to monthly frequency.

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

# Core functionality imports
from .data_manager import DataManager, TransformationResult
from .core.spline_detrending import SplineDetrending
from .core.regressors import Regressors
from .core.kalman_filter import KalmanFilter
from .core.kalman_optimization import run_kalman_optimization

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
        logger.info(f"Transforming data for interpolation")
        
        # Get transformed data from result
        df_Y = result.quarterly_data
        df_X = result.monthly_data
        date_cols_Y = ['Year', 'Quarter']
        date_cols_X = ['Year', 'Month']
        
        # Save for debugging if needed
        with pd.ExcelWriter(temp_excel) as writer:
            df_Y.to_excel(writer, sheet_name='Quarterly', index=False)
            df_X.to_excel(writer, sheet_name='Monthly', index=False)
        
        # Print DataFrame shapes and columns for debug
        logger.info(f"Quarterly data columns for save: {list(df_Y.columns)}")
        logger.info(f"Monthly data columns for save: {list(df_X.columns)}")
        print(f"Transformed data saved to {temp_excel}")
        
        # Detrend data with cubic splines
        # Step 1: Initialize components
        logger.info(f"Quarterly data shape: {df_Y.shape}, Monthly data shape: {df_X.shape}")
        logger.info(f"Quarterly data columns: {list(df_Y.columns)}")
        logger.info(f"Monthly data columns: {list(df_X.columns)}")
        
        # Step 2: Detrend data using cubic splines
        logger.info("Detrending data with cubic splines")
        spline_detrending = SplineDetrending()
        
        # Setup variables to store detrended data
        Y_dates = df_Y[date_cols_Y].copy()
        X_dates = df_X[date_cols_X].copy()
        
        # Detrend quarterly and monthly data
        quarterly_series = df_Y.drop(date_cols_Y, axis=1)
        monthly_series = df_X.drop(date_cols_X, axis=1)
        
        series_mapping = {
            'PCE': ['PCE1'],
            'I_NS': ['I_NS1', 'I_NS2'],
            'I_ES': ['I_ES1', 'I_ES2', 'I_ES3', 'I_ES4', 'I_ES5'],
            'I_RS': ['I_RS1', 'I_RS2'],
            'I_chPI': ['I_chPI1', 'I_chPI2'],
            'X': ['X1', 'X2', 'X3', 'X4'],
            'IM': ['IM1', 'IM2', 'IM3', 'IM4'],
            'G': ['G1', 'G2', 'G3', 'G4', 'G5'],
            'PGDP': ['nine']
        }
        
        quarterly_cols = {
            'PCE': 'PCEC',
            'I_NS': 'NR_STRUC_Q',
            'I_ES': 'equip_q',
            'I_RS': 'RES_Q',
            'I_chPI': 'INVTCHANGE_Q',
            'X': 'exports_q',
            'IM': 'imports_q',
            'G': 'GOV_Q',
            'PGDP': 'GDPDEF'
        }
        
        # Detrend each series
        df_Y_detrended = pd.DataFrame()
        df_X_detrended = pd.DataFrame()
        df_Y_monthly = pd.DataFrame()
        
        for key, monthly_cols in series_mapping.items():
            quarterly_col = quarterly_cols.get(key)
            if quarterly_col and quarterly_col in quarterly_series.columns:
                # Detrend quarterly series
                quarterly_data = quarterly_series[quarterly_col].values
                
                # Only find monthly columns that exist in the data
                existing_monthly_cols = [col for col in monthly_cols if col in monthly_series.columns]
                
                if not existing_monthly_cols:
                    logger.warning(f"Could not find matching column for series key: {key}")
                    continue
                
                # Detrend each series
                try:
                    y_quarterly_detrended, y_monthly_detrended = spline_detrending.detrend_quarterly_and_monthly(
                        quarterly_data, nknots=5
                    )
                    df_Y_detrended[key] = y_quarterly_detrended
                    df_Y_monthly[key] = y_monthly_detrended
                    
                    # Detrend each monthly regressor
                    for col in existing_monthly_cols:
                        if col in monthly_series.columns:
                            x_detrended = spline_detrending.detrend_monthly(
                                monthly_series[col].values, nknots=4
                            )
                            df_X_detrended[col] = x_detrended
                except Exception as e:
                    logger.error(f"Error detrending {key}: {e}")
        
        # Expand quarterly data to monthly frequency
        logger.info("Expanding quarterly data to monthly frequency")
        df_Y_expanded = pd.DataFrame()
        
        # Expand each quarterly series to monthly frequency
        for col in df_Y_detrended.columns:
            expanded_values = np.repeat(df_Y_detrended[col].values, 3)
            # Ensure the length matches the monthly data
            df_Y_expanded[col] = expanded_values[:len(df_X)]
        
        # Print expanded shape
        logger.info(f"Expanded quarterly data shape: {df_Y_expanded.shape}")
        
        # Construct regressors for Kalman Filter
        logger.info("Constructing regressors for Kalman Filter")
        regressors = Regressors()
        
        kalman_regressors = {}
        component_b_start = {}
        
        # Construct regressors for each component
        component_names = list(df_Y_detrended.columns)
        
        for i, component in enumerate(component_names):
            # Find matching monthly indicators
            matching_cols = []
            for col in df_X_detrended.columns:
                # Extract the prefix before any numbers
                prefix = ''.join([c for c in col if not c.isdigit()])
                if prefix.strip('_') in component or component in prefix:
                    matching_cols.append(col)
            
            if matching_cols:
                # Construct regressors
                regressors_data = df_X_detrended[matching_cols].values
                y_expanded = df_Y_expanded[component].values
                
                try:
                    component_regressors, b_start = regressors.prepare(
                        regressors_data, y_expanded, df_X_detrended.values
                    )
                    kalman_regressors[component] = component_regressors
                    component_b_start[component] = b_start
                except Exception as e:
                    logger.error(f"Error constructing regressors for {component}: {e}")
                    kalman_regressors[component] = None
                    component_b_start[component] = None
        
        # Check if we have regressors
        if kalman_regressors:
            logger.info(f"Successfully constructed regressors for {len(kalman_regressors)} components")
        else:
            logger.warning("No regressors could be constructed")
        
        # Run Kalman Filter interpolation
        logger.info("Running Kalman Filter interpolation")
        
        # Use Kalman Filter to interpolate each component
        component_names = list(df_Y_detrended.columns)
        results = []
        
        logger.info(f"Components to interpolate: {component_names}")
        
        for i, component in enumerate(component_names):
            logger.info(f"Processing component {i+1}/{len(component_names)}: {component}")
            
            # Check if we have monthly data for this component (like PCE)
            if component in df_Y_monthly.columns:
                logger.info(f"Using existing monthly data for {component}")
                monthly_component = df_Y_monthly[component]
                results.append(monthly_component)
                continue
                
            # Get regressors for this component
            component_regressors = kalman_regressors.get(component)
            component_b_start = component_b_start.get(component)
            
            if component_regressors is not None and component_b_start is not None:
                logger.info(f"Optimizing parameters for {component}")
                
                # Get monthly values for this component if available
                df_X_monthly = df_X_detrended
                
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