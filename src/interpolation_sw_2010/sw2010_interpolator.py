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
import matplotlib.pyplot as plt

# Core functionality imports
from .core.data_manager import DataManager, TransformationResult
from .core.spline_detrending import SplineDetrending
from .core.regressors import Regressors
from .core.kalman_filter import KalmanFilter
from .core.kalman_optimization import run_kalman_optimization

# Utilities import
from .utils.visualization import Visualization

# Set up simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class SW2010Interpolator:
    """
    Stock-Watson (2010) interpolation method for GDP components.
    
    This class implements the Stock-Watson (2010) methodology for interpolating
    quarterly GDP components to monthly frequency using related monthly indicators.
    """
    
    def __init__(self, data_path: str = "data/raw_data.xlsx"):
        """
        Initialize the SW2010Interpolator.
        
        Args:
            data_path: Path to the raw data file
        """
        self.data_path = data_path
        self.spline_detrending = SplineDetrending()
        self.regressors = Regressors()
        self.kalman_filter = KalmanFilter()
        
    def run_interpolation(self):
        """Run the Stock-Watson interpolation procedure."""
        # Load and transform data
        logger.info("Loading data from %s", self.data_path)
        df_q, df_m = self._load_and_transform_data()
        
        # Identify quarterly and monthly data columns for saving
        q_cols = [col for col in df_q.columns if col not in ['Year', 'Quarter']]
        m_cols = [col for col in df_m.columns if col not in ['Year', 'Month']]
        
        logger.info("Quarterly data shape: %s", df_q[q_cols].shape)
        logger.info("Monthly data shape: %s", df_m[m_cols].shape)
        
        # Detrend data with cubic splines
        logger.info("Detrending data with cubic splines")
        df_q_detrended, df_m_detrended, df_q_trend, df_m_trend = self.spline_detrending.detrend_data(df_q, df_m)
        
        # Expand quarterly data to monthly frequency
        logger.info("Expanding quarterly data to monthly frequency")
        df_X = self.spline_detrending.expand_quarterly_to_monthly(df_q, df_m)
        logger.info("Expanded shape: %s", df_X.shape)
        
        # Prepare regressors for Kalman filter
        logger.info("Constructing regressors for Kalman filter")
        
        # Extract the detrended data as numpy arrays
        y_reg = df_q_detrended.values
        x_reg = df_m_detrended.values
        y_m_trend = df_m_trend.values
        y_q_trend = df_q_trend.values
        
        # Prepare the regressors
        try:
            component_regressors, b_start = self.regressors.prepare(
                x_reg, y_reg, df_X.values, y_m_trend, debug=True
            )
            logger.info("Successfully constructed regressors for %d components", len(component_regressors))
        except Exception as e:
            logger.error("Error constructing regressors: %s", str(e))
            logger.warning("Falling back to cubic spline interpolation")
            return self._fallback_to_spline_interpolation(df_q, df_m, df_X)
        
        # Run Kalman filter interpolation for each component
        logger.info("Running Kalman filter interpolation")
        results = []
        
        # Define the components to interpolate
        components = [
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
        
        # Interpolate each component
        for i, component in enumerate(components):
            logger.info("Interpolating %s", component)
            
            # Get the regressors for this component
            if component in component_regressors:
                x = component_regressors[component]
                
                # Get the quarterly series for this component
                y_q = y_reg[:, i]
                
                # Get the starting values for this component
                b_start_component = b_start[:, i]
                
                try:
                    # Run the Kalman filter interpolation
                    smoothed_monthly, optimized_params, log_likelihood = self.kalman_filter.interpolate_and_smooth(
                        y_q, x, b_start_component, component, debug=True
                    )
                    
                    # Add back the trend
                    monthly_values = smoothed_monthly + y_m_trend[:, i]
                    
                    logger.info("Successfully interpolated %s with log-likelihood: %f", component, log_likelihood)
                    results.append(monthly_values)
                except Exception as e:
                    logger.error("Error interpolating %s: %s", component, str(e))
                    logger.warning("Falling back to cubic spline interpolation for %s", component)
                    
                    # Fallback to cubic spline interpolation for this component
                    monthly_values = self._interpolate_component_with_spline(df_q, df_m, component, i)
                    results.append(monthly_values)
            else:
                logger.warning("No regressors found for %s, using cubic spline interpolation", component)
                
                # Fallback to cubic spline interpolation for this component
                monthly_values = self._interpolate_component_with_spline(df_q, df_m, component, i)
                results.append(monthly_values)
        
        # Calculate GDP from components
        logger.info("Calculating GDP from components")
        gdp = self._calculate_gdp_from_components(results)
        results.append(gdp)
        
        # Create a DataFrame with the results
        logger.info("Creating results DataFrame")
        results_df = self._create_results_dataframe(df_m, results, components)
        
        # Save the results
        logger.info("Saving results to data/monthly_gdp.csv")
        results_df.to_csv("data/monthly_gdp.csv", index=False)
        
        # Create comparison plots
        logger.info("Creating comparison plots")
        self._create_comparison_plots(df_q, results_df, components)
        
        return results_df
    
    def _load_and_transform_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and transform data from the raw data file.
        
        Returns:
            Tuple containing:
                - DataFrame with quarterly data
                - DataFrame with monthly data
        """
        # Load data from separate sheets
        logger.info(f"Loading data from {self.data_path}")
        df_m = pd.read_excel(self.data_path, sheet_name='Monthly')
        df_q = pd.read_excel(self.data_path, sheet_name='Quarterly')
        
        # Ensure we have Year and Quarter columns for quarterly data
        if 'Year' not in df_q.columns:
            logger.warning("Year column not found in quarterly data, creating it")
            df_q['Year'] = np.floor(np.arange(len(df_q)) / 4) + 1959
        
        if 'Quarter' not in df_q.columns:
            logger.warning("Quarter column not found in quarterly data, creating it")
            df_q['Quarter'] = np.mod(np.arange(len(df_q)), 4) + 1
        
        # Ensure we have Year and Month columns for monthly data
        if 'Year' not in df_m.columns:
            logger.warning("Year column not found in monthly data, creating it")
            df_m['Year'] = np.floor(np.arange(len(df_m)) / 12) + 1959
        
        if 'Month' not in df_m.columns:
            logger.warning("Month column not found in monthly data, creating it")
            df_m['Month'] = np.mod(np.arange(len(df_m)), 12) + 1
        
        # Reorder columns to have Year and Quarter/Month first
        q_cols = [col for col in df_q.columns if col not in ['Year', 'Quarter']]
        df_q = df_q[['Year', 'Quarter'] + q_cols]
        
        m_cols = [col for col in df_m.columns if col not in ['Year', 'Month']]
        df_m = df_m[['Year', 'Month'] + m_cols]
        
        # Handle NaN values in the data
        # For quarterly data, forward fill NaN values
        df_q_numeric = df_q.select_dtypes(include=[np.number])
        df_q_non_numeric = df_q.select_dtypes(exclude=[np.number])
        df_q_numeric = df_q_numeric.ffill()
        df_q = pd.concat([df_q_non_numeric, df_q_numeric], axis=1)
        
        # For monthly data, forward fill NaN values
        df_m_numeric = df_m.select_dtypes(include=[np.number])
        df_m_non_numeric = df_m.select_dtypes(exclude=[np.number])
        df_m_numeric = df_m_numeric.ffill()
        df_m = pd.concat([df_m_non_numeric, df_m_numeric], axis=1)
        
        # Reorder columns again after concatenation
        df_q = df_q[['Year', 'Quarter'] + q_cols]
        df_m = df_m[['Year', 'Month'] + m_cols]
        
        # Log the shapes and column names
        logger.info(f"Quarterly data shape: {df_q.shape}")
        logger.info(f"Monthly data shape: {df_m.shape}")
        logger.info(f"Quarterly data columns: {df_q.columns.tolist()}")
        logger.info(f"Monthly data columns: {df_m.columns.tolist()}")
        
        return df_q, df_m
    
    def _fallback_to_spline_interpolation(self, df_q: pd.DataFrame, df_m: pd.DataFrame, df_X: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback to cubic spline interpolation when Kalman filter fails.
        
        Args:
            df_q: DataFrame with quarterly data
            df_m: DataFrame with monthly data
            df_X: DataFrame with expanded quarterly data
            
        Returns:
            DataFrame with interpolated monthly data
        """
        logger.info("Using cubic spline interpolation as fallback")
        
        # Define the components to interpolate
        components = [
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
        
        # Interpolate each component
        results = []
        for i, component in enumerate(components):
            logger.info("Interpolating %s with cubic spline", component)
            monthly_values = self._interpolate_component_with_spline(df_q, df_m, component, i)
            results.append(monthly_values)
        
        # Calculate GDP from components
        logger.info("Calculating GDP from components")
        gdp = self._calculate_gdp_from_components(results)
        results.append(gdp)
        
        # Create a DataFrame with the results
        logger.info("Creating results DataFrame")
        results_df = self._create_results_dataframe(df_m, results, components)
        
        # Save the results
        logger.info("Saving results to data/monthly_gdp.csv")
        results_df.to_csv("data/monthly_gdp.csv", index=False)
        
        # Create comparison plots
        logger.info("Creating comparison plots")
        self._create_comparison_plots(df_q, results_df, components)
        
        return results_df
    
    def _interpolate_component_with_spline(self, df_q: pd.DataFrame, df_m: pd.DataFrame, component: str, index: int) -> np.ndarray:
        """
        Interpolate a component using cubic spline.
        
        Args:
            df_q: DataFrame with quarterly data
            df_m: DataFrame with monthly data
            component: Name of the component to interpolate
            index: Index of the component in the quarterly data
            
        Returns:
            Numpy array with interpolated monthly values
        """
        logger.info("Interpolating %s with cubic spline", component)
        
        # Get the quarterly series for this component
        q_col = df_q.columns[index + 2]  # Skip Year and Quarter columns
        quarterly_series = df_q[q_col].values
        
        # Create a quarterly date index
        quarterly_dates = pd.to_datetime(df_q['Year'].astype(str) + 'Q' + df_q['Quarter'].astype(str))
        quarterly_series_pd = pd.Series(quarterly_series, index=quarterly_dates)
        
        # Create a monthly date range
        start_date = quarterly_dates.min()
        end_date = quarterly_dates.max() + pd.DateOffset(months=2)
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Reindex to monthly frequency and interpolate
        monthly_series = quarterly_series_pd.reindex(monthly_dates).interpolate(method='cubic')
        
        return monthly_series.values
    
    def _calculate_gdp_from_components(self, results: List[np.ndarray]) -> np.ndarray:
        """
        Calculate GDP from its components.
        
        Args:
            results: List of interpolated component series
            
        Returns:
            Numpy array with calculated GDP
        """
        # Extract components
        pce = results[0]
        i_ns = results[1]
        i_es = results[2]
        i_rs = results[3]
        i_chpi = results[4]
        x = results[5]
        im = results[6]
        g = results[7]
        
        # Calculate GDP
        gdp = pce + i_ns + i_es + i_rs + i_chpi + x - im + g
        
        return gdp
    
    def _create_results_dataframe(self, df_m: pd.DataFrame, results: List[np.ndarray], components: List[str]) -> pd.DataFrame:
        """
        Create a DataFrame with the interpolation results.
        
        Args:
            df_m: DataFrame with monthly data
            results: List of interpolated component series
            components: List of component names
            
        Returns:
            DataFrame with interpolated monthly data
        """
        # Create a date index
        dates = pd.to_datetime(df_m['Year'].astype(str) + '-' + df_m['Month'].astype(str) + '-01')
        
        # Create a DataFrame with the results
        results_df = pd.DataFrame({
            'Date': dates
        })
        
        # Add components
        for i, component in enumerate(components):
            results_df[component] = results[i]
        
        # Add GDP
        results_df['GDP'] = results[-1]
        
        return results_df
    
    def _create_comparison_plots(self, df_q: pd.DataFrame, results_df: pd.DataFrame, components: List[str]):
        """
        Create comparison plots of quarterly and monthly series.
        
        Args:
            df_q: DataFrame with quarterly data
            results_df: DataFrame with interpolated monthly data
            components: List of component names
        """
        # Create output directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)
        
        # Create quarterly date index
        quarterly_dates = pd.to_datetime(df_q['Year'].astype(str) + 'Q' + df_q['Quarter'].astype(str))
        
        # Create monthly date index
        monthly_dates = results_df['Date']
        
        # Create comparison plots for each component
        for i, component in enumerate(components):
            plt.figure(figsize=(12, 6))
            
            # Get quarterly series
            q_col = df_q.columns[i + 2]  # Skip Year and Quarter columns
            quarterly_series = df_q[q_col].values
            
            # Get monthly series
            monthly_series = results_df[component].values
            
            # Plot quarterly series
            plt.plot(quarterly_dates, quarterly_series, 'o-', label='Quarterly')
            
            # Plot monthly series
            plt.plot(monthly_dates, monthly_series, '-', label='Monthly (Interpolated)')
            
            # Set title and labels
            plt.title(f'Comparison of Quarterly and Monthly {component}')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plt.savefig(f'figures/comparison_{component}.png')
            plt.close()
        
        # Create comparison plot for GDP
        plt.figure(figsize=(12, 6))
        
        # Get quarterly GDP
        quarterly_gdp = df_q['GDP'].values if 'GDP' in df_q.columns else None
        
        # Get monthly GDP
        monthly_gdp = results_df['GDP'].values
        
        # Plot quarterly GDP if available
        if quarterly_gdp is not None:
            plt.plot(quarterly_dates, quarterly_gdp, 'o-', label='Quarterly')
        
        # Plot monthly GDP
        plt.plot(monthly_dates, monthly_gdp, '-', label='Monthly (Interpolated)')
        
        # Set title and labels
        plt.title('Comparison of Quarterly and Monthly GDP')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig('figures/comparison_GDP.png')
        plt.close()
        
        logger.info("Created %d comparison plots in the figures directory", len(components) + 1)


def main():
    """Simple entry point with no arguments needed."""
    try:
        interpolator = SW2010Interpolator()
        interpolator.run_interpolation()
        logger.info("Interpolation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error during interpolation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 