#!/usr/bin/env python
"""
Stock-Watson Interpolation Data Transformation Script

This script loads raw economic data and transforms it according to the
Stock-Watson procedure for temporal disaggregation of quarterly GDP to monthly values.
"""

from interpolation_sw_2010.data_manager import DataManager, TransformationResult
from interpolation_sw_2010.spline_detrending import SplineDetrending
from interpolation_sw_2010.regressors import Regressors
from interpolation_sw_2010.visualization import Visualization
from interpolation_sw_2010.validation import (
    validate_dataframe,
    validate_transformation_result,
    DataValidationError
)
from interpolation_sw_2010.config import Config

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import argparse
import sys
from typing import Tuple, Dict, Any
from tqdm import tqdm


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('transform_data.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Transform economic data using Stock-Watson procedure'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input Excel file with Monthly and Quarterly sheets'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output Excel file'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug output'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualizations of the transformed data'
    )
    parser.add_argument(
        '--detrend', '-t',
        action='store_true',
        help='Apply cubic spline detrending to the data'
    )
    parser.add_argument(
        '--regressors', '-r',
        action='store_true',
        help='Construct regressors for Stock-Watson interpolation'
    )
    return parser.parse_args()


def configure_logging(debug: bool = False):
    """Configure logging based on debug flag."""
    config = Config()
    log_config = config.get('logging', {})
    
    level = logging.DEBUG if debug else getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('transform_data.log')
        ]
    )


def initialize_data_manager(args: argparse.Namespace) -> DataManager:
    """Initialize the DataManager with configuration."""
    config = Config()
    data_config = config.get('data', {})
    
    input_file = args.input or data_config.get('input_file')
    if not input_file:
        raise ValueError("Input file path not provided in arguments or configuration")
    
    return DataManager(data_path=input_file)


def process_data(data_manager: DataManager, debug: bool = False) -> TransformationResult:
    """Process the data using the Stock-Watson procedure."""
    logger.info("Starting data processing")
    
    # Load and validate raw data
    monthly_df = data_manager.raw_data['monthly']
    quarterly_df = data_manager.raw_data['quarterly']
    
    validate_dataframe(monthly_df, numeric_only=True)
    validate_dataframe(quarterly_df, numeric_only=True)
    
    # Transform the data
    result = data_manager.transform(debug=debug)
    validate_transformation_result(result)
    
    return result


def run_detrending_pipeline(
    df_Y: pd.DataFrame,
    df_X: pd.DataFrame,
    args: argparse.Namespace,
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the detrending pipeline."""
    logger.info("Starting detrending pipeline")
    
    # Create SplineDetrending instance
    spline_detrending = SplineDetrending()
    
    # Detrend the data
    Yreg, Xreg = spline_detrending.detrend_data(df_Y, df_X, debug=debug)
    
    # Save detrended data
    save_detrended_data(Yreg, Xreg, args.output)
    
    # Construct regressors if requested
    if args.regressors:
        # Expand quarterly data to monthly frequency
        df_Y_monthly = spline_detrending.expand_quarterly_to_monthly(df_Y, debug=debug)
        
        # Construct regressors
        regressors_dict, b_start = construct_regressors(
            Xreg.values,
            df_Y_monthly.values,
            df_X.values,
            df_Y_monthly.values,
            debug=debug
        )
        
        # Save regressors
        save_regressors(regressors_dict, b_start, args.output)
    
    return Yreg, Xreg


def save_detrended_data(df_Y: pd.DataFrame, df_X: pd.DataFrame, output_file: str):
    """Save detrended data to Excel file."""
    output_path = Path(output_file)
    detrended_path = output_path.parent / f"{output_path.stem}_detrended{output_path.suffix}"
    
    logger.info(f"Saving detrended data to {detrended_path}")
    
    try:
        with pd.ExcelWriter(detrended_path) as writer:
            df_Y.to_excel(writer, sheet_name="Quarterly_Detrended", index=False)
            df_X.to_excel(writer, sheet_name="Monthly_Detrended", index=False)
    except Exception as e:
        logger.error(f"Error saving detrended data: {e}")
        raise


def construct_regressors(
    Xreg: np.ndarray,
    Y_ext: np.ndarray,
    X_tr: np.ndarray,
    Y_m: np.ndarray,
    debug: bool = False
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Construct regressors for Stock-Watson interpolation."""
    logger.info("Constructing regressors")
    
    regressors = Regressors()
    regressors_dict, b_start = regressors.construct_regressors(
        Xreg, Y_ext, X_tr, Y_m, debug=debug
    )
    
    return regressors_dict, b_start


def save_regressors(regressors_dict: Dict[str, Any], b_start: np.ndarray, output_file: str):
    """Save regressors and starting values."""
    output_path = Path(output_file)
    regressors_path = output_path.parent / f"{output_path.stem}_regressors.npz"
    b_start_path = output_path.parent / f"{output_path.stem}_b_start.npy"
    
    logger.info(f"Saving regressors to {regressors_path}")
    
    # Convert dictionary to saveable format
    save_dict = {}
    for indicator, regressor_list in regressors_dict.items():
        for i, regressor in enumerate(regressor_list):
            if regressor is not None and len(regressor) > 0:
                save_dict[f"{indicator}_{i}"] = regressor
    
    np.savez(regressors_path, **save_dict)
    np.save(b_start_path, b_start)


def generate_visualizations(df_Y: pd.DataFrame, df_X: pd.DataFrame, output_dir: Path):
    """Generate visualizations using the Visualization class."""
    logger.info("Generating visualizations")
    
    viz = Visualization(output_dir)
    plots = viz.generate_all_plots(df_X)  # Generate plots for monthly data
    
    logger.info(f"Generated visualizations: {list(plots.keys())}")
    return plots


def main():
    """Main function to transform and save economic data."""
    try:
        # Parse arguments and configure
        args = parse_arguments()
        configure_logging(args.debug)
        
        logger.info("Starting Stock-Watson Interpolation Data Transformation")
        
        # Initialize components
        data_manager = initialize_data_manager(args)
        
        # Process data
        with tqdm(total=5, desc="Processing") as pbar:
            # Transform data
            result = process_data(data_manager, args.debug)
            pbar.update(1)
            
            # Save transformed data
            df_Y, df_X = data_manager.save_transformed_data(args.output)
            pbar.update(1)
            
            # Run detrending if requested
            if args.detrend:
                df_Y_reg, df_X_reg = run_detrending_pipeline(df_Y, df_X, args, args.debug)
                pbar.update(1)
            
            # Generate visualizations if requested
            if args.visualize:
                output_dir = Path(args.output).parent / "visualizations"
                plots = generate_visualizations(df_Y, df_X, output_dir)
                pbar.update(1)
            
            pbar.update(1)
        
        logger.info("Process completed successfully!")
        return 0
        
    except (DataValidationError, ValueError) as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())