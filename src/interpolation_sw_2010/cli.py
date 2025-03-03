"""
Command-line interface for Stock-Watson interpolation procedure.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from .data_manager import DataManager, TransformationResult
from .spline_detrending import SplineDetrending
from .regressors import Regressors
from .visualization import Visualization
from .validation import (
    validate_dataframe,
    validate_transformation_result,
    validate_config,
    DataValidationError
)
from .config import Config


# Set up logging
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Transform economic data using Stock-Watson procedure'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input Excel file with Monthly and Quarterly sheets (default: data/raw_data.xlsx)',
        default=None
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output Excel file (default: data/transformed_data.xlsx)',
        default=None
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
        '--skip-detrend',
        action='store_true',
        help='Skip cubic spline detrending step'
    )
    parser.add_argument(
        '--skip-regressors',
        action='store_true',
        help='Skip constructing regressors for Stock-Watson interpolation'
    )
    return parser.parse_args()


def configure_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag."""
    config = Config()
    log_config = config.get('logging', {})
    
    level = logging.DEBUG if debug else getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Reset the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('transform_data.log')
        ]
    )
    
    # Force matplotlib logger to INFO level
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger('PIL').setLevel(logging.INFO)


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
    
    # Convert result to dict for validation
    result_dict = {
        'quarterly_data': result.quarterly_data,
        'monthly_data': result.monthly_data,
        'variable_names': result.variable_names
    }
    validate_transformation_result(result_dict)
    
    return result


def get_output_path(args: argparse.Namespace) -> str:
    """Get the output file path from args or config."""
    config = Config()
    data_config = config.get('data', {})
    
    output_file = args.output or data_config.get('output_file')
    if not output_file:
        output_file = 'data/transformed_data.xlsx'
        logger.warning(f"Output file path not provided, using default: {output_file}")
    
    # Make sure the directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    return str(output_path)


def run_detrending_pipeline(
    df_Y: pd.DataFrame,
    df_X: pd.DataFrame,
    args: argparse.Namespace,
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the detrending pipeline for Stock-Watson interpolation procedure."""
    logger.info("Starting detrending pipeline")
    
    # Print column info for debugging
    logger.info(f"Before processing - df_Y columns: {df_Y.columns.tolist()}")
    logger.info(f"Before processing - df_X columns: {df_X.columns.tolist()}")
    
    # Preserve date columns from both dataframes if they exist
    date_cols_Y = {}
    date_cols_X = {}
    
    for col in ['Year', 'Month']:
        # Check and save from df_Y
        if col in df_Y.columns:
            logger.info(f"Found and preserving column in df_Y: {col}")
            date_cols_Y[col] = df_Y[col].copy()
            df_Y = df_Y.drop(columns=[col])
            
        # Check and save from df_X
        if col in df_X.columns:
            logger.info(f"Found and preserving column in df_X: {col}")
            date_cols_X[col] = df_X[col].copy()
            df_X = df_X.drop(columns=[col])
    
    logger.info(f"After removing date columns - df_Y columns: {df_Y.columns.tolist()}")
    logger.info(f"After removing date columns - df_X columns: {df_X.columns.tolist()}")
    
    # Create SplineDetrending instance
    spline_detrending = SplineDetrending()
    
    # Detrend the data
    Yreg, Xreg = spline_detrending.detrend_data(df_Y, df_X, debug=debug)
    
    logger.info(f"After detrending - Yreg columns: {Yreg.columns.tolist()}")
    logger.info(f"After detrending - Xreg columns: {Xreg.columns.tolist()}")
    
    # Restore date columns to Yreg if they were preserved
    for col, data in date_cols_Y.items():
        if col in Yreg.columns:
            logger.warning(f"Column {col} already exists in Yreg! Using existing column.")
        else:
            if col == 'Year':
                Yreg.insert(0, col, data)
            elif col == 'Month':
                loc = 1 if 'Year' in Yreg.columns else 0
                Yreg.insert(loc, col, data)
    
    # Restore date columns to Xreg if they were preserved
    for col, data in date_cols_X.items():
        if col in Xreg.columns:
            logger.warning(f"Column {col} already exists in Xreg! Using existing column.")
        else:
            if col == 'Year':
                Xreg.insert(0, col, data)
            elif col == 'Month':
                loc = 1 if 'Year' in Xreg.columns else 0
                Xreg.insert(loc, col, data)
    
    logger.info(f"After restoring date columns - Yreg columns: {Yreg.columns.tolist()}")
    logger.info(f"After restoring date columns - Xreg columns: {Xreg.columns.tolist()}")
    
    # Save detrended data
    output_file = get_output_path(args)
    save_detrended_data(Yreg, Xreg, output_file)
    
    # Skip regressors if requested
    if args.skip_regressors:
        return Yreg, Xreg
    
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
    save_regressors(regressors_dict, b_start, output_file)
    
    return Yreg, Xreg


def save_detrended_data(df_Y: pd.DataFrame, df_X: pd.DataFrame, output_file: str) -> None:
    """Save detrended data to Excel file."""
    output_path = Path(output_file)
    detrended_path = output_path.parent / f"{output_path.stem}_detrended{output_path.suffix}"
    
    logger.info(f"Saving detrended data to {detrended_path}")
    
    # Log the column structure for debugging
    logger.info(f"Detrended quarterly data columns: {df_Y.columns.tolist()}")
    logger.info(f"Detrended monthly data columns: {df_X.columns.tolist()}")
    
    try:
        # No need to add Year and Month columns here as they should already be in df_X
        # from the run_detrending_pipeline function
        logger.info(f"Columns in df_X being saved: {df_X.columns.tolist()}")
        
        with pd.ExcelWriter(detrended_path) as writer:
            df_Y.to_excel(writer, sheet_name="Quarterly_Detrended", index=False)
            df_X.to_excel(writer, sheet_name="Monthly_Detrended", index=False)
            
        logger.info(f"Successfully saved detrended data to {detrended_path}")
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


def save_regressors(regressors_dict: Dict[str, Any], b_start: np.ndarray, output_file: str) -> None:
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


def generate_visualizations(df_Y: pd.DataFrame, df_X: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Generate visualizations using the Visualization class."""
    logger.info("Generating visualizations")
    
    viz = Visualization(output_dir)
    plots = viz.generate_all_plots(df_X)  # Generate plots for monthly data
    
    logger.info(f"Generated visualizations: {list(plots.keys())}")
    return plots


def calculate_gdp_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate GDP results from the monthly indicators.
    
    This follows the Stock-Watson procedure:
    1. GDP_nominal = sum(all_components) - 2*imports
    2. GDP_real = GDP_nominal / price_index * 100
    
    Args:
        results_df: DataFrame containing monthly indicators
        
    Returns:
        DataFrame with GDP results (GDP_nominal, GDP_real, Price_index)
    """
    logger.info("Calculating GDP results")
    
    # Copy date columns if they exist
    date_cols = {}
    for col in ['Year', 'Month']:
        if col in results_df.columns:
            date_cols[col] = results_df[col].copy()
    
    # Remove date columns for calculation
    calc_df = results_df.copy()
    for col in date_cols:
        if col in calc_df.columns:
            calc_df = calc_df.drop(columns=[col])
    
    # Log columns for debugging
    logger.info(f"Columns available for GDP calculation: {calc_df.columns.tolist()}")
    
    # Identify imports column (should be IM1, IM2, IM3, IM4)
    import_cols = [col for col in calc_df.columns if col.startswith('IM')]
    
    # Fallback: try other common import column naming patterns
    if not import_cols:
        import_fallbacks = ['imports', 'Imports', 'import']
        for prefix in import_fallbacks:
            import_cols = [col for col in calc_df.columns if prefix in col]
            if import_cols:
                logger.info(f"Using fallback import columns: {import_cols}")
                break
    
    # Identify price index column (should be PGDP1)
    price_index_col = [col for col in calc_df.columns if col.startswith('PGDP')]
    
    # Fallback: try other common price index column naming patterns
    if not price_index_col:
        price_fallbacks = ['price', 'Price', 'deflator', 'Deflator', 'GDPDEF']
        for prefix in price_fallbacks:
            price_index_col = [col for col in calc_df.columns if prefix in col]
            if price_index_col:
                logger.info(f"Using fallback price index column: {price_index_col}")
                break
    
    if not import_cols:
        logger.warning("No import columns found. GDP calculation may be incorrect.")
        imports = pd.Series(0, index=calc_df.index)
    else:
        # In Stock-Watson, imports are in column 7 of Results
        # We'll use all import columns
        imports = calc_df[import_cols].sum(axis=1)
        logger.info(f"Using import columns: {import_cols}")
    
    if not price_index_col:
        logger.warning("No price index column found. Using 1 for GDP_real calculation.")
        price_index = pd.Series(1, index=calc_df.index)
    else:
        price_index = calc_df[price_index_col[-1]]
        logger.info(f"Using price index column: {price_index_col[-1]}")
    
    # Calculate GDP_nominal = sum(all) - 2*imports
    # This follows the MATLAB approach: GDP_nominal=sum(Results(:,1:end-1),2)-2*Results(:,7)
    # All components are added, then imports subtracted twice (once because they're in the sum, once to exclude)
    gdp_nominal = calc_df.sum(axis=1) - 2 * imports
    
    # Calculate GDP_real = GDP_nominal / price_index * 100
    gdp_real = gdp_nominal / price_index * 100
    
    # Create results DataFrame
    gdp_results = pd.DataFrame({
        'GDP_nominal': gdp_nominal,
        'GDP_real': gdp_real,
        'Price_index': price_index
    })
    
    # Add date columns back
    for col, data in date_cols.items():
        if col == 'Year':
            gdp_results.insert(0, col, data)
        elif col == 'Month':
            loc = 1 if 'Year' in gdp_results.columns else 0
            gdp_results.insert(loc, col, data)
    
    logger.info("GDP results calculated successfully")
    return gdp_results


def save_gdp_results(results_df: pd.DataFrame, gdp_df: pd.DataFrame, output_file: str) -> None:
    """
    Save GDP results and monthly indicators to files.
    
    Args:
        results_df: DataFrame containing monthly indicators
        gdp_df: DataFrame containing GDP results
        output_file: Base path for output files
    """
    output_path = Path(output_file)
    indicators_path = output_path.parent / f"{output_path.stem}_monthly_indicators.xlsx"
    gdp_path = output_path.parent / f"{output_path.stem}_gdp_results.csv"
    
    logger.info(f"Saving monthly indicators to {indicators_path}")
    logger.info(f"Saving GDP results to {gdp_path}")
    
    try:
        # Save monthly indicators to Excel
        results_df.to_excel(indicators_path, index=False)
        
        # Save GDP results to CSV
        gdp_df.to_csv(gdp_path, index=False)
        
        logger.info("GDP results saved successfully")
        print(f"GDP results saved to {gdp_path}")
    except Exception as e:
        logger.error(f"Error saving GDP results: {e}")
        raise


def main() -> int:
    """Main function to transform and save economic data."""
    try:
        # Parse arguments and configure
        args = parse_arguments()
        configure_logging(args.debug)
        
        logger.info("Starting Stock-Watson Interpolation Data Transformation")
        
        # Initialize components
        data_manager = initialize_data_manager(args)
        
        # Process data
        with tqdm(total=6, desc="Processing") as pbar:
            # Transform data
            result = process_data(data_manager, args.debug)
            pbar.update(1)
            
            # Save transformed data
            output_path = get_output_path(args)
            df_Y, df_X = data_manager.save_transformed_data(output_path)
            pbar.update(1)
            
            # Run detrending by default unless skipped
            if not args.skip_detrend:
                df_Y_reg, df_X_reg = run_detrending_pipeline(df_Y, df_X, args, args.debug)
                pbar.update(1)
                
                # Calculate and save GDP results
                gdp_results = calculate_gdp_results(df_X_reg)
                save_gdp_results(df_X_reg, gdp_results, output_path)
                pbar.update(1)
            
            # Generate visualizations if requested
            if args.visualize:
                output_dir = Path(output_path).parent / "visualizations"
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