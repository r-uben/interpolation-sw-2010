"""
Data validation module for the Stock-Watson interpolation procedure.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Base class for validation errors."""
    pass

class DataValidationError(ValidationError):
    """Raised when data validation fails."""
    pass

class ConfigValidationError(ValidationError):
    """Raised when configuration validation fails."""
    pass

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    allow_missing: bool = True,
    numeric_only: bool = True,
    frequency: Optional[str] = None,
    missing_threshold: float = 0.5
) -> None:
    """
    Validate a pandas DataFrame for required properties.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        allow_missing: Whether to allow missing values
        numeric_only: Whether to require all columns to be numeric
        frequency: Expected frequency of the time series data
        missing_threshold: Maximum allowed proportion of missing values per column (0-1)
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            raise DataValidationError("DataFrame is empty")
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise DataValidationError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        if not allow_missing:
            cols_with_missing = df.columns[df.isna().any()].tolist()
            if cols_with_missing:
                raise DataValidationError(f"Missing values found in columns: {cols_with_missing}")
        else:
            # Only check if missing values exceed threshold
            missing_ratios = df.isna().mean()
            cols_exceeding_threshold = missing_ratios[missing_ratios > missing_threshold].index.tolist()
            if cols_exceeding_threshold:
                logger.warning(
                    f"Columns with more than {missing_threshold*100}% missing values: {cols_exceeding_threshold}"
                )
        
        # Check numeric columns
        if numeric_only:
            non_numeric = df.select_dtypes(exclude=[np.number]).columns
            if not non_numeric.empty:
                # Log a warning instead of raising an error
                logger.warning(f"Non-numeric columns found: {non_numeric.tolist()}")
        
        # Check time series frequency if specified
        if frequency and isinstance(df.index, pd.DatetimeIndex):
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq != frequency:
                # Log a warning instead of raising an error
                logger.warning(
                    f"Invalid time series frequency. Expected {frequency}, got {inferred_freq}"
                )
        
        logger.debug(f"DataFrame validation passed: {df.shape}")
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Unexpected error during validation: {str(e)}")

def validate_transformation_result(result: Dict[str, Any]) -> None:
    """
    Validate the results of data transformation.
    
    Args:
        result: Dictionary containing transformation results
        
    Raises:
        DataValidationError: If validation fails
    """
    required_keys = ['quarterly_data', 'monthly_data', 'variable_names']
    
    try:
        # Check required keys
        missing_keys = set(required_keys) - set(result.keys())
        if missing_keys:
            raise DataValidationError(f"Missing required keys in transformation result: {missing_keys}")
        
        # Validate quarterly data
        if isinstance(result['quarterly_data'], pd.DataFrame):
            validate_dataframe(
                result['quarterly_data'],
                allow_missing=True,
                numeric_only=True,
                frequency='Q'
            )
        elif not isinstance(result['quarterly_data'], np.ndarray):
            raise DataValidationError("Quarterly data must be DataFrame or ndarray")
        
        # Validate monthly data
        if isinstance(result['monthly_data'], pd.DataFrame):
            validate_dataframe(
                result['monthly_data'],
                allow_missing=True,
                numeric_only=True,
                frequency='M'
            )
        elif not isinstance(result['monthly_data'], np.ndarray):
            raise DataValidationError("Monthly data must be DataFrame or ndarray")
        
        # Validate variable names
        if not isinstance(result['variable_names'], list):
            raise DataValidationError("Variable names must be a list")
        if not all(isinstance(name, str) for name in result['variable_names']):
            raise DataValidationError("All variable names must be strings")
        
        logger.debug("Transformation result validation passed")
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Unexpected error during validation: {str(e)}")

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ConfigValidationError: If validation fails
    """
    required_sections = ['data', 'visualization', 'logging']
    
    try:
        # Check required sections
        missing_sections = set(required_sections) - set(config.keys())
        if missing_sections:
            raise ConfigValidationError(f"Missing required config sections: {missing_sections}")
        
        # Validate data section
        if 'default_voa' not in config['data']:
            raise ConfigValidationError("Missing default_voa in data configuration")
        if not isinstance(config['data']['default_voa'], list):
            raise ConfigValidationError("default_voa must be a list")
        
        # Validate visualization section
        if 'matplotlib' not in config['visualization']:
            raise ConfigValidationError("Missing matplotlib configuration")
        if 'figure_sizes' not in config['visualization']:
            raise ConfigValidationError("Missing figure_sizes in visualization configuration")
        
        # Validate logging section
        if 'level' not in config['logging']:
            raise ConfigValidationError("Missing logging level configuration")
        if 'format' not in config['logging']:
            raise ConfigValidationError("Missing logging format configuration")
        
        logger.debug("Configuration validation passed")
        
    except ConfigValidationError:
        raise
    except Exception as e:
        raise ConfigValidationError(f"Unexpected error during config validation: {str(e)}") 