import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
from contextlib import contextmanager


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    """Container for the results of the data transformation."""
    quarterly_data: Union[np.ndarray, pd.DataFrame]
    monthly_data: Union[np.ndarray, pd.DataFrame]
    variable_names: List[str]
    
    @property
    def shape(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return the shapes of quarterly and monthly data."""
        return self.quarterly_data.shape, self.monthly_data.shape
    
    def to_dataframes(self, quarterly_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert the arrays to pandas DataFrames."""
        if isinstance(self.quarterly_data, pd.DataFrame) and isinstance(self.monthly_data, pd.DataFrame):
            return self.quarterly_data, self.monthly_data
            
        df_quarterly = pd.DataFrame(self.quarterly_data, columns=quarterly_columns)
        df_monthly = pd.DataFrame(self.monthly_data, columns=self.variable_names)
        return df_quarterly, df_monthly


class DataManager:
    """
    Manages data for the Stock-Watson interpolation procedure.
    
    This class handles loading, transforming, and saving data for the
    interpolation of quarterly GDP to monthly values using the Stock-Watson approach.
    """
    
    def __init__(self, data_path: str = "data/raw_data.xlsx"):
        self.__raw_data = None
        self.data_path = Path(data_path)
        
        # Default Vector of Aggregation
        self.default_voa = [1, 4, 5, 0, 2, 4, 4, 7, 1]
    
    @property
    def raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from Excel files if not already loaded.
        
        Returns:
            Dict containing 'monthly' and 'quarterly' DataFrames
        """
        if self.__raw_data is None:
            try:
                self.__raw_data = {
                    "monthly": pd.read_excel(self.data_path, sheet_name="Monthly"),
                    "quarterly": pd.read_excel(self.data_path, sheet_name="Quarterly")
                }
                logger.info(f"Loaded data from {self.data_path}")
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                raise
        return self.__raw_data
    
    @contextmanager
    def debug_context(self, debug: bool = False):
        """Context manager for debug operations."""
        if debug:
            logger.setLevel(logging.DEBUG)
        try:
            yield
        finally:
            logger.setLevel(logging.INFO)
    
    def transform(self, voa: Optional[List[int]] = None, debug: bool = False) -> TransformationResult:
        """
        Transform raw data into variables needed for Kalman Filter interpolations.
        
        This is a Python implementation of the MATLAB transform.m function from the
        Stock-Watson procedure for temporal disaggregation.
        
        Args:
            voa: Vector of Aggregation - specifies how many monthly indices 
                correspond to each quarterly series. Default is [1, 4, 5, 0, 2, 4, 4, 7, 1].
            debug: If True, print debug information.
            
        Returns:
            TransformationResult containing quarterly data, monthly data, and variable names
        """
        with self.debug_context(debug):
            # Default Vector of Aggregation if not provided
            voa = voa or self.default_voa
                
            # Get the data as DataFrames
            qdata_df = self.raw_data["quarterly"]
            mdata_df = self.raw_data["monthly"]
            
            if debug:
                logger.debug(f"Monthly data shape: {mdata_df.shape}")
                logger.debug(f"Monthly data columns: {mdata_df.columns.tolist()}")
                logger.debug(f"First few rows of monthly data:\n{mdata_df.head()}")
            
            # Initialize output DataFrame for transformed monthly data
            transformed_monthly = pd.DataFrame(index=mdata_df.index)
            
            try:
                # Process each indicator and add to transformed_monthly
                transformed_monthly = self._process_all_indicators(mdata_df, transformed_monthly, voa, debug)
                
                # Replace NaN values with zeros
                transformed_monthly = transformed_monthly.fillna(0)
                
                if debug:
                    logger.debug(f"Transformation complete: {transformed_monthly.shape}")
                    logger.debug(f"Transformed columns: {transformed_monthly.columns.tolist()}")
                
            except Exception as e:
                logger.error(f"Error during transformation: {e}")
                if debug:
                    print(f"Error: {e}")
                raise
                
            return TransformationResult(
                qdata_df, 
                transformed_monthly, 
                transformed_monthly.columns.tolist()
            )
    
    def _process_all_indicators(self, mdata_df: pd.DataFrame, 
                                       result_df: pd.DataFrame, 
                                       voa: List[int], 
                                       debug: bool) -> pd.DataFrame:
        """Process all economic indicators using pandas operations."""
        # Process Personal Consumption Expenditures
        result_df = self._process_pce(mdata_df, result_df, voa, debug)
        
        # Process Nonresidential Structures
        result_df = self._process_nonresidential_structures(mdata_df, result_df)
        
        # Process Equipment and Software
        result_df = self._process_equipment_software(mdata_df, result_df, voa)
        
        # Process Residential Structures
        result_df = self._process_residential_structures(mdata_df, result_df)
        
        # Process Private Inventories
        result_df = self._process_private_inventories(mdata_df, result_df, voa)
        
        # Process Exports
        result_df = self._process_exports(mdata_df, result_df, voa)
        
        # Process Imports
        result_df = self._process_imports(mdata_df, result_df, voa)
        
        # Process Government
        result_df = self._process_government(mdata_df, result_df, voa)
        
        # Process Price Index
        result_df = self._process_price_index(mdata_df, result_df, voa, debug)
        
        return result_df
    
    def _get_indicator_columns(self, mdata_df: pd.DataFrame, index: int, voa: List[int], count: int = 1) -> List[str]:
        """Get column names for an indicator based on the Vector of Aggregation."""
        base_idx = 2 + sum(voa[:index])
        return mdata_df.columns[base_idx:base_idx + count].tolist()
    
    def _process_pce(self, mdata_df: pd.DataFrame, result_df: pd.DataFrame, 
                           voa: List[int], debug: bool) -> pd.DataFrame:
        """Process Personal Consumption Expenditures indicator using pandas."""
        if debug:
            logger.debug("Processing PCE indicator")
        
        # Get the column for PCE
        pce_col = self._get_indicator_columns(mdata_df, 0, voa)[0]
        
        # Add to result DataFrame
        result_df['PCE1'] = mdata_df[pce_col]
        
        return result_df
    
    def _process_nonresidential_structures(self, mdata_df: pd.DataFrame, 
                                                result_df: pd.DataFrame) -> pd.DataFrame:
        """Process Investment - Nonresidential Structures indicator using pandas."""
        # Using column names instead of indices
        if 'CONP' in mdata_df.columns and 'CONFR' in mdata_df.columns:
            result_df['I_NS1'] = (mdata_df['CONP'] - mdata_df['CONFR']) / 1000
        else:
            # Fallback to positional columns if named columns not found
            result_df['I_NS1'] = (mdata_df.iloc[:, 5] - mdata_df.iloc[:, 3]) / 1000
            
        if 'PRIV' in mdata_df.columns and 'RES' in mdata_df.columns:
            result_df['I_NS2'] = (mdata_df['PRIV'] - mdata_df['RES']) / 1000
        else:
            # Fallback to positional columns if named columns not found
            result_df['I_NS2'] = (mdata_df.iloc[:, 6] - mdata_df.iloc[:, 4]) / 1000
        
        return result_df
    
    def _process_equipment_software(self, mdata_df: pd.DataFrame, 
                                         result_df: pd.DataFrame, 
                                         voa: List[int]) -> pd.DataFrame:
        """Process Investment - Equipment and Software indicator using pandas."""
        # Get the columns for Equipment and Software
        cols = self._get_indicator_columns(mdata_df, 2, voa, count=5)
        
        # Process each column and add to result
        for i, col in enumerate(cols):
            result_df[f'I_ES{i+1}'] = 12 * mdata_df[col] / 1000
        
        return result_df
    
    def _process_residential_structures(self, mdata_df: pd.DataFrame, 
                                             result_df: pd.DataFrame) -> pd.DataFrame:
        """Process Investment in Residential Structures indicator using pandas."""
        if 'CONFR' in mdata_df.columns:
            result_df['I_RS1'] = mdata_df['CONFR'] / 1000
        else:
            # Fallback to positional columns if named columns not found
            result_df['I_RS1'] = mdata_df.iloc[:, 3] / 1000
            
        if 'RES' in mdata_df.columns:
            result_df['I_RS2'] = mdata_df['RES'] / 1000
        else:
            # Fallback to positional columns if named columns not found
            result_df['I_RS2'] = mdata_df.iloc[:, 4] / 1000
        
        return result_df
    
    def _process_private_inventories(self, mdata_df: pd.DataFrame, 
                                          result_df: pd.DataFrame, 
                                          voa: List[int]) -> pd.DataFrame:
        """Process Investment - Change in private inventories indicator using pandas."""
        # Get the columns for Private Inventories
        cols = self._get_indicator_columns(mdata_df, 4, voa, count=2)
        
        # First regressor is simple scaling
        result_df['I_chPI1'] = mdata_df[cols[0]] / 1000
        
        # Second regressor requires differencing
        xreg2 = mdata_df[cols[1]]
        
        # Create the differenced series
        # First value is 0.041, then use diff() for the rest
        xreg2_diff = pd.Series(0.041, index=[xreg2.index[0]])
        xreg2_diff = pd.concat([xreg2_diff, xreg2.diff().iloc[1:]])
        
        # Convert to annual rate
        result_df['I_chPI2'] = 12 * xreg2_diff
        
        return result_df
    
    def _process_exports(self, mdata_df: pd.DataFrame, 
                              result_df: pd.DataFrame, 
                              voa: List[int]) -> pd.DataFrame:
        """Process Exports indicator using pandas."""
        # Get the columns for Exports
        cols = self._get_indicator_columns(mdata_df, 5, voa, count=4)
        
        # Process each column and add to result
        for i, col in enumerate(cols):
            result_df[f'X{i+1}'] = 12 * mdata_df[col] / 1000
        
        return result_df
    
    def _process_imports(self, mdata_df: pd.DataFrame, 
                              result_df: pd.DataFrame, 
                              voa: List[int]) -> pd.DataFrame:
        """Process Imports indicator using pandas."""
        # Get the columns for Imports
        cols = self._get_indicator_columns(mdata_df, 6, voa, count=4)
        
        # Process each column and add to result
        for i, col in enumerate(cols):
            result_df[f'IM{i+1}'] = 12 * mdata_df[col] / 1000
        
        return result_df
    
    def _process_government(self, mdata_df: pd.DataFrame, 
                                 result_df: pd.DataFrame, 
                                 voa: List[int]) -> pd.DataFrame:
        """Process Government indicator using pandas."""
        # Get the columns for Government
        cols = self._get_indicator_columns(mdata_df, 7, voa, count=7)
        
        # First three regressors
        result_df['G1'] = mdata_df[cols[0]]  # Wages_G
        result_df['G2'] = mdata_df[cols[1]] / 1000  # conq/1000
        result_df['G3'] = mdata_df[cols[2]] / 1000  # con_gov/1000
        
        # Fourth and fifth regressors require subtraction
        result_df['G4'] = 12 * (mdata_df[cols[3]] - mdata_df[cols[4]]) / 1000  # man_ship_def_1
        result_df['G5'] = 12 * (mdata_df[cols[5]] - mdata_df[cols[6]]) / 1000  # man_ship_def_2
        
        return result_df
    
    def _process_price_index(self, mdata_df: pd.DataFrame, 
                                  result_df: pd.DataFrame, 
                                  voa: List[int], 
                                  debug: bool) -> pd.DataFrame:
        """Process Price index indicator using pandas."""
        try:
            # Get the column for Price Index
            cols = self._get_indicator_columns(mdata_df, 8, voa, count=1)
            result_df['nine'] = mdata_df[cols[0]]  # No log transformation
        except IndexError:
            if debug:
                logger.warning(f"Price index column not found, using placeholder")
            # Use a placeholder if the index is out of bounds
            result_df['nine_placeholder'] = 0
        
        return result_df
    
    def save_transformed_data(self, output_file: str = "data/transformed_data.xlsx") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform the data and save it to an Excel file.
        
        Args:
            output_file: Path to the output Excel file.
            
        Returns:
            Tuple containing:
                - df_Y: DataFrame of quarterly data
                - df_X: DataFrame of transformed monthly data
        """
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Transform the data
        result = self.transform()
        
        # Convert to DataFrames if needed
        df_Y, df_X = result.to_dataframes(self.raw_data["quarterly"].columns)
        
        # For debugging purposes
        logger.debug(f"Quarterly data columns before save: {df_Y.columns.tolist()}")
        logger.debug(f"Monthly data columns before save: {df_X.columns.tolist()}")
        
        # Add Year and Month columns from original monthly data if they don't exist
        monthly_data = self.raw_data["monthly"]
        
        # Make a copy to avoid modifying the original
        df_X_with_dates = df_X.copy()
        df_Y_with_dates = df_Y.copy()
        
        # Make sure Year and Quarter exist in quarterly data
        if 'Year' not in df_Y_with_dates.columns and 'Year' in self.raw_data["quarterly"].columns:
            df_Y_with_dates.insert(0, "Year", self.raw_data["quarterly"]["Year"])
        if 'Quarter' not in df_Y_with_dates.columns and 'Quarter' in self.raw_data["quarterly"].columns:
            loc = 1 if 'Year' in df_Y_with_dates.columns else 0
            df_Y_with_dates.insert(loc, "Quarter", self.raw_data["quarterly"]["Quarter"])
        
        # Add Year and Month columns to monthly data if they don't exist
        if 'Year' not in df_X_with_dates.columns and 'Year' in monthly_data.columns:
            df_X_with_dates.insert(0, "Year", monthly_data["Year"])
        if 'Month' not in df_X_with_dates.columns and 'Month' in monthly_data.columns:
            loc = 1 if 'Year' in df_X_with_dates.columns else 0
            df_X_with_dates.insert(loc, "Month", monthly_data["Month"])
        
        # Log the final column structures
        logger.info(f"Quarterly data columns for save: {df_Y_with_dates.columns.tolist()}")
        logger.info(f"Monthly data columns for save: {df_X_with_dates.columns.tolist()}")
        
        # Save to Excel
        try:
            with pd.ExcelWriter(output_path) as writer:
                df_Y_with_dates.to_excel(writer, sheet_name="Quarterly", index=False)
                df_X_with_dates.to_excel(writer, sheet_name="Monthly_Transformed", index=False)
                
            logger.info(f"Transformed data saved to {output_path}")
            print(f"Transformed data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {e}")
            raise
        
        return df_Y_with_dates, df_X_with_dates
    
    def load_data(self, file_path: Union[str, Path], sheet_name: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load data from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to load
            
        Returns:
            Tuple containing:
                - numpy array of data
                - list of column names
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df.values, df.columns.tolist()

    def transform_data(self, qdata: np.ndarray, mdata: np.ndarray, voa: List[int], 
                      qtxt: List[str], mtxt: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Transform raw data into variables needed for Kalman Filter interpolations.
        
        Args:
            qdata: Quarterly data array
            mdata: Monthly data array
            voa: Vector of Aggregation - specifies how many monthly indices
                correspond to each quarterly series
            qtxt: Column names for quarterly data
            mtxt: Column names for monthly data
                
        Returns:
            Tuple containing:
                - transformed quarterly data (Y)
                - transformed monthly data (X)
                - list of variable names
        """
        # Initialize DataManager with the data
        dm = DataManager()
        dm._raw_data = {
            "quarterly": pd.DataFrame(qdata, columns=qtxt),
            "monthly": pd.DataFrame(mdata, columns=mtxt)
        }
        
        # Transform the data
        result = dm.transform(voa=voa)
        
        return (
            result.quarterly_data.values,
            result.monthly_data.values,
            result.variable_names
        )
    
    
    
