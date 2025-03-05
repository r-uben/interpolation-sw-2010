#!/usr/bin/env python3
"""
Script to upload GDP-related data files to AWS S3.

This script uploads raw_data.xlsx and gdp.csv to the 'macroeconomic_data' bucket
in the 'usa/national_accounting/' folder with appropriate metadata.
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, Optional

from interpolation_sw_2010.aws_manager.bucket_manager import BucketManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BUCKET_NAME = "macroeconomic-data"
DESTINATION_FOLDER = "usa/national_accounting/"
DATA_FILES = {
    "raw_data": {
        "local_path": "data/raw_data.xlsx",
        "s3_path": f"{DESTINATION_FOLDER}raw_data.xlsx",
        "description": "Raw data for Stock-Watson (2010) GDP interpolation method"
    },
    "gdp": {
        "local_path": "data/gdp.csv",
        "s3_path": f"{DESTINATION_FOLDER}gdp.csv",
        "description": "Interpolated monthly GDP data using Stock-Watson (2010) method"
    }
}


def create_metadata_for_raw_data(filepath: str) -> Dict[str, str]:
    """
    Create metadata for raw_data.xlsx file.
    
    Args:
        filepath: Path to the raw data file
        
    Returns:
        Dict with metadata as simple key-value pairs
    """
    try:
        # Read the Excel file to extract basic information
        xls = pd.ExcelFile(filepath)
        sheet_names = xls.sheet_names
        
        # Try to read the first sheet to get column names and date ranges
        first_sheet = pd.read_excel(filepath, sheet_name=0)
        columns = list(first_sheet.columns)
        
        # Get date range if applicable
        min_year = max_year = None
        if 'Year' in first_sheet.columns:
            min_year = int(first_sheet['Year'].min()) if not pd.isna(first_sheet['Year'].min()) else None
            max_year = int(first_sheet['Year'].max()) if not pd.isna(first_sheet['Year'].max()) else None
        
        # Create flattened metadata
        metadata = {
            "filename": os.path.basename(filepath),
            "file_type": "Excel",
            "sheets": ", ".join(sheet_names[:5]) + (", ..." if len(sheet_names) > 5 else ""),
            "sheet_count": str(len(sheet_names)),
            "columns": ", ".join(columns[:5]) + (", ..." if len(columns) > 5 else ""),
            "column_count": str(len(columns)),
            "row_count": str(len(first_sheet)),
            "year_min": str(min_year) if min_year else "",
            "year_max": str(max_year) if max_year else "",
            "description": "Raw data for Stock-Watson (2010) GDP interpolation",
            "purpose": "Contains quarterly GDP components and monthly indicators for temporal disaggregation",
            "methodology": "Stock-Watson (2010) interpolation method",
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error creating metadata for {filepath}: {e}")
        return {
            "error": str(e),
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def create_metadata_for_gdp(filepath: str) -> Dict[str, str]:
    """
    Create metadata for gdp.csv file.
    
    Args:
        filepath: Path to the GDP CSV file
        
    Returns:
        Dict with metadata as simple key-value pairs
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        columns = list(df.columns)
        
        # Extract date range if possible
        start_date = end_date = None
        date_column = next((col for col in columns if 'date' in col.lower()), None)
        
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            start_date = df[date_column].min()
            end_date = df[date_column].max()
        
        # Get components (excluding date columns)
        components = [col for col in columns if col not in ['Year', 'Month', 'Date', 'date', date_column] if date_column]
        
        # Create flattened metadata
        metadata = {
            "filename": os.path.basename(filepath),
            "file_type": "CSV",
            "columns": ", ".join(columns[:5]) + (", ..." if len(columns) > 5 else ""),
            "column_count": str(len(columns)),
            "row_count": str(len(df)),
            "start_date": start_date.strftime("%Y-%m-%d") if start_date and not pd.isna(start_date) else "",
            "end_date": end_date.strftime("%Y-%m-%d") if end_date and not pd.isna(end_date) else "",
            "components": ", ".join(components[:5]) + (", ..." if len(components) > 5 else ""),
            "description": "Monthly interpolated GDP data",
            "methodology": "Stock-Watson (2010) interpolation method",
            "frequency": "Monthly",
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "interpolated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add simple data quality metrics
        missing_values_total = df.isna().sum().sum()
        metadata["missing_values"] = str(missing_values_total)
        metadata["completeness"] = f"{(1 - missing_values_total / (len(df) * len(columns))) * 100:.1f}%"
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error creating metadata for {filepath}: {e}")
        return {
            "error": str(e),
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def upload_file_with_metadata(
    bucket_manager: BucketManager,
    local_path: str,
    s3_path: str,
    metadata_function
) -> bool:
    """
    Upload a file to S3 with metadata.
    
    Args:
        bucket_manager: Initialized BucketManager
        local_path: Path to local file
        s3_path: Destination path in S3
        metadata_function: Function to create metadata for this file
        
    Returns:
        True if upload successful, False otherwise
    """
    try:
        file_path = Path(local_path)
        if not file_path.exists():
            logger.error(f"File not found: {local_path}")
            return False
        
        # Read file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Create metadata - now flattened as simple key-value pairs
        metadata = metadata_function(local_path)
        
        # No need to convert to JSON strings - use simple string values directly
        # AWS S3 metadata values must be strings
        string_metadata = {k: str(v) for k, v in metadata.items()}
        
        # Upload to S3
        logger.info(f"Uploading {local_path} to s3://{bucket_manager.bucket_name}/{s3_path}")
        bucket_manager.upload_file(file_content, s3_path, string_metadata)
        logger.info(f"Successfully uploaded {local_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error uploading {local_path}: {e}")
        return False


def main():
    """Main function to upload data files to AWS."""
    try:
        logger.info("Initializing AWS bucket manager")
        bucket_manager = BucketManager(BUCKET_NAME)
        
        # Upload raw_data.xlsx
        raw_data_info = DATA_FILES["raw_data"]
        raw_data_success = upload_file_with_metadata(
            bucket_manager,
            raw_data_info["local_path"],
            raw_data_info["s3_path"],
            create_metadata_for_raw_data
        )
        
        # Upload gdp.csv
        gdp_info = DATA_FILES["gdp"]
        gdp_success = upload_file_with_metadata(
            bucket_manager,
            gdp_info["local_path"],
            gdp_info["s3_path"],
            create_metadata_for_gdp
        )
        
        # Summary
        logger.info("Upload summary:")
        logger.info(f"  - raw_data.xlsx: {'Success' if raw_data_success else 'Failed'}")
        logger.info(f"  - gdp.csv: {'Success' if gdp_success else 'Failed'}")
        
    except Exception as e:
        logger.error(f"Error in upload process: {e}")


if __name__ == "__main__":
    main()
