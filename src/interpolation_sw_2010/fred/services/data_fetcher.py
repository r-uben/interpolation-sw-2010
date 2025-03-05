"""
Service for fetching data from FRED.
"""
import logging
from typing import Optional
import pandas as pd
import json
from datetime import datetime
from interpolation_sw_2010.fred.core.fred_client import FREDClient
from interpolation_sw_2010.aws_manager.bucket_manager import BucketManager
from pathlib import Path

logger = logging.getLogger(__name__)

class DataFetcher:
    """Service for fetching and managing FRED data."""
    
    # Common FRED series IDs for frequently requested variables
    COMMON_SERIES = {
        'real gdp': 'GDPC1',
        'nominal gdp': 'GDP',
        'unemployment rate': 'UNRATE',
        'cpi': 'CPIAUCSL',
        'core cpi': 'CPILFESL',
        'industrial production': 'INDPRO',
        'federal funds rate': 'FEDFUNDS',
        'consumer price index': 'CPIAUCSL',
        'core inflation': 'CPILFESL',
    }
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.client = FREDClient()
        self.bucket_manager = BucketManager(bucket_name="macroeconomic-data")
    
    def _find_series_id(self, query: str) -> str:
        """Find the appropriate FRED series ID for a query."""
        try:
            # Clean the query
            query = query.lower().strip()
            
            # If query is already a FRED series ID, use it
            if query.isupper() and not ' ' in query:
                logger.info(f"Using provided series ID: {query}")
                return query
            
            # Check if it's a common series
            if query in self.COMMON_SERIES:
                series_id = self.COMMON_SERIES[query]
                logger.info(f"Found common series ID for '{query}': {series_id}")
                return series_id
                
            # Check if any common series contains the query
            for key, series_id in self.COMMON_SERIES.items():
                if query in key:
                    logger.info(f"Found matching common series ID for '{query}': {series_id} ({key})")
                    return series_id
            
            # Search FRED database
            logger.info(f"Searching FRED database for '{query}'...")
            search_results = self.client.search_series(query)
            if not search_results.empty:
                series_id = search_results.index[0]
                logger.info(f"Found series ID in FRED database: {series_id}")
                return series_id
                
            raise ValueError(f"Could not find a matching FRED series for query: {query}")
            
        except Exception as e:
            logger.error(f"Error finding series ID for query '{query}': {str(e)}")
            raise
    
    def _save_to_s3(self, data: pd.DataFrame, series_id: str, series_info: dict):
        """Save data and metadata to S3."""
        try:
            # Create metadata
            metadata = {
                'series_id': series_id,
                'title': series_info.get('title', ''),
                'units': series_info.get('units', ''),
                'frequency': series_info.get('frequency', ''),
                'last_updated': datetime.now().isoformat(),
                'source': 'Federal Reserve Economic Data (FRED)',
                'observation_start': data.index.min().isoformat(),
                'observation_end': data.index.max().isoformat(),
                'notes': series_info.get('notes', '')
            }
            
            # Save data
            data_path = f"fred/{series_id}/data.csv"
            self.bucket_manager.upload_file(
                file_content=data.to_csv().encode(),
                file_path=data_path,
                metadata={'content_type': 'text/csv'}
            )
            
            # Save metadata
            metadata_path = f"fred/{series_id}/metadata.json"
            self.bucket_manager.upload_file(
                file_content=json.dumps(metadata, indent=2).encode(),
                file_path=metadata_path,
                metadata={'content_type': 'application/json'}
            )
            
            logger.info(f"Successfully saved {series_id} data and metadata to S3")
            
        except Exception as e:
            logger.error(f"Error saving {series_id} to S3: {str(e)}")
            raise
    
    def _save_locally(self, data: pd.DataFrame, series_id: str, metadata: dict):
        """Save data and metadata locally."""
        try:
            # Create base directory
            base_dir = Path('data/fred') / series_id
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Save data
            data_path = base_dir / "data.csv"
            data.to_csv(data_path)
            
            # Save metadata
            metadata_path = base_dir / "metadata.txt"
            with metadata_path.open('w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                    
            logger.info(f"Successfully saved {series_id} data locally")
            
        except Exception as e:
            logger.error(f"Error saving {series_id} locally: {str(e)}")
            raise

    def get_series(self, query: str) -> pd.DataFrame:
        """Get time series data for a given query."""
        try:
            # Find the appropriate series ID
            series_id = self._find_series_id(query)
            logger.info(f"Fetching data for series {series_id}...")
            
            # Fetch the data and series info
            data = self.client.get_series(series_id)
            if data is None or data.empty:
                raise ValueError(f"No data found for series {series_id}")
                
            series_info = self.client.get_series_info(series_id)
            if series_info is None:
                raise ValueError(f"No metadata found for series {series_id}")
            
            # Log data info
            logger.info(f"Retrieved {len(data)} observations for {series_id}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"Latest value: {data['value'].iloc[-1]:.2f}")
            
            # Create metadata
            metadata = {
                'series_id': series_id,
                'title': series_info.get('title', ''),
                'units': series_info.get('units', ''),
                'frequency': series_info.get('frequency', ''),
                'last_updated': datetime.now().isoformat(),
                'source': 'Federal Reserve Economic Data (FRED)',
                'observation_start': data.index.min().isoformat(),
                'observation_end': data.index.max().isoformat(),
                'notes': series_info.get('notes', '')
            }
            
            # Save to S3
            self._save_to_s3(data, series_id, series_info)
            
            # Save locally
            self._save_locally(data, series_id, metadata)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for query '{query}': {str(e)}")
            raise 