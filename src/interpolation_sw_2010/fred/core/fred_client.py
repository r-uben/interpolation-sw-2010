"""
Core FRED client implementation.
"""
import logging
import pandas as pd
from fredapi import Fred
from src.interpolation_sw_2010.aws_manager.s3 import S3

logger = logging.getLogger(__name__)

class FREDClient:
    """Client for interacting with FRED API."""
    
    def __init__(self):
        """Initialize FRED client with API key from AWS Secrets Manager."""
        try:
            # Get API key from AWS Secrets Manager
            api_key = S3.get_secret('FRED_API_KEY', key='api_key')
            
            # Handle different formats of the API key
            if isinstance(api_key, dict):
                api_key = api_key.get('api_key')
            elif isinstance(api_key, str):
                # If it's a string, try to parse it as JSON first
                try:
                    import json
                    key_dict = json.loads(api_key)
                    api_key = key_dict.get('api_key')
                except json.JSONDecodeError:
                    # If it's not JSON, use the string as is
                    pass
            
            if not api_key:
                raise ValueError("Could not retrieve valid FRED API key")
                
            self.client = Fred(api_key=api_key)
            logger.info("Successfully initialized FRED client")
            
        except Exception as e:
            logger.error(f"Failed to initialize FRED client: {str(e)}")
            raise
    
    def get_series(self, series_id: str, start_date=None, end_date=None) -> pd.DataFrame:
        """Fetch and standardize data series from FRED"""
        try:
            # Get the raw series
            series = self.client.get_series(series_id, start_date, end_date)
            if series is None:
                raise ValueError(f"No data found for series {series_id}")
            
            # Get series info
            info = self.client.get_series_info(series_id)
            if info is None:
                raise ValueError(f"No metadata found for series {series_id}")
            
            # Convert to DataFrame
            df = series.to_frame(name='value')
            df.index.name = 'date'
            
            # Get frequency
            frequency = info.get('frequency_short', 'M')  # Default to monthly if not specified
            
            # Standardize the index to end-of-period dates
            if frequency == 'Q':
                # For quarterly data, resample to end of quarter
                df = df.resample('QE-DEC').last()
                # Ensure it's end of quarter (Q4)
                df.index = df.index + pd.offsets.QuarterEnd(0)
            elif frequency == 'M':
                # For monthly data, ensure it's end of month
                df = df.resample('ME').last()
                df.index = df.index + pd.offsets.MonthEnd(0)
            
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort index in ascending order
            df = df.sort_index()
            
            # Add metadata as attributes
            df.attrs['series_id'] = series_id
            df.attrs['title'] = info.get('title', '')
            df.attrs['frequency'] = frequency
            df.attrs['units'] = info.get('units', '')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching series {series_id}: {str(e)}")
            raise
    
    def get_series_info(self, series_id: str) -> dict:
        """Get metadata information about a FRED series."""
        try:
            info = self.client.get_series_info(series_id)
            if info is None:
                raise ValueError(f"No metadata found for series {series_id}")
            return info
        except Exception as e:
            logger.error(f"Error fetching series info for {series_id}: {str(e)}")
            raise
    
    def search_series(self, search_text: str) -> pd.DataFrame:
        """Search for series in FRED"""
        try:
            results = self.client.search(search_text)
            if results is None:
                return pd.DataFrame()  # Return empty DataFrame if no results
            return results
        except Exception as e:
            logger.error(f"Error searching for series with query '{search_text}': {str(e)}")
            raise 