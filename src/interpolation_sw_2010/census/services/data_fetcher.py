"""
Service for fetching data from Census Bureau.
"""
import logging
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any, List
from pathlib import Path
from ..core.census_client import CensusClient
from ...aws_manager.bucket_manager import BucketManager

logger = logging.getLogger(__name__)

class DataFetcher:
    """Service for fetching and managing Census Bureau data."""
    
    # Mapping of Stock-Watson components to Census Bureau data sources
    SW_DATA_SOURCES = {
        # NonResidential Structures: CONSTPRIVUS - CONSTRESUS
        'nonresidential_structures': {
            'type': 'construction',
            'series': {
                'private': 'CONSTPRIVUS',  # Private construction
                'residential': 'CONSTRESUS'  # Residential construction
            },
            'operation': 'subtract',  # Private - Residential = NonResidential
            'description': 'Non-Residential Structures Investment'
        },
        # Residential Structures: CONSTRESUS
        'residential_structures': {
            'type': 'construction',
            'series': {'residential': 'CONSTRESUS'},  # Direct use
            'description': 'Residential Structures Investment'
        },
        # Equipment and Software: MFGNONDFCS + MFGCEPS
        'equipment_software': {
            'type': 'manufacturing',
            'series': {
                'nondurable': 'MFGNONDFCS',  # Non-durable manufacturing
                'durable': 'MFGCEPS'  # Durable manufacturing
            },
            'operation': 'add',  # Nondurable + Durable
            'description': 'Equipment and Software Investment'
        },
        # Change in Private Inventories: First difference of MFGINVT
        'inventories': {
            'type': 'manufacturing',
            'series': {'inventories': 'MFGINVT'},  # Manufacturing inventories
            'operation': 'difference',  # First difference
            'description': 'Change in Private Inventories'
        },
        # Government: CONSTPUBUS (construction)
        'government_construction': {
            'type': 'construction',
            'series': {'public': 'CONSTPUBUS'},  # Public construction
            'description': 'Government Construction Expenditures'
        }
    }
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.client = CensusClient()
        self.bucket_manager = BucketManager(bucket_name="macroeconomic-data")
    
    def _save_to_s3(self, data: pd.DataFrame, component: str, metadata: dict):
        """Save data and metadata to S3."""
        # Create metadata
        full_metadata = {
            'component': component,
            'title': metadata.get('title', ''),
            'source_series': metadata.get('source_series', ''),
            'frequency': 'M',  # Census data is monthly
            'last_updated': datetime.now().isoformat(),
            'source': 'Census Bureau',
            'observation_start': data.index.min().isoformat() if not data.empty else '',
            'observation_end': data.index.max().isoformat() if not data.empty else '',
            'notes': metadata.get('notes', '')
        }
        
        # Save data
        data_path = f"census/{component}/data.csv"
        self.bucket_manager.upload_file(
            file_content=data.to_csv().encode(),
            file_path=data_path,
            metadata={'content_type': 'text/csv'}
        )
        
        # Save metadata
        metadata_path = f"census/{component}/metadata.json"
        self.bucket_manager.upload_file(
            file_content=json.dumps(full_metadata, indent=2).encode(),
            file_path=metadata_path,
            metadata={'content_type': 'application/json'}
        )
    
    def _save_locally(self, data: pd.DataFrame, component: str, metadata: dict):
        """Save data and metadata locally."""
        try:
            # Create base directory
            base_dir = Path('data/census') / component
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Save data
            data_path = base_dir / "data.csv"
            data.to_csv(data_path)
            
            # Save metadata
            metadata_path = base_dir / "metadata.txt"
            with metadata_path.open('w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                    
            logger.info(f"Successfully saved {component} data locally")
            
        except Exception as e:
            logger.error(f"Error saving {component} locally: {str(e)}")
            raise

    def _fetch_single_series(self, series_type: str, series_id: str, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Fetch a single data series from Census Bureau.
        
        Args:
            series_type: Type of series ('construction', 'manufacturing', etc.)
            series_id: Series ID
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            DataFrame with the series data
        """
        if series_type == 'construction':
            return self.client.get_construction_data(series_id, start_year, end_year)
        elif series_type == 'manufacturing':
            return self.client.get_manufacturing_data(series_id, start_year, end_year)
        elif series_type == 'economic_indicators':
            return self.client.get_economic_indicators(series_id, start_year, end_year)
        else:
            raise ValueError(f"Unknown series type: {series_type}")

    def get_component_data(self, component: str, start_year: int = 2000, end_year: int = None) -> pd.DataFrame:
        """
        Get data for a specific Stock-Watson component.
        
        Args:
            component: One of the components defined in SW_DATA_SOURCES
            start_year: Start year for the data (default: 2000)
            end_year: End year for the data (default: current year)
            
        Returns:
            DataFrame with the component data
        """
        # Set default end_year to current year if not specified
        if end_year is None:
            end_year = datetime.now().year
            
        # Check if the component exists
        if component not in self.SW_DATA_SOURCES:
            raise ValueError(f"Unknown component: {component}. Valid components are: {list(self.SW_DATA_SOURCES.keys())}")
        
        # Get component configuration
        config = self.SW_DATA_SOURCES[component]
        series_type = config['type']
        
        try:
            # Different handling based on the operation type
            if 'operation' in config and config['operation'] == 'subtract':
                # Handle subtraction operation (e.g., nonresidential_structures)
                first_key = list(config['series'].keys())[0]
                second_key = list(config['series'].keys())[1]
                
                first_df = self._fetch_single_series(
                    series_type, 
                    config['series'][first_key], 
                    start_year, 
                    end_year
                )
                
                second_df = self._fetch_single_series(
                    series_type, 
                    config['series'][second_key], 
                    start_year, 
                    end_year
                )
                
                # Join the dataframes
                combined_df = pd.merge(
                    first_df[['value']], 
                    second_df[['value']], 
                    left_index=True, 
                    right_index=True,
                    suffixes=(f'_{first_key}', f'_{second_key}')
                )
                
                # Calculate the difference
                combined_df['value'] = combined_df[f'value_{first_key}'] - combined_df[f'value_{second_key}']
                
                # Final dataframe with just the value column
                df = combined_df[['value']]
                
            elif 'operation' in config and config['operation'] == 'add':
                # Handle addition operation (e.g., equipment_software)
                first_key = list(config['series'].keys())[0]
                second_key = list(config['series'].keys())[1]
                
                first_df = self._fetch_single_series(
                    series_type, 
                    config['series'][first_key], 
                    start_year, 
                    end_year
                )
                
                second_df = self._fetch_single_series(
                    series_type, 
                    config['series'][second_key], 
                    start_year, 
                    end_year
                )
                
                # Join the dataframes
                combined_df = pd.merge(
                    first_df[['value']], 
                    second_df[['value']], 
                    left_index=True, 
                    right_index=True,
                    suffixes=(f'_{first_key}', f'_{second_key}')
                )
                
                # Calculate the sum
                combined_df['value'] = combined_df[f'value_{first_key}'] + combined_df[f'value_{second_key}']
                
                # Final dataframe with just the value column
                df = combined_df[['value']]
                
            elif 'operation' in config and config['operation'] == 'difference':
                # Handle first difference operation (e.g., inventories)
                series_key = list(config['series'].keys())[0]
                
                original_df = self._fetch_single_series(
                    series_type, 
                    config['series'][series_key], 
                    start_year, 
                    end_year
                )
                
                # Calculate first difference
                df = original_df.copy()
                df['value'] = df['value'].diff()
                
                # Drop the first row (NaN due to diff)
                df = df.dropna()
                
            else:
                # Handle direct use of a single series
                series_key = list(config['series'].keys())[0]
                
                df = self._fetch_single_series(
                    series_type, 
                    config['series'][series_key], 
                    start_year, 
                    end_year
                )
            
            # Create metadata
            metadata = {
                'component': component,
                'title': config['description'],
                'source_series': str(config['series']),
                'frequency': 'M',  # Census data is monthly
                'last_updated': datetime.now().isoformat(),
                'source': 'Census Bureau',
                'notes': f"Stock-Watson component: {component}"
            }
            
            # Save to S3
            self._save_to_s3(df, component, metadata)
            
            # Save locally
            self._save_locally(df, component, metadata)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for component '{component}': {str(e)}")
            raise
    
    def get_all_components(self, start_year: int = 2000, end_year: int = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for all Stock-Watson components.
        
        Args:
            start_year: Start year for the data (default: 2000)
            end_year: End year for the data (default: current year)
            
        Returns:
            Dictionary mapping component names to DataFrames
        """
        # Set default end_year to current year if not specified
        if end_year is None:
            end_year = datetime.now().year
            
        components_data = {}
        
        for component in self.SW_DATA_SOURCES.keys():
            logger.info(f"Fetching {component} data...")
            try:
                df = self.get_component_data(component, start_year, end_year)
                components_data[component] = df
                logger.info(f"Successfully fetched {component} data")
            except Exception as e:
                logger.error(f"Error fetching {component} data: {str(e)}")
                # Continue with other components even if one fails
        
        return components_data 