"""
Service for fetching data from BEA.
"""
import logging
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any, List
from pathlib import Path
from src.interpolation_sw_2010.bea.core.bea_client import BEAClient
from src.interpolation_sw_2010.aws_manager.bucket_manager import BucketManager

logger = logging.getLogger(__name__)

class DataFetcher:
    """Service for fetching and managing BEA data."""
    
    # Mapping of Stock-Watson components to BEA data sources
    SW_DATA_SOURCES = {
        # PCE
        'pce': {
            'table': 'T20600',  # Table 2.6 Personal Income and Disposable Personal Income
            'frequency': 'M',
            'description': 'Personal Consumption Expenditures',
            'line_number': '1'  # Line number in Table 2.6
        },
        # Government
        'government': {
            'table': 'T20700',  # Table 2.7 Wages and Salaries by Industry and Government
            'frequency': 'M',
            'description': 'Government Consumption Expenditures and Gross Investment',
            'line_number': '23'  # Line number for government wages
        },
        # Exports
        'exports': {
            'dataset': 'international',
            'table': 'ita',  # International Transactions Accounts
            'frequency': 'M',
            'description': 'Exports of Goods and Services',
            'direction': 'exports'
        },
        # Imports
        'imports': {
            'dataset': 'international',
            'table': 'ita',  # International Transactions Accounts
            'frequency': 'M',
            'description': 'Imports of Goods and Services',
            'direction': 'imports'
        },
        # Employee Compensation (GDI)
        'employee_compensation': {
            'table': 'T20600',  # Table 2.6
            'frequency': 'M',
            'description': 'Compensation of Employees',
            'line_number': '2'
        },
        # Proprietors' Income (GDI)
        'proprietors_income': {
            'table': 'T20600',  # Table 2.6
            'frequency': 'M',
            'description': 'Proprietors\' Income',
            'line_number': '7'
        },
        # Rental Income (GDI)
        'rental_income': {
            'table': 'T20600',  # Table 2.6
            'frequency': 'M',
            'description': 'Rental Income with Capital Consumption Adjustment',
            'line_number': '10'
        },
        # Net Interest (GDI)
        'net_interest': {
            'table': 'T20600',  # Table 2.6
            'frequency': 'M',
            'description': 'Personal Interest Income',
            'line_number': '12'
        },
        # Corporate Profits (GDI) - Quarterly only, needs special handling
        'corporate_profits': {
            'table': 'T11200',  # Table 1.12
            'frequency': 'Q',
            'description': 'Corporate Profits with Inventory Valuation and Capital Consumption Adjustments',
            'monthly_method': 'distribute_equally'  # Distribute quarterly values equally over months
        }
    }
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.client = BEAClient()
        self.bucket_manager = BucketManager(bucket_name="macroeconomic-data")
    
    def _save_to_s3(self, data: pd.DataFrame, component: str, metadata: dict):
        """Save data and metadata to S3."""
        # Create metadata
        full_metadata = {
            'component': component,
            'title': metadata.get('title', ''),
            'table': metadata.get('table', ''),
            'frequency': metadata.get('frequency', ''),
            'last_updated': datetime.now().isoformat(),
            'source': 'Bureau of Economic Analysis (BEA)',
            'observation_start': data.index.min().isoformat() if not data.empty else '',
            'observation_end': data.index.max().isoformat() if not data.empty else '',
            'notes': metadata.get('notes', '')
        }
        
        # Save data
        data_path = f"bea/{component}/data.csv"
        self.bucket_manager.upload_file(
            file_content=data.to_csv().encode(),
            file_path=data_path,
            metadata={'content_type': 'text/csv'}
        )
        
        # Save metadata
        metadata_path = f"bea/{component}/metadata.json"
        self.bucket_manager.upload_file(
            file_content=json.dumps(full_metadata, indent=2).encode(),
            file_path=metadata_path,
            metadata={'content_type': 'application/json'}
        )
    
    def _save_locally(self, data: pd.DataFrame, component: str, metadata: dict):
        """Save data and metadata locally."""
        try:
            # Create base directory
            base_dir = Path('data/bea') / component
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

    def get_component_data(self, component: str, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Get data for a specific Stock-Watson component.
        
        Args:
            component: One of the components defined in SW_DATA_SOURCES
            start_date: Start date for the data (optional)
            end_date: End date for the data (optional)
            
        Returns:
            DataFrame with the component data
        """
        # Check if the component exists
        if component not in self.SW_DATA_SOURCES:
            raise ValueError(f"Unknown component: {component}. Valid components are: {list(self.SW_DATA_SOURCES.keys())}")
        
        # Get component configuration
        config = self.SW_DATA_SOURCES[component]
        
        try:
            # Different handling based on the source dataset
            if 'dataset' in config and config['dataset'] == 'international':
                # Handle international data (exports, imports)
                df = self.client.get_international_data(
                    table_name=config['table'],
                    frequency=config['frequency'],
                    direction=config.get('direction')
                )
            elif config.get('frequency') == 'Q' and config.get('monthly_method') == 'distribute_equally':
                # Handle quarterly data that needs to be distributed to monthly
                # (e.g., corporate profits)
                quarterly_df = self.client.get_nipa_table(
                    table_name=config['table'],
                    frequency='Q'
                )
                
                # Filter by line number if specified
                if 'line_number' in config:
                    quarterly_df = quarterly_df[quarterly_df['LineNumber'] == config['line_number']]
                
                # Convert quarterly to monthly by distributing values equally
                df = self._distribute_quarterly_to_monthly(quarterly_df)
            else:
                # Standard NIPA table data
                df = self.client.get_nipa_table(
                    table_name=config['table'],
                    frequency=config['frequency']
                )
                
                # Filter by line number if specified
                if 'line_number' in config:
                    df = df[df['LineNumber'] == config['line_number']]
            
            # Apply date filtering if specified
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
                
            # Create metadata
            metadata = {
                'component': component,
                'title': config['description'],
                'table': config['table'],
                'frequency': config['frequency'],
                'last_updated': datetime.now().isoformat(),
                'source': 'Bureau of Economic Analysis (BEA)',
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
    
    def _distribute_quarterly_to_monthly(self, quarterly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Distribute quarterly values equally across months.
        This is used for corporate profits which are only reported quarterly.
        
        Args:
            quarterly_df: DataFrame with quarterly data
            
        Returns:
            DataFrame with monthly data
        """
        monthly_data = []
        
        for idx, row in quarterly_df.iterrows():
            # Get the quarter
            year = idx.year
            quarter = idx.quarter
            
            # Determine the months in this quarter
            if quarter == 1:
                months = [1, 2, 3]
            elif quarter == 2:
                months = [4, 5, 6]
            elif quarter == 3:
                months = [7, 8, 9]
            else:  # quarter == 4
                months = [10, 11, 12]
            
            # Create a row for each month with value = quarterly_value / 3
            for month in months:
                monthly_data.append({
                    'date': pd.to_datetime(f"{year}-{month}-01") + pd.offsets.MonthEnd(0),
                    'value': row['value'] / 3,  # Distribute equally
                    'SeriesName': row.get('SeriesName', ''),
                    'LineNumber': row.get('LineNumber', ''),
                    'LineDescription': row.get('LineDescription', '')
                })
        
        # Convert to DataFrame
        monthly_df = pd.DataFrame(monthly_data)
        monthly_df = monthly_df.set_index('date')
        
        # Sort by date
        monthly_df = monthly_df.sort_index()
        
        return monthly_df
    
    def get_all_components(self, start_date=None, end_date=None) -> Dict[str, pd.DataFrame]:
        """
        Get data for all Stock-Watson components.
        
        Args:
            start_date: Start date for the data (optional)
            end_date: End date for the data (optional)
            
        Returns:
            Dictionary mapping component names to DataFrames
        """
        components_data = {}
        
        for component in self.SW_DATA_SOURCES.keys():
            logger.info(f"Fetching {component} data...")
            try:
                df = self.get_component_data(component, start_date, end_date)
                components_data[component] = df
                logger.info(f"Successfully fetched {component} data")
            except Exception as e:
                logger.error(f"Error fetching {component} data: {str(e)}")
                # Continue with other components even if one fails
        
        return components_data 