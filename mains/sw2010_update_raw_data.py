#!/usr/bin/env python
"""
SW2010 Raw Data Update Script

This script updates raw data for the Stock-Watson 2010 interpolation project.
It fetches data from FRED for sources with FRED codes and will later be extended
to scrape web data for other sources.
"""

import json
import os
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
import re

from interpolation_sw_2010.fred.services.data_fetcher import DataFetcher
import beaapi  # Import the beaapi package

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
SOURCES_FILE = DATA_DIR / "sources.json"
VERSION = "1.0.0"

# Source types
SOURCE_TYPE_FRED = "fred"
SOURCE_TYPE_BEA = "bea"

# Get BEA API key from environment or AWS Secrets Manager
def get_bea_api_key():
    """
    Get the BEA API key from environment variable or AWS Secrets Manager.
    
    Returns:
        str: BEA API key
    """
    try:
        # Try to get from environment variable first
        api_key = os.environ.get("BEA_API_KEY")
        if api_key:
            return api_key
        
        # If not in environment, try AWS Secrets Manager
        from interpolation_sw_2010.aws_manager.s3 import S3
        api_key = json.loads(S3.get_secret("BEA-API-KEY"))["api_key"]
        return api_key
    except Exception as e:
        logger.error(f"Error getting BEA API key: {str(e)}")
        raise


def load_sources() -> List[Dict[str, Any]]:
    """
    Load data sources from the sources.json file.
    
    Returns:
        List[Dict[str, Any]]: List of data source configurations
    
    Raises:
        FileNotFoundError: If sources.json file is not found
        json.JSONDecodeError: If sources.json is not valid JSON
    """
    try:
        with open(SOURCES_FILE, "r") as file:
            data = json.load(file)
        logger.info(f"Successfully loaded {len(data)} sources from {SOURCES_FILE}")
        return data
    except FileNotFoundError:
        logger.error(f"Sources file not found: {SOURCES_FILE}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in sources file: {e}")
        raise


def setup_directories() -> None:
    """
    Create necessary directories for data storage if they don't exist.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {RAW_DATA_DIR}")


def calculate_data_quality_metrics(series_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate quality metrics for a data series.
    
    Args:
        series_data (pd.DataFrame): The data series to analyze
        
    Returns:
        Dict[str, Any]: Dictionary of quality metrics
    """
    if series_data.empty:
        return {
            'num_observations': 0,
            'missing_values': 0,
            'missing_pct': 0,
            'min_value': None,
            'max_value': None,
            'mean_value': None,
            'std_dev': None
        }
    
    return {
        'num_observations': len(series_data),
        'missing_values': series_data.isna().sum().iloc[0],
        'missing_pct': round(series_data.isna().sum().iloc[0] / len(series_data) * 100, 2),
        'min_value': float(series_data.min().iloc[0]),
        'max_value': float(series_data.max().iloc[0]),
        'mean_value': float(series_data.mean().iloc[0]),
        'std_dev': float(series_data.std().iloc[0])
    }


def create_metadata(
    source: Dict[str, Any], 
    series_data: pd.DataFrame, 
    series_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create comprehensive metadata for a data series.
    
    Args:
        source (Dict[str, Any]): Source configuration
        series_data (pd.DataFrame): The data series
        series_info (Dict[str, Any]): FRED series information
        
    Returns:
        Dict[str, Any]: Structured metadata
    """
    # Determine actual date range from the data
    actual_start_date = series_data.index.min().strftime('%Y-%m-%d') if not series_data.empty else None
    actual_end_date = series_data.index.max().strftime('%Y-%m-%d') if not series_data.empty else None
    
    # Calculate data quality metrics
    data_quality = calculate_data_quality_metrics(series_data)
    
    return {
        # Source information
        'source_info': {
            'id': source.get('id'),
            'name': source.get('name'),
            'varname': source.get('varname'),
            'source': source.get('source'),
            'release': source.get('release'),
            'link': f"{source.get('link', '')}{source.get('fredcode', '')}" if source.get('link') else None,
            'notes': source.get('notes')
        },
        
        # Data characteristics
        'data_characteristics': {
            'description': source.get('description'),
            'units': source.get('units', series_info.get('units', '')),
            'frequency': source.get('frequency', series_info.get('frequency_short', '')),
            'seasonal_adjustment': series_info.get('seasonal_adjustment', ''),
            'tables': source.get('tables')
        },
        
        # Date information
        'date_info': {
            'expected_start_date': source.get('start_date'),
            'expected_end_date': source.get('end_date'),
            'actual_start_date': actual_start_date,
            'actual_end_date': actual_end_date,
            'last_updated': datetime.now().strftime('%Y-%m-%d'),
            'fred_last_updated': series_info.get('last_updated', '')
        },
        
        # FRED specific information
        'fred_info': {
            'fred_code': source.get('fredcode'),
            'fred_id': series_info.get('id', ''),
            'fred_title': series_info.get('title', ''),
            'fred_notes': series_info.get('notes', '')
        },
        
        # Data quality metrics
        'data_quality': data_quality,
        
        # Processing information
        'processing_info': {
            'processed_at': datetime.now().isoformat(),
            'processor_version': VERSION
        }
    }


def create_bea_metadata(
    source: Dict[str, Any], 
    series_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Create comprehensive metadata for a BEA data series.
    
    Args:
        source (Dict[str, Any]): Source configuration
        series_data (pd.DataFrame): The data series
        
    Returns:
        Dict[str, Any]: Structured metadata
    """
    # Determine actual date range from the data
    actual_start_date = series_data.index.min().strftime('%Y-%m-%d') if not series_data.empty else None
    actual_end_date = series_data.index.max().strftime('%Y-%m-%d') if not series_data.empty else None
    
    # Calculate data quality metrics
    data_quality = calculate_data_quality_metrics(series_data)
    
    return {
        # Source information
        'source_info': {
            'id': source.get('id'),
            'name': source.get('name'),
            'varname': source.get('varname'),
            'source': source.get('source'),
            'table': source.get('table'),
            'link': source.get('link'),
            'notes': source.get('notes')
        },
        
        # Data characteristics
        'data_characteristics': {
            'description': source.get('description'),
            'units': source.get('units', ''),
            'frequency': source.get('frequency', ''),
            'time_window': source.get('time_window', '')
        },
        
        # Date information
        'date_info': {
            'expected_start_date': source.get('start_date'),
            'expected_end_date': source.get('end_date'),
            'actual_start_date': actual_start_date,
            'actual_end_date': actual_end_date,
            'last_updated': datetime.now().strftime('%Y-%m-%d'),
            'source_last_updated': source.get('last_updated', '')
        },
        
        # Data quality metrics
        'data_quality': data_quality,
        
        # Processing information
        'processing_info': {
            'processed_at': datetime.now().isoformat(),
            'processor_version': VERSION
        }
    }


def save_data_and_metadata(
    varname: str, 
    series_data: pd.DataFrame, 
    metadata: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Save data series and its metadata to files.
    
    Args:
        varname (str): Variable name
        series_data (pd.DataFrame): Data to save
        metadata (Dict[str, Any]): Metadata to save
        
    Returns:
        Tuple[str, str]: Paths to saved data and metadata files
    """
    # Save to CSV with appropriate date formatting
    data_file = RAW_DATA_DIR / f"{varname}.csv"
    series_data.to_csv(data_file, date_format='%Y-%m-%d')
    logger.info(f"Saved data to {data_file}")
    
    # Save metadata
    metadata_file = RAW_DATA_DIR / f"{varname}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)  # default=str handles any non-serializable objects
    logger.info(f"Saved metadata to {metadata_file}")
    
    return str(data_file), str(metadata_file)


def process_fred_source(
    data_fetcher: DataFetcher, 
    source: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Process a single FRED data source.
    
    Args:
        data_fetcher (DataFetcher): FRED data fetcher instance
        source (Dict[str, Any]): Source configuration
        
    Returns:
        Optional[Dict[str, Any]]: Result dictionary or None if error
    """
    varname = source.get('varname')
    fredcode = source.get('fredcode')
    
    if not fredcode:
        logger.warning(f"Skipping source {source.get('name')} - no FRED code provided")
        return None
    
    try:
        logger.info(f"Fetching data for {source['name']} (FRED code: {fredcode})")
        
        # Get the data from FRED
        series_data = data_fetcher.client.get_series(fredcode)
        
        # Rename column to match variable name
        series_data.columns = [varname]
        
        # Get series info
        series_info = data_fetcher.client.get_series_info(fredcode)
        
        # Create metadata
        metadata = create_metadata(source, series_data, series_info)
        
        # Save data and metadata
        data_file, metadata_file = save_data_and_metadata(varname, series_data, metadata)
        
        return {
            'varname': varname,
            'data': series_data,
            'metadata': metadata,
            'files': {
                'data': data_file,
                'metadata': metadata_file
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing {source.get('name')}: {str(e)}")
        return {
            'varname': varname,
            'error': str(e),
            'source': source.get('name', 'Unknown'),
            'fred_code': fredcode
        }


def fetch_fred_data(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fetch data from FRED for sources with FRED codes.
    
    Args:
        sources (List[Dict[str, Any]]): List of data source configurations
        
    Returns:
        Dict[str, Any]: Dictionary of results by variable name
    """
    data_fetcher = DataFetcher()
    results = {}
    
    # Filter sources with FRED codes
    fred_sources = [s for s in sources if s.get('fredcode')]
    logger.info(f"Processing {len(fred_sources)} FRED sources")
    
    # Process each source
    for source in fred_sources:
        result = process_fred_source(data_fetcher, source)
        if result:
            results[source.get('varname')] = result
    
    return results


def fetch_bea_page_content(url: str) -> Optional[str]:
    """
    Fetch the HTML content from a BEA.gov URL.
    
    Args:
        url (str): The URL to fetch
        
    Returns:
        Optional[str]: HTML content or None if error
    """
    try:
        logger.info(f"Fetching content from URL: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        
        return response.text
    
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return None


def analyze_bea_page_structure(source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the structure of a BEA.gov page to understand how to extract data.
    This is a diagnostic function to help understand the page structure.
    
    Args:
        source (Dict[str, Any]): Source configuration
        
    Returns:
        Dict[str, Any]: Information about the page structure
    """
    url = source.get('link')
    if not url:
        logger.warning(f"No URL provided for source: {source.get('name')}")
        return {'error': 'No URL provided'}
    
    html_content = fetch_bea_page_content(url)
    if not html_content:
        return {'error': 'Failed to fetch page content'}
    
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Analyze the page structure
    analysis = {
        'title': soup.title.text if soup.title else 'No title found',
        'tables_count': len(soup.find_all('table')),
        'forms_count': len(soup.find_all('form')),
        'iframes_count': len(soup.find_all('iframe')),
        'scripts_count': len(soup.find_all('script')),
        'page_size': len(html_content),
    }
    
    # Check for specific BEA elements
    bea_table_container = soup.find(id='beaTableContainer')
    if bea_table_container:
        analysis['bea_table_container_found'] = True
        analysis['bea_table_container_children'] = len(bea_table_container.find_all())
    else:
        analysis['bea_table_container_found'] = False
    
    # Look for data tables
    tables = soup.find_all('table')
    table_info = []
    for i, table in enumerate(tables[:5]):  # Limit to first 5 tables for brevity
        headers = [th.text.strip() for th in table.find_all('th')]
        rows = len(table.find_all('tr'))
        table_info.append({
            'table_index': i,
            'headers': headers[:10],  # First 10 headers
            'rows_count': rows,
            'has_class': bool(table.get('class')),
            'has_id': bool(table.get('id')),
            'class_value': table.get('class', []),
            'id_value': table.get('id', '')
        })
    
    analysis['tables_info'] = table_info
    
    # Look for API endpoints in JavaScript
    api_endpoints = []
    for script in soup.find_all('script'):
        script_text = script.string
        if script_text:
            # Look for potential API URLs
            api_matches = re.findall(r'(https?://[^\s"\']+api[^\s"\']*)', script_text)
            api_endpoints.extend(api_matches)
            
            # Look for BEA-specific endpoints
            bea_matches = re.findall(r'(https?://[^\s"\']*bea\.gov[^\s"\']*)', script_text)
            api_endpoints.extend(bea_matches)
    
    analysis['potential_api_endpoints'] = list(set(api_endpoints))  # Remove duplicates
    
    # Look for JSON data embedded in the page
    json_data_patterns = []
    for script in soup.find_all('script'):
        script_text = script.string
        if script_text:
            # Look for JSON objects
            json_matches = re.findall(r'(\{["\'].*?["\']:[^\}]+\})', script_text)
            json_data_patterns.extend(json_matches[:5])  # Limit to first 5 matches
    
    analysis['json_data_samples'] = json_data_patterns
    
    # Check for AJAX requests
    ajax_patterns = []
    for script in soup.find_all('script'):
        script_text = script.string
        if script_text:
            # Look for AJAX calls
            ajax_matches = re.findall(r'(ajax\([^\)]+\))', script_text)
            ajax_patterns.extend(ajax_matches)
            
            # Look for fetch calls
            fetch_matches = re.findall(r'(fetch\([^\)]+\))', script_text)
            ajax_patterns.extend(fetch_matches)
    
    analysis['ajax_patterns'] = ajax_patterns
    
    return analysis


def extract_bea_table_info(source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract information about the BEA table structure.
    
    Args:
        source (Dict[str, Any]): Source configuration
        
    Returns:
        Dict[str, Any]: Information about the table structure
    """
    # For BEA tables, we need to understand the URL structure
    url = source.get('link', '')
    table_id = source.get('table')
    
    # Extract parameters from the URL
    url_info = {
        'full_url': url,
        'base_url': url.split('#')[0] if '#' in url else url,
        'hash_params': url.split('#')[1] if '#' in url else None,
        'table_id': table_id
    }

    # If there's a hash part, try to decode it
    if url_info['hash_params']:
        try:
            # Some BEA URLs use base64 encoded JSON in the hash
            import base64
            decoded = None
            
            # Try different decoding approaches
            try:
                # Try standard base64 decoding
                decoded = base64.b64decode(url_info['hash_params']).decode('utf-8')
            except:
                # If that fails, try URL-safe base64 decoding
                try:
                    decoded = base64.urlsafe_b64decode(url_info['hash_params']).decode('utf-8')
                except:
                    # If that fails too, try to parse it as is
                    decoded = url_info['hash_params']
            
            # Try to parse as JSON
            try:
                if decoded:
                    import json
                    parsed_json = json.loads(decoded)
                    url_info['decoded_params'] = parsed_json
            except:
                url_info['decoded_params'] = None
                url_info['raw_decoded'] = decoded
                
        except Exception as e:
            url_info['decoding_error'] = str(e)
    
    return url_info


def fetch_bea_data(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze BEA data sources and prepare for fetching.
    
    Args:
        sources (List[Dict[str, Any]]): List of data source configurations
        
    Returns:
        Dict[str, Any]: Dictionary of results by variable name
    """
    results = {}
    
    # Filter sources with BEA links (those without FRED codes but with links to BEA)
    bea_sources = [s for s in sources if not s.get('fredcode') and 'bea.gov' in s.get('link', '')]
    logger.info(f"Analyzing {len(bea_sources)} BEA sources")
    
    # For each BEA source, extract table information
    for source in bea_sources:
        varname = source.get('varname')
        logger.info(f"Analyzing BEA source for {source.get('name')} ({varname})")
        
        try:
            # Extract table information
            table_info = extract_bea_table_info(source)
            
            # Save the table info
            info_file = RAW_DATA_DIR / f"{varname}_bea_info.json"
            with open(info_file, 'w') as f:
                json.dump(table_info, f, indent=2, default=str)
            
            logger.info(f"Saved BEA table info to {info_file}")
            
            results[varname] = {
                'varname': varname,
                'table_info': table_info,
                'files': {
                    'info': str(info_file)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing BEA source for {source.get('name')}: {str(e)}")
            results[varname] = {
                'varname': varname,
                'error': str(e),
                'source': source.get('name', 'Unknown')
            }
    
    return results


def fetch_bea_api_data(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fetch data from BEA API using the beaapi package.
    
    Args:
        sources (List[Dict[str, Any]]): List of data source configurations
        
    Returns:
        Dict[str, Any]: Dictionary of results by variable name
    """
    results = {}
    
    # Get BEA API key
    try:
        api_key = get_bea_api_key()
    except Exception as e:
        logger.error(f"Failed to get BEA API key: {str(e)}")
        return results
    
    # Filter sources with BEA tables (those without FRED codes but with BEA table info)
    bea_sources = [s for s in sources if not s.get('fredcode') and (s.get('table') or 'bea.gov' in s.get('link', ''))]
    logger.info(f"Processing {len(bea_sources)} BEA API sources")
    
    for source in bea_sources:
        varname = source.get('varname')
        table_name = source.get('table')
        series_code = source.get('series_code')
        line_number = source.get('line_number')
        hierarchy_level = source.get('hierarchy_level', 'top')  # Default to top level if not specified
        
        # Skip if no table name is provided
        if not table_name:
            logger.warning(f"Skipping {varname}: No table name provided")
            continue
        
        logger.info(f"Fetching BEA data for {source.get('name')} ({varname}) from table {table_name}")
        
        try:
            # Map table names to BEA API table names
            # The BEA API uses specific table name formats
            api_table_name = table_name
            
            # Handle specific table name formats
            if table_name == "5.7.5AM1":
                api_table_name = "U50705AM1"  # Underlying detail table for 5.7.5A Monthly
            elif table_name == "5.7.5BM1":
                api_table_name = "U50705BM1"  # Underlying detail table for 5.7.5B Monthly
            elif "2.7A" in table_name:
                api_table_name = "T20700A"  # Table 2.7A
            elif "2.7B" in table_name:
                api_table_name = "T20700B"  # Table 2.7B
            
            # Determine the dataset name based on the table name
            dataset_name = 'NIPA'
            
            # Determine the frequency (default to monthly)
            frequency = source.get('frequency', 'monthly').startswith('m') and 'M' or 'A'
            
            # Try to get the data with error handling
            try:
                logger.info(f"Requesting BEA data for {varname} with TableName={api_table_name}, Frequency={frequency}")
                bea_data = beaapi.get_data(
                    api_key, 
                    datasetname=dataset_name, 
                    TableName=api_table_name, 
                    Frequency=frequency, 
                    Year="ALL"
                )

                # Safely inspect the data type and structure
                logger.info(f"BEA API returned data of type: {type(bea_data)}")
                
            except Exception as e:
                error_message = f"BEA API request failed: {str(e)}"
                logger.warning(f"Error fetching data for {varname}: {str(e)}")


            # Filter out unwanted categories
            if 'LineDescription' in bea_data.columns:
                # Define patterns to exclude
                exclude_patterns = [
                    'durable goods', 
                    'nondurable goods', 
                    'other',
                    '1'
                ]
                
                # Count rows before filtering
                rows_before = len(bea_data)
                
                # Create a filter mask - initialize as a Series of True values
                mask = pd.Series(True, index=bea_data.index, dtype=bool)
                
                # Apply pattern filters
                for pattern in exclude_patterns:
                    # Case-insensitive pattern matching
                    pattern_filter = ~bea_data['LineDescription'].str.lower().str.contains(pattern.lower())
                    mask = mask & pattern_filter
                
                # Apply the filter
                bea_data = bea_data[mask]
                
                # Aggregate by TimePeriod
                bea_data = bea_data.groupby('TimePeriod')['DataValue'].sum().reset_index()
            
            # Check if we have data
            if len(bea_data) == 0:
                raise Exception(f"No data available for {varname}")
            
            # Convert TimePeriod to datetime and set as index
            if 'TimePeriod' in bea_data.columns:
                # Convert TimePeriod to datetime
                logger.info(f"Converting TimePeriod to datetime for {varname}")
                bea_data['date'] = pd.to_datetime(bea_data['TimePeriod'], 
                                                format='%YM%m' if 'M' in bea_data['TimePeriod'].iloc[0] else '%Y')
                bea_data = bea_data.set_index('date')
                bea_data = bea_data.sort_index()
            else:
                logger.warning(f"TimePeriod column not found in BEA data for {varname}")
            
            # Extract the DataValue column as our series data
            if 'DataValue' in bea_data.columns:
                # Create a new DataFrame with just the DataValue column
                logger.info(f"Extracting DataValue column for {varname}")
                series_data = pd.DataFrame(bea_data['DataValue'])
                # Rename the column to match the variable name
                series_data.columns = [varname]
            else:
                logger.warning(f"DataValue column not found in BEA data for {varname}")
                # Try to find any numeric column
                numeric_cols = bea_data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    # Use the first numeric column
                    series_data = pd.DataFrame(bea_data[numeric_cols[0]])
                    # Rename the column to match the variable name
                    series_data.columns = [varname]
                else:
                    raise Exception(f"No numeric data columns found in BEA data for {varname}")
            
            # Create metadata with additional BEA API info
            metadata = create_bea_metadata(source, series_data)
            
            # Add BEA API specific information
            metadata['bea_api_info'] = {
                'dataset': dataset_name,
                'table_name': api_table_name,
                'frequency': frequency,
                'series_code': series_code,
                'line_number': line_number,
                'excluded_patterns': exclude_patterns,
                'time_window': source.get('time_window', ''),
                'retrieved_at': datetime.now().isoformat()
            }
            
            # Save data and metadata
            data_file, metadata_file = save_data_and_metadata(varname, series_data, metadata)
            
            results[varname] = {
                'varname': varname,
                'data': series_data,
                'metadata': metadata,
                'files': {
                    'data': data_file,
                    'metadata': metadata_file
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching BEA data for {source.get('name')}: {str(e)}")
            results[varname] = {
                'varname': varname,
                'error': str(e),
                'source': source
            }
    
    return results


def main(fetch_fred=None, fetch_bea=None, fetch_all=None):
    """
    Main execution function.
    
    Args:
        fetch_fred (bool): Whether to fetch FRED data
        fetch_bea (bool): Whether to fetch BEA data
        fetch_all (bool): Whether to fetch all data (overrides other flags)
    """
    # Parse command-line arguments if not provided
    if fetch_fred is None and fetch_bea is None and fetch_all is None:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Update raw data for the Stock-Watson 2010 interpolation project.')
        parser.add_argument('--bea', action='store_true', help='Fetch BEA data only')
        parser.add_argument('--fred', action='store_true', help='Fetch FRED data only')
        parser.add_argument('--all', action='store_true', help='Fetch all data (default if no flags are specified)')
        
        # Parse arguments
        args = parser.parse_args()
        
        # If no flags are specified, default to fetching all data
        if not (args.bea or args.fred or args.all):
            args.all = True
        
        fetch_fred = args.fred
        fetch_bea = args.bea
        fetch_all = args.all
    
    try:
        # Setup
        setup_directories()
        
        # Load sources
        sources = load_sources()
        logger.info(f"Loaded {len(sources)} data sources")
        
        # If fetch_all is True, set both fetch_fred and fetch_bea to True
        if fetch_all:
            fetch_fred = True
            fetch_bea = True
        
        # Fetch data from FRED if requested
        if fetch_fred:
            logger.info("Fetching FRED data...")
            fred_results = fetch_fred_data(sources)
            successful_fred = sum(1 for r in fred_results.values() if 'error' not in r)
            logger.info(f"Fetched data for {successful_fred} FRED sources successfully")
        else:
            logger.info("Skipping FRED data fetching")
        
        # Fetch data from BEA if requested
        if fetch_bea:
            logger.info("Fetching BEA data...")
            
            # Analyze BEA sources
            bea_info_results = fetch_bea_data(sources)
            successful_bea_info = sum(1 for r in bea_info_results.values() if 'error' not in r)
            logger.info(f"Analyzed {successful_bea_info} BEA sources successfully")
            
            # Fetch data from BEA API
            try:
                bea_api_results = fetch_bea_api_data(sources)
                successful_bea_api = sum(1 for r in bea_api_results.values() if 'error' not in r)
                logger.info(f"Fetched data for {successful_bea_api} BEA API sources successfully")
            except Exception as e:
                logger.error(f"Error fetching data from BEA API: {str(e)}")
                logger.info("BEA API data fetching failed")
        else:
            logger.info("Skipping BEA data fetching")
        
        logger.info("Data update completed")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    # When run directly, call main() without arguments to trigger argument parsing
    main()
