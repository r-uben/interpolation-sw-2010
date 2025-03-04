#!/usr/bin/env python
"""
Fetch all data needed for Stock-Watson (2010) interpolation.

This script fetches data from all sources needed for the Stock-Watson interpolation:
- Federal Reserve Economic Data (FRED)
- Bureau of Economic Analysis (BEA)
- Census Bureau

Usage:
    python fetch_all_sw_data.py [--start_year YEAR] [--end_year YEAR] [--interactive]
"""
import argparse
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import data fetchers
from src.interpolation_sw_2010.fred import DataFetcher as FREDFetcher
from src.interpolation_sw_2010.bea import DataFetcher as BEAFetcher
from src.interpolation_sw_2010.census import DataFetcher as CensusFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_fetch.log')
    ]
)
logger = logging.getLogger(__name__)

# FRED series needed for Stock-Watson interpolation
FRED_SERIES = {
    # Construction series
    'constprivus': {
        'id': 'CONSTPRIVUS',  # Private Construction Spending
        'description': 'Private Construction Spending',
        'notes': 'Used for NonResidential Structures calculation (CONSTPRIVUS - CONSTRESUS)'
    },
    'constresus': {
        'id': 'CONSTRESUS',  # Residential Construction Spending
        'description': 'Residential Construction Spending',
        'notes': 'Used directly for Residential Structures and in NonResidential calculation'
    },
    'constpubus': {
        'id': 'CONSTPUBUS',  # Public Construction Spending
        'description': 'Public Construction Spending',
        'notes': 'Used for Government Construction component'
    },
    
    # Manufacturing series
    'mfgnondfcs': {
        'id': 'MFGNONDFCS',  # Non-Durable Manufacturing
        'description': 'Non-Durable Manufacturing',
        'notes': 'Used for Equipment and Software calculation (MFGNONDFCS + MFGCEPS)'
    },
    'mfgceps': {
        'id': 'MFGCEPS',  # Durable Manufacturing
        'description': 'Durable Manufacturing',
        'notes': 'Used for Equipment and Software calculation (MFGNONDFCS + MFGCEPS)'
    },
    'mfginvt': {
        'id': 'MFGINVT',  # Manufacturing Inventories
        'description': 'Manufacturing Inventories',
        'notes': 'First difference used for Change in Private Inventories'
    }
}

def ensure_directory(path: str):
    """Ensure a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def fetch_fred_data(start_year=None, end_year=None):
    """Fetch all required FRED data."""
    logger.info("Fetching data from Federal Reserve (FRED)...")
    
    try:
        fetcher = FREDFetcher()
        fred_data = {}
        
        for name, info in FRED_SERIES.items():
            logger.info(f"Fetching FRED series: {name} ({info['id']})")
            try:
                data = fetcher.get_series(info['id'])
                if data is not None and not data.empty:
                    fred_data[name] = data
                    logger.info(f"Successfully fetched {name}")
                    logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
                    logger.info(f"Number of observations: {len(data)}")
                    logger.info(f"Latest value: {data['value'].iloc[-1]:.2f}")
                else:
                    logger.error(f"No data returned for {name}")
            except Exception as e:
                logger.error(f"Error fetching {name}: {str(e)}")
        
        # Save summary
        ensure_directory('data/summary')
        with open('data/summary/fred_summary.txt', 'w') as f:
            f.write(f"FRED Data Fetch Summary - {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            for name, df in fred_data.items():
                if df is not None and not df.empty:
                    f.write(f"{name}:\n")
                    f.write(f"Description: {FRED_SERIES[name]['description']}\n")
                    f.write(f"Series ID: {FRED_SERIES[name]['id']}\n")
                    f.write(f"Notes: {FRED_SERIES[name]['notes']}\n")
                    f.write(f"Date range: {df.index.min()} to {df.index.max()}\n")
                    f.write(f"Observations: {len(df)}\n")
                    f.write(f"Latest value: {df['value'].iloc[-1]:.2f}\n\n")
        
        return fred_data
        
    except Exception as e:
        logger.error(f"Error fetching FRED data: {str(e)}")
        return {}

def fetch_bea_data(start_year=None, end_year=None):
    """Fetch all required BEA data."""
    logger.info("Fetching data from Bureau of Economic Analysis (BEA)...")
    
    try:
        fetcher = BEAFetcher()
        
        # Fetch all BEA components defined in the data fetcher
        bea_data = fetcher.get_all_components(start_year, end_year)
        
        # Save summary
        ensure_directory('data/summary')
        with open('data/summary/bea_summary.txt', 'w') as f:
            f.write(f"BEA Data Fetch Summary - {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            for name, df in bea_data.items():
                if df is not None and not df.empty:
                    f.write(f"{name}:\n")
                    f.write(f"  Date range: {df.index.min()} to {df.index.max()}\n")
                    f.write(f"  Observations: {len(df)}\n")
                    f.write(f"  Latest value: {df['value'].iloc[-1]:.2f}\n\n")
        
        return bea_data
        
    except Exception as e:
        logger.error(f"Error fetching BEA data: {str(e)}")
        return {}

def fetch_census_data(start_year=None, end_year=None):
    """Fetch all required Census Bureau data."""
    logger.info("Fetching data from Census Bureau...")
    
    try:
        fetcher = CensusFetcher()
        
        # Fetch all Census components defined in the data fetcher
        census_data = fetcher.get_all_components(start_year, end_year)
        
        # Save summary
        ensure_directory('data/summary')
        with open('data/summary/census_summary.txt', 'w') as f:
            f.write(f"Census Bureau Data Fetch Summary - {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            for name, df in census_data.items():
                if df is not None and not df.empty:
                    f.write(f"{name}:\n")
                    f.write(f"  Date range: {df.index.min()} to {df.index.max()}\n")
                    f.write(f"  Observations: {len(df)}\n")
                    f.write(f"  Latest value: {df['value'].iloc[-1]:.2f}\n\n")
        
        return census_data
        
    except Exception as e:
        logger.error(f"Error fetching Census data: {str(e)}")
        return {}

def interactive_mode():
    """Interactive mode for fetching specific components."""
    logger.info("\nEntering interactive mode...")
    logger.info("Type 'exit' or 'quit' to end the session")
    
    # Create fetchers
    fred_fetcher = FREDFetcher()
    bea_fetcher = BEAFetcher()
    census_fetcher = CensusFetcher()
    
    # Show available components
    print("\nAvailable components:")
    print("FRED components:")
    for name in FRED_SERIES:
        print(f"  - fred:{name}")
    
    print("\nBEA components:")
    for name in bea_fetcher.SW_DATA_SOURCES:
        print(f"  - bea:{name}")
    
    print("\nCensus components:")
    for name in census_fetcher.SW_DATA_SOURCES:
        print(f"  - census:{name}")
    
    while True:
        try:
            query = input("\nEnter component to fetch (e.g., fred:gdp, bea:pce, census:nonresidential_structures): ").strip().lower()
            
            if query in ['exit', 'quit']:
                logger.info("Exiting interactive mode...")
                break
            
            if not query:
                continue
                
            # Parse the query
            if ':' not in query:
                print("Invalid format. Use source:component format (e.g., fred:gdp)")
                continue
                
            source, component = query.split(':')
            
            if source == 'fred':
                if component not in FRED_SERIES:
                    print(f"Unknown FRED component: {component}")
                    continue
                df = fred_fetcher.get_series(FRED_SERIES[component]['id'])
                
            elif source == 'bea':
                if component not in bea_fetcher.SW_DATA_SOURCES:
                    print(f"Unknown BEA component: {component}")
                    continue
                df = bea_fetcher.get_component_data(component)
                
            elif source == 'census':
                if component not in census_fetcher.SW_DATA_SOURCES:
                    print(f"Unknown Census component: {component}")
                    continue
                df = census_fetcher.get_component_data(component)
                
            else:
                print(f"Unknown source: {source}")
                continue
            
            # Print summary statistics
            print("\n=== Data Summary ===")
            print(f"Component: {source}:{component}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Observations: {len(df)}")
            print(f"Latest value: {df['value'].iloc[-1]:.2f}")
            
            # Ask if user wants to see more details
            show_more = input("\nWould you like to see more details? (y/n): ").strip().lower()
            if show_more == 'y':
                print("\nLast 5 observations:")
                print(df.tail())
                print("\nBasic statistics:")
                print(df.describe())
            
        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode...")
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")

def combine_data_sources(fred_data, bea_data, census_data):
    """Combine all data sources into a single integrated dataset."""
    logger.info("Combining all data sources...")
    
    try:
        # Create a summary dictionary
        all_data = {
            'fred': fred_data,
            'bea': bea_data,
            'census': census_data
        }
        
        # Create a combined summary
        ensure_directory('data/summary')
        with open('data/summary/all_data_summary.txt', 'w') as f:
            f.write(f"Stock-Watson Data Fetch Summary - {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
            
            for source, data_dict in all_data.items():
                f.write(f"{source.upper()} Data:\n")
                f.write("-" * 40 + "\n")
                
                for name, df in data_dict.items():
                    if df is not None and not df.empty:
                        f.write(f"{name}:\n")
                        f.write(f"Description: {FRED_SERIES[name]['description']}\n")
                        f.write(f"Series ID: {FRED_SERIES[name]['id']}\n")
                        f.write(f"Notes: {FRED_SERIES[name]['notes']}\n")
                        f.write(f"Date range: {df.index.min()} to {df.index.max()}\n")
                        f.write(f"Observations: {len(df)}\n")
                        f.write(f"Latest value: {df['value'].iloc[-1]:.2f}\n\n")
                
                f.write("\n")
        
        logger.info(f"Data summary saved to data/summary/all_data_summary.txt")
        return all_data
        
    except Exception as e:
        logger.error(f"Error combining data sources: {str(e)}")
        return {}

def main():
    """Main function to fetch all data needed for Stock-Watson interpolation."""
    parser = argparse.ArgumentParser(description="Fetch data for Stock-Watson (2010) interpolation")
    parser.add_argument('--start_year', type=int, help='Start year for data')
    parser.add_argument('--end_year', type=int, help='End year for data')
    parser.add_argument('--interactive', action='store_true', help='Enter interactive mode')
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return 0
    
    logger.info("Starting Stock-Watson data fetch...")
    
    # Fetch data from all sources
    fred_data = fetch_fred_data(args.start_year, args.end_year)
    bea_data = fetch_bea_data(args.start_year, args.end_year)
    census_data = fetch_census_data(args.start_year, args.end_year)
    
    # Combine all data sources
    all_data = combine_data_sources(fred_data, bea_data, census_data)
    
    logger.info("Stock-Watson data fetch complete!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 