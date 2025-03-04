import logging
from macroeconomic_data.fred import DataFetcher
import pandas as pd
import argparse
from pathlib import Path
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def ensure_directory(path: str):
    """Ensure a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_to_local(data: pd.DataFrame, name: str, description: str):
    """Save data to local fred directory."""
    base_dir = "data/fred"
    var_dir = os.path.join(base_dir, name)
    ensure_directory(var_dir)
    
    # Save data
    filename = os.path.join(var_dir, "data.csv")
    data.to_csv(filename, index=True)
    
    # Save metadata
    with open(os.path.join(var_dir, "metadata.txt"), "w") as f:
        f.write(f"Variable: {name}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Last updated: {pd.Timestamp.now()}\n")
        f.write(f"Frequency: {data.index.freq if data.index.freq else 'Inferred from data'}\n")
        f.write(f"Date range: {data.index.min()} to {data.index.max()}\n")
        f.write(f"Latest value: {data['value'].iloc[-1]:.2f}\n")

def fetch_key_economic_indicators():
    """Fetch key economic indicators from Federal Reserve Economic Data (FRED)"""
    logger.info("Fetching key economic indicators from FRED...")
    
    try:
        fetcher = DataFetcher()
        
        # Dictionary of indicators with their descriptions
        indicators = {
            'gdp': ('real gdp', 'Real Gross Domestic Product'),
            'industrial_production': ('industrial production', 'Industrial Production Index'),
            'unemployment': ('unemployment rate', 'Unemployment Rate'),
            'cpi': ('consumer price index', 'Consumer Price Index'),
            'core_inflation': ('core inflation', 'Core Inflation Rate'),
            'fed_funds': ('federal funds rate', 'Federal Funds Rate')
        }
        
        # Fetch and save data
        data_dict = {}
        
        # Real Economy
        logger.info("\n=== Real Economy Indicators ===")
        for name in ['gdp', 'industrial_production', 'unemployment']:
            query, description = indicators[name]
            data = fetcher.get_series(query)
            data_dict[name] = data
            save_to_local(data, name, description)
            logger.info(f"Saved {name} data")
        
        # Prices and Inflation
        logger.info("\n=== Price and Inflation Indicators ===")
        for name in ['cpi', 'core_inflation']:
            query, description = indicators[name]
            data = fetcher.get_series(query)
            data_dict[name] = data
            save_to_local(data, name, description)
            logger.info(f"Saved {name} data")
        
        # Financial Conditions
        logger.info("\n=== Financial Indicators ===")
        name = 'fed_funds'
        query, description = indicators[name]
        data = fetcher.get_series(query)
        data_dict[name] = data
        save_to_local(data, name, description)
        logger.info(f"Saved {name} data")
        
        # Print summary statistics
        logger.info("\n=== Data Summary ===")
        for name, data in data_dict.items():
            logger.info(f"\n{name.upper()}:")
            logger.info(f"Frequency: {data.index.freq if data.index.freq else 'Inferred from data'}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"Latest value: {data['value'].iloc[-1]:.2f}")
        
        return data_dict
        
    except Exception as e:
        logger.error(f"Error fetching economic data: {str(e)}")
        raise

def interactive_mode():
    """Interactive mode for querying FRED data"""
    logger.info("\nEntering interactive mode...")
    logger.info("Type 'exit' or 'quit' to end the session")
    logger.info("Example queries:")
    logger.info("  - real gdp")
    logger.info("  - unemployment rate")
    logger.info("  - consumer price index")
    logger.info("  - federal funds rate")
    
    fetcher = DataFetcher()
    
    while True:
        try:
            query = input("\nEnter your query: ").strip().lower()
            
            if query in ['exit', 'quit']:
                logger.info("Exiting interactive mode...")
                break
            
            if not query:
                continue
                
            data = fetcher.get_series(query)
            
            # Save the data
            name = query.replace(" ", "_")
            save_to_local(data, name, query)
            logger.info(f"\nSaved data to data/fred/{name}/")
            
            # Print summary statistics
            logger.info("\n=== Data Summary ===")
            logger.info(f"Query: {query}")
            logger.info(f"Frequency: {data.index.freq if data.index.freq else 'Inferred from data'}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"Latest value: {data['value'].iloc[-1]:.2f}")
            
            # Ask if user wants to see more details
            show_more = input("\nWould you like to see more details? (y/n): ").strip().lower()
            if show_more == 'y':
                print("\nLast 5 observations:")
                print(data.tail())
                print("\nBasic statistics:")
                print(data.describe())
            
        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode...")
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")

def main():
    """Main function to fetch and display Federal Reserve economic data"""
    parser = argparse.ArgumentParser(description="Fetch economic data from Federal Reserve (FRED)")
    parser.add_argument('-i', '--interactive', action='store_true', 
                       help='Enter interactive mode for querying FRED data')
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return 0
    
    logger.info("Starting Federal Reserve data fetch...")
    try:
        indicators = fetch_key_economic_indicators()
        logger.info("\nSuccessfully retrieved all economic indicators")
        return 0
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 