"""Command-line interface for the interpolation-sw-2010 package."""

import sys
import logging
import argparse
from pathlib import Path

from interpolation_sw_2010.sw2010_interpolator import SW2010Interpolator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Interpolate quarterly GDP to monthly frequency using Stock-Watson (2010) methodology.'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw_data.xlsx',
        help='Path to input data file (default: data/raw_data.xlsx)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/monthly_gdp.csv',
        help='Path to output file (default: data/monthly_gdp.csv)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug output'
    )
    parser.add_argument(
        '--no-plots', '-np',
        action='store_true',
        help='Disable creation of comparison plots'
    )
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figures directory if creating plots
    if not args.no_plots:
        figures_dir = Path("figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize the interpolator
        interpolator = SW2010Interpolator(data_path=args.input)
        
        # Run the interpolation
        results_df = interpolator.run_interpolation()
        
        # Save the results
        results_df.to_csv(args.output, index=False)
        
        logger.info(f"Interpolation completed successfully! Results saved to {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Error during interpolation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 