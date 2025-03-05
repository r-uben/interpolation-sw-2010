# Stock-Watson (2010) GDP Interpolation

This package implements the Stock-Watson (2010) procedure for temporal disaggregation of quarterly GDP to monthly values. It provides tools to interpolate quarterly economic data to monthly frequency using various methods, with a focus on the approach described in Stock and Watson (2010).

## Features

- Load and transform economic data from Excel files
- Apply cubic spline and linear interpolation methods
- Implement the Stock-Watson (2010) Kalman filter interpolation method
- Generate comparison visualizations between quarterly and monthly data
- Fetch data directly from FRED and BEA (Bureau of Economic Analysis) using web scraping or API
- Configurable via YAML configuration files
- Comprehensive data validation and quality metrics
- Progress tracking and logging
- Automated data update workflow

## Requirements

- Python 3.8+
- Poetry (dependency management)
- Internet connection (for data fetching features)
- BEA API key (for accessing BEA data)

## Installation

This project uses Poetry for dependency management. To install:

1. First, install Poetry if you haven't already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/interpolation-sw-2010.git
cd interpolation-sw-2010
```

3. Install dependencies:

```bash
poetry install
```

4. Set up environment variables:

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env to add your BEA API key
```

## Usage

The package provides several command-line interfaces:

### GDP Interpolation

```bash
# Activate the poetry environment
poetry shell

# Run the Stock-Watson GDP interpolator
poetry run sw2010-gdp-interpolator

# With specific options
poetry run sw2010-gdp-interpolator --data-path custom/path/to/data.xlsx --output custom/output/path.csv
```

### Data Fetching

```bash
# Update raw data from all sources
poetry run sw2010-update-raw-data --all --final

# Fetch only FRED data
poetry run sw2010-update-raw-data --fred

# Fetch only BEA data
poetry run sw2010-update-raw-data --bea

# Fetch specific frequency data
poetry run sw2010-update-raw-data --quarterly
poetry run sw2010-update-raw-data --monthly
```

### Configuration

The application uses YAML configuration files located in `src/interpolation_sw_2010/config/`:

- `sources.yaml`: Defines data sources and their parameters
- `config.yaml`: General application configuration

Example of `sources.yaml` structure:
```yaml
sources:
  - name: gdp
    type: bea
    table_id: T10101
    description: "Gross Domestic Product"
    frequency: quarterly
  - name: ip
    type: fred
    series_id: "INDPRO"
    description: "Industrial Production Index"
    frequency: monthly
```

### Interpolation Methods

The package supports multiple interpolation methods:

1. **Cubic Spline Interpolation**: Uses scipy's cubic spline interpolation to generate monthly values from quarterly data. Implemented in `SplineDetrending` class.

2. **Linear Interpolation**: Uses linear interpolation between quarterly data points.

3. **Kalman Filter Interpolation**: Implements the Stock-Watson (2010) approach using a state-space model and Kalman filter. This is the primary method and is implemented in the `KalmanFilter` class.

The implementation follows the procedure described in Stock and Watson (2010), which:
- Places quarterly values at the middle month of each quarter
- Uses related monthly indicators to guide the interpolation
- Ensures that the sum of three monthly values equals the quarterly value
- Applies a state-space model and Kalman filter for optimal interpolation

### Data Requirements

The input data should be in Excel format with the following structure:

- Quarterly sheet with GDP components (PCE, I_NS, I_ES, I_RS, I_chPI, X, IM, G, PGDP)
- Monthly sheet with related monthly indicators

Alternatively, you can fetch data directly from FRED and BEA using the built-in data fetchers.

### Output

The script generates:

1. A CSV file with interpolated monthly GDP values
2. Comparison plots for each GDP component showing quarterly vs. monthly values
3. A time series plot of the interpolated monthly GDP

## Troubleshooting

### Common Issues

1. **Missing or invalid BEA API key**: Ensure your BEA API key is valid and set correctly in the .env file
2. **Data fetching failures**: Check your internet connection; BEA and FRED websites occasionally change their structure
3. **Interpolation errors**: Ensure your data has sufficient coverage and no extreme outliers

### Performance Considerations

- The Kalman filter optimization can be computationally intensive for large datasets
- For very large datasets, consider increasing the RAM allocation to Python

## Implementation Details

### Core Components

- `sw2010_interpolator.py`: Main implementation of the Stock-Watson interpolation method
- `core/kalman_filter.py`: Implementation of the Kalman filter for temporal disaggregation
- `core/spline_detrending.py`: Cubic spline detrending implementation
- `core/regressors.py`: Handles regression analysis for indicators
- `core/data_manager.py`: Manages data loading, validation, and transformation
- `core/kalman_optimization.py`: Optimizes Kalman filter parameters
- `utils/visualization.py`: Generates comparison plots and visualizations

### BEA Data Fetcher

The BEA data fetcher provides two methods for retrieving data:

1. **Web Scraping**: Uses requests and BeautifulSoup to navigate to the BEA website, extract table structures, and parse the data.
2. **Direct API Access**: Uses the BEA API to directly request the data using the table ID.

## References

Stock, J. H., & Watson, M. W. (2010). "Distribution of quarterly GDP growth rates," NBER Working Paper No. 15862.

Chow, G. C., & Lin, A. L. (1971). "Best linear unbiased interpolation, distribution, and extrapolation of time series by related series," The Review of Economics and Statistics, 53(4), 372-375.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

This project is open source and contributions are welcome. Please submit a pull request or open an issue to discuss proposed changes.
