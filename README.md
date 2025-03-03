# Stock-Watson (2010) GDP Interpolation

This package implements the Stock-Watson (2010) procedure for temporal disaggregation of quarterly GDP to monthly values. It provides tools to interpolate quarterly economic data to monthly frequency using various methods, with a focus on the approach described in Stock and Watson (2010).

## Features

- Load and transform economic data from Excel files
- Apply cubic spline and linear interpolation methods
- Implement the Stock-Watson (2010) Kalman filter interpolation method
- Generate comparison visualizations between quarterly and monthly data
- Configurable via YAML configuration files
- Comprehensive data validation
- Progress tracking and logging

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

## Usage

The package provides a command-line interface for interpolating quarterly GDP data to monthly frequency:

```bash
# Activate the poetry environment
poetry shell

# Run the Stock-Watson GDP interpolator
poetry run sw2010-gdp-interpolator

# Or directly from the script
poetry run python mains/sw2010_gdp_interpolator.py
```

### Interpolation Methods

The package supports multiple interpolation methods:

1. **Cubic Spline Interpolation**: Uses scipy's cubic spline interpolation to generate monthly values from quarterly data.
2. **Linear Interpolation**: Uses linear interpolation between quarterly data points.
3. **Kalman Filter Interpolation**: Implements the Stock-Watson (2010) approach using a state-space model and Kalman filter.

### Data Requirements

The input data should be in Excel format with the following structure:
- Quarterly sheet with GDP components (PCE, I_NS, I_ES, I_RS, I_chPI, X, IM, G, PGDP)
- Monthly sheet with related monthly indicators

### Output

The script generates:
1. A CSV file with interpolated monthly GDP values
2. Comparison plots for each GDP component showing quarterly vs. monthly values
3. A time series plot of the interpolated monthly GDP

## Implementation Details

### Stock-Watson (2010) Method

The Stock-Watson method for interpolating quarterly GDP to monthly frequency involves:

1. Placing quarterly values at the middle month of each quarter
2. Using related monthly indicators to guide the interpolation
3. Ensuring that the sum of three monthly values equals the quarterly value
4. Applying a state-space model and Kalman filter for optimal interpolation

### Key Components

- `sw2010_gdp_interpolator.py`: Main script for interpolation
- `sw2010_interpolator.py`: Implementation of the Stock-Watson interpolation method
- `visualization.py`: Tools for generating comparison plots
- `spline_detrending.py`: Implementation of cubic spline detrending

## References

Stock, J. H., & Watson, M. W. (2010). Distribution of quarterly GDP growth rates.

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]
