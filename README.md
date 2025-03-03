# Stock-Watson Interpolation

This package implements the Stock-Watson procedure for temporal disaggregation of quarterly GDP to monthly values.

## Features

- Load and transform economic data from Excel files
- Apply cubic spline detrending
- Construct regressors for Stock-Watson interpolation
- Generate visualizations of the data
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

The package provides a command-line interface for data transformation:

```bash
# Activate the poetry environment
poetry shell

# Basic usage
get-monthly-us-data --input data/raw_data.xlsx --output data/transformed_data.xlsx

# With all options
get-monthly-us-data \
    --input data/raw_data.xlsx \
    --output data/transformed_data.xlsx \
    --debug \
    --visualize \
    --detrend \
    --regressors
```

### Command-line Options

- `--input`, `-i`: Path to input Excel file with Monthly and Quarterly sheets
- `--output`, `-o`: Path to output Excel file
- `--debug`, `-d`: Enable debug output
- `--visualize`, `-v`: Generate visualizations of the transformed data
- `--detrend`, `-t`: Apply cubic spline detrending to the data
- `--regressors`, `-r`: Construct regressors for Stock-Watson interpolation

### Configuration

The package can be configured via a YAML file located at `src/interpolation_sw_2010/config.yaml`. If the file doesn't exist, it will be created with default values.

Example configuration:

```yaml
data:
  default_voa: [1, 4, 5, 0, 2, 4, 4, 7, 1]
  input_file: data/raw_data.xlsx
  output_file: data/transformed_data.xlsx

visualization:
  matplotlib:
    font.family: serif
    font.serif: ["Latin Modern Roman"]
    mathtext.fontset: cm
    axes.titlesize: 12
    axes.labelsize: 10
    xtick.labelsize: 9
    ytick.labelsize: 9
  seaborn_style: whitegrid
  dpi: 300
  figure_sizes:
    correlation: [12, 10]
    time_series: [12, 8]
    distribution: [15, 10]
    boxplot: [15, 8]

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
# Format code with black
poetry run black .

# Sort imports
poetry run isort .

# Run type checking
poetry run mypy .

# Run linting
poetry run flake8 .
```

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]
