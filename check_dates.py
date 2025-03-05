import pandas as pd

# Read the Excel file
df = pd.read_excel('data/raw_data.xlsx', sheet_name='Monthly')

# Print the last year with valid data for each column
print('Last year for each column:')
for col in df.columns:
    if col not in ['Year', 'Month']:
        last_valid = df[df[col].notna()]['Year'].max()
        print(f'{col}: {last_valid}') 