from interpolation_sw_2010.data_manager import DataManager
import numpy as np

def main():
    # Create a DataManager instance
    data_manager = DataManager()
    
    # Call the transform method
    Y, X, names = data_manager.transform()
    
    # Print some information about the transformed data
    print("Quarterly data (Y) shape:", Y.shape)
    print("Monthly data (X) shape:", X.shape)
    print("Number of variable names:", len(names))
    print("\nVariable names:")
    for i, name in enumerate(names):
        print(f"{i+1}. {name}")
    
    # Print a sample of the transformed data
    print("\nSample of transformed monthly data (first 5 rows, first 5 columns):")
    print(X[:5, :5])

if __name__ == "__main__":
    main() 