
import sys
import os
import warnings
from erfa import ErfaWarning
import pandas as pd

def suppress_output(func, *args, **kwargs):
    # Save the original stdout
    original_stdout = sys.stdout
    try:
        # Suppress stdout by redirecting to devnull
        sys.stdout = open(os.devnull, 'w')
        
        # Suppress specific warnings (ErfaWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            result = func(*args, **kwargs)
    finally:
        # Always restore stdout, even if an error occurs
        sys.stdout = original_stdout
    return result

def spacecraft_ID(ID, ID_number=False):
    """
    Retrieve the spacecraft ID or name based on the provided DataFrame or number.

    Parameters:
        ID (DataFrame or int): DataFrame containing 'Spacecraft_ID' or an ID number.
        ID_number (bool): Whether to return the ID number instead of the name.

    Returns:
        str or int: Spacecraft name or ID number.
    """
    df = pd.DataFrame({
        'ID': ['PSP', 'SolO', 'BepiC', 'STEREO-A', 'STEREO-B', 'OMNI', 'MAVEN']
    }, index=[1, 2, 3, 4, 5, 6, 7])

    if isinstance(ID, pd.DataFrame):
        # Check if DataFrame is empty
        if ID.empty:
            raise ValueError("The input DataFrame is empty. Cannot determine spacecraft ID.")
        
        # Check for 'Spacecraft_ID' column
        if 'Spacecraft_ID' in ID.columns:
            ID.dropna(subset=['Spacecraft_ID'], inplace=True)  # Drop rows where 'Spacecraft_ID' is NaN
            if ID.empty:  # Check again if DataFrame is empty after dropping NaN
                raise ValueError("The 'Spacecraft_ID' column is empty after dropping NaN values.")
            
            # Safely access the first value
            number = ID.iloc[0]['Spacecraft_ID']
        else:
            raise ValueError("DataFrame must contain a 'Spacecraft_ID' column.")
    elif isinstance(ID, (int, float)):
        number = int(ID)
    else:
        raise TypeError("ID must be a DataFrame, int, or float.")

    # Return ID or name
    if ID_number:
        return number
    else:
        return df.loc[number, 'ID']
