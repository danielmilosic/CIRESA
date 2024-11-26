
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

import os
from PIL import Image


def glue_together(folder_a, folder_b, output_folder):

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get sorted lists of file names from both folders
    files_a = sorted(f for f in os.listdir(folder_a) if f.endswith(".png"))
    files_b = sorted(f for f in os.listdir(folder_b) if f.endswith(".png"))

    # Ensure both folders have the same number of files
    if len(files_a) != len(files_b):
        print("Warning: The number of files in folder A and folder B do not match!")

    # Process files
    for file_a, file_b in zip(files_a, files_b):
        # Open images from both folders
        img_a = Image.open(os.path.join(folder_a, file_a))
        img_b = Image.open(os.path.join(folder_b, file_b))
        
        # Ensure the images have the same height by resizing (optional, remove if not needed)
        if img_a.height != img_b.height:
            img_b = img_b.resize((img_b.width, img_a.height), Image.ANTIALIAS)
        
        # Create a new image with the combined width and same height
        combined_width = img_a.width + img_b.width
        combined_image = Image.new("RGB", (combined_width, img_a.height))
        
        # Paste the images side by side
        combined_image.paste(img_a, (0, 0))
        combined_image.paste(img_b, (img_a.width, 0))
        
        # Save the combined image in the output folder
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file_a)  # Use file_a name for the output
        combined_image.save(output_path)

        print(f"Saved combined image: {output_path}")

    print("All images have been processed and saved.")
