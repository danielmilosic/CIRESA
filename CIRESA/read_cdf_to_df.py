# Function to read a list of cdf files

def read_cdf_files_to_dataframe(cdf_file_list, variables_to_read):
    import pandas as pd
    import cdflib

    # Initialize lists to store the data
    data = {var: [] for var in variables_to_read}

    # Loop through the CDF file list
    for cdf_file_path in cdf_file_list:
        cdf_file = cdflib.CDF(cdf_file_path)
        for var in variables_to_read:
            data[var].extend(cdf_file.varget(var).tolist())
        cdf_file.close()

    # Create a pandas DataFrame from the collected data
    df = pd.DataFrame(data)
    return df