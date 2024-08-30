import os
import re
from datetime import datetime

def extract_date_from_filename(filename):
    # Regular expression to match dates in 'YYYYMMDD' format
    match = re.search(r'(\d{8})', filename)
    date = match.group(1) if match else None
    #print(f"Extracted date from '{filename}': {date}")
    return date

def is_date_in_range(date_str, start_date_str, end_date_str):
    # Convert string dates to datetime objects
    date = datetime.strptime(date_str, '%Y%m%d')
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    is_in_range = start_date <= date <= end_date
    #print(f"Date '{date_str}' in range ({start_date_str} to {end_date_str}): {is_in_range}")
    return is_in_range

def find_files_in_timeframe(root_dir, start_date_str, end_date_str):
    matching_files = []
    
    for root, dirs, files in os.walk(root_dir):
        #print(f"Checking directory: {root}")
        for file in files:
            date_str = extract_date_from_filename(file)
            if date_str and is_date_in_range(date_str, start_date_str, end_date_str):
                matching_files.append(os.path.join(root, file))
    
    return matching_files

