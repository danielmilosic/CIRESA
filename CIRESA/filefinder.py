import os
import re
from datetime import datetime, timedelta

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
    # Add one day
    end_date += timedelta(days=1)
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

def find_parquet_files(root_dir, month):
    import os
    # Iterate through the directory structure
    for root, dirs, files in os.walk(root_dir):
        # Iterate over each file in the current directory
        for file in files:
            # Check if the file is a Parquet file and matches the month string
            if file.endswith('.parquet') and month in file:
                # Return the full path to the matching file
                return os.path.join(root, file)
    
    # Return None if no matching file is found
    return None


def get_month_dates(month_str):
    from datetime import datetime
    import calendar

    # Parse the string to get year and month
    year, month = map(int, month_str.split('-'))

    # Get the first day of the month
    first_day = datetime(year, month, 1)

    # Get the last day of the month
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])

    # Format the dates as 'YYYY-MM-DD'
    first_day_str = first_day.strftime('%Y-%m-%d')
    last_day_str = last_day.strftime('%Y-%m-%d')

    # Return a list with the formatted first and last days of the month
    return [first_day_str, last_day_str]