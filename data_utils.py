
import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", verify=False)

def get_percent_above_50_80(probMap):
    total_pixels = probMap.size
    above_50_percent = np.sum(probMap > 0.5)
    above_80_percent = np.sum(probMap > 0.8)

    percent_above_50 = (above_50_percent / total_pixels) * 100
    percent_above_80 = (above_80_percent / total_pixels) * 100
    return percent_above_50, percent_above_80


def extract_stats_from_last_matching_row(file_path, resolution):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Reverse the DataFrame and find the first row with the matching resolution
    reversed_df = df.iloc[::-1]  # Reverse the DataFrame
    matching_row = reversed_df[reversed_df['Resolution'] == resolution].head(1)

    # Check if a matching row was found
    if matching_row.empty:
        print(f"No data found for resolution: {resolution}")
        return None, None, None
    else:
        print(f"Last matching row for resolution {resolution}:\n", matching_row)
        max_probability = matching_row["Max Probability"].values[0]
        percent_above_50 = matching_row["Percent Above 50%"].values[0]
        percent_above_80 = matching_row["Percent Above 80%"].values[0]
        
        print(f"Values for resolution {resolution}:")
        print("Max Probability:", max_probability)
        print("Percent Above 50%:", percent_above_50)
        print("Percent Above 80%:", percent_above_80)
        
        return max_probability, percent_above_50, percent_above_80