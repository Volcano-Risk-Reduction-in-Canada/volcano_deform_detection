import argparse
import numpy as np
import os
import csv
import pandas as pd

from calculate_ai_confidence import calculate_confidence_score, read_tif
from get_probability_map_func_single_resolution import run_volcano_deformation_detection_single
from osgeo import gdal
from datetime import datetime


from get_disp_and_wrp_images_from_s3 import download_images_from_s3

def main():
    args = parse_args()
    output_disp_stats_dir = f"disp_stats"
    latlong = args.resolution < 1
    image_name_set = download_images_from_s3(
        args.bucket_name,
        args.site,
        args.beam,
        args.disp_dir,
        args.wrp_dir
    )
    # generate ai model outputs
    os.makedirs(output_disp_stats_dir, exist_ok=True)

    for image in image_name_set:
        run_volcano_deformation_detection_single(f"{image}.adf.wrp.geo.tif", args.site, args.beam, f"models/model{args.model}.pd", latlong, args.resolution)

    for image in image_name_set:
        stats = get_disp_image_statistics(args.site, args.beam, image, latlong, args.resolution, args.model)
        write_stats_to_csv(args.site, args.beam, image, output_disp_stats_dir, stats)


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

def get_disp_image_statistics(site, beam, image_name, latlong, resolution, model):
    print("GETTING DISP IMG STATS")
    file_path = os.path.join("disp_images", site, beam, f"{image_name}.disp.geo.tif")
    ai_output_folder = os.path.join(
        'probability_map',
        site,
        beam,
        image_name,
        'latlong' if latlong else 'utm',
        f'm{model}'
    )
    ai_output_path = os.path.join(
        ai_output_folder,
        f'{resolution}',
        f'{image_name}_probmap.tif'
    )
    # Open the TIF file using GDAL
    dataset = gdal.Open(file_path)
    if not dataset:
        raise FileNotFoundError(f"Unable to open file: {file_path}")
    
    # Read the first band
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    
    # Mask out any NoData values
    nodata_value = band.GetNoDataValue()
    if nodata_value is not None:
        array = np.ma.masked_equal(array, nodata_value)
    
    # Calculate statistics
    min_value = np.min(array)
    max_value = np.max(array)
    abs_max_value = np.max(np.abs(array))  # Absolute maximum value
    value_range = max_value - min_value
    percentiles = np.percentile(array.compressed(), [25, 50, 75])  # 25th, 50th (median), and 75th percentiles

    # ai confidence

    # Read the AI output and displacement maps
    ai_output_map = read_tif(ai_output_path, resolution, latlong)
    # displacement map is always in Longitude/Latitude
    displacement_map = read_tif(file_path, resolution, True)

    # Calculate the confidence score
    confidence_score_80, confidence_score_50 = calculate_confidence_score(displacement_map, ai_output_map)

    # extract "Max Probability", "Percent Above 50%", "Percent Above 80%" with the input resolution from csv file
    # Example usage
    ai_csv_path = os.path.join(
        ai_output_folder,
        f'{image_name}_probability.csv'
    )
    max_prob, percent_above_50, percent_above_80 = extract_stats_from_last_matching_row(ai_csv_path, resolution)

    # Print or return the results
    stats = {
        "Minimum Value": min_value,
        "Maximum Value": max_value,
        "Absolute Maximum Value": abs_max_value,
        "Range": value_range,
        "25th Percentile": percentiles[0],
        "50th Percentile (Median)": percentiles[1],
        "75th Percentile": percentiles[2],
        "Confidence Score (50% Threshold)": confidence_score_50,
        "Confidence Score (80% Threshold)": confidence_score_80,
        "[AI] Max Probability": max_prob,
        "[AI] Percent Above 50%": percent_above_50,
        "[AI] Percent Above 80%": percent_above_80
    }
    
    return stats

def write_stats_to_csv(site, beam, image_name, output_disp_stats_dir, stats):
    print("WRITING TO CSV")
    # Set the file path
    file_path = os.path.join(output_disp_stats_dir, f'{site}_{beam}_stats.csv')
    # Assuming image_name is YYYYMMDD_HH_YYYYMMDD_HH
    # Split by '_HH_'
    start_date, end_date_with_hh = image_name.split('_HH_')[:2]
    # Remove '_HH' from the end date
    end_date = end_date_with_hh.split('_HH')[0]
    # Open file in append mode
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if os.stat(file_path).st_size == 0:
            writer.writerow([
                "Start Date",
                "End Date",
                "Minimum Value",
                "Maximum Value",
                "Absolute Maximum Value",
                "Range",
                "25th Percentile",
                "50th Percentile (Median)",
                "75th Percentile",
                "Confidence Score (50% Threshold)",
                "Confidence Score (80% Threshold)",
                "[AI] Max Probability",
                "[AI] Percent Above 50%",
                "[AI] Percent Above 80%"
            ])

        writer.writerow([
            datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d"),
            datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d"),
            stats["Minimum Value"],
            stats["Maximum Value"],
            stats["Absolute Maximum Value"],
            stats["Range"],
            stats["25th Percentile"],
            stats["50th Percentile (Median)"],
            stats["75th Percentile"],
            stats["Confidence Score (50% Threshold)"],
            stats["Confidence Score (80% Threshold)"],
            stats["[AI] Max Probability"],
            stats["[AI] Percent Above 50%"],
            stats["[AI] Percent Above 80%"]
        ])


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Download paired .disp.geo.tif and .adf.wrp.geo.tif images from S3")
    parser.add_argument("--bucket_name",
                        type=str,
                        help="Name of S3 Bucket",
                        required=True),
    parser.add_argument("--site",
                        type=str,
                        help="Site folder in S3 bucket",
                        required=True),
    parser.add_argument("--beam",
                        type=str,
                        help="Beam folder in S3 bucket",
                        required=True),
    parser.add_argument("--disp_dir",
                        type=str,
                        help="Local directory to save disp images",
                        required=True),
    parser.add_argument("--wrp_dir",
                        type=str,
                        help="Local directory to save wrp images",
                        required=True),
    parser.add_argument("--model",
                        type=int,
                        help="which AI Model to use",
                        required=True),
    parser.add_argument("--resolution",
                        type=int,
                        help="which resolution to use ",
                        required=True),
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()