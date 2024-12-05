import argparse
import numpy as np
import os
import csv
import pandas as pd

from calculate_ai_confidence import calculate_confusion_matrix, calculate_normalized_displacement_in_50_and_80_regions, read_tif
from data_utils import extract_stats_from_last_matching_row
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

    # Run the confusion matrix calculation
    results = calculate_confusion_matrix(displacement_map, ai_output_map)

    # Print the results
    print("Confusion Matrix Results:")
    print(results)

    TP_uplift_50 = results[50]["Uplift"]["TP"]
    TP_subsidence_50 = results[50]["Subsidence"]["TP"]
    TP_uplift_80 = results[80]["Uplift"]["TP"]
    TP_subsidence_80 = results[80]["Subsidence"]["TP"]

    TN_uplift_50 = results[50]["Uplift"]["TN"]
    TN_subsidence_50 = results[50]["Subsidence"]["TN"]
    TN_uplift_80 = results[80]["Uplift"]["TN"]
    TN_subsidence_80 = results[80]["Subsidence"]["TN"]

    FP_uplift_50 = results[50]["Uplift"]["FP"]
    FP_subsidence_50 = results[50]["Subsidence"]["FP"]
    FP_uplift_80 = results[80]["Uplift"]["FP"]
    FP_subsidence_80 = results[80]["Subsidence"]["FP"]

    FN_uplift_50 = results[50]["Uplift"]["FN"]
    FN_subsidence_50 = results[50]["Subsidence"]["FN"]
    FN_uplift_80 = results[80]["Uplift"]["FN"]
    FN_subsidence_80 = results[80]["Subsidence"]["FN"]


    accuracy_50 = (
        (
            TP_uplift_50 + TP_subsidence_50 + TN_uplift_50 + TN_subsidence_50
        ) / (
            TP_uplift_50 + TP_subsidence_50 + TN_uplift_50 + TN_subsidence_50 + FP_uplift_50 + FP_subsidence_50 + FN_uplift_50 + FN_subsidence_50
        )
    )

    # sensitivity
    recall_50 = (
        (
            TP_uplift_50 + TP_subsidence_50
        ) / (
            TP_uplift_50 + TP_subsidence_50 + FN_uplift_50 + FN_subsidence_50
        )
    )

    # positive predictive value
    precision_50 = (
        (
            TP_uplift_50 + TP_subsidence_50
        ) / (
            TP_uplift_50 + TP_subsidence_50 + FP_uplift_50 + FP_subsidence_50
        )
    )

    accuracy_80 = (
        (
            TP_uplift_80 + TP_subsidence_80 + TN_uplift_80 + TN_subsidence_80
        ) / (
            TP_uplift_80 + TP_subsidence_80 + TN_uplift_80 + TN_subsidence_80 + FP_uplift_80 + FP_subsidence_80 + FN_uplift_80 + FN_subsidence_80
        )
    )

    # sensitivity
    recall_80 = (
        (
            TP_uplift_80 + TP_subsidence_80
        ) / (
            TP_uplift_80 + TP_subsidence_80 + FN_uplift_80 + FN_subsidence_80
        )
    )

    # positive predictive value
    precision_80 = (
        (
            TP_uplift_80 + TP_subsidence_80
        ) / (
            TP_uplift_80 + TP_subsidence_80 + FP_uplift_80 + FP_subsidence_80
        )
    )


    # Calculate the normalized displacement
    normalized_displacement_80, normalized_displacement_50 = calculate_normalized_displacement_in_50_and_80_regions(displacement_map, ai_output_map)

    # extract "Max Probability", "Percent Above 50%", "Percent Above 80%" with the input resolution from csv file
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
        "Normalized Displacement (50% Threshold)": normalized_displacement_50,
        "Normalized Displacement (80% Threshold)": normalized_displacement_80,
        "[AI] Max Probability": max_prob,
        "[AI] Percent Above 50%": percent_above_50,
        "[AI] Percent Above 80%": percent_above_80,
        "Accuracy (50% Threshold)": accuracy_50,
        "Recall (50% Threshold)": recall_50,
        "Precision (50% Threshold)": precision_50,
        "Accuracy (80% Threshold)": accuracy_80,
        "Recall (80% Threshold)": recall_80,
        "Precision (80% Threshold)": precision_80
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
                "Normalized Displacement (50% Threshold)",
                "Normalized Displacement (80% Threshold)",
                "[AI] Max Probability",
                "[AI] Percent Above 50%",
                "[AI] Percent Above 80%",
                "Accuracy (50% Threshold)",
                "Recall (50% Threshold)",
                "Precision (50% Threshold)",
                "Accuracy (80% Threshold)",
                "Recall (80% Threshold)",
                "Precision (80% Threshold)"
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
            stats["Normalized Displacement (50% Threshold)"],
            stats["Normalized Displacement (80% Threshold)"],
            stats["[AI] Max Probability"],
            stats["[AI] Percent Above 50%"],
            stats["[AI] Percent Above 80%"],
            stats["Accuracy (50% Threshold)"],
            stats["Recall (50% Threshold)"],
            stats["Precision (50% Threshold)"],
            stats["Accuracy (80% Threshold)"],
            stats["Recall (80% Threshold)"],
            stats["Precision (80% Threshold)"]
            
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