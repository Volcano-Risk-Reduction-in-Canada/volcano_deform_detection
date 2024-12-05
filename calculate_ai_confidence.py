import numpy as np
from osgeo import gdal


def read_tif(file_path, resolution, latlong):
    print("READ TIF", latlong)
    # Open the TIF file and resample to the desired resolution
    ds = gdal.Open(file_path, gdal.GA_ReadOnly)
    # Check if the dataset opened correctly
    if ds is None:
        raise ValueError(f"Could not open file {file_path}")
    
    # Print dataset dimensions
    print("Dataset dimensions:", ds.RasterXSize, "x", ds.RasterYSize)
    if ds.RasterXSize == 0 or ds.RasterYSize == 0:
        raise ValueError("The input dataset has zero dimensions.")

    if latlong:
        print('lATLONG')
        print(ds.GetGeoTransform())
        utm_zone = int(1+(ds.GetGeoTransform()[0]+180.0)/6.0)
        south = ds.GetGeoTransform()[3] <= 0
        print(utm_zone, south)
        epsg_code = 32600 + utm_zone + (100 if south else 0)
        print(epsg_code)
        # 4326
        # espg = f'EPSG:{4326}'
        espg = f'EPSG:{epsg_code}'
        warp_options = gdal.WarpOptions(
            format='MEM',
            xRes=resolution,
            yRes=resolution,
            dstSRS=espg,
            srcNodata=0,
            resampleAlg=gdal.gdalconst.GRA_Average,
        )
    else:
        # No coordinate transformation, but still apply resolution and resampling
        warp_options = gdal.WarpOptions(
            format='MEM',
            xRes=resolution,
            yRes=resolution,
            srcNodata=0,
            resampleAlg=gdal.gdalconst.GRA_Average,
        )
    warp_ds = gdal.Warp('', ds, options=warp_options)
    if warp_ds is None:
        raise ValueError("Warping failed. Check your warp options.")
    img_array = warp_ds.ReadAsArray()
    img_array[img_array == 0] = np.nan  # Set NoData values to NaN

    return img_array

def calculate_normalized_displacement_in_50_and_80_regions(displacement_map, ai_output_map):
    print("CALC CONFIDENCE SCORE")
    # Create masks for the 80% and 50% confidence regions
    disp_80_mask = ai_output_map >= 0.8
    disp_50_mask = ai_output_map >= 0.5

    # Extract displacement values within the confidence regions
    displacement_80_region = displacement_map[disp_80_mask]
    displacement_50_region = displacement_map[disp_50_mask]

    # Calculate average displacement in each region
    average_displacement_80 = np.nanmean(displacement_80_region)  # Use nanmean to ignore NaN values
    average_displacement_50 = np.nanmean(displacement_50_region)

    # Calculate the maximum absolute displacement in the entire map
    max_absolute_displacement = np.nanmax(np.abs(displacement_map))

    # Calculate confidence scores as a fraction of the maximum displacement (normalization)
    normalized_displacement_80 = average_displacement_80 / max_absolute_displacement
    normalized_displacement_50 = average_displacement_50 / max_absolute_displacement

    # normalized measure of displacement within high-confidence regions
    # an indicator of how significant the ground movement is in regions flagged by the model
    return normalized_displacement_80, normalized_displacement_50


def calculate_confusion_matrix(displacement_map, ai_prob_map, uplift_threshold=0, subsidence_threshold=0):
    # Define ground truth areas based on the displacement map
    uplift_area = displacement_map > uplift_threshold  # TRUE uplift area
    subsidence_area = displacement_map < subsidence_threshold  # TRUE subsidence area
    
    # Define AI model regions (50% and 80% thresholds)
    confidence_50_area = ai_prob_map >= 0.5  # AI 50% confidence area (blue circle)
    confidence_80_area = ai_prob_map >= 0.8  # AI 80% confidence area (green circle)

    # Calculate TP, FP, TN, and FN for the 50% confidence area
    tp_50_uplift = np.sum((confidence_50_area & uplift_area))  # Uplift TP for 50%
    fp_50_uplift = np.sum((confidence_50_area & ~uplift_area))  # Uplift FP for 50%
    tn_50_uplift = np.sum((~confidence_50_area & ~uplift_area))  # Uplift TN for 50%
    fn_50_uplift = np.sum((~confidence_50_area & uplift_area))  # Uplift FN for 50%

    tp_50_subsidence = np.sum((confidence_50_area & subsidence_area))  # Subsidence TP for 50%
    fp_50_subsidence = np.sum((confidence_50_area & ~subsidence_area))  # Subsidence FP for 50%
    tn_50_subsidence = np.sum((~confidence_50_area & ~subsidence_area))  # Subsidence TN for 50%
    fn_50_subsidence = np.sum((~confidence_50_area & subsidence_area))  # Subsidence FN for 50%

    # Calculate TP, FP, TN, and FN for the 80% confidence area
    tp_80_uplift = np.sum((confidence_80_area & uplift_area))  # Uplift TP for 80%
    fp_80_uplift = np.sum((confidence_80_area & ~uplift_area))  # Uplift FP for 80%
    tn_80_uplift = np.sum((~confidence_80_area & ~uplift_area))  # Uplift TN for 80%
    fn_80_uplift = np.sum((~confidence_80_area & uplift_area))  # Uplift FN for 80%

    tp_80_subsidence = np.sum((confidence_80_area & subsidence_area))  # Subsidence TP for 80%
    fp_80_subsidence = np.sum((confidence_80_area & ~subsidence_area))  # Subsidence FP for 80%
    tn_80_subsidence = np.sum((~confidence_80_area & ~subsidence_area))  # Subsidence TN for 80%
    fn_80_subsidence = np.sum((~confidence_80_area & subsidence_area))  # Subsidence FN for 80%

    # Results in dictionary format for better readability
    results = {
        50: {
            "Uplift": {"TP": tp_50_uplift, "FP": fp_50_uplift, "TN": tn_50_uplift, "FN": fn_50_uplift},
            "Subsidence": {"TP": tp_50_subsidence, "FP": fp_50_subsidence, "TN": tn_50_subsidence, "FN": fn_50_subsidence},
        },
        80: {
            "Uplift": {"TP": tp_80_uplift, "FP": fp_80_uplift, "TN": tn_80_uplift, "FN": fn_80_uplift},
            "Subsidence": {"TP": tp_80_subsidence, "FP": fp_80_subsidence, "TN": tn_80_subsidence, "FN": fn_80_subsidence},
        }
    }

    return results