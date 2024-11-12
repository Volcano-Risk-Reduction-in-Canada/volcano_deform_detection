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

def calculate_confidence_score(displacement_map, ai_output_map):
    print("CALC CONFIDENCE SCORE")
    # Create masks for the 80% and 50% confidence regions
    confidence_80_mask = ai_output_map >= 0.8
    confidence_50_mask = ai_output_map >= 0.5

    # Extract displacement values within the confidence regions
    displacement_80_region = displacement_map[confidence_80_mask]
    displacement_50_region = displacement_map[confidence_50_mask]

    # Calculate average displacement in each region
    average_displacement_80 = np.nanmean(displacement_80_region)  # Use nanmean to ignore NaN values
    average_displacement_50 = np.nanmean(displacement_50_region)

    # Calculate the maximum absolute displacement in the entire map
    max_absolute_displacement = np.nanmax(np.abs(displacement_map))

    # Calculate confidence scores as a fraction of the maximum displacement (normalization)
    confidence_score_80 = average_displacement_80 / max_absolute_displacement
    confidence_score_50 = average_displacement_50 / max_absolute_displacement

    return confidence_score_80, confidence_score_50
