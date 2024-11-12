#!/usr/bin/python3
"""
Volcano Deformation Detection Using COMET pre-trained model

SPDX-License-Identifier: MIT

Copyright (C) 2021-2023 Government of Canada

Authors:
  - Drew Rotheram <drew.rotheram-clarke@nrcan-rncan.gc.ca>
"""

import csv
import os
import time
import gc
import sys
import numpy as np
import tensorflow as tf

from data_utils import get_percent_above_50_80
from osgeo import gdal, osr
from scipy.stats import norm
from skimage import morphology


def run_volcano_deformation_detection_single(image_file_name, site, beam, model, latlong, resolution):
    """
    Run Volcano Deformation Detection
    
    Parameters
    - image_file_name: str - name of InSAR image
    - site:
    - beam: 
    - model: str - Path to detection model
    - latlong: boolean - coordinate system used (Latitude/Longitude = True, UTM = False)
    - resolution: int (multiple of 5, between 5 and 100 inclusive) - the resolution of the image
    """
    image_name = image_file_name.split('.')[0]
    image_path = os.path.join(
        "wrp_images",
        site,
        beam,
        image_file_name
    )
    output_directory = os.path.join(
        "probability_map",
        site,
        beam,
        image_name,
        'latlong' if latlong else 'utm',
        'm1' if model == 'models/model1.pd' else 'm2'
    )
    ai_rgb_probmap = os.path.join(
        output_directory,
        f"{resolution}",
        f"{image_name}_rgb_probmap.tif"
    )
    print(ai_rgb_probmap, os.path.isfile(ai_rgb_probmap))
    if not os.path.exists(ai_rgb_probmap):
        start = time.time()

        os.makedirs(output_directory, exist_ok=True)

        #mean of imagenet dataset in BGR
        imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
        # crop input size
        overlapRatio = 1./4.
        hpatch = 227
        wpatch = 227
        hgap = int(float(hpatch)*overlapRatio)
        wgap = int(float(wpatch)*overlapRatio)

        # Create weight for each patch
        a = norm(hpatch/2, hpatch/6).pdf(np.arange(hpatch))
        b = norm(wpatch/2, wpatch/6).pdf(np.arange(wpatch))
        wmap = np.matmul(a[np.newaxis].T,b[np.newaxis])
        wmap = wmap/wmap.sum()

        # Import image with GDAL
        ds = gdal.Open(image_path,
                    gdal.GA_ReadOnly)
        
        # Project to UTM for consistent pixel spacing
        utm_zone = int(1+(ds.GetGeoTransform()[0]+180.0)/6.0)
        south = False if ds.GetGeoTransform()[3] > 0 else True
        epsg_code = 32600 + utm_zone + (100 if south else 0)
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(epsg_code)

        # Reset the graph before loading each model
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            with tf.io.gfile.GFile(model, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
        
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

            x = sess.graph.get_tensor_by_name('data:0')
            out = sess.graph.get_tensor_by_name('softmax:0')

            os.makedirs(f"{output_directory}/{resolution}", exist_ok=True)

            # Resample image to 100m x 100m equivalent decimel degrees
            warp_options = gdal.WarpOptions(
                format='MEM',
                xRes=resolution,
                yRes=resolution,
                dstSRS=f'+init=epsg:{4326}' if latlong else f'+init=epsg:{epsg_code}',
                srcNodata=0,
                resampleAlg=gdal.gdalconst.GRA_Average,
            )
            warp_ds = gdal.Warp('', ds, options=warp_options)
            img_array = warp_ds.ReadAsArray()
            img_array [img_array == 0] = np.nan
            mask  = img_array == 0
            seedmask = morphology.disk(5)
            mask  = morphology.binary_closing(~mask, seedmask)

            # convert from phase raster to grayscale image
            img_array = (img_array + np.pi)/(2*np.pi) * 255
            img_array = np.dstack((img_array, img_array, img_array))

            # subtract ImageNet mean
            img = img_array - imagenet_mean

            # break image into overlapping patches and run through model
            himg, wimg = img.shape[:2]
            weightMap = np.zeros((himg, wimg), np.float32) + 0.00001
            probMap = np.zeros((himg,wimg), np.float32)

            for starty in np.concatenate((np.arange(0, himg - hpatch, hgap), np.array([himg - hpatch])), axis=0):
                for startx in np.concatenate((np.arange(0, wimg - wpatch, wgap), np.array([wimg - wpatch])), axis=0):
                    crop_img = img[starty:starty + hpatch, startx:startx + wpatch]
                    curmask = mask[starty:starty + hpatch, startx:startx + wpatch]

                    weightMap[starty:starty + hpatch, startx:startx + wpatch] += wmap
                    
                    testimg = crop_img + imagenet_mean
                    testimg[testimg!=0.] = 1.

                    if ((testimg.sum() / hpatch / wpatch / 3) > 0.5) and ((curmask.sum() / hpatch / wpatch) > 0.25):
                        # Reshape as needed to feed into model
                        crop_img = np.transpose(crop_img, (2, 0, 1))
                        crop_img = crop_img.reshape((1, 3, 227, 227))
                        # Run the session and calculate the class probability
                        probs = sess.run(out, feed_dict={x: crop_img})
                        #  Put in prob map
                        if np.isnan(probs[0,0]):
                            probs[0,0] = 0.0
                        probMap[starty:starty + hpatch, startx:startx + wpatch] += probs[0,0]* wmap * (testimg.sum()/ hpatch/ wpatch / 3) 

            # Normalised weight
            probMap /= weightMap

            process_output_files(image_name, img_array, probMap, f"{output_directory}", warp_ds, resolution)

            endt = time.time()
            print("time elapsed:" + str(endt - start))

            # Explicitly close datasets to free memory
            warp_ds = None
            
            # Force garbage collection
            gc.collect()
    else:
        print(f"File already exists: {ai_rgb_probmap} - Skipping")
        return

def process_output_files(image_name, img_array, probMap, output_directory, warp_ds, resolution):
    """Helper function to process and write output files."""
    # Calculate percentages of pixels above 50% and 80% probability
    percent_above_50, percent_above_80 = get_percent_above_50_80(probMap)

    # Set the file path
    file_path = os.path.join(output_directory, f'{image_name}_probability.csv')

    # Open file in append mode
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if os.stat(file_path).st_size == 0:
            writer.writerow(["Resolution", "Max Probability", "Percent Above 50%", "Percent Above 80%"])

        writer.writerow([
            resolution, 
            probMap.max() * 100,
            percent_above_50,
            percent_above_80
        ])

    if probMap.max() > 0.1:
        im_scale = img_array/255.
        im_scale[:,:,2] = im_scale[:,:,2]*(1-probMap) + probMap
        im_scale[:,:,1] = im_scale[:,:,1]*(1-probMap) + probMap
        # Draw contour of high prob
        psbound = np.logical_and(probMap>0.5,probMap<0.525)
        im_scale[:,:,2] -= psbound
        im_scale[:,:,1] = im_scale[:,:,1]*(1-psbound) + 0.5*psbound
        im_scale[:,:,0] = im_scale[:,:,0]*(1-psbound) + 0.75*psbound
        psbound = np.logical_and(probMap>0.8,probMap<0.825)
        im_scale[:,:,0] -= psbound
        im_scale[:,:,1] += psbound
        im_scale[:,:,2] -= psbound
        # Cap values
        im_scale[im_scale<0] = 0.
        im_scale[im_scale>1] = 1.
        
        # Write RGB probmap image
        driver = gdal.GetDriverByName("GTiff")
        output_rgb = driver.Create(f'{output_directory}/{resolution}/{image_name}_rgb_probmap.tif',
                                    im_scale.shape[1],
                                    im_scale.shape[0],
                                    3,
                                    gdal.GDT_Byte,
                                    options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF',])
        output_rgb.SetProjection(warp_ds.GetProjection())
        output_rgb.SetGeoTransform(warp_ds.GetGeoTransform())
        output_rgb.GetRasterBand(1).WriteArray(im_scale[:, :, 2]*255 )
        output_rgb.GetRasterBand(1).FlushCache()
        output_rgb.GetRasterBand(1).SetNoDataValue(0)
        output_rgb.GetRasterBand(2).WriteArray(im_scale[:, :, 1]*255 )
        output_rgb.GetRasterBand(2).FlushCache()
        output_rgb.GetRasterBand(2).SetNoDataValue(0)
        output_rgb.GetRasterBand(3).WriteArray(im_scale[:, :, 0]*255 )
        output_rgb.GetRasterBand(3).FlushCache()
        output_rgb.GetRasterBand(2).SetNoDataValue(0)
        output_rgb = None

        print(probMap.shape)
        output_probmap = driver.Create(f'{output_directory}/{resolution}/{image_name}_probmap.tif',
                                    probMap.shape[1],
                                    probMap.shape[0],
                                    1,
                                    gdal.GDT_Float32)
        output_probmap.SetProjection(warp_ds.GetProjection())
        output_probmap.SetGeoTransform(warp_ds.GetGeoTransform())
        output_probmap.GetRasterBand(1).WriteArray(probMap)
        output_probmap.GetRasterBand(1).FlushCache()
        output_probmap.GetRasterBand(1).SetNoDataValue(0)
        output_probmap = None