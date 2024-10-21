#!/usr/bin/python3
"""
Volcano Deformation Detection Using COMET pre-trained model

SPDX-License-Identifier: MIT

Copyright (C) 2021-2023 Government of Canada

Authors:
  - Drew Rotheram <drew.rotheram-clarke@nrcan-rncan.gc.ca>
"""

import argparse
import csv
import glob
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from osgeo import gdal, osr
from scipy.stats import norm
from skimage import morphology


def main():
    start = time.time()
    args = parse_args()



    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

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
    ds = gdal.Open(args.image_path,
                   gdal.GA_ReadOnly)
    
    # Project to UTM for consistent pixel spacing
    utm_zone = int(1+(ds.GetGeoTransform()[0]+180.0)/6.0)
    south = False if ds.GetGeoTransform()[3] > 0 else True
    epsg_code = 32600 +utm_zone
    if south is True:
        epsg_code += 100
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(epsg_code)

    # Resample image to 100m x 100m equivalent decimel degrees
    warp_options = gdal.WarpOptions(
        format='MEM',
        xRes=5,
        yRes=5,
        dstSRS=f'+init=epsg:{epsg_code}',
        srcNodata=0,
        resampleAlg=gdal.gdalconst.GRA_Average,
    )
    warp_ds = gdal.Warp('',
                        ds,
                        options=warp_options)
    img_array = warp_ds.ReadAsArray()
    img_array [img_array == 0] = np.nan
    mask  = img_array == 0
    seedmask = morphology.disk(5)
    mask  = morphology.binary_closing(~mask, seedmask)

    # convert from phase raster to grayscale image
    img_array = (img_array + np.pi)/(2*np.pi)*255
    img_array = np.dstack((img_array, img_array, img_array))

    # subtract ImageNet mean
    img = img_array - imagenet_mean

    # break image into overlapping patches and run through model
    himg = img.shape[0]
    wimg = img.shape[1]
    weightMap = np.zeros((himg,wimg),np.float32) + 0.00001
    probMap = np.zeros((himg,wimg),np.float32)

    with tf.compat.v1.Session() as sess:
        with tf.io.gfile.GFile(args.model, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
    
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        x = sess.graph.get_tensor_by_name('data:0')
        out = sess.graph.get_tensor_by_name('softmax:0')

        path = os.path.normpath(args.image_path)

        file=open(os.path.join(args.output_directory,
                f'{os.path.split(args.image_path)[1].split(".")[0]}_probability.csv'),
                'w',
                newline='')
        writer = csv.writer(file)

        for starty in np.concatenate((np.arange(0, himg-hpatch,hgap), np.array([himg-hpatch])), axis=0):
            for startx in np.concatenate((np.arange(0, wimg-wpatch, wgap), np.array([wimg-wpatch])), axis=0):
                crop_img = img[starty:starty+hpatch,startx:startx+wpatch]
                curmask = mask[starty:starty+hpatch,startx:startx+wpatch]
                
                weightMap[starty:starty+hpatch,startx:startx+wpatch] += wmap
                
                testimg = crop_img + imagenet_mean
                testimg[testimg!=0.] = 1.

                if ((testimg.sum()/hpatch/wpatch/3) > 0.5) and ((curmask.sum()/hpatch/wpatch) > 0.25):
                    # Reshape as needed to feed into model
                    crop_img = np.transpose(crop_img, (2, 0, 1))
                    crop_img = crop_img.reshape((1,3, 227,227))
                    # Run the session and calculate the class probability
                    
                    probs = sess.run(out, feed_dict={x: crop_img})
                    #  Put in prob map
                    if np.isnan(probs[0,0]):
                        probs[0,0] = 0.0
                    probMap[starty:starty+hpatch, startx:startx+wpatch] += probs[0,0]*wmap*(testimg.sum()/hpatch/wpatch/3) 

        # Normalised weight
        probMap /= weightMap
        
        # record max prob
        filename = os.path.split(args.image_path)[1].split(".")[0]
        writer.writerow([filename,
                        probMap.max()])

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
            output_rgb = driver.Create(f'{args.output_directory}/{os.path.split(args.image_path)[1].split(".")[0]}_rgb_probmap.tif',
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
            output_probmap = driver.Create(f'{args.output_directory}/{os.path.split(args.image_path)[1].split(".")[0]}_probmap.tif',
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


            endt = time.time()
            print("time elapsed:" + str(endt - start))
            

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Volcano Deformation Detection")
    parser.add_argument("--image_path",
                        type=str,
                        help="Path to InSAR image",
                        required=True),
    parser.add_argument("--output_directory",
                        type=str,
                        help="Path to output directory",
                        default="probability_map/",
                        required=True),
    parser.add_argument("--model",
                        type=str,
                        help="Path to detection model",
                        default="models/model2.pd",
                        required=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
