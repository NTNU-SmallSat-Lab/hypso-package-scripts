#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create dataset for IGARSS Chl-a PLSR estimation

Author: Cameron Penne
Date: 2024-01-06
"""

import sys
sys.path.insert(0, '/home/cameron/Projects/hypso-package')

from hypso import Hypso1
import os
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from datetime import datetime
import csv
from pyresample import load_area
import glob
from satpy import Scene
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler
from pyresample.bilinear.xarr import XArrayBilinearResampler 
import pickle



def decode_labels(file_path):
    # Open the binary file and read its content
    with open(file_path, 'rb') as fileID:
        fileContent = fileID.read()
    # Extract the required values from the binary data
    classification_execution_time = int.from_bytes(fileContent[0:4], byteorder='little', signed=True)
    loading_execution_time = int.from_bytes(fileContent[4:8], byteorder='little', signed=True)
    classes_holder = fileContent[8:24]
    labels_holder = fileContent[24:]
    classes = []
    labels = []
    # Decode the labels and convert them back to original classes.
    for i in range(len(classes_holder)):
        if classes_holder[i] != 255:
            classes.append(classes_holder[i])
    if len(classes) <= 2:
        for i in range(len(labels_holder)):
            pixel_str = format(labels_holder[i], '08b')
            for j in range(8):
                labels.append(int(pixel_str[j]))
    if 2 < len(classes) <= 4:
        for i in range(len(labels_holder)):
            pixel_str = format(labels_holder[i], '08b')
            for j in range(4):
                labels.append(int(pixel_str[2 * j:2 * j + 2], 2))
    if 4 < len(classes) <= 16:
        for i in range(len(labels_holder)):
            pixel_str = format(labels_holder[i], '08b')
            for j in range(2):
                labels.append(int(pixel_str[4 * j:4 * j + 4], 2))
    # Corrected label conversion
    for i in range(len(labels)):
        labels[i] = classes[labels[i]]
    # Save 'labels' as a CSV file with a comma delimiter
    # with open('labels.csv', 'w') as csv_file:
    #    csv_file.write(','.join(map(str, labels)))
    return labels

def sanitize_data_array(data_array):

    for attr, value in data_array.attrs.items():
        if value is None:
            data_array.attrs[attr] = 'None'

    return data_array

labels = {'water': 0,
        'strange_water': 1,
        'light_forest': 2,
        'dark_forest': 3,
        'urban': 4,
        'rock': 5,
        'ice': 6,
        'sand': 7,
        'thick_clouds': 8,
        'thin_clouds': 9,
        'shadows': 10}


captures_csv_path = '/home/cameron/Dropbox/IGARSS/captures.csv'

entry_list = read_csv(captures_csv_path)
points_file_base_dir = "/home/cameron/Projects/hypso1-qgis-gcps/png/bin3"

area_def = load_area("/home/cameron/Dropbox/IGARSS/frohavet.yaml")

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "dataset")

for entry in entry_list:

    hypso_path = entry['hypso']
    sentinel_path = entry['sentinel']

    parts = hypso_path.rstrip('/').split('/')
    capture_target_name = parts[-2]
    timestamp = parts[-1].split('_')[1]

    base_labels_path = os.path.join(hypso_path, "processing-temp")

    for item in os.listdir(base_labels_path):
        if os.path.isdir(os.path.join(base_labels_path, item)) and item.startswith(capture_target_name):
            subdir_name = item
            break

    labels_path = os.path.join(base_labels_path, subdir_name, "labels.bin")
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H-%M-%SZ")
    new_timestamp = dt.strftime("%Y-%m-%d_%H%MZ")
    points_file = os.path.join(points_file_base_dir, capture_target_name, f"{capture_target_name}_{new_timestamp}-bin3.points")

    name = f"{capture_target_name}_{timestamp}-l1a"
    l1a_nc_file = os.path.join(hypso_path, name + '.nc')

    print("Hypso Path: ", hypso_path)
    print("Labels Path: ", labels_path)
    print("Points File: ", points_file)
    print("Sentinel Path: ", sentinel_path)
    #input("Pause...")

    satobj = Hypso1(path=l1a_nc_file, verbose=True)
    satobj.load_points_file(path=points_file, image_mode='standard', origin_mode='cube')


    satobj.flip_l1a_cube()
    satobj.generate_geometry()
    satobj.generate_l1b_cube()





    labels_data = decode_labels(labels_path)
    lsc = np.array(labels_data).reshape(satobj.spatial_dimensions)

    thick_cloud_key = labels['thick_clouds']
    thin_cloud_key = labels['thin_clouds']
    water_key = labels['water']
    ice_key = labels['ice']

    cloud_mask = ((lsc == thick_cloud_key) | (lsc == thin_cloud_key))
    land_mask = ~(lsc == water_key)

    satobj.cloud_mask = cloud_mask
    satobj.land_mask = land_mask

    # Must run cloud and land mask code before this part
    satobj.generate_toa_reflectance()
    #satobj.generate_l2a_cube('machi')

    #z = satobj.l2a_cube[:,:,30].to_numpy()
    #plt.imshow(z, interpolation='nearest')
    #plt.savefig('TEST_l2a_'+str(30)+'.png')

    #print(type(satobj.toa_reflectance_cube))
    #input("pause")



    #resampled_hypso = satobj.resample_l2a_cube(area_def=area_def)
    resampled_hypso = satobj.resample_toa_reflectance_cube(area_def=area_def)

    #for band in range(0,120):
    #    plt.imsave('resampled_l2a_'+str(band)+'.png', resampled_hypso[:,:,band])




    #plt.imsave(os.path.join(output_dir, satobj.capture_name + "_l2a.png"), resampled_hypso[:,:,40] )

    plt.imshow(resampled_hypso[:,:,40])
    plt.savefig(os.path.join(output_dir, satobj.capture_name + "_l2a.png"))
    plt.clf()

    '''
    from pyresample.kd_tree import XArrayResamplerNN

    swath_def = satobj._generate_swath_definition()
    nnrs = XArrayResamplerNN(source_geo_def=swath_def, 
                            target_geo_def=area_def,
                            radius_of_influence=1000,
                            neighbours=1,
                            epsilon=0)
    
    nnrs.get_neighbour_info()

    data = satobj.l2a_cube
    num_bands = 120
    resampled_hypso = np.zeros((area_def.shape[0], area_def.shape[1], num_bands))
    resampled_hypso = xr.DataArray(resampled_hypso, dims=satobj.dim_names_3d)
    resampled_hypso.attrs.update(data.attrs)

    for band in range(0,num_bands):
        resampled_hypso[:,:,band] = nnrs.get_sample_from_neighbour_info(data=data[:,:,band], 
                                                                        fill_value=np.nan)
 

    z = resampled_hypso[:,:,30].to_numpy()
    plt.imshow(z, interpolation='nearest')
    plt.savefig('resampled_l2a_'+str(30)+'.png')
    #plt.clf()

    input("pause")
    '''


    '''
    brs = XArrayBilinearResampler(source_geo_def=swath_def, target_geo_def=area_def, radius_of_influence=50000)
    brs.get_bil_info()

    data = satobj.l2a_cube
    num_bands = 120
    resampled_hypso = np.zeros((area_def.shape[0], area_def.shape[1], num_bands))
    resampled_hypso = xr.DataArray(resampled_hypso, dims=satobj.dim_names_3d)
    resampled_hypso.attrs.update(data.attrs)

    for band in range(0,num_bands):
        resampled_hypso[:,:,band] = brs.get_sample_from_bil_info(data=data[:,:,band], fill_value=np.nan, output_shape=area_def.shape)
        plt.imsave('out_'+str(band)+'.png', resampled_hypso[:,:,band])
        plt.imsave('l2a_'+str(band)+'.png', satobj.l2a_cube[:,:,band])
        plt.imsave('l1b_'+str(band)+'.png', satobj.l1b_cube[:,:,band])
        plt.imsave('l1a_'+str(band)+'.png', satobj.l1a_cube[:,:,band])
    '''


    filenames = []
    filenames = filenames + glob.glob(sentinel_path + '/geo_coordinates.nc')
    filenames = filenames + glob.glob(sentinel_path + '/chl_nn.nc')
    sentinel_scene = Scene(filenames=filenames, reader='olci_l2')
    sentinel_scene.load(['chl_nn'])


    swath_def = satobj._generate_swath_definition()
    nnrs = KDTreeNearestXarrayResampler(source_geo_def=swath_def, target_geo_def=area_def)
    resampled_unified_mask = nnrs.resample(satobj.unified_mask, fill_value=np.nan, radius_of_influence=None, epsilon=0)


    resampled_sentinel_scene = sentinel_scene.resample(area_def, resampler='bilinear', fill_value=np.NaN)
    #resampled_sentinel = resampled_sentinel_scene.to_xarray()
    resampled_sentinel = resampled_sentinel_scene['chl_nn']


    plt.imshow(resampled_sentinel)
    plt.savefig(os.path.join(output_dir, satobj.capture_name + "_chl.png"))
    plt.clf()


    print(type(resampled_unified_mask))
    print(type(resampled_hypso))
    print(type(resampled_sentinel))


    #plt.imshow(resampled_hypso[:,:,40])


    basename = satobj.capture_name

    resampled_mask_path = os.path.join(output_dir, basename + "_mask.pkl")
    resampled_hypso_path = os.path.join(output_dir, basename + "_hypso.pkl")
    resampled_sentinel_path = os.path.join(output_dir, basename + "_sentinel.pkl")


    with open(resampled_mask_path, 'wb') as file:
        pickle.dump(resampled_unified_mask, file)

    with open(resampled_hypso_path, 'wb') as file:
        pickle.dump(resampled_hypso, file)

    with open(resampled_sentinel_path, 'wb') as file:
        pickle.dump(resampled_sentinel.to_numpy(), file)



    #input("Press Enter to continue...")
