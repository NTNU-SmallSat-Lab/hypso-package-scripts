# Development use only:
#import sys
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso/')



from hypso import Hypso
import os
import matplotlib.pyplot as plt

from hypso.satpy import get_l1a_satpy_scene, \
                        get_l1b_satpy_scene, \
                        get_l1c_satpy_scene, \
                        get_l1d_satpy_scene

from hypso.spectral_analysis import get_closest_wavelength_index
from hypso.geometry import compute_bbox
from hypso.geometry_definition import generate_area_def

#from satpy.composites import GenericCompositor
#from satpy.writers import to_image

from PIL import Image
#from pycoast import ContourWriterAGG

import numpy as np

from hypso.write import write_l1c_nc_file


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
import netCDF4 as nc
from hypso.geometry_definition import generate_hypso_swath_def

from hypso.classification import decode_jon_cnn_water_mask, decode_jon_cnn_land_mask, decode_jon_cnn_cloud_mask

from global_land_mask import globe

from scipy import linalg



class HypsoCapture:
    def __init__(self, nc_file=None):

        self.nc_file = nc_file


        dir_path = os.path.dirname(nc_file)
        print(dir_path)

        self.cnn_labels_file = os.path.join(dir_path, '/processing-temp', 'sea-land-cloud.labels')
        self.latitudes_file = os.path.join(dir_path, 'processing-temp', 'latitudes_indirectgeoref.dat')
        self.longitudes_file = os.path.join(dir_path, 'processing-temp', 'longitudes_indirectgeoref.dat')
        self.cnn_labels = os.path.join(dir_path, 'processing-temp', 'sea-land-cloud.labels')


#dir_path = '/home/cameron/Dokumenter/Data/'


#capture_1 = HypsoCapture('/home/cameron/Nedlastinger/gulfofcalifornia_2025-04-25T18-48-51Z/gulfofcalifornia_2025-04-25T18-48-51Z-l1a.nc')
#capture_2 = HypsoCapture('/home/cameron/Nedlastinger/gulfofcalifornia_2025-04-19T18-17-20Z/gulfofcalifornia_2025-04-19T18-17-20Z-l1a.nc')

capture_1 = HypsoCapture('/media/veracrypt3/HYPSO/aeronetvenice_2025-06-12T09-58-02Z/aeronetvenice_2025-06-12T09-58-02Z-l1a.nc')
capture_2 = HypsoCapture('/media/veracrypt3/HYPSO/aeronetvenice_2025-06-22T10-46-15Z/aeronetvenice_2025-06-22T10-46-15Z-l1a.nc')
bd_cloud_mask = False
mask_clouds = False

#capture_1 = HypsoCapture('/media/veracrypt3/HYPSO/aeronetvenice_2025-07-22T09-57-52Z/aeronetvenice_2025-07-22T09-57-52Z-l1a.nc')
#capture_2 = HypsoCapture('/media/veracrypt3/HYPSO/aeronetvenice_2025-07-23T10-02-32Z/aeronetvenice_2025-07-23T10-02-32Z-l1a.nc')

#capture_1 = HypsoCapture('/media/veracrypt3/HYPSO/oslofjord_2025-06-20T10-33-03Z/oslofjord_2025-06-20T10-33-03Z-l1a.nc')
#capture_2 = HypsoCapture('/media/veracrypt3/HYPSO/oslofjord_2025-06-27T11-06-40Z/oslofjord_2025-06-27T11-06-40Z-l1a.nc')
#bd_cloud_mask = False
#mask_clouds = False


#capture_1 = HypsoCapture('/media/veracrypt3/HYPSO/blackseabloom3_2025-07-22T08-22-09Z/blackseabloom3_2025-07-22T08-22-09Z-l1a.nc')
#capture_2 = HypsoCapture('/media/veracrypt3/HYPSO/blackseabloom3_2025-07-23T08-26-49Z/blackseabloom3_2025-07-23T08-26-49Z-l1a.nc')
#bd_cloud_mask = True
#mask_clouds = True


#capture_1 = HypsoCapture('/home/cameron/Nedlastinger/aeronetvenice_2025-06-12T09-58-02Z/aeronetvenice_2025-06-12T09-58-02Z-l1a.nc')
#capture_1.cnn_labels_file = '/home/cameron/Nedlastinger/aeronetvenice_2025-06-12T09-58-02Z/processing-temp/sea-land-cloud.labels'
#capture_1.latitudes_file = '/home/cameron/Nedlastinger/aeronetvenice_2025-06-12T09-58-02Z/processing-temp/latitudes_indirectgeoref.dat'
#capture_1.longitudes_file = '/home/cameron/Nedlastinger/aeronetvenice_2025-06-12T09-58-02Z/processing-temp/longitudes_indirectgeoref.dat'

#capture_2 = HypsoCapture('/home/cameron/Nedlastinger/aeronetvenice_2025-06-22T10-46-15Z/aeronetvenice_2025-06-22T10-46-15Z-l1a.nc')
#capture_2.cnn_labels_file = '/home/cameron/Nedlastinger/aeronetvenice_2025-06-22T10-46-15Z/processing-temp/sea-land-cloud.labels'
#capture_2.latitudes_file = '/home/cameron/Nedlastinger/aeronetvenice_2025-06-22T10-46-15Z/processing-temp/latitudes_indirectgeoref.dat'
#capture_2.longitudes_file = '/home/cameron/Nedlastinger/aeronetvenice_2025-06-22T10-46-15Z/processing-temp/longitudes_indirectgeoref.dat'

captures = [capture_1, capture_2]
#nc_file_1 = os.path.join(dir_path, capture_1)
#cnn_labels_file_1 = os.path.join(dir_path, cnn_labels_1)
#svm_labels_file_1 = os.path.join(dir_path, svm_labels_1)

#hypso_base_dir = "/home/_shared/ARIEL/HYPSO/"
#dir_path = os.path.join(hypso_base_dir, scene_name)
#h1_l1a_nc_file = os.path.join(dir_path, scene_name+'-l1a.nc')
#h1_points_file = os.path.join(dir_path, 'processing-temp', 'gcp.points')
#h1_lats_file = os.path.join(dir_path, 'processing-temp', 'latitudes_indirectgeoref.dat')
#h1_lons_file = os.path.join(dir_path, 'processing-temp', 'longitudes_indirectgeoref.dat')

processed_captures = []

for capture in captures:
    
    # Load HYPSO-1 Capture
    satobj = Hypso(path=capture.nc_file, verbose=True)

    satobj.generate_l1b_cube()
    satobj.generate_l1c_cube()
    satobj.generate_l1d_cube()

    # Read from latitudes_indirectgeoref.dat
    with open(capture.latitudes_file, mode='rb') as file:
        file_content = file.read()

    lats = np.frombuffer(file_content, dtype=np.float32)

    lats = lats.reshape(satobj.spatial_dimensions)

    # Read from longitudes_indirectgeoref.dat
    with open(capture.longitudes_file, mode='rb') as file:
        file_content = file.read()

    lons = np.frombuffer(file_content, dtype=np.float32)

    lons = lons.reshape(satobj.spatial_dimensions)

    # Directly provide the indirect lat/lons loaded from the file. This function will run the track geometry computations.
    satobj.run_indirect_georeferencing(latitudes=lats, longitudes=lons)

    # Generate land mask
    grid_x_dim, grid_y_dim = satobj.spatial_dimensions

    land_mask = np.zeros(satobj.spatial_dimensions, dtype=bool)

    for x_idx in range(0,grid_x_dim):
        for y_idx in range(0,grid_y_dim):

            grid_lat = satobj.latitudes_indirect[x_idx, y_idx]
            grid_lon = satobj.longitudes_indirect[x_idx, y_idx]

            if globe.is_land(grid_lat, grid_lon):
                land_mask[x_idx, y_idx] = True

    satobj.land_mask = land_mask

    if mask_clouds:



        cloud_mask = np.sum(satobj.l1d_cube.to_numpy(), axis=2)**2

        #cloud_mask_threshold = 0.075e8
        cloud_mask_threshold = np.quantile(cloud_mask, 0.075)

        cloud_mask = np.sum(satobj.l1d_cube.to_numpy(), axis=2)**2 > cloud_mask_threshold

        plt.imsave('cloud_mask_quantile_' + satobj.capture_name + '.png', cloud_mask)


        cloud_mask = decode_jon_cnn_cloud_mask(file_path=capture.cnn_labels, spatial_dimensions=satobj.spatial_dimensions)


        if bd_cloud_mask:
            from scipy.ndimage import binary_dilation
            # Create a structuring element (disk-shaped for 2D)
            from skimage.morphology import disk
            structuring_element = disk(30)  # Radius of 30 pixels
            cloud_mask = binary_dilation(cloud_mask, structure=structuring_element)

        satobj.cloud_mask = cloud_mask

        #plt.imshow(cloud_mask)
        plt.imsave('cloud_mask_' + satobj.capture_name + '.png', cloud_mask)
        #input('Continue...')


    processed_captures.append(satobj)

    print(satobj.l1a_cube.shape)

    #land_mask = decode_jon_cnn_land_mask(file_path=cnn_labels_file_1, spatial_dimensions=satobj.spatial_dimensions)
    #cloud_mask = decode_jon_cnn_cloud_mask(file_path=cnn_labels_file_1, spatial_dimensions=satobj.spatial_dimensions)
    #water_mask = decode_jon_cnn_water_mask(file_path=cnn_labels_file_1, spatial_dimensions=satobj.spatial_dimensions)
    #satobj.cloud_mask = cloud_mask
    #satobj.land_mask = land_mask #| lut_land_mask[:,:]
    #mask = satobj._unified_mask()


    


import pickle


data_1 = {}
data_2 = {}

data_1_name = processed_captures[0].capture_name
data_2_name = processed_captures[1].capture_name

data_1['data'] = processed_captures[0].masked_l1d_cube
data_2['data']  = processed_captures[1].masked_l1d_cube

data_1['lats'] = processed_captures[0].latitudes_indirect
data_2['lats']  = processed_captures[1].latitudes_indirect

data_1['lons'] = processed_captures[0].longitudes_indirect
data_2['lons']  = processed_captures[1].longitudes_indirect


with open(data_1_name + '.pickle', 'wb') as handle:
    pickle.dump(data_1, handle)

with open(data_2_name + '.pickle', 'wb') as handle:
    pickle.dump(data_2, handle)

