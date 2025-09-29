import xarray as xr
import numpy as np
from pyresample.geometry import SwathDefinition
from pyresample.geometry import SwathDefinition, AreaDefinition
from pyresample.bilinear.xarr import XArrayBilinearResampler 
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler

latitudes_xr = xr.DataArray(satobj.latitudes_indirect, dims=satobj.dim_names_2d)
longitudes_xr = xr.DataArray(satobj.longitudes_indirect, dims=satobj.dim_names_2d)
dst_swath_def = SwathDefinition(lons=longitudes_xr, lats=latitudes_xr)

latitudes_xr = xr.DataArray(satobj.latitudes, dims=satobj.dim_names_2d)
longitudes_xr = xr.DataArray(satobj.longitudes, dims=satobj.dim_names_2d)
src_swath_def = SwathDefinition(lons=longitudes_xr, lats=latitudes_xr)

data = satobj.l1a_cube

#brs = XArrayBilinearResampler(source_geo_def=src_swath_def, target_geo_def=dst_swath_def, radius_of_influence=50000)
brs = KDTreeNearestXarrayResampler(source_geo_def=src_swath_def, target_geo_def=dst_swath_def)

num_bands = data.shape[2]

resampled_data = np.zeros((dst_swath_def.shape[0], dst_swath_def.shape[1], num_bands))
resampled_data = xr.DataArray(resampled_data, dims=satobj.dim_names_3d)
resampled_data.attrs.update(data.attrs)

for band in range(0,num_bands):
    
    # Resample using pre-computed resampling LUTs
    #resampled_data[:,:,band] = brs.get_sample_from_bil_info(data=data[:,:,band], 
    #                                                        fill_value=np.nan, 
    #                                                        output_shape=dst_swath_def.shape)

    resampled_data[:,:,band] = brs.resample(data=data[:,:,band], fill_value=np.nan)
