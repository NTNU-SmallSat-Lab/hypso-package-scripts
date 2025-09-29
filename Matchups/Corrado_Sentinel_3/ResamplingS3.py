import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr                 # a library that supports the use of multi-dimensional arrays in Python
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains
from cartopy.feature import ShapelyFeature
from datetime import datetime
from netCDF4 import Dataset
from pyresample import geometry, kd_tree
from pyresample.image import ImageContainer
from pyresample import bilinear, geometry

def grid_to_polygon(lat_matrix, lon_matrix):
    """Convert the external points of lat/lon matrices into a Shapely polygon."""
    # Extract boundary points
    top = list(zip(lon_matrix[0, :], lat_matrix[0, :]))
    right = list(zip(lon_matrix[:, -1], lat_matrix[:, -1]))
    bottom = list(zip(lon_matrix[-1, ::-1], lat_matrix[-1, ::-1]))
    left = list(zip(lon_matrix[::-1, 0], lat_matrix[::-1, 0]))
    # Combine in order and create polygon
    return Polygon(top + right + bottom + left)
# Define polygons representing the S3 Image and SINMOD Operative Area

def filter_points_out_polygon(lat_matrix, lon_matrix, chl_matrix, polygon):
    """Filter out points outside the polygon using Shapely's vectorized operations."""
    # Flatten the matrices into 1D arrays
    lon_flat = lon_matrix.flatten()
    lat_flat = lat_matrix.flatten()
    chl_flat = chl_matrix.flatten()

    # Vectorized filtering with Shapely
    mask = contains(polygon, lon_flat, lat_flat)  # Boolean mask for points inside polygon
    
    return lon_flat[mask], lat_flat[mask], chl_flat[mask]

def extract_time_from_filename(filename):
    try:
        # Split the filename into parts
        filename_parts = filename.split('_')
        
        # Ensure there are enough parts and extract the timestamp
        if len(filename_parts) > 3:
            time_part = filename_parts[7]  # Extract '20250311T104817'
            
            # Try to parse the time string
            extracted_time = datetime.strptime(time_part, '%Y%m%dT%H%M%S')
            
            # Convert to seconds since epoch
            time_in_seconds = (extracted_time - datetime(1970, 1, 1)).total_seconds()
            return time_part, time_in_seconds
        else:
            raise ValueError(f"Filename format is incorrect: {filename}")
    except Exception as e:
        print(f"Error extracting time: {e}")
        return None

def copy_and_update_nc(input_file, output_file, new_chl, new_time):
    with Dataset(input_file, 'r') as src:
        with Dataset(output_file, 'w') as dst:
            # Copy only the TITLE attribute if it exists
            if 'TITLE' in src.ncattrs():
                dst.setncattr('TITLE', 'S3_chl_interpolateddata')

            # Copy dimensions
            for name, dimension in src.dimensions.items():
                dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

            for name in ['time', 'CHL']:
                if name in src.variables:
                    var = src.variables[name]
                    new_var = dst.createVariable(name, var.datatype, var.dimensions)
                    
                    # Copy attributes
                    new_var.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs()})
                    
                    if name == 'time':
                        new_var[:] = new_time
                    elif name == 'CHL':
                        new_var[:] = new_chl  # Direct assignment without chunking

            print(f"New NetCDF file saved as '{output_file}' with updated values")


def copy_and_update_nc_daniel(input_file, output_file, new_chl, new_time):
    with Dataset(input_file, 'r') as src:
        with Dataset(output_file, 'w') as dst:
            # Copy global attributes
            for attr in src.ncattrs():
                dst.setncattr(attr, src.getncattr(attr))
            
            # Copy dimensions
            for name, dimension in src.dimensions.items():
                dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
            
            # Copy variables and update chl_a and time
            for name, var in src.variables.items():
                new_var = dst.createVariable(name, var.datatype, var.dimensions)

                # Copy variable attributes
                new_var.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs()})
                
                if name == 'time':
                    new_var[:] = new_time  # Update time
                elif name == 'chl_a':
                    new_var[:] = new_chl  # Update chlorophyll-a data
                else:
                    new_var[:] = var[:]  # Copy everything else unchanged
            
            print(f"✅ New NetCDF file saved as '{output_file}' with updated CHL and time values")




def resampling_S3(SAFE_directory, SINMODgrid_file, sample_netcdf, sample_netcdf_daniel, output_directory):

    geo_file = os.path.join(SAFE_directory, 'geo_coordinates.nc')
    chl_file = os.path.join(SAFE_directory, 'chl_nn.nc')
    flag_file = os.path.join(SAFE_directory, 'wqsf.nc')

    ################################################################################################################

    # Open the Grid for SINMOD
    dataset_SINMOD = nc.Dataset(SINMODgrid_file, 'r')
    # Access the gridLats variable
    grid_lats = dataset_SINMOD.variables['gridLats'][:]
    grid_lons = dataset_SINMOD.variables['gridLons'][:]

    dataset_SINMOD.close()
    
    dataset_geo = nc.Dataset(geo_file, 'r')

    longitude = dataset_geo.variables['longitude'][:]
    latitude = dataset_geo.variables['latitude'][:]
    dataset_geo.close()



    #dataset = nc.Dataset(chl_file, 'r')
    band_vars = xr.open_mfdataset(chl_file)
    band_vars.close() 
    chl_a = band_vars["CHL_NN"][:]

    # Mask variables
    flag_vars = xr.open_dataset(flag_file)
    flag_vars.close()


    ii = np.argsort(flag_vars["WQSF"].flag_masks)
    bitvals = np.array(flag_vars["WQSF"].flag_masks)[ii]
    meanings = np.array(flag_vars["WQSF"].flag_meanings.split(' '))[ii]

    wqsf = flag_vars["WQSF"]

    # Access the `flag_masks` and `flag_meanings` attributes
    flag_masks = wqsf.attrs["flag_masks"]
    flag_meanings = wqsf.attrs["flag_meanings"].split()

    # Define the target flags
    #target_flags = ['CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'CLOUD']
    '''
    target_flags = ['LAND', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'INVALID', 'COSMETIC', 'SATURATED', 'SUSPECT', 'HISOLZEN', 'HIGHGLINT', 'SNOW_ICE', 'AC_FAIL',
                                                        'WHITECAPS', 'ADJAC', 'RWNEG_O2', 'RWNEG_O3', 'RWNEG_O4', 'RWNEG_O5', 'RWNEG_O6', 'RWNEG_O7', 'RWNEG_O8',
                                                        'OCNN_FAIL']

    # Set all targets
    '''
    target_flags = ['LAND', 'CLOUD', 'TURBID_ATM', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 
                    'INVALID', 'COSMETIC', 'SATURATED', 'MEGLINT', 'SUSPECT', 'HISOLZEN', 
                    'HIGHGLINT', 'SNOW_ICE', 'AC_FAIL', 'WHITECAPS', 'ADJAC', 'RWNEG_O2', 
                    'RWNEG_O3', 'RWNEG_O4', 'RWNEG_O5', 'RWNEG_O6', 'RWNEG_O7', 'RWNEG_O8',
                    'OCNN_FAIL']
    


    # Find the corresponding masks for the target flags
    target_masks = [flag_masks[flag_meanings.index(flag)] for flag in target_flags]

    # Initialize a mask for all target flags
    combined_mask = np.zeros_like(wqsf, dtype=bool)

    # Loop through each target mask and combine them
    for mask in target_masks:
        combined_mask |= (wqsf & mask) > 0  # Bitwise OR to combine the masks

    combined_mask_np = combined_mask.values

    # Set the values to NaN using NumPy indexing
    chl_a_updated = chl_a.values
    chl_a_updated[combined_mask_np == True] = np.nan
    

    polygon1 = grid_to_polygon(grid_lats, grid_lons)



    ####### Filter out points that are close to mask and greater han cut_off value
    cut_off = 10
    radius = 20
    temp = 10**chl_a_updated
    indexes = np.where(temp > cut_off)
    mask = combined_mask_np
    for row, col in zip(indexes[0], indexes[1]):    
        # Define search boundaries
        row_start, row_end = max(0, row - radius), min(mask.shape[0], row + radius + 1)
        col_start, col_end = max(0, col - radius), min(mask.shape[1], col + radius + 1)
        
        # Check and modify if there's a 1 in the surrounding area
        nearby_area = mask[row_start:row_end, col_start:col_end]
        if np.any(nearby_area == 1):
            temp[row, col] = np.nan

    chl_a_updated = temp
    #temp2 = temp.flatten()
    #print(np.max(temp2[~np.isnan(temp2)]))






    # Filter out points that are outside the SINMOD grid
    lon_cut, lat_cut, chl_cut = filter_points_out_polygon(latitude, longitude, chl_a_updated, polygon1)

    if np.isnan(chl_cut).all():
        print("Nothing to save here")
        return
    non_nan_count = np.count_nonzero(~np.isnan(chl_cut))
    print(f"Number of non-NaN values: {non_nan_count}")
    if non_nan_count < 1000:
        print("Too few points, ignoring")
        return



    skip = 1
    points=(lon_cut[::skip], lat_cut[::skip])
    orig_def = geometry.SwathDefinition(lons=points[0], lats=points[1])
    tgt_def = geometry.GridDefinition(lons=grid_lons, lats=grid_lats)
    # Define fill value as a float (not an integer!)
    fill_value = np.nan  

    # Perform nearest-neighbor resampling
    z_pyresample = kd_tree.resample_nearest(
        orig_def, chl_cut[::skip], tgt_def, 
        radius_of_influence=1000, fill_value=fill_value
    )

    #  FIX: Convert to float BEFORE assigning NaN
    z_pyresample = z_pyresample.astype(np.float32)  # Ensure it's float-compatible
    #z_pyresample = 10**z_pyresample
    z_pyresample[z_pyresample> 10] = 10

    
    ## SAVE THE NETCDF FILES!

    last_folder = os.path.basename(os.path.normpath(os.path.basename(os.path.normpath(SAFE_directory))))

    time_part, extracted_time = extract_time_from_filename(last_folder)
    print(f"Extracted time in seconds: {extracted_time}")
    print(extracted_time)
    # File paths
    output_file = os.path.join(output_directory, 'chl_S3_' + time_part + '.nc') 
    output_file_daniel = os.path.join(output_directory, 'chl_S3_' + time_part + '_daniel.nc') 


    # Delete the file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Deleted existing file: {output_file}")

    ################## Usage
    new_chl = np.expand_dims(z_pyresample, axis=0)  # Add a time dimension if missing
    #copy_and_update_nc(sample_netcdf, output_file, new_chl, extracted_time)
    #print(f"New file saved as '{output_file}' with CHL_uncertainty set to NaN.")

    copy_and_update_nc_daniel(sample_netcdf_daniel, output_file_daniel, new_chl, extracted_time)
    print(f"New file saved as '{output_file_daniel}' with CHL_uncertainty set to NaN and Daniel file approach.")
    
    
    
#    dataset = nc.Dataset(output_file, 'r')
#    print(dataset)










    
    
    ########################################
    '''
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    # Create the figure
    plt.figure(figsize=(16, 8))

    # Set up the map with PlateCarree projection
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set the extent based on the grid's longitude and latitude
    ax.set_extent([np.min(grid_lons) - 2, np.max(grid_lons) + 2, np.min(grid_lats) - 1, np.max(grid_lats) + 1], crs=ccrs.PlateCarree())

    # Plot the resampled data (this assumes z_pyresample is your data to plot)
    mesh = ax.pcolormesh(grid_lons, grid_lats, z_pyresample, shading='auto', cmap='viridis', transform=ccrs.PlateCarree())

    # Add basemap features
    ax.coastlines(resolution='10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Assuming you have the latitude and longitude data that form the polygon
    polygon2 = grid_to_polygon(latitude, longitude)

    # Create the ShapelyFeature for the polygon (ensure CRS is provided)
    shape_feature1 = ShapelyFeature([polygon1], ccrs.PlateCarree(), edgecolor='orange', facecolor=(1.0, 0.647, 0.0, 0.1), linewidth=2)
    shape_feature2 = ShapelyFeature([polygon2], ccrs.PlateCarree(), edgecolor='blue', facecolor=(0.678, 0.847, 0.902, 0.1), linewidth=2)

    # Add the feature to the map
    ax.add_feature(shape_feature1)
    ax.add_feature(shape_feature2)

    # Add colorbar and labels
    plt.colorbar(mesh, ax=ax, orientation='vertical', label='Chlorophyll-a (mg/m^3)')
    plt.title('Resampled Chlorophyll-a Concentration')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Show the plot
    plt.show()
    ###### Save NetCDF file ######
    # latitude
    # longitude
    # chl-a
    ########################################
    '''

'''
if __name__ == "__main__":
        ################################################################################################################
    # Load geo_coordinates.nc for longitude and latitude, chl-a, mask file, and SINMOD grid
    download_folder = '/mnt/c/Users/corradoc/data'
    SINMODgrid_file = 'midnor_grid.nc'  # Static file (assumed to be always the same)

    SAFE_directory = '/mnt/c/Users/corradoc/data/S3A_OL_2_WFR____20240403T101439_20240403T101739_20240404T181033_0179_111_008_1800_MAR_O_NT_003.SEN3/S3A_OL_2_WFR____20240403T101439_20240403T101739_20240404T181033_0179_111_008_1800_MAR_O_NT_003.SEN3'
    # Geocoordinates file
    output_directory = os.path.join(download_folder, 'results')
    sample_netcdf = 'cmems_chl.nc'
    print(f'Running main() for {SAFE_directory}...')
    resampling_S3(SAFE_directory, SINMODgrid_file, sample_netcdf, output_directory)
    # Chl_nn.nc for Chlorophyll-a data
    #chl_file = '/mnt/c/Users/corradoc/Downloads/S3A_OL_2_WFR____20240326T102208_20240326T102508_20240327T182958_0179_110_279_1800_MAR_O_NT_003/S3A_OL_2_WFR____20240326T102208_20240326T102508_20240327T182958_0179_110_279_1800_MAR_O_NT_003.SEN3/chl_nn.nc'

    # SINMOD grid
    SINMODgrid_file = 'midnor_grid.nc'
    flag_file = os.path.join(SAFE_directory,'wqsf.nc')    
    resampling_S3(SAFE_directory, SINMODgrid_file, sample_netcdf, output_directory)
'''