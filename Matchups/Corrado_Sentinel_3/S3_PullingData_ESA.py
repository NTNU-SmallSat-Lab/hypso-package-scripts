from datetime import datetime, timedelta
import netCDF4 as nc
import geopandas as gpd
from dotenv import load_dotenv
import os
from ResamplingS3 import resampling_S3, grid_to_polygon
# New packages
import boto3
from pathlib import Path
from botocore.exceptions import ClientError
import requests


def function_pulldata(start, end):
    load_dotenv()

    # Read keys from environment
    S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
    S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')

    # Change accordingly to where you wan to store the data
    download_folder = Path('/mnt/c/Users/corradoc/data/S3')
    output_directory = Path('../../Frohavet_campaign/S3_processed')

    # Load SINMOD grid for AOI definition
    SINMODgrid_file = 'midnor_grid.nc'  # Static file (assumed to be always the same)
    sample_netcdf = 'cmems_chl.nc'
    sample_netcdf_Daniel = 'Daniel_netcdf.nc'


    dataset_SINMOD = nc.Dataset(SINMODgrid_file, 'r')

    grid_lats = dataset_SINMOD.variables['gridLats'][:]
    grid_lons = dataset_SINMOD.variables['gridLons'][:]

    # Convert grid to polygon
    polygon = grid_to_polygon(grid_lats, grid_lons)
    gdf = gpd.GeoDataFrame({'geometry': [polygon]})



    gdf.set_crs('EPSG:32633', inplace=True)  # Example: UTM projection
    gdf = gdf.to_crs('EPSG:4326')
    geometry = [list(coord) for coord in gdf.geometry[0].exterior.coords]
    # Generate a simplified polygon to filter out data that is not in the SINMOD grid 
    simplified_polygon = polygon.simplify(tolerance=0.01, preserve_topology=True)
    # Extract the simplified geometry coordinates
    simplified_geometry = [list(coord) for coord in simplified_polygon.exterior.coords]

    polygon_wkt = "POLYGON((" + ", ".join([f"{lon} {lat}" for lon, lat in simplified_geometry]) + "))"


    # Time window to filter data
    timewindow = timedelta(hours=3)
    date_from = (start - timewindow).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    date_to = (end + timewindow).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # Define the OData base URL for Sentinel-3 WFR product type
    odata_base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    # Construct the OData URL with filtering parameters
    odata_url = (
        f"{odata_base_url}?$filter=("
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'OL_2_WFR___')"
        f") and ContentDate/Start ge {date_from} and "
        f"ContentDate/End le {date_to} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon_wkt}')"
        f"&$orderby=PublicationDate desc"
        f"&$expand=Attributes"
    )

    #print(f"Requesting data from: {odata_url}")

    # Making the request to the OData service
    try:
        response = requests.get(odata_url)
        response.raise_for_status()  # Will raise an exception for HTTP errors
        request_data = response.json()
        #print(request_data)
    except requests.exceptions.RequestException as e:
        print(f"Failed to get data: {e}")
        request_data = {}

    if "value" in request_data:
        for result in request_data["value"]:
            try:
                s3_url = result["S3Path"].strip("/eodata/")
                #print(f"Found download URL: {s3_url}")
            except KeyError:
                print("S3Path not found in result.")

    session = boto3.session.Session()
    s3 = session.resource(
        's3',
        endpoint_url='https://eodata.dataspace.copernicus.eu',
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name='default'
    )
    bucket = s3.Bucket("eodata")

    # Files to keep
    files_to_keep = {'chl_nn.nc', 'geo_coordinates.nc', 'wqsf.nc'}
    extracted_folders = set()

    if "value" in request_data:
        for result in request_data["value"]:
            try:
                s3_url = result["S3Path"].strip("/eodata/")

                # ✅ Filter only NR files
                if "NR" not in s3_url.split("_")[-2]:
                    #print(f"Skipping NT file: {s3_url}")
                    continue
                print(f"Found download URL: {s3_url}")
                files = bucket.objects.filter(Prefix=s3_url)
                if not list(files):
                    print(f"No files found for: {s3_url}")
                    continue

                for file in files:
                    if file.key and not file.key.endswith("/"):
                        file_name = Path(file.key).name
                        
                        # ✅ Keep only specific files
                        if file_name in files_to_keep:
                            inner_most_folder = Path(file.key).parent.name
                            folder_path = download_folder / inner_most_folder
                            folder_path.mkdir(parents=True, exist_ok=True)

                            local_file_path = folder_path / file_name
                            if not local_file_path.exists():
                                #print(f"Downloading {file.key}...")
                                bucket.download_file(file.key, str(local_file_path))
                                
                                # ✅ Track extracted folders
                                extracted_folders.add(folder_path)
                            else:
                                print(f"File already exists: {local_file_path}")
                        
                print("Downloaded files successfully.")

            except KeyError:
                print("S3Path not found in result.")
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except ClientError as e:
                if e.response['Error']['Code'] == '403':
                    print("Access Denied: Check your S3 credentials and permissions.")
                else:
                    print(f"ClientError: {e}")
            except Exception as e:
                print(f"Error processing file: {e}")


    # ✅ Now iterate over the extracted folders and process them
    if not extracted_folders:
        print('No captures to download.')
    else:
        for SAFE_directory in extracted_folders:
            print(f'Running main() for {SAFE_directory}...')
            resampling_S3(SAFE_directory, SINMODgrid_file, sample_netcdf, sample_netcdf_Daniel, output_directory)
            #resampling_S3(SAFE_directory, SINMODgrid_file, sample_netcdf_Daniel, output_directory)
        print('All products processed successfully.')


from datetime import datetime, timedelta

start_date = datetime(2025, 5, 19)
end_date = datetime(2025, 5, 22)

# Set time window (e.g., 6:00 to 18:00 daily)
for day in range((end_date - start_date).days + 1):
    current_day = start_date + timedelta(days=day)
    start = current_day.replace(hour=6, minute=0)
    end = current_day.replace(hour=18, minute=0)

    print(f"Processing data from {start} to {end}")
    function_pulldata(start, end)
