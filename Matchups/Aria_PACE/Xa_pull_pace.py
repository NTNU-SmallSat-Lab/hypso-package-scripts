#!/usr/bin/env python3

import os
import sys
import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
import geopandas as gpd
from pathlib import Path
from hypso import Hypso
from hypso.write import write_l1d_nc_file
import requests
import boto3
from botocore.exceptions import ClientError
import earthaccess


results = earthaccess.search_data(
    short_name="PACE_OCI_L1B_SCI",
    temporal=tspan,
    bounding_box=bbox,
)
print(results)


scores = np.zeros(len(results))
for j in range(len(results)):
    # retrieve the bounding box latitides and longitudes
    bbox = results[j]['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]['Boundary']['Points']
    # calculate the distance from the target latitude and longitude to the bounding box points
    for point in bbox:
        a = point['Latitude'] - target_latitude[i]
        b = point['Longitude'] - target_longitude[i]
        scores[j] += np.sqrt(a**2 + b**2)
# choose the capture with the lowest score, since this gives the most centered point
# why the center gives the lowest score: https://stackoverflow.com/questions/4609846/find-point-which-sum-of-distances-to-set-of-other-points-is-minimal
best_capture_idx = np.argmin(scores)

# download data
full_pace_file = earthaccess.download(results[best_capture_idx]) # take the capture where the target lat-lon is most centered