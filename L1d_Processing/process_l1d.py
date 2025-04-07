#!/usr/bin/env python3

import os
import sys
import numpy as np

from pathlib import Path
from hypso import Hypso

from hypso.write import write_l1d_nc_file

def main(l1a_nc_path, lats_path=None, lons_path=None):
    # Check if the first file exists
    if not os.path.isfile(l1a_nc_path):
        print(f"Error: The file '{l1a_nc_path}' does not exist.")
        return

    # Process the first file
    print(f"Processing file: {l1a_nc_path}")

    nc_file = Path(l1a_nc_path)

    satobj = Hypso(path=nc_file, verbose=True)


    # Run indirect georeferencing
    if lats_path is not None and lons_path is not None:
        try:

            with open(lats_path, mode='rb') as file:
                file_content = file.read()
            
            lats = np.frombuffer(file_content, dtype=np.float32)

            lats = lats.reshape(satobj.spatial_dimensions)

            with open(lons_path, mode='rb') as file:
                file_content = file.read()
            
            lons = np.frombuffer(file_content, dtype=np.float32)
  
            lons = lons.reshape(satobj.spatial_dimensions)


            # Directly provide the indirect lat/lons loaded from the file. This function will run the track geometry computations.
            satobj.run_indirect_georeferencing(latitudes=lats, longitudes=lons)

            satobj.generate_l1b_cube()
            satobj.generate_l1c_cube()
            satobj.generate_l1d_cube(use_indirect_georef=True)

        except Exception as ex:
            print(ex)
            print('Indirect georeferencing has failed. Defaulting to direct georeferencing.')

            satobj.run_direct_georeferencing()

    else:
        satobj.run_direct_georeferencing()

        satobj.generate_l1b_cube()
        satobj.generate_l1c_cube()
        satobj.generate_l1d_cube(use_indirect_georef=False)
        

    write_l1d_nc_file(satobj, overwrite=True, datacube=False)



if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python process_l1d.py <l1a_nc_path> [lats_path] [lons_path]")
        sys.exit(1)

    l1a_nc_path = sys.argv[1]
    
    lats_path = sys.argv[2] if len(sys.argv) == 4 else None
    lons_path = sys.argv[3] if len(sys.argv) == 4 else None

    main(l1a_nc_path, lats_path, lons_path)


