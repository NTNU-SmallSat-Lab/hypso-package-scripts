#!/usr/bin/env python3

import os
import subprocess

# Path to the base directory
base_dir = "/home/_shared/ARIEL/HYPSO"

# Iterate over all entries in the base directory
for entry in os.listdir(base_dir):
    full_path = os.path.join(base_dir, entry)
    
    # Check if the entry is a directory
    if os.path.isdir(full_path):
        # Construct the command
        command = ["python", "process_l1d_ocsmart.py", full_path]
        
        # Run the command
        try:
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command for {full_path}: {e}")
