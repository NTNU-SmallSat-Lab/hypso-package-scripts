import sys
from polymer.main_v5 import run_polymer, default_output_datasets
from eoread.hypso import Level1_HYPSO
from pathlib import Path
import shutil
 
# List of base names
base_names = [
    "image61N6E_2025-03-13T11-27-56Z",
    "image62N5E_2025-05-16T10-51-16Z",
    "image62N5E_2025-05-17T10-56-18Z",
    "image62N6E_2025-04-19T11-42-57Z",
    "image62N6E_2025-05-04T11-25-43Z",
    "image62N6E_2025-05-15T10-46-12Z",
    "image62N7E_2025-04-06T10-32-55Z",
    "image63N6E_2025-04-02T11-46-04Z",
    "image63N6E_2025-05-11T10-25-37Z",
    "image63N8E_2025-04-20T10-12-24Z",
    "image64N9E_2025-04-14T11-15-42Z",
    "image64N9E_2025-04-23T10-28-01Z",
    "image64N9E_2025-04-23T12-03-20Z",
    "image64N9E_2025-05-11T12-00-37Z",
    "richard_2025-05-18T11-01-12Z",
    "frohavet_2025-05-19T11-06-03Z"
]
 
downloads = Path.home() / "Downloads"
hypso_package = downloads / "hypso-package-scripts-main"
polymer_base_dir = downloads / "HYPSO-2-Polymer"
 
for base_name in base_names:
    try:
        print(f"\n--- Processing {base_name} ---")
        source_file = hypso_package / base_name / f"{base_name}-l1c.nc"
        polymer_dir = polymer_base_dir / base_name
        output_dir = polymer_dir / "Output"
        l1c_path = polymer_dir / f"{base_name}-l1c.nc"
 
        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"Output already exists for {base_name}, skipping.")
            continue
 
        # Setup output directory
        polymer_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
 
        # Copy L1C file to Polymer directory
        if not l1c_path.exists():
            if not source_file.exists():
                print(f"Source file missing for {base_name}, skipping.")
                continue
            shutil.copy(source_file, l1c_path)
 
        # Run Polymer
        run_polymer(
            Level1_HYPSO(l1c_path),
            dir_out=str(output_dir),
            output_datasets=default_output_datasets + ['SPM']
        )
        print(f"Completed {base_name}")
 
    except Exception as e:
        print(f"Error processing {base_name}: {e}")
