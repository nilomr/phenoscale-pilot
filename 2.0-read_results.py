import json
import multiprocessing
import os
from datetime import datetime, timedelta
from multiprocessing import Manager
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
import pyproj
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import euclidean_distances

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = Path(REPO_ROOT) / "metadata"
RAW_DIR = Path("/media/nilomr/SONGDATA/raw/phenoscale_2024_pilot/")
DETECTIONS_DIR = Path(REPO_ROOT, "data", "derived")
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# read in the file metadata json file
with open(Path(METADATA_DIR, "fileindex.json"), "r", encoding="utf-8") as jf:
    fileindex = json.load(jf)

# Read in the metadata
metadata = pl.read_csv(Path(METADATA_DIR, "pilot_metadata.csv"))

existing_files = {
    (file.stem.split("_", 1)[0], file.stem.split("_", 1)[1], file)
    for file in DETECTIONS_DIR.iterdir()
    if file.suffix == ".json"
}


# Initialize a list to store the extracted data
def process_file(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        try:
            content = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file}: {e}")
            return data
        file_name = content["file_name"]
        dir_name = content["dir_name"]
        device = int(dir_name)
        for detection in content["detections"]:
            common_name = detection["common_name"]
            confidence = detection["confidence"]
            start_time = detection["start_time"]
            detection_time = datetime.strptime(file_name, "%Y%m%d_%H%M%S") + timedelta(
                seconds=start_time
            )
            data.append([file_name, device, detection_time, common_name, confidence])
    return data


def collect_results(result):
    data.extend(result)


if __name__ == "__main__":
    manager = Manager()
    data = manager.list()

    with multiprocessing.Pool(processes=4) as pool:
        for file in DETECTIONS_DIR.glob("*.json"):
            pool.apply_async(process_file, args=(file,), callback=collect_results)
        pool.close()
        pool.join()

    data = list(data)


# get the pilot code for each device
device_to_experiment = []
for device, experiments in fileindex.items():
    for experiment, files in experiments.items():
        for file in files:
            device_to_experiment.append([Path(file).stem, int(device), experiment])

# Buld the dataset
df = pl.DataFrame(
    data,
    orient="row",
    schema=["file_name", "device", "detection_time", "common_name", "confidence"],
)
df = df.join(
    pl.DataFrame(
        device_to_experiment, orient="row", schema=["file_name", "device", "experiment"]
    ),
    on=["file_name", "device"],
)

# filter out if confidence is less than 0.9
df = df.filter(pl.col("confidence") >= 0.90)

# Save this df
df.write_csv(DETECTIONS_DIR / "detections.csv")
