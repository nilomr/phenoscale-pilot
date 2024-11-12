import contextlib
import json
import multiprocessing
import os
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path

import pandas as pd
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = Path(REPO_ROOT) / "metadata"
RAW_DIR = Path("/media/nilomr/SONGDATA/raw/phenoscale_2024_pilot/")
DETECTIONS_DIR = Path(REPO_ROOT, "data", "derived")
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# read in the metadata json file
with open(Path(METADATA_DIR, "fileindex.json"), "r", encoding="utf-8") as jf:
    fileindex = json.load(jf)

file_paths = [
    Path(RAW_DIR, path)
    for k, v in fileindex.items()
    for sk, sv in v.items()
    for path in sv
]

# print folder names in file_paths
print("Folder names in file_paths:")
print(pd.Series([path.parent.name for path in file_paths]).value_counts())

existing_files = {
    (file.stem.split("_", 1)[0], file.stem.split("_", 1)[1])
    for file in DETECTIONS_DIR.iterdir()
    if file.suffix == ".json"
}

file_paths = [
    path
    for path in tqdm(file_paths)
    if path.suffix in [".wav", ".WAV"]
    and os.path.getsize(path) > 10
    and (path.parent.name, path.stem) not in existing_files
]

print(f"Found {len(file_paths)} files")

# ──── MAIN ───────────────────────────────────────────────────────────────────

# Load and initialize the BirdNET-Analyzer model
analyzer = Analyzer(version="2.4")

# Create a shared queue to store the detections
manager = Manager()
detections_queue = manager.Queue()


def process_file(file_path):
    date = datetime.strptime(file_path.stem.split("_")[0], "%Y%m%d")
    dir_name = file_path.parent.name

    recording = Recording(
        analyzer,
        str(file_path),
        lat=51.775036,
        lon=-1.336488,
        date=date,
        min_conf=0.80,
    )
    log_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    log_file = Path(REPO_ROOT, "logs", f"{log_date}.log")
    with open(log_file, "a", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            recording.analyze()
            recording.extract_embeddings()

    detections_queue.put(
        [file_path.stem, dir_name, recording.detections, recording.embeddings]
    )


def save_detections(data):
    with open(
        Path(DETECTIONS_DIR, f"{data[1]}_{data[0]}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "file_name": data[0],
                "dir_name": data[1],
                "detections": data[2],
                "embeddings": data[3],
            },
            f,
            indent=4,
        )


def process_file_and_save(file_path):
    process_file(file_path)
    while not detections_queue.empty():
        data = detections_queue.get()
        save_detections(data)


def process_files(file_paths):
    with tqdm(total=len(file_paths), desc="Processing files") as pbar:
        ncore = os.cpu_count()
        print(f"Using {ncore} cpus")
        pool = multiprocessing.Pool(processes=ncore)
        for _ in pool.imap_unordered(process_file_and_save, file_paths):
            pbar.update(1)
        pool.close()
        pool.join()


# ──── RUN PROCESS ────────────────────────────────────────────────────────────

process_files(file_paths)
