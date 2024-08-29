"""
This script reads the recorder deployment metadata file, plots a map of 
the spatial locations. It then filters files based on deployment dates, 
and saves the file index as a JSON for later use.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define constants for paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = Path(REPO_ROOT) / "metadata"
RAW_DIR = Path("/media/nilomr/SONGDATA/raw/phenoscale_2024_pilot/")


def read_metadata():
    """Read the 'pilot_metadata.csv' file."""
    return pd.read_csv(METADATA_DIR / "pilot_metadata.csv")


def plot_map(df):
    """Plot map of Latitudes and Longitudes, colored by 'test' column."""
    sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="test")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def get_folders():
    """Read the names of the folders in the raw directory."""
    return [folder for folder in RAW_DIR.iterdir() if folder.is_dir()]


def get_foldernames(folders):
    """Extract folder names as integers."""
    return [int(folder.name) for folder in folders]


def get_files_in_folder(folder):
    """Get sorted list of files in a folder."""
    return sorted(folder.iterdir(), key=lambda x: x.name)


def format_file_dates(files):
    """Format file dates from filenames."""
    file_dates = [file.name.split("_")[0] for file in files]
    return [f"{date[:4]}-{date[4:6]}-{date[6:]} 00:00:00" for date in file_dates]


def filter_files(files, file_dates, deployment_dates):
    """Filter files based on deployment dates."""
    files = [
        file
        for file, date in zip(files, file_dates)
        if date.split(" ")[0] not in [date.split(" ")[0] for date in deployment_dates]
    ]
    mask01 = [True if date < max(deployment_dates) else False for date in file_dates]
    mask02 = [True if date > max(deployment_dates) else False for date in file_dates]
    pilot01 = [file for file, mask in zip(files, mask01) if mask]
    pilot02 = [file for file, mask in zip(files, mask02) if mask]
    return {"pilot01": pilot01, "pilot02": pilot02}


def save_fileindex(fileindex):
    """Save the file index as a JSON file in the metadata folder."""
    with open(METADATA_DIR / "fileindex.json", "w", encoding="utf-8") as f:
        json.dump(fileindex, f, indent=4, default=str)


def main():
    df = read_metadata()
    plot_map(df)

    folders = get_folders()
    foldernames = get_foldernames(folders)

    fileindex = {}
    for folder in foldernames:
        folder_rows = df[df["n"] == folder]
        deployment_dates = folder_rows["deployed"].unique()
        files = get_files_in_folder(folders[foldernames.index(folder)])
        file_dates = format_file_dates(files)
        fileindex[folder] = filter_files(files, file_dates, deployment_dates)

    save_fileindex(fileindex)


if __name__ == "__main__":
    main()
