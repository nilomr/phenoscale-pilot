"""
Extracts the pilot points from the Organic Maps backup file and saves them
to a csv file. This then needs to be cleaned up manually.
"""

import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from bs4 import BeautifulSoup

# Define constants for paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
KMZ_FILEPATH = os.path.join(REPO_ROOT, "metadata", "OrganicMapsBackup_240821.kmz")
OUTPUT_CSV = Path(REPO_ROOT, "metadata", "wytham_pilot_points.csv")


def extract_kml_from_kmz(filepath):
    """Extract the KML file from the KMZ archive."""
    with ZipFile(filepath, "r") as kmz:
        kml_files = [file for file in kmz.namelist() if "Wytham" in file]
        if kml_files:
            return kmz.open(kml_files[0]).read()
        else:
            print("No KML file containing 'Wytham' found.")
            return None


def parse_kml(kml):
    """Parse the KML content and extract placemarks."""
    soup = BeautifulSoup(kml, features="lxml")
    placemarks = soup.find_all("placemark")
    return [
        {
            "name": placemark.find("name").text,
            "description": (
                placemark.find("description").text.split("\n")
                if placemark.find("description")
                else None
            ),
            "coords": tuple(map(float, placemark.find("coordinates").text.split(","))),
        }
        for placemark in placemarks
        if placemark.find("description") is not None
        and "doc" not in placemark.find("name").text.lower()
    ]


def create_dataframe(pilot_placemarks):
    """Create a DataFrame from the extracted placemarks."""
    df = pd.DataFrame(pilot_placemarks)
    df = (
        df.map(lambda x: x.strip() if isinstance(x, str) else x)
        .join(
            df["description"]
            .apply(pd.Series)
            .rename(columns={0: "test", 1: "device", 2: "comments"})
        )
        .drop(columns="description")
        .replace(to_replace="Pilot", value="", regex=True)
    )
    df = df.join(
        df["device"]
        .str.split(" ", expand=True)
        .rename(columns={0: "am_version", 1: "n", 2: "extra"})
    )
    df[["Latitude", "Longitude"]] = pd.DataFrame(
        df["coords"].apply(lambda x: tuple(reversed(x))).tolist()
    )
    df = df.drop(columns="coords")
    return df


def save_dataframe(df, output_path):
    """Save the DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)


def main():
    kml = extract_kml_from_kmz(KMZ_FILEPATH)
    if kml:
        pilot_placemarks = parse_kml(kml)
        df = create_dataframe(pilot_placemarks)
        save_dataframe(df, OUTPUT_CSV)


if __name__ == "__main__":
    main()
