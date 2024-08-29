# 2024 Phenoscale Pilot

![Title
Badge](https://img.shields.io/badge/2024_phenoscale_pilot_|_species_detection-k?style=for-the-badge&labelColor=d99c2b&color=d99c2b)
![Python
version](https://img.shields.io/badge/v3.10-4295B3?style=for-the-badge&logo=python&logoColor=white)



Code to process and analyze audio recordings from the 2024 Phenoscale Pilot project.

## Table of Contents
- [2024 Phenoscale Pilot](#2024-phenoscale-pilot)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Scripts Overview](#scripts-overview)
  - [Contributing](#contributing)
  - [License](#license)

## Project Structure

    .
    ├── .gitignore
    ├── 0.0-read_kmz.py
    ├── 0.1-metadata.py
    ├── 1.0-inference.py
    ├── 2.0-read_results.py
    ├── data/
    │   ├── derived/
    │   └── ...
    ├── logs/
    │   └── ...
    ├── metadata/
    │   ├── pilot_metadata.csv
    │   └── ...
    ├── output/
    ├── src/
    └── README.md

## Installation

1. Clone the repository:
   git clone https://github.com/nilomr/phenoscale-pilot.git
   cd phenoscale-pilot

2. Create a virtual environment and activate it, say using `venv`:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Scripts Overview

| Script Name        | Description                                                                                                                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0.0-read_kmz.py`  | This script extracts pilot points from an Organic Maps backup file (.kmz) and saves them to a CSV file.                                                                                         |
| `0.1-metadata.py`  | This script reads the recorder deployment metadata file, plots a map of the spatial locations, filters files based on deployment dates, and saves the file index as a JSON for later use.       |
| `1.0-inference.py` | This script performs inference on audio recordings using the BirdNET-Analyzer model. It reads metadata, filters relevant audio files, and processes them to generate detections and embeddings. |

## Contributing

Contributions are welcome! Please read the CONTRIBUTING.md file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT licence. See the LICENSE file for more details.