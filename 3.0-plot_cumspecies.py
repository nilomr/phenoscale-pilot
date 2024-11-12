import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import pyproj
import seaborn as sns
from labellines import labelLines
from matplotlib.lines import Line2D
from sklearn.metrics import euclidean_distances

from src.plot import (
    cm,
    plot_aes,
    set_plot_title,
    set_x_axis_title,
    set_x_ticks,
    set_y_axis_title,
    set_y_ticks,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = Path(REPO_ROOT) / "metadata"
DETECTIONS_DIR = Path(REPO_ROOT, "data", "derived")

plot_aes()


# ──── VERY SPECIFIC FUNCTIONS ────────────────────────────────────────────────


def plot_cumlines(experiment_df, palette):
    for i, device in enumerate(experiment_df["device"].unique().to_list()):
        device_df = experiment_df.filter(pl.col("device") == device)
        device_df = device_df.with_columns(
            n=pl.col("cumulative_unique_species").list.len()
        )
        plt.plot(
            device_df["day"],
            device_df["n"],
            label=f"Device {device}",
            color=palette[i],
        )


def plot_cumrate(experiment_df, column_name, palette):
    for i, device in enumerate(experiment_df["device"].unique().to_list()):
        device_df = experiment_df.filter(pl.col("device") == device)
        plt.plot(
            device_df["day"],
            device_df[column_name],
            label=f"Device {device}",
            color=palette[i],
        )


# ──── PLOTTING ───────────────────────────────────────────────────────────────


# read the data
df = pl.read_csv(DETECTIONS_DIR / "detections.csv", try_parse_dates=True)

# Read in the metadata
metadata = pl.read_csv(Path(METADATA_DIR, "pilot_metadata.csv"))

# extract detection date to a detection_date column from the detection_time
# column
df = df.with_columns(pl.col("detection_time").dt.date().alias("detection_date"))

unique_species = (
    df.group_by(["detection_date", "device", "experiment"])
    .agg(pl.col("common_name").unique().alias("unique_species"))
    .sort(["detection_date", "experiment", "device"])
)

cum_unique_species = unique_species.with_columns(
    pl.col("unique_species")
    .cumulative_eval(pl.element().explode().unique().sort().implode())
    .list.drop_nulls()
    .over(["device", "experiment"])
    .alias("cumulative_unique_species")
)

# Calculate the day number from the start of the experiment for each device
cum_unique_species = cum_unique_species.with_columns(
    (
        pl.col("detection_date")
        - pl.col("detection_date").min().over(["experiment", "device"])
    )
    .dt.total_days()
    .alias("day")
)

# Calculate a windowed rate of change in the number of unique species detected
wsize = 7
change_rates = (
    cum_unique_species.with_columns(
        pl.col("cumulative_unique_species")
        .list.len()
        .over("detection_date")
        .alias("n")
    )
    .with_columns(
        pl.col("n").diff().over(["device", "experiment"]).alias("change_rate")
    )
    .sort(["experiment", "device", "detection_date"])
    .with_columns(
        pl.col("change_rate")
        .rolling_mean(wsize)
        .over(["device", "experiment"])
        .alias("rolling_mean")
    )
)


# plot the cumulative number of unique species detected over time for each
# device and site in each experiment

output_dir = Path(REPO_ROOT) / "output" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

width = 8
aspect_ratio = 0.8

palette = [
    "#EF7157",
    "#EFB559",
    "#EF8B4E",
    "#F05A5C",
    "#c2a045",
    "#e6ac3a",
    "#8B7A75",
    "#6d5b47",
    "#885a55",
    "#8C8F7C",
]


for experiment in df["experiment"].unique().to_list():
    experiment_df = cum_unique_species.filter(
        pl.col("experiment") == experiment
    )
    plt.figure(figsize=(width * cm, (width * cm) / aspect_ratio))
    plot_cumlines(experiment_df, palette)
    ax = plt.gca()
    set_y_ticks(ax, num_ticks=5, decimals=0, ymin=0)
    set_x_ticks(ax, num_ticks=6, decimals=0, xmin=0)
    set_y_axis_title(ax, "Species", offset=2)
    set_plot_title(
        ax,
        f"{experiment}".capitalize(),
        "Cumulative Number of Unique Species Detected",
        subtitle_pad=0.6,
        title_pad=0.13,
    )
    labelLines(plt.gca().get_lines(), align=True, fontsize=8)
    plt.savefig(
        output_dir / f"cumulative_species_{experiment}.svg",
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )


for experiment in df["experiment"].unique().to_list():
    experiment_df = change_rates.filter(pl.col("experiment") == experiment)
    experiment_df = experiment_df.filter(pl.col("rolling_mean").is_not_null())
    plt.figure(figsize=(width * cm, (width * cm) / aspect_ratio))
    plot_cumrate(experiment_df, "rolling_mean", palette)
    ax = plt.gca()
    set_y_ticks(ax, num_ticks=5, decimals=1, ymin=0)
    set_x_ticks(
        ax, num_ticks=7, decimals=0, xmin=wsize, xmax=experiment_df["day"].max()
    )
    set_y_axis_title(ax, "Species", offset=2)
    set_plot_title(
        ax,
        f"{experiment}".capitalize(),
        "Rate of Change in New Species",
        subtitle_pad=0.6,
        title_pad=0.13,
    )
    labelLines(plt.gca().get_lines(), align=True, fontsize=8)
    plt.savefig(
        output_dir / f"rate_of_change_{experiment}.svg",
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )


# Now we plot the cumulative number of unique species detected but if we only
# use data from alternate days in unique_species to calculate cum_unique_species

# Add day column and filter for alternate days
filtered_df = unique_species.with_columns(
    (
        pl.col("detection_date")
        - pl.col("detection_date").min().over(["experiment", "device"])
    )
    .dt.total_days()
    .alias("day")
).filter(pl.col("day") % 2 == 0)

# Recalculate cumulative species for alternate days
alt_cum_species = filtered_df.with_columns(
    pl.col("unique_species")
    .cumulative_eval(pl.element().explode().unique().sort().implode())
    .list.drop_nulls()
    .over(["device", "experiment"])
    .alias("cumulative_unique_species")
)

# Plot original and subsampled data side by side for each experiment
for experiment in df["experiment"].unique().to_list():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(width * 2 * cm, (width * cm) / aspect_ratio)
    )

    # Plot original data
    experiment_df = cum_unique_species.filter(
        pl.col("experiment") == experiment
    )
    plt.sca(ax1)
    plot_cumlines(experiment_df, palette)
    ymax = experiment_df.select(
        (pl.col("cumulative_unique_species").list.len())
    ).max()[0, 0]
    set_y_ticks(ax1, num_ticks=5, decimals=0, ymin=0, ymax=ymax)
    set_x_ticks(ax1, num_ticks=6, decimals=0, xmin=0)
    set_y_axis_title(ax1, "Species", offset=2)
    set_plot_title(
        ax1,
        f"{experiment}".capitalize(),
        "All Days",
        subtitle_pad=0.6,
        title_pad=0.13,
    )
    labelLines(ax1.get_lines(), align=True, fontsize=8)

    # Plot subsampled data
    experiment_df_alt = alt_cum_species.filter(
        pl.col("experiment") == experiment
    )
    plt.sca(ax2)
    plot_cumlines(experiment_df_alt, palette)
    set_y_ticks(ax2, num_ticks=5, decimals=0, ymin=0, ymax=ymax)
    set_x_ticks(ax2, num_ticks=6, decimals=0, xmin=0)
    set_y_axis_title(ax2, "Species", offset=2)
    set_plot_title(
        ax2,
        f"{experiment}".capitalize(),
        "Alternate Days",
        subtitle_pad=0.6,
        title_pad=0.13,
    )
    labelLines(ax2.get_lines(), align=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"cumulative_species_{experiment}_comparison.svg",
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )


# Bar chart of species counts

species_counts = (
    df.group_by(["common_name", "experiment"])
    .agg(pl.col("common_name").count().alias("count"))
    .sort(["count"], descending=True)
    .filter(pl.col("count") > 3)
)

# Create plot
aspect_ratio = 0.4
fig, ax = plt.subplots(figsize=(width * cm, (width * cm) / aspect_ratio))

# Get unique experiments and speciess
experiments = species_counts["experiment"].unique().to_list()
mean_counts = (
    species_counts.group_by("common_name")
    .agg(pl.col("count").mean().alias("mean_count"))
    .sort("mean_count", descending=False)
)
species = mean_counts["common_name"].to_list()

# Plot parameters
width_bar = 0.4
y_positions = np.arange(len(species))

# Plot bars for each experiment
for i, exp in enumerate(experiments):
    exp_data = species_counts.filter(pl.col("experiment") == exp)
    counts = []
    for sp in species:
        count = exp_data.filter(pl.col("common_name") == sp).select("count")
        counts.append(count[0, 0] if len(count) > 0 else 0)

    ax.barh(
        y_positions + i * width_bar,
        counts,
        width_bar,
        label=exp,
        color=palette[i],
    )

# Set labels and titles
plt.yticks(y_positions + width_bar / 2, species)
set_x_ticks(ax, num_ticks=6, decimals=0, xmin=0)
set_y_axis_title(ax, "Species", offset=2)
set_x_axis_title(ax, "Number of Detections")
set_plot_title(
    ax,
    "All Species Detected",
    "Number of Detections",
    subtitle_pad=0.6,
    title_pad=0.13,
)
plt.legend()

# Save plot
plt.savefig(
    output_dir / "species_counts.svg",
    format="svg",
    bbox_inches="tight",
    transparent=True,
)

# Plot when in the day detections are made for each species and experiment
# Extract hour from detection_time
# Extract hour and minute,
# Then create a time bin for each 10 minute interval, summing detections in each
# bin fore each species and experiment

df_times = df.with_columns(
    [
        pl.col("detection_time").dt.hour().alias("hour"),
        pl.col("detection_time").dt.minute().alias("minute"),
        (
            pl.col("detection_time").dt.hour() * 6
            + pl.col("detection_time").dt.minute() // 10
        ).alias("time_bin"),
    ]
)


# Group by species, experiment, and time bin to get counts
binned_counts = (
    df_times.group_by(["common_name", "experiment", "time_bin"])
    .len()
    .sort(["experiment", "common_name", "time_bin"])
)

# Create all possible time bins (0 to 143 for 24 hours * 6 bins per hour)
all_bins = pl.DataFrame({"time_bin": pl.Series(range(144), dtype=pl.Int16)})

# Create all combinations of species, experiments and time bins
species_df = pl.DataFrame({"common_name": df["common_name"].unique().to_list()})
experiment_df = pl.DataFrame(
    {"experiment": df["experiment"].unique().to_list()}
)
template = species_df.join(experiment_df, how="cross").join(
    all_bins, how="cross"
)

# Join with actual counts and fill missing values with 0
# ensure both time bin columns are of the same type (int16)
template = template.with_columns(pl.col("time_bin").cast(pl.Int16))
binned_counts = binned_counts.with_columns(pl.col("time_bin").cast(pl.Int16))

binned_counts = template.join(
    binned_counts, on=["common_name", "experiment", "time_bin"], how="left"
).with_columns(pl.col("len").fill_null(0))

# TODO
# preserve the date column, then at this point calculate how far from sunrise or
# sunset the detections are made. Then plot in two sections, one for AM and one
# for PM, not as 'time_bin' but as 'time_from_sunrise' and 'time_from_sunset' as
# this will be more informative.


# Get species with more than 3 detections for the plot
species_to_plot = species_counts["common_name"].unique().to_list()

# Plot detections by time bin for one sample species in one experiment
sample_species = species_to_plot[0]
sample_experiment = experiments[0]

sample_data = hourly_counts.filter(
    (pl.col("common_name") == sample_species)
    & (pl.col("experiment") == sample_experiment)
)

# Create plot with specified dimensions
plt.figure(figsize=(width * cm, (width * cm) / aspect_ratio))

# Convert time bins to actual time for x-axis
time_labels = [f"{(bin // 6):02d}:{(bin % 6 * 10):02d}" for bin in range(144)]

# Plot the data
plt.plot(sample_data["time_bin"], sample_data["len"])

# Customize plot
ax = plt.gca()
set_x_ticks(ax, num_ticks=8, decimals=0, xmin=0, xmax=143)
tick_positions = ax.get_xticks().astype(int)
tick_positions = np.clip(tick_positions, 0, 143)
ax.set_xticklabels([time_labels[i] for i in tick_positions])
set_y_ticks(ax, num_ticks=5, decimals=0, ymin=0)
set_y_axis_title(ax, "Number of Detections", offset=2)
set_x_axis_title(ax, "Time of Day")
set_plot_title(
    ax,
    f"{sample_species} - {sample_experiment}",
    "Daily Detection Pattern",
    subtitle_pad=0.6,
    title_pad=0.13,
)

# Add vertical lines to separate AM and PM recording periods
# First, analyze the data to find the actual recording periods. the second part
# of the file_name in df contains the recording times '20240625_031700'. we can
# use this to find the start and end times of the recording periods
