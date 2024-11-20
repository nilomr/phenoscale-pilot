import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.nonparametric.smoothers_lowess as lowess
from astral import LocationInfo
from astral.sun import sun
from labellines import labelLines

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

# Calculate the total proportion reduction in the number of unique species detected if we had sampled every other day (for each experiment and device)

# Calculate the total number of unique species detected for each experiment and device
total_species = cum_unique_species.group_by(["experiment", "device"]).agg(
    pl.col("cumulative_unique_species").list.len().alias("total_species")
)

# Calculate the total number of unique species detected for each experiment and device when sampled every other day
alt_total_species = alt_cum_species.group_by(["experiment", "device"]).agg(
    pl.col("cumulative_unique_species").list.len().alias("alt_total_species")
)

# Merge the two dataframes
merged_species = total_species.join(
    alt_total_species, on=["experiment", "device"], how="inner"
)

# Calculate the proportion reduction
merged_species = merged_species.with_columns(
    (
        (
            pl.col("total_species").list.max()
            - pl.col("alt_total_species").list.max()
        )
        / pl.col("total_species").list.max()
    ).alias("proportion_reduction")
)

# add columns for max species detected and max species detected on alternate
# days, and difference between the two
merged_species = merged_species.with_columns(
    pl.col("total_species").list.max().alias("max_species"),
    pl.col("alt_total_species").list.max().alias("max_species_alt"),
)

merged_species = merged_species.with_columns(
    (pl.col("max_species") - pl.col("max_species_alt")).alias(
        "species_difference"
    ),
)

# Print the results
print(merged_species)

# plot max species detected and max species detected on alternate days for each
# device and experiment. plot each number as a point and join them with a line.
# species count in the y axis and device in the x axis. color by experiment.

# Create plot
aspect_ratio = 0.8
plt.figure(figsize=(width * cm, (width * cm) / aspect_ratio))

# Plot data for each experiment
# Pre-calculate x positions for all experiments
x_positions = []
current_x = 0
experiment_start_positions = {}

for experiment in merged_species["experiment"].unique():
    exp_data = merged_species.filter(pl.col("experiment") == experiment)
    positions = [current_x + j * 0.6 for j in range(len(exp_data))]
    x_positions.extend(positions)
    experiment_start_positions[experiment] = positions
    current_x = positions[-1] + 1.2  # Add larger gap between experiments

# Plot data using pre-calculated positions
for i, experiment in enumerate(merged_species["experiment"].unique()):
    exp_data = merged_species.filter(pl.col("experiment") == experiment)
    x_pos = experiment_start_positions[experiment]

    # Plot arrows showing direction of change
    for x, full, alt in zip(
        x_pos, exp_data["max_species"], exp_data["max_species_alt"]
    ):
        # Determine arrow direction
        if alt > full:
            plt.arrow(
                x,
                full,
                0,
                alt - full,
                head_width=0.4,
                head_length=0.8,
                color=palette[i],
                alpha=0.7,
            )
        elif alt < full:
            plt.arrow(
                x,
                full,
                0,
                alt - full,
                head_width=0.4,
                head_length=0.8,
                color=palette[i],
                alpha=0.7,
            )
        elif alt == full:
            plt.plot(x, full, "o", color=palette[i])

        # Plot only the higher point
        plt.plot(
            x,
            max(full, alt),
            "o",
            color=palette[i],
            label=experiment if x == x_pos[0] else "",
        )

# Customize plot
ax = plt.gca()
set_y_ticks(ax, num_ticks=5, decimals=0, ymin=0)
set_y_axis_title(ax, "Species", offset=2)
# set x axes ticks (experiment, with each tick at the mean of the x positions for that experiment)
x_ticks = [
    np.mean(experiment_start_positions[exp])
    for exp in merged_species["experiment"].unique()
]
plt.xticks(x_ticks, merged_species["experiment"].unique())
set_plot_title(
    ax,
    "Species Detection Comparison",
    "Full vs. Alternate Day Sampling",
    subtitle_pad=0.6,
    title_pad=0.13,
)
plt.legend()

# Save plot
plt.savefig(
    output_dir / "species_comparison_by_device.svg",
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
# Group by species, experiment, date, and time bin to get counts
df_times = df_times.with_columns(
    pl.col("detection_time").dt.date().alias("detection_date")
)

binned_counts = (
    df_times.group_by(
        ["common_name", "experiment", "detection_date", "time_bin"]
    )
    .len()
    .sort(["experiment", "common_name", "detection_date", "time_bin"])
)

# Fill any missing values with 0
binned_counts = binned_counts.with_columns(
    [
        pl.col("time_bin").cast(pl.Int16),
        # Add bin start time
        (
            pl.col("time_bin")
            .map_elements(lambda x: f"{x//6:02d}:{(x%6)*10:02d}")
            .str.strptime(pl.Time, "%H:%M")
        ).alias("bin_start"),
    ]
)


# Calculate sunrise and sunset times for each date

# Get location from metadata (first row)
lat = metadata["Latitude"][0]
lon = metadata["Longitude"][0]
location = LocationInfo("Site", "UK", "UTC", lat, lon)


# Create function to get sun times for a date
def get_sun_times(date):
    s = sun(location.observer, date)
    return s["sunrise"], s["sunset"]


# Calculate sunrise/sunset for each unique date
dates = df["detection_time"].dt.date().unique().to_list()
sun_times = {date: get_sun_times(date) for date in dates}

# Add sunrise/sunset times for that day as a new column to binned_counts
binned_counts = binned_counts.with_columns(
    pl.col("detection_date")
    .map_elements(lambda x: sun_times[x][0])
    .alias("sunrise"),
    pl.col("detection_date")
    .map_elements(lambda x: sun_times[x][1])
    .alias("sunset"),
)

# For each row calculate the time from bin start to sunrise and sunset. just
# subtract the bin_start column from the sunrise and sunset columns (bin_start -
# sunrise, bin_start - sunset). 2

# You can try using the Expr.sub() function instead of the - operator:

#  (pl.from_records([{'start': '2021-01-01', 'end': '2022-01-01'}])
#  .with_columns(pl.col(['start', 'end']).str.to_date('%Y-%m-%d'))
#  .with_columns(delta = pl.col('end').sub(pl.col('start'))))

# Calculate minutes from sunrise and sunset
binned_counts = binned_counts.with_columns(
    [
        pl.col("detection_date")
        .dt.combine(pl.col("bin_start"))
        .sub(pl.col("sunrise"))
        .dt.total_minutes()
        .alias("minutes_from_sunrise"),
        pl.col("detection_date")
        .dt.combine(pl.col("bin_start"))
        .sub(pl.col("sunset"))
        .dt.total_minutes()
        .alias("minutes_from_sunset"),
    ]
)

# Get absolute values and add AM/PM flag
binned_counts = binned_counts.with_columns(
    [
        pl.col("minutes_from_sunrise").abs().alias("minutes_from_sunrise_abs"),
        pl.col("minutes_from_sunset").abs().alias("minutes_from_sunset_abs"),
    ]
)

# Add time of day flag, AM if minutes_from_sunrise_abs < minutes_from_sunset_abs
binned_counts = binned_counts.with_columns(
    pl.when(
        pl.col("minutes_from_sunrise_abs") < pl.col("minutes_from_sunset_abs")
    )
    .then(pl.lit("AM"))
    .otherwise(pl.lit("PM"))
    .alias("time_of_day")
)


# order based on experiment, count.
binned_counts = binned_counts.sort(["experiment", "len"])

# For AM data, check what the largest negative value is
binned_counts.filter(pl.col("time_of_day") == "AM").select(
    "minutes_from_sunrise"
).min()


# Take a sample species from the first experiment and plot 'count' vs
# 'minutes_from_sunrise_abs' or 'minutes_from_sunset_abs' for AM and PM
# as a time series. with a subplot for AM and PM detections

# Get first experiment and a sample species
experiment = binned_counts["experiment"].unique()[1]
# get unique species, in order of total count
species = (
    binned_counts.filter(pl.col("experiment") == experiment)
    .group_by("common_name")
    .agg(pl.col("len").sum())
    .sort("len", descending=True)
    .select("common_name")
)
species = species["common_name"][5]


def plot_detection_timing(ax, data, period, palette, y_max):
    """Plot detection timing data for a given period (AM/PM)."""
    # Plot raw data points
    ax.plot(
        data[f"minutes_from_{period[1]}"], data["len"], "|", color=palette[0]
    )

    # Calculate and plot LOWESS smoothing
    x = data[f"minutes_from_{period[1]}"].to_numpy()
    y = data["len"].to_numpy()
    loess = lowess.lowess(y, x, frac=1, it=5)
    ax.plot(loess[:, 0], loess[:, 1], "-", color=palette[0], alpha=0.8)

    # Add reference line at sunrise/sunset
    ax.axvline(x=0, color="orange", linestyle="-", alpha=0.4, linewidth=1.4)

    # Set axis properties
    set_y_ticks(ax, num_ticks=5, decimals=0, ymin=0, ymax=y_max)
    if period[0] == "AM":
        ax.set_xticks([0, 50, 100])
        ax.set_xlim(-40, 130)
    else:
        ax.set_xticks([-100, -50, 0])
        ax.set_xlim(-130, 40)

    # Set labels
    ax.set_xlabel(f"Minutes from {period[1]}")
    ax.set_ylabel("Detections")
    ax.set_title(f"{'Morning' if period[0]=='AM' else 'Evening'} detections")


def get_species_list(binned_counts, experiment):
    """Get ordered list of species for an experiment."""
    return (
        binned_counts.filter(pl.col("experiment") == experiment)
        .group_by("common_name")
        .agg(pl.col("len").sum())
        .sort("len", descending=True)
        .select("common_name")
    )["common_name"]


def process_species_data(binned_counts, experiment, species):
    """Get filtered data for a species in an experiment."""
    return binned_counts.filter(
        (pl.col("experiment") == experiment)
        & (pl.col("common_name") == species)
        & (pl.col("len") > 0)
    )


def setup_timing_plot(width, aspect_ratio):
    """Create and setup the figure and axes."""
    return plt.subplots(
        1, 2, figsize=(width * 2 * cm, (width * cm) / aspect_ratio)
    )


def save_timing_plot(fig, output_dir, experiment, species):
    """Save the plot to file."""
    plt.suptitle(f"{species} - {experiment}")
    plt.tight_layout()
    filename = f"timing_{experiment}_{species.replace(' ', '_')}.svg"
    plt.savefig(
        output_dir / filename,
        format="svg",
        bbox_inches="tight",
        transparent=False,
    )
    plt.close()


def create_timing_plots(
    binned_counts, output_dir, width, aspect_ratio, palette
):
    """Create timing plots for all experiments and species."""
    for experiment in binned_counts["experiment"].unique():
        species_list = get_species_list(binned_counts, experiment)

        for species in species_list:
            data = process_species_data(binned_counts, experiment, species)
            if len(data) <= 10:
                continue

            fig, (ax1, ax2) = setup_timing_plot(width, aspect_ratio)

            # Plot morning and evening data
            plot_detection_timing(
                ax1,
                data.filter(pl.col("time_of_day") == "AM"),
                ("AM", "sunrise"),
                palette,
                data["len"].max(),
            )
            plot_detection_timing(
                ax2,
                data.filter(pl.col("time_of_day") == "PM"),
                ("PM", "sunset"),
                palette,
                data["len"].max(),
            )

            save_timing_plot(fig, output_dir, experiment, species)


# Main execution
create_timing_plots(binned_counts, output_dir, width, aspect_ratio, palette)
