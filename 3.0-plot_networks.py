import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
import pyproj
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import euclidean_distances

from src.plot import cm, plot_aes

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = Path(REPO_ROOT) / "metadata"
DETECTIONS_DIR = Path(REPO_ROOT, "data", "derived")

plot_aes()

# ──── PLOTTING ───────────────────────────────────────────────────────────────


# read the data
df = pl.read_csv(DETECTIONS_DIR / "detections.csv", try_parse_dates=True)

# Read in the metadata
metadata = pl.read_csv(Path(METADATA_DIR, "pilot_metadata.csv"))


# Plot the frequency distribution of detections by date for each device
df = df.with_columns(pl.col("detection_time").dt.date().alias("detection_date"))
timesdf = df.group_by(["detection_date", "device", "experiment"]).agg(
    pl.col("detection_date").count().alias("count")
)


plt.figure(figsize=(12, 6))
experiments = df["experiment"].unique().to_list()

for experiment in experiments:
    plt.figure(figsize=(12, 6))
    experiment_df = df.filter(pl.col("experiment") == experiment)
    timesdf_experiment = (
        experiment_df.group_by(["detection_date", "device"])
        .agg(pl.col("detection_date").count().alias("count"))
        .sort("detection_date")
    )  # Ensure data is sorted by detection_date
    for device in timesdf_experiment["device"].unique().to_list():
        subset = timesdf_experiment.filter(pl.col("device") == device)
        plt.plot(
            subset["detection_date"], subset["count"], label=f"Device {device}"
        )
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.title(f"Frequency distribution of detections by date for {experiment}")
    plt.legend()
    plt.show()


# get a list of unique "common_name" values for each device and site
unique_birds = df.group_by(["device", "experiment"]).agg(pl.col("common_name"))
unique_birds = unique_birds.sort(["device", "experiment"])

# Calculate the Jaccard similarity between each pair of devices in each experiment
similarity = []
for experiment in experiments:
    experiment_df = unique_birds.filter(pl.col("experiment") == experiment)
    devices = experiment_df["device"].unique().to_list()
    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            device1 = set(
                experiment_df.filter(pl.col("device") == devices[i])[
                    "common_name"
                ][0].to_list()
            )
            device2 = set(
                experiment_df.filter(pl.col("device") == devices[j])[
                    "common_name"
                ][0].to_list()
            )
            intersection = len(device1 & device2)
            union = len(device1 | device2)
            jaccard_similarity = intersection / union if union != 0 else 0
            similarity.append(
                {
                    "experiment": experiment,
                    "device1": devices[i],
                    "device2": devices[j],
                    "similarity": jaccard_similarity,
                }
            )


# to a polars dataframe
similarity_df = pl.DataFrame(similarity)


# plot as a network graph
def plot_network_graph(
    experiment,
    similarity_df,
    metadata,
    df,
    node_color="#cf9a4a",
    edge_color="#333333",
    palette="viridis",
):
    G = nx.Graph()
    experiment_similarity_df = similarity_df.filter(
        pl.col("experiment") == experiment
    )
    for row in experiment_similarity_df.iter_rows(named=True):
        G.add_edge(row["device1"], row["device2"], weight=row["similarity"])

    pos = get_device_coordinates(experiment, metadata)
    unique_species_counts = get_unique_species_counts(experiment, df)
    node_rarity = get_nodewise_rarity(experiment, df)

    # get edgecolors based on node rarity (scale across a palette)
    norm = plt.Normalize(min(node_rarity.values()), max(node_rarity.values()))
    cmap = plt.colormaps.get_cmap(palette)
    edgecolors = [cmap(norm(node_rarity[node])) for node in G.nodes]

    linewidths = [node_rarity[node] for node in G.nodes]
    linewidths = [
        1 + 5 * (w - min(linewidths)) / (max(linewidths) - min(linewidths))
        for w in linewidths
    ]
    node_sizes = scale_node_sizes(unique_species_counts)
    edges, scaled_weights = scale_edge_weights(G)

    plt.figure(figsize=(5, 5))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=[node_sizes[node] for node in G.nodes],
        node_color=node_color,
        edgecolors=edgecolors,
        linewidths=linewidths,
    )
    draw_edges_with_weights(G, pos, edges, scaled_weights, edge_color)
    annotate_nodes_with_species_counts(pos, unique_species_counts, node_rarity)

    plt.title(
        f"Similarity between unique bird lists for each pair of devices in {experiment}"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.gca().set_aspect("equal", adjustable="box")  # Ensure axes are equal

    # Add legend for edge widths
    add_edge_width_legend(G)

    # Add a 100m scale bar
    add_scale_bar(plt.gca(), 100)


def draw_edges_with_weights(G, pos, edges, scaled_weights, edge_color):
    for edge, weight in zip(edges, scaled_weights):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[edge],
            width=weight,
            edge_color=edge_color,
            alpha=min(1, 0.1 + 0.9 * (weight / 10)),
        )


def convert_to_osgb36(lon, lat):
    wgs84 = pyproj.Proj("epsg:4326")
    osgb36 = pyproj.Proj("epsg:27700")
    transformer = pyproj.Transformer.from_proj(wgs84, osgb36)
    easting, northing = transformer.transform(lat, lon)
    return easting, northing


def get_device_coordinates(experiment, metadata):
    experiment_metadata = metadata.filter(pl.col("test") == experiment)
    return {
        row["n"]: convert_to_osgb36(row["Longitude"], row["Latitude"])
        for row in experiment_metadata.iter_rows(named=True)
    }


def get_unique_species_counts(experiment, df):
    unique_species_counts = (
        df.filter(pl.col("experiment") == experiment)
        .group_by("device")
        .agg(pl.col("common_name").n_unique().alias("unique_species_count"))
    )
    return {
        row["device"]: row["unique_species_count"]
        for row in unique_species_counts.iter_rows(named=True)
    }


def get_nodewise_rarity(experiment, df):
    species_rarity = (
        df.filter(pl.col("experiment") == experiment)
        .group_by(["common_name", "device"])
        .agg(pl.col("common_name").count().alias("count"))
        .group_by("common_name")
        .agg(pl.col("device").count().alias("device_count"))
        .with_columns(
            (1 - (pl.col("device_count") / len(df["device"].unique()))).alias(
                "rarity"
            )
        )
    )
    device_rarity = (
        df.filter(pl.col("experiment") == experiment)
        .join(species_rarity, on="common_name")
        .group_by("device")
        .agg(pl.col("rarity").mean().alias("average_rarity"))
    )

    return {
        row["device"]: row["average_rarity"]
        for row in device_rarity.iter_rows(named=True)
    }


def scale_node_sizes(unique_species_counts):
    return {
        device: count * 50 for device, count in unique_species_counts.items()
    }


def scale_edge_weights(G):
    edges = G.edges(data=True)
    weights = [edge[2]["weight"] for edge in edges]
    min_weight = min(weights)
    max_weight = max(weights)
    scaled_weights = [
        8 * (w - min_weight) / (max_weight - min_weight) for w in weights
    ]
    return edges, scaled_weights


def annotate_nodes_with_species_counts(pos, unique_species_counts, rarity):
    for node, (x, y) in pos.items():
        us = unique_species_counts[node]
        r = rarity[node]
        plt.annotate(
            f"Unique Species: {us}\nRarity: {r:.2f}",
            xy=(x, y),
            xytext=(15, 15),
            textcoords="offset points",
        )


def add_edge_width_legend(G):
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    min_weight = min(weights)
    max_weight = max(weights)
    legend_elements = [
        Line2D([0], [0], color="#333333", lw=1, label=f"{min_weight:.2f}"),
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=5,
            label=f"{(min_weight + max_weight) / 2:.2f}",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=10,
            label=f"{max_weight:.2f}",
        ),
    ]
    plt.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Jaccard\nSimilarity",
    )


def add_scale_bar(ax, length):
    """Add a scale bar to the plot."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    scale_bar_x = x0 + (x1 - x0) * 0.05
    scale_bar_y = y0 + (y1 - y0) * 0.05
    ax.plot(
        [scale_bar_x, scale_bar_x + length],
        [scale_bar_y, scale_bar_y],
        color="k",
    )
    ax.text(
        scale_bar_x + length / 2,
        scale_bar_y,
        f"{length} m",
        va="bottom",
        ha="center",
        fontsize=10,
    )


for experiment in experiments:
    plot_network_graph(
        experiment, similarity_df, metadata, df, palette="Spectral_r"
    )
    output_dir = Path(REPO_ROOT) / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"network_graph_{experiment}.svg", format="svg")

# print a list of all unique species detected across experiment pilot01
unique_species = df["common_name"].unique()
print(unique_species.to_list())


# Plot similarity between devices vs geographic distance in metres between devices in each
# experiment. convert lat long to os grid first so we can just do euclidean
# distance between devices
# Define plot aesthetics configuration


def set_spine_bounds(ax, x):
    padding = (max(x) - min(x)) * 0.03  # Add 5% padding on either side
    ax.spines["bottom"].set_bounds(min(x) - padding, max(x) + padding)
    ax.spines["left"].set_bounds(min(ax.get_yticks()), max(ax.get_yticks()))
    ax.set_xlim(min(x) - padding, max(x) + padding)
    ax.set_ylim(min(ax.get_yticks()), max(ax.get_yticks()))


def calculate_distance_between_devices(experiment, metadata):
    experiment_metadata = metadata.filter(pl.col("test") == experiment)
    device_coordinates = {
        row["n"]: convert_to_osgb36(row["Longitude"], row["Latitude"])
        for row in experiment_metadata.iter_rows(named=True)
    }
    return {
        (device1, device2): euclidean_distances(
            [device_coordinates[device1]], [device_coordinates[device2]]
        )[0][0]
        for device1 in device_coordinates
        for device2 in device_coordinates
        if device1 != device2
    }


def plot_distance_vs_similarity(experiment, similarity_df, metadata):
    distances = calculate_distance_between_devices(experiment, metadata)
    experiment_similarity_df = similarity_df.filter(
        pl.col("experiment") == experiment
    )
    distances = [
        distances[(row["device1"], row["device2"])[::-1]]
        for row in experiment_similarity_df.iter_rows(named=True)
    ]
    similarities = experiment_similarity_df["similarity"].to_list()
    width = 5
    aspect_ratio = 1.5
    plt.figure(figsize=(width, width / aspect_ratio))
    ax = plt.gca()
    ax.set_axisbelow(
        True
    )  # Ensure grid lines are below the scatter plot points
    sns.regplot(
        x=distances,
        y=similarities,
        ci=95,
        scatter_kws={"s": 15},
        line_kws={"color": "#43616d"},
        ax=ax,
    )
    plt.xlabel("Distance between devices (m)")
    plt.ylabel("Jaccard similarity")
    plt.title(f"Distance between devices vs similarity in {experiment}")
    set_spine_bounds(ax, distances)
    # fix y range to between 0.2 and 0.8
    ax.set_ylim(0.2, 0.8)


for experiment in experiments:
    plot_distance_vs_similarity(experiment, similarity_df, metadata)
    output_dir = Path(REPO_ROOT) / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / f"distance_vs_similarity_{experiment}.svg", format="svg"
    )
