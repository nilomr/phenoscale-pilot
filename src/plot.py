"""
Usage Example:

import matplotlib.pyplot as plt
from plot import set_y_ticks, set_x_ticks, set_y_axis_title, set_x_axis_title, set_plot_title

fig, ax = plt.subplots()
ax.scatter([1, 2, 3, 4], [1, 4, 6, 12], label="Scatter Plot")
set_y_ticks(ax, step=2, decimals=0)
set_x_ticks(ax, num_ticks=4, decimals=2)
set_y_axis_title(ax, "Y axis title")
set_x_axis_title(ax, "X axis title")
set_plot_title(ax, "Plot title", "Plot subtitle")
plt.show()
"""

import math
import warnings

import matplotlib
import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import TransformedBbox
from numpy.typing import ArrayLike

plot_aesthetics = {
    "title_size": 12,
    "subtitle_size": 10,
    "tick_size": 10,
    "background_color": "white",
    "font_family": "Roboto Condensed",
}


def plot_aes():
    plt.rcParams.update(
        {
            "axes.titlesize": plot_aesthetics["title_size"],
            "axes.titleweight": "bold",
            "axes.labelsize": plot_aesthetics["subtitle_size"],
            "xtick.labelsize": plot_aesthetics["tick_size"],
            "ytick.labelsize": plot_aesthetics["tick_size"],
            "axes.facecolor": plot_aesthetics["background_color"],
            "figure.facecolor": plot_aesthetics["background_color"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.axisbelow": True,
            "grid.color": "#d3d3d3",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "xtick.major.pad": 8,
            "ytick.major.pad": 15,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "ytick.color": "black",  # Add y-axis tick values
            "ytick.left": False,  # Remove y-axis tick marks
            "font.family": plot_aesthetics["font_family"],
            "text.usetex": False,
            "svg.fonttype": "none",
        }
    )


def set_y_ticks(
    ax: matplotlib.axes.Axes,
    step: int = None,
    num_ticks: int = None,
    decimals: int = None,
    **kwargs,
) -> None:
    """
    Adjust the number of y-axis ticks on a given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis object to modify.
    step (int, optional): Step size to thin the y-ticks. If provided, every 'step'-th tick will be kept.
    num_ticks (int, optional): Desired number of y-ticks. If provided, the y-ticks will be set to this number.
    decimals (int, optional): Number of decimal places to round the tick labels to. If None, no rounding is applied.
    **kwargs: Additional keyword arguments for y-axis range (e.g., ymin, ymax).

    Note:
    Only one of 'step' or 'num_ticks' should be provided. If both are provided, 'step' will be used.
    """
    ymin = kwargs.get("ymin", None)
    ymax = kwargs.get("ymax", None)

    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)

    yticks = ax.get_yticks()

    if step is not None:
        ax.set_yticks(yticks[::step])
    elif num_ticks is not None:
        ax.set_yticks(np.linspace(yticks[0], yticks[-1], num_ticks))
    else:
        raise ValueError("Either 'step' or 'num_ticks' must be provided.")

    if decimals is not None:
        ax.set_yticklabels([f"{tick:.{decimals}f}" for tick in ax.get_yticks()])

    # Left-align all Y-axis tick labels
    for label in ax.get_yticklabels():
        label.set_ha("left")


def set_x_ticks(
    ax: plt.Axes,
    step: int = None,
    num_ticks: int = None,
    decimals: int = None,
    **kwargs,
) -> None:
    """
    Adjust the x-axis ticks on a given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis object to modify.
    step (int, optional): The step size between ticks. If provided, sets the x-ticks at intervals of `step`.
    num_ticks (int, optional): The number of ticks to display. If provided, sets the x-ticks to be evenly spaced.
    decimals (int, optional): Number of decimal places to round the tick labels to. If None, no rounding is applied.
    **kwargs: Additional keyword arguments to pass to ax.set_xlim().

    Raises:
    ValueError: If neither 'step' nor 'num_ticks' is provided.
    """
    xticks = ax.get_xticks()

    if step is not None:
        ax.set_xticks(xticks[::step])
    elif num_ticks is not None:
        ax.set_xticks(np.linspace(xticks[0], xticks[-1], num_ticks))
    else:
        warnings.warn(
            "Neither 'step' nor 'num_ticks' provided. Using default x-ticks."
        )

    if decimals is not None:
        ticks = ax.get_xticks()
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
        ax.set_xticklabels([f"{tick:.{decimals}f}" for tick in ticks])

    if kwargs:
        ax.set_xlim(**kwargs)


# Assuming plot_aes, yticks, and xticks functions are already defined


def tight_bbox(ax: Axes) -> TransformedBbox:
    """
    Calculate the tight bounding box of an Axes object in figure coordinates.

    Parameters:
    ax (Axes): The Axes object for which to calculate the tight bounding box.

    Returns:
    TransformedBbox: The tight bounding box of the Axes in figure coordinates.
    """
    fig = ax.get_figure()
    tight_bbox_raw = ax.get_tightbbox(fig.canvas.get_renderer())
    tight_bbox_fig = TransformedBbox(tight_bbox_raw, fig.transFigure.inverted())
    return tight_bbox_fig


def set_y_axis_title(
    ax: Axes, title: str, offset: float = 2.5, style: str = "italic"
) -> None:
    """
    Set the title for the y-axis of a given Axes object with a specified offset and style.

    Parameters:
    ax (Axes): The Axes object to set the y-axis title for.
    title (str): The title text to set for the y-axis.
    offset (float, optional): The offset for the title position relative to the y-axis tick labels. Default is 2.5.
    style (str, optional): The style of the title text. Default is "italic".

    Returns:
    None
    """
    x0, y0, _, height = (
        ax.yaxis.get_ticklabels()[-1]
        .get_window_extent()
        .transformed(ax.transData.inverted())
        .bounds
    )
    ax.text(
        x0,
        y0 + height * offset,
        title,
        ha="left",
        va="bottom",
        style=style,
    )


def set_x_axis_title(ax: Axes, title: str, style: str = "italic") -> None:
    """
    Set the title for the x-axis of a given Axes object with a specified style.

    Parameters:
    ax (Axes): The Axes object to set the x-axis title for.
    title (str): The title text to set for the x-axis.
    style (str, optional): The style of the title text. Default is "italic".

    Returns:
    None
    """
    ax.set_xlabel(title, ha="center", va="top", style=style)


def set_plot_title(
    ax: Axes,
    title: str,
    subtitle: str = None,
    subtitle_pad: float = 0.6,
    title_pad: float = 0.13,
) -> None:
    """
    Set the main title and optional subtitle for a given Axes object.

    Parameters:
    ax (Axes): The Axes object to set the titles for.
    title (str): The main title text to set for the plot.
    subtitle (str, optional): The subtitle text to set for the plot. Default is None.
    subtitle_pad (float, optional): The padding between the figure and the subtitle. Default is 1.
    title_pad (float, optional): The padding between the subtitle and the title. Default is 0.9.

    Returns:
    None
    """

    # Calculate the x position for the title and subtitle
    bbox = ax.get_yticklabels()[-1].get_window_extent()
    x, _ = ax.figure.transFigure.inverted().transform([bbox.x0, bbox.y0])

    # Calculate the y positions for the subtitle and title
    bbox = ax.get_position()
    figure_height = ax.figure.get_size_inches()[1]
    subtitle_y = bbox.y1 + subtitle_pad / figure_height
    title_y = subtitle_y + title_pad / figure_height

    ax.set_title(
        title, ha="left", x=x, y=title_y, transform=ax.figure.transFigure
    )
    if subtitle:
        ax.text(
            x,
            subtitle_y,
            subtitle,
            ha="left",
            va="center",
            transform=ax.figure.transFigure,
        )


cm = 1 / 2.54

# Test the set_plot_title function on different aspect ratio plots
aspect_ratios = [(4, 3), (16, 9), (1, 1), (3, 4)]

for aspect_ratio in aspect_ratios:
    fig, ax = plt.subplots(figsize=aspect_ratio)
    ax.plot([1, 2, 3], [1, 4, 9])  # Example plot
    set_plot_title(
        ax,
        f"Plot with aspect ratio {aspect_ratio[0]}:{aspect_ratio[1]}",
        "Subtitle example",
    )
    plt.show()
