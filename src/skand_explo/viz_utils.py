import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, StrMethodFormatter

DEFAULT_FIGSIZE = (13, 7)
DEFAULT_DPI = 200


def line_chart(
    dates: list[int],
    country_stats: dict[str, dict[str, list[int]]],
    tags_colours: dict[str, str],
    title: str,
) -> None:
    """Make line chart of chosen statistic"""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    for tag, vals in country_stats.items():
        ax.plot(vals["dates"], vals["stats"], color=tags_colours[tag])

        if world_data:
            annotation = f"{tag} | {vals['stats'][-1]:.4%}"
        else:
            annotation = f"{tag} | {vals['stats'][-1]:_.2f}"

        ax.annotate(
            f"{annotation}",
            (max(dates), vals["stats"][-1]),
            fontsize=10,
            color=tags_colours[tag],
            bbox=dict(facecolor="black", edgecolor="white", boxstyle="round"),
        )

    ax.set_yscale("log")
    formatter = FuncFormatter(lambda y, pos: f"{y:_.2f}")
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)

    ax.legend(country_stats.keys(), loc="upper left")
    ax.set_title(f"{title}{f' as % of world {title}' if world_data else ''}")
    ax.set_xticks(dates)

    if world_data:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.4%}"))

    ax.grid(visible=True, axis="y", alpha=0.25)
    fig.tight_layout()

    return fig


def customize_table(
    data: pd.DataFrame,
    tag_colours: dict[str, str],
    title: str,
    year,
    world_data=False,
) -> None:
    full = data

    if world_data:
        nr_formats = dict.fromkeys(full.keys(), "{:,.4%}")
    else:
        nr_formats = {x: "{:,.1f}" if int(x) > 1_000 else "{:.1%}" for x in full}

    full = full.sort_values(full.columns[-1], ascending=False)

    formatter = full.style

    # customize background and text colour
    formatter = formatter.set_properties(
        **{"background-color": "black", "color": "white"},
    )

    # customize index
    colours_for_index = {k: f"background-color: black; color: {v}" for k, v in tag_colours.items()}
    formatter = formatter.map_index(
        lambda x: colours_for_index.get(x, "background-color: black; color: white"),
        axis=0,
    )
    # customize columns
    formatter = formatter.map_index(
        lambda x: "background-color: black; color: white; font-size: 125%",
        axis=1,
    )
    # customize header
    formatter = formatter.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "black"),
                    ("font-size", "125%"),
                ],
            },
            {
                "selector": "caption",
                "props": [
                    ("background-color", "black"),
                    ("color", "white"),
                    ("font-size", "175%"),
                ],
            },
        ],
    )
    # set caption
    caption = f" as % of world's {title}" if world_data else ""
    formatter = formatter.set_caption(title + caption)

    subset = pd.IndexSlice[[indx for indx in full.index if indx != "WORLD"], :]
    formatter = formatter.highlight_max(color="darkgreen", subset=subset)
    formatter = formatter.highlight_min(color="darkred", subset=subset)

    return formatter.format(nr_formats)
