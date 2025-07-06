"""Skanderbeg Analyzer.

This module provides tools for downloading (via Skanderbeg.pm),
processing, and analyzing EU4 save data,
including country and province statistics, visualizations, and data exports.
"""

import argparse
import base64
import datetime
import itertools
import locale
import os
import pickle
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from itertools import pairwise
from pathlib import Path
from typing import NoReturn

import dataframe_image as dfi
import geopandas as gpd
import matplotlib.colors
import matplotlib.patheffects as path_effects
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from pandas.io.formats.style import Styler
from PIL import Image
from requests.exceptions import RequestException

locale.setlocale(locale.LC_ALL, ("Polish_Poland", "1250"))


SKANDERBEG_LINK = "https://skanderbeg.pm/api.php"
SKANDERBEG_API_KEY_ENV_VAR_NAME = "SKANDERBEG_API_KEY"

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
CHARTS_PATH = ROOT_PATH / "charts"
CONFIG_PATH = ROOT_PATH / "config"
OUTPUT_PATH = ROOT_PATH / "output"
DATA_PATH = ROOT_PATH / "data"
CACHE_PATH = ROOT_PATH / "cache"

CHARTS_PATH.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True)
CACHE_PATH.mkdir(exist_ok=True)

ALL_STATS = Path(CONFIG_PATH / "available_stats.txt").read_text(encoding="utf-8").splitlines()
ALL_STATS.sort()


@dataclass
class Save:
    """Represents a game save with associated metadata and data.

    Attributes
    ----------
    id: str
        Unique identifier for the save.
    hash: str
        Hash value of the save file.
    timestamp: datetime
        Timestamp when the save was created or uploaded.
    name: str
        Name of the save.
    uploaded_by: str
        Name of the uploader.
    version: str
        Game version associated with the save.
    player: str
        Name of the player.
    multiplayer: bool
        Indicates if the save is from a multiplayer game.
    date: date
        In-game date of the save.
    custom_name: str
        Custom name assigned to the save.
    game: str
        Name of the game.
    mods: str
        List or description of mods used.
    linked_sheet: str
        Link to an associated spreadsheet or document.
    options: str
        Game options or settings used.
    last_visited: datetime
        Timestamp of the last visit or access.
    view_count: int
        Number of times the save has been viewed.
    augmented_campaign: str
        Information about any augmented campaign.
    year: int
        In-game year of the save.
    country_data: dict or None, optional
        Data related to countries in the save.
    provinces_data: dict or pandas.DataFrame or None, optional
        Data related to provinces in the save.

    """

    id: str
    hash: str
    timestamp: datetime.datetime
    name: str
    uploaded_by: str
    version: str
    player: str
    multiplayer: bool
    date: datetime.date
    custom_name: str
    game: str
    mods: str
    linked_sheet: str
    options: str
    last_visited: datetime.datetime
    view_count: int
    augmented_campaign: str
    year: int
    country_data: dict | None = None
    provinces_data: dict | pd.DataFrame | None = None

    def __repr__(self) -> str:
        """Return a string representation of the Save instance.

        Returns
        -------
        str
            A string summarizing the object's class and main attributes.

        """
        return (
            f"{self.__class__.__name__}("
            f"year={self.year}, player={self.player}, hash={self.hash}, "
            f"country_data={self.country_data is not None}, "
            f"provinces_data={self.provinces_data is not None})"
        )


@dataclass(frozen=True)
class SkandStat:
    """Represent a statistic with convenient methods.

    A class for statistical analysis and visualization of time series data,
    with support for custom tags, CAGR calculation, and styled export.

    Parameters
    ----------
    statistic: str
        The name of the statistic being analyzed.
    data: pd.DataFrame
        The main data, indexed by year, with columns representing different tags/entities.
    tags: dict[str, dict], optional
        A dictionary mapping tag names to metadata dictionaries (e.g., color, flag).
    save_dates: list[int], optional
        List of years to include in visualizations and tables. If None, uses all years in `data`.
    include_world: bool, default=True
        Whether to include the "WORLD" column in analysis and visualizations.

    Attributes
    ----------
    DEFAULT_FIGSIZE: tuple[int, int]
        Default figure size for visualizations.
    DEFAULT_DPI: int
        Default DPI for exported figures.
    DEFAULT_STYLE: str
        Default matplotlib style for plots.
    DEFAULT_EXPORT_EXTENSION: str
        Default file extension for exported visualizations.

    Methods
    -------
    __repr__()
        Return a string representation of the SkandStat instance.
    current_year
        Return the most recent year in `save_dates` or in the data index.
    line_chart(y_scale='log', figsize=None, style=None, emphesize_tag=None)
        Generate a matplotlib Figure with a line chart of the data.
    data_with_cagr()
        Return a DataFrame with Compound Annual Growth Rate (CAGR) columns inserted between years.
    table(with_cagr=True)
        Return a pandas Styler object for the data, optionally including CAGR columns
        and custom formatting.
    export_viz(viz, dpi=None, extension=None, **kwargs)
        Export a visualization (matplotlib Figure or pandas Styler) to a file.

    Notes
    -----
    - The class is designed for statistical exploration and presentation of time series data,
      with a focus on visual clarity and customizability.
    - Tag metadata can include colors and flags for enhanced visualization and table styling.
    - CAGR calculations are inserted between years and for the total period.
    - Exported visualizations are saved to a predefined `CHARTS_PATH`.

    Examples
    --------
    >>> stat = Analyzer.get_statistic('max_manpower')
    >>> fig = stat.line_chart(emphesize_tag="FRA")
    >>> stat.export_viz(fig)
    >>> styler = stat.table()
    >>> stat.export_viz(styler)

    """

    statistic: str
    data: pd.DataFrame
    tags: dict[str, dict] | None = None
    save_dates: list[int] | None = None
    include_world: bool = True

    DEFAULT_FIGSIZE: tuple[int, int] = field(default=(13, 7), init=False, repr=False)
    DEFAULT_DPI: int = field(default=200, init=False, repr=False)
    DEFAULT_STYLE: str = field(default="dark_background", init=False, repr=False)
    DEFAULT_EXPORT_EXTENSION: str = field(default=".png", init=False, repr=False)

    def __repr__(self) -> str:
        """Return a string representation of the object for debugging.

        Returns
        -------
        str
            A string representation of the object.

        """
        return (
            f"{self.__class__.__name__}"
            f"({self.statistic=}, "
            f"{self.save_dates=}, "
            f"{self.include_world=})"
        )

    @property
    def current_year(self) -> int:
        """Returns the most recent year from the available data.

        If `save_dates` is not set, the method returns the maximum value from the data index.
        Otherwise, it returns the maximum value from `save_dates`.

        Returns
        -------
        int
            The most recent year available.

        Notes
        -----
        Assumes that `self.data.index` and `self.save_dates` contain year values.

        """
        if self.save_dates is None:
            return self.data.index.max()
        return max(self.save_dates)

    def line_chart(
        self,
        y_scale: str = "log",
        figsize: tuple[int, int] | None = None,
        style: str | None = None,
        emphesize_tag: str | None = None,
    ) -> Figure:
        """Generate a matplotlib Figure with a line chart of the data.

        Parameters
        ----------
        y_scale : str, default="log"
            Y-axis scale (e.g., "log", "linear").
        figsize : tuple[int, int], optional
            Figure size. If None, uses default.
        style : str, optional
            Matplotlib style to use. If None, uses default.
        emphesize_tag : str, optional
            Tag to emphasize in the plot.

        Returns
        -------
        Figure
            The matplotlib Figure object containing the line chart.

        """
        figsize = figsize or self.DEFAULT_FIGSIZE
        colours = {} if not self.tags else self.tags
        dates = self.data.index if not self.save_dates else self.save_dates
        style = style or self.DEFAULT_STYLE

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            for tag in self.data.columns:
                if tag == "WORLD":
                    continue

                colour = colours.get(tag, {}).get("colour", "#ffffff")
                rgb = matplotlib.colors.to_rgb(colour)

                if not emphesize_tag:
                    zorder = 2
                    viz_effects = []
                    text_effects = []
                elif tag == emphesize_tag:
                    zorder = 10
                    max_alpha = 20
                    min_alpha = 10
                    viz_effects: list[path_effects.AbstractPathEffect] = [
                        path_effects.Stroke(
                            foreground=rgb,
                            linewidth=max_alpha - i,
                            alpha=i / 100,
                        )
                        for i in range(min_alpha, max_alpha)
                    ]
                    viz_effects.extend([
                        path_effects.withSimplePatchShadow(
                            shadow_rgbFace=rgb,
                            alpha=min_alpha / 100,
                        ),
                    ])
                    text_effects = []
                else:
                    zorder = 2
                    viz_effects = [path_effects.Stroke(foreground=rgb, alpha=0.25)]
                    text_effects = [path_effects.Stroke(foreground=rgb, alpha=0.25)]

                ax.plot(
                    self.data[tag],
                    color=rgb,
                    zorder=zorder,
                    path_effects=viz_effects,
                )

                last_point = self.data[tag].iloc[-1]
                formatted_last_point = locale.format_string("%.0f", last_point, grouping=True)
                annotation = f"{tag} | {formatted_last_point}"
                ann = ax.annotate(
                    f"{annotation}",
                    (self.data.index.max() + 2, last_point),
                    fontsize=10,
                    color=rgb,
                    zorder=zorder,
                    bbox={
                        "facecolor": "black",
                        "edgecolor": rgb,
                        "boxstyle": "round",
                    },
                    path_effects=text_effects,
                )
                bbox_patch = ann.get_bbox_patch()
                if bbox_patch:
                    bbox_patch.set_path_effects(viz_effects)

            ax.set_yscale(y_scale)
            scalar_formatter = ScalarFormatter(useLocale=True)
            scalar_formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(scalar_formatter)
            ax.yaxis.set_minor_formatter(scalar_formatter)
            ax.legend(
                self.data.keys(),
                loc="upper left",
                frameon=False,
                shadow=True,
            )
            ax.set_title(
                f"{self.statistic.upper().replace('_', ' ')}",
                fontsize="xx-large",
                fontstretch="semi-expanded",
                fontweight="roman",
            )
            ax.set_xticks(dates)
            ax.grid(visible=True, which="major", axis="y", alpha=0.25)
            ax.grid(visible=True, which="minor", axis="y", alpha=0.05)
            ax.set_frame_on(False)
            ax.tick_params("y", colors="grey", which="minor")
            ax.xaxis.set_tick_params(color="none")
            ax.yaxis.set_tick_params(color="none", which="both")
            fig.tight_layout()

            return fig

    def data_with_cagr(self) -> pd.DataFrame:
        """Return a DataFrame with CAGR columns inserted between years and for the total period.

        Returns
        -------
        pd.DataFrame
            DataFrame with CAGR columns between years and for the total period.

        """
        data = self.data.T

        min_year = data.columns.min()
        max_year = data.columns.max()

        for i, (start, end) in enumerate(pairwise(data), start=0):
            year_diff = end - start  # type: ignore
            period_cagr = (data[end] / data[start]) ** (1 / year_diff) - 1
            data.insert(2 * i + 1, year_diff, period_cagr, allow_duplicates=True)

        total_year_diff = max_year - min_year
        total_cagr = (data[max_year] / data[min_year]) ** (1 / total_year_diff) - 1
        data.insert(len(data.columns), total_year_diff, total_cagr)
        return data

    def table(self, *, with_cagr: bool = True) -> Styler:
        """Return a custom pandas Styler object for the data, optionally including CAGR columns.

        Parameters
        ----------
        with_cagr : bool, default=True
            Whether to include CAGR columns in the table.

        Returns
        -------
        Styler
            A pandas Styler object for the data table.

        """
        data = self.data_with_cagr() if with_cagr else self.data.T

        # sort by CAGR from whole campaign
        # or there's no CAGR, by last value
        last_column = data.columns[-1]
        data = data.copy(deep=True).sort_values(by=last_column, ascending=False)

        old_cols = data.columns.copy(deep=True)

        # it is required for applying highlights that every index is unique
        # every row index always will be unique, but it's not enforced for columns
        # to make every column unique a invisible character will be added
        # to the end of the column name
        counter = {}
        new_cols = []
        for col in data.columns:
            if col in counter:
                spaces = counter[col]
                counter[col] += 1
            else:
                spaces = 0
                counter[col] = 1
            new_cols.append(str(col) + "\u200b" * spaces)

        data.columns = new_cols

        # ignoring this warning, the same is in the docs
        # https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.io.formats.style.Styler.format.html
        barrier = 1000
        nr_formats = {
            new: "{:,.1f}" if old > barrier else "{:.1%}"  # type: ignore
            for old, new in zip(old_cols, new_cols, strict=False)
        }

        htmls = []
        flags = {} if not self.tags else self.tags
        for tag in data.index:
            flag = flags.get(tag, {}).get("flag", None)
            html = "" if flag is None else f'<img src="data:image/png;base64,{flag}" width="24">'
            htmls.append(html)
        data.insert(0, " ", htmls)

        styler = data.style
        styler.format(nr_formats, na_rep="-", precision=1, decimal=",", thousands=" ")  # type: ignore

        # ignoring this warning, the same is in the docs
        # https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.io.formats.style.Styler.set_properties.html
        styler.set_properties(**{
            "background-color": "black",
            "color": "white",
        })  # type: ignore

        # customize index
        colours = {} if not self.tags else self.tags
        colours_for_index = {k: f"color: {v.get('colour')}" for k, v in colours.items()}
        styler = styler.map_index(
            lambda x: colours_for_index.get(x, "color: white"),  # type: ignore
            axis=0,
        )

        # customize header
        styler = styler.set_table_styles(
            [
                {
                    "selector": "",  # An empty string '' targets the <table> element
                    "props": [
                        ("border", "none"),
                        ("border-collapse", "collapse"),
                        ("color", "white"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "black"),
                        ("font-size", "125%"),
                        ("font-weight", "bold"),
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
        caption = (
            self.statistic.replace(
                "_",
                " ",
            )  # + f"<br>as % of world's {self.statistic}" if self.include_world else ""
        )
        styler = styler.set_caption(caption.upper())

        # add highlitights
        subset = pd.IndexSlice[
            [indx for indx in data.index if indx != "WORLD"],
            new_cols,
        ]
        styler = styler.highlight_max(color="darkgreen", subset=subset, axis="index")  # type: ignore
        return styler.highlight_min(color="darkred", subset=subset, axis="index")  # type: ignore

    def export_viz(
        self,
        viz: Figure | Styler,
        dpi: int | None = None,
        extension: str | None = None,
        **kwargs,
    ) -> Path:
        """Export a visualization (matplotlib Figure or pandas Styler) to a file.

        Parameters
        ----------
        viz : Figure or Styler
            The visualization object to export.
        dpi : int, optional
            Dots per inch for the export. If None, uses default.
        extension : str, optional
            File extension for the export. If None, uses default.
        **kwargs
            Additional keyword arguments for export.

        Returns
        -------
        Path
            Path to the exported file.

        Raises
        ------
        TypeError
            If `viz` is not a matplotlib Figure or pandas Styler.

        """
        extension = extension if extension is not None else self.DEFAULT_EXPORT_EXTENSION
        dpi = dpi if dpi is not None else self.DEFAULT_DPI

        if isinstance(viz, Figure):
            emphesize_tag = kwargs.get("emphesize_tag", "")
            emphesize_tag = "_" + emphesize_tag if emphesize_tag else ""

            file_name = f"{self.statistic}_by_{self.current_year}_line_chart{emphesize_tag}".upper()
            export_path = CHARTS_PATH / Path(file_name).with_suffix(extension)

            viz.savefig(export_path, dpi=dpi)
            return export_path
        if isinstance(viz, Styler):
            file_name = f"{self.statistic}_by_{self.current_year}_table".upper()
            export_path = CHARTS_PATH / Path(file_name).with_suffix(extension)

            dfi.export(
                viz,  # type: ignore
                export_path,
                dpi=dpi,
            )
            return export_path

        msg = "`viz` is not an instance of `Styler` or `Figure`"
        raise TypeError(msg)


class Analyzer:
    """Analyzer for Skanderbeg EU4 data.

    Provides methods to download, process, and analyze EU4 save data, including country and province statistics.
    """

    _EU4_START_DATE: int = 1444

    def __init__(self, api_key: str | None = None, *, force_offline: bool = False) -> None:
        """Initialize the Analyzer.

        Parameters
        ----------
        api_key : str, optional
            API key for Skanderbeg.pm.
        force_offline : bool, default=False
            If True, only use cached data and do not attempt to download.

        """
        self._api_key: str | None = api_key
        self._save_dates: list[int] | None = None
        self._tags: dict[str, dict] | None = None
        self.saves: dict[int, Save] = {}
        self.force_offline: bool = force_offline
        self._downloaded_country_data: bool = False
        self._downloaded_province_data: bool = False
        self._downloaded_flags: bool = False
        self._processed_country_data: bool = False
        self._processed_province_data: bool = False

    @staticmethod
    def read_cached_data() -> dict[int, Save]:
        """Read cached save data from disk.

        Returns
        -------
        dict[int, Save]
            Dictionary of saves loaded from cache, or empty dict if not found.

        """
        cached_saves = Path(CACHE_PATH / "saves.pkl")
        if cached_saves.exists():
            with cached_saves.open("rb") as f:
                return pickle.load(f)
        return {}

    @staticmethod
    def read_cached_tags() -> dict[str, dict]:
        """Read cached tags from disk.

        Returns
        -------
        dict[str, dict]
            Dictionary of tags loaded from cache, or empty dict if not found.

        """
        cached_tags = Path(CACHE_PATH / "tags.pkl")
        if cached_tags.exists():
            with cached_tags.open("rb") as f:
                return pickle.load(f)
        return {}

    @property
    def tags(self) -> dict[str, dict]:
        """Return tags dictionary, loading from cache or config if necessary.

        Returns
        -------
        dict[str, dict]
            Dictionary of tags and their metadata.

        """
        if self._tags:
            return self._tags

        cached_tags = self.read_cached_tags()
        if cached_tags:
            self._tags = cached_tags
            return self._tags

        tags = (CONFIG_PATH / "countries.txt").read_text(encoding="utf-8").split(",")
        tags = {tag: {} for tag in tags}
        self._tags = {tag: {} for tag in tags}
        return self._tags

    @property
    def save_dates(self) -> list[int]:
        """Return sorted list of save years.

        Returns
        -------
        list[int]
            Sorted list of years for which saves are available.

        """
        if self._save_dates:
            return self._save_dates
        if not self.saves:
            self.get_save_metadata()
        self._save_dates = sorted(self.saves.keys())
        return self._save_dates

    @property
    def current_year(self) -> int:
        """Return the most recent year for which a save is available.

        Returns
        -------
        int
            The most recent year.

        """
        if not self.save_dates:
            self.get_save_metadata()
        return max(self.save_dates)

    @property
    def current_save(self) -> Save:
        """Return the Save object for the most recent year.

        Returns
        -------
        Save
            The Save object for the current year.

        """
        if not self.save_dates:
            self.get_save_metadata()
        return self.saves[self.current_year]

    def get_save_metadata(self) -> dict[int, Save]:
        """Download or load metadata for all saves.

        Returns
        -------
        dict[int, Save]
            Dictionary of saves indexed by year.

        Raises
        ------
        RequestException
            If the request for a flag fails (e.g., network error, invalid response).

        """
        if self.saves:
            return self.saves

        if self.force_offline:
            self.saves = self.read_cached_data()
            return self.saves

        params = {"key": self._api_key, "scope": "fetchUserSaves"}
        response = requests.get(SKANDERBEG_LINK, params=params, timeout=10)

        if not response.ok or response.text == "Err":
            raise RequestException(response=response)

        saves = {}
        for save_data in response.json():
            # don't care about timezone awareness

            timestamp = save_data["timestamp"]
            # timestamp format '2025-06-17 01:00:09'
            timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007

            date = save_data["date"]
            # date format '1615.2.15'
            date = datetime.datetime.strptime(date, "%Y.%m.%d").date()  # noqa: DTZ007
            year = date.year

            multiplayer = save_data["multiplayer"] == "yes"

            last_visited = save_data["lastVisited"]
            # date format '2025-06-17 01:00:09'
            last_visited = datetime.datetime.strptime(last_visited, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007

            view_count = save_data["viewCount"]
            view_count = int(view_count)

            saves[year] = Save(
                id=save_data["id"],
                hash=save_data["hash"],
                timestamp=timestamp,
                name=save_data["name"],
                uploaded_by=save_data["uploadedBy"],
                version=save_data["version"],
                player=save_data["player"],
                multiplayer=multiplayer,
                date=date,
                custom_name=save_data["customname"],
                game=save_data["game"],
                mods=save_data["mods"],
                linked_sheet=save_data["linkedSheet"],
                options=save_data["options"],
                last_visited=last_visited,
                view_count=view_count,
                augmented_campaign=save_data["augmentedCampaign"],
                year=year,
            )

        # sort saves ones for all the future uses
        saves = dict(sorted(saves.items()))
        self.saves = saves
        return saves

    def get_country_flags(self) -> dict[str, dict]:
        """Download and cache country flags for all tags.

        Returns
        -------
        dict[str, dict]
            Dictionary of tags with flag data added.

        Raises
        ------
        RequestException
            If the request for a flag fails (e.g., network error, invalid response).

        """
        if self.force_offline or self._downloaded_flags:
            return self.tags

        if self._tags is None:
            self._tags = {}

        with requests.Session() as session:
            for tag in self.tags:
                params = {
                    "scope": "getCountryFlag",
                    "save": self.current_save.hash,
                    "key": self._api_key,
                    "format": "base64",
                    "country": tag,
                }
                request = requests.Request(
                    method="GET",
                    url=SKANDERBEG_LINK,
                    params=params,
                ).prepare()
                response = session.send(request, timeout=10)
                if not response.ok or response.text == "Err":
                    raise RequestException(request=request, response=response)

                response = response.text
                self._tags[tag]["flag"] = response

                flag = base64.b64decode(response)
                flag = BytesIO(flag)

                try:
                    picture = plt.imread(flag, format="png")
                except SyntaxError:
                    pass
                else:
                    plt.imsave(OUTPUT_PATH / f"{tag}.png", picture)

            self._downloaded_flags = True

        with Path(CACHE_PATH / "tags.pkl").open("wb") as f:
            pickle.dump(self._tags, f)

        return self._tags

    def get_country_data(self) -> dict[int, Save]:
        """Download and process country data for all saves.

        Returns
        -------
        dict[int, Save]
            Dictionary of saves with country data processed.

        """
        if self._processed_country_data:
            return self.saves

        self._get_data(data_type="countriesData")
        self._process_country_data()

        self.get_country_flags()

        return self.saves

    def get_provinces_data(self, *, add_base: bool = False) -> dict[int, Save]:
        """Download and process provinces data for all saves.

        Parameters
        ----------
        add_base : bool, default=False
            Whether to merge with base province data from geojson.

        Returns
        -------
        dict[int, Save]
            Dictionary of saves with provinces data processed.

        """
        self._get_data(data_type="provincesData")
        self._process_provinces_data(add_base=add_base)

        return self.saves

    def _get_data(self, *, data_type: str) -> dict[int, Save]:
        if data_type not in {"countriesData", "provincesData"}:
            raise ValueError

        if not self.saves:
            self.get_save_metadata()

        self.saves.update(self.read_cached_data())

        # return cached data (if there's any) when force_offline
        # when no cache and force_offline -> error
        if self.force_offline and self.saves:
            return self.saves
        if self.force_offline and not self.saves:
            raise ValueError

        # if the data was already downloaded return
        if data_type == "countriesData" and self._downloaded_country_data:
            return self.saves
        if data_type == "provincesData" and self._downloaded_province_data:
            return self.saves

        if data_type == "countriesData":
            saves_to_download = {
                date: save for date, save in self.saves.items() if not save.country_data
            }
        elif data_type == "provincesData":
            saves_to_download = {
                date: save for date, save in self.saves.items() if not save.provinces_data
            }
        else:
            raise ValueError

        with requests.Session() as session:
            for date, save in saves_to_download.items():
                params = {
                    "scope": "getSaveDataDump",
                    "save": save.hash,
                    "key": self._api_key,
                    "type": data_type,
                }
                request = requests.Request(
                    method="GET",
                    url=SKANDERBEG_LINK,
                    params=params,
                ).prepare()
                response = session.send(request, timeout=10)
                if not response.ok or response.text == "Err":
                    raise RequestException(request=request, response=response)

                response = response.json()

                if data_type == "countriesData":
                    self.saves[date].country_data = response
                    self._downloaded_country_data = True
                elif data_type == "provincesData":
                    self.saves[date].provinces_data = response
                    self._downloaded_province_data = True

        with Path(CACHE_PATH / "saves.pkl").open("wb") as f:
            pickle.dump(self.saves, f)

        return self.saves

    def _process_country_data(self) -> None:
        if (current_save_data := self.current_save.country_data) is None:
            raise ValueError

        if self._processed_country_data:
            return

        # overwrite tags (with current ones) that players used to form other tags
        # this will ensure continuity of tags in the game
        # and easier access to data

        # first check the current save what countries players formed
        # and make basic dict with reformations that look like this
        # tag as key and list of tuples {'AUH': [('CRO', 1531), ('HUN', 1689)]}
        reformations = {}
        for tag in self.tags:
            tag_data = current_save_data.get(tag)
            if tag_data is None:
                raise ValueError

            reforms = tag_data.get("reformations")

            if reforms:
                for reform in reforms:
                    prev_tag = reform["tag"]
                    formation_year = int(reform["date"][:4])

                    reformations.setdefault(tag, [])
                    reformations[tag].append((prev_tag, formation_year))

            # save current colour of the tag
            self.tags[tag]["colour"] = current_save_data[tag]["hex"]

        # purely for 'income_stats' create new reformation dict
        # that will simplify combining correct income stats
        # dict {'BAV': {'UBV': (1444, 1448), 'BAV': (1448, 1705)}
        # current tags as keys and previous tags with years when existed as values
        reformations_for_income_stats = deepcopy(reformations)
        for tag, refs in reformations_for_income_stats.items():
            first_tag = refs[0][0]

            refs.insert(0, (first_tag, self._EU4_START_DATE))
            refs.append((tag, self.current_year))

            result = {}
            for (_, earlier_year), (later_tag, later_year) in itertools.pairwise(refs):
                result[later_tag] = (earlier_year, later_year)

            reformations_for_income_stats[tag] = result

        print(reformations_for_income_stats)

        # algorithm to get income_stats
        for current_tag, formation in reformations_for_income_stats.items():
            inc_stats = {}
            for save_date in self.saves:
                if (country_data := self.saves[save_date].country_data) is None:
                    raise ValueError

                for prev_tag, (start, end) in formation.items():
                    if save_date == self._EU4_START_DATE:
                        continue

                    if prev_tag not in country_data:
                        continue

                    if save_date < start:
                        continue

                    if end > save_date:
                        continue

                    data = {
                        int(k): v
                        for k, v in country_data[prev_tag]["income_stats"].items()
                        if start < int(k) <= end
                    }
                    inc_stats.update(data)

            current_save_data[current_tag]["income_stats"] = inc_stats

        # algorithm to relabel tags
        # for each tag that was formed, check all saves
        for current_tag, formation in reformations.items():
            for save_date in self.saves:
                country_data = self.saves[save_date].country_data
                if country_data is None:
                    raise ValueError
                for prev_tag, formation_year in formation:
                    if save_date == self._EU4_START_DATE:
                        prev_tag_data = country_data[prev_tag]
                        country_data[current_tag] = prev_tag_data
                        break

                    if formation_year >= save_date:
                        prev_tag_data = country_data[prev_tag]
                        country_data[current_tag] = prev_tag_data
                        break

                    # delete previous tag data, so it doesn't pollute the data
                    del country_data[prev_tag]

        self._processed_country_data = True

    def _process_provinces_data(self, *, add_base: bool = True) -> None:
        base_provinces_data = None
        if add_base:
            base_provinces_path = DATA_PATH / "eu4provinces.geojson"

            if not base_provinces_path.exists():
                raise FileNotFoundError
            base_provinces_data = gpd.read_file(base_provinces_path).set_index("province_id")

        for save in self.saves.values():
            if save.provinces_data is None:
                raise ValueError

            provinces = {
                int(province_id): data
                for province_id, data in save.provinces_data.items()
                if isinstance(province_id, (str, int)) and str(province_id).isnumeric()
            }

            provinces_df = pd.DataFrame(provinces).T

            dtype_map = {
                "monument": pd.StringDtype(),
                "name": pd.StringDtype(),
                "owner": pd.StringDtype(),
                "controller": pd.StringDtype(),
                "culture": pd.StringDtype(),
                "religion": pd.StringDtype(),
                "base_tax": pd.Int64Dtype(),
                "base_production": pd.Int64Dtype(),
                "base_manpower": pd.Int64Dtype(),
                "trade_goods": pd.StringDtype(),
                "buildings_value": pd.Int64Dtype(),
                # 'ownership_changes':,
                # 'dev_improvement_values':,
                # 'dev_improvements_values':,
                # 'army':,
                "casualties": pd.Int64Dtype(),
                "improveCount": pd.Int64Dtype(),
                "prosperity": pd.Float64Dtype(),
                "devastation": pd.Float64Dtype(),
                "fort_level": pd.Int64Dtype(),
                "original_coloniser": pd.StringDtype(),
                "active_trade_company": pd.BooleanDtype(),
            }

            existing_dtype_map = {k: v for k, v in dtype_map.items() if k in provinces_df.columns}
            provinces_df = provinces_df.astype(existing_dtype_map, errors="ignore").fillna(pd.NA)

            if add_base and base_provinces_data is not None:
                provinces_df = provinces_df.merge(
                    base_provinces_data,
                    right_index=True,
                    left_index=True,
                    validate="one_to_one",
                    suffixes=("", "__DROP"),
                )
                cols_to_drop = [col for col in provinces_df.columns if col.endswith("__DROP")]
                provinces_df = provinces_df.drop(columns=cols_to_drop)

            save.provinces_data = provinces_df

        self._processed_province_data = True

    # selecting data
    def get_statistic(
        self,
        statistic: str,
        *,
        tags: dict[str, dict] | None = None,
        include_world: bool = True,
    ) -> SkandStat:
        """Return a SkandStat object for the given statistic.

        Parameters
        ----------
        statistic : str
            Name of the statistic to retrieve.
        tags : dict[str, dict], optional
            Tags to include. If None, uses all tags.
        include_world : bool, default=True
            Whether to include the 'WORLD' column in the result.

        Returns
        -------
        SkandStat
            SkandStat object containing the requested statistic.

        Raises
        ------
        ValueError
            If required country data is missing for any save.

        """
        if tags is None:
            tags = self.tags

        dates = []
        chosen_tags = {tag: [] for tag in tags}

        # special case for income_stats
        if statistic == "income_stats":
            return self._get_statistic__income_stats(tags=tags)

        world = []
        for date, save_data in self.saves.items():
            dates.append(date)
            country_data = save_data.country_data
            if country_data is None:
                raise ValueError

            world_stat = 0
            for tag, stats in country_data.items():
                stat = stats.get(statistic, 0)
                world_stat += float(stat)

                if tag in chosen_tags:
                    chosen_tags[tag].append(stat)

            world.append(world_stat)

        if include_world:
            chosen_tags["WORLD"] = world

        data = (
            pd.DataFrame(data=chosen_tags, index=dates)
            .astype("float64", errors="ignore")
            .sort_index(axis="index", ascending=True)
        )

        return SkandStat(
            statistic=statistic,
            data=data,
            tags=self.tags,
            save_dates=self.save_dates,
            include_world=include_world,
        )

    def _get_statistic__income_stats(self, tags: dict[str, dict]) -> SkandStat:
        current_country_data = self.current_save.country_data
        if current_country_data is None:
            raise ValueError

        series_to_concat = []
        for tag in tags:
            data = current_country_data[tag]["income_stats"]
            series = pd.Series(data=data, name=tag)
            series.index = series.index.astype("int64")
            series_to_concat.append(series)

        data = pd.concat(series_to_concat, axis="columns", sort=True).astype(
            "float64",
            errors="ignore",
        )
        return SkandStat(
            statistic="income_stats",
            data=data,
            tags=self.tags,
            save_dates=self.save_dates,
            include_world=False,
        )

    # TODO: export data
    def export_data(self) -> None:
        """Export data to file (not yet implemented)."""

    # MAIN LOOP
    # TODO: work on the main loop
    def run(self) -> NoReturn:
        """Run the main interactive loop for the analyzer."""

        def clear_console() -> None:
            """Clear the console based on the operating system."""
            # Clear console based on the operating system
            if os.name == "nt":
                command = "cls"
                os.system(command)  # For Windows  # noqa: S605
            else:
                command = "clear"
                os.system(command)  # For Unix/Linux/Mac  # noqa: S605

        def prompt_menu() -> str:
            """Prompt the user for the main menu option.

            Returns
            -------
            str
                The user's menu choice.

            """
            print(
                "\nSkanderbeg Analyzer Main Menu:",
                "1. Download, process, and analyze country data",
                "2. Download, process, and save provinces data",
                "3. Exit",
                sep="\n",
            )
            inp = input("Choose an option (1-3): ").strip()
            clear_console()
            return inp

        def prompt_statistic() -> str | None:
            """Prompt the user to select a statistic.

            Returns
            -------
            str or None
                The selected statistic, or None if the user quits.

            """
            print("\nAvailable statistics:")
            for i, stat in enumerate(ALL_STATS, 1):
                print(f"{i}. {stat}")
            while True:
                choice = input(f"Select statistic (1 - {len(ALL_STATS)}): ").strip()
                if choice == "q":
                    return None

                if choice.isdigit() and 1 <= int(choice) <= len(ALL_STATS):
                    clear_console()
                    return ALL_STATS[int(choice) - 1]
                print("Invalid choice. Try again.")

        def country_data_segment() -> None:
            """Handle the country data analysis segment."""
            print("\n--- Country Data Analysis ---")
            try:
                self.get_country_data()
            except Exception as e:
                print(f"Error downloading/processing country data: {e}")
                return

            while True:
                chosen_stat = prompt_statistic()
                if chosen_stat is None:
                    clear_console()
                    break
                try:
                    stat = self.get_statistic(statistic=chosen_stat)
                except Exception as e:
                    clear_console()
                    print(f"Error getting statistic: {e}")
                    continue

                # table part
                table = None
                try:
                    table = stat.table()
                    if table is not None:
                        table_path = stat.export_viz(table)
                        print(f"Table saved to: {table_path}")
                        img = Image.open(table_path)
                        img.show()

                except Exception as e:
                    clear_console()
                    print(f"Table error: {e}")

                # chart part
                fig = None
                try:
                    fig = stat.line_chart()
                    plt.show()
                    if fig is not None:
                        fig_path = stat.export_viz(fig)
                        print(f"Table saved to: {fig_path}")
                except Exception as e:
                    clear_console()
                    print(f"Chart error: {e}")

        def province_data_segment() -> None:
            """Handle the provinces data download/process segment."""
            print("\n--- Provinces Data Download/Process ---")
            add_base = input("Merge with base province data? (y/n): ").strip().lower() == "y"
            try:
                self.get_provinces_data(add_base=add_base)
                print("Provinces data downloaded and processed.")
            except Exception as e:
                clear_console()
                print(f"Error: {e}")

        while True:
            choice = prompt_menu()
            if choice == "1":
                country_data_segment()
            elif choice == "2":
                province_data_segment()
            elif choice in {"3", "q"}:
                print("Exiting.")
                sys.exit()
            else:
                clear_console()
                print("Invalid option. Try again.")
                sys.exit()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Skanderbeg Analyzer.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(description="SkandParser")
    parser.add_argument("--api-key", type=str, help="api key to skanderbeg.pm")
    parser.add_argument("--force-offline", action="store_true", help="run in offline mode")
    return parser.parse_args()


def main() -> None:
    """Read command-line arguments, initialize Analyzer and run it."""
    args = parse_args()

    api_key = args.api_key or os.environ.get(SKANDERBEG_API_KEY_ENV_VAR_NAME)
    force_offline = args.force_offline

    analyzer = Analyzer(api_key=api_key, force_offline=force_offline)
    analyzer.run()


if __name__ == "__main__":
    main()
