import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pandas.io.formats.style import Styler

SKANDERBEG_LINK = "https://skanderbeg.pm/api.php"

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
CHARTS_PATH = ROOT_PATH / "charts"
CONFIG_PATH = ROOT_PATH / "config"
OUTPUT_PATH = ROOT_PATH / "output"
CACHE_PATH = ROOT_PATH / "cache"

CHARTS_PATH.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True)
CACHE_PATH.mkdir(exist_ok=True)

ALL_STATS = Path(CONFIG_PATH / "available_stats.txt").read_text(encoding="utf-8").splitlines()

# TODO: docs
# TODO: type hints


@dataclass
class Save:
    id: str
    hash: str
    timestamp: str
    name: str
    uploadedBy: str
    version: str
    player: str
    multiplayer: str
    date: str
    customname: str
    game: str
    mods: str
    linkedSheet: str
    options: str
    lastVisited: str
    viewCount: str
    augmentedCampaign: str
    year: int
    country_data: dict | None = None
    provinces_data: dict | None = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"year={self.year}, player={self.player}, hash={self.hash}, "
            f"country_data={self.country_data is not None}, "
            f"provinces_data={self.provinces_data is not None})"
        )


@dataclass(frozen=True)
class SkandStat:
    statistic: str
    data: pd.DataFrame
    tag_colours: dict[str, str] | None = None
    save_dates: list[int] | None = None
    include_world: bool = True

    DEFAULT_FIGSIZE: tuple[int, int] = field(default=(13, 7), init=False, repr=False)
    DEFAULT_DPI: int = field(default=200, init=False, repr=False)
    DEFAULT_STYLE: str = field(default="dark_background", init=False, repr=False)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({self.statistic=}, "
            f"{self.save_dates=}, "
            f"{self.include_world=})"
        )

    @property
    def current_year(self) -> int:
        if self.save_dates is None:
            return self.data.index.max()
        return max(self.save_dates)

    def line_chart(
        self,
        y_scale: str = "log",
        figsize: tuple[int, int] | None = None,
        style: str | None = None,
    ) -> Figure:
        figsize = figsize or self.DEFAULT_FIGSIZE
        colours = {} if not self.tag_colours else self.tag_colours
        dates = self.data.index if not self.save_dates else self.save_dates
        style = style or self.DEFAULT_STYLE

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            for tag in self.data.columns:
                if tag == "WORLD":
                    continue

                ax.plot(self.data[tag], color=colours.get(tag))

                annotation = f"{tag} | {self.data[tag].iloc[-1]:_.2f}"
                ax.annotate(
                    f"{annotation}",
                    (self.data.index.max() + 2, self.data[tag].iloc[-1]),
                    fontsize=10,
                    color=colours.get(tag),
                    bbox={"facecolor": "black", "edgecolor": "white", "boxstyle": "round"},
                )

            ax.set_yscale(y_scale)
            formatter = FuncFormatter(lambda y, pos: f"{y:_.2f}")
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(formatter)
            ax.legend(self.data.keys(), loc="upper left")
            ax.set_title(f"{self.statistic} by {self.current_year}")
            ax.set_xticks(dates)
            ax.grid(visible=True, axis="y", alpha=0.25)
            fig.tight_layout()

            return fig

    def data_with_cagr_values(self) -> pd.DataFrame:
        data = self.data.T.sort_index(axis="columns")

        min_year = data.columns.min()
        max_year = data.columns.max()

        for i, (start, end) in enumerate(pairwise(data), start=0):
            year_diff = int(str(end)) - int(str(start))  # Convert to int to avoid type error
            period_cagr = (data[end] / data[start]) ** (1 / year_diff) - 1
            data.insert(2 * i + 1, year_diff, period_cagr, allow_duplicates=True)

        total_year_diff = max_year - min_year
        total_cagr = (data[max_year] / data[min_year]) ** (1 / total_year_diff) - 1
        data.insert(len(data.columns), total_year_diff, total_cagr)
        return data

    def pretty_table(self, *, with_cagr: bool = True) -> Styler:
        data = self.data_with_cagr_values() if with_cagr else self.data.T
        data = data.copy(deep=True)
        old_cols = data.columns.copy(deep=True)

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

        styler = data.style

        # ignoring this warning, the same is in the docs
        # https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.io.formats.style.Styler.format.html
        barrier = 1000
        nr_formats = {
            new: "{:,.1f}" if old > barrier else "{:.1%}"  # type: ignore
            for old, new in zip(old_cols, new_cols, strict=False)
        }
        styler.format(nr_formats, na_rep="-", precision=1, decimal=",", thousands=" ")  # type: ignore

        # ignoring this warning, the same is in the docs
        # https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.io.formats.style.Styler.set_properties.html
        styler.set_properties(**{
            "background-color": "black",
            "color": "white",
        })  # type: ignore

        # customize index
        colours = {} if not self.tag_colours else self.tag_colours
        colours_for_index = {k: f"color: {v}" for k, v in colours.items()}
        styler = styler.map_index(
            lambda x: colours_for_index.get(x, "color: white"),  # type: ignore
            axis=0,
        )
        # customize header
        styler = styler.set_table_styles(
            [
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
        caption = f" as % of world's {self.statistic}" if self.include_world else ""
        styler = styler.set_caption(self.statistic + caption)
        styler = styler.set_caption(self.statistic.upper())

        # add highlitights
        subset = pd.IndexSlice[[indx for indx in data.index if indx != "WORLD"], :]
        styler = styler.highlight_max(color="darkgreen", subset=subset, axis="index")  # type: ignore
        styler = styler.highlight_min(color="darkred", subset=subset, axis="index")  # type: ignore

        return styler


class Analyzer:
    _EU4_START_DATE: int = 1444

    def __init__(self, *, api_key: str | None = None, force_offline: bool = False):
        self._api_key: str = (
            api_key or (CONFIG_PATH / "apis.json").open(encoding="utf-8").read()
        )  # TODO: replace with txt
        self._save_dates: list[int] | None = None
        self._tags: list[str] | None = None
        self._downloaded_country_data: bool = False
        self._downloaded_province_data: bool = False
        self.saves: dict[int, Save] = {}
        self.force_offline: bool = force_offline
        self._tag_coulours: dict[str, str] | None = None

    def read_cached_data(self) -> dict[str, Save]:
        cached_saves = Path(CACHE_PATH / "saves.pkl")
        if cached_saves.exists():
            with cached_saves.open("rb") as f:
                return pickle.load(f)
        return {}

    @property
    def tags(self) -> list[str]:
        if self._tags:
            return self._tags
        tags = (CONFIG_PATH / "countries.txt").read_text(encoding="utf-8").split(",")
        self._tags = tags
        return tags

    @property
    def save_dates(self) -> list[int]:
        if self._save_dates:
            return self._save_dates
        if not self.saves:
            self.get_save_metadata()
        self._save_dates = sorted(self.saves.keys())
        return self._save_dates

    @property
    def current_year(self) -> int:
        if not self.save_dates:
            self.get_save_metadata()
        return max(self.save_dates)

    @property
    def current_save(self) -> Save:
        if not self.save_dates:
            self.get_save_metadata()
        return self.saves[self.current_year]

    @property
    def tag_colours(self) -> dict[str, str] | None:
        if self._tag_coulours:
            return self._tag_coulours
        return None

    def get_save_metadata(self) -> dict[int, Save]:
        if self.saves:
            return self.saves

        if self.force_offline:
            self.saves = self.read_cached_data()
            return self.saves

        params = {"key": self._api_key, "scope": "fetchUserSaves"}
        response = requests.get(SKANDERBEG_LINK, params=params, timeout=10)

        if response.status_code != requests.codes.all_good or response.text == "Err":
            raise requests.exceptions.HTTPError

        saves = {}
        for save_data in response.json():
            year = int(save_data["date"][:4])
            saves[year] = Save(
                **save_data,
                year=year,
            )

        self.saves = saves
        return saves

    def get_country_data(self) -> dict[int, Save]:
        if self._downloaded_country_data:
            return self.saves

        self._get_data(data_type="countriesData")
        self._process_country_data()
        return self.saves

    def get_provinces_data(self) -> dict[int, Save]:
        if self._downloaded_province_data:
            return self.saves

        return self._get_data(data_type="provincesData")

    def _get_data(self, *, data_type: str) -> dict[int, Save]:
        if data_type not in {"countriesData", "provincesData"}:
            raise ValueError

        if not self.saves:
            self.get_save_metadata()

        self.saves.update(self.read_cached_data())

        if self.force_offline:
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

        for date, save in saves_to_download.items():
            params = {
                "scope": "getSaveDataDump",
                "save": save.hash,
                "key": self._api_key,
                "type": data_type,
            }
            response = requests.get(SKANDERBEG_LINK, params=params, timeout=10)
            if response.status_code != requests.codes.all_good or response.text == "Err":
                raise requests.exceptions.HTTPError

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

        # overwrite tags (with current ones) that players used to form other tags
        # this will ensure continuity of tags in the game
        # and easier access to data

        # first check the current save what countries players formed
        reformations = {}
        tag_colours = {}

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
            tag_colours[tag] = current_save_data[tag]["hex"]

        self._tag_coulours = tag_colours

        # algorithm to get income_stats
        ref = deepcopy(reformations)
        for current_tag, formation in ref.items():
            inc_stats = {}
            left = self._EU4_START_DATE
            formation.append((current_tag, self.current_year))
            for save_date in sorted(self.saves):
                if (country_data := self.saves[save_date].country_data) is None:
                    raise ValueError

                for prev_tag, formation_year in formation:
                    if save_date == self._EU4_START_DATE:
                        continue

                    if save_date == self.current_year:
                        data = {
                            k: v
                            for k, v in country_data[current_tag]["income_stats"].items()
                            if int(k) > left
                        }
                        inc_stats.update(data)

                    if formation_year > save_date:
                        break
                    if formation_year == left:
                        break

                    data = {
                        k: v
                        for k, v in country_data[prev_tag]["income_stats"].items()
                        if formation_year >= int(k) > left
                    }
                    inc_stats.update(data)
                    left = formation_year

            inc_stats = {int(k): v for k, v in inc_stats.items()}

            current_save_data[current_tag]["income_stats"] = inc_stats

        # algorithm to relabel tags
        # for each tag that was formed, check all saves
        for current_tag, formation in reformations.items():
            for save_date in sorted(self.saves):
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

    # selecting data
    def get_statistic(
        self,
        statistic: str,
        *,
        tags: list[str] | None = None,
        include_world: bool = True,
    ) -> SkandStat:
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
            tag_colours=self.tag_colours,
            save_dates=self.save_dates,
            include_world=include_world,
        )

    def _get_statistic__income_stats(self, tags: list[str]) -> SkandStat:
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
            tag_colours=self.tag_colours,
            save_dates=self.save_dates,
            include_world=False,
        )


# TODO: provinces data


def prepare_provinces_data(data: dict[int, dict]) -> pd.DataFrame:
    """Prepares province data from skanderbeg

    Args:
        data (dict[int, dict]): dict with raw province data from skanderbeg

    Returns:
        pd.DataFrame: dataframe with statistics for each province for each year

    """
    dates = list(data.keys())
    master = pd.DataFrame()
    for date in dates:
        provinces = {}
        provs = [prov_num for prov_num in data[date] if prov_num.isnumeric()]
        (
            owner,
            culture,
            religion,
            tax,
            prod,
            manp,
            trade_good,
            buildings_value,
            improve_count,
            casualties,
            prosperity,
        ) = ([], [], [], [], [], [], [], [], [], [], [])
        for prov in provs:
            prov = data[date][prov]
            owner.append(prov.get("owner", np.nan))
            culture.append(prov.get("culture", np.nan))
            religion.append(prov.get("religion", np.nan))
            tax.append(prov.get("base_tax", np.nan))
            prod.append(prov.get("base_production", np.nan))
            manp.append(prov.get("base_manpower", np.nan))
            trade_good.append(prov.get("trade_goods", np.nan))
            buildings_value.append(prov.get("buildings_value", np.nan))
            improve_count.append(prov.get("improveCount", np.nan))
            casualties.append(prov.get("casualties", np.nan))
            prosperity.append(prov.get("prosperity", np.nan))

        provinces["id"] = provs
        provinces["year"] = date
        provinces["tag"] = owner
        provinces["culture"] = culture
        provinces["religion"] = religion
        provinces["tax"] = tax
        provinces["prod"] = prod
        provinces["manp"] = manp
        provinces["trade_good"] = trade_good
        provinces["buildings_value"] = buildings_value
        provinces["improve_count"] = improve_count
        provinces["casualties"] = casualties
        provinces["prosperity"] = prosperity
        df = pd.DataFrame(provinces)

        if master.empty:
            master = df
        else:
            master = pd.concat([master, df])

    master = master.astype(
        {
            "id": int,
            "tax": float,
            "prod": float,
            "manp": float,
            "improve_count": float,
            "casualties": float,
            "prosperity": float,
        },
    )

    master["dev"] = master["tax"] + master["prod"] + master["manp"]

    return master


def _export_provinces_data(data: pd.DataFrame) -> None:
    data.to_csv(OUTPUT_PATH / "provinces_data.csv", index=False)
    print("exported")


# TODO: main loop


def _inp() -> str:
    value = input("""\nchoose metric or say "help": """)
    print()

    if value.lower() == "help":
        print(ALL_STATS)
        return _inp()
    if value.lower() == "q":
        exit()
    elif value not in ALL_STATS:
        print("not in available statistics")
        return _inp()
    else:
        return value


def country_data_segment():
    """Loop with country data analysis segment"""


def provinces_data_segment():
    """Loop with province data analysis segment"""


# TODO: rewrite the main function
def main(): ...


if __name__ == "__main__":
    main()
