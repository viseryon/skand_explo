import argparse
import base64
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO
from itertools import pairwise
from pathlib import Path

import dataframe_image as dfi
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pandas.io.formats.style import Styler
from requests.exceptions import RequestException

SKANDERBEG_LINK = "https://skanderbeg.pm/api.php"
SKANDERBEG_API_KEY_ENV_VAR_NAME = "SKANDERBEG_API_KEY"

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
    timestamp: datetime
    name: str
    uploaded_by: str
    version: str
    player: str
    multiplayer: bool
    date: date
    custom_name: str
    game: str
    mods: str
    linked_sheet: str
    options: str
    last_visited: datetime
    view_count: int
    augmented_campaign: str
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
    tags: dict[str, dict] | None = None
    save_dates: list[int] | None = None
    include_world: bool = True

    DEFAULT_FIGSIZE: tuple[int, int] = field(default=(13, 7), init=False, repr=False)
    DEFAULT_DPI: int = field(default=200, init=False, repr=False)
    DEFAULT_STYLE: str = field(default="dark_background", init=False, repr=False)
    DEFAULT_EXPORT_EXTENSION: str = field(default=".png", init=False, repr=False)

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
        colours = {} if not self.tags else self.tags
        dates = self.data.index if not self.save_dates else self.save_dates
        style = style or self.DEFAULT_STYLE

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            for tag in self.data.columns:
                if tag == "WORLD":
                    continue

                colour = colours.get(tag, {}).get("colour")
                ax.plot(self.data[tag], color=colour)

                annotation = f"{tag} | {self.data[tag].iloc[-1]:_.2f}"
                ax.annotate(
                    f"{annotation}",
                    (self.data.index.max() + 2, self.data[tag].iloc[-1]),
                    fontsize=10,
                    color=colour,
                    bbox={"facecolor": "black", "edgecolor": "white", "boxstyle": "round"},
                )

            ax.set_yscale(y_scale)
            formatter = FuncFormatter(lambda y, pos: f"{y:_.2f}")
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(formatter)
            ax.legend(
                self.data.keys(),
                loc="upper left",
                frameon=False,
                shadow=True,
            )
            ax.set_title(
                f"{self.statistic.upper().replace('_', ' ')} BY {self.current_year}",
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
            self.statistic  # + f"<br>as % of world's {self.statistic}" if self.include_world else ""
        )
        styler = styler.set_caption(caption.upper())

        # add highlitights
        subset = pd.IndexSlice[
            [indx for indx in data.index if indx != "WORLD"],
            new_cols,
        ]
        styler = styler.highlight_max(color="darkgreen", subset=subset, axis="index")  # type: ignore
        styler = styler.highlight_min(color="darkred", subset=subset, axis="index")  # type: ignore

        return styler

    def export_viz(
        self,
        viz: Figure | Styler,
        dpi: int | None = None,
        extension: str | None = None,
    ) -> Path:
        extension = extension if extension is not None else self.DEFAULT_EXPORT_EXTENSION
        dpi = dpi if dpi is not None else self.DEFAULT_DPI

        file_name = f"{self.statistic}_by_{self.current_year}".upper()
        export_path = CHARTS_PATH / Path(file_name).with_suffix(extension)

        if isinstance(viz, Figure):
            viz.savefig(export_path, dpi=dpi)
            return export_path
        if isinstance(viz, Styler):
            dfi.export(
                viz,  # type: ignore
                export_path,
                dpi=dpi,
            )
            return export_path
        msg = "`viz` is not an instance of `Styler` or `Figure`"
        raise TypeError(msg)


class Analyzer:
    _EU4_START_DATE: int = 1444

    def __init__(self, api_key: str | None = None, *, force_offline: bool = False):
        self._api_key: str | None = api_key
        self._save_dates: list[int] | None = None
        self._tags: dict[str, dict] | None = None
        self._downloaded_country_data: bool = False
        self._downloaded_province_data: bool = False
        self.saves: dict[int, Save] = {}
        self.force_offline: bool = force_offline
        self._downloaded_flags: bool = False

    def read_cached_data(self) -> dict[int, Save]:
        cached_saves = Path(CACHE_PATH / "saves.pkl")
        if cached_saves.exists():
            with cached_saves.open("rb") as f:
                return pickle.load(f)
        return {}

    def read_cached_tags(self) -> dict[str, dict]:
        cached_tags = Path(CACHE_PATH / "tags.pkl")
        if cached_tags.exists():
            with cached_tags.open("rb") as f:
                return pickle.load(f)
        return {}

    @property
    def tags(self) -> dict[str, dict]:
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

    def get_save_metadata(self) -> dict[int, Save]:
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
            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007

            date = save_data["date"]
            # date format '1615.2.15'
            date = datetime.strptime(date, "%Y.%m.%d").date()  # noqa: DTZ007
            year = date.year

            multiplayer = save_data["multiplayer"] == "yes"

            last_visited = save_data["lastVisited"]
            # date format '2025-06-17 01:00:09'
            last_visited = datetime.strptime(last_visited, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007

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

                picture = plt.imread(flag, format="png")
                plt.imsave(OUTPUT_PATH / f"{tag}.png", picture)

            self._downloaded_flags = True

        with Path(CACHE_PATH / "tags.pkl").open("wb") as f:
            pickle.dump(self._tags, f)

        return self._tags

    def get_country_data(self) -> dict[int, Save]:
        if self._downloaded_country_data:
            return self.saves

        self._get_data(data_type="countriesData")
        self._process_country_data()

        self.get_country_flags()

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

        # return cached data (if there's any) when force_offline
        # when no cache and force_offline -> error
        if self.force_offline and self.saves:
            return self.saves
        if self.force_offline and not self.saves:
            raise ValueError

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

        # overwrite tags (with current ones) that players used to form other tags
        # this will ensure continuity of tags in the game
        # and easier access to data

        # first check the current save what countries players formed
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

        # algorithm to get income_stats
        ref = deepcopy(reformations)
        for current_tag, formation in ref.items():
            inc_stats = {}
            left = self._EU4_START_DATE
            formation.append((current_tag, self.current_year))
            for save_date in self.saves:
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

    # selecting data
    def get_statistic(
        self,
        statistic: str,
        *,
        tags: dict[str, dict] | None = None,
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

    # MAIN LOOP
    # TODO: main loop
    def run(self): ...  # TODO: write `run()` function that will handle user interaction


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SkandParser")
    parser.add_argument("--api-key", type=str, help="api key to skanderbeg.pm")
    parser.add_argument("--force-offline", action="store_true", help="run in offline mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get(SKANDERBEG_API_KEY_ENV_VAR_NAME) or args.api_key
    force_offline = args.force_offline

    analyzer = Analyzer(api_key=api_key, force_offline=force_offline)
    analyzer.run()


if __name__ == "__main__":
    main()
