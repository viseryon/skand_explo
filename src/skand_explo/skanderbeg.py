import json
import pickle
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt

plt.style.use("dark_background")

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


class Analyzer:
    def __init__(self, *, api_key: str | None = None, force_offline: bool = False):
        self._api_key = (
            api_key or (CONFIG_PATH / "apis.json").open(encoding="utf-8").read()
        )  # TODO: replace with txt
        self._save_dates = None
        self._tags = None
        self._downloaded_country_data = False
        self._downloaded_province_data = False
        self.saves: dict = {}
        self.force_offline = force_offline
        self._tag_coulours = None

    def read_cached_data(self):
        cached_saves = Path(CACHE_PATH / "saves.pkl")
        if cached_saves.exists():
            with cached_saves.open("rb") as f:
                self.saves = pickle.load(f)
            return self.saves
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
    def tag_colours(self):
        if self._tag_coulours:
            return self._tag_coulours
        return None

    def get_save_metadata(self):
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

    def get_country_data(self, *, force_offline: bool = False):
        if self._downloaded_country_data:
            return self.saves

        self.get_data(data_type="countriesData", force_offline=force_offline)
        self._process_country_data()
        return self.saves

    def get_provinces_data(self, *, force_offline: bool = False):
        if self._downloaded_province_data:
            return self.saves

        return self.get_data(data_type="provincesData", force_offline=force_offline)

    def get_data(self, *, data_type: str, force_offline: bool = False):
        if data_type not in {"countriesData", "provincesData"}:
            raise ValueError

        if not self.saves and not force_offline:
            self.get_save_metadata()

        self.saves.update(self.read_cached_data())

        if force_offline:
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

    def _process_country_data(self):
        # overwrite tags (with current ones) that players used to form other tags
        # this will ensure continuity of tags in the game
        # and easier access to data

        # first check the current save what countries players formed
        reformations = {}
        tag_colours = {}

        for tag in self.tags:
            reforms: list[dict] | None = self.current_save.country_data[tag].get(
                "reformations",
            )  # TODO: resolve this warning

            if reforms:
                for reform in reforms:
                    prev_tag = reform["tag"]
                    formation_year = int(reform["date"][:4])

                    reformations.setdefault(tag, [])
                    reformations[tag].append((prev_tag, formation_year))

            # save current colour of the tag
            tag_colours[tag] = self.current_save.country_data[tag][
                "hex"
            ]  # TODO: resolve this warning

        self._tag_coulours = tag_colours

        # algorithm to relabel tags
        # for each tag that was formed, check all saves
        for current_tag, formation in reformations.items():
            for save_date in sorted(self.saves):
                for prev_tag, formation_year in formation:
                    if save_date == 1444:
                        prev_tag_data = self.saves[save_date].country_data[prev_tag]
                        self.saves[save_date].country_data[current_tag] = prev_tag_data

                    # income_stats fix
                    # this has to be done separately
                    # income_stats is a different structure than other stats
                    if formation_year <= save_date:
                        income_stats = (
                            self.saves[save_date].country_data[current_tag].get("income_stats", {})
                        )
                        prev_tag_data = self.saves[save_date].country_data[prev_tag]

                        income_stats.update(prev_tag_data["income_stats"])
                        prev_tag_data["income_stats"] = income_stats

                    if formation_year >= save_date:
                        prev_tag_data = self.saves[save_date].country_data[prev_tag]
                        self.saves[save_date].country_data[current_tag] = prev_tag_data

                    # delete previous tag data, so it doesn't pollute the data
                    del self.saves[save_date].country_data[prev_tag]

    # selecting data
    def get_statistic(
        self,
        statistic: str,
        *,
        tags: list[str] | None = None,
        include_world: bool = True,
    ):
        if tags is None:
            tags = self.tags
        # TODO: include world, sum of all tags for specific date
        dates = []
        chosen_tags = {tag: [] for tag in tags}

        # special case for income_stats
        if statistic == "income_stats":
            return self._get_statistic__income_stats(tags=tags)

        for date, save_data in self.saves.items():
            dates.append(date)
            country_data = save_data.country_data

            for tag, tag_stats in chosen_tags.items():
                stat = country_data[tag].get(statistic, pd.NA)
                tag_stats.append(stat)

        return pd.DataFrame(data=chosen_tags, index=dates).astype("float64", errors="ignore")

    def _get_statistic__income_stats(self, tags: list[str]) -> pd.DataFrame:
        series_to_concat = []
        for tag in tags:
            data = self.current_save.country_data[tag]["income_stats"]  # TODO: resolve this warning
            series = pd.Series(data=data, name=tag)
            series.index = series.index.astype("int64")
            series_to_concat.append(series)

        return pd.concat(series_to_concat, axis="columns", sort=True).astype(
            "float64",
            errors="ignore",
        )


def create_df_with_cagr_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_index(axis="columns")

    min_year = data.columns.min()
    max_year = data.columns.max()

    for i, (start, end) in enumerate(pairwise(data), start=0):
        year_diff = end - start  # TODO: resolve this warning
        period_cagr = (data[end] / data[start]) ** (1 / year_diff) - 1
        data.insert(2 * i + 1, year_diff, period_cagr, allow_duplicates=True)

    total_year_diff = max_year - min_year
    total_cagr = (data[max_year] / data[min_year]) ** (1 / total_year_diff) - 1
    data.insert(len(data.columns), total_year_diff, total_cagr)
    return data


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


def country_data_segment(data, tags_colours: dict[str, str]) -> None:
    """Loop with country data analysis segment

    Args:
        data (dict[int, dict]): prepared country data
        tags_colours (dict[str, str]): dict with players' tags and corresponding colours

    """
    while True:
        statistic = _inp()

        dates, country_stats, income_stats = get_country_statistic(
            statistic=statistic,
            data=data,
            tags=tags_colours.keys(),
        )

        try:
            vals_only = _make_table(country_stats, dates)
            curr_year = vals_only.columns.max()

            if not income_stats:
                wd = world_data(statistic, data)
                vals_only = pd.concat([vals_only, wd])

            growth_rates = _calculate_growth_rates(vals_only)
            vals_and_growths = _combine_vals_and_growth_rates(vals_only, growth_rates)
            print(vals_and_growths, end="\n\n")
        except TypeError as e:
            print(f"this doesn't work: {e}")
            continue

        inp = input("save df? (y/n/q): ")
        if inp.lower() == "y":
            _export_table(
                vals_and_growths,
                tags_colours,
                title=statistic,
                year=curr_year,
            )
            print("df saved", end="\n\n")
        elif inp.lower() == "q":
            return

        inp = input("chart? (y/s/n/q): ")
        if inp.lower() == "y":
            if income_stats:
                _line_chart(dates, income_stats, tags_colours, statistic)
            else:
                _line_chart(dates, country_stats, tags_colours, statistic)
        elif inp.lower() == "s":
            if income_stats:
                _line_chart(
                    dates,
                    income_stats,
                    tags_colours,
                    statistic,
                    only_save=True,
                )
            else:
                _line_chart(
                    dates,
                    country_stats,
                    tags_colours,
                    statistic,
                    only_save=True,
                )
        elif inp.lower() == "q":
            return

        # world share segment
        if income_stats:
            inp = input("new stat? (y/n/q): ")
            if inp.lower() == "y":
                continue
            return

        inp = input("world share? (y/n/q): ")
        if inp.lower() == "y":
            wd = world_data(statistic, data)
            players_world_share = vals_only.drop("WORLD") / wd.loc["WORLD"]
            print(players_world_share, end="\n\n")

            inp = input("save df? (y/n/q): ")
            if inp.lower() == "y":
                _export_table(
                    players_world_share,
                    tags_colours,
                    title=statistic,
                    year=curr_year,
                    world_data=True,
                )
                print("df saved", end="\n\n")
            elif inp.lower() == "q":
                return

            inp = input("chart? (y/s/n/q): ")
            prepd_for_chart = _players_vs_world_data_for_chart(players_world_share)
            if inp.lower() == "y":
                _line_chart(
                    dates,
                    prepd_for_chart,
                    tags_colours,
                    statistic,
                    world_data=True,
                )
            elif inp.lower() == "s":
                _line_chart(
                    dates,
                    prepd_for_chart,
                    tags_colours,
                    statistic,
                    only_save=True,
                    world_data=True,
                )
            elif inp.lower() == "q":
                return
        elif inp == "q":
            return

        inp = input("new stat? (y/n/q): ")
        if inp.lower() == "y":
            pass
        else:
            return


def provinces_data_segment(data: pd.DataFrame) -> None:
    """Loop with province data analysis segment

    Args:
        data (pd.DataFrame): prepared province data

    """
    while True:
        inp = input("export provinces data (y/n/q): ")
        if inp == "y":
            _export_provinces_data(data)
        elif inp == "q":
            return


# TODO: rewrite the main function
def main():
    """Main function

    main function with the interface, can choose which api key to use and what to analyse
    """
    tags = get_tags()
    api_num = bool(int(input("\nchoose api (1 -> alan, 0 -> michal): ")))
    saves, api_key = get_saves(my_api=api_num)

    global_provinces_data = None
    global_countries_data = None
    tags_colours = None

    while True:
        inp = input("\nprovinces_data/tradenodes or country_data (0, 1, q): ")
        print()
        if inp == "0":
            if not global_provinces_data:
                print("requesting data...")
                data = get_global_provinces_data(saves, api_key)

                print("processing data...")
                global_provinces_data = prepare_provinces_data(data)

            provinces_data_segment(global_provinces_data)

        elif inp == "1":
            if not global_countries_data or not tags_colours:
                print("requesting data...")
                data = get_global_countries_data(saves, api_key)

                print("processing data...")
                global_countries_data, tags_colours = prepare_countries_data(data, tags)

                with open(OUTPUT_PATH / "tags_colours.json", "w") as f:
                    json.dump(tags_colours, f)
                print("tags_colours.json saved")

            country_data_segment(global_countries_data, tags_colours)

        elif inp == "q":
            return


if __name__ == "__main__":
    main()
