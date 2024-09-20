import json
from pathlib import Path

import dataframe_image as dfi
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

plt.style.use("dark_background")


class Analyser:

    SKANDERBEG_LINK = "https://skanderbeg.pm/api.php"

    ALL_STATS = set(
        [
            "buildings_value",
            "totalRulerManaGain",
            "real_development",
            "total_development",
            "technology",
            "gp_score",
            "naval_strength",
            "warscore_cost",
            "provinces",
            "forts",
            "score_ranking",
            "prestige",
            "stability",
            "treasury",
            "monthly_income",
            "army_tradition",
            "income",
            "subsidies",
            "inc_no_subs",
            "interest",
            "expense",
            "expenses",
            "army_professionalism",
            "government_reforms",
            "manpower",
            "max_manpower",
            "total_army",
            "army_size",
            "total_ships",
            "total_navy",
            "total_casualties",
            "battleCasualties",
            "attritionCasualties",
            "navalCasualties",
            "innovativeness",
            "manpower_dev",
            "armyStrength",
            "totalManaGainAverage",
            "qualityScore",
            "FL",
            "shock_damage",
            "fire_damage",
            "land_morale",
            "naval_morale",
            "discipline",
            "at",
            "addedTraditionDisci",
            "score",
            "manpower_percentage",
            "manpower_dev_ratio",
            "total_mana_spent",
            "total_mana_spent_on_deving",
            "total_mana_spent_on_barrages",
            "total_mana_spent_on_culture_converting",
            "total_mana_spent_on_reducing_we",
            "total_mana_spent_on_hiring_generals",
            "total_mana_spent_on_force_marching",
            "total_mana_spent_on_militarization",
            "total_mana_spent_on_scorching",
            "total_mana_spent_on_stabbing_up",
            "total_mana_on_teching_up",
            "average_mana_spent_on_tech",
            "total_mana_on_teching_up_ex_mil",
            "total_mana_spent_on_assaulting",
            "total_mana_spent_on_coring",
            "spent_total",
            "spent_on_advisors",
            "spent_on_interest",
            "spent_on_states",
            "spent_on_subsidies",
            "spent_on_army_recruitment",
            "spent_on_loans",
            "spent_on_gifts",
            "spent_on_forts",
            "spent_on_army_maintenance",
            "spent_on_navies",
            "spent_on_buildings",
            "total_development_ratio",
            "real_development_ratio",
            "income_ratio",
            "fdp",
            "merc_fl",
            "manpower_recovery",
            "buildings_value_avg",
            "dev_ratio",
            "deving_stats",
            "deving_ratios",
            "adjustedEffectiveDisci",
            "quantityScore",
            "adjustedArmyStrength",
            "strengthTest",
            "overall_strength",
            "armyStrengthRatio",
            "army_str_to_dev",
            "average_monarch_total",
            "average_monarch",
            "weighted_avg_monarch",
            "technologyTD",
            "distinct_score",
            "quantityScore_ratio",
            "average_autonomy",
            "spent_on_forts_build",
            "dev_total",
            "manpower_recoveryy",
            "income_stats",
        ]
    )

    @staticmethod
    def get_tags() -> list[str]:
        """gets tags of chosen countries"""

        with open("countries.txt", "r") as file:
            tags = file.read()

        tags = tags.split(",")
        return tags

    ### INIT

    def __init__(
        self,
        tags: list[str],
        saves: dict,
        countries_data: dict | None = None,
        provinces_data: dict | None = None,
        **kwargs,
    ) -> None:
        self.tags = tags
        self.saves = saves
        self.countries_data = countries_data
        self.provinces_data = provinces_data
        self.__dict__.update(kwargs)

    @classmethod
    def from_file(cls, tags: list[str], obj):
        ...  # reading file
        return NotImplementedError

    @classmethod
    def from_web(cls, tags: list[str], api_key: str, countries_data: bool = True):

        saves = cls._get_saves(api_key)
        with open("saves_metadata.json", "w") as file:
            json.dump(saves, file)

        if countries_data:
            c_data = cls._get_countries_data(saves, api_key)
            c_data, colours, saves_dates, curr_year = cls._prepare_countries_data(c_data, tags)
            return cls(
                tags,
                saves=saves,
                countries_data=c_data,
                colours=colours,
                saves_dates=saves_dates,
                curr_year=curr_year,
                api_key=api_key,
            )
        return cls(
            tags,
            saves=saves,
            api_key=api_key,
        )

    @classmethod
    def _get_saves(cls, api_key: str):

        params = {"key": api_key, "scope": "fetchUserSaves"}
        response = requests.get(cls.SKANDERBEG_LINK, params=params).json()

        saves = dict()
        for data in response:
            year = int(data["date"][:4])
            hash_ = data["hash"]
            saves[year] = hash_

        return saves

    @classmethod
    def _get_countries_data(cls, saves: dict[int, str], api_key: str) -> dict[int, dict]:

        data = dict()
        with requests.Session() as session:
            for date, save in saves.items():
                params = dict(
                    scope="getSaveDataDump", save=save, api_key=api_key, type="countriesData"
                )
                response = session.get(cls.SKANDERBEG_LINK, params=params)
                if response.status_code != 200:
                    print(response)
                    exit()
                response = response.json()

                data[date] = response

        return data

    @classmethod
    def _get_provinces_data(cls, saves: dict[int, str], api_key: str):
        """get global provinces data"""

        data = dict()
        for date, save in saves.items():
            params = dict(scope="getSaveDataDump", save=save, api_key=api_key, type="provincesData")
            response = requests.get(cls.SKANDERBEG_LINK, params=params).json()
            data[date] = response

        return data

    @classmethod
    def _prepare_countries_data(cls, data, tags):
        """prepares countries data by changing tags to current ones"""

        save_dates = data.keys()
        save_dates = sorted(save_dates)
        curr_save = max(save_dates)

        years_tags = {year: [] for year in save_dates}
        tag_swaps = {}
        colours = {}
        for tag in tags:
            reforms = data[curr_save][tag].get("reformations")

            if reforms:
                for reform in reforms:
                    prev_tag = reform["tag"]
                    formation_year = int(reform["date"][:4])

                    tag_swaps.setdefault(tag, [])
                    tag_swaps[tag].append((prev_tag, formation_year))
            else:
                for year, val in years_tags.items():
                    years_tags[year].append(tag)

            colours[tag] = data[curr_save][tag]["hex"]

        for curr_tag, tag_formation in tag_swaps.items():
            tag_formation = sorted(tag_formation, key=lambda x: x[1])
            tag_formation.append((curr_tag, 2_000))
            for save_date in save_dates:
                for prev_tag, formation_year in tag_formation:
                    if save_date == 1444:
                        new_tag = prev_tag
                        break

                    if formation_year > save_date:
                        new_tag = prev_tag
                        break

                data[save_date][curr_tag] = data[save_date][new_tag]
                if new_tag != curr_tag:
                    del data[save_date][new_tag]

        return data, colours, save_dates, curr_save

    ### CALCULATIONS

    @staticmethod
    def _make_table(data: dict[str, dict], dates: list[int]) -> pd.DataFrame:
        """creates table with tags and statistic values"""

        df = pd.DataFrame(
            [[stat for stat in vals["stats"]] for tag, vals in data.items()],
            index=data,
            columns=dates,
        )
        df = df.round(2)
        df = df.sort_index(axis=1)

        return df

    def _world_data(self, statistic: str):
        data = {}
        for date, val in self.countries_data.items():
            data[date] = {"tags": (tag for tag in val if len(tag) == 3)}
            year_value = 0

            for tag in data[date]["tags"]:
                v = self.countries_data[date][tag].get(statistic, 0)
                year_value += float(v)

            data[date] = [year_value]

        df = pd.DataFrame(data, index=["WORLD"])
        df = df.sort_index(axis=1)

        return df

    @staticmethod
    def _players_vs_world_data_for_chart(world_share):
        dates = world_share.columns
        prepared_for_chart = {}
        for tag, vals in world_share.T.items():
            prepared_for_chart[tag] = {"stats": vals.values.tolist(), "dates": dates}

        return prepared_for_chart

    @staticmethod
    def _calculate_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
        years = df.columns
        mx_year = years.max()
        mn_year = years.min()

        growths = pd.DataFrame()
        for i in range(len(df.columns) - 1):
            start_date, end_date = int(years[i]), int(years[i + 1])
            start, end = df[start_date], df[end_date]
            years_diff = end_date - start_date
            growth = (end / start) ** (1 / years_diff) - 1

            col_title = str(years_diff)
            while col_title in growths:
                col_title += " "

            growths[col_title] = growth

        full_growth = (df[mx_year] / df[mn_year]) ** (1 / (mx_year - mn_year)) - 1
        growths[f"{mx_year - mn_year} "] = full_growth

        return growths

    @staticmethod
    def _combine_vals_and_growth_rates(vals: pd.DataFrame, growths: pd.DataFrame) -> pd.DataFrame:
        """combines df of values with df of growth rates"""

        full_df = pd.DataFrame()

        for (k1, v1), (k2, v2) in zip(vals.items(), growths.items()):
            full_df[k1] = v1
            full_df[k2] = v2

        return full_df

    def calculate_statistic(self, statistic: str):

        dates = self.countries_data.keys()
        dates = sorted(dates)

        country_stats = {tag: {"dates": [], "stats": []} for tag in self.tags}

        for date in dates:
            vals = self.countries_data[date]

            for tag in self.tags:
                stat = vals[tag].get(statistic)
                if stat in {"0.001", "0", "1", "1000"}:
                    stat = None
                country_stats[tag]["dates"].append(date)
                country_stats[tag]["stats"].append(stat if stat is None else float(stat))

        vals_only = self._make_table(country_stats, dates)

        wd = self._world_data(statistic)
        vals_only = pd.concat([vals_only, wd])

        growth_rates = self._calculate_growth_rates(vals_only)
        vals_and_growths = self._combine_vals_and_growth_rates(vals_only, growth_rates)
        print(vals_and_growths, end="\n\n")

        return dates, vals_only, vals_and_growths

    # ANALYSIS
    @classmethod
    def _stat_input(cls):
        value = input("""choose metric or say "help": """)
        print()

        if value.lower() == "help":
            print(
                "\nhere are all available statistics listed on skanderbeg.pm, not all of them will work here",
                cls.ALL_STATS,
                sep="\n",
                end="\n\n",
            )
            return cls._stat_input()
        elif value.lower() == "q":
            exit()

        if value not in cls.ALL_STATS:
            print("not in available statistics")
            return cls._stat_input()
        else:
            return value

    def analyse(self):
        while True:
            inp = input("""provinces_data/tradenodes -> 0\ncountry_data -> 1\nexit -> q\n: """)
            if inp == "0":
                if not self.provinces_data:
                    print("no provinces data available")
                    continue
                self._provinces_data_segment()
            elif inp == "1":
                if not self.countries_data:
                    print("no countries data available")
                    continue
                self._country_data_segment()
            elif inp == "q":
                return

    def _country_data_segment(self):

        while True:
            statistic = self._stat_input()

            try:
                dates, values, country_stats = self.calculate_statistic(statistic=statistic)
            except TypeError as e:
                print(f"this doesn't work: {e}")
                continue

            inp = input("save df? (y/n/q): ")
            if inp.lower() == "y":
                self._export_table(country_stats, title=statistic)
                print("df saved", end="\n\n")
            elif inp.lower() == "q":
                return

            inp = input("chart? (y/s/n/q): ")
            if inp.lower() == "y":
                if income_stats:
                    self._line_chart(dates, income_stats, statistic)
                else:
                    self._line_chart(dates, country_stats, statistic)
            elif inp.lower() == "s":
                if income_stats:
                    self._line_chart(dates, income_stats, statistic, only_save=True)
                else:
                    self._line_chart(dates, country_stats, statistic, only_save=True)
            elif inp.lower() == "q":
                return

            # world share segment
            if income_stats:
                inp = input("new stat? (y/n/q): ")
                if inp.lower() == "y":
                    continue
                else:
                    return

            inp = input("world share? (y/n/q): ")
            if inp.lower() == "y":
                wd = self._world_data(statistic)
                players_world_share = values.drop("WORLD") / wd.loc["WORLD"]
                print(players_world_share, end="\n\n")

                inp = input("save df? (y/n/q): ")
                if inp.lower() == "y":
                    self._export_table(
                        players_world_share,
                        title=statistic,
                        world_data=True,
                    )
                    print("df saved", end="\n\n")
                elif inp.lower() == "q":
                    return

                inp = input("chart? (y/s/n/q): ")
                prepd_for_chart = self._players_vs_world_data_for_chart(players_world_share)
                if inp.lower() == "y":
                    self._line_chart(dates, prepd_for_chart, statistic, world_data=True)
                elif inp.lower() == "s":
                    self._line_chart(
                        dates,
                        prepd_for_chart,
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

    def _provinces_data_segment(self):
        while True:
            inp = input("export provinces data (y/n/q): ")
            if inp == "y":
                _export_provinces_data(data)
            elif inp == "q":
                return

    ### EXPORTING

    def _export_table(
        self,
        vals_and_growths: pd.DataFrame,
        title: str,
        world_data=False,
    ) -> None:
        """make it beautiful and export df to png"""

        full = vals_and_growths

        if world_data:
            nr_formats = {x: "{:,.4%}" for x in full.keys()}
        else:
            nr_formats = {x: "{:,.1f}" if int(x) > 1_000 else "{:.1%}" for x in full.keys()}

        full = full.sort_values(full.columns[-1], ascending=False)

        colours_for_index = {
            k: f"background-color: black; color: {v}" for k, v, in self.colours.items()
        }

        formater = full.style
        formater = formater.set_properties(
            **{"background-color": "black", "color": "white", "border-color": "white"}
        )

        formater = formater.map_index(
            lambda x: colours_for_index.get(x, "background-color: black; color: white"),
            axis=0,
        )
        formater = formater.map_index(
            lambda x: "background-color: black; color: white; font-size: 125%", axis=1
        )

        formater = formater.set_table_styles(
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
            ]
        )

        caption = f" as % of world's {title}" if world_data else ""
        formater = formater.set_caption(title + caption)

        subset = pd.IndexSlice[[indx for indx in full.index if indx != "WORLD"], :]
        formater = formater.highlight_max(color="darkgreen", subset=subset)
        formater = formater.highlight_min(color="darkred", subset=subset)

        formater = formater.format(nr_formats)

        dfi.export(
            formater,
            f"charts/{title}_by_{self.curr_year}{'_as_%world' if world_data else ''}.png",
            dpi=200,
        )

        return

    def _line_chart(
        self,
        dates: list[int],
        country_stats: dict[str, dict[str, list[int]]],
        title: str,
        only_save=False,
        world_data=False,
    ) -> None:
        """make line chart of chosen statistic"""

        fig, ax = plt.subplots(figsize=(13, 7))
        for tag, vals in country_stats.items():
            ax.plot(vals["dates"], vals["stats"], color=self.colours[tag])

            if world_data:
                annotation = f"{tag} | {vals['stats'][-1]:.4%}"
            else:
                annotation = f"{tag} | {vals['stats'][-1]:_.2f}"

            ax.annotate(
                f"{annotation}",
                (max(dates), vals["stats"][-1]),
                fontsize=10,
                color=self.colours[tag],
                bbox=dict(facecolor="black", edgecolor="white", boxstyle="round"),
            )

        ax.legend(country_stats.keys(), loc="upper left")
        ax.set_title(f"{title}{f' as % of world {title}' if world_data else ''}")
        ax.set_xticks(dates)

        if world_data:
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.4%}"))

        ax.grid(visible=True, axis="y", alpha=0.25)
        fig.tight_layout()

        if only_save:
            fig.savefig(fname=f"charts/{title}{'_as_%world' if world_data else ''}.jpg", dpi=200)
            print("fig saved", end="\n\n")
            return

        fig.show()

        inp = input("save fig? (y/n/q): ")
        if inp.lower() == "y":
            fig.savefig(fname=f"charts/{title}{'_as_%world' if world_data else ''}.jpg", dpi=200)
            print("fig saved", end="\n\n")

        return

    def save_data(self):

        if self.countries_data:
            with open("countries_data.json", "w") as file:
                json.dump(self.countries_data, file)
            print("countries data saved", end="\n\n")

        inp = input("request and save provinces data (y/n): ")
        if inp == "y":
            provinces_data = self._get_provinces_data(self.saves, self.api_key)
            with open("provinces_data.json", "w") as file:
                json.dump(provinces_data, file)
            print("provinces data saved", end="\n\n")

        if self.colours:
            with open("tags_colours.json", "w") as file:
                json.dump(self.colours, file)
            print("colours saved", end="\n\n")


def main():

    with open("api.txt") as file:
        api_key = file.read()

    tags = Analyser.get_tags()

    analyser = Analyser.from_web(tags, api_key, countries_data=True)
    analyser.analyse()
    analyser.save_data()


if __name__ == "__main__":
    main()
