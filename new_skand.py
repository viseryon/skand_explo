import json
from pathlib import Path

import dataframe_image as dfi
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

plt.style.use("dark_background")


def get_tags() -> list[str]:
    """gets tags of chosen countries"""

    with open("countries.txt", "r") as file:
        tags = file.read()

    tags = tags.split(",")
    return tags


class SkanderbegAnalyzer:

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

    def __init__(self, tags: list[str], **kwargs) -> None:
        self._tags = tags

    @property
    def tags(self) -> list[str]:
        return self._tags

    @classmethod
    def from_file(cls, tags: list[str], obj):
        ...  # reading file
        return cls(tags=tags)

    @classmethod
    def from_web(cls, tags: list[str], api_key: str, countries_data: bool, provinces_data: bool):

        saves = cls._get_saves(api_key)
        with open("saves_metadata.json", "w") as file:
            json.dump(saves, file)

        if countries_data:
            c_data = cls._get_countries_data(saves, api_key)
        else:
            c_data = None

        if provinces_data:
            p_data = cls._get_provinces_data(saves, api_key)
        else:
            p_data = None

        return cls(tags, saves=saves, countries_data=c_data, provinces_data=p_data)

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


    # analysis

    def analyse(self):
        while True:
            ...

def main():

    with open("apis.json") as file:
        apis = json.load(file)

    tags = get_tags()

    analyser = SkanderbegAnalyzer.from_web(tags, apis["alan"], countries_data=True, provinces_data=False)
    analyser.analyse()


if __name__ == "__main__":
    main()
