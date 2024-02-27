import json

import dataframe_image as dfi
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

plt.style.use("dark_background")

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

# data request


def get_apis():
    with open("apis.json", "r") as f:
        apis = json.load(f)

    return apis


def get_saves(my_api=True) -> tuple[dict[int, str], str]:
    """get save ids and correct api key"""

    apis = get_apis()

    if my_api:
        api_key = apis["alan"]
    else:
        api_key = apis["michal"]

    params = {"key": api_key, "scope": "fetchUserSaves"}
    r = requests.get(SKANDERBEG_LINK, params=params).json()
    saves = dict()

    for info in r:
        saves[int(info["date"][:4])] = info["hash"]

    return saves, api_key


def get_global_provinces_data(
    saves: dict[int, str], api_key: str, global_provinces_data=None
) -> dict[int, dict]:
    """get global provinces data"""

    if global_provinces_data:
        return global_provinces_data

    data = dict()

    for date, save in saves.items():
        params = dict(
            scope="getSaveDataDump", save=save, api_key=api_key, type="provincesData"
        )
        response = requests.get(SKANDERBEG_LINK, params=params).json()
        data[date] = response

    return data


def get_global_countries_data(
    saves: dict[int, str], api_key: str, global_countries_data=None
) -> dict[int, dict]:
    """get global countries data"""

    if global_countries_data:
        return global_countries_data

    data = dict()

    for date, save in saves.items():
        params = dict(
            scope="getSaveDataDump", save=save, api_key=api_key, type="countriesData"
        )
        response = requests.get(SKANDERBEG_LINK, params=params)
        if response.status_code != 200:
            print(response)
            exit()
        response = response.json()

        data[date] = response

    return data


# chosen countries


def get_tags() -> list[str]:
    """gets tags of chosen countries"""

    with open("countries.txt", "r") as file:
        tags = file.read()

    tags = tags.split(",")
    return tags


# countries data


def prepare_countries_data(data: dict[int, dict], tags: list[str]) -> tuple[dict, dict]:
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

    return data, colours


def get_country_statistic(
    statistic: str, data: dict[int, dict], tags: list[str]
) -> tuple:
    """get time series of a statistic for tags"""

    dates = data.keys()
    dates = sorted(dates)

    if statistic == "income_stats":
        return _income_stats(data, tags)

    country_stats = {tag: {"dates": [], "stats": []} for tag in tags}

    for date in dates:
        vals = data[date]

        for tag in tags:
            stat = vals[tag].get(statistic)
            if stat == "0.001" or stat == "0" or stat == "1" or stat == "1000":
                stat = None
            country_stats[tag]["dates"].append(date)
            country_stats[tag]["stats"].append(stat if stat == None else float(stat))

    return dates, country_stats, None


def _income_stats(data, tags):
    dates = data.keys()
    dates = sorted(dates)

    income_stats = {}
    for date in dates:
        vals = data[date]

        for tag in tags:
            stat = vals[tag].get("income_stats")

            if not stat:
                continue

            income_stats.setdefault(tag, {})
            income_stats[tag].update({int(year): v for year, v in stat.items()})

    income_stats = {
        tag: {"dates": list(v.keys()), "stats": list(v.values())}
        for tag, v in income_stats.items()
    }

    tmp = {}
    for tag, vals in income_stats.items():
        tmp[tag] = {k: v for k, v in zip(vals["dates"], vals["stats"])}

    country_stats = {}

    dates[0] = 1445
    for date in dates:
        for tag in tags:
            country_stats.setdefault(tag, {"dates": list(), "stats": list()})
            country_stats[tag]["dates"].append(date - 1)
            country_stats[tag]["stats"].append(tmp[tag].get(date))

    dates[0] = 1444
    return dates, country_stats, income_stats


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


def _combine_vals_and_growth_rates(
    vals: pd.DataFrame, growths: pd.DataFrame
) -> pd.DataFrame:
    """combines df of values with df of growth rates"""

    full_df = pd.DataFrame()

    for (k1, v1), (k2, v2) in zip(vals.items(), growths.items()):
        full_df[k1] = v1
        full_df[k2] = v2

    return full_df


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


def _export_table(
    vals_and_growths: pd.DataFrame,
    colours: dict[str, str],
    title: str,
    year,
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
        k: f"background-color: black; color: {v}" for k, v, in colours.items()
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

    formater = formater.set_caption(
        f"{title}{f' as % of world {title}' if world_data else ''}"
    )

    subset = pd.IndexSlice[[indx for indx in full.index if indx != "WORLD"], :]
    formater = formater.highlight_max(color="darkgreen", subset=subset)
    formater = formater.highlight_min(color="darkred", subset=subset)

    formater = formater.format(nr_formats)

    dfi.export(
        formater,
        f"charts/{title}_by_{year}{'_as_%world' if world_data else ''}.png",
        dpi=200,
    )

    return


def _line_chart(
    dates: list[int],
    country_stats: dict[str, dict[str, list[int]]],
    tags_colours: dict[str, str],
    title: str,
    only_save=False,
    world_data=False,
) -> None:
    """make line chart of chosen statistic"""

    fig, ax = plt.subplots(figsize=(13, 7))
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

    ax.legend(country_stats.keys(), loc="upper left")
    ax.set_title(f"{title}{f' as % of world {title}' if world_data else ''}")
    ax.set_xticks(dates)

    if world_data:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:.4%}"))

    ax.grid(visible=True, axis="y", alpha=0.25)
    fig.tight_layout()

    if only_save:
        fig.savefig(
            fname=f"charts/{title}{'_as_%world' if world_data else ''}.jpg", dpi=200
        )
        print("fig saved", end="\n\n")
        return

    fig.show()

    inp = input("save fig? (y/n/q): ")
    if inp.lower() == "y":
        fig.savefig(
            fname=f"charts/{title}{'_as_%world' if world_data else ''}.jpg", dpi=200
        )
        print("fig saved", end="\n\n")

    return


def world_data(statistic: str, global_countries_data: dict):
    data = {}
    for date, val in global_countries_data.items():
        data[date] = {"tags": (tag for tag in val if len(tag) == 3)}
        year_value = 0

        for tag in data[date]["tags"]:
            v = global_countries_data[date][tag].get(statistic, 0)
            year_value += float(v)

        data[date] = [year_value]

    df = pd.DataFrame(data, index=["WORLD"])
    df = df.sort_index(axis=1)

    return df


def _players_vs_world_data_for_chart(world_share):
    dates = world_share.columns
    prepared_for_chart = {}
    for tag, vals in world_share.T.items():
        prepared_for_chart[tag] = {"stats": vals.values.tolist(), "dates": dates}

    return prepared_for_chart


# provinces data


def prepare_provinces_data(data: dict[int, dict]) -> pd.DataFrame:
    """
    prepares province data from skanderbeg

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
        }
    )

    master["dev"] = master["tax"] + master["prod"] + master["manp"]

    return master


def _export_provinces_data(data: pd.DataFrame) -> None:
    data.to_csv("provinces_data.csv", index=False)
    print("exported")


# main loop


def _inp() -> str:
    value = input("""\nchoose metric or say "help": """)
    print()

    if value.lower() == "help":
        print(ALL_STATS)
        return _inp()
    elif value.lower() == "q":
        exit()
    else:
        if value not in ALL_STATS:
            print("not in available statistics")
            return _inp()
        else:
            return value


def country_data_segment(data, tags_colours: dict[str, str]) -> None:
    """
    loop with country data analysis segment

    Args:
        data (dict[int, dict]): prepared country data
        tags_colours (dict[str, str]): dict with players' tags and corresponding colours
    """

    while True:
        statistic = _inp()

        dates, country_stats, income_stats = get_country_statistic(
            statistic=statistic, data=data, tags=tags_colours.keys()
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
                vals_and_growths, tags_colours, title=statistic, year=curr_year
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
                    dates, income_stats, tags_colours, statistic, only_save=True
                )
            else:
                _line_chart(
                    dates, country_stats, tags_colours, statistic, only_save=True
                )
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
                    dates, prepd_for_chart, tags_colours, statistic, world_data=True
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
    """
    loop with province data analysis segment

    Args:
        data (pd.DataFrame): prepared province data
    """

    while True:
        inp = input("export provinces data (y/n/q): ")
        if inp == "y":
            _export_provinces_data(data)
        elif inp == "q":
            return


def main():
    """
    main function

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

                with open("tags_colours.json", "w") as f:
                    json.dump(tags_colours, f)

            country_data_segment(global_countries_data, tags_colours)

        elif inp == "q":
            return


if __name__ == "__main__":
    main()
