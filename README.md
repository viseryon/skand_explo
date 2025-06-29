# My EU4 Save Exploration Tool

![Expansion](/assets/expansion_without_bg.png)

## Project Overview

This project analyzes EUIV save files using data from the [skanderbeg.pm](https://skanderbeg.pm) API. It generates tables and charts for various in-game statistics across multiple saves, helping you visualize campaign progress and compare stats over time.

## Example Visualizations

### Table

The script lets you create tables with the players and other tags to compare progress over time. Tables may include CAGR values to measure how well the session went.

![Max Manpower](/assets/MAX_MANPOWER_BY_1660_TABLE.png)

### Line chart

![Income stats](/assets/INCOME_STATS_BY_1660_LINE_CHART.png)

![Income stats WES](/assets/INCOME_STATS_BY_1660_LINE_CHART_WES.png)

### Maps

If you download the provinces data via this script you can later use it to chart maps with geopandas. This repo includes `/eu4provinces.geojson` file that has data that stays mostly fixed during a campaign (no dev, culture, owner data), which is merged with the data downloaded from [skanderbeg.pm](https://skanderbeg.pm).

![Scandinavia](/assets/SCA_1609.png)

## Features

- Downloads and processes all uploaded saves from skanderbeg.pm.
- Generates tables showing stats (e.g., development, income, manpower) for selected dates and calculates CAGR between them.
- Creates line charts and summary tables for key statistics.
- Optionally displays values as a percentage of the world's total.
- Supports map visualizations using provided `.geojson` files and `geopandas`.
- Includes pre-generated charts and images for quick reference.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and installation.

Make sure you have `uv` installed. See the [uv documentation](https://github.com/astral-sh/uv) for installation instructions.

## Usage Notes

- If you have saves from different campaigns in your skanderbeg.pm account, results may be inconsistent.
- Downloaded provinces data can be used for custom map visualizations.
- The `data/` folder contains `eu4provinces.geojson`: mostly static data (e.g., province shapes, base info)

## Acknowledgements

Thanks to [Skanderbeg.pm](https://skanderbeg.pm) for the API and data resources.
