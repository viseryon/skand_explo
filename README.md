# My Skanderbeg.pm save exploration code
![Expansion](/assets/expansion_without_bg.png)
## About code
It's terrrible, but does the job. I plan to rewrite it at some point. You can create tables with a particular stat across time (multiple saves) and chart it.

<b>This script requests data from skanderbeg.pm API and downloads all uploaded saves. If there are saves from different campaings it may produce weird results!</b>


![Development](/assets/real_development_by_1630.png)

The table show values for given save date and CAGR between dates with the last one being from the beginning of the game. There's also an option to show values as % of the world's.

![Income](/assets/income_stats.jpg)

If you download the provinces data via this script you can later use it to chart maps with geopandas. This repo includes .geojson files. `eu4base.geojson` only has data that stays mostly fixed during a campaign (no dev, culture, owner data). `eu4.geojson` has data as of starting date.

![Scandinavia](/assets/SCA_1609.png)


## Thanks to Skanderbeg.pm
### dfi.export bug
In case of this bug  
`\dataframe_image\converter\browser\chrome_converter.py", line 140, in screenshot
    subprocess.run(
  File "\Lib\subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['--enable-logging', '--disable-gpu', '--headless', '--crash-dumps-dir=\\.dataframe_image\\tmp69jtb92e', '--force-device-scale-factor=2.0', '--window-size=1400,900', '--hide-scrollbars', '--screenshot=\\.dataframe_image\\tmp69jtb92e\\temp.png', '\\.dataframe_image\\tmp69jtb92e\\temp.html']' returned non-zero exit status 21.`  
replace '--headless' by '--headless=old'.