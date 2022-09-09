# ClimateBench

ClimateBench is a benchmark dataset for climate model emulation inspired by [WeatherBench](https://github.com/pangeo-data/WeatherBench). It consists of NorESM2 simulation outputs with associated forcing data processed in to a consistent format from a variety of experiments performed for CMIP6. Multiple ensemble members are included where available. 

The processed training, validation and test data can be obtained from Zenodo: [10.5281/zenodo.5196512](https://doi.org/10.5281/zenodo.5196512).

A pre-print of the paper describing ClimateBench and the baseline models can be found here: <https://www.essoar.org/doi/10.1002/essoar.10509765.2>

## Leaderboard

The spatial, global and total NRMSE of the different baseline emulators for the years 2080-2100 against the ClimateBench task of estimating key climate variables under future scenario SSP245. The models  are ranked in order of the mean of the total NRMSE across all tasks. 

|                  |   ('tas', 'Spatial') |   ('tas', 'Global') |   ('tas', 'Total') |   ('diurnal_temperature_range', 'Spatial') |   ('diurnal_temperature_range', 'Global') |   ('diurnal_temperature_range', 'Total') |   ('pr', 'Spatial') |   ('pr', 'Global') |   ('pr', 'Total') |   ('pr90', 'Spatial') |   ('pr90', 'Global') |   ('pr90', 'Total') |
|------------------|----------------------|---------------------|--------------------|--------------------------------------------|-------------------------------------------|------------------------------------------|---------------------|--------------------|-------------------|-----------------------|----------------------|---------------------|
| Neural Network   |             0.107294 |           0.0440271 |           0.327429 |                                    9.91735 |                                   1.37219 |                                  16.7783 |             2.1281  |           0.2093   |           3.1746  |               2.61022 |             0.345709 |             4.33876 |
| Gaussian Process |             0.109106 |           0.0738238 |           0.478225 |                                    9.20713 |                                   2.67495 |                                  22.5819 |             2.34092 |           0.341453 |           4.04818 |               2.5559  |             0.429154 |             4.70167 |
| Random Forest    |             0.107574 |           0.0584057 |           0.399602 |                                    9.19503 |                                   2.65241 |                                  22.4571 |             2.52431 |           0.502126 |           5.03494 |               2.68209 |             0.543375 |             5.39896 |


## Installation
The example scripts provided here require [ESEm](https://github.com/duncanwp/ESEm) and a few other packages. It is recommended to first create a conda environment with iris or xarray::

    $ conda install -c conda-forge iris

Then pip install the additional requirements:

    $ pip install esem[gpflow,keras,scikit-learn] eofs

