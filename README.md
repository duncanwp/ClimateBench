# ClimateBench

ClimateBench is a benchmark dataset for climate model emulation inspired by [WeatherBench](https://github.com/pangeo-data/WeatherBench). It consists of NorESM2 simulation outputs with associated forcing data processed in to a consistent format from a variety of experiments performed for CMIP6. Multiple ensemble members are included where available. 

The processed training, validation and test data can be obtained from Zenodo: [10.5281/zenodo.5196512](https://doi.org/10.5281/zenodo.5196512).

## Leaderboard

| Model | TAS RMSE (2050 / 2100) [K] | DTR RMSE (2050 / 2100) [K] | Pr RMSE (2050 / 2100) [mm/day] | P90 RMSE (2050 / 2100) [mm/day] | 
|--------------------|----------------------------------|----------------------------|----------------------|------------------|
| Baseline GP | 0.32 / 0.41 | 0.14 / 0.15 | 0.42 / 0.62 | 1.29 / 1.82 |
| UNet | a / b | a / b | a /b | a /b |
| Random Forest | a / b | a/ b | a /b | a /b |
| UKESM | 1.71 / 2.70 | 1.17 / 1.34 | 0.59 / 0.82 | 1.77 / 2.48 |
| (Variability) | 0.53 / 0.59 | 0.21 / 0.22 | 0.74 / 0.86 | 2.26 / 2.55 |


## Installation
The example scripts provided here require [ESEm](https://github.com/duncanwp/ESEm) and a few other packages. It is recommended to first create a conda environment with iris::

    $ conda install -c conda-forge iris

Then pip install the additional requirements:

    $ pip install esem[gpflow,keras,scikit-learn] eofs

