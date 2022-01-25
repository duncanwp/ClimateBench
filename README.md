# ClimateBench

ClimateBench is a benchmark dataset for climate model emulation inspired by [WeatherBench](https://github.com/pangeo-data/WeatherBench). It consists of NorESM2 simulation outputs with associated forcing data processed in to a consistent format from a variety of experiments performed for CMIP6. Multiple ensemble members are included where available. 

The processed training, validation and test data can be obtained from Zenodo: [10.5281/zenodo.5196512](https://doi.org/10.5281/zenodo.5196512).

## Leaderboard

The average root mean square error (RMSE) of the different baseline emulators for the years 2050-2100 against the ClimateBench task of estimating key climate variables under future scenario SSP245. Another state-of-the-art model (UKESM1) and the average RMSE between NorESM ensemble members as an estimate of internal variability are included for comparison.

| Model | TAS RMSE [K] | DTR RMSE [K] | Pr RMSE [mm/day] | P90 RMSE [mm/day] | 
|--------------------|----------------------------------|----------------------------|----------------------|------------------|
| GP regression | 0.36 (CRPS: 0.33) | 0.15 (CRPS: 0.12) | 0.53 (CRPS: 0.42) | 1.54 (CRPS: 1.27) |
| CNN+LSTM | 0.38 | 0.17 | 0.58 | 1.64 |
| Random Forest | 0.42 | 0.15 | 0.53 | 1.54 |
| UKESM | 2.20 | 1.28 | 0.89 | 2.57 |
| (Variability) | 0.80 | 0.31 | 1.20 | 3.52 |


## Installation
The example scripts provided here require [ESEm](https://github.com/duncanwp/ESEm) and a few other packages. It is recommended to first create a conda environment with iris or xarray::

    $ conda install -c conda-forge iris

Then pip install the additional requirements:

    $ pip install esem[gpflow,keras,scikit-learn] eofs

