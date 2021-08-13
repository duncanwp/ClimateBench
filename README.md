# ClimateBench

ClimateBench is a benchmark dataset for climate model emulation inspired by [WeatherBench](https://github.com/pangeo-data/WeatherBench). It consists of NorESM2 simulation outputs with asociated forcing data processed in to a consistent format from a variety of experiments performed for CMIP6. Multiple ensemble members are included where available. 

The processed data can be obtained from Zenodo: 10.5281/zenodo.5196513

## Hackathon

This benchmark dataset is currently being used for a hackathon during the NOAA AI [workshop](https://2021noaaaiworkshop.sched.com). Test data used for evaluation of these submissions will be released upon it's conclusion.

We hope to keep a ranking of participating models on this page.

## Installation
The example scripts provided here require [ESEm](https://github.com/duncanwp/ESEm) and a few other packages. It is recommended to first create a conda environment with iris:: 

    $ conda install -c conda-forge iris

Then pip install the aditional requirements::

    $ pip install esem[gpflow,keras,scikit-learn] eofs

