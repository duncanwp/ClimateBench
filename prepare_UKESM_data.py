#!/usr/bin/env python
"""
Prepare the NorESM2 output by getting it directly from the http://noresg.nird.sigma2.no THREDDS server
"""

from siphon import catalog
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster
import os.path
overwrite = False

model = 'UKESM1-0-LL'
#experiments = [
#               '1pctCO2', 'abrupt-4xCO2', 'historical', 'piControl', # CMIP
#               'hist-GHG', 'hist-aer', # DAMIP
#               'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF', 'ssp585' #	ScenarioMIP
#]
experiments = ['ssp245', 'piControl']
variables = [
             'tas', 'tasmin', 'tasmax', 'pr'
]


# So use the CEDA data directly 
# At e.g.: /badc/cmip6/data/CMIP6/ScenarioMIP/MOHC/UKESM1-0-LL/ssp245/r1i1p1f2/day/tasmin/gn/latest/


def get_MIP(experiment):
  if experiment == 'ssp245-covid':
    return 'DAMIP'
  elif experiment == 'ssp370-lowNTCF':
    return 'AerChemMIP'
  elif experiment.startswith('ssp'):
    return 'ScenarioMIP'
  elif experiment.startswith('hist-'):
    return 'DAMIP'
  else:
    return 'CMIP'


def get_ceda_data(variable, experiment, ensemble_member):

  path = f"/badc/cmip6/data/CMIP6/{get_MIP(experiment)}/MOHC/{model}/{experiment}/{ensemble_member}/day/{variable}/gn/latest/"
  ds = xr.open_mfdataset(path+"*.nc", combine_attrs='drop', chunks={})
  return ds[variable]

if __name__ == '__main__':
    #cluster = LocalCluster(n_workers=4, processes=True, diagnostics_port=None, scheduler_port=0, silence_logs=10, worker_dashboard_address=':0', dashboard_address=':0', threads_per_worker=1)
    #print(cluster)
    #client = Client(cluster, worker_dashboard_address=':0', dashboard_address=':0', local_directory='/tmp')
    #Loop over experiments and members creating one (annual mean) file with all variables in for each one
    for experiment in experiments:
      # Just take three ensemble members (there are more in the COVID simulation but we don't need them all)
      for i in range(3):
        physics = 2 
        member = f"r{i+1}i1p1f{physics}"
        print(f"Processing {member} of {experiment}...")
        outfile = f"{model}_{experiment}_{member}.nc"
        if (not overwrite) and os.path.isfile(outfile):
          print("File already exists, skipping.")
          continue

        try: 
          tasmin = get_ceda_data('tasmin', experiment, member)
          tasmax = get_ceda_data('tasmax', experiment, member)
          tas = get_ceda_data('tas', experiment, member)
          pr = get_ceda_data('pr', experiment, member).persist()  # Since we need to process it twice
        except IndexError:
          print("Skipping this realisation as no data present")
          continue
        
        # Derive additional vars
        dtr = tasmax-tasmin
        ds = xr.Dataset({'diurnal_temperature_range': dtr.groupby('time.year').mean('time'),
                         'tas': tas.groupby('time.year').mean('time'),
                         'pr': pr.groupby('time.year').mean('time'),
                         'pr90': pr.groupby('time.year').quantile(0.9, skipna=True)})
        ds.to_netcdf(f"{model}_{experiment}_{member}.nc")

