#!/usr/bin/env python
"""

"""

from siphon import catalog
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster
import os.path
overwrite = False

model = 'NorESM2-LM'
experiments = [
               '1pctCO2', 'abrupt-4xCO2', 'historical', 'piControl', # CMIP
               'hist-GHG', 'hist-aer', # DAMIP
               'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF', 'ssp585' #	ScenarioMIP
]
variables = [
             'tas', 'tasmin', 'tasmax', 'pr'
]

# Check the PANGEO holdings
# for AWS S3:
#df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6-noQC.csv")

# Unfortuntaly they're missing a couple of the scenarios. 
#set(experiments) - set(df.query("source_id==@model & experiment_id in @experiments").experiment_id.unique())

# And the ones that are there don't have all the variables I need

# So use the NIRD ESGF node directly 
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


def get_esgf_data(variable, experiment, ensemble_member):
  """
  Inspired by https://github.com/rabernat/pangeo_esgf_demo/blob/master/narr_noaa_thredds.ipynb
  """

  # Get the relevant catalog references
  cat_refs = list({k:v for k,v in full_catalog.catalog_refs.items() if k.startswith(f"CMIP6.{get_MIP(experiment)}.NCC.NorESM2-LM.{experiment}.{ensemble_member}.day.{variable}.")}.values()) 
  # Get the latest version (in case there are multiple)
  print(cat_refs)
  cat_ref = sorted(cat_refs, key=lambda x: str(x))[-1]
  print(cat_ref)
  sub_cat = cat_ref.follow().datasets
  datasets = []
  # Filter and fix the datasets
  for cds in sub_cat[:]:
    # Only pull out the (un-aggregated) NetCDF files
    if (str(cds).endswith('.nc') and ('aggregated' not in str(cds))):
      # For some reason these OpenDAP Urls are not referred to as Siphon expects...
      cds.access_urls['OPENDAP'] = cds.access_urls['OpenDAPServer']
      datasets.append(cds)
  dsets = [(cds.remote_access(use_xarray=True)
             .reset_coords(drop=True)
             .chunk({'time': 365}))
         for cds in datasets]
  ds = xr.combine_by_coords(dsets, combine_attrs='drop')
  return ds[variable]

if __name__ == '__main__':
    #cluster = LocalCluster(n_workers=4, processes=True, diagnostics_port=None, scheduler_port=0, silence_logs=10, worker_dashboard_address=':0', dashboard_address=':0', threads_per_worker=1)
    #print(cluster)
    #client = Client(cluster, worker_dashboard_address=':0', dashboard_address=':0', local_directory='/tmp')
    #print(client)
    print("starting")
    # Cache the full catalogue from NorESG
    full_catalog = catalog.TDSCatalog('http://noresg.nird.sigma2.no/thredds/catalog/esgcet/catalog.xml')
    print("Read full catalogue")
    #Loop over experiments and members creating one (annual mean) file with all variables in for each one
    for experiment in experiments:
      # Just take three ensemble members (there are more in the COVID simulation but we don't need them all)
      for i in range(3):
        physics = 2 if experiment == 'ssp245-covid' else 1  # The COVID simulation uses a different physics setup 
        # TODO - check the differences...
        member = f"r{i+1}i1p1f{physics}"
        print(f"Processing {member} of {experiment}...")
        outfile = f"{model}_{experiment}_{member}.nc"
        if (not overwrite) and os.path.isfile(outfile):
          print("File already exists, skipping.")
          continue

        try: 
          tasmin = get_esgf_data('tasmin', experiment, member)
          tasmax = get_esgf_data('tasmax', experiment, member)
          tas = get_esgf_data('tas', experiment, member)
          pr = get_esgf_data('pr', experiment, member).persist()  # Since we need to process it twice
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

