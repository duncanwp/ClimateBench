import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof
data_path = './data/train_val/'


def create_predictor_data(data_set, n_eofs=5):
    """
    Args:
        data_set (str): name of dataset
        n_eofs (int): number of eofs to create for aerosol variables

    """
    X = xr.open_dataset(data_path + "inputs_{}.nc".format(data_set)).compute()

    if data_set == "hist-aer":
        X = X.rename_vars({"CO4": "CO2"})
        X = X.sel(time=slice(1850, 2014))

    if data_set == "hist-GHG":
        X = X.sel(time=slice(1850, 2014))

    if "ssp" in data_set or data_set == "hist-aer":
        # Compute EOFs for BC
        bc_solver = Eof(X['BC'])
        bc_eofs = bc_solver.eofsAsCorrelation(neofs=n_eofs)
        bc_pcs = bc_solver.pcs(npcs=n_eofs, pcscaling=1)

        # Compute EOFs for SO2
        so2_solver = Eof(X['SO2'])
        so2_eofs = so2_solver.eofsAsCorrelation(neofs=n_eofs)
        so2_pcs = so2_solver.pcs(npcs=n_eofs, pcscaling=1)

        # Convert to pandas
        bc_df = bc_pcs.to_dataframe().unstack('mode')
        bc_df.columns = [f"BC_{i}" for i in range(n_eofs)]

        so2_df = so2_pcs.to_dataframe().unstack('mode')
        so2_df.columns = [f"SO2_{i}" for i in range(n_eofs)]

    else:
        # all values are zero, fill up eofs so we have same inputs as for other datasets
        timesteps = len(X.time)
        zeros = np.zeros(shape=(timesteps, n_eofs))
        bc_df = pd.DataFrame(zeros, columns=[f"BC_{i}" for i in range(n_eofs)], index=X["BC"].coords['time'].data)
        so2_df = pd.DataFrame(zeros, columns=[f"SO2_{i}" for i in range(n_eofs)], index=X["BC"].coords['time'].data)

    # Bring the emissions data back together again and normalise
    inputs = pd.DataFrame({
        "CO2": X["CO2"].data,
        "CH4": X["CH4"].data
    }, index=X["CO2"].coords['time'].data)

    # Combine with aerosol EOFs
    inputs = pd.concat([inputs, bc_df, so2_df], axis=1)
    return inputs


def create_predictdand_data(data_set):
    Y = xr.open_dataset(data_path + "outputs_{}.nc".format(data_set)).mean("member")
    # Convert the precip values to mm/day
    Y["pr"] *= 86400
    Y["pr90"] *= 86400
    return Y


def get_rmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.lat))
    return np.sqrt(((truth - pred)**2).weighted(weights).mean(['lat', 'lon'])).data
