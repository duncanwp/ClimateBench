#!/usr/bin/env python
"""
A simple GPFlow based GP regression model
"""
from pathlib import Path
import click
import pickle
import os
import numpy as np

from esem import gp_model

from utils import create_test_data, create_predictor_data, create_predictdand_data, get_mean_rmse


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "job-dir",
    type=click.Path(),
    nargs=1,
    default=Path.cwd(),
)
def setup(job_dir):
    """
        Download and unzip all the ClimateBench data to the current working directory, or another specified location
    """
    from utils import get_data

    get_data(job_dir, "train_val.tar.gz")
    get_data(job_dir, "test.tar.gz")

    print("All files downloaded and extracted")


@cli.command()
@click.argument(
    "experiments",
    type=str,
    required=True,
    nargs=-1,
)
@click.option(
    "--variable",
    type=str,
    required=True,
    help="ClimateBench variable to use. E.g., 'tas'",
)
@click.option(
    "--job-dir",
    type=click.Path(),
    default=Path.cwd(),
    help="location for writing checkpoints and results",
)
@click.option("--n_estimators", default=200, type=int, help="number of estimators, default=200")
@click.option("--min_samples_split", default=10, type=int, help="number of trials, default=10")
@click.option("--min_samples_leaf", default=1, type=int, help="number of trials, default=1")
@click.option("--max_depth", default=1, type=int, help="number of trials, default=1")
@click.option("--verbose/--no-verbose", default=False, help="verbosity default=False")
@click.option("--RSCV/--no-RSCV", default=True, help="RandomizedSearchCV default=True")
@click.option("--bootstrap/--no-bootstrap", default=True, help="Bootstrap default=True")
@click.option(
    "--seed", default=1331, type=int, help="random number generator seed, default=1331",
)
def train(
    experiments, variable, job_dir, n_estimators, min_samples_split, min_samples_leaf, max_depth, verbose,
        rscv, bootstrap, seed
):
    """
        Train a GP model against a set of ClimateBench EXPERIMENT(s). E.g., `historical ssp126
        ssp370`.",
    """
    # Create training and testing arrays
    X, solvers = create_predictor_data(job_dir, experiments)
    Y = create_predictdand_data(job_dir, experiments)

    #TODO: This should pickle the ESEm model and parameters then recreate the GPFLOW and reload the params
    # See: https://gpflow.github.io/GPflow/2.5.1/notebooks/intro_to_gpflow2.html#Copying-(hyper)parameter-values-between-models
    # The CNN model can just use model.save and tf.keras.models.load_model
    gp = gp_model(X, Y[variable], **kwargs)

    gp.train(verbose=verbose)

    if rscv:
        print(gp.model.model.best_params_)

    with open(job_dir + Path(f"gp_model_{variable}.pkl"), "wb") as f:
        pickle.dump(gp, f)

    with open(job_dir + Path(f"gp_model_eofs_{variable}.pkl"), "wb") as f:
        pickle.dump(solvers, f)


@cli.command()
@click.argument(
    "experiment",
    type=str,
    required=True,
    nargs='1',
)
@click.option(
    "--variable",
    type=str,
    required=True,
    nargs='1',
    help="Model variable to evaluate. E.g., `tas`.",
)
@click.option(
    "--output",
    type=click.File(),
    help="Filename for prediction outputs",
)
@click.option(
    "--job-dir",
    type=click.Path(),
    default=Path.cwd(),
    help="Location for writing predictions",
)
@click.option("--overwrite", default=False, type=bool, help="Overwrite output file default=False")
def evaluate(
    experiment, variable, output, job_dir, overwrite
):
    """
        Evaluate a particular ClimateBench EXPERIMENT. E.g., `ssp245`.
    """

    if output is None:
        output = f"gp_model_prediction_{variable}.nc"

    if os.path.isfile(output) and not overwrite:
        raise IOError("Output file ({output}) already exists, quitting. Set --overwrite to overwrite.")

    with open(job_dir + Path(f"gp_model_{variable}.pkl"), "wb") as f:
        gp = pickle.load(f)

    with open(job_dir + Path(f"gp_model_eofs_{variable}.pkl"), "wb") as f:
        solvers = pickle.load(f)

    X_test = create_test_data(job_dir, experiment, solvers)
    Y_test = create_predictdand_data(job_dir, [experiment])

    m_out, _ = gp.predict(X_test)

    m_out.assign_coords(time=m_out.sample + 2014)

    # save output to netcdf
    m_out.to_netcdf(job_dir + output, 'w')

    print(f"{variable} RMSE: {get_mean_rmse(Y_test[variable], m_out)}")


if __name__ == '__main__':
    cli()
