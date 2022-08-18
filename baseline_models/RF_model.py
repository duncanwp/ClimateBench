#!/usr/bin/env python
"""
A simple sklearn based RandomForest regression model
"""
from pathlib import Path
import click
import pickle
import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from esem import rf_model

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


@cli.command("train")
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
    type=click.Path(path_type=Path),
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
        Train a RF model against a set of ClimateBench EXPERIMENT(s). E.g., `historical ssp126
        ssp370`.",
    """
    # Create training and testing arrays
    X, solvers = create_predictor_data(job_dir, experiments)
    Y = create_predictdand_data(job_dir, experiments)

    rf = rf_model(X, Y[variable], random_state=seed, bootstrap=bootstrap, max_features='auto',
                  n_estimators=n_estimators, min_samples_split=min_samples_split,
                  min_samples_leaf=min_samples_leaf, max_depth=max_depth)

    if rscv:
        rf_random = RandomizedSearchCV(estimator=rf.model.model, param_distributions=get_random_grid(),
                                       n_iter=29, cv=3, verbose=int(verbose)+1, n_jobs=-1)
        rf.model.model = rf_random

    rf.train(verbose=verbose)

    if rscv:
        print(rf.model.model.best_params_)

    #TODO: This still isn't working even with tf 2.8.0
    # I should use the same approach as for the GP and just pickle the parameters and load them again
    # This has the advantage that I don't pickle all the training data too. It's probably a better pattern for ESEm
    # if I can figure it out
    with open(job_dir / f"rf_model_{variable}.pkl", "wb") as f:
        pickle.dump(rf, f)

    with open(job_dir / f"rf_model_eofs_{variable}.pkl", "wb") as f:
        pickle.dump(solvers, f)


def get_random_grid():

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=300, num=5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 55, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10, 15, 25]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [4, 8, 12]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    return {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap}


@cli.command("evaluate")
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
        output = f"rf_model_prediction_{variable}.nc"

    if os.path.isfile(output) and not overwrite:
        raise IOError("Output file ({output}) already exists, quitting. Set --overwrite to overwrite.")

    with open(job_dir / f"rf_model_{variable}.pkl", "wb") as f:
        rf = pickle.load(f)

    with open(job_dir / f"rf_model_eofs_{variable}.pkl", "wb") as f:
        solvers = pickle.load(f)

    X_test = create_test_data(job_dir, experiment, solvers)
    Y_test = create_predictdand_data(job_dir, [experiment])

    m_out, _ = rf.predict(X_test)

    m_out.assign_coords(time=m_out.sample + 2014)

    # save output to netcdf
    m_out.to_netcdf(job_dir + output, 'w')

    print(f"{variable} RMSE: {get_mean_rmse(Y_test[variable], m_out)}")


if __name__ == '__main__':
    cli()
