import argparse
import yaml
import optuna
import wandb
from src.train.main_editDistance import main_editDistance
from src.train.main_geneFinder import main_geneFinder
from src.train.parsers import (
    edit_distance_parser,
    coding_metagenomics_parser,
    assert_arguments_correctness,
    convert_string_args_to_list,
)


def training_function(
    trial: optuna.Trial,
    search_space: dict,
    optimisation_goal: dict,
    downstream_task: str,
):
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters:
    - trial (optuna.Trial): The trial object.
    - search_space (dict): The hyperparameter search space.
    - optimisation_goal (dict): The optimization goal.
    - downstream_task (str): The specific task ("edit_distance" or "gene_finder").

    Returns:
    - float: The optimization goal (objective value) for the given hyperparameters.

    Raises:
    - ValueError: If the `downstream_task` is not recognized.
    - optuna.exceptions.TrialPruned: If the trial encounters an exception.
    """

    # Get default argument values
    if downstream_task == "edit_distance":
        parser = edit_distance_parser()
    elif downstream_task == "gene_finder":
        parser = coding_metagenomics_parser()
    else:
        raise ValueError(f"Invalid downstream task: {downstream_task}")
    args = vars(parser.parse_args([]))

    # Get trial parameters as a dictionary
    for param_name, param_values in search_space.items():
        if isinstance(param_values[0], int):
            args[param_name] = trial.suggest_int(
                param_name, min(param_values), max(param_values)
            )
        elif isinstance(param_values[0], float):
            args[param_name] = trial.suggest_float(
                param_name, min(param_values), max(param_values)
            )
        elif isinstance(param_values[0], str):
            args[param_name] = trial.suggest_categorical(param_name, param_values)
        elif isinstance(param_values[0], bool):
            args[param_name] = trial.suggest_categorical(param_name, param_values)

    try:
        args = argparse.Namespace(**args)
        assert_arguments_correctness(args)
        args = convert_string_args_to_list(args)
        args.optimization_goal = optimisation_goal

        if downstream_task == "edit_distance":
            goal = main_editDistance(args, optuna_training=True)
        elif downstream_task == "gene_finder":
            goal = main_geneFinder(args, optuna_training=True)
        else:
            raise ValueError(f"Invalid downstream task: {downstream_task}")

        return goal

    except Exception as e:
        wandb.finish()
        print(f"Encountered exception: {e}")
        raise optuna.exceptions.TrialPruned()


def main(args_main):
    """
    Main function for running the hyperparameter optimization.

    Parameters:
    - args_main (Namespace): Parsed command-line arguments containing configuration file path, downstream task, and number of trials.

    Raises:
    - ValueError: If the optimization method specified in the configuration is not recognized.
    """

    with open(args_main.config_file) as file:
        config = yaml.safe_load(file)

    # Extract the direction, method, and parameters from the config
    direction = config.pop("direction")
    method = config.pop("method")

    search_space = {k: v["values"] for k, v in config["parameters"].items()}

    # Create the sampler based on the method
    if method == "grid":
        sampler = optuna.samplers.GridSampler(search_space)
    elif method == "random":
        sampler = optuna.samplers.RandomSampler(search_space)
    elif method == "bo":
        sampler = optuna.samplers.TPESampler(search_space)
    else:
        raise ValueError(f"Invalid method: {method}")

    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(
        lambda trial: training_function(
            trial, search_space, config["optimization_goal"], args_main.downstream_task
        ),
        n_trials=args_main.n_trials,
    )
    best_trial = study.best_trial

    print(f"Best Value: {best_trial.value}")
    print("Best hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    """
    Script entry point.

    Invokes the hyperparameter optimization process based on the provided YAML configuration file.

    Usage:
    python param_search_optuna.py --config_file PATH_TO_CONFIG --downstream_task TASK_NAME [--n_trials NUMBER_OF_TRIALS]

    Arguments:
    --config_file: Path to the YAML configuration file detailing the search space, optimization goal, and method.
    --downstream_task: Name of the downstream task:
      "edit_distance" for Edit distance approximation and Closest String Retrieval or "gene_finder" for Gene Prediction.
    --n_trials (optional): Number of trials for optuna optimization. Default is 10000.
    """

    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument(
        "--config_file", type=str, help="Path to the yaml configuration file"
    )
    parser.add_argument(
        "--downstream_task", type=str, help="Path to the yaml configuration file"
    )
    parser.add_argument(
        "--n_trials", type=int, help="Number of trials to run", default=10000
    )
    args_main = parser.parse_args()

    main(args_main)
