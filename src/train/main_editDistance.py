import wandb
import hashlib

from src.utils import set_all_seeds
from src.representations.representations_factory import RepresentationFactory
from src.downstream_tasks.datasets_factory.edit_distance import (
    edit_distance_benchmark,
    retrieval_benchmark,
)
from src.downstream_tasks.edit_distance_models.train import train
from src.downstream_tasks.edit_distance_models.zero_shot_model import (
    zero_shot_model,
)
from src.downstream_tasks.edit_distance_models.retrieval_test import retrieval_test
from src.train.parsers import edit_distance_parser, convert_string_args_to_list


def main_editDistance(args, optuna_training=False):
    """
    Main function for the Edit Distance Aproximation and Closest String Retrieval tasks.
    It initializes the Weights and Biases run, loads data, computes representations,
    trains models, and performs retrieval testing.

    Parameters:
    - args (Namespace): Parsed command-line arguments containing configuration parameters and hyperparameters.
    - optuna_training (bool, optional): Flag indicating if the function is being called as part of an Optuna optimization trial. Default is False.

    Returns:
    - float or None: The optimization goal value (objective value) if `optuna_training` is True; otherwise None.
    """
    group_id = generate_group_id(args)
    run = wandb.init(
        project=f"editDistance__{args.data.split('/')[-1]}",
        dir=args.wandb_dir,
        name=args.representation_method,
        config=args,
        group=group_id,
        mode="offline",
    )
    run.define_metric("best_val_loss", summary="min,last")
    (
        train_df,
        train_dist,
        val_df,
        val_dist,
        test_df,
        test_dist,
    ) = edit_distance_benchmark(ds_path=args.data)

    set_all_seeds(int(args.seed))

    dfs_list_inference = []
    if args.retrieval_data:
        references_df, queries_df = retrieval_benchmark(ds_path=args.retrieval_data)
        dfs_list_inference.append(references_df)
        dfs_list_inference.append(queries_df)

    representations = RepresentationFactory(method=args.representation_method)

    if args.include_val_test_unsupervised:
        dfs_list_train = [train_df, val_df, test_df]
    else:
        dfs_list_train = [train_df, val_df]
        dfs_list_inference = [*dfs_list_inference, test_df]

    dfs_list_train = representations.fit_transform(dfs_list_train, args=vars(args))

    dfs_list_inference = representations.transform(dfs_list_inference, args=vars(args))

    embedding_layers_list = representations.layers_list

    if args.include_val_test_unsupervised:
        train_df, val_df, test_df = dfs_list_train
    else:
        train_df = dfs_list_train[0]
        test_df = dfs_list_inference.pop()
        val_df = dfs_list_train[1]
    if args.retrieval_data:
        references_df, queries_df = dfs_list_inference[:2]

    models = []
    if args.zero_shot_method:
        for method in args.zero_shot_method.split(","):
            model = zero_shot_model(
                embedding_layers_list,
                method,
            )
            models.append(model)
    else:
        model = train(
            train_df,
            train_dist,
            val_df,
            val_dist,
            test_df,
            test_dist,
            embedding_layers_list,
            args=vars(args),
        )
        models.append(model)

    if args.retrieval_data:
        for model in models:
            retrieval_test(
                model,
                references_df,
                queries_df,
                args=vars(args),
                distance=args.distance_type,
            )

    goal = None

    if optuna_training:
        if args.optimization_goal == "val_loss":
            goal = run.summary["best_val_loss"].min
        else:
            goal = run.summary[args.optimization_goal]

    wandb.finish()

    return goal


def generate_group_id(args):
    """
    Generates a unique group ID based on the provided arguments.
    The group ID is determined by hashing a string representation of the arguments (excluding the seed).

    Parameters:
    - args (Namespace): Parsed command-line arguments or configurations.

    Returns:
    - str: A MD5 hash representing the unique group ID.
    """
    params = vars(args).copy()
    params.pop("seed", None)
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    group_id = hashlib.md5(param_str.encode()).hexdigest()
    return group_id


if __name__ == "__main__":
    """
    Script entry point.

    Usage:
    python main_editDistance.py [ARGUMENTS]

    See 'edit_distance_parser' in parsers.py for available arguments and their details.
    """
    parser = edit_distance_parser()
    args = parser.parse_args()
    args = convert_string_args_to_list(args)

    main_editDistance(args)
