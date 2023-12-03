import wandb
import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split


from src.utils import set_all_seeds
from src.representations.representations_factory import RepresentationFactory
from src.downstream_tasks.datasets_factory.coding_metagenomics import (
    load_all_coding_datasets,
)
from src.downstream_tasks.coding_metagenomics.train import train
from src.downstream_tasks.coding_metagenomics.coding_datasets import create_dataloaders
from src.train.parsers import coding_metagenomics_parser, convert_string_args_to_list


def main_geneFinder(args, optuna_training=False):
    """
    Executes the main workflow for the Gene Prediction task,
    encompassing data loading, representation learning, model training, and evaluation.

    Parameters:
    - args (Namespace): Parsed command-line arguments or configurations containing various parameters like data paths, representation methods, and model hyperparameters.
    - optuna_training (bool, optional): Indicates if the function is being used within an Optuna hyperparameter optimization trial. Defaults to False.

    Returns:
    - float or None: The optimization goal if `optuna_training` is True, otherwise None.
    """

    group_id = generate_group_id(args)
    run = wandb.init(
        dir=args.wandb_dir,
        name=args.representation_method,
        project=f"geneFinder__{args.data.split('/')[-1]}__{args.data_max_sequence_length}__{args.proportion_samples_to_keep_supervised_training}",
        config=args,
        group=group_id,
        mode="offline",
    )
    run.define_metric("val_loss", summary="min")
    run.define_metric("val_acc", summary="max")
    run.define_metric("val_auroc", summary="max")
    set_all_seeds(int(args.seed))

    (
        training_1_df,
        training_2_df,
        validation_df,
        test_1_df,
        test_2_low_df,
        test_2_medium_df,
        test_2_high_df,
    ) = load_all_coding_datasets(
        datasets_folder=args.data,
        max_sequence_length=args.data_max_sequence_length,
    )
    if args.include_training_1:
        training_df = pd.concat([training_1_df, training_2_df])
    else:
        training_df = training_2_df

    if args.only_CAMI_dataset:

        def split_dataframe(df, train_size=0.25, val_size=0.25):
            train, temp = train_test_split(
                df, test_size=1 - train_size, random_state=42
            )
            test, val = train_test_split(
                temp, test_size=val_size / (1 - train_size), random_state=42
            )
            return train, val, test

        train_low, val_low, test_2_low_df = split_dataframe(test_2_low_df)
        train_medium, val_medium, test_2_medium_df = split_dataframe(test_2_medium_df)
        train_high, val_high, test_2_high_df = split_dataframe(test_2_high_df)

        training_df = pd.concat([train_low, train_medium, train_high])
        validation_df = pd.concat([val_low, val_medium, val_high])

    representations = RepresentationFactory(method=args.representation_method)

    if args.include_val_test_unsupervised:
        dfs_list = representations.fit_transform(
            dfs_list=[
                training_df,
                validation_df,
                test_1_df,
                test_2_low_df,
                test_2_medium_df,
                test_2_high_df,
            ],
            args=vars(args),
        )
        (
            training_df,
            validation_df,
            test_1_df,
            test_2_low_df,
            test_2_medium_df,
            test_2_high_df,
        ) = dfs_list
    else:
        training_df = representations.fit_transform(
            dfs_list=[training_df], args=vars(args)
        )[0]
        (
            validation_df,
            test_1_df,
            test_2_low_df,
            test_2_medium_df,
            test_2_high_df,
        ) = representations.transform(
            [validation_df, test_1_df, test_2_low_df, test_2_medium_df, test_2_high_df],
            args=vars(args),
        )
    embedding_layers_list = representations.layers_list

    print("Size of entire training set:", len(training_df.index))
    training_df = training_df.sample(
        frac=args.proportion_samples_to_keep_supervised_training
    ).reset_index(drop=True)
    print("Size of supervised training set:", len(training_df.index))
    validation_df = validation_df.sample(frac=1).reset_index(drop=True)
    test_1_df = test_1_df.sample(frac=1).reset_index(drop=True)
    test_2_low_df = test_2_low_df.sample(frac=1).reset_index(drop=True)
    test_2_medium_df = test_2_medium_df.sample(frac=1).reset_index(drop=True)
    test_2_high_df = test_2_high_df.sample(frac=1).reset_index(drop=True)

    ## create datasets
    (
        train_dataloader,
        val_dataloader,
        test_1_dataloader,
        test_2_low_dataloader,
        test_2_medium_dataloader,
        test_2_high_dataloader,
        test_oveall_dataloader,
        input_embedding_size,
    ) = create_dataloaders(
        training_df,
        validation_df,
        test_1_df,
        test_2_low_df,
        test_2_medium_df,
        test_2_high_df,
        vars(args),
    )

    del training_df
    del validation_df
    del test_1_df
    del test_2_low_df
    del test_2_medium_df
    del test_2_high_df

    model = train(
        train_dataloader,
        val_dataloader,
        test_1_dataloader,
        test_2_low_dataloader,
        test_2_medium_dataloader,
        test_2_high_dataloader,
        test_oveall_dataloader,
        input_embedding_size,
        embedding_layers_list,
        args=vars(args),
    )

    goal = None
    if optuna_training:
        goal = run.summary[args.optimization_goal].max

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
    python main_geneFinder.py [ARGUMENTS]

    See 'coding_metagenomics_parser' in parsers.py for available arguments and their details.
    """
    parser = coding_metagenomics_parser()
    args = parser.parse_args()
    args = convert_string_args_to_list(args)

    main_geneFinder(args)
