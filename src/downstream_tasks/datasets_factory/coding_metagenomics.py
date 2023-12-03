import pandas as pd

LABEL_ENCODER = {"yes": 1, "no": 0}


def load_all_coding_datasets(datasets_folder: str, max_sequence_length=600):
    """
    Load all coding datasets related to the GENE PREDICTION task.

    This function reads multiple datasets from the specified folder path
    and processes them to ensure they meet specific criteria for the GENE PREDICTION task.

    Parameters:
        datasets_folder (str): The directory path containing the datasets.
        max_sequence_length (int, optional): The maximum sequence length for filtering. Default is 600.

    Returns:
        tuple: A tuple containing dataframes for each dataset (training_1, training_2, validation,
               test_1, test_2_low, test_2_medium, test_2_high).
    """
    training_1_df = coding_regions_generic_dataset(
        f"{datasets_folder}/training1/groundtruth.csv"
    ).rename(columns={"sequences": "sequence"})
    training_2_df = coding_regions_generic_dataset(
        f"{datasets_folder}/training2/groundtruth.csv"
    ).rename(columns={"sequences": "sequence"})
    validation_df = coding_regions_generic_dataset(
        f"{datasets_folder}/validation/groundtruth.csv"
    ).rename(columns={"sequences": "sequence"})
    test_1_df = coding_regions_generic_dataset(
        f"{datasets_folder}/test1/groundtruth.csv"
    ).rename(columns={"sequences": "sequence"})
    test_2_low_df = coding_regions_generic_dataset(
        f"{datasets_folder}/test2low/groundtruth.csv"
    ).rename(columns={"sequences": "sequence"})
    test_2_medium_df = coding_regions_generic_dataset(
        f"{datasets_folder}/test2medium/groundtruth.csv"
    ).rename(columns={"sequences": "sequence"})
    test_2_high_df = coding_regions_generic_dataset(
        f"{datasets_folder}/test2high1/groundtruth.csv"
    )
    test_2_high_df = pd.concat(
        (
            test_2_high_df,
            coding_regions_generic_dataset(
                f"{datasets_folder}/test2high2/groundtruth.csv"
            ),
        )
    )
    test_2_high_df = pd.concat(
        (
            test_2_high_df,
            coding_regions_generic_dataset(
                f"{datasets_folder}/test2high3/groundtruth.csv"
            ),
        )
    ).rename(columns={"sequences": "sequence"})

    training_1_df = filter_dataframe(training_1_df, "training_1", max_sequence_length)
    training_2_df = filter_dataframe(training_2_df, "training_2", max_sequence_length)
    validation_df = filter_dataframe(validation_df, "validation", max_sequence_length)
    test_1_df = filter_dataframe(test_1_df, "test_1", max_sequence_length)
    test_2_low_df = filter_dataframe(test_2_low_df, "test_2_low", max_sequence_length)
    test_2_medium_df = filter_dataframe(
        test_2_medium_df, "test_2_medium", max_sequence_length
    )
    test_2_high_df = filter_dataframe(
        test_2_high_df, "test_2_high", max_sequence_length
    )

    return (
        training_1_df,
        training_2_df,
        validation_df,
        test_1_df,
        test_2_low_df,
        test_2_medium_df,
        test_2_high_df,
    )


def coding_regions_generic_dataset(
    df_path: str,
):
    """
    Loads and preprocesses a single dataset.

    Reads the dataset from a CSV file and encodes the class labels.

    Parameters:
        df_path (str): The path to the dataset CSV file.

    Returns:
        DataFrame: A processed dataframe with encoded class labels.
    """
    df = pd.read_csv(df_path, delimiter=";", index_col=0)

    def encode(l):
        return LABEL_ENCODER[l]

    df["class"] = df["class"].apply(encode)
    return df


def filter_dataframe(df, name: str, max_sequence_length: int):
    """
    Filters and pads sequences in the dataframe.

    The function removes sequences containing characters other than ACTG and pads or trims
    sequences to the specified maximum length. It also prints the proportion of the dataset
    retained after filtering.

    Parameters:
        df (DataFrame): The dataframe containing sequence data.
        name (str): The name or identifier of the dataset (used for printing purposes).
        max_sequence_length (int): The maximum sequence length for filtering and padding.

    Returns:
        DataFrame: The filtered and padded dataframe.
    """
    df = df[~df["sequence"].str.contains("[^ACTG]", na=False)]
    if max_sequence_length:
        length = df.shape[0]
        mask = df["sequence"].str.len() <= max_sequence_length
        df = df.loc[mask]
        new_length = df.shape[0]

    df["sequence"] = df["sequence"].str.pad(
        width=max_sequence_length, side="right", fillchar="N"
    )

    print("Proportion of ", name, " dataset left: ", new_length / length)

    return df
