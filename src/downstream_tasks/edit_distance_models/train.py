import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from src.downstream_tasks.edit_distance_models.mlp import MLP
from src.downstream_tasks.edit_distance_models.cnn1d import CNN1D
from src.downstream_tasks.edit_distance_models.distance_datasets import (
    DistanceDataset,
    SampledDistanceDataset,
)

AVAILABLE_MODELS = {
    "mlp": MLP,
    "cnn1d": CNN1D,
}


def train(
    train_df,
    train_dist,
    val_df,
    val_dist,
    test_df,
    test_dist,
    embedding_layers_list,
    args: dict,
):
    """
    Trains a model to estimate pairwise sequence distances using provided training, validation, and test data.
    EDIT DISTANCE APPROXIMATION task

    Parameters:
    -----------
    train_df : pd.DataFrame
        DataFrame containing training data.
    train_dist : np.array
        Pairwise distances for training data.
    val_df : pd.DataFrame
        DataFrame containing validation data.
    val_dist : np.array
        Pairwise distances for validation data.
    test_df : pd.DataFrame
        DataFrame containing test data.
    test_dist : np.array
        Pairwise distances for test data.
    embedding_layers_list : list[nn.Module]
        List of embedding layers to be used in the model.
    args : dict
        Dictionary containing hyperparameters and other training configurations.

    Returns:
    --------
    model : pl.LightningModule
        The trained model.
    """
    epochs = args["epochs"]
    check_val_every_n_epoch = args["check_val_every_n_epoch"]
    batch_size = args["batch_size"]
    ks_str = args["representation_k"]
    accelerator = args["accelerator"]
    representation_size = args["representation_size"]
    representation_method = args["representation_method"]
    model_class = AVAILABLE_MODELS[args["model_class"]]
    log_dir = args["wandb_dir"]

    ## datasets

    train_dataset = SampledDistanceDataset(df=train_df, dist=train_dist, ks_str=ks_str)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataset = DistanceDataset(val_df, val_dist, ks_str=ks_str)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = DistanceDataset(test_df, test_dist, ks_str=ks_str)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    input_embedding_size = [
        representation_size * l for l in train_dataset.max_seq_lengths
    ]

    ##callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    ## logging

    trainer = pl.Trainer(
        default_root_dir=log_dir,
        max_epochs=epochs,
        accelerator=accelerator,
        check_val_every_n_epoch=check_val_every_n_epoch,
        devices=1,
        callbacks=[early_stop, checkpoint_callback, TQDMProgressBar(refresh_rate=1000)],
        # gradient_clip_val=1.0,
    )

    model = model_class(
        embedding_layers_list=embedding_layers_list,
        input_embedding_sizes_list=input_embedding_size,
        args=args,
        distance=args["distance_type"],
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    model = model_class.load_from_checkpoint(
        checkpoint_callback.best_model_path,  #### after that other arguments
        embedding_layers_list=embedding_layers_list,
        input_embedding_sizes_list=input_embedding_size,
        args=args,
    )
    losses = trainer.test(model, test_dataloader)
    wandb.log(losses[0])

    return model
