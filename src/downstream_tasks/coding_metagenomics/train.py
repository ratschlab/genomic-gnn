import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.downstream_tasks.coding_metagenomics.cnn1d import CNN1D

AVAILABLE_MODELS = {"cnn1d": CNN1D}


def train(
    train_dataloader,
    val_dataloader,
    test_1_dataloader,
    test_2_low_dataloader,
    test_2_medium_dataloader,
    test_2_high_dataloader,
    test_oveall_dataloader,
    input_embedding_size,
    embedding_layers_list,
    args: dict,
):
    """
    Train a model for the coding metagenomics task.

    Parameters:
    -----------
    - train_dataloader, val_dataloader, test_1_dataloader, test_2_low_dataloader,
      test_2_medium_dataloader, test_2_high_dataloader, test_oveall_dataloader (DataLoader):
        PyTorch DataLoaders for training, validation, and testing datasets.
    - input_embedding_size (list): The input sizes for the embeddings.
    - embedding_layers_list (list): A list of embedding layers.
    - args (dict): Dictionary containing hyperparameters and configurations.

    Returns:
    --------
    - model: Trained model.

    Callbacks:
    ----------
    - EarlyStopping: Stops training early if "val_loss" does not improve for 10 epochs.
    - ModelCheckpoint: Saves the best model based on "val_loss".

    Logging:
    --------
    - WandbLogger: Logs training and validation metrics to Weights & Biases.

    Usage:
    ------
    >>> model = train(train_loader, val_loader, test_loader, input_embedding_size, embedding_layers_list, args)
    """
    epochs = args["epochs"]
    check_val_every_n_epoch = args["check_val_every_n_epoch"]
    accelerator = args["accelerator"]
    model_class = AVAILABLE_MODELS[args["model_class"]]
    log_dir = args["wandb_dir"]

    ##callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    ## logging
    wandb_logger = WandbLogger(save_dir=log_dir)

    trainer = pl.Trainer(
        default_root_dir=log_dir,
        max_epochs=epochs,
        accelerator=accelerator,
        check_val_every_n_epoch=check_val_every_n_epoch,
        devices=1,
        callbacks=[early_stop, checkpoint_callback],
        logger=wandb_logger,
    )

    model = model_class(
        embedding_layers_list=embedding_layers_list,
        input_embedding_sizes_list=input_embedding_size,
        args=args,
    )
    print(model)

    trainer.fit(model, train_dataloader, val_dataloader)

    model = model_class.load_from_checkpoint(
        checkpoint_callback.best_model_path,  #### after that other arguments
        embedding_layers_list=embedding_layers_list,
        input_embedding_sizes_list=input_embedding_size,
        args=args,
    )
    trainer.test(
        model,
        [
            test_1_dataloader,
            test_2_low_dataloader,
            test_2_medium_dataloader,
            test_2_high_dataloader,
            test_oveall_dataloader,
        ],
    )

    return model
