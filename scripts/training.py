import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from cnn_model import CNNModel
from data_module import FashionMNISTDataModule

def train_on_chunk(chunk_id, max_epochs=1):
    print(f"Starting training on chunk {chunk_id}")

    model = CNNModel()
    data_module = FashionMNISTDataModule()
    data_module.prepare_data()
    data_module.setup()

    logger = TensorBoardLogger('logs', name=f'fashionmnist_experiment_chunk_{chunk_id}')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/chunk_{chunk_id}',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=[0],
        accelerator='gpu',
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, data_module)
    print(f"Training completed on chunk {chunk_id}")

    # Create a flag file to indicate completion
    open(f'training_complete_chunk_{chunk_id}.flag', 'w').close()
