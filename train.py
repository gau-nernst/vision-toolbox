from pytorch_lightning.utilities.cli import LightningCLI

from classifier import ImageClassifier
from data import ImageDataModule


if __name__ == "__main__":
    cli = LightningCLI(
        ImageClassifier,
        ImageDataModule,
        save_config_overwrite=True,
        save_config_filename="saved_config.yaml"
    )
