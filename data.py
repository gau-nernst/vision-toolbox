import torch
import torch.distributed
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
import webdataset as wds


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str = None,
        val_dir: str = None,
        batch_size: int = 128,
        num_workers: int = 4,
        train_crop_size: int = 176,
        val_resize_size: int = 232,
        val_crop_size: int = 224,
        webdataset: bool = False,
        train_size: int = 0,
        val_size: int = 0,
        random_erasing_p: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _get_dataset(self, data_dir, transform, training):
        if self.hparams.webdataset:
            # https://webdataset.github.io/webdataset/multinode/
            # https://github.com/webdataset/webdataset-lightning/blob/main/train.py
            ds = (
                wds.WebDataset(data_dir, shardshuffle=training)
                .shuffle(1000 if training else 0)
                .decode("pil")
                .to_tuple("jpg;jpeg;png cls")
                .map_tuple(transform, lambda x: x)
            )
        else:
            ds = torchvision.datasets.ImageFolder(data_dir, transform=transform)
        return ds

    def setup(self, stage=None):
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(self.hparams.train_crop_size),
                T.RandomHorizontalFlip(),
                T.TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.RandomErasing(p=self.hparams.random_erasing_p, value="random"),
            ]
        )
        val_transform = T.Compose(
            [
                T.Resize(self.hparams.val_resize_size),
                T.CenterCrop(self.hparams.val_crop_size),
                T.ToTensor(),
            ]
        )
        self.train_ds = self._get_dataset(self.hparams.train_dir, train_transform, True)
        self.val_ds = self._get_dataset(self.hparams.val_dir, val_transform, False)

    def _get_dataloader(self, ds, pin_memory=True, training=False):
        batch_size = self.hparams.batch_size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            batch_size //= torch.distributed.get_world_size()

        if self.hparams.webdataset:
            dataloader = wds.WebLoader(
                ds.batched(batch_size, partial=not training),
                batch_size=None,
                num_workers=self.hparams.num_workers,
                pin_memory=pin_memory,
            )
            if training:
                num_batches = self.hparams.train_size // batch_size
                dataloader = dataloader.ddp_equalize(num_batches)
        else:
            dataloader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=training,
                num_workers=self.hparams.num_workers,
                pin_memory=pin_memory,
            )
        return dataloader

    def train_dataloader(self):
        return self._get_dataloader(self.train_ds, training=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_ds, training=False)
