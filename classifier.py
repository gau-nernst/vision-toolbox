from functools import partial
from typing import Callable, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
import webdataset as wds
try:
    import timm.optim as timm_optim
except ImportError:
    timm_optim = None

from vision_toolbox import backbones
from extras import RandomCutMixMixUp

# https://github.com/pytorch/vision/blob/main/references/classification/train.py
# https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/

_optimizers = {
    "SGD": partial(torch.optim.SGD, momentum=0.9),
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RMSprop": partial(torch.optim.RMSprop, momentum=0.9)
}
if timm_optim is not None:
    _timm_optimizers = {
        "Lamb": timm_optim.Lamb,
        "Lars": timm_optim.Lars
    }
    _optimizers.update(_timm_optimizers)


def image_loader(path, device='cpu'):
    img = torchvision.io.read_file(path)
    img = torchvision.io.decode_jpeg(img, device=device)
    return img


class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        # model
        backbone: Union[str, backbones.BaseBackbone],
        num_classes: int,
        include_pool: bool=True,
        
        # data
        train_dir: str=None,
        val_dir: str=None,
        batch_size: int=128,
        num_workers: int=4,
        train_crop_size: int=176,
        val_resize_size: int=232,
        val_crop_size: int=224,
        
        # augmentation
        random_erasing_p: float=0.1,
        mixup_alpha: float=0.2,
        cutmix_alpha: float=1.0,
        
        # optimizer and scheduler
        optimizer: str="SGD",
        lr: float=0.05,
        weight_decay: float=2e-5,
        norm_weight_decay: float=0,
        label_smoothing: float=0.1,
        warmup_epochs: int=5,
        warmup_decay: float=0.01,

        # others
        jit: bool=False,
        channels_last: bool=False,
        webdataset: bool=False,
        train_size: int=0,
        val_size: int=0
        ):
        super().__init__()
        self.save_hyperparameters()
        backbone = backbones.__dict__[backbone]() if isinstance(backbone, str) else backbone
        layers = [backbone]
        if include_pool:
            layers.append(nn.AdaptiveAvgPool2d((1,1)))
            layers.append(nn.Flatten())
        layers.append(nn.Linear(backbone.get_out_channels()[-1], num_classes))
        self.model = nn.Sequential(*layers)

        train_transforms = [
            T.RandomHorizontalFlip(),
            T.autoaugment.TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=(0,0,0), std=(1,1,1)),
        ]
        if random_erasing_p > 0:
            train_transforms.append(T.RandomErasing(p=random_erasing_p, value="random"))
        
        self.train_transforms = nn.Sequential(*train_transforms)
        self.mixup_cutmix = RandomCutMixMixUp(num_classes, cutmix_alpha, mixup_alpha) if cutmix_alpha > 0 and mixup_alpha > 0 else None

        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if jit:
            self.model = torch.jit.script(self.model)

    def get_dataloader(self, transform, training=False, pin_memory=True):
        data_dir = self.hparams.train_dir if training else self.hparams.val_dir
        if self.hparams.webdataset:
            # https://webdataset.github.io/webdataset/multinode/
            # https://github.com/webdataset/webdataset-lightning/blob/main/train.py
            ds = (
                wds.WebDataset(data_dir, shardshuffle=training)
                .shuffle(1000 if training else 0)
                .decode("pil")
                .to_tuple("jpg;jpeg;png cls")
                .map_tuple(transform, lambda x: x)
                .batched(self.hparams.batch_size, partial=not training)
            )
            dataloader = wds.WebLoader(ds, batch_size=None, num_workers=self.hparams.num_workers, pin_memory=pin_memory)
            if training:
                dataloader = dataloader.ddp_equalize(self.hparams.train_size//self.hparams.batch_size)
        else:
            ds = torchvision.datasets.ImageFolder(data_dir, transform=transform)
            dataloader = DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=training, num_workers=self.hparams.num_workers, pin_memory=pin_memory)
        return dataloader

    def train_dataloader(self):
        transform = T.Compose([
            T.RandomResizedCrop(self.hparams.train_crop_size),
            T.PILToTensor()
        ])
        return self.get_dataloader(transform, training=True)

    def val_dataloader(self):
        transform = T.Compose([
            T.Resize(self.hparams.val_resize_size),
            T.CenterCrop(self.hparams.val_crop_size),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=(0,0,0), std=(1,1,1))
        ])
        return self.get_dataloader(transform, training=False)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = self.train_transforms(images)
        
        if self.mixup_cutmix is not None:
            images, labels = self.mixup_cutmix(images, labels)
        if self.hparams.channels_last:
            images = images.to(memory_format=torch.channels_last)

        logits = self.model(images)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.hparams.label_smoothing)
        self.log("train/loss", loss, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        if self.hparams.channels_last:
            images = images.to(memory_format=torch.channels_last)

        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        self.log("val/loss", loss, sync_dist=True)

        preds = torch.argmax(logits, dim=-1)
        correct = (labels == preds).sum()
        acc = correct / labels.numel()
        self.log("val/acc", acc, sync_dist=True)

    def configure_optimizers(self):
        if self.hparams.norm_weight_decay is not None:
            # https://github.com/pytorch/vision/blob/main/torchvision/ops/_utils.py
            norm_classes = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm)
            
            norm_params = []
            other_params = []
            for module in self.modules():
                if next(module.children(), None):
                    other_params.extend(p for p in module.parameters(recurse=False) if p.requires_grad)
                elif isinstance(module, norm_classes):
                    norm_params.extend(p for p in module.parameters() if p.requires_grad)
                else:
                    other_params.extend(p for p in module.parameters() if p.requires_grad)

            param_groups = (norm_params, other_params)
            wd_groups = (self.hparams.norm_weight_decay, self.hparams.weight_decay)
            parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

        else:
            parameters = self.parameters()

        optimizer = _optimizers[self.hparams.optimizer](parameters, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs-self.hparams.warmup_epochs)
        if self.hparams.warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=self.hparams.warmup_decay, total_iters=self.hparams.warmup_epochs)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[self.hparams.warmup_epochs])
            
            # https://github.com/pytorch/pytorch/issues/67318
            if not hasattr(lr_scheduler, "optimizer"):
                setattr(lr_scheduler, "optimizer", optimizer)

        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler
        }
