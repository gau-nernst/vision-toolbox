from functools import partial
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torchvision
from torchvision.ops import StochasticDepth
import torchvision.transforms as T
import pytorch_lightning as pl
import timm.optim

from vision_toolbox import backbones
from extras import RandomCutMixMixUp

# https://github.com/pytorch/vision/blob/main/references/classification/train.py
# https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/


def image_loader(path, device="cpu"):
    img = torchvision.io.read_file(path)
    img = torchvision.io.decode_jpeg(img, device=device)
    return img


class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        # model
        backbone: Union[str, backbones.BaseBackbone],
        num_classes: int,
        include_pool: bool = True,
        # augmentation and regularization
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        # regularization
        weight_decay: float = 2e-5,
        norm_weight_decay: float = 0,
        bias_weight_decay: float = 0,
        label_smoothing: float = 0.1,
        drop_out: float = None,
        drop_path: float = None,
        # optimizer and scheduler
        optimizer: str = "SGD",
        momentum: float = 0.9,
        lr: float = 0.05,
        decay_factor: float = 0,
        warmup_epochs: int = 5,
        warmup_factor: float = 0.01,
        # others
        jit: bool = False,
        channels_last: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        backbone = (
            backbones.__dict__[backbone]() if isinstance(backbone, str) else backbone
        )
        layers = [backbone]
        if include_pool:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            layers.append(nn.Flatten())
        layers.append(nn.Linear(backbone.get_last_out_channels(), num_classes))
        self.model = nn.Sequential(*layers)

        self.mixup_cutmix = (
            RandomCutMixMixUp(num_classes, cutmix_alpha, mixup_alpha)
            if cutmix_alpha > 0 and mixup_alpha > 0
            else None
        )
        if drop_out is not None:
            for m in self.model.modules():
                if isinstance(m, nn.modules.dropout._DropoutNd):
                    m.p = drop_out
        if drop_path is not None:
            for m in self.model.modules():
                if isinstance(m, StochasticDepth):
                    m.p = drop_path

        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if jit:
            self.model = torch.jit.script(self.model)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        if self.mixup_cutmix is not None:
            images, labels = self.mixup_cutmix(images, labels)
        if self.hparams.channels_last:
            images = images.to(memory_format=torch.channels_last)

        logits = self.model(images)
        loss = F.cross_entropy(
            logits, labels, label_smoothing=self.hparams.label_smoothing
        )
        self.log("train/loss", loss)

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
        self.log("val/acc", acc)

    def configure_optimizers(self):
        # split parameters
        # https://github.com/pytorch/vision/blob/main/torchvision/ops/_utils.py
        norm_classes = (
            nn.modules.batchnorm._BatchNorm,
            nn.modules.instancenorm._InstanceNorm,
            nn.LayerNorm,
            nn.GroupNorm,
        )
        layer_classes = (nn.Linear, nn.modules.conv._ConvNd)

        norm_params = []
        bias_params = []
        other_params = []
        for module in self.modules():
            if next(module.children(), None):
                other_params.extend(
                    p for p in module.parameters(recurse=False) if p.requires_grad
                )

            elif isinstance(module, norm_classes):
                norm_params.extend(p for p in module.parameters() if p.requires_grad)

            elif isinstance(module, layer_classes):
                if module.weight.requires_grad:
                    other_params.append(module.weight)
                if module.bias is not None and module.bias.requires_grad:
                    bias_params.append(module.bias)

            else:
                other_params.extend(p for p in module.parameters() if p.requires_grad)

        wd = self.hparams.weight_decay
        norm_wd = self.hparams.norm_weight_decay
        bias_wd = self.hparams.bias_weight_decay
        parameters = [
            {
                "params": norm_params,
                "weight_decay": norm_wd if norm_wd is not None else wd,
            },
            {
                "params": bias_params,
                "weight_decay": bias_wd if bias_wd is not None else wd,
            },
            {"params": other_params, "weight_decay": wd},
        ]
        parameters = [
            x for x in parameters if x["params"]
        ]  # remove empty params groups

        # build optimizer
        optimizer_name = self.hparams.optimizer
        lr = self.hparams.lr
        momentum = self.hparams.momentum
        if optimizer_name in ("SGD", "RMSprop"):
            optimizer_cls = partial(
                getattr(torch.optim, optimizer_name), momentum=momentum
            )
        elif hasattr(torch.optim, optimizer_name):
            optimizer_cls = getattr(optimizer_name)
        elif hasattr(timm.optim, optimizer_name):
            optimizer_cls = getattr(optimizer_name)
        else:
            raise ValueError(f"{optimizer_name} optimizer is not supported")
        optimizer = optimizer_cls(parameters, lr=lr, weight_decay=wd)

        # build scheduler
        warmup_epochs = self.hparams.warmup_epochs
        warmup_factor = self.hparams.warmup_factor
        decay_factor = self.hparams.decay_factor
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - warmup_epochs,
            eta_min=lr * decay_factor,
        )
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_epochs
            )
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, lr_scheduler],
                milestones=[warmup_epochs],
            )

            # https://github.com/pytorch/pytorch/issues/67318
            if not hasattr(lr_scheduler, "optimizer"):
                setattr(lr_scheduler, "optimizer", optimizer)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
