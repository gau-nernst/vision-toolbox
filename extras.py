import os
import io
import math
import time
import hashlib

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
class RandomMixup(nn.Module):
    def __init__(self, num_classes, p=0.5, alpha=1, inplace=False):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: torch.Tensor, target: torch.Tensor):
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.num_classes).to(
                dtype=batch.dtype
            )

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target


class RandomCutmix(nn.Module):
    def __init__(self, num_classes, p=0.5, alpha=1, inplace=False):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: torch.Tensor, target: torch.Tensor):
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.num_classes).to(
                dtype=batch.dtype
            )

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        W, H = TF.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target


class RandomCutMixMixUp(nn.Module):
    def __init__(self, num_classes, cutmix_alpha, mixup_alpha, inplace=False):
        super().__init__()
        if cutmix_alpha == 0 and mixup_alpha == 0:
            raise ValueError

        self.cutmix = (
            RandomCutmix(num_classes, p=1, alpha=cutmix_alpha, inplace=inplace)
            if cutmix_alpha > 0
            else None
        )
        self.mixup = (
            RandomMixup(num_classes, p=1, alpha=mixup_alpha, inplace=inplace)
            if mixup_alpha > 0
            else None
        )

    def forward(self, batch, target):
        if self.cutmix is None or torch.rand(1).item() >= 0.5:
            return self.mixup(batch, target)

        return self.cutmix(batch, target)


def extract_backbone_weights(lightning_ckpt_path, save_name, save_dir=None):
    if save_dir == None:
        save_dir = os.getcwd()

    ckpt = torch.load(lightning_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    backbone_token = "model.0."
    backbone_weights = {
        k[len(backbone_token) :]: v
        for k, v in state_dict.items()
        if k.startswith(backbone_token)
    }

    buffer = io.BytesIO()
    torch.save(backbone_weights, buffer)
    bin_data = buffer.getvalue()
    hash = hashlib.sha256(bin_data).hexdigest()[:8]

    name = os.path.join(save_dir, f"{save_name}-{hash}.pth")
    with open(name, "wb") as f:
        f.write(bin_data)


# Modified from YOLOv5 utils/torch_utils.py
def profile(module: nn.Module, input: torch.Tensor = None, n: int = 10, device="cpu"):
    from fvcore.nn import FlopCountAnalysis

    if input is None:
        input = torch.randn((1, 3, 224, 224))

    input = input.to(device)
    module = module.to(device)
    input.requires_grad = True

    flops = FlopCountAnalysis(module, input).total() / 1e9 * 2  # GFLOPs

    def time_sync(device):
        if device != "cpu":
            torch.cuda.synchronize(device=device)
        return time.time()

    tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
    for _ in range(n):
        t[0] = time_sync(device)
        out = module(input)
        t[1] = time_sync(device)

        out.sum().backward()
        t[2] = time_sync(device)

        tf += t[1] - t[0]
        tb += t[2] - t[1]

    tf *= 1000 / n  # convert to ms and take average
    tb *= 1000 / n

    mem = (
        torch.cuda.memory_reserved(device) / 1e9 if torch.cuda.is_available() else 0
    )  # GB
    params = sum([x.numel() for x in module.parameters()]) / 1e6  # M
    torch.cuda.empty_cache()

    return params, flops, mem, tf, tb
