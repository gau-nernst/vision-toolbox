import math

import torch
from torch import nn
import torchvision.transforms.functional as TF

# https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
class RandomMixup(nn.Module):
    def __init__(self, num_classes, p=0.5, alpha=1, inplace=False):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch, target):
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
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

    def forward(self, batch, target):
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
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
        
        self.cutmix = RandomCutmix(num_classes, p=1, alpha=cutmix_alpha, inplace=inplace) if cutmix_alpha > 0 else None
        self.mixup = RandomMixup(num_classes, p=1, alpha=mixup_alpha, inplace=inplace) if mixup_alpha > 0 else None

    def forward(self, batch, target):
        if self.cutmix is None or torch.rand(1).item() >= 0.5:
            return self.mixup(batch, target)
            
        return self.cutmix(batch, target)
