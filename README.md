# Vision Backbones

Implemented backbones:

- [Darknet](#darknet)
- [VoVNet](#vovnet)

## ImageNet

Download ImageNet Challenge here: https://www.kaggle.com/c/imagenet-object-localization-challenge/

```
unzip imagenet-object-localization-challenge
tar -xf imagenet-object-localization-challenge -C ImageNet
python ./scripts/imagenet.py --val_solution_path ./ImageNet/LOC_val_solution.csv --val_image_dir ./ImageNet/ILSVRC/Data/CLS-LOC/val
```

Reference training recipe:

- https://github.com/pytorch/vision/blob/main/references/classification/train.py
- https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
- Ross Wightman, ResNet strikesback: https://arxiv.org/abs/2110.00476

Training recipe

- Optimizer: SGD, learning rate = 0.5, weight decay = 2e-5, 100 epochs
- LR schedule: Linear warmup for 5 epochs, then cosine annealing
- Batch size: 1024 (512 per GPU, 2x RTX3090, DDP)
- Augmentations: Random Resized Crop, Trivial Augmentation, Randome Erasing (p=0.1), CutMix (alpha=1.0), and MixUp (alpha=0.2). For each batch, either CutMix or MixUp is applied, but not both.
- Label smoothing = 0.1
- Train res: 176
- Val resize: 232, Val crop: 224
- Mixed-precision training

Note: All hyperparameters are adopted from torchvision's recipe, except number of epochs (600 in torchvision's vs 100 in mine). Since the training time is shorter, augmentations should be reduced.

PyTorch Lightning is used to train the models (see `classifier.py`). The easiest way to run training is to use Lightning CLI with a config file (see below). Note that PyTorch Lightning is not required to create, run, and load the models.

```bash
python train.py fit --config config.yaml
```

## Darknet

Paper: [[YOLOv2]](https://arxiv.org/abs/1612.08242) [[YOLOv3]](https://arxiv.org/abs/1804.02767)

- Darknet-{19,53}
- CSPDarknet-{19,53}

Darknet-53 is from YOLOv3. Darknet-19 is modified from YOLOv2 with improvements from YOLOv3 (replace stride 2 max pooling + 3x3 conv with single stride 2 3x3 conv and add skip connections).

Backbone               | Top-1 acc | #Params(M) | FLOPS(G)*
-----------------------|-----------|------------|----------
Darknet-53             | 77.3      | 40.64      | 14.33
Darknet-53 (paper)^    | 77.2
CSPDarknet-53          |           | 26.28      | 9.42
CSPDarknet-53 (paper)^ | 77.2

^Paper uses 256x256 image

## VoVNet

Paper: [[VoVNetV1]](https://arxiv.org/abs/1904.09730) [[VoVNetV2]](https://arxiv.org/abs/1911.06667)

- VoVNet-{19-slim,39,57,99}

All models use V2 by default (with skip connection + effective Squeeze-Excitation). To create V1 models, pass `residual=False` and `ese=False` to model constructor.

Implementation notes:

- Original implementation ([here](https://github.com/youngwanLEE/vovnet-detectron2/blob/master/vovnet/vovnet.py)) only applies eSE for stage 2 and 3 (each only has 1 block). timm ([here](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vovnet.py)) applies eSE for the last block of each stage. This implementation applies eSE for all blocks. This has more impact for deeper models (e.g. VoVNet-99) as they have more blocks per stage. Profiling shows that applying eSE for all blocks incur at most extra ~10% forward time for VoVNet-99.
- Both original implementation and timm merge max pool in stage 2 to the stem's last convolution (stage 1). This differs from VoVNetV1 paper. This implementation keeps max pool in stage 2. A few reasons for this: keep the code simple; stride 2 (stem) output is sufficiently good with 3 convs (although in practice rarely stride 2 output is used); stay faithful to the paper's specifications.
- VoVNet with depth-wise separable convolution is not implemented.


Backbone  | Top-1 acc | #Params(M) | FLOPS(G)*
----------|-----------|------------|----------
VoVNet-39 |           | 25.18      | 15.62
VoVNet-57 |           | 41.45      | 19.35

## torchvision

Port some torchvision models to output multiple feature maps.

ResNet:

- ResNet-{18,34,50,101,152}
- ResNeXt-{50,101}
- Wide ResNet-{50_2,101_2}

MobileNet:

- MobileNetV2
- MobileNetV3-{large,small}

EfficientNet:

- EfficientNet-{B0-B7}

Backbone          | #Params(M) | FLOPS(G)*
------------------|------------|----------
ResNet-18         | 11.18      | 3.64
ResNet-34         | 21.28      | 7.34
ResNet-50         | 23.51      | 8.22
ResNet-101        | 42.50      | 15.66
ResNet-152        | 58.14      | 23.11
ResNeXt-50 32x4d  | 22.98      | 8.51
ResNeXt-101 32x8d | 86.74      | 32.95
MobileNetV2       | 2.22       | 0.6
MobileNetV3 large | 2.97       | 0.45
MobileNetV3 small | 0.93       | 0.12
EfficientNet B0   | 4.01       | 0.80
EfficientNet B1   | 6.51       | 1.18
EfficientNet B2   | 7.70       | 1.36
EfficientNet B3   | 10.70      | 1.98
EfficientNet B4   | 17.55      | 3.09
EfficientNet B5   | 28.34      | 4.82
EfficientNet B6   | 40.74      | 6.86
EfficientNet B7   | 63.79      | 10.53

*FLOPS is measured with `(1,3,224,224)` input.
