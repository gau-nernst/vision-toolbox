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

Note: All hyperparameters are adopted from torchvision's recipe, except number of epochs (600 in torchvision's vs 100 in mine). Since I train for much shorter time, I should have reduced regularizations.

I use PyTorch Lightning to train the models (see `classifier.py`). The easiest way to run training is to use Lightning CLI and use a config file.

```bash
python train.py fit --config config.yaml
```

## Darknet

Paper: [[YOLOv2]](https://arxiv.org/abs/1612.08242) [[YOLOv3]](https://arxiv.org/abs/1804.02767)

- Darknet-{19,53}
- CSPDarknet-{19,53}

Darknet-53 is from YOLOv3. Darknet-19 is modified from YOLOv2 with improvements from YOLOv3 (replace stride 2 max pooling + 3x3 conv with single stride 2 3x3 conv and add skip connections).

For Cross Stage Partial (CPS) networks, there are two approaches to split the feature maps:

- Use 2 separate convolutions. YOLOv5 and YOLOX follow this
- Use 1 convolution and slice the tensor. CSP authors and YOLOv4 follow this

This implementation follows the first approach. More information about the two approaches: https://github.com/WongKinYiu/CrossStagePartialNetworks/issues/18

Backbone | Top-1 acc
---------|----------
Darknet-53 (224x224) | 77.3
Darknet-53 (paper, 256x256) | 77.2

## VoVNet

Paper: [[VoVNetV1]](https://arxiv.org/abs/1904.09730) [[VoVNetV2]](https://arxiv.org/abs/1911.06667)

- VoVNet-{19-slim,39,57,99}

All models use V2 by default (with skip connection + effective Squeeze-Excitation). To create V1 models, pass `residual=False` and `ese=False` to model constructor.

Backbone | Top-1 acc
---------|----------
VoVNet-57 (224x224) | 79.5

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
