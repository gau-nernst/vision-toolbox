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

Training tricks from torchvision's recipe:

- LR schedule: linear warmup with cosine annealing
- Label smoothing
- Augmentations: Trivial Augmentation, Randome Erasing, CutMix, and MixUp
- FixRes
- Model EMA (not implemented)

## Darknet

Paper: [[YOLOv2]](https://arxiv.org/abs/1612.08242) [[YOLOv3]](https://arxiv.org/abs/1804.02767)

- Darknet-{19,53}
- CSPDarknet-{19,53}

Darknet-53 is from YOLOv3. Darknet-19 is modified from YOLOv2 with improvements from YOLOv3 (replace max pooling with stride 2 convolution and add residual connections)

For Cross Stage Partial (CPS) networks, there are two approaches to split the feature maps:

- Use 2 separate convolutions. YOLOv5 and YOLOX follow this
- Use 1 convolution and slice the tensor. CSP authors and YOLOv4 follow this

This implementation follows the first approach. More information about the two approaches: https://github.com/WongKinYiu/CrossStagePartialNetworks/issues/18

## VoVNet

Paper: [[VoVNetV1]](https://arxiv.org/abs/1904.09730) [[VoVNetV2]](https://arxiv.org/abs/1911.06667)

- VoVNet-{19-slim,39,57,99}

All models use V2 by default (with skip connection + effective Squeeze-Excitation). To create V1 models, pass `residual=False` and `ese=False` to model constructor.

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
