# Vision Toolbox

Backbones, necks, and useful modules for Vision tasks.

## Backbones

Implemented backbones:

- [Darknet](#darknet)
- [VoVNet](#vovnet)

### ImageNet pre-training

Download ImageNet Challenge here: https://www.kaggle.com/c/imagenet-object-localization-challenge/

```bash
kaggle competitions download -c imagenet-object-localization-challenge
unzip imagenet-object-localization-challenge.zip -d ImageNet
tar -xf ImageNet/imagenet_object_localization_patched2019.tar.gz -C ImageNet
python ./scripts/imagenet.py --val_solution_path ./ImageNet/LOC_val_solution.csv --val_image_dir ./ImageNet/ILSVRC/Data/CLS-LOC/val
```

To create WebDataset shards

```bash
python ./scripts/wds.py --data_dir ./ImageNet/ILSVRC/Data/CLS-LOC/train --save_dir ./ImageNet/webdataset/train --shuffle True
python ./scripts/wds.py --data_dir ./ImageNet/ILSVRC/Data/CLS-LOC/val --save_dir ./ImageNet/webdataset/val --shuffle False
```

Reference training recipe:

- https://github.com/pytorch/vision/blob/main/references/classification/train.py
- https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
- Ross Wightman, ResNet strikesback: https://arxiv.org/abs/2110.00476

Training recipe

- Optimizer: SGD, learning rate = 0.5, weight decay = 2e-5, 100 epochs
- LR schedule: Linear warmup for 5 epochs, then cosine annealing
- Batch size: 1024 (512 per GPU, 2x RTX3090, DDP, no SyncBN)
- Augmentations: Random Resized Crop, Trivial Augmentation, Randome Erasing (p=0.1), CutMix (alpha=1.0), and MixUp (alpha=0.2). For each batch, either CutMix or MixUp is applied, but not both.
  - Random Erasing, CutMix, and Mixup are removed for small models (Darknet-19, VoVNet-19-slim)
- Label smoothing = 0.1
- Train res: 176
- Val resize: 232, Val crop: 224
- Mixed-precision training

Note: All hyperparameters are adopted from torchvision's recipe, except number of epochs (600 in torchvision's vs 100 in mine). Since the training time is shorter, augmentations should be reduced. Model EMA is not used.

PyTorch Lightning is used to train the models (see `classifier.py`). The easiest way to run training is to use Lightning CLI with a config file (see below). Note that PyTorch Lightning is not required to create, run, and load the models.

```bash
python train.py fit --config config.yaml
```

### Darknet

Paper: [[YOLOv2]](https://arxiv.org/abs/1612.08242) [[YOLOv3]](https://arxiv.org/abs/1804.02767)

- Darknet-{19,53}
- CSPDarknet-{19,53}

Darknet-53 is from YOLOv3. Darknet-19 is modified from YOLOv2 with improvements from YOLOv3 (replace stride 2 max pooling + 3x3 conv with single stride 2 3x3 conv and add skip connections). All LeakyReLU is replaced with ReLU.

Backbone                  | Top-1 acc | #Params(M) | FLOPS(G)*
--------------------------|-----------|------------|----------
Darknet-19                | 73.5      | 19.82      | 5.50
Darknet-19 (official^)    | 72.9      |            | 7.29
Darknet-53                | 77.3      | 40.64      | 14.33
Darknet-53 (official^)    | 77.2      | 41.57      | 18.57
CSPDarknet-53             | 77.1      | 26.28      | 9.42
CSPDarknet-53 (official^) | 77.2      | 27.61      | 13.07

*FLOPS is measured with `(1,3,224,224)` input.

^Sources: [[pjreddie's website]](https://pjreddie.com/darknet/imagenet/) [[WongKinYiu's CSP GitHub repo]](https://github.com/WongKinYiu/CrossStagePartialNetworks). Official Darknet models use 256x256 image, thus their FLOPS are slightly higher. The 1000-class classification head is probaby included in their Parameters and FLOPS count, resulting in slightly higher numbers.

### VoVNet

Paper: [[VoVNetV1]](https://arxiv.org/abs/1904.09730) [[VoVNetV2]](https://arxiv.org/abs/1911.06667)

- VoVNet-{19-slim,39,57,99}

All models use V2 by default (with skip connection + effective Squeeze-Excitation). To create V1 models, pass `residual=False` and `ese=False` to model constructor.

Implementation notes:

- Original implementation ([here](https://github.com/youngwanLEE/vovnet-detectron2/blob/master/vovnet/vovnet.py)) only applies eSE for stage 2 and 3 (each only has 1 block). timm ([here](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vovnet.py)) applies eSE for the last block of each stage. This implementation applies eSE for all blocks. This has more impact for deeper models (e.g. VoVNet-99) as they have more blocks per stage. Profiling shows that applying eSE for all blocks incur at most extra ~10% forward time for VoVNet-99.
- Both original implementation and timm merge max pool in stage 2 to the stem's last convolution (stage 1). This differs from VoVNetV1 paper. This implementation keeps max pool in stage 2. A few reasons for this: keep the code simple; stride 2 (stem) output is sufficiently good with 3 convs (although in practice rarely stride 2 output is used); stay faithful to the paper's specifications.
- VoVNet with depth-wise separable convolution is not implemented.

Backbone       | Top-1 acc | #Params(M) | FLOPS(G)*
---------------|-----------|------------|----------
VoVNet-19-slim | 70.7      | 2.65       | 4.77
VoVNet-39      | 78.1      | 25.18      | 15.57
VoVNet-57      |           | 41.45      | 19.30
VoVNet-99      |           | 69.52      | 34.43

*FLOPS is measured with `(1,3,224,224)` input.

### torchvision

Some torchvision classification models are ported to use with the toolbox. They can output multiple feature map levels.

ResNet:

- ResNet-{18,34,50,101,152}
- ResNeXt-{50,101}
- Wide ResNet-{50_2,101_2}

MobileNet:

- MobileNetV2
- MobileNetV3-{large,small}

EfficientNet:

- EfficientNet-{B0-B7}

Backbone          | Top-1 acc^ | #Params(M) | FLOPS(G)*
------------------|------------|------------|----------
ResNet-18         | 69.8       | 11.18      | 3.64
ResNet-34         | 73.3       | 21.28      | 7.34
ResNet-50         | 76.1       | 23.51      | 8.22
ResNet-101        | 77.4       | 42.50      | 15.66
ResNet-152        | 78.3       | 58.14      | 23.11
ResNeXt-50 32x4d  | 77.6       | 22.98      | 8.51
ResNeXt-101 32x8d | 79.3       | 86.74      | 32.95
Wide ResNet-50-2  | 78.5       | 66.83      | 22.85
Wide ResNet-101-2 | 78.9       | 124.84     | 45.59
MobileNetV2       | 71.9       | 2.22       | 0.6
MobileNetV3 large | 74.0       | 2.97       | 0.45
MobileNetV3 small | 67.7       | 0.93       | 0.12
EfficientNet B0   | 77.7       | 4.01       | 0.80
EfficientNet B1   | 78.6       | 6.51       | 1.18
EfficientNet B2   | 80.6       | 7.70       | 1.36
EfficientNet B3   | 82.0       | 10.70      | 1.98
EfficientNet B4   | 83.4       | 17.55      | 3.09
EfficientNet B5   | 83.4       | 28.34      | 4.82
EfficientNet B6   | 84.0       | 40.74      | 6.86
EfficientNet B7   | 84.1       | 63.79      | 10.53

*FLOPS is measured with `(1,3,224,224)` input.

^Top-1 accuracy is copied from [torchvision's documentation](https://pytorch.org/vision/stable/models.html) (0.11.0 at the time of writing)
