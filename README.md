# Vision Toolbox

Backbones, necks, and useful modules for Vision tasks.

Special thanks to [MLDA@EEE Lab](https://www.ntu.edu.sg/eee/student-life/mlda) for providing GPU resources to train the models.

## Installation

Install PyTorch and torchvision from conda. Then install this GitHub repo directly

```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/gau-nernst/vision-toolbox.git
```

## Usage

```python
from vision_toolbox import backbones

model = backbones.cspdarknet53(pretrained=True)
model(inputs)                       # last feature map, stride 32
model.forward_features(inputs)      # list of 5 feature maps, stride 2, 4, 8, 16, 32
model.get_out_channels()            # channels of output feature maps
```

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
pip install webdataset        # install webdataset
python ./scripts/wds.py --data_dir ./ImageNet/ILSVRC/Data/CLS-LOC/train --save_dir ./ImageNet/webdataset/train --shuffle True
python ./scripts/wds.py --data_dir ./ImageNet/ILSVRC/Data/CLS-LOC/val --save_dir ./ImageNet/webdataset/val --shuffle False
```

There should be 147 shards of training set, and 7 shards of validation set. Each shard is 1GB.

Reference training recipe:

- https://github.com/pytorch/vision/blob/main/references/classification/train.py
- https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
- Ross Wightman, ResNet strikesback: https://arxiv.org/abs/2110.00476

Training recipe

- Optimizer: SGD
- Epochs: 100
- Learning rate: 0.5 for batch size 1024, linear scaling for other batch sizes
- Weight decay: 2e-5
- Batch size: 1024, unless stated otherwise (for larger models)
- LR schedule: Linear warmup for 5 epochs, then cosine annealing
- Augmentations: (1) Random Resized Crop, (2) Trivial Augmentation, (3) Random Erasing (p=0.1), (4) CutMix (alpha=1.0), and (5) MixUp (alpha=0.2). For each batch, either CutMix or MixUp is applied, but not both.
- Label smoothing = 0.1
- Val resize: 232, Val crop: 224
- Train resized crop: 176 (FixRes)
- Mixed-precision training
- For small models, Random Erasing, CutMix, Mixup, and Label smoothing are removed (e.g. Darknet-19, VoVNet-19)
- For large models, FixRes is removed -> train resized crop is 176 (e.g. VoVNet-99)

Note: All hyperparameters are adopted from torchvision's recipe, except number of epochs (600 in torchvision's vs 100 in mine). Since the training time is shorter, augmentations should be reduced. Model EMA is not used.

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is used to train the models (see `classifier.py`). The easiest way to run training is to use [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html) with a config file (see below, `jsonargparse[signatures]` is required). Note that PyTorch Lightning is not required to create, run, and load the models. Some config files are provided in this repo.

```bash
pip install pytorch-lightning jsonargparse[signatures]    # dependencies for Lightning CLI
python train.py fit --config config.yaml
python train.py fit --config config.yaml --model.backbone cspdarknet53    # change backbone to train
python train.py fit --config config.yaml --config config_wds.yaml         # use webdataset
```

### Darknet

Paper: [[YOLOv2]](https://arxiv.org/abs/1612.08242) [[YOLOv3]](https://arxiv.org/abs/1804.02767) [[CSPNet]](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf)

- Darknet-{19,53}
- CSPDarknet-53
- Darknet-YOLOv5{n,s,m,l,x}

Darknet-53 is from YOLOv3. Darknet-19 is modified from YOLOv2 with improvements from YOLOv3 (replace stride 2 max pooling + 3x3 conv with single stride 2 3x3 conv and add skip connections). All LeakyReLU are replaced with ReLU.

Darknet-YOLOv5 is adapted from Ultralytics' [YOLOv5](https://github.com/ultralytics/yolov5). It is the `backbone` section in the [config files](https://github.com/ultralytics/yolov5/blob/master/models/yolov5l.yaml) without the SPPF module. All SiLU are replaced with ReLU.

Backbone                  | Top-1 acc | #Params(M) | FLOPS(G)* | Train recipe
--------------------------|-----------|------------|-----------|--------------
Darknet-19                | 73.5      | 19.82      |  5.53     | small
Darknet-19 (official^)    | 72.9      |            |  7.29     | NA
Darknet-53                | 77.3      | 40.64      | 14.38     | default
Darknet-53 (official^)    | 77.2      | 41.57      | 18.57     | NA
CSPDarknet-53             | 77.1      | 26.28      |  9.48     | default
CSPDarknet-53 (official^) | 77.2      | 27.61      | 13.07     | NA
Darknet-YOLOv5n           |           |  0.88      |  0.33     | small
Darknet-YOLOv5s           |           |  3.51      |  1.23     | small
Darknet-YOLOv5m           |           | 10.69      |  3.70     | small
Darknet-YOLOv5l           |           | 23.96      |  8.31     | default
Darknet-YOLOv5x           |           | 45.18      | 15.73     | default

*FLOPS is measured with `(1,3,224,224)` input.

^Sources: [[pjreddie's website]](https://pjreddie.com/darknet/imagenet/) [[WongKinYiu's CSP GitHub repo]](https://github.com/WongKinYiu/CrossStagePartialNetworks). Official Darknet models use 256x256 image, thus their FLOPS are slightly higher. The 1000-class classification head is probaby included in their Parameters and FLOPS count, resulting in slightly higher numbers.

### VoVNet

Paper: [[VoVNetV1]](https://arxiv.org/abs/1904.09730) [[VoVNetV2]](https://arxiv.org/abs/1911.06667)

- VoVNet-{19-slim,19,39,57,99}

All models use V2 by default (with skip connection + effective Squeeze-Excitation). To create V1 models, pass `residual=False` and `ese=False` to model constructor.

Implementation notes:

- Original implementation ([here](https://github.com/youngwanLEE/vovnet-detectron2/blob/master/vovnet/vovnet.py)) only applies eSE for stage 2 and 3 (each only has 1 block). timm ([here](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vovnet.py)) applies eSE for the last block of each stage. This implementation applies eSE for all blocks. This has more impact for deeper models (e.g. VoVNet-99) as they have more blocks per stage. Profiling shows that applying eSE for all blocks incur at most extra ~10% forward time for VoVNet-99.
- Both original implementation and timm merge max pool in stage 2 to the stem's last convolution (stage 1). This is not mentioned in the papers. This repo's implementation keeps max pool in stage 2. A few reasons for this: keep the code simple; stride 2 (stem) output is sufficiently good with 3 convs (although in practice rarely stride 2 output is used).
- VoVNet with depth-wise separable convolution is not implemented.

Backbone       | Top-1 acc | #Params(M) | FLOPS(G)* | Train recipe
---------------|-----------|------------|-----------|--------------
VoVNet-19-slim | 70.7      |  2.65      |  4.80     | small
VoVNet-19      | 75.4      | 10.18      |  9.70     | small
VoVNet-39      | 78.1      | 25.18      | 15.62     | default
VoVNet-57      | 79.2      | 41.45      | 19.35     | default
VoVNet-99      | 80.7      | 69.52      | 34.51     | large, batch_size=512

*FLOPS is measured with `(1,3,224,224)` input.

### torchvision

Some torchvision classification models are ported to use with the toolbox. They can output multiple feature map levels. `torchvision>=0.11.0` is required since its feature extraction API is used to do this.

- For MobileNet and EfficientNet models, intermediate outputs are taken after the first 1x1 conv expansion layer of the strided MBConv block. See Section 6.2 of [MobileNetv2 paper](https://arxiv.org/abs/1801.04381) and Section 6.3 of [MobileNetv3 paper](https://arxiv.org/abs/1905.02244).
- To use weights from the new PyTorch training recipe, go to torchvision's [prototype](https://github.com/pytorch/vision/tree/main/torchvision/prototype/models) directory and copy weights URLs (labelled as `ImageNet1K_V2`) from their respective models' files.

ResNet:

- ResNet-{18,34,50,101,152}
- ResNeXt-{50,101}
- Wide ResNet-{50_2,101_2}

MobileNet:

- MobileNetV2
- MobileNetV3-{large,small}

RegNet:

- RegNetX-{400MF,800MF,1.6GF,3.2GF,8GF,16GF,32GF}
- RegNetY-(400MF,800MF,1.6GF,3.2GF,8GF,16GF,32GF) (with SE)

EfficientNet:

- EfficientNet-{B0-B7}

Backbone          | Top-1 acc^ | #Params(M) | FLOPS(G)*
------------------|------------|------------|----------
ResNet-18         | 69.8       |  11.18     |  3.64
ResNet-34         | 73.3       |  21.28     |  7.34
ResNet-50         | 76.1       |  23.51     |  8.22
ResNet-101        | 77.4       |  42.50     | 15.66
ResNet-152        | 78.3       |  58.14     | 23.11
ResNeXt-50 32x4d  | 77.6       |  22.98     |  8.51
ResNeXt-101 32x8d | 79.3       |  86.74     | 32.95
Wide ResNet-50-2  | 78.5       |  66.83     | 22.85
Wide ResNet-101-2 | 78.9       | 124.84     | 45.59
MobileNetV2       | 71.9       |   2.22     |  0.63
MobileNetV3 large | 74.0       |   2.97     |  0.45
MobileNetV3 small | 67.7       |   0.93     |  0.12
EfficientNet B0   | 77.7       |   4.01     |  0.80
EfficientNet B1   | 78.6       |   6.51     |  1.18
EfficientNet B2   | 80.6       |   7.70     |  1.36
EfficientNet B3   | 82.0       |  10.70     |  1.98
EfficientNet B4   | 83.4       |  17.55     |  3.09
EfficientNet B5   | 83.4       |  28.34     |  4.82
EfficientNet B6   | 84.0       |  40.74     |  6.86
EfficientNet B7   | 84.1       |  63.79     | 10.53

*FLOPS is measured with `(1,3,224,224)` input.

^Top-1 accuracy is copied from [torchvision's documentation](https://pytorch.org/vision/stable/models.html) (0.11.0 at the time of writing)

## Necks

- [FPN](https://arxiv.org/abs/1612.03144)
- [SemanticFPN](https://arxiv.org/abs/1901.02446)
- [PAN](https://arxiv.org/abs/1803.01534)
- [BiFPN](https://arxiv.org/abs/1911.09070)

Implementation notes

- FPN: no batch norm and activation are used in lateral connections. Output convolutions (with batch norm and ReLU) are inside the top-down path..

  ```python
  P5 = lateral_conv5(C5)
  P4 = out_conv4(lateral_conv4(C4) + upsample(P5))
  P3 = out_conv4(lateral_conv4(C3) + upsample(P4))
  ```

- BiFPN: weighted fusion is not implemented. Normal 3x3 convolutions (with batch norm and ReLU) are used by default.
