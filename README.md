# Vision Toolbox

Backbones, necks, and useful modules for Vision tasks.

Special thanks to [MLDA@EEE Lab](https://www.ntu.edu.sg/eee/student-life/mlda) for providing GPU resources to train the models.

## Installation

Install PyTorch and torchvision from conda. Then install this GitHub repo directly

```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/gau-nernst/vision-toolbox.git
```

To update

```bash
pip install -U --force-reinstall --no-deps git+https://github.com/gau-nernst/vision-toolbox.git
```

## Usage

```python
from vision_toolbox import backbones

model = backbones.cspdarknet53(pretrained=True)
model(inputs)                       # last feature map, stride 32
model.get_feature_maps(inputs)      # list of 5 feature maps, stride 2, 4, 8, 16, 32
model.out_channels_list             # tuple of output channels, corresponding to each feature map
```

For object detection, usually only the last 4 feature maps are used. It is the responsibility of the user to select the last 4 feature maps from `.get_feature_maps()`

```python
outputs = model.get_feature_maps(inputs)[-4:]   # last 4 feature maps
```

## Backbones with ported weights

- ViT: AugReg (Google), DeiT/DeiT3 (Facebook), SigLIP (Google)
- MLP-Mixer
- CaiT
- Swin 
- ConvNeXt and ConvNeXt-V2

## Backbones trained by me

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
python ./scripts/wds.py --data_dir ./ImageNet/ILSVRC/Data/CLS-LOC/train --save_dir ./ImageNet/webdataset/train --shuffle True --name imagenet-train
python ./scripts/wds.py --data_dir ./ImageNet/ILSVRC/Data/CLS-LOC/val --save_dir ./ImageNet/webdataset/val --shuffle False --name imagenet-val
```

There should be 147 shards of training set, and 7 shards of validation set. Each shard is 1GB.

Reference recipes:

- https://github.com/pytorch/vision/blob/main/references/classification/train.py
- https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
- Ross Wightman, ResNet strikesback: https://arxiv.org/abs/2110.00476

Recipe used in this repo:

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
- For large models, FixRes is removed -> train resized crop is 224 (e.g. VoVNet-99)

Note: All hyperparameters are adopted from torchvision's recipe, except number of epochs (600 in torchvision's vs 100 in mine). Since the training time is shorter, augmentations should be reduced. Model EMA is not used.

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is used to train the models (see `classifier.py`). The easiest way to run training is to use [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html) with a config file (see below, `jsonargparse[signatures]` is required). Note that PyTorch Lightning is not required to create, run, and load the models. Some config files are provided in this repo.

```bash
git clone https://github.com/gau-nernst/vision-toolbox.git
cd vision-toolbox
pip install pytorch-lightning jsonargparse[signatures]    # dependencies for Lightning CLI
python train.py fit --config config.yaml
python train.py fit --config config.yaml --model.backbone cspdarknet53    # change backbone to train
python train.py fit --config config.yaml --config config_wds.yaml         # use webdataset
```

### Darknet

Paper: [[YOLOv2]](https://arxiv.org/abs/1612.08242) [[YOLOv3]](https://arxiv.org/abs/1804.02767) [[CSPNet]](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf) [[YOLOv4]](https://arxiv.org/abs/2004.10934)

- Darknet-{19,53}
- CSPDarknet-53
- Darknet-YOLOv5{n,s,m,l,x}

<details><summary>Implementation notes</summary>

- Darknet-53 is from YOLOv3. Darknet-19 is modified from YOLOv2 with improvements from YOLOv3 (replace stride 2 max pooling + 3x3 conv with single stride 2 3x3 conv and add skip connections). CSPDarknet-53 originates from CSPNet and is used in YOLOv4, though it is not discussed in either paper. All LeakyReLU are replaced with ReLU.
- In CSPDarknet-53's [original config](https://github.com/WongKinYiu/CrossStagePartialNetworks/blob/master/cfg/csdarknet53.cfg), stage 1 does not have reduced width even though it has cross-stage connection. This implementation maintains the width reduction in stage 1.
- Darknet-YOLOv5 is adapted from Ultralytics' [YOLOv5](https://github.com/ultralytics/yolov5). It is the `backbone` section in the [config files](https://github.com/ultralytics/yolov5/blob/master/models/yolov5l.yaml) without the SPPF module. All SiLU are replaced with ReLU. For batch norm layers, YOLOv5 set `eps=1e-3` and `momentum=0.03`, but this implementation keeps the default PyTorch values.

</details>

Backbone                  | Top-1 acc | #Params(M) | FLOPS(G)* | Train recipe
--------------------------|-----------|------------|-----------|--------------
Darknet-19                | 73.5      | 19.82      |  5.53     | small
Darknet-53                | 76.9      | 40.58      | 14.31     | default
CSPDarknet-53             | 77.5      | 26.24      |  9.44     | default
Darknet-YOLOv5n           | 56.3      |  0.88      |  0.33     | small
Darknet-YOLOv5s           | 67.3      |  3.51      |  1.23     | small
Darknet-YOLOv5m           | 73.8      | 10.69      |  3.70     | small
Darknet-YOLOv5l           | 76.9      | 23.96      |  8.31     | default
Darknet-YOLOv5x           | 78.6      | 45.18      | 15.73     | large, batch_size = 512

*FLOPS is measured with `(1,3,224,224)` input.

For reference, official Darknet models (Sources: [[pjreddie's website]](https://pjreddie.com/darknet/imagenet/) [[WongKinYiu's CSP GitHub repo]](https://github.com/WongKinYiu/CrossStagePartialNetworks)):

- Darknet-19: top-1 72.9%, 7.29G FLOPS
- Darknet-53: top-1 77.2%, 41.57M params, 18.57 FLOPS
- CSPDarknet-53: top-1 77.2%, 27.61M params, 13.07 FLOPS
- These models use 256x256 image, thus their FLOPS are slightly higher
- The 1000-class classification head is probaby included in their Parameters and FLOPS count, resulting in slightly higher numbers.

#### Convert YOLOv5 backbone weights to use with [Ultralytics' repo](https://github.com/ultralytics/yolov5) (WIP)

```bash
python scripts/convert_yolov5_weights.py {weights_from_this_repo.pth} {save_path.pth}
```

The weights will be renamed to be compatiable with [Ultralytics' repo](https://github.com/ultralytics/yolov5). Note that the converted `.pth` file only contains the renamed state dict, up to `model.8` (the backbone part, without the SPPF layer). You will need to modify their [train script](https://github.com/ultralytics/yolov5/blob/master/train.py#L123) to treat the loaded file as a state dict, instead of a dictionary with key `model` containing the model object.

I haven't tested training a full YOLOv5 object detector with the converted weights, so this function is not guaranteed to work correctly.

### VoVNet

Paper: [[VoVNetV1]](https://arxiv.org/abs/1904.09730) [[VoVNetV2]](https://arxiv.org/abs/1911.06667)

- VoVNet-(27-slim,39,57)
- VoVNet-{19-slim,19,39,57,99}-ese

<details><summary>Implementation notes</summary>

- All models have skip connections, which are not present in the original VoVNetV1. Non-ese (effective Squeeze-and-Excitation) models are from V1, while models with ese are from V2. The decision to keep V1 models is to have better compatibility for edge accelerators.
- Original implementation ([here](https://github.com/youngwanLEE/vovnet-detectron2/blob/master/vovnet/vovnet.py)) only applies eSE for stage 2 and 3 (each only has 1 block). timm ([here](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vovnet.py)) applies eSE for the last block of each stage. This implementation applies eSE for all blocks. This has more impact for deeper models (e.g. VoVNet-99) as they have more blocks per stage. Profiling shows that applying eSE for all blocks incur at most extra ~10% forward time for VoVNet-99.
- VoVNet-19-slim output channels in stage 2 is changed to 128. Original implementation is 112.
- Both original implementation and timm merge max pool in stage 2 to the stem's last convolution (stage 1). This is not mentioned in the papers. This repo's implementation keeps max pool in stage 2. A few reasons for this: keep the code simple; stride 2 (stem) output is sufficiently good with 3 convs (although in practice rarely stride 2 output is used).
- VoVNet with depth-wise separable convolution is not implemented.

</details>

Backbone           | Top-1 acc | #Params(M) | FLOPS(G)* | Train recipe
-------------------|-----------|------------|-----------|--------------
VoVNet-27-slim     | 71.6      |  3.01      |  5.79     | small
VoVNet-39          | 78.2      | 21.58      | 15.61     | default
VoVNet-57          | 79.3      | 35.62      | 19.33     | default
VoVNet-19-slim-ese | 70.8      |  2.68      |  4.85     | small
VoVNet-19-ese      | 75.4      | 10.18      |  9.70     | small
VoVNet-39-ese      | 78.1      | 25.18      | 15.62     | default
VoVNet-57-ese      | 79.2      | 41.45      | 19.35     | default
VoVNet-99-ese      | 80.7      | 69.52      | 34.51     | large, batch_size=512

*FLOPS is measured with `(1,3,224,224)` input.

### torchvision

Some torchvision classification models are ported to use with the toolbox. They can output multiple feature map levels. `torchvision>=0.11.0` is required since its feature extraction API is used to do this.

- For MobileNet and EfficientNet models, intermediate outputs are taken after the first 1x1 conv expansion layer of the strided MBConv block. See Section 6.2 of [MobileNetv2 paper](https://arxiv.org/abs/1801.04381) and Section 6.3 of [MobileNetv3 paper](https://arxiv.org/abs/1905.02244).
- To use weights from the new PyTorch training recipe (which are significantly better), go to torchvision's [prototype](https://github.com/pytorch/vision/tree/main/torchvision/prototype/models) directory and copy weights URLs (labelled as `ImageNet1K_V2`) from their respective models' files.

ResNet:

- ResNet-{18,34,50,101,152}
- ResNeXt-{50,101}
- Wide ResNet-{50_2,101_2}

RegNet:

- RegNetX-{400MF,800MF,1.6GF,3.2GF,8GF,16GF,32GF}
- RegNetY-{400MF,800MF,1.6GF,3.2GF,8GF,16GF,32GF} (with SE)

MobileNet:

- MobileNetV2
- MobileNetV3-{large,small}

EfficientNet:

- EfficientNet-{B0-B7}

<details><summary>Profiling</summary>

Backbone          | Top-1 acc^ | #Params(M) | FLOPS(G)*
------------------|------------|------------|----------
**ResNet family**
ResNet-18         | 69.8       |  11.18     |  3.64
ResNet-34         | 73.3       |  21.28     |  7.34
ResNet-50         | 76.1       |  23.51     |  8.22
ResNet-101        | 77.4       |  42.50     | 15.66
ResNet-152        | 78.3       |  58.14     | 23.11
ResNeXt-50 32x4d  | 77.6       |  22.98     |  8.51
ResNeXt-101 32x8d | 79.3       |  86.74     | 32.95
Wide ResNet-50-2  | 78.5       |  66.83     | 22.85
Wide ResNet-101-2 | 78.9       | 124.84     | 45.59
**RegNet family**
RegNetX-400MF     | 72.8       |   5.10     |  0.84
RegNetX-800MF     | 75.2       |   6.59     |  1.62
RegNetX-1.6GF     | 77.0       |   8.28     |  3.24
RegNetX-3.2GF     | 78.4       |  14.29     |  6.40
RegNetX-8GF       | 79.3       |  37.65     | 16.04
RegNetX-16GF      | 80.1       |  52.23     | 31.98
RegNetX-32GF      | 80.6       | 105.29     | 63.61
RegNetY-400MF     | 74.0       |   3.90     |  0.82
RegNetY-800MF     | 76.4       |   5.65     |  1.69
RegNetY-1.6GF     | 78.0       |  10.31     |  3.26
RegNetY-3.2GF     | 78.9       |  17.92     |  6.40
RegNetY-8GF       | 80.0       |  37.36     | 17.03
RegNetY-16GF      | 80.4       |  80.57     | 31.92
RegNetY-32GF      | 80.9       | 141.33     | 64.69
**MobileNet family**
MobileNetV2       | 71.9       |   2.22     |  0.63
MobileNetV3 large | 74.0       |   2.97     |  0.45
MobileNetV3 small | 67.7       |   0.93     |  0.12
**EfficientNet family**
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

</details>

## Necks

- [FPN](https://arxiv.org/abs/1612.03144)
- [SemanticFPN](https://arxiv.org/abs/1901.02446)
- [PAN](https://arxiv.org/abs/1803.01534)
- [BiFPN](https://arxiv.org/abs/1911.09070)

Implementation notes

- FPN: no batch norm and activation are used in lateral connections. Output convolutions (with batch norm and ReLU) are inside the top-down path.

  ```python
  P5 = lateral_conv5(C5)
  P4 = out_conv4(lateral_conv4(C4) + upsample(P5))
  P3 = out_conv4(lateral_conv4(C3) + upsample(P4))
  ```

- BiFPN: weighted fusion is not implemented. Normal 3x3 convolutions (with batch norm and ReLU) are used by default.
