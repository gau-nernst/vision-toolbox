# Vision Backbones

Download ImageNet Challenge here: https://www.kaggle.com/c/imagenet-object-localization-challenge/

Implemented backbones:

- Darknet
- VoVNet

## Darknet

Paper: [[YOLOv2]](https://arxiv.org/abs/1612.08242) [[YOLOv3]](https://arxiv.org/abs/1804.02767)

- Darknet-53: from YOLOv3
- Darknet-19: modified from YOLOv2 with improvements from YOLOv3 (replace max pooling with stride 2 convolution and add residual connections)
- CSPDarknet-53
- CSPDarknet-19

For Cross Stage Partial (CPS) networks, there are two approaches to split the feature maps:

- Use 2 separate convolutions. YOLOv5 and YOLOX follow this
- Use 1 convolution and slice the tensor. CSP authors and YOLOv4 follow this

This implementation follows the first approach. More information about the two approaches: https://github.com/WongKinYiu/CrossStagePartialNetworks/issues/18

## VoVNet

Paper: [[VoVNet]](https://arxiv.org/abs/1904.09730) [[VoVNet2]](https://arxiv.org/abs/1911.06667)
