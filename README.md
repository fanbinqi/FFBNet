## FFBNet
FFBNET : LIGHTWEIGHT BACKBONE FOR OBJECT DETECTION BASED FEATURE FUSION BLOCK

## Our paper has been accepted by IEEE ICIP2019 for presentention.

### VOC2007 Test
| System                                   |  *mAP*   | **FPS** (1080Ti) |
| :--------------------------------------- | :------: | :-----------------------: |
| Mob-SSD |   68   |            190             |
| Tiny-Yolo v3 |   61.3   |           220             |
| Pelee |   70.9   |            -             |
| SSD |   77.2   |            160            |
| DSSD |  78.6   |            9.5             |
| STDN | 78.1 |            41             |
| FSSD | 78.8 |            140             |
| RefineDet |  80.0  |     40      |
| FFBNet |   73.54   |       185        |
| VGG-FFB |   80.2   |      142        |

## Installation
- Install [PyTorch 0.3.1](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository. This repository is mainly based on[FSSD](https://github.com/lzx1413/PytorchSSD), and a huge thank to him.
  * Note: We currently only support Python 3.5.
- Compile the nms and coco tools:
```Shell
./make.sh
```

Note: For training, we currently support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://mscoco.org/). 

## Datasets
To make things easy, we provide simple VOC and COCO dataset loader that inherits `torch.utils.data.Dataset` making it fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

### VOC Dataset
##### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training
```Shell
# Put vgg16_reducedfc.pth, and mobilenet_1.pth in a new folder weights and 
python train_test_mob.py or python train_test_vgg.py
```

If you are interested in this project, please QQ me (374873360)
