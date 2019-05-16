# FFBNet
FFBNET : LIGHTWEIGHT BACKBONE FOR OBJECT DETECTION BASED FEATURE FUSION BLOCK




## Installation
- Install [PyTorch-0.2.0-0.3.1](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository. This repository is mainly based on[RFBNet](https://github.com/ruinmessi/RFBNet), [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [Chainer-ssd](https://github.com/Hakuyume/chainer-ssd), a huge thank to them.
  * Note: We currently only support Python 3+.
- Compile the nms and coco tools:
```Shell
./make.sh
```
Note*: Check you GPU architecture support in utils/build.py, line 131. Default is:

``` 
'nvcc': ['-arch=sm_52',
```
- Install [pyinn](https://github.com/szagoruyko/pyinn) for MobileNet backbone:
```Shell
pip install git+https://github.com/szagoruyko/pyinn.git@master
```
- Then download the dataset by following the [instructions](#download-voc2007-trainval--test) below and install opencv. 
```Shell
conda install opencv
```
Note: For training, we currently  support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://mscoco.org/). 

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
