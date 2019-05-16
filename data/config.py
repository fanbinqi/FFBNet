# config.py

# gets home dir cross platform
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# note: if you used our download scripts, this should be right
VOCroot = '/home/zdh1901/data/VOCdevkit'  # path to VOCdevkit root dir

COCOroot = '/home/zdh1901/data/coco'

# RFB CONFIGS
VOC_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    #'feature_maps': [1, 3, 5, 10, 19, 38],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],
    #'steps': [300, 100, 64, 32, 16, 8],

    'min_sizes': [30, 60, 111, 162, 213, 264],

    'max_sizes': [60, 111, 162, 213, 264, 315],
    # 'min_sizes': [264, 213, 163, 111, 60, 30],
    #
    # 'max_sizes': [315, 264, 213, 163, 111, 60],

    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    #'aspect_ratios' : [[2], [2], [2, 3], [2, 3], [2, 3], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

VOC_512 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 512,

    'steps': [14, 27, 51, 102, 170, 512],

    'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0],

    'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8],

    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

COCO_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [21, 45, 99, 153, 207, 261],

    'max_sizes': [45, 99, 153, 207, 261, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

COCO_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],

    'min_dim': 512,

    'steps': [8, 16, 32, 64, 128, 256, 512],

    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

COCO_mobile_300 = {
    'feature_maps': [19, 10, 5, 3, 2, 1],

    'min_dim': 300,

    'steps': [16, 32, 64, 100, 150, 300],

    'min_sizes': [45, 90, 135, 180, 225, 270],

    'max_sizes': [90, 135, 180, 225, 270, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

VOC_320 = {
    'feature_maps': [40, 20, 10, 5],

    'min_dim': 320,

    'steps': [8, 16, 32, 64],

    'min_sizes': [32, 64, 128, 256],

    'max_sizes': [],

    'aspect_ratios': [[2], [2], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}
