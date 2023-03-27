import numba
import numpy as np
from numba import jit
from numpy import isin
import torch
from PIL import Image
from .registry import DATASETS, PIPELINES, PIPELINES_WITH_INFO
from .base import BaseDataset
from .utils import to_numpy
from torch.utils.data import Dataset
from openselfsup.utils import print_log, build_from_cfg
from openselfsup.datasets.pipelines.transform_utils import Compose
from .builder import build_datasource
import torchvision.transforms.functional as F

@DATASETS.register_module
class ContrastiveDatasetBoxes(Dataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, resized_size=(224,224), grid_size=7, prefetch=False,
                 with_trans_info=True):
        data_source['return_label'] = False
        self.data_source = build_datasource(data_source)
        pipeline = [build_from_cfg(p, PIPELINES_WITH_INFO) for p in pipeline]
        self.pipeline = Compose(pipeline, with_trans_info=with_trans_info)
        self.prefetch = prefetch
        self.resized_size = resized_size
        self.grid_size = grid_size
        
        img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor_and_normalized = Compose(
            [build_from_cfg(p, PIPELINES) for p in 
                [dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)]]
        )

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        view1 = self.pipeline(img)
        view2 = self.pipeline(img)
        
        if isinstance(view1, Image.Image):
            if self.prefetch:
                img1 = torch.from_numpy(to_numpy(view1))
                img2 = torch.from_numpy(to_numpy(view2))
            img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
            return dict(img=img_cat)
        else:
            # with transformation information.
            img1, transf1, ratio1, size1 = view1.image, view1.transf, view1.ratio, view1.size
            img2, transf2, ratio2, size2 = view2.image, view2.transf, view2.ratio, view2.size

            w, h = img.size

            view_1_crop_range_and_flip = [
                transf1[0] / h,  # top
                transf1[1] / w,  # left
                transf1[2] / h,  # height
                transf1[3] / w,  # width
                transf1[4]
            ]

            view_2_crop_range_and_flip = [
                transf2[0] / h,
                transf2[1] / w,
                transf2[2] / h,
                transf2[3] / w,
                transf2[4]
            ]
            boxes1, h_step_1, w_step_1 = gen_bbox(*view_1_crop_range_and_flip, self.grid_size)
            boxes2, h_step_2, w_step_2 = gen_bbox(*view_2_crop_range_and_flip, self.grid_size)

            if self.prefetch:
                img1 = torch.from_numpy(to_numpy(img1))
                img2 = torch.from_numpy(to_numpy(img2))

            img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
            boxes = torch.cat((torch.from_numpy(boxes1).unsqueeze(0),
                               torch.from_numpy(boxes2).unsqueeze(0)), dim=0)
            return dict(img=img_cat, boxes=boxes)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented


@jit(nopython=True)
def gen_bbox(t: float, l: float, h: float, w: float, flip: bool, N: int=7):
    h_step = h / N
    w_step = w / N

    bbox = np.zeros((N, N, 4), dtype=numba.float32)
    for h_i in range(N):
        for w_i in range(N):
            h_st = t + h_i * h_step
            w_st = l + w_i * w_step
            bbox[h_i, w_i, :] = h_st, w_st, h_st + h_step, w_st + w_step

    if flip:
        bbox = np.fliplr(bbox)
        bbox = bbox.copy()

    return bbox, h_step, w_step