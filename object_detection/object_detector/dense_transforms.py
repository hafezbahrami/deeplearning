# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, *args):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            args = tuple(np.array([(image.width-x1, y0, image.width-x0, y1) for x0, y0, x1, y1 in boxes])
                         for boxes in args)
        return (image,) + args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)


class Normalize(T.Normalize):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args


class ToHeatmap(object):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, image, *dets):
        peak, size = detections_to_heatmap(dets, image.shape[1:], radius=self.radius)
        return image, peak, size

def detections_to_heatmap(dets, shape, radius=2, device=None):
    with torch.no_grad():
        size = torch.zeros((2, shape[0], shape[1]), device=device)
        peak = torch.zeros((len(dets), shape[0], shape[1]), device=device)
        for i, det in enumerate(dets):
            if len(det):
                det = torch.tensor(det.astype(float), dtype=torch.float32, device=device) # det contains 4-ponit for boxes within an image
                # Let's locate the boxes with the pixel of the input image. gx and gx will be non-zero at boxes' centers
                cx, cy = (det[:, 0] + det[:, 2] - 1) / 2, (det[:, 1] + det[:, 3] - 1) / 2  # cx and cy stores the (x,y) centres of boxes
                x = torch.arange(shape[1], dtype=cx.dtype, device=cx.device)
                y = torch.arange(shape[0], dtype=cy.dtype, device=cy.device)
                gx = (-((x[:, None] - cx[None, :]) / radius)**2).exp()  # gx is non-zero (or larger value) close to the x-centre of boxes
                gy = (-((y[:, None] - cy[None, :]) / radius)**2).exp()  # gy is non-zero (or larger value) close to the y-centre of boxes
                gaussian, id = (gx[None] * gy[:, None]).max(dim=-1) # image of x * y pixels, every x-value is multiplied to y-value
                mask = gaussian > peak.max(dim=0)[0] # mask the boxes with non-zero values
                det_size = (det[:, 2:] - det[:, :2]).T / 2 # (x_center, y_center) for each box
                size[:, mask] = det_size[:, id[mask]]
                peak[i] = gaussian
        return peak, size


# test: let's assume a 10 x 5 pixel, and two boxes in it: one at (0,0)&(2,2) the other (7,2)&(9,4)
# import numpy
# two_boxes=[[[0,0,2,2], [7,0,9,4]]]
# aa = detections_to_heatmap(numpy.array(two_boxes), torch.randn(5,10).shape, radius=0.2)
# print(aa)